//! Benchmark ops shared by both targets: the wasm cdylib (timed from node by
//! bench.mjs) and the native bin (timed with std::time::Instant). Identical
//! code runs on both sides, so the ratio is apples-to-apples.
//!
//! Protocol: call `setup(n)` once per size (allocates + fills inputs,
//! untimed), then time repeated calls of `run_*()`. Each run returns a probe
//! double so the work can't be dead-code-eliminated.

#![cfg_attr(target_arch = "wasm32", no_std)]

extern crate alloc;

use faer::prelude::*;
use faer::Mat;

struct State {
    a: Mat<f64>,
    b: Mat<f64>,
    sym: Mat<f64>,
    rhs: Mat<f64>,
}

struct StateCell(core::cell::UnsafeCell<Option<State>>);
unsafe impl Sync for StateCell {}
static STATE: StateCell = StateCell(core::cell::UnsafeCell::new(None));

fn state() -> &'static State {
    unsafe { (*STATE.0.get()).as_ref().expect("call setup(n) first") }
}

// deterministic fill (splitmix-style LCG), values in [-1, 1]
fn fill(nrows: usize, ncols: usize, mut s: u64) -> Mat<f64> {
    Mat::from_fn(nrows, ncols, |_, _| {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((s >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    })
}

#[no_mangle]
pub extern "C" fn setup(n: usize) {
    let a = fill(n, n, 0x9E3779B97F4A7C15);
    let b = fill(n, n, 0xD1B54A32D192ED03);
    // symmetric, diagonally dominant: well-conditioned for the EVD op
    let at = a.transpose().to_owned();
    let mut sym = &a + &at;
    for i in 0..n {
        sym[(i, i)] += 2.0 * n as f64;
    }
    let rhs = fill(n, 1, 0x853C49E6748FEA9B);
    unsafe { *STATE.0.get() = Some(State { a, b, sym, rhs }) }
}

#[no_mangle]
pub extern "C" fn run_matmul() -> f64 {
    let s = state();
    let c = &s.a * &s.b;
    let n = c.nrows();
    c[(0, 0)] + c[(n - 1, n - 1)]
}

#[no_mangle]
pub extern "C" fn run_lu_solve() -> f64 {
    let s = state();
    let x = s.a.partial_piv_lu().solve(&s.rhs);
    x[(0, 0)]
}

#[no_mangle]
pub extern "C" fn run_qr() -> f64 {
    let s = state();
    s.a.qr().R()[(0, 0)]
}

#[no_mangle]
pub extern "C" fn run_svd() -> f64 {
    let s = state();
    s.a.svd().unwrap().S()[0]
}

#[no_mangle]
pub extern "C" fn run_sa_evd() -> f64 {
    let s = state();
    s.sym.self_adjoint_eigen(faer::Side::Lower).unwrap().S()[0]
}

#[no_mangle]
pub extern "C" fn run_gen_evd() -> f64 {
    let s = state();
    let e: alloc::vec::Vec<faer::c64> = s.a.eigenvalues().unwrap();
    e[0].re
}

// --- blocking-parameter tuning probes (Phase 3) -------------------------
// Factor-only entry points with caller-controlled blocking parameters, so
// tune.mjs can sweep them on wasm. Passing 0 for a parameter selects the
// library default, making the same export usable as the baseline.

use faer::dyn_stack::{MemBuffer, MemStack};
use faer::linalg::lu::partial_pivoting::factor as lu_pp;
use faer::linalg::qr::no_pivoting::factor as qr_np;
use faer::{Auto, Par, Spec};

#[no_mangle]
pub extern "C" fn run_lu_factor_tuned(recursion_threshold: usize, block_size: usize) -> f64 {
    let s = state();
    let n = s.a.nrows();
    let mut a = s.a.to_owned();
    let mut perm = alloc::vec![0usize; n];
    let mut perm_inv = alloc::vec![0usize; n];
    let dflt: lu_pp::PartialPivLuParams = Auto::<f64>::auto();
    let params = lu_pp::PartialPivLuParams {
        recursion_threshold: if recursion_threshold == 0 {
            dflt.recursion_threshold
        } else {
            recursion_threshold
        },
        block_size: if block_size == 0 { dflt.block_size } else { block_size },
        ..dflt
    };
    let mut mem = MemBuffer::new(lu_pp::lu_in_place_scratch::<usize, f64>(
        n,
        n,
        Par::Seq,
        Spec::new(params),
    ));
    lu_pp::lu_in_place(
        a.as_mut(),
        &mut perm,
        &mut perm_inv,
        Par::Seq,
        MemStack::new(&mut mem),
        Spec::new(params),
    );
    a[(0, 0)]
}

#[no_mangle]
pub extern "C" fn run_qr_factor_tuned(block_size: usize, blocking_threshold: usize) -> f64 {
    let s = state();
    let n = s.a.nrows();
    let mut a = s.a.to_owned();
    let bs = if block_size == 0 {
        qr_np::recommended_block_size::<f64>(n, n)
    } else {
        block_size.min(n)
    };
    let dflt: qr_np::QrParams = Auto::<f64>::auto();
    let params = qr_np::QrParams {
        blocking_threshold: if blocking_threshold == 0 {
            dflt.blocking_threshold
        } else {
            blocking_threshold
        },
        ..dflt
    };
    let mut h = Mat::<f64>::zeros(bs, n);
    let mut mem = MemBuffer::new(qr_np::qr_in_place_scratch::<f64>(
        n,
        n,
        bs,
        Par::Seq,
        Spec::new(params),
    ));
    qr_np::qr_in_place(
        a.as_mut(),
        h.as_mut(),
        Par::Seq,
        MemStack::new(&mut mem),
        Spec::new(params),
    );
    a[(0, 0)]
}

#[cfg(target_arch = "wasm32")]
mod wasm_shim {
    use core::alloc::{GlobalAlloc, Layout};

    // leak-only bump allocator over memory.grow (same as smoke-test): the
    // module needs zero imports. Benchmarks leak per run; node re-instantiates
    // per size to reset memory.
    struct Bump;

    extern "C" {
        static __heap_base: u8;
    }

    static mut OFFSET: usize = 0;

    #[inline]
    unsafe fn offset() -> usize {
        if OFFSET == 0 {
            OFFSET = &__heap_base as *const u8 as usize;
        }
        OFFSET
    }

    unsafe impl GlobalAlloc for Bump {
        unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
            let align = layout.align().max(16);
            let base = (offset() + align - 1) & !(align - 1);
            let end = base + layout.size();
            let cur_pages = core::arch::wasm32::memory_size(0);
            let need = (end + 0xffff) / 0x10000;
            if need > cur_pages
                && core::arch::wasm32::memory_grow(0, need - cur_pages) == usize::MAX
            {
                return core::ptr::null_mut();
            }
            OFFSET = end;
            base as *mut u8
        }

        unsafe fn dealloc(&self, _ptr: *mut u8, _layout: Layout) {}
    }

    #[global_allocator]
    static A: Bump = Bump;

    #[panic_handler]
    fn panic(_: &core::panic::PanicInfo) -> ! {
        core::arch::wasm32::unreachable()
    }
}
