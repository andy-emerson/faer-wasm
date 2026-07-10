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
    ac: Mat<faer::c64>,
    bc: Mat<faer::c64>,
    rhsc: Mat<faer::c64>,
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
    // c64 twins of a/b/rhs for the complex ops
    let re = fill(n, n, 0x2545F4914F6CDD1D);
    let im = fill(n, n, 0x94D049BB133111EB);
    let ac = Mat::from_fn(n, n, |i, j| faer::c64::new(re[(i, j)], im[(i, j)]));
    let re = fill(n, n, 0xBF58476D1CE4E5B9);
    let im = fill(n, n, 0x9E3779B97F4A7C15);
    let bc = Mat::from_fn(n, n, |i, j| faer::c64::new(re[(i, j)], im[(i, j)]));
    let re = fill(n, 1, 0xD6E8FEB86659FD93);
    let im = fill(n, 1, 0xCA5A826395121157);
    let rhsc = Mat::from_fn(n, 1, |i, j| faer::c64::new(re[(i, j)], im[(i, j)]));
    unsafe { *STATE.0.get() = Some(State { a, b, sym, rhs, ac, bc, rhsc }) }
}

#[no_mangle]
pub extern "C" fn run_schur() -> f64 {
    use faer::Par;
    let s = state();
    let sc = faer_schur::real::real_schur(s.a.as_ref(), Par::Seq).unwrap();
    let n = sc.t.nrows();
    sc.t[(0, 0)] + sc.z[(n - 1, n - 1)] + sc.w_re[0]
}

#[no_mangle]
pub extern "C" fn run_schur_c64() -> f64 {
    use faer::Par;
    let s = state();
    let sc = faer_schur::complex::complex_schur(s.ac.as_ref(), Par::Seq).unwrap();
    let n = sc.t.nrows();
    sc.t[(0, 0)].re + sc.z[(n - 1, n - 1)].im + sc.w[0].re
}

// Schur with an explicit blocking threshold (0 = library default), for the
// wasm crossover sweep — mirrors run_lu_factor_tuned / run_qr_factor_tuned.
// Isolated rank-k trailing update at LU shapes: C(m-k,m-k) -= A(m-k,k)*B(k,m-k)
#[no_mangle]
pub extern "C" fn run_rank_update(k: usize) -> f64 {
    use faer::linalg::matmul::matmul;
    use faer::{Accum, Par};
    let s = state();
    let n = s.a.nrows();
    let k = k.min(n / 2);
    let mut c = s.b.to_owned();
    let a21 = s.a.as_ref().submatrix(k, 0, n - k, k);
    let a12 = s.a.as_ref().submatrix(0, k, k, n - k);
    matmul(
        c.as_mut().submatrix_mut(k, k, n - k, n - k),
        Accum::Add,
        a21,
        a12,
        -1.0,
        Par::Seq,
    );
    c[(k, k)] + c[(n - 1, n - 1)]
}

// The wasm-shaped kernels (kernels/ crate): lean panels + faer-gemm bulk.
#[no_mangle]
pub extern "C" fn run_lu_factor_wk(nb: usize) -> f64 {
    let s = state();
    let n = s.a.nrows();
    let mut f = s.a.to_owned();
    let mut piv = alloc::vec![0usize; n];
    faer_wasm_kernels::lu::lu_factor_in_place(f.as_mut(), &mut piv, nb);
    f[(0, 0)] + f[(n - 1, n - 1)] + piv[n / 2] as f64
}

#[no_mangle]
pub extern "C" fn run_lu_factor_rec(crossover: usize) -> f64 {
    let s = state();
    let n = s.a.nrows();
    let mut f = s.a.to_owned();
    let mut piv = alloc::vec![0usize; n];
    faer_wasm_kernels::lu::lu_factor_recursive_in_place(f.as_mut(), &mut piv, crossover);
    f[(0, 0)] + f[(n - 1, n - 1)] + piv[n / 2] as f64
}

// Both recursive tunables exposed for the on-runner sweep (lu-tune.yml):
// crossover = 0 → size-dependent default, trsm_base = 0 → RECOMMENDED_TRSM_BASE.
#[no_mangle]
pub extern "C" fn run_lu_factor_rec_tuned(crossover: usize, trsm_base: usize) -> f64 {
    let s = state();
    let n = s.a.nrows();
    let mut f = s.a.to_owned();
    let mut piv = alloc::vec![0usize; n];
    faer_wasm_kernels::lu::lu_factor_recursive_in_place_tuned(f.as_mut(), &mut piv, crossover, trsm_base);
    f[(0, 0)] + f[(n - 1, n - 1)] + piv[n / 2] as f64
}

// Full Ax=b via the wasm-shaped kernels: flat LU factor + our forward/back
// substitution (kernels' lu_solve_in_place). Comparable to np.linalg.solve
// (factor + solve), unlike run_lu_solve which uses faer's default path.
#[no_mangle]
pub extern "C" fn run_lu_solve_wk() -> f64 {
    let s = state();
    let n = s.a.nrows();
    let mut f = s.a.to_owned();
    let mut piv = alloc::vec![0usize; n];
    faer_wasm_kernels::lu::lu_factor_recursive_in_place(f.as_mut(), &mut piv, 0);
    let mut x = alloc::vec![0.0f64; n];
    for i in 0..n {
        x[i] = s.rhs[(i, 0)];
    }
    faer_wasm_kernels::lu::lu_solve_in_place(f.as_ref(), &piv, &mut x);
    x[0] + x[n - 1]
}

#[no_mangle]
pub extern "C" fn run_schur_tuned(blocking_threshold: usize) -> f64 {
    use faer::dyn_stack::{MemBuffer, MemStack};
    use faer::{Auto, Col, Mat, Par};
    use faer_schur::real::{real_schur_in_place, real_schur_scratch, SchurParams};
    let s = state();
    let n = s.a.nrows();
    let mut params: SchurParams = Auto::<f64>::auto();
    if blocking_threshold != 0 {
        params.blocking_threshold = blocking_threshold;
    }
    let mut t = s.a.to_owned();
    let mut z = Mat::<f64>::zeros(n, n);
    let mut w_re = Col::<f64>::zeros(n);
    let mut w_im = Col::<f64>::zeros(n);
    let mut buf = MemBuffer::new(real_schur_scratch(n, Par::Seq, params));
    let stack = MemStack::new(&mut buf);
    real_schur_in_place(
        t.as_mut(),
        Some(z.as_mut()),
        w_re.as_mut(),
        w_im.as_mut(),
        Par::Seq,
        stack,
        params,
    )
    .unwrap();
    t[(0, 0)] + z[(n - 1, n - 1)] + w_re[0]
}

#[no_mangle]
pub extern "C" fn run_schur_c64_tuned(blocking_threshold: usize) -> f64 {
    use faer::dyn_stack::{MemBuffer, MemStack};
    use faer::{Auto, Col, Mat, Par};
    use faer_schur::complex::{complex_schur_in_place, complex_schur_scratch, SchurParams};
    let s = state();
    let n = s.ac.nrows();
    let mut params: SchurParams = Auto::<faer::c64>::auto();
    if blocking_threshold != 0 {
        params.blocking_threshold = blocking_threshold;
    }
    let mut t = s.ac.to_owned();
    let mut z = Mat::<faer::c64>::zeros(n, n);
    let mut w = Col::<faer::c64>::zeros(n);
    let mut buf = MemBuffer::new(complex_schur_scratch(n, Par::Seq, params));
    let stack = MemStack::new(&mut buf);
    complex_schur_in_place(t.as_mut(), Some(z.as_mut()), w.as_mut(), Par::Seq, stack, params).unwrap();
    t[(0, 0)].re + z[(n - 1, n - 1)].im + w[0].re
}

#[no_mangle]
pub extern "C" fn run_matmul_c64() -> f64 {
    let s = state();
    let c = &s.ac * &s.bc;
    let n = c.nrows();
    c[(0, 0)].re + c[(n - 1, n - 1)].im
}

#[no_mangle]
pub extern "C" fn run_lu_solve_c64() -> f64 {
    use faer::prelude::*;
    let s = state();
    let x = s.ac.partial_piv_lu().solve(&s.rhsc);
    let n = x.nrows();
    x[(0, 0)].re + x[(n - 1, 0)].im
}

#[no_mangle]
pub extern "C" fn run_qr_c64() -> f64 {
    let s = state();
    let qr = s.ac.qr();
    let r = qr.R();
    let n = r.nrows();
    r[(0, 0)].re + r[(n - 1, n - 1)].im
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

// --- SVD roofline profiling (architect direction 2026-07-10: locate the
// machine ceiling and where faer's SVD time actually goes, before deciding
// tune-bidiag vs build-Jacobi). ---

// faer's bidiagonalization ALONE (the reduction), mirroring svd/mod.rs's
// internal setup. Timed against full run_svd to get the reduction/solve
// split on the runner.
#[no_mangle]
pub extern "C" fn run_bidiag_only() -> f64 {
    use faer::linalg::svd::bidiag::{bidiag_in_place, bidiag_in_place_scratch, BidiagParams};
    let s = state();
    let n = s.a.nrows();
    let bs = qr_np::recommended_block_size::<f64>(n, n);
    let mut bid = s.a.to_owned();
    let mut hl = Mat::<f64>::zeros(bs, n);
    let mut hr = Mat::<f64>::zeros(bs, n - 1);
    let params: BidiagParams = Auto::<f64>::auto();
    let mut mem = MemBuffer::new(bidiag_in_place_scratch::<f64>(n, n, Par::Seq, Spec::new(params)));
    bidiag_in_place(
        bid.as_mut(),
        hl.as_mut(),
        hr.as_mut(),
        Par::Seq,
        MemStack::new(&mut mem),
        Spec::new(params),
    );
    bid[(0, 0)] + bid[(n - 1, n - 1)]
}

// STREAM triad over the packed n*n buffers (a += 1.5*b): reads a, reads b,
// writes a = 24 bytes/elem, memory-bound. GB/s = 24*n*n / time is the
// achievable-bandwidth roofline anchor at the SVD working-set size.
#[no_mangle]
pub extern "C" fn run_stream() -> f64 {
    let s = state();
    let n = s.a.nrows();
    let len = n * n; // faer Mat is packed column-major (col_stride == nrows)
    unsafe {
        let ap = s.a.as_ptr() as *mut f64;
        let bp = s.b.as_ptr();
        let mut i = 0usize;
        while i < len {
            *ap.add(i) += 1.5 * *bp.add(i);
            i += 1;
        }
        *ap
    }
}

// One-sided Jacobi SVD probe (kernels/src/svd.rs): the bidiagonalization-
// avoiding full SVD. run_svd_jacobi times it; run_svd_jacobi_sweeps returns
// the sweep count so the profiler can report it per size.
#[no_mangle]
pub extern "C" fn run_svd_jacobi() -> f64 {
    let s = state();
    let n = s.a.nrows();
    let mut u = s.a.to_owned();
    let mut v = Mat::<f64>::zeros(n, n);
    let mut sv = alloc::vec![0.0f64; n];
    faer_wasm_kernels::svd::jacobi_svd_in_place(u.as_mut(), v.as_mut(), &mut sv, 60, 1e-13);
    sv[0] + sv[n - 1]
}

#[no_mangle]
pub extern "C" fn run_svd_jacobi_sweeps() -> f64 {
    let s = state();
    let n = s.a.nrows();
    let mut u = s.a.to_owned();
    let mut v = Mat::<f64>::zeros(n, n);
    let mut sv = alloc::vec![0.0f64; n];
    faer_wasm_kernels::svd::jacobi_svd_in_place(u.as_mut(), v.as_mut(), &mut sv, 60, 1e-13) as f64
}

// Standalone GEMV y = A*x — the memory-bound kernel bidiagonalization spends
// ~half its flops in. GB/s = 8*n*n / time; compare to run_stream to see if a
// raw matvec already saturates bandwidth.
#[no_mangle]
pub extern "C" fn run_gemv() -> f64 {
    let s = state();
    let y = &s.a * &s.rhs;
    y[(0, 0)]
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

// The wasm-shaped unblocked Householder QR (kernels/src/qr.rs): fused
// dlarfg + dlarf in flat simd128, no compact-WY/T-matrix/gemm.
#[no_mangle]
pub extern "C" fn run_qr_factor_wk() -> f64 {
    let s = state();
    let n = s.a.nrows();
    let mut f = s.a.to_owned();
    let mut tau = alloc::vec![0.0f64; n];
    faer_wasm_kernels::qr::qr_factor_in_place(f.as_mut(), &mut tau);
    f[(0, 0)] + f[(n - 1, n - 1)] + tau[n / 2]
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
