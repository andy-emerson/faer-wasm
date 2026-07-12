#![cfg_attr(target_arch = "wasm32", no_std)]

extern crate alloc;

use faer::Mat;

// stage "matmul": just matrix multiply
// stage "lu": + partial-pivot LU solve
// stage "full": + QR, SVD, self-adjoint EVD, general EVD,
//               Schur + eigenvalue reordering (faer-schur companion crate)

#[no_mangle]
pub extern "C" fn matmul_trace() -> f64 {
    let a = Mat::from_fn(3, 3, |i, j| (i + 2 * j) as f64 + 1.0);
    let b = Mat::from_fn(3, 3, |i, j| (3 * i + j) as f64 - 2.0);
    let c = &a * &b;
    c[(0, 0)] + c[(1, 1)] + c[(2, 2)]
}

#[cfg(any(feature = "lu", feature = "full"))]
#[no_mangle]
pub extern "C" fn lu_solve_sum() -> f64 {
    use faer::prelude::*;
    let a = faer::mat![[4.0, 3.0, 2.0], [2.0, 5.0, 1.0], [1.0, 1.0, 3.0f64]];
    let rhs = faer::mat![[1.0], [2.0], [3.0f64]];
    let lu = a.partial_piv_lu();
    let x = lu.solve(&rhs);
    x[(0, 0)] + x[(1, 0)] + x[(2, 0)]
}

#[cfg(feature = "full")]
#[no_mangle]
pub extern "C" fn qr_svd_evd_probe() -> f64 {
    let a = faer::mat![[4.0, 1.0, 0.5], [1.0, 3.0, 0.2], [0.5, 0.2, 2.0f64]];
    let qr = a.qr();
    let r00 = qr.R()[(0, 0)];
    let svd = a.svd().unwrap();
    let s0 = svd.S()[0];
    let evd = a.self_adjoint_eigen(faer::Side::Lower).unwrap();
    let e0 = evd.S()[0];
    // general (non-hermitian) eigendecomposition
    let g = faer::mat![[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0f64]];
    let ge: alloc::vec::Vec<faer::c64> = g.eigenvalues().unwrap();
    r00 + s0 + e0 + ge[0].re
}

// Unlike the 3×3 probes above, the Schur pipeline at 8×8 is NOT bit-identical
// across targets: pulp reduction order differs with SIMD width (native AVX vs
// wasm simd128), and relaxed-SIMD FMA can steer the QR iteration down a
// different path, landing the (canonically unordered) eigenvalues in a
// different diagonal order. So this probe scores integer-valued correctness
// properties — backward error, orthogonality, exact structure, reorder
// invariants — whose exactness IS the cross-target gate. Tolerances sit ~2-3
// orders of magnitude above the observed errors (~3e-15) and the eigenvalue
// selection margin (min |Re λ| = 0.023) is far above ulp noise, so rounding
// differences can't flip the score. Reference value: 11 (6 checks + m = 5).
#[cfg(feature = "full")]
#[no_mangle]
pub extern "C" fn schur_probe() -> f64 {
    use faer::Par;
    use faer_schur::real::{real_schur, real_schur_select};

    // deterministic fill (same LCG as bench/)
    let mut vals = [0.0f64; 64];
    let mut s = 0x9E3779B97F4A7C15u64;
    for v in vals.iter_mut() {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        *v = ((s >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0;
    }

    let n = 8usize;
    let a = Mat::from_fn(n, n, |i, j| vals[n * i + j]);
    let mut score = 0.0f64;

    let sc = real_schur(a.as_ref(), Par::Seq).unwrap();
    let max_abs = |m: &Mat<f64>, target: &dyn Fn(usize, usize) -> f64| {
        let mut e = 0.0f64;
        for j in 0..m.ncols() {
            for i in 0..m.nrows() {
                let d = m[(i, j)] - target(i, j);
                e = if d.abs() > e { d.abs() } else { e };
            }
        }
        e
    };
    // backward error ||A - Z T Zᵀ||, orthogonality ||ZᵀZ - I||
    let recon = &sc.z * &sc.t * sc.z.transpose();
    if max_abs(&recon, &|i, j| a[(i, j)]) < 1e-12 {
        score += 1.0;
    }
    let ztz = sc.z.transpose() * &sc.z;
    if max_abs(&ztz, &|i, j| if i == j { 1.0 } else { 0.0 }) < 1e-13 {
        score += 1.0;
    }
    // quasi-triangular structure: exact zeros below the subdiagonal
    let mut structural = true;
    for j in 0..n {
        for i in j + 2..n {
            structural &= sc.t[(i, j)] == 0.0;
        }
    }
    if structural {
        score += 1.0;
    }
    // trace invariant (order-independent): sum of eigenvalues == trace
    let tr: f64 = (0..n).map(|k| a[(k, k)]).sum();
    let ws: f64 = (0..n).map(|k| sc.w_re[k]).sum();
    if (tr - ws).abs() < 1e-12 {
        score += 1.0;
    }
    // reorder the right-half-plane eigenvalues to the top; for this seed the
    // smallest |Re λ| is far above ulp noise, so the selection (and m) is
    // stable across targets
    let mut t = sc.t;
    let mut z = sc.z;
    let select: [bool; 8] = core::array::from_fn(|k| sc.w_re[k] > 0.0);
    let m = real_schur_select(t.as_mut(), Some(z.as_mut()), &select).unwrap();
    score += m as f64;
    let recon = &z * &t * z.transpose();
    if max_abs(&recon, &|i, j| a[(i, j)]) < 1e-12 {
        score += 1.0;
    }
    // leading m rows carry the selected (Re λ > 0) eigenvalues
    let mut split_ok = true;
    let mut k = 0usize;
    while k < n {
        let re = if k + 1 < n && t[(k + 1, k)] != 0.0 {
            let v = 0.5 * (t[(k, k)] + t[(k + 1, k + 1)]);
            k += 2;
            v
        } else {
            let v = t[(k, k)];
            k += 1;
            v
        };
        split_ok &= (re > 0.0) == (k <= m);
    }
    if split_ok {
        score += 1.0;
    }

    score
}

// Complex (c64) Schur, scored the same way: 3 property checks, reference
// value 3. Kept as a SEPARATE export because it doubles as the regression
// guard for patches/pulp/0003: pulp's wasm RelaxedSimd backend shipped its
// complex mul_add_e/mul_e kernels with transposed FMA arguments (NEON
// accumulator-first order passed to accumulator-last relaxed_madd), which
// made every c64 computation past matmul grossly wrong under
// `+relaxed-simd` — including faer's own `.eigenvalues()` on complex input.
// Found + root-caused + fixed 2026-07-08; this probe failing on a re-pin
// means the pulp patch was dropped while upstream is still broken.
#[cfg(feature = "full")]
#[no_mangle]
pub extern "C" fn schur_probe_cplx() -> f64 {
    use faer::Par;
    use faer_schur::complex::complex_schur;

    let mut vals = [0.0f64; 64];
    let mut s = 0x9E3779B97F4A7C15u64;
    for v in vals.iter_mut() {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        *v = ((s >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0;
    }

    let nc = 4usize;
    let ac = Mat::from_fn(nc, nc, |i, j| faer::c64::new(vals[nc * i + j], vals[32 + nc * i + j]));
    let cs = complex_schur(ac.as_ref(), Par::Seq).unwrap();
    let recon = &cs.z * &cs.t * cs.z.adjoint();
    let uzz = cs.z.adjoint() * &cs.z;
    let mut cerr = 0.0f64;
    let mut uerr = 0.0f64;
    let mut ctri = true;
    for j in 0..nc {
        for i in 0..nc {
            let d = recon[(i, j)] - ac[(i, j)];
            let e = d.re * d.re + d.im * d.im;
            cerr = if e > cerr { e } else { cerr };
            let id = if i == j { 1.0 } else { 0.0 };
            let d = uzz[(i, j)] - faer::c64::new(id, 0.0);
            let e = d.re * d.re + d.im * d.im;
            uerr = if e > uerr { e } else { uerr };
            if i > j {
                ctri &= cs.t[(i, j)] == faer::c64::new(0.0, 0.0);
            }
        }
    }
    let mut score = 0.0f64;
    if cerr < 1e-24 {
        score += 1.0;
    }
    if uerr < 1e-26 {
        score += 1.0;
    }
    if ctri {
        score += 1.0;
    }
    score
}

#[cfg(feature = "full")]
mod dense_probes;

#[cfg(feature = "full")]
pub fn dense_probes_f64_errors(n: usize) -> [f64; 13] {
    dense_probes::f64_errors(n)
}

#[cfg(feature = "full")]
pub fn dense_probes_c64_errors(n: usize) -> [f64; 12] {
    dense_probes::c64_errors(n)
}

// Foundation gate: the "simple" decompositions (LU/QR/LLT/SVD/EVD) property-
// checked at n=33 (SIMD tail lanes) and n=96 (blocked paths), f64 and c64.
// Integer scores, exact across targets; see src/dense_probes.rs.
#[cfg(feature = "full")]
#[no_mangle]
pub extern "C" fn dense_f64_probe() -> f64 {
    dense_probes::f64_score()
}

#[cfg(feature = "full")]
#[no_mangle]
pub extern "C" fn dense_c64_probe() -> f64 {
    dense_probes::c64_score()
}

#[cfg(target_arch = "wasm32")]
mod wasm_shim {
    use core::alloc::{GlobalAlloc, Layout};

    // NB deliberately duplicated in bench/src/lib.rs — this crate is the
    // zero-import consumer example and must stay self-contained; keep the
    // two shims in sync (2026-07-12 sweep note).
    // LIFO-rewind bump allocator over memory.grow, so the module needs no
    // imports at all. Upgraded from leak-only 2026-07-11: faer's c64 matmul
    // allocates per-call temporaries through the global allocator (measured
    // 15.4 GB cumulative inside ONE c64 multishift at n=600, peak live
    // ~19 MB), so a leak-only bump OOMs on large-n complex work; rewinding
    // on top-of-stack frees reclaims those nested temporaries. See
    // docs/wasm.md §2.
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
            if need > cur_pages {
                if core::arch::wasm32::memory_grow(0, need - cur_pages) == usize::MAX {
                    return core::ptr::null_mut();
                }
            }
            OFFSET = end;
            base as *mut u8
        }
        unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
            // LIFO rewind: reclaim iff this is the top allocation
            if ptr as usize + layout.size() == OFFSET {
                OFFSET = ptr as usize;
            }
        }
    }

    #[global_allocator]
    static A: Bump = Bump;

    #[panic_handler]
    fn panic(_: &core::panic::PanicInfo) -> ! {
        core::arch::wasm32::unreachable()
    }
}
