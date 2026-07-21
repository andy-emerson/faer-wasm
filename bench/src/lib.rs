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
    // f32 twins (same values as a/b/rhs, cast) for the f32/c32 phase rows
    a32: Mat<f32>,
    b32: Mat<f32>,
    rhs32: Mat<f32>,
    // c32 twins (ac/bc cast) for the complex market races
    ac32: Mat<faer::c32>,
    bc32: Mat<faer::c32>,
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
    let a32 = Mat::from_fn(n, n, |i, j| a[(i, j)] as f32);
    let b32 = Mat::from_fn(n, n, |i, j| b[(i, j)] as f32);
    let rhs32 = Mat::from_fn(n, 1, |i, j| rhs[(i, j)] as f32);
    let ac32 = Mat::from_fn(n, n, |i, j| faer::c32::new(ac[(i, j)].re as f32, ac[(i, j)].im as f32));
    let bc32 = Mat::from_fn(n, n, |i, j| faer::c32::new(bc[(i, j)].re as f32, bc[(i, j)].im as f32));
    unsafe {
        *STATE.0.get() =
            Some(State { a, b, sym, rhs, ac, bc, rhsc, a32, b32, rhs32, ac32, bc32 })
    }
}

/// faer's blocked complex gemm rows for the market race against the
/// blas layer's zgemm/cgemm (close-out campaign): matmul() into a
/// per-call zeroed Mat, mirroring run_blas_ab(4,0)'s shape. variant:
/// 0 = c64, 1 = c32.
#[no_mangle]
pub extern "C" fn run_gemm_faer_cplx(variant: usize) -> f64 {
    use faer::linalg::matmul::matmul;
    use faer::Accum;
    let s = state();
    match variant {
        0 => {
            let n = s.ac.nrows();
            let mut c = Mat::<faer::c64>::zeros(n, n);
            matmul(
                c.as_mut(),
                Accum::Replace,
                s.ac.as_ref(),
                s.bc.as_ref(),
                faer::c64::new(1.0, 0.0),
                Par::Seq,
            );
            c[(0, 0)].re + c[(n - 1, n - 1)].im
        }
        1 => {
            let n = s.ac32.nrows();
            let mut c = Mat::<faer::c32>::zeros(n, n);
            matmul(
                c.as_mut(),
                Accum::Replace,
                s.ac32.as_ref(),
                s.bc32.as_ref(),
                faer::c32::new(1.0, 0.0),
                Par::Seq,
            );
            (c[(0, 0)].re + c[(n - 1, n - 1)].im) as f64
        }
        _ => f64::NAN,
    }
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

// SVD with a tunable divide-and-conquer recursion_threshold. Deep research
// (2026-07-10) found faer's default 128 forces bidiagonal blocks up to 127
// through the SCALAR qr_algorithm (Givens vector accumulation, no SIMD),
// where LAPACK dbdsdc uses ~25-leaves + gemm merges — the root cause of
// faer SVD losing to scipy at small n. Sweep this to test the fix.
// recursion_threshold = 0 -> faer default (128).
#[no_mangle]
pub extern "C" fn run_svd_tuned(recursion_threshold: usize) -> f64 {
    use faer::diag::Diag;
    use faer::linalg::svd::{svd, svd_scratch, ComputeSvdVectors, SvdParams};
    let s = state();
    let n = s.a.nrows();
    let mut sv = Diag::<f64>::zeros(n);
    let mut u = Mat::<f64>::zeros(n, n);
    let mut v = Mat::<f64>::zeros(n, n);
    let mut params: SvdParams = Auto::<f64>::auto();
    if recursion_threshold != 0 {
        params.recursion_threshold = recursion_threshold;
    }
    let mut mem = MemBuffer::new(svd_scratch::<f64>(
        n,
        n,
        ComputeSvdVectors::Full,
        ComputeSvdVectors::Full,
        Par::Seq,
        Spec::new(params),
    ));
    svd(
        s.a.as_ref(),
        sv.as_mut(),
        Some(u.as_mut()),
        Some(v.as_mut()),
        Par::Seq,
        MemStack::new(&mut mem),
        Spec::new(params),
    )
    .unwrap();
    sv.column_vector()[0] + u[(0, 0)] + v[(n - 1, n - 1)]
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

// --- EVD (eigvals) phase profiling + parameter probes (architect direction
// 2026-07-10: deep-research the eigen flank; faer eigvals is 0.3-0.4x scipy,
// the worst ratio in the suite). Known prior: faer's blocked multishift/AED
// Schur path measured 2-13x SLOWER than its own scalar lahqr on wasm
// (2026-07-09, schur/src/real.rs recommended_params) while the same
// algorithm is reference LAPACK's fast path -- these probes locate where
// the cycles go: Hessenberg vs QR-iteration, per-sweep cost vs sweep count,
// and whether LAPACK's iparmq-style parameters close the gap. ---

use faer::linalg::evd::hessenberg;
use faer::linalg::evd::schur::{real_schur, SchurParams};

extern "C" fn shift_count_lapack(_dim: usize, active: usize) -> usize {
    let nh = Ord::max(active, 4);
    let ns = if nh < 30 {
        2
    } else if nh < 60 {
        4
    } else if nh < 150 {
        10
    } else if nh < 590 {
        Ord::max(10, nh / nh.ilog2() as usize)
    } else if nh < 3000 {
        64
    } else if nh < 6000 {
        128
    } else {
        256
    };
    Ord::max(2, ns - ns % 2)
}

// LAPACK iparmq-style AED window (ISPEC=13: NW = NS, 3*NS/2 above n=500).
extern "C" fn deflation_window_lapack(dim: usize, active: usize) -> usize {
    let ns = shift_count_lapack(dim, active);
    if active <= 500 {
        ns
    } else {
        3 * ns / 2
    }
}

fn evd_schur_params(blocking: usize, nibble: usize, profile: usize) -> SchurParams {
    let mut params: SchurParams = Auto::<f64>::auto();
    if blocking != 0 {
        params.blocking_threshold = blocking;
    }
    if nibble != 0 {
        params.nibble_threshold = nibble;
    }
    if profile == 1 {
        params.recommended_shift_count = shift_count_lapack;
        params.recommended_deflation_window = deflation_window_lapack;
    }
    params
}

// Eigenvalues-only pipeline (Hessenberg -> multishift QR, want_t=false,
// no Z: LAPACK JOB='E' equivalent, same shape as faer's eigenvalues())
// with caller-controlled SchurParams. 0 = library default per knob;
// blocking=1<<30 pins every size to the scalar lahqr kernel (the
// faer-schur wasm recommendation). Returns an eigenvalue probe.
fn eigvals_tuned_imp(blocking: usize, nibble: usize, profile: usize) -> (f64, usize, usize) {
    let s = state();
    let n = s.a.nrows();
    let params = evd_schur_params(blocking, nibble, profile);
    let bs = qr_np::recommended_block_size::<f64>(n - 1, n - 1);
    let mut h = s.a.to_owned();
    let mut hh = Mat::<f64>::zeros(bs, n - 1);
    let mut w_re = faer::Col::<f64>::zeros(n);
    let mut w_im = faer::Col::<f64>::zeros(n);
    let scratch = faer::dyn_stack::StackReq::any_of(&[
        hessenberg::hessenberg_in_place_scratch::<f64>(n, bs, Par::Seq, Default::default()),
        faer::linalg::evd::schur::multishift_qr_scratch::<f64>(n, n, false, false, Par::Seq, params),
    ]);
    let mut mem = MemBuffer::new(scratch);
    let stack = MemStack::new(&mut mem);
    hessenberg::hessenberg_in_place(h.as_mut(), hh.as_mut(), Par::Seq, stack, Default::default());
    for j in 0..n {
        for i in j + 2..n {
            h[(i, j)] = 0.0;
        }
    }
    let (info, count_aed, count_sweep) = real_schur::multishift_qr::<f64>(
        false,
        h.as_mut(),
        None,
        w_re.as_mut(),
        w_im.as_mut(),
        0,
        n,
        Par::Seq,
        stack,
        params,
    );
    assert!(info == 0, "eigvals did not converge");
    (w_re[0] + w_im[n - 1], count_aed, count_sweep)
}

#[no_mangle]
pub extern "C" fn run_eigvals_tuned(blocking: usize, nibble: usize, profile: usize) -> f64 {
    eigvals_tuned_imp(blocking, nibble, profile).0
}

// The full fix-1+2+3 pipeline: kernel Hessenberg (fix 2) + hand double-shift
// hqr iteration (fix 3) below the measured 480 crossover (fix 1), repaired
// faer multishift above it.
#[no_mangle]
pub extern "C" fn run_eigvals_k3() -> f64 {
    let s = state();
    let n = s.a.nrows();
    let mut h = s.a.to_owned();
    let mut tau = alloc::vec![0.0f64; n.saturating_sub(2).max(1)];
    let mut work = alloc::vec![0.0f64; n];
    faer_wasm_kernels::hessenberg::hessenberg_factor_in_place(h.as_mut(), &mut tau, &mut work);
    for j in 0..n {
        for i in j + 2..n {
            h[(i, j)] = 0.0;
        }
    }
    if n < 480 {
        let mut w_re = alloc::vec![0.0f64; n];
        let mut w_im = alloc::vec![0.0f64; n];
        let info =
            faer_wasm_kernels::schur_small::hqr_eigvals_in_place(h.as_mut(), &mut w_re, &mut w_im);
        assert!(info == 0, "eigvals_k3 (hqr) did not converge");
        w_re[0] + w_im[n - 1]
    } else {
        let params = faer_schur::real::recommended_eigenvalues_params(n);
        let mut w_re = faer::Col::<f64>::zeros(n);
        let mut w_im = faer::Col::<f64>::zeros(n);
        let mut mem = MemBuffer::new(faer::linalg::evd::schur::multishift_qr_scratch::<f64>(
            n,
            n,
            false,
            false,
            Par::Seq,
            params,
        ));
        let (info, _, _) = real_schur::multishift_qr::<f64>(
            false,
            h.as_mut(),
            None,
            w_re.as_mut(),
            w_im.as_mut(),
            0,
            n,
            Par::Seq,
            MemStack::new(&mut mem),
            params,
        );
        assert!(info == 0, "eigvals_k3 (multishift) did not converge");
        w_re[0] + w_im[n - 1]
    }
}

// ---- Schur campaign (2026-07-11) / eigenvector campaign (2026-07-12):
// the full-Schur and full-eig kernel pipelines. Kernel Hessenberg (fix 2)
// + backward-accumulated Q formation + hqr want_t+Z below the 480
// crossover; above it, kernel Hessenberg front-end + faer's repaired
// multishift (want_t, Z seeded with the accumulated Q) — the multishift
// path also inherits the accumulated-U gemm batching the research
// identified (docs/research-schur-wasm-2026-07.md). One generic body
// serves the f64 and f32 exports (they began as hand-copies; folded in
// the 2026-07-12 sweep), and a c64 body serves the complex twins.

/// A → (T, Z): the real Schur pipeline at either precision. `z` is
/// allocated either way but only formed when `want_z` (matching what the
/// exports always did).
fn schur_pipeline<T: faer_wasm_kernels::scalar::WasmScalarRefl>(
    a: faer::MatRef<'_, T>,
    want_t: bool,
    want_z: bool,
) -> (faer::Mat<T>, faer::Mat<T>) {
    let n = a.nrows();
    let mut h = a.to_owned();
    let mut tau = alloc::vec![T::ZERO; n.saturating_sub(2).max(1)];
    let mut work = alloc::vec![T::ZERO; n];
    faer_wasm_kernels::hessenberg::hessenberg_factor_in_place(h.as_mut(), &mut tau, &mut work);
    let mut z = faer::Mat::<T>::zeros(n, n);
    if want_z {
        faer_wasm_kernels::hessenberg::hessenberg_form_q(h.as_ref(), &tau, z.as_mut());
    }
    for j in 0..n {
        for i in j + 2..n {
            h[(i, j)] = T::ZERO;
        }
    }
    if n < 480 {
        let mut w_re = alloc::vec![T::ZERO; n];
        let mut w_im = alloc::vec![T::ZERO; n];
        let info = faer_wasm_kernels::schur_small::hqr_schur_in_place(
            h.as_mut(),
            if want_z { Some(z.as_mut()) } else { None },
            &mut w_re,
            &mut w_im,
            want_t,
        );
        assert!(info == 0, "schur pipeline (hqr) did not converge");
    } else {
        let params = faer_schur::real::recommended_params(n);
        let mut w_re = faer::Col::<T>::zeros(n);
        let mut w_im = faer::Col::<T>::zeros(n);
        let mut mem = MemBuffer::new(faer::linalg::evd::schur::multishift_qr_scratch::<T>(
            n,
            n,
            want_t,
            want_z,
            Par::Seq,
            params,
        ));
        let (info, _, _) = real_schur::multishift_qr::<T>(
            want_t,
            h.as_mut(),
            if want_z { Some(z.as_mut()) } else { None },
            w_re.as_mut(),
            w_im.as_mut(),
            0,
            n,
            Par::Seq,
            MemStack::new(&mut mem),
            params,
        );
        assert!(info == 0, "schur pipeline (multishift) did not converge");
        // faer's blocked path leaves workspace junk below the subdiagonal
        // (faer-schur zeroes it too); include the cleanup in the timed cost
        if want_t {
            for j in 0..n {
                for i in j + 2..n {
                    h[(i, j)] = T::ZERO;
                }
            }
        }
    }
    (h, z)
}

/// A → (T, Z): the c64 Schur pipeline. Complex Hessenberg +
/// backward-accumulated Q + single-shift chqr below the 480 crossover
/// (measured identical to the real crossover, run 29134291933,
/// provisional under the tuning freeze); faer's complex multishift above.
fn schur_c64_pipeline(
    a: faer::MatRef<'_, faer::c64>,
    want_t: bool,
    want_z: bool,
) -> (faer::Mat<faer::c64>, faer::Mat<faer::c64>) {
    use faer::c64;
    let n = a.nrows();
    let mut h = a.to_owned();
    let mut tau = alloc::vec![c64::new(0.0, 0.0); n.saturating_sub(2).max(1)];
    let mut work = alloc::vec![c64::new(0.0, 0.0); n];
    faer_wasm_kernels::hessenberg_cplx::hessenberg_cplx_factor_in_place(
        h.as_mut(),
        &mut tau,
        &mut work,
    );
    let mut z = faer::Mat::<c64>::zeros(n, n);
    if want_z {
        faer_wasm_kernels::hessenberg_cplx::hessenberg_cplx_form_q(h.as_ref(), &tau, z.as_mut());
    }
    for j in 0..n {
        for i in j + 2..n {
            h[(i, j)] = c64::new(0.0, 0.0);
        }
    }
    if n < 480 {
        let mut w = alloc::vec![c64::new(0.0, 0.0); n];
        let info = faer_wasm_kernels::schur_small_cplx::chqr_schur_in_place(
            h.as_mut(),
            if want_z { Some(z.as_mut()) } else { None },
            &mut w,
            want_t,
        );
        assert!(info == 0, "c64 schur pipeline (chqr) did not converge");
    } else {
        let params = faer_schur::complex::recommended_params(n);
        let mut w = faer::Col::<c64>::zeros(n);
        let mut mem = MemBuffer::new(faer::linalg::evd::schur::multishift_qr_scratch::<c64>(
            n,
            n,
            want_t,
            want_z,
            Par::Seq,
            params,
        ));
        let (info, _, _) = faer::linalg::evd::schur::complex_schur::multishift_qr::<c64>(
            want_t,
            h.as_mut(),
            if want_z { Some(z.as_mut()) } else { None },
            w.as_mut(),
            0,
            n,
            Par::Seq,
            MemStack::new(&mut mem),
            params,
        );
        assert!(info == 0, "c64 schur pipeline (multishift) did not converge");
        if want_t {
            for j in 0..n {
                for i in j + 2..n {
                    h[(i, j)] = c64::new(0.0, 0.0);
                }
            }
        }
    }
    (h, z)
}

/// The shipping full-Schur pipeline: T + Z.
#[no_mangle]
pub extern "C" fn run_schur_k() -> f64 {
    let s = state();
    let (h, z) = schur_pipeline::<f64>(s.a.as_ref(), true, true);
    let n = h.nrows();
    h[(0, 0)] + z[(n - 1, n - 1)]
}

/// Instrumentation toggles for the eigvals→Schur cost split (research open
/// question 1): mode 0 = eigvals-only (baseline), 1 = +want_t, 2 = +Z
/// (with Q formation), 3 = both (== run_schur_k).
#[no_mangle]
pub extern "C" fn run_schur_k_mode(mode: usize) -> f64 {
    let s = state();
    let want_z = mode & 2 != 0;
    let (h, z) = schur_pipeline::<f64>(s.a.as_ref(), mode & 1 != 0, want_z);
    let n = h.nrows();
    h[(0, 0)] + if want_z { z[(n - 1, n - 1)] } else { 0.0 }
}

/// f32 full-Schur pipeline (coverage rule: every number type the kernels
/// support gets a row). SchurParams is type-free, so faer-schur's wasm
/// routing applies as-is.
#[no_mangle]
pub extern "C" fn run_schur_k_f32() -> f64 {
    let s = state();
    let (h, z) = schur_pipeline::<f32>(s.a32.as_ref(), true, true);
    let n = h.nrows();
    (h[(0, 0)] + z[(n - 1, n - 1)]) as f64
}

/// The c64 full-Schur pipeline: T + Z.
#[no_mangle]
pub extern "C" fn run_schur_c64_k() -> f64 {
    let s = state();
    let (h, z) = schur_c64_pipeline(s.ac.as_ref(), true, true);
    let n = h.nrows();
    h[(0, 0)].re + z[(n - 1, n - 1)].im
}

// ---- eigenvector campaign (2026-07-12): full eig = the Schur pipeline +
// the trevc-shaped eigenvector kernels (back-substitution on T,
// triangular-matmul back-transform through Z). Scoreboard opponent:
// np.linalg.eig.

#[no_mangle]
pub extern "C" fn run_eig_k() -> f64 {
    let s = state();
    let (h, z) = schur_pipeline::<f64>(s.a.as_ref(), true, true);
    let n = h.nrows();
    let mut v = faer::Mat::<f64>::zeros(n, n);
    faer_wasm_kernels::eigvec::trevc_in_place(h.as_ref(), z.as_ref(), v.as_mut());
    v[(0, 0)] + v[(n - 1, n - 1)] + h[(0, 0)]
}

/// f32 twin of run_eig_k.
#[no_mangle]
pub extern "C" fn run_eig_k_f32() -> f64 {
    let s = state();
    let (h, z) = schur_pipeline::<f32>(s.a32.as_ref(), true, true);
    let n = h.nrows();
    let mut v = faer::Mat::<f32>::zeros(n, n);
    faer_wasm_kernels::eigvec::trevc_in_place(h.as_ref(), z.as_ref(), v.as_mut());
    (v[(0, 0)] + v[(n - 1, n - 1)] + h[(0, 0)]) as f64
}

/// c64 twin of run_eig_k (complex Schur pipeline + ctrevc).
#[no_mangle]
pub extern "C" fn run_eig_c64_k() -> f64 {
    let s = state();
    let (h, z) = schur_c64_pipeline(s.ac.as_ref(), true, true);
    let n = h.nrows();
    let mut v = faer::Mat::<faer::c64>::zeros(n, n);
    faer_wasm_kernels::eigvec_cplx::ctrevc_in_place(h.as_ref(), z.as_ref(), v.as_mut());
    v[(0, 0)].re + v[(n - 1, n - 1)].im + h[(0, 0)].re
}

/// faer's own full eigendecomposition (values + vectors) — the baseline
/// our eig_k pipeline is measured against, same role as run_schur for
/// the Schur rows.
#[no_mangle]
pub extern "C" fn run_eig() -> f64 {
    let s = state();
    let n = s.a.nrows();
    let e = s.a.eigen().unwrap();
    let u = e.U();
    u[(0, 0)].re + u[(n - 1, n - 1)].im
}

// ---- f32 rows (f32/c32 phase, architect direction 2026-07-11): the four
// headliners in f32 -- ~2x mechanism pair on wasm SIMD128 (4 lanes for
// compute-bound, half traffic for bandwidth-bound). matmul rides faer's
// generic f32 path; the rest ride the now-generic kernels.
#[no_mangle]
pub extern "C" fn run_matmul_f32() -> f64 {
    let s = state();
    let c = &s.a32 * &s.b32;
    let n = c.nrows();
    (c[(0, 0)] + c[(n - 1, n - 1)]) as f64
}

#[no_mangle]
pub extern "C" fn run_lu_solve_wk_f32() -> f64 {
    let s = state();
    let n = s.a32.nrows();
    let mut f = s.a32.to_owned();
    let mut piv = alloc::vec![0usize; n];
    faer_wasm_kernels::lu::lu_factor_recursive_in_place(f.as_mut(), &mut piv, 0);
    let mut x = alloc::vec![0.0f32; n];
    for i in 0..n {
        x[i] = s.rhs32[(i, 0)];
    }
    faer_wasm_kernels::lu::lu_solve_in_place(f.as_ref(), &piv, &mut x);
    (x[0] + x[n - 1]) as f64
}

#[no_mangle]
pub extern "C" fn run_qr_factor_wk_f32() -> f64 {
    let s = state();
    let n = s.a32.nrows();
    let mut f = s.a32.to_owned();
    let mut tau = alloc::vec![0.0f32; n];
    faer_wasm_kernels::qr::qr_factor_in_place(f.as_mut(), &mut tau);
    (f[(0, 0)] + f[(n - 1, n - 1)] + tau[n / 2]) as f64
}

#[no_mangle]
pub extern "C" fn run_eigvals_k3_f32() -> f64 {
    let s = state();
    let n = s.a32.nrows();
    let mut h = s.a32.to_owned();
    let mut tau = alloc::vec![0.0f32; n.saturating_sub(2).max(1)];
    let mut work = alloc::vec![0.0f32; n];
    faer_wasm_kernels::hessenberg::hessenberg_factor_in_place(h.as_mut(), &mut tau, &mut work);
    for j in 0..n {
        for i in j + 2..n {
            h[(i, j)] = 0.0;
        }
    }
    if n < 480 {
        let mut w_re = alloc::vec![0.0f32; n];
        let mut w_im = alloc::vec![0.0f32; n];
        let info =
            faer_wasm_kernels::schur_small::hqr_eigvals_in_place(h.as_mut(), &mut w_re, &mut w_im);
        assert!(info == 0, "eigvals_k3_f32 (hqr) did not converge");
        (w_re[0] + w_im[n - 1]) as f64
    } else {
        let params: faer::linalg::evd::schur::SchurParams = Auto::<f32>::auto();
        let mut w_re = faer::Col::<f32>::zeros(n);
        let mut w_im = faer::Col::<f32>::zeros(n);
        let mut mem = MemBuffer::new(faer::linalg::evd::schur::multishift_qr_scratch::<f32>(
            n,
            n,
            false,
            false,
            Par::Seq,
            params,
        ));
        let (info, _, _) = real_schur::multishift_qr::<f32>(
            false,
            h.as_mut(),
            None,
            w_re.as_mut(),
            w_im.as_mut(),
            0,
            n,
            Par::Seq,
            MemStack::new(&mut mem),
            params,
        );
        assert!(info == 0, "eigvals_k3_f32 (multishift) did not converge");
        (w_re[0] + w_im[n - 1]) as f64
    }
}

// KEPT by the 2026-07-12 sweep (no live script, but cited as the R-ratio
// measurement instrument in docs/research-eig-wasm-2026-07.md — the tool
// behind the gemm-vs-streaming throughput ratio the routing decisions and
// a possible measurement paper rest on).
// Fix-3 profiling: faer matmul at the multishift sweep's accumulated-update
// shapes -- C(nb x ib) = U2^T(nb x nb) * A(nb x ib) plus the copy-back the
// sweep does after every such call. At n=128 the sweep uses nb ~ 24 and
// chunks ib <= wh cols; if per-call dispatch overhead dominates these tiny
// gemms, that's the 8x-per-sweep deficit. One export call = `reps`
// matmul+copy pairs so node-side timing amortizes its own overhead.
#[no_mangle]
pub extern "C" fn run_sweep_gemm(nb: usize, ib: usize, reps: usize) -> f64 {
    use faer::linalg::matmul::matmul;
    use faer::Accum;
    let s = state();
    let n = s.a.nrows();
    let nb = nb.min(n);
    let ib = ib.min(n);
    let u2 = s.a.as_ref().submatrix(0, 0, nb, nb);
    let mut a_slice = s.b.as_ref().submatrix(0, 0, nb, ib).to_owned();
    let mut wh = Mat::<f64>::zeros(nb, ib);
    for _ in 0..reps.max(1) {
        matmul(
            wh.as_mut(),
            Accum::Replace,
            u2.transpose(),
            a_slice.as_ref(),
            1.0,
            Par::Seq,
        );
        a_slice.copy_from(wh.as_ref());
    }
    a_slice[(0, 0)] + a_slice[(nb - 1, ib - 1)]
}

// KEPT by the 2026-07-12 sweep (no live script, but this is the instrument
// that root-caused patch 0004 — README evidence grid — and the tool for
// re-verifying that patch on every faer release adoption).
// Iteration-count probe: AED calls and multishift sweeps for one solve,
// packed as aed*100000 + sweeps. Distinguishes "faer converges slower"
// (count explosion) from "each sweep is slower" (counts match LAPACK's
// expectations but wall time doesn't).
#[no_mangle]
pub extern "C" fn run_eigvals_counters(blocking: usize, nibble: usize, profile: usize) -> f64 {
    let (_, count_aed, count_sweep) = eigvals_tuned_imp(blocking, nibble, profile);
    (count_aed * 100000 + count_sweep) as f64
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

// ---- BLAS-layer A/B (architect-directed, 2026-07-13): for every
// "unchanged" BLAS-level operation, a streaming-loop variant built on our
// SIMD primitives, timed interleaved against the faer path on one machine
// (bench/blas-ab.mjs). Tests the standing policy that shaping only pays
// where an op is hot AND arithmetic-bound: prediction is parity for the
// bandwidth-bound Level-1/2 rows and a faer win of ~R for the Level-3
// rows (the gemm row IS the R measurement). op codes:
//   0 copy   1 gemv   2 ger    3 trsv (upper, 1 rhs)
//   4 gemm   5 syrk (lower C=A·Aᵀ)   6 trmm (C=U·B)   7 trsm (U·X=B)
// variant: 0 = faer path, 1 = streaming loops. Returns a probe double.
// Fused (FMA) twins for the streaming loops, compiled only when the build
// enables relaxed-simd: `dst[i] += alpha * src[i]` in one rounding via
// `f64x2_relaxed_madd`. Variant 2 of run_blas_ab uses these; in a plain
// build variant 2 falls back to the plain loop (race it only on the FMA
// build). Ceiling probes at the bottom follow the same pattern.
#[cfg(all(target_arch = "wasm32", target_feature = "relaxed-simd"))]
#[target_feature(enable = "simd128", enable = "relaxed-simd")]
unsafe fn axpy_fma(dst: *mut f64, src: *const f64, alpha: f64, len: usize) {
    use core::arch::wasm32::*;
    let va = f64x2_splat(alpha);
    let mut i = 0usize;
    while i + 4 <= len {
        let d0 = v128_load(dst.add(i) as *const v128);
        let s0 = v128_load(src.add(i) as *const v128);
        let d1 = v128_load(dst.add(i + 2) as *const v128);
        let s1 = v128_load(src.add(i + 2) as *const v128);
        v128_store(dst.add(i) as *mut v128, f64x2_relaxed_madd(va, s0, d0));
        v128_store(dst.add(i + 2) as *mut v128, f64x2_relaxed_madd(va, s1, d1));
        i += 4;
    }
    while i < len {
        *dst.add(i) += *src.add(i) * alpha;
        i += 1;
    }
}

/// `dst += alpha·src` — fused when the build has relaxed-simd, otherwise the
/// plain kernel primitive (note the sign: kernels' axpy is dst -= src·alpha).
#[inline(always)]
unsafe fn axpy_acc(dst: *mut f64, src: *const f64, alpha: f64, len: usize) {
    #[cfg(all(target_arch = "wasm32", target_feature = "relaxed-simd"))]
    {
        axpy_fma(dst, src, alpha, len);
    }
    #[cfg(not(all(target_arch = "wasm32", target_feature = "relaxed-simd")))]
    {
        use faer_wasm_kernels::scalar::WasmScalar;
        f64::axpy(dst, src, -alpha, len);
    }
}

#[no_mangle]
pub extern "C" fn run_blas_ab(op: usize, variant: usize) -> f64 {
    use faer::linalg::matmul::matmul;
    use faer::linalg::matmul::triangular::{self, BlockStructure};
    use faer::linalg::triangular_solve;
    use faer::Accum;
    use faer_wasm_kernels::scalar::WasmScalar;
    let s = state();
    let n = s.a.nrows();
    // an upper-triangular, well-conditioned U for the triangular ops
    let mut u = s.a.to_owned();
    for j in 0..n {
        for i in j + 1..n {
            u[(i, j)] = 0.0;
        }
        u[(j, j)] += 4.0; // diagonal dominance keeps the solves tame
    }
    match (op, variant) {
        // ---- copy: C <- B
        (0, 0) => {
            let mut c = Mat::<f64>::zeros(n, n);
            c.copy_from(s.b.as_ref());
            c[(0, 0)] + c[(n - 1, n - 1)]
        }
        (0, 1) => {
            let mut c = Mat::<f64>::zeros(n, n);
            let ccs = c.as_ref().col_stride() as usize;
            let cp = c.as_mut().as_ptr_mut();
            let bp = s.b.as_ref().as_ptr();
            let bcs = s.b.col_stride() as usize;
            for j in 0..n {
                unsafe {
                    core::ptr::copy_nonoverlapping(bp.add(j * bcs), cp.add(j * ccs), n);
                }
            }
            c[(0, 0)] + c[(n - 1, n - 1)]
        }
        // ---- gemv: y <- A·x
        (1, 0) => {
            let mut y = Mat::<f64>::zeros(n, 1);
            matmul(y.as_mut(), Accum::Replace, s.a.as_ref(), s.rhs.as_ref(), 1.0, Par::Seq);
            y[(0, 0)] + y[(n - 1, 0)]
        }
        (1, 1) => {
            let mut y = alloc::vec![0.0f64; n];
            let ap = s.a.as_ref().as_ptr();
            let acs = s.a.col_stride() as usize;
            for j in 0..n {
                unsafe { f64::axpy(y.as_mut_ptr(), ap.add(j * acs), -s.rhs[(j, 0)], n) };
            }
            y[0] + y[n - 1]
        }
        // ---- ger: A <- A + x·yᵀ  (y = x here; rank-1 update)
        (2, 0) => {
            let mut c = s.b.to_owned();
            matmul(
                c.as_mut(),
                Accum::Add,
                s.rhs.as_ref(),
                s.rhs.transpose(),
                1.0,
                Par::Seq,
            );
            c[(0, 0)] + c[(n - 1, n - 1)]
        }
        (2, 1) => {
            let mut c = s.b.to_owned();
            let ccs = c.as_ref().col_stride() as usize;
            let cp = c.as_mut().as_ptr_mut();
            let xp = s.rhs.as_ref().as_ptr();
            for j in 0..n {
                unsafe { f64::axpy(cp.add(j * ccs), xp, -s.rhs[(j, 0)], n) };
            }
            c[(0, 0)] + c[(n - 1, n - 1)]
        }
        (1, 2) => {
            let mut y = alloc::vec![0.0f64; n];
            let ap = s.a.as_ref().as_ptr();
            let acs = s.a.col_stride() as usize;
            for j in 0..n {
                unsafe { axpy_acc(y.as_mut_ptr(), ap.add(j * acs), s.rhs[(j, 0)], n) };
            }
            y[0] + y[n - 1]
        }
        // ---- trsv: U·x = b, one right-hand side
        (3, 0) => {
            let mut x = s.rhs.to_owned();
            triangular_solve::solve_upper_triangular_in_place(u.as_ref(), x.as_mut(), Par::Seq);
            x[(0, 0)] + x[(n - 1, 0)]
        }
        (3, 1) => {
            let mut x = alloc::vec![0.0f64; n];
            for i in 0..n {
                x[i] = s.rhs[(i, 0)];
            }
            let up = u.as_ref().as_ptr();
            let ucs = u.as_ref().col_stride() as usize;
            for j in (0..n).rev() {
                x[j] /= u[(j, j)];
                unsafe { f64::axpy(x.as_mut_ptr(), up.add(j * ucs), x[j], j) };
            }
            x[0] + x[n - 1]
        }
        // ---- gemm: C <- A·B
        (4, 0) => {
            let mut c = Mat::<f64>::zeros(n, n);
            matmul(c.as_mut(), Accum::Replace, s.a.as_ref(), s.b.as_ref(), 1.0, Par::Seq);
            c[(0, 0)] + c[(n - 1, n - 1)]
        }
        (4, 1) => {
            let mut c = Mat::<f64>::zeros(n, n);
            let ccs = c.as_ref().col_stride() as usize;
            let cp = c.as_mut().as_ptr_mut();
            let ap = s.a.as_ref().as_ptr();
            let acs = s.a.col_stride() as usize;
            for j in 0..n {
                for l in 0..n {
                    unsafe { f64::axpy(cp.add(j * ccs), ap.add(l * acs), -s.b[(l, j)], n) };
                }
            }
            c[(0, 0)] + c[(n - 1, n - 1)]
        }
        (4, 2) => {
            let mut c = Mat::<f64>::zeros(n, n);
            let ccs = c.as_ref().col_stride() as usize;
            let cp = c.as_mut().as_ptr_mut();
            let ap = s.a.as_ref().as_ptr();
            let acs = s.a.col_stride() as usize;
            for j in 0..n {
                for l in 0..n {
                    unsafe { axpy_acc(cp.add(j * ccs), ap.add(l * acs), s.b[(l, j)], n) };
                }
            }
            c[(0, 0)] + c[(n - 1, n - 1)]
        }
        // ---- syrk: C (lower) <- A·Aᵀ
        (5, 0) => {
            let mut c = Mat::<f64>::zeros(n, n);
            triangular::matmul(
                c.as_mut(),
                BlockStructure::TriangularLower,
                Accum::Replace,
                s.a.as_ref(),
                BlockStructure::Rectangular,
                s.a.transpose(),
                BlockStructure::Rectangular,
                1.0,
                Par::Seq,
            );
            c[(0, 0)] + c[(n - 1, n - 1)]
        }
        (5, 1) => {
            let mut c = Mat::<f64>::zeros(n, n);
            let ccs = c.as_ref().col_stride() as usize;
            let cp = c.as_mut().as_ptr_mut();
            let ap = s.a.as_ref().as_ptr();
            let acs = s.a.col_stride() as usize;
            for j in 0..n {
                for l in 0..n {
                    // rows j.. of column j only (lower half)
                    unsafe {
                        f64::axpy(cp.add(j + j * ccs), ap.add(j + l * acs), -s.a[(j, l)], n - j)
                    };
                }
            }
            c[(0, 0)] + c[(n - 1, n - 1)]
        }
        (5, 2) => {
            let mut c = Mat::<f64>::zeros(n, n);
            let ccs = c.as_ref().col_stride() as usize;
            let cp = c.as_mut().as_ptr_mut();
            let ap = s.a.as_ref().as_ptr();
            let acs = s.a.col_stride() as usize;
            for j in 0..n {
                for l in 0..n {
                    unsafe { axpy_acc(cp.add(j + j * ccs), ap.add(j + l * acs), s.a[(j, l)], n - j) };
                }
            }
            c[(0, 0)] + c[(n - 1, n - 1)]
        }
        // ---- trmm: C <- U·B
        (6, 0) => {
            let mut c = Mat::<f64>::zeros(n, n);
            triangular::matmul(
                c.as_mut(),
                BlockStructure::Rectangular,
                Accum::Replace,
                u.as_ref(),
                BlockStructure::TriangularUpper,
                s.b.as_ref(),
                BlockStructure::Rectangular,
                1.0,
                Par::Seq,
            );
            c[(0, 0)] + c[(n - 1, n - 1)]
        }
        (6, 1) => {
            let mut c = Mat::<f64>::zeros(n, n);
            let ccs = c.as_ref().col_stride() as usize;
            let cp = c.as_mut().as_ptr_mut();
            let up = u.as_ref().as_ptr();
            let ucs = u.as_ref().col_stride() as usize;
            for j in 0..n {
                for l in 0..n {
                    // U's column l has rows 0..=l
                    unsafe { f64::axpy(cp.add(j * ccs), up.add(l * ucs), -s.b[(l, j)], l + 1) };
                }
            }
            c[(0, 0)] + c[(n - 1, n - 1)]
        }
        (6, 2) => {
            let mut c = Mat::<f64>::zeros(n, n);
            let ccs = c.as_ref().col_stride() as usize;
            let cp = c.as_mut().as_ptr_mut();
            let up = u.as_ref().as_ptr();
            let ucs = u.as_ref().col_stride() as usize;
            for j in 0..n {
                for l in 0..n {
                    unsafe { axpy_acc(cp.add(j * ccs), up.add(l * ucs), s.b[(l, j)], l + 1) };
                }
            }
            c[(0, 0)] + c[(n - 1, n - 1)]
        }
        // ---- trsm: U·X = B, n right-hand sides
        (7, 0) => {
            let mut x = s.b.to_owned();
            triangular_solve::solve_upper_triangular_in_place(u.as_ref(), x.as_mut(), Par::Seq);
            x[(0, 0)] + x[(n - 1, n - 1)]
        }
        (7, 1) => {
            let mut x = s.b.to_owned();
            let xcs = x.as_ref().col_stride() as usize;
            let xp = x.as_mut().as_ptr_mut();
            let up = u.as_ref().as_ptr();
            let ucs = u.as_ref().col_stride() as usize;
            for k in 0..n {
                let col = unsafe { xp.add(k * xcs) };
                for j in (0..n).rev() {
                    unsafe {
                        *col.add(j) /= u[(j, j)];
                        f64::axpy(col, up.add(j * ucs), *col.add(j), j);
                    }
                }
            }
            x[(0, 0)] + x[(n - 1, n - 1)]
        }
        (7, 2) => {
            let mut x = s.b.to_owned();
            let xcs = x.as_ref().col_stride() as usize;
            let xp = x.as_mut().as_ptr_mut();
            let up = u.as_ref().as_ptr();
            let ucs = u.as_ref().col_stride() as usize;
            for k in 0..n {
                let col = unsafe { xp.add(k * xcs) };
                for j in (0..n).rev() {
                    unsafe {
                        *col.add(j) /= u[(j, j)];
                        axpy_acc(col, up.add(j * ucs), -*col.add(j), j);
                    }
                }
            }
            x[(0, 0)] + x[(n - 1, n - 1)]
        }
        _ => f64::NAN,
    }
}

// ---- L1 assumption race (architect, 2026-07-18): swap/asum/iamax were
// carried as "hand-SIMD buys nothing" WITHOUT a measurement — exactly what
// the race-the-foundation rule forbids. op: 0 swap, 1 asum, 2 iamax.
// variant: 0 plain scalar loop, 1 hand-SIMD. Streams run per column (the
// matrices may be stride-padded, so a flat n² walk would cross padding).
#[no_mangle]
pub extern "C" fn run_l1_ab(op: usize, variant: usize) -> f64 {
    let s = state();
    let n = s.a.nrows();
    match (op, variant) {
        (0, v) => {
            // swap: exchange the columns of two working copies
            let mut x = s.a.to_owned();
            let mut y = s.b.to_owned();
            let (xcs, ycs) = (x.as_ref().col_stride() as usize, y.as_ref().col_stride() as usize);
            let (xp, yp) = (x.as_mut().as_ptr_mut(), y.as_mut().as_ptr_mut());
            for j in 0..n {
                unsafe {
                    if v == 0 {
                        l1_swap_plain(xp.add(j * xcs), yp.add(j * ycs), n);
                    } else {
                        l1_swap_simd(xp.add(j * xcs), yp.add(j * ycs), n);
                    }
                }
            }
            x[(0, 0)] + y[(n - 1, n - 1)]
        }
        (1, v) => {
            let ap = s.a.as_ref().as_ptr();
            let acs = s.a.col_stride() as usize;
            let mut total = 0.0f64;
            for j in 0..n {
                total += unsafe {
                    if v == 0 {
                        l1_asum_plain(ap.add(j * acs), n)
                    } else {
                        l1_asum_simd(ap.add(j * acs), n)
                    }
                };
            }
            total
        }
        (2, v) => {
            let ap = s.a.as_ref().as_ptr();
            let acs = s.a.col_stride() as usize;
            let mut best = -1.0f64;
            let mut best_idx = 0usize;
            for j in 0..n {
                let (m, i) = unsafe {
                    if v == 0 {
                        l1_iamax_plain(ap.add(j * acs), n)
                    } else {
                        l1_iamax_simd(ap.add(j * acs), n)
                    }
                };
                if m > best {
                    best = m;
                    best_idx = j * n + i;
                }
            }
            best + best_idx as f64
        }
        _ => f64::NAN,
    }
}

// ---- the shipped BLAS layer (the blas.wasm repo since the split), Level 1: roofline rows +
// cross-target determinism probes. run_l1_layer streams every column of
// the n×n state through the layer's function; the script scores GB/s
// against the same-run bandwidth ceiling. Mutating ops run in place on
// the persistent state (no per-call allocation — the allocator-tax
// lesson); the chosen constants keep values bounded across iterations
// (scal by -1, rot by an orthogonal pair, axpy with small alpha).
fn state_mut() -> &'static mut State {
    unsafe { (*STATE.0.get()).as_mut().expect("call setup(n) first") }
}


unsafe fn l1_swap_plain(x: *mut f64, y: *mut f64, len: usize) {
    for i in 0..len {
        let t = *x.add(i);
        *x.add(i) = *y.add(i);
        *y.add(i) = t;
    }
}

unsafe fn l1_asum_plain(x: *const f64, len: usize) -> f64 {
    let mut s = 0.0;
    for i in 0..len {
        s += (*x.add(i)).abs();
    }
    s
}

unsafe fn l1_iamax_plain(x: *const f64, len: usize) -> (f64, usize) {
    let mut m = -1.0;
    let mut mi = 0usize;
    for i in 0..len {
        let v = (*x.add(i)).abs();
        if v > m {
            m = v;
            mi = i;
        }
    }
    (m, mi)
}

#[cfg(target_arch = "wasm32")]
#[target_feature(enable = "simd128")]
unsafe fn l1_swap_simd(x: *mut f64, y: *mut f64, len: usize) {
    use core::arch::wasm32::*;
    let mut i = 0usize;
    while i + 4 <= len {
        let x0 = v128_load(x.add(i) as *const v128);
        let y0 = v128_load(y.add(i) as *const v128);
        let x1 = v128_load(x.add(i + 2) as *const v128);
        let y1 = v128_load(y.add(i + 2) as *const v128);
        v128_store(x.add(i) as *mut v128, y0);
        v128_store(y.add(i) as *mut v128, x0);
        v128_store(x.add(i + 2) as *mut v128, y1);
        v128_store(y.add(i + 2) as *mut v128, x1);
        i += 4;
    }
    while i < len {
        let t = *x.add(i);
        *x.add(i) = *y.add(i);
        *y.add(i) = t;
        i += 1;
    }
}

#[cfg(not(target_arch = "wasm32"))]
unsafe fn l1_swap_simd(x: *mut f64, y: *mut f64, len: usize) {
    l1_swap_plain(x, y, len)
}

#[cfg(target_arch = "wasm32")]
#[target_feature(enable = "simd128")]
unsafe fn l1_asum_simd(x: *const f64, len: usize) -> f64 {
    use core::arch::wasm32::*;
    let mut a0 = f64x2_splat(0.0);
    let mut a1 = f64x2_splat(0.0);
    let mut i = 0usize;
    while i + 4 <= len {
        a0 = f64x2_add(a0, f64x2_abs(v128_load(x.add(i) as *const v128)));
        a1 = f64x2_add(a1, f64x2_abs(v128_load(x.add(i + 2) as *const v128)));
        i += 4;
    }
    let a = f64x2_add(a0, a1);
    let mut s = f64x2_extract_lane::<0>(a) + f64x2_extract_lane::<1>(a);
    while i < len {
        s += (*x.add(i)).abs();
        i += 1;
    }
    s
}

#[cfg(not(target_arch = "wasm32"))]
unsafe fn l1_asum_simd(x: *const f64, len: usize) -> f64 {
    l1_asum_plain(x, len)
}

#[cfg(target_arch = "wasm32")]
#[target_feature(enable = "simd128")]
unsafe fn l1_iamax_simd(x: *const f64, len: usize) -> (f64, usize) {
    use core::arch::wasm32::*;
    // branch-free vector pass for the max VALUE, then one scalar rescan for
    // its index — the realistic SIMD strategy for an argmax
    let mut m0 = f64x2_splat(-1.0);
    let mut m1 = f64x2_splat(-1.0);
    let mut i = 0usize;
    while i + 4 <= len {
        m0 = f64x2_pmax(m0, f64x2_abs(v128_load(x.add(i) as *const v128)));
        m1 = f64x2_pmax(m1, f64x2_abs(v128_load(x.add(i + 2) as *const v128)));
        i += 4;
    }
    let m = f64x2_pmax(m0, m1);
    let mut best = f64x2_extract_lane::<0>(m).max(f64x2_extract_lane::<1>(m));
    while i < len {
        best = best.max((*x.add(i)).abs());
        i += 1;
    }
    let mut mi = 0usize;
    for k in 0..len {
        if (*x.add(k)).abs() == best {
            mi = k;
            break;
        }
    }
    (best, mi)
}

#[cfg(not(target_arch = "wasm32"))]
unsafe fn l1_iamax_simd(x: *const f64, len: usize) -> (f64, usize) {
    l1_iamax_plain(x, len)
}

// ---- ceiling probes (roofline metric, adopted 2026-07-18): the two
// machine limits every benchmark row is scored against.

/// Memory-bandwidth probe: triad c = a + 2.5·b over the n×n state
/// matrices. Bytes moved per call = 3·8·n² (read a, read b, write c).
#[no_mangle]
pub extern "C" fn run_ceiling_bw() -> f64 {
    // v2 (2026-07-18): the original version allocated + copied an n×n
    // matrix INSIDE the timed region (to_owned per call) and streamed c
    // twice — depressing the measured ceiling and miscounting traffic.
    // Now a pure single-pass triad sym ← a + 2.5·b over persistent
    // state: exactly 3·8·n² bytes per call, no allocation. (sym is
    // sacrificed as the destination — don't run symmetric-eigen benches
    // on the same instance after this probe.)
    let s = state_mut();
    let n = s.a.nrows();
    let acs = s.a.as_ref().col_stride() as usize;
    let bcs = s.b.as_ref().col_stride() as usize;
    let ccs = s.sym.as_ref().col_stride() as usize;
    let ap = s.a.as_ref().as_ptr();
    let bp = s.b.as_ref().as_ptr();
    let cp = s.sym.as_mut().as_ptr_mut();
    for j in 0..n {
        unsafe {
            triad(cp.add(j * ccs), ap.add(j * acs), bp.add(j * bcs), n);
        }
    }
    s.sym[(0, 0)] + s.sym[(n - 1, n - 1)]
}

/// c ← a + 2.5·b, one pass, 2 lanes 2× unrolled.
#[cfg_attr(target_arch = "wasm32", target_feature(enable = "simd128"))]
unsafe fn triad(c: *mut f64, a: *const f64, b: *const f64, len: usize) {
    #[cfg(target_arch = "wasm32")]
    {
        use core::arch::wasm32::*;
        let k = f64x2_splat(2.5);
        let mut i = 0usize;
        while i + 4 <= len {
            let a0 = v128_load(a.add(i) as *const v128);
            let b0 = v128_load(b.add(i) as *const v128);
            let a1 = v128_load(a.add(i + 2) as *const v128);
            let b1 = v128_load(b.add(i + 2) as *const v128);
            v128_store(c.add(i) as *mut v128, f64x2_add(a0, f64x2_mul(b0, k)));
            v128_store(c.add(i + 2) as *mut v128, f64x2_add(a1, f64x2_mul(b1, k)));
            i += 4;
        }
        while i < len {
            *c.add(i) = *a.add(i) + *b.add(i) * 2.5;
            i += 1;
        }
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        for i in 0..len {
            *c.add(i) = *a.add(i) + *b.add(i) * 2.5;
        }
    }
}

/// Peak-arithmetic probe: register-resident mul+add chains, 8 independent
/// v128 accumulators × `iters` rounds. FLOPs per call = iters · 8 · 2 lanes
/// · 2 ops. Fused (one relaxed_madd = still 2 FLOPs) when the build has
/// relaxed-simd — so this probe measures the ceiling OF THE BUILD.
#[no_mangle]
pub extern "C" fn run_ceiling_flops(iters: usize) -> f64 {
    #[cfg(target_arch = "wasm32")]
    unsafe {
        ceiling_flops_imp(iters)
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        let _ = iters;
        f64::NAN
    }
}

#[cfg(target_arch = "wasm32")]
#[target_feature(enable = "simd128")]
unsafe fn ceiling_flops_imp(iters: usize) -> f64 {
    use core::arch::wasm32::*;
    let m = f64x2_splat(1.000000001);
    let a = f64x2_splat(1e-9);
    let mut acc = [f64x2_splat(1.0); 8];
    for _ in 0..iters {
        for k in 0..8 {
            #[cfg(target_feature = "relaxed-simd")]
            {
                acc[k] = f64x2_relaxed_madd(acc[k], m, a);
            }
            #[cfg(not(target_feature = "relaxed-simd"))]
            {
                acc[k] = f64x2_add(f64x2_mul(acc[k], m), a);
            }
        }
    }
    let mut s = 0.0;
    for k in 0..8 {
        s += f64x2_extract_lane::<0>(acc[k]) + f64x2_extract_lane::<1>(acc[k]);
    }
    s
}


// The BLAS-layer rows, determinism probes, and their f32 twins moved
// to the blas.wasm repo's bench module (2026-07-19/20) — the layer measures
// itself; this harness keeps only the faer-dependent benches (and
// run_ceiling_bw/run_ceiling_flops for the legacy ceilings.mjs).
// gemm-tune-ab.mjs loads both wasm modules for the faer race.

#[cfg(target_arch = "wasm32")]
mod wasm_shim {
    use core::alloc::{GlobalAlloc, Layout};

    // NB deliberately duplicated in smoke-test/src/lib.rs (that crate is
    // the zero-import consumer example and must stay self-contained) —
    // keep the two shims in sync (2026-07-12 sweep note).
    // LIFO-rewind bump allocator over memory.grow (upgraded from leak-only
    // 2026-07-11): freeing the MOST RECENT allocation rewinds the bump
    // pointer, so nested temporaries — the pattern of faer's per-call c64
    // matmul temps, measured at 15.4 GB cumulative / 25K allocations inside
    // ONE c64 multishift at n=600, peak live only ~19 MB — are reclaimed.
    // Non-LIFO frees still leak (bounded by the old behavior). The module
    // still needs zero imports; node still re-instantiates per size.
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
