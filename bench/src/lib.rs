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
    unsafe { *STATE.0.get() = Some(State { a, b, sym, rhs, ac, bc, rhsc, a32, b32, rhs32 }) }
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
