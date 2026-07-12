//! Correctness gate for the c64 eigenvector kernel: complex Schur pipeline
//! → `ctrevc_in_place`. Ground truth is the per-eigenpair residual
//! ‖A·v − λ·v‖ with λ read from diag(T). Also gates the n=512
//! multishift-composed route and a defective-matrix smoke (repeated
//! eigenvalues on a triangular T must stay finite through the
//! perturbed-pivot path).

use faer::{c64, Mat};
use faer_wasm_kernels::eigvec_cplx::ctrevc_in_place;
use faer_wasm_kernels::hessenberg_cplx::{hessenberg_cplx_factor_in_place, hessenberg_cplx_form_q};
use faer_wasm_kernels::schur_small_cplx::chqr_schur_in_place;

fn fill_c(n: usize, mut s: u64) -> Mat<c64> {
    let mut next = move || {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((s >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    };
    Mat::from_fn(n, n, |_, _| {
        let re = next();
        let im = next();
        c64::new(re, im)
    })
}

fn cabs1(z: c64) -> f64 {
    z.re.abs() + z.im.abs()
}

fn kernel_schur_c64(a: &Mat<c64>) -> (Mat<c64>, Mat<c64>) {
    let n = a.nrows();
    let mut t = a.clone();
    let k = n.saturating_sub(2);
    let mut tau = vec![c64::new(0.0, 0.0); k.max(1)];
    let mut work = vec![c64::new(0.0, 0.0); n];
    hessenberg_cplx_factor_in_place(t.as_mut(), &mut tau, &mut work);
    let mut z = Mat::<c64>::zeros(n, n);
    hessenberg_cplx_form_q(t.as_ref(), &tau, z.as_mut());
    for j in 0..n {
        for i in j + 2..n {
            t[(i, j)] = c64::new(0.0, 0.0);
        }
    }
    let mut w = vec![c64::new(0.0, 0.0); n];
    let info = chqr_schur_in_place(t.as_mut(), Some(z.as_mut()), &mut w, true);
    assert!(info == 0, "n={n}: chqr_schur did not converge");
    (t, z)
}

/// max over eigenpairs of ‖A·v − λ·v‖ in cabs1; columns are normalized to
/// max cabs1-component 1, so no ‖v‖ division needed
fn max_eig_residual(a: &Mat<c64>, t: &Mat<c64>, v: &Mat<c64>) -> f64 {
    let n = a.nrows();
    let av = a * v;
    let mut worst = 0.0f64;
    for k in 0..n {
        let lam = t[(k, k)];
        for r in 0..n {
            worst = worst.max(cabs1(av[(r, k)] - lam * v[(r, k)]));
        }
    }
    worst
}

#[test]
fn eigvec_c64_residuals() {
    for &n in &[1usize, 2, 3, 4, 5, 6, 7, 8, 12, 16, 33, 64, 96, 128, 200, 256] {
        let a = fill_c(n, 0x9E3779B97F4A7C15 ^ (n as u64));
        let (t, z) = kernel_schur_c64(&a);
        let mut v = Mat::<c64>::zeros(n, n);
        ctrevc_in_place(t.as_ref(), z.as_ref(), v.as_mut());
        // normalization: max cabs1 component of every column == 1
        for k in 0..n {
            let mut emax = 0.0f64;
            for r in 0..n {
                emax = emax.max(cabs1(v[(r, k)]));
            }
            assert!(
                (emax - 1.0).abs() < 1e-12,
                "n={n}: column {k} not normalized (emax={emax:.3e})"
            );
        }
        let res = max_eig_residual(&a, &t, &v);
        assert!(res < 1e-10 * (n.max(4) as f64), "n={n}: residual {res:.2e}");
    }
}

#[test]
fn eigvec_c64_multishift_composition_above_crossover() {
    use faer::dyn_stack::{MemBuffer, MemStack};
    use faer::linalg::evd::schur::{self, complex_schur};
    use faer::{Auto, Par};

    let n = 512usize;
    let a = fill_c(n, 0x94D049BB133111EB);
    let mut t = a.clone();
    let mut tau = vec![c64::new(0.0, 0.0); n - 2];
    let mut work = vec![c64::new(0.0, 0.0); n];
    hessenberg_cplx_factor_in_place(t.as_mut(), &mut tau, &mut work);
    let mut z = Mat::<c64>::zeros(n, n);
    hessenberg_cplx_form_q(t.as_ref(), &tau, z.as_mut());
    for j in 0..n {
        for i in j + 2..n {
            t[(i, j)] = c64::new(0.0, 0.0);
        }
    }
    let params: schur::SchurParams = Auto::<c64>::auto();
    let mut w = faer::Col::<c64>::zeros(n);
    let mut mem = MemBuffer::new(schur::multishift_qr_scratch::<c64>(
        n,
        n,
        true,
        true,
        Par::Seq,
        params,
    ));
    let (info, _, _) = complex_schur::multishift_qr::<c64>(
        true,
        t.as_mut(),
        Some(z.as_mut()),
        w.as_mut(),
        0,
        n,
        Par::Seq,
        MemStack::new(&mut mem),
        params,
    );
    assert!(info == 0, "complex multishift did not converge");
    for j in 0..n {
        for i in j + 2..n {
            t[(i, j)] = c64::new(0.0, 0.0);
        }
    }
    let mut v = Mat::<c64>::zeros(n, n);
    ctrevc_in_place(t.as_ref(), z.as_ref(), v.as_mut());
    let res = max_eig_residual(&a, &t, &v);
    assert!(res < 1e-10 * (n as f64), "n={n}: residual {res:.2e}");
}

#[test]
fn eigvec_c64_defective_smoke() {
    // triangular T with a repeated eigenvalue and a coupling entry — the
    // second solve hits an exactly zero shifted pivot (perturbed path).
    // Contract: finite, normalized output; the true eigenvector e1 exact.
    let n = 4usize;
    let mut t = Mat::<c64>::zeros(n, n);
    for k in 0..n {
        t[(k, k)] = c64::new(1.0, 1.0);
        if k + 1 < n {
            t[(k, k + 1)] = c64::new(1.0, 0.0);
        }
    }
    let z = Mat::<c64>::identity(n, n);
    let mut v = Mat::<c64>::zeros(n, n);
    ctrevc_in_place(t.as_ref(), z.as_ref(), v.as_mut());
    for j in 0..n {
        let mut emax = 0.0f64;
        for i in 0..n {
            assert!(
                v[(i, j)].re.is_finite() && v[(i, j)].im.is_finite(),
                "non-finite at ({i},{j})"
            );
            emax = emax.max(cabs1(v[(i, j)]));
        }
        assert!((emax - 1.0).abs() < 1e-12, "column {j} not normalized");
    }
    assert!(v[(0, 0)] == c64::new(1.0, 0.0));
    for i in 1..n {
        assert!(v[(i, 0)] == c64::new(0.0, 0.0));
    }
}
