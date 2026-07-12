//! Correctness gate for the trevc-shaped eigenvector kernel (eigenvector
//! campaign, 2026-07-12): kernel Schur pipeline → `trevc_in_place`.
//! Ground truth is the per-eigenpair residual ‖A·v − λ·v‖ — for a complex
//! pair λ = a±bi packed as columns (re, im) that means
//! A·re = a·re − b·im and A·im = a·im + b·re. Also gates: the n=512
//! multishift-composed route, the f32 twins at eps32 tolerances, and a
//! defective-matrix smoke (perturbed-pivot path must stay finite).

use faer::Mat;
use faer_wasm_kernels::eigvec::trevc_in_place;
use faer_wasm_kernels::hessenberg::{hessenberg_factor_in_place, hessenberg_form_q};
use faer_wasm_kernels::schur_small::hqr_schur_in_place;

fn fill(n: usize, mut s: u64) -> Mat<f64> {
    Mat::from_fn(n, n, |_, _| {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((s >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    })
}

fn kernel_schur(a: &Mat<f64>) -> (Mat<f64>, Mat<f64>) {
    let n = a.nrows();
    let mut t = a.clone();
    let k = n.saturating_sub(2);
    let mut tau = vec![0.0f64; k.max(1)];
    let mut work = vec![0.0f64; n];
    hessenberg_factor_in_place(t.as_mut(), &mut tau, &mut work);
    let mut z = Mat::<f64>::zeros(n, n);
    hessenberg_form_q(t.as_ref(), &tau, z.as_mut());
    for j in 0..n {
        for i in j + 2..n {
            t[(i, j)] = 0.0;
        }
    }
    let mut w_re = vec![0.0f64; n];
    let mut w_im = vec![0.0f64; n];
    let info = hqr_schur_in_place(t.as_mut(), Some(z.as_mut()), &mut w_re, &mut w_im, true);
    assert!(info == 0, "n={n}: hqr_schur did not converge");
    (t, z)
}

/// max over eigenpairs of ‖A·v − λ·v‖_∞, reading λ and the pair structure
/// from T's diagonal the same way the kernel does. Columns are normalized
/// to max-component 1, so no ‖v‖ division is needed.
fn max_eig_residual(a: &Mat<f64>, t: &Mat<f64>, v: &Mat<f64>) -> f64 {
    let n = a.nrows();
    let av = a * v;
    let mut worst = 0.0f64;
    let mut k = 0usize;
    while k < n {
        let pair = k + 1 < n && t[(k + 1, k)] != 0.0;
        if !pair {
            let lam = t[(k, k)];
            for r in 0..n {
                worst = worst.max((av[(r, k)] - lam * v[(r, k)]).abs());
            }
            k += 1;
        } else {
            let wr = t[(k, k)];
            let wi = t[(k + 1, k)].abs().sqrt() * t[(k, k + 1)].abs().sqrt();
            for r in 0..n {
                let re = av[(r, k)] - (wr * v[(r, k)] - wi * v[(r, k + 1)]);
                let im = av[(r, k + 1)] - (wr * v[(r, k + 1)] + wi * v[(r, k)]);
                worst = worst.max(re.abs()).max(im.abs());
            }
            k += 2;
        }
    }
    worst
}

#[test]
fn eigvec_residuals() {
    for &n in &[1usize, 2, 3, 4, 5, 6, 7, 8, 12, 16, 33, 64, 96, 128, 200, 256] {
        let a = fill(n, 0x9E3779B97F4A7C15 ^ (n as u64));
        let (t, z) = kernel_schur(&a);
        let mut v = Mat::<f64>::zeros(n, n);
        trevc_in_place(t.as_ref(), z.as_ref(), v.as_mut());
        // every column normalized: max |re|(+|im|) component == 1
        let mut k = 0usize;
        while k < n {
            let pair = k + 1 < n && t[(k + 1, k)] != 0.0;
            let cols = if pair { 2 } else { 1 };
            let mut emax = 0.0f64;
            for r in 0..n {
                let m = if pair {
                    v[(r, k)].abs() + v[(r, k + 1)].abs()
                } else {
                    v[(r, k)].abs()
                };
                emax = emax.max(m);
            }
            assert!(
                (emax - 1.0).abs() < 1e-12,
                "n={n}: column {k} not normalized (emax={emax:.3e})"
            );
            k += cols;
        }
        let res = max_eig_residual(&a, &t, &v);
        assert!(res < 1e-10 * (n.max(4) as f64), "n={n}: residual {res:.2e}");
    }
}

#[test]
fn eigvec_multishift_composition_above_crossover() {
    // the n >= 480 route: kernel Hessenberg + Q seeding faer's multishift
    // (want_t, Z), then trevc on that (T, Z)
    use faer::dyn_stack::{MemBuffer, MemStack};
    use faer::linalg::evd::schur::{self, real_schur};
    use faer::{Auto, Par};

    let n = 512usize;
    let a = fill(n, 0x94D049BB133111EB);
    let mut t = a.clone();
    let mut tau = vec![0.0f64; n - 2];
    let mut work = vec![0.0f64; n];
    hessenberg_factor_in_place(t.as_mut(), &mut tau, &mut work);
    let mut z = Mat::<f64>::zeros(n, n);
    hessenberg_form_q(t.as_ref(), &tau, z.as_mut());
    for j in 0..n {
        for i in j + 2..n {
            t[(i, j)] = 0.0;
        }
    }
    let params: schur::SchurParams = Auto::<f64>::auto();
    let mut w_re = faer::Col::<f64>::zeros(n);
    let mut w_im = faer::Col::<f64>::zeros(n);
    let mut mem = MemBuffer::new(schur::multishift_qr_scratch::<f64>(
        n,
        n,
        true,
        true,
        Par::Seq,
        params,
    ));
    let (info, _, _) = real_schur::multishift_qr::<f64>(
        true,
        t.as_mut(),
        Some(z.as_mut()),
        w_re.as_mut(),
        w_im.as_mut(),
        0,
        n,
        Par::Seq,
        MemStack::new(&mut mem),
        params,
    );
    assert!(info == 0, "multishift did not converge");
    for j in 0..n {
        for i in j + 2..n {
            t[(i, j)] = 0.0;
        }
    }
    let mut v = Mat::<f64>::zeros(n, n);
    trevc_in_place(t.as_ref(), z.as_ref(), v.as_mut());
    let res = max_eig_residual(&a, &t, &v);
    assert!(res < 1e-10 * (n as f64), "n={n}: residual {res:.2e}");
}

#[test]
fn eigvec_f32() {
    for &n in &[4usize, 16, 33, 64, 96] {
        let a64 = fill(n, 0x2545F4914F6CDD1D ^ (n as u64));
        let a = Mat::from_fn(n, n, |i, j| a64[(i, j)] as f32);
        let mut t = a.clone();
        let k = n.saturating_sub(2);
        let mut tau = vec![0.0f32; k.max(1)];
        let mut work = vec![0.0f32; n];
        hessenberg_factor_in_place(t.as_mut(), &mut tau, &mut work);
        let mut z = Mat::<f32>::zeros(n, n);
        hessenberg_form_q(t.as_ref(), &tau, z.as_mut());
        for j in 0..n {
            for i in j + 2..n {
                t[(i, j)] = 0.0;
            }
        }
        let mut w_re = vec![0.0f32; n];
        let mut w_im = vec![0.0f32; n];
        let info = hqr_schur_in_place(t.as_mut(), Some(z.as_mut()), &mut w_re, &mut w_im, true);
        assert!(info == 0, "n={n}: f32 hqr_schur did not converge");
        let mut v = Mat::<f32>::zeros(n, n);
        trevc_in_place(t.as_ref(), z.as_ref(), v.as_mut());
        let av = &a * &v;
        let mut worst = 0.0f32;
        let mut k = 0usize;
        while k < n {
            let pair = k + 1 < n && t[(k + 1, k)] != 0.0;
            if !pair {
                let lam = t[(k, k)];
                for r in 0..n {
                    worst = worst.max((av[(r, k)] - lam * v[(r, k)]).abs());
                }
                k += 1;
            } else {
                let wr = t[(k, k)];
                let wi = t[(k + 1, k)].abs().sqrt() * t[(k, k + 1)].abs().sqrt();
                for r in 0..n {
                    let re = av[(r, k)] - (wr * v[(r, k)] - wi * v[(r, k + 1)]);
                    let im = av[(r, k + 1)] - (wr * v[(r, k + 1)] + wi * v[(r, k)]);
                    worst = worst.max(re.abs()).max(im.abs());
                }
                k += 2;
            }
        }
        assert!(worst < 1e-4 * (n as f32), "n={n}: f32 residual {worst:.2e}");
    }
}

#[test]
fn eigvec_defective_smoke() {
    // A Jordan block has one eigenvector; the second solve hits an exactly
    // singular shifted diagonal and rides dlaln2's perturbed-pivot path.
    // Contract: finite output, normalized columns — accuracy is undefined
    // for defective matrices (same as LAPACK).
    let n = 4usize;
    let mut t = Mat::<f64>::zeros(n, n);
    for k in 0..n {
        t[(k, k)] = 1.0;
        if k + 1 < n {
            t[(k, k + 1)] = 1.0;
        }
    }
    let z = Mat::<f64>::identity(n, n);
    let mut v = Mat::<f64>::zeros(n, n);
    trevc_in_place(t.as_ref(), z.as_ref(), v.as_mut());
    for j in 0..n {
        let mut emax = 0.0f64;
        for i in 0..n {
            assert!(v[(i, j)].is_finite(), "non-finite at ({i},{j})");
            emax = emax.max(v[(i, j)].abs());
        }
        assert!((emax - 1.0).abs() < 1e-12, "column {j} not normalized");
    }
    // the true eigenvector e1 must be exact
    assert!(v[(0, 0)] == 1.0);
    for i in 1..n {
        assert!(v[(i, 0)] == 0.0);
    }
}
