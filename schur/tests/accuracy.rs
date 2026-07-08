//! Accuracy gate for the Schur companion crate: backward error,
//! orthogonality, quasi-triangular structure, eigenvalue cross-checks
//! (real vs complex driver, trace invariant), and reordering behavior.

use faer::prelude::*;
use faer::Mat;
use faer_schur::complex::{complex_schur, complex_schur_move, complex_schur_select};
use faer_schur::real::{real_schur, real_schur_move, real_schur_select};

const SIZES: &[usize] = &[0, 1, 2, 3, 5, 8, 16, 33, 64, 96, 150];

// deterministic fill (same LCG as bench/), values in [-1, 1]
fn fill(n: usize, mut s: u64) -> Mat<f64> {
    Mat::from_fn(n, n, |_, _| {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((s >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    })
}

fn max_abs(m: MatRef<f64>) -> f64 {
    let mut v = 0.0f64;
    for j in 0..m.ncols() {
        for i in 0..m.nrows() {
            v = v.max(m[(i, j)].abs());
        }
    }
    v
}

fn max_abs_c(m: MatRef<c64>) -> f64 {
    let mut v = 0.0f64;
    for j in 0..m.ncols() {
        for i in 0..m.nrows() {
            v = v.max(m[(i, j)].re.hypot(m[(i, j)].im));
        }
    }
    v
}

/// eigenvalues read off the quasi-triangular diagonal blocks; also asserts
/// the structure is valid (strictly below the subdiagonal is exactly zero,
/// no two consecutive nonzero subdiagonal entries)
fn quasi_tri_eigenvalues(t: MatRef<f64>) -> Vec<(f64, f64)> {
    let n = t.nrows();
    for j in 0..n {
        for i in j + 2..n {
            assert_eq!(t[(i, j)], 0.0, "junk below subdiagonal at ({i},{j})");
        }
    }
    let mut w = Vec::new();
    let mut k = 0;
    while k < n {
        if k + 1 < n && t[(k + 1, k)] != 0.0 {
            if k + 2 < n {
                assert_eq!(t[(k + 2, k + 1)], 0.0, "overlapping 2x2 blocks at {k}");
            }
            let (a, b, c, d) = (t[(k, k)], t[(k, k + 1)], t[(k + 1, k)], t[(k + 1, k + 1)]);
            let tr2 = 0.5 * (a + d);
            let disc = 0.25 * (a - d) * (a - d) + b * c;
            if disc < 0.0 {
                let im = (-disc).sqrt();
                w.push((tr2, im));
                w.push((tr2, -im));
            } else {
                let s = disc.sqrt();
                w.push((tr2 + s, 0.0));
                w.push((tr2 - s, 0.0));
            }
            k += 2;
        } else {
            w.push((t[(k, k)], 0.0));
            k += 1;
        }
    }
    w
}

/// multiset equality via greedy nearest-match (sorting is fragile when
/// conjugate pairs differ in the last ulp of the sort key)
fn assert_eig_sets_match(a: Vec<(f64, f64)>, b: Vec<(f64, f64)>, tol: f64, ctx: &str) {
    assert_eq!(a.len(), b.len(), "{ctx}: eigenvalue count");
    let mut used = vec![false; b.len()];
    for x in &a {
        let found = b.iter().enumerate().any(|(j, y)| {
            if !used[j] && (x.0 - y.0).abs() < tol && (x.1 - y.1).abs() < tol {
                used[j] = true;
                true
            } else {
                false
            }
        });
        assert!(found, "{ctx}: unmatched eigenvalue {x:?} (tol {tol:.2e})");
    }
}

fn check_real_factorization(a: MatRef<f64>, t: MatRef<f64>, z: MatRef<f64>, ctx: &str) {
    let n = a.nrows();
    let scale = max_abs(a).max(1.0);
    let tol = 1e-13 * scale * (n.max(1) as f64);
    let recon = z * t * z.transpose();
    let mut err = 0.0f64;
    for j in 0..n {
        for i in 0..n {
            err = err.max((a[(i, j)] - recon[(i, j)]).abs());
        }
    }
    assert!(err < tol, "{ctx}: backward error {err:.2e} >= {tol:.2e}");
    let ztz = z.transpose() * z;
    let mut oerr = 0.0f64;
    for j in 0..n {
        for i in 0..n {
            let id = if i == j { 1.0 } else { 0.0 };
            oerr = oerr.max((ztz[(i, j)] - id).abs());
        }
    }
    let otol = 1e-14 * (n.max(1) as f64);
    assert!(oerr < otol, "{ctx}: Z not orthogonal, err {oerr:.2e} >= {otol:.2e}");
}

fn check_complex_factorization(a: MatRef<c64>, t: MatRef<c64>, z: MatRef<c64>, ctx: &str) {
    let n = a.nrows();
    let scale = max_abs_c(a).max(1.0);
    let tol = 1e-13 * scale * (n.max(1) as f64);
    // strictly triangular
    for j in 0..n {
        for i in j + 1..n {
            assert_eq!(t[(i, j)], c64::new(0.0, 0.0), "{ctx}: T not triangular at ({i},{j})");
        }
    }
    let recon = z * t * z.adjoint();
    let mut err = 0.0f64;
    for j in 0..n {
        for i in 0..n {
            let d = a[(i, j)] - recon[(i, j)];
            err = err.max(d.re.hypot(d.im));
        }
    }
    assert!(err < tol, "{ctx}: backward error {err:.2e} >= {tol:.2e}");
    let ztz = z.adjoint() * z;
    let mut oerr = 0.0f64;
    for j in 0..n {
        for i in 0..n {
            let id = if i == j { 1.0 } else { 0.0 };
            let d = ztz[(i, j)] - c64::new(id, 0.0);
            oerr = oerr.max(d.re.hypot(d.im));
        }
    }
    let otol = 1e-14 * (n.max(1) as f64);
    assert!(oerr < otol, "{ctx}: Z not unitary, err {oerr:.2e} >= {otol:.2e}");
}

#[test]
fn real_schur_factorization_and_eigenvalues() {
    for &n in SIZES {
        let a = fill(n, 0x9E3779B97F4A7C15 ^ (n as u64));
        let s = real_schur(a.as_ref(), Par::Seq).unwrap();
        let ctx = format!("real n={n}");
        check_real_factorization(a.as_ref(), s.t.as_ref(), s.z.as_ref(), &ctx);

        // w_re/w_im agree with the diagonal blocks of T
        let from_t = quasi_tri_eigenvalues(s.t.as_ref());
        let from_w: Vec<_> = (0..n).map(|k| (s.w_re[k], s.w_im[k])).collect();
        let tol = 1e-10 * (n.max(1) as f64);
        assert_eig_sets_match(from_t, from_w.clone(), tol, &ctx);

        // trace invariant: sum of eigenvalues == trace(A)
        let tr: f64 = (0..n).map(|k| a[(k, k)]).sum();
        let ws: f64 = from_w.iter().map(|w| w.0).sum();
        assert!(
            (tr - ws).abs() < 1e-11 * (n.max(1) as f64),
            "{ctx}: trace {tr} vs eigensum {ws}"
        );

        // cross-check against the complex driver on the same (real) matrix
        let ac = Mat::from_fn(n, n, |i, j| c64::new(a[(i, j)], 0.0));
        let sc = complex_schur(ac.as_ref(), Par::Seq).unwrap();
        check_complex_factorization(ac.as_ref(), sc.t.as_ref(), sc.z.as_ref(), &ctx);
        let from_c: Vec<_> = (0..n).map(|k| (sc.w[k].re, sc.w[k].im)).collect();
        assert_eig_sets_match(from_w, from_c, tol, &format!("{ctx} vs complex"));
    }
}

#[test]
fn complex_schur_factorization() {
    for &n in SIZES {
        let re = fill(n, 0xD1B54A32D192ED03 ^ (n as u64));
        let im = fill(n, 0x853C49E6748FEA9B ^ (n as u64));
        let a = Mat::from_fn(n, n, |i, j| c64::new(re[(i, j)], im[(i, j)]));
        let s = complex_schur(a.as_ref(), Par::Seq).unwrap();
        let ctx = format!("complex n={n}");
        check_complex_factorization(a.as_ref(), s.t.as_ref(), s.z.as_ref(), &ctx);
        // diagonal of T == w
        for k in 0..n {
            assert_eq!(s.t[(k, k)], s.w[k], "{ctx}: w[{k}] != T diagonal");
        }
        let tr = (0..n).fold(c64::new(0.0, 0.0), |acc, k| acc + a[(k, k)]);
        let ws = (0..n).fold(c64::new(0.0, 0.0), |acc, k| acc + s.w[k]);
        let d = tr - ws;
        assert!(
            d.re.hypot(d.im) < 1e-11 * (n.max(1) as f64),
            "{ctx}: trace {tr:?} vs eigensum {ws:?}"
        );
    }
}

#[test]
fn real_reorder_select() {
    for &n in &[2usize, 3, 5, 8, 16, 33, 64, 96] {
        let a = fill(n, 0x2545F4914F6CDD1D ^ (n as u64));
        let s = real_schur(a.as_ref(), Par::Seq).unwrap();
        let mut t = s.t.clone();
        let mut z = s.z.clone();
        let before = quasi_tri_eigenvalues(t.as_ref());

        // select eigenvalues with positive real part (pair-consistent since
        // conjugate pairs share their real part)
        let select: Vec<bool> = (0..n).map(|k| s.w_re[k] > 0.0).collect();
        let expected_m = select.iter().filter(|&&x| x).count();
        let m = real_schur_select(t.as_mut(), Some(z.as_mut()), &select).unwrap();
        assert_eq!(m, expected_m, "n={n}: m");

        let ctx = format!("real reorder n={n}");
        check_real_factorization(a.as_ref(), t.as_ref(), z.as_ref(), &ctx);
        let after = quasi_tri_eigenvalues(t.as_ref());
        assert_eig_sets_match(before, after.clone(), 1e-9 * n as f64, &ctx);

        // the leading m eigenvalues are exactly the selected ones
        let tol = 1e-9 * n as f64;
        for (k, w) in after.iter().enumerate() {
            if k < m {
                assert!(w.0 > -tol, "{ctx}: leading eigenvalue {k} has re {}", w.0);
            } else {
                assert!(w.0 < tol, "{ctx}: trailing eigenvalue {k} has re {}", w.0);
            }
        }
    }
}

#[test]
fn real_reorder_move() {
    let n = 12;
    let a = fill(n, 0x94D049BB133111EB);
    let s = real_schur(a.as_ref(), Par::Seq).unwrap();
    let mut t = s.t.clone();
    let mut z = s.z.clone();
    let before = quasi_tri_eigenvalues(t.as_ref());
    // move the block containing the last row to the top
    let dst = real_schur_move(t.as_mut(), Some(z.as_mut()), n - 1, 0).unwrap();
    assert!(dst <= 1, "landed at {dst}");
    let ctx = "real move";
    check_real_factorization(a.as_ref(), t.as_ref(), z.as_ref(), ctx);
    assert_eig_sets_match(
        before,
        quasi_tri_eigenvalues(t.as_ref()),
        1e-10 * n as f64,
        ctx,
    );
}

#[test]
fn complex_reorder() {
    for &n in &[2usize, 3, 5, 8, 16, 33, 64] {
        let re = fill(n, 0xBF58476D1CE4E5B9 ^ (n as u64));
        let im = fill(n, 0x9E3779B97F4A7C15 ^ (n as u64));
        let a = Mat::from_fn(n, n, |i, j| c64::new(re[(i, j)], im[(i, j)]));
        let s = complex_schur(a.as_ref(), Par::Seq).unwrap();
        let mut t = s.t.clone();
        let mut z = s.z.clone();
        let before: Vec<_> = (0..n).map(|k| (s.w[k].re, s.w[k].im)).collect();

        // select the upper half plane
        let select: Vec<bool> = (0..n).map(|k| s.w[k].im > 0.0).collect();
        let expected_m = select.iter().filter(|&&x| x).count();
        let m = complex_schur_select(t.as_mut(), Some(z.as_mut()), &select).unwrap();
        assert_eq!(m, expected_m, "n={n}: m");

        let ctx = format!("complex reorder n={n}");
        check_complex_factorization(a.as_ref(), t.as_ref(), z.as_ref(), &ctx);
        let after: Vec<_> = (0..n).map(|k| (t[(k, k)].re, t[(k, k)].im)).collect();
        let tol = 1e-9 * n as f64;
        for (k, w) in after.iter().enumerate() {
            if k < m {
                assert!(w.1 > -tol, "{ctx}: leading eigenvalue {k} has im {}", w.1);
            } else {
                assert!(w.1 < tol, "{ctx}: trailing eigenvalue {k} has im {}", w.1);
            }
        }
        assert_eig_sets_match(before, after, tol, &ctx);

        // single move round-trip
        if n >= 3 {
            let w_last = t[(n - 1, n - 1)];
            complex_schur_move(t.as_mut(), Some(z.as_mut()), n - 1, 0).unwrap();
            let d = t[(0, 0)] - w_last;
            assert!(d.re.hypot(d.im) < tol, "moved eigenvalue changed");
            check_complex_factorization(a.as_ref(), t.as_ref(), z.as_ref(), "complex move");
        }
    }
}

#[test]
fn rejects_non_finite() {
    let mut a = fill(4, 1);
    a[(2, 1)] = f64::NAN;
    assert!(matches!(
        real_schur(a.as_ref(), Par::Seq),
        Err(faer_schur::SchurError::NonFinite)
    ));
}
