//! Correctness gate for the wasm-shaped Hessenberg kernel: reconstruct Q
//! from the stored reflectors and require (a) similarity ‖A·Q − Q·H‖ small,
//! (b) Q orthogonal, (c) H genuinely upper Hessenberg, (d) eigenvalues of H
//! (via faer's full EVD of the reduced matrix) match eigenvalues of A —
//! the property the eigvals pipeline actually consumes.

use faer::prelude::*;
use faer::Mat;
use faer_wasm_kernels::hessenberg::hessenberg_factor_in_place;

fn fill(n: usize, mut s: u64) -> Mat<f64> {
    Mat::from_fn(n, n, |_, _| {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((s >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    })
}

#[test]
fn hessenberg_similarity_orthogonality_eigenvalues() {
    for &n in &[2usize, 3, 4, 8, 16, 33, 64, 96, 128] {
        let a = fill(n, 0x853C49E6748FEA9B ^ (n as u64));
        let mut fac = a.clone();
        let k = n.saturating_sub(2);
        let mut tau = vec![0.0f64; k.max(1)];
        let mut work = vec![0.0f64; n];
        hessenberg_factor_in_place(fac.as_mut(), &mut tau, &mut work);

        // H = upper-Hessenberg part of fac (below-subdiagonal holds v's)
        let h = Mat::from_fn(n, n, |i, j| if i > j + 1 { 0.0 } else { fac[(i, j)] });

        // Q = H_0 · H_1 · … applied to I from the right
        let mut q = Mat::<f64>::identity(n, n);
        for j in 0..k {
            let tj = tau[j];
            if tj == 0.0 {
                continue;
            }
            // v supported on rows j+1..n: v[j+1] = 1, tail from fac
            let mut v = vec![0.0f64; n];
            v[j + 1] = 1.0;
            for t in 0..n - j - 2 {
                v[j + 2 + t] = fac[(j + 2 + t, j)];
            }
            // Q := Q - tj * (Q v) v^T
            let mut qv = vec![0.0f64; n];
            for r in 0..n {
                let mut s = 0.0;
                for c in j + 1..n {
                    s += q[(r, c)] * v[c];
                }
                qv[r] = s;
            }
            for c in j + 1..n {
                let f = tj * v[c];
                for r in 0..n {
                    q[(r, c)] -= qv[r] * f;
                }
            }
        }

        // similarity: ‖A·Q − Q·H‖_max
        let aq = &a * &q;
        let qh = &q * &h;
        let mut serr = 0.0f64;
        let mut scale = 0.0f64;
        for j in 0..n {
            for i in 0..n {
                serr = serr.max((aq[(i, j)] - qh[(i, j)]).abs());
                scale = scale.max(a[(i, j)].abs());
            }
        }
        assert!(
            serr < 1e-12 * scale.max(1.0) * (n as f64),
            "n={n}: ||AQ - QH|| = {serr:.2e}"
        );

        // orthogonality
        let qtq = q.transpose() * &q;
        let mut oerr = 0.0f64;
        for j in 0..n {
            for i in 0..n {
                oerr = oerr.max((qtq[(i, j)] - if i == j { 1.0 } else { 0.0 }).abs());
            }
        }
        assert!(oerr < 1e-12 * (n as f64), "n={n}: QᵀQ-I = {oerr:.2e}");

        // eigenvalues preserved (sorted complex compare vs faer's EVD of A)
        let ea: Vec<faer::c64> = a.eigenvalues().unwrap();
        let eh: Vec<faer::c64> = h.eigenvalues().unwrap();
        let key = |z: &faer::c64| (z.re, z.im);
        let mut ea: Vec<_> = ea.iter().map(key).collect();
        let mut eh: Vec<_> = eh.iter().map(key).collect();
        let cmp = |a: &(f64, f64), b: &(f64, f64)| {
            a.0.partial_cmp(&b.0).unwrap().then(a.1.partial_cmp(&b.1).unwrap())
        };
        ea.sort_by(cmp);
        eh.sort_by(cmp);
        for i in 0..n {
            let d = ((ea[i].0 - eh[i].0).powi(2) + (ea[i].1 - eh[i].1).powi(2)).sqrt();
            assert!(
                d < 1e-9 * (n as f64),
                "n={n}: eigenvalue {i} moved by {d:.2e}"
            );
        }
    }
}
