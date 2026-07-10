//! Correctness gate for the one-sided Jacobi SVD probe: reconstruction
//! ‖A − U·Σ·Vᵀ‖, orthogonality of U and V, and singular-value agreement with
//! faer's own SVD — plus it prints sweeps-to-convergence per size (the number
//! that decides whether the full build is worth it).

use faer::prelude::*;
use faer::Mat;
use faer_wasm_kernels::svd::jacobi_svd_in_place;

fn fill(m: usize, n: usize, mut s: u64) -> Mat<f64> {
    Mat::from_fn(m, n, |_, _| {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((s >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    })
}

#[test]
fn jacobi_svd_correct_and_sweep_counts() {
    for &n in &[2usize, 3, 4, 8, 16, 31, 32, 33, 64, 96, 128, 256] {
        let a = fill(n, n, 0x243F6A8885A308D3 ^ (n as u64));
        let mut u = a.clone(); // overwritten with U
        let mut v = Mat::<f64>::zeros(n, n);
        let mut s = vec![0.0f64; n];
        let sweeps = jacobi_svd_in_place(u.as_mut(), v.as_mut(), &mut s, 60, 1e-14);
        println!("n={n}: {sweeps} sweeps");
        assert!(sweeps < 60, "n={n}: did not converge (hit sweep cap)");

        // reconstruction ‖A − U·diag(s)·Vᵀ‖_max
        let us = Mat::from_fn(n, n, |i, j| u[(i, j)] * s[j]);
        let usvt = &us * v.transpose();
        let mut rerr = 0.0f64;
        let mut scale = 0.0f64;
        for j in 0..n {
            for i in 0..n {
                rerr = rerr.max((a[(i, j)] - usvt[(i, j)]).abs());
                scale = scale.max(a[(i, j)].abs());
            }
        }
        assert!(
            rerr < 1e-11 * scale.max(1.0) * (n as f64),
            "n={n}: ||A-UΣVᵀ|| = {rerr:.2e}"
        );

        // orthogonality of U and V
        for (name, q) in [("U", &u), ("V", &v)] {
            let mut oerr = 0.0f64;
            for c in 0..n {
                for r in 0..n {
                    let mut d = 0.0;
                    for i in 0..n {
                        d += q[(i, r)] * q[(i, c)];
                    }
                    oerr = oerr.max((d - if r == c { 1.0 } else { 0.0 }).abs());
                }
            }
            assert!(oerr < 1e-10 * (n as f64), "n={n}: {name}ᵀ{name}-I = {oerr:.2e}");
        }

        // singular values agree with faer (both sorted descending)
        let mut mine = s.clone();
        mine.sort_by(|x, y| y.partial_cmp(x).unwrap());
        let fsvd = a.svd().unwrap();
        let fs = fsvd.S();
        let mut serr = 0.0f64;
        for i in 0..n {
            serr = serr.max((mine[i] - fs[i]).abs());
        }
        assert!(
            serr < 1e-9 * mine[0].max(1.0),
            "n={n}: singular values disagree with faer by {serr:.2e}"
        );
    }
}
