//! Property gate for the wasm-shaped LU: ‖P·A − L·U‖ residual, solve
//! residual, and cross-check against faer's own solver — across sizes that
//! hit the panel-only path, block boundaries, odd sizes, and multi-block.

use faer::prelude::*;
use faer::Mat;
use faer_wasm_kernels::lu::{lu_factor_in_place, lu_factor_recursive_in_place, lu_solve_in_place};

fn fill(n: usize, mut s: u64) -> Mat<f64> {
    Mat::from_fn(n, n, |_, _| {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((s >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    })
}

/// Shared checks for any factorization with the LAPACK ipiv contract:
/// ‖P·A − L·U‖, solve residual, agreement with faer's own solver.
fn check_factors(a: &Mat<f64>, f: &Mat<f64>, piv: &[usize], what: &str) {
    let n = a.nrows();
    // reconstruct L and U
    let l = Mat::from_fn(n, n, |i, j| {
        if i == j {
            1.0
        } else if i > j {
            f[(i, j)]
        } else {
            0.0
        }
    });
    let u = Mat::from_fn(n, n, |i, j| if i <= j { f[(i, j)] } else { 0.0 });
    // P·A: apply recorded swaps in order
    let mut pa = a.clone();
    for k in 0..n {
        if piv[k] != k {
            for c in 0..n {
                let t = pa[(k, c)];
                pa[(k, c)] = pa[(piv[k], c)];
                pa[(piv[k], c)] = t;
            }
        }
    }
    let lu = &l * &u;
    let mut err = 0.0f64;
    for j in 0..n {
        for i in 0..n {
            err = err.max((pa[(i, j)] - lu[(i, j)]).abs());
        }
    }
    let tol = 1e-13 * (n.max(4) as f64);
    assert!(err < tol, "{what}: ||PA-LU|| = {err:.2e} >= {tol:.2e}");

    // solve residual + agreement with faer
    let b = fill(n, 0xD1B54A32D192ED03 ^ (n as u64));
    let mut x = (0..n).map(|i| b[(i, 0)]).collect::<Vec<f64>>();
    lu_solve_in_place(f.as_ref(), piv, &mut x);
    let xm = Mat::from_fn(n, 1, |i, _| x[i]);
    let r = a * &xm;
    let mut rerr = 0.0f64;
    for i in 0..n {
        rerr = rerr.max((r[(i, 0)] - b[(i, 0)]).abs());
    }
    // random square matrices can be mildly ill-conditioned; residual
    // scales with cond(A), so keep a generous but meaningful bound
    assert!(
        rerr < 1e-9 * (n.max(4) as f64),
        "{what}: solve residual {rerr:.2e}"
    );

    let bf = Mat::from_fn(n, 1, |i, _| b[(i, 0)]);
    let xf = a.partial_piv_lu().solve(&bf);
    let mut derr = 0.0f64;
    let mut scale = 0.0f64;
    for i in 0..n {
        derr = derr.max((xf[(i, 0)] - x[i]).abs());
        scale = scale.max(xf[(i, 0)].abs());
    }
    assert!(
        derr < 1e-8 * scale.max(1.0),
        "{what}: disagrees with faer by {derr:.2e} (scale {scale:.2e})"
    );
}

#[test]
fn factorization_and_solve() {
    for &n in &[1usize, 2, 3, 5, 8, 31, 32, 33, 64, 96, 257, 512] {
        for &nb in &[0usize, 8, 33] {
            let a = fill(n, 0x9E3779B97F4A7C15 ^ (n as u64) ^ ((nb as u64) << 32));
            let mut f = a.clone();
            let mut piv = vec![0usize; n];
            lu_factor_in_place(f.as_mut(), &mut piv, nb);
            check_factors(&a, &f, &piv, &format!("blocked n={n} nb={nb}"));
        }
    }
}

#[test]
fn recursive_factorization_and_solve() {
    for &n in &[1usize, 2, 3, 5, 8, 31, 32, 33, 64, 96, 257, 512] {
        for &co in &[0usize, 8, 33] {
            let a = fill(n, 0xC2B2AE3D27D4EB4F ^ (n as u64) ^ ((co as u64) << 32));
            let mut f = a.clone();
            let mut piv = vec![0usize; n];
            lu_factor_recursive_in_place(f.as_mut(), &mut piv, co);
            check_factors(&a, &f, &piv, &format!("recursive n={n} crossover={co}"));
        }
    }
}

/// The recursive and blocked factorizations must produce *identical* pivots
/// and factors — same pivoting rule, different evaluation order only up to
/// gemm reassociation.
#[test]
fn recursive_matches_blocked_pivots() {
    for &n in &[33usize, 96, 257] {
        let a = fill(n, 0xA0761D6478BD642F ^ (n as u64));
        let mut fb = a.clone();
        let mut fr = a.clone();
        let mut pb = vec![0usize; n];
        let mut pr = vec![0usize; n];
        lu_factor_in_place(fb.as_mut(), &mut pb, 0);
        lu_factor_recursive_in_place(fr.as_mut(), &mut pr, 0);
        assert_eq!(pb, pr, "n={n}: pivot sequences differ");
        let mut derr = 0.0f64;
        for j in 0..n {
            for i in 0..n {
                derr = derr.max((fb[(i, j)] - fr[(i, j)]).abs());
            }
        }
        assert!(derr < 1e-11, "n={n}: factors differ by {derr:.2e}");
    }
}

