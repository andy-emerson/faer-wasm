//! f32 correctness gate for the generic kernels (f32/c32 phase): the same
//! invariants as the f64 suites at f32-appropriate tolerances (eps ≈ 6e-8;
//! residual bounds scale with eps·n). The reference is faer's own f32
//! paths, so kernel-vs-faer agreement is precision-consistent.

use faer::prelude::*;
use faer::Mat;
use faer_wasm_kernels::hessenberg::hessenberg_factor_in_place;
use faer_wasm_kernels::lu::{lu_factor_in_place, lu_solve_in_place};
use faer_wasm_kernels::qr::qr_factor_in_place;
use faer_wasm_kernels::schur_small::hqr_eigvals_in_place;

fn fill(n: usize, mut s: u64) -> Mat<f32> {
    Mat::from_fn(n, n, |_, _| {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (((s >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0) as f32
    })
}

#[test]
fn lu_f32_factorization_and_solve() {
    for &n in &[4usize, 16, 33, 64, 128] {
        let a = fill(n, 0x9E3779B97F4A7C15 ^ (n as u64));
        let mut f = a.clone();
        let mut piv = vec![0usize; n];
        lu_factor_in_place(f.as_mut(), &mut piv, 0);

        // reconstruct P·A and compare against L·U
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
        let mut lu = Mat::<f32>::zeros(n, n);
        for j in 0..n {
            for i in 0..n {
                let mut s = 0.0f32;
                for k in 0..=usize::min(i, j) {
                    let l = if i == k { 1.0 } else if i > k { f[(i, k)] } else { 0.0 };
                    let u = if k <= j { f[(k, j)] } else { 0.0 };
                    s += l * u;
                }
                lu[(i, j)] = s;
            }
        }
        let mut err = 0.0f32;
        for j in 0..n {
            for i in 0..n {
                err = err.max((pa[(i, j)] - lu[(i, j)]).abs());
            }
        }
        assert!(err < 1e-4 * (n as f32), "n={n}: ||PA-LU|| = {err:.2e}");

        // solve against a known x
        let xtrue: Vec<f32> = (0..n).map(|i| ((i % 7) as f32) - 3.0).collect();
        let mut b = vec![0.0f32; n];
        for i in 0..n {
            let mut s = 0.0f32;
            for j in 0..n {
                s += a[(i, j)] * xtrue[j];
            }
            b[i] = s;
        }
        lu_solve_in_place(f.as_ref(), &piv, &mut b);
        let mut xerr = 0.0f32;
        for i in 0..n {
            xerr = xerr.max((b[i] - xtrue[i]).abs());
        }
        assert!(xerr < 5e-3 * (n as f32), "n={n}: solve error {xerr:.2e}");
    }
}

#[test]
fn qr_f32_backward_error() {
    for &n in &[4usize, 16, 33, 64, 128] {
        let a = fill(n, 0xD1B54A32D192ED03 ^ (n as u64));
        let mut f = a.clone();
        let mut tau = vec![0.0f32; n];
        qr_factor_in_place(f.as_mut(), &mut tau);

        // form Q by applying reflectors to I, then check ||A - QR||
        let mut q = Mat::<f32>::identity(n, n);
        for j in (0..n).rev() {
            let tj = tau[j];
            if tj == 0.0 {
                continue;
            }
            for c in 0..n {
                let mut w = q[(j, c)];
                for i in j + 1..n {
                    w += f[(i, j)] * q[(i, c)];
                }
                w *= tj;
                q[(j, c)] -= w;
                for i in j + 1..n {
                    q[(i, c)] -= f[(i, j)] * w;
                }
            }
        }
        let r = Mat::from_fn(n, n, |i, j| if i <= j { f[(i, j)] } else { 0.0 });
        let qr = &q * &r;
        let mut err = 0.0f32;
        for j in 0..n {
            for i in 0..n {
                err = err.max((a[(i, j)] - qr[(i, j)]).abs());
            }
        }
        assert!(err < 1e-4 * (n as f32), "n={n}: ||A-QR|| = {err:.2e}");
    }
}

#[test]
fn eigvals_f32_pipeline_matches_f64_truth() {
    for &n in &[4usize, 16, 32, 64, 96, 128] {
        let a32 = fill(n, 0xA24BAED4963EE407 ^ (n as u64));
        // f64 truth on the SAME matrix values
        let a64 = Mat::from_fn(n, n, |i, j| a32[(i, j)] as f64);

        let mut h = a32.clone();
        let mut tau = vec![0.0f32; n.saturating_sub(2).max(1)];
        let mut work = vec![0.0f32; n];
        hessenberg_factor_in_place(h.as_mut(), &mut tau, &mut work);
        for j in 0..n {
            for i in j + 2..n {
                h[(i, j)] = 0.0;
            }
        }
        let mut w_re = vec![0.0f32; n];
        let mut w_im = vec![0.0f32; n];
        let info = hqr_eigvals_in_place(h.as_mut(), &mut w_re, &mut w_im);
        assert!(info == 0, "n={n}: f32 hqr did not converge");

        let fe: Vec<faer::c64> = a64.eigenvalues().unwrap();
        let mut mine: Vec<(f64, f64)> =
            (0..n).map(|i| (w_re[i] as f64, w_im[i] as f64)).collect();
        let mut truth: Vec<(f64, f64)> = fe.iter().map(|z| (z.re, z.im)).collect();
        let cmp = |a: &(f64, f64), b: &(f64, f64)| {
            a.0.partial_cmp(&b.0).unwrap().then(a.1.partial_cmp(&b.1).unwrap())
        };
        mine.sort_by(cmp);
        truth.sort_by(cmp);
        // eigenvalue perturbation for f32 arithmetic scales with eps_32 * ||A||
        // times conditioning; random dense at these sizes stays comfortably
        // under this loose-but-meaningful bound
        for i in 0..n {
            let d = ((mine[i].0 - truth[i].0).powi(2) + (mine[i].1 - truth[i].1).powi(2)).sqrt();
            assert!(
                d < 2e-3 * (n as f64).max(4.0),
                "n={n}: eigenvalue {i} off f64 truth by {d:.2e}"
            );
        }
    }
}
