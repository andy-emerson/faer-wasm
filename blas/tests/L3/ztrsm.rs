use super::common::*;
use faer_wasm_blas::L3::*;
use faer_wasm_blas::C64;

#[test]
fn ztrsm_both_sides_residual() {
	let mut rng = Lcg(138);
	let alpha = C64::new(0.9, 0.3);
	for &n in NS {
		let m = n;
		for upper in [true, false] {
			for unit in [true, false] {
				let acs = n + 1;
				// dominant complex diagonal exercises Smith division
				let mut a = rng.mat_c64(n, n, acs);
				for j in 0..n {
					a[j * acs + j] = C64::new(2.0 * (n as f64) + 1.0 + j as f64, 0.7);
				}
				let bcs = m + 1;
				let b0 = rng.mat_c64(m, n, bcs);
				let tri_at = |i: usize, j: usize| -> C64 {
					let in_tri = if upper { i <= j } else { i >= j };
					if !in_tri {
						C64::ZERO
					} else if unit && i == j {
						C64::ONE
					} else {
						a[j * acs + i]
					}
				};
				let bound = 4.0 * (n as f64 + 1.0);

				// left: A·X = α·B0
				let mut x = b0.clone();
				ztrsm_left(alpha, m, n, &a, acs, upper, unit, &mut x, bcs);
				for j in 0..n {
					let xmax =
						(0..n).fold(0.0f64, |acc, l| acc.max(x[j * bcs + l].abs1()));
					for i in 0..m {
						let ax = comp_sum_c((0..n).map(|l| tri_at(i, l) * x[j * bcs + l]));
						let want = alpha * b0[j * bcs + i];
						let tol =
							f64::EPSILON * (n.max(1) as f64) * 16.0 * bound * xmax + 1e-300;
						assert!(
							(ax.re - want.re).abs() + (ax.im - want.im).abs() <= tol,
							"ztrsm_left residual upper={upper} unit={unit} n={n} ({i},{j})"
						);
					}
				}

				// right: X·A = α·B0
				let mut x = b0.clone();
				ztrsm_right(alpha, m, n, &a, acs, upper, unit, &mut x, bcs);
				for j in 0..n {
					for i in 0..m {
						let xa = comp_sum_c((0..n).map(|l| x[l * bcs + i] * tri_at(l, j)));
						let want = alpha * b0[j * bcs + i];
						let xmax =
							(0..n).fold(0.0f64, |acc, l| acc.max(x[l * bcs + i].abs1()));
						let tol =
							f64::EPSILON * (n.max(1) as f64) * 16.0 * bound * xmax + 1e-300;
						assert!(
							(xa.re - want.re).abs() + (xa.im - want.im).abs() <= tol,
							"ztrsm_right residual upper={upper} unit={unit} n={n} ({i},{j})"
						);
					}
				}
			}
		}
	}
}

#[test]
fn ztrsm_bit_replay_both_sides() {
	let mut rng = Lcg(139);
	let alpha = C64::new(0.9, 0.3);
	for &(m, n) in TRI_DIMS {
		for upper in [true, false] {
			for unit in [true, false] {
				let bcs = m + 2;

				// left: B <- α·inv(A)·B, plain αscal-then-ztrsv replay
				let acs = m + 1;
				let mut a = rng.mat_c64(m, m, acs);
				for j in 0..m {
					a[j * acs + j] = C64::new(2.0 * (m as f64) + 1.0 + j as f64, 0.7);
				}
				let b0 = rng.mat_c64(m, n, bcs);
				let mut bt = b0.clone();
				ztrsm_left(alpha, m, n, &a, acs, upper, unit, &mut bt, bcs);
				let mut br = b0.clone();
				for j in 0..n {
					for i in 0..m {
						br[j * bcs + i] = alpha * br[j * bcs + i];
					}
					if upper {
						for l in (0..m).rev() {
							if !unit {
								br[j * bcs + l] = br[j * bcs + l] / a[l * acs + l];
							}
							let t = -br[j * bcs + l];
							for i in 0..l {
								br[j * bcs + i] = br[j * bcs + i] + t * a[l * acs + i];
							}
						}
					} else {
						for l in 0..m {
							if !unit {
								br[j * bcs + l] = br[j * bcs + l] / a[l * acs + l];
							}
							let t = -br[j * bcs + l];
							for i in l + 1..m {
								br[j * bcs + i] = br[j * bcs + i] + t * a[l * acs + i];
							}
						}
					}
				}
				for (x, y) in bt.iter().zip(&br) {
					if !x.re.is_nan() {
						assert!(bits_eq_c(*x, *y), "ztrsm_left {m}x{n} u={upper}");
					}
				}

				// right: B <- α·B·inv(A)
				let acs = n + 1;
				let mut a = rng.mat_c64(n, n, acs);
				for j in 0..n {
					a[j * acs + j] = C64::new(2.0 * (n as f64) + 1.0 + j as f64, 0.7);
				}
				let b0 = rng.mat_c64(m, n, bcs);
				let mut bt = b0.clone();
				ztrsm_right(alpha, m, n, &a, acs, upper, unit, &mut bt, bcs);
				let mut br = b0.clone();
				let elim = |br: &mut Vec<C64>, j: usize, k: usize| {
					let t = -a[j * acs + k];
					for i in 0..m {
						br[j * bcs + i] = br[j * bcs + i] + t * br[k * bcs + i];
					}
				};
				let finish = |br: &mut Vec<C64>, j: usize| {
					if !unit {
						let s = C64::ONE / a[j * acs + j];
						for i in 0..m {
							br[j * bcs + i] = s * br[j * bcs + i];
						}
					}
				};
				if upper {
					// plain ascending replay — the grouping keeps upper
					// fully ascending, so this asserts bit-identity to it
					for j in 0..n {
						for i in 0..m {
							br[j * bcs + i] = alpha * br[j * bcs + i];
						}
						for k in 0..j {
							elim(&mut br, j, k);
						}
						finish(&mut br, j);
					}
				} else {
					// grouped replay: out-of-group solved columns first,
					// then in-group elimination descending
					let r = n % 4;
					let mut gs = n;
					while gs >= r + 4 {
						gs -= 4;
						for u in 0..4 {
							for i in 0..m {
								br[(gs + u) * bcs + i] = alpha * br[(gs + u) * bcs + i];
							}
						}
						for k in gs + 4..n {
							for u in 0..4 {
								elim(&mut br, gs + u, k);
							}
						}
						for tc in (gs..gs + 4).rev() {
							for k in tc + 1..gs + 4 {
								elim(&mut br, tc, k);
							}
							finish(&mut br, tc);
						}
					}
					for j in (0..r).rev() {
						for i in 0..m {
							br[j * bcs + i] = alpha * br[j * bcs + i];
						}
						for k in j + 1..n {
							elim(&mut br, j, k);
						}
						finish(&mut br, j);
					}
				}
				for (x, y) in bt.iter().zip(&br) {
					if !x.re.is_nan() {
						assert!(bits_eq_c(*x, *y), "ztrsm_right {m}x{n} u={upper}");
					}
				}
			}
		}
	}
}
