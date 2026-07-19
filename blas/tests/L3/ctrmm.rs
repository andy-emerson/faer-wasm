use super::common::*;
use faer_wasm_blas::L3::*;
use faer_wasm_blas::C32;

#[test]
fn ctrmm_both_sides_bounded() {
	let mut rng = Lcg(135);
	let alpha = C32::new(0.9, -0.2);
	for &n in NS {
		let m = n;
		for upper in [true, false] {
			for unit in [true, false] {
				let acs = n + 1;
				let a = rng.mat_c32(n, n, acs);
				let bcs = m + 2;
				let b0 = rng.mat_c32(m, n, bcs);
				let tri_at = |i: usize, j: usize| -> C32 {
					let in_tri = if upper { i <= j } else { i >= j };
					if !in_tri {
						C32::ZERO
					} else if unit && i == j {
						C32::ONE
					} else {
						a[j * acs + i]
					}
				};

				let mut b = b0.clone();
				ctrmm_left(alpha, m, n, &a, acs, upper, unit, &mut b, bcs);
				for j in 0..n {
					for i in 0..m {
						let want =
							c_up(alpha) * comp_sum_cc((0..n).map(|l| tri_at(i, l) * b0[j * bcs + l]));
						let scale = comp_scale_cc((0..n).map(|l| tri_at(i, l) * b0[j * bcs + l]))
							* (alpha.abs1() as f64 + 1.0);
						let tol = EPS * (n.max(1) as f64) * 16.0 * scale + 1e-30;
						assert!(
							(b[j * bcs + i].re as f64 - want.re).abs()
								+ (b[j * bcs + i].im as f64 - want.im).abs()
								<= tol,
							"ctrmm_left upper={upper} unit={unit} n={n} ({i},{j})"
						);
					}
				}

				let mut b = b0.clone();
				ctrmm_right(alpha, m, n, &a, acs, upper, unit, &mut b, bcs);
				for j in 0..n {
					for i in 0..m {
						let want =
							c_up(alpha) * comp_sum_cc((0..n).map(|l| b0[l * bcs + i] * tri_at(l, j)));
						let scale = comp_scale_cc((0..n).map(|l| b0[l * bcs + i] * tri_at(l, j)))
							* (alpha.abs1() as f64 + 1.0);
						let tol = EPS * (n.max(1) as f64) * 16.0 * scale + 1e-30;
						assert!(
							(b[j * bcs + i].re as f64 - want.re).abs()
								+ (b[j * bcs + i].im as f64 - want.im).abs()
								<= tol,
							"ctrmm_right upper={upper} unit={unit} n={n} ({i},{j})"
						);
					}
				}
			}
		}
	}
}

#[test]
fn ctrmm_bit_replay_both_sides() {
	let mut rng = Lcg(137);
	let alpha = C32::new(0.9, -0.2);
	for &(m, n) in TRI_DIMS {
		for upper in [true, false] {
			for unit in [true, false] {
				let bcs = m + 2;

				// left: B <- α·A·B, plain ctrmv-per-column replay then αscal
				let acs = m + 1;
				let a = rng.mat_c32(m, m, acs);
				let b0 = rng.mat_c32(m, n, bcs);
				let mut bt = b0.clone();
				ctrmm_left(alpha, m, n, &a, acs, upper, unit, &mut bt, bcs);
				let mut br = b0.clone();
				for j in 0..n {
					if upper {
						for l in 0..m {
							let t = br[j * bcs + l];
							for i in 0..l {
								br[j * bcs + i] = br[j * bcs + i] + t * a[l * acs + i];
							}
							if !unit {
								br[j * bcs + l] = t * a[l * acs + l];
							}
						}
					} else {
						for l in (0..m).rev() {
							let t = br[j * bcs + l];
							for i in l + 1..m {
								br[j * bcs + i] = br[j * bcs + i] + t * a[l * acs + i];
							}
							if !unit {
								br[j * bcs + l] = t * a[l * acs + l];
							}
						}
					}
					for i in 0..m {
						br[j * bcs + i] = alpha * br[j * bcs + i];
					}
				}
				for (x, y) in bt.iter().zip(&br) {
					if !x.re.is_nan() {
						assert!(bits_eq_cc(*x, *y), "ctrmm_left {m}x{n} u={upper}");
					}
				}

				// right: B <- α·B·A
				let acs = n + 1;
				let a = rng.mat_c32(n, n, acs);
				let b0 = rng.mat_c32(m, n, bcs);
				let mut bt = b0.clone();
				ctrmm_right(alpha, m, n, &a, acs, upper, unit, &mut bt, bcs);
				let mut br = b0.clone();
				let dcol = |br: &mut Vec<C32>, j: usize, lo: usize, hi: usize| {
					let d = if unit { C32::ONE } else { a[j * acs + j] };
					let s = alpha * d;
					for i in 0..m {
						br[j * bcs + i] = s * br[j * bcs + i];
					}
					for k in lo..hi {
						let t = alpha * a[j * acs + k];
						for i in 0..m {
							br[j * bcs + i] = br[j * bcs + i] + t * br[k * bcs + i];
						}
					}
				};
				if upper {
					// grouped replay: in-group (descending targets,
					// ascending k) first, then out-of-group sources
					let r = n % 4;
					let mut gs = n;
					while gs >= r + 4 {
						gs -= 4;
						for tc in (gs..gs + 4).rev() {
							dcol(&mut br, tc, gs, tc);
						}
						for k in 0..gs {
							for u in 0..4 {
								let t = alpha * a[(gs + u) * acs + k];
								for i in 0..m {
									br[(gs + u) * bcs + i] =
										br[(gs + u) * bcs + i] + t * br[k * bcs + i];
								}
							}
						}
					}
					for j in (0..r).rev() {
						dcol(&mut br, j, 0, j);
					}
				} else {
					// plain ascending replay — the grouping keeps lower
					// fully ascending, so this asserts bit-identity to it
					for j in 0..n {
						dcol(&mut br, j, j + 1, n);
					}
				}
				for (x, y) in bt.iter().zip(&br) {
					if !x.re.is_nan() {
						assert!(bits_eq_cc(*x, *y), "ctrmm_right {m}x{n} u={upper}");
					}
				}
			}
		}
	}
}
