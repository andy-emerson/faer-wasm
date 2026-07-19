//! Level 3 correctness per the testing contract (../README.md):
//! bit-for-bit against same-order scalar references (every op is pure
//! column-axpy / divide-then-column-axpy), independent
//! compensated-summation bounds computed in a different accumulation
//! order, and residual checks for both trsm sides on diagonally
//! dominant systems.

use faer_wasm_blas::level3::*;

struct Lcg(u64);
impl Lcg {
	fn next_f64(&mut self) -> f64 {
		self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
		let bits = (self.0 >> 11) as f64 / (1u64 << 53) as f64;
		4.0 * bits - 2.0
	}
	fn vec(&mut self, n: usize) -> Vec<f64> {
		(0..n).map(|_| self.next_f64()).collect()
	}
	fn mat(&mut self, nrows: usize, ncols: usize, cs: usize) -> Vec<f64> {
		assert!(cs >= nrows);
		let mut a = vec![f64::NAN; if ncols == 0 { 0 } else { cs * (ncols - 1) + nrows }];
		for j in 0..ncols {
			for i in 0..nrows {
				a[j * cs + i] = self.next_f64();
			}
		}
		a
	}
}

fn comp_sum(it: impl Iterator<Item = f64>) -> f64 {
	let mut s = 0.0f64;
	let mut c = 0.0f64;
	for v in it {
		let t = s + v;
		if s.abs() >= v.abs() {
			c += (s - t) + v;
		} else {
			c += (v - t) + s;
		}
		s = t;
	}
	s + c
}

const DIMS: &[(usize, usize, usize)] =
	&[(0, 0, 0), (1, 1, 1), (3, 2, 4), (5, 5, 5), (8, 3, 7), (17, 12, 9), (33, 33, 33)];
const NS: &[usize] = &[0, 1, 2, 3, 5, 8, 17, 33];

#[test]
fn gemm_bit_for_bit_and_bounded() {
	let mut rng = Lcg(31);
	for &(m, k, n) in DIMS {
		let (acs, bcs, ccs) = (m + 1, k + 2, m + 3);
		let a = rng.mat(m, k, acs);
		let b = rng.mat(k, n, bcs);
		let c0 = rng.mat(m, n, ccs);
		for (alpha, beta) in [(1.0, 0.0), (-0.7, 0.4)] {
			let mut c = c0.clone();
			gemm(alpha, m, k, n, &a, acs, &b, bcs, beta, &mut c, ccs);
			// same-order scalar replay (gemv per column = column-axpy)
			let mut cr = c0.clone();
			for j in 0..n {
				if beta == 0.0 {
					for i in 0..m {
						cr[j * ccs + i] = 0.0;
					}
				} else if beta != 1.0 {
					for i in 0..m {
						cr[j * ccs + i] *= beta;
					}
				}
				for l in 0..k {
					let t = alpha * b[j * bcs + l];
					for i in 0..m {
						cr[j * ccs + i] += a[l * acs + i] * t;
					}
				}
			}
			for j in 0..n {
				for i in 0..m {
					assert_eq!(
						c[j * ccs + i].to_bits(),
						cr[j * ccs + i].to_bits(),
						"gemm bits {m}x{k}x{n} ({i},{j})"
					);
				}
			}
			// independent bound, different accumulation order
			for j in 0..n {
				for i in 0..m {
					let want = alpha * comp_sum((0..k).map(|l| a[l * acs + i] * b[j * bcs + l]))
						+ if beta == 0.0 { 0.0 } else { beta * c0[j * ccs + i] };
					let scale = comp_sum((0..k).map(|l| (a[l * acs + i] * b[j * bcs + l]).abs()))
						+ c0[j * ccs + i].abs();
					let tol = f64::EPSILON * (k.max(1) as f64) * 8.0 * scale + 1e-300;
					assert!((c[j * ccs + i] - want).abs() <= tol, "gemm bound ({i},{j})");
				}
			}
		}
	}
}

#[test]
fn symm_both_sides_bounded() {
	let mut rng = Lcg(32);
	for &n in NS {
		let m = n; // square B keeps the reference simple
		for upper in [true, false] {
			let acs = n + 1;
			// full symmetric ground truth, one triangle exposed
			let mut full = vec![0.0; n * n];
			for j in 0..n {
				for i in 0..=j {
					let v = rng.next_f64();
					full[j * n + i] = v;
					full[i * n + j] = v;
				}
			}
			let mut a = vec![f64::NAN; if n == 0 { 0 } else { acs * (n - 1) + n }];
			for j in 0..n {
				for i in 0..n {
					if if upper { i <= j } else { i >= j } {
						a[j * acs + i] = full[j * n + i];
					}
				}
			}
			let (bcs, ccs) = (m + 2, m + 1);
			let b = rng.mat(m, n, bcs);
			let c0 = rng.mat(m, n, ccs);

			let mut c = c0.clone();
			symm_left(0.8, m, n, &a, acs, upper, &b, bcs, 0.3, &mut c, ccs);
			for j in 0..n {
				for i in 0..m {
					let want = 0.8 * comp_sum((0..n).map(|l| full[l * n + i] * b[j * bcs + l]))
						+ 0.3 * c0[j * ccs + i];
					let scale = comp_sum((0..n).map(|l| (full[l * n + i] * b[j * bcs + l]).abs()))
						+ c0[j * ccs + i].abs();
					let tol = f64::EPSILON * (n.max(1) as f64) * 8.0 * scale + 1e-300;
					assert!((c[j * ccs + i] - want).abs() <= tol, "symm_left ({i},{j})");
				}
			}

			let mut c = c0.clone();
			symm_right(0.8, m, n, &a, acs, upper, &b, bcs, 0.3, &mut c, ccs);
			for j in 0..n {
				for i in 0..m {
					let want = 0.8 * comp_sum((0..n).map(|l| b[l * bcs + i] * full[j * n + l]))
						+ 0.3 * c0[j * ccs + i];
					let scale = comp_sum((0..n).map(|l| (b[l * bcs + i] * full[j * n + l]).abs()))
						+ c0[j * ccs + i].abs();
					let tol = f64::EPSILON * (n.max(1) as f64) * 8.0 * scale + 1e-300;
					assert!((c[j * ccs + i] - want).abs() <= tol, "symm_right ({i},{j})");
				}
			}
			// same-order scalar replay (β scale, then k ascending: one
			// α·a rounding + one mul-add per element — the fan-out
			// grouping does not change any element's sequence)
			let mut cr = c0.clone();
			for j in 0..n {
				for i in 0..m {
					cr[j * ccs + i] *= 0.3;
				}
				for k in 0..n {
					let t = 0.8 * full[j * n + k];
					for i in 0..m {
						cr[j * ccs + i] += b[k * bcs + i] * t;
					}
				}
			}
			for j in 0..n {
				for i in 0..m {
					assert_eq!(
						c[j * ccs + i].to_bits(),
						cr[j * ccs + i].to_bits(),
						"symm_right bits n={n} ({i},{j})"
					);
				}
			}
		}
	}
}

#[test]
fn syrk_syr2k_bit_for_bit() {
	let mut rng = Lcg(33);
	for &n in NS {
		let k = (n / 2).max(1);
		for upper in [true, false] {
			let (acs, bcs, ccs) = (n + 1, n + 2, n + 1);
			let a = rng.mat(n, k, acs);
			let b = rng.mat(n, k, bcs);
			let c0 = rng.mat(n, n, ccs);

			let mut c = c0.clone();
			syrk(0.5, n, k, &a, acs, 0.25, &mut c, ccs, upper);
			let mut cr = c0.clone();
			for j in 0..n {
				let (lo, hi) = if upper { (0, j + 1) } else { (j, n) };
				for i in lo..hi {
					cr[j * ccs + i] *= 0.25;
				}
				for l in 0..k {
					let t = 0.5 * a[l * acs + j];
					for i in lo..hi {
						cr[j * ccs + i] += a[l * acs + i] * t;
					}
				}
			}
			for j in 0..n {
				for i in 0..n {
					assert_eq!(c[j * ccs + i].to_bits(), cr[j * ccs + i].to_bits(), "syrk ({i},{j})");
				}
			}

			let mut c = c0.clone();
			syr2k(0.5, n, k, &a, acs, &b, bcs, 0.25, &mut c, ccs, upper);
			let mut cr = c0.clone();
			for j in 0..n {
				let (lo, hi) = if upper { (0, j + 1) } else { (j, n) };
				for i in lo..hi {
					cr[j * ccs + i] *= 0.25;
				}
				for l in 0..k {
					let tb = 0.5 * b[l * bcs + j];
					for i in lo..hi {
						cr[j * ccs + i] += a[l * acs + i] * tb;
					}
					let ta = 0.5 * a[l * acs + j];
					for i in lo..hi {
						cr[j * ccs + i] += b[l * bcs + i] * ta;
					}
				}
			}
			for j in 0..n {
				for i in 0..n {
					assert_eq!(c[j * ccs + i].to_bits(), cr[j * ccs + i].to_bits(), "syr2k ({i},{j})");
				}
			}
		}
	}
}

#[test]
fn trmm_both_sides_bounded() {
	let mut rng = Lcg(34);
	for &n in NS {
		let m = n;
		for upper in [true, false] {
			for unit in [true, false] {
				let acs = n + 1;
				let a = rng.mat(n, n, acs);
				let bcs = m + 2;
				let b0 = rng.mat(m, n, bcs);
				let tri_at = |i: usize, j: usize| -> f64 {
					let in_tri = if upper { i <= j } else { i >= j };
					if !in_tri {
						0.0
					} else if unit && i == j {
						1.0
					} else {
						a[j * acs + i]
					}
				};

				let mut b = b0.clone();
				trmm_left(0.9, m, n, &a, acs, upper, unit, &mut b, bcs);
				for j in 0..n {
					for i in 0..m {
						let want = 0.9 * comp_sum((0..n).map(|l| tri_at(i, l) * b0[j * bcs + l]));
						let scale = comp_sum((0..n).map(|l| (tri_at(i, l) * b0[j * bcs + l]).abs()));
						let tol = f64::EPSILON * (n.max(1) as f64) * 8.0 * scale + 1e-300;
						assert!(
							(b[j * bcs + i] - want).abs() <= tol,
							"trmm_left upper={upper} unit={unit} n={n} ({i},{j})"
						);
					}
				}

				let mut b = b0.clone();
				trmm_right(0.9, m, n, &a, acs, upper, unit, &mut b, bcs);
				for j in 0..n {
					for i in 0..m {
						let want = 0.9 * comp_sum((0..n).map(|l| b0[l * bcs + i] * tri_at(l, j)));
						let scale = comp_sum((0..n).map(|l| (b0[l * bcs + i] * tri_at(l, j)).abs()));
						let tol = f64::EPSILON * (n.max(1) as f64) * 8.0 * scale + 1e-300;
						assert!(
							(b[j * bcs + i] - want).abs() <= tol,
							"trmm_right upper={upper} unit={unit} n={n} ({i},{j})"
						);
					}
				}
			}
		}
	}
}

#[test]
fn trsm_both_sides_residual() {
	let mut rng = Lcg(35);
	for &n in NS {
		let m = n;
		for upper in [true, false] {
			for unit in [true, false] {
				let acs = n + 1;
				let mut a = rng.mat(n, n, acs);
				for j in 0..n {
					a[j * acs + j] = 2.0 * (n as f64) + 1.0 + j as f64;
				}
				let bcs = m + 1;
				let b0 = rng.mat(m, n, bcs);
				let tri_at = |i: usize, j: usize| -> f64 {
					let in_tri = if upper { i <= j } else { i >= j };
					if !in_tri {
						0.0
					} else if unit && i == j {
						1.0
					} else {
						a[j * acs + i]
					}
				};
				let bound = 3.0 * (n as f64 + 1.0);

				// left: A·X = α·B0
				let mut x = b0.clone();
				trsm_left(0.9, m, n, &a, acs, upper, unit, &mut x, bcs);
				for j in 0..n {
					for i in 0..m {
						let ax = comp_sum((0..n).map(|l| tri_at(i, l) * x[j * bcs + l]));
						let want = 0.9 * b0[j * bcs + i];
						let xmax = (0..n).fold(0.0f64, |acc, l| acc.max(x[j * bcs + l].abs()));
						let tol = f64::EPSILON * (n.max(1) as f64) * 8.0 * bound * xmax + 1e-300;
						assert!(
							(ax - want).abs() <= tol,
							"trsm_left residual upper={upper} unit={unit} n={n} ({i},{j})"
						);
					}
				}

				// right: X·A = α·B0
				let mut x = b0.clone();
				trsm_right(0.9, m, n, &a, acs, upper, unit, &mut x, bcs);
				for j in 0..n {
					for i in 0..m {
						let xa = comp_sum((0..n).map(|l| x[l * bcs + i] * tri_at(l, j)));
						let want = 0.9 * b0[j * bcs + i];
						let xmax = (0..n).fold(0.0f64, |acc, l| acc.max(x[l * bcs + i].abs()));
						let tol = f64::EPSILON * (n.max(1) as f64) * 8.0 * bound * xmax + 1e-300;
						assert!(
							(xa - want).abs() <= tol,
							"trsm_right residual upper={upper} unit={unit} n={n} ({i},{j})"
						);
					}
				}
			}
		}
	}
}

#[test]
fn gemm_tiled_bit_identical_to_gemm() {
	let mut rng = Lcg(36);
	// sizes crossing every tile boundary: exact multiples, tails in m,
	// n, both, and tiny
	for &(m, k, n) in &[(8usize, 8usize, 8usize), (4, 4, 4), (12, 7, 8), (9, 5, 10), (7, 3, 6), (3, 2, 3), (1, 1, 1), (0, 0, 0), (16, 16, 5), (5, 16, 16)] {
		let (acs, bcs, ccs) = (m + 1, k + 2, m + 3);
		let a = rng.mat(m, k, acs);
		let b = rng.mat(k, n, bcs);
		let c0 = rng.mat(m, n, ccs);
		for (alpha, beta) in [(1.0, 0.0), (-0.7, 0.4), (0.3, 1.0)] {
			let mut c1 = c0.clone();
			gemm_colaxpy(alpha, m, k, n, &a, acs, &b, bcs, beta, &mut c1, ccs);
			let mut cd = c0.clone();
			gemm(alpha, m, k, n, &a, acs, &b, bcs, beta, &mut cd, ccs);
			for j in 0..n {
				for i in 0..m {
					assert_eq!(c1[j * ccs + i].to_bits(), cd[j * ccs + i].to_bits(), "dispatcher vs colaxpy");
				}
			}
			let mut c2 = c0.clone();
			gemm_tiled(alpha, m, k, n, &a, acs, &b, bcs, beta, &mut c2, ccs);
			let mut c3 = c0.clone();
			gemm_col4(alpha, m, k, n, &a, acs, &b, bcs, beta, &mut c3, ccs);
			for j in 0..n {
				for i in 0..m {
					assert_eq!(
						c1[j * ccs + i].to_bits(),
						c2[j * ccs + i].to_bits(),
						"tiled vs column-axpy {m}x{k}x{n} ({i},{j})"
					);
					assert_eq!(
						c1[j * ccs + i].to_bits(),
						c3[j * ccs + i].to_bits(),
						"col4 vs column-axpy {m}x{k}x{n} ({i},{j})"
					);
				}
			}
		}
	}
}

// Same-order scalar replays for the triangular in-place ops. For
// trmm_left / trsm_left / trmm_right-lower / trsm_right-upper the
// replay is the PLAIN pre-tuning order — asserting the shipped
// lockstep/fan-out grouping is bit-identical to it. For
// trmm_right-upper / trsm_right-lower the grouping reorders the adds
// (documented in the op files); the replay mirrors the shipped group
// structure exactly.
const TRI_DIMS: &[(usize, usize)] =
	&[(0, 0), (1, 1), (3, 5), (5, 3), (4, 4), (8, 8), (9, 13), (16, 7), (20, 33), (33, 20)];

#[test]
fn trmm_bit_replay_both_sides() {
	let mut rng = Lcg(37);
	for &(m, n) in TRI_DIMS {
		for upper in [true, false] {
			for unit in [true, false] {
				let bcs = m + 2;

				// left: B <- alpha * A * B, plain trmv-per-column replay
				let acs = m + 1;
				let a = rng.mat(m, m, acs);
				let b0 = rng.mat(m, n, bcs);
				let mut bt = b0.clone();
				trmm_left(0.9, m, n, &a, acs, upper, unit, &mut bt, bcs);
				let mut br = b0.clone();
				for j in 0..n {
					if upper {
						for l in 0..m {
							let t = br[j * bcs + l];
							for i in 0..l {
								br[j * bcs + i] += a[l * acs + i] * t;
							}
							if !unit {
								br[j * bcs + l] = t * a[l * acs + l];
							}
						}
					} else {
						for l in (0..m).rev() {
							let t = br[j * bcs + l];
							for i in l + 1..m {
								br[j * bcs + i] += a[l * acs + i] * t;
							}
							if !unit {
								br[j * bcs + l] = t * a[l * acs + l];
							}
						}
					}
					for i in 0..m {
						br[j * bcs + i] *= 0.9;
					}
				}
				for (x, y) in bt.iter().zip(&br) {
					if !x.is_nan() {
						assert_eq!(x.to_bits(), y.to_bits(), "trmm_left {m}x{n} u={upper}");
					}
				}

				// right: B <- alpha * B * A
				let acs = n + 1;
				let a = rng.mat(n, n, acs);
				let b0 = rng.mat(m, n, bcs);
				let mut bt = b0.clone();
				trmm_right(0.9, m, n, &a, acs, upper, unit, &mut bt, bcs);
				let mut br = b0.clone();
				let dcol = |br: &mut Vec<f64>, j: usize, lo: usize, hi: usize| {
					let d = if unit { 1.0 } else { a[j * acs + j] };
					let s = 0.9 * d;
					for i in 0..m {
						br[j * bcs + i] *= s;
					}
					for k in lo..hi {
						let t = 0.9 * a[j * acs + k];
						for i in 0..m {
							br[j * bcs + i] += br[k * bcs + i] * t;
						}
					}
				};
				if upper {
					// grouped replay: in-group (descending targets, ascending
					// k) first, then out-of-group sources ascending
					let r = n % 4;
					let mut gs = n;
					while gs >= r + 4 {
						gs -= 4;
						for tc in (gs..gs + 4).rev() {
							dcol(&mut br, tc, gs, tc);
						}
						for k in 0..gs {
							for u in 0..4 {
								let t = 0.9 * a[(gs + u) * acs + k];
								for i in 0..m {
									br[(gs + u) * bcs + i] += br[k * bcs + i] * t;
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
					if !x.is_nan() {
						assert_eq!(x.to_bits(), y.to_bits(), "trmm_right {m}x{n} u={upper}");
					}
				}
			}
		}
	}
}

#[test]
fn trsm_bit_replay_both_sides() {
	let mut rng = Lcg(38);
	for &(m, n) in TRI_DIMS {
		for upper in [true, false] {
			for unit in [true, false] {
				let bcs = m + 2;

				// left: B <- alpha * inv(A) * B, plain scal-then-trsv replay
				let acs = m + 1;
				let mut a = rng.mat(m, m, acs);
				for j in 0..m {
					a[j * acs + j] = 2.0 * (m as f64) + 1.0 + j as f64;
				}
				let b0 = rng.mat(m, n, bcs);
				let mut bt = b0.clone();
				trsm_left(0.9, m, n, &a, acs, upper, unit, &mut bt, bcs);
				let mut br = b0.clone();
				for j in 0..n {
					for i in 0..m {
						br[j * bcs + i] *= 0.9;
					}
					if upper {
						for l in (0..m).rev() {
							if !unit {
								br[j * bcs + l] /= a[l * acs + l];
							}
							let t = -br[j * bcs + l];
							for i in 0..l {
								br[j * bcs + i] += a[l * acs + i] * t;
							}
						}
					} else {
						for l in 0..m {
							if !unit {
								br[j * bcs + l] /= a[l * acs + l];
							}
							let t = -br[j * bcs + l];
							for i in l + 1..m {
								br[j * bcs + i] += a[l * acs + i] * t;
							}
						}
					}
				}
				for (x, y) in bt.iter().zip(&br) {
					if !x.is_nan() {
						assert_eq!(x.to_bits(), y.to_bits(), "trsm_left {m}x{n} u={upper}");
					}
				}

				// right: B <- alpha * B * inv(A)
				let acs = n + 1;
				let mut a = rng.mat(n, n, acs);
				for j in 0..n {
					a[j * acs + j] = 2.0 * (n as f64) + 1.0 + j as f64;
				}
				let b0 = rng.mat(m, n, bcs);
				let mut bt = b0.clone();
				trsm_right(0.9, m, n, &a, acs, upper, unit, &mut bt, bcs);
				let mut br = b0.clone();
				let elim = |br: &mut Vec<f64>, j: usize, k: usize| {
					let t = -a[j * acs + k];
					for i in 0..m {
						br[j * bcs + i] += br[k * bcs + i] * t;
					}
				};
				let finish = |br: &mut Vec<f64>, j: usize| {
					if !unit {
						let s = 1.0 / a[j * acs + j];
						for i in 0..m {
							br[j * bcs + i] *= s;
						}
					}
				};
				if upper {
					// plain ascending replay — the grouping keeps upper
					// fully ascending, so this asserts bit-identity to it
					for j in 0..n {
						for i in 0..m {
							br[j * bcs + i] *= 0.9;
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
								br[(gs + u) * bcs + i] *= 0.9;
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
							br[j * bcs + i] *= 0.9;
						}
						for k in j + 1..n {
							elim(&mut br, j, k);
						}
						finish(&mut br, j);
					}
				}
				for (x, y) in bt.iter().zip(&br) {
					if !x.is_nan() {
						assert_eq!(x.to_bits(), y.to_bits(), "trsm_right {m}x{n} u={upper}");
					}
				}
			}
		}
	}
}

#[test]
#[should_panic(expected = "storage too short")]
fn gemm_short_storage_panics() {
	gemm(1.0, 2, 2, 2, &[1.0; 4], 2, &[1.0; 3], 2, 0.0, &mut [0.0; 4], 2);
}
