//! Level 2 correctness per the testing contract (../README.md).
//! Two layers of evidence per operation:
//! - bit-for-bit against a same-order scalar reference wherever the
//!   implementation is pure column-axpy (gemv, ger, syr, syr2, trmv,
//!   trsv) — the SIMD stream must not change any element's rounding
//!   sequence;
//! - error-bounded against an INDEPENDENT compensated-summation
//!   reference (different accumulation order) for everything,
//!   including the dot-involving forms (gemv_t, symv) and a trsv
//!   residual check.

use faer_wasm_blas::level1::asum;
use faer_wasm_blas::level2::*;

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
	/// column-major nrows×ncols at stride cs (padding filled with junk
	/// that must never be read)
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

// (nrows, ncols) shapes: empty, single, tails, unroll boundaries, big
const SHAPES: &[(usize, usize)] = &[(0, 0), (1, 1), (3, 2), (4, 4), (8, 5), (17, 9), (64, 64), (130, 33)];
// square sizes for the symmetric/triangular ops
const NS: &[usize] = &[0, 1, 2, 3, 5, 8, 17, 64, 130];
const PADS: &[usize] = &[0, 3];

#[test]
fn gemv_bit_for_bit_and_bounded() {
	let mut rng = Lcg(21);
	for &(m, n) in SHAPES {
		for &pad in PADS {
			let cs = m + pad;
			let a = rng.mat(m, n, cs);
			let x = rng.vec(n);
			let y0 = rng.vec(m);
			for (alpha, beta) in [(1.0, 0.0), (-0.5, 1.0), (0.3, -2.0), (0.0, 0.5)] {
				let mut y = y0.clone();
				gemv(alpha, m, n, &a, cs, &x, beta, &mut y);
				// same-order scalar reference: exact replay of column-axpy
				let mut yr = y0.clone();
				if beta == 0.0 {
					yr.fill(0.0);
				} else if beta != 1.0 {
					for v in yr.iter_mut() {
						*v *= beta;
					}
				}
				for j in 0..n {
					let t = alpha * x[j];
					for i in 0..m {
						yr[i] += a[j * cs + i] * t;
					}
				}
				for i in 0..m {
					assert_eq!(y[i].to_bits(), yr[i].to_bits(), "gemv bits {m}x{n} pad={pad} i={i}");
				}
				// independent bound: Kahan row-dots in a different order
				for i in 0..m {
					let want = comp_sum((0..n).map(|j| alpha * x[j] * a[j * cs + i]))
						+ if beta == 0.0 { 0.0 } else { beta * y0[i] };
					let scale = comp_sum((0..n).map(|j| (alpha * x[j] * a[j * cs + i]).abs()))
						+ (beta * y0[i]).abs();
					let tol = f64::EPSILON * (n.max(1) as f64) * 4.0 * scale + 1e-300;
					assert!((y[i] - want).abs() <= tol, "gemv bound {m}x{n} i={i}");
				}
			}
		}
	}
}

#[test]
fn gemv_t_bounded() {
	let mut rng = Lcg(22);
	for &(m, n) in SHAPES {
		let cs = m + 2;
		let a = rng.mat(m, n, cs);
		let x = rng.vec(m);
		let y0 = rng.vec(n);
		let mut y = y0.clone();
		gemv_t(0.7, m, n, &a, cs, &x, -0.3, &mut y);
		for j in 0..n {
			let want = 0.7 * comp_sum((0..m).map(|i| a[j * cs + i] * x[i])) - 0.3 * y0[j];
			let scale = comp_sum((0..m).map(|i| (a[j * cs + i] * x[i]).abs())) + y0[j].abs();
			let tol = f64::EPSILON * (m.max(1) as f64) * 4.0 * scale + 1e-300;
			assert!((y[j] - want).abs() <= tol, "gemv_t {m}x{n} j={j}");
		}
	}
}

#[test]
fn ger_bit_for_bit() {
	let mut rng = Lcg(23);
	for &(m, n) in SHAPES {
		let cs = m + 1;
		let a0 = rng.mat(m, n, cs);
		let x = rng.vec(m);
		let y = rng.vec(n);
		let mut a = a0.clone();
		ger(-1.3, m, n, &mut a, cs, &x, &y);
		for j in 0..n {
			let t = -1.3 * y[j];
			for i in 0..m {
				let want = a0[j * cs + i] + x[i] * t;
				assert_eq!(a[j * cs + i].to_bits(), want.to_bits(), "ger {m}x{n} ({i},{j})");
			}
		}
	}
}

#[test]
fn symv_bounded_both_triangles() {
	let mut rng = Lcg(24);
	for &n in NS {
		for upper in [true, false] {
			let cs = n + 1;
			// build a full symmetric matrix, then expose only one triangle
			let full = {
				let mut f = vec![0.0; n * n];
				for j in 0..n {
					for i in 0..=j {
						let v = rng.next_f64();
						f[j * n + i] = v;
						f[i * n + j] = v;
					}
				}
				f
			};
			let mut a = vec![f64::NAN; if n == 0 { 0 } else { cs * (n - 1) + n }];
			for j in 0..n {
				for i in 0..n {
					let stored = if upper { i <= j } else { i >= j };
					if stored {
						a[j * cs + i] = full[j * n + i];
					}
				}
			}
			let x = rng.vec(n);
			let y0 = rng.vec(n);
			let mut y = y0.clone();
			symv(0.9, n, &a, cs, upper, &x, 0.4, &mut y);
			for i in 0..n {
				let want =
					0.9 * comp_sum((0..n).map(|j| full[j * n + i] * x[j])) + 0.4 * y0[i];
				let scale =
					comp_sum((0..n).map(|j| (full[j * n + i] * x[j]).abs())) + y0[i].abs();
				let tol = f64::EPSILON * (n.max(1) as f64) * 4.0 * scale + 1e-300;
				assert!((y[i] - want).abs() <= tol, "symv upper={upper} n={n} i={i}");
			}
		}
	}
}

#[test]
fn syr_syr2_bit_for_bit() {
	let mut rng = Lcg(25);
	for &n in NS {
		for upper in [true, false] {
			let cs = n + 2;
			let a0 = rng.mat(n, n, cs);
			let x = rng.vec(n);
			let y = rng.vec(n);

			let mut a = a0.clone();
			syr(0.6, n, &mut a, cs, upper, &x);
			for j in 0..n {
				let t = 0.6 * x[j];
				for i in 0..n {
					let stored = if upper { i <= j } else { i >= j };
					let want = if stored { a0[j * cs + i] + x[i] * t } else { a0[j * cs + i] };
					assert_eq!(a[j * cs + i].to_bits(), want.to_bits(), "syr n={n} ({i},{j})");
				}
			}

			let mut a = a0.clone();
			syr2(0.6, n, &mut a, cs, upper, &x, &y);
			for j in 0..n {
				let (ty, tx) = (0.6 * y[j], 0.6 * x[j]);
				for i in 0..n {
					let stored = if upper { i <= j } else { i >= j };
					let want = if stored {
						(a0[j * cs + i] + x[i] * ty) + y[i] * tx
					} else {
						a0[j * cs + i]
					};
					assert_eq!(a[j * cs + i].to_bits(), want.to_bits(), "syr2 n={n} ({i},{j})");
				}
			}
		}
	}
}

#[test]
fn trmv_bit_for_bit_all_variants() {
	let mut rng = Lcg(26);
	for &n in NS {
		for upper in [true, false] {
			for unit in [true, false] {
				let cs = n + 1;
				let a = rng.mat(n, n, cs);
				let x0 = rng.vec(n);
				let mut x = x0.clone();
				trmv(n, &a, cs, upper, unit, &mut x);
				// same-order scalar replay
				let mut xr = x0.clone();
				if upper {
					for j in 0..n {
						let t = xr[j];
						for i in 0..j {
							xr[i] += a[j * cs + i] * t;
						}
						if !unit {
							xr[j] = t * a[j * cs + j];
						}
					}
				} else {
					for j in (0..n).rev() {
						let t = xr[j];
						for i in j + 1..n {
							xr[i] += a[j * cs + i] * t;
						}
						if !unit {
							xr[j] = t * a[j * cs + j];
						}
					}
				}
				for i in 0..n {
					assert_eq!(
						x[i].to_bits(),
						xr[i].to_bits(),
						"trmv upper={upper} unit={unit} n={n} i={i}"
					);
				}
			}
		}
	}
}

#[test]
fn trsv_bit_for_bit_and_residual() {
	let mut rng = Lcg(27);
	for &n in NS {
		for upper in [true, false] {
			for unit in [true, false] {
				let cs = n + 1;
				// diagonally dominant triangle: solves stay well-conditioned
				let mut a = rng.mat(n, n, cs);
				for j in 0..n {
					a[j * cs + j] = 2.0 * (n as f64) + 1.0 + j as f64;
				}
				let b = rng.vec(n);
				let mut x = b.clone();
				trsv(n, &a, cs, upper, unit, &mut x);

				// same-order scalar replay: bit-for-bit
				let mut xr = b.clone();
				if upper {
					for j in (0..n).rev() {
						if !unit {
							xr[j] /= a[j * cs + j];
						}
						let t = xr[j];
						for i in 0..j {
							xr[i] += a[j * cs + i] * -t;
						}
					}
				} else {
					for j in 0..n {
						if !unit {
							xr[j] /= a[j * cs + j];
						}
						let t = xr[j];
						for i in j + 1..n {
							xr[i] += a[j * cs + i] * -t;
						}
					}
				}
				for i in 0..n {
					assert_eq!(
						x[i].to_bits(),
						xr[i].to_bits(),
						"trsv bits upper={upper} unit={unit} n={n} i={i}"
					);
				}

				// independent residual: A·x must reproduce b
				for i in 0..n {
					let ax = comp_sum((0..n).map(|j| {
						let in_tri = if upper { i <= j } else { i >= j };
						if !in_tri {
							return 0.0;
						}
						let aij = if unit && i == j { 1.0 } else { a[j * cs + i] };
						aij * x[j]
					}));
					let scale = asum(&x) * (2.0 * n as f64 + n as f64) + b[i].abs();
					let tol = f64::EPSILON * (n.max(1) as f64) * 8.0 * scale + 1e-300;
					assert!(
						(ax - b[i]).abs() <= tol,
						"trsv residual upper={upper} unit={unit} n={n} i={i}: {ax} vs {}",
						b[i]
					);
				}
			}
		}
	}
}

#[test]
#[should_panic(expected = "length mismatch")]
fn gemv_length_mismatch_panics() {
	gemv(1.0, 2, 2, &[1.0, 2.0, 3.0, 4.0], 2, &[1.0], 0.0, &mut [0.0, 0.0]);
}

#[test]
#[should_panic(expected = "storage too short")]
fn gemv_short_storage_panics() {
	gemv(1.0, 2, 2, &[1.0, 2.0, 3.0], 2, &[1.0, 1.0], 0.0, &mut [0.0, 0.0]);
}
