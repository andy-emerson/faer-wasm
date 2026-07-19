use super::common::*;
use faer_wasm_blas::L2::*;
use faer_wasm_blas::C32;

#[test]
fn ctrsv_bit_for_bit_and_residual() {
	let mut rng = Lcg(129);
	for &n in NS {
		for upper in [true, false] {
			for unit in [true, false] {
				let cs = n + 1;
				// diagonally dominant triangle with a genuinely complex
				// diagonal, so Smith's division is exercised
				let mut a = rng.mat_c32(n, n, cs);
				for j in 0..n {
					a[j * cs + j] = C32::new(2.0 * (n as f32) + 1.0 + j as f32, 0.7);
				}
				let b = rng.vec_c32(n);
				let mut x = b.clone();
				ctrsv(n, &a, cs, upper, unit, &mut x);

				// same-order scalar replay: bit-for-bit
				let mut xr = b.clone();
				if upper {
					for j in (0..n).rev() {
						if !unit {
							xr[j] = xr[j] / a[j * cs + j];
						}
						let t = -xr[j];
						for i in 0..j {
							xr[i] = xr[i] + t * a[j * cs + i];
						}
					}
				} else {
					for j in 0..n {
						if !unit {
							xr[j] = xr[j] / a[j * cs + j];
						}
						let t = -xr[j];
						for i in j + 1..n {
							xr[i] = xr[i] + t * a[j * cs + i];
						}
					}
				}
				for i in 0..n {
					assert!(
						bits_eq_cc(x[i], xr[i]),
						"ctrsv bits upper={upper} unit={unit} n={n} i={i}"
					);
				}

				// independent residual: A·x must reproduce b
				let xmax = x.iter().fold(0.0f64, |m, v| m.max(v.abs1() as f64));
				for i in 0..n {
					let ax = comp_sum_cc((0..n).map(|j| {
						let in_tri = if upper { i <= j } else { i >= j };
						if !in_tri {
							return C32::ZERO;
						}
						let aij = if unit && i == j { C32::ONE } else { a[j * cs + i] };
						aij * x[j]
					}));
					let scale = xmax * (3.0 * n as f64 + 2.0) + b[i].abs1() as f64;
					let tol = EPS * (n.max(1) as f64) * 16.0 * scale + 1e-30;
					assert!(
						(ax.re - b[i].re as f64).abs() + (ax.im - b[i].im as f64).abs() <= tol,
						"ctrsv residual upper={upper} unit={unit} n={n} i={i}"
					);
				}
			}
		}
	}
}
