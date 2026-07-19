use super::common::*;
use faer_wasm_blas::L2::*;
use faer_wasm_blas::C64;

#[test]
fn zher2_bit_for_bit_and_real_diagonal() {
	let mut rng = Lcg(127);
	let alpha = C64::new(0.6, -0.9);
	for &n in NS {
		for upper in [true, false] {
			let cs = n + 1;
			let a0 = rng.mat_c64(n, n, cs);
			let x = rng.vec_c64(n);
			let y = rng.vec_c64(n);
			let mut a = a0.clone();
			zher2(alpha, n, &mut a, cs, upper, &x, &y);
			for j in 0..n {
				let t1 = alpha * y[j].conj();
				let t2 = (alpha * x[j]).conj();
				let (lo, hi) = if upper { (0, j) } else { (j + 1, n) };
				for i in lo..hi {
					// two zaxpy passes in order: x-sourced then y-sourced
					let want = (a0[j * cs + i] + t1 * x[i]) + t2 * y[i];
					assert!(
						bits_eq_c(a[j * cs + i], want),
						"zher2 upper={upper} n={n} ({i},{j})"
					);
				}
				let wd = C64::new(
					a0[j * cs + j].re + ((x[j] * t1).re + (y[j] * t2).re),
					0.0,
				);
				assert!(bits_eq_c(a[j * cs + j], wd), "zher2 diag upper={upper} n={n} j={j}");
				assert_eq!(a[j * cs + j].im.to_bits(), 0.0f64.to_bits());
			}
		}
	}
}
