use super::common::*;
use faer_wasm_blas::L1::*;
use faer_wasm_blas::C64;

#[test]
fn zaxpy_bit_for_bit() {
	let mut rng = Lcg(101);
	for &n in SIZES {
		let x = rng.vec_c64(n);
		let y0 = rng.vec_c64(n);
		for alpha in [
			C64::new(-0.7, 0.3),
			C64::new(1.5, 0.0),
			C64::new(0.0, -1.1),
			C64::ZERO,
		] {
			let mut y = y0.clone();
			zaxpy(alpha, &x, &mut y);
			for i in 0..n {
				let want = y0[i] + alpha * x[i];
				assert!(bits_eq_c(y[i], want), "zaxpy n={n} i={i} α={alpha:?}");
			}
		}
	}
}

#[test]
#[should_panic(expected = "length mismatch")]
fn zaxpy_length_mismatch_panics() {
	zaxpy(C64::ONE, &[C64::ONE], &mut []);
}
