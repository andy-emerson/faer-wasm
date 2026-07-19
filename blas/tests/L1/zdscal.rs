use super::common::*;
use faer_wasm_blas::L1::*;
use faer_wasm_blas::C64;

#[test]
fn zdscal_bit_for_bit() {
	let mut rng = Lcg(103);
	for &n in SIZES {
		let x0 = rng.vec_c64(n);
		for alpha in [-0.7f64, 2.0, 0.0] {
			let mut x = x0.clone();
			zdscal(alpha, &mut x);
			for i in 0..n {
				// one real multiply per component — NOT the full
				// complex product (zscal with imag 0), per module doc
				let want = C64::new(x0[i].re * alpha, x0[i].im * alpha);
				assert!(bits_eq_c(x[i], want), "zdscal n={n} i={i} α={alpha}");
			}
		}
	}
}
