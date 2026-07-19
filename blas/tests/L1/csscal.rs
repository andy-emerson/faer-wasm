use super::common::*;
use faer_wasm_blas::L1::*;
use faer_wasm_blas::C32;

#[test]
fn csscal_bit_for_bit() {
	let mut rng = Lcg(103);
	for &n in SIZES {
		let x0 = rng.vec_c32(n);
		for alpha in [-0.7f32, 2.0, 0.0] {
			let mut x = x0.clone();
			csscal(alpha, &mut x);
			for i in 0..n {
				// one real multiply per component — NOT the full
				// complex product (cscal with imag 0), per module doc
				let want = C32::new(x0[i].re * alpha, x0[i].im * alpha);
				assert!(bits_eq_cc(x[i], want), "csscal n={n} i={i} α={alpha}");
			}
		}
	}
}
