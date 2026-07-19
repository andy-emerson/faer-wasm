use super::common::*;
use faer_wasm_blas::L1::*;
use faer_wasm_blas::C32;

#[test]
fn cscal_bit_for_bit() {
	let mut rng = Lcg(102);
	for &n in SIZES {
		let x0 = rng.vec_c32(n);
		for alpha in [C32::new(-0.7, 0.3), C32::new(2.0, 0.0), C32::ZERO] {
			let mut x = x0.clone();
			cscal(alpha, &mut x);
			for i in 0..n {
				let want = alpha * x0[i];
				assert!(bits_eq_cc(x[i], want), "cscal n={n} i={i} α={alpha:?}");
			}
		}
	}
}
