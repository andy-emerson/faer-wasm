use super::common::*;
use faer_wasm_blas::L1::*;
use faer_wasm_blas::C64;

#[test]
fn zscal_bit_for_bit() {
	let mut rng = Lcg(102);
	for &n in SIZES {
		let x0 = rng.vec_c64(n);
		for alpha in [C64::new(-0.7, 0.3), C64::new(2.0, 0.0), C64::ZERO] {
			let mut x = x0.clone();
			zscal(alpha, &mut x);
			for i in 0..n {
				let want = alpha * x0[i];
				assert!(bits_eq_c(x[i], want), "zscal n={n} i={i} α={alpha:?}");
			}
		}
	}
}
