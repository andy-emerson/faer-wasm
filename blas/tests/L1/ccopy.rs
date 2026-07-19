use super::common::*;
use faer_wasm_blas::L1::*;
use faer_wasm_blas::C32;

#[test]
fn ccopy_exact() {
	let mut rng = Lcg(104);
	for &n in SIZES {
		let x = rng.vec_c32(n);
		let mut y = vec![C32::new(f32::NAN, f32::NAN); n];
		ccopy(&x, &mut y);
		for i in 0..n {
			assert!(bits_eq_cc(y[i], x[i]), "ccopy n={n} i={i}");
		}
	}
}

#[test]
#[should_panic(expected = "length mismatch")]
fn ccopy_length_mismatch_panics() {
	ccopy(&[C32::ONE], &mut []);
}
