use super::common::*;
use faer_wasm_blas::L1::*;

#[test]
fn zdotu_bounded() {
	let mut rng = Lcg(107);
	for &n in SIZES {
		let x = rng.vec_c64(n);
		let y = rng.vec_c64(n);
		let s = zdotu(&x, &y);
		// products formed in C64 as the implementation forms them,
		// accumulated component-wise with compensation
		let want = comp_sum_c((0..n).map(|i| x[i] * y[i]));
		let scale = comp_scale_c((0..n).map(|i| x[i] * y[i]));
		let tol = f64::EPSILON * (n.max(1) as f64) * 4.0 * scale + 1e-300;
		assert!((s.re - want.re).abs() <= tol, "zdotu re n={n}");
		assert!((s.im - want.im).abs() <= tol, "zdotu im n={n}");
	}
}

#[test]
#[should_panic(expected = "length mismatch")]
fn zdotu_length_mismatch_panics() {
	zdotu(&[faer_wasm_blas::C64::ONE], &[]);
}
