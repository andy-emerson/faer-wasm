use super::common::*;
use faer_wasm_blas::L1::*;

#[test]
fn cdotu_bounded() {
	let mut rng = Lcg(107);
	for &n in SIZES {
		let x = rng.vec_c32(n);
		let y = rng.vec_c32(n);
		let s = cdotu(&x, &y);
		// products formed in C32 as the implementation forms them,
		// accumulated component-wise with compensation
		let want = comp_sum_cc((0..n).map(|i| x[i] * y[i]));
		let scale = comp_scale_cc((0..n).map(|i| x[i] * y[i]));
		let tol = EPS * (n.max(1) as f64) * 4.0 * scale + 1e-30;
		assert!((s.re as f64 - want.re).abs() <= tol, "cdotu re n={n}");
		assert!((s.im as f64 - want.im).abs() <= tol, "cdotu im n={n}");
	}
}

#[test]
#[should_panic(expected = "length mismatch")]
fn cdotu_length_mismatch_panics() {
	cdotu(&[faer_wasm_blas::C32::ONE], &[]);
}
