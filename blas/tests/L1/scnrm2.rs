use super::common::*;
use faer_wasm_blas::L1::*;
use faer_wasm_blas::C32;

#[test]
fn scnrm2_bounded() {
	let mut rng = Lcg(110);
	for &n in SIZES {
		let x = rng.vec_c32(n);
		let v = scnrm2(&x) as f64;
		// squares formed in f32 (as the implementation forms them),
		// summed exactly in f64
		let ss = comp_sum32((0..n).flat_map(|i| [x[i].re * x[i].re, x[i].im * x[i].im]));
		let want = ss.sqrt();
		let tol = EPS * (2 * n.max(1)) as f64 * 4.0 * want + 1e-30;
		assert!((v - want).abs() <= tol, "scnrm2 n={n}: {v} vs {want}");
	}
}

#[test]
fn scnrm2_overflow_underflow_guards() {
	// squares overflow: |x| ~ 1e30 — the rescued pass must survive
	let big = vec![C32::new(3e30, 4e30); 5];
	let vb = scnrm2(&big) as f64;
	let want_b = 5e30 * (5.0f64).sqrt();
	assert!((vb - want_b).abs() <= 1e-5 * want_b, "overflow rescue: {vb} vs {want_b}");
	// squares underflow to subnormals: |x| ~ 1e-30
	let small = vec![C32::new(3e-30, 4e-30); 5];
	let vs = scnrm2(&small) as f64;
	let want_s = 5e-30 * (5.0f64).sqrt();
	assert!((vs - want_s).abs() <= 1e-5 * want_s, "underflow rescue: {vs} vs {want_s}");
	assert_eq!(scnrm2(&[]), 0.0);
	assert_eq!(scnrm2(&[C32::ZERO; 3]), 0.0);
}
