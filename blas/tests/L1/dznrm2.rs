use super::common::*;
use faer_wasm_blas::L1::*;
use faer_wasm_blas::C64;

#[test]
fn dznrm2_bounded() {
	let mut rng = Lcg(110);
	for &n in SIZES {
		let x = rng.vec_c64(n);
		let v = dznrm2(&x);
		let ss = comp_sum((0..n).flat_map(|i| [x[i].re * x[i].re, x[i].im * x[i].im]));
		let want = ss.sqrt();
		let tol = f64::EPSILON * (2 * n.max(1)) as f64 * 4.0 * want + 1e-300;
		assert!((v - want).abs() <= tol, "dznrm2 n={n}: {v} vs {want}");
	}
}

#[test]
fn dznrm2_overflow_underflow_guards() {
	// squares overflow: |x| ~ 1e300 — the rescued pass must survive
	let big = vec![C64::new(3e300, 4e300); 5];
	let vb = dznrm2(&big);
	let want_b = 5e300 * (5.0f64).sqrt();
	assert!((vb - want_b).abs() <= 1e-10 * want_b, "overflow rescue: {vb} vs {want_b}");
	// squares underflow to subnormals: |x| ~ 1e-300
	let small = vec![C64::new(3e-300, 4e-300); 5];
	let vs = dznrm2(&small);
	let want_s = 5e-300 * (5.0f64).sqrt();
	assert!((vs - want_s).abs() <= 1e-10 * want_s, "underflow rescue: {vs} vs {want_s}");
	assert_eq!(dznrm2(&[]), 0.0);
	assert_eq!(dznrm2(&[C64::ZERO; 3]), 0.0);
}
