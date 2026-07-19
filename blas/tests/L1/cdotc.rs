use super::common::*;
use faer_wasm_blas::L1::*;

#[test]
fn cdotc_bounded() {
	let mut rng = Lcg(108);
	for &n in SIZES {
		let x = rng.vec_c32(n);
		let y = rng.vec_c32(n);
		let s = cdotc(&x, &y);
		let want = comp_sum_cc((0..n).map(|i| x[i].conj() * y[i]));
		let scale = comp_scale_cc((0..n).map(|i| x[i].conj() * y[i]));
		let tol = EPS * (n.max(1) as f64) * 4.0 * scale + 1e-30;
		assert!((s.re as f64 - want.re).abs() <= tol, "cdotc re n={n}");
		assert!((s.im as f64 - want.im).abs() <= tol, "cdotc im n={n}");
	}
}

#[test]
fn cdotc_is_cdotu_of_conjugated_x_bitwise() {
	// the conjugation folds into the lane signs exactly, so the two
	// paths must agree to the bit — a strong cross-check of the lane
	// algebra
	let mut rng = Lcg(109);
	for &n in SIZES {
		let x = rng.vec_c32(n);
		let y = rng.vec_c32(n);
		let xc: Vec<_> = x.iter().map(|v| v.conj()).collect();
		assert!(bits_eq_cc(cdotc(&x, &y), cdotu(&xc, &y)), "n={n}");
	}
}
