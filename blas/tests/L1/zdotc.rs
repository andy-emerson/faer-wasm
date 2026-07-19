use super::common::*;
use faer_wasm_blas::L1::*;

#[test]
fn zdotc_bounded() {
	let mut rng = Lcg(108);
	for &n in SIZES {
		let x = rng.vec_c64(n);
		let y = rng.vec_c64(n);
		let s = zdotc(&x, &y);
		let want = comp_sum_c((0..n).map(|i| x[i].conj() * y[i]));
		let scale = comp_scale_c((0..n).map(|i| x[i].conj() * y[i]));
		let tol = f64::EPSILON * (n.max(1) as f64) * 4.0 * scale + 1e-300;
		assert!((s.re - want.re).abs() <= tol, "zdotc re n={n}");
		assert!((s.im - want.im).abs() <= tol, "zdotc im n={n}");
	}
}

#[test]
fn zdotc_is_zdotu_of_conjugated_x_bitwise() {
	// the conjugation folds into the lane signs exactly, so the two
	// paths must agree to the bit — a strong cross-check of the lane
	// algebra
	let mut rng = Lcg(109);
	for &n in SIZES {
		let x = rng.vec_c64(n);
		let y = rng.vec_c64(n);
		let xc: Vec<_> = x.iter().map(|v| v.conj()).collect();
		assert!(bits_eq_c(zdotc(&x, &y), zdotu(&xc, &y)), "n={n}");
	}
}
