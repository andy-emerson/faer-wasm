use super::common::*;
use faer_wasm_blas::L1::*;

#[test]
fn dzasum_bounded() {
	let mut rng = Lcg(111);
	for &n in SIZES {
		let x = rng.vec_c64(n);
		let v = dzasum(&x);
		// reference semantics: component magnitudes, NOT moduli
		let want = comp_sum((0..n).flat_map(|i| [x[i].re.abs(), x[i].im.abs()]));
		let tol = f64::EPSILON * (2 * n.max(1)) as f64 * 4.0 * want + 1e-300;
		assert!((v - want).abs() <= tol, "dzasum n={n}: {v} vs {want}");
	}
	assert_eq!(dzasum(&[]), 0.0);
}
