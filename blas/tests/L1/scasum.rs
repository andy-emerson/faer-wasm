use super::common::*;
use faer_wasm_blas::L1::*;

#[test]
fn scasum_bounded() {
	let mut rng = Lcg(111);
	for &n in SIZES {
		let x = rng.vec_c32(n);
		let v = scasum(&x) as f64;
		// reference semantics: component magnitudes, NOT moduli
		let want = comp_sum32((0..n).flat_map(|i| [x[i].re.abs(), x[i].im.abs()]));
		let tol = EPS * (2 * n.max(1)) as f64 * 4.0 * want + 1e-30;
		assert!((v - want).abs() <= tol, "scasum n={n}: {v} vs {want}");
	}
	assert_eq!(scasum(&[]), 0.0f32);
}
