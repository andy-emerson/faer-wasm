use super::common::*;
use faer_wasm_blas::L1::*;

#[test]
fn cswap_exact() {
	let mut rng = Lcg(105);
	for &n in SIZES {
		let x0 = rng.vec_c32(n);
		let y0 = rng.vec_c32(n);
		let mut x = x0.clone();
		let mut y = y0.clone();
		cswap(&mut x, &mut y);
		for i in 0..n {
			assert!(bits_eq_cc(x[i], y0[i]), "cswap x n={n} i={i}");
			assert!(bits_eq_cc(y[i], x0[i]), "cswap y n={n} i={i}");
		}
	}
}
