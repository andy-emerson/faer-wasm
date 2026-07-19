use super::common::*;
use faer_wasm_blas::L1::*;

#[test]
fn zswap_exact() {
	let mut rng = Lcg(105);
	for &n in SIZES {
		let x0 = rng.vec_c64(n);
		let y0 = rng.vec_c64(n);
		let mut x = x0.clone();
		let mut y = y0.clone();
		zswap(&mut x, &mut y);
		for i in 0..n {
			assert!(bits_eq_c(x[i], y0[i]), "zswap x n={n} i={i}");
			assert!(bits_eq_c(y[i], x0[i]), "zswap y n={n} i={i}");
		}
	}
}
