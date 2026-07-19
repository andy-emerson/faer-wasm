use super::common::*;
use faer_wasm_blas::L1::*;
use faer_wasm_blas::C64;

#[test]
fn zdrot_bit_for_bit() {
	let mut rng = Lcg(106);
	let (c, s) = (0.8, -0.6);
	for &n in SIZES {
		let x0 = rng.vec_c64(n);
		let y0 = rng.vec_c64(n);
		let mut x = x0.clone();
		let mut y = y0.clone();
		zdrot(&mut x, &mut y, c, s);
		for i in 0..n {
			// real rotation acts on re and im independently — the
			// scalar drot definition per component
			let wx = C64::new(x0[i].re * c + y0[i].re * s, x0[i].im * c + y0[i].im * s);
			let wy = C64::new(y0[i].re * c - x0[i].re * s, y0[i].im * c - x0[i].im * s);
			assert!(bits_eq_c(x[i], wx), "zdrot x n={n} i={i}");
			assert!(bits_eq_c(y[i], wy), "zdrot y n={n} i={i}");
		}
	}
}
