use super::common::*;
use faer_wasm_blas::L2::*;
use faer_wasm_blas::C64;

#[test]
fn zgeru_bit_for_bit() {
	let mut rng = Lcg(123);
	let alpha = C64::new(-1.3, 0.7);
	for &(m, n) in SHAPES {
		let cs = m + 1;
		let a0 = rng.mat_c64(m, n, cs);
		let x = rng.vec_c64(m);
		let y = rng.vec_c64(n);
		let mut a = a0.clone();
		zgeru(alpha, m, n, &mut a, cs, &x, &y);
		for j in 0..n {
			let t = alpha * y[j];
			for i in 0..m {
				let want = a0[j * cs + i] + t * x[i];
				assert!(bits_eq_c(a[j * cs + i], want), "zgeru {m}x{n} ({i},{j})");
			}
		}
	}
}
