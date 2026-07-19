use super::common::*;
use faer_wasm_blas::L2::*;
use faer_wasm_blas::C64;

#[test]
fn zgerc_bit_for_bit_and_geru_cross_check() {
	let mut rng = Lcg(124);
	let alpha = C64::new(0.9, -0.4);
	for &(m, n) in SHAPES {
		let cs = m + 1;
		let a0 = rng.mat_c64(m, n, cs);
		let x = rng.vec_c64(m);
		let y = rng.vec_c64(n);
		let mut a = a0.clone();
		zgerc(alpha, m, n, &mut a, cs, &x, &y);
		for j in 0..n {
			let t = alpha * y[j].conj();
			for i in 0..m {
				let want = a0[j * cs + i] + t * x[i];
				assert!(bits_eq_c(a[j * cs + i], want), "zgerc {m}x{n} ({i},{j})");
			}
		}
		// zgerc(y) must be bit-identical to zgeru(conj(y))
		let yc: Vec<C64> = y.iter().map(|v| v.conj()).collect();
		let mut a2 = a0.clone();
		zgeru(alpha, m, n, &mut a2, cs, &x, &yc);
		for (p, q) in a.iter().zip(&a2) {
			if !p.re.is_nan() {
				assert!(bits_eq_c(*p, *q), "zgerc vs zgeru(conj y) {m}x{n}");
			}
		}
	}
}
