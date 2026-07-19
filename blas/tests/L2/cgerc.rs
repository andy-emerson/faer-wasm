use super::common::*;
use faer_wasm_blas::L2::*;
use faer_wasm_blas::C32;

#[test]
fn cgerc_bit_for_bit_and_geru_cross_check() {
	let mut rng = Lcg(124);
	let alpha = C32::new(0.9, -0.4);
	for &(m, n) in SHAPES {
		let cs = m + 1;
		let a0 = rng.mat_c32(m, n, cs);
		let x = rng.vec_c32(m);
		let y = rng.vec_c32(n);
		let mut a = a0.clone();
		cgerc(alpha, m, n, &mut a, cs, &x, &y);
		for j in 0..n {
			let t = alpha * y[j].conj();
			for i in 0..m {
				let want = a0[j * cs + i] + t * x[i];
				assert!(bits_eq_cc(a[j * cs + i], want), "cgerc {m}x{n} ({i},{j})");
			}
		}
		// cgerc(y) must be bit-identical to cgeru(conj(y))
		let yc: Vec<C32> = y.iter().map(|v| v.conj()).collect();
		let mut a2 = a0.clone();
		cgeru(alpha, m, n, &mut a2, cs, &x, &yc);
		for (p, q) in a.iter().zip(&a2) {
			if !p.re.is_nan() {
				assert!(bits_eq_cc(*p, *q), "cgerc vs cgeru(conj y) {m}x{n}");
			}
		}
	}
}
