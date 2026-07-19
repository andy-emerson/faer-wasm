use super::common::*;
use faer_wasm_blas::L2::*;
use faer_wasm_blas::C64;

#[test]
fn zher_bit_for_bit_and_real_diagonal() {
	let mut rng = Lcg(126);
	let alpha = -0.8f64;
	for &n in NS {
		for upper in [true, false] {
			let cs = n + 1;
			let a0 = rng.mat_c64(n, n, cs);
			let x = rng.vec_c64(n);
			let mut a = a0.clone();
			zher(alpha, n, &mut a, cs, upper, &x);
			for j in 0..n {
				// replay: zaxpy over the strict stored segment with
				// t = α·conj(x[j]), then the DBLE diagonal update
				let t = x[j].conj().scale(alpha);
				let (lo, hi) = if upper { (0, j) } else { (j + 1, n) };
				for i in lo..hi {
					let want = a0[j * cs + i] + t * x[i];
					assert!(
						bits_eq_c(a[j * cs + i], want),
						"zher upper={upper} n={n} ({i},{j})"
					);
				}
				let wd = C64::new(a0[j * cs + j].re + (x[j] * t).re, 0.0);
				assert!(bits_eq_c(a[j * cs + j], wd), "zher diag upper={upper} n={n} j={j}");
				// Hermitian invariant: diagonal imag is exactly +0.0
				assert_eq!(a[j * cs + j].im.to_bits(), 0.0f64.to_bits());
			}
		}
	}
}
