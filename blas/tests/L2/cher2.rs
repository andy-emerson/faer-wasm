use super::common::*;
use faer_wasm_blas::L2::*;
use faer_wasm_blas::C32;

#[test]
fn cher2_bit_for_bit_and_real_diagonal() {
	let mut rng = Lcg(127);
	let alpha = C32::new(0.6, -0.9);
	for &n in NS {
		for upper in [true, false] {
			let cs = n + 1;
			let a0 = rng.mat_c32(n, n, cs);
			let x = rng.vec_c32(n);
			let y = rng.vec_c32(n);
			let mut a = a0.clone();
			cher2(alpha, n, &mut a, cs, upper, &x, &y);
			for j in 0..n {
				let t1 = alpha * y[j].conj();
				let t2 = (alpha * x[j]).conj();
				let (lo, hi) = if upper { (0, j) } else { (j + 1, n) };
				for i in lo..hi {
					// two caxpy passes in order: x-sourced then y-sourced
					let want = (a0[j * cs + i] + t1 * x[i]) + t2 * y[i];
					assert!(
						bits_eq_cc(a[j * cs + i], want),
						"cher2 upper={upper} n={n} ({i},{j})"
					);
				}
				let wd = C32::new(
					a0[j * cs + j].re + ((x[j] * t1).re + (y[j] * t2).re),
					0.0,
				);
				assert!(bits_eq_cc(a[j * cs + j], wd), "cher2 diag upper={upper} n={n} j={j}");
				assert_eq!(a[j * cs + j].im.to_bits(), 0.0f32.to_bits());
			}
		}
	}
}
