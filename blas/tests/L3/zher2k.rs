use super::common::*;
use faer_wasm_blas::L3::*;
use faer_wasm_blas::C64;

#[test]
fn zher2k_bit_for_bit_and_real_diagonal() {
	let mut rng = Lcg(134);
	let alpha = C64::new(0.5, -0.7);
	let beta = 0.25f64;
	for &n in NS {
		let k = (n / 2).max(1);
		for upper in [true, false] {
			let (acs, bcs, ccs) = (n + 1, n + 2, n + 1);
			let a = rng.mat_c64(n, k, acs);
			let b = rng.mat_c64(n, k, bcs);
			let c0 = rng.mat_c64(n, n, ccs);

			let mut c = c0.clone();
			zher2k(alpha, n, k, &a, acs, &b, bcs, beta, &mut c, ccs, upper);
			// replay: real-β component scale; per l the A-sourced adds
			// (t = α·conj(B[j,l])) precede the B-sourced adds
			// (t = conj(α·A[j,l])); diagonal imag forced to 0 at the end
			let mut cr = c0.clone();
			for j in 0..n {
				let (lo, hi) = if upper { (0, j + 1) } else { (j, n) };
				for i in lo..hi {
					let v = cr[j * ccs + i];
					cr[j * ccs + i] = C64::new(v.re * beta, v.im * beta);
				}
				for l in 0..k {
					let tb = alpha * b[l * bcs + j].conj();
					for i in lo..hi {
						cr[j * ccs + i] = cr[j * ccs + i] + tb * a[l * acs + i];
					}
					let ta = (alpha * a[l * acs + j]).conj();
					for i in lo..hi {
						cr[j * ccs + i] = cr[j * ccs + i] + ta * b[l * bcs + i];
					}
				}
				let d = cr[j * ccs + j];
				cr[j * ccs + j] = C64::new(d.re, 0.0);
			}
			for j in 0..n {
				for i in 0..n {
					assert!(
						bits_eq_c(c[j * ccs + i], cr[j * ccs + i]),
						"zher2k upper={upper} n={n} ({i},{j})"
					);
				}
			}
			for j in 0..n {
				assert_eq!(c[j * ccs + j].im.to_bits(), 0.0f64.to_bits(), "diag j={j}");
			}
		}
	}
}
