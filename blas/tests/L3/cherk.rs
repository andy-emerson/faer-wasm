use super::common::*;
use faer_wasm_blas::L3::*;
use faer_wasm_blas::C32;

#[test]
fn cherk_bit_for_bit_and_real_diagonal() {
	let mut rng = Lcg(133);
	let (alpha, beta) = (0.5f32, 0.25f32);
	for &n in NS {
		let k = (n / 2).max(1);
		for upper in [true, false] {
			let (acs, ccs) = (n + 1, n + 1);
			let a = rng.mat_c32(n, k, acs);
			let c0 = rng.mat_c32(n, n, ccs);

			let mut c = c0.clone();
			cherk(alpha, n, k, &a, acs, beta, &mut c, ccs, upper);
			// replay: real-β component scale, l ascending with
			// t = α·conj(A[j,l]), then the diagonal's imag forced to 0
			let mut cr = c0.clone();
			for j in 0..n {
				let (lo, hi) = if upper { (0, j + 1) } else { (j, n) };
				for i in lo..hi {
					let v = cr[j * ccs + i];
					cr[j * ccs + i] = C32::new(v.re * beta, v.im * beta);
				}
				for l in 0..k {
					let t = a[l * acs + j].conj().scale(alpha);
					for i in lo..hi {
						cr[j * ccs + i] = cr[j * ccs + i] + t * a[l * acs + i];
					}
				}
				let d = cr[j * ccs + j];
				cr[j * ccs + j] = C32::new(d.re, 0.0);
			}
			for j in 0..n {
				for i in 0..n {
					assert!(
						bits_eq_cc(c[j * ccs + i], cr[j * ccs + i]),
						"cherk upper={upper} n={n} ({i},{j})"
					);
				}
			}
			// Hermitian invariant: diagonal imag exactly +0.0
			for j in 0..n {
				assert_eq!(c[j * ccs + j].im.to_bits(), 0.0f32.to_bits(), "diag j={j}");
			}
		}
	}
}
