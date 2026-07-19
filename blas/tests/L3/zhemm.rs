use super::common::*;
use faer_wasm_blas::L3::*;
use faer_wasm_blas::C64;

#[test]
fn zhemm_both_sides_bounded_right_bit_replay() {
	let mut rng = Lcg(132);
	for &n in NS {
		let m = n; // square B keeps the reference simple
		for upper in [true, false] {
			let acs = n + 1;
			// full Hermitian ground truth (real diagonal), one
			// triangle exposed; stored diagonal imag poisoned to prove
			// it is ignored
			let mut full = vec![C64::ZERO; n * n];
			for j in 0..n {
				for i in 0..j {
					let v = rng.next_c64();
					full[j * n + i] = v;
					full[i * n + j] = v.conj();
				}
				full[j * n + j] = C64::new(rng.next_f64(), 0.0);
			}
			let nan = C64::new(f64::NAN, f64::NAN);
			let mut a = vec![nan; if n == 0 { 0 } else { acs * (n - 1) + n }];
			for j in 0..n {
				for i in 0..n {
					if if upper { i <= j } else { i >= j } {
						a[j * acs + i] = full[j * n + i];
					}
				}
			}
			for j in 0..n {
				a[j * acs + j].im = 42.0;
			}
			let (bcs, ccs) = (m + 2, m + 1);
			let b = rng.mat_c64(m, n, bcs);
			let c0 = rng.mat_c64(m, n, ccs);
			let alpha = C64::new(0.8, -0.3);
			let beta = C64::new(0.3, 0.5);

			let mut c = c0.clone();
			zhemm_left(alpha, m, n, &a, acs, upper, &b, bcs, beta, &mut c, ccs);
			for j in 0..n {
				for i in 0..m {
					let want = alpha
						* comp_sum_c((0..n).map(|l| full[l * n + i] * b[j * bcs + l]))
						+ beta * c0[j * ccs + i];
					let scale = comp_scale_c((0..n).map(|l| full[l * n + i] * b[j * bcs + l]))
						* (alpha.abs1() + 1.0)
						+ (beta * c0[j * ccs + i]).abs1();
					let tol = f64::EPSILON * (n.max(1) as f64) * 16.0 * scale + 1e-300;
					assert!(
						(c[j * ccs + i].re - want.re).abs()
							+ (c[j * ccs + i].im - want.im).abs()
							<= tol,
						"zhemm_left ({i},{j})"
					);
				}
			}

			let mut c = c0.clone();
			zhemm_right(alpha, m, n, &a, acs, upper, &b, bcs, beta, &mut c, ccs);
			for j in 0..n {
				for i in 0..m {
					let want = alpha
						* comp_sum_c((0..n).map(|l| b[l * bcs + i] * full[j * n + l]))
						+ beta * c0[j * ccs + i];
					let scale = comp_scale_c((0..n).map(|l| b[l * bcs + i] * full[j * n + l]))
						* (alpha.abs1() + 1.0)
						+ (beta * c0[j * ccs + i]).abs1();
					let tol = f64::EPSILON * (n.max(1) as f64) * 16.0 * scale + 1e-300;
					assert!(
						(c[j * ccs + i].re - want.re).abs()
							+ (c[j * ccs + i].im - want.im).abs()
							<= tol,
						"zhemm_right ({i},{j})"
					);
				}
			}
			// same-order scalar replay for the right side (β scale,
			// then k ascending — the fan-out grouping does not change
			// any element's sequence)
			let mut cr = c0.clone();
			for j in 0..n {
				for i in 0..m {
					cr[j * ccs + i] = beta * cr[j * ccs + i];
				}
				for k in 0..n {
					let t = alpha * full[j * n + k];
					for i in 0..m {
						cr[j * ccs + i] = cr[j * ccs + i] + t * b[k * bcs + i];
					}
				}
			}
			for j in 0..n {
				for i in 0..m {
					assert!(
						bits_eq_c(c[j * ccs + i], cr[j * ccs + i]),
						"zhemm_right bits n={n} ({i},{j})"
					);
				}
			}
		}
	}
}
