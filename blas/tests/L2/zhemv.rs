use super::common::*;
use faer_wasm_blas::L2::*;
use faer_wasm_blas::C64;

#[test]
fn zhemv_bounded_both_triangles() {
	let mut rng = Lcg(125);
	for &n in NS {
		for upper in [true, false] {
			let cs = n + 1;
			// full Hermitian ground truth (real diagonal), one
			// triangle exposed — the hidden triangle's slots carry
			// junk that must never be read
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
			let mut a = vec![nan; if n == 0 { 0 } else { cs * (n - 1) + n }];
			for j in 0..n {
				for i in 0..n {
					let stored = if upper { i <= j } else { i >= j };
					if stored {
						a[j * cs + i] = full[j * n + i];
					}
				}
			}
			// stored diagonal imaginary parts are IGNORED by contract —
			// poison them to prove it
			for j in 0..n {
				a[j * cs + j].im = 42.0;
			}
			let x = rng.vec_c64(n);
			let y0 = rng.vec_c64(n);
			let alpha = C64::new(0.9, 0.2);
			let beta = C64::new(0.4, -0.1);
			let mut y = y0.clone();
			zhemv(alpha, n, &a, cs, upper, &x, beta, &mut y);
			for i in 0..n {
				let want = alpha * comp_sum_c((0..n).map(|j| full[j * n + i] * x[j]))
					+ beta * y0[i];
				let scale = comp_scale_c((0..n).map(|j| full[j * n + i] * x[j]))
					* (alpha.abs1() + 1.0)
					+ (beta * y0[i]).abs1();
				let tol = f64::EPSILON * (n.max(1) as f64) * 16.0 * scale + 1e-300;
				assert!(
					(y[i].re - want.re).abs() + (y[i].im - want.im).abs() <= tol,
					"zhemv upper={upper} n={n} i={i}"
				);
			}
		}
	}
}
