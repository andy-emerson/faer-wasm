use super::common::*;
use faer_wasm_blas::L2::*;
use faer_wasm_blas::C32;

#[test]
fn chemv_bounded_both_triangles() {
	let mut rng = Lcg(125);
	for &n in NS {
		for upper in [true, false] {
			let cs = n + 1;
			// full Hermitian ground truth (real diagonal), one
			// triangle exposed — the hidden triangle's slots carry
			// junk that must never be read
			let mut full = vec![C32::ZERO; n * n];
			for j in 0..n {
				for i in 0..j {
					let v = rng.next_c32();
					full[j * n + i] = v;
					full[i * n + j] = v.conj();
				}
				full[j * n + j] = C32::new(rng.next_f32(), 0.0);
			}
			let nan = C32::new(f32::NAN, f32::NAN);
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
			let x = rng.vec_c32(n);
			let y0 = rng.vec_c32(n);
			let alpha = C32::new(0.9, 0.2);
			let beta = C32::new(0.4, -0.1);
			let mut y = y0.clone();
			chemv(alpha, n, &a, cs, upper, &x, beta, &mut y);
			for i in 0..n {
				let want = c_up(alpha) * comp_sum_cc((0..n).map(|j| full[j * n + i] * x[j]))
					+ c_up(beta * y0[i]);
				let scale = comp_scale_cc((0..n).map(|j| full[j * n + i] * x[j]))
					* (alpha.abs1() as f64 + 1.0)
					+ (beta * y0[i]).abs1() as f64;
				let tol = EPS * (n.max(1) as f64) * 16.0 * scale + 1e-30;
				assert!(
					(y[i].re as f64 - want.re).abs() + (y[i].im as f64 - want.im).abs() <= tol,
					"chemv upper={upper} n={n} i={i}"
				);
			}
		}
	}
}
