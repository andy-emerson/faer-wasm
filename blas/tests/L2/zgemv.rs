use super::common::*;
use faer_wasm_blas::L2::*;
use faer_wasm_blas::C64;

#[test]
fn zgemv_bit_for_bit_and_bounded() {
	let mut rng = Lcg(121);
	for &(m, n) in SHAPES {
		for &pad in PADS {
			let cs = m + pad;
			let a = rng.mat_c64(m, n, cs);
			let x = rng.vec_c64(n);
			let y0 = rng.vec_c64(m);
			for (alpha, beta) in [
				(C64::ONE, C64::ZERO),
				(C64::new(-0.5, 0.4), C64::ONE),
				(C64::new(0.3, -0.2), C64::new(-1.1, 0.6)),
			] {
				let mut y = y0.clone();
				zgemv(alpha, m, n, &a, cs, &x, beta, &mut y);
				// same-order scalar replay of column-zaxpy
				let mut yr = y0.clone();
				if beta == C64::ZERO {
					yr.fill(C64::ZERO);
				} else if beta != C64::ONE {
					for v in yr.iter_mut() {
						*v = beta * *v;
					}
				}
				for j in 0..n {
					let t = alpha * x[j];
					for i in 0..m {
						yr[i] = yr[i] + t * a[j * cs + i];
					}
				}
				for i in 0..m {
					assert!(bits_eq_c(y[i], yr[i]), "zgemv bits {m}x{n} pad={pad} i={i}");
				}
				// independent bound: compensated row-dots, different order
				for i in 0..m {
					let bt = if beta == C64::ZERO { C64::ZERO } else { beta * y0[i] };
					let want = comp_sum_c((0..n).map(|j| alpha * x[j] * a[j * cs + i])) + bt;
					let scale = comp_scale_c((0..n).map(|j| alpha * x[j] * a[j * cs + i]))
						+ bt.abs1();
					let tol = f64::EPSILON * (n.max(1) as f64) * 8.0 * scale + 1e-300;
					assert!(
						(y[i].re - want.re).abs() + (y[i].im - want.im).abs() <= tol,
						"zgemv bound {m}x{n} i={i}"
					);
				}
			}
		}
	}
}

#[test]
fn zgemv_t_and_c_bounded_and_cross_checked() {
	let mut rng = Lcg(122);
	for &(m, n) in SHAPES {
		let cs = m + 2;
		let a = rng.mat_c64(m, n, cs);
		let x = rng.vec_c64(m);
		let y0 = rng.vec_c64(n);
		let alpha = C64::new(0.7, -0.3);
		let beta = C64::new(-0.3, 0.1);
		let mut yt = y0.clone();
		zgemv_t(alpha, m, n, &a, cs, &x, beta, &mut yt);
		let mut yc = y0.clone();
		zgemv_c(alpha, m, n, &a, cs, &x, beta, &mut yc);
		for j in 0..n {
			let bt = beta * y0[j];
			// transpose form: Σ a[i,j]·x[i]
			let want_t = alpha * comp_sum_c((0..m).map(|i| a[j * cs + i] * x[i])) + bt;
			// conjugate-transpose form: Σ conj(a[i,j])·x[i]
			let want_c = alpha * comp_sum_c((0..m).map(|i| a[j * cs + i].conj() * x[i])) + bt;
			let scale_t = comp_scale_c((0..m).map(|i| a[j * cs + i] * x[i]))
				.max(1.0) * (alpha.abs1() + bt.abs1() + 1.0);
			let tol = f64::EPSILON * (m.max(1) as f64) * 16.0 * scale_t + 1e-300;
			assert!(
				(yt[j].re - want_t.re).abs() + (yt[j].im - want_t.im).abs() <= tol,
				"zgemv_t {m}x{n} j={j}"
			);
			assert!(
				(yc[j].re - want_c.re).abs() + (yc[j].im - want_c.im).abs() <= tol,
				"zgemv_c {m}x{n} j={j}"
			);
		}
		// conjugation folds into the lane signs exactly: zgemv_c on A
		// must be bit-identical to zgemv_t on conj(A)
		let ac: Vec<C64> = a.iter().map(|v| v.conj()).collect();
		let mut yt2 = y0.clone();
		zgemv_t(alpha, m, n, &ac, cs, &x, beta, &mut yt2);
		for j in 0..n {
			assert!(bits_eq_c(yc[j], yt2[j]), "zgemv_c vs zgemv_t(conj A) {m}x{n} j={j}");
		}
	}
}

#[test]
#[should_panic(expected = "length mismatch")]
fn zgemv_length_mismatch_panics() {
	let a = [C64::ONE; 4];
	zgemv(C64::ONE, 2, 2, &a, 2, &[C64::ONE], C64::ZERO, &mut [C64::ZERO; 2]);
}
