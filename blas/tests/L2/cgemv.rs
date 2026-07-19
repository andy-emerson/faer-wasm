use super::common::*;
use faer_wasm_blas::L2::*;
use faer_wasm_blas::C64;
use faer_wasm_blas::C32;

#[test]
fn cgemv_bit_for_bit_and_bounded() {
	let mut rng = Lcg(121);
	for &(m, n) in SHAPES {
		for &pad in PADS {
			let cs = m + pad;
			let a = rng.mat_c32(m, n, cs);
			let x = rng.vec_c32(n);
			let y0 = rng.vec_c32(m);
			for (alpha, beta) in [
				(C32::ONE, C32::ZERO),
				(C32::new(-0.5, 0.4), C32::ONE),
				(C32::new(0.3, -0.2), C32::new(-1.1, 0.6)),
			] {
				let mut y = y0.clone();
				cgemv(alpha, m, n, &a, cs, &x, beta, &mut y);
				// same-order scalar replay of column-caxpy
				let mut yr = y0.clone();
				if beta == C32::ZERO {
					yr.fill(C32::ZERO);
				} else if beta != C32::ONE {
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
					assert!(bits_eq_cc(y[i], yr[i]), "cgemv bits {m}x{n} pad={pad} i={i}");
				}
				// independent bound: compensated row-dots, different order
				for i in 0..m {
					let bt = if beta == C32::ZERO { C64::ZERO } else { c_up(beta * y0[i]) };
					let want = comp_sum_cc((0..n).map(|j| alpha * x[j] * a[j * cs + i])) + bt;
					let scale = comp_scale_cc((0..n).map(|j| alpha * x[j] * a[j * cs + i]))
						+ bt.abs1();
					let tol = EPS * (n.max(1) as f64) * 8.0 * scale + 1e-30;
					assert!(
						(y[i].re as f64 - want.re).abs() + (y[i].im as f64 - want.im).abs() <= tol,
						"cgemv bound {m}x{n} i={i}"
					);
				}
			}
		}
	}
}

#[test]
fn cgemv_t_and_c_bounded_and_cross_checked() {
	let mut rng = Lcg(122);
	for &(m, n) in SHAPES {
		let cs = m + 2;
		let a = rng.mat_c32(m, n, cs);
		let x = rng.vec_c32(m);
		let y0 = rng.vec_c32(n);
		let alpha = C32::new(0.7, -0.3);
		let beta = C32::new(-0.3, 0.1);
		let mut yt = y0.clone();
		cgemv_t(alpha, m, n, &a, cs, &x, beta, &mut yt);
		let mut yc = y0.clone();
		cgemv_c(alpha, m, n, &a, cs, &x, beta, &mut yc);
		for j in 0..n {
			let bt = c_up(beta * y0[j]);
			// transpose form: Σ a[i,j]·x[i]
			let want_t = c_up(alpha) * comp_sum_cc((0..m).map(|i| a[j * cs + i] * x[i])) + bt;
			// conjugate-transpose form: Σ conj(a[i,j])·x[i]
			let want_c = c_up(alpha) * comp_sum_cc((0..m).map(|i| a[j * cs + i].conj() * x[i])) + bt;
			let scale_t = comp_scale_cc((0..m).map(|i| a[j * cs + i] * x[i]))
				.max(1.0) * (alpha.abs1() as f64 + bt.abs1() + 1.0);
			let tol = EPS * (m.max(1) as f64) * 16.0 * scale_t + 1e-30;
			assert!(
				(yt[j].re as f64 - want_t.re).abs() + (yt[j].im as f64 - want_t.im).abs() <= tol,
				"cgemv_t {m}x{n} j={j}"
			);
			assert!(
				(yc[j].re as f64 - want_c.re).abs() + (yc[j].im as f64 - want_c.im).abs() <= tol,
				"cgemv_c {m}x{n} j={j}"
			);
		}
		// conjugation folds into the lane signs exactly: cgemv_c on A
		// must be bit-identical to cgemv_t on conj(A)
		let ac: Vec<C32> = a.iter().map(|v| v.conj()).collect();
		let mut yt2 = y0.clone();
		cgemv_t(alpha, m, n, &ac, cs, &x, beta, &mut yt2);
		for j in 0..n {
			assert!(bits_eq_cc(yc[j], yt2[j]), "cgemv_c vs cgemv_t(conj A) {m}x{n} j={j}");
		}
	}
}

#[test]
#[should_panic(expected = "length mismatch")]
fn cgemv_length_mismatch_panics() {
	let a = [C32::ONE; 4];
	cgemv(C32::ONE, 2, 2, &a, 2, &[C32::ONE], C32::ZERO, &mut [C32::ZERO; 2]);
}
