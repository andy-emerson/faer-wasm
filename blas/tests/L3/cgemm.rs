use super::common::*;
use faer_wasm_blas::L3::*;
use faer_wasm_blas::C64;
use faer_wasm_blas::C32;

#[test]
fn cgemm_bit_for_bit_and_bounded() {
	let mut rng = Lcg(131);
	for &(m, k, n) in DIMS {
		let (acs, bcs, ccs) = (m + 1, k + 2, m + 3);
		let a = rng.mat_c32(m, k, acs);
		let b = rng.mat_c32(k, n, bcs);
		let c0 = rng.mat_c32(m, n, ccs);
		for (alpha, beta) in [
			(C32::ONE, C32::ZERO),
			(C32::new(-0.7, 0.4), C32::new(0.4, -0.2)),
		] {
			let mut c = c0.clone();
			cgemm(alpha, m, k, n, &a, acs, &b, bcs, beta, &mut c, ccs);
			// same-order scalar replay (cgemv per column = column-caxpy)
			let mut cr = c0.clone();
			for j in 0..n {
				if beta == C32::ZERO {
					for i in 0..m {
						cr[j * ccs + i] = C32::ZERO;
					}
				} else if beta != C32::ONE {
					for i in 0..m {
						cr[j * ccs + i] = beta * cr[j * ccs + i];
					}
				}
				for l in 0..k {
					let t = alpha * b[j * bcs + l];
					for i in 0..m {
						cr[j * ccs + i] = cr[j * ccs + i] + t * a[l * acs + i];
					}
				}
			}
			for j in 0..n {
				for i in 0..m {
					assert!(
						bits_eq_cc(c[j * ccs + i], cr[j * ccs + i]),
						"cgemm bits {m}x{k}x{n} ({i},{j})"
					);
				}
			}
			// independent bound, different accumulation order
			for j in 0..n {
				for i in 0..m {
					let bt = if beta == C32::ZERO { C64::ZERO } else { c_up(beta * c0[j * ccs + i]) };
					let want =
						c_up(alpha) * comp_sum_cc((0..k).map(|l| a[l * acs + i] * b[j * bcs + l])) + bt;
					let scale = comp_scale_cc((0..k).map(|l| a[l * acs + i] * b[j * bcs + l]))
						* (alpha.abs1() as f64 + 1.0)
						+ bt.abs1();
					let tol = EPS * (k.max(1) as f64) * 16.0 * scale + 1e-30;
					assert!(
						(c[j * ccs + i].re as f64 - want.re).abs() + (c[j * ccs + i].im as f64 - want.im).abs()
							<= tol,
						"cgemm bound ({i},{j})"
					);
				}
			}
		}
	}
}

#[test]
fn cgemm_col4_bit_identical_to_colaxpy() {
	let mut rng = Lcg(136);
	// sizes crossing the 4-column group boundary: multiples, tails, tiny
	for &(m, k, n) in &[
		(8usize, 8usize, 8usize),
		(4, 4, 4),
		(12, 7, 8),
		(9, 5, 10),
		(7, 3, 6),
		(3, 2, 3),
		(1, 1, 1),
		(0, 0, 0),
		(16, 16, 5),
		(5, 16, 16),
	] {
		let (acs, bcs, ccs) = (m + 1, k + 2, m + 3);
		let a = rng.mat_c32(m, k, acs);
		let b = rng.mat_c32(k, n, bcs);
		let c0 = rng.mat_c32(m, n, ccs);
		for (alpha, beta) in [
			(C32::ONE, C32::ZERO),
			(C32::new(-0.7, 0.2), C32::new(0.4, 0.1)),
			(C32::new(0.3, -0.6), C32::ONE),
		] {
			let mut c1 = c0.clone();
			cgemm_colaxpy(alpha, m, k, n, &a, acs, &b, bcs, beta, &mut c1, ccs);
			let mut c2 = c0.clone();
			cgemm(alpha, m, k, n, &a, acs, &b, bcs, beta, &mut c2, ccs);
			for j in 0..n {
				for i in 0..m {
					assert!(
						bits_eq_cc(c1[j * ccs + i], c2[j * ccs + i]),
						"cgemm col4 vs colaxpy {m}x{k}x{n} ({i},{j})"
					);
				}
			}
		}
	}
}

#[test]
#[should_panic(expected = "storage too short")]
fn cgemm_short_storage_panics() {
	let a = [C32::ONE; 4];
	let b = [C32::ONE; 3];
	cgemm(C32::ONE, 2, 2, 2, &a, 2, &b, 2, C32::ZERO, &mut [C32::ZERO; 4], 2);
}

#[test]
fn cgemm_packed_bit_identical_to_colaxpy() {
	let mut rng = Lcg(142);
	// sizes crossing the packed-path boundaries: KC (256) exact / +1 /
	// with remainder, MC row-blocking, MR=4 row tails, column tails
	for &(m, k, n) in &[
		(8usize, 8usize, 8usize),
		(12, 7, 8),
		(9, 5, 10),
		(1, 1, 1),
		(0, 0, 0),
		(5, 16, 16),
		(8, 300, 8),
		(140, 260, 12),
		(16, 256, 4),
		(33, 257, 7),
	] {
		let (acs, bcs, ccs) = (m + 1, k + 2, m + 3);
		let a = rng.mat_c32(m, k, acs);
		let b = rng.mat_c32(k, n, bcs);
		let c0 = rng.mat_c32(m, n, ccs);
		for (alpha, beta) in [
			(C32::ONE, C32::ZERO),
			(C32::new(-0.7, 0.2), C32::new(0.4, 0.1)),
			(C32::new(0.3, -0.6), C32::ONE),
		] {
			let mut c1 = c0.clone();
			cgemm_colaxpy(alpha, m, k, n, &a, acs, &b, bcs, beta, &mut c1, ccs);
			let mut c2 = c0.clone();
			cgemm_packed(alpha, m, k, n, &a, acs, &b, bcs, beta, &mut c2, ccs);
			for j in 0..n {
				for i in 0..m {
					assert_eq!(
						c1[j * ccs + i].re.to_bits(),
						c2[j * ccs + i].re.to_bits(),
						"packed vs column-caxpy re {m}x{k}x{n} ({i},{j})"
					);
					assert_eq!(
						c1[j * ccs + i].im.to_bits(),
						c2[j * ccs + i].im.to_bits(),
						"packed vs column-caxpy im {m}x{k}x{n} ({i},{j})"
					);
				}
			}
		}
	}
}

#[test]
fn cgemm_dispatch_packed_zone_bit_identical() {
	// A = 1040x1024x8B = 8.1 MB >= the 8 MB packed threshold — the
	// dispatcher routes packed; replay against colaxpy
	let mut rng = Lcg(144);
	let (m, k, n) = (1040, 1024, 8);
	let (acs, bcs, ccs) = (m, k, m);
	let a = rng.mat_c32(m, k, acs);
	let b = rng.mat_c32(k, n, bcs);
	let c0 = rng.mat_c32(m, n, ccs);
	let (alpha, beta) = (C32::new(-0.7, 0.2), C32::new(0.4, 0.1));
	let mut c1 = c0.clone();
	cgemm_colaxpy(alpha, m, k, n, &a, acs, &b, bcs, beta, &mut c1, ccs);
	let mut c2 = c0.clone();
	cgemm(alpha, m, k, n, &a, acs, &b, bcs, beta, &mut c2, ccs);
	for i in 0..c1.len() {
		assert_eq!(c1[i].re.to_bits(), c2[i].re.to_bits(), "dispatch(packed) re @{i}");
		assert_eq!(c1[i].im.to_bits(), c2[i].im.to_bits(), "dispatch(packed) im @{i}");
	}
}
