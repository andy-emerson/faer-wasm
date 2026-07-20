use super::common::*;
use faer_wasm_blas::L3::*;

#[test]
fn gemm_bit_for_bit_and_bounded() {
	let mut rng = Lcg(31);
	for &(m, k, n) in DIMS {
		let (acs, bcs, ccs) = (m + 1, k + 2, m + 3);
		let a = rng.mat_f32(m, k, acs);
		let b = rng.mat_f32(k, n, bcs);
		let c0 = rng.mat_f32(m, n, ccs);
		for (alpha, beta) in [(1.0, 0.0), (-0.7, 0.4)] {
			let mut c = c0.clone();
			sgemm(alpha, m, k, n, &a, acs, &b, bcs, beta, &mut c, ccs);
			// same-order scalar replay (sgemv per column = column-saxpy)
			let mut cr = c0.clone();
			for j in 0..n {
				if beta == 0.0 {
					for i in 0..m {
						cr[j * ccs + i] = 0.0;
					}
				} else if beta != 1.0 {
					for i in 0..m {
						cr[j * ccs + i] *= beta;
					}
				}
				for l in 0..k {
					let t = alpha * b[j * bcs + l];
					for i in 0..m {
						cr[j * ccs + i] += a[l * acs + i] * t;
					}
				}
			}
			for j in 0..n {
				for i in 0..m {
					assert_eq!(
						c[j * ccs + i].to_bits(),
						cr[j * ccs + i].to_bits(),
						"sgemm bits {m}x{k}x{n} ({i},{j})"
					);
				}
			}
			// independent bound, different accumulation order
			for j in 0..n {
				for i in 0..m {
					let want = alpha as f64 * comp_sum32((0..k).map(|l| a[l * acs + i] * b[j * bcs + l]))
						+ if beta == 0.0 { 0.0 } else { (beta * c0[j * ccs + i]) as f64 };
					let scale = comp_sum32((0..k).map(|l| (a[l * acs + i] * b[j * bcs + l]).abs()))
						+ c0[j * ccs + i].abs() as f64;
					let tol = f32::EPSILON as f64 * (k.max(1) as f64) * 8.0 * scale + 1e-40;
					assert!((c[j * ccs + i] as f64 - want).abs() <= tol, "sgemm bound ({i},{j})");
				}
			}
		}
	}
}

#[test]
fn gemm_tiled_bit_identical_to_gemm() {
	let mut rng = Lcg(36);
	// sizes crossing every tile boundary: exact multiples, tails in m,
	// n, both, and tiny
	for &(m, k, n) in &[(8usize, 8usize, 8usize), (4, 4, 4), (12, 7, 8), (9, 5, 10), (7, 3, 6), (3, 2, 3), (1, 1, 1), (0, 0, 0), (16, 16, 5), (5, 16, 16)] {
		let (acs, bcs, ccs) = (m + 1, k + 2, m + 3);
		let a = rng.mat_f32(m, k, acs);
		let b = rng.mat_f32(k, n, bcs);
		let c0 = rng.mat_f32(m, n, ccs);
		for (alpha, beta) in [(1.0, 0.0), (-0.7, 0.4), (0.3, 1.0)] {
			let mut c1 = c0.clone();
			sgemm_colaxpy(alpha, m, k, n, &a, acs, &b, bcs, beta, &mut c1, ccs);
			let mut cd = c0.clone();
			sgemm(alpha, m, k, n, &a, acs, &b, bcs, beta, &mut cd, ccs);
			for j in 0..n {
				for i in 0..m {
					assert_eq!(c1[j * ccs + i].to_bits(), cd[j * ccs + i].to_bits(), "dispatcher vs colaxpy");
				}
			}
			let mut c2 = c0.clone();
			sgemm_tiled(alpha, m, k, n, &a, acs, &b, bcs, beta, &mut c2, ccs);
			let mut c3 = c0.clone();
			sgemm_col4(alpha, m, k, n, &a, acs, &b, bcs, beta, &mut c3, ccs);
			for j in 0..n {
				for i in 0..m {
					assert_eq!(
						c1[j * ccs + i].to_bits(),
						c2[j * ccs + i].to_bits(),
						"tiled vs column-saxpy {m}x{k}x{n} ({i},{j})"
					);
					assert_eq!(
						c1[j * ccs + i].to_bits(),
						c3[j * ccs + i].to_bits(),
						"col4 vs column-saxpy {m}x{k}x{n} ({i},{j})"
					);
				}
			}
		}
	}
}

#[test]
#[should_panic(expected = "storage too short")]
fn gemm_short_storage_panics() {
	sgemm(1.0, 2, 2, 2, &[1.0; 4], 2, &[1.0; 3], 2, 0.0, &mut [0.0; 4], 2);
}

#[test]
fn gemm_packed_bit_identical_to_colaxpy() {
	let mut rng = Lcg(42);
	// sizes crossing the packed-path boundaries: KC (256) exact / +1 /
	// with remainder, MC row-blocking, MR=8 row tails, 4-column tails
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
		let a = rng.mat_f32(m, k, acs);
		let b = rng.mat_f32(k, n, bcs);
		let c0 = rng.mat_f32(m, n, ccs);
		for (alpha, beta) in [(1.0f32, 0.0f32), (-0.7, 0.4), (0.3, 1.0)] {
			let mut c1 = c0.clone();
			sgemm_colaxpy(alpha, m, k, n, &a, acs, &b, bcs, beta, &mut c1, ccs);
			let mut c2 = c0.clone();
			sgemm_packed(alpha, m, k, n, &a, acs, &b, bcs, beta, &mut c2, ccs);
			for j in 0..n {
				for i in 0..m {
					assert_eq!(
						c1[j * ccs + i].to_bits(),
						c2[j * ccs + i].to_bits(),
						"packed vs column-saxpy {m}x{k}x{n} ({i},{j})"
					);
				}
			}
		}
	}
}

#[test]
fn gemm_dispatch_packed_zone_bit_identical() {
	// A = 1032x800x4B = 3.3 MB > the 3 MB tiled threshold — the
	// dispatcher routes packed; replay against colaxpy
	let mut rng = Lcg(44);
	let (m, k, n) = (1032, 800, 8);
	let (acs, bcs, ccs) = (m, k, m);
	let a = rng.mat_f32(m, k, acs);
	let b = rng.mat_f32(k, n, bcs);
	let c0 = rng.mat_f32(m, n, ccs);
	let mut c1 = c0.clone();
	sgemm_colaxpy(-0.7, m, k, n, &a, acs, &b, bcs, 0.4, &mut c1, ccs);
	let mut c2 = c0.clone();
	sgemm(-0.7, m, k, n, &a, acs, &b, bcs, 0.4, &mut c2, ccs);
	for i in 0..c1.len() {
		assert_eq!(c1[i].to_bits(), c2[i].to_bits(), "dispatch(packed zone) vs colaxpy @{i}");
	}
}
