use super::common::*;
use faer_wasm_blas::L2::*;
use faer_wasm_blas::C64;

#[test]
fn ztrmv_bit_for_bit_all_variants() {
	let mut rng = Lcg(128);
	for &n in NS {
		for upper in [true, false] {
			for unit in [true, false] {
				let cs = n + 1;
				let a = rng.mat_c64(n, n, cs);
				let x0 = rng.vec_c64(n);
				let mut x = x0.clone();
				ztrmv(n, &a, cs, upper, unit, &mut x);
				// same-order scalar replay
				let mut xr = x0.clone();
				if upper {
					for j in 0..n {
						let t = xr[j];
						for i in 0..j {
							xr[i] = xr[i] + t * a[j * cs + i];
						}
						if !unit {
							xr[j] = t * a[j * cs + j];
						}
					}
				} else {
					for j in (0..n).rev() {
						let t = xr[j];
						for i in j + 1..n {
							xr[i] = xr[i] + t * a[j * cs + i];
						}
						if !unit {
							xr[j] = t * a[j * cs + j];
						}
					}
				}
				for i in 0..n {
					assert!(
						bits_eq_c(x[i], xr[i]),
						"ztrmv upper={upper} unit={unit} n={n} i={i}"
					);
				}
			}
		}
	}
}
