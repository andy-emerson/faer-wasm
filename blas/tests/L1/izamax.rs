use super::common::*;
use faer_wasm_blas::L1::*;
use faer_wasm_blas::C64;

#[test]
fn izamax_exact_index() {
	let mut rng = Lcg(112);
	for &n in SIZES {
		let x = rng.vec_c64(n);
		let got = izamax(&x);
		// scalar reference: first occurrence of the max |re| + |im|
		let mut want = 0usize;
		let mut best = -1.0f64;
		for (k, v) in x.iter().enumerate() {
			let m = v.re.abs() + v.im.abs();
			if m > best {
				best = m;
				want = k;
			}
		}
		if n == 0 {
			assert_eq!(got, 0, "empty");
		} else {
			assert_eq!(got, want, "izamax n={n}");
		}
	}
}

#[test]
fn izamax_first_occurrence_ties() {
	// four elements, all |re|+|im| = 3 — must return 0
	let x = [
		C64::new(1.0, 2.0),
		C64::new(-2.0, 1.0),
		C64::new(3.0, 0.0),
		C64::new(0.0, -3.0),
	];
	assert_eq!(izamax(&x), 0);
	// tie later in the vector: the first of the tied pair wins
	let y = [C64::new(1.0, 0.0), C64::new(2.0, 2.0), C64::new(-2.0, -2.0)];
	assert_eq!(izamax(&y), 1);
}
