//! Level 1 correctness per the testing contract (../README.md):
//! elementwise streams bit-for-bit against the scalar definition;
//! reductions error-bounded against a compensated-summation reference;
//! iamax's index exact including first-occurrence tie-breaking;
//! nrm2's over/underflow guards exercised explicitly; rotg checked
//! against its defining identities and the reference edge cases.

use faer_wasm_blas::level1::*;

// Deterministic pseudo-random data (no external crates): LCG over u64,
// mapped to roughly [-2, 2] with varied magnitudes.
struct Lcg(u64);
impl Lcg {
	fn next_f64(&mut self) -> f64 {
		self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
		let bits = (self.0 >> 11) as f64 / (1u64 << 53) as f64; // [0,1)
		4.0 * bits - 2.0
	}
	fn vec(&mut self, n: usize) -> Vec<f64> {
		(0..n).map(|_| self.next_f64()).collect()
	}
}

// Sizes that exercise the empty case, the pure-tail path (< 4), unroll
// boundaries, and odd tails.
const SIZES: &[usize] = &[0, 1, 2, 3, 4, 5, 7, 8, 16, 33, 100, 257, 1000];

// Neumaier compensated summation — the higher-precision reference for
// reduction bounds.
fn comp_sum(it: impl Iterator<Item = f64>) -> f64 {
	let mut s = 0.0f64;
	let mut c = 0.0f64;
	for v in it {
		let t = s + v;
		if s.abs() >= v.abs() {
			c += (s - t) + v;
		} else {
			c += (v - t) + s;
		}
		s = t;
	}
	s + c
}

// ---- elementwise streams: bit-for-bit ----

#[test]
fn copy_bit_for_bit() {
	let mut rng = Lcg(1);
	for &n in SIZES {
		let x = rng.vec(n);
		let mut y = vec![0.0; n];
		copy(&x, &mut y);
		for i in 0..n {
			assert_eq!(x[i].to_bits(), y[i].to_bits(), "copy n={n} i={i}");
		}
	}
}

#[test]
fn swap_bit_for_bit() {
	let mut rng = Lcg(2);
	for &n in SIZES {
		let x0 = rng.vec(n);
		let y0 = rng.vec(n);
		let mut x = x0.clone();
		let mut y = y0.clone();
		swap(&mut x, &mut y);
		for i in 0..n {
			assert_eq!(x[i].to_bits(), y0[i].to_bits(), "swap n={n} i={i}");
			assert_eq!(y[i].to_bits(), x0[i].to_bits(), "swap n={n} i={i}");
		}
	}
}

#[test]
fn scal_bit_for_bit() {
	let mut rng = Lcg(3);
	for &n in SIZES {
		for alpha in [0.0, 1.0, -1.5, 0.3333333333333333, 1e100] {
			let x0 = rng.vec(n);
			let mut x = x0.clone();
			scal(alpha, &mut x);
			for i in 0..n {
				assert_eq!(x[i].to_bits(), (x0[i] * alpha).to_bits(), "scal n={n} i={i}");
			}
		}
	}
}

#[test]
fn axpy_bit_for_bit() {
	let mut rng = Lcg(4);
	for &n in SIZES {
		for alpha in [0.0, 1.0, -2.5, 0.1] {
			let x = rng.vec(n);
			let y0 = rng.vec(n);
			let mut y = y0.clone();
			axpy(alpha, &x, &mut y);
			for i in 0..n {
				let want = y0[i] + x[i] * alpha;
				assert_eq!(y[i].to_bits(), want.to_bits(), "axpy n={n} i={i}");
			}
		}
	}
}

#[test]
fn rot_bit_for_bit() {
	let mut rng = Lcg(5);
	let (c, s) = (0.8, 0.6);
	for &n in SIZES {
		let x0 = rng.vec(n);
		let y0 = rng.vec(n);
		let mut x = x0.clone();
		let mut y = y0.clone();
		rot(&mut x, &mut y, c, s);
		for i in 0..n {
			let wx = x0[i] * c + y0[i] * s;
			let wy = y0[i] * c - x0[i] * s;
			assert_eq!(x[i].to_bits(), wx.to_bits(), "rot x n={n} i={i}");
			assert_eq!(y[i].to_bits(), wy.to_bits(), "rot y n={n} i={i}");
		}
	}
}

// ---- reduction streams: error-bounded vs compensated reference ----

#[test]
fn dot_error_bounded() {
	let mut rng = Lcg(6);
	for &n in SIZES {
		let x = rng.vec(n);
		let y = rng.vec(n);
		let got = dot(&x, &y);
		let reference = comp_sum((0..n).map(|i| x[i] * y[i]));
		let scale = comp_sum((0..n).map(|i| (x[i] * y[i]).abs()));
		let tol = f64::EPSILON * (n.max(1) as f64) * scale + f64::MIN_POSITIVE;
		assert!(
			(got - reference).abs() <= tol,
			"dot n={n}: got {got}, ref {reference}, tol {tol}"
		);
	}
}

#[test]
fn asum_error_bounded() {
	let mut rng = Lcg(7);
	for &n in SIZES {
		let x = rng.vec(n);
		let got = asum(&x);
		let reference = comp_sum(x.iter().map(|v| v.abs()));
		let tol = f64::EPSILON * (n.max(1) as f64) * reference + f64::MIN_POSITIVE;
		assert!(
			(got - reference).abs() <= tol,
			"asum n={n}: got {got}, ref {reference}, tol {tol}"
		);
		assert!(got >= 0.0);
	}
}

#[test]
fn nrm2_error_bounded() {
	let mut rng = Lcg(8);
	for &n in SIZES {
		let x = rng.vec(n);
		let got = nrm2(&x);
		let m = x.iter().fold(0.0f64, |a, v| a.max(v.abs()));
		let reference = if m == 0.0 {
			0.0
		} else {
			m * comp_sum(x.iter().map(|v| (v / m) * (v / m))).sqrt()
		};
		let tol = f64::EPSILON * (n.max(1) as f64) * reference.max(f64::MIN_POSITIVE);
		assert!(
			(got - reference).abs() <= tol,
			"nrm2 n={n}: got {got}, ref {reference}, tol {tol}"
		);
	}
}

#[test]
fn nrm2_overflow_underflow_guards() {
	// naive sum of squares overflows: values ~1e300
	let big = vec![1e300, -1e300, 1e300];
	let got = nrm2(&big);
	let want = 1e300 * 3.0f64.sqrt();
	assert!((got - want).abs() <= 1e287, "overflow rescue: got {got}, want {want}");
	assert!(got.is_finite());

	// naive squares underflow to zero: values ~1e-300
	let tiny = vec![3e-300, 4e-300];
	let got = nrm2(&tiny);
	let want = 5e-300;
	assert!(
		(got - want).abs() <= 1e-313,
		"underflow rescue: got {got}, want {want}"
	);
	assert!(got > 0.0);

	// mixed sizes across the rescue boundary, checked against the scaled
	// reference
	let mixed = vec![1e300, 1.0, 1e-300, -2e299];
	let m = 1e300f64;
	let want = m * comp_sum(mixed.iter().map(|v| (v / m) * (v / m))).sqrt();
	let got = nrm2(&mixed);
	assert!((got - want).abs() <= 1e287, "mixed rescue: got {got}, want {want}");

	// exact zeros and empty
	assert_eq!(nrm2(&[]), 0.0);
	assert_eq!(nrm2(&[0.0, -0.0, 0.0]), 0.0);
	// infinity in, infinity out
	assert_eq!(nrm2(&[1.0, f64::INFINITY]), f64::INFINITY);
}

// ---- iamax: exact index semantics ----

#[test]
fn iamax_exact_semantics() {
	// first-occurrence tie-breaking, negatives, single element, empty
	assert_eq!(iamax(&[]), 0);
	assert_eq!(iamax(&[7.0]), 0);
	assert_eq!(iamax(&[1.0, -3.0, 3.0, 2.0]), 1, "tie: first occurrence");
	assert_eq!(iamax(&[2.0, 2.0, 2.0]), 0, "all equal");
	assert_eq!(iamax(&[0.0, 0.0]), 0, "all zero");
	assert_eq!(iamax(&[-0.5, 0.25, -0.75]), 2);

	// agreement with the plain-loop definition on random data, all sizes
	let mut rng = Lcg(9);
	for &n in SIZES {
		let x = rng.vec(n);
		let got = iamax(&x);
		let mut m = -1.0f64;
		let mut mi = 0usize;
		for (i, v) in x.iter().enumerate() {
			if v.abs() > m {
				m = v.abs();
				mi = i;
			}
		}
		assert_eq!(got, mi, "iamax n={n}");
	}
}

// ---- rotg: defining identities + reference edge cases ----

#[test]
fn rotg_identities() {
	let cases = [
		(3.0, 4.0),
		(4.0, 3.0),
		(-3.0, 4.0),
		(3.0, -4.0),
		(-3.0, -4.0),
		(5.0, 0.0),
		(0.0, 5.0),
		(0.0, -5.0),
		(1e300, 1e300),   // would overflow unguarded
		(1e-308, 1e-308), // r² would underflow unguarded (normal-range inputs)
		(1.0, 1e-200),
	];
	for (a, b) in cases {
		let g = rotg(a, b);
		let hyp = (a / g.r).hypot(b / g.r); // c² + s² via stable hypot
		assert!((hyp - 1.0).abs() < 1e-12, "({a},{b}): c²+s² = {hyp}");
		// the rotation maps (a,b) to (r,0)
		let r1 = g.c * a + g.s * b;
		let z = g.c * b - g.s * a;
		assert!(
			(r1 - g.r).abs() <= 1e-12 * g.r.abs().max(f64::MIN_POSITIVE),
			"({a},{b}): c·a+s·b = {r1}, r = {}",
			g.r
		);
		assert!(
			z.abs() <= 1e-12 * g.r.abs().max(f64::MIN_POSITIVE),
			"({a},{b}): residual {z}"
		);
		// r carries the sign of the larger-magnitude input
		let roe = if a.abs() > b.abs() { a } else { b };
		assert_eq!(g.r < 0.0, roe < 0.0, "({a},{b}): sign of r");
	}
	// subnormal inputs: reference drotg legitimately loses precision
	// (subnormals carry ~13 bits) — require only a sane, finite result
	let g = rotg(1e-320, 1e-320);
	assert!(g.r.is_finite() && g.r > 0.0);
	assert!((g.c * g.c + g.s * g.s - 1.0).abs() < 1e-3, "subnormal: c²+s² far off");

	// the zero case: identity rotation
	let g = rotg(0.0, 0.0);
	assert_eq!((g.c, g.s, g.r), (1.0, 0.0, 0.0));
	// classic 3-4-5 exactness
	let g = rotg(3.0, 4.0);
	assert!((g.r - 5.0).abs() < 1e-15 && (g.c - 0.6).abs() < 1e-15 && (g.s - 0.8).abs() < 1e-15);
}

// ---- panics on length mismatch (the safe-API contract) ----

#[test]
#[should_panic(expected = "length mismatch")]
fn axpy_length_mismatch_panics() {
	axpy(1.0, &[1.0, 2.0], &mut [1.0]);
}
