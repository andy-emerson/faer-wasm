use super::common::*;
use faer_wasm_blas::L1::zrotg;
use faer_wasm_blas::L1::zrotg::ZGivens;
use faer_wasm_blas::C64;

fn check_identities(a: C64, b: C64, g: ZGivens, ctx: &str) {
	// c² + |s|² = 1
	let unit = g.c * g.c + (g.s.re * g.s.re + g.s.im * g.s.im);
	assert!((unit - 1.0).abs() <= 32.0 * f64::EPSILON, "{ctx}: c²+|s|² = {unit}");
	// c·a + s·b = r
	let r = C64::new(g.c * a.re, g.c * a.im) + g.s * b;
	let rscale = r.abs1().max(g.r.abs1()).max(1e-300);
	assert!(
		(r.re - g.r.re).abs() + (r.im - g.r.im).abs() <= 64.0 * f64::EPSILON * rscale,
		"{ctx}: c·a+s·b ≠ r"
	);
	// −conj(s)·a + c·b = 0
	let z = -(g.s.conj() * a) + C64::new(g.c * b.re, g.c * b.im);
	let zscale = a.abs1().max(b.abs1()).max(1e-300);
	assert!(z.abs1() <= 64.0 * f64::EPSILON * zscale, "{ctx}: elimination residual {z:?}");
}

#[test]
fn zrotg_identities() {
	let mut rng = Lcg(113);
	for _ in 0..200 {
		let a = rng.next_c64();
		let b = rng.next_c64();
		if a.abs1() == 0.0 {
			continue;
		}
		check_identities(a, b, zrotg::zrotg(a, b), "random");
	}
}

#[test]
fn zrotg_zero_a_reference_case() {
	// reference zrotg: a = 0 → c=0, s=1, r=b (exactly)
	let b = C64::new(-1.25, 0.5);
	let g = zrotg::zrotg(C64::ZERO, b);
	assert_eq!(g.c, 0.0);
	assert!(bits_eq_c(g.s, C64::ONE));
	assert!(bits_eq_c(g.r, b));
}

#[test]
fn zrotg_extreme_magnitudes() {
	// the scaled-norm guard must survive magnitudes whose squares
	// would overflow/underflow
	for (a, b) in [
		(C64::new(3e150, 4e150), C64::new(-1e150, 2e150)),
		(C64::new(3e-150, 4e-150), C64::new(1e-150, -2e-150)),
		(C64::new(1e300, 0.0), C64::new(0.0, 1e300)),
		(C64::new(5e-300, 0.0), C64::new(1e-300, 1e-300)),
	] {
		let g = zrotg::zrotg(a, b);
		assert!(g.c.is_finite() && g.r.re.is_finite() && g.r.im.is_finite());
		check_identities(a, b, g, "extreme");
	}
}
