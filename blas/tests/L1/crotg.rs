use super::common::*;
use faer_wasm_blas::L1::crotg;
use faer_wasm_blas::L1::crotg::CGivens;
use faer_wasm_blas::C32;

fn check_identities(a: C32, b: C32, g: CGivens, ctx: &str) {
	// c² + |s|² = 1
	let unit = g.c * g.c + (g.s.re * g.s.re + g.s.im * g.s.im);
	assert!((unit - 1.0).abs() <= 32.0 * f32::EPSILON, "{ctx}: c²+|s|² = {unit}");
	// c·a + s·b = r
	let r = C32::new(g.c * a.re, g.c * a.im) + g.s * b;
	let rscale = r.abs1().max(g.r.abs1()).max(1e-30);
	assert!(
		(r.re - g.r.re).abs() + (r.im - g.r.im).abs() <= 64.0 * f32::EPSILON * rscale,
		"{ctx}: c·a+s·b ≠ r"
	);
	// −conj(s)·a + c·b = 0
	let z = -(g.s.conj() * a) + C32::new(g.c * b.re, g.c * b.im);
	let cscale = a.abs1().max(b.abs1()).max(1e-30);
	assert!(z.abs1() <= 64.0 * f32::EPSILON * cscale, "{ctx}: elimination residual {z:?}");
}

#[test]
fn crotg_identities() {
	let mut rng = Lcg(113);
	for _ in 0..200 {
		let a = rng.next_c32();
		let b = rng.next_c32();
		if a.abs1() == 0.0 {
			continue;
		}
		check_identities(a, b, crotg::crotg(a, b), "random");
	}
}

#[test]
fn crotg_zero_a_reference_case() {
	// reference crotg: a = 0 → c=0, s=1, r=b (exactly)
	let b = C32::new(-1.25, 0.5);
	let g = crotg::crotg(C32::ZERO, b);
	assert_eq!(g.c, 0.0);
	assert!(bits_eq_cc(g.s, C32::ONE));
	assert!(bits_eq_cc(g.r, b));
}

#[test]
fn crotg_extreme_magnitudes() {
	// the scaled-norm guard must survive magnitudes whose squares
	// would overflow/underflow
	for (a, b) in [
		(C32::new(3e15, 4e15), C32::new(-1e15, 2e15)),
		(C32::new(3e-15, 4e-15), C32::new(1e-15, -2e-15)),
		(C32::new(1e30, 0.0), C32::new(0.0, 1e30)),
		(C32::new(5e-30, 0.0), C32::new(1e-30, 1e-30)),
	] {
		let g = crotg::crotg(a, b);
		assert!(g.c.is_finite() && g.r.re.is_finite() && g.r.im.is_finite());
		check_identities(a, b, g, "extreme");
	}
}
