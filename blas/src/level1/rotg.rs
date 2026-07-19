//! `rotg` — generate a plane rotation: given (a, b), find (c, s, r) with
//! c·a + s·b = r and c·b − s·a = 0, c² + s² = 1.
//!
//! Implementation: no arrays = no SIMD. Guarded scalar arithmetic —
//! reference BLAS `drotg`'s anti-overflow scaling kept verbatim
//! (proven numerics; an ancestor convention that earns its place).
//! `#[inline]` so sweep loops pay no call. The classic `z`
//! reconstruction output is omitted — no consumer wants it.

/// A generated plane rotation: cosine, sine, and the resulting radius.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Givens {
	pub c: f64,
	pub s: f64,
	pub r: f64,
}

/// Generate the rotation that maps (a, b) to (r, 0).
#[inline]
pub fn rotg(a: f64, b: f64) -> Givens {
	let scale = a.abs() + b.abs();
	if scale == 0.0 {
		return Givens { c: 1.0, s: 0.0, r: 0.0 };
	}
	// reference drotg: r carries the sign of the larger-magnitude input
	let roe = if a.abs() > b.abs() { a } else { b };
	let ta = a / scale;
	let tb = b / scale;
	let mut r = scale * libm::sqrt(ta * ta + tb * tb);
	if roe < 0.0 {
		r = -r;
	}
	Givens { c: a / r, s: b / r, r }
}
