//! `crotg` — generate a complex f32 plane rotation: given (a, b),
//! find (c real, s complex, r complex) with c·a + s·b = r and
//! −conj(s)·a + c·b = 0, c² + |s|² = 1.
//!
//! Implementation: the `zrotg` shape in f32 — guarded scalar
//! arithmetic, moduli via overflow-safe `hypotf`, the norm formed
//! from scaled moduli, the phase factor α = a/|a| carrying r's sign.
//! The a = 0 special case (c=0, s=1, r=b) is reference behavior.

use crate::c32::C32;

/// A generated complex plane rotation: real cosine, complex sine, and
/// the resulting complex radius.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CGivens {
	pub c: f32,
	pub s: C32,
	pub r: C32,
}

/// Generate the rotation that maps (a, b) to (r, 0).
#[inline]
pub fn crotg(a: C32, b: C32) -> CGivens {
	let aabs = a.abs();
	if aabs == 0.0 {
		// reference crotg: rotate b straight into r
		return CGivens { c: 0.0, s: C32::ONE, r: b };
	}
	let babs = b.abs();
	let scale = aabs + babs;
	let ta = aabs / scale;
	let tb = babs / scale;
	let norm = scale * libm::sqrtf(ta * ta + tb * tb);
	// α = a/|a| — the phase of a; r inherits it (reference convention)
	let alpha = C32::new(a.re / aabs, a.im / aabs);
	let c = aabs / norm;
	let sb = alpha * b.conj();
	CGivens {
		c,
		s: C32::new(sb.re / norm, sb.im / norm),
		r: C32::new(alpha.re * norm, alpha.im * norm),
	}
}
