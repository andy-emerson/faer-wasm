//! `zrotg` — generate a complex plane rotation: given (a, b), find
//! (c real, s complex, r complex) with c·a + s·b = r and
//! −conj(s)·a + c·b = 0, c² + |s|² = 1.
//!
//! Implementation: no arrays = no SIMD. Guarded scalar arithmetic —
//! reference BLAS `zrotg`'s shape kept (proven numerics): moduli via
//! overflow-safe `hypot`, the norm formed from scaled moduli, the
//! phase factor α = a/|a| carrying r's complex sign. The a = 0
//! special case (c=0, s=1, r=b) is reference behavior. Consumers:
//! the complex-s rotation application (`zrot`) has no consumer yet —
//! explicit gap; `zdrot` applies the real-c,s form.

use crate::c64::C64;

/// A generated complex plane rotation: real cosine, complex sine, and
/// the resulting complex radius.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ZGivens {
	pub c: f64,
	pub s: C64,
	pub r: C64,
}

/// Generate the rotation that maps (a, b) to (r, 0).
#[inline]
pub fn zrotg(a: C64, b: C64) -> ZGivens {
	let aabs = a.abs();
	if aabs == 0.0 {
		// reference zrotg: rotate b straight into r
		return ZGivens { c: 0.0, s: C64::ONE, r: b };
	}
	let babs = b.abs();
	let scale = aabs + babs;
	let ta = aabs / scale;
	let tb = babs / scale;
	let norm = scale * libm::sqrt(ta * ta + tb * tb);
	// α = a/|a| — the phase of a; r inherits it (reference convention)
	let alpha = C64::new(a.re / aabs, a.im / aabs);
	let c = aabs / norm;
	let sb = alpha * b.conj();
	ZGivens {
		c,
		s: C64::new(sb.re / norm, sb.im / norm),
		r: C64::new(alpha.re * norm, alpha.im * norm),
	}
}
