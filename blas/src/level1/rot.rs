//! `rot` — apply a plane rotation: (xᵢ, yᵢ) ← (c·xᵢ + s·yᵢ, c·yᵢ − s·xᵢ).
//!
//! Implementation: elementwise stream (2 lanes, 2× unrolled).
//!
//! Rounding contract: each output is two multiply roundings then one
//! add/sub rounding, in that order — bit-identical to the scalar
//! definition on every target. A fused variant is a future per-op
//! measurement.

use crate::lanes::F64x2;

/// Apply the rotation with cosine `c` and sine `s` to the vector pair.
/// Panics on length mismatch.
pub fn rot(x: &mut [f64], y: &mut [f64], c: f64, s: f64) {
	assert_eq!(x.len(), y.len(), "rot: length mismatch");
	unsafe { imp(x.as_mut_ptr(), y.as_mut_ptr(), c, s, x.len()) }
}

#[cfg_attr(target_arch = "wasm32", target_feature(enable = "simd128"))]
unsafe fn imp(xp: *mut f64, yp: *mut f64, c: f64, s: f64, len: usize) {
	let vc = F64x2::splat(c);
	let vs = F64x2::splat(s);
	let mut i = 0usize;
	while i + 4 <= len {
		let x0 = F64x2::load(xp.add(i));
		let y0 = F64x2::load(yp.add(i));
		let x1 = F64x2::load(xp.add(i + 2));
		let y1 = F64x2::load(yp.add(i + 2));
		x0.mul(vc).add(y0.mul(vs)).store(xp.add(i));
		y0.mul(vc).sub(x0.mul(vs)).store(yp.add(i));
		x1.mul(vc).add(y1.mul(vs)).store(xp.add(i + 2));
		y1.mul(vc).sub(x1.mul(vs)).store(yp.add(i + 2));
		i += 4;
	}
	while i < len {
		let xi = *xp.add(i);
		let yi = *yp.add(i);
		*xp.add(i) = xi * c + yi * s;
		*yp.add(i) = yi * c - xi * s;
		i += 1;
	}
}
