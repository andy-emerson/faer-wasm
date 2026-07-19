//! `axpy` — scaled vector addition: y ← αx + y.
//!
//! Implementation: elementwise stream (2 lanes, 2× unrolled).
//!
//! Rounding contract: each element is `y[i] + α·x[i]` — one multiply
//! rounding, one add rounding, bit-identical to the scalar definition
//! on every target. An FMA variant (single rounding) is a future
//! per-op measurement, not built yet.

use crate::lanes::F64x2;

/// y ← αx + y. Panics on length mismatch.
pub fn axpy(alpha: f64, x: &[f64], y: &mut [f64]) {
	assert_eq!(x.len(), y.len(), "axpy: length mismatch");
	unsafe { imp(alpha, x.as_ptr(), y.as_mut_ptr(), x.len()) }
}

#[cfg_attr(target_arch = "wasm32", target_feature(enable = "simd128"))]
unsafe fn imp(alpha: f64, xp: *const f64, yp: *mut f64, len: usize) {
	let va = F64x2::splat(alpha);
	let mut i = 0usize;
	while i + 4 <= len {
		let x0 = F64x2::load(xp.add(i));
		let y0 = F64x2::load(yp.add(i));
		let x1 = F64x2::load(xp.add(i + 2));
		let y1 = F64x2::load(yp.add(i + 2));
		y0.add(x0.mul(va)).store(yp.add(i));
		y1.add(x1.mul(va)).store(yp.add(i + 2));
		i += 4;
	}
	while i < len {
		*yp.add(i) += *xp.add(i) * alpha;
		i += 1;
	}
}
