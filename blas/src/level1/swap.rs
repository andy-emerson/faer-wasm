//! `swap` — exchange two vectors: x ↔ y.
//!
//! Implementation: elementwise stream (2 lanes, 2× unrolled). Ported
//! from the raced variant — SIMD beat the auto-vectorized plain loop
//! 1.15–1.33× on all three runner draws (docs/blas-ab-2026-07.md,
//! step 2).
//!
//! Rounding contract: none — bytes move unchanged, bit-for-bit.

use crate::lanes::F64x2;

/// x ↔ y. Panics on length mismatch.
pub fn swap(x: &mut [f64], y: &mut [f64]) {
	assert_eq!(x.len(), y.len(), "swap: length mismatch");
	unsafe { imp(x.as_mut_ptr(), y.as_mut_ptr(), x.len()) }
}

#[cfg_attr(target_arch = "wasm32", target_feature(enable = "simd128"))]
unsafe fn imp(xp: *mut f64, yp: *mut f64, len: usize) {
	let mut i = 0usize;
	while i + 4 <= len {
		let x0 = F64x2::load(xp.add(i));
		let y0 = F64x2::load(yp.add(i));
		let x1 = F64x2::load(xp.add(i + 2));
		let y1 = F64x2::load(yp.add(i + 2));
		y0.store(xp.add(i));
		x0.store(yp.add(i));
		y1.store(xp.add(i + 2));
		x1.store(yp.add(i + 2));
		i += 4;
	}
	while i < len {
		let t = *xp.add(i);
		*xp.add(i) = *yp.add(i);
		*yp.add(i) = t;
		i += 1;
	}
}
