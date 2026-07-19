//! `copy` — vector copy: y ← x.
//!
//! Implementation: elementwise stream (2 lanes, 2× unrolled) — an
//! architect consistency decision (Andy, 2026-07-18) on top of a
//! measured no-harm: copy runs at the machine's bandwidth ceiling
//! either way, so no speed claim attaches to the loop over memcpy.
//!
//! Rounding contract: none — bytes move unchanged, bit-for-bit.

use crate::lanes::F64x2;

/// y ← x. Panics on length mismatch.
pub fn copy(x: &[f64], y: &mut [f64]) {
	assert_eq!(x.len(), y.len(), "copy: length mismatch");
	unsafe { imp(x.as_ptr(), y.as_mut_ptr(), x.len()) }
}

#[cfg_attr(target_arch = "wasm32", target_feature(enable = "simd128"))]
unsafe fn imp(xp: *const f64, yp: *mut f64, len: usize) {
	let mut i = 0usize;
	while i + 4 <= len {
		let x0 = F64x2::load(xp.add(i));
		let x1 = F64x2::load(xp.add(i + 2));
		x0.store(yp.add(i));
		x1.store(yp.add(i + 2));
		i += 4;
	}
	while i < len {
		*yp.add(i) = *xp.add(i);
		i += 1;
	}
}
