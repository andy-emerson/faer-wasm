//! `cdotc` — conjugated complex f32 dot product: xᴴy = Σ conj(xᵢ)·yᵢ.
//!
//! Implementation: reduction stream, 4 accumulator registers × two
//! packed complexes; the conjugated pair-wise product form
//! (dup_even(x)·y + neg_odd(dup_odd(x)·swap_pairs(y))) is bit-exactly
//! `conj(x)·y` in the canonical `C32` order (sign-folding is exact).
//!
//! Rounding contract: same as `cdotu` — bounds-tested vs an
//! f64-accumulated reference, native ↔ wasm bit-identical by
//! construction.

use crate::c32::C32;
use crate::lanes::F32x4;

/// Returns xᴴy (x conjugated). Panics on length mismatch.
pub fn cdotc(x: &[C32], y: &[C32]) -> C32 {
	assert_eq!(x.len(), y.len(), "cdotc: length mismatch");
	unsafe { imp(x.as_ptr(), y.as_ptr(), x.len()) }
}

#[cfg_attr(target_arch = "wasm32", target_feature(enable = "simd128"))]
unsafe fn imp(x: *const C32, y: *const C32, len: usize) -> C32 {
	let xp = x as *const f32;
	let yp = y as *const f32;
	let mut acc0 = F32x4::splat(0.0);
	let mut acc1 = F32x4::splat(0.0);
	let mut acc2 = F32x4::splat(0.0);
	let mut acc3 = F32x4::splat(0.0);
	let mut i = 0usize;
	while i + 8 <= len {
		let x0 = F32x4::load(xp.add(2 * i));
		let y0 = F32x4::load(yp.add(2 * i));
		let x1 = F32x4::load(xp.add(2 * i + 4));
		let y1 = F32x4::load(yp.add(2 * i + 4));
		let x2 = F32x4::load(xp.add(2 * i + 8));
		let y2 = F32x4::load(yp.add(2 * i + 8));
		let x3 = F32x4::load(xp.add(2 * i + 12));
		let y3 = F32x4::load(yp.add(2 * i + 12));
		acc0 = acc0.add(x0.dup_even().mul(y0).add(x0.dup_odd().mul(y0.swap_pairs()).neg_odd()));
		acc1 = acc1.add(x1.dup_even().mul(y1).add(x1.dup_odd().mul(y1.swap_pairs()).neg_odd()));
		acc2 = acc2.add(x2.dup_even().mul(y2).add(x2.dup_odd().mul(y2.swap_pairs()).neg_odd()));
		acc3 = acc3.add(x3.dup_even().mul(y3).add(x3.dup_odd().mul(y3.swap_pairs()).neg_odd()));
		i += 8;
	}
	let f = acc0.add(acc1).add(acc2.add(acc3));
	let mut s = C32::new(f.lane0() + f.lane2(), f.lane1() + f.lane3());
	while i < len {
		s = s + (*x.add(i)).conj() * *y.add(i);
		i += 1;
	}
	s
}
