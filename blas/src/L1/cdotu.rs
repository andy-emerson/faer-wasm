//! `cdotu` — unconjugated complex f32 dot product: xᵀy = Σ xᵢ·yᵢ.
//!
//! Implementation: reduction stream, 4 accumulator registers × two
//! packed complexes; the pair-wise elementwise product form
//! (dup_even(x)·y + neg_even(dup_odd(x)·swap_pairs(y))) is
//! bit-exactly the canonical `C32` product order per complex.
//!
//! Rounding contract: register-parallel accumulation with a fixed
//! cross-pair fold — bounds-tested against an f64-accumulated
//! reference (products formed in C32, as the implementation forms
//! them); native ↔ wasm bit-identical by the lane-emulation
//! construction.

use crate::c32::C32;
use crate::lanes::F32x4;

/// Returns xᵀy (no conjugation). Panics on length mismatch.
pub fn cdotu(x: &[C32], y: &[C32]) -> C32 {
	assert_eq!(x.len(), y.len(), "cdotu: length mismatch");
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
		acc0 = acc0.add(x0.dup_even().mul(y0).add(x0.dup_odd().mul(y0.swap_pairs()).neg_even()));
		acc1 = acc1.add(x1.dup_even().mul(y1).add(x1.dup_odd().mul(y1.swap_pairs()).neg_even()));
		acc2 = acc2.add(x2.dup_even().mul(y2).add(x2.dup_odd().mul(y2.swap_pairs()).neg_even()));
		acc3 = acc3.add(x3.dup_even().mul(y3).add(x3.dup_odd().mul(y3.swap_pairs()).neg_even()));
		i += 8;
	}
	let f = acc0.add(acc1).add(acc2.add(acc3));
	let mut s = C32::new(f.lane0() + f.lane2(), f.lane1() + f.lane3());
	while i < len {
		s = s + *x.add(i) * *y.add(i);
		i += 1;
	}
	s
}
