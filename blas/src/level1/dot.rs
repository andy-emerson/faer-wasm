//! `dot` — dot product: xᵀy.
//!
//! Implementation: reduction stream (2 accumulator registers = 4 lanes,
//! folded in a fixed order at the end).
//!
//! Rounding contract: lane-parallel accumulation reorders the additions
//! relative to a sequential loop — a legitimately different, equally
//! valid rounding sequence. Tested against a compensated-summation
//! reference within n-scaled error bounds; native ↔ wasm bit-identical
//! by the lane-emulation construction (see `lanes.rs`).

use crate::lanes::F64x2;

/// Returns xᵀy. Panics on length mismatch.
pub fn dot(x: &[f64], y: &[f64]) -> f64 {
	assert_eq!(x.len(), y.len(), "dot: length mismatch");
	unsafe { imp(x.as_ptr(), y.as_ptr(), x.len()) }
}

#[cfg_attr(target_arch = "wasm32", target_feature(enable = "simd128"))]
unsafe fn imp(xp: *const f64, yp: *const f64, len: usize) -> f64 {
	let mut acc0 = F64x2::splat(0.0);
	let mut acc1 = F64x2::splat(0.0);
	let mut acc2 = F64x2::splat(0.0);
	let mut acc3 = F64x2::splat(0.0);
	let mut i = 0usize;
	// 4 accumulator registers (tuning lever, 2026-07-19)
	while i + 8 <= len {
		acc0 = acc0.add(F64x2::load(xp.add(i)).mul(F64x2::load(yp.add(i))));
		acc1 = acc1.add(F64x2::load(xp.add(i + 2)).mul(F64x2::load(yp.add(i + 2))));
		acc2 = acc2.add(F64x2::load(xp.add(i + 4)).mul(F64x2::load(yp.add(i + 4))));
		acc3 = acc3.add(F64x2::load(xp.add(i + 6)).mul(F64x2::load(yp.add(i + 6))));
		i += 8;
	}
	let acc = acc0.add(acc1).add(acc2.add(acc3));
	let mut s = acc.lane0() + acc.lane1();
	while i < len {
		s += *xp.add(i) * *yp.add(i);
		i += 1;
	}
	s
}
