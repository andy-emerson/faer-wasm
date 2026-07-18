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
	let len = x.len();
	let xp = x.as_ptr();
	let yp = y.as_ptr();
	let mut acc0 = F64x2::splat(0.0);
	let mut acc1 = F64x2::splat(0.0);
	let mut i = 0usize;
	unsafe {
		while i + 4 <= len {
			let x0 = F64x2::load(xp.add(i));
			let y0 = F64x2::load(yp.add(i));
			let x1 = F64x2::load(xp.add(i + 2));
			let y1 = F64x2::load(yp.add(i + 2));
			acc0 = acc0.add(x0.mul(y0));
			acc1 = acc1.add(x1.mul(y1));
			i += 4;
		}
		let acc = acc0.add(acc1);
		let mut s = acc.lane0() + acc.lane1();
		while i < len {
			s += *xp.add(i) * *yp.add(i);
			i += 1;
		}
		s
	}
}
