//! `asum` — sum of absolute values (ℓ¹ norm): Σ|xᵢ|.
//!
//! Implementation: reduction stream (2 accumulator registers = 4 lanes,
//! `abs` in-lane, folded in a fixed order at the end). Ported from the
//! raced variant — 3.5–4× faster than the plain loop on all three
//! runner draws (docs/blas-ab-2026-07.md, step 2): the plain loop's
//! serial dependent-add chain is latency-bound, not bandwidth-bound.
//!
//! Rounding contract: lane-parallel accumulation reorders the additions
//! — tested against a compensated-summation reference within n-scaled
//! error bounds; native ↔ wasm bit-identical by construction.

use crate::lanes::F64x2;

/// Returns Σ|xᵢ|.
pub fn asum(x: &[f64]) -> f64 {
	let len = x.len();
	let xp = x.as_ptr();
	let mut a0 = F64x2::splat(0.0);
	let mut a1 = F64x2::splat(0.0);
	let mut i = 0usize;
	unsafe {
		while i + 4 <= len {
			a0 = a0.add(F64x2::load(xp.add(i)).abs());
			a1 = a1.add(F64x2::load(xp.add(i + 2)).abs());
			i += 4;
		}
		let a = a0.add(a1);
		let mut s = a.lane0() + a.lane1();
		while i < len {
			s += (*xp.add(i)).abs();
			i += 1;
		}
		s
	}
}
