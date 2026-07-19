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
	unsafe { imp(x.as_ptr(), x.len()) }
}

#[cfg_attr(target_arch = "wasm32", target_feature(enable = "simd128"))]
unsafe fn imp(xp: *const f64, len: usize) -> f64 {
	let mut a0 = F64x2::splat(0.0);
	let mut a1 = F64x2::splat(0.0);
	let mut a2 = F64x2::splat(0.0);
	let mut a3 = F64x2::splat(0.0);
	let mut i = 0usize;
	// 4 accumulator registers (tuning lever, 2026-07-19): the 2-register
	// version left the reductions latency-bound at 60-80%% of triad
	while i + 8 <= len {
		a0 = a0.add(F64x2::load(xp.add(i)).abs());
		a1 = a1.add(F64x2::load(xp.add(i + 2)).abs());
		a2 = a2.add(F64x2::load(xp.add(i + 4)).abs());
		a3 = a3.add(F64x2::load(xp.add(i + 6)).abs());
		i += 8;
	}
	let a = a0.add(a1).add(a2.add(a3));
	let mut s = a.lane0() + a.lane1();
	while i < len {
		s += (*xp.add(i)).abs();
		i += 1;
	}
	s
}
