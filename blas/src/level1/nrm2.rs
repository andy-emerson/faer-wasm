//! `nrm2` — Euclidean length (ℓ² norm): √(Σxᵢ²).
//!
//! Implementation: reduction stream (2 accumulator registers = 4 lanes),
//! with a two-pass rescue for over/underflow — the wasm reshaping of
//! LAPACK's scaled-sum-of-squares guard. Fast path: one streaming
//! sum-of-squares pass; if the sum is non-finite (overflow) or small
//! enough that squares may have lost bits to subnormals (underflow),
//! rescale by the largest magnitude and stream again. The guard
//! semantics are LAPACK-proven; the shape (branch once between two
//! streaming passes, instead of a per-element branchy update) is ours.
//!
//! Rounding contract: lane-parallel accumulation — tested against a
//! scaled compensated-summation reference within n-scaled error bounds;
//! native ↔ wasm bit-identical by construction. Overflow/underflow
//! inputs (‖x‖ near 1e±300) are tested explicitly.

use crate::lanes::F64x2;

// Below this, individual squares may have underflowed to subnormals and
// lost precision: MIN_POSITIVE / EPSILON ≈ 1.002e-292.
const RESCUE_FLOOR: f64 = f64::MIN_POSITIVE / f64::EPSILON;

/// Returns √(Σxᵢ²), safe against overflow and underflow.
pub fn nrm2(x: &[f64]) -> f64 {
	if x.is_empty() {
		return 0.0;
	}
	let ss = unsafe { sumsq(x.as_ptr(), x.len()) };
	if ss.is_finite() && ss > RESCUE_FLOOR {
		return libm::sqrt(ss);
	}
	// rescue: scale by the largest magnitude, then stream again
	let m = unsafe { maxabs(x.as_ptr(), x.len()) };
	if m == 0.0 {
		// all zeros → ss = 0.0; NaNs present (dropped by pmax) → ss = NaN
		return ss;
	}
	if m.is_infinite() {
		return m;
	}
	m * libm::sqrt(unsafe { sumsq_scaled(x.as_ptr(), x.len(), m) })
}

#[cfg_attr(target_arch = "wasm32", target_feature(enable = "simd128"))]
unsafe fn sumsq(xp: *const f64, len: usize) -> f64 {
	let mut acc0 = F64x2::splat(0.0);
	let mut acc1 = F64x2::splat(0.0);
	let mut acc2 = F64x2::splat(0.0);
	let mut acc3 = F64x2::splat(0.0);
	let mut i = 0usize;
	// 4 accumulator registers (tuning lever, 2026-07-19)
	while i + 8 <= len {
		let x0 = F64x2::load(xp.add(i));
		let x1 = F64x2::load(xp.add(i + 2));
		let x2 = F64x2::load(xp.add(i + 4));
		let x3 = F64x2::load(xp.add(i + 6));
		acc0 = acc0.add(x0.mul(x0));
		acc1 = acc1.add(x1.mul(x1));
		acc2 = acc2.add(x2.mul(x2));
		acc3 = acc3.add(x3.mul(x3));
		i += 8;
	}
	let acc = acc0.add(acc1).add(acc2.add(acc3));
	let mut s = acc.lane0() + acc.lane1();
	while i < len {
		let v = *xp.add(i);
		s += v * v;
		i += 1;
	}
	s
}

/// Σ(xᵢ/m)² — the rescued pass.
#[cfg_attr(target_arch = "wasm32", target_feature(enable = "simd128"))]
unsafe fn sumsq_scaled(xp: *const f64, len: usize, m: f64) -> f64 {
	let vm = F64x2::splat(m);
	let mut acc0 = F64x2::splat(0.0);
	let mut acc1 = F64x2::splat(0.0);
	let mut i = 0usize;
	while i + 4 <= len {
		let x0 = F64x2::load(xp.add(i)).div(vm);
		let x1 = F64x2::load(xp.add(i + 2)).div(vm);
		acc0 = acc0.add(x0.mul(x0));
		acc1 = acc1.add(x1.mul(x1));
		i += 4;
	}
	let acc = acc0.add(acc1);
	let mut s = acc.lane0() + acc.lane1();
	while i < len {
		let v = *xp.add(i) / m;
		s += v * v;
		i += 1;
	}
	s
}

/// The branch-free max-|xᵢ| value pass (iamax's vector pass).
#[cfg_attr(target_arch = "wasm32", target_feature(enable = "simd128"))]
unsafe fn maxabs(xp: *const f64, len: usize) -> f64 {
	let mut m0 = F64x2::splat(-1.0);
	let mut m1 = F64x2::splat(-1.0);
	let mut i = 0usize;
	while i + 4 <= len {
		m0 = m0.pmax(F64x2::load(xp.add(i)).abs());
		m1 = m1.pmax(F64x2::load(xp.add(i + 2)).abs());
		i += 4;
	}
	let m = m0.pmax(m1);
	let mut best = m.lane0().max(m.lane1());
	while i < len {
		best = best.max((*xp.add(i)).abs());
		i += 1;
	}
	// len >= 1 guaranteed by caller; splat(-1) never survives a nonempty
	// pass because |x| >= 0
	best.max(0.0)
}
