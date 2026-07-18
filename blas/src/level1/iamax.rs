//! `iamax` — index of the largest element by magnitude.
//!
//! Implementation: reduction stream — a branch-free lane-parallel `pmax`
//! pass finds the max VALUE, then one scalar rescan finds its first
//! index (the realistic SIMD strategy for an argmax). Ported from the
//! raced variant — 1.4–1.6× faster than the branching plain loop on all
//! three runner draws (docs/blas-ab-2026-07.md, step 2).
//!
//! Semantics contract (exact, tested): returns the 0-based index of the
//! first occurrence of the maximum |xᵢ| — BLAS's tie-breaking rule.
//! Returns 0 for an empty slice. Behavior on NaN input is unspecified
//! (wasm `pmax` is not NaN-propagating; reference BLAS is quirky here
//! too).

use crate::lanes::F64x2;

/// Returns the 0-based index of the first element with maximum |xᵢ|
/// (0 if `x` is empty).
pub fn iamax(x: &[f64]) -> usize {
	let len = x.len();
	let xp = x.as_ptr();
	let mut m0 = F64x2::splat(-1.0);
	let mut m1 = F64x2::splat(-1.0);
	let mut i = 0usize;
	unsafe {
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
		for k in 0..len {
			if (*xp.add(k)).abs() == best {
				return k;
			}
		}
	}
	0
}
