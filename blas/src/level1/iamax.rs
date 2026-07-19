//! `iamax` — index of the largest element by magnitude.
//!
//! Implementation: fused single-pass reduction stream (tuned
//! 2026-07-19) — each lane track carries its running max VALUE and the
//! f64-encoded INDEX of that value's first occurrence, updated
//! branch-free per step (`gt` mask + bitselect; indices below 2⁵³ are
//! exact in f64). One pass over the data replaces the old two-pass
//! shape (pmax value scan + scalar first-index rescan, itself 1.4–1.6×
//! over the branching plain loop on all three step-2 runner draws).
//! The cross-track fold takes the lower index on value ties, keeping
//! the first-occurrence contract exactly.
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
	unsafe { imp(x.as_ptr(), x.len()) }
}

#[cfg_attr(target_arch = "wasm32", target_feature(enable = "simd128"))]
unsafe fn imp(xp: *const f64, len: usize) -> usize {
	let mut best = -1.0f64;
	let mut bi = 0usize;
	let mut i = 0usize;
	if len >= 4 {
		let mut m0 = F64x2::splat(-1.0);
		let mut m1 = F64x2::splat(-1.0);
		let mut i0 = F64x2::splat(0.0);
		let mut i1 = F64x2::splat(0.0);
		let mut c0 = F64x2::pair(0.0, 1.0);
		let mut c1 = F64x2::pair(2.0, 3.0);
		let four = F64x2::splat(4.0);
		while i + 4 <= len {
			let a0 = F64x2::load(xp.add(i)).abs();
			let a1 = F64x2::load(xp.add(i + 2)).abs();
			(m0, i0) = F64x2::argmax_step(m0, i0, a0, c0);
			(m1, i1) = F64x2::argmax_step(m1, i1, a1, c1);
			c0 = c0.add(four);
			c1 = c1.add(four);
			i += 4;
		}
		// cross-track fold: lower index wins value ties, preserving
		// the first-occurrence contract (each track already keeps its
		// own first occurrence via the strict `>` step)
		for (v, ix) in [
			(m0.lane0(), i0.lane0()),
			(m0.lane1(), i0.lane1()),
			(m1.lane0(), i1.lane0()),
			(m1.lane1(), i1.lane1()),
		] {
			let ix = ix as usize;
			if v > best || (v == best && ix < bi) {
				best = v;
				bi = ix;
			}
		}
	}
	// tail indices are all larger than any lane index, so strict `>`
	// keeps first occurrence here too
	while i < len {
		let v = (*xp.add(i)).abs();
		if v > best {
			best = v;
			bi = i;
		}
		i += 1;
	}
	bi
}
