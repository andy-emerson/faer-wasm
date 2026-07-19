//! `izamax` — index of the largest complex element by the BLAS
//! magnitude |re| + |im| (the ℓ¹ metric — reference BLAS's choice for
//! complex pivoting: monotone enough, no square roots).
//!
//! Implementation: reduction stream, the idamax two-pass shape —
//! a branch-free lane pass finds the max VALUE, one scalar rescan
//! finds its first index. Per complex: abs then add(swap) puts
//! |re| + |im| in both lanes (addition commutes bit-exactly), so
//! `pmax` accumulates the per-complex magnitudes directly. The fused
//! single-pass shape was REFUTED for idamax on both runner draws
//! (docs step 9) — not re-tried here without new evidence.
//!
//! Semantics contract (exact, tested): 0-based index of the FIRST
//! occurrence of the maximum |re| + |im|; 0 for an empty slice; NaN
//! behavior unspecified (as idamax).

use crate::c64::C64;
use crate::lanes::F64x2;

/// Returns the 0-based index of the first element with maximum
/// |re| + |im| (0 if `x` is empty).
pub fn izamax(x: &[C64]) -> usize {
	unsafe { imp(x.as_ptr(), x.len()) }
}

#[cfg_attr(target_arch = "wasm32", target_feature(enable = "simd128"))]
unsafe fn imp(x: *const C64, len: usize) -> usize {
	let xp = x as *const f64;
	let mut m0 = F64x2::splat(-1.0);
	let mut m1 = F64x2::splat(-1.0);
	let mut i = 0usize;
	while i + 2 <= len {
		let a0 = F64x2::load(xp.add(2 * i)).abs();
		let a1 = F64x2::load(xp.add(2 * i + 2)).abs();
		m0 = m0.pmax(a0.add(a0.swap()));
		m1 = m1.pmax(a1.add(a1.swap()));
		i += 2;
	}
	let m = m0.pmax(m1);
	let mut best = m.lane0().max(m.lane1());
	while i < len {
		best = best.max((*x.add(i)).abs1());
		i += 1;
	}
	for k in 0..len {
		if (*x.add(k)).abs1() == best {
			return k;
		}
	}
	0
}
