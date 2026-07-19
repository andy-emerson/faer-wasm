//! `icamax` — index of the largest complex f32 element by |re| + |im|
//! (the BLAS complex pivoting metric — see `izamax`).
//!
//! Implementation: the idamax two-pass shape — a branch-free lane
//! pass finds the max VALUE (abs + add(swap_pairs) puts each
//! complex's |re|+|im| in both of its lanes; addition commutes
//! bit-exactly), one scalar rescan finds its first index.
//!
//! Semantics contract (exact, tested): 0-based index of the FIRST
//! occurrence of the maximum |re| + |im|; 0 for an empty slice; NaN
//! behavior unspecified.

use crate::c32::C32;
use crate::lanes::F32x4;

/// Returns the 0-based index of the first element with maximum
/// |re| + |im| (0 if `x` is empty).
pub fn icamax(x: &[C32]) -> usize {
	unsafe { imp(x.as_ptr(), x.len()) }
}

#[cfg_attr(target_arch = "wasm32", target_feature(enable = "simd128"))]
unsafe fn imp(x: *const C32, len: usize) -> usize {
	let xp = x as *const f32;
	let mut m0 = F32x4::splat(-1.0);
	let mut m1 = F32x4::splat(-1.0);
	let mut i = 0usize;
	while i + 4 <= len {
		let a0 = F32x4::load(xp.add(2 * i)).abs();
		let a1 = F32x4::load(xp.add(2 * i + 4)).abs();
		m0 = m0.pmax(a0.add(a0.swap_pairs()));
		m1 = m1.pmax(a1.add(a1.swap_pairs()));
		i += 4;
	}
	let m = m0.pmax(m1);
	let mut best = (m.lane0().max(m.lane1())).max(m.lane2().max(m.lane3()));
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
