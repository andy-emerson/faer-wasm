//! `scal` — scalar × vector: x ← αx.
//!
//! Implementation: elementwise stream (2 lanes, 2× unrolled).
//!
//! Rounding contract: one multiply rounding per element — bit-identical
//! to the scalar definition on every target.

use crate::lanes::F64x2;

/// x ← αx.
pub fn scal(alpha: f64, x: &mut [f64]) {
	let len = x.len();
	let xp = x.as_mut_ptr();
	let va = F64x2::splat(alpha);
	let mut i = 0usize;
	unsafe {
		while i + 4 <= len {
			let x0 = F64x2::load(xp.add(i));
			let x1 = F64x2::load(xp.add(i + 2));
			x0.mul(va).store(xp.add(i));
			x1.mul(va).store(xp.add(i + 2));
			i += 4;
		}
		while i < len {
			*xp.add(i) *= alpha;
			i += 1;
		}
	}
}
