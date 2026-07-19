//! `cswap` — exchange two complex f32 vectors. Pure delegation:
//! `sswap` on the 2n-real views.

use crate::c32::{as_re_mut, C32};
use crate::L1::sswap;

/// x ↔ y. Panics on length mismatch.
pub fn cswap(x: &mut [C32], y: &mut [C32]) {
	sswap(as_re_mut(x), as_re_mut(y));
}
