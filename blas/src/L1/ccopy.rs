//! `ccopy` — complex f32 vector copy: y ← x. Pure delegation:
//! `scopy` on the 2n-real view.

use crate::c32::{as_re, as_re_mut, C32};
use crate::L1::scopy;

/// y ← x. Panics on length mismatch.
pub fn ccopy(x: &[C32], y: &mut [C32]) {
	scopy(as_re(x), as_re_mut(y));
}
