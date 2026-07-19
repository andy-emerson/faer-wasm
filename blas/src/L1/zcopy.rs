//! `zcopy` — complex vector copy: y ← x.
//!
//! Implementation: pure delegation — copying interleaved complex IS
//! `dcopy` on the 2n-real view (bit moves have no complex structure).
//! The tuned d-stream carries over for free.

use crate::c64::{as_re, as_re_mut, C64};
use crate::L1::dcopy;

/// y ← x. Panics on length mismatch.
pub fn zcopy(x: &[C64], y: &mut [C64]) {
	dcopy(as_re(x), as_re_mut(y));
}
