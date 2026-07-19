//! `zswap` — exchange two complex vectors.
//!
//! Implementation: pure delegation — swapping interleaved complex IS
//! `dswap` on the 2n-real view. The tuned d-stream (SIMD swap raced
//! 1.2–1.3× over plain, step 2) carries over for free.

use crate::c64::{as_re_mut, C64};
use crate::L1::dswap;

/// x ↔ y. Panics on length mismatch.
pub fn zswap(x: &mut [C64], y: &mut [C64]) {
	dswap(as_re_mut(x), as_re_mut(y));
}
