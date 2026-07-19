//! `zdrot` — apply a REAL plane rotation to complex vectors:
//! (xᵢ, yᵢ) ← (c·xᵢ + s·yᵢ, c·yᵢ − s·xᵢ), c and s real.
//!
//! Implementation: pure delegation — a real rotation acts on re and im
//! independently, so it IS `drot` on the 2n-real views. The tuned
//! d-stream carries over for free. (The complex-s rotation LAPACK
//! calls `zrot` has no consumer yet — explicit gap; `zrotg` already
//! produces its complex s for when one arrives.)

use crate::c64::{as_re_mut, C64};
use crate::L1::drot;

/// Apply the rotation with real cosine `c` and real sine `s`.
/// Panics on length mismatch.
pub fn zdrot(x: &mut [C64], y: &mut [C64], c: f64, s: f64) {
	drot(as_re_mut(x), as_re_mut(y), c, s);
}
