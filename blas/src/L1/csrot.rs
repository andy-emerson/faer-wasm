//! `csrot` — apply a REAL plane rotation to complex f32 vectors.
//! Pure delegation: a real rotation acts on re and im independently,
//! so it IS `srot` on the 2n-real views (see `zdrot`).

use crate::c32::{as_re_mut, C32};
use crate::L1::srot;

/// Apply the rotation with real cosine `c` and real sine `s`.
/// Panics on length mismatch.
pub fn csrot(x: &mut [C32], y: &mut [C32], c: f32, s: f32) {
	srot(as_re_mut(x), as_re_mut(y), c, s);
}
