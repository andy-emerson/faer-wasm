//! `zdscal` — REAL scalar × complex vector, in place: x ← αx, α ∈ ℝ.
//!
//! Implementation: pure delegation — a real scale of an interleaved
//! complex vector IS `dscal` on the 2n-real view (each component gets
//! exactly one multiply rounding, the same as reference `zdscal`'s
//! `DCMPLX(DA,0)·zx` never quite is: the full complex product would
//! add a redundant ±0·im term with signed-zero hazards). The tuned
//! d-stream carries over for free.

use crate::c64::{as_re_mut, C64};
use crate::L1::dscal;

/// x ← αx with real α.
pub fn zdscal(alpha: f64, x: &mut [C64]) {
	dscal(alpha, as_re_mut(x));
}
