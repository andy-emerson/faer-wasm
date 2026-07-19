//! `csscal` — REAL scalar × complex f32 vector, in place: x ← αx,
//! α ∈ ℝ. Pure delegation: `sscal` on the 2n-real view (one multiply
//! rounding per component — see `zdscal` for the rationale).

use crate::c32::{as_re_mut, C32};
use crate::L1::sscal;

/// x ← αx with real α.
pub fn csscal(alpha: f32, x: &mut [C32]) {
	sscal(alpha, as_re_mut(x));
}
