//! `scnrm2` — Euclidean length of a complex f32 vector: √(Σ|xᵢ|²).
//! Pure delegation: `snrm2` on the 2n-real view — the tuned stream
//! AND the over/underflow rescue carry over (see `dznrm2`).

use crate::c32::{as_re, C32};
use crate::L1::snrm2;

/// Returns √(Σ|xᵢ|²), safe against overflow and underflow.
pub fn scnrm2(x: &[C32]) -> f32 {
	snrm2(as_re(x))
}
