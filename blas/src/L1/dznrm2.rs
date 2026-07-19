//! `dznrm2` — Euclidean length of a complex vector: √(Σ|xᵢ|²).
//!
//! Implementation: pure delegation — Σ|xᵢ|² = Σ(reᵢ² + imᵢ²) IS the
//! sum of squares of the 2n-real view, so the whole routine is
//! `dnrm2` on it: the tuned 4-accumulator stream AND the LAPACK-grade
//! two-pass over/underflow rescue carry over for free.

use crate::c64::{as_re, C64};
use crate::L1::dnrm2;

/// Returns √(Σ|xᵢ|²), safe against overflow and underflow.
pub fn dznrm2(x: &[C64]) -> f64 {
	dnrm2(as_re(x))
}
