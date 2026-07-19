//! `dzasum` — Σ(|reᵢ| + |imᵢ|), the complex BLAS ℓ¹-style norm
//! (reference BLAS sums COMPONENT magnitudes, not moduli — kept: it's
//! the interop contract and it's cheaper).
//!
//! Implementation: pure delegation — the component sum IS `dasum` on
//! the 2n-real view; the tuned 4-accumulator stream carries over for
//! free. Reduction order therefore interleaves re/im pairs exactly as
//! the d-stream folds its lanes — bounds-tested, deterministic by the
//! same construction.

use crate::c64::{as_re, C64};
use crate::L1::dasum;

/// Returns Σ(|reᵢ| + |imᵢ|).
pub fn dzasum(x: &[C64]) -> f64 {
	dasum(as_re(x))
}
