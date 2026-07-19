//! `scasum` — Σ(|reᵢ| + |imᵢ|), the complex f32 ℓ¹-style norm
//! (component magnitudes, reference semantics — see `dzasum`).
//! Pure delegation: `sasum` on the 2n-real view.

use crate::c32::{as_re, C32};
use crate::L1::sasum;

/// Returns Σ(|reᵢ| + |imᵢ|).
pub fn scasum(x: &[C32]) -> f32 {
	sasum(as_re(x))
}
