//! `cher` — Hermitian rank-1 update: A ← A + αx·xᴴ, α REAL, one
//! triangle stored.
//!
//! Implementation: column-caxpy over the stored strict segment
//! (column j's scalar is α·conj(x[j])); the diagonal update
//! re(x[j]·t) is real and is stored with imaginary part exactly 0 —
//! reference `cher`'s DBLE() convention, kept as the layer's
//! Hermitian invariant (stored diagonals stay exactly real).

use super::check_mat;
use crate::c32::C32;
use crate::L1::caxpy;

/// A ← A + αx·xᴴ with real α. A is Hermitian n×n at column stride
/// `cs`, `upper` (or lower) triangle stored.
pub fn cher(alpha: f32, n: usize, a: &mut [C32], cs: usize, upper: bool, x: &[C32]) {
	check_mat(a.len(), n, n, cs);
	assert_eq!(x.len(), n, "cher: x length mismatch");
	for j in 0..n {
		let cj = j * cs;
		let t = x[j].conj().scale(alpha);
		if upper {
			caxpy(t, &x[..j], &mut a[cj..cj + j]);
			a[cj + j] = C32::new(a[cj + j].re + (x[j] * t).re, 0.0);
		} else {
			a[cj + j] = C32::new(a[cj + j].re + (x[j] * t).re, 0.0);
			caxpy(t, &x[j + 1..], &mut a[cj + j + 1..cj + n]);
		}
	}
}
