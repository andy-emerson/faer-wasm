//! `zher` — Hermitian rank-1 update: A ← A + αx·xᴴ, α REAL, one
//! triangle stored.
//!
//! Implementation: column-zaxpy over the stored strict segment
//! (column j's scalar is α·conj(x[j])); the diagonal update
//! re(x[j]·t) is real and is stored with imaginary part exactly 0 —
//! reference `zher`'s DBLE() convention, kept as the layer's
//! Hermitian invariant (stored diagonals stay exactly real).

use super::check_mat;
use crate::c64::C64;
use crate::L1::zaxpy;

/// A ← A + αx·xᴴ with real α. A is Hermitian n×n at column stride
/// `cs`, `upper` (or lower) triangle stored.
pub fn zher(alpha: f64, n: usize, a: &mut [C64], cs: usize, upper: bool, x: &[C64]) {
	check_mat(a.len(), n, n, cs);
	assert_eq!(x.len(), n, "zher: x length mismatch");
	for j in 0..n {
		let cj = j * cs;
		let t = x[j].conj().scale(alpha);
		if upper {
			zaxpy(t, &x[..j], &mut a[cj..cj + j]);
			a[cj + j] = C64::new(a[cj + j].re + (x[j] * t).re, 0.0);
		} else {
			a[cj + j] = C64::new(a[cj + j].re + (x[j] * t).re, 0.0);
			zaxpy(t, &x[j + 1..], &mut a[cj + j + 1..cj + n]);
		}
	}
}
