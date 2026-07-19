//! `cher2` — Hermitian rank-2 update: A ← A + αx·yᴴ + conj(α)y·xᴴ,
//! one triangle stored.
//!
//! Implementation: column-caxpy — two Level 1 `caxpy` streams per
//! stored strict segment (scalars α·conj(y[j]) and conj(α·x[j]); the
//! conjugations land on the scalars). Diagonal update is real and
//! stored with imaginary part exactly 0 — reference `cher2`'s DBLE()
//! convention, the layer's Hermitian invariant.

use super::check_mat;
use crate::c32::C32;
use crate::L1::caxpy;

/// A ← A + αx·yᴴ + conj(α)y·xᴴ. A is Hermitian n×n at column stride
/// `cs`, `upper` (or lower) triangle stored.
#[allow(clippy::too_many_arguments)]
pub fn cher2(alpha: C32, n: usize, a: &mut [C32], cs: usize, upper: bool, x: &[C32], y: &[C32]) {
	check_mat(a.len(), n, n, cs);
	assert_eq!(x.len(), n, "cher2: x length mismatch");
	assert_eq!(y.len(), n, "cher2: y length mismatch");
	for j in 0..n {
		let cj = j * cs;
		let t1 = alpha * y[j].conj();
		let t2 = (alpha * x[j]).conj();
		let diag = (x[j] * t1).re + (y[j] * t2).re;
		if upper {
			caxpy(t1, &x[..j], &mut a[cj..cj + j]);
			caxpy(t2, &y[..j], &mut a[cj..cj + j]);
			a[cj + j] = C32::new(a[cj + j].re + diag, 0.0);
		} else {
			a[cj + j] = C32::new(a[cj + j].re + diag, 0.0);
			caxpy(t1, &x[j + 1..], &mut a[cj + j + 1..cj + n]);
			caxpy(t2, &y[j + 1..], &mut a[cj + j + 1..cj + n]);
		}
	}
}
