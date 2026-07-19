//! `zher2` — Hermitian rank-2 update: A ← A + αx·yᴴ + conj(α)y·xᴴ,
//! one triangle stored.
//!
//! Implementation: column-zaxpy — two Level 1 `zaxpy` streams per
//! stored strict segment (scalars α·conj(y[j]) and conj(α·x[j]); the
//! conjugations land on the scalars). Diagonal update is real and
//! stored with imaginary part exactly 0 — reference `zher2`'s DBLE()
//! convention, the layer's Hermitian invariant.

use super::check_mat;
use crate::c64::C64;
use crate::L1::zaxpy;

/// A ← A + αx·yᴴ + conj(α)y·xᴴ. A is Hermitian n×n at column stride
/// `cs`, `upper` (or lower) triangle stored.
#[allow(clippy::too_many_arguments)]
pub fn zher2(alpha: C64, n: usize, a: &mut [C64], cs: usize, upper: bool, x: &[C64], y: &[C64]) {
	check_mat(a.len(), n, n, cs);
	assert_eq!(x.len(), n, "zher2: x length mismatch");
	assert_eq!(y.len(), n, "zher2: y length mismatch");
	for j in 0..n {
		let cj = j * cs;
		let t1 = alpha * y[j].conj();
		let t2 = (alpha * x[j]).conj();
		let diag = (x[j] * t1).re + (y[j] * t2).re;
		if upper {
			zaxpy(t1, &x[..j], &mut a[cj..cj + j]);
			zaxpy(t2, &y[..j], &mut a[cj..cj + j]);
			a[cj + j] = C64::new(a[cj + j].re + diag, 0.0);
		} else {
			a[cj + j] = C64::new(a[cj + j].re + diag, 0.0);
			zaxpy(t1, &x[j + 1..], &mut a[cj + j + 1..cj + n]);
			zaxpy(t2, &y[j + 1..], &mut a[cj + j + 1..cj + n]);
		}
	}
}
