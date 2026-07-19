//! `syr` — symmetric rank-1 update: A ← A + αxxᵀ, one triangle stored.
//!
//! Implementation: column-axpy — one Level 1 `axpy` stream per stored
//! column segment.

use super::check_mat;
use crate::level1::axpy;

/// A ← A + αxxᵀ, A symmetric n×n at column stride `cs`, `upper` (or
/// lower) triangle stored.
pub fn syr(alpha: f64, n: usize, a: &mut [f64], cs: usize, upper: bool, x: &[f64]) {
	check_mat(a.len(), n, n, cs);
	assert_eq!(x.len(), n, "syr: x length mismatch");
	for j in 0..n {
		let t = alpha * x[j];
		if upper {
			axpy(t, &x[..=j], &mut a[j * cs..j * cs + j + 1]);
		} else {
			axpy(t, &x[j..], &mut a[j * cs + j..j * cs + n]);
		}
	}
}
