//! `syr2` — symmetric rank-2 update: A ← A + α(xyᵀ + yxᵀ), one
//! triangle stored.
//!
//! Implementation: column-axpy — two Level 1 `axpy` streams per stored
//! column segment (the second rides the cache the first just warmed).

use super::check_mat;
use crate::level1::axpy;

/// A ← A + α(xyᵀ + yxᵀ), A symmetric n×n at column stride `cs`,
/// `upper` (or lower) triangle stored.
pub fn syr2(alpha: f64, n: usize, a: &mut [f64], cs: usize, upper: bool, x: &[f64], y: &[f64]) {
	check_mat(a.len(), n, n, cs);
	assert_eq!(x.len(), n, "syr2: x length mismatch");
	assert_eq!(y.len(), n, "syr2: y length mismatch");
	for j in 0..n {
		let (ty, tx) = (alpha * y[j], alpha * x[j]);
		if upper {
			let col = &mut a[j * cs..j * cs + j + 1];
			axpy(ty, &x[..=j], col);
			axpy(tx, &y[..=j], col);
		} else {
			let col = &mut a[j * cs + j..j * cs + n];
			axpy(ty, &x[j..], col);
			axpy(tx, &y[j..], col);
		}
	}
}
