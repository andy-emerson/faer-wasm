//! `zgeru` — unconjugated rank-1 update: A ← A + αxyᵀ.
//!
//! Implementation: column-zaxpy — one Level 1 `zaxpy` stream per
//! column (column j gets α·y[j] times x), exactly the `dger` shape.

use super::check_mat;
use crate::c64::C64;
use crate::L1::zaxpy;

/// A ← A + αxyᵀ. A is nrows×ncols at column stride `cs`;
/// x has nrows elements, y has ncols.
pub fn zgeru(alpha: C64, nrows: usize, ncols: usize, a: &mut [C64], cs: usize, x: &[C64], y: &[C64]) {
	check_mat(a.len(), nrows, ncols, cs);
	assert_eq!(x.len(), nrows, "zgeru: x length mismatch");
	assert_eq!(y.len(), ncols, "zgeru: y length mismatch");
	for j in 0..ncols {
		zaxpy(alpha * y[j], x, &mut a[j * cs..j * cs + nrows]);
	}
}
