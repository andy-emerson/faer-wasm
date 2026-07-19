//! `cgeru` — unconjugated rank-1 update: A ← A + αxyᵀ.
//!
//! Implementation: column-caxpy — one Level 1 `caxpy` stream per
//! column (column j gets α·y[j] times x), exactly the `dger` shape.

use super::check_mat;
use crate::c32::C32;
use crate::L1::caxpy;

/// A ← A + αxyᵀ. A is nrows×ncols at column stride `cs`;
/// x has nrows elements, y has ncols.
pub fn cgeru(alpha: C32, nrows: usize, ncols: usize, a: &mut [C32], cs: usize, x: &[C32], y: &[C32]) {
	check_mat(a.len(), nrows, ncols, cs);
	assert_eq!(x.len(), nrows, "cgeru: x length mismatch");
	assert_eq!(y.len(), ncols, "cgeru: y length mismatch");
	for j in 0..ncols {
		caxpy(alpha * y[j], x, &mut a[j * cs..j * cs + nrows]);
	}
}
