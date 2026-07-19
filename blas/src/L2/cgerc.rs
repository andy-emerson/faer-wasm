//! `cgerc` — conjugated rank-1 update: A ← A + αx·yᴴ.
//!
//! Implementation: column-caxpy — identical to `cgeru` except column
//! j's scalar is α·conj(y[j]) (the conjugation lands on the scalar,
//! never on a stream).

use super::check_mat;
use crate::c32::C32;
use crate::L1::caxpy;

/// A ← A + αx·yᴴ. A is nrows×ncols at column stride `cs`;
/// x has nrows elements, y has ncols.
pub fn cgerc(alpha: C32, nrows: usize, ncols: usize, a: &mut [C32], cs: usize, x: &[C32], y: &[C32]) {
	check_mat(a.len(), nrows, ncols, cs);
	assert_eq!(x.len(), nrows, "cgerc: x length mismatch");
	assert_eq!(y.len(), ncols, "cgerc: y length mismatch");
	for j in 0..ncols {
		caxpy(alpha * y[j].conj(), x, &mut a[j * cs..j * cs + nrows]);
	}
}
