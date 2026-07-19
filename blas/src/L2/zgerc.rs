//! `zgerc` — conjugated rank-1 update: A ← A + αx·yᴴ.
//!
//! Implementation: column-zaxpy — identical to `zgeru` except column
//! j's scalar is α·conj(y[j]) (the conjugation lands on the scalar,
//! never on a stream).

use super::check_mat;
use crate::c64::C64;
use crate::L1::zaxpy;

/// A ← A + αx·yᴴ. A is nrows×ncols at column stride `cs`;
/// x has nrows elements, y has ncols.
pub fn zgerc(alpha: C64, nrows: usize, ncols: usize, a: &mut [C64], cs: usize, x: &[C64], y: &[C64]) {
	check_mat(a.len(), nrows, ncols, cs);
	assert_eq!(x.len(), nrows, "zgerc: x length mismatch");
	assert_eq!(y.len(), ncols, "zgerc: y length mismatch");
	for j in 0..ncols {
		zaxpy(alpha * y[j].conj(), x, &mut a[j * cs..j * cs + nrows]);
	}
}
