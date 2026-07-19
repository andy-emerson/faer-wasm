//! Level 2: matrix–vector operations, every one a composition of the
//! Level 1 streams over column slices — "column-axpy" (and its
//! dot-per-column transpose twin), or "divide-then-column-axpy" for the
//! triangular solve. Since the 2026-07-19 tuning campaign the
//! multiply-vector family runs its columns four at a time through the
//! shared blocked kernels (`crate::kernels`) — same per-element math,
//! 4× less vector read-modify-write traffic; tails and the rank-1/2
//! updates stay one Level 1 call per column. The SIMD hot loops live
//! in level1/kernels under their `target_feature` annotations.
//!
//! Matrix convention (the whole crate): column-major slice with a
//! column stride — column `j` of an `nrows × ncols` matrix `a` with
//! stride `cs ≥ nrows` is `a[j*cs .. j*cs + nrows]`. Unit row stride
//! only; callers with padded faer matrices pass `col_stride()` as `cs`.

pub mod gemv;
pub mod ger;
pub mod symv;
pub mod syr;
pub mod syr2;
pub mod trmv;
pub mod trsv;

pub use gemv::{gemv, gemv_t};
pub use ger::ger;
pub use symv::symv;
pub use syr::syr;
pub use syr2::syr2;
pub use trmv::trmv;
pub use trsv::trsv;

/// Shared entry checks: the storage really contains an nrows×ncols
/// matrix at stride cs.
#[inline]
pub(crate) fn check_mat(a_len: usize, nrows: usize, ncols: usize, cs: usize) {
	assert!(cs >= nrows, "column stride below row count");
	if ncols > 0 {
		assert!(
			a_len >= cs * (ncols - 1) + nrows,
			"matrix storage too short for its dimensions"
		);
	}
}

/// y ← βy with BLAS β=0 semantics (a hard zero-fill, so stale NaN/inf
/// in y cannot leak through 0·y).
#[inline]
pub(crate) fn scale_y(beta: f64, y: &mut [f64]) {
	if beta == 0.0 {
		y.fill(0.0);
	} else if beta != 1.0 {
		crate::level1::scal(beta, y);
	}
}
