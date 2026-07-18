//! Level 3: matrix–matrix operations, every one a loop of Level 1/2
//! calls over the right-hand matrix's columns — gemm is gemv per
//! column, symm is symv per column, trmm/trsm (left side) are
//! trmv/trsv per column, the rank-k updates and right-side triangular
//! forms are truncated column-axpy sweeps. Same matrix convention as
//! Level 2 (column-major slice + column stride, unit row stride).
//!
//! Scope (gap-lined in ../../README.md): no-transpose forms only —
//! Aᵀ variants of gemm have no consumer yet (syrk covers A·Aᵀ).

pub mod gemm;
pub mod symm;
pub mod syr2k;
pub mod syrk;
pub mod trmm;
pub mod trsm;

pub use gemm::{gemm, gemm_col4, gemm_tiled};
pub use symm::{symm_left, symm_right};
pub use syr2k::syr2k;
pub use syrk::syrk;
pub use trmm::{trmm_left, trmm_right};
pub use trsm::{trsm_left, trsm_right};

pub(crate) use super::level2::check_mat;

/// Symmetric-triangle element lookup: A[i,j] with only one triangle
/// stored.
#[inline]
pub(crate) fn sym_at(a: &[f64], cs: usize, upper: bool, i: usize, j: usize) -> f64 {
	let stored = if upper { i <= j } else { i >= j };
	if stored {
		a[j * cs + i]
	} else {
		a[i * cs + j]
	}
}
