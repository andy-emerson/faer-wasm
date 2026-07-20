//! Level 3: matrix–matrix operations, one file per BLAS routine
//! (netlib naming — src/README.md), every one structurally a loop
//! of Level 1/2 calls over the right-hand matrix's columns. Since the
//! 2026-07-19 tuning campaign every op runs its columns four at a
//! time through the shared blocked kernels (`crate::kernels`) or a
//! lockstep column walk — bit-identical to the plain composition
//! wherever the elimination order allows (the two right-side reorders
//! are documented in their files); gemm additionally size-dispatches
//! a register tile. Same matrix convention as Level 2 (column-major
//! slice + column stride, unit row stride).
//!
//! Scope (gap-lined in ../../README.md): no-transpose forms only —
//! Aᵀ variants of gemm have no consumer yet (syrk covers A·Aᵀ).

pub mod dgemm;
pub mod dsymm;
pub mod dsyr2k;
pub mod dsyrk;
pub mod dtrmm;
pub mod dtrsm;
pub mod sgemm;
pub mod ssymm;
pub mod ssyr2k;
pub mod ssyrk;
pub mod strmm;
pub mod strsm;
pub mod zgemm;
pub mod zhemm;
pub mod zher2k;
pub mod zherk;
pub mod ztrmm;
pub mod ztrsm;
pub mod cgemm;
pub mod chemm;
pub mod cher2k;
pub mod cherk;
pub mod ctrmm;
pub mod ctrsm;

pub use dgemm::{dgemm, dgemm_col4, dgemm_colaxpy, dgemm_packed, dgemm_tiled};
pub use dsymm::{dsymm_left, dsymm_right};
pub use dsyr2k::dsyr2k;
pub use dsyrk::dsyrk;
pub use dtrmm::{dtrmm_left, dtrmm_right};
pub use dtrsm::{dtrsm_left, dtrsm_right};
pub use sgemm::{sgemm, sgemm_col4, sgemm_colaxpy, sgemm_packed, sgemm_tiled};
pub use ssymm::{ssymm_left, ssymm_right};
pub use ssyr2k::ssyr2k;
pub use ssyrk::ssyrk;
pub use strmm::{strmm_left, strmm_right};
pub use strsm::{strsm_left, strsm_right};
pub use zgemm::{zgemm, zgemm_colaxpy, zgemm_packed};
pub use zhemm::{zhemm_left, zhemm_right};
pub use zher2k::zher2k;
pub use zherk::zherk;
pub use ztrmm::{ztrmm_left, ztrmm_right};
pub use ztrsm::{ztrsm_left, ztrsm_right};
pub use cgemm::{cgemm, cgemm_colaxpy, cgemm_packed};
pub use chemm::{chemm_left, chemm_right};
pub use cher2k::cher2k;
pub use cherk::cherk;
pub use ctrmm::{ctrmm_left, ctrmm_right};
pub use ctrsm::{ctrsm_left, ctrsm_right};

pub(crate) use super::L2::check_mat;

/// Symmetric-triangle element lookup: A[i,j] with only one triangle
/// stored (f64).
#[inline]
pub(crate) fn dsym_at(a: &[f64], cs: usize, upper: bool, i: usize, j: usize) -> f64 {
	let stored = if upper { i <= j } else { i >= j };
	if stored {
		a[j * cs + i]
	} else {
		a[i * cs + j]
	}
}

/// f32 twin of `dsym_at`.
#[inline]
pub(crate) fn ssym_at(a: &[f32], cs: usize, upper: bool, i: usize, j: usize) -> f32 {
	let stored = if upper { i <= j } else { i >= j };
	if stored {
		a[j * cs + i]
	} else {
		a[i * cs + j]
	}
}

/// Hermitian-triangle element lookup: A[i,j] with only one triangle
/// stored — conjugated on the reflected side, and REAL on the
/// diagonal (stored diagonal imaginary parts are ignored, per the
/// LAPACK Hermitian storage convention).
#[inline]
pub(crate) fn zher_at(
	a: &[crate::c64::C64],
	cs: usize,
	upper: bool,
	i: usize,
	j: usize,
) -> crate::c64::C64 {
	use crate::c64::C64;
	if i == j {
		return C64::new(a[j * cs + j].re, 0.0);
	}
	let stored = if upper { i < j } else { i > j };
	if stored {
		a[j * cs + i]
	} else {
		a[i * cs + j].conj()
	}
}

/// c32 twin of `zher_at`.
#[inline]
pub(crate) fn cher_at(
	a: &[crate::c32::C32],
	cs: usize,
	upper: bool,
	i: usize,
	j: usize,
) -> crate::c32::C32 {
	use crate::c32::C32;
	if i == j {
		return C32::new(a[j * cs + j].re, 0.0);
	}
	let stored = if upper { i < j } else { i > j };
	if stored {
		a[j * cs + i]
	} else {
		a[i * cs + j].conj()
	}
}
