//! Complex Schur decomposition `A = Z T Zᴴ` for `c64`.
//!
//! `T` is upper triangular with the eigenvalues on the diagonal. Same driver
//! shape as [`crate::real`], with the simpler 1×1-blocks-only reordering.

use crate::{ReorderError, SchurError};
use faer::dyn_stack::{MemBuffer, MemStack, StackReq};
use faer::linalg::evd::schur::{self, complex_schur};
use faer::linalg::evd::hessenberg;
use faer::linalg::householder;
use faer::linalg::qr::no_pivoting::factor::recommended_block_size;
use faer::linalg::{temp_mat_scratch, temp_mat_zeroed};
use faer::mat::AsMatMut;
use faer::prelude::*;
use faer::{Auto, Conj};

pub use faer::linalg::evd::schur::SchurParams;

/// `SchurParams` tuned for the compilation target. The 2026-07-09 "blocked
/// multishift/AED loses to `lahqr` by 2–13× on wasm" measurement (which
/// pinned this to `usize::MAX`) was the no_std AED-window bug for n ≥ 150,
/// fixed by carried `patches/faer-rs/0004`; post-fix the crossover grid
/// (run 29134291933) has `lahqr` winning through n = 448 (11.8× at 96 down
/// to 1.16× at 448) and multishift winning from n = 512 (1.07×) — the same
/// crossover as the real pipeline. 480 routes every measured winner
/// correctly; re-sweep (`bench/evd-tune.mjs`) before relying on it beyond
/// n = 512. On other targets this is faer's `Auto` default, unchanged.
pub fn recommended_params() -> SchurParams {
	#[allow(unused_mut)]
	let mut params: SchurParams = Auto::<c64>::auto();
	#[cfg(target_arch = "wasm32")]
	{
		params.blocking_threshold = 480;
	}
	params
}

/// scratch space required by [`complex_schur_in_place`]
pub fn complex_schur_scratch(n: usize, par: Par, params: SchurParams) -> StackReq {
	if n <= 1 {
		return StackReq::EMPTY;
	}
	let bs = recommended_block_size::<c64>(n - 1, n - 1);
	StackReq::all_of(&[
		temp_mat_scratch::<c64>(bs, n - 1),
		StackReq::any_of(&[
			hessenberg::hessenberg_in_place_scratch::<c64>(n, bs, par, Default::default()),
			householder::apply_block_householder_sequence_on_the_right_in_place_scratch::<c64>(
				n - 1,
				bs,
				n - 1,
			),
			schur::multishift_qr_scratch::<c64>(n, n, true, true, par, params),
		]),
	])
}

/// Computes the complex Schur form in place.
///
/// On entry `h` is a general square matrix `A`; on exit it is the upper
/// triangular `T`. If `z` is provided it must be `n×n` and is overwritten
/// with the unitary `Z` such that `A = Z T Zᴴ`. Eigenvalues land in `w`.
pub fn complex_schur_in_place(
	mut h: MatMut<'_, c64>,
	mut z: Option<MatMut<'_, c64>>,
	mut w: ColMut<'_, c64>,
	par: Par,
	stack: &mut MemStack,
	params: SchurParams,
) -> Result<(), SchurError> {
	let n = h.nrows();
	assert!(h.ncols() == n);
	assert!(w.nrows() == n);
	if let Some(z) = z.rb() {
		assert!(z.nrows() == n);
		assert!(z.ncols() == n);
	}
	for j in 0..n {
		for i in 0..n {
			let v = h[(i, j)];
			if !(v.re.is_finite() && v.im.is_finite()) {
				return Err(SchurError::NonFinite);
			}
		}
	}
	if let Some(z) = z.rb_mut() {
		let mut z = z;
		z.fill(c64::new(0.0, 0.0));
		z.diagonal_mut().fill(c64::new(1.0, 0.0));
	}
	if n == 0 {
		return Ok(());
	}
	if n == 1 {
		w[0] = h[(0, 0)];
		return Ok(());
	}

	let bs = recommended_block_size::<c64>(n - 1, n - 1);
	{
		let (mut hh, stack) = temp_mat_zeroed::<c64, _, _>(bs, n - 1, &mut *stack);
		let mut hh = hh.as_mat_mut();
		hessenberg::hessenberg_in_place(
			h.rb_mut(),
			hh.rb_mut(),
			par,
			stack,
			Default::default(),
		);
		if let Some(mut z) = z.rb_mut() {
			householder::apply_block_householder_sequence_on_the_right_in_place_with_conj(
				h.rb().submatrix(1, 0, n - 1, n - 1),
				hh.rb(),
				Conj::No,
				z.rb_mut().submatrix_mut(1, 1, n - 1, n - 1),
				par,
				stack,
			);
		}
	}
	for j in 0..n {
		for i in j + 2..n {
			h[(i, j)] = c64::new(0.0, 0.0);
		}
	}

	let (info, _, _) = complex_schur::multishift_qr::<c64>(
		true,
		h.rb_mut(),
		z.rb_mut(),
		w.rb_mut(),
		0,
		n,
		par,
		stack,
		params,
	);
	if info != 0 {
		return Err(SchurError::NoConvergence);
	}
	// faer's blocked path uses the region below the subdiagonal as workspace
	// and does not clean it up; once converged every subdiagonal entry is
	// deflated, so T is upper triangular — zero the full strict lower part
	for j in 0..n {
		for i in j + 1..n {
			h[(i, j)] = c64::new(0.0, 0.0);
		}
	}
	Ok(())
}

/// result of [`complex_schur`]: `a = z * t * z.adjoint()`
pub struct ComplexSchur {
	/// upper triangular Schur form
	pub t: Mat<c64>,
	/// unitary Schur vectors
	pub z: Mat<c64>,
	/// eigenvalues
	pub w: Col<c64>,
}

/// Allocating convenience wrapper around [`complex_schur_in_place`].
pub fn complex_schur(a: MatRef<'_, c64>, par: Par) -> Result<ComplexSchur, SchurError> {
	let n = a.nrows();
	assert!(a.ncols() == n);
	let params = recommended_params();
	let mut t = a.to_owned();
	let mut z = Mat::zeros(n, n);
	let mut w = Col::zeros(n);
	let mut buf = MemBuffer::new(complex_schur_scratch(n, par, params));
	let stack = MemStack::new(&mut buf);
	complex_schur_in_place(t.as_mut(), Some(z.as_mut()), w.as_mut(), par, stack, params)?;
	Ok(ComplexSchur { t, z, w })
}

/// Moves the eigenvalue at row `ifst` to row `ilst` (`ztrexc`-equivalent),
/// updating `t` and, if given, the Schur vectors `z`.
pub fn complex_schur_move(
	t: MatMut<'_, c64>,
	z: Option<MatMut<'_, c64>>,
	ifst: usize,
	ilst: usize,
) -> Result<usize, ReorderError> {
	let n = t.nrows();
	assert!(t.ncols() == n);
	assert!(ifst < n);
	assert!(ilst < n);
	if let Some(z) = z.as_ref() {
		assert!(z.nrows() == n && z.ncols() == n);
	}
	let mut ilst = ilst;
	let ierr = complex_schur::schur_move(t, z, ifst, &mut ilst);
	if ierr != 0 {
		return Err(ReorderError::SwapRejected { at: ilst });
	}
	Ok(ilst)
}

/// Reorders the Schur form so that the eigenvalues selected by `select`
/// occupy the leading rows of `t` (`ztrsen`-equivalent, reordering only).
/// Returns `m`, the dimension of the leading invariant subspace.
pub fn complex_schur_select(
	mut t: MatMut<'_, c64>,
	mut z: Option<MatMut<'_, c64>>,
	select: &[bool],
) -> Result<usize, ReorderError> {
	let n = t.nrows();
	assert!(t.ncols() == n);
	assert!(select.len() == n);
	if let Some(z) = z.as_ref() {
		assert!(z.nrows() == n && z.ncols() == n);
	}
	let mut ks = 0usize;
	for k in 0..n {
		if select[k] {
			if k != ks {
				let mut ilst = ks;
				let ierr = complex_schur::schur_move(t.rb_mut(), z.rb_mut(), k, &mut ilst);
				if ierr != 0 {
					return Err(ReorderError::SwapRejected { at: ilst });
				}
			}
			ks += 1;
		}
	}
	Ok(ks)
}
