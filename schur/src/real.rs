//! Real Schur decomposition `A = Z T Zᵀ` for `f64`.
//!
//! `T` is quasi upper triangular: 1×1 diagonal blocks are real eigenvalues,
//! 2×2 diagonal blocks (nonzero subdiagonal entry) are complex-conjugate
//! pairs. The driver mirrors faer's own EVD pipeline exactly (Hessenberg →
//! block-Householder accumulation of `Z` → multishift QR with `want_t`), so
//! `T`/`Z` here and eigenvalues from `faer`'s EVD come from the same kernels.

use crate::{ReorderError, SchurError};
use faer::dyn_stack::{MemBuffer, MemStack, StackReq};
use faer::linalg::evd::schur::{self, real_schur};
use faer::linalg::evd::hessenberg;
use faer::linalg::householder;
use faer::linalg::qr::no_pivoting::factor::recommended_block_size;
use faer::linalg::{temp_mat_scratch, temp_mat_zeroed};
use faer::mat::AsMatMut;
use faer::prelude::*;
use faer::{Auto, Conj};

pub use faer::linalg::evd::schur::SchurParams;

/// scratch space required by [`real_schur_in_place`]
pub fn real_schur_scratch(n: usize, par: Par, params: SchurParams) -> StackReq {
	if n <= 1 {
		return StackReq::EMPTY;
	}
	let bs = recommended_block_size::<f64>(n - 1, n - 1);
	StackReq::all_of(&[
		temp_mat_scratch::<f64>(bs, n - 1),
		StackReq::any_of(&[
			hessenberg::hessenberg_in_place_scratch::<f64>(n, bs, par, Default::default()),
			householder::apply_block_householder_sequence_on_the_right_in_place_scratch::<f64>(
				n - 1,
				bs,
				n - 1,
			),
			schur::multishift_qr_scratch::<f64>(n, n, true, true, par, params),
		]),
	])
}

/// Computes the real Schur form in place.
///
/// On entry `h` is a general square matrix `A`; on exit it is the quasi
/// upper triangular `T`. If `z` is provided it must be `n×n` and is
/// overwritten with the orthogonal `Z` such that `A = Z T Zᵀ`. Eigenvalues
/// land in `w_re`/`w_im` (a 2×2 block at `k` yields the conjugate pair at
/// `k`, `k+1`).
pub fn real_schur_in_place(
	mut h: MatMut<'_, f64>,
	mut z: Option<MatMut<'_, f64>>,
	mut w_re: ColMut<'_, f64>,
	mut w_im: ColMut<'_, f64>,
	par: Par,
	stack: &mut MemStack,
	params: SchurParams,
) -> Result<(), SchurError> {
	let n = h.nrows();
	assert!(h.ncols() == n);
	assert!(w_re.nrows() == n);
	assert!(w_im.nrows() == n);
	if let Some(z) = z.rb() {
		assert!(z.nrows() == n);
		assert!(z.ncols() == n);
	}
	for j in 0..n {
		for i in 0..n {
			if !h[(i, j)].is_finite() {
				return Err(SchurError::NonFinite);
			}
		}
	}
	if let Some(z) = z.rb_mut() {
		let mut z = z;
		z.fill(0.0);
		z.diagonal_mut().fill(1.0);
	}
	if n == 0 {
		return Ok(());
	}
	if n == 1 {
		w_re[0] = h[(0, 0)];
		w_im[0] = 0.0;
		return Ok(());
	}

	let bs = recommended_block_size::<f64>(n - 1, n - 1);
	{
		let (mut hh, stack) = temp_mat_zeroed::<f64, _, _>(bs, n - 1, &mut *stack);
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
			h[(i, j)] = 0.0;
		}
	}

	let (info, _, _) = real_schur::multishift_qr::<f64>(
		true,
		h.rb_mut(),
		z.rb_mut(),
		w_re.rb_mut(),
		w_im.rb_mut(),
		0,
		n,
		par,
		stack,
		params,
	);
	if info != 0 {
		return Err(SchurError::NoConvergence);
	}
	// faer's blocked path (n ≥ blocking threshold) uses the region strictly
	// below the subdiagonal as workspace and does not clean it up (its EVD
	// only reads the upper part); T is only quasi triangular after this
	for j in 0..n {
		for i in j + 2..n {
			h[(i, j)] = 0.0;
		}
	}
	Ok(())
}

/// result of [`real_schur`]: `a = z * t * z.transpose()`
pub struct RealSchur {
	/// quasi upper triangular Schur form
	pub t: Mat<f64>,
	/// orthogonal Schur vectors
	pub z: Mat<f64>,
	/// eigenvalue real parts
	pub w_re: Col<f64>,
	/// eigenvalue imaginary parts
	pub w_im: Col<f64>,
}

/// Allocating convenience wrapper around [`real_schur_in_place`].
pub fn real_schur(a: MatRef<'_, f64>, par: Par) -> Result<RealSchur, SchurError> {
	let n = a.nrows();
	assert!(a.ncols() == n);
	let params: SchurParams = Auto::<f64>::auto();
	let mut t = a.to_owned();
	let mut z = Mat::zeros(n, n);
	let mut w_re = Col::zeros(n);
	let mut w_im = Col::zeros(n);
	let mut buf = MemBuffer::new(real_schur_scratch(n, par, params));
	let stack = MemStack::new(&mut buf);
	real_schur_in_place(
		t.as_mut(),
		Some(z.as_mut()),
		w_re.as_mut(),
		w_im.as_mut(),
		par,
		stack,
		params,
	)?;
	Ok(RealSchur { t, z, w_re, w_im })
}

/// Moves the diagonal block containing row `ifst` to row `ilst`
/// (`dtrexc`-equivalent), updating `t` and, if given, the Schur vectors `z`.
///
/// Returns the row index the block actually landed at (block boundaries can
/// shift both indices by one, exactly as in LAPACK).
pub fn real_schur_move(
	t: MatMut<'_, f64>,
	z: Option<MatMut<'_, f64>>,
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
	let ierr = real_schur::schur_move(t, z, ifst, &mut ilst);
	if ierr != 0 {
		return Err(ReorderError::SwapRejected { at: ilst });
	}
	Ok(ilst)
}

/// Reorders the Schur form so that the eigenvalues selected by `select`
/// occupy the leading rows of `t` (`dtrsen`-equivalent, reordering only).
///
/// `select[k]` refers to the eigenvalue at row `k` of the *input* `t`; for a
/// complex-conjugate 2×2 block, selecting either member moves the pair.
/// Returns `m`, the dimension of the leading invariant subspace (the first
/// `m` columns of the updated `z` span it).
pub fn real_schur_select(
	mut t: MatMut<'_, f64>,
	mut z: Option<MatMut<'_, f64>>,
	select: &[bool],
) -> Result<usize, ReorderError> {
	let n = t.nrows();
	assert!(t.ncols() == n);
	assert!(select.len() == n);
	if let Some(z) = z.as_ref() {
		assert!(z.nrows() == n && z.ncols() == n);
	}
	// LAPACK dtrsen scan: positions ≥ k in the current T still hold the
	// input's eigenvalue k (moving a block from k to ks < k only slides the
	// already-scanned, unselected blocks in [ks, k) down), so indexing
	// `select` by the loop position is exact, with the `pair` flag skipping
	// the second row of a moved 2×2 block.
	let mut ks = 0usize;
	let mut pair = false;
	for k in 0..n {
		if pair {
			pair = false;
			continue;
		}
		let mut swap = select[k];
		if k + 1 < n && t[(k + 1, k)] != 0.0 {
			pair = true;
			swap = swap || select[k + 1];
		}
		if swap {
			if k != ks {
				let mut ilst = ks;
				let ierr = real_schur::schur_move(t.rb_mut(), z.rb_mut(), k, &mut ilst);
				if ierr != 0 {
					return Err(ReorderError::SwapRejected { at: ilst });
				}
			}
			ks += if pair { 2 } else { 1 };
		}
	}
	Ok(ks)
}
