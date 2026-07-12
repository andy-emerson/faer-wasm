//! c64 twin of [`crate::eigvec`] (eigenvector campaign, 2026-07-12):
//! right eigenvectors from the *complex* Schur form. Structurally simpler
//! than the real kernel — complex T is truly upper triangular, so there
//! are no 2×2 blocks and every back-substitution step is one guarded
//! complex division.
//!
//! Shape notes vs LAPACK's `ztrevc3`: the destination semantics are the
//! same (per-column back-substitution on T − λI, back-transform through
//! Z, largest-`cabs1`-component normalization), but ztrevc3 delegates the
//! triangular solve to `zlatrs`. We instead run the same per-element
//! guard set the real kernel ports from `dlaln2`'s complex 1×1 branch
//! (tiny-pivot perturbation to `smin`, rhs pre-scale when the pivot is
//! small, dtrevc's column-norm/BIGNUM growth guard) — the identical
//! protection style at a fraction of the code, in the well-conditioned
//! dense regime this project targets. The back-transform is one
//! triangular matmul, exactly like the real kernel (the eigenvector
//! matrix of a triangular T is itself upper triangular).

use faer::linalg::matmul::triangular::{self, BlockStructure};
use faer::prelude::*;
use faer::{c64, Accum, MatMut, MatRef};

use crate::cplx::{cabs1, caxpy, cmul_real, cscale_re};
use crate::eigvec::ladiv;

/// All right eigenvectors of the matrix whose complex Schur form is
/// `(t, z)` — `t` upper triangular, `z` unitary. Writes the eigenvectors
/// of A = Z·T·Zᴴ into `v` (one column per eigenvalue, in T's diagonal
/// order), each normalized so its largest `|re| + |im|` component is 1
/// (ztrevc's convention). Column-major, unit row stride required.
pub fn ctrevc_in_place(t: MatRef<'_, c64>, z: MatRef<'_, c64>, mut v: MatMut<'_, c64>) {
	let n = t.nrows();
	assert!(t.ncols() == n && z.nrows() == n && z.ncols() == n);
	assert!(v.nrows() == n && v.ncols() == n);
	assert!(t.row_stride() == 1 && z.row_stride() == 1 && v.row_stride() == 1);
	if n == 0 {
		return;
	}

	let ulp = f64::EPSILON;
	let smlnum = f64::MIN_POSITIVE * (n as f64 / ulp);
	let bignum = (1.0 - ulp) / smlnum;
	// dlaln2's floor for the perturbed pivot
	let laln_smlnum = 2.0 * f64::MIN_POSITIVE;

	// strictly-above-diagonal cabs1 norms of T's columns (the growth guard)
	let mut colnorm = alloc::vec![0.0f64; n];
	for j in 1..n {
		let mut s = 0.0;
		for i in 0..j {
			s += cabs1(t[(i, j)]);
		}
		colnorm[j] = s;
	}

	// X: eigenvectors of T, upper triangular
	let mut x = faer::Mat::<c64>::zeros(n, n);
	let xcs = x.as_ref().col_stride() as usize;
	debug_assert!(x.as_ref().row_stride() == 1);
	let xp = x.as_mut().as_ptr_mut();
	let tp = t.as_ptr();
	let tcs = t.col_stride() as usize;

	for ki in (0..n).rev() {
		let lam = t[(ki, ki)];
		let smin = (ulp * cabs1(lam)).max(smlnum).max(laln_smlnum);
		let xcol = unsafe { xp.add(ki * xcs) };
		unsafe {
			*xcol.add(ki) = c64::new(1.0, 0.0);
			for r in 0..ki {
				*xcol.add(r) = -t[(r, ki)];
			}
		}
		for j in (0..ki).rev() {
			// solve (t[j,j] − λ)·x = rhs with dlaln2's 1×1-complex guards
			let mut den = t[(j, j)] - lam;
			if cabs1(den) < smin {
				den = c64::new(smin, 0.0);
			}
			let cnorm = cabs1(den);
			let rhs = unsafe { *xcol.add(j) };
			let bnorm = cabs1(rhs);
			let mut scale = 1.0f64;
			if cnorm < 1.0 && bnorm > 1.0 && bnorm > bignum * cnorm {
				scale = 1.0 / bnorm;
			}
			let (xr, xi) = ladiv(scale * rhs.re, scale * rhs.im, den.re, den.im);
			let mut xj = c64::new(xr, xi);
			// dtrevc-style growth guard against the column norm
			let xnorm = cabs1(xj);
			if xnorm > 1.0 && colnorm[j] > bignum / xnorm {
				xj = cmul_real(xj, 1.0 / xnorm);
				scale /= xnorm;
			}
			if scale != 1.0 {
				unsafe { cscale_re(xcol, scale, ki + 1) };
			}
			unsafe {
				*xcol.add(j) = xj;
				// rhs[0..j] -= xj · t[0..j, j]
				caxpy(xcol, tp.add(j * tcs), xj, j);
			}
		}
	}

	// back-transform: V = Z · X (upper triangular X) — the gemm-shaped bulk
	triangular::matmul(
		v.rb_mut(),
		BlockStructure::Rectangular,
		Accum::Replace,
		z,
		BlockStructure::Rectangular,
		x.as_ref(),
		BlockStructure::TriangularUpper,
		c64::new(1.0, 0.0),
		Par::Seq,
	);

	// ztrevc normalization: largest cabs1 component → 1
	let vp = v.as_ptr_mut();
	let vcs = v.col_stride() as usize;
	for k in 0..n {
		let mut emax = 0.0f64;
		for r in 0..n {
			emax = emax.max(cabs1(v[(r, k)]));
		}
		if emax > 0.0 {
			unsafe { cscale_re(vp.add(k * vcs), 1.0 / emax, n) };
		}
	}
}
