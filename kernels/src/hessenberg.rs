//! Unblocked Householder Hessenberg reduction (`dgehd2`-shape) for f64 on
//! wasm — fix-2 of the eigen plan (docs/research-eig-wasm-2026-07.md).
//!
//! faer's Hessenberg runs unblocked below n=256 and its blocked panel is
//! gemv-bound either way; the measured GEMV runs at ~30% of STREAM
//! bandwidth on the reference runner (3× headroom), and Hessenberg is
//! ~36% of the repaired eigvals pipeline. This kernel is the same recipe
//! that beat scipy 2.5–3× on QR: per reflector, stream the trailing
//! columns exactly twice per side with flat simd128 `dot`/`axpy`, no
//! blocking, no T-matrix.
//!
//! Per column `j` (reflector from `x = A[j+1.., j]`, `v` stored in
//! `A[j+2.., j]` with implicit `v[0]=1`, `β` at `A[j+1, j]`, LAPACK
//! `dgehrd` storage):
//! - right-apply to ALL rows (`A[0..n, j+1..n] · H`): one gaxpy pass
//!   accumulating `w = A[:, j+1] + Σ v[t]·A[:, j+2+t]` into `work`, one
//!   axpy pass `A[:, c] -= (τ·v_c)·w`;
//! - left-apply to the trailing block (`H · A[j+1..n, j+1..n]`): per
//!   column a `dot` (`s = vᵀ·c`) then an `axpy` (`c -= τ·s·v`).
//!
//! Like the QR/LU kernels this targets the well-conditioned dense regime
//! the gate exercises (no `dlarfg` small-β rescaling path).

use faer::MatMut;

use crate::qr::{axpy, dot, scale};

/// Reduces square `A` (n×n, column-major, unit row stride) to upper
/// Hessenberg form in place. On exit the Hessenberg matrix occupies the
/// upper triangle plus the first subdiagonal; the Householder vectors sit
/// below the first subdiagonal (`dgehrd` storage) with `tau[j]` their
/// scalars (`k = n-2` reflectors; `tau` needs at least that). `work`
/// needs at least `n` scratch f64s.
pub fn hessenberg_factor_in_place(mut a: MatMut<'_, f64>, tau: &mut [f64], work: &mut [f64]) {
	let n = a.nrows();
	assert!(a.ncols() == n, "square input required");
	let k = n.saturating_sub(2);
	assert!(tau.len() >= k, "tau must hold n-2 scalars");
	assert!(work.len() >= n, "work must hold n scratch f64s");
	assert!(a.row_stride() == 1, "column-major with unit row stride required");
	let cs = a.col_stride() as usize;
	let base = a.as_ptr_mut();
	let w = work.as_mut_ptr();

	unsafe {
		for j in 0..k {
			let col = base.add(j * cs);
			let alpha = *col.add(j + 1);
			let tail = n - j - 2; // length of the stored v tail

			let xnorm_sq = if tail > 0 { dot(col.add(j + 2), col.add(j + 2), tail) } else { 0.0 };
			if xnorm_sq == 0.0 {
				tau[j] = 0.0;
				continue;
			}

			// dlarfg: beta = -sign(alpha)*hypot(alpha,‖x‖); v = x/(alpha-beta)
			let anorm = libm::sqrt(alpha * alpha + xnorm_sq);
			let beta = if alpha >= 0.0 { -anorm } else { anorm };
			let tj = (beta - alpha) / beta;
			let inv = 1.0 / (alpha - beta);
			scale(col.add(j + 2), inv, tail);
			tau[j] = tj;
			*col.add(j + 1) = beta;

			// ---- right-apply to all rows: A[:, j+1..n] := A[:, j+1..n]·H.
			// gaxpy pass: w = A[:, j+1] + Σ_t v[t]·A[:, j+2+t]
			let first = base.add((j + 1) * cs);
			core::ptr::copy_nonoverlapping(first, w, n);
			for t in 0..tail {
				let vt = *col.add(j + 2 + t);
				if vt != 0.0 {
					// w -= (-v[t]) * A[:, j+2+t]
					axpy(w, base.add((j + 2 + t) * cs), -vt, n);
				}
			}
			// update pass: A[:, j+1] -= τ·w; A[:, j+2+t] -= (τ·v[t])·w
			axpy(first, w, tj, n);
			for t in 0..tail {
				let vt = *col.add(j + 2 + t);
				if vt != 0.0 {
					axpy(base.add((j + 2 + t) * cs), w, tj * vt, n);
				}
			}

			// ---- left-apply to the trailing block: A[j+1.., j+1..] := H·(...).
			// per column: s = τ·(vᵀ·c); c[j+1] -= s; c[j+2..] -= s·v
			let mut c = j + 1;
			while c < n {
				let ac = base.add(c * cs);
				let mut s = *ac.add(j + 1);
				if tail > 0 {
					s += dot(col.add(j + 2), ac.add(j + 2), tail);
				}
				s *= tj;
				*ac.add(j + 1) -= s;
				if tail > 0 {
					axpy(ac.add(j + 2), col.add(j + 2), s, tail);
				}
				c += 1;
			}
		}
	}
}
