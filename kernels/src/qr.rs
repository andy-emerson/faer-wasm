//! Unblocked Householder QR (`dgeqr2`-shape) on wasm — the wasm-shaped
//! answer for QR at n ≤ 512 argued in docs/research-qr-wasm-2026-07.md.
//! Generic over [`WasmScalar`] (f64/f32) since the f32/c32 phase.
//!
//! Unlike LU, QR does **not** want blocking/recursion on wasm: the
//! compact-WY block-apply (`dlarfb`) carries a ~2× flop penalty that a
//! narrow-lane SIMD can't earn back until well past n=512, and measurement
//! already showed faer's unblocked (`block_size = 1`) path beating scipy
//! 1.3–1.7×. So this is a *fully unblocked* panel: per column, generate the
//! Householder reflector (`dlarfg`) and immediately apply it to the trailing
//! columns (`dlarf`) — the hot loop is a `dot` (vᵀ·c) then an `axpy`
//! (c −= τ·(vᵀc)·v), both in flat SIMD. No compact-WY, no T-matrix, no
//! trailing gemm.
//!
//! On exit `a` holds LAPACK `dgeqrf` storage: `R` in the upper triangle
//! (incl. diagonal), the Householder vectors `v` (implicit `v[0]=1`) below
//! the diagonal, and `tau[j]` the reflector scalars — so a companion
//! `apply Qᵀ` / `form Q` can reuse it. Column-major, unit row stride.

use faer::MatMut;

use crate::scalar::WasmScalar;

/// Factors `A` (m×n, m ≥ n or m < n both allowed; `k = min(m,n)` reflectors)
/// in place into the `dgeqrf` representation described in the module docs.
/// `tau` receives the `k` reflector scalars.
///
/// Uses the standard `dlarfg` reflector (`H = I − τ·v·vᵀ`, `v[0]=1`,
/// `β = −sign(α)·‖x‖`); the LAPACK small-`β` rescaling path is skipped —
/// like the LU kernels, this targets the well-conditioned dense regime the
/// gate exercises, not a general-purpose LAPACK drop-in.
pub fn qr_factor_in_place<T: WasmScalar>(a: MatMut<'_, T>, tau: &mut [T]) {
	let m = a.nrows();
	let n = a.ncols();
	let k = Ord::min(m, n);
	assert!(tau.len() >= k, "tau must hold min(m,n) scalars");
	assert!(a.row_stride() == 1, "column-major with unit row stride required");
	let cs = a.col_stride() as usize;
	let base = a.as_ptr_mut();

	unsafe {
		for j in 0..k {
			let col = base.add(j * cs);
			let alpha = *col.add(j);
			let tail = m - j - 1; // length of x = A[j+1.., j]

			// ‖x‖² over the sub-diagonal tail
			let xnorm_sq = if tail > 0 {
				T::dot(col.add(j + 1), col.add(j + 1), tail)
			} else {
				T::ZERO
			};

			if xnorm_sq == T::ZERO {
				// column already upper-triangular here: H = I
				tau[j] = T::ZERO;
				// R[j,j] = alpha stays as-is; no trailing update needed
				continue;
			}

			// dlarfg: beta = -sign(alpha)*hypot(alpha, ‖x‖); v = x/(alpha-beta)
			let anorm = (alpha * alpha + xnorm_sq).sqrt();
			let beta = if alpha >= T::ZERO { -anorm } else { anorm };
			let tj = (beta - alpha) / beta;
			let inv = T::ONE / (alpha - beta);
			T::scale(col.add(j + 1), inv, tail); // v tail (v[0] is implicit 1)
			tau[j] = tj;
			*col.add(j) = beta; // R[j,j]

			// apply H = I - tj * v vᵀ to each trailing column A[j.., c]:
			//   w = tj * (vᵀ · A[j.., c]);  A[j.., c] -= w * v
			let mut c = j + 1;
			while c < n {
				let ac = base.add(c * cs);
				// vᵀ·col: the v[0]=1 term is A[j,c], the tail is dot(v_tail, .)
				let mut w = *ac.add(j);
				if tail > 0 {
					w += T::dot(col.add(j + 1), ac.add(j + 1), tail);
				}
				w *= tj;
				*ac.add(j) -= w; // v[0]=1 component
				if tail > 0 {
					T::axpy(ac.add(j + 1), col.add(j + 1), w, tail);
				}
				c += 1;
			}
		}
	}
}
