//! Unblocked Householder QR (`dgeqr2`-shape) for f64 on wasm — the
//! wasm-shaped answer for QR at n ≤ 512 argued in
//! docs/research-qr-wasm-2026-07.md.
//!
//! Unlike LU, QR does **not** want blocking/recursion on wasm: the
//! compact-WY block-apply (`dlarfb`) carries a ~2× flop penalty that a
//! 2-lane f64 SIMD can't earn back until well past n=512, and measurement
//! already showed faer's unblocked (`block_size = 1`) path beating scipy
//! 1.3–1.7×. So this is a *fully unblocked* panel: per column, generate the
//! Householder reflector (`dlarfg`) and immediately apply it to the trailing
//! columns (`dlarf`) — the hot loop is a `dot` (vᵀ·c) then an `axpy`
//! (c −= τ·(vᵀc)·v), both in flat simd128. No compact-WY, no T-matrix, no
//! trailing gemm.
//!
//! On exit `a` holds LAPACK `dgeqrf` storage: `R` in the upper triangle
//! (incl. diagonal), the Householder vectors `v` (implicit `v[0]=1`) below
//! the diagonal, and `tau[j]` the reflector scalars — so a companion
//! `apply Qᵀ` / `form Q` can reuse it. Column-major, unit row stride.

use faer::MatMut;

/// `dst[i] -= src[i] * alpha`, simd128 on wasm (2 lanes, 2× unrolled;
/// `v128_load`/`store` are alignment-free by spec), scalar elsewhere.
#[inline(always)]
pub(crate) unsafe fn axpy(dst: *mut f64, src: *const f64, alpha: f64, len: usize) {
	#[cfg(target_arch = "wasm32")]
	{
		axpy_simd128(dst, src, alpha, len);
	}
	#[cfg(not(target_arch = "wasm32"))]
	{
		for i in 0..len {
			*dst.add(i) -= *src.add(i) * alpha;
		}
	}
}

/// `sum_i a[i]*b[i]` — the reflector's `vᵀ·c` (and `‖x‖²` when `a == b`).
#[inline(always)]
pub(crate) unsafe fn dot(a: *const f64, b: *const f64, len: usize) -> f64 {
	#[cfg(target_arch = "wasm32")]
	{
		dot_simd128(a, b, len)
	}
	#[cfg(not(target_arch = "wasm32"))]
	{
		let mut s = 0.0;
		for i in 0..len {
			s += *a.add(i) * *b.add(i);
		}
		s
	}
}

/// `dst[i] *= alpha` — scales the reflector tail into `v`.
#[inline(always)]
pub(crate) unsafe fn scale(dst: *mut f64, alpha: f64, len: usize) {
	#[cfg(target_arch = "wasm32")]
	{
		scale_simd128(dst, alpha, len);
	}
	#[cfg(not(target_arch = "wasm32"))]
	{
		for i in 0..len {
			*dst.add(i) *= alpha;
		}
	}
}

#[cfg(target_arch = "wasm32")]
#[target_feature(enable = "simd128")]
unsafe fn axpy_simd128(dst: *mut f64, src: *const f64, alpha: f64, len: usize) {
	use core::arch::wasm32::*;
	let va = f64x2_splat(alpha);
	let mut i = 0usize;
	while i + 4 <= len {
		let d0 = v128_load(dst.add(i) as *const v128);
		let s0 = v128_load(src.add(i) as *const v128);
		let d1 = v128_load(dst.add(i + 2) as *const v128);
		let s1 = v128_load(src.add(i + 2) as *const v128);
		v128_store(dst.add(i) as *mut v128, f64x2_sub(d0, f64x2_mul(s0, va)));
		v128_store(dst.add(i + 2) as *mut v128, f64x2_sub(d1, f64x2_mul(s1, va)));
		i += 4;
	}
	while i < len {
		*dst.add(i) -= *src.add(i) * alpha;
		i += 1;
	}
}

#[cfg(target_arch = "wasm32")]
#[target_feature(enable = "simd128")]
unsafe fn dot_simd128(a: *const f64, b: *const f64, len: usize) -> f64 {
	use core::arch::wasm32::*;
	let mut acc0 = f64x2_splat(0.0);
	let mut acc1 = f64x2_splat(0.0);
	let mut i = 0usize;
	while i + 4 <= len {
		let a0 = v128_load(a.add(i) as *const v128);
		let b0 = v128_load(b.add(i) as *const v128);
		let a1 = v128_load(a.add(i + 2) as *const v128);
		let b1 = v128_load(b.add(i + 2) as *const v128);
		acc0 = f64x2_add(acc0, f64x2_mul(a0, b0));
		acc1 = f64x2_add(acc1, f64x2_mul(a1, b1));
		i += 4;
	}
	let acc = f64x2_add(acc0, acc1);
	let mut s = f64x2_extract_lane::<0>(acc) + f64x2_extract_lane::<1>(acc);
	while i < len {
		s += *a.add(i) * *b.add(i);
		i += 1;
	}
	s
}

#[cfg(target_arch = "wasm32")]
#[target_feature(enable = "simd128")]
unsafe fn scale_simd128(dst: *mut f64, alpha: f64, len: usize) {
	use core::arch::wasm32::*;
	let va = f64x2_splat(alpha);
	let mut i = 0usize;
	while i + 4 <= len {
		let d0 = v128_load(dst.add(i) as *const v128);
		let d1 = v128_load(dst.add(i + 2) as *const v128);
		v128_store(dst.add(i) as *mut v128, f64x2_mul(d0, va));
		v128_store(dst.add(i + 2) as *mut v128, f64x2_mul(d1, va));
		i += 4;
	}
	while i < len {
		*dst.add(i) *= alpha;
		i += 1;
	}
}

/// Factors `A` (m×n, m ≥ n or m < n both allowed; `k = min(m,n)` reflectors)
/// in place into the `dgeqrf` representation described in the module docs.
/// `tau` receives the `k` reflector scalars.
///
/// Uses the standard `dlarfg` reflector (`H = I − τ·v·vᵀ`, `v[0]=1`,
/// `β = −sign(α)·‖x‖`); the LAPACK small-`β` rescaling path is skipped —
/// like the LU kernels, this targets the well-conditioned dense regime the
/// gate exercises, not a general-purpose LAPACK drop-in.
pub fn qr_factor_in_place(mut a: MatMut<'_, f64>, tau: &mut [f64]) {
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
			let xnorm_sq = if tail > 0 { dot(col.add(j + 1), col.add(j + 1), tail) } else { 0.0 };

			if xnorm_sq == 0.0 {
				// column already upper-triangular here: H = I
				tau[j] = 0.0;
				// R[j,j] = alpha stays as-is; no trailing update needed
				continue;
			}

			// dlarfg: beta = -sign(alpha)*hypot(alpha, ‖x‖); v = x/(alpha-beta)
			let anorm = libm::sqrt(alpha * alpha + xnorm_sq);
			let beta = if alpha >= 0.0 { -anorm } else { anorm };
			let tj = (beta - alpha) / beta;
			let inv = 1.0 / (alpha - beta);
			scale(col.add(j + 1), inv, tail); // v tail (v[0] is implicit 1)
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
					w += dot(col.add(j + 1), ac.add(j + 1), tail);
				}
				w *= tj;
				*ac.add(j) -= w; // v[0]=1 component
				if tail > 0 {
					axpy(ac.add(j + 1), col.add(j + 1), w, tail);
				}
				c += 1;
			}
		}
	}
}
