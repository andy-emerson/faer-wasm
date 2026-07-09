//! Blocked LU with partial pivoting, `dgetrf`-shaped, for f64 on wasm.
//!
//! Structure per block step `j` (block width `nb`):
//!   1. **panel**: factor columns `j..j+kb` over rows `j..m` with flat
//!      scalar loops (pivot search, row swap, scale, rank-1 updates) — the
//!      O(n²·nb) part, written in the code shape wasm engines compile well;
//!   2. **pivots**: apply the panel's row swaps to the columns outside it;
//!   3. **trsm**: `A12 ← L11⁻¹ A12` (unit lower, forward substitution),
//!      lean loops over contiguous columns;
//!   4. **gemm**: `A22 ← A22 − A21·A12` via `faer::linalg::matmul` — the
//!      O(n³) bulk rides the fast microkernels.
//!
//! Pivot indices follow LAPACK `ipiv` semantics: at step `k`, rows `k` and
//! `piv[k]` were swapped (`piv[k] ≥ k`).

use faer::linalg::matmul::matmul;
use faer::prelude::*;
use faer::{Accum, MatMut};

/// `dst[i] -= src[i] * alpha` — the panel/trsm/solve workhorse. On wasm this
/// is an explicit simd128 kernel (2 lanes × 2-unrolled; `v128_load`/`store`
/// are alignment-free by spec); elsewhere a plain scalar loop. This is the
/// "shaping": the same axpy faer expresses through generic SIMD dispatch,
/// written the way the target actually executes it.
#[inline(always)]
unsafe fn axpy(dst: *mut f64, src: *const f64, alpha: f64, len: usize) {
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

/// Block width. Swept on wasm 2026-07-09 (see docs/benchmarks-2026-07.md):
/// panels narrower than ~24 waste gemm efficiency, wider ones spend too
/// long in the O(n²·nb) panel at these sizes.
pub const RECOMMENDED_BLOCK_SIZE: usize = 64;

/// Factors a square `A` in place into `P·A = L·U` (`L` unit lower, `U`
/// upper, both stored in `a`), recording LAPACK-style pivots in `piv`.
/// `nb = 0` uses [`RECOMMENDED_BLOCK_SIZE`].
pub fn lu_factor_in_place(mut a: MatMut<'_, f64>, piv: &mut [usize], nb: usize) {
	let n = a.nrows();
	assert!(a.ncols() == n, "square only for now");
	assert!(piv.len() >= n);
	assert!(a.row_stride() == 1, "column-major with unit row stride required");
	// pure-panel mode through n=128: the lean panel alone matches scipy at
	// small n, and skinny-k gemm calls cost more than they save there
	let nb = if nb == 0 {
		if n <= 128 { n.max(1) } else { RECOMMENDED_BLOCK_SIZE }
	} else {
		nb
	};
	let cs = a.col_stride() as usize;
	let base = a.as_ptr_mut();

	let mut j = 0usize;
	while j < n {
		let kb = Ord::min(nb, n - j);

		// 1. panel over rows j..n, columns j..j+kb
		unsafe {
			for k in 0..kb {
				let jc = j + k;
				let col = base.add(jc * cs);
				// pivot search below (and including) the diagonal
				let mut p = jc;
				let mut mx = (*col.add(jc)).abs();
				let mut i = jc + 1;
				while i < n {
					let v = (*col.add(i)).abs();
					if v > mx {
						mx = v;
						p = i;
					}
					i += 1;
				}
				piv[jc] = p;
				// swap rows jc <-> p across the panel columns
				if p != jc {
					let mut c = j;
					while c < j + kb {
						let pc = base.add(c * cs);
						let t = *pc.add(jc);
						*pc.add(jc) = *pc.add(p);
						*pc.add(p) = t;
						c += 1;
					}
				}
				let d = *col.add(jc);
				if d != 0.0 {
					// scale the multipliers
					let inv = 1.0 / d;
					let mut i = jc + 1;
					while i < n {
						*col.add(i) *= inv;
						i += 1;
					}
				}
				// rank-1 update of the remaining panel columns
				let mut l = k + 1;
				while l < kb {
					let colr = base.add((j + l) * cs);
					let alk = *colr.add(jc);
					if alk != 0.0 {
						axpy(colr.add(jc + 1), col.add(jc + 1), alk, n - jc - 1);
					}
					l += 1;
				}
			}

			// 2. apply the panel's row swaps to columns outside the panel
			for k in 0..kb {
				let jc = j + k;
				let p = piv[jc];
				if p != jc {
					let mut c = 0usize;
					while c < n {
						if c == j {
							c = j + kb; // skip the panel: already swapped
							continue;
						}
						let pc = base.add(c * cs);
						let t = *pc.add(jc);
						*pc.add(jc) = *pc.add(p);
						*pc.add(p) = t;
						c += 1;
					}
				}
			}

			// 3. trsm: A12 <- L11^-1 A12 (unit lower forward substitution),
			// one contiguous column of A12 at a time
			if j + kb < n {
				let mut c = j + kb;
				while c < n {
					let xc = base.add(c * cs);
					for k in 0..kb {
						let xk = *xc.add(j + k);
						if xk != 0.0 {
							let lc = base.add((j + k) * cs);
							axpy(xc.add(j + k + 1), lc.add(j + k + 1), xk, kb - k - 1);
						}
					}
					c += 1;
				}
			}
		}

		// 4. gemm: A22 -= A21 * A12 — the O(n³) bulk on faer's microkernels
		if j + kb < n {
			let (_, a12_full, a21_full, a22) = a.rb_mut().split_at_mut(j + kb, j + kb);
			let a12 = a12_full.rb().subrows(j, kb);
			let a21 = a21_full.rb().subcols(j, kb);
			matmul(a22, Accum::Add, a21, a12, -1.0, Par::Seq);
		}

		j += kb;
	}
}

/// Solves `A·x = b` in place using factors from [`lu_factor_in_place`]:
/// applies the row swaps to `b`, then unit-lower forward substitution and
/// upper back substitution. `b` is a single column.
pub fn lu_solve_in_place(a: MatRef<'_, f64>, piv: &[usize], b: &mut [f64]) {
	let n = a.nrows();
	assert!(a.ncols() == n && b.len() == n && piv.len() >= n);
	assert!(a.row_stride() == 1);
	let cs = a.col_stride() as usize;
	let base = a.as_ptr();
	unsafe {
		for k in 0..n {
			let p = piv[k];
			if p != k {
				b.swap(k, p);
			}
		}
		// L y = P b (unit lower)
		for k in 0..n {
			let yk = b[k];
			if yk != 0.0 {
				let col = base.add(k * cs);
				axpy(b.as_mut_ptr().add(k + 1), col.add(k + 1), yk, n - k - 1);
			}
		}
		// U x = y
		for k in (0..n).rev() {
			let col = base.add(k * cs);
			let xk = b[k] / *col.add(k);
			b[k] = xk;
			if xk != 0.0 {
				axpy(b.as_mut_ptr(), col, xk, k);
			}
		}
	}
}

/// Base-case width for [`lu_factor_recursive_in_place`] — below this the
/// recursion switches to flat right-looking loops. **Set to `usize::MAX`:
/// the default never recurses.**
///
/// The recursion was measured on the GitHub runner (the reference machine
/// for the scipy comparison) over 3 rounds plus a dedicated large-n probe
/// to n=1024 (`lu-largen.mjs`). Verdict: **pure-flat wins at every size
/// 64–1024** — a tie at n=512 and 6.5–8.5% *ahead* of the best recursion
/// at 640/768/1024, with no sign of the margin closing. On 2-lane wasm the
/// flat simd128 rank-1 panel runs so close to gemm rate that routing flops
/// into gemm never pays, and the skinny-gemm penalty makes it lose. (A dev
/// box showed recursion winning at large n — a machine-specific cache
/// quirk that did NOT reproduce on the runner; kept as the cautionary
/// reason we tune on the runner, not locally.) The recursion machinery is
/// retained for explicit-`crossover` opt-in on a machine that profiles
/// differently, but is off by default.
pub const RECOMMENDED_CROSSOVER: usize = usize::MAX;

/// Recursive LU (`dgetrf2`/Toledo shape) — the top-ranked technique from
/// docs/research-lu-wasm-2026-07.md: splitting at w/2 casts the panel's
/// memory-bound rank-1 work into trsm + gemm at growing ranks; the
/// `crossover`-wide base case runs the flat simd128 loops. **By default it
/// does not recurse at all** ([`RECOMMENDED_CROSSOVER`] = `usize::MAX`): the
/// runner large-n probe (to n=1024) showed the pure-flat panel winning at
/// every size, so the default is that flat panel — the fastest wasm LU
/// measured, at scipy parity around n=256. Pass an explicit finite
/// `crossover` to opt into recursion (loses on the reference runner; may
/// help a machine with a different cache/bandwidth balance).
///
/// Base-case width for the recursive unit-lower trsm (`trsm_rec`) — below
/// this the trsm runs flat simd128 substitution instead of splitting into
/// gemms. Swept on the runner 2026-07-09 (`lu-tune.yml`, 3 rounds): the
/// n=512 winners all used a *non-recursing* trsm (tb ≥ 256), same lesson as
/// the crossover — flat simd128 substitution beats splitting into small
/// gemms on 2-lane wasm through the swept range.
pub const RECOMMENDED_TRSM_BASE: usize = 256;

/// Same output contract as [`lu_factor_in_place`] (LAPACK ipiv semantics):
/// identical pivot sequence, factors equal up to gemm reassociation.
/// `crossover = 0` uses [`RECOMMENDED_CROSSOVER`].
pub fn lu_factor_recursive_in_place(a: MatMut<'_, f64>, piv: &mut [usize], crossover: usize) {
	lu_factor_recursive_in_place_tuned(a, piv, crossover, RECOMMENDED_TRSM_BASE);
}

/// As [`lu_factor_recursive_in_place`] but with both tunables exposed for
/// on-target parameter sweeps. `crossover = 0` → size-dependent default;
/// `trsm_base = 0` → [`RECOMMENDED_TRSM_BASE`]. Consumers should call the
/// non-`_tuned` entry; this exists so the bench harness can sweep on the
/// runner rather than trusting dev-box numbers.
pub fn lu_factor_recursive_in_place_tuned(
	mut a: MatMut<'_, f64>,
	piv: &mut [usize],
	crossover: usize,
	trsm_base: usize,
) {
	let n = a.nrows();
	assert!(a.ncols() == n, "square only for now");
	assert!(piv.len() >= n);
	assert!(a.row_stride() == 1, "column-major with unit row stride required");
	// default (crossover == 0) never recurses — pure-flat won at every size
	// 64–1024 on the runner (see RECOMMENDED_CROSSOVER). An explicit finite
	// crossover re-enables recursion for machines that profile differently.
	let crossover = if crossover == 0 { RECOMMENDED_CROSSOVER } else { crossover };
	let trsm_base = if trsm_base == 0 { RECOMMENDED_TRSM_BASE } else { trsm_base };
	let cs = a.col_stride() as usize;
	unsafe {
		lu_rec(a.rb_mut(), cs, 0, n, piv, crossover, trsm_base);
	}
}

/// swap rows r1 <-> r2 across columns [c0, c1)
#[inline(always)]
unsafe fn swap_rows(base: *mut f64, cs: usize, c0: usize, c1: usize, r1: usize, r2: usize) {
	let mut c = c0;
	while c < c1 {
		let pc = base.add(c * cs);
		let t = *pc.add(r1);
		*pc.add(r1) = *pc.add(r2);
		*pc.add(r2) = t;
		c += 1;
	}
}

/// factor the `w`-wide block starting at diagonal position `d` over rows
/// `d..m`, columns `d..d+w`, recording absolute pivots; swaps are applied
/// only within columns [d, d+w) — the caller handles siblings
unsafe fn lu_rec(
	mut a: MatMut<'_, f64>,
	cs: usize,
	d: usize,
	w: usize,
	piv: &mut [usize],
	crossover: usize,
	trsm_base: usize,
) {
	let m = a.nrows();
	let base = a.rb_mut().as_ptr_mut();
	if w <= crossover {
		// base case: flat-loop right-looking factorization (same shape as
		// the blocked driver's panel)
		for k in 0..w {
			let jc = d + k;
			let col = base.add(jc * cs);
			let mut p = jc;
			let mut mx = (*col.add(jc)).abs();
			let mut i = jc + 1;
			while i < m {
				let v = (*col.add(i)).abs();
				if v > mx {
					mx = v;
					p = i;
				}
				i += 1;
			}
			piv[jc] = p;
			if p != jc {
				swap_rows(base, cs, d, d + w, jc, p);
			}
			let dv = *col.add(jc);
			if dv != 0.0 {
				let inv = 1.0 / dv;
				let mut i = jc + 1;
				while i < m {
					*col.add(i) *= inv;
					i += 1;
				}
			}
			let mut l = k + 1;
			while l < w {
				let colr = base.add((d + l) * cs);
				let alk = *colr.add(jc);
				if alk != 0.0 {
					axpy(colr.add(jc + 1), col.add(jc + 1), alk, m - jc - 1);
				}
				l += 1;
			}
		}
		return;
	}

	let n1 = w / 2;
	// 1. factor the left half over all rows
	lu_rec(a.rb_mut(), cs, d, n1, piv, crossover, trsm_base);
	let base = a.rb_mut().as_ptr_mut();
	// 2. apply the left half's pivots to the right-half columns
	for k in d..d + n1 {
		if piv[k] != k {
			swap_rows(base, cs, d + n1, d + w, k, piv[k]);
		}
	}
	// 3. A12 = L11^{-1} A12 (recursive unit-lower trsm: off-diagonal via gemm)
	trsm_rec(a.rb_mut(), cs, d, n1, d + n1, d + w, trsm_base);
	// 4. A22 -= A21 * A12 (rows d+n1..m)
	{
		let (_, a12_full, a21_full, a22_full) = a.rb_mut().split_at_mut(d + n1, d + n1);
		// a12_full: rows 0..d+n1 × cols d+n1..ncols ; want rows d..d+n1, cols 0..(w-n1)
		let a12 = a12_full.rb().subrows(d, n1).subcols(0, w - n1);
		// a21_full: rows d+n1..m × cols 0..d+n1 ; want cols d..d+n1
		let a21 = a21_full.rb().subcols(d, n1);
		// a22_full: rows d+n1..m × cols d+n1.. ; want cols 0..(w-n1)
		let a22 = a22_full.subcols_mut(0, w - n1);
		matmul(a22, Accum::Add, a21, a12, -1.0, Par::Seq);
	}
	// 5. factor the right half over rows d+n1..m
	lu_rec(a.rb_mut(), cs, d + n1, w - n1, piv, crossover, trsm_base);
	let base = a.rb_mut().as_ptr_mut();
	// 6. apply the right half's pivots back to the left-half columns
	for k in d + n1..d + w {
		if piv[k] != k {
			swap_rows(base, cs, d, d + n1, k, piv[k]);
		}
	}
}

/// X = L^{-1} X where L is the unit-lower k×k block at (d, d) and X spans
/// rows d..d+k, columns [c0, c1). Recursive: diagonal blocks by lean
/// substitution, the off-diagonal update by gemm.
unsafe fn trsm_rec(mut a: MatMut<'_, f64>, cs: usize, d: usize, k: usize, c0: usize, c1: usize, trsm_base: usize) {
	if k <= trsm_base {
		let base = a.rb_mut().as_ptr_mut();
		let mut c = c0;
		while c < c1 {
			let xc = base.add(c * cs);
			for kk in 0..k {
				let xk = *xc.add(d + kk);
				if xk != 0.0 {
					let lc = base.add((d + kk) * cs);
					axpy(xc.add(d + kk + 1), lc.add(d + kk + 1), xk, k - kk - 1);
				}
			}
			c += 1;
		}
		return;
	}
	let k1 = k / 2;
	// solve top: L11 X1
	trsm_rec(a.rb_mut(), cs, d, k1, c0, c1, trsm_base);
	// X2 -= L21 * X1
	{
		let (_, x1_full, l21_full, x2_full) = a.rb_mut().split_at_mut(d + k1, d + k1);
		let x1 = x1_full.rb().subrows(d, k1).subcols(c0 - (d + k1), c1 - c0);
		let l21 = l21_full.rb().subrows(0, k - k1).subcols(d, k1);
		let x2 = x2_full.subrows_mut(0, k - k1).subcols_mut(c0 - (d + k1), c1 - c0);
		matmul(x2, Accum::Add, l21, x1, -1.0, Par::Seq);
	}
	// solve bottom: L22 X2
	trsm_rec(a.rb_mut(), cs, d + k1, k - k1, c0, c1, trsm_base);
}
