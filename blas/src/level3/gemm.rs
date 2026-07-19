//! `gemm` — matrix multiplication: C ← αAB + βC.
//!
//! Implementation: size-dispatched column-axpy family (tuned
//! 2026-07-18): 4×4 register tile below ~1.5 MB of A, 4-column fused
//! stream above — all bit-identical to the plain gemv-per-column
//! reference, which is kept as `gemm_colaxpy`. Both tuned shapes beat
//! faer's blocked gemm at every measured size (1.4–1.8× small, ~1.25×
//! large; docs/blas-ab-2026-07.md step 6). Transpose forms: not built
//! — no consumer yet (syrk covers A·Aᵀ).

use super::check_mat;
use crate::kernels::axpy4;
use crate::level2::gemv;

/// C ← αAB + βC. A is m×k, B is k×n, C is m×n; each matrix has its own
/// column stride.
///
/// Dispatches by size (tuning campaign, 2026-07-18, two runner draws
/// agreeing within 3%): the 4×4 register tile wins while A stays small
/// enough that its column-strided k-walk rides the caches (best
/// 128–384, 1.4–1.8× over faer); the 4-column fused stream wins above
/// (512+, ~1.25× over faer). All three shapes are bit-for-bit
/// identical, so the dispatch is invisible to results.
#[allow(clippy::too_many_arguments)]
pub fn gemm(
	alpha: f64,
	m: usize,
	k: usize,
	n: usize,
	a: &[f64],
	acs: usize,
	b: &[f64],
	bcs: usize,
	beta: f64,
	c: &mut [f64],
	ccs: usize,
) {
	// measured crossover between n=384 (A ≈ 1.2 MB: tiled) and n=512
	// (A = 2 MB: col4) on both reference draws and the container
	const TILED_MAX_A_BYTES: usize = 3 << 19; // 1.5 MB
	if m * k * 8 <= TILED_MAX_A_BYTES {
		gemm_tiled(alpha, m, k, n, a, acs, b, bcs, beta, c, ccs);
	} else {
		gemm_col4(alpha, m, k, n, a, acs, b, bcs, beta, c, ccs);
	}
}

/// The original column-axpy shape (gemv per column) — kept as the
/// plain reference the tuned shapes are raced and bit-checked against.
#[allow(clippy::too_many_arguments)]
pub fn gemm_colaxpy(
	alpha: f64,
	m: usize,
	k: usize,
	n: usize,
	a: &[f64],
	acs: usize,
	b: &[f64],
	bcs: usize,
	beta: f64,
	c: &mut [f64],
	ccs: usize,
) {
	check_mat(a.len(), m, k, acs);
	check_mat(b.len(), k, n, bcs);
	check_mat(c.len(), m, n, ccs);
	for j in 0..n {
		gemv(alpha, m, k, a, acs, &b[j * bcs..j * bcs + k], beta, &mut c[j * ccs..j * ccs + m]);
	}
}

/// Tuning-campaign candidate (2026-07-18): 4×4 register-tiled gemm.
/// The column-axpy gemm re-reads and re-writes each C element once per
/// k step; this micro-kernel holds a 4-row × 4-column tile of C in 8
/// SIMD registers across the whole k loop — one C load and one C store
/// per tile, 16 FLOPs per 2 A-register loads. Rounding sequence per
/// element is IDENTICAL to the column-axpy path (β first, then one
/// α·b rounding and one multiply-add per k, ascending), so the two are
/// bit-for-bit interchangeable — tested. Tails (m%4, n%4) fall back to
/// the gemv path.
#[allow(clippy::too_many_arguments)]
pub fn gemm_tiled(
	alpha: f64,
	m: usize,
	k: usize,
	n: usize,
	a: &[f64],
	acs: usize,
	b: &[f64],
	bcs: usize,
	beta: f64,
	c: &mut [f64],
	ccs: usize,
) {
	check_mat(a.len(), m, k, acs);
	check_mat(b.len(), k, n, bcs);
	check_mat(c.len(), m, n, ccs);
	let mut j = 0usize;
	while j + 4 <= n {
		let mut i = 0usize;
		while i + 4 <= m {
			unsafe {
				tile_4x4(
					alpha,
					k,
					a.as_ptr().add(i),
					acs,
					b.as_ptr().add(j * bcs),
					bcs,
					beta,
					c.as_mut_ptr().add(j * ccs + i),
					ccs,
				);
			}
			i += 4;
		}
		// row tail for these four columns: same per-element sequence as
		// the gemv path over the remaining rows
		if i < m {
			for jj in j..j + 4 {
				let seg = &mut c[jj * ccs + i..jj * ccs + m];
				crate::level2::scale_y(beta, seg);
				for l in 0..k {
					crate::level1::axpy(
						alpha * b[jj * bcs + l],
						&a[l * acs + i..l * acs + m],
						seg,
					);
				}
			}
		}
		j += 4;
	}
	// column tail: plain gemv columns
	while j < n {
		gemv(alpha, m, k, a, acs, &b[j * bcs..j * bcs + k], beta, &mut c[j * ccs..j * ccs + m]);
		j += 1;
	}
}

/// Tuning-campaign candidate 2 (2026-07-18): 4-column fused gemm.
/// The 4×4 tile's k-loop walks A at column stride (TLB/prefetch-hostile
/// at large n — measured losing above n≈512); this shape instead
/// streams each A column SEQUENTIALLY, once per group of four C
/// columns — A traffic drops 4× vs column-axpy while the four hot C
/// columns ride the near caches. Rounding sequence per element is
/// identical to the column-axpy path — bit-for-bit interchangeable,
/// tested.
#[allow(clippy::too_many_arguments)]
pub fn gemm_col4(
	alpha: f64,
	m: usize,
	k: usize,
	n: usize,
	a: &[f64],
	acs: usize,
	b: &[f64],
	bcs: usize,
	beta: f64,
	c: &mut [f64],
	ccs: usize,
) {
	check_mat(a.len(), m, k, acs);
	check_mat(b.len(), k, n, bcs);
	check_mat(c.len(), m, n, ccs);
	let mut j = 0usize;
	while j + 4 <= n {
		{
			let cp = c.as_mut_ptr();
			for u in 0..4 {
				let col =
					unsafe { core::slice::from_raw_parts_mut(cp.add((j + u) * ccs), m) };
				crate::level2::scale_y(beta, col);
			}
			for l in 0..k {
				let t = [
					alpha * b[j * bcs + l],
					alpha * b[(j + 1) * bcs + l],
					alpha * b[(j + 2) * bcs + l],
					alpha * b[(j + 3) * bcs + l],
				];
				unsafe {
					axpy4(
						a.as_ptr().add(l * acs),
						t,
						cp.add(j * ccs),
						cp.add((j + 1) * ccs),
						cp.add((j + 2) * ccs),
						cp.add((j + 3) * ccs),
						m,
					);
				}
			}
		}
		j += 4;
	}
	while j < n {
		gemv(alpha, m, k, a, acs, &b[j * bcs..j * bcs + k], beta, &mut c[j * ccs..j * ccs + m]);
		j += 1;
	}
}

/// One 4×4 tile: rows i..i+4 (two f64x2 registers) × columns j..j+4.
/// # Safety
/// `ap` points at A[i, 0] (stride acs, k columns), `bp` at B[0, j]
/// (stride bcs), `cp` at C[i, j] (stride ccs); 4 rows and 4 B/C
/// columns must be in bounds.
#[cfg_attr(target_arch = "wasm32", target_feature(enable = "simd128"))]
#[allow(clippy::too_many_arguments)]
unsafe fn tile_4x4(
	alpha: f64,
	k: usize,
	ap: *const f64,
	acs: usize,
	bp: *const f64,
	bcs: usize,
	beta: f64,
	cp: *mut f64,
	ccs: usize,
) {
	use crate::lanes::F64x2;
	let zero = F64x2::splat(0.0);
	let vb = F64x2::splat(beta);
	let mut acc = [[zero; 2]; 4];
	if beta != 0.0 {
		for (u, au) in acc.iter_mut().enumerate() {
			au[0] = F64x2::load(cp.add(u * ccs)).mul(vb);
			au[1] = F64x2::load(cp.add(u * ccs + 2)).mul(vb);
		}
	}
	for l in 0..k {
		let a0 = F64x2::load(ap.add(l * acs));
		let a1 = F64x2::load(ap.add(l * acs + 2));
		for (u, au) in acc.iter_mut().enumerate() {
			let t = F64x2::splat(alpha * *bp.add(u * bcs + l));
			au[0] = au[0].add(a0.mul(t));
			au[1] = au[1].add(a1.mul(t));
		}
	}
	for (u, au) in acc.iter().enumerate() {
		au[0].store(cp.add(u * ccs));
		au[1].store(cp.add(u * ccs + 2));
	}
}
