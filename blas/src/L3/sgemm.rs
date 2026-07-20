//! `sgemm` — matrix multiplication: C ← αAB + βC.
//!
//! Implementation: size-dispatched family (tuned 2026-07-18, packed
//! shape added 2026-07-20): 8×4 register tile below ~3 MB of A,
//! BLIS-style packed-panel shape above (replaced the 4-column fused
//! stream: +3–5% at 1024³ and 512×4096×1024, two runner draws
//! unanimous; the tile keeps everything below, where packed measured
//! noise to −2%) — all bit-identical to the plain sgemv-per-column
//! reference, which is kept as `sgemm_colaxpy`; col4 stays as the
//! raced reference. f32 measurements: docs/blas-ab-2026-07.md steps
//! 10 and 14. Transpose forms: not built — no consumer yet (ssyrk
//! covers A·Aᵀ).

use super::check_mat;
use crate::kernels::saxpy4;
use crate::L2::sgemv;

/// C ← αAB + βC. A is m×k, B is k×n, C is m×n; each matrix has its own
/// column stride.
///
/// Dispatches by A's byte size at the runner-raced f32 crossover (the
/// register tile wins while A rides the caches, the 4-column fused
/// stream wins above; two draws, docs step 10). All three shapes are
/// bit-for-bit identical, so the dispatch is invisible to results.
#[allow(clippy::too_many_arguments)]
pub fn sgemm(
	alpha: f32,
	m: usize,
	k: usize,
	n: usize,
	a: &[f32],
	acs: usize,
	b: &[f32],
	bcs: usize,
	beta: f32,
	c: &mut [f32],
	ccs: usize,
) {
	// f32 crossover from the step-10 runner draws (which rule over the
	// container, where col4 led from n=512): tiled wins unanimously
	// through n=512, n=768 splits within noise — the 8-row tile
	// survives LONGER in bytes than the f64 4-row tile's 1.5 MB.
	// Above the threshold, the packed-panel shape replaced col4
	// (packed-gemm race 2026-07-20, two runner draws: +3–4% at 1024³,
	// +5% at 512x4096x1024, unanimous; 512³/256³ stay tiled, where
	// packed measured noise/−2%). col4 stays as the raced reference.
	const TILED_MAX_A_BYTES: usize = 3 << 20; // 3 MB
	if m * k * 4 <= TILED_MAX_A_BYTES {
		sgemm_tiled(alpha, m, k, n, a, acs, b, bcs, beta, c, ccs);
	} else {
		sgemm_packed(alpha, m, k, n, a, acs, b, bcs, beta, c, ccs);
	}
}

/// The original column-saxpy shape (sgemv per column) — kept as the
/// plain reference the tuned shapes are raced and bit-checked against.
#[allow(clippy::too_many_arguments)]
pub fn sgemm_colaxpy(
	alpha: f32,
	m: usize,
	k: usize,
	n: usize,
	a: &[f32],
	acs: usize,
	b: &[f32],
	bcs: usize,
	beta: f32,
	c: &mut [f32],
	ccs: usize,
) {
	check_mat(a.len(), m, k, acs);
	check_mat(b.len(), k, n, bcs);
	check_mat(c.len(), m, n, ccs);
	for j in 0..n {
		sgemv(alpha, m, k, a, acs, &b[j * bcs..j * bcs + k], beta, &mut c[j * ccs..j * ccs + m]);
	}
}

/// Tuning-campaign shape (ported from f64): 8×4 register-tiled sgemm.
/// The column-saxpy sgemm re-reads and re-writes each C element once per
/// k step; this micro-kernel holds an 8-row × 4-column tile of C in 8
/// SIMD registers across the whole k loop — one C load and one C store
/// per tile, 32 FLOPs per 2 A-register loads. Rounding sequence per
/// element is IDENTICAL to the column-saxpy path (β first, then one
/// α·b rounding and one multiply-add per k, ascending), so the two are
/// bit-for-bit interchangeable — tested. Tails (m%8, n%4) fall back to
/// the sgemv path.
#[allow(clippy::too_many_arguments)]
pub fn sgemm_tiled(
	alpha: f32,
	m: usize,
	k: usize,
	n: usize,
	a: &[f32],
	acs: usize,
	b: &[f32],
	bcs: usize,
	beta: f32,
	c: &mut [f32],
	ccs: usize,
) {
	check_mat(a.len(), m, k, acs);
	check_mat(b.len(), k, n, bcs);
	check_mat(c.len(), m, n, ccs);
	let mut j = 0usize;
	while j + 4 <= n {
		let mut i = 0usize;
		while i + 8 <= m {
			unsafe {
				tile_8x4(
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
			i += 8;
		}
		// row tail for these four columns: same per-element sequence as
		// the sgemv path over the remaining rows
		if i < m {
			for jj in j..j + 4 {
				let seg = &mut c[jj * ccs + i..jj * ccs + m];
				crate::L2::sscale_y(beta, seg);
				for l in 0..k {
					crate::L1::saxpy(
						alpha * b[jj * bcs + l],
						&a[l * acs + i..l * acs + m],
						seg,
					);
				}
			}
		}
		j += 4;
	}
	// column tail: plain sgemv columns
	while j < n {
		sgemv(alpha, m, k, a, acs, &b[j * bcs..j * bcs + k], beta, &mut c[j * ccs..j * ccs + m]);
		j += 1;
	}
}

/// Tuning-campaign candidate 2 (2026-07-18): 4-column fused sgemm.
/// The 4×4 tile's k-loop walks A at column stride (TLB/prefetch-hostile
/// at large n — measured losing above n≈512); this shape instead
/// streams each A column SEQUENTIALLY, once per group of four C
/// columns — A traffic drops 4× vs column-saxpy while the four hot C
/// columns ride the near caches. Rounding sequence per element is
/// identical to the column-saxpy path — bit-for-bit interchangeable,
/// tested.
#[allow(clippy::too_many_arguments)]
pub fn sgemm_col4(
	alpha: f32,
	m: usize,
	k: usize,
	n: usize,
	a: &[f32],
	acs: usize,
	b: &[f32],
	bcs: usize,
	beta: f32,
	c: &mut [f32],
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
				crate::L2::sscale_y(beta, col);
			}
			for l in 0..k {
				let t = [
					alpha * b[j * bcs + l],
					alpha * b[(j + 1) * bcs + l],
					alpha * b[(j + 2) * bcs + l],
					alpha * b[(j + 3) * bcs + l],
				];
				unsafe {
					saxpy4(
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
		sgemv(alpha, m, k, a, acs, &b[j * bcs..j * bcs + k], beta, &mut c[j * ccs..j * ccs + m]);
		j += 1;
	}
}

/// Tuning-campaign candidate 3 (2026-07-20): packed-panel sgemm —
/// BLIS/Goto structure around the same 8×4 microkernel math; see
/// `dgemm_packed` for the full rationale (deep-K motivation, cache
/// roles of the two panels, bit-compat argument). Rounding sequence
/// per element is IDENTICAL to the column-saxpy path — k-blocking
/// resumes each element's accumulation from its stored partial —
/// bit-for-bit interchangeable, tested.
#[allow(clippy::too_many_arguments)]
pub fn sgemm_packed(
	alpha: f32,
	m: usize,
	k: usize,
	n: usize,
	a: &[f32],
	acs: usize,
	b: &[f32],
	bcs: usize,
	beta: f32,
	c: &mut [f32],
	ccs: usize,
) {
	check_mat(a.len(), m, k, acs);
	check_mat(b.len(), k, n, bcs);
	check_mat(c.len(), m, n, ccs);
	// Ap = MC·KC·4 = 128 KB (L2-resident), one Bp strip = KC·4·4 = 4 KB
	const KC: usize = 256;
	const MC: usize = 128;
	const MR: usize = 8;
	if k == 0 {
		for j in 0..n {
			crate::L2::sscale_y(beta, &mut c[j * ccs..j * ccs + m]);
		}
		return;
	}
	let n4 = n - n % 4;
	let m_mr = m - m % MR;
	if n4 > 0 && m_mr > 0 {
		let mut bp = alloc::vec![0.0f32; KC * n4];
		let mut ap = alloc::vec![0.0f32; MC * KC];
		let mut pc = 0usize;
		while pc < k {
			let kc = KC.min(k - pc);
			for jg in 0..n4 / 4 {
				let dst = &mut bp[jg * kc * 4..(jg + 1) * kc * 4];
				let j = jg * 4;
				for l in 0..kc {
					for u in 0..4 {
						dst[l * 4 + u] = b[(j + u) * bcs + pc + l];
					}
				}
			}
			let first = pc == 0;
			let mut ic = 0usize;
			while ic < m_mr {
				let mc = MC.min(m_mr - ic); // both multiples of MR
				for g in 0..mc / MR {
					let base = g * kc * MR;
					for l in 0..kc {
						let src = (pc + l) * acs + ic + g * MR;
						ap[base + l * MR..base + l * MR + MR]
							.copy_from_slice(&a[src..src + MR]);
					}
				}
				for jg in 0..n4 / 4 {
					for g in 0..mc / MR {
						unsafe {
							tile_8x4_packed(
								alpha,
								kc,
								ap.as_ptr().add(g * kc * MR),
								bp.as_ptr().add(jg * kc * 4),
								beta,
								first,
								c.as_mut_ptr().add(jg * 4 * ccs + ic + g * MR),
								ccs,
							);
						}
					}
				}
				ic += mc;
			}
			pc += kc;
		}
	}
	if m_mr < m {
		for j in 0..n4 {
			let seg = &mut c[j * ccs + m_mr..j * ccs + m];
			crate::L2::sscale_y(beta, seg);
			for l in 0..k {
				crate::L1::saxpy(alpha * b[j * bcs + l], &a[l * acs + m_mr..l * acs + m], seg);
			}
		}
	}
	for j in n4..n {
		sgemv(alpha, m, k, a, acs, &b[j * bcs..j * bcs + k], beta, &mut c[j * ccs..j * ccs + m]);
	}
}

/// One 8×4 microkernel step over PACKED panels: `ap` walks MR=8
/// contiguous row values per k step, `bp` walks 4 contiguous column
/// values per k step. Math and per-element rounding order identical to
/// [`tile_8x4`]; `first` selects β-application vs continuing from the
/// stored partial.
/// # Safety
/// `ap` must be valid for `kc*8` f32s, `bp` for `kc*4`; `cp` points at
/// C[i, j] (stride ccs) with 8 rows and 4 columns in bounds.
#[cfg_attr(target_arch = "wasm32", target_feature(enable = "simd128"))]
#[allow(clippy::too_many_arguments)]
unsafe fn tile_8x4_packed(
	alpha: f32,
	kc: usize,
	ap: *const f32,
	bp: *const f32,
	beta: f32,
	first: bool,
	cp: *mut f32,
	ccs: usize,
) {
	use crate::lanes::F32x4;
	let zero = F32x4::splat(0.0);
	let mut acc = [[zero; 2]; 4];
	if first {
		if beta != 0.0 {
			let vb = F32x4::splat(beta);
			for (u, au) in acc.iter_mut().enumerate() {
				au[0] = F32x4::load(cp.add(u * ccs)).mul(vb);
				au[1] = F32x4::load(cp.add(u * ccs + 4)).mul(vb);
			}
		}
	} else {
		for (u, au) in acc.iter_mut().enumerate() {
			au[0] = F32x4::load(cp.add(u * ccs));
			au[1] = F32x4::load(cp.add(u * ccs + 4));
		}
	}
	for l in 0..kc {
		let a0 = F32x4::load(ap.add(l * 8));
		let a1 = F32x4::load(ap.add(l * 8 + 4));
		for (u, au) in acc.iter_mut().enumerate() {
			let t = F32x4::splat(alpha * *bp.add(l * 4 + u));
			au[0] = au[0].add(a0.mul(t));
			au[1] = au[1].add(a1.mul(t));
		}
	}
	for (u, au) in acc.iter().enumerate() {
		au[0].store(cp.add(u * ccs));
		au[1].store(cp.add(u * ccs + 4));
	}
}

/// One 8×4 tile: rows i..i+8 (two f32x4 registers) × columns j..j+4 —
/// the f64 4×4 tile with the same register count at twice the lane
/// width.
/// # Safety
/// `ap` points at A[i, 0] (stride acs, k columns), `bp` at B[0, j]
/// (stride bcs), `cp` at C[i, j] (stride ccs); 8 rows and 4 B/C
/// columns must be in bounds.
#[cfg_attr(target_arch = "wasm32", target_feature(enable = "simd128"))]
#[allow(clippy::too_many_arguments)]
unsafe fn tile_8x4(
	alpha: f32,
	k: usize,
	ap: *const f32,
	acs: usize,
	bp: *const f32,
	bcs: usize,
	beta: f32,
	cp: *mut f32,
	ccs: usize,
) {
	use crate::lanes::F32x4;
	let zero = F32x4::splat(0.0);
	let vb = F32x4::splat(beta);
	let mut acc = [[zero; 2]; 4];
	if beta != 0.0 {
		for (u, au) in acc.iter_mut().enumerate() {
			au[0] = F32x4::load(cp.add(u * ccs)).mul(vb);
			au[1] = F32x4::load(cp.add(u * ccs + 4)).mul(vb);
		}
	}
	for l in 0..k {
		let a0 = F32x4::load(ap.add(l * acs));
		let a1 = F32x4::load(ap.add(l * acs + 4));
		for (u, au) in acc.iter_mut().enumerate() {
			let t = F32x4::splat(alpha * *bp.add(u * bcs + l));
			au[0] = au[0].add(a0.mul(t));
			au[1] = au[1].add(a1.mul(t));
		}
	}
	for (u, au) in acc.iter().enumerate() {
		au[0].store(cp.add(u * ccs));
		au[1].store(cp.add(u * ccs + 4));
	}
}
