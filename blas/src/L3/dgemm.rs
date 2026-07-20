//! `dgemm` — matrix multiplication: C ← αAB + βC.
//!
//! Implementation: size-dispatched family (tuned 2026-07-18, packed
//! shape added 2026-07-20): 4×4 register tile below ~1.5 MB of A,
//! BLIS-style packed-panel shape above (replaced the 4-column fused
//! stream after two unanimous runner draws: 1.24× at 512³, 1.44× at
//! 1024³, 1.10–1.23× deep-K) — all bit-identical to the plain
//! dgemv-per-column reference, which is kept as `dgemm_colaxpy`; col4
//! stays as the raced mid-size reference. The pre-packed shapes beat
//! faer's blocked dgemm at every measured size (1.4–1.8× small, ~1.25×
//! large; docs/blas-ab-2026-07.md step 6). Transpose forms: not built
//! — no consumer yet (dsyrk covers A·Aᵀ).

use super::check_mat;
use crate::kernels::daxpy4;
use crate::L2::dgemv;

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
pub fn dgemm(
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
	// (A = 2 MB) on both reference draws and the container. Above it,
	// the packed-panel shape replaced col4 (packed-gemm race 2026-07-20,
	// two runner draws unanimous: 1.24x at 512³, 1.44x at 1024³,
	// 1.10–1.23x on deep-K prefill shapes; col4 stays as the raced
	// reference). Packed also edged tiled +3% at 256³ — recorded, not
	// routed: the tiled zone below 256³ is unmeasured.
	const TILED_MAX_A_BYTES: usize = 3 << 19; // 1.5 MB
	if m * k * 8 <= TILED_MAX_A_BYTES {
		dgemm_tiled(alpha, m, k, n, a, acs, b, bcs, beta, c, ccs);
	} else {
		dgemm_packed(alpha, m, k, n, a, acs, b, bcs, beta, c, ccs);
	}
}

/// The original column-daxpy shape (dgemv per column) — kept as the
/// plain reference the tuned shapes are raced and bit-checked against.
#[allow(clippy::too_many_arguments)]
pub fn dgemm_colaxpy(
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
		dgemv(alpha, m, k, a, acs, &b[j * bcs..j * bcs + k], beta, &mut c[j * ccs..j * ccs + m]);
	}
}

/// Tuning-campaign candidate (2026-07-18): 4×4 register-tiled dgemm.
/// The column-daxpy dgemm re-reads and re-writes each C element once per
/// k step; this micro-kernel holds a 4-row × 4-column tile of C in 8
/// SIMD registers across the whole k loop — one C load and one C store
/// per tile, 16 FLOPs per 2 A-register loads. Rounding sequence per
/// element is IDENTICAL to the column-daxpy path (β first, then one
/// α·b rounding and one multiply-add per k, ascending), so the two are
/// bit-for-bit interchangeable — tested. Tails (m%4, n%4) fall back to
/// the dgemv path.
#[allow(clippy::too_many_arguments)]
pub fn dgemm_tiled(
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
		// the dgemv path over the remaining rows
		if i < m {
			for jj in j..j + 4 {
				let seg = &mut c[jj * ccs + i..jj * ccs + m];
				crate::L2::dscale_y(beta, seg);
				for l in 0..k {
					crate::L1::daxpy(
						alpha * b[jj * bcs + l],
						&a[l * acs + i..l * acs + m],
						seg,
					);
				}
			}
		}
		j += 4;
	}
	// column tail: plain dgemv columns
	while j < n {
		dgemv(alpha, m, k, a, acs, &b[j * bcs..j * bcs + k], beta, &mut c[j * ccs..j * ccs + m]);
		j += 1;
	}
}

/// Tuning-campaign candidate 2 (2026-07-18): 4-column fused dgemm.
/// The 4×4 tile's k-loop walks A at column stride (TLB/prefetch-hostile
/// at large n — measured losing above n≈512); this shape instead
/// streams each A column SEQUENTIALLY, once per group of four C
/// columns — A traffic drops 4× vs column-daxpy while the four hot C
/// columns ride the near caches. Rounding sequence per element is
/// identical to the column-daxpy path — bit-for-bit interchangeable,
/// tested.
#[allow(clippy::too_many_arguments)]
pub fn dgemm_col4(
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
				crate::L2::dscale_y(beta, col);
			}
			for l in 0..k {
				let t = [
					alpha * b[j * bcs + l],
					alpha * b[(j + 1) * bcs + l],
					alpha * b[(j + 2) * bcs + l],
					alpha * b[(j + 3) * bcs + l],
				];
				unsafe {
					daxpy4(
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
		dgemv(alpha, m, k, a, acs, &b[j * bcs..j * bcs + k], beta, &mut c[j * ccs..j * ccs + m]);
		j += 1;
	}
}

/// Tuning-campaign candidate 3 (2026-07-20): packed-panel dgemm —
/// BLIS/Goto structure around the same 4×4 microkernel math. The tile's
/// weakness is its k-walk over A at column stride (TLB/prefetch-hostile
/// once A outgrows the near caches — why the dispatch abandons it at
/// 1.5 MB); this shape first copies a KC-deep block of B and an MC×KC
/// block of A into contiguous packed panels, so the microkernel streams
/// both operands sequentially regardless of matrix size. Motivating
/// evidence: deep-K prefill shapes (k ≥ 2048), where a packed-panel
/// competitor beat the col4 stream ~2× in a container probe — runner
/// race pending.
///
/// Rounding sequence per element is IDENTICAL to the column-daxpy path:
/// k-blocking resumes each element's accumulation from its stored
/// partial, so the additions happen in the same ascending-k order with
/// the same once-rounded α·b terms, β applied exactly once (first
/// k-block) — bit-for-bit interchangeable, tested. Tails (m%4 rows,
/// n%4 columns) fall back to the same per-column sequences the tile
/// path uses.
#[allow(clippy::too_many_arguments)]
pub fn dgemm_packed(
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
	// Panel sizes: Ap = MC·KC·8 = 256 KB (an L2-resident block), one
	// Bp column-group strip = KC·4·8 = 8 KB (L1-resident). First-guess
	// parameters — the race measures the shape, a later sweep can tune
	// the numbers.
	const KC: usize = 256;
	const MC: usize = 128;
	const MR: usize = 4;
	if k == 0 {
		// degenerate case the packed loops never reach: C ← βC
		for j in 0..n {
			crate::L2::dscale_y(beta, &mut c[j * ccs..j * ccs + m]);
		}
		return;
	}
	let n4 = n - n % 4;
	let m_mr = m - m % MR;
	if n4 > 0 && m_mr > 0 {
		let mut bp = alloc::vec![0.0f64; KC * n4];
		let mut ap = alloc::vec![0.0f64; MC * KC];
		let mut pc = 0usize;
		while pc < k {
			let kc = KC.min(k - pc);
			// pack B[pc.., 0..n4]: per 4-column group, k-major ×4
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
				// pack A[ic..ic+mc, pc..pc+kc]: per MR-row group, k-major ×MR
				for g in 0..mc / MR {
					let base = g * kc * MR;
					for l in 0..kc {
						let src = (pc + l) * acs + ic + g * MR;
						ap[base + l * MR..base + l * MR + MR]
							.copy_from_slice(&a[src..src + MR]);
					}
				}
				// jg outer / row-group inner: Ap rides L2 across column
				// groups, each 8 KB Bp strip rides L1 across row groups
				for jg in 0..n4 / 4 {
					for g in 0..mc / MR {
						unsafe {
							tile_4x4_packed(
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
	// row tail (m_mr..m) × packed columns: per-column daxpy over full k
	if m_mr < m {
		for j in 0..n4 {
			let seg = &mut c[j * ccs + m_mr..j * ccs + m];
			crate::L2::dscale_y(beta, seg);
			for l in 0..k {
				crate::L1::daxpy(alpha * b[j * bcs + l], &a[l * acs + m_mr..l * acs + m], seg);
			}
		}
	}
	// column tail: plain dgemv columns over all rows
	for j in n4..n {
		dgemv(alpha, m, k, a, acs, &b[j * bcs..j * bcs + k], beta, &mut c[j * ccs..j * ccs + m]);
	}
}

/// One 4×4 microkernel step over PACKED panels: `ap` walks MR=4
/// contiguous row values per k step, `bp` walks 4 contiguous column
/// values per k step. Math and per-element rounding order are identical
/// to [`tile_4x4`]; `first` selects β-application (first k-block) vs
/// continuing accumulation from the stored partial.
/// # Safety
/// `ap` must be valid for `kc*4` f64s, `bp` for `kc*4`; `cp` points at
/// C[i, j] (stride ccs) with 4 rows and 4 columns in bounds.
#[cfg_attr(target_arch = "wasm32", target_feature(enable = "simd128"))]
#[allow(clippy::too_many_arguments)]
unsafe fn tile_4x4_packed(
	alpha: f64,
	kc: usize,
	ap: *const f64,
	bp: *const f64,
	beta: f64,
	first: bool,
	cp: *mut f64,
	ccs: usize,
) {
	use crate::lanes::F64x2;
	let zero = F64x2::splat(0.0);
	let mut acc = [[zero; 2]; 4];
	if first {
		if beta != 0.0 {
			let vb = F64x2::splat(beta);
			for (u, au) in acc.iter_mut().enumerate() {
				au[0] = F64x2::load(cp.add(u * ccs)).mul(vb);
				au[1] = F64x2::load(cp.add(u * ccs + 2)).mul(vb);
			}
		}
	} else {
		for (u, au) in acc.iter_mut().enumerate() {
			au[0] = F64x2::load(cp.add(u * ccs));
			au[1] = F64x2::load(cp.add(u * ccs + 2));
		}
	}
	for l in 0..kc {
		let a0 = F64x2::load(ap.add(l * 4));
		let a1 = F64x2::load(ap.add(l * 4 + 2));
		for (u, au) in acc.iter_mut().enumerate() {
			let t = F64x2::splat(alpha * *bp.add(l * 4 + u));
			au[0] = au[0].add(a0.mul(t));
			au[1] = au[1].add(a1.mul(t));
		}
	}
	for (u, au) in acc.iter().enumerate() {
		au[0].store(cp.add(u * ccs));
		au[1].store(cp.add(u * ccs + 2));
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
