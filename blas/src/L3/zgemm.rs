//! `zgemm` — complex matrix multiplication: C ← αAB + βC.
//!
//! Implementation: size-dispatched (packed shape added 2026-07-20):
//! 4-column fused column-zaxpy (`kernels::zaxpy4`) below 1 MB of A,
//! BLIS-style packed-panel 2×4 complex register tile at and above
//! (two runner draws unanimous at every measured size: 1.08–1.15× at
//! 256³ to 1.33× at 1024³). The packed tile is the c64 register
//! geometry the col4-era doc recorded as a non-starter — packing
//! removes the strided k-walk that made it one. The plain
//! zgemv-per-column loop stays as `zgemm_colaxpy`, the raced-and-
//! bit-checked reference; all shapes are bit-for-bit identical,
//! per-element sequence βC then ascending k. Transpose/conjugate
//! forms: not built — no consumer yet (zherk covers A·Aᴴ).

use super::check_mat;
use crate::c64::C64;
use crate::kernels::zaxpy4;
use crate::L2::zgemv;

/// C ← αAB + βC. A is m×k, B is k×n, C is m×n; each matrix has its
/// own column stride.
#[allow(clippy::too_many_arguments)]
pub fn zgemm(
	alpha: C64,
	m: usize,
	k: usize,
	n: usize,
	a: &[C64],
	acs: usize,
	b: &[C64],
	bcs: usize,
	beta: C64,
	c: &mut [C64],
	ccs: usize,
) {
	check_mat(a.len(), m, k, acs);
	check_mat(b.len(), k, n, bcs);
	check_mat(c.len(), m, n, ccs);
	// packed-gemm race 2026-07-20 (two runner draws, unanimous at every
	// measured size): packed wins 1.08–1.15x at 256³ (A = 1 MB, the
	// smallest measured point) up to 1.33x at 1024³. Below 1 MB of A is
	// unmeasured — col4 keeps it.
	const PACKED_MIN_A_BYTES: usize = 1 << 20; // 1 MB
	if m * k * 16 >= PACKED_MIN_A_BYTES {
		return zgemm_packed(alpha, m, k, n, a, acs, b, bcs, beta, c, ccs);
	}
	let mut j = 0usize;
	while j + 4 <= n {
		{
			let cp = c.as_mut_ptr();
			for u in 0..4 {
				let col =
					unsafe { core::slice::from_raw_parts_mut(cp.add((j + u) * ccs), m) };
				crate::L2::zscale_y(beta, col);
			}
			for l in 0..k {
				let t = [
					alpha * b[j * bcs + l],
					alpha * b[(j + 1) * bcs + l],
					alpha * b[(j + 2) * bcs + l],
					alpha * b[(j + 3) * bcs + l],
				];
				unsafe {
					zaxpy4(
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
		zgemv(alpha, m, k, a, acs, &b[j * bcs..j * bcs + k], beta, &mut c[j * ccs..j * ccs + m]);
		j += 1;
	}
}

/// Tuning-campaign candidate (2026-07-20): packed-panel zgemm — the
/// BLIS/Goto structure of `dgemm_packed` at complex register geometry.
/// This is the c64 register tile the col4 module doc recorded as "a
/// different register geometry, not a mechanical port": 2 complexes ×
/// 4 columns of C held in 8 F64x2 accumulators, fed from packed panels
/// so the k-walk is sequential (the strided-A weakness that killed the
/// idea of an unpacked complex tile never arises). α is folded into
/// the B panel at pack time (same single α·b rounding as col4's `t`),
/// stored pre-expanded as [re,re] / [−im,im] lane pairs so the kernel
/// loads its multipliers instead of rebuilding them per step.
///
/// Per-element sequence: βC first (shared `zscale_y`, applied to whole
/// columns up front), then ascending-k adds of the proven bit-exact
/// product form `av·vre + swap(av)·vim` — identical to col4/colaxpy;
/// bit-for-bit interchangeable, tested. Tails (m%2 rows, n%4 columns)
/// fall back to the plain per-column sequences.
#[allow(clippy::too_many_arguments)]
pub fn zgemm_packed(
	alpha: C64,
	m: usize,
	k: usize,
	n: usize,
	a: &[C64],
	acs: usize,
	b: &[C64],
	bcs: usize,
	beta: C64,
	c: &mut [C64],
	ccs: usize,
) {
	check_mat(a.len(), m, k, acs);
	check_mat(b.len(), k, n, bcs);
	check_mat(c.len(), m, n, ccs);
	// Ap = MC·KC complexes = 256 KB (L2-resident); one Bp column-group
	// strip = KC·16 f64 = 32 KB (L1-resident).
	const KC: usize = 256;
	const MC: usize = 64;
	const MR: usize = 2;
	if k == 0 {
		for j in 0..n {
			crate::L2::zscale_y(beta, &mut c[j * ccs..j * ccs + m]);
		}
		return;
	}
	let n4 = n - n % 4;
	let m_mr = m - m % MR;
	// β once per element, before any accumulation — same order as the
	// per-column passes
	for j in 0..n4 {
		crate::L2::zscale_y(beta, &mut c[j * ccs..j * ccs + m]);
	}
	if n4 > 0 && m_mr > 0 {
		// bp holds, per (l, column u): [t.re, t.re, −t.im, t.im] with
		// t = α·b already rounded — 4 f64 per element
		let mut bp = alloc::vec![0.0f64; KC * n4 * 4];
		let mut ap = alloc::vec![C64::ZERO; MC * KC];
		let mut pc = 0usize;
		while pc < k {
			let kc = KC.min(k - pc);
			for jg in 0..n4 / 4 {
				for l in 0..kc {
					for u in 0..4 {
						let t = alpha * b[(jg * 4 + u) * bcs + pc + l];
						let o = jg * kc * 16 + l * 16 + u * 4;
						bp[o] = t.re;
						bp[o + 1] = t.re;
						bp[o + 2] = -t.im;
						bp[o + 3] = t.im;
					}
				}
			}
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
							ztile_2x4_packed(
								kc,
								ap.as_ptr().add(g * kc * MR) as *const f64,
								bp.as_ptr().add(jg * kc * 16),
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
	// row tail (β already applied above): plain zaxpy per k step
	if m_mr < m {
		for j in 0..n4 {
			for l in 0..k {
				crate::L1::zaxpy(
					alpha * b[j * bcs + l],
					&a[l * acs + m_mr..l * acs + m],
					&mut c[j * ccs + m_mr..j * ccs + m],
				);
			}
		}
	}
	// column tail: plain zgemv columns (handle their own β)
	for j in n4..n {
		zgemv(alpha, m, k, a, acs, &b[j * bcs..j * bcs + k], beta, &mut c[j * ccs..j * ccs + m]);
	}
}

/// One 2×4 complex microkernel step over PACKED panels: 2 complexes
/// (2 F64x2 accumulator rows) × 4 columns. `ap` walks 2 complexes
/// (4 f64) per k step; `bp` walks 4 pre-expanded multiplier pairs
/// (16 f64) per k step. Product form and per-element rounding order
/// identical to `zaxpy4` (`c + (av·vre + swap(av)·vim)`).
/// # Safety
/// `ap` valid for `kc*4` f64, `bp` for `kc*16` f64; `cp` points at
/// C[i, j] (stride `ccs` complexes) with 2 rows and 4 columns in
/// bounds.
#[cfg_attr(target_arch = "wasm32", target_feature(enable = "simd128"))]
unsafe fn ztile_2x4_packed(
	kc: usize,
	ap: *const f64,
	bp: *const f64,
	cp: *mut C64,
	ccs: usize,
) {
	use crate::lanes::F64x2;
	let cpf = cp as *mut f64;
	let mut acc = [[F64x2::splat(0.0); 2]; 4];
	for (u, au) in acc.iter_mut().enumerate() {
		au[0] = F64x2::load(cpf.add(u * ccs * 2));
		au[1] = F64x2::load(cpf.add(u * ccs * 2 + 2));
	}
	for l in 0..kc {
		let a0 = F64x2::load(ap.add(l * 4));
		let a1 = F64x2::load(ap.add(l * 4 + 2));
		let a0s = a0.swap();
		let a1s = a1.swap();
		for (u, au) in acc.iter_mut().enumerate() {
			let vre = F64x2::load(bp.add(l * 16 + u * 4));
			let vim = F64x2::load(bp.add(l * 16 + u * 4 + 2));
			au[0] = au[0].add(a0.mul(vre).add(a0s.mul(vim)));
			au[1] = au[1].add(a1.mul(vre).add(a1s.mul(vim)));
		}
	}
	for (u, au) in acc.iter().enumerate() {
		au[0].store(cpf.add(u * ccs * 2));
		au[1].store(cpf.add(u * ccs * 2 + 2));
	}
}

/// The plain column-zaxpy shape (zgemv per column) — kept as the
/// reference the fused shape is bit-checked against.
#[allow(clippy::too_many_arguments)]
pub fn zgemm_colaxpy(
	alpha: C64,
	m: usize,
	k: usize,
	n: usize,
	a: &[C64],
	acs: usize,
	b: &[C64],
	bcs: usize,
	beta: C64,
	c: &mut [C64],
	ccs: usize,
) {
	check_mat(a.len(), m, k, acs);
	check_mat(b.len(), k, n, bcs);
	check_mat(c.len(), m, n, ccs);
	for j in 0..n {
		zgemv(alpha, m, k, a, acs, &b[j * bcs..j * bcs + k], beta, &mut c[j * ccs..j * ccs + m]);
	}
}
