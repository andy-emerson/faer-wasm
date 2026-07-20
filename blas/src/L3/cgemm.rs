//! `cgemm` — complex matrix multiplication: C ← αAB + βC.
//!
//! Implementation: size-dispatched (packed shape added 2026-07-20):
//! 4-column fused column-caxpy (`kernels::caxpy4`) below 8 MB of A,
//! BLIS-style packed-panel 4×4 complex register tile at and above —
//! the only zone where the packed shape measured a win (+12% at
//! 1024³, two runner draws unanimous; 256³–512³ and deep-K were
//! noise, so col4 keeps them). The plain cgemv-per-column loop stays
//! as `cgemm_colaxpy`, the raced-and-bit-checked reference; all
//! shapes are bit-for-bit identical, per-element sequence βC then
//! ascending k. Transpose/conjugate forms: not built — no consumer
//! yet (cherk covers A·Aᴴ).

use super::check_mat;
use crate::c32::C32;
use crate::kernels::caxpy4;
use crate::L2::cgemv;

/// C ← αAB + βC. A is m×k, B is k×n, C is m×n; each matrix has its
/// own column stride.
#[allow(clippy::too_many_arguments)]
pub fn cgemm(
	alpha: C32,
	m: usize,
	k: usize,
	n: usize,
	a: &[C32],
	acs: usize,
	b: &[C32],
	bcs: usize,
	beta: C32,
	c: &mut [C32],
	ccs: usize,
) {
	check_mat(a.len(), m, k, acs);
	check_mat(b.len(), k, n, bcs);
	check_mat(c.len(), m, n, ccs);
	// packed-gemm race 2026-07-20 (two runner draws): packed wins only
	// at 1024³ (+12%, A = 8 MB, unanimous); 256³/512³/deep-K measured
	// noise. Routed at the measured win only — the (2 MB, 8 MB) gap is
	// unmeasured and col4 keeps it.
	const PACKED_MIN_A_BYTES: usize = 8 << 20; // 8 MB
	if m * k * 8 >= PACKED_MIN_A_BYTES {
		return cgemm_packed(alpha, m, k, n, a, acs, b, bcs, beta, c, ccs);
	}
	let mut j = 0usize;
	while j + 4 <= n {
		{
			let cp = c.as_mut_ptr();
			for u in 0..4 {
				let col =
					unsafe { core::slice::from_raw_parts_mut(cp.add((j + u) * ccs), m) };
				crate::L2::cscale_y(beta, col);
			}
			for l in 0..k {
				let t = [
					alpha * b[j * bcs + l],
					alpha * b[(j + 1) * bcs + l],
					alpha * b[(j + 2) * bcs + l],
					alpha * b[(j + 3) * bcs + l],
				];
				unsafe {
					caxpy4(
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
		cgemv(alpha, m, k, a, acs, &b[j * bcs..j * bcs + k], beta, &mut c[j * ccs..j * ccs + m]);
		j += 1;
	}
}

/// The plain column-caxpy shape (cgemv per column) — kept as the
/// reference the fused shape is bit-checked against.
#[allow(clippy::too_many_arguments)]
pub fn cgemm_colaxpy(
	alpha: C32,
	m: usize,
	k: usize,
	n: usize,
	a: &[C32],
	acs: usize,
	b: &[C32],
	bcs: usize,
	beta: C32,
	c: &mut [C32],
	ccs: usize,
) {
	check_mat(a.len(), m, k, acs);
	check_mat(b.len(), k, n, bcs);
	check_mat(c.len(), m, n, ccs);
	for j in 0..n {
		cgemv(alpha, m, k, a, acs, &b[j * bcs..j * bcs + k], beta, &mut c[j * ccs..j * ccs + m]);
	}
}

/// Tuning-campaign candidate (2026-07-20): packed-panel cgemm — the
/// `zgemm_packed` structure at c32 pair geometry: 4 complexes (2 F32x4
/// accumulator rows) × 4 columns of C, fed from packed panels. α folds
/// into the B panel at pack time (same single α·b rounding as col4's
/// `t`), stored pre-expanded as [re×4] / [−im,im,−im,im] lanes so the
/// kernel loads its multipliers. Per-element sequence: βC first
/// (shared `cscale_y`), then ascending-k adds of the proven product
/// form `av·vre + swap_pairs(av)·vim` — identical to col4/colaxpy;
/// bit-for-bit interchangeable, tested. Tails (m%4 rows, n%4 columns)
/// fall back to the plain per-column sequences.
#[allow(clippy::too_many_arguments)]
pub fn cgemm_packed(
	alpha: C32,
	m: usize,
	k: usize,
	n: usize,
	a: &[C32],
	acs: usize,
	b: &[C32],
	bcs: usize,
	beta: C32,
	c: &mut [C32],
	ccs: usize,
) {
	check_mat(a.len(), m, k, acs);
	check_mat(b.len(), k, n, bcs);
	check_mat(c.len(), m, n, ccs);
	// Ap = MC·KC complexes = 256 KB (L2-resident); one Bp column-group
	// strip = KC·32 f32 = 32 KB (L1-resident).
	const KC: usize = 256;
	const MC: usize = 128;
	const MR: usize = 4;
	if k == 0 {
		for j in 0..n {
			crate::L2::cscale_y(beta, &mut c[j * ccs..j * ccs + m]);
		}
		return;
	}
	let n4 = n - n % 4;
	let m_mr = m - m % MR;
	for j in 0..n4 {
		crate::L2::cscale_y(beta, &mut c[j * ccs..j * ccs + m]);
	}
	if n4 > 0 && m_mr > 0 {
		// bp holds, per (l, column u): [t.re ×4] then [−t.im, t.im,
		// −t.im, t.im] with t = α·b already rounded — 8 f32 per element
		let mut bp = alloc::vec![0.0f32; KC * n4 * 8];
		let mut ap = alloc::vec![C32::ZERO; MC * KC];
		let mut pc = 0usize;
		while pc < k {
			let kc = KC.min(k - pc);
			for jg in 0..n4 / 4 {
				for l in 0..kc {
					for u in 0..4 {
						let t = alpha * b[(jg * 4 + u) * bcs + pc + l];
						let o = jg * kc * 32 + l * 32 + u * 8;
						bp[o..o + 4].fill(t.re);
						bp[o + 4] = -t.im;
						bp[o + 5] = t.im;
						bp[o + 6] = -t.im;
						bp[o + 7] = t.im;
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
							ctile_4x4_packed(
								kc,
								ap.as_ptr().add(g * kc * MR) as *const f32,
								bp.as_ptr().add(jg * kc * 32),
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
	// row tail (β already applied above): plain caxpy per k step
	if m_mr < m {
		for j in 0..n4 {
			for l in 0..k {
				crate::L1::caxpy(
					alpha * b[j * bcs + l],
					&a[l * acs + m_mr..l * acs + m],
					&mut c[j * ccs + m_mr..j * ccs + m],
				);
			}
		}
	}
	// column tail: plain cgemv columns (handle their own β)
	for j in n4..n {
		cgemv(alpha, m, k, a, acs, &b[j * bcs..j * bcs + k], beta, &mut c[j * ccs..j * ccs + m]);
	}
}

/// One 4×4 complex microkernel step over PACKED panels: 4 complexes
/// (2 F32x4 accumulator rows, 2 complexes per register) × 4 columns.
/// `ap` walks 4 complexes (8 f32) per k step; `bp` walks 4
/// pre-expanded multiplier pairs (32 f32) per k step. Product form and
/// per-element rounding order identical to `caxpy4`
/// (`c + (av·vre + swap_pairs(av)·vim)`).
/// # Safety
/// `ap` valid for `kc*8` f32, `bp` for `kc*32` f32; `cp` points at
/// C[i, j] (stride `ccs` complexes) with 4 rows and 4 columns in
/// bounds.
#[cfg_attr(target_arch = "wasm32", target_feature(enable = "simd128"))]
unsafe fn ctile_4x4_packed(
	kc: usize,
	ap: *const f32,
	bp: *const f32,
	cp: *mut C32,
	ccs: usize,
) {
	use crate::lanes::F32x4;
	let cpf = cp as *mut f32;
	let mut acc = [[F32x4::splat(0.0); 2]; 4];
	for (u, au) in acc.iter_mut().enumerate() {
		au[0] = F32x4::load(cpf.add(u * ccs * 2));
		au[1] = F32x4::load(cpf.add(u * ccs * 2 + 4));
	}
	for l in 0..kc {
		let a0 = F32x4::load(ap.add(l * 8));
		let a1 = F32x4::load(ap.add(l * 8 + 4));
		let a0s = a0.swap_pairs();
		let a1s = a1.swap_pairs();
		for (u, au) in acc.iter_mut().enumerate() {
			let vre = F32x4::load(bp.add(l * 32 + u * 8));
			let vim = F32x4::load(bp.add(l * 32 + u * 8 + 4));
			au[0] = au[0].add(a0.mul(vre).add(a0s.mul(vim)));
			au[1] = au[1].add(a1.mul(vre).add(a1s.mul(vim)));
		}
	}
	for (u, au) in acc.iter().enumerate() {
		au[0].store(cpf.add(u * ccs * 2));
		au[1].store(cpf.add(u * ccs * 2 + 4));
	}
}
