//! `cgemm` — complex matrix multiplication: C ← αAB + βC.
//!
//! Implementation: 4-column fused column-caxpy (the dgemm `col4`
//! shape — each A column streams once per group of four C columns,
//! `kernels::caxpy4`), with the plain cgemv-per-column loop kept as
//! `cgemm_colaxpy`, the raced-and-bit-checked reference. Both shapes
//! are bit-for-bit identical, per-element sequence βC then ascending
//! k. The f32 layer's small-size register tile has NO c32 twin yet —
//! a complex tile is a different register geometry (one complex per
//! register), so it's a recorded tuning lever, not a mechanical port;
//! cgemm currently routes everything through col4. Transpose/
//! conjugate forms: not built — no consumer yet (cherk covers A·Aᴴ).

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
