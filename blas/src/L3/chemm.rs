//! `chemm` — Hermitian matrix multiply: C ← αAB + βC (left) or
//! C ← αBA + βC (right), A Hermitian with one triangle stored
//! (diagonal treated as real, per the storage convention).
//!
//! Implementation: the `dsymm` shapes — left side is `chemv` per
//! column of B/C (riding the fused column pass); right side is a
//! 4-column fan-out column-caxpy sweep with the Hermitian element
//! looked up across the stored triangle (`cher_at` — conjugating on
//! the reflected side, real on the diagonal); per-element rounding
//! sequence identical to the plain sweep, bit-for-bit tested.

use super::{check_mat, cher_at};
use crate::c32::C32;
use crate::kernels::caxpy4;
use crate::L1::caxpy;
use crate::L2::{chemv, cscale_y};

/// C ← αAB + βC. A is m×m Hermitian (`upper` triangle stored), B and
/// C are m×n.
#[allow(clippy::too_many_arguments)]
pub fn chemm_left(
	alpha: C32,
	m: usize,
	n: usize,
	a: &[C32],
	acs: usize,
	upper: bool,
	b: &[C32],
	bcs: usize,
	beta: C32,
	c: &mut [C32],
	ccs: usize,
) {
	check_mat(b.len(), m, n, bcs);
	check_mat(c.len(), m, n, ccs);
	for j in 0..n {
		chemv(alpha, m, a, acs, upper, &b[j * bcs..j * bcs + m], beta, &mut c[j * ccs..j * ccs + m]);
	}
}

/// C ← αBA + βC. A is n×n Hermitian (`upper` triangle stored), B and
/// C are m×n.
#[allow(clippy::too_many_arguments)]
pub fn chemm_right(
	alpha: C32,
	m: usize,
	n: usize,
	a: &[C32],
	acs: usize,
	upper: bool,
	b: &[C32],
	bcs: usize,
	beta: C32,
	c: &mut [C32],
	ccs: usize,
) {
	check_mat(a.len(), n, n, acs);
	check_mat(b.len(), m, n, bcs);
	check_mat(c.len(), m, n, ccs);
	let mut j = 0usize;
	while j + 4 <= n {
		let cp = c.as_mut_ptr();
		for u in 0..4 {
			let col = unsafe { core::slice::from_raw_parts_mut(cp.add((j + u) * ccs), m) };
			cscale_y(beta, col);
		}
		for k in 0..n {
			let t = [
				alpha * cher_at(a, acs, upper, k, j),
				alpha * cher_at(a, acs, upper, k, j + 1),
				alpha * cher_at(a, acs, upper, k, j + 2),
				alpha * cher_at(a, acs, upper, k, j + 3),
			];
			unsafe {
				caxpy4(
					b.as_ptr().add(k * bcs),
					t,
					cp.add(j * ccs),
					cp.add((j + 1) * ccs),
					cp.add((j + 2) * ccs),
					cp.add((j + 3) * ccs),
					m,
				);
			}
		}
		j += 4;
	}
	while j < n {
		let cj = &mut c[j * ccs..j * ccs + m];
		cscale_y(beta, cj);
		for k in 0..n {
			caxpy(alpha * cher_at(a, acs, upper, k, j), &b[k * bcs..k * bcs + m], cj);
		}
		j += 1;
	}
}
