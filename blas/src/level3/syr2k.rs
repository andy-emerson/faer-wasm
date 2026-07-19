//! `syr2k` — symmetric rank-2k update: C ← α(ABᵀ + BAᵀ) + βC, one
//! triangle of C stored.
//!
//! Implementation: truncated column-axpy, 4-column fan-out (tuned
//! 2026-07-19, same shape as syrk) — four stored columns grouped so
//! each inner A and B column streams once per group over the common
//! triangle segment; ragged edges scalar in the same per-element
//! order (bit-for-bit tested). Within each inner index the A-sourced
//! add still precedes the B-sourced add per element, as in the plain
//! two-stream sweep.

use super::check_mat;
use crate::kernels::axpy4;
use crate::level1::axpy;
use crate::level2::scale_y;

/// C ← α(ABᵀ + BAᵀ) + βC. A and B are n×k; C is n×n with the `upper`
/// (or lower) triangle stored.
#[allow(clippy::too_many_arguments)]
pub fn syr2k(
	alpha: f64,
	n: usize,
	k: usize,
	a: &[f64],
	acs: usize,
	b: &[f64],
	bcs: usize,
	beta: f64,
	c: &mut [f64],
	ccs: usize,
	upper: bool,
) {
	check_mat(a.len(), n, k, acs);
	check_mat(b.len(), n, k, bcs);
	check_mat(c.len(), n, n, ccs);
	let mut j = 0usize;
	while j + 4 <= n {
		let cp = c.as_mut_ptr();
		for u in 0..4 {
			let (lo, hi) = if upper { (0, j + u + 1) } else { (j + u, n) };
			let seg =
				unsafe { core::slice::from_raw_parts_mut(cp.add((j + u) * ccs + lo), hi - lo) };
			scale_y(beta, seg);
		}
		let (clo, chi) = if upper { (0, j + 1) } else { (j + 3, n) };
		let ragged =
			|u: usize| if upper { (chi, j + u + 1) } else { (j + u, clo) };
		for l in 0..k {
			let tb = [
				alpha * b[l * bcs + j],
				alpha * b[l * bcs + j + 1],
				alpha * b[l * bcs + j + 2],
				alpha * b[l * bcs + j + 3],
			];
			unsafe {
				axpy4(
					a.as_ptr().add(l * acs + clo),
					tb,
					cp.add(j * ccs + clo),
					cp.add((j + 1) * ccs + clo),
					cp.add((j + 2) * ccs + clo),
					cp.add((j + 3) * ccs + clo),
					chi - clo,
				);
			}
			for (u, &tbu) in tb.iter().enumerate() {
				let (lo, hi) = ragged(u);
				for i in lo..hi {
					unsafe {
						*cp.add((j + u) * ccs + i) += a[l * acs + i] * tbu;
					}
				}
			}
			let ta = [
				alpha * a[l * acs + j],
				alpha * a[l * acs + j + 1],
				alpha * a[l * acs + j + 2],
				alpha * a[l * acs + j + 3],
			];
			unsafe {
				axpy4(
					b.as_ptr().add(l * bcs + clo),
					ta,
					cp.add(j * ccs + clo),
					cp.add((j + 1) * ccs + clo),
					cp.add((j + 2) * ccs + clo),
					cp.add((j + 3) * ccs + clo),
					chi - clo,
				);
			}
			for (u, &tau) in ta.iter().enumerate() {
				let (lo, hi) = ragged(u);
				for i in lo..hi {
					unsafe {
						*cp.add((j + u) * ccs + i) += b[l * bcs + i] * tau;
					}
				}
			}
		}
		j += 4;
	}
	while j < n {
		let (lo, hi) = if upper { (0, j + 1) } else { (j, n) };
		let seg = &mut c[j * ccs + lo..j * ccs + hi];
		scale_y(beta, seg);
		for l in 0..k {
			axpy(alpha * b[l * bcs + j], &a[l * acs + lo..l * acs + hi], seg);
			axpy(alpha * a[l * acs + j], &b[l * bcs + lo..l * bcs + hi], seg);
		}
		j += 1;
	}
}
