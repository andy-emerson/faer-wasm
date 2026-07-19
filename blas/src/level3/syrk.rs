//! `syrk` — symmetric rank-k (Gram-matrix) update:
//! C ← αAAᵀ + βC, one triangle of C stored.
//!
//! Implementation: truncated column-axpy, 4-column fan-out (tuned
//! 2026-07-19) — four stored columns of C are grouped so each inner
//! A column streams once per group over the common triangle segment
//! (source traffic 4× down); the ≤3-row ragged edge per column is
//! handled scalar in the same per-element order, so results stay
//! bit-for-bit identical to the plain sweep (tested). The plain
//! (non-FMA) variant is the measured choice: fusing HARMED syrk on
//! all three step-1 draws.

use super::check_mat;
use crate::kernels::axpy4;
use crate::level1::axpy;
use crate::level2::scale_y;

/// C ← αAAᵀ + βC. A is n×k; C is n×n with the `upper` (or lower)
/// triangle stored.
#[allow(clippy::too_many_arguments)]
pub fn syrk(
	alpha: f64,
	n: usize,
	k: usize,
	a: &[f64],
	acs: usize,
	beta: f64,
	c: &mut [f64],
	ccs: usize,
	upper: bool,
) {
	check_mat(a.len(), n, k, acs);
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
		// intersection of the four stored segments; the rest is the
		// ≤3-row ragged edge per column
		let (clo, chi) = if upper { (0, j + 1) } else { (j + 3, n) };
		for l in 0..k {
			let t = [
				alpha * a[l * acs + j],
				alpha * a[l * acs + j + 1],
				alpha * a[l * acs + j + 2],
				alpha * a[l * acs + j + 3],
			];
			unsafe {
				axpy4(
					a.as_ptr().add(l * acs + clo),
					t,
					cp.add(j * ccs + clo),
					cp.add((j + 1) * ccs + clo),
					cp.add((j + 2) * ccs + clo),
					cp.add((j + 3) * ccs + clo),
					chi - clo,
				);
			}
			for (u, &tu) in t.iter().enumerate() {
				let (lo, hi) = if upper { (chi, j + u + 1) } else { (j + u, clo) };
				for i in lo..hi {
					unsafe {
						*cp.add((j + u) * ccs + i) += a[l * acs + i] * tu;
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
			axpy(alpha * a[l * acs + j], &a[l * acs + lo..l * acs + hi], seg);
		}
		j += 1;
	}
}
