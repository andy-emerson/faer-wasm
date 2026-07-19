//! `zher2k` — Hermitian rank-2k update: C ← αABᴴ + conj(α)BAᴴ + βC,
//! β REAL, one triangle of C stored.
//!
//! Implementation: the `dsyr2k` shape — truncated column-zaxpy,
//! 4-column fan-out over the common triangle segment for both the
//! A-sourced and B-sourced streams, ragged edges scalar in the same
//! per-element order (bit-for-bit tested). Column j's inner-l scalars
//! are α·conj(B[j,l]) (applied to A's column) then conj(α·A[j,l])
//! (applied to B's column); within each inner index the A-sourced add
//! precedes the B-sourced add per element, as in the plain two-stream
//! sweep. Diagonals end exactly real (reference DBLE() convention,
//! the layer's Hermitian invariant).

use super::check_mat;
use crate::c64::C64;
use crate::kernels::zaxpy4;
use crate::L1::zaxpy;
use crate::L2::zdscale_y;

/// C ← αABᴴ + conj(α)BAᴴ + βC with real β. A and B are n×k; C is n×n
/// with the `upper` (or lower) triangle stored.
#[allow(clippy::too_many_arguments)]
pub fn zher2k(
	alpha: C64,
	n: usize,
	k: usize,
	a: &[C64],
	acs: usize,
	b: &[C64],
	bcs: usize,
	beta: f64,
	c: &mut [C64],
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
			zdscale_y(beta, seg);
		}
		let (clo, chi) = if upper { (0, j + 1) } else { (j + 3, n) };
		let ragged = |u: usize| if upper { (chi, j + u + 1) } else { (j + u, clo) };
		for l in 0..k {
			let tb = [
				alpha * b[l * bcs + j].conj(),
				alpha * b[l * bcs + j + 1].conj(),
				alpha * b[l * bcs + j + 2].conj(),
				alpha * b[l * bcs + j + 3].conj(),
			];
			unsafe {
				zaxpy4(
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
						let p = cp.add((j + u) * ccs + i);
						*p = *p + a[l * acs + i] * tbu;
					}
				}
			}
			let ta = [
				(alpha * a[l * acs + j]).conj(),
				(alpha * a[l * acs + j + 1]).conj(),
				(alpha * a[l * acs + j + 2]).conj(),
				(alpha * a[l * acs + j + 3]).conj(),
			];
			unsafe {
				zaxpy4(
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
						let p = cp.add((j + u) * ccs + i);
						*p = *p + b[l * bcs + i] * tau;
					}
				}
			}
		}
		// Hermitian invariant: the four diagonals end exactly real
		for u in 0..4 {
			let d = &mut c[(j + u) * ccs + j + u];
			*d = C64::new(d.re, 0.0);
		}
		j += 4;
	}
	while j < n {
		let (lo, hi) = if upper { (0, j + 1) } else { (j, n) };
		let seg = &mut c[j * ccs + lo..j * ccs + hi];
		zdscale_y(beta, seg);
		for l in 0..k {
			zaxpy(
				alpha * b[l * bcs + j].conj(),
				&a[l * acs + lo..l * acs + hi],
				&mut c[j * ccs + lo..j * ccs + hi],
			);
			zaxpy(
				(alpha * a[l * acs + j]).conj(),
				&b[l * bcs + lo..l * bcs + hi],
				&mut c[j * ccs + lo..j * ccs + hi],
			);
		}
		let d = &mut c[j * ccs + j];
		*d = C64::new(d.re, 0.0);
		j += 1;
	}
}
