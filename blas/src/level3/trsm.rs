//! `trsm` — triangular solve, many right-hand sides, in place:
//! B ← αA⁻¹B (left) or B ← αBA⁻¹ (right).
//!
//! Implementation (tuned 2026-07-19): left side is divide-then-
//! column-axpy — αscal, then four B columns walk the `trsv`
//! elimination in lockstep so each A column streams once per group of
//! four (A traffic 4× down; the columns are independent, so results
//! are bit-for-bit the plain trsv-per-column loop, tested). Right
//! side groups four destination columns: already-solved columns
//! outside the group fan out via one shared stream each, then the
//! in-group elimination runs in the original order (ascending for
//! upper, descending for lower), including reference `dtrsm`'s
//! multiply-by-reciprocal on the diagonal. For UPPER the per-element
//! add order stays fully ascending — bit-identical to the plain
//! sweep; for LOWER the out-of-group adds now precede the in-group
//! adds — a deterministic, documented reorder (the tests' scalar
//! replay mirrors it). The fused-FMA variant is DEFERRED at the f64
//! campaign close (relaxed-madd rounding is implementation-dependent
//! — architect decision, recorded in ROADMAP). Transpose forms: not
//! built — no consumer yet.

use super::check_mat;
use crate::kernels::axpy4;
use crate::level1::{axpy, scal};
use crate::level2::trsv;

/// B ← αA⁻¹B. A is m×m triangular, B is m×n. No singularity check —
/// a zero diagonal yields inf/NaN, as in reference BLAS.
#[allow(clippy::too_many_arguments)]
pub fn trsm_left(
	alpha: f64,
	m: usize,
	n: usize,
	a: &[f64],
	acs: usize,
	upper: bool,
	unit: bool,
	b: &mut [f64],
	bcs: usize,
) {
	check_mat(a.len(), m, m, acs);
	check_mat(b.len(), m, n, bcs);
	let mut j = 0usize;
	while j + 4 <= n {
		if alpha != 1.0 {
			for u in 0..4 {
				scal(alpha, &mut b[(j + u) * bcs..(j + u) * bcs + m]);
			}
		}
		let bp = b.as_mut_ptr();
		let cols: [*mut f64; 4] = [
			unsafe { bp.add(j * bcs) },
			unsafe { bp.add((j + 1) * bcs) },
			unsafe { bp.add((j + 2) * bcs) },
			unsafe { bp.add((j + 3) * bcs) },
		];
		// the trsv elimination, four right-hand sides in lockstep
		// sharing each A column read
		if upper {
			for l in (0..m).rev() {
				let d = a[l * acs + l];
				let mut t = [0.0f64; 4];
				for (u, cu) in cols.iter().enumerate() {
					unsafe {
						if !unit {
							*cu.add(l) /= d;
						}
						t[u] = -*cu.add(l);
					}
				}
				unsafe {
					axpy4(a.as_ptr().add(l * acs), t, cols[0], cols[1], cols[2], cols[3], l);
				}
			}
		} else {
			for l in 0..m {
				let d = a[l * acs + l];
				let mut t = [0.0f64; 4];
				for (u, cu) in cols.iter().enumerate() {
					unsafe {
						if !unit {
							*cu.add(l) /= d;
						}
						t[u] = -*cu.add(l);
					}
				}
				unsafe {
					axpy4(
						a.as_ptr().add(l * acs + l + 1),
						t,
						cols[0].add(l + 1),
						cols[1].add(l + 1),
						cols[2].add(l + 1),
						cols[3].add(l + 1),
						m - l - 1,
					);
				}
			}
		}
		j += 4;
	}
	while j < n {
		let col = &mut b[j * bcs..j * bcs + m];
		if alpha != 1.0 {
			scal(alpha, col);
		}
		trsv(m, a, acs, upper, unit, col);
		j += 1;
	}
}

/// B ← αBA⁻¹. A is n×n triangular, B is m×n. Reference `dtrsm`
/// multiplies by the diagonal's reciprocal (not a division) — kept.
#[allow(clippy::too_many_arguments)]
pub fn trsm_right(
	alpha: f64,
	m: usize,
	n: usize,
	a: &[f64],
	acs: usize,
	upper: bool,
	unit: bool,
	b: &mut [f64],
	bcs: usize,
) {
	check_mat(a.len(), n, n, acs);
	check_mat(b.len(), m, n, bcs);
	// X·A = αB solved column-wise: X[:,j] = (αB[:,j] − Σ X[:,k]·a[k,j])
	// / a[j,j], where the sum runs over already-solved columns k
	// (k < j for upper, k > j for lower). Groups of four apply the
	// solved columns outside the group first (one shared stream each),
	// then finish the in-group elimination in the original order.
	// Columns never alias.
	let bp = b.as_mut_ptr();
	let col_mut = |idx: usize| unsafe { core::slice::from_raw_parts_mut(bp.add(idx * bcs), m) };
	let col_ref =
		|idx: usize| unsafe { core::slice::from_raw_parts(bp.add(idx * bcs) as *const f64, m) };
	let do_col = |j: usize, lo: usize, hi: usize| {
		if alpha != 1.0 {
			scal(alpha, col_mut(j));
		}
		for k in lo..hi {
			axpy(-a[j * acs + k], col_ref(k), col_mut(j));
		}
		if !unit {
			scal(1.0 / a[j * acs + j], col_mut(j));
		}
	};
	let fan_out = |gs: usize, k: usize| {
		let t = [
			-a[gs * acs + k],
			-a[(gs + 1) * acs + k],
			-a[(gs + 2) * acs + k],
			-a[(gs + 3) * acs + k],
		];
		unsafe {
			axpy4(
				bp.add(k * bcs) as *const f64,
				t,
				bp.add(gs * bcs),
				bp.add((gs + 1) * bcs),
				bp.add((gs + 2) * bcs),
				bp.add((gs + 3) * bcs),
				m,
			);
		}
	};
	if upper {
		let mut gs = 0usize;
		while gs + 4 <= n {
			if alpha != 1.0 {
				for u in 0..4 {
					scal(alpha, col_mut(gs + u));
				}
			}
			// solved columns before the group, one shared stream each —
			// ascending k, so the per-element order stays the plain
			// sweep's (bit-identical for upper)
			for k in 0..gs {
				fan_out(gs, k);
			}
			// in-group elimination, original ascending order
			for tc in gs..gs + 4 {
				for k in gs..tc {
					axpy(-a[tc * acs + k], col_ref(k), col_mut(tc));
				}
				if !unit {
					scal(1.0 / a[tc * acs + tc], col_mut(tc));
				}
			}
			gs += 4;
		}
		for j in gs..n {
			do_col(j, 0, j);
		}
	} else {
		let r = n % 4;
		let mut gs = n;
		while gs >= r + 4 {
			gs -= 4;
			if alpha != 1.0 {
				for u in 0..4 {
					scal(alpha, col_mut(gs + u));
				}
			}
			// solved columns after the group, one shared stream each
			for k in gs + 4..n {
				fan_out(gs, k);
			}
			// in-group elimination, original descending order
			for tc in (gs..gs + 4).rev() {
				for k in tc + 1..gs + 4 {
					axpy(-a[tc * acs + k], col_ref(k), col_mut(tc));
				}
				if !unit {
					scal(1.0 / a[tc * acs + tc], col_mut(tc));
				}
			}
		}
		for j in (0..r).rev() {
			do_col(j, j + 1, n);
		}
	}
}
