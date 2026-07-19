//! `trmm` — triangular matrix multiply, in place:
//! B ← αAB (left) or B ← αBA (right), A triangular.
//!
//! Implementation (tuned 2026-07-19): left side walks four B columns
//! in lockstep through the `trmv` column sweep, so each A column
//! streams once per group of four instead of once per column (A
//! traffic 4× down; per-column rounding sequence unchanged — the
//! columns are independent — so results are bit-for-bit the plain
//! trmv-per-column loop, tested). Right side groups four destination
//! columns: contributions from inside the group run first in the
//! original elimination order (descending for upper, ascending for
//! lower — each original column consumed before it is overwritten),
//! then the columns outside the group fan out via one stream each.
//! For LOWER the resulting per-element add order is identical to the
//! plain sweep (ascending k throughout); for UPPER the in-group adds
//! now precede the out-of-group adds — a deterministic, documented
//! reorder (the tests' scalar replay mirrors it). A fused-FMA variant
//! measured better in the step-1 race but is DEFERRED at the f64
//! campaign close: wasm relaxed-madd rounding is implementation-
//! dependent, so it trades away cross-target bit-identity — an
//! architect decision, recorded in ROADMAP. Transpose forms: not
//! built — no consumer yet.

use super::check_mat;
use crate::kernels::axpy4;
use crate::level1::{axpy, scal};
use crate::level2::trmv;

/// B ← αAB. A is m×m triangular, B is m×n.
#[allow(clippy::too_many_arguments)]
pub fn trmm_left(
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
		let bp = b.as_mut_ptr();
		let cols: [*mut f64; 4] = [
			unsafe { bp.add(j * bcs) },
			unsafe { bp.add((j + 1) * bcs) },
			unsafe { bp.add((j + 2) * bcs) },
			unsafe { bp.add((j + 3) * bcs) },
		];
		// the trmv column sweep, four x vectors in lockstep sharing
		// each A column read
		if upper {
			for l in 0..m {
				let t = unsafe { [*cols[0].add(l), *cols[1].add(l), *cols[2].add(l), *cols[3].add(l)] };
				unsafe {
					axpy4(a.as_ptr().add(l * acs), t, cols[0], cols[1], cols[2], cols[3], l);
				}
				if !unit {
					let d = a[l * acs + l];
					for (u, cu) in cols.iter().enumerate() {
						unsafe { *cu.add(l) = t[u] * d };
					}
				}
			}
		} else {
			for l in (0..m).rev() {
				let t = unsafe { [*cols[0].add(l), *cols[1].add(l), *cols[2].add(l), *cols[3].add(l)] };
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
				if !unit {
					let d = a[l * acs + l];
					for (u, cu) in cols.iter().enumerate() {
						unsafe { *cu.add(l) = t[u] * d };
					}
				}
			}
		}
		if alpha != 1.0 {
			for u in 0..4 {
				scal(alpha, &mut b[(j + u) * bcs..(j + u) * bcs + m]);
			}
		}
		j += 4;
	}
	while j < n {
		let col = &mut b[j * bcs..j * bcs + m];
		trmv(m, a, acs, upper, unit, col);
		if alpha != 1.0 {
			scal(alpha, col);
		}
		j += 1;
	}
}

/// B ← αBA. A is n×n triangular, B is m×n.
#[allow(clippy::too_many_arguments)]
pub fn trmm_right(
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
	// result column j = α·Σ_{k in triangle} a[k,j]·B_orig[:,k]. The
	// column order (descending for upper, ascending for lower) touches
	// each original column before anything overwrites it; groups of
	// four keep that order internally, then share one stream per
	// out-of-group source column. Columns j and k never alias
	// (k ≠ j always), so the views below are disjoint — raw pointers
	// express what split_at_mut can't say cleanly for arbitrary
	// (j, k) pairs.
	let bp = b.as_mut_ptr();
	let col_mut = |idx: usize| unsafe { core::slice::from_raw_parts_mut(bp.add(idx * bcs), m) };
	let col_ref =
		|idx: usize| unsafe { core::slice::from_raw_parts(bp.add(idx * bcs) as *const f64, m) };
	let do_col = |j: usize| {
		let diag = if unit { 1.0 } else { a[j * acs + j] };
		scal(alpha * diag, col_mut(j));
		let (lo, hi) = if upper { (0, j) } else { (j + 1, n) };
		for k in lo..hi {
			axpy(alpha * a[j * acs + k], col_ref(k), col_mut(j));
		}
	};
	let fan_out = |gs: usize, k: usize| {
		let t = [
			alpha * a[gs * acs + k],
			alpha * a[(gs + 1) * acs + k],
			alpha * a[(gs + 2) * acs + k],
			alpha * a[(gs + 3) * acs + k],
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
		let r = n % 4;
		let mut gs = n;
		while gs >= r + 4 {
			gs -= 4;
			// in-group first, original descending order (sources still
			// original when read)
			for tc in (gs..gs + 4).rev() {
				let diag = if unit { 1.0 } else { a[tc * acs + tc] };
				scal(alpha * diag, col_mut(tc));
				for k in gs..tc {
					axpy(alpha * a[tc * acs + k], col_ref(k), col_mut(tc));
				}
			}
			// out-of-group sources, one shared stream each
			for k in 0..gs {
				fan_out(gs, k);
			}
		}
		for j in (0..r).rev() {
			do_col(j);
		}
	} else {
		let mut gs = 0usize;
		while gs + 4 <= n {
			// in-group ascending: per-element add order stays fully
			// ascending — bit-identical to the plain sweep for lower
			for tc in gs..gs + 4 {
				let diag = if unit { 1.0 } else { a[tc * acs + tc] };
				scal(alpha * diag, col_mut(tc));
				for k in tc + 1..gs + 4 {
					axpy(alpha * a[tc * acs + k], col_ref(k), col_mut(tc));
				}
			}
			for k in gs + 4..n {
				fan_out(gs, k);
			}
			gs += 4;
		}
		for j in gs..n {
			do_col(j);
		}
	}
}
