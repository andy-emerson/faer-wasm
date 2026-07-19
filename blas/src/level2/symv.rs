//! `symv` — symmetric matrix × vector: y ← αAx + βy, A symmetric with
//! only one triangle stored.
//!
//! Implementation: fused 4-column pass (tuned 2026-07-19) — four
//! stored columns at a time, ONE stream over x and y serves both
//! triangles' contributions for all four (`kernels::axpy_dot4`; the
//! ≤3-row ragged edge and diagonals scalar per column), where the
//! plain shape streamed each column twice (`axpy` + `dot`). symv has
//! no cross-column dependencies (y is only ever accumulated into, x
//! and A only read), so the grouping is free. Tail columns use the
//! single-column fused pass (`kernels::axpy_dot`). Accumulation
//! order differs from reference `dsymv` (symv is bounds-tested, not
//! bit-locked; cross-target determinism holds through the lane
//! emulation as everywhere else).

use super::{check_mat, scale_y};
use crate::kernels::{axpy_dot, axpy_dot4};

/// y ← αAx + βy, A symmetric n×n at column stride `cs`, with the
/// `upper` (or lower) triangle stored.
#[allow(clippy::too_many_arguments)]
pub fn symv(
	alpha: f64,
	n: usize,
	a: &[f64],
	cs: usize,
	upper: bool,
	x: &[f64],
	beta: f64,
	y: &mut [f64],
) {
	check_mat(a.len(), n, n, cs);
	assert_eq!(x.len(), n, "symv: x length mismatch");
	assert_eq!(y.len(), n, "symv: y length mismatch");
	scale_y(beta, y);
	let ap = a.as_ptr();
	let mut j = 0usize;
	while j + 4 <= n {
		let t = [alpha * x[j], alpha * x[j + 1], alpha * x[j + 2], alpha * x[j + 3]];
		let cols = unsafe {
			[ap.add(j * cs), ap.add((j + 1) * cs), ap.add((j + 2) * cs), ap.add((j + 3) * cs)]
		};
		if upper {
			// common strict segment rows 0..j fused for all four; then
			// the ragged rows j..j+u and the diagonal, scalar per column
			let mut d =
				unsafe { axpy_dot4(cols, t, x.as_ptr(), y.as_mut_ptr(), j) };
			for u in 0..4 {
				let cj = (j + u) * cs;
				for i in j..j + u {
					y[i] += t[u] * a[cj + i];
					d[u] += a[cj + i] * x[i];
				}
				y[j + u] += t[u] * a[cj + j + u] + alpha * d[u];
			}
		} else {
			// common strict segment rows j+4..n; ragged rows j+u+1..j+4
			let mut d = unsafe {
				axpy_dot4(
					[cols[0].add(j + 4), cols[1].add(j + 4), cols[2].add(j + 4), cols[3].add(j + 4)],
					t,
					x.as_ptr().add(j + 4),
					y.as_mut_ptr().add(j + 4),
					n - j - 4,
				)
			};
			for u in 0..4 {
				let cj = (j + u) * cs;
				for i in j + u + 1..j + 4 {
					y[i] += t[u] * a[cj + i];
					d[u] += a[cj + i] * x[i];
				}
				y[j + u] += t[u] * a[cj + j + u] + alpha * d[u];
			}
		}
		j += 4;
	}
	while j < n {
		let col = &a[j * cs..j * cs + n];
		let t = alpha * x[j];
		if upper {
			let d = unsafe { axpy_dot(col.as_ptr(), t, x.as_ptr(), y.as_mut_ptr(), j) };
			y[j] += t * col[j] + alpha * d;
		} else {
			let d = unsafe {
				axpy_dot(
					col.as_ptr().add(j + 1),
					t,
					x.as_ptr().add(j + 1),
					y.as_mut_ptr().add(j + 1),
					n - j - 1,
				)
			};
			y[j] += t * col[j] + alpha * d;
		}
		j += 1;
	}
}
