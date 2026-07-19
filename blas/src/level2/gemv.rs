//! `gemv` — matrix × vector: y ← αAx + βy (and the transpose twin
//! y ← αAᵀx + βy).
//!
//! Implementation: 4-column fan-in column-axpy (tuned 2026-07-19) —
//! groups of four A columns stream through one pass over y, so y's
//! read-modify-write traffic drops 4× versus one `axpy` per column;
//! the rounding sequence per element is identical to the plain
//! column-axpy loop (bit-for-bit tested). Tail columns fall back to
//! Level 1 `axpy`. The transpose form is one `dot` reduction stream
//! per column. Evidence: docs/blas-ab-2026-07.md. The fused-FMA
//! variant (measured better in the step-1 race) is DEFERRED at the
//! f64 campaign close (relaxed-madd rounding is implementation-
//! dependent — architect decision, recorded in ROADMAP).

use super::{check_mat, scale_y};
use crate::kernels::axpy4in;
use crate::level1::{axpy, dot};

/// y ← αAx + βy. A is nrows×ncols at column stride `cs`;
/// x has ncols elements, y has nrows.
#[allow(clippy::too_many_arguments)]
pub fn gemv(
	alpha: f64,
	nrows: usize,
	ncols: usize,
	a: &[f64],
	cs: usize,
	x: &[f64],
	beta: f64,
	y: &mut [f64],
) {
	check_mat(a.len(), nrows, ncols, cs);
	assert_eq!(x.len(), ncols, "gemv: x length mismatch");
	assert_eq!(y.len(), nrows, "gemv: y length mismatch");
	scale_y(beta, y);
	let mut j = 0usize;
	while j + 4 <= ncols {
		let t = [alpha * x[j], alpha * x[j + 1], alpha * x[j + 2], alpha * x[j + 3]];
		let ap = a.as_ptr();
		unsafe {
			axpy4in(
				ap.add(j * cs),
				ap.add((j + 1) * cs),
				ap.add((j + 2) * cs),
				ap.add((j + 3) * cs),
				t,
				y.as_mut_ptr(),
				nrows,
			);
		}
		j += 4;
	}
	while j < ncols {
		axpy(alpha * x[j], &a[j * cs..j * cs + nrows], y);
		j += 1;
	}
}

/// y ← αAᵀx + βy. A is nrows×ncols at column stride `cs`;
/// x has nrows elements, y has ncols.
#[allow(clippy::too_many_arguments)]
pub fn gemv_t(
	alpha: f64,
	nrows: usize,
	ncols: usize,
	a: &[f64],
	cs: usize,
	x: &[f64],
	beta: f64,
	y: &mut [f64],
) {
	check_mat(a.len(), nrows, ncols, cs);
	assert_eq!(x.len(), nrows, "gemv_t: x length mismatch");
	assert_eq!(y.len(), ncols, "gemv_t: y length mismatch");
	scale_y(beta, y);
	for j in 0..ncols {
		y[j] += alpha * dot(&a[j * cs..j * cs + nrows], x);
	}
}
