//! `zhemv` — Hermitian matrix × vector: y ← αAx + βy, A Hermitian
//! with only one triangle stored (A[j,i] = conj(A[i,j]); diagonal
//! imaginary parts are ignored, per the LAPACK storage convention).
//!
//! Implementation: fused column pass — one stream over the stored
//! strict segment serves both triangles' contributions
//! (`kernels::zaxpy_dotc`: y[i] += t·a[i] elementwise while
//! acc += conj(a[i])·x[i] reduces), the complex twin of the `dsymv`
//! fused shape at single-column width. The diagonal contributes
//! t·re(a[j,j]) (real by convention). Accumulation order is the fused
//! pass's own — zhemv is bounds-tested, not bit-locked; cross-target
//! determinism holds through the lane emulation as everywhere else.
//! The 4-column fused grouping that pushed `dsymv` to 2× is a
//! recorded tuning lever, not yet built for c64.

use super::{check_mat, zscale_y};
use crate::c64::C64;
use crate::kernels::zaxpy_dotc;

/// y ← αAx + βy, A Hermitian n×n at column stride `cs`, with the
/// `upper` (or lower) triangle stored.
#[allow(clippy::too_many_arguments)]
pub fn zhemv(
	alpha: C64,
	n: usize,
	a: &[C64],
	cs: usize,
	upper: bool,
	x: &[C64],
	beta: C64,
	y: &mut [C64],
) {
	check_mat(a.len(), n, n, cs);
	assert_eq!(x.len(), n, "zhemv: x length mismatch");
	assert_eq!(y.len(), n, "zhemv: y length mismatch");
	zscale_y(beta, y);
	for j in 0..n {
		let cj = j * cs;
		let t = alpha * x[j];
		let d = if upper {
			unsafe { zaxpy_dotc(a.as_ptr().add(cj), t, x.as_ptr(), y.as_mut_ptr(), j) }
		} else {
			unsafe {
				zaxpy_dotc(
					a.as_ptr().add(cj + j + 1),
					t,
					x.as_ptr().add(j + 1),
					y.as_mut_ptr().add(j + 1),
					n - j - 1,
				)
			}
		};
		y[j] = y[j] + t.scale(a[cj + j].re) + alpha * d;
	}
}
