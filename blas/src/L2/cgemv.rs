//! `cgemv` — complex matrix × vector: y ← αAx + βy, with the
//! transpose twin (Aᵀx, unconjugated) and the conjugate-transpose
//! twin (Aᴴx — the form complex algorithms actually consume).
//!
//! Implementation: same shapes as `dgemv` — the no-transpose form is
//! 4-column fan-in column-caxpy (`kernels::caxpy4in`; per-element
//! rounding sequence identical to the plain column loop, bit-for-bit
//! tested), the transpose forms are one `cdotu`/`cdotc` reduction
//! stream per column and inherit the 4-accumulator dot through
//! composition.

use super::{check_mat, cscale_y};
use crate::c32::C32;
use crate::kernels::caxpy4in;
use crate::L1::{caxpy, cdotc, cdotu};

/// y ← αAx + βy. A is nrows×ncols at column stride `cs`;
/// x has ncols elements, y has nrows.
#[allow(clippy::too_many_arguments)]
pub fn cgemv(
	alpha: C32,
	nrows: usize,
	ncols: usize,
	a: &[C32],
	cs: usize,
	x: &[C32],
	beta: C32,
	y: &mut [C32],
) {
	check_mat(a.len(), nrows, ncols, cs);
	assert_eq!(x.len(), ncols, "cgemv: x length mismatch");
	assert_eq!(y.len(), nrows, "cgemv: y length mismatch");
	cscale_y(beta, y);
	let mut j = 0usize;
	while j + 4 <= ncols {
		let t = [alpha * x[j], alpha * x[j + 1], alpha * x[j + 2], alpha * x[j + 3]];
		let ap = a.as_ptr();
		unsafe {
			caxpy4in(
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
		caxpy(alpha * x[j], &a[j * cs..j * cs + nrows], y);
		j += 1;
	}
}

/// y ← αAᵀx + βy (unconjugated transpose). A is nrows×ncols at column
/// stride `cs`; x has nrows elements, y has ncols.
#[allow(clippy::too_many_arguments)]
pub fn cgemv_t(
	alpha: C32,
	nrows: usize,
	ncols: usize,
	a: &[C32],
	cs: usize,
	x: &[C32],
	beta: C32,
	y: &mut [C32],
) {
	check_mat(a.len(), nrows, ncols, cs);
	assert_eq!(x.len(), nrows, "cgemv_t: x length mismatch");
	assert_eq!(y.len(), ncols, "cgemv_t: y length mismatch");
	cscale_y(beta, y);
	for j in 0..ncols {
		y[j] = y[j] + alpha * cdotu(&a[j * cs..j * cs + nrows], x);
	}
}

/// y ← αAᴴx + βy (conjugate transpose). A is nrows×ncols at column
/// stride `cs`; x has nrows elements, y has ncols.
#[allow(clippy::too_many_arguments)]
pub fn cgemv_c(
	alpha: C32,
	nrows: usize,
	ncols: usize,
	a: &[C32],
	cs: usize,
	x: &[C32],
	beta: C32,
	y: &mut [C32],
) {
	check_mat(a.len(), nrows, ncols, cs);
	assert_eq!(x.len(), nrows, "cgemv_c: x length mismatch");
	assert_eq!(y.len(), ncols, "cgemv_c: y length mismatch");
	cscale_y(beta, y);
	for j in 0..ncols {
		y[j] = y[j] + alpha * cdotc(&a[j * cs..j * cs + nrows], x);
	}
}
