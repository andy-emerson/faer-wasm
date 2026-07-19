//! `zgemv` — complex matrix × vector: y ← αAx + βy, with the
//! transpose twin (Aᵀx, unconjugated) and the conjugate-transpose
//! twin (Aᴴx — the form complex algorithms actually consume).
//!
//! Implementation: same shapes as `dgemv` — the no-transpose form is
//! 4-column fan-in column-zaxpy (`kernels::zaxpy4in`; per-element
//! rounding sequence identical to the plain column loop, bit-for-bit
//! tested), the transpose forms are one `zdotu`/`zdotc` reduction
//! stream per column and inherit the 4-accumulator dot through
//! composition.

use super::{check_mat, zscale_y};
use crate::c64::C64;
use crate::kernels::zaxpy4in;
use crate::L1::{zaxpy, zdotc, zdotu};

/// y ← αAx + βy. A is nrows×ncols at column stride `cs`;
/// x has ncols elements, y has nrows.
#[allow(clippy::too_many_arguments)]
pub fn zgemv(
	alpha: C64,
	nrows: usize,
	ncols: usize,
	a: &[C64],
	cs: usize,
	x: &[C64],
	beta: C64,
	y: &mut [C64],
) {
	check_mat(a.len(), nrows, ncols, cs);
	assert_eq!(x.len(), ncols, "zgemv: x length mismatch");
	assert_eq!(y.len(), nrows, "zgemv: y length mismatch");
	zscale_y(beta, y);
	let mut j = 0usize;
	while j + 4 <= ncols {
		let t = [alpha * x[j], alpha * x[j + 1], alpha * x[j + 2], alpha * x[j + 3]];
		let ap = a.as_ptr();
		unsafe {
			zaxpy4in(
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
		zaxpy(alpha * x[j], &a[j * cs..j * cs + nrows], y);
		j += 1;
	}
}

/// y ← αAᵀx + βy (unconjugated transpose). A is nrows×ncols at column
/// stride `cs`; x has nrows elements, y has ncols.
#[allow(clippy::too_many_arguments)]
pub fn zgemv_t(
	alpha: C64,
	nrows: usize,
	ncols: usize,
	a: &[C64],
	cs: usize,
	x: &[C64],
	beta: C64,
	y: &mut [C64],
) {
	check_mat(a.len(), nrows, ncols, cs);
	assert_eq!(x.len(), nrows, "zgemv_t: x length mismatch");
	assert_eq!(y.len(), ncols, "zgemv_t: y length mismatch");
	zscale_y(beta, y);
	for j in 0..ncols {
		y[j] = y[j] + alpha * zdotu(&a[j * cs..j * cs + nrows], x);
	}
}

/// y ← αAᴴx + βy (conjugate transpose). A is nrows×ncols at column
/// stride `cs`; x has nrows elements, y has ncols.
#[allow(clippy::too_many_arguments)]
pub fn zgemv_c(
	alpha: C64,
	nrows: usize,
	ncols: usize,
	a: &[C64],
	cs: usize,
	x: &[C64],
	beta: C64,
	y: &mut [C64],
) {
	check_mat(a.len(), nrows, ncols, cs);
	assert_eq!(x.len(), nrows, "zgemv_c: x length mismatch");
	assert_eq!(y.len(), ncols, "zgemv_c: y length mismatch");
	zscale_y(beta, y);
	for j in 0..ncols {
		y[j] = y[j] + alpha * zdotc(&a[j * cs..j * cs + nrows], x);
	}
}
