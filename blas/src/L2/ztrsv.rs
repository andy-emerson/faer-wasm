//! `ztrsv` — triangular solve, one right-hand side, in place:
//! x ← A⁻¹x (complex).
//!
//! Implementation: the `dtrsv` shape verbatim — divide-then-
//! column-zaxpy, 4-column fan-in (`kernels::zaxpy4in`): the four
//! divisions (Smith's algorithm — `C64`'s guarded division, the
//! `dladiv` shape) and the ≤3-row in-group elimination band run
//! scalar in reference order, then the four solved unknowns eliminate
//! from the common remaining segment in one shared pass. Per-element
//! rounding sequence unchanged — bit-for-bit the plain loop, locked
//! by the test. No singularity check — a zero diagonal yields
//! inf/NaN, as in reference BLAS. Transposed/conjugate forms: not
//! built — no consumer yet (explicit gap).

use super::check_mat;
use crate::c64::C64;
use crate::kernels::zaxpy4in;
use crate::L1::zaxpy;

/// x ← A⁻¹x, A triangular n×n at column stride `cs`. `upper` selects
/// the triangle; `unit` treats the diagonal as ones (stored values
/// ignored).
pub fn ztrsv(n: usize, a: &[C64], cs: usize, upper: bool, unit: bool, x: &mut [C64]) {
	check_mat(a.len(), n, n, cs);
	assert_eq!(x.len(), n, "ztrsv: x length mismatch");
	let ap = a.as_ptr();
	if upper {
		let r = n % 4;
		// high leftover rows first, original descending order
		for j in (n - r..n).rev() {
			let cj = j * cs;
			if !unit {
				x[j] = x[j] / a[cj + j];
			}
			let t = x[j];
			zaxpy(-t, &a[cj..cj + j], &mut x[..j]);
		}
		let mut g = n - r;
		while g >= 4 {
			g -= 4;
			// in-group solve, original descending order (each t depends
			// on the adds from the steps above it)
			let mut t = [C64::ZERO; 4];
			for u in (0..4).rev() {
				let cj = (g + u) * cs;
				if !unit {
					x[g + u] = x[g + u] / a[cj + g + u];
				}
				t[u] = -x[g + u];
				for i in g..g + u {
					x[i] = x[i] + a[cj + i] * t[u];
				}
			}
			// common prefix x[..g], sources in descending step order
			unsafe {
				zaxpy4in(
					ap.add((g + 3) * cs),
					ap.add((g + 2) * cs),
					ap.add((g + 1) * cs),
					ap.add(g * cs),
					[t[3], t[2], t[1], t[0]],
					x.as_mut_ptr(),
					g,
				);
			}
		}
	} else {
		let mut j = 0usize;
		while j + 4 <= n {
			// in-group solve, original ascending order
			let mut t = [C64::ZERO; 4];
			for u in 0..4 {
				let cj = (j + u) * cs;
				if !unit {
					x[j + u] = x[j + u] / a[cj + j + u];
				}
				t[u] = -x[j + u];
				for i in j + u + 1..j + 4 {
					x[i] = x[i] + a[cj + i] * t[u];
				}
			}
			// common suffix x[j+4..], sources in ascending step order
			unsafe {
				zaxpy4in(
					ap.add(j * cs + j + 4),
					ap.add((j + 1) * cs + j + 4),
					ap.add((j + 2) * cs + j + 4),
					ap.add((j + 3) * cs + j + 4),
					t,
					x.as_mut_ptr().add(j + 4),
					n - j - 4,
				);
			}
			j += 4;
		}
		while j < n {
			let cj = j * cs;
			if !unit {
				x[j] = x[j] / a[cj + j];
			}
			let t = x[j];
			zaxpy(-t, &a[cj + j + 1..cj + n], &mut x[j + 1..]);
			j += 1;
		}
	}
}
