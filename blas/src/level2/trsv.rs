//! `trsv` — triangular solve, one right-hand side, in place: x ← A⁻¹x.
//!
//! Implementation: divide-then-column-axpy, 4-column fan-in (tuned
//! 2026-07-19) — the four divisions and the ≤3-row in-group
//! elimination band run scalar in reference `dtrsv`'s order (the
//! t-values are sequentially dependent), then the four solved
//! unknowns eliminate from the common remaining segment in one shared
//! pass (`kernels::axpy4in`, sources in the same step order), cutting
//! x's read-modify-write traffic 4×. Per-element rounding sequence
//! unchanged — bit-for-bit the plain loop, locked by the existing
//! test. Transposed form: not built — no consumer yet (explicit gap).

use super::check_mat;
use crate::kernels::axpy4in;
use crate::level1::axpy;

/// x ← A⁻¹x, A triangular n×n at column stride `cs`. `upper` selects
/// the triangle; `unit` treats the diagonal as ones (stored values
/// ignored). No singularity check — a zero diagonal yields inf/NaN,
/// as in reference BLAS.
pub fn trsv(n: usize, a: &[f64], cs: usize, upper: bool, unit: bool, x: &mut [f64]) {
	check_mat(a.len(), n, n, cs);
	assert_eq!(x.len(), n, "trsv: x length mismatch");
	let ap = a.as_ptr();
	if upper {
		let r = n % 4;
		// high leftover rows first, original descending order
		for j in (n - r..n).rev() {
			let col = &a[j * cs..j * cs + n];
			if !unit {
				x[j] /= col[j];
			}
			let t = x[j];
			axpy(-t, &col[..j], &mut x[..j]);
		}
		let mut g = n - r;
		while g >= 4 {
			g -= 4;
			// in-group solve, original descending order (each t depends
			// on the adds from the steps above it)
			let mut t = [0.0f64; 4];
			for u in (0..4).rev() {
				let cj = (g + u) * cs;
				if !unit {
					x[g + u] /= a[cj + g + u];
				}
				t[u] = -x[g + u];
				for i in g..g + u {
					x[i] += a[cj + i] * t[u];
				}
			}
			// common prefix x[..g], sources in descending step order
			unsafe {
				axpy4in(
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
			let mut t = [0.0f64; 4];
			for u in 0..4 {
				let cj = (j + u) * cs;
				if !unit {
					x[j + u] /= a[cj + j + u];
				}
				t[u] = -x[j + u];
				for i in j + u + 1..j + 4 {
					x[i] += a[cj + i] * t[u];
				}
			}
			// common suffix x[j+4..], sources in ascending step order
			unsafe {
				axpy4in(
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
			let col = &a[j * cs..j * cs + n];
			if !unit {
				x[j] /= col[j];
			}
			let t = x[j];
			axpy(-t, &col[j + 1..], &mut x[j + 1..]);
			j += 1;
		}
	}
}
