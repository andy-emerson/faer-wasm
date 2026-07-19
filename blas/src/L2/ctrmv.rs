//! `ctrmv` — triangular matrix × vector, in place: x ← Ax (complex).
//!
//! Implementation: the `dtrmv` shape verbatim — 4-column fan-in
//! column-caxpy (`kernels::caxpy4in`, sources in reference column
//! order: ascending for upper, descending for lower), ragged band and
//! diagonal writes scalar in the original step order. Per-element
//! rounding sequence unchanged — bit-for-bit the plain column loop,
//! locked by the test. Transposed/conjugate forms: not built — no
//! consumer yet (explicit gap).

use super::check_mat;
use crate::c32::C32;
use crate::kernels::caxpy4in;
use crate::L1::caxpy;

/// x ← Ax, A triangular n×n at column stride `cs`. `upper` selects the
/// triangle; `unit` treats the diagonal as ones (stored values
/// ignored).
pub fn ctrmv(n: usize, a: &[C32], cs: usize, upper: bool, unit: bool, x: &mut [C32]) {
	check_mat(a.len(), n, n, cs);
	assert_eq!(x.len(), n, "ctrmv: x length mismatch");
	let ap = a.as_ptr();
	if upper {
		let mut j = 0usize;
		while j + 4 <= n {
			// all four t's are original: step j+u writes only x[..=j+u']
			// for u' < u, never x[j+u]
			let t = [x[j], x[j + 1], x[j + 2], x[j + 3]];
			unsafe {
				caxpy4in(
					ap.add(j * cs),
					ap.add((j + 1) * cs),
					ap.add((j + 2) * cs),
					ap.add((j + 3) * cs),
					t,
					x.as_mut_ptr(),
					j,
				);
			}
			// ragged band + diagonals, original ascending step order
			for u in 0..4 {
				let cj = (j + u) * cs;
				for i in j..j + u {
					x[i] = x[i] + t[u] * a[cj + i];
				}
				if !unit {
					x[j + u] = t[u] * a[cj + j + u];
				}
			}
			j += 4;
		}
		while j < n {
			let cj = j * cs;
			let t = x[j];
			caxpy(t, &a[cj..cj + j], &mut x[..j]);
			if !unit {
				x[j] = t * a[cj + j];
			}
			j += 1;
		}
	} else {
		let r = n % 4;
		// high leftover columns first, original descending order
		for j in (n - r..n).rev() {
			let cj = j * cs;
			let t = x[j];
			caxpy(t, &a[cj + j + 1..cj + n], &mut x[j + 1..]);
			if !unit {
				x[j] = t * a[cj + j];
			}
		}
		let mut g = n - r;
		while g >= 4 {
			g -= 4;
			// steps g+3 down to g; all four t's are original (step g+u'
			// writes only x[g+u'+1..], never x[g..=g+u'])
			let t = [x[g], x[g + 1], x[g + 2], x[g + 3]];
			// common suffix x[g+4..], sources in descending step order
			unsafe {
				caxpy4in(
					ap.add((g + 3) * cs + g + 4),
					ap.add((g + 2) * cs + g + 4),
					ap.add((g + 1) * cs + g + 4),
					ap.add(g * cs + g + 4),
					[t[3], t[2], t[1], t[0]],
					x.as_mut_ptr().add(g + 4),
					n - g - 4,
				);
			}
			// ragged band + diagonals, original descending step order
			for u in (0..4).rev() {
				let cj = (g + u) * cs;
				for i in g + u + 1..g + 4 {
					x[i] = x[i] + t[u] * a[cj + i];
				}
				if !unit {
					x[g + u] = t[u] * a[cj + g + u];
				}
			}
		}
	}
}
