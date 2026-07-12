//! One-sided Jacobi SVD (cyclic, unpreconditioned) for f64 on wasm.
//!
//! **Status: measured and KILLED 2026-07-10; retained as the evidence
//! artifact, not a shipping path.** This was the probe built to decide
//! whether a preconditioned Jacobi SVD could beat faer's bidiag→DC
//! pipeline on wasm (the roofline said ~70% of faer's SVD time exists
//! only because it bidiagonalizes, and Jacobi's column dot/rotation core
//! is the simd128-native shape). Measurement answered no: 12–15 sweeps
//! to converge at n≤512 ≈ 8× the flops = 0.1–0.2× scipy on the runner
//! (docs/research-svd-wasm-2026-07.md). The kernel and its correctness
//! test stay so the refutation remains reproducible (2026-07-12 sweep
//! decision); its bench exports were dropped in that sweep.

use faer::MatMut;

/// `sum_k a[k]*b[k]` over `m` contiguous rows.
#[inline(always)]
unsafe fn cdot(a: *const f64, b: *const f64, m: usize) -> f64 {
	let mut s = 0.0f64;
	let mut k = 0usize;
	while k < m {
		s += *a.add(k) * *b.add(k);
		k += 1;
	}
	s
}

/// Apply the plane rotation `[[c, s], [-s, c]]` to the column pair (x, y):
/// `x' = c·x − s·y`, `y' = s·x + c·y`, over `m` contiguous rows.
#[inline(always)]
unsafe fn rot_cols(x: *mut f64, y: *mut f64, c: f64, s: f64, m: usize) {
	let mut k = 0usize;
	while k < m {
		let xk = *x.add(k);
		let yk = *y.add(k);
		*x.add(k) = c * xk - s * yk;
		*y.add(k) = s * xk + c * yk;
		k += 1;
	}
}

/// One-sided Jacobi SVD of `a` (m×n, **m ≥ n**). On exit:
/// - `a`'s columns hold the left singular vectors `U` (unit norm),
/// - `s[j]` holds the singular values (in `a`'s column order, unsorted),
/// - `v` holds the right singular vectors `V` (n×n orthogonal),
///
/// so `A_orig = U · diag(s) · Vᵀ`. Returns the number of sweeps run.
/// `max_sweeps` caps iteration (well-conditioned inputs converge in a
/// handful); `tol` is the relative off-diagonal threshold for convergence.
pub fn jacobi_svd_in_place(
	a: MatMut<'_, f64>,
	v: MatMut<'_, f64>,
	s: &mut [f64],
	max_sweeps: usize,
	tol: f64,
) -> usize {
	let m = a.nrows();
	let n = a.ncols();
	assert!(m >= n, "one-sided Jacobi wants m >= n");
	assert!(v.nrows() == n && v.ncols() == n, "v must be n×n");
	assert!(s.len() >= n);
	assert!(a.row_stride() == 1 && v.row_stride() == 1, "column-major, unit row stride");
	let acs = a.col_stride() as usize;
	let vcs = v.col_stride() as usize;
	let abase = a.as_ptr_mut();
	let vbase = v.as_ptr_mut();

	let mut sweeps = 0usize;
	unsafe {
		// V = I
		for j in 0..n {
			let col = vbase.add(j * vcs);
			for i in 0..n {
				*col.add(i) = if i == j { 1.0 } else { 0.0 };
			}
		}

		for _ in 0..max_sweeps {
			sweeps += 1;
			let mut max_off = 0.0f64; // largest relative off-diagonal this sweep
			for i in 0..n {
				let ci = abase.add(i * acs);
				for j in (i + 1)..n {
					let cj = abase.add(j * acs);
					let alpha = cdot(ci, ci, m);
					let beta = cdot(cj, cj, m);
					let gamma = cdot(ci, cj, m);
					let denom = libm::sqrt(alpha * beta);
					if denom == 0.0 {
						continue;
					}
					let rel = gamma.abs() / denom;
					if rel > max_off {
						max_off = rel;
					}
					if rel <= tol {
						continue;
					}
					// Jacobi rotation zeroing gamma (Golub–Van Loan):
					let zeta = (beta - alpha) / (2.0 * gamma);
					let t = if zeta == 0.0 {
						1.0
					} else {
						zeta.signum() / (zeta.abs() + libm::sqrt(1.0 + zeta * zeta))
					};
					let c = 1.0 / libm::sqrt(1.0 + t * t);
					let sn = t * c;
					rot_cols(ci, cj, c, sn, m); // columns of A
					rot_cols(vbase.add(i * vcs), vbase.add(j * vcs), c, sn, n); // V
				}
			}
			if max_off <= tol {
				break;
			}
		}

		// singular values = column norms; normalize columns of A into U
		for j in 0..n {
			let col = abase.add(j * acs);
			let nrm = libm::sqrt(cdot(col, col, m));
			s[j] = nrm;
			if nrm > 0.0 {
				let inv = 1.0 / nrm;
				for k in 0..m {
					*col.add(k) *= inv;
				}
			}
		}
	}
	sweeps
}
