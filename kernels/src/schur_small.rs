//! Fix-3 of the eigen plan: a wasm-shaped small-n eigenvalue iteration.
//!
//! Double-shift Francis QR on an upper-Hessenberg matrix (`dlahqr`-shape),
//! **eigenvalues only** (`want_t = false`, no `Z`), ported from faer's
//! `lahqr` (the pinned base's `real_schur.rs`) into flat pointer code, generic over [`WasmScalar`] (f64/f32).
//! Same shifts, same deflation criteria, same exceptional-shift constants
//! (0.75 / −0.4375 every 10 stalled iterations), same two-consecutive-
//! small-subdiagonals early start — so convergence behavior matches faer's;
//! only the inner reflector-application loops change: raw column-major
//! pointer arithmetic with the three contiguous row entries per column kept
//! in registers, instead of generic-indexing abstractions. The eigenvalues-
//! only mode also drops the 2×2-block standardization (`lahqr_schur22` /
//! `lasy2`) entirely: a deflated block is never re-read, so only its
//! eigenvalues (`eig22`) are recorded.
//!
//! Like the other kernels this targets the well-conditioned dense regime
//! (no `dlarfg` small-β rescaling path).

use faer::MatMut;

use crate::scalar::WasmScalar;

/// eigenvalues of [[a00,a01],[a10,a11]] (faer `lahqr_eig22` / LAPACK-style
/// scaled 2×2 solve); returns ((re1,im1),(re2,im2))
#[inline]
fn eig22<T: WasmScalar>(a00: T, a01: T, a10: T, a11: T) -> ((T, T), (T, T)) {
	let s = a00.abs() + a01.abs() + a10.abs() + a11.abs();
	if s == T::ZERO {
		return ((T::ZERO, T::ZERO), (T::ZERO, T::ZERO));
	}
	let a00 = a00 / s;
	let a01 = a01 / s;
	let a10 = a10 / s;
	let a11 = a11 / s;
	let tr = (a00 + a11) * T::from_f64(0.5);
	let det = (a00 - tr) * (a00 - tr) + a01 * a10;
	if det >= T::ZERO {
		let rtdisc = det.sqrt();
		((s * (tr + rtdisc), T::ZERO), (s * (tr - rtdisc), T::ZERO))
	} else {
		let rtdisc = (-det).sqrt();
		let re = s * tr;
		let im = s * rtdisc;
		((re, im), (re, -im))
	}
}

/// first column of (H − s1·I)(H − s2·I) for the 3×3 (or 2×2) leading block
/// (faer `lahqr_shiftcolumn` / LAPACK `dlaqr1`), scaled for safety.
#[inline]
#[allow(clippy::too_many_arguments)]
fn shiftcolumn3<T: WasmScalar>(
	h00: T,
	h01: T,
	h02: T,
	h10: T,
	h11: T,
	h12: T,
	h20: T,
	h21: T,
	h22: T,
	s1: (T, T),
	s2: (T, T),
) -> (T, T, T) {
	let s = (h00 - s2.0).abs() + s2.1.abs() + h10.abs() + h20.abs();
	if s == T::ZERO {
		return (T::ZERO, T::ZERO, T::ZERO);
	}
	let h10s = h10 / s;
	let h20s = h20 / s;
	let v0 = (h00 - s1.0) * ((h00 - s2.0) / s) - s1.1 * (s2.1 / s) + h01 * h10s + h02 * h20s;
	let v1 = h10s * (h00 + h11 - s1.0 - s2.0) + h12 * h20s;
	let v2 = h20s * (h00 + h22 - s1.0 - s2.0) + h21 * h10s;
	(v0, v1, v2)
}

#[inline]
fn shiftcolumn2<T: WasmScalar>(h00: T, h01: T, h10: T, h11: T, s1: (T, T), s2: (T, T)) -> (T, T) {
	let s = (h00 - s2.0).abs() + s2.1.abs() + h10.abs();
	if s == T::ZERO {
		return (T::ZERO, T::ZERO);
	}
	let h10s = h10 / s;
	let v0 = h10s * h01 + (h00 - s1.0) * ((h00 - s2.0) / s) - s1.1 * (s2.1 / s);
	let v1 = h10s * (h00 + h11 - s1.0 - s2.0);
	(v0, v1)
}

/// dlarfg-style reflector from (b0, b1, b2): returns (beta, tau, v1, v2)
/// with H = I − τ·[1,v1,v2][1,v1,v2]ᵀ. Pass b2 = 0.0 for the 2-vector case
/// (v2 comes back 0).
#[inline]
fn householder3<T: WasmScalar>(b0: T, b1: T, b2: T) -> (T, T, T, T) {
	let xnorm_sq = b1 * b1 + b2 * b2;
	if xnorm_sq == T::ZERO {
		return (b0, T::ZERO, T::ZERO, T::ZERO);
	}
	let anorm = (b0 * b0 + xnorm_sq).sqrt();
	let beta = if b0 >= T::ZERO { -anorm } else { anorm };
	let tau = (beta - b0) / beta;
	let inv = T::ONE / (b0 - beta);
	(beta, tau, b1 * inv, b2 * inv)
}

/// Computes the eigenvalues of an upper-Hessenberg `h` in place (contents
/// destroyed), conjugate pairs adjacent in `w_re`/`w_im`. Returns 0 on
/// success, or (LAPACK-style) the failing index+1 on non-convergence.
pub fn hqr_eigvals_in_place<T: WasmScalar>(h: MatMut<'_, T>, w_re: &mut [T], w_im: &mut [T]) -> isize {
	let n = h.nrows();
	assert!(h.ncols() == n, "square input required");
	assert!(w_re.len() >= n && w_im.len() >= n);
	assert!(h.row_stride() == 1, "column-major with unit row stride required");
	let cs = h.col_stride() as usize;
	let p = h.as_ptr_mut();

	let eps = T::EPS;
	let small_num = T::SMALL_NUM;
	let dat1 = T::from_f64(0.75);
	let dat2 = T::from_f64(-0.4375);
	const NON_CONVERGENCE_LIMIT: usize = 10;

	if n == 0 {
		return 0;
	}
	if n == 1 {
		unsafe {
			w_re[0] = *p;
		}
		w_im[0] = T::ZERO;
		return 0;
	}

	unsafe {
		// entry (r, c)
		macro_rules! at {
			($r:expr, $c:expr) => {
				*p.add(($r) + ($c) * cs)
			};
		}

		let itmax = 30 * Ord::max(10, n);
		let mut k_defl = 0usize;
		let mut istop = n;
		let mut istart = 0usize;

		for iter in 0..itmax + 1 {
			if iter == itmax {
				return istop as isize;
			}
			if istart + 1 >= istop {
				if istart + 1 == istop {
					w_re[istart] = at!(istart, istart);
					w_im[istart] = T::ZERO;
				}
				break;
			}

			// deflation scan: find a negligible subdiagonal to split at
			for i in (istart + 1..istop).rev() {
				let sub = at!(i, i - 1);
				if sub.abs() < small_num {
					at!(i, i - 1) = T::ZERO;
					istart = i;
					break;
				}
				let mut tst = at!(i - 1, i - 1).abs() + at!(i, i).abs();
				if tst == T::ZERO {
					if i >= 2 {
						tst += at!(i - 1, i - 2).abs();
					}
					if i + 1 < n {
						tst += at!(i + 1, i).abs();
					}
				}
				if sub.abs() <= eps * tst {
					// Ahues–Tisseur small-subdiagonal test
					let sup = at!(i - 1, i);
					let ab = sub.abs().maxs(sup.abs());
					let ba = sub.abs().mins(sup.abs());
					let d = at!(i, i) - at!(i - 1, i - 1);
					let aa = at!(i, i).abs().maxs(d.abs());
					let bb = at!(i, i).abs().mins(d.abs());
					let s = aa + ab;
					if ba * (ab / s) <= (eps * (bb * (aa / s))).maxs(small_num) {
						at!(i, i - 1) = T::ZERO;
						istart = i;
						break;
					}
				}
			}

			// 1×1 / 2×2 tail deflation
			if istart + 2 >= istop {
				if istart + 1 == istop {
					k_defl = 0;
					w_re[istart] = at!(istart, istart);
					w_im[istart] = T::ZERO;
					istop = istart;
					istart = 0;
					continue;
				}
				if istart + 2 == istop {
					// eigenvalues-only: record eig22, no standardization
					let (s1, s2) = eig22(
						at!(istart, istart),
						at!(istart, istart + 1),
						at!(istart + 1, istart),
						at!(istart + 1, istart + 1),
					);
					w_re[istart] = s1.0;
					w_im[istart] = s1.1;
					w_re[istart + 1] = s2.0;
					w_im[istart + 1] = s2.1;
					k_defl = 0;
					istop = istart;
					istart = 0;
					continue;
				}
			}

			// shifts: trailing 2×2, or exceptional every 10 stalled rounds
			let (a00, a01, a10, a11);
			k_defl += 1;
			if k_defl % NON_CONVERGENCE_LIMIT == 0 {
				let mut s = at!(istop - 1, istop - 2).abs();
				if istop > 2 {
					s += at!(istop - 2, istop - 3).abs();
				}
				a00 = dat1 * s + at!(istop - 1, istop - 1);
				a01 = dat2 * s;
				a10 = s;
				a11 = a00;
			} else {
				a00 = at!(istop - 2, istop - 2);
				a01 = at!(istop - 2, istop - 1);
				a10 = at!(istop - 1, istop - 2);
				a11 = at!(istop - 1, istop - 1);
			}
			let (mut s1, mut s2) = eig22(a00, a01, a10, a11);
			if s1.1 == T::ZERO && s2.1 == T::ZERO {
				// prefer the shift closer to the trailing entry, doubled
				let t = at!(istop - 1, istop - 1);
				if (s1.0 - t).abs() <= (s2.0 - t).abs() {
					s2 = s1;
				} else {
					s1 = s2;
				}
			}

			// two-consecutive-small-subdiagonals early start
			let mut istart2 = istart;
			if istart + 3 < istop {
				for i in (istart + 1..istop - 2).rev() {
					let (v0, v1, v2) = shiftcolumn3(
						at!(i, i),
						at!(i, i + 1),
						at!(i, i + 2),
						at!(i + 1, i),
						at!(i + 1, i + 1),
						at!(i + 1, i + 2),
						at!(i + 2, i),
						at!(i + 2, i + 1),
						at!(i + 2, i + 2),
						s1,
						s2,
					);
					let (_, tau, w1, w2) = householder3(v0, v1, v2);
					let refsum = tau * at!(i, i - 1) + w1 * at!(i + 1, i - 1);
					if (at!(i + 1, i - 1) - refsum * w1).abs() + (refsum * w2).abs()
						<= eps
							* (at!(i, i - 1).abs()
								+ at!(i, i + 1).abs() + at!(i + 1, i + 2).abs())
					{
						istart2 = i;
						break;
					}
				}
			}

			// the double-shift bulge chase over [istart2, istop)
			for i in istart2..istop - 1 {
				let nr = Ord::min(3, istop - i);
				let (tau, v1, v2);
				if i == istart2 {
					let (b0, b1, b2) = if nr == 3 {
						shiftcolumn3(
							at!(i, i),
							at!(i, i + 1),
							at!(i, i + 2),
							at!(i + 1, i),
							at!(i + 1, i + 1),
							at!(i + 1, i + 2),
							at!(i + 2, i),
							at!(i + 2, i + 1),
							at!(i + 2, i + 2),
							s1,
							s2,
						)
					} else {
						let (b0, b1) = shiftcolumn2(
							at!(i, i),
							at!(i, i + 1),
							at!(i + 1, i),
							at!(i + 1, i + 1),
							s1,
							s2,
						);
						(b0, b1, T::ZERO)
					};
					let (_, t, w1, w2) = householder3(b0, b1, b2);
					tau = t;
					v1 = w1;
					v2 = w2;
					if i > istart {
						at!(i, i - 1) *= T::ONE - tau;
					}
				} else {
					let b0 = at!(i, i - 1);
					let b1 = at!(i + 1, i - 1);
					let b2 = if nr == 3 { at!(i + 2, i - 1) } else { T::ZERO };
					let (beta, t, w1, w2) = householder3(b0, b1, b2);
					tau = t;
					v1 = w1;
					v2 = w2;
					at!(i, i - 1) = beta;
					at!(i + 1, i - 1) = T::ZERO;
					if nr == 3 {
						at!(i + 2, i - 1) = T::ZERO;
					}
				}
				let t2 = tau * v1;
				if nr == 3 {
					let t3 = tau * v2;
					// rows i, i+1, i+2 across columns i..istop: the three
					// row entries per column are CONTIGUOUS in memory
					let mut pj = p.add(i + i * cs);
					let mut j = i;
					while j < istop {
						let r0 = *pj;
						let r1 = *pj.add(1);
						let r2 = *pj.add(2);
						let sum = r0 + v1 * r1 + v2 * r2;
						*pj = r0 - sum * tau;
						*pj.add(1) = r1 - sum * t2;
						*pj.add(2) = r2 - sum * t3;
						pj = pj.add(cs);
						j += 1;
					}
					// columns i, i+1, i+2 over rows istart..min(i+4, istop)
					let c0 = p.add(i * cs);
					let c1 = p.add((i + 1) * cs);
					let c2 = p.add((i + 2) * cs);
					let jend = Ord::min(i + 4, istop);
					for r in istart..jend {
						let x0 = *c0.add(r);
						let x1 = *c1.add(r);
						let x2 = *c2.add(r);
						let sum = x0 + v1 * x1 + v2 * x2;
						*c0.add(r) = x0 - sum * tau;
						*c1.add(r) = x1 - sum * t2;
						*c2.add(r) = x2 - sum * t3;
					}
				} else {
					let mut pj = p.add(i + i * cs);
					let mut j = i;
					while j < istop {
						let r0 = *pj;
						let r1 = *pj.add(1);
						let sum = r0 + v1 * r1;
						*pj = r0 - sum * tau;
						*pj.add(1) = r1 - sum * t2;
						pj = pj.add(cs);
						j += 1;
					}
					let c0 = p.add(i * cs);
					let c1 = p.add((i + 1) * cs);
					let jend = Ord::min(i + 3, istop);
					for r in istart..jend {
						let x0 = *c0.add(r);
						let x1 = *c1.add(r);
						let sum = x0 + v1 * x1;
						*c0.add(r) = x0 - sum * tau;
						*c1.add(r) = x1 - sum * t2;
					}
				}
			}
		}
	}
	0
}
