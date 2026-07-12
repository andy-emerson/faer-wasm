//! Right eigenvectors from the real Schur form (eigenvector campaign,
//! 2026-07-12): the `dtrevc3` shape — back-substitution on the
//! quasi-triangular T for the eigenvectors of T, then one blocked
//! back-transform through Z for the eigenvectors of A.
//!
//! Why this splits well on wasm: the back-substitution is O(n²)-ish
//! streaming work (flat axpy columns — our native shape), and the
//! back-transform V = Z·X is a single triangular matmul — the gemm-shaped
//! bulk that faer's matmul already wins by 4–20×. LAPACK's dtrevc3 blocks
//! the transform into chunked gemvs/gemms for workspace reasons we don't
//! have; X here is materialized whole (n² scratch, same as Z) and multiplied
//! once. X is *exactly* upper triangular in dtrevc3's packing — complex-pair
//! columns don't dip below the diagonal — so the multiply rides
//! `faer::linalg::matmul::triangular` (gemm-grade blocking, none of the 2×
//! zero-half flops a full gemm would pay).
//!
//! The numerics are ported, not invented (identity rule: proven machinery
//! earns its place): `dlaln2`'s guarded 1×1/2×2 solves with tiny-pivot
//! perturbation and anti-overflow scaling, `dladiv`'s robust complex
//! division, dtrevc3's column-norm/BIGNUM growth guards and its
//! max-component normalization. Specializations baked in: `ltrans = false`,
//! `ca = 1`, `d1 = d2 = 1` (the only way dtrevc calls dlaln2), and
//! `howmny = 'A'` (all vectors — no SELECT plumbing).
//!
//! Eigenvector packing is LAPACK `dgeev`'s (interop contract): a real
//! eigenvalue's vector occupies one column; a complex-conjugate pair
//! λ = a±bi occupies two adjacent columns (re, im), with re+i·im the
//! vector for a+bi and re−i·im for a−bi.

use faer::linalg::matmul::triangular::{self, BlockStructure};
use faer::prelude::*;
use faer::{Accum, MatMut, MatRef};

use crate::scalar::WasmScalar;

// ---------------------------------------------------------------------------
// dladiv: robust complex division (a + ib)/(c + id) — Baudin–Smith with
// the reference's pre-scaling.

fn ladiv2<T: WasmScalar>(a: T, b: T, c: T, d: T, r: T, t: T) -> T {
	if r != T::ZERO {
		let br = b * r;
		if br != T::ZERO {
			(a + br) * t
		} else {
			a * t + (b * t) * r
		}
	} else {
		(a + d * (b / c)) * t
	}
}

fn ladiv1<T: WasmScalar>(a: T, b: T, c: T, d: T) -> (T, T) {
	let r = d / c;
	let t = T::ONE / (c + d * r);
	let p = ladiv2(a, b, c, d, r, t);
	let q = ladiv2(b, -a, c, d, r, t);
	(p, q)
}

/// (p, q) = (a + ib) / (c + id) — also used by the c64 twin
/// ([`crate::eigvec_cplx`]) at `T = f64`
pub(crate) fn ladiv<T: WasmScalar>(a: T, b: T, c: T, d: T) -> (T, T) {
	let half = T::from_f64(0.5);
	let two = T::from_f64(2.0);
	let (mut aa, mut bb, mut cc, mut dd) = (a, b, c, d);
	let ab = a.abs().maxs(b.abs());
	let cd = c.abs().maxs(d.abs());
	let mut s = T::ONE;
	let ov = T::MAX_POS;
	let un = T::MIN_POS;
	let eps = T::EPS;
	let be = two / (eps * eps);
	if ab >= half * ov {
		aa = half * aa;
		bb = half * bb;
		s = two * s;
	}
	if cd >= half * ov {
		cc = half * cc;
		dd = half * dd;
		s = half * s;
	}
	if ab <= un * two / eps {
		aa = aa * be;
		bb = bb * be;
		s = s / be;
	}
	if cd <= un * two / eps {
		cc = cc * be;
		dd = dd * be;
		s = s * be;
	}
	let (p, q) = if d.abs() <= c.abs() {
		ladiv1(aa, bb, cc, dd)
	} else {
		let (p, q) = ladiv1(bb, aa, dd, cc);
		(p, -q)
	};
	(p * s, q * s)
}

// ---------------------------------------------------------------------------
// dlaln2, specialized to dtrevc's calls: solve (A − (wr + i·wi)·I)·x = b for
// na ∈ {1,2} (real block size) and nw ∈ {1,2} (real or complex shift/rhs),
// perturbing pivots below `smin` and scaling the rhs to prevent overflow.
//
// Layout mirrors the Fortran: `a = [a11, a21, a12, a22]` (column-major CRV
// order), `b[r + 2*c]` and `x[r + 2*c]` with column 0 = real part, column
// 1 = imaginary part. Returns (x, scale, xnorm).

const IPIVOT: [[usize; 4]; 4] = [[0, 1, 2, 3], [1, 0, 3, 2], [2, 3, 0, 1], [3, 2, 1, 0]];
const ZSWAP: [bool; 4] = [false, false, true, true];
const RSWAP: [bool; 4] = [false, true, false, true];

#[allow(clippy::too_many_arguments)]
fn laln2<T: WasmScalar>(
	na: usize,
	nw: usize,
	smin: T,
	a: [T; 4],
	b: [T; 4],
	wr: T,
	wi: T,
) -> ([T; 4], T, T) {
	let smlnum = T::from_f64(2.0) * T::MIN_POS;
	let bignum = T::ONE / smlnum;
	let smini = smin.maxs(smlnum);
	let mut scale = T::ONE;
	let mut x = [T::ZERO; 4];

	if na == 1 {
		if nw == 1 {
			// 1×1 real: (a11 − wr)·x = b
			let mut csr = a[0] - wr;
			let mut cnorm = csr.abs();
			if cnorm < smini {
				csr = smini;
				cnorm = smini;
			}
			let bnorm = b[0].abs();
			if cnorm < T::ONE && bnorm > T::ONE && bnorm > bignum * cnorm {
				scale = T::ONE / bnorm;
			}
			x[0] = (b[0] * scale) / csr;
			(x, scale, x[0].abs())
		} else {
			// 1×1 complex: (a11 − wr − i·wi)·x = b
			let mut csr = a[0] - wr;
			let mut csi = -wi;
			let cnorm = csr.abs() + csi.abs();
			if cnorm < smini {
				csr = smini;
				csi = T::ZERO;
			}
			let cnorm = cnorm.maxs(smini);
			let bnorm = b[0].abs() + b[2].abs();
			if cnorm < T::ONE && bnorm > T::ONE && bnorm > bignum * cnorm {
				scale = T::ONE / bnorm;
			}
			let (xr, xi) = ladiv(scale * b[0], scale * b[2], csr, csi);
			x[0] = xr;
			x[2] = xi;
			(x, scale, xr.abs() + xi.abs())
		}
	} else if nw == 1 {
		// 2×2 real: complete-pivoting Gaussian elimination
		let crv = [a[0] - wr, a[1], a[2], a[3] - wr];
		let mut cmax = T::ZERO;
		let mut icmax = 0usize;
		for (j, c) in crv.iter().enumerate() {
			if c.abs() > cmax {
				cmax = c.abs();
				icmax = j;
			}
		}
		if cmax < smini {
			let bnorm = b[0].abs().maxs(b[1].abs());
			if smini < T::ONE && bnorm > T::ONE && bnorm > bignum * smini {
				scale = T::ONE / bnorm;
			}
			let temp = scale / smini;
			x[0] = temp * b[0];
			x[1] = temp * b[1];
			return (x, scale, temp * bnorm);
		}
		let ur11 = crv[icmax];
		let cr21 = crv[IPIVOT[icmax][1]];
		let ur12 = crv[IPIVOT[icmax][2]];
		let cr22 = crv[IPIVOT[icmax][3]];
		let ur11r = T::ONE / ur11;
		let lr21 = ur11r * cr21;
		let mut ur22 = cr22 - ur12 * lr21;
		if ur22.abs() < smini {
			ur22 = smini;
		}
		let (br1, mut br2) = if RSWAP[icmax] { (b[1], b[0]) } else { (b[0], b[1]) };
		br2 = br2 - lr21 * br1;
		let bbnd = (br1 * (ur22 * ur11r)).abs().maxs(br2.abs());
		if bbnd > T::ONE && ur22.abs() < T::ONE && bbnd >= bignum * ur22.abs() {
			scale = T::ONE / bbnd;
		}
		let xr2 = (br2 * scale) / ur22;
		let xr1 = (scale * br1) * ur11r - xr2 * (ur11r * ur12);
		if ZSWAP[icmax] {
			x[0] = xr2;
			x[1] = xr1;
		} else {
			x[0] = xr1;
			x[1] = xr2;
		}
		let mut xnorm = xr1.abs().maxs(xr2.abs());
		if xnorm > T::ONE && cmax > T::ONE && xnorm > bignum / cmax {
			let temp = cmax / bignum;
			x[0] = temp * x[0];
			x[1] = temp * x[1];
			xnorm = temp * xnorm;
			scale = temp * scale;
		}
		(x, scale, xnorm)
	} else {
		// 2×2 complex: real 2×2 block, complex shift wr + i·wi
		let crv = [a[0] - wr, a[1], a[2], a[3] - wr];
		let civ = [-wi, T::ZERO, T::ZERO, -wi];
		let mut cmax = T::ZERO;
		let mut icmax = 0usize;
		for j in 0..4 {
			if crv[j].abs() + civ[j].abs() > cmax {
				cmax = crv[j].abs() + civ[j].abs();
				icmax = j;
			}
		}
		if cmax < smini {
			let bnorm = (b[0].abs() + b[2].abs()).maxs(b[1].abs() + b[3].abs());
			if smini < T::ONE && bnorm > T::ONE && bnorm > bignum * smini {
				scale = T::ONE / bnorm;
			}
			let temp = scale / smini;
			for k in 0..4 {
				x[k] = temp * b[k];
			}
			return (x, scale, temp * bnorm);
		}
		let ur11 = crv[icmax];
		let ui11 = civ[icmax];
		let cr21 = crv[IPIVOT[icmax][1]];
		let ci21 = civ[IPIVOT[icmax][1]];
		let ur12 = crv[IPIVOT[icmax][2]];
		let ui12 = civ[IPIVOT[icmax][2]];
		let cr22 = crv[IPIVOT[icmax][3]];
		let ci22 = civ[IPIVOT[icmax][3]];
		let (ur11r, ui11r, lr21, li21, ur12s, ui12s, mut ur22, mut ui22);
		if icmax == 0 || icmax == 3 {
			// pivot is a diagonal element (complex part −wi ≠ 0)
			if ur11.abs() > ui11.abs() {
				let temp = ui11 / ur11;
				ur11r = T::ONE / (ur11 * (T::ONE + temp * temp));
				ui11r = -temp * ur11r;
			} else {
				let temp = ur11 / ui11;
				ui11r = -T::ONE / (ui11 * (T::ONE + temp * temp));
				ur11r = -temp * ui11r;
			}
			lr21 = cr21 * ur11r;
			li21 = cr21 * ui11r;
			ur12s = ur12 * ur11r;
			ui12s = ur12 * ui11r;
			ur22 = cr22 - ur12 * lr21;
			ui22 = ci22 - ur12 * li21;
		} else {
			// pivot is an off-diagonal (purely real)
			ur11r = T::ONE / ur11;
			ui11r = T::ZERO;
			lr21 = cr21 * ur11r;
			li21 = ci21 * ur11r;
			ur12s = ur12 * ur11r;
			ui12s = ui12 * ur11r;
			ur22 = cr22 - ur12 * lr21 + ui12 * li21;
			ui22 = -ur12 * li21 - ui12 * lr21;
		}
		// NB: dlaln2 does NOT refresh u22abs after perturbing the pivot —
		// the downstream bbnd guard intentionally sees the tiny value
		let u22abs = ur22.abs() + ui22.abs();
		if u22abs < smini {
			ur22 = smini;
			ui22 = T::ZERO;
		}
		let (br1, mut br2, bi1, mut bi2) = if RSWAP[icmax] {
			(b[1], b[0], b[3], b[2])
		} else {
			(b[0], b[1], b[2], b[3])
		};
		br2 = br2 - lr21 * br1 + li21 * bi1;
		bi2 = bi2 - li21 * br1 - lr21 * bi1;
		let bbnd = ((br1.abs() + bi1.abs()) * (u22abs * (ur11r.abs() + ui11r.abs())))
			.maxs(br2.abs() + bi2.abs());
		let (mut br1, mut bi1) = (br1, bi1);
		if bbnd > T::ONE && u22abs < T::ONE && bbnd >= bignum * u22abs {
			scale = T::ONE / bbnd;
			br1 = scale * br1;
			bi1 = scale * bi1;
			br2 = scale * br2;
			bi2 = scale * bi2;
		}
		let (xr2, xi2) = ladiv(br2, bi2, ur22, ui22);
		let xr1 = ur11r * br1 - ui11r * bi1 - ur12s * xr2 + ui12s * xi2;
		let xi1 = ui11r * br1 + ur11r * bi1 - ui12s * xr2 - ur12s * xi2;
		if ZSWAP[icmax] {
			x[0] = xr2;
			x[1] = xr1;
			x[2] = xi2;
			x[3] = xi1;
		} else {
			x[0] = xr1;
			x[1] = xr2;
			x[2] = xi1;
			x[3] = xi2;
		}
		let mut xnorm = (xr1.abs() + xi1.abs()).maxs(xr2.abs() + xi2.abs());
		if xnorm > T::ONE && cmax > T::ONE && xnorm > bignum / cmax {
			let temp = cmax / bignum;
			for k in 0..4 {
				x[k] = temp * x[k];
			}
			xnorm = temp * xnorm;
			scale = temp * scale;
		}
		(x, scale, xnorm)
	}
}

// ---------------------------------------------------------------------------
// the driver

/// All right eigenvectors of the matrix whose real Schur form is
/// `(t, z)` — `t` quasi-upper-triangular with dlanv2-standardized 2×2
/// blocks (what [`crate::schur_small::hqr_schur_in_place`] and faer's
/// multishift both produce), `z` the accumulated orthogonal factor.
/// Writes the eigenvectors of A = Z·T·Zᵀ into `v` (n×n) in LAPACK dgeev
/// packing, each normalized so its largest `|re| + |im|` component is 1
/// (dtrevc3's convention). Column-major, unit row stride required.
pub fn trevc_in_place<T: WasmScalar>(t: MatRef<'_, T>, z: MatRef<'_, T>, mut v: MatMut<'_, T>) {
	let n = t.nrows();
	assert!(t.ncols() == n && z.nrows() == n && z.ncols() == n);
	assert!(v.nrows() == n && v.ncols() == n);
	assert!(t.row_stride() == 1 && z.row_stride() == 1 && v.row_stride() == 1);
	if n == 0 {
		return;
	}

	let ulp = T::EPS;
	let smlnum = T::MIN_POS * (T::from_f64(n as f64) / ulp);
	let bignum = (T::ONE - ulp) / smlnum;

	// strictly-above-diagonal 1-norms of T's columns (dtrevc3's WORK(J)),
	// used by the BIGNUM growth guards
	let mut colnorm = alloc::vec![T::ZERO; n];
	for j in 1..n {
		let mut s = T::ZERO;
		for i in 0..j {
			s += t[(i, j)].abs();
		}
		colnorm[j] = s;
	}

	// X: eigenvectors of T, exactly upper triangular (pair columns included)
	let mut x = faer::Mat::<T>::zeros(n, n);
	let xcs = x.as_ref().col_stride() as usize;
	debug_assert!(x.as_ref().row_stride() == 1);
	let xp = x.as_mut().as_ptr_mut();
	let tp = t.as_ptr();
	let tcs = t.col_stride() as usize;

	let mut ki = n; // 1-based bottom of the current eigenvalue group
	while ki > 0 {
		let kb = ki - 1;
		let is_pair = kb > 0 && t[(kb, kb - 1)] != T::ZERO;
		if !is_pair {
			// ---- real eigenvalue at diagonal index kb
			let wr = t[(kb, kb)];
			let smin = (ulp * wr.abs()).maxs(smlnum);
			unsafe {
				*xp.add(kb + kb * xcs) = T::ONE;
			}
			for r in 0..kb {
				unsafe {
					*xp.add(r + kb * xcs) = -t[(r, kb)];
				}
			}
			let xcol = unsafe { xp.add(kb * xcs) };
			let mut jj = kb as isize - 1;
			while jj >= 0 {
				let j0 = jj as usize;
				let two_block = j0 > 0 && t[(j0, j0 - 1)] != T::ZERO;
				if !two_block {
					let (xs, mut scale, xnorm) = laln2(
						1,
						1,
						smin,
						[t[(j0, j0)], T::ZERO, T::ZERO, T::ZERO],
						[unsafe { *xcol.add(j0) }, T::ZERO, T::ZERO, T::ZERO],
						wr,
						T::ZERO,
					);
					let mut x11 = xs[0];
					if xnorm > T::ONE && colnorm[j0] > bignum / xnorm {
						x11 = x11 / xnorm;
						scale = scale / xnorm;
					}
					if scale != T::ONE {
						unsafe { T::scale(xcol, scale, ki) };
					}
					unsafe {
						*xcol.add(j0) = x11;
						// x[0..j0] -= x11 · t[0..j0, j0]
						T::axpy(xcol, tp.add(j0 * tcs), x11, j0);
					}
					jj -= 1;
				} else {
					// 2×2 block at rows (j0-1, j0)
					let jt = j0 - 1;
					let a = [t[(jt, jt)], t[(j0, jt)], t[(jt, j0)], t[(j0, j0)]];
					let b = [
						unsafe { *xcol.add(jt) },
						unsafe { *xcol.add(j0) },
						T::ZERO,
						T::ZERO,
					];
					let (xs, mut scale, xnorm) = laln2(2, 1, smin, a, b, wr, T::ZERO);
					let (mut x1, mut x2) = (xs[0], xs[1]);
					if xnorm > T::ONE {
						let beta = colnorm[jt].maxs(colnorm[j0]);
						if beta > bignum / xnorm {
							x1 = x1 / xnorm;
							x2 = x2 / xnorm;
							scale = scale / xnorm;
						}
					}
					if scale != T::ONE {
						unsafe { T::scale(xcol, scale, ki) };
					}
					unsafe {
						*xcol.add(jt) = x1;
						*xcol.add(j0) = x2;
						T::axpy(xcol, tp.add(jt * tcs), x1, jt);
						T::axpy(xcol, tp.add(j0 * tcs), x2, jt);
					}
					jj -= 2;
				}
			}
			ki -= 1;
		} else {
			// ---- complex-conjugate pair at diagonal indices (kt, kb)
			let kt = kb - 1;
			let wr = t[(kb, kb)];
			let wi = t[(kb, kt)].abs().sqrt() * t[(kt, kb)].abs().sqrt();
			let smin = (ulp * (wr.abs() + wi)).maxs(smlnum);
			// real part in column kt, imaginary part in column kb
			let xr = unsafe { xp.add(kt * xcs) };
			let xi = unsafe { xp.add(kb * xcs) };
			unsafe {
				if t[(kt, kb)].abs() >= t[(kb, kt)].abs() {
					*xr.add(kt) = T::ONE;
					*xi.add(kb) = wi / t[(kt, kb)];
				} else {
					*xr.add(kt) = -wi / t[(kb, kt)];
					*xi.add(kb) = T::ONE;
				}
				*xr.add(kb) = T::ZERO;
				*xi.add(kt) = T::ZERO;
				let xrkt = *xr.add(kt);
				let xikb = *xi.add(kb);
				for r in 0..kt {
					*xr.add(r) = -xrkt * t[(r, kt)];
					*xi.add(r) = -xikb * t[(r, kb)];
				}
			}
			let mut jj = kt as isize - 1;
			while jj >= 0 {
				let j0 = jj as usize;
				let two_block = j0 > 0 && t[(j0, j0 - 1)] != T::ZERO;
				if !two_block {
					let b = [unsafe { *xr.add(j0) }, T::ZERO, unsafe { *xi.add(j0) }, T::ZERO];
					let (xs, mut scale, xnorm) = laln2(
						1,
						2,
						smin,
						[t[(j0, j0)], T::ZERO, T::ZERO, T::ZERO],
						b,
						wr,
						wi,
					);
					let (mut x11r, mut x11i) = (xs[0], xs[2]);
					if xnorm > T::ONE && colnorm[j0] > bignum / xnorm {
						x11r = x11r / xnorm;
						x11i = x11i / xnorm;
						scale = scale / xnorm;
					}
					if scale != T::ONE {
						unsafe {
							T::scale(xr, scale, ki);
							T::scale(xi, scale, ki);
						}
					}
					unsafe {
						*xr.add(j0) = x11r;
						*xi.add(j0) = x11i;
						T::axpy(xr, tp.add(j0 * tcs), x11r, j0);
						T::axpy(xi, tp.add(j0 * tcs), x11i, j0);
					}
					jj -= 1;
				} else {
					let jt = j0 - 1;
					let a = [t[(jt, jt)], t[(j0, jt)], t[(jt, j0)], t[(j0, j0)]];
					let b = [
						unsafe { *xr.add(jt) },
						unsafe { *xr.add(j0) },
						unsafe { *xi.add(jt) },
						unsafe { *xi.add(j0) },
					];
					let (xs, mut scale, xnorm) = laln2(2, 2, smin, a, b, wr, wi);
					let (mut x1r, mut x2r, mut x1i, mut x2i) = (xs[0], xs[1], xs[2], xs[3]);
					if xnorm > T::ONE {
						let beta = colnorm[jt].maxs(colnorm[j0]);
						if beta > bignum / xnorm {
							let rec = T::ONE / xnorm;
							x1r = x1r * rec;
							x2r = x2r * rec;
							x1i = x1i * rec;
							x2i = x2i * rec;
							scale = scale * rec;
						}
					}
					if scale != T::ONE {
						unsafe {
							T::scale(xr, scale, ki);
							T::scale(xi, scale, ki);
						}
					}
					unsafe {
						*xr.add(jt) = x1r;
						*xr.add(j0) = x2r;
						*xi.add(jt) = x1i;
						*xi.add(j0) = x2i;
						T::axpy(xr, tp.add(jt * tcs), x1r, jt);
						T::axpy(xr, tp.add(j0 * tcs), x2r, jt);
						T::axpy(xi, tp.add(jt * tcs), x1i, jt);
						T::axpy(xi, tp.add(j0 * tcs), x2i, jt);
					}
					jj -= 2;
				}
			}
			ki -= 2;
		}
	}

	// back-transform: V = Z · X, X upper triangular — the gemm-shaped bulk
	triangular::matmul(
		v.rb_mut(),
		BlockStructure::Rectangular,
		Accum::Replace,
		z,
		BlockStructure::Rectangular,
		x.as_ref(),
		BlockStructure::TriangularUpper,
		T::ONE,
		Par::Seq,
	);

	// dtrevc3 normalization: largest |re| (+ |im| for pairs) component → 1
	let vp = v.as_ptr_mut();
	let vcs = v.col_stride() as usize;
	let mut k = 0usize;
	while k < n {
		let pair = k + 1 < n && t[(k + 1, k)] != T::ZERO;
		if !pair {
			let mut emax = T::ZERO;
			for r in 0..n {
				emax = emax.maxs(v[(r, k)].abs());
			}
			if emax > T::ZERO {
				unsafe { T::scale(vp.add(k * vcs), T::ONE / emax, n) };
			}
			k += 1;
		} else {
			let mut emax = T::ZERO;
			for r in 0..n {
				emax = emax.maxs(v[(r, k)].abs() + v[(r, k + 1)].abs());
			}
			if emax > T::ZERO {
				let remax = T::ONE / emax;
				unsafe {
					T::scale(vp.add(k * vcs), remax, n);
					T::scale(vp.add((k + 1) * vcs), remax, n);
				}
			}
			k += 2;
		}
	}
}
