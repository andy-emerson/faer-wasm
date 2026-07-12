//! c64 helpers for the complex kernels (Schur campaign decision point
//! (e), 2026-07-11; eigenvector campaign 2026-07-12). Scalar building
//! blocks over `faer::c64` (= `num_complex::Complex<f64>`) plus the
//! simd128 streaming primitives that measurement earned: the `crot`
//! rotation applies (kept on same-machine A/B evidence, 1.17–1.25×
//! where measurement separates) and the `caxpy`/`cscale_re` streams of
//! the ctrevc back-substitution — one c64 per 128-bit lane. The O(n³)
//! bulk still routes through faer's c64 gemm (a measured 4.5–5.4× win).

use faer::c64;

/// |re| + |im| — LAPACK `CABS1`, the cheap magnitude used by all the
/// deflation tests
#[inline(always)]
pub fn cabs1(z: c64) -> f64 {
	z.re.abs() + z.im.abs()
}

/// modulus. The kernels only call this on deflation-scan and shift values
/// (never accumulating), and `rotg` does its own scaling, so the naive
/// formula's overflow window (~1e154) is irrelevant in our regime.
#[inline(always)]
pub fn cabs(z: c64) -> f64 {
	libm::sqrt(z.re * z.re + z.im * z.im)
}

#[inline(always)]
pub fn cabs2(z: c64) -> f64 {
	z.re * z.re + z.im * z.im
}

#[inline(always)]
pub fn conj(z: c64) -> c64 {
	c64::new(z.re, -z.im)
}

#[inline(always)]
pub fn cmul_real(z: c64, s: f64) -> c64 {
	c64::new(z.re * s, z.im * s)
}

/// principal complex square root
#[inline]
pub fn csqrt(z: c64) -> c64 {
	if z.re == 0.0 && z.im == 0.0 {
		return c64::new(0.0, 0.0);
	}
	let r = cabs(z);
	let re = libm::sqrt(0.5 * (r + z.re));
	let im = libm::sqrt(0.5 * (r - z.re));
	c64::new(re, if z.im < 0.0 { -im } else { im })
}

/// Complex Givens rotation, `zlartg`-shape — a port of faer's
/// `JacobiRotation::rotg` complex branch at the pinned base (same scaled
/// branches, so convergence behavior matches), returning `(c, s, r)` with
/// `c` real. Applying the rotation to `(a, b)` as
/// `x' = c·x − conj(s)·y, y' = s·x + c·y` gives `(r, 0)`.
///
/// One deliberate deviation, recorded in the ROADMAP upstream ledger:
/// faer returns `r = 1` when `b == 0` (its `lahqr` then writes that `r`
/// over the subdiagonal — wrong for a measure-zero input class); LAPACK
/// `zlartg` returns `r = a`, and so do we.
pub fn rotg(a: c64, b: c64) -> (f64, c64, c64) {
	if b.re == 0.0 && b.im == 0.0 {
		return (1.0, c64::new(0.0, 0.0), a);
	}
	let eps = f64::EPSILON;
	let sml = f64::MIN_POSITIVE;
	let big = 1.0 / sml;
	let rtmin = libm::sqrt(sml / eps);
	let rtmax = 1.0 / rtmin;
	let (c, s, r);
	if a.re == 0.0 && a.im == 0.0 {
		c = 0.0;
		let g1 = b.re.abs().max(b.im.abs());
		if g1 > rtmin && g1 < rtmax {
			let g2 = cabs2(b);
			let d = libm::sqrt(g2);
			s = cmul_real(conj(b), 1.0 / d);
			r = c64::new(d, 0.0);
		} else {
			let u = g1.max(sml).min(big);
			let uu = 1.0 / u;
			let gs = cmul_real(b, uu);
			let g2 = cabs2(gs);
			let d = libm::sqrt(g2);
			s = cmul_real(conj(gs), 1.0 / d);
			r = c64::new(d * u, 0.0);
		}
	} else {
		let f1 = a.re.abs().max(a.im.abs());
		let g1 = b.re.abs().max(b.im.abs());
		if f1 > rtmin && f1 < rtmax && g1 > rtmin && g1 < rtmax {
			let f2 = cabs2(a);
			let g2 = cabs2(b);
			let h2 = f2 + g2;
			let d = if f2 > rtmin && h2 < rtmax {
				libm::sqrt(f2 * h2)
			} else {
				libm::sqrt(f2) * libm::sqrt(h2)
			};
			let p = 1.0 / d;
			c = f2 * p;
			s = conj(b) * cmul_real(a, p);
			r = cmul_real(a, h2 * p);
		} else {
			let u = f1.max(g1).max(sml).min(big);
			let uu = 1.0 / u;
			let gs = cmul_real(b, uu);
			let g2 = cabs2(gs);
			let (f2, h2, w);
			let fs;
			if f1 * uu < rtmin {
				let v = f1.max(sml).min(big);
				let vv = 1.0 / v;
				w = v * uu;
				fs = cmul_real(a, vv);
				f2 = cabs2(fs);
				h2 = (f2 * w) * w + g2;
			} else {
				w = 1.0;
				fs = cmul_real(a, uu);
				f2 = cabs2(fs);
				h2 = f2 + g2;
			}
			let d = if f2 > rtmin && h2 < rtmax {
				libm::sqrt(f2 * h2)
			} else {
				libm::sqrt(f2) * libm::sqrt(h2)
			};
			let p = 1.0 / d;
			c = (f2 * p) * w;
			s = conj(gs) * cmul_real(fs, p);
			r = cmul_real(cmul_real(fs, h2 * p), u);
		}
	}
	// faer stores the rotation as {c, -conj(s)} and its left-apply computes
	// x' = c·x − conj(s_field)·y = c·x + s·y … our apply convention above
	// absorbs that: return s_field = −conj(s)
	(c, -conj(s), r)
}

// ---------------------------------------------------------------------------
// SIMD rotation applies (Schur close-out, 2026-07-11). One c64 fills one
// 128-bit lane, so a complex multiply is a splat/swap/multiply dance rather
// than a lane-parallel win — but it still halves the instruction count vs
// scalar (both components move per op), and the measured gap it closes is
// chqr's rotation applies vs faer's pulp-SIMD ones (the c64@256 loss).
//
// Core form (matches `rotg`'s convention; c is real):
//   x' = c·x − s·y
//   y' = conj(s)·x + c·y
// The chase's left-apply (x' = c·x − conj(s)·y ; y' = s·x + c·y) is the same
// form with `s` replaced by `conj(s)` — the wrappers below handle that.

/// `x' = c·x − s·y ; y' = conj(s)·x + c·y` over `len` elements of two
/// contiguous c64 streams (the right-apply / Z-apply shape).
#[inline(always)]
pub unsafe fn crot_streams(x: *mut c64, y: *mut c64, c: f64, s: c64, len: usize) {
	#[cfg(target_arch = "wasm32")]
	{
		crot_streams_simd(x, y, c, s, len);
	}
	#[cfg(not(target_arch = "wasm32"))]
	{
		let sc = conj(s);
		for i in 0..len {
			let xi = *x.add(i);
			let yi = *y.add(i);
			*x.add(i) = cmul_real(xi, c) - s * yi;
			*y.add(i) = sc * xi + cmul_real(yi, c);
		}
	}
}

/// Left-apply over a strided row pair: elements at `p + j·stride` (x) and
/// `p + j·stride + 1` (y) for `j in 0..len`; applies
/// `x' = c·x − conj(s)·y ; y' = s·x + c·y`.
#[inline(always)]
pub unsafe fn crot_row_pair(p: *mut c64, stride: usize, c: f64, s: c64, len: usize) {
	#[cfg(target_arch = "wasm32")]
	{
		crot_row_pair_simd(p, stride, c, s, len);
	}
	#[cfg(not(target_arch = "wasm32"))]
	{
		let sc = conj(s);
		for j in 0..len {
			let q = p.add(j * stride);
			let xi = *q;
			let yi = *q.add(1);
			*q = cmul_real(xi, c) - sc * yi;
			*q.add(1) = s * xi + cmul_real(yi, c);
		}
	}
}

#[cfg(target_arch = "wasm32")]
#[target_feature(enable = "simd128")]
unsafe fn crot_streams_simd(x: *mut c64, y: *mut c64, c: f64, s: c64, len: usize) {
	use core::arch::wasm32::*;
	// s·z      = s.re·(re,im) + (−s.im·im, s.im·re)
	// conj(s)·z = s.re·(re,im) + ( s.im·im, −s.im·re)
	let vc = f64x2_splat(c);
	let vsre = f64x2_splat(s.re);
	let vsim_n = f64x2(-s.im, s.im); // for s·z
	let vsim_p = f64x2(s.im, -s.im); // for conj(s)·z
	let mut i = 0usize;
	while i < len {
		let xv = v128_load(x.add(i) as *const v128);
		let yv = v128_load(y.add(i) as *const v128);
		let xsw = i64x2_shuffle::<1, 0>(xv, xv);
		let ysw = i64x2_shuffle::<1, 0>(yv, yv);
		let sy = f64x2_add(f64x2_mul(vsre, yv), f64x2_mul(vsim_n, ysw));
		let scx = f64x2_add(f64x2_mul(vsre, xv), f64x2_mul(vsim_p, xsw));
		v128_store(x.add(i) as *mut v128, f64x2_sub(f64x2_mul(vc, xv), sy));
		v128_store(y.add(i) as *mut v128, f64x2_add(scx, f64x2_mul(vc, yv)));
		i += 1;
	}
}

#[cfg(target_arch = "wasm32")]
#[target_feature(enable = "simd128")]
unsafe fn crot_row_pair_simd(p: *mut c64, stride: usize, c: f64, s: c64, len: usize) {
	use core::arch::wasm32::*;
	// left form = core form with s -> conj(s)
	let vc = f64x2_splat(c);
	let vsre = f64x2_splat(s.re);
	let vsim_n = f64x2(s.im, -s.im); // conj(s)·z
	let vsim_p = f64x2(-s.im, s.im); // conj(conj(s))·z = s·z
	let mut j = 0usize;
	while j < len {
		let q = p.add(j * stride);
		let xv = v128_load(q as *const v128);
		let yv = v128_load(q.add(1) as *const v128);
		let xsw = i64x2_shuffle::<1, 0>(xv, xv);
		let ysw = i64x2_shuffle::<1, 0>(yv, yv);
		let scy = f64x2_add(f64x2_mul(vsre, yv), f64x2_mul(vsim_n, ysw));
		let sx = f64x2_add(f64x2_mul(vsre, xv), f64x2_mul(vsim_p, xsw));
		v128_store(q as *mut v128, f64x2_sub(f64x2_mul(vc, xv), scy));
		v128_store(q.add(1) as *mut v128, f64x2_add(sx, f64x2_mul(vc, yv)));
		j += 1;
	}
}

// ---------------------------------------------------------------------------
// Streaming c64 primitives for the complex eigenvector kernel (eigenvector
// campaign, 2026-07-12): the back-substitution's column updates and the
// scale-by-real used by anti-overflow rescaling and normalization. Same
// one-c64-per-128-bit-lane encoding as the `crot` applies.

/// `dst[i] -= alpha · src[i]` over `len` contiguous c64 elements (the
/// complex axpy of the trevc column updates).
#[inline(always)]
pub unsafe fn caxpy(dst: *mut c64, src: *const c64, alpha: c64, len: usize) {
	#[cfg(target_arch = "wasm32")]
	{
		caxpy_simd(dst, src, alpha, len);
	}
	#[cfg(not(target_arch = "wasm32"))]
	{
		for i in 0..len {
			*dst.add(i) -= alpha * *src.add(i);
		}
	}
}

/// `dst[i] *= s` (real scale) over `len` contiguous c64 elements.
#[inline(always)]
pub unsafe fn cscale_re(dst: *mut c64, s: f64, len: usize) {
	#[cfg(target_arch = "wasm32")]
	{
		cscale_re_simd(dst, s, len);
	}
	#[cfg(not(target_arch = "wasm32"))]
	{
		for i in 0..len {
			*dst.add(i) = cmul_real(*dst.add(i), s);
		}
	}
}

#[cfg(target_arch = "wasm32")]
#[target_feature(enable = "simd128")]
unsafe fn caxpy_simd(dst: *mut c64, src: *const c64, alpha: c64, len: usize) {
	use core::arch::wasm32::*;
	// alpha·z = alpha.re·(re,im) + (−alpha.im·im, alpha.im·re)
	let vre = f64x2_splat(alpha.re);
	let vim = f64x2(-alpha.im, alpha.im);
	let mut i = 0usize;
	while i < len {
		let s = v128_load(src.add(i) as *const v128);
		let d = v128_load(dst.add(i) as *const v128);
		let ssw = i64x2_shuffle::<1, 0>(s, s);
		let az = f64x2_add(f64x2_mul(vre, s), f64x2_mul(vim, ssw));
		v128_store(dst.add(i) as *mut v128, f64x2_sub(d, az));
		i += 1;
	}
}

#[cfg(target_arch = "wasm32")]
#[target_feature(enable = "simd128")]
unsafe fn cscale_re_simd(dst: *mut c64, s: f64, len: usize) {
	use core::arch::wasm32::*;
	let vs = f64x2_splat(s);
	let mut i = 0usize;
	while i < len {
		let d = v128_load(dst.add(i) as *const v128);
		v128_store(dst.add(i) as *mut v128, f64x2_mul(d, vs));
		i += 1;
	}
}
