//! The scalar abstraction behind the wasm-shaped kernels (f32/c32 phase,
//! architect direction 2026-07-11).
//!
//! Every real kernel in this crate is generic over [`WasmScalar`]: the
//! driver logic is written once, and the hot flat-SIMD primitives (`axpy`,
//! `dot`, `scale`) are implemented per type at the natural lane width —
//! `f64x2` (2 lanes, 2× unrolled) and `f32x4` (4 lanes, 2× unrolled). f32
//! exists for the ~2× mechanism pair on wasm SIMD128: double the lanes for
//! compute-bound work, half the memory traffic for bandwidth-bound work,
//! at ~7 significant digits (`EPS` ≈ 6e-8). Complex kernels are c64-typed
//! rather than generic (see `hessenberg_cplx` / `schur_small_cplx` /
//! `eigvec_cplx`, with their simd128 primitives in `cplx`); c32 kernels
//! do not exist — c32 runs through faer's generic paths.
//!
//! The `RealField` supertrait is what lets the same generic code hand the
//! O(n³) bulk to `faer::linalg::matmul` and its friends.

use faer_traits::RealField;

/// Scalar for the flat wasm kernels: arithmetic + the SIMD primitives at
/// the type's natural lane width. Implemented for `f64` and `f32`.
pub trait WasmScalar:
	RealField
	+ Copy
	+ PartialOrd
	+ core::ops::Add<Output = Self>
	+ core::ops::Sub<Output = Self>
	+ core::ops::Mul<Output = Self>
	+ core::ops::Div<Output = Self>
	+ core::ops::Neg<Output = Self>
	+ core::ops::AddAssign
	+ core::ops::SubAssign
	+ core::ops::MulAssign
{
	const ZERO: Self;
	const ONE: Self;
	/// machine epsilon (LAPACK `ulp`)
	const EPS: Self;
	/// `MIN_POSITIVE / EPS` — LAPACK's `smlnum`-style deflation floor
	const SMALL_NUM: Self;
	/// smallest positive normal (LAPACK `safmin`)
	const MIN_POS: Self;
	/// largest finite value (LAPACK overflow threshold)
	const MAX_POS: Self;

	fn from_f64(x: f64) -> Self;
	fn abs(self) -> Self;
	fn sqrt(self) -> Self;
	fn maxs(self, o: Self) -> Self;
	fn mins(self, o: Self) -> Self;

	/// `dst[i] -= src[i] * alpha` over `len` contiguous elements
	unsafe fn axpy(dst: *mut Self, src: *const Self, alpha: Self, len: usize);
	/// `Σ a[i]·b[i]` over `len` contiguous elements
	unsafe fn dot(a: *const Self, b: *const Self, len: usize) -> Self;
	/// `dst[i] *= alpha` over `len` contiguous elements
	unsafe fn scale(dst: *mut Self, alpha: Self, len: usize);
}

impl WasmScalar for f64 {
	const ZERO: Self = 0.0;
	const ONE: Self = 1.0;
	const EPS: Self = f64::EPSILON;
	const SMALL_NUM: Self = f64::MIN_POSITIVE / f64::EPSILON;
	const MIN_POS: Self = f64::MIN_POSITIVE;
	const MAX_POS: Self = f64::MAX;

	#[inline(always)]
	fn from_f64(x: f64) -> Self {
		x
	}
	#[inline(always)]
	fn abs(self) -> Self {
		f64::abs(self)
	}
	#[inline(always)]
	fn sqrt(self) -> Self {
		libm::sqrt(self)
	}
	#[inline(always)]
	fn maxs(self, o: Self) -> Self {
		f64::max(self, o)
	}
	#[inline(always)]
	fn mins(self, o: Self) -> Self {
		f64::min(self, o)
	}

	#[inline(always)]
	unsafe fn axpy(dst: *mut Self, src: *const Self, alpha: Self, len: usize) {
		#[cfg(target_arch = "wasm32")]
		{
			axpy_f64x2(dst, src, alpha, len);
		}
		#[cfg(not(target_arch = "wasm32"))]
		{
			for i in 0..len {
				*dst.add(i) -= *src.add(i) * alpha;
			}
		}
	}
	#[inline(always)]
	unsafe fn dot(a: *const Self, b: *const Self, len: usize) -> Self {
		#[cfg(target_arch = "wasm32")]
		{
			dot_f64x2(a, b, len)
		}
		#[cfg(not(target_arch = "wasm32"))]
		{
			let mut s = 0.0;
			for i in 0..len {
				s += *a.add(i) * *b.add(i);
			}
			s
		}
	}
	#[inline(always)]
	unsafe fn scale(dst: *mut Self, alpha: Self, len: usize) {
		#[cfg(target_arch = "wasm32")]
		{
			scale_f64x2(dst, alpha, len);
		}
		#[cfg(not(target_arch = "wasm32"))]
		{
			for i in 0..len {
				*dst.add(i) *= alpha;
			}
		}
	}
}

impl WasmScalar for f32 {
	const ZERO: Self = 0.0;
	const ONE: Self = 1.0;
	const EPS: Self = f32::EPSILON;
	const SMALL_NUM: Self = f32::MIN_POSITIVE / f32::EPSILON;
	const MIN_POS: Self = f32::MIN_POSITIVE;
	const MAX_POS: Self = f32::MAX;

	#[inline(always)]
	fn from_f64(x: f64) -> Self {
		x as f32
	}
	#[inline(always)]
	fn abs(self) -> Self {
		f32::abs(self)
	}
	#[inline(always)]
	fn sqrt(self) -> Self {
		libm::sqrtf(self)
	}
	#[inline(always)]
	fn maxs(self, o: Self) -> Self {
		f32::max(self, o)
	}
	#[inline(always)]
	fn mins(self, o: Self) -> Self {
		f32::min(self, o)
	}

	#[inline(always)]
	unsafe fn axpy(dst: *mut Self, src: *const Self, alpha: Self, len: usize) {
		#[cfg(target_arch = "wasm32")]
		{
			axpy_f32x4(dst, src, alpha, len);
		}
		#[cfg(not(target_arch = "wasm32"))]
		{
			for i in 0..len {
				*dst.add(i) -= *src.add(i) * alpha;
			}
		}
	}
	#[inline(always)]
	unsafe fn dot(a: *const Self, b: *const Self, len: usize) -> Self {
		#[cfg(target_arch = "wasm32")]
		{
			dot_f32x4(a, b, len)
		}
		#[cfg(not(target_arch = "wasm32"))]
		{
			let mut s = 0.0;
			for i in 0..len {
				s += *a.add(i) * *b.add(i);
			}
			s
		}
	}
	#[inline(always)]
	unsafe fn scale(dst: *mut Self, alpha: Self, len: usize) {
		#[cfg(target_arch = "wasm32")]
		{
			scale_f32x4(dst, alpha, len);
		}
		#[cfg(not(target_arch = "wasm32"))]
		{
			for i in 0..len {
				*dst.add(i) *= alpha;
			}
		}
	}
}

// ---- f64x2 primitives (moved here from qr.rs; 2 lanes, 2× unrolled;
// v128_load/store are alignment-free by spec) ----

#[cfg(target_arch = "wasm32")]
#[target_feature(enable = "simd128")]
unsafe fn axpy_f64x2(dst: *mut f64, src: *const f64, alpha: f64, len: usize) {
	use core::arch::wasm32::*;
	let va = f64x2_splat(alpha);
	let mut i = 0usize;
	while i + 4 <= len {
		let d0 = v128_load(dst.add(i) as *const v128);
		let s0 = v128_load(src.add(i) as *const v128);
		let d1 = v128_load(dst.add(i + 2) as *const v128);
		let s1 = v128_load(src.add(i + 2) as *const v128);
		v128_store(dst.add(i) as *mut v128, f64x2_sub(d0, f64x2_mul(s0, va)));
		v128_store(dst.add(i + 2) as *mut v128, f64x2_sub(d1, f64x2_mul(s1, va)));
		i += 4;
	}
	while i < len {
		*dst.add(i) -= *src.add(i) * alpha;
		i += 1;
	}
}

#[cfg(target_arch = "wasm32")]
#[target_feature(enable = "simd128")]
unsafe fn dot_f64x2(a: *const f64, b: *const f64, len: usize) -> f64 {
	use core::arch::wasm32::*;
	let mut acc0 = f64x2_splat(0.0);
	let mut acc1 = f64x2_splat(0.0);
	let mut i = 0usize;
	while i + 4 <= len {
		let a0 = v128_load(a.add(i) as *const v128);
		let b0 = v128_load(b.add(i) as *const v128);
		let a1 = v128_load(a.add(i + 2) as *const v128);
		let b1 = v128_load(b.add(i + 2) as *const v128);
		acc0 = f64x2_add(acc0, f64x2_mul(a0, b0));
		acc1 = f64x2_add(acc1, f64x2_mul(a1, b1));
		i += 4;
	}
	let acc = f64x2_add(acc0, acc1);
	let mut s = f64x2_extract_lane::<0>(acc) + f64x2_extract_lane::<1>(acc);
	while i < len {
		s += *a.add(i) * *b.add(i);
		i += 1;
	}
	s
}

#[cfg(target_arch = "wasm32")]
#[target_feature(enable = "simd128")]
unsafe fn scale_f64x2(dst: *mut f64, alpha: f64, len: usize) {
	use core::arch::wasm32::*;
	let va = f64x2_splat(alpha);
	let mut i = 0usize;
	while i + 4 <= len {
		let d0 = v128_load(dst.add(i) as *const v128);
		let d1 = v128_load(dst.add(i + 2) as *const v128);
		v128_store(dst.add(i) as *mut v128, f64x2_mul(d0, va));
		v128_store(dst.add(i + 2) as *mut v128, f64x2_mul(d1, va));
		i += 4;
	}
	while i < len {
		*dst.add(i) *= alpha;
		i += 1;
	}
}

// ---- f32x4 primitives (4 lanes, 2× unrolled = 8 elements/iter) ----

#[cfg(target_arch = "wasm32")]
#[target_feature(enable = "simd128")]
unsafe fn axpy_f32x4(dst: *mut f32, src: *const f32, alpha: f32, len: usize) {
	use core::arch::wasm32::*;
	let va = f32x4_splat(alpha);
	let mut i = 0usize;
	while i + 8 <= len {
		let d0 = v128_load(dst.add(i) as *const v128);
		let s0 = v128_load(src.add(i) as *const v128);
		let d1 = v128_load(dst.add(i + 4) as *const v128);
		let s1 = v128_load(src.add(i + 4) as *const v128);
		v128_store(dst.add(i) as *mut v128, f32x4_sub(d0, f32x4_mul(s0, va)));
		v128_store(dst.add(i + 4) as *mut v128, f32x4_sub(d1, f32x4_mul(s1, va)));
		i += 8;
	}
	while i < len {
		*dst.add(i) -= *src.add(i) * alpha;
		i += 1;
	}
}

#[cfg(target_arch = "wasm32")]
#[target_feature(enable = "simd128")]
unsafe fn dot_f32x4(a: *const f32, b: *const f32, len: usize) -> f32 {
	use core::arch::wasm32::*;
	let mut acc0 = f32x4_splat(0.0);
	let mut acc1 = f32x4_splat(0.0);
	let mut i = 0usize;
	while i + 8 <= len {
		let a0 = v128_load(a.add(i) as *const v128);
		let b0 = v128_load(b.add(i) as *const v128);
		let a1 = v128_load(a.add(i + 4) as *const v128);
		let b1 = v128_load(b.add(i + 4) as *const v128);
		acc0 = f32x4_add(acc0, f32x4_mul(a0, b0));
		acc1 = f32x4_add(acc1, f32x4_mul(a1, b1));
		i += 8;
	}
	let acc = f32x4_add(acc0, acc1);
	let mut s = f32x4_extract_lane::<0>(acc)
		+ f32x4_extract_lane::<1>(acc)
		+ f32x4_extract_lane::<2>(acc)
		+ f32x4_extract_lane::<3>(acc);
	while i < len {
		s += *a.add(i) * *b.add(i);
		i += 1;
	}
	s
}

#[cfg(target_arch = "wasm32")]
#[target_feature(enable = "simd128")]
unsafe fn scale_f32x4(dst: *mut f32, alpha: f32, len: usize) {
	use core::arch::wasm32::*;
	let va = f32x4_splat(alpha);
	let mut i = 0usize;
	while i + 8 <= len {
		let d0 = v128_load(dst.add(i) as *const v128);
		let d1 = v128_load(dst.add(i + 4) as *const v128);
		v128_store(dst.add(i) as *mut v128, f32x4_mul(d0, va));
		v128_store(dst.add(i + 4) as *mut v128, f32x4_mul(d1, va));
		i += 8;
	}
	while i < len {
		*dst.add(i) *= alpha;
		i += 1;
	}
}

// ---- fused 3-column reflector apply (Schur campaign): the Z-update /
// widened-column-apply shape of the full-Schur hqr kernel. For each row i:
//   sum = c0[i] + v1·c1[i] + v2·c2[i]
//   c0[i] -= sum·t1;  c1[i] -= sum·t2;  c2[i] -= sum·t3
// Three contiguous column streams — elementwise across rows, so it
// vectorizes at the natural lane width (the mode-split instrumentation
// showed the Z updates are the dominant eigvals→Schur delta cost).

pub trait WasmScalarRefl: WasmScalar {
	/// fused reflector apply to three contiguous column streams (see module
	/// comment); `len` rows
	unsafe fn refl3(
		c0: *mut Self,
		c1: *mut Self,
		c2: *mut Self,
		v1: Self,
		v2: Self,
		t1: Self,
		t2: Self,
		t3: Self,
		len: usize,
	);
	/// two-column variant: `sum = c0[i] + v1·c1[i]`, `c0 -= sum·t1`,
	/// `c1 -= sum·t2`
	unsafe fn refl2(c0: *mut Self, c1: *mut Self, v1: Self, t1: Self, t2: Self, len: usize);
}

impl WasmScalarRefl for f64 {
	#[inline(always)]
	unsafe fn refl3(
		c0: *mut Self,
		c1: *mut Self,
		c2: *mut Self,
		v1: Self,
		v2: Self,
		t1: Self,
		t2: Self,
		t3: Self,
		len: usize,
	) {
		#[cfg(target_arch = "wasm32")]
		{
			refl3_f64x2(c0, c1, c2, v1, v2, t1, t2, t3, len);
		}
		#[cfg(not(target_arch = "wasm32"))]
		{
			for i in 0..len {
				let sum = *c0.add(i) + v1 * *c1.add(i) + v2 * *c2.add(i);
				*c0.add(i) -= sum * t1;
				*c1.add(i) -= sum * t2;
				*c2.add(i) -= sum * t3;
			}
		}
	}
	#[inline(always)]
	unsafe fn refl2(c0: *mut Self, c1: *mut Self, v1: Self, t1: Self, t2: Self, len: usize) {
		#[cfg(target_arch = "wasm32")]
		{
			refl2_f64x2(c0, c1, v1, t1, t2, len);
		}
		#[cfg(not(target_arch = "wasm32"))]
		{
			for i in 0..len {
				let sum = *c0.add(i) + v1 * *c1.add(i);
				*c0.add(i) -= sum * t1;
				*c1.add(i) -= sum * t2;
			}
		}
	}
}

impl WasmScalarRefl for f32 {
	#[inline(always)]
	unsafe fn refl3(
		c0: *mut Self,
		c1: *mut Self,
		c2: *mut Self,
		v1: Self,
		v2: Self,
		t1: Self,
		t2: Self,
		t3: Self,
		len: usize,
	) {
		#[cfg(target_arch = "wasm32")]
		{
			refl3_f32x4(c0, c1, c2, v1, v2, t1, t2, t3, len);
		}
		#[cfg(not(target_arch = "wasm32"))]
		{
			for i in 0..len {
				let sum = *c0.add(i) + v1 * *c1.add(i) + v2 * *c2.add(i);
				*c0.add(i) -= sum * t1;
				*c1.add(i) -= sum * t2;
				*c2.add(i) -= sum * t3;
			}
		}
	}
	#[inline(always)]
	unsafe fn refl2(c0: *mut Self, c1: *mut Self, v1: Self, t1: Self, t2: Self, len: usize) {
		#[cfg(target_arch = "wasm32")]
		{
			refl2_f32x4(c0, c1, v1, t1, t2, len);
		}
		#[cfg(not(target_arch = "wasm32"))]
		{
			for i in 0..len {
				let sum = *c0.add(i) + v1 * *c1.add(i);
				*c0.add(i) -= sum * t1;
				*c1.add(i) -= sum * t2;
			}
		}
	}
}

#[cfg(target_arch = "wasm32")]
#[target_feature(enable = "simd128")]
#[allow(clippy::too_many_arguments)]
unsafe fn refl3_f64x2(
	c0: *mut f64,
	c1: *mut f64,
	c2: *mut f64,
	v1: f64,
	v2: f64,
	t1: f64,
	t2: f64,
	t3: f64,
	len: usize,
) {
	use core::arch::wasm32::*;
	let vv1 = f64x2_splat(v1);
	let vv2 = f64x2_splat(v2);
	let vt1 = f64x2_splat(t1);
	let vt2 = f64x2_splat(t2);
	let vt3 = f64x2_splat(t3);
	let mut i = 0usize;
	while i + 2 <= len {
		let x0 = v128_load(c0.add(i) as *const v128);
		let x1 = v128_load(c1.add(i) as *const v128);
		let x2 = v128_load(c2.add(i) as *const v128);
		let sum = f64x2_add(x0, f64x2_add(f64x2_mul(vv1, x1), f64x2_mul(vv2, x2)));
		v128_store(c0.add(i) as *mut v128, f64x2_sub(x0, f64x2_mul(sum, vt1)));
		v128_store(c1.add(i) as *mut v128, f64x2_sub(x1, f64x2_mul(sum, vt2)));
		v128_store(c2.add(i) as *mut v128, f64x2_sub(x2, f64x2_mul(sum, vt3)));
		i += 2;
	}
	while i < len {
		let sum = *c0.add(i) + v1 * *c1.add(i) + v2 * *c2.add(i);
		*c0.add(i) -= sum * t1;
		*c1.add(i) -= sum * t2;
		*c2.add(i) -= sum * t3;
		i += 1;
	}
}

#[cfg(target_arch = "wasm32")]
#[target_feature(enable = "simd128")]
unsafe fn refl2_f64x2(c0: *mut f64, c1: *mut f64, v1: f64, t1: f64, t2: f64, len: usize) {
	use core::arch::wasm32::*;
	let vv1 = f64x2_splat(v1);
	let vt1 = f64x2_splat(t1);
	let vt2 = f64x2_splat(t2);
	let mut i = 0usize;
	while i + 2 <= len {
		let x0 = v128_load(c0.add(i) as *const v128);
		let x1 = v128_load(c1.add(i) as *const v128);
		let sum = f64x2_add(x0, f64x2_mul(vv1, x1));
		v128_store(c0.add(i) as *mut v128, f64x2_sub(x0, f64x2_mul(sum, vt1)));
		v128_store(c1.add(i) as *mut v128, f64x2_sub(x1, f64x2_mul(sum, vt2)));
		i += 2;
	}
	while i < len {
		let sum = *c0.add(i) + v1 * *c1.add(i);
		*c0.add(i) -= sum * t1;
		*c1.add(i) -= sum * t2;
		i += 1;
	}
}

#[cfg(target_arch = "wasm32")]
#[target_feature(enable = "simd128")]
#[allow(clippy::too_many_arguments)]
unsafe fn refl3_f32x4(
	c0: *mut f32,
	c1: *mut f32,
	c2: *mut f32,
	v1: f32,
	v2: f32,
	t1: f32,
	t2: f32,
	t3: f32,
	len: usize,
) {
	use core::arch::wasm32::*;
	let vv1 = f32x4_splat(v1);
	let vv2 = f32x4_splat(v2);
	let vt1 = f32x4_splat(t1);
	let vt2 = f32x4_splat(t2);
	let vt3 = f32x4_splat(t3);
	let mut i = 0usize;
	while i + 4 <= len {
		let x0 = v128_load(c0.add(i) as *const v128);
		let x1 = v128_load(c1.add(i) as *const v128);
		let x2 = v128_load(c2.add(i) as *const v128);
		let sum = f32x4_add(x0, f32x4_add(f32x4_mul(vv1, x1), f32x4_mul(vv2, x2)));
		v128_store(c0.add(i) as *mut v128, f32x4_sub(x0, f32x4_mul(sum, vt1)));
		v128_store(c1.add(i) as *mut v128, f32x4_sub(x1, f32x4_mul(sum, vt2)));
		v128_store(c2.add(i) as *mut v128, f32x4_sub(x2, f32x4_mul(sum, vt3)));
		i += 4;
	}
	while i < len {
		let sum = *c0.add(i) + v1 * *c1.add(i) + v2 * *c2.add(i);
		*c0.add(i) -= sum * t1;
		*c1.add(i) -= sum * t2;
		*c2.add(i) -= sum * t3;
		i += 1;
	}
}

#[cfg(target_arch = "wasm32")]
#[target_feature(enable = "simd128")]
unsafe fn refl2_f32x4(c0: *mut f32, c1: *mut f32, v1: f32, t1: f32, t2: f32, len: usize) {
	use core::arch::wasm32::*;
	let vv1 = f32x4_splat(v1);
	let vt1 = f32x4_splat(t1);
	let vt2 = f32x4_splat(t2);
	let mut i = 0usize;
	while i + 4 <= len {
		let x0 = v128_load(c0.add(i) as *const v128);
		let x1 = v128_load(c1.add(i) as *const v128);
		let sum = f32x4_add(x0, f32x4_mul(vv1, x1));
		v128_store(c0.add(i) as *mut v128, f32x4_sub(x0, f32x4_mul(sum, vt1)));
		v128_store(c1.add(i) as *mut v128, f32x4_sub(x1, f32x4_mul(sum, vt2)));
		i += 4;
	}
	while i < len {
		let sum = *c0.add(i) + v1 * *c1.add(i);
		*c0.add(i) -= sum * t1;
		*c1.add(i) -= sum * t2;
		i += 1;
	}
}
