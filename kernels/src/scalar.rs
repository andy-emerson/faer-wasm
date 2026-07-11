//! The scalar abstraction behind the wasm-shaped kernels (f32/c32 phase,
//! architect direction 2026-07-11).
//!
//! Every kernel in this crate is generic over [`WasmScalar`]: the driver
//! logic is written once, and the hot flat-SIMD primitives (`axpy`, `dot`,
//! `scale`) are implemented per type at the natural lane width — `f64x2`
//! (2 lanes, 2× unrolled) and `f32x4` (4 lanes, 2× unrolled). f32 exists
//! for the ~2× mechanism pair on wasm SIMD128: double the lanes for
//! compute-bound work, half the memory traffic for bandwidth-bound work,
//! at ~7 significant digits (`EPS` ≈ 6e-8). Complex kernels do not exist
//! yet for either width; c32/c64 run through faer's generic paths.
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
