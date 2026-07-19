//! `C32` — the single-precision complex scalar the c-routines operate
//! on. Same contract as `C64` (see `c64.rs`): a **fixed rounding
//! order** per operation, reproduced bit-exactly by the SIMD lane
//! forms (two complexes per `F32x4` register — see `kernels.rs`), so
//! native ↔ wasm bit-identity holds by the same construction.
//!
//! Layout: `#[repr(C)]` `{ re, im }` — identical to C99 `float
//! _Complex`, Fortran `COMPLEX`, `num_complex::Complex32`, and
//! faer's c32; consumers cast slices. The crate views `&[C32]` as
//! `&[f32]` of twice the length for the delegating c-routines.

/// Single-precision complex number (see module docs for layout and
/// rounding contracts).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct C32 {
	pub re: f32,
	pub im: f32,
}

impl C32 {
	pub const ZERO: C32 = C32 { re: 0.0, im: 0.0 };
	pub const ONE: C32 = C32 { re: 1.0, im: 0.0 };

	#[inline(always)]
	pub const fn new(re: f32, im: f32) -> Self {
		Self { re, im }
	}

	/// Complex conjugate (exact — sign flip only).
	#[inline(always)]
	pub const fn conj(self) -> Self {
		Self { re: self.re, im: -self.im }
	}

	/// The ℓ¹ magnitude |re| + |im| — the `icamax`/`scasum` metric.
	#[inline(always)]
	pub fn abs1(self) -> f32 {
		self.re.abs() + self.im.abs()
	}

	/// The modulus √(re² + im²), overflow/underflow-safe (libm hypotf
	/// — deterministic across targets).
	#[inline(always)]
	pub fn abs(self) -> f32 {
		libm::hypotf(self.re, self.im)
	}

	/// Scale by a real (two independent roundings).
	#[inline(always)]
	pub fn scale(self, s: f32) -> Self {
		Self { re: self.re * s, im: self.im * s }
	}
}

/// The canonical product rounding order — identical shape to `C64`'s;
/// the SIMD kernels reproduce exactly this sequence.
impl core::ops::Mul for C32 {
	type Output = C32;
	#[inline(always)]
	fn mul(self, o: C32) -> C32 {
		C32 {
			re: self.re * o.re - self.im * o.im,
			im: self.re * o.im + self.im * o.re,
		}
	}
}

impl core::ops::Add for C32 {
	type Output = C32;
	#[inline(always)]
	fn add(self, o: C32) -> C32 {
		C32 { re: self.re + o.re, im: self.im + o.im }
	}
}

impl core::ops::Sub for C32 {
	type Output = C32;
	#[inline(always)]
	fn sub(self, o: C32) -> C32 {
		C32 { re: self.re - o.re, im: self.im - o.im }
	}
}

impl core::ops::Neg for C32 {
	type Output = C32;
	#[inline(always)]
	fn neg(self) -> C32 {
		C32 { re: -self.re, im: -self.im }
	}
}

/// Complex division by Smith's algorithm — same guarded shape as
/// `C64`'s, in f32.
impl core::ops::Div for C32 {
	type Output = C32;
	#[inline(always)]
	fn div(self, o: C32) -> C32 {
		if o.re.abs() >= o.im.abs() {
			let t = o.im / o.re;
			let d = o.re + o.im * t;
			C32 { re: (self.re + self.im * t) / d, im: (self.im - self.re * t) / d }
		} else {
			let t = o.re / o.im;
			let d = o.re * t + o.im;
			C32 { re: (self.re * t + self.im) / d, im: (self.im * t - self.re) / d }
		}
	}
}

/// View a complex slice as the interleaved real slice it is — the
/// bridge for the delegating c-routines. Sound: `C32` is `repr(C)`
/// with no padding.
#[inline(always)]
pub(crate) fn as_re(x: &[C32]) -> &[f32] {
	unsafe { core::slice::from_raw_parts(x.as_ptr() as *const f32, 2 * x.len()) }
}

/// Mutable twin of [`as_re`].
#[inline(always)]
pub(crate) fn as_re_mut(x: &mut [C32]) -> &mut [f32] {
	unsafe { core::slice::from_raw_parts_mut(x.as_mut_ptr() as *mut f32, 2 * x.len()) }
}
