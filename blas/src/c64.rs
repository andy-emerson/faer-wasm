//! `C64` — the double-precision complex scalar the z-routines operate
//! on. Defined here (zero dependencies) with a **fixed rounding order**
//! for every operation, so complex arithmetic is as deterministic as
//! the real layers: the SIMD paths reproduce these exact sequences
//! lane-for-lane (see `kernels.rs`), and native ↔ wasm bit-identity
//! holds by the same construction as everywhere else.
//!
//! Layout: `#[repr(C)]` `{ re, im }` — identical to C99 `double
//! _Complex`, Fortran `COMPLEX*16`, `num_complex::Complex64`, and
//! faer's `c64`, so consumers can cast slices instead of copying.
//! Internally the crate views `&[C64]` as `&[f64]` of twice the
//! length (sound: no padding, alignment 8) — that view is how the
//! elementwise/interleavable z-routines delegate to the tuned
//! d-routines.

/// Double-precision complex number (see module docs for layout and
/// rounding contracts).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct C64 {
	pub re: f64,
	pub im: f64,
}

impl C64 {
	pub const ZERO: C64 = C64 { re: 0.0, im: 0.0 };
	pub const ONE: C64 = C64 { re: 1.0, im: 0.0 };

	#[inline(always)]
	pub const fn new(re: f64, im: f64) -> Self {
		Self { re, im }
	}

	/// Complex conjugate (exact — sign flip only).
	#[inline(always)]
	pub const fn conj(self) -> Self {
		Self { re: self.re, im: -self.im }
	}

	/// The ℓ¹ magnitude |re| + |im| — the metric BLAS's `izamax` and
	/// `dzasum` use (one rounding; cheaper than the modulus and
	/// monotone enough for pivoting).
	#[inline(always)]
	pub fn abs1(self) -> f64 {
		self.re.abs() + self.im.abs()
	}

	/// The modulus √(re² + im²), overflow/underflow-safe (libm hypot —
	/// deterministic across targets, same code both sides).
	#[inline(always)]
	pub fn abs(self) -> f64 {
		libm::hypot(self.re, self.im)
	}

	/// Scale by a real (two independent roundings).
	#[inline(always)]
	pub fn scale(self, s: f64) -> Self {
		Self { re: self.re * s, im: self.im * s }
	}
}

/// The canonical product rounding order — re: (a·c) − (b·d), im:
/// (a·d) + (b·c), each parenthesis one rounding. The SIMD kernels
/// reproduce exactly this sequence (their sign-folded multiplies are
/// bit-exact rewrites: (−b)·d ≡ −(b·d), x − y ≡ x + (−y)).
impl core::ops::Mul for C64 {
	type Output = C64;
	#[inline(always)]
	fn mul(self, o: C64) -> C64 {
		C64 {
			re: self.re * o.re - self.im * o.im,
			im: self.re * o.im + self.im * o.re,
		}
	}
}

impl core::ops::Add for C64 {
	type Output = C64;
	#[inline(always)]
	fn add(self, o: C64) -> C64 {
		C64 { re: self.re + o.re, im: self.im + o.im }
	}
}

impl core::ops::Sub for C64 {
	type Output = C64;
	#[inline(always)]
	fn sub(self, o: C64) -> C64 {
		C64 { re: self.re - o.re, im: self.im - o.im }
	}
}

impl core::ops::Neg for C64 {
	type Output = C64;
	#[inline(always)]
	fn neg(self) -> C64 {
		C64 { re: -self.re, im: -self.im }
	}
}

/// Complex division by Smith's algorithm (the shape inside LAPACK's
/// `dladiv`): scale by the ratio of the divisor's larger component so
/// intermediate products can't overflow when the plain formula would.
/// Deterministic — one fixed branch on |c.re| vs |c.im|, then a fixed
/// rounding sequence per side.
impl core::ops::Div for C64 {
	type Output = C64;
	#[inline(always)]
	fn div(self, o: C64) -> C64 {
		if o.re.abs() >= o.im.abs() {
			let t = o.im / o.re;
			let d = o.re + o.im * t;
			C64 { re: (self.re + self.im * t) / d, im: (self.im - self.re * t) / d }
		} else {
			let t = o.re / o.im;
			let d = o.re * t + o.im;
			C64 { re: (self.re * t + self.im) / d, im: (self.im * t - self.re) / d }
		}
	}
}

/// View a complex slice as the interleaved real slice it is (re₀, im₀,
/// re₁, …) — the bridge the elementwise z-routines use to delegate to
/// the tuned d-streams. Sound: `C64` is `repr(C)` with no padding.
#[inline(always)]
pub(crate) fn as_re(x: &[C64]) -> &[f64] {
	unsafe { core::slice::from_raw_parts(x.as_ptr() as *const f64, 2 * x.len()) }
}

/// Mutable twin of [`as_re`].
#[inline(always)]
pub(crate) fn as_re_mut(x: &mut [C64]) -> &mut [f64] {
	unsafe { core::slice::from_raw_parts_mut(x.as_mut_ptr() as *mut f64, 2 * x.len()) }
}
