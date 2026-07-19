//! Tuned multi-stream kernels shared across levels and types (tuning
//! campaign, 2026-07; d-prefixed = f64 on F64x2 lanes, s-prefixed =
//! f32 on F32x4 lanes — same shapes at each lane width): the loop
//! shapes that survived racing on the reference runners. `daxpy4` fans one source column OUT into four destinations
//! (the gemm `col4` shape — cuts source traffic 4×); `daxpy4in` fans
//! four source columns IN to one destination (cuts destination
//! read-modify-write traffic 4×). Both preserve the per-element
//! rounding sequence of the four plain `axpy` passes they replace, so
//! callers stay bit-for-bit interchangeable with their column-axpy
//! references.

use crate::lanes::F64x2;

/// One sequential pass over a source column updating four destination
/// columns: cᵤ[i] += a[i]·tᵤ.
///
/// # Safety
/// All five column pointers must be valid for `len` f64s; the
/// destination columns must not alias each other or `a`.
#[cfg_attr(target_arch = "wasm32", target_feature(enable = "simd128"))]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn daxpy4(
	a: *const f64,
	t: [f64; 4],
	c0: *mut f64,
	c1: *mut f64,
	c2: *mut f64,
	c3: *mut f64,
	len: usize,
) {
	let v = [
		F64x2::splat(t[0]),
		F64x2::splat(t[1]),
		F64x2::splat(t[2]),
		F64x2::splat(t[3]),
	];
	let mut i = 0usize;
	while i + 2 <= len {
		let av = F64x2::load(a.add(i));
		F64x2::load(c0.add(i)).add(av.mul(v[0])).store(c0.add(i));
		F64x2::load(c1.add(i)).add(av.mul(v[1])).store(c1.add(i));
		F64x2::load(c2.add(i)).add(av.mul(v[2])).store(c2.add(i));
		F64x2::load(c3.add(i)).add(av.mul(v[3])).store(c3.add(i));
		i += 2;
	}
	while i < len {
		let av = *a.add(i);
		*c0.add(i) += av * t[0];
		*c1.add(i) += av * t[1];
		*c2.add(i) += av * t[2];
		*c3.add(i) += av * t[3];
		i += 1;
	}
}

/// Fused symv column pass (tuned 2026-07-19): one load of `a` per
/// element serves both halves of the symmetric update —
/// y[i] += t·a[i] (elementwise) and acc += a[i]·x[i] (reduction,
/// two lane-pair accumulators) — where the plain shape streamed the
/// column twice (`axpy` + `dot`). Returns the dot part. The reduction
/// order differs from `dot`'s 4-accumulator fold, which is fine: symv
/// is bounds-tested, not bit-locked, and determinism holds through
/// the lane emulation as everywhere else.
///
/// # Safety
/// `a`, `x`, `y` must each be valid for `len` f64s; `y` must not
/// alias `a` or `x`.
#[cfg_attr(target_arch = "wasm32", target_feature(enable = "simd128"))]
pub(crate) unsafe fn daxpy_dot(
	a: *const f64,
	t: f64,
	x: *const f64,
	y: *mut f64,
	len: usize,
) -> f64 {
	let tv = F64x2::splat(t);
	let mut acc0 = F64x2::splat(0.0);
	let mut acc1 = F64x2::splat(0.0);
	let mut i = 0usize;
	while i + 4 <= len {
		let a0 = F64x2::load(a.add(i));
		let a1 = F64x2::load(a.add(i + 2));
		F64x2::load(y.add(i)).add(a0.mul(tv)).store(y.add(i));
		F64x2::load(y.add(i + 2)).add(a1.mul(tv)).store(y.add(i + 2));
		acc0 = acc0.add(a0.mul(F64x2::load(x.add(i))));
		acc1 = acc1.add(a1.mul(F64x2::load(x.add(i + 2))));
		i += 4;
	}
	let f = acc0.add(acc1);
	let mut s = f.lane0() + f.lane1();
	while i < len {
		let av = *a.add(i);
		*y.add(i) += av * t;
		s += av * *x.add(i);
		i += 1;
	}
	s
}

/// Four fused symv column passes sharing one stream over x and y:
/// y[i] += Σᵤ tᵤ·aᵤ[i] (accumulated in u order into the loaded y
/// value) while each accᵤ += aᵤ[i]·x[i]. One y load/store and one x
/// load per pair of elements serve four columns. Returns the four dot
/// parts.
///
/// # Safety
/// All six array pointers must be valid for `len` f64s; `y` must not
/// alias any `aᵤ` or `x`.
#[cfg_attr(target_arch = "wasm32", target_feature(enable = "simd128"))]
pub(crate) unsafe fn daxpy_dot4(
	a: [*const f64; 4],
	t: [f64; 4],
	x: *const f64,
	y: *mut f64,
	len: usize,
) -> [f64; 4] {
	let tv = [
		F64x2::splat(t[0]),
		F64x2::splat(t[1]),
		F64x2::splat(t[2]),
		F64x2::splat(t[3]),
	];
	let mut acc = [F64x2::splat(0.0); 4];
	let mut i = 0usize;
	while i + 2 <= len {
		let xv = F64x2::load(x.add(i));
		let mut yv = F64x2::load(y.add(i));
		for u in 0..4 {
			let av = F64x2::load(a[u].add(i));
			yv = yv.add(av.mul(tv[u]));
			acc[u] = acc[u].add(av.mul(xv));
		}
		yv.store(y.add(i));
		i += 2;
	}
	let mut s = [
		acc[0].lane0() + acc[0].lane1(),
		acc[1].lane0() + acc[1].lane1(),
		acc[2].lane0() + acc[2].lane1(),
		acc[3].lane0() + acc[3].lane1(),
	];
	while i < len {
		let xi = *x.add(i);
		let mut yi = *y.add(i);
		for u in 0..4 {
			let av = *a[u].add(i);
			yi += av * t[u];
			s[u] += av * xi;
		}
		*y.add(i) = yi;
		i += 1;
	}
	s
}

/// One pass over the destination with four source columns fanned in:
/// c[i] ← ((((c[i] + a0[i]·t0) + a1[i]·t1) + a2[i]·t2) + a3[i]·t3) —
/// the same rounding sequence as four consecutive plain `axpy` passes
/// in that order, with the destination loaded and stored once.
///
/// # Safety
/// All five column pointers must be valid for `len` f64s; the source
/// columns must not alias the destination.
#[cfg_attr(target_arch = "wasm32", target_feature(enable = "simd128"))]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn daxpy4in(
	a0: *const f64,
	a1: *const f64,
	a2: *const f64,
	a3: *const f64,
	t: [f64; 4],
	c: *mut f64,
	len: usize,
) {
	let v = [
		F64x2::splat(t[0]),
		F64x2::splat(t[1]),
		F64x2::splat(t[2]),
		F64x2::splat(t[3]),
	];
	let mut i = 0usize;
	while i + 2 <= len {
		let cv = F64x2::load(c.add(i))
			.add(F64x2::load(a0.add(i)).mul(v[0]))
			.add(F64x2::load(a1.add(i)).mul(v[1]))
			.add(F64x2::load(a2.add(i)).mul(v[2]))
			.add(F64x2::load(a3.add(i)).mul(v[3]));
		cv.store(c.add(i));
		i += 2;
	}
	while i < len {
		let cv = (((*c.add(i) + *a0.add(i) * t[0]) + *a1.add(i) * t[1]) + *a2.add(i) * t[2])
			+ *a3.add(i) * t[3];
		*c.add(i) = cv;
		i += 1;
	}
}

// ---- c64 kernels (z-prefixed, one complex per F64x2 register) ----
//
// A complex multiply by a fixed scalar t is two lane multiplies and
// one add: with vre = [t.re, t.re] and vim = [−t.im, t.im],
// x·t = x·vre + swap(x)·vim — lane0 gives t.re·x.re + (−t.im)·x.im,
// lane1 gives t.re·x.im + t.im·x.re, bit-exactly the canonical C64
// product order (sign-folding is exact; see c64.rs). Every complex
// element is one whole v128, so there are no ragged lane tails.
use crate::c64::C64;

/// One sequential pass over a source column updating four destination
/// columns: cᵤ[i] += a[i]·tᵤ (complex).
///
/// # Safety
/// All five column pointers must be valid for `len` C64s; the
/// destination columns must not alias each other or `a`.
#[cfg_attr(target_arch = "wasm32", target_feature(enable = "simd128"))]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn zaxpy4(
	a: *const C64,
	t: [C64; 4],
	c0: *mut C64,
	c1: *mut C64,
	c2: *mut C64,
	c3: *mut C64,
	len: usize,
) {
	let vre = [
		F64x2::splat(t[0].re),
		F64x2::splat(t[1].re),
		F64x2::splat(t[2].re),
		F64x2::splat(t[3].re),
	];
	let vim = [
		F64x2::pair(-t[0].im, t[0].im),
		F64x2::pair(-t[1].im, t[1].im),
		F64x2::pair(-t[2].im, t[2].im),
		F64x2::pair(-t[3].im, t[3].im),
	];
	let ap = a as *const f64;
	let cp = [c0 as *mut f64, c1 as *mut f64, c2 as *mut f64, c3 as *mut f64];
	for i in 0..len {
		let av = F64x2::load(ap.add(2 * i));
		let asw = av.swap();
		for u in 0..4 {
			F64x2::load(cp[u].add(2 * i))
				.add(av.mul(vre[u]).add(asw.mul(vim[u])))
				.store(cp[u].add(2 * i));
		}
	}
}

/// One pass over the destination with four source columns fanned in:
/// c[i] ← ((((c[i] + a0[i]·t0) + a1[i]·t1) + a2[i]·t2) + a3[i]·t3)
/// (complex; each product fully formed before its add) — the same
/// rounding sequence as four consecutive plain `zaxpy` passes in that
/// order.
///
/// # Safety
/// All five column pointers must be valid for `len` C64s; the source
/// columns must not alias the destination.
#[cfg_attr(target_arch = "wasm32", target_feature(enable = "simd128"))]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn zaxpy4in(
	a0: *const C64,
	a1: *const C64,
	a2: *const C64,
	a3: *const C64,
	t: [C64; 4],
	c: *mut C64,
	len: usize,
) {
	let vre = [
		F64x2::splat(t[0].re),
		F64x2::splat(t[1].re),
		F64x2::splat(t[2].re),
		F64x2::splat(t[3].re),
	];
	let vim = [
		F64x2::pair(-t[0].im, t[0].im),
		F64x2::pair(-t[1].im, t[1].im),
		F64x2::pair(-t[2].im, t[2].im),
		F64x2::pair(-t[3].im, t[3].im),
	];
	let ap = [a0 as *const f64, a1 as *const f64, a2 as *const f64, a3 as *const f64];
	let cp = c as *mut f64;
	for i in 0..len {
		let mut cv = F64x2::load(cp.add(2 * i));
		for u in 0..4 {
			let av = F64x2::load(ap[u].add(2 * i));
			cv = cv.add(av.mul(vre[u]).add(av.swap().mul(vim[u])));
		}
		cv.store(cp.add(2 * i));
	}
}

/// Fused zhemv column pass: one load of `a` per element serves both
/// halves of the Hermitian update — y[i] += t·a[i] (elementwise) and
/// acc += conj(a[i])·x[i] (reduction, two register accumulators).
/// The conjugated product is dup0(a)·x + neg1(dup1(a)·swap(x)):
/// lane0 = a.re·x.re + a.im·x.im, lane1 = a.re·x.im − a.im·x.re —
/// bit-exactly `a.conj() * x` in the canonical order. Returns the
/// accumulated dot part. Fold order (acc0+acc1, then scalar tail)
/// differs from a sequential loop — zhemv is bounds-tested, not
/// bit-locked, and determinism holds through the lane emulation as
/// everywhere else.
///
/// # Safety
/// `a`, `x`, `y` must each be valid for `len` C64s; `y` must not
/// alias `a` or `x`.
#[cfg_attr(target_arch = "wasm32", target_feature(enable = "simd128"))]
pub(crate) unsafe fn zaxpy_dotc(
	a: *const C64,
	t: C64,
	x: *const C64,
	y: *mut C64,
	len: usize,
) -> C64 {
	let vre = F64x2::splat(t.re);
	let vim = F64x2::pair(-t.im, t.im);
	let ap = a as *const f64;
	let xp = x as *const f64;
	let yp = y as *mut f64;
	let mut acc0 = F64x2::splat(0.0);
	let mut acc1 = F64x2::splat(0.0);
	let mut i = 0usize;
	while i + 2 <= len {
		let a0 = F64x2::load(ap.add(2 * i));
		let a1 = F64x2::load(ap.add(2 * i + 2));
		F64x2::load(yp.add(2 * i))
			.add(a0.mul(vre).add(a0.swap().mul(vim)))
			.store(yp.add(2 * i));
		F64x2::load(yp.add(2 * i + 2))
			.add(a1.mul(vre).add(a1.swap().mul(vim)))
			.store(yp.add(2 * i + 2));
		let x0 = F64x2::load(xp.add(2 * i));
		let x1 = F64x2::load(xp.add(2 * i + 2));
		acc0 = acc0.add(a0.dup0().mul(x0).add(a0.dup1().mul(x0.swap()).neg1()));
		acc1 = acc1.add(a1.dup0().mul(x1).add(a1.dup1().mul(x1.swap()).neg1()));
		i += 2;
	}
	let f = acc0.add(acc1);
	let mut s = C64::new(f.lane0(), f.lane1());
	while i < len {
		let av = *a.add(i);
		*y.add(i) = *y.add(i) + t * av;
		s = s + av.conj() * *x.add(i);
		i += 1;
	}
	s
}

// ---- f32 twins (s-prefixed, F32x4 lanes) ----
use crate::lanes::F32x4;

/// One sequential pass over a source column updating four destination
/// columns: cᵤ[i] += a[i]·tᵤ.
///
/// # Safety
/// All five column pointers must be valid for `len` f32s; the
/// destination columns must not alias each other or `a`.
#[cfg_attr(target_arch = "wasm32", target_feature(enable = "simd128"))]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn saxpy4(
	a: *const f32,
	t: [f32; 4],
	c0: *mut f32,
	c1: *mut f32,
	c2: *mut f32,
	c3: *mut f32,
	len: usize,
) {
	let v = [
		F32x4::splat(t[0]),
		F32x4::splat(t[1]),
		F32x4::splat(t[2]),
		F32x4::splat(t[3]),
	];
	let mut i = 0usize;
	while i + 4 <= len {
		let av = F32x4::load(a.add(i));
		F32x4::load(c0.add(i)).add(av.mul(v[0])).store(c0.add(i));
		F32x4::load(c1.add(i)).add(av.mul(v[1])).store(c1.add(i));
		F32x4::load(c2.add(i)).add(av.mul(v[2])).store(c2.add(i));
		F32x4::load(c3.add(i)).add(av.mul(v[3])).store(c3.add(i));
		i += 4;
	}
	while i < len {
		let av = *a.add(i);
		*c0.add(i) += av * t[0];
		*c1.add(i) += av * t[1];
		*c2.add(i) += av * t[2];
		*c3.add(i) += av * t[3];
		i += 1;
	}
}

/// Fused symv column pass (tuned 2026-07-19): one load of `a` per
/// element serves both halves of the symmetric update —
/// y[i] += t·a[i] (elementwise) and acc += a[i]·x[i] (reduction,
/// two lane-pair accumulators) — where the plain shape streamed the
/// column twice (`axpy` + `dot`). Returns the dot part. The reduction
/// order differs from `dot`'s 4-accumulator fold, which is fine: symv
/// is bounds-tested, not bit-locked, and determinism holds through
/// the lane emulation as everywhere else.
///
/// # Safety
/// `a`, `x`, `y` must each be valid for `len` f32s; `y` must not
/// alias `a` or `x`.
#[cfg_attr(target_arch = "wasm32", target_feature(enable = "simd128"))]
pub(crate) unsafe fn saxpy_dot(
	a: *const f32,
	t: f32,
	x: *const f32,
	y: *mut f32,
	len: usize,
) -> f32 {
	let tv = F32x4::splat(t);
	let mut acc0 = F32x4::splat(0.0);
	let mut acc1 = F32x4::splat(0.0);
	let mut i = 0usize;
	while i + 8 <= len {
		let a0 = F32x4::load(a.add(i));
		let a1 = F32x4::load(a.add(i + 4));
		F32x4::load(y.add(i)).add(a0.mul(tv)).store(y.add(i));
		F32x4::load(y.add(i + 4)).add(a1.mul(tv)).store(y.add(i + 4));
		acc0 = acc0.add(a0.mul(F32x4::load(x.add(i))));
		acc1 = acc1.add(a1.mul(F32x4::load(x.add(i + 4))));
		i += 8;
	}
	let f = acc0.add(acc1);
	let mut s = (f.lane0() + f.lane1()) + (f.lane2() + f.lane3());
	while i < len {
		let av = *a.add(i);
		*y.add(i) += av * t;
		s += av * *x.add(i);
		i += 1;
	}
	s
}

/// Four fused symv column passes sharing one stream over x and y:
/// y[i] += Σᵤ tᵤ·aᵤ[i] (accumulated in u order into the loaded y
/// value) while each accᵤ += aᵤ[i]·x[i]. One y load/store and one x
/// load per pair of elements serve four columns. Returns the four dot
/// parts.
///
/// # Safety
/// All six array pointers must be valid for `len` f32s; `y` must not
/// alias any `aᵤ` or `x`.
#[cfg_attr(target_arch = "wasm32", target_feature(enable = "simd128"))]
pub(crate) unsafe fn saxpy_dot4(
	a: [*const f32; 4],
	t: [f32; 4],
	x: *const f32,
	y: *mut f32,
	len: usize,
) -> [f32; 4] {
	let tv = [
		F32x4::splat(t[0]),
		F32x4::splat(t[1]),
		F32x4::splat(t[2]),
		F32x4::splat(t[3]),
	];
	let mut acc = [F32x4::splat(0.0); 4];
	let mut i = 0usize;
	while i + 4 <= len {
		let xv = F32x4::load(x.add(i));
		let mut yv = F32x4::load(y.add(i));
		for u in 0..4 {
			let av = F32x4::load(a[u].add(i));
			yv = yv.add(av.mul(tv[u]));
			acc[u] = acc[u].add(av.mul(xv));
		}
		yv.store(y.add(i));
		i += 4;
	}
	let mut s = [
		(acc[0].lane0() + acc[0].lane1()) + (acc[0].lane2() + acc[0].lane3()),
		(acc[1].lane0() + acc[1].lane1()) + (acc[1].lane2() + acc[1].lane3()),
		(acc[2].lane0() + acc[2].lane1()) + (acc[2].lane2() + acc[2].lane3()),
		(acc[3].lane0() + acc[3].lane1()) + (acc[3].lane2() + acc[3].lane3()),
	];
	while i < len {
		let xi = *x.add(i);
		let mut yi = *y.add(i);
		for u in 0..4 {
			let av = *a[u].add(i);
			yi += av * t[u];
			s[u] += av * xi;
		}
		*y.add(i) = yi;
		i += 1;
	}
	s
}

/// One pass over the destination with four source columns fanned in:
/// c[i] ← ((((c[i] + a0[i]·t0) + a1[i]·t1) + a2[i]·t2) + a3[i]·t3) —
/// the same rounding sequence as four consecutive plain `axpy` passes
/// in that order, with the destination loaded and stored once.
///
/// # Safety
/// All five column pointers must be valid for `len` f32s; the source
/// columns must not alias the destination.
#[cfg_attr(target_arch = "wasm32", target_feature(enable = "simd128"))]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn saxpy4in(
	a0: *const f32,
	a1: *const f32,
	a2: *const f32,
	a3: *const f32,
	t: [f32; 4],
	c: *mut f32,
	len: usize,
) {
	let v = [
		F32x4::splat(t[0]),
		F32x4::splat(t[1]),
		F32x4::splat(t[2]),
		F32x4::splat(t[3]),
	];
	let mut i = 0usize;
	while i + 4 <= len {
		let cv = F32x4::load(c.add(i))
			.add(F32x4::load(a0.add(i)).mul(v[0]))
			.add(F32x4::load(a1.add(i)).mul(v[1]))
			.add(F32x4::load(a2.add(i)).mul(v[2]))
			.add(F32x4::load(a3.add(i)).mul(v[3]));
		cv.store(c.add(i));
		i += 4;
	}
	while i < len {
		let cv = (((*c.add(i) + *a0.add(i) * t[0]) + *a1.add(i) * t[1]) + *a2.add(i) * t[2])
			+ *a3.add(i) * t[3];
		*c.add(i) = cv;
		i += 1;
	}
}

// ---- c32 kernels (c-prefixed, TWO complexes per F32x4 register,
// packed [re0, im0, re1, im1]) ----
//
// Same product form as the z-kernels at pair granularity: with
// vre = splat(t.re) and vim = [−t.im, t.im, −t.im, t.im],
// x·t = x·vre + swap_pairs(x)·vim — bit-exactly the canonical C32
// product order per complex (sign-folding is exact; see c32.rs).
// Odd lengths leave a one-complex scalar tail computed with C32 ops
// (the same rounding sequence).
use crate::c32::C32;

/// One sequential pass over a source column updating four destination
/// columns: cᵤ[i] += a[i]·tᵤ (complex f32).
///
/// # Safety
/// All five column pointers must be valid for `len` C32s; the
/// destination columns must not alias each other or `a`.
#[cfg_attr(target_arch = "wasm32", target_feature(enable = "simd128"))]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn caxpy4(
	a: *const C32,
	t: [C32; 4],
	c0: *mut C32,
	c1: *mut C32,
	c2: *mut C32,
	c3: *mut C32,
	len: usize,
) {
	let vre = [
		F32x4::splat(t[0].re),
		F32x4::splat(t[1].re),
		F32x4::splat(t[2].re),
		F32x4::splat(t[3].re),
	];
	let vim = [
		F32x4::quad(-t[0].im, t[0].im, -t[0].im, t[0].im),
		F32x4::quad(-t[1].im, t[1].im, -t[1].im, t[1].im),
		F32x4::quad(-t[2].im, t[2].im, -t[2].im, t[2].im),
		F32x4::quad(-t[3].im, t[3].im, -t[3].im, t[3].im),
	];
	let ap = a as *const f32;
	let cp = [c0 as *mut f32, c1 as *mut f32, c2 as *mut f32, c3 as *mut f32];
	let mut i = 0usize;
	while i + 2 <= len {
		let av = F32x4::load(ap.add(2 * i));
		let asw = av.swap_pairs();
		for u in 0..4 {
			F32x4::load(cp[u].add(2 * i))
				.add(av.mul(vre[u]).add(asw.mul(vim[u])))
				.store(cp[u].add(2 * i));
		}
		i += 2;
	}
	if i < len {
		let av = *a.add(i);
		*c0.add(i) = *c0.add(i) + t[0] * av;
		*c1.add(i) = *c1.add(i) + t[1] * av;
		*c2.add(i) = *c2.add(i) + t[2] * av;
		*c3.add(i) = *c3.add(i) + t[3] * av;
	}
}

/// One pass over the destination with four source columns fanned in:
/// c[i] ← ((((c[i] + a0[i]·t0) + a1[i]·t1) + a2[i]·t2) + a3[i]·t3)
/// (complex f32; each product fully formed before its add) — the same
/// rounding sequence as four consecutive plain `caxpy` passes in that
/// order.
///
/// # Safety
/// All five column pointers must be valid for `len` C32s; the source
/// columns must not alias the destination.
#[cfg_attr(target_arch = "wasm32", target_feature(enable = "simd128"))]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn caxpy4in(
	a0: *const C32,
	a1: *const C32,
	a2: *const C32,
	a3: *const C32,
	t: [C32; 4],
	c: *mut C32,
	len: usize,
) {
	let vre = [
		F32x4::splat(t[0].re),
		F32x4::splat(t[1].re),
		F32x4::splat(t[2].re),
		F32x4::splat(t[3].re),
	];
	let vim = [
		F32x4::quad(-t[0].im, t[0].im, -t[0].im, t[0].im),
		F32x4::quad(-t[1].im, t[1].im, -t[1].im, t[1].im),
		F32x4::quad(-t[2].im, t[2].im, -t[2].im, t[2].im),
		F32x4::quad(-t[3].im, t[3].im, -t[3].im, t[3].im),
	];
	let ap = [a0 as *const f32, a1 as *const f32, a2 as *const f32, a3 as *const f32];
	let cp = c as *mut f32;
	let mut i = 0usize;
	while i + 2 <= len {
		let mut cv = F32x4::load(cp.add(2 * i));
		for u in 0..4 {
			let av = F32x4::load(ap[u].add(2 * i));
			cv = cv.add(av.mul(vre[u]).add(av.swap_pairs().mul(vim[u])));
		}
		cv.store(cp.add(2 * i));
		i += 2;
	}
	if i < len {
		let cv = (((*c.add(i) + t[0] * *a0.add(i)) + t[1] * *a1.add(i)) + t[2] * *a2.add(i))
			+ t[3] * *a3.add(i);
		*c.add(i) = cv;
	}
}

/// Fused chemv column pass: y[i] += t·a[i] while acc += conj(a[i])·x[i]
/// — the c32 twin of `zaxpy_dotc` (conjugated product =
/// dup_even(a)·x + neg_odd(dup_odd(a)·swap_pairs(x)), bit-exactly
/// `a.conj() * x` per complex). Returns the accumulated dot part;
/// fold order (two register accumulators, cross-pair add, scalar
/// tail) is the pass's own — chemv is bounds-tested, not bit-locked.
///
/// # Safety
/// `a`, `x`, `y` must each be valid for `len` C32s; `y` must not
/// alias `a` or `x`.
#[cfg_attr(target_arch = "wasm32", target_feature(enable = "simd128"))]
pub(crate) unsafe fn caxpy_dotc(
	a: *const C32,
	t: C32,
	x: *const C32,
	y: *mut C32,
	len: usize,
) -> C32 {
	let vre = F32x4::splat(t.re);
	let vim = F32x4::quad(-t.im, t.im, -t.im, t.im);
	let ap = a as *const f32;
	let xp = x as *const f32;
	let yp = y as *mut f32;
	let mut acc0 = F32x4::splat(0.0);
	let mut acc1 = F32x4::splat(0.0);
	let mut i = 0usize;
	while i + 4 <= len {
		let a0 = F32x4::load(ap.add(2 * i));
		let a1 = F32x4::load(ap.add(2 * i + 4));
		F32x4::load(yp.add(2 * i))
			.add(a0.mul(vre).add(a0.swap_pairs().mul(vim)))
			.store(yp.add(2 * i));
		F32x4::load(yp.add(2 * i + 4))
			.add(a1.mul(vre).add(a1.swap_pairs().mul(vim)))
			.store(yp.add(2 * i + 4));
		let x0 = F32x4::load(xp.add(2 * i));
		let x1 = F32x4::load(xp.add(2 * i + 4));
		acc0 = acc0.add(a0.dup_even().mul(x0).add(a0.dup_odd().mul(x0.swap_pairs()).neg_odd()));
		acc1 = acc1.add(a1.dup_even().mul(x1).add(a1.dup_odd().mul(x1.swap_pairs()).neg_odd()));
		i += 4;
	}
	let f = acc0.add(acc1);
	// fold the two packed complexes: pair 0 + pair 1
	let mut s = C32::new(f.lane0() + f.lane2(), f.lane1() + f.lane3());
	while i < len {
		let av = *a.add(i);
		*y.add(i) = *y.add(i) + t * av;
		s = s + av.conj() * *x.add(i);
		i += 1;
	}
	s
}
