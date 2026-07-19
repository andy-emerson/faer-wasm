//! Tuned multi-stream kernels shared across levels (tuning campaign,
//! 2026-07): the two loop shapes that survived racing on the reference
//! runners. `axpy4` fans one source column OUT into four destinations
//! (the gemm `col4` shape — cuts source traffic 4×); `axpy4in` fans
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
pub(crate) unsafe fn axpy4(
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

/// One pass over the destination with four source columns fanned in:
/// c[i] ← ((((c[i] + a0[i]·t0) + a1[i]·t1) + a2[i]·t2) + a3[i]·t3) —
/// the same rounding sequence as four consecutive plain `axpy` passes
/// in that order, with the destination loaded and stored once.
///
/// # Safety
/// All five column pointers must be valid for `len` f64s; the source
/// columns must not alias the destination.
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
pub(crate) unsafe fn axpy_dot(
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
pub(crate) unsafe fn axpy_dot4(
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

#[cfg_attr(target_arch = "wasm32", target_feature(enable = "simd128"))]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn axpy4in(
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
