//! Two f64 SIMD lanes: wasm simd128 `v128` on wasm32, a bit-identical
//! two-element emulation everywhere else. Reductions built on this fold
//! their accumulator lanes in a fixed order, so native and wasm produce
//! the same bits by construction — the determinism guarantee held
//! structurally, not by luck.
//!
//! Every method is `unsafe` and, on wasm, carries
//! `#[target_feature(enable = "simd128")]`: simd128 is NOT in rustc's
//! default wasm32 feature set (measured 2026-07-18 — an unannotated
//! wrapper compiled every lane op as an out-of-line call and ran the
//! reductions 6× slower), so the feature must be enabled on the whole
//! call chain for the intrinsics to inline. Callers keep the chain by
//! annotating their own inner loops (see any level1 stream).
//! (`v128_load`/`v128_store` are alignment-free by spec; the emulation
//! reads/writes elementwise.)

#[cfg(target_arch = "wasm32")]
mod imp {
	use core::arch::wasm32::*;

	#[derive(Clone, Copy)]
	pub struct F64x2(v128);

	#[allow(clippy::missing_safety_doc)]
	impl F64x2 {
		#[inline]
		#[target_feature(enable = "simd128")]
		pub unsafe fn splat(v: f64) -> Self {
			Self(f64x2_splat(v))
		}
		#[inline]
		#[target_feature(enable = "simd128")]
		pub unsafe fn load(p: *const f64) -> Self {
			Self(v128_load(p as *const v128))
		}
		#[inline]
		#[target_feature(enable = "simd128")]
		pub unsafe fn store(self, p: *mut f64) {
			v128_store(p as *mut v128, self.0)
		}
		#[inline]
		#[target_feature(enable = "simd128")]
		pub unsafe fn add(self, o: Self) -> Self {
			Self(f64x2_add(self.0, o.0))
		}
		#[inline]
		#[target_feature(enable = "simd128")]
		pub unsafe fn sub(self, o: Self) -> Self {
			Self(f64x2_sub(self.0, o.0))
		}
		#[inline]
		#[target_feature(enable = "simd128")]
		pub unsafe fn mul(self, o: Self) -> Self {
			Self(f64x2_mul(self.0, o.0))
		}
		#[inline]
		#[target_feature(enable = "simd128")]
		pub unsafe fn div(self, o: Self) -> Self {
			Self(f64x2_div(self.0, o.0))
		}
		#[inline]
		#[target_feature(enable = "simd128")]
		pub unsafe fn abs(self) -> Self {
			Self(f64x2_abs(self.0))
		}
		#[inline]
		#[target_feature(enable = "simd128")]
		pub unsafe fn pmax(self, o: Self) -> Self {
			Self(f64x2_pmax(self.0, o.0))
		}
		#[inline]
		#[target_feature(enable = "simd128")]
		pub unsafe fn lane0(self) -> f64 {
			f64x2_extract_lane::<0>(self.0)
		}
		#[inline]
		#[target_feature(enable = "simd128")]
		pub unsafe fn lane1(self) -> f64 {
			f64x2_extract_lane::<1>(self.0)
		}
		#[inline]
		#[target_feature(enable = "simd128")]
		pub unsafe fn pair(a: f64, b: f64) -> Self {
			Self(f64x2(a, b))
		}
		/// Branch-free argmax step, per lane: if `a > m` take `(a, ai)`,
		/// else keep `(m, mi)`. Strict `>` keeps the first occurrence
		/// within each lane track; NaN never replaces.
		#[inline]
		#[target_feature(enable = "simd128")]
		pub unsafe fn argmax_step(m: Self, mi: Self, a: Self, ai: Self) -> (Self, Self) {
			let gt = f64x2_gt(a.0, m.0);
			(Self(v128_bitselect(a.0, m.0, gt)), Self(v128_bitselect(ai.0, mi.0, gt)))
		}
	}
}

#[cfg(not(target_arch = "wasm32"))]
mod imp {
	#[derive(Clone, Copy)]
	pub struct F64x2([f64; 2]);

	// `unsafe fn` for signature parity with the wasm impl — the level1
	// streams call these inside one feature-annotated unsafe inner fn.
	#[allow(clippy::missing_safety_doc)]
	impl F64x2 {
		#[inline(always)]
		pub unsafe fn splat(v: f64) -> Self {
			Self([v, v])
		}
		#[inline(always)]
		pub unsafe fn load(p: *const f64) -> Self {
			Self([*p, *p.add(1)])
		}
		#[inline(always)]
		pub unsafe fn store(self, p: *mut f64) {
			*p = self.0[0];
			*p.add(1) = self.0[1];
		}
		#[inline(always)]
		pub unsafe fn add(self, o: Self) -> Self {
			Self([self.0[0] + o.0[0], self.0[1] + o.0[1]])
		}
		#[inline(always)]
		pub unsafe fn sub(self, o: Self) -> Self {
			Self([self.0[0] - o.0[0], self.0[1] - o.0[1]])
		}
		#[inline(always)]
		pub unsafe fn mul(self, o: Self) -> Self {
			Self([self.0[0] * o.0[0], self.0[1] * o.0[1]])
		}
		#[inline(always)]
		pub unsafe fn div(self, o: Self) -> Self {
			Self([self.0[0] / o.0[0], self.0[1] / o.0[1]])
		}
		#[inline(always)]
		pub unsafe fn abs(self) -> Self {
			Self([self.0[0].abs(), self.0[1].abs()])
		}
		// wasm f64x2_pmax is lane-wise `a < b ? b : a` (NOT NaN-propagating
		// like fmax) — emulated with exactly that comparison.
		#[inline(always)]
		pub unsafe fn pmax(self, o: Self) -> Self {
			#[inline(always)]
			fn pm(a: f64, b: f64) -> f64 {
				if a < b {
					b
				} else {
					a
				}
			}
			Self([pm(self.0[0], o.0[0]), pm(self.0[1], o.0[1])])
		}
		#[inline(always)]
		pub unsafe fn lane0(self) -> f64 {
			self.0[0]
		}
		#[inline(always)]
		pub unsafe fn lane1(self) -> f64 {
			self.0[1]
		}
		#[inline(always)]
		pub unsafe fn pair(a: f64, b: f64) -> Self {
			Self([a, b])
		}
		// wasm side is f64x2_gt + v128_bitselect — lane-wise strict `>`
		// blend, emulated with exactly that predicate.
		#[inline(always)]
		pub unsafe fn argmax_step(m: Self, mi: Self, a: Self, ai: Self) -> (Self, Self) {
			#[inline(always)]
			fn step(m: f64, mi: f64, a: f64, ai: f64) -> (f64, f64) {
				if a > m {
					(a, ai)
				} else {
					(m, mi)
				}
			}
			let (v0, i0) = step(m.0[0], mi.0[0], a.0[0], ai.0[0]);
			let (v1, i1) = step(m.0[1], mi.0[1], a.0[1], ai.0[1]);
			(Self([v0, v1]), Self([i0, i1]))
		}
	}
}

pub(crate) use imp::F64x2;
