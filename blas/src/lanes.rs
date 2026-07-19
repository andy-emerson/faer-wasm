//! The SIMD lane types — F64x2 (two f64 lanes) and F32x4 (four f32
//! lanes): wasm simd128 `v128` on wasm32, a bit-identical elementwise
//! emulation everywhere else. Reductions built on these fold their
//! accumulator lanes in a fixed order, so native and wasm produce the
//! same bits by construction — the determinism guarantee held
//! structurally, not by luck.
//!
//! Every method is `unsafe` and, on wasm, carries
//! `#[target_feature(enable = "simd128")]`: simd128 is NOT in rustc's
//! default wasm32 feature set (measured 2026-07-18 — an unannotated
//! wrapper compiled every lane op as an out-of-line call and ran the
//! reductions 6× slower), so the feature must be enabled on the whole
//! call chain for the intrinsics to inline. Callers keep the chain by
//! annotating their own inner loops (see any l1 stream).
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
		// ---- lane rearrangement/sign ops (the c64 layer's complex
		// multiply is built from these; they are generic lane ops, no
		// complex semantics here) ----
		#[inline]
		#[target_feature(enable = "simd128")]
		pub unsafe fn pair(a: f64, b: f64) -> Self {
			Self(f64x2(a, b))
		}
		#[inline]
		#[target_feature(enable = "simd128")]
		pub unsafe fn swap(self) -> Self {
			Self(i64x2_shuffle::<1, 0>(self.0, self.0))
		}
		#[inline]
		#[target_feature(enable = "simd128")]
		pub unsafe fn dup0(self) -> Self {
			Self(i64x2_shuffle::<0, 0>(self.0, self.0))
		}
		#[inline]
		#[target_feature(enable = "simd128")]
		pub unsafe fn dup1(self) -> Self {
			Self(i64x2_shuffle::<1, 1>(self.0, self.0))
		}
		#[inline]
		#[target_feature(enable = "simd128")]
		pub unsafe fn neg0(self) -> Self {
			Self(v128_xor(self.0, f64x2(-0.0, 0.0)))
		}
		#[inline]
		#[target_feature(enable = "simd128")]
		pub unsafe fn neg1(self) -> Self {
			Self(v128_xor(self.0, f64x2(0.0, -0.0)))
		}
	}
}

#[cfg(not(target_arch = "wasm32"))]
mod imp {
	#[derive(Clone, Copy)]
	pub struct F64x2([f64; 2]);

	// `unsafe fn` for signature parity with the wasm impl — the l1
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
		// ---- lane rearrangement/sign ops (exact — moves and sign
		// flips only, so emulation is trivially bit-identical) ----
		#[inline(always)]
		pub unsafe fn pair(a: f64, b: f64) -> Self {
			Self([a, b])
		}
		#[inline(always)]
		pub unsafe fn swap(self) -> Self {
			Self([self.0[1], self.0[0]])
		}
		#[inline(always)]
		pub unsafe fn dup0(self) -> Self {
			Self([self.0[0], self.0[0]])
		}
		#[inline(always)]
		pub unsafe fn dup1(self) -> Self {
			Self([self.0[1], self.0[1]])
		}
		#[inline(always)]
		pub unsafe fn neg0(self) -> Self {
			Self([-self.0[0], self.0[1]])
		}
		#[inline(always)]
		pub unsafe fn neg1(self) -> Self {
			Self([self.0[0], -self.0[1]])
		}
	}
}

pub(crate) use imp::F64x2;

#[cfg(target_arch = "wasm32")]
mod imp32 {
	use core::arch::wasm32::*;

	#[derive(Clone, Copy)]
	pub struct F32x4(v128);

	#[allow(clippy::missing_safety_doc)]
	impl F32x4 {
		#[inline]
		#[target_feature(enable = "simd128")]
		pub unsafe fn splat(v: f32) -> Self {
			Self(f32x4_splat(v))
		}
		#[inline]
		#[target_feature(enable = "simd128")]
		pub unsafe fn load(p: *const f32) -> Self {
			Self(v128_load(p as *const v128))
		}
		#[inline]
		#[target_feature(enable = "simd128")]
		pub unsafe fn store(self, p: *mut f32) {
			v128_store(p as *mut v128, self.0)
		}
		#[inline]
		#[target_feature(enable = "simd128")]
		pub unsafe fn add(self, o: Self) -> Self {
			Self(f32x4_add(self.0, o.0))
		}
		#[inline]
		#[target_feature(enable = "simd128")]
		pub unsafe fn sub(self, o: Self) -> Self {
			Self(f32x4_sub(self.0, o.0))
		}
		#[inline]
		#[target_feature(enable = "simd128")]
		pub unsafe fn mul(self, o: Self) -> Self {
			Self(f32x4_mul(self.0, o.0))
		}
		#[inline]
		#[target_feature(enable = "simd128")]
		pub unsafe fn div(self, o: Self) -> Self {
			Self(f32x4_div(self.0, o.0))
		}
		#[inline]
		#[target_feature(enable = "simd128")]
		pub unsafe fn abs(self) -> Self {
			Self(f32x4_abs(self.0))
		}
		#[inline]
		#[target_feature(enable = "simd128")]
		pub unsafe fn pmax(self, o: Self) -> Self {
			Self(f32x4_pmax(self.0, o.0))
		}
		#[inline]
		#[target_feature(enable = "simd128")]
		pub unsafe fn lane0(self) -> f32 {
			f32x4_extract_lane::<0>(self.0)
		}
		#[inline]
		#[target_feature(enable = "simd128")]
		pub unsafe fn lane1(self) -> f32 {
			f32x4_extract_lane::<1>(self.0)
		}
		#[inline]
		#[target_feature(enable = "simd128")]
		pub unsafe fn lane2(self) -> f32 {
			f32x4_extract_lane::<2>(self.0)
		}
		#[inline]
		#[target_feature(enable = "simd128")]
		pub unsafe fn lane3(self) -> f32 {
			f32x4_extract_lane::<3>(self.0)
		}
	}
}

#[cfg(not(target_arch = "wasm32"))]
mod imp32 {
	#[derive(Clone, Copy)]
	pub struct F32x4([f32; 4]);

	// `unsafe fn` for signature parity with the wasm impl — the f32
	// streams call these inside one feature-annotated unsafe inner fn.
	#[allow(clippy::missing_safety_doc)]
	impl F32x4 {
		#[inline(always)]
		pub unsafe fn splat(v: f32) -> Self {
			Self([v; 4])
		}
		#[inline(always)]
		pub unsafe fn load(p: *const f32) -> Self {
			Self([*p, *p.add(1), *p.add(2), *p.add(3)])
		}
		#[inline(always)]
		pub unsafe fn store(self, p: *mut f32) {
			*p = self.0[0];
			*p.add(1) = self.0[1];
			*p.add(2) = self.0[2];
			*p.add(3) = self.0[3];
		}
		#[inline(always)]
		pub unsafe fn add(self, o: Self) -> Self {
			Self(core::array::from_fn(|i| self.0[i] + o.0[i]))
		}
		#[inline(always)]
		pub unsafe fn sub(self, o: Self) -> Self {
			Self(core::array::from_fn(|i| self.0[i] - o.0[i]))
		}
		#[inline(always)]
		pub unsafe fn mul(self, o: Self) -> Self {
			Self(core::array::from_fn(|i| self.0[i] * o.0[i]))
		}
		#[inline(always)]
		pub unsafe fn div(self, o: Self) -> Self {
			Self(core::array::from_fn(|i| self.0[i] / o.0[i]))
		}
		#[inline(always)]
		pub unsafe fn abs(self) -> Self {
			Self(core::array::from_fn(|i| self.0[i].abs()))
		}
		// wasm f32x4_pmax is lane-wise `a < b ? b : a` (NOT
		// NaN-propagating like fmax) — emulated with exactly that
		// comparison.
		#[inline(always)]
		pub unsafe fn pmax(self, o: Self) -> Self {
			#[inline(always)]
			fn pm(a: f32, b: f32) -> f32 {
				if a < b {
					b
				} else {
					a
				}
			}
			Self(core::array::from_fn(|i| pm(self.0[i], o.0[i])))
		}
		#[inline(always)]
		pub unsafe fn lane0(self) -> f32 {
			self.0[0]
		}
		#[inline(always)]
		pub unsafe fn lane1(self) -> f32 {
			self.0[1]
		}
		#[inline(always)]
		pub unsafe fn lane2(self) -> f32 {
			self.0[2]
		}
		#[inline(always)]
		pub unsafe fn lane3(self) -> f32 {
			self.0[3]
		}
	}
}

pub(crate) use imp32::F32x4;
