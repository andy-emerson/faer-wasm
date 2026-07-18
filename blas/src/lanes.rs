//! Two f64 SIMD lanes: wasm simd128 `v128` on wasm32, a bit-identical
//! two-element emulation everywhere else. Reductions built on this fold
//! their accumulator lanes in a fixed order, so native and wasm produce
//! the same bits by construction — the determinism guarantee held
//! structurally, not by luck. (`v128_load`/`v128_store` are
//! alignment-free by spec; the emulation reads/writes elementwise.)

#[cfg(target_arch = "wasm32")]
mod imp {
	use core::arch::wasm32::*;

	#[derive(Clone, Copy)]
	pub struct F64x2(v128);

	impl F64x2 {
		#[inline(always)]
		pub fn splat(v: f64) -> Self {
			Self(f64x2_splat(v))
		}
		/// # Safety
		/// `p` must be valid for reading two f64s.
		#[inline(always)]
		pub unsafe fn load(p: *const f64) -> Self {
			Self(v128_load(p as *const v128))
		}
		/// # Safety
		/// `p` must be valid for writing two f64s.
		#[inline(always)]
		pub unsafe fn store(self, p: *mut f64) {
			v128_store(p as *mut v128, self.0)
		}
		#[inline(always)]
		pub fn add(self, o: Self) -> Self {
			Self(f64x2_add(self.0, o.0))
		}
		#[inline(always)]
		pub fn sub(self, o: Self) -> Self {
			Self(f64x2_sub(self.0, o.0))
		}
		#[inline(always)]
		pub fn mul(self, o: Self) -> Self {
			Self(f64x2_mul(self.0, o.0))
		}
		#[inline(always)]
		pub fn div(self, o: Self) -> Self {
			Self(f64x2_div(self.0, o.0))
		}
		#[inline(always)]
		pub fn abs(self) -> Self {
			Self(f64x2_abs(self.0))
		}
		#[inline(always)]
		pub fn pmax(self, o: Self) -> Self {
			Self(f64x2_pmax(self.0, o.0))
		}
		#[inline(always)]
		pub fn lane0(self) -> f64 {
			f64x2_extract_lane::<0>(self.0)
		}
		#[inline(always)]
		pub fn lane1(self) -> f64 {
			f64x2_extract_lane::<1>(self.0)
		}
	}
}

#[cfg(not(target_arch = "wasm32"))]
mod imp {
	#[derive(Clone, Copy)]
	pub struct F64x2([f64; 2]);

	impl F64x2 {
		#[inline(always)]
		pub fn splat(v: f64) -> Self {
			Self([v, v])
		}
		/// # Safety
		/// `p` must be valid for reading two f64s.
		#[inline(always)]
		pub unsafe fn load(p: *const f64) -> Self {
			Self([*p, *p.add(1)])
		}
		/// # Safety
		/// `p` must be valid for writing two f64s.
		#[inline(always)]
		pub unsafe fn store(self, p: *mut f64) {
			*p = self.0[0];
			*p.add(1) = self.0[1];
		}
		#[inline(always)]
		pub fn add(self, o: Self) -> Self {
			Self([self.0[0] + o.0[0], self.0[1] + o.0[1]])
		}
		#[inline(always)]
		pub fn sub(self, o: Self) -> Self {
			Self([self.0[0] - o.0[0], self.0[1] - o.0[1]])
		}
		#[inline(always)]
		pub fn mul(self, o: Self) -> Self {
			Self([self.0[0] * o.0[0], self.0[1] * o.0[1]])
		}
		#[inline(always)]
		pub fn div(self, o: Self) -> Self {
			Self([self.0[0] / o.0[0], self.0[1] / o.0[1]])
		}
		#[inline(always)]
		pub fn abs(self) -> Self {
			Self([self.0[0].abs(), self.0[1].abs()])
		}
		// wasm f64x2_pmax is lane-wise `a < b ? b : a` (NOT NaN-propagating
		// like fmax) — emulated with exactly that comparison.
		#[inline(always)]
		pub fn pmax(self, o: Self) -> Self {
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
		pub fn lane0(self) -> f64 {
			self.0[0]
		}
		#[inline(always)]
		pub fn lane1(self) -> f64 {
			self.0[1]
		}
	}
}

pub(crate) use imp::F64x2;
