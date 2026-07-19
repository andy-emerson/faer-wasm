//! Shared test helpers for this level: the deterministic data
//! generator (LCG, no external crates), the higher-precision
//! reference summers, and the size lists. `_f64`/`_f32` method pairs
//! draw from the same integer stream, so a given seed produces the
//! same underlying values in either type.

#![allow(dead_code)] // each per-routine test file uses its own subset

pub struct Lcg(pub u64);
impl Lcg {
	pub fn next_f64(&mut self) -> f64 {
		self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
		let bits = (self.0 >> 11) as f64 / (1u64 << 53) as f64; // [0,1)
		4.0 * bits - 2.0
	}
	pub fn next_f32(&mut self) -> f32 {
		self.next_f64() as f32
	}
	pub fn vec_f64(&mut self, n: usize) -> Vec<f64> {
		(0..n).map(|_| self.next_f64()).collect()
	}
	pub fn vec_f32(&mut self, n: usize) -> Vec<f32> {
		(0..n).map(|_| self.next_f32()).collect()
	}
	/// column-major nrows×ncols at stride cs (padding filled with junk
	/// that must never be read)
	pub fn mat_f64(&mut self, nrows: usize, ncols: usize, cs: usize) -> Vec<f64> {
		assert!(cs >= nrows);
		let mut a = vec![f64::NAN; if ncols == 0 { 0 } else { cs * (ncols - 1) + nrows }];
		for j in 0..ncols {
			for i in 0..nrows {
				a[j * cs + i] = self.next_f64();
			}
		}
		a
	}
	pub fn mat_f32(&mut self, nrows: usize, ncols: usize, cs: usize) -> Vec<f32> {
		assert!(cs >= nrows);
		let mut a = vec![f32::NAN; if ncols == 0 { 0 } else { cs * (ncols - 1) + nrows }];
		for j in 0..ncols {
			for i in 0..nrows {
				a[j * cs + i] = self.next_f32();
			}
		}
		a
	}
}

/// Neumaier compensated summation — the higher-precision reference
/// for the f64 reduction bounds.
pub fn comp_sum(it: impl Iterator<Item = f64>) -> f64 {
	let mut s = 0.0f64;
	let mut c = 0.0f64;
	for v in it {
		let t = s + v;
		if s.abs() >= v.abs() {
			c += (s - t) + v;
		} else {
			c += (v - t) + s;
		}
		s = t;
	}
	s + c
}

/// f32 items (products formed in f32, as the implementation forms
/// them), accumulated exactly in f64 — the higher-precision reference
/// for the f32 bounds.
pub fn comp_sum32(it: impl Iterator<Item = f32>) -> f64 {
	let mut s = 0.0f64;
	for v in it {
		s += v as f64;
	}
	s
}

/// f32 epsilon as f64, for f32 tolerance arithmetic.
pub const EPS: f64 = f32::EPSILON as f64;

pub const DIMS: &[(usize, usize, usize)] =
	&[(0, 0, 0), (1, 1, 1), (3, 2, 4), (5, 5, 5), (8, 3, 7), (17, 12, 9), (33, 33, 33)];
pub const NS: &[usize] = &[0, 1, 2, 3, 5, 8, 17, 33];
// (m, n) shapes for the in-place triangular replays: tile boundaries
// and tails on both sides
pub const TRI_DIMS: &[(usize, usize)] =
	&[(0, 0), (1, 1), (3, 5), (5, 3), (4, 4), (8, 8), (9, 13), (16, 7), (20, 33), (33, 20)];

// ---- c64 helpers ----
use faer_wasm_blas::C64;

impl Lcg {
	/// re then im from the same integer stream (so a seed's draw
	/// order is stable across types)
	pub fn next_c64(&mut self) -> C64 {
		let re = self.next_f64();
		let im = self.next_f64();
		C64::new(re, im)
	}
	pub fn vec_c64(&mut self, n: usize) -> Vec<C64> {
		(0..n).map(|_| self.next_c64()).collect()
	}
	/// column-major nrows×ncols at stride cs (padding filled with junk
	/// that must never be read)
	pub fn mat_c64(&mut self, nrows: usize, ncols: usize, cs: usize) -> Vec<C64> {
		assert!(cs >= nrows);
		let nan = C64::new(f64::NAN, f64::NAN);
		let mut a = vec![nan; if ncols == 0 { 0 } else { cs * (ncols - 1) + nrows }];
		for j in 0..ncols {
			for i in 0..nrows {
				a[j * cs + i] = self.next_c64();
			}
		}
		a
	}
}

/// Component-wise Neumaier compensated summation over complex items
/// (products formed in C64, as the implementation forms them) — the
/// higher-precision reference for the c64 reduction bounds.
pub fn comp_sum_c(it: impl Iterator<Item = C64>) -> C64 {
	let (mut sr, mut cr) = (0.0f64, 0.0f64);
	let (mut si, mut ci) = (0.0f64, 0.0f64);
	for v in it {
		let t = sr + v.re;
		if sr.abs() >= v.re.abs() {
			cr += (sr - t) + v.re;
		} else {
			cr += (v.re - t) + sr;
		}
		sr = t;
		let t = si + v.im;
		if si.abs() >= v.im.abs() {
			ci += (si - t) + v.im;
		} else {
			ci += (v.im - t) + si;
		}
		si = t;
	}
	C64::new(sr + cr, si + ci)
}

/// Exact bit agreement for complex values (both components).
pub fn bits_eq_c(a: C64, b: C64) -> bool {
	a.re.to_bits() == b.re.to_bits() && a.im.to_bits() == b.im.to_bits()
}

/// |z|₁ magnitude of each item summed (Neumaier) — the scale term for
/// c64 tolerances.
pub fn comp_scale_c(it: impl Iterator<Item = C64>) -> f64 {
	comp_sum(it.map(|v| v.re.abs() + v.im.abs()))
}

// ---- c32 helpers ----
use faer_wasm_blas::C32;

impl Lcg {
	/// re then im from the same integer stream, in f32
	pub fn next_c32(&mut self) -> C32 {
		let re = self.next_f32();
		let im = self.next_f32();
		C32::new(re, im)
	}
	pub fn vec_c32(&mut self, n: usize) -> Vec<C32> {
		(0..n).map(|_| self.next_c32()).collect()
	}
	pub fn mat_c32(&mut self, nrows: usize, ncols: usize, cs: usize) -> Vec<C32> {
		assert!(cs >= nrows);
		let nan = C32::new(f32::NAN, f32::NAN);
		let mut a = vec![nan; if ncols == 0 { 0 } else { cs * (ncols - 1) + nrows }];
		for j in 0..ncols {
			for i in 0..nrows {
				a[j * cs + i] = self.next_c32();
			}
		}
		a
	}
}

/// Up-convert a C32 to C64 for f64 reference arithmetic.
pub fn c_up(v: C32) -> C64 {
	C64::new(v.re as f64, v.im as f64)
}

/// C32 items (products formed in f32, as the implementation forms
/// them) accumulated exactly in f64 per component — the
/// higher-precision reference for the c32 bounds (the complex analog
/// of `comp_sum32`).
pub fn comp_sum_cc(it: impl Iterator<Item = C32>) -> C64 {
	let (mut sr, mut si) = (0.0f64, 0.0f64);
	for v in it {
		sr += v.re as f64;
		si += v.im as f64;
	}
	C64::new(sr, si)
}

/// Exact bit agreement for c32 values (both components).
pub fn bits_eq_cc(a: C32, b: C32) -> bool {
	a.re.to_bits() == b.re.to_bits() && a.im.to_bits() == b.im.to_bits()
}

/// Σ|z|₁ of the f32 items, accumulated in f64 — the scale term for
/// c32 tolerances.
pub fn comp_scale_cc(it: impl Iterator<Item = C32>) -> f64 {
	let mut s = 0.0f64;
	for v in it {
		s += (v.re.abs() + v.im.abs()) as f64;
	}
	s
}
