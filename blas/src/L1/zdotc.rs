//! `zdotc` — conjugated complex dot product: xᴴy = Σ conj(xᵢ)·yᵢ.
//!
//! Implementation: reduction stream, 4 complex accumulator registers.
//! The conjugated elementwise product is the dup/swap lane form:
//! dup0(x)·y + neg1(dup1(x)·swap(y)) — lane0 = x.re·y.re + x.im·y.im,
//! lane1 = x.re·y.im − x.im·y.re, bit-exactly `conj(x)·y` in the
//! canonical `C64` product order (sign-folding is exact).
//!
//! Rounding contract: same as `zdotu` — register-parallel accumulation,
//! bounds-tested against a compensated complex reference, native ↔ wasm
//! bit-identical by construction.

use crate::c64::C64;
use crate::lanes::F64x2;

/// Returns xᴴy (x conjugated). Panics on length mismatch.
pub fn zdotc(x: &[C64], y: &[C64]) -> C64 {
	assert_eq!(x.len(), y.len(), "zdotc: length mismatch");
	unsafe { imp(x.as_ptr(), y.as_ptr(), x.len()) }
}

#[cfg_attr(target_arch = "wasm32", target_feature(enable = "simd128"))]
unsafe fn imp(x: *const C64, y: *const C64, len: usize) -> C64 {
	let xp = x as *const f64;
	let yp = y as *const f64;
	let mut acc0 = F64x2::splat(0.0);
	let mut acc1 = F64x2::splat(0.0);
	let mut acc2 = F64x2::splat(0.0);
	let mut acc3 = F64x2::splat(0.0);
	let mut i = 0usize;
	while i + 4 <= len {
		let x0 = F64x2::load(xp.add(2 * i));
		let y0 = F64x2::load(yp.add(2 * i));
		let x1 = F64x2::load(xp.add(2 * i + 2));
		let y1 = F64x2::load(yp.add(2 * i + 2));
		let x2 = F64x2::load(xp.add(2 * i + 4));
		let y2 = F64x2::load(yp.add(2 * i + 4));
		let x3 = F64x2::load(xp.add(2 * i + 6));
		let y3 = F64x2::load(yp.add(2 * i + 6));
		acc0 = acc0.add(x0.dup0().mul(y0).add(x0.dup1().mul(y0.swap()).neg1()));
		acc1 = acc1.add(x1.dup0().mul(y1).add(x1.dup1().mul(y1.swap()).neg1()));
		acc2 = acc2.add(x2.dup0().mul(y2).add(x2.dup1().mul(y2.swap()).neg1()));
		acc3 = acc3.add(x3.dup0().mul(y3).add(x3.dup1().mul(y3.swap()).neg1()));
		i += 4;
	}
	let f = acc0.add(acc1).add(acc2.add(acc3));
	let mut s = C64::new(f.lane0(), f.lane1());
	while i < len {
		s = s + (*x.add(i)).conj() * *y.add(i);
		i += 1;
	}
	s
}
