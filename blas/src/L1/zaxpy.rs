//! `zaxpy` — scaled vector addition, complex: y ← αx + y.
//!
//! Implementation: elementwise stream, one complex per F64x2 register,
//! 2× unrolled. The complex multiply is the two-multiply lane form
//! (see `kernels.rs` header) — bit-exactly the canonical `C64`
//! product order, so the whole stream is bit-identical to the scalar
//! definition `y[i] + α·x[i]` on every target.

use crate::c64::C64;
use crate::lanes::F64x2;

/// y ← αx + y. Panics on length mismatch.
pub fn zaxpy(alpha: C64, x: &[C64], y: &mut [C64]) {
	assert_eq!(x.len(), y.len(), "zaxpy: length mismatch");
	unsafe { imp(alpha, x.as_ptr(), y.as_mut_ptr(), x.len()) }
}

#[cfg_attr(target_arch = "wasm32", target_feature(enable = "simd128"))]
unsafe fn imp(alpha: C64, x: *const C64, y: *mut C64, len: usize) {
	let vre = F64x2::splat(alpha.re);
	let vim = F64x2::pair(-alpha.im, alpha.im);
	let xp = x as *const f64;
	let yp = y as *mut f64;
	let mut i = 0usize;
	while i + 2 <= len {
		let x0 = F64x2::load(xp.add(2 * i));
		let x1 = F64x2::load(xp.add(2 * i + 2));
		F64x2::load(yp.add(2 * i))
			.add(x0.mul(vre).add(x0.swap().mul(vim)))
			.store(yp.add(2 * i));
		F64x2::load(yp.add(2 * i + 2))
			.add(x1.mul(vre).add(x1.swap().mul(vim)))
			.store(yp.add(2 * i + 2));
		i += 2;
	}
	while i < len {
		let x0 = F64x2::load(xp.add(2 * i));
		F64x2::load(yp.add(2 * i))
			.add(x0.mul(vre).add(x0.swap().mul(vim)))
			.store(yp.add(2 * i));
		i += 1;
	}
}
