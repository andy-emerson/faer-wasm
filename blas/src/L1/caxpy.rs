//! `caxpy` — scaled vector addition, complex f32: y ← αx + y.
//!
//! Implementation: elementwise stream, two complexes per F32x4
//! register, 2× unrolled; the pair-wise two-multiply product form
//! (see `kernels.rs`) — bit-exactly the canonical `C32` product
//! order, so the stream (including the one-complex scalar tail) is
//! bit-identical to the scalar definition on every target.

use crate::c32::C32;
use crate::lanes::F32x4;

/// y ← αx + y. Panics on length mismatch.
pub fn caxpy(alpha: C32, x: &[C32], y: &mut [C32]) {
	assert_eq!(x.len(), y.len(), "caxpy: length mismatch");
	unsafe { imp(alpha, x.as_ptr(), y.as_mut_ptr(), x.len()) }
}

#[cfg_attr(target_arch = "wasm32", target_feature(enable = "simd128"))]
unsafe fn imp(alpha: C32, x: *const C32, y: *mut C32, len: usize) {
	let vre = F32x4::splat(alpha.re);
	let vim = F32x4::quad(-alpha.im, alpha.im, -alpha.im, alpha.im);
	let xp = x as *const f32;
	let yp = y as *mut f32;
	let mut i = 0usize;
	while i + 4 <= len {
		let x0 = F32x4::load(xp.add(2 * i));
		let x1 = F32x4::load(xp.add(2 * i + 4));
		F32x4::load(yp.add(2 * i))
			.add(x0.mul(vre).add(x0.swap_pairs().mul(vim)))
			.store(yp.add(2 * i));
		F32x4::load(yp.add(2 * i + 4))
			.add(x1.mul(vre).add(x1.swap_pairs().mul(vim)))
			.store(yp.add(2 * i + 4));
		i += 4;
	}
	while i + 2 <= len {
		let x0 = F32x4::load(xp.add(2 * i));
		F32x4::load(yp.add(2 * i))
			.add(x0.mul(vre).add(x0.swap_pairs().mul(vim)))
			.store(yp.add(2 * i));
		i += 2;
	}
	if i < len {
		*y.add(i) = *y.add(i) + alpha * *x.add(i);
	}
}
