//! `cscal` — complex scalar × complex f32 vector, in place: x ← αx.
//!
//! Implementation: elementwise stream, two complexes per F32x4
//! register, 2× unrolled; pair-wise product form (bit-exactly the
//! canonical `C32` order). For a REAL α use `csscal`.

use crate::c32::C32;
use crate::lanes::F32x4;

/// x ← αx.
pub fn cscal(alpha: C32, x: &mut [C32]) {
	unsafe { imp(alpha, x.as_mut_ptr(), x.len()) }
}

#[cfg_attr(target_arch = "wasm32", target_feature(enable = "simd128"))]
unsafe fn imp(alpha: C32, x: *mut C32, len: usize) {
	let vre = F32x4::splat(alpha.re);
	let vim = F32x4::quad(-alpha.im, alpha.im, -alpha.im, alpha.im);
	let xp = x as *mut f32;
	let mut i = 0usize;
	while i + 4 <= len {
		let x0 = F32x4::load(xp.add(2 * i));
		let x1 = F32x4::load(xp.add(2 * i + 4));
		x0.mul(vre).add(x0.swap_pairs().mul(vim)).store(xp.add(2 * i));
		x1.mul(vre).add(x1.swap_pairs().mul(vim)).store(xp.add(2 * i + 4));
		i += 4;
	}
	while i + 2 <= len {
		let x0 = F32x4::load(xp.add(2 * i));
		x0.mul(vre).add(x0.swap_pairs().mul(vim)).store(xp.add(2 * i));
		i += 2;
	}
	if i < len {
		*x.add(i) = alpha * *x.add(i);
	}
}
