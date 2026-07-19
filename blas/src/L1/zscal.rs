//! `zscal` — complex scalar × complex vector, in place: x ← αx.
//!
//! Implementation: elementwise stream, one complex per F64x2 register,
//! 2× unrolled; the two-multiply lane form of the complex product
//! (bit-exactly the canonical `C64` order — see `kernels.rs` header).
//! For a REAL α use `zdscal`, which is cheaper and rounds differently
//! (one multiply per component instead of a full complex product).

use crate::c64::C64;
use crate::lanes::F64x2;

/// x ← αx.
pub fn zscal(alpha: C64, x: &mut [C64]) {
	unsafe { imp(alpha, x.as_mut_ptr(), x.len()) }
}

#[cfg_attr(target_arch = "wasm32", target_feature(enable = "simd128"))]
unsafe fn imp(alpha: C64, x: *mut C64, len: usize) {
	let vre = F64x2::splat(alpha.re);
	let vim = F64x2::pair(-alpha.im, alpha.im);
	let xp = x as *mut f64;
	let mut i = 0usize;
	while i + 2 <= len {
		let x0 = F64x2::load(xp.add(2 * i));
		let x1 = F64x2::load(xp.add(2 * i + 2));
		x0.mul(vre).add(x0.swap().mul(vim)).store(xp.add(2 * i));
		x1.mul(vre).add(x1.swap().mul(vim)).store(xp.add(2 * i + 2));
		i += 2;
	}
	while i < len {
		let x0 = F64x2::load(xp.add(2 * i));
		x0.mul(vre).add(x0.swap().mul(vim)).store(xp.add(2 * i));
		i += 1;
	}
}
