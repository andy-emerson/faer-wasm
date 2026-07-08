//! Schur decomposition and eigenvalue reordering as a companion crate to the
//! carried faer — the first Phase 2 item (LAPACK-parity tail).
//!
//! - [`real`]: `A = Z T Zᵀ` for `f64`, `T` quasi upper triangular (`dgees` /
//!   `dtrexc` / `dtrsen`-shaped).
//! - [`complex`]: `A = Z T Zᴴ` for `c64`, `T` upper triangular (`zgees` /
//!   `ztrexc` / `ztrsen`-shaped).
//!
//! Everything is built over faer's public modules (`hessenberg`,
//! `evd::schur::{real_schur, complex_schur}`); the latter two are public only
//! with `patches/0002-expose-schur-kernels.patch` applied (visibility-only,
//! no behavior change). No faer code is duplicated here — this crate is the
//! driver (Hessenberg → Z accumulation → multishift QR) plus the
//! LAPACK-flavored reordering loops.
//!
//! `no_std`: needs only a global allocator (see `smoke-test/src/lib.rs` for
//! the zero-import wasm pattern).

#![no_std]
extern crate alloc;

pub mod complex;
pub mod real;

pub use faer::linalg::evd::schur::SchurParams;

/// failure of the Schur decomposition itself
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SchurError {
	/// the input contained a NaN or infinity
	NonFinite,
	/// the QR iteration failed to converge
	NoConvergence,
}

/// failure of an eigenvalue-reordering request
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReorderError {
	/// a direct 2×2 block swap was rejected as too ill-conditioned
	/// (LAPACK `INFO = 1` semantics): `T` is still a valid Schur form of the
	/// same matrix but is only partially reordered; `at` is the row the
	/// moving block stopped at
	SwapRejected { at: usize },
}
