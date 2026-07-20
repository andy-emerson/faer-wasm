//! faer-wasm-blas: the wasm-native BLAS layer, one file per BLAS
//! routine per number type in netlib naming (d = f64, s = f32,
//! z = c64, c = c32 — naming convention and call graph in
//! src/README.md). All four type layers are built; unit stride —
//! callers pass contiguous column slices (strided access defeats
//! streaming and no consumer wants it). The tuned multi-column loop
//! shapes live in `kernels`; the SIMD lane types with their
//! bit-identical native emulation live in `lanes`; the complex
//! scalars (`C64`/`C32`, exported) in `c64`/`c32`. Plan of record:
//! README.md in the crate root.
#![no_std]

// Only the packed-gemm paths allocate (pack buffers); everything else
// stays allocation-free. Consumers must link a global allocator iff they
// call a `*gemm_packed` routine — all current consumers already do.
extern crate alloc;

mod c32;
mod c64;
mod kernels;
mod lanes;

pub use c32::C32;
pub use c64::C64;

// Module names match the folder and table spelling (BLAS "Level 1"
// convention) — deliberately not snake_case.
#[allow(non_snake_case)]
pub mod L1;
#[allow(non_snake_case)]
pub mod L2;
#[allow(non_snake_case)]
pub mod L3;
