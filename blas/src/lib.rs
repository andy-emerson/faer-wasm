//! faer-wasm-blas: the wasm-native BLAS layer, one file per function.
//! Plan of record: README.md in this folder. The f64 layer is complete
//! and tuned (all three levels; unit stride — callers pass contiguous
//! column slices; strided access defeats streaming and no consumer
//! wants it). The tuned multi-column loop shapes live in `kernels`;
//! the SIMD lane type with its bit-identical native emulation lives in
//! `lanes`. Other number types (f32, c64) are the next campaign.
#![no_std]

mod kernels;
mod lanes;

pub mod level1;
pub mod level2;
pub mod level3;
