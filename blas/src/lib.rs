//! faer-wasm-blas: the wasm-native BLAS layer, one file per function.
//! Plan of record: README.md in this folder. Level 1 is implemented
//! (f64, unit stride — callers pass contiguous column slices; strided
//! access defeats streaming and no consumer wants it). Levels 2–3 are
//! scaffold.
#![no_std]

mod lanes;

pub mod level1;
pub mod level2;
pub mod level3;
