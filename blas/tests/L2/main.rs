//! Level 2 correctness per the testing contract (../../README.md),
//! one test file per BLAS routine mirroring src/L2/ (naming
//! convention: src/L1/README.md). Shared helpers in common.rs.
#![allow(non_snake_case)] // test target named for its folder (L1/L2/L3)

mod common;

mod dgemv;
mod dger;
mod dsymv;
mod dsyr;
mod dsyr2;
mod dtrmv;
mod dtrsv;
mod sgemv;
mod sger;
mod ssymv;
mod ssyr;
mod ssyr2;
mod strmv;
mod strsv;

mod zgemv;
mod zgerc;
mod zgeru;
mod zhemv;
mod zher;
mod zher2;
mod ztrmv;
mod ztrsv;
