//! Level 3 correctness per the testing contract (../../README.md),
//! one test file per BLAS routine mirroring src/L3/ (naming
//! convention: src/L1/README.md). Shared helpers in common.rs.
#![allow(non_snake_case)] // test target named for its folder (L1/L2/L3)

mod common;

mod dgemm;
mod dsymm;
mod dsyr2k;
mod dsyrk;
mod dtrmm;
mod dtrsm;
mod sgemm;
mod ssymm;
mod ssyr2k;
mod ssyrk;
mod strmm;
mod strsm;

mod zgemm;
mod zhemm;
mod zher2k;
mod zherk;
mod ztrmm;
mod ztrsm;

mod cgemm;
mod chemm;
mod cher2k;
mod cherk;
mod ctrmm;
mod ctrsm;
