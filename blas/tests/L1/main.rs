//! Level 1 correctness per the testing contract (../../README.md),
//! one test file per BLAS routine mirroring src/L1/ (naming
//! convention: src/L1/README.md). Shared helpers in common.rs.
#![allow(non_snake_case)] // test target named for its folder (L1/L2/L3)

mod common;

mod dasum;
mod daxpy;
mod dcopy;
mod ddot;
mod dnrm2;
mod drot;
mod drotg;
mod dscal;
mod dswap;
mod idamax;
mod isamax;
mod sasum;
mod saxpy;
mod scopy;
mod sdot;
mod snrm2;
mod srot;
mod srotg;
mod sscal;
mod sswap;

mod dzasum;
mod dznrm2;
mod izamax;
mod zaxpy;
mod zcopy;
mod zdotc;
mod zdotu;
mod zdrot;
mod zdscal;
mod zrotg;
mod zscal;
mod zswap;

mod caxpy;
mod ccopy;
mod cdotc;
mod cdotu;
mod crotg;
mod cscal;
mod csrot;
mod csscal;
mod cswap;
mod icamax;
mod scasum;
mod scnrm2;
