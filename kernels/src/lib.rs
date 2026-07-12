//! Wasm-shaped dense kernels (Phase-2 companion code, architect-directed
//! 2026-07-09): the operations the Pyodide head-to-head showed losing to
//! Pyodide's LAPACK (OpenBLAS 0.3.28 built as generic C, no arch
//! microkernels), rebuilt the way the one winning kernel (gemm) was built —
//! *for* this target.
//!
//! Design rule, learned from the benchmarks: wasm engines reward the code
//! shape reference LAPACK happens to have (flat scalar loops over contiguous
//! columns, no deep abstraction, no native-cache blocking assumptions), and
//! faer's gemm is 4–20× faster than any wasm BLAS the competition links.
//! So every kernel here is LAPACK's blocked *structure* — lean panels doing
//! the O(n²) work in plain f64 loops — with the O(n³) trailing updates
//! routed through `faer::linalg::matmul`. Reference LAPACK caps out on its
//! slow dgemm; these kernels inherit our fast one.
//!
//! `no_std`; column-major with unit row stride (asserted). Real kernels
//! are generic over f64/f32 ([`scalar::WasmScalar`]); complex kernels are
//! c64-typed.

#![no_std]
extern crate alloc;

pub mod cplx;
pub mod eigvec;
pub mod eigvec_cplx;
pub mod hessenberg;
pub mod hessenberg_cplx;
pub mod scalar;
pub mod schur_small;
pub mod schur_small_cplx;
pub mod lu;
pub mod qr;
pub mod svd;
