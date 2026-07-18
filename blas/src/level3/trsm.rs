//! `trsm` — triangular solve, many right-hand sides.
//!
//! Implementation: divide-then-column-axpy (fused-FMA variant, measured).
//!
//! STATUS: not yet built — scaffold only. Ported from the raced bench
//! variant during the build campaign; lands with its correctness test
//! and benchmark row (coverage rule). Evidence: docs/blas-ab-2026-07.md.
