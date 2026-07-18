//! `syrk` — Gram-matrix update (alpha*A*A^T + beta*C).
//!
//! Implementation: column-axpy (plain variant, measured: FMA harms).
//!
//! STATUS: not yet built — scaffold only. Ported from the raced bench
//! variant during the build campaign; lands with its correctness test
//! and benchmark row (coverage rule). Evidence: docs/blas-ab-2026-07.md.
