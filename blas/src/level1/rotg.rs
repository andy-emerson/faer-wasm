//! `rotg` — generate a plane rotation.
//!
//! Implementation: no arrays = no SIMD (guarded scalar arithmetic, branch-free, inlined at call sites).
//!
//! STATUS: not yet built — scaffold only. Ported from the raced bench
//! variant during the build campaign; lands with its correctness test
//! and benchmark row (coverage rule). Evidence: docs/blas-ab-2026-07.md.
