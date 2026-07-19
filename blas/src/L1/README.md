# Routine naming — the netlib convention (applies to L1/, L2/, L3/)

One file per BLAS routine per number type, named exactly as the
reference BLAS names them — the layout of the netlib source tree.
The file name is the function it exports: `daxpy.rs` exports
`pub fn daxpy`.

## The type prefix

The first letter encodes the number type, matching the columns of the
tables in the crate README:

| prefix | type | example |
|---|---|---|
| `d` | f64 (double) | `daxpy`, `dgemm` |
| `s` | f32 (single) | `saxpy`, `sgemm` |
| `z` | c64 (double complex) | `zaxpy`, `zgemm` — built 2026-07-19 |
| `c` | c32 (single complex) | `caxpy`, `cgemm` — built 2026-07-19 |

The index routines put `i` first and the type second: `idamax` /
`isamax` / `izamax` / `icamax` (index of the largest element — |x|
for real, |re|+|im| for complex). The complex routines that return a
real carry both letters, exactly as reference BLAS spells them:
`dznrm2`/`scnrm2`, `dzasum`/`scasum`; and the real-scalar/
real-rotation forms keep reference names too: `zdscal`/`csscal`
(real α), `zdrot`/`csrot` (real c, s).

## Where we deviate from reference BLAS — deliberately, per routine

The names are netlib's; the signatures are ours (documented per file):

- **No trans/side/uplo character arguments.** Variants that reference
  BLAS folds into flag parameters are separate functions:
  `dgemv`/`dgemv_t` (plus `zgemv_c` for the conjugate transpose),
  `dsymm_left`/`dsymm_right`, `dtrmm_left`/`dtrmm_right`,
  `dtrsm_left`/`dtrsm_right` (and their z twins). Triangle and
  unit-diagonal selection stay as `bool` parameters where the loop
  structure is shared (`upper`, `unit`).
- **Unit stride only.** Callers pass contiguous column slices with a
  column stride per matrix — no `incx`/`incy` (strided access defeats
  streaming and no consumer wants it).
- **`drotg`/`srotg` return a `Givens` struct** (c, s, r) instead of
  writing through pointers, and omit the classic `z` reconstruction
  output — no consumer wants it. `zrotg` likewise returns a `ZGivens`
  (real c, complex s, complex r).
- **Tuned-variant exports.** Where a routine ships raced alternates,
  they carry a suffix: `dgemm_colaxpy` (the plain reference shape),
  `dgemm_tiled`, `dgemm_col4` — `dgemm` itself is the size dispatcher.

Everything else about a routine — what it computes, its rounding
contract, which tuned loop shape it ships and why — lives in its own
file's module docs. The per-level composition rules are in each
level's `mod.rs`; who-calls-whom is mapped in `../README.md`.
