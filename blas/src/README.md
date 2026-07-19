# src/ — the call graph

Which operations call which operations. **One node = one
mathematically distinct operation** — 27 of them: the 23 netlib-named
routines (= the files per type) with the flag variants split out,
because a transposed product or a side swap is a different result
(gemv_t; symm/trmm/trsm left and right). gemm's internal loop shapes
are NOT extra nodes — they produce bit-identical results. This graph
covers the f64 and f32 layers, which have identical structure; the
c64 layer's differences are listed below it.

```mermaid
graph TD
  subgraph L3
    gemm
    symm_left
    symm_right
    syrk
    syr2k
    trmm_left
    trmm_right
    trsm_left
    trsm_right
  end

  subgraph L2
    gemv
    gemv_t
    ger
    symv
    trmv
    trsv
    syr
    syr2
  end

  subgraph L1
    axpy
    dot
    scal
    copy
    swap
    rot
    nrm2
    asum
    iamax
    rotg
  end

  gemm --> gemv
  symm_left --> symv
  symm_right --> axpy
  syrk --> axpy
  syr2k --> axpy
  trmm_left --> trmv
  trmm_left --> scal
  trmm_right --> axpy
  trmm_right --> scal
  trsm_left --> trsv
  trsm_left --> scal
  trsm_right --> axpy
  trsm_right --> scal

  gemv --> axpy
  gemv_t --> dot
  ger --> axpy
  trmv --> axpy
  trsv --> axpy
  syr --> axpy
  syr2 --> axpy
```

Notes:

- Below the operations sits shared plumbing, deliberately not in the
  graph: the private SIMD kernels (`kernels.rs` — blocked hot loops
  several operations share), the lane types (`lanes.rs`), gemm's
  internal shapes and dispatcher (inside the gemm files), and the
  small helpers (`check_mat`, `{d,s}scale_y`, `{d,s}sym_at`).
- `symv` has no outgoing arrows: its fused kernel replaced the
  axpy+dot composition it used to be.
- `rotg` and copy/swap/rot/nrm2/asum/iamax are leaves — nothing in
  the crate calls them; consumers do.

## The complex layers (z- and c-routines) — same graph, renamed nodes

The two complex layers (c64 z-prefixed, c32 c-prefixed — identical
structure) are the same picture with node substitutions — **26
routines / 31 operations** each (counting convention: crate README):

- *Splits*: `dot` → `dotu` + `dotc` (unconjugated/conjugated are
  different results); `ger` → `geru` + `gerc`; `gemv` gains a third
  form `gemv_c` (y ← αAᴴx, one `dotc` per column) beside `gemv` and
  `gemv_t`; `scal` gains the real-α `dscal` form.
- *Renames*: the symmetric family becomes Hermitian — `symv` → `hemv`
  (fused, no outgoing arrows, same as symv), `syr`/`syr2` →
  `her`/`her2`, `symm` → `hemm`, `syrk` → `herk`, `syr2k` → `her2k` —
  with the same edges as their real twins.
- *Delegations, not nodes*: copy/swap/rot/the real-α scal/nrm2/asum
  are one-line calls to the tuned same-precision real routines on the
  interleaved 2n-real view (`c64.rs`/`c32.rs`) — they inherit the
  real stream's speed, guards, and determinism rather than
  duplicating the loops (c64: zcopy/zswap/zdrot/zdscal/dznrm2/dzasum;
  c32: ccopy/cswap/csrot/csscal/scnrm2/scasum).
- `rot` is the real-c,s form (`zdrot`/`csrot`); `zrotg`/`crotg`
  generate the complex Givens (c real, s complex) — the complex-s
  application (`zrot`) has no consumer yet.
