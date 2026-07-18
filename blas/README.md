# faer-wasm-blas — the BLAS layer

The wasm-native BLAS layer, built as its own finished product per the
2026-07-18 direction reset: the LAPACK-layer kernels re-route their
bulk work onto this crate as it fills in. One file per function, one
folder per level; this README is the plan of record for the layer.

**Status: Level 1 implemented** (f64, unit stride — callers pass
contiguous column slices; strided access defeats streaming and no
consumer wants it). All ten functions shipped with correctness tests
(`tests/level1.rs`, 12 tests) and runner-measured roofline rows
(`../bench/l1-roofline.mjs`): the read-modify-write streams run at
81–100% of the machine's fastest same-run stream, copy/dot at the
read-path (triad) ceiling, reductions at 60–80% of triad (recorded
lever: more accumulator registers). Reductions are bit-identical
native ↔ wasm by construction (`src/lanes.rs` emulates the SIMD lane
structure elementwise off-wasm — verified 4/4 probes on the container
and both runner draws). Full record: `../docs/blas-ab-2026-07.md`
step 3. Levels 2–3 are scaffold. Gaps: f32 and c64 variants queued
behind the f64 layer; FMA variants per-op-measured as built; the
`cd blas && cargo test` CI gate line still needs adding to the
workflow (session tokens can't edit workflow files).

Hard-won build rule: simd128 is NOT in rustc's default wasm32 feature
set — every SIMD path must sit under `#[target_feature(enable =
"simd128")]` on the whole call chain (see `src/lanes.rs`), or the
intrinsics compile as out-of-line calls (measured 6.4× slowdown).

## Testing contract — two axes, both required to land

**Correctness — `tests/` in this crate** (`cd blas && cargo test
--release`). Each function is tested to the strongest standard its
math allows:

- *Elementwise streams* (copy, swap, scal, axpy, rot): **bit-for-bit**
  against the scalar definition — SIMD lanes don't change the rounding
  sequence of any individual element, so there is no excuse for any
  difference. An FMA variant is checked bit-for-bit against the *fused*
  scalar definition (one rounding instead of two — a different, equally
  valid reference, documented per variant).
- *Reduction streams* (dot, nrm2, asum): lane-parallel accumulation
  legitimately reorders the additions, so bit-for-bit against
  sequential reference BLAS is mathematically the wrong demand. The
  standard is agreement with a higher-precision reference within
  n-scaled floating-point error bounds. `iamax` is the exception that
  IS exact: the returned index, including BLAS's first-occurrence
  tie-breaking rule, must match precisely.
- *Level 2/3*: agreement with a reference implementation within
  n-scaled error bounds.
- *Everything*: **native ↔ wasm bit-identical for our own code** — the
  project's standing determinism guarantee. Cross-target difference is
  a bug, not noise.

**Performance — `../bench/`** (timing runs in the wasm runtime on the
reference CI machines, so it lives in the bench harness, not in cargo
tests). The score is **distance from the machine's measured ceiling**:
streaming ops against the bandwidth ceiling, multiply-class ops against
the arithmetic peak — per the re-derived success metric. Method: the
ceiling probes (`bench/ceilings.mjs`) plus same-machine interleaved
A/B rows, verdict-stability rule throughout.

## Implementation taxonomy

The whole layer reduces to **four SIMD streaming-loop shapes plus one
scalar function**:

- **elementwise stream** — one pass over the vector(s); lanes are
  transformed and written back (includes the fused y ← αx + y form).
- **reduction stream** — one pass; parallel accumulator lanes, folded
  to a single number at the end.
- **column-axpy** — the matrix operation runs as one elementwise/axpy
  stream per column.
- **divide-then-column-axpy** — triangular solves: divide by the
  diagonal entry, then stream the elimination update through the
  remaining columns.

`rotg` is the sole exception: no arrays = no SIMD. Guarded scalar
arithmetic, inlined branch-free into the sweep loops that call it
(LAPACK's overflow guards kept — proven numerics).

Per-operation FMA variant choice (measured, step-1 three-way race):
fused for `trmm`/`trsm`/`gemv`, plain for `gemm`/`syrk`; the rest
measured as built. Banded/packed forms are not planned — no consumer
demand. Evidence per row: `../docs/blas-ab-2026-07.md`.

## Level 1 — `src/level1/`

| BLAS | mathematical name | implementation |
|---|---|---|
| `axpy` | scaled vector addition (y ← αx + y) | elementwise stream |
| `scal` | scalar × vector | elementwise stream |
| `copy` | vector copy | elementwise stream |
| `swap` | exchange two vectors | elementwise stream |
| `rot` | apply a plane rotation | elementwise stream |
| `dot` | dot product | reduction stream |
| `nrm2` | Euclidean length (ℓ² norm) | reduction stream |
| `asum` | sum of absolute values (ℓ¹ norm) | reduction stream |
| `iamax` | index of the largest element | reduction stream |
| `rotg` | generate a plane rotation | no arrays = no SIMD |

## Level 2 — `src/level2/`

| BLAS | mathematical name | implementation |
|---|---|---|
| `gemv` | matrix × vector | column-axpy |
| `ger` | outer-product update (rank-1) | column-axpy |
| `symv` | symmetric matrix × vector | column-axpy |
| `trmv` | triangular matrix × vector | column-axpy |
| `syr` / `syr2` | symmetric rank-1/2 updates | column-axpy |
| `trsv` | triangular solve, one vector | divide-then-column-axpy |

## Level 3 — `src/level3/`

| BLAS | mathematical name | implementation |
|---|---|---|
| `gemm` | matrix multiplication | column-axpy |
| `syrk` | Gram-matrix update (αAAᵀ + βC) | column-axpy |
| `trmm` | triangular matrix multiplication | column-axpy |
| `symm` / `syr2k` | symmetric multiply / rank-2k update | column-axpy |
| `trsm` | triangular solve, many right-hand sides | divide-then-column-axpy |
