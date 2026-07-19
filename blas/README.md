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
read-path (triad) ceiling; the reductions run 4 accumulator registers
since tuning lever 2 — runner-confirmed on both draws: dot AT the
triad read ceiling, nrm2/asum at 73–97% of it (was 60–80%).
Reductions are bit-identical
native ↔ wasm by construction (`src/lanes.rs` emulates the SIMD lane
structure elementwise off-wasm — verified 4/4 probes on the container
and both runner draws). Full record: `../docs/blas-ab-2026-07.md`
step 3.

**Level 2 is implemented** the same way (f64, 9 tests, 8/8
determinism probes bit-identical, runner roofline in the same doc,
step 4): every function is a loop of Level 1 calls over column slices
— the classification table below is literally the module structure.
The rank-1 updates run at 83–100% of ceiling; gemv ships 4-column
fan-in blocking since the tuning campaign — runner-confirmed 29–31
GB/s vs ~17 before — and gemv_t rose to 22–29 GB/s untouched, by
inheriting the 4-accumulator dot through composition. Remaining
recorded levers: fused symv pass (in test), trmv/trsv blocking.

**Level 3 is implemented — the f64 layer is complete** (2026-07-18):
23 functions, 30 tests, 21 cross-target determinism probes (all
bit-identical native ↔ wasm on the container and every runner draw),
roofline rows for every operation (`../docs/blas-ab-2026-07.md` steps
3–5).

**The f64 tuning campaign is in progress** (steps 6–7 in the same
doc): gemm now dispatches a 4×4 register tile / 4-column fan-out by
size and beats faer's blocked gemm at every measured size (two runner
draws); the reductions run 4 accumulators; the shared blocked kernels
(`src/kernels.rs`) carry the fan-out/fan-in shapes through gemv and
the rest of Level 3 — all runner-confirmed on two draws (L3 family
48–56% of arithmetic peak, was 34–44%; runs 29669397117/29669395117).
Remaining levers are listed in ROADMAP.

Sequencing (Andy, 2026-07-18, revised same day; ROADMAP "BLAS
campaign sequencing"): finish tuning + benchmarking f64 first, then
clone the tuned layer into the other number types (f32, c64), then —
only then — LAPACK work resumes.

Gaps: f32/c64 variants queued; c32 undecided (never shipped anywhere
in the project); FMA variants per-op-measured as built; transpose
forms of gemm/trmv/trsv/trmm/trsm not built (no consumer yet); the
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
- *Level 2/3*: **bit-for-bit against a same-order scalar replay**
  wherever the operation's add order is deterministic (gemv, ger,
  syr/syr2, trmv, trsv, and all of Level 3 except `symm_left`) — this
  is also what lets tuned loop shapes ship invisibly: a blocked
  variant must reproduce the replay's bits or document its reorder
  and mirror it in the replay. Plus independent n-scaled error bounds
  computed in a different accumulation order (and residual checks for
  the solves).
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

The tuning campaign (2026-07-18/19) blocks the column shapes four
columns at a time without changing any of them — two shared kernels in
`src/kernels.rs` do all the work: **fan-out** (one source column
streamed once into four destination columns — source traffic 4× down)
and **fan-in** (four source columns accumulated in one pass over the
destination — destination read-modify-write traffic 4× down); gemm
additionally carries a 4×4 register tile for small matrices. Both
kernels preserve the per-element rounding sequence, so tuned ops stay
bit-for-bit against their plain column-axpy replays (the two
triangular right-side cases whose elimination order forbids that
document their reorder and the tests mirror it). The table rows note
which blocked shape each op ships with.

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
| `gemv` | matrix × vector | column-axpy, 4-column fan-in |
| `ger` | outer-product update (rank-1) | column-axpy |
| `symv` | symmetric matrix × vector | fused stream, 4-column grouped |
| `trmv` | triangular matrix × vector | column-axpy, 4-column fan-in |
| `syr` / `syr2` | symmetric rank-1/2 updates | column-axpy |
| `trsv` | triangular solve, one vector | divide-then-column-axpy, 4-column fan-in |

## Level 3 — `src/level3/`

| BLAS | mathematical name | implementation |
|---|---|---|
| `gemm` | matrix multiplication | column-axpy, size-dispatched 4×4 tile / 4-column fan-out |
| `syrk` | Gram-matrix update (αAAᵀ + βC) | column-axpy, 4-column fan-out |
| `trmm` | triangular matrix multiplication | column-axpy, 4-column fan-out |
| `symm` / `syr2k` | symmetric multiply / rank-2k update | column-axpy, 4-column fan-out (left symm: symv per column) |
| `trsm` | triangular solve, many right-hand sides | divide-then-column-axpy, 4-column fan-out |
