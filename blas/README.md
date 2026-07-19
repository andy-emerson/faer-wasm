# faer-wasm-blas — the BLAS layer

The wasm-native BLAS layer, built as its own finished product per the
2026-07-18 direction reset: the LAPACK-layer kernels re-route their
bulk work onto this crate as it fills in. One file per BLAS routine
per number type in netlib naming (`daxpy.rs`, `saxpy.rs`, … —
convention: `src/L1/README.md`), one folder per level; this README is
the plan of record for the layer. Companion maps: `src/README.md`
(who calls whom), `bench/README.md` (the measured benchmarks),
`tests/README.md` (the bit-identical test results).

**Status: the f64 layer is COMPLETE and TUNED — campaign closed
2026-07-19** (f64, unit stride — callers pass contiguous column
slices; strided access defeats streaming and no consumer wants it).
23 functions, 30 tests, 21 cross-target determinism probes, a
runner-measured roofline row for every operation, every tuning verdict
backed by two reference-machine draws. Full record:
`../docs/blas-ab-2026-07.md` steps 3–9. Where the layer landed:

- **Level 1**: read-modify-write streams at 81–100% of the machine's
  fastest same-run stream; dot AT the triad read ceiling, nrm2/asum at
  73–97% of it (4-accumulator reductions). Reductions are
  bit-identical native ↔ wasm BY CONSTRUCTION (`src/lanes.rs` emulates
  the SIMD lane structure elementwise off-wasm).
- **Level 2**: rank-1 updates 79–100% of ceiling; gemv 29–31 GB/s
  (was ~17) via 4-column fan-in; gemv_t 22–29 GB/s untouched — it
  inherited the 4-accumulator dot through composition; symv ~2× via
  the fused 4-column pass; trmv/trsv ~1.3× via fan-in blocking.
- **Level 3**: the family runs at 48–56% of the machine's arithmetic
  peak (was 34–44%), gemm beats faer's blocked gemm at every measured
  size (1.25–1.8×, size-dispatched tile/fan-out), and symm_left —
  riding the fused symv — reaches **84–86% of peak**, the best
  matrix–matrix row on the board.

One candidate was raced and REFUTED (fused single-pass iamax — slower
than the shipped two-pass shape on both draws; reverted, loss recorded
in its module docs). The per-op fused-FMA variants are **deferred by
the campaign close**: wasm relaxed-madd rounding is
implementation-dependent, so shipping them trades away cross-target
bit-identity — an architect decision, recorded in ROADMAP.

**The f32 layer is built** (2026-07-19, Andy: "same treatment as
f64") — the tuned f64 layer cloned one-for-one (the s-routines): same
shapes, same testing contract, on four f32 lanes per
register (`lanes::F32x4`, bit-identical native emulation). 23
functions, 30 mirrored tests (60 crate-wide), 21 f32 determinism
probes; reduction bounds check against an f64-accumulated reference.
Deliberate differences, both measured: the gemm register tile covers
8 rows × 4 columns (same register count, double lane width), and the
tile/col4 dispatch threshold is 3 MB of A — the f32 crossover raced
on both runner draws (tiled unanimous through n=512, col4 unanimous
at 1024; the container said the opposite and was overruled).
Consumer path: the s-prefixed routines in `faer_wasm_blas::L{1,2,3}`. Runner rooflines
in `../docs/blas-ab-2026-07.md` step 10: f32 arithmetic peak ~1.8×
f64's, the L3 family at the same fractions of it (48–58%, symm_left
79–82%), reductions on the read path, 21 probes bit-identical on
both draws.

Sequencing (Andy, 2026-07-18, revised same day; ROADMAP "BLAS
campaign sequencing"): f64 tuned first — DONE; the tuned layer is
being cloned into the other number types (f32 — DONE; c64 next),
then — only then — LAPACK work resumes.

Gaps: c64 variant queued; c32 undecided (never shipped anywhere
in the project); FMA variants deferred (above); transpose forms of
gemm/trmv/trsv/trmm/trsm not built (no consumer yet, both types); the
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

**Performance — `bench/` in this crate** (timing runs in the wasm
runtime on the reference CI machines, so it lives in a harness, not
in cargo tests; the harness is self-contained — no faer — and its
README documents the method). The score is **distance from the
machine's measured ceiling**: streaming ops against the bandwidth
ceiling, multiply-class ops against the arithmetic peak — per the
re-derived success metric, with same-machine interleaved A/B rows and
the verdict-stability rule throughout. Current results:
`bench/README.md` (the scoreboard). Market races against faer need
faer and stay in `../bench/`.

## Implementation taxonomy

The whole layer reduces to **four SIMD streaming-loop shapes plus one
scalar function** — the bold codes are the shorthand the tables below
use:

- **ES** — *elementwise stream*: one pass over the vector(s); lanes
  are transformed and written back (includes the fused y ← αx + y
  form).
- **RS** — *reduction stream*: one pass; parallel accumulator lanes,
  folded to a single number at the end.
- **CA** — *column-axpy*: the matrix operation runs as one
  elementwise/axpy stream per column.
- **DCA** — *divide-then-column-axpy*: triangular solves — divide by
  the diagonal entry, then stream the elimination update through the
  remaining columns.

**G** (`rotg`) is the sole exception: no arrays = no SIMD. Guarded
scalar arithmetic, inlined branch-free into the sweep loops that call
it (LAPACK's overflow guards kept — proven numerics).

The tuning campaign (2026-07-18/19) blocks the column shapes four
columns at a time without changing any of them — the shared kernels in
`src/kernels.rs` do the work — the remaining table codes:

- **+FO** — *4-column fan-out* (`axpy4`): one source column streamed
  once into four destination columns — source traffic 4× down.
- **+FI** — *4-column fan-in* (`axpy4in`): four source columns
  accumulated in one pass over the destination — destination
  read-modify-write traffic 4× down.
- **FS** — *fused symv pass* (`axpy_dot`/`axpy_dot4`): one column
  load serves both triangles of the symmetric update, four columns
  grouped per pass over x and y.
- **RT** — *register tile*: gemm's small-matrix micro-kernel (4×4 in
  f64, 8×4 in f32 — same register count at each lane width);
  `CA+RT/FO` means size-dispatched between the tile and fan-out at
  the measured crossover.
- **—** — not built for that type (c64 is the next campaign; c32
  undecided — nothing has ever shipped c32). The fan-out/fan-in
kernels preserve the per-element rounding sequence, so tuned ops stay
bit-for-bit against their plain column-axpy replays (the two
triangular right-side cases whose elimination order forbids that
document their reorder and the tests mirror it; symv's fused pass
legitimately re-folds its reduction and is bounds-tested). The tables
give each op's shorthand per number type — matching f64/f32 cells are
the clone doing its job.

The step-1 three-way race measured fused-FMA better for
`trmm`/`trsm`/`gemv` and harmful for `syrk` — those variants are
deferred at the campaign close (determinism trade-off; see status
above). Banded/packed forms are not planned — no consumer demand.
Evidence per row: `../docs/blas-ab-2026-07.md`.

The type columns map to the routine-name prefixes (d/s/z/c — the
full convention with the per-routine deviations from reference BLAS:
`src/L1/README.md`); a row names the routine family, so the f64 cell
of `axpy` describes `daxpy`, the f32 cell `saxpy`.

## Level 1 — `src/L1/`

| BLAS | mathematical name | f64 | f32 | c64 | c32 |
|---|---|---|---|---|---|
| `axpy` | scaled vector addition (y ← αx + y) | ES | ES | — | — |
| `scal` | scalar × vector | ES | ES | — | — |
| `copy` | vector copy | ES | ES | — | — |
| `swap` | exchange two vectors | ES | ES | — | — |
| `rot` | apply a plane rotation | ES | ES | — | — |
| `dot` | dot product | RS | RS | — | — |
| `nrm2` | Euclidean length (ℓ² norm) | RS | RS | — | — |
| `asum` | sum of absolute values (ℓ¹ norm) | RS | RS | — | — |
| `iamax` | index of the largest element | RS | RS | — | — |
| `rotg` | generate a plane rotation | G | G | — | — |

## Level 2 — `src/L2/`

| BLAS | mathematical name | f64 | f32 | c64 | c32 |
|---|---|---|---|---|---|
| `gemv` | matrix × vector | CA+FI | CA+FI | — | — |
| `gemv_t` | transposed-matrix × vector (Aᵀx, transpose never formed) | RS per column | RS per column | — | — |
| `ger` | outer-product update (rank-1) | CA | CA | — | — |
| `symv` | symmetric matrix × vector | FS | FS | — | — |
| `trmv` | triangular matrix × vector | CA+FI | CA+FI | — | — |
| `syr` / `syr2` | symmetric rank-1/2 updates | CA | CA | — | — |
| `trsv` | triangular solve, one vector | DCA+FI | DCA+FI | — | — |

## Level 3 — `src/L3/`

| BLAS | mathematical name | f64 | f32 | c64 | c32 |
|---|---|---|---|---|---|
| `gemm` | matrix multiplication | CA+RT/FO | CA+RT/FO | — | — |
| `syrk` | Gram-matrix update (αAAᵀ + βC) | CA+FO | CA+FO | — | — |
| `trmm` | triangular matrix multiplication | CA+FO | CA+FO | — | — |
| `symm` / `syr2k` | symmetric multiply / rank-2k update | CA+FO ¹ | CA+FO ¹ | — | — |
| `trsm` | triangular solve, many right-hand sides | DCA+FO | DCA+FO | — | — |

¹ left-side symm is FS per column of B.
