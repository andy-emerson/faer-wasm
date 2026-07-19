# faer-wasm-blas — the BLAS layer

The wasm-native BLAS layer, built as its own finished product per the
2026-07-18 direction reset: the LAPACK-layer kernels re-route their
bulk work onto this crate as it fills in. One file per BLAS routine
per number type in netlib naming (`daxpy.rs`, `saxpy.rs`, … —
convention: `src/L1/README.md`), one folder per level; this README is
the plan of record for the layer. Counting convention, used
everywhere: **23 routines** per real type (netlib names = files; flag
variants folded, so gemv covers N/T and symm/trmm/trsm cover both
sides) which split into **27 distinct operations** (the call-graph
nodes). The complex layers count differently because reference BLAS
itself splits there: **26 routines / 31 operations** each for c64 and
c32 (dot → dotu/dotc, ger → geru/gerc, scal gains a real-α form, gemv
adds the conjugate-transpose form, and the symmetric family becomes
Hermitian). Companion maps: `src/README.md` (who calls whom),
`bench/README.md` (the measured benchmarks), `tests/README.md` (the
bit-identical test results).

**Status: the f64 layer is COMPLETE and TUNED — campaign closed
2026-07-19** (f64, unit stride — callers pass contiguous column
slices; strided access defeats streaming and no consumer wants it).
23 routines (27 operations), 30 tests, 21 determinism probes, a
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
routines, 30 mirrored tests (60 crate-wide), 21 f32 determinism
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

**The c64 layer is built** (2026-07-19) — the first NON-mechanical
clone: 26 routines / 31 operations (counting note above) over a new
`C64` scalar (`src/c64.rs`, layout-compatible with
num_complex/C99/faer — consumers cast slices) with one complex per
F64x2 register. The complex product is a two-multiply sign-folded
lane form that is bit-exactly the canonical scalar order, so the
whole real-layer testing contract carried over: 40 new tests (104
crate-wide), every deterministic-order op bit-locked to a same-order
replay, and 24 c64 determinism probes bit-identical native ↔ wasm on
the container and both runner draws. Six L1 routines
(zdscal/zcopy/zswap/zdrot/dznrm2/dzasum) are pure delegations to the
tuned d-streams over the interleaved 2n-real view — speed and guards
inherited, zswap/zdrot measure 97–100% of ceiling. Runner rooflines
(2 draws, `bench/README.md`): the L3 family at **74–86% of the f64
arithmetic peak** (complex does 4× the FLOPs per byte, so the
fan-out shapes run compute-bound — zgemm 74–79% with NO register
tile), with two honest weak rows tied to one recorded lever: zhemv
ships the single-column fused pass (19–21%), and hemm_left rides it
(39–41%) — the 4-column fused grouping that pushed dsymv to 2× is
the obvious first c64 tuning candidate.

**The c32 layer is built** (2026-07-19, Andy: "C32") — the c64 layer
cloned into single precision with one real lane-geometry change: TWO
complexes per F32x4 register, so the product form works pair-wise
(`swap_pairs`/`dup_even`/`dup_odd`/`neg_even`/`neg_odd`) and odd
lengths leave a one-complex scalar tail — bit-identical by the same
sign-folding proof. 26 routines / 31 operations over `C32`
(`src/c32.rs`); six delegations to the tuned s-streams; 40 new tests
(144 crate-wide, all green); 24 determinism probes bit-identical
native ↔ wasm on the container and both runner draws. Runner
rooflines (2 draws): the L3 family at **75–87% of the f32 arithmetic
peak** — cgemm's 85% of ~30.5 GFLOP/s (≈26 GFLOP/s) is the fastest
absolute row on the whole board — with the same recorded lever
(chemv's un-grouped fused pass at 12%, hemm_left riding it at
55–56%) and icamax inheriting isamax's rescan weakness (13–16%).

Sequencing (Andy, 2026-07-18, revised same day; ROADMAP "BLAS
campaign sequencing"): f64 tuned first — DONE; the tuned layer
cloned into the other number types — f32, c64, c32 all DONE
2026-07-19. **The four-type grid is complete**; per the sequencing,
LAPACK-layer work (the kernel re-route onto this crate) is now
unblocked.

Gaps: FMA variants deferred (above); transpose forms of
gemm/trmv/trsv/trmm/trsm not built (no consumer yet, any type);
complex tuning levers recorded but not raced (register-tile
zgemm/cgemm, 4-column fused zhemv/chemv, the i*amax rescans); the
complex-symmetric symm/syrk/syr2k forms and the complex-s rotation
apply (`zrot`) not built (no consumer); the `cd blas && cargo test`
CI gate line still needs adding to the workflow (session tokens
can't edit workflow files).

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
- **→d** / **→s** — *delegation*: the complex routine is a one-line
  call onto the tuned same-precision real routine over the
  interleaved 2n-real view (`src/c64.rs` / `src/c32.rs`) — a real
  scale/copy/swap/rotation/reduction of complex data IS the real
  operation on twice the elements, so speed, guards, and determinism
  are inherited rather than duplicated.

The fan-out/fan-in kernels preserve the per-element rounding
sequence, so tuned ops stay bit-for-bit against their plain
column-axpy replays (the two triangular right-side cases whose
elimination order forbids that document their reorder and the tests
mirror it; symv's fused pass legitimately re-folds its reduction and
is bounds-tested). The tables give each op's shorthand per number
type — matching cells across type columns are the clones doing their
job.

The step-1 three-way race measured fused-FMA better for
`trmm`/`trsm`/`gemv` and harmful for `syrk` — those variants are
deferred at the campaign close (determinism trade-off; see status
above). Banded/packed forms are not planned — no consumer demand.
Evidence per row: `../docs/blas-ab-2026-07.md`.

The type columns map to the routine-name prefixes (d/s/z/c — the
full convention with the per-routine deviations from reference BLAS:
`src/L1/README.md`); a row names the routine family, so the f64 cell
of `axpy` describes `daxpy`, the f32 cell `saxpy`. In the c64 column
the symmetric rows describe the **Hermitian** z-routine (symv →
`zhemv`, syr/syr2 → `zher`/`zher2`, symm → `zhemm`, syrk → `zherk`,
syr2k → `zher2k`), per the counting note above.

## Level 1 — `src/L1/`

| BLAS | mathematical name | f64 | f32 | c64 | c32 |
|---|---|---|---|---|---|
| `axpy` | scaled vector addition (y ← αx + y) | ES | ES | ES | ES |
| `scal` | scalar × vector | ES | ES | ES ² | ES ² |
| `copy` | vector copy | ES | ES | →d | →s |
| `swap` | exchange two vectors | ES | ES | →d | →s |
| `rot` | apply a plane rotation | ES | ES | →d ³ | →s ³ |
| `dot` | dot product | RS | RS | RS ⁴ | RS ⁴ |
| `nrm2` | Euclidean length (ℓ² norm) | RS | RS | →d | →s |
| `asum` | sum of absolute values (ℓ¹ norm) | RS | RS | →d ⁵ | →s ⁵ |
| `iamax` | index of the largest element | RS | RS | RS ⁶ | RS ⁶ |
| `rotg` | generate a plane rotation | G | G | G | G |

² `zscal`/`cscal` (complex α); the real-α forms `zdscal`/`csscal`
are delegations.
³ `zdrot`/`csrot` — the real-c,s rotation; a real rotation acts on
re/im independently, so it IS `drot`/`srot` on the 2n-real view. The
complex-s apply (`zrot`) has no consumer yet (gap).
⁴ splits into `dotu` (xᵀy) and `dotc` (xᴴy) — different results,
both 4-accumulator reduction streams.
⁵ reference semantics Σ(|re|+|im|) — component magnitudes ARE
`dasum`/`sasum` of the 2n-real view.
⁶ `izamax`/`icamax` maximize |re|+|im| per element — own lane pass
(abs + swap-add), idamax's two-pass shape.

## Level 2 — `src/L2/`

| BLAS | mathematical name | f64 | f32 | c64 | c32 |
|---|---|---|---|---|---|
| `gemv` | matrix × vector | CA+FI | CA+FI | CA+FI | CA+FI |
| `gemv_t` | transposed-matrix × vector (Aᵀx, transpose never formed) | RS per column | RS per column | RS per column ⁷ | RS per column ⁷ |
| `ger` | outer-product update (rank-1) | CA | CA | CA ⁸ | CA ⁸ |
| `symv` | symmetric matrix × vector | FS | FS | FS ⁹ | FS ⁹ |
| `trmv` | triangular matrix × vector | CA+FI | CA+FI | CA+FI | CA+FI |
| `syr` / `syr2` | symmetric rank-1/2 updates | CA | CA | CA | CA |
| `trsv` | triangular solve, one vector | DCA+FI | DCA+FI | DCA+FI ¹⁰ | DCA+FI ¹⁰ |

⁷ two forms: `_t` (Aᵀx, `dotu` per column) and `_c` (Aᴴx, `dotc`
per column) — the conjugate transpose is the one complex algorithms
actually consume.
⁸ splits into `geru` (αxyᵀ) and `gerc` (αxyᴴ) — the conjugation
lands on the per-column scalar, never on a stream.
⁹ the Hermitian `hemv` fused pass is single-column
(`zaxpy_dotc`/`caxpy_dotc`); the 4-column grouping that pushed dsymv
to 2× is a recorded complex tuning lever.
¹⁰ complex division by Smith's algorithm (`C64`/`C32`'s guarded `/`
— the `dladiv` shape).

## Level 3 — `src/L3/`

| BLAS | mathematical name | f64 | f32 | c64 | c32 |
|---|---|---|---|---|---|
| `gemm` | matrix multiplication | CA+RT/FO | CA+RT/FO | CA+FO ¹¹ | CA+FO ¹¹ |
| `syrk` | Gram-matrix update (αAAᵀ + βC) | CA+FO | CA+FO | CA+FO ¹² | CA+FO ¹² |
| `trmm` | triangular matrix multiplication | CA+FO | CA+FO | CA+FO | CA+FO |
| `symm` / `syr2k` | symmetric multiply / rank-2k update | CA+FO ¹ | CA+FO ¹ | CA+FO ¹ | CA+FO ¹ |
| `trsm` | triangular solve, many right-hand sides | DCA+FO | DCA+FO | DCA+FO | DCA+FO |

¹ left-side symm/hemm is FS per column of B.
¹¹ no register tile for the complex types yet — a complex tile is a
different register geometry, a recorded tuning lever, not a
mechanical port; `zgemm`/`cgemm` route everything through the col4
fan-out with the `_colaxpy` reference kept bit-checked.
¹² the Hermitian `herk`/`her2k` take real α (herk) / real β per the
reference signatures, and maintain the Hermitian invariant: stored
diagonals end exactly real.
