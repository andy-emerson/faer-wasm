# faer-wasm-blas — the BLAS layer

The wasm-native BLAS layer, built as its own finished product per the
2026-07-18 direction reset: the LAPACK-layer kernels re-route their
bulk work onto this crate as it fills in. One file per BLAS routine
per number type in netlib naming (`daxpy.rs`, `saxpy.rs`, … —
convention: `src/README.md`), one folder per level; this README is
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

**Status — all four type layers are BUILT; f64 TUNED (campaign
closed 2026-07-19); complex tuning levers recorded.** Unit stride
throughout — callers pass contiguous column slices (strided access
defeats streaming and no consumer wants it). One-line history, full
plain-English record in `../STATUS.md`, evidence per step in
`../docs/blas-ab-2026-07.md`, current numbers in `bench/README.md`:

- **f64** (built 2026-07-18, tuned + campaign closed 2026-07-19,
  docs steps 3–9): every verdict backed by two runner draws; gemm
  beats faer at every measured size (1.25–1.8×, size-dispatched
  tile/col4); dot AT the triad read ceiling; symm_left 84–86% of the
  arithmetic peak via the fused symv. One candidate raced and
  REFUTED (fused single-pass iamax — reverted, loss recorded in its
  module docs). Fused-FMA variants DEFERRED at the close
  (relaxed-madd rounding is implementation-dependent — trades away
  cross-target bit-identity; architect decision, in ROADMAP).
- **f32** (built 2026-07-19, docs step 10): the tuned layer cloned
  one-for-one at double lane width; two measured deviations (8×4
  gemm tile, 3 MB dispatch threshold — runner-raced, container
  overruled); same ceiling fractions as f64.
- **c64** (built 2026-07-19, docs step 11; tuned at the close-out,
  step 13): the first non-mechanical clone — own `C64` scalar,
  sign-folded lane product bit-exact to the scalar order, six L1
  delegations onto the tuned d-streams; L3 at 74–86% of the f64
  arithmetic peak (complex is compute-bound at 4× FLOPs/byte). The
  4-column grouped fused zhemv won its race (~13%, both draws) and
  ships; hemm_left rode it from 39–41% to 54–61% of peak. Market:
  zgemm beats faer's blocked complex gemm 1.49–1.71× at n≥256.
- **c32** (built 2026-07-19, docs step 12; close-out race step 13):
  c64 at two-complexes-per-register lane geometry; L3 at 75–87% of
  the f32 peak — cgemm ~26 GFLOP/s, the fastest absolute row in the
  library, and 3.1–3.7× over faer's blocked complex gemm at every
  size. The hemv grouping LOST for c32 (~2%, both draws — refuted,
  recorded in `src/L2/chemv.rs`); chemv keeps the single-column
  pass.

Sequencing (Andy, 2026-07-18, revised same day; ROADMAP "BLAS
campaign sequencing"): f64 tuned first — DONE; the tuned layer
cloned into the other number types — f32, c64, c32 all DONE
2026-07-19. **The four-type grid is complete**; per the sequencing,
LAPACK-layer work (the kernel re-route onto this crate) is now
unblocked.

Gaps: FMA variants deferred (above); transpose forms of
gemm/trmv/trsv/trmm/trsm not built (no consumer yet, any type);
remaining recorded-not-raced levers — register-tile zgemm/cgemm (a
design, and gemm already runs 74–94% of peak in complex, so the
headroom is thin) and the i*amax rescans (worst: isamax 8%); the
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
--release`): every routine is tested to the strongest standard its
math allows — bit-for-bit against a same-order scalar replay
wherever the rounding order is deterministic, n-scaled error bounds
against higher-precision references where lane parallelism
legitimately reorders, exact-index/identity checks for iamax/rotg,
residuals for the solves, and native ↔ wasm bit-identity throughout.
The full rationale (what standard applies to which routine class and
why) and the per-routine table live in `tests/README.md`.

**Performance — `bench/` in this crate**: the score is **distance
from the machine's measured ceiling** — streaming ops against the
bandwidth ceiling, multiply-class ops against the arithmetic peak —
per the re-derived success metric, with same-machine interleaved A/B
rows and the verdict-stability rule (two runner draws per claim; the
dev container is iteration only). The harness is self-contained (no
faer); method and current scoreboard: `bench/README.md`. Market
races against faer need faer and stay in `../bench/`.

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
`src/README.md`); a row names the routine family, so the f64 cell
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
