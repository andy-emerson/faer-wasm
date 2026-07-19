# BLAS-layer implementation plan (architect-approved format, 2026-07-18)

> **Historical document.** This is the pre-build plan; the layer has
> since been built and tuned. The living plan of record — including
> the tuned per-op loop shapes — is `blas/README.md`; the campaign
> evidence is `docs/blas-ab-2026-07.md` steps 3–9.

The build list for the BLAS campaign: every operation the layer ships,
classified by implementation. The whole layer reduces to **four SIMD
streaming-loop shapes plus one scalar function**:

- **elementwise stream** — one pass over the vector(s); lanes are
  transformed and written back (includes the fused y ← αx + y form).
- **reduction stream** — one pass; parallel accumulator lanes, folded
  to a single number at the end.
- **column-axpy** — the matrix operation runs as one elementwise/axpy
  stream per column.
- **divide-then-column-axpy** — triangular solves: divide by the
  diagonal entry, then stream the elimination update through the
  remaining columns.

`rotg` is the sole exception: no arrays = no SIMD. It ships as guarded
scalar arithmetic, inlined branch-free into the sweep loops that call
it (LAPACK's overflow guards kept — proven numerics).

Evidence per row is in `docs/blas-ab-2026-07.md`: the three rows that
used to rest on assumptions (`swap`, `asum`, `iamax`) were raced on
2026-07-18 and all three assumptions lost. `copy` (Andy, 2026-07-18)
is a streaming loop by architect consistency decision on top of a
measured no-harm — copy runs at the bandwidth ceiling, so no speed
claim attaches.

Per-operation FMA variant choice (from the step-1 three-way race):
fused for `trmm`/`trsm`/`gemv`, plain for `gemm`/`syrk`; the remaining
ops get their variant measured as they are built. Banded/packed forms
(`gbmv`, `sbmv`, `spmv`, `tb*`, `tp*`) are not planned — no consumer
demand.

## Level 1

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

## Level 2

| BLAS | mathematical name | implementation |
|---|---|---|
| `gemv` | matrix × vector | column-axpy |
| `ger` | outer-product update (rank-1) | column-axpy |
| `symv` | symmetric matrix × vector | column-axpy |
| `trmv` | triangular matrix × vector | column-axpy |
| `syr` / `syr2` | symmetric rank-1/2 updates | column-axpy |
| `trsv` | triangular solve, one vector | divide-then-column-axpy |

## Level 3

| BLAS | mathematical name | implementation |
|---|---|---|
| `gemm` | matrix multiplication | column-axpy |
| `syrk` | Gram-matrix update (αAAᵀ + βC) | column-axpy |
| `trmm` | triangular matrix multiplication | column-axpy |
| `symm` / `syr2k` | symmetric multiply / rank-2k update | column-axpy |
| `trsm` | triangular solve, many right-hand sides | divide-then-column-axpy |

## Evidence trail

- Level 2/3 rows: the streaming-vs-faer A/B, three runner draws
  (streaming ≥ faer through n = 512 on the reference class, gemm
  1.07–1.33×) — `docs/blas-ab-2026-07.md`.
- `swap` / `asum` / `iamax`: the Level-1 assumption race, three runner
  draws (1.15–1.33× / 3.5–4× / 1.4–1.6× SIMD wins) — same doc, step 2.
- `copy`: raced in the original A/B on four machines (never separated)
  and clocked at the measured bandwidth ceiling in the step-1 probes;
  the streaming loop is an architect consistency decision on top of a
  measured no-harm, not a speed claim.
- FMA per-op verdicts: step-1 three-way race, three runner draws.

Verdict-stability rule applies throughout: only same-machine
interleaved comparisons count, and sub-1.3× margins are trusted for
direction only when unanimous across draws.
