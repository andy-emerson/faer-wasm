# BLAS-layer implementation plan (architect-approved format, 2026-07-18)

The build list for the BLAS campaign: every operation the layer ships,
and whether its implementation is a SIMD streaming loop. A **why** entry
appears only where the answer is no, and must begin with its evidence
class: **measured** (raced, verdict in `docs/blas-ab-2026-07.md`) or
**structural** (the operation's shape rules the technique out — no
performance claim being made).

Every "no" in this table is earned, not presumed: the three rows that
used to rest on assumptions (`swap`, `asum`, `iamax`) were raced on
2026-07-18 and all three assumptions lost — those rows are now "yes"
(see the Level-1 assumption race in `docs/blas-ab-2026-07.md`).

`copy` (Andy, 2026-07-18): implemented as a SIMD streaming loop for
consistency and completeness. The measurement stands — copy runs at
the bandwidth ceiling and the loop cannot be *faster* than memcpy —
but the race also showed the loop is never slower, so the layer keeps
one uniform implementation family rather than one special case. No
speed claim attaches to this choice.

Per-operation FMA variant choice (from the step-1 three-way race):
fused for `trmm`/`trsm`/`gemv`, plain for `gemm`/`syrk`; the remaining
ops get their variant measured as they are built. Banded/packed forms
(`gbmv`, `sbmv`, `spmv`, `tb*`, `tp*`) are not planned — no consumer
demand.

## Level 1

| BLAS | mathematical name | SIMD streaming loop? | why |
|---|---|---|---|
| `axpy` | scaled vector addition (y ← αx + y) | yes | |
| `dot` | dot product | yes | |
| `scal` | scalar × vector | yes | |
| `rot` | apply a plane rotation | yes | |
| `swap` | exchange two vectors | yes | |
| `asum` | sum of absolute values (ℓ¹ norm) | yes | |
| `iamax` | index of the largest element | yes | |
| `nrm2` | Euclidean length (ℓ² norm) | yes | |
| `copy` | vector copy | yes | |
| `rotg` | generate a plane rotation | no | structural: no arrays — guarded scalar arithmetic on two numbers |

## Level 2

| BLAS | mathematical name | SIMD streaming loop? | why |
|---|---|---|---|
| `gemv` | matrix × vector | yes | |
| `ger` | outer-product update (rank-1) | yes | |
| `trsv` | triangular solve, one vector | yes | |
| `symv` | symmetric matrix × vector | yes | |
| `trmv` | triangular matrix × vector | yes | |
| `syr` / `syr2` | symmetric rank-1/2 updates | yes | |

## Level 3

| BLAS | mathematical name | SIMD streaming loop? | why |
|---|---|---|---|
| `gemm` | matrix multiplication | yes | |
| `syrk` | Gram-matrix update (αAAᵀ + βC) | yes | |
| `trmm` | triangular matrix multiplication | yes | |
| `trsm` | triangular solve, many right-hand sides | yes | |
| `symm` / `syr2k` | symmetric multiply / rank-2k update | yes | |

## Evidence trail

- Level 2/3 "yes" rows: the streaming-vs-faer A/B, three runner draws
  (streaming ≥ faer through n = 512 on the reference class, gemm
  1.07–1.33×) — `docs/blas-ab-2026-07.md`.
- `swap` / `asum` / `iamax`: the Level-1 assumption race, three runner
  draws (1.15–1.33× / 3.5–4× / 1.4–1.6× SIMD wins) — same doc, step 2.
- `copy`: raced in the original A/B on four machines (never separated)
  and clocked at the measured bandwidth ceiling in the step-1 probes;
  the "yes" is an architect consistency decision on top of a measured
  no-harm, not a speed claim.
- FMA per-op verdicts: step-1 three-way race, three runner draws.

Verdict-stability rule applies throughout: only same-machine
interleaved comparisons count, and sub-1.3× margins are trusted for
direction only when unanimous across draws.
