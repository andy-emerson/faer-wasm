# Eigvals (nonsymmetric EVD) wasm research — 2026-07-10

Architect direction: deep-research eigvals for speed and correctness on
wasm. faer's `eigenvalues()` measured **0.3–0.4× scipy** — the worst ratio
in the suite. Two tracks ran in parallel: the adversarially-verified
deep-research harness (results appended when it completes) and the
empirical track below (faer source reading + runner phase/parameter
probes, `bench/evd-tune.mjs`, run 29118452323). The empirical track found
the root cause before the harness returned.

## What faer's eigvals pipeline actually is (source-verified)

- `eigenvalues()` → `evd_real` with `ComputeEigenvectors::No`: Hessenberg
  reduction, then `real_schur::multishift_qr` with `want_t=false`,
  `Z=None` — the LAPACK `JOB='E'` equivalent. **No unrequested vector
  work** (hypothesis eliminated).
- faer HAS modern small-bulge multishift QR + aggressive early deflation
  (`multishift_qr`, `aggressive_early_deflation`, scalar `lahqr`
  fallback), structurally mirroring `dlaqr0`/`dlaqr2-5`, with
  `SchurParams { recommended_shift_count, recommended_deflation_window,
  blocking_threshold=75, nibble_threshold=50 }`.
- Quirks vs LAPACK found by source diff: the AED window is *always*
  solved by scalar `lahqr` (`real_schur.rs:837` — `if true ||` makes the
  recursive-multishift branch dead code; harmless at n≤512 where LAPACK
  would also use `dlahqr`); shift count / window are evaluated once on
  the full n (LAPACK's `iparmq` re-evaluates on the shrinking active
  block); the shift table caps at 32 for n<590 (LAPACK grows to
  ~n/log₂n ≈ 56 by n=512); `nibble=50` vs LAPACK's 14; Hessenberg blocks
  only from n≥256.

## THE ROOT CAUSE — a 1-line upstream bug, no_std only (patch 0004)

`faer/src/linalg/evd/schur/mod.rs:98`, the default AED deflation window
for 150 ≤ n < 590:

```rust
#[cfg(feature = "std")]      { (n as f64 / (n as f64).log2()) as usize } // n/log2(n) ≈ 56 at n=512
#[cfg(not(feature = "std"))] { libm::log2(n as f64 / (n as f64)) as usize } // log2(n/n) = 0 !!
```

The `no_std` branch computes **log₂(n/n) = 0** instead of n/log₂(n). All
typical wasm builds are `no_std` (ours: `default-features=false`), so on
wasm the AED window silently degenerates to 2 (the `max(nwr,2)` clamp)
for every 150 ≤ n < 590: AED can deflate at most ~2 eigenvalues per call
and supplies at most 2 shifts, so every "multishift" sweep runs as a
degenerate 2-shift bulge chase with full blocked-sweep overhead.

**Measured on the runner (run 29118452323, eigenvalues-only, min-of-N):**

| n | variant | ms | AED calls / sweeps |
| - | - | -: | -: |
| 128 | faer default | 93.3 | 52 / 39 |
| 128 | lahqr-pinned | **14.6** | — |
| 128 | iparmq-style fns | 92.5 | 52 / 39 |
| 256 | faer default | 173.3 | **540 / 420** |
| 256 | lahqr-pinned | 109.8 | — |
| 256 | iparmq-style fns | 146.9 | **25 / 17** |
| 512 | faer default | 1459.4 | **1091 / 852** |
| 512 | lahqr-pinned | 929.6 | — |
| 512 | iparmq-style fns | **597.5** | **17 / 10** |

The iparmq-style replacement functions compute the *same intended table*
without the bug — the ~50–85× iteration collapse (852 → 10 sweeps at
n=512) is the bug's signature, not a tuning effect. n=128 is unaffected
(the window table's n<150 branch has no log₂), which also explains why
the two variants tie there. Dev-box verification with patch 0004 applied:
faer's *unmodified defaults* now converge in 25/17 (n=256) and 26/22
(n=512) — the defaults were never the problem; the arithmetic was.

**This bug also explains (for n≥150) the 2026-07-09 finding** that faer's
blocked multishift/AED path lost to its own scalar `lahqr` by 2–13× on
wasm (recorded in `schur/src/real.rs` `recommended_params`, which pins
faer-schur to `lahqr`). That pin should be re-evaluated after 0004: at
n=512 the repaired multishift path (597 ms) beats lahqr-pinned (930 ms)
by 1.56×; at n≤256 lahqr still wins (110 vs 147 ms at 256, 15 vs 93 ms at
128) — the wasm crossover sits between 256 and 512, far above the
`nmin=75` default. Still to tune: `blocking_threshold` on wasm.

## Phase split (same run)

| n | Hessenberg | full eigvals | Hessenberg share |
| - | -: | -: | -: |
| 64 | 0.27 ms | 3.11 ms | 9% |
| 128 | 3.32 ms | 91.6 ms | 4% |
| 256 | 46.2 ms | 170.8 ms | 27% |
| 512 | 215.3 ms | 1509.6 ms | 14% |

(Shares computed against the *buggy* totals; against the repaired 597 ms
at n=512, Hessenberg is ~36% — it becomes a first-class target once the
QR-iteration side is fixed. The blocked-Hessenberg panel is GEMV-rich and
our measured GEMV runs at ~30% of bandwidth: the flat-simd128 panel +
block-apply rebuild, task #17's deferred half, addresses exactly this.)

## Scoreboard context

Run 8 (three-way, 2026-07-10): faer eigvals 0.3–0.4× scipy. The repaired
n=512 time (597 ms vs buggy 1459 ms, 2.44×) projects eigvals to roughly
parity with scipy before any wasm-shaping work (Hessenberg kernel,
blocking_threshold tuning, shift-table shaping) — to be confirmed by a
full three-way re-run on the runner with 0004 applied.

## Runner confirmation of 0004 (run 29119237626)

With the patch applied, on the reference machine, eigenvalues-only:

| n | default (buggy → fixed) | counters (buggy → fixed) | lahqr-pinned |
| - | - | - | - |
| 128 | 94.3 ms (unchanged) | 52/39 (unchanged) | **14.7 ms** (6.4×) |
| 256 | 173.3 → **149.7 ms** | 540/420 → **25/17** | **110.2 ms** (1.36×) |
| 512 | 1459.4 → **598.1 ms** (2.44×) | 1091/852 → **26/22** | 960.6 ms (0.62×) |

Post-fix, faer's defaults match the iparmq-style profile exactly (597 vs
598 ms at n=512) — parameter re-shaping is NOT needed; only the bug was.
The multishift-vs-lahqr **sign flips at large n**: multishift now wins at
n=512 (1.61×), lahqr still wins at n≤256 (6.4× at 128, 1.36× at 256). So
the wasm-right `blocking_threshold` sits between 256 and 512 (default 75
is far too low for wasm; n=128 is untouched by the bug and still 6.4×
better on lahqr — that residual gap is real wasm shaping, not the bug).
faer-schur's `recommended_params` lahqr pin (usize::MAX) must be
re-swept for the Schur-with-vectors case post-0004.

## Three-way with 0004 (pyodide run 29119238887)

| op | n | faer | scipy | ratio (pre-0004) |
| - | -: | -: | -: | - |
| eigvals | 128 | 131.9 ms | 11.2 ms | 0.1× (unchanged — bug starts at 150) |
| eigvals | 256 | 204.1 ms | 85.3 ms | 0.4× (0.3×) |
| eigvals | 512 | 835.9 ms | 592.5 ms | **0.7×** (0.3–0.4×) |
| schur | 512 | 2083.5 ms | 744.8 ms | 0.4× (lahqr-pinned, pays at large n) |

(The eigvals@512 wall time differs between runner instances — 598 ms on
the evd-tune machine vs 836 ms here; ratios within a run are the honest
comparison.) What the full grid adds beyond the bug:

1. **faer's multishift is ~8× slower than LAPACK's multishift at n=128**
   (94 vs 11 ms) where the window bug never applied and iteration counts
   are modest (52 AED/39 sweeps) — a genuine per-sweep implementation gap
   on wasm (suspects: `multishift_qr_sweep`'s small-gemm block updates
   and workspace copies on 2 lanes). Meanwhile faer's scalar lahqr
   (14.7 ms) is within 1.3× of scipy — so at n≤256 the cheap move is the
   threshold, and the expensive question is the sweep kernel.
2. **A wasm-tuned `blocking_threshold` ≈ 384** (lahqr below, multishift
   above) projects eigvals to ~0.7–0.8× scipy across all sizes with no
   further code: 14.7 ms @128 (0.76×), 110 @256 (0.77×), 598 @512
   (0.99× on matched hardware). Companion-level params (faer's public
   `evd_real` takes `Spec<EvdParams>`) — no additional patch needed.
3. **faer-schur's lahqr pin now costs 2.8× at n=512** for Schur-with-Z
   (2083 vs ~750 ms) — re-sweep the pin post-0004.

## Deep-research harness findings (wf_a92bf7f7-da9; 104 agents, 22 sources, 22/25 claims confirmed 3-vote)

The adversarially-verified harness ran in parallel with the empirical
track (it launched before the bug was found; its evidence base is
source-reading + primary literature, explicitly no runtime measurement).
Where its inferences meet our runner data, measurement wins. Verified
findings and their post-measurement status:

- **The opponent runs multishift+AED at every benchmark size** (3-0×2):
  dhseqr dispatches to dlaqr0 above NMIN=75 in OpenBLAS 0.3.28's bundled
  LAPACK, Fortran and f2c/C alike. faer competes against the modern
  algorithm, not dlahqr.
- **Missing algorithm RULED OUT** (3-0×3) and **unrequested work RULED
  OUT** (3-0×5): faer implements full dlaqr0-class multishift+AED, and
  its eigenvalues-only path does strictly the reduced work (want_t
  gating verified line-by-line against LAPACK's WANTT mechanism).
  Independently confirms our source reading.
- **Harness prime suspect #1 — parameter mistuning (nibble 50 vs 14,
  32 vs ~56 shifts, window ladder)** (3-0×5): mechanistically real but
  **REFUTED as the explanation by our post-0004 runner sweep** —
  nibble=14 measures slightly *worse* (0.92× at n=512), and the
  iparmq-style shift/window profile is indistinguishable from faer's
  fixed defaults (597.4 vs 598.1 ms). The actual culprit was the no_std
  `log2(n/n)`=0 window bug, which the harness's 3-vote source reading
  did not catch — its agents verified the *std* branch's intended
  semantics and never flagged the cfg-gated discrepancy. Instructive
  failure mode: iteration *counters* caught what source review missed.
- **Harness prime suspect #2 — Hessenberg shaping** (3-0): unblocked
  below n=256, gemv-bound, measured 3× bandwidth headroom, and the
  proven flat-simd128 panel + block-apply recipe applies. Matches our
  measured phase split (~36% of the repaired pipeline at n=512).
- **dlaqr5's sweep is level-3 via KACC22** (3-0×4): LAPACK's
  far-from-diagonal updates run as two DGEMMs per chase step over
  accumulated local reflections; small-bulge chains avoid shift
  blurring. This is the surviving frame for our residual per-sweep gap:
  faer's sweep also accumulates (wh/wv workspaces) but at n=128 its
  blocks are tiny and tiny-gemm is a known faer-on-wasm weak spot —
  measured 8× vs LAPACK's multishift at n=128 (94 vs 11 ms) at modest
  iteration counts. The one structural question the harness left open
  (its own flag): whether faer's gemm routing matches dlaqr5's
  effectiveness at NS=12–32 on 2 lanes.
- **No algorithm replacement exists** (3-0 + 2-1): Hessenberg +
  multishift QR + AED is settled serial-optimal at n≤512; even the 2022
  state of the art (Algorithm 1019/StarNEig) base-cases to *sequential
  dhseqr* below ~300. Same shape as the SVD verdict: implementation
  shaping wins, replacement loses.
- **Correctness guards (Q5)** (3-0): faer's convergence safeguards match
  LAPACK's exactly — exceptional shifts DAT1/DAT2 = 0.75/−0.4375, every
  6 stalled cycles in the multishift driver / 10 in lahqr, AED window
  adaptation after 5 stalls. No documented reference-LAPACK-on-wasm
  eigensolver failures, no FMA/denormal-specific hazards in any
  surviving claim (the correctness question is answered at
  guard-inventory level; no wasm-specific horror stories exist in the
  literature). Our exact-value smoke probes + faer-schur accuracy gates
  passing with 0004 are the local correctness evidence.

Caveats (harness's own): BBM paper bodies and StarNEig PDF were
proxy-blocked (LAPACK source substituted — canonical-equivalent); three
claims refuted 0-3 including an over-precise iparmq ladder and an IACC22
assertion (LAPACK ≥3.10 changed how IACC22 is honored inside dlaqr5 —
re-verify before leaning on it).

## Status / next

- [x] Patch 0004 minted, round-trip verified (`git apply` clean on
  pin+0001+0002), full gate green (smoke-test exact values, faer-schur
  6/6, kernels 5/5).
- [x] Runner re-run of evd-tune with 0004: confirmed (table above).
- [x] Deep-research harness landed; parameter-mistuning suspect refuted
  by measurement, Hessenberg + sweep-gemm-routing survive as the two
  levers; algorithm replacement ruled out; correctness guard inventory
  complete.
- [ ] Three-way pyodide re-run: where does repaired eigvals land vs scipy?
- [ ] Re-evaluate faer-schur's lahqr pin (crossover now between 256/512);
  sweep `blocking_threshold` on wasm.
- [ ] Deep-research harness findings to append here (running,
  wf_a92bf7f7-da9): dgeev phase splits, iparmq semantics, correctness
  guards for wasm, algorithm-replacement candidates.
- [ ] Hessenberg flat-panel kernel (task #17's deferred half) — now ~36%
  of the repaired pipeline.
