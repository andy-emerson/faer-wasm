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

## Status / next

- [x] Patch 0004 minted, round-trip verified (`git apply` clean on
  pin+0001+0002), full gate green (smoke-test exact values, faer-schur
  6/6, kernels 5/5).
- [ ] Runner re-run of evd-tune with 0004 to confirm default-params
  collapse on the reference machine.
- [ ] Three-way pyodide re-run: where does repaired eigvals land vs scipy?
- [ ] Re-evaluate faer-schur's lahqr pin (crossover now between 256/512);
  sweep `blocking_threshold` on wasm.
- [ ] Deep-research harness findings to append here (running,
  wf_a92bf7f7-da9): dgeev phase splits, iparmq semantics, correctness
  guards for wasm, algorithm-replacement candidates.
- [ ] Hessenberg flat-panel kernel (task #17's deferred half) — now ~36%
  of the repaired pipeline.
