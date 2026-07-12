# faer-wasm vs Pyodide (scipy/numpy on wasm) — 2026-07-09

> **Status note (2026-07-11):** this file is the chronological run log;
> the early "Pyodide wins most of this suite" headline describes Run 1
> and is long superseded. Current replication-graded state: matmul
> 4–20× wins (n ≥ 128), QR kernel 2.5–3×, LU-solve 1.4–1.7×, eigvals
> kernel pipeline wins at ALL five sizes n=64–1024 (1.51–2.08×; the
> old 512-parity verdict was leak-allocator tax, see below), an f32
> column at 2–9× (scipy's s-routines are no faster than its d-routines
> on wasm), SVD 0.7–1.5× post-fix, real Schur via the schur_k kernel
> pipeline winning 1.08–1.67× at n=64–512 with 0.99× at 1024 (was
> 0.2–0.6× everywhere), c64 Schur via the kernel twins winning
> 1.38×/1.34×/1.10×/1.03× at n=64/128/512/1024 with 0.90× at 256 (was
> 0.4–0.9×). ⚠ All rows BEFORE run 29157035070 (2026-07-11) carry
> leak-only-bump allocator tax on the faer side (cold pages +
> memory.grow per temp; scipy's own allocator was unaffected) — the
> LIFO-rewind allocator fix moved allocation-heavy faer rows 1.6–1.8×;
> treat 29157035070 as the reference run. See
> `research-eig-wasm-2026-07.md`, `research-svd-wasm-2026-07.md`, and
> `research-schur-wasm-2026-07.md` for the current tables and verdicts.

The first external head-to-head, per the architect's benchmark direction:
the same problems solved by the incumbent scientific-computing-on-wasm
stack, in the same V8, both sides single-threaded and paying the same
wasm tax. Run on a GitHub Actions runner (`pyodide-bench.yml`,
`bench/pyodide-vs-faer.mjs`): faer bench-harness wasm (opt-level 3,
carried patches, `faer-schur` for Schur) vs Pyodide 314.0.2 with
numpy 2.4.3 / scipy 1.18.0 under node v22.23.1. Adaptive min-of-3 per
measurement; n = 64 / 128 / 256.

## Headline — honest version

**Pyodide wins most of this suite.** Geomean speedup of faer-wasm over
Pyodide: **0.41×** — i.e. scipy/numpy's LAPACK stack on wasm is ~2.4×
faster overall at these sizes. [Correction from Run 4: the stack is
OpenBLAS 0.3.28 with generic C kernels, not f2c'd reference LAPACK —
the "simple loops compile to lean wasm" reading below survives, the
library attribution does not.] faer-wasm wins exactly where its
hand-written gemm microkernels dominate (matmul, 4–5× at n ≥ 128, real
and complex); it loses the factorizations and eigensolvers, typically
2–10×.

| op | n | faer-wasm | pyodide | speedup |
| - | -: | -: | -: | -: |
| matmul | 64 | 0.44 ms | 0.25 ms | 0.6× |
| lu_solve | 64 | 1.01 ms | 0.09 ms | 0.1× |
| qr_r | 64 | 3.64 ms | 0.21 ms | 0.1× |
| svd | 64 | 11.54 ms | 1.65 ms | 0.1× |
| eigvals | 64 | 3.12 ms | 1.50 ms | 0.5× |
| schur | 64 | 10.44 ms | 1.68 ms | 0.2× |
| matmul_c64 | 64 | 0.84 ms | 0.41 ms | 0.5× |
| lu_solve_c64 | 64 | 1.77 ms | 0.19 ms | 0.1× |
| qr_r_c64 | 64 | 3.95 ms | 0.81 ms | 0.2× |
| schur_c64 | 64 | 13.09 ms | 5.42 ms | 0.4× |
| matmul | 128 | 0.82 ms | 3.24 ms | **4.0×** |
| lu_solve | 128 | 3.21 ms | 0.56 ms | 0.2× |
| qr_r | 128 | 7.77 ms | 1.31 ms | 0.2× |
| svd | 128 | 20.71 ms | 9.13 ms | 0.4× |
| eigvals | 128 | 116.80 ms | 11.44 ms | 0.1× |
| schur | 128 | 24.91 ms | 14.18 ms | 0.6× |
| matmul_c64 | 128 | 2.47 ms | 3.45 ms | **1.4×** |
| lu_solve_c64 | 128 | 3.82 ms | 1.29 ms | 0.3× |
| qr_r_c64 | 128 | 12.06 ms | 5.85 ms | 0.5× |
| schur_c64 | 128 | 37.74 ms | 36.79 ms | 1.0× |
| matmul | 256 | 5.98 ms | 30.86 ms | **5.2×** |
| lu_solve | 256 | 13.51 ms | 4.11 ms | 0.3× |
| qr_r | 256 | 42.68 ms | 10.06 ms | 0.2× |
| svd | 256 | 127.39 ms | 68.31 ms | 0.5× |
| eigvals | 256 | 227.77 ms | 79.40 ms | 0.3× |
| schur | 256 | 209.27 ms | 88.12 ms | 0.4× |
| matmul_c64 | 256 | 18.98 ms | 84.53 ms | **4.5×** |
| lu_solve_c64 | 256 | 21.10 ms | 9.61 ms | 0.5× |
| qr_r_c64 | 256 | 66.94 ms | 49.80 ms | 0.7× |
| schur_c64 | 256 | 376.70 ms | 278.01 ms | 0.7× |

## Reading it

- **The pattern is one story told twice.** faer's kernels are shaped for
  native machines: wide SIMD, deep blocking, cache-oblivious recursion.
  On wasm those shapes become overhead — this is the same disease as
  every blocking cliff this repo has measured, now visible externally.
  Their stack is simple loops that compile to lean wasm and, at
  n ≤ 256, win. [Run 4 identified it: OpenBLAS generic-C kernels, not
  f2c'd Fortran — same shape argument, corrected attribution.]
- **matmul is the counterexample that proves it**: the gemm crate's
  microkernels are the one place faer's approach carries to wasm, and
  there faer beats numpy's wasm matmul 4–5×. That gap is also the
  long-game lever — LAPACK's own blocked factorizations bottom out in
  dgemm, ours in the fast gemm, so the crossover should flip toward
  faer as n grows past this table. Unmeasured; next question.
- **The §7 tuned kernels close some gap but not all of it**: tuned LU at
  n=128 is ~0.24× of the default (≈ 0.77 ms) vs Pyodide's 0.56 ms; QR
  panel-1 ≈ 1.9 ms vs 1.31 ms. The comparison above uses faer's
  *default* high-level calls, which is what a naive consumer gets.
- **eigvals @128 (0.1×) is faer's `.eigenvalues()` blocking cliff**
  (`benchmarks-2026-07.md`) — no public params to fix it; `faer-schur`'s
  tuned Schur is the workaround and scores 0.6× rather than 0.1×.

## Caveats

- Each side does its idiomatic call for the same problem (e.g. QR is
  R-only on both sides via `mode='r'`; SVD computes full U/V on both).
- One run, one shared runner, min-of-3 — treat entries within ~±30% as
  ties; the 3×+ gaps are real.
- Pyodide numbers include no marshalling: timing loops run entirely in
  Python inside the wasm VM, as ours run entirely in wasm.
- n ≤ 256 only (leak-capped harness). The matmul advantage says the
  large-n story likely reverses; not yet evidence.

## Follow-ups recorded in ROADMAP

1. Extend to n = 512/1024 for the crossover (factorizations ride gemm).
2. Re-run with §7-tuned faer calls to quantify the honest best-case.
3. The strategic option nobody has taken: faer's algorithms with
   wasm-shaped parameters everywhere (what `faer-schur` did for Schur).

## Regenerate

Actions → "pyodide bench" → Run workflow (manual; the dev container
blocks the Pyodide CDN, runners don't). Results: job log + artifact.

## Run 2 — crossover sweep to n=512 + tuned rows (2026-07-09, run 28994239085)

Same protocol, sizes extended to 512, plus tuned-parameter faer rows
(docs/wasm.md §7 params; factor-only LU vs `scipy.linalg.lu_factor`,
R-only QR both sides). Highlights (full table in the workflow artifact):

| op | n=64 | n=128 | n=256 | n=512 |
| - | -: | -: | -: | -: |
| matmul | 0.7× | 4.1× | 6.1× | **21.0×** |
| matmul_c64 | 0.5× | 1.4× | 4.0× | 4.9× |
| **qr_r_tuned** | **1.3×** | **1.5×** | **1.7×** | **1.7×** |
| lu_factor_tuned | 1.3× | 0.8× | 0.7× | 0.6× |
| qr_r (default) | 0.1× | 0.1× | 0.2× | 0.4× |
| lu_factor (default) | 0.1× | 0.1× | 0.2× | 0.2× |
| svd | 0.2× | 0.4× | 0.5× | 0.8× |
| qr_r_c64 | 0.2× | 0.4× | 0.8× | 1.1× |
| eigvals | 0.5× | 0.1× | 0.4× | 0.3× |
| schur | 0.2× | 0.6× | 0.4× | 0.3× |
| schur_c64 | 0.4× | 1.0× | 0.7× | 0.7× |

(speedup = pyodide/faer; > 1 means faer-wasm wins. Geomean over all
rows incl. defaults: 0.52×.)

What this changes:

- **QR is already won — by parameters alone.** Panel-width-1 QR beats
  scipy at every size measured, 1.3–1.7×. The "QR panel kernel rewrite"
  collapses to a packaging problem: consumers must get these params by
  default (the faer-schur `recommended_params` treatment), not to a
  kernel-implementation problem.
- **LU's residual gap is now precisely bounded**: tuned LU wins at n=64
  and sits at 0.6–0.8× for 128–512 — the panel-kernel rewrite is chasing
  a 1.25–1.7× residual, not the 5–10× default-path gap.
- **Crossovers exist but are matmul-led**: matmul 21× at n=512 (numpy's
  wasm matmul collapses at cache-unfriendly sizes), complex QR crosses
  to 1.1×, SVD approaches parity (0.8×). Per the architect's caveat,
  large-n-only wins are scoping data, not victory — recorded as such.
- **Eigen-pipelines stay the weak flank at every size** (0.1–0.6×):
  unchanged conclusion — the lahqr-class kernel work is where their gap
  lives, queued after QR/LU per the architect's ordering.

## Run 3 — first wasm-shaped kernel enters the ring (2026-07-09, run 28995103525)

`lu_factor_wk` is `faer-wasm-kernels`' blocked LU: lean simd128 panel +
faer-gemm trailing updates (kernels/src/lu.rs). Same box, four LU
implementations against scipy's `lu_factor`:

| n | wk (shaped) | faer tuned | faer default | scipy | wk vs scipy |
| -: | -: | -: | -: | -: | -: |
| 64 | **0.06 ms** | 0.07 ms | 0.87 ms | 0.08 ms | **1.5×** |
| 128 | **0.35 ms** | 0.45 ms | 3.22 ms | 0.38 ms | **1.1×** |
| 256 | 3.27 ms | 3.26 ms | 14.48 ms | 2.43 ms | 0.7× |
| 512 | **27.69 ms** | 30.50 ms | 68.42 ms | 18.37 ms | 0.7× |

Approach-validation read (new-faer vs old-faer vs incumbent):

- The shaped kernel is the **fastest LU on wasm among all faer paths at
  every size** — 9–14× over faer's defaults at 64–128, 2.5–4.4× at
  256–512, and it beats the best tuning by 10–30% except a tie at 256.
- It **beats scipy at n=64 and n=128** — the first faer LU ever to do so
  at any size. The shaping approach is validated in the regime where
  parameters alone had plateaued at 0.8–1.3×.
- The n ≥ 256 residual is real on equal hardware: 1.35× at 256, 1.5× at
  512. Flop accounting places it in the O(n²·nb) parts (panel + trsm at
  ~3 GF vs the gemm bulk's ~4.7): closing it needs higher-intensity
  panel updates (rank-2+), not more parameter search. Bounded,
  understood, and diminishing — the eigen flank's 2.5–3× gaps are the
  bigger target.

## Run 4 — recursive LU + the BLAS reveal (2026-07-09, run 29023029694)

`lu_factor_rec` is the recursive (`dgetrf2`/Toledo-shape) LU from
`docs/research-lu-wasm-2026-07.md` plan item 1: the panel's memory-bound
rank-1 work recast as trsm + gemm at growing ranks, flat simd128 loops
only below the crossover. Same box, five LU implementations vs scipy:

| n | rec | wk (blocked) | faer tuned | faer default | scipy | rec vs scipy |
| -: | -: | -: | -: | -: | -: | -: |
| 64 | **0.06 ms** | 0.06 ms | 0.08 ms | 1.48 ms | 0.09 ms | **1.5×** |
| 128 | **0.35 ms** | 0.36 ms | 0.45 ms | 3.17 ms | 0.38 ms | **1.1×** |
| 256 | **2.78 ms** | 3.26 ms | 3.29 ms | 13.88 ms | 2.44 ms | 0.9× |
| 512 | **22.40 ms** | 27.39 ms | 29.45 ms | 83.49 ms | 18.45 ms | 0.8× |

- **The recursion delivered what the research projected** (~20–22 ms at
  n=512; measured 22.40): 15% over the blocked wk driver at 256, 18% at
  512, identical below 256 (both go flat-loop-only there). scipy's lead
  shrank from 1.35×/1.5× to 1.14×/1.21×.
- **But not the way the theory said**: narrow crossovers (8–32) *lose* —
  each skinny gemm's call overhead exceeds the flat-loop cost it
  replaces. The winning shape is coarse: crossover 128, trsm base 64.
  Measurement beat theory on the constants; the structure survived.
- **The incumbent is now identified, not guessed** (this run prints
  `numpy/scipy.show_config()`): Pyodide's scipy links **OpenBLAS 0.3.28
  built with the generic C `RISCV64_GENERIC` target** (config string:
  `... NO_AFFINITY=1 USE_OPENMP= RISCV64_GENERIC MAX_THREADS=4`), via
  Emscripten. No architecture microkernels. Run 1's "f2c'd reference
  LAPACK" read was wrong about the library and right about the shape:
  their hot loops are autovectorized generic C, which explains both why
  simple faer paths lost to it and why our gemm beats it 4–20×.
- Suite geomean: **0.57×** (Run 1: 0.41×) — the LU/QR wins and the
  matmul crossover pull it up; the eigen flank (0.1–0.6×) is what holds
  it down and is the queued target.

## Run 5 — recursive LU tuned on the runner (2026-07-09, run 29045340390)

The recursive LU's crossover/trsm_base were dev-box-swept in Run 4; the
`lu-tune.yml` sweep re-tuned them **on the runner** (run 29044864243) and
the corrected defaults (crossover 256, trsm_base 128 — *less* recursion,
wider flat base cases than the dev box picked) were baked and re-raced:

| n | rec (tuned) | wk (blocked) | faer tuned | scipy | rec vs scipy | Δ vs Run 4 |
| -: | -: | -: | -: | -: | -: | -: |
| 64 | **0.06 ms** | 0.06 | 0.07 | 0.08 | **1.5×** | = |
| 128 | **0.34 ms** | 0.35 | 0.45 | 0.38 | **1.1×** | = |
| 256 | **2.49 ms** | 3.26 | 3.34 | 2.43 | **1.0×** | 2.78→2.49, **scipy parity** |
| 512 | 22.33 ms | 27.28 | 29.37 | 18.39 | 0.8× | ≈ (22.40→22.33) |

- **The tuning delivered at n=256**: recursive dropped 2.78→2.49 ms and
  reached **scipy parity** (1.0×, was 0.9×), now 24% ahead of the blocked
  driver (the co=256 default makes n=256 pure-flat, no recursion — which
  the runner sweep showed wins there).
- **n=512 barely moved** (22.40→22.33, still 0.8× scipy). The runner
  sweep predicted only ~5% at 512 and per-instance variance covers it.
  Honest read: the LU-tuning win is real through n=256, marginal at 512.
- **The dev box had mis-tuned toward more recursion** than the runner
  wants — the reason the sweep moved to CI (`lu-tune.yml`) and the reason
  the defaults are now gated (`gate.mjs`: recursive ≤ blocked ≤ faer-tuned
  at n=256) so a re-pin or bad tune can't silently undo it.
- Suite geomean **0.58×** (Run 4: 0.57×). Unchanged conclusion overall;
  the eigen flank (eigvals/schur 0.3–0.4×) remains the dominant gap.

## Run 6 — wasm-shaped QR kernel + LU re-tune (2026-07-09, run 29048492853)

Two things: the unblocked Householder QR kernel (`kernels/src/qr.rs`,
`qr_r_wk`) enters the ring, and the recursive-LU knobs were re-swept over
3 rounds (`lu-tune.yml` run 29048491497) because the first single-round
sweep wasn't trustworthy. **This was a faster runner instance than Runs
4–5** (scipy LU@512 17.8 ms vs 18.4; matmul@512 only 9.7× because numpy's
matmul didn't collapse as hard) — so read the *ratios*, which are same-run
valid, not the absolutes against earlier runs.

**QR — the kernel is a decisive win at every size:**

| n | qr_r_wk | faer bs=1 | faer default | scipy | wk vs scipy | wk vs faer-bs1 |
| -: | -: | -: | -: | -: | -: | -: |
| 64 | **0.08 ms** | 0.28 | 4.59 | 0.20 | **2.5×** | 3.5× |
| 128 | **0.52 ms** | 1.40 | 10.76 | 1.35 | **2.6×** | 2.7× |
| 256 | **4.11 ms** | 8.28 | 55.04 | 11.02 | **2.7×** | 2.0× |
| 512 | **28.74 ms** | 58.02 | 246.34 | 86.78 | **3.0×** | 2.0× |

- Beats scipy **2.5–3.0× at every size**, including n=512 where the
  factorizations usually lose — the first faer factorization to beat scipy
  decisively across the whole range. Confirms the research call
  empirically: the fused unblocked dot+axpy panel is 2–3.5× leaner than
  faer's own `block_size=1` machinery (compact-WY setup even at width 1),
  and there is no compact-WY/T-matrix/gemm in it at all.
- The projection ("large QR lead") held and then some. Gated at
  `wk ≤ 0.8×faer-tuned` so it can't silently regress.

## Run 8 — wasm-shaped LU solve wired in (2026-07-10, run 29061871875)

The full Ax=b solve (`run_lu_solve_wk` = flat factor + our forward/back
substitution, `lu_solve_in_place`) against `np.linalg.solve` — the solve
kernel already existed and was correctness-gated; this only wired it into
the bench (the prior `lu_solve` row used faer's default path, hence 0.1–0.5×).

| n | lu_solve_wk | scipy (np.linalg.solve) | speedup |
| -: | -: | -: | -: |
| 64 | **0.06 ms** | 0.09 | **1.6×** |
| 128 | **0.34 ms** | 0.56 | **1.6×** |
| 256 | **2.44 ms** | 4.07 | **1.7×** |
| 512 | **22.79 ms** | 32.69 | **1.4×** |

- **A clean win at every size, 1.4–1.7×** — up from the 0.1–0.5× the
  default-path `lu_solve` row still shows. No new kernel: LU-solve's only
  dependency was LU-factor, which we already had. Solve = flat factor
  (fastest wasm LU) + O(n²) substitution (cheap), so it inherits the
  factor's standing and adds negligible cost.
- It edges past our LU-*factor*-vs-scipy ratio because it's built on the
  flat factor (the default now) and `np.linalg.solve` is heavier than
  scipy's `lu_factor`; still an honest factor+solve vs factor+solve match.
- Suite geomean **0.67×** (Run 6: 0.58×), pulled up by the qr_wk and
  lu_solve_wk wins. The eigen flank (SVD/eigvals/Schur, 0.2–0.4×) is now
  unambiguously the sole remaining loss.

## Run 7 — recursion verdict: removed from the LU default (2026-07-09, run 29051443596)

Follow-up to the architect's doubt ("recursion is active exactly where we
lose to scipy — does it belong at all?"). A dedicated large-n probe
(`lu-largen.mjs`, to n=1024) settled it on the runner:

| n | pure-flat | best recursion | verdict |
| -: | -: | -: | -: |
| 512 | 23.36 ms | 23.29 ms | tie |
| 640 | 33.35 ms | 36.18 ms | **flat wins 8.5%** |
| 768 | 58.11 ms | 61.90 ms | **flat wins 6.5%** |
| 1024 | 155.78 ms | 168.61 ms | **flat wins 8.2%** |

**Pure-flat wins at every size 64–1024 on the runner; recursion never
wins, and the gap does not close toward 1024.** (A dev box had shown
recursion winning 5–10% at large n — a machine-specific cache quirk that
did *not* reproduce on the reference runner. Exactly why tuning lives on
the runner, not locally.) So `RECOMMENDED_CROSSOVER` was set to
`usize::MAX`: **the default LU never recurses** — it is the pure-flat
unblocked simd128 panel, the fastest wasm LU measured at every size. The
recursion machinery stays in the tree for explicit-`crossover` opt-in on a
differently-balanced machine, but is off by default. Net: the LU story is
now simply "a lean flat simd128 panel, at scipy parity around n=256" — no
recursion claim to defend.

## LU re-tune history — the recursion never earned its keep

**LU re-tune — the recursion barely earns its keep (honest):** the 3-round
sweep (top-3 per size printed) shows pure-flat winning outright at
n=192/256/384 and tying within 0.1% at n=512 — the flat simd128 panel is
the engine, gemm-fed recursion adds almost nothing through n=512. The
first sweep's `co=256` was reproduced as good-but-not-optimal (7.6% behind
pure-flat at n=384; a coin-flip at 512). Re-baked `crossover=384`,
`trsm_base=256` — optimal-or-tied at every swept size, recursion retained
only for the n>512 regime the sweep can't reach. On this instance
`lu_factor_rec` read 1.0–1.2× scipy across sizes, but that's the fast
instance talking; the durable claim stays "parity-ish at 256, ~0.8× at
512" from the controlled Run 5.
