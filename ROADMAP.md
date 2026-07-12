# faer-on-wasm roadmap

A phased plan for the carried faer, scoped **language-agnostically**: every item
is something any wasm-targeting (or LAPACK-replacing) consumer of faer would
plausibly want — nothing Julia- or Ruju-specific lives here. The consumer-side
integration (the LBT/BLAS-ABI shim that presents `dgemm_64_`-style symbols)
belongs to the consumer's repo, not this one; see the boundary note at the end.

Grounded in empirical verification (2026-07, faer 0.24.4 @0539947 /
pulp 0.22.3): faer fails to build on `wasm32-unknown-unknown` only because of
three `(n >> 32)`-on-32-bit-`usize` sites; after a 4-line fix it builds with
`default-features = false, features = ["linalg"]`, runs matmul / LU / QR /
SVD / self-adjoint EVD / general complex EVD in node **bit-identical to native
x86-64**, at 51 KiB (matmul-only) to ~396 KiB (full suite) pre-wasm-opt.
pulp already ships a complete wasm backend (`Simd128` + `RelaxedSimd` with
`f64x2` FMA); the `gemm` crate ships wasm microkernels. Disassembly confirms
real `f64x2.mul` / `relaxed_madd` in the output. The `rayon` feature does not
build on wasm (`atomic-wait` has no port); `Par::Seq` is first-class.

**Prime directive: keep the carry thin.** Upstreaming is de-prioritized,
not forbidden (Andy, 2026-07-11, revising the 2026-07-08 "nothing goes
upstream"): nothing is submitted now, but upstream-worthy findings are
tracked in the **upstream ledger** below and the question reopens when
the project settles toward completeness. We vendor the minimum patch set
against a pinned upstream base, re-verify on every faer release, and
drop patches the moment a release doesn't need them. New capability is
built *alongside* faer (companion crates / consumer shim over public
APIs), not inside it.

## Phase 0 — Carry the enabler ✅ (maintenance mode)

- [x] The **4-line 32-bit fix**: `(n >> 32)` → `((n as u64) >> 32) as u32` in
      `operator/{eigen, self_adjoint_eigen, svd}`. Carried in
      `patches/faer-rs/0001-*.patch`; zero-`v0` regression tests exist in the
      archived `upstream/0001-*.patch` (they double as our own tests).
- [x] Verification gate green (2026-07-07): wasm build + node smoke test
      reproduce the reference values exactly, bit-identical to native.
- [x] **CI in this repo** (GitHub Actions): clone → pin → patch → build
      `wasm32-unknown-unknown` → node smoke test with exact-value checks
      (`.github/workflows/wasm-gate.yml` + `smoke-test/check.mjs`), so a
      faer bump or patch drift can't silently break us. (Replaces the
      shelved upstream CI job.)
- Recurring: **evaluate** each faer release — adopt (re-pin, re-apply,
  re-run the gate) only when it advances us; slight accommodations to
  upstream changes are fine; if upstream deviates from our needs, stay on
  the pinned base. If an adopted release builds on 32-bit without our
  patch, delete the patch and note it here.

## Phase 1 — Wasm consumer ergonomics ✅ (2026-07-08)

- [x] A documented **wasm recipe** (`docs/wasm.md`): feature set, build
      profile, per-decomposition sizes, the `no_std` zero-import path, and
      the determinism note (bit-identical to native).
- [x] **Relaxed-SIMD route documented** (`docs/wasm.md` §4) and built in
      CI on every push: pulp `relaxed-simd` + `-C target-feature=
      +simd128,+relaxed-simd`; probe values bit-identical to the plain
      build.
- [x] **Size regression tracking** in CI: all four variants built per
      push, checked against `smoke-test/size-budgets.json` (~10% over
      2026-07-08 measured sizes: 59,207 / 123,751 / 447,270 / 440,441 B).

## Phase 2 — Coverage growth (the LAPACK-parity tail) — underway

Prioritized by what a LAPACK-replacing consumer hits first; each item is
built **alongside faer** (a companion crate or the consumer's shim, over
faer's public low-level modules) with accuracy tests. Status:

- [x] **Schur decomposition** (`gees`-equivalent, f64 + c64) +
      **eigenvalue reordering** (`trexc`/`trsen`-shaped) — landed
      2026-07-08 as the `schur/` companion crate (`faer-schur`), driving
      faer's own internal kernels through
      `patches/faer-rs/0002-expose-schur-kernels.patch` (6 visibility-only
      lines; engineer's call, architect can veto: the alternative was
      porting ~500 lines of swap/QR-iteration internals into the
      companion crate). Accuracy CI-tested (backward error ~1e-15,
      orthogonality, eigenvalue agreement with faer's EVD, reorder
      invariants, n ≤ 150 covering the blocked/AED path); wasm-gated via
      integer property probes in the full variants. Findings along the
      way: (a) faer's blocked multishift leaves workspace junk below the
      subdiagonal — our driver zeroes it; (b) Schur raw doubles are NOT
      bit-identical across targets at 8×8 (docs/wasm.md §5) — the
      determinism claim stays scoped to the fixed probes; (c) the
      c64×relaxed-simd bug below.
- [x] **c64 × relaxed-SIMD miscomputation: root-caused and fixed**
      (found 2026-07-08 by the Schur gate, dug out same day on Andy's
      call). Cause: pulp's wasm `RelaxedSimd` backend ported its complex
      `mul_add_e`/`mul_e` kernels from the aarch64 NEON backend but kept
      NEON's accumulator-first FMA argument order when calling the
      accumulator-last `relaxed_madd` — four transposed-argument
      functions (`mul_add_e_c32s/c64s`, `mul_e_c32s/c64s`), each
      computing `(c·b)·a + x` instead of `c + b·x + a·y`-style complex
      FMA. Everything c64 past matmul was garbage under `+relaxed-simd`
      (faer's own `.eigenvalues()` included); f64 untouched (its `_e`
      ops are single-instruction, no porting involved). Fix carried as
      `patches/pulp/0003` (4 lines); `schur_probe_cplx == 3` on the
      full-relaxed variant is the CI regression guard. Drop the patch
      when a pulp release fixes it upstream.
- [ ] **Sylvester solver** (`trsyl`) — pairs with Schur; next in line.
- [ ] **LQ adapter** (thin; QR-of-transpose plumbing).
- [ ] `geevx`-style balancing / condition-number extras.
- [ ] Generalized SVD (`ggsvd3`) — lowest priority; rare call sites.

Already covered (no work): LU (partial+full pivot), LLT/pivoted-LLT/LDLT/
Bunch-Kaufman, QR ± column pivoting, SVD, self-adjoint + general EVD,
generalized EVD (QZ), triangular solve/inverse, full complex support.

## Foundation gate — correctness × efficiency for the simple ops ✅ (2026-07-09)

Architect-directed (2026-07-09): get the simple stuff verifiably right —
correct AND efficient — before building more complex layers on it.

- [x] **Dense correctness probes** (`smoke-test/src/dense_probes.rs`):
      LU / QR / LLT / SVD / self-adjoint EVD / general eigenvalues,
      **f64 and c64**, at n=33 (SIMD tail lanes) and n=96 (blocked
      paths) — the regime where the pulp c64 bug and faer's workspace
      junk lived, invisible to the 3×3 probes. Property-scored
      (residuals/orthogonality/structure vs tolerances ≥2 orders above
      measured errors — see the `margins` bin); scores 26/24 identical
      on native, node, and Chrome, all full variants, every push.
- [x] **Efficiency gate** (`bench/gate.mjs`, in CI): op/matmul ratio
      bands (×3) vs recorded `expected-ratios.json`, O(n³) scaling
      windows, and tuned-vs-default guards that fail if the
      docs/wasm.md §7 guidance ever stops winning. Bands sized to the
      cliff-class regressions actually observed (3–10×), robust to
      shared-runner noise. Bench harness gained c64 ops (c64 matmul ≈
      3.1× f64 at n=128; c64 LU ≈ 4.1×; c64 QR ≈ 12.9× — recorded).
- [x] **Browser gate** (`smoke-test/browser-check.mjs`, in CI): all
      probes exact in headless Chrome via raw CDP (no npm deps),
      including the relaxed-SIMD build — closes Phase 3's browser-run
      refinement; "runs in real browsers" is now CI-enforced.
- [x] **Complexity verification** (2026-07-09, second architect pass on
      "efficiency"): `bench/complexity.mjs` fits the empirical exponent
      p per op (log-log LSQ, n ≥ 96) and detects step jumps
      (consecutive ratio ≤ 4×(n₂/n₁)³) — the jump detector found the
      **Schur/EVD blocking cliff** on its first run: faer's blocked
      multishift loses to unblocked `lahqr` by 2–13× through n=384 on
      wasm, and the default threshold (75) hits 13× at n=96. Fixed in
      `faer-schur::recommended_params()` (wasm-tuned defaults, native
      unchanged); faer's own `.eigenvalues()` keeps the cliff
      (no public params) — documented in docs/wasm.md §7. Post-fix
      exponents all in [1.8, 3.2]; `--gate` mode runs per push.
- [x] **External baseline: Pyodide head-to-head** (architect direction
      2026-07-09; supersedes the parked OpenBLAS/pure-JS question
      below). First results (`docs/benchmarks-vs-pyodide-2026-07.md`,
      re-runnable via the manual `pyodide-bench` workflow): **honest
      and unflattering** — geomean 0.41× at n=64–256; faer-wasm wins
      matmul 4–5× (gemm microkernels) and loses the factorizations/
      eigensolvers 2–10× to scipy's LAPACK stack (identified in Run 4
      as OpenBLAS-generic-C, not f2c'd reference LAPACK), which
      compiles to leaner wasm than faer's native-shaped kernels. Open
      follow-ups: (a) n=512/1024 crossover — LAPACK's blocked paths
      bottom out in slow dgemm, ours in fast gemm, so large-n likely
      flips; (b) re-run with §7-tuned calls; (c) the strategic option:
      wasm-shaped parameters across all of faer's factorizations, the
      way faer-schur did for Schur. [(a) and (b) done below — Run 2;
      (c) superseded by the kernels crate.]

## Wasm-shaped kernels (`kernels/` = `faer-wasm-kernels`) — underway (2026-07-09)

Architect direction: not tuning — *implementation shaping*. Kernels
written in the code shape wasm engines compile well (flat loops, explicit
simd128), with the O(n³) bulk routed through faer's gemm microkernels
(the one faer component that already beats Pyodide everywhere). LU first,
then QR, then the eigen flank; correctness gated in CI (`kernels/tests`)
alongside the wasm gate.

**Tuning freeze (architect decision, 2026-07-11).** Shape before tune:
parameters are only optimal relative to an implementation, so no more
tuning *campaigns* (sweeps as deliverables) until the full surface
exists — f32/c32 layer, shaped Schur (+Z kernels), the SVD small-n
rotation path, both eigenvector functions. Existing thresholds (the 480
eig crossover, LU trsm base, etc.) remain as *provisional routing
scaffolding* so builds stay benchmarkable; the known re-tune debts (the
hqr-vs-multishift crossover at n=512 was measured against the pre-kernel
lahqr; f32's halved cache footprint moves every threshold) are deferred
to one global, replication-graded tuning pass at the end.

- [x] **Pyodide Run 2 — crossover sweep to n=512 + tuned rows**: tuned
      QR already beats scipy 1.3–1.7× at *every* size (a packaging
      problem now, not a kernel problem); tuned LU bounded at 0.6–1.3×;
      matmul 21× at n=512. Large-n-only wins are scoping data, not
      victory (architect's caveat).
- [x] **Blocked LU with lean simd128 panel** (`kernels/src/lu.rs`,
      Run 3): fastest faer LU at every size; **first faer LU to beat
      scipy** (1.5× at n=64, 1.1× at 128); 0.7× at 256–512.
- [x] **Deep research on LU optimization**
      (`docs/research-lu-wasm-2026-07.md`): claims graded
      confirmed/refuted/unverified; recursive (Toledo/dgetrf2) panel
      confirmed as the canonical fix for our measured panel wall.
- [x] **Recursive LU** (`lu_factor_recursive_in_place`, Run 4): the
      research's narrow-crossover theory *refuted by measurement*
      (skinny gemms lose to flat loops; crossover 128 + TRSM base 64
      won the sweep). On the runner: 2.78 ms at n=256 / 22.40 ms at
      n=512 — 15–18% over the blocked wk driver, moving scipy's lead
      to 0.9×/0.8× (was 0.7×/0.7×). Suite geomean now 0.57×
      (was 0.41× at Run 1). Projection from the research (20–22 ms)
      essentially held. Identical-pivot cross-check vs the blocked
      driver is CI-gated.
- [x] **Pyodide BLAS identity settled** (Run 4 prints
      `scipy.show_config()`): OpenBLAS 0.3.28 built with the generic C
      `RISCV64_GENERIC` target — no arch microkernels, no threads.
      Their factorizations ride autovectorized generic C; ours ride
      hand-written simd128 gemm. Routing more flops into gemm is the
      whole game, confirmed.
- [x] **Deep research on QR optimization**
      (`docs/research-qr-wasm-2026-07.md`): search/fetch completed, but
      the 3-vote verify died on a Fable-5 usage-credit exhaustion —
      claims verified by hand on Opus 4.8 against LAPACK/OpenBLAS
      primary source (graded proven/by-hand for structural facts,
      observed/sourced-unverified for single-paper perf numbers). Two
      load-bearing results: (i) Pyodide's QR is *doubly* un-optimized —
      OpenBLAS ships **no** QR routines, so scipy runs reference-netlib
      `dgeqrf` over generic-C BLAS, which is *why* `qr_r_tuned` already
      wins 1.3–1.7× structurally; (ii) **recursion is the wrong move for
      QR** (opposite of LU) — ReLAPACK excludes it, `dgeqrt3` recurses to
      skinny gemms, both concede blocking loses at small n. The durable
      lever is the block-apply `(I−V T Vᵀ)·C` kernel, whose real payoff
      is Hessenberg (the eigen flank), not standalone QR.
- [x] **Wasm-shaped unblocked QR kernel** (`kernels/src/qr.rs`, Run 6):
      the research's positive answer, built and measured. Fully unblocked
      dgeqr2-shape — fused `dlarfg` + `dlarf` (dot + axpy) in flat
      simd128, no compact-WY/T-matrix/gemm. **Beats scipy 2.5–3.0× at
      every size n=64–512** and faer's own `block_size=1` path 2–3.5× —
      the first faer factorization to beat scipy decisively across the
      whole range. Correctness-gated (‖A−QR‖, ‖QᵀQ−I‖, |R| vs faer);
      efficiency-gated (`wk ≤ 0.8×faer-tuned`). This *is* the "package
      width-1 as default" win and then some — a naive consumer would get
      `qr_r_wk`'s speed, not faer's native default.
- [x] **Hessenberg front-end — resolved differently than predicted**
      (2026-07-11): the planned block-apply `(I−V T Vᵀ)·C` kernel was
      never needed; the flat *unblocked* Hessenberg kernel
      (`kernels/src/hessenberg.rs`, gaxpy right-apply + dot/axpy
      left-apply) beat faer's blocked reduction 3.2×/7.0× at n=512/1024
      and sidesteps faer's blocked-hess machine cliff (7–95× across
      runners at 1024). The LU lesson held a third time: flat loops over
      block structure on 2-lane wasm.
- [x] **Focused research: SVD + eig wasm-shaping**
      (`docs/research-eig-svd-wasm-2026-07.md`, "focused" tier — one
      agent, ~2 min). Key finding: the eigen flank is dominated by the
      Householder **reduction** (`dgebrd`/`dsytrd`/`dgehrd`), all the same
      memory-bound-panel disease our flat kernels win; ~50% level-2 for
      SVD/sym-eig (flat simd128 GEMV/SYMV is the lever), ~80% level-3 for
      gen-eig (mostly routing into faer gemm). OpenBLAS optimizes *none*
      of it → same weak opponent as QR → win-big potential. Back-ends
      (`bdsqr`/`stedc`/`hseqr`) left alone (`laed3` already gemm-shaped;
      `lahqr` fix shipped).
- [x] **Pivotal pre-build measurement — answered for 2 of 3**:
      `gehrd` → flat-panel rebuild won (kernel above); `gebrd` (SVD) →
      measured as only ~30% of SVD wall time with a ~10–15% realizable
      sliver, deliberately unpursued (`research-svd-wasm-2026-07.md`);
      `sytrd` (symmetric eig) → still unprofiled, queued in the
      next-sessions plan.
- [x] **General-eig flank COMPLETE** (2026-07-10/11, full record in
      `research-eig-wasm-2026-07.md`): upstream bug 0004 found via
      iteration counters and carried as a 1-line patch (~50–85× iteration
      collapse); per-n lahqr/multishift routing at the measured 480
      crossover; flat Hessenberg kernel; hand double-shift `hqr` kernel
      (dlahqr-shape, eigenvalues-only). Replication-gated verdicts vs
      scipy: **wins 1.75×/2.05×/1.83×/1.24× at n=64/128/256/1024, parity
      at 512**. Symmetric-eig panel still open (next-sessions plan).
- [x] **f32/c32 phase, first cut** (2026-07-11): all four kernels generic
      over `WasmScalar` (f64x2/f32x4), f64 bit-unchanged, f32 gates green;
      runner f32 column vs scipy float32: matmul 4.3–9.1× (n≥128),
      LU-solve 2.4–3.0×, QR 3.7–5.1×, eigvals 2.0–4.3× — scipy's
      s-routines measure no faster than its d-routines on wasm. c32 hand
      kernels remain future work; c64 kernels landed in the later Schur
      and eigenvector campaigns (Phase 2 items 1(e) and 2).
- [ ] LU residual at n≥256 (0.8–0.9× vs scipy): next levers per the
      research plan are relaxed-FMA base kernels and packing the panel
      columns; diminishing returns vs the eigen flank.
- [x] **Tuned then *removed* the recursion from the LU default**
      (`lu-tune.yml` + `lu-largen.mjs`, Runs 5–7): a single-round sweep
      (crossover 256), distrusted by the architect, was redone over 3
      rounds (showed recursion marginal), then a dedicated large-n probe
      to n=1024 settled it: **pure-flat wins at every size 64–1024 on the
      runner** (tie at 512, flat 6.5–8.5% ahead at 640–1024). A dev box
      had shown recursion winning at large n — a cache quirk that didn't
      reproduce on the runner. So `RECOMMENDED_CROSSOVER = usize::MAX`:
      the default LU is now the pure-flat simd128 panel, no recursion.
      The machinery stays for explicit opt-in. The LU story simplified to
      "lean flat panel, scipy parity ~n=256" with no recursion claim to
      defend. Gated (`gate.mjs`). Docs: benchmarks Runs 5–7.

## Next sessions — build-out plan (drafted 2026-07-11, for architect review)

Sequence set by the architect: sweep ✅ → f32/c32 refactor ✅ → the
items below, with the **tuning freeze** in force until all of them exist,
then one global replication-graded tuning pass.

1. **Schur campaign** (the first advanced function; mirrors the eigvals
   playbook). **Deep research done 2026-07-11**
   (`docs/research-schur-wasm-2026-07.md`, 21/25 claims 3-vote
   confirmed): the eigvals→Schur delta is exactly want_t range-widening
   + Z updates; LAPACK holds it to ~1.26× via accumulated-U gemm
   batching (inner dim only 2·NS ≈ 32–128 — whether that pays at our
   gemm≈2×-flat regime is THE open measurement); the opponent runs
   verbatim reference-netlib for the whole Schur path (same soft bar as
   QR); no serial post-2015 replacement exists (StarNEig =
   concurrency-only, the QDWH-trap analog; RQR beat only unblocked
   zlahqr — watch-list note); Kressner LAWN 171 block reordering is a
   real serial lever (up to 4×, sourced-unverified) if reordering shows
   up hot; c64's small kernel (zlahqr) is single-shift 2×1 — a different
   cost model than real. Steps (a)–(d) **built and benchmarked same day**
   (commit `eb98432`, run 29146566266; full record in the research doc):
   (a) ✅ schur_k rows in the replication gate incl. n=1024; (b) ✅
   Z-accumulating Hessenberg (`hessenberg_form_q`, dorghr-shape backward
   accumulation — also removes the shipping Schur's exposure to faer's
   blocked-Hessenberg machine cliff); (c) ✅ hqr want_t+Z sibling with
   dlanv2 standardization + fused simd128 `refl3`/`refl2` applies;
   (d) ✅ replication-gated verdicts, post-allocator-fix reference (run
   29157035070): **WIN 1.24×/1.67×/1.08×/1.10× at n=64–512, 0.99× at
   1024** (was 0.2–0.6× at every size; the first-run 512/1024 "losses"
   were the leak-allocator tax, see 1(e)). The projection ("wins probable below n=512") held below the
   crossover; the 512/1024 losses are measured to live in the
   +Z cost of faer's multishift path (our eigvals→Schur delta there is
   1.90–2.05× vs scipy's 1.06–1.30×; below the crossover our delta is
   LAPACK-grade 1.50–1.61×). Next levers recorded in the research doc:
   hand hqr+Z past 480 (re-tune debt, frozen), wasm-shaped multishift
   Z-accumulation, or a 0004-class profile of faer's want_t/Z internals.
   (e) ✅ c64 kernel twins built same day (architect go): complex
   Hessenberg + backward-accumulated Q + Givens-based single-shift chqr
   (want_t/Z), flat scalar complex loops. Replication verdicts vs scipy
   complex Schur (run 29157035070): **WIN 1.38×/1.34× at 64/128, LOSS
   0.90× at 256, WIN 1.10×/1.03× at 512/1024** — from the 0.4–0.9×
   baseline. The 256 loss is located: our rotation applies are scalar
   while faer's lahqr applies ride pulp SIMD — next lever is a simd128
   complex-rotation primitive (one c64 per lane, the refl3 twin).
   Fallout finding — the biggest of the campaign: faer's c64 matmul
   allocates per-call temps via GlobalAlloc (15.4 GB cumulative in one
   n=600 multishift) — OOM'd the leak-only bump allocator at n≥590;
   fixed by LIFO-rewind in both wasm shims (docs/wasm.md §2),
   regression-guarded, upstream-ledgered. The fix then revealed the
   leak-only allocator had been TAXING every allocation-heavy row on our
   side of the benchmarks (cold pages + memory.grow in the timing loop;
   scipy unaffected): post-fix (run 29157035070, the new reference)
   **real schur_k WINS 1.24×/1.67×/1.08×/1.10× at n=64–512 with 0.99× at
   1024, and eigvals_k3 WINS at all five sizes incl. 512 (1.52×) and
   1024 (1.51×)** — the former 512-parity and 512/1024-loss verdicts
   were allocator tax. Gate baselines re-recorded
   (`expected-ratios.json`); LU-solve win-guard re-margined 0.6→0.85
   (part of its old margin was the tax on faer's default).
   (f) ✅ **campaign closed 2026-07-12** (commit `6c3fb49` + doc pass):
   the predicted simd128 complex-rotation primitive built and wired
   (`crot_streams`/`crot_row_pair`); f32 Schur row lands 1.7–2.5× to
   256, 1.1× at 512. crot judged by same-machine A/B
   (`bench/ab-crot.mjs`, with an untouched-op control row) after the
   cross-run reading proved untrustworthy — CI machines drift 7–15% on
   identical binaries — and KEPT: 1.17–1.25× wherever measurement
   separates, never a measured loss. It did not flip c64@256, which
   closes as the campaign's one recorded residual (0.76–0.90× vs scipy,
   machine-dependent; mechanism not yet located). Full close-out in the
   research doc.
2. ✅ **Eigenvectors (nonsymmetric `eig`)** — built and closed 2026-07-12
   (commit `62c95d1`, verdicts run 29175738677). `kernels/eigvec`:
   dtrevc3-shaped back-substitution on T (dlaln2/dladiv guards ported)
   + one *triangular* matmul back-transform through Z (X is exactly
   upper triangular in dtrevc3's packing, so faer's blocked triangular
   multiply does the gemm bulk with no zero-half waste). Generic
   f64/f32. Replication verdicts vs `np.linalg.eig`: **WIN at all five
   sizes, ranges separate — 1.80×/2.00×/1.79×/1.55×/1.64× at
   n=64–1024**; f32 row 2.9–4.4×. Wins at 1024 despite schur_k's 0.99×
   there: our eigenvector step costs ~300 ms over Schur at 1024 where
   scipy's dtrevc3+balancing tail costs ~2.7 s. **c64 twin built same
   day** (commit `459205e`, verdicts run 29177564170): `ctrevc` — one
   guarded complex division per step (no 2×2 blocks), SIMD `caxpy`/
   `cscale_re` streams, same triangular back-transform. **WIN at all
   five sizes, 3.24×/2.78×/2.61×/2.18×/2.11× at n=64–1024 — the widest
   replicated margins in the project.** That run also minted the
   verdict-stability rule: sub-1.3×-margin rows flip WIN/LOSS with the
   CI machine drawn (c64 Schur@256 read 0.89×/0.89×/1.21× across three
   machines); ≥1.4× margins replicate everywhere.
3. **SVD small-n rotation path** — profile how much of faer's small-n SVD
   (its worst losses: 0.2× @64, 0.4–0.5× @128) sits in the scalar
   `qr_algorithm` rotation application; if large, reuse the rotation
   kernel minted in 1(c). Plausible transfer, unverified — profile before
   building (the algorithm-replacement door stays closed per the
   adversarially-verified research).
4. **Symmetric eigen probe** — never scoreboarded or profiled; add the
   row + a phase profile. Cheap, and the 0004 precedent says unprofiled
   faer pipelines can hide landmines.
5. **Global tuning pass** (ends the freeze): re-sweep every threshold
   against the final shaped implementations — hqr-vs-multishift crossover
   (the 480 was measured against pre-kernel lahqr; hqr@512 never
   measured), LU/QR parameters, all f32 twins (halved cache footprint
   moves every crossover) — replication-graded, full 64–1024 grid.
6. **Packaging/budget decision (architect)**: f32/c32 monomorphizations
   cannot fit the 1,014 KB full-variant budget (c64 alone cost +443 KB);
   choose per-precision build features + new witness variants/budgets vs
   a deliberate budget raise. Note: adding CI variants needs
   workflow-file edits the current session token cannot push (lost
   `workflow` scope mid-session 2026-07-11) — architect pushes those or
   the token gets refreshed.
7. **Watch list**: faer's generic f32 gemm gains only 1.1–1.3× over f64
   on wasm (far off lane-doubling — a shaping target); wasm FP16 proposal
   (8 lanes, revisit when engines ship it); Demmel serial block-Jacobi
   Part Two (the one live SVD research thread); each upstream faer
   release re-evaluated per the release policy (0004 especially — drop
   the patch if upstream fixes the no_std window); **the 2025–26
   fast-matmul wave** (architect flag 2026-07-11, recorded unmeasured):
   AlphaEvolve's 4×4-in-48-mults (May 2025) has been rationalized to
   real/rational coefficients (arXiv 2506.13242 / 2602.13171 /
   2603.18699), flip-graph search improved the known rank for 207 small
   formats ≤ 16×16×16 (arXiv 2606.02480, June 2026), and
   alternative-basis Strassen (Numerische Mathematik, 2026) attains the
   optimal leading coefficient *with* Strassen-class error bounds.
   Candidate large-n matmul lever only: a 1-level
   alternative-basis/Strassen–Winograd wrapper over faer-gemm blocks
   saves ≤ 12.5% of mults at n ≥ 1024 for O(n²) extra adds/traffic —
   cheap measure-first probe, but it changes the error profile
   (norm-wise, not component-wise backward stable), so if it ever wins
   it ships as an opt-in path, never the default gemm; **allocator-tax
   re-verification debt (2026-07-11)**: every large-n number measured
   before the LIFO-rewind allocator fix (blocked-Hessenberg cliff
   magnitudes, the 480 crossover, LU large-n verdicts, the pre-fix
   Pyodide rows) carries leak-allocator tax on the faer side — re-read
   against post-fix runs before citing; within-run ratios are less
   contaminated than cross-run comparisons.

## Considered option — WebGPU for the large-n tail (architect Q, 2026-07-09; deferred)

The audience logic holds — the people who hit large-n dense linear
algebra are the ones with a GPU — but two facts gate the decision, so
it is recorded here rather than started:

- **WebGPU is effectively f32-only.** WGSL's numeric types are
  `i32/u32/f32/f16`; there is **no f64** in the WebGPU/WGSL spec and no
  standard `shader-f64` feature. This repo's entire contract is f64
  (LAPACK-parity accuracy, bit-identical native↔wasm determinism, the
  property probes). A WebGPU path is therefore an **f32 tier** (or
  emulated double-single, slow enough to defeat the point) — a *different
  numerical contract*, not a faster version of what we have.
- **GPU wins matmul, not the panels.** The factorization panels are
  sequential and GPU-hostile; the standard hybrid (MAGMA-style) keeps the
  panel on the CPU — i.e. it keeps *our* fast wasm-simd panel kernel. So
  even a GPU path reuses the CPU kernel work happening now.

Framing if pursued (after the CPU kernels are shaped, per the architect):
a **separate f32, matmul-and-solve-bound project for large-n power
users**, living *alongside* faer (not a faer patch — consistent with the
prime directive), explicitly a distinct numerical tier rather than a
drop-in replacement for the f64 CPU path. Not started; not re-litigated
unless the architect raises it.

## Phase 3 — Wasm performance ✅ (2026-07-08; browser gate closed it 2026-07-09)

- [x] A **benchmark harness** (`bench/`): wasm-vs-native per decomposition
      across sizes under node, results in `docs/benchmarks-2026-07.md`;
      CI keeps it buildable. Browser run: done via the foundation gate's
      browser-check.mjs (2026-07-09).
- [x] Relaxed-SIMD vs baseline deltas, published: ~11% geomean, up to
      ~25% on SVD/self-adjoint EVD at n≥128. Also measured: `opt-level`
      z→3 is ~1.75× overall — the biggest knob we control.
- [x] Single-thread **blocking-parameter tuning** for wasm — swept via
      `bench/tune.mjs`, resolved: unblocked kernels win through n=256.
      LU with `recursion_threshold ≥ n` → 1.25–1.5× native (was up to
      9.6×); QR with Householder panel width 1 → ≈0.9× of faer's default
      native time (was ~8–10× slower). Consumer guidance in
      `docs/wasm.md` §7; tables in `docs/benchmarks-2026-07.md`.
      Open residue: SVD/EVD internal stages (untuned, ~3.3×+) and
      re-sweeping beyond n=256.
- [x] **Determinism guarantee** enforced in CI: native vs wasm probe
      values compared bit-for-bit on every push (plain *and* relaxed-SIMD
      builds), via `smoke-test/src/bin/native.rs` + `determinism.mjs`.
- Open question (architect, 2026-07-08, undecided): what should external
  benchmarks compare against? Candidates identified, none chosen —
  (a) OpenBLAS/LAPACK single-thread (algorithmic apples-to-apples vs
  what Julia calls; feasible in-container via numpy), (b) OpenBLAS
  default multithread (the gap desktop users would actually feel;
  quantifies what Phase 4 threads would buy), (c) a pure-JS matrix
  library (brackets from below; the "why wasm at all" number). Depends
  on audience: consumer tuning vs Ruju-vs-Julia vs wasm value pitch.

## Phase 4 — Threads, later and optional

`Par::Seq` is the wasm story until wasm threads (SharedArrayBuffer +
atomics) justify more. If demand appears: a `Par` backend over wasm threads
without `atomic-wait` (busy-wait or `memory.atomic.wait32` where available).
Not needed for any current consumer; keep as a recorded non-goal until it is.

## Upstream ledger (started 2026-07-11 — de-prioritized, not forbidden)

Everything found so far that would plausibly help upstream, recorded as
discovered per the revised prime directive. Nothing here is being
prepared for submission yet; when the project settles, the architect
picks from this list. Canonical target is Codeberg
(`codeberg.org/sarah-quinones/faer` — GitHub is a mirror); the archived
`upstream/` package (fix + regression tests + wasm CI job for 0001) is
the submission template.

| finding | kind | evidence |
| - | - | - |
| `patches/faer-rs/0001`: `(n >> 32)` on 32-bit `usize` is a compile error — faer does not build on ANY 32-bit target (wasm32, armv7, i686) | build fix, 4 lines | ready-made package archived in `upstream/`; CI-verified every push |
| `patches/pulp/0003`: wasm `RelaxedSimd` complex `mul_add_e`/`mul_e` pass NEON accumulator-first FMA args to accumulator-last `relaxed_madd` — ALL c32/c64 compute wrong under `+relaxed-simd` | correctness fix, 4 lines | root-caused 2026-07-08; `schur_probe_cplx == 3` CI guard; docs/wasm.md §4 |
| `patches/faer-rs/0004`: `no_std` AED deflation window computes `log2(n/n)` = 0 instead of `n/log2(n)` — eigensolver iterations explode ~50–85× for 150 ≤ n < 590 on every `no_std` build | perf fix, 1 line | iteration counters pre/post on runner; docs/research-eig-wasm-2026-07.md |
| `patches/faer-rs/0002`: Schur kernels are `pub(crate)` — better framed upstream as a feature request (public Schur API + `trexc`/`trsen` reordering, the `faer-schur` shape) than as the raw visibility patch | API gap | `schur/` crate implements the shape; accuracy CI-tested |
| latent 64-bit bug: `operator/*` splits `n` into u32 halves then recombines as `n0 as f64 + n1 as f64` without scaling `n1` by 2³² — looks wrong for n ≥ 2³² (intent UNCERTAIN) | bug report candidate | research-faer-wasm-2026-07.md §8, source reading only |
| blocked multishift QR leaves workspace junk below the subdiagonal of `T` (faer's own EVD never reads it; anyone consuming `T` directly gets garbage) | bug report candidate | found 2026-07-08 building `faer-schur`, which zeroes it; accuracy tests |
| dead code: `real_schur.rs:837` `if true \|\|` disables the recursive-multishift AED branch — possibly intentional, worth asking | question upstream | source diff vs LAPACK, research-eig-wasm-2026-07.md |
| blocked Hessenberg has a machine-sensitive cache cliff (7–95× slower than an unblocked kernel at n=1024 across runner instances) | perf report candidate | phase-split probe run 29136868733, cross-checked on 3 machines |
| complex `JacobiRotation::rotg(a, b)` returns `r = 1` when `b == 0` (LAPACK `zlartg` returns `r = a`); the complex `lahqr` chase writes that `r` over the subdiagonal — wrong output for the measure-zero input class where a bulge entry is exactly 0 | bug report candidate | source reading during the c64 kernel port (our port uses LAPACK semantics); not yet reproduced with a concrete input |
| c64 matmul allocates per-call temporaries via GlobalAlloc (f64 path doesn't): one c64 multishift call at n=600 = 15.4 GB cumulative / ~25K allocations, peak live ~19 MB — a `no_std` perf hazard (25K allocs per solve) and fatal on allocator-less/arena wasm patterns; also, the complex AED recurses into `multishift_qr` where the real path dead-codes that branch (`if true \|\|`) — an intentional(?) real/complex asymmetry | perf/no_std report candidate | counting-allocator probe `kernels/tests/alloc_probe.rs`, 2026-07-11; wasm OOM reproduced + fixed by LIFO-rewind shim |

## Boundary note — what does NOT live in this repo

The BLAS/LAPACK **ABI shim** (Fortran symbol names, column-major leading
dimensions, transpose/uplo flags, ILP64 widths, `info` codes, workspace-query
protocol) is consumer-side. Upstream already ships **`faer-ffi`** (a C-ABI
crate) — the natural foundation to build against or extend in companion
code, while the Fortran-flavored LBT layer stays with the consumer (for
Ruju: inside its runtime, registered in its internal ccall symbol table).

## Cadence

Upstream is a resource, not an obligation: releases are evaluated and
adopted only when they advance us (then re-pin, re-apply `patches/`,
re-run the gate); when upstream deviates from our needs, we stay pinned.
The carry stays minimal — if an adopted release builds on 32-bit targets
without our patch, the patch is deleted. A phase is "done" when its
capability is available to consumers with the gate green **and** it has
had its closing doc pass (truth-seeking: docs vs evidence — see the
working contract in CLAUDE.md, incl. the evidence grid in README.md).
