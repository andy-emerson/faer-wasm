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

**Prime directive: keep the carry thin.** Strategy settled 2026-07-08:
nothing is submitted upstream (Andy's decision — do not revisit
unprompted). We vendor the minimum patch set against a pinned upstream
base, re-verify on every faer release, and drop patches the moment a
release doesn't need them. New capability is built *alongside* faer
(companion crates / consumer shim over public APIs), not inside it.

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
- [ ] **Wasm-shaped block-apply kernel `(I−V T Vᵀ)·C` in
      `faer-wasm-kernels`** — coarse-only (no recursion, no skinny-gemm
      base case; the LU lesson), routed into faer gemm. Buys little on
      already-won QR; built as the reusable inner loop of Hessenberg.
      Note the QR result reinforces the LU lesson: on 2-lane wasm the
      flat simd128 loop is so close to gemm rate that routing into gemm
      barely helps below ~512 — the block-apply's value is Hessenberg
      structure reuse, not raw QR speed.
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
- [ ] **Pivotal pre-build measurement**: does faer's own
      `gebrd`/`sytrd`/`gehrd` just need param-tuning (faer-schur style) or
      a flat-panel rebuild? Same fork QR faced. One profiling run +
      `faer-rs/` read decides tuning-job vs kernel-job. **Do this before
      building.**
- [ ] **Eigen-flank kernel(s)** per the research: (1) flat simd128
      reduction panel + block-apply→gemm — unblocks SVD *and* general-eig;
      (2) symmetric `symv`/`syr2k` panel — unblocks symmetric eig. Scope
      after the measurement above.
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
   playbook). (a) Add Schur real+c64 to the replication gate incl.
   n=1024 — baseline to standard (current informal: ~0.4–0.6× scipy).
   (b) Z-accumulating Hessenberg: extend the kernel to form/apply Q from
   its stored reflectors (the reconstruction code shape already exists in
   the test suite); this also removes the shipping Schur's exposure to
   faer's blocked-Hessenberg machine cliff. (c) Z-accumulating iteration:
   the hqr kernel's `want_t=true` + Z sibling — Z updates are contiguous
   3-column axpys, friendlier than the eigvals shapes. (d) Benchmark each
   step. (e) Decision point: c64 kernel twins (a NEW build — no complex
   hand kernels exist at either precision). Projection from measured
   structure: parity very likely, wins probable below n=512 (scipy pays
   1.26× to go eigvals→Schur; faer currently pays 2.6×).
2. **Eigenvectors (nonsymmetric `eig`)** — needs Schur first:
   `trevc`-shaped back-substitution on T + back-transform through Z, both
   kernel-shaped; scoreboard row vs `np.linalg.eig`.
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
   the patch if upstream fixes the no_std window).

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
