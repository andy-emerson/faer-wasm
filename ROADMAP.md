# faer-on-wasm roadmap

A phased plan for the faer clone, scoped **language-agnostically**: every item
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

**Prime directive: keep the fork thin.** Every phase's endpoint is an
upstream PR (faer's home is Codeberg; GitHub is a mirror). Zero long-lived
divergence is the success criterion — precedent (faer-rs#222) shows real
wasm users and a responsive maintainer.

## Phase 0 — Land the enabler (days-scale)

- [ ] The **4-line 32-bit fix**: `(n >> 32)` → `((n as u64) >> 32) as u32` in
      `operator/{eigen, self_adjoint_eigen, svd}`. Benefits wasm32, armv7,
      i686 — a genuine 32-bit-correctness fix, not a wasm accommodation.
      Submit upstream with a regression test on a 32-bit target.
      *(prepared 2026-07-07 — patch + zero-`v0` regression tests in
      `upstream/0001-*.patch`; awaiting Codeberg filing)*
- [ ] A **wasm32 CI job**: build `--target wasm32-unknown-unknown` with
      `default-features = false, features = ["linalg"]` + a headless node
      smoke test (LU solve, hand-verified values), so wasm can't silently
      regress again.
      *(prepared 2026-07-07 — `.woodpecker/wasm.yaml` + `faer-wasm-test`
      crate in `upstream/0002-*.patch`; awaiting Codeberg filing)*
- [ ] Open the conversation with the maintainer: wasm as a supported tier,
      referencing #222's existing demand.

## Phase 1 — Wasm consumer ergonomics

- [ ] A documented **wasm recipe** (README section or `docs/wasm.md`):
      feature set, `panic = "abort"`, `opt-level = "z"` + fat LTO, expected
      sizes per decomposition (publish the measured 51 KiB → 396 KiB table),
      the `no_std` path, and the determinism note (bit-identical to native —
      worth advertising as a feature).
- [ ] **Relaxed-SIMD passthrough**: consumers can already reach pulp's
      `RelaxedSimd` via feature unification (verified); either document that
      route or add an explicit faer feature that forwards to it. FMA via
      `relaxed_madd` is the single biggest wasm perf lever (+~10% size).
- [ ] **Size regression tracking** in CI: the per-decomposition wasm sizes as
      a checked budget, so dependency creep is caught at PR time.

## Phase 2 — Coverage growth (the LAPACK-parity tail)

Prioritized by what a LAPACK-replacing consumer hits first; each item lands
with accuracy tests against LAPACK reference values. Verified-missing today:

- [ ] **Schur decomposition** (`gees`-equivalent) + **eigenvalue reordering**
      (`trexc`/`trsen`) — the largest gap; unlocks matrix functions
      (`exp(A)`, `log(A)`) for every consumer.
- [ ] **Sylvester solver** (`trsyl`) — pairs with Schur.
- [ ] **LQ adapter** (thin; QR-of-transpose plumbing).
- [ ] `geevx`-style balancing / condition-number extras.
- [ ] Generalized SVD (`ggsvd3`) — lowest priority; rare call sites.

Already covered (no work): LU (partial+full pivot), LLT/pivoted-LLT/LDLT/
Bunch-Kaufman, QR ± column pivoting, SVD, self-adjoint + general EVD,
generalized EVD (QZ), triangular solve/inverse, full complex support.

## Phase 3 — Wasm performance

- [ ] A **benchmark harness**: wasm-vs-native per decomposition across sizes
      (node + a browser run), tracked over time.
- [ ] Relaxed-SIMD vs baseline simd128 deltas, published.
- [ ] Single-thread **blocking-parameter tuning** for wasm's flat memory
      (cache-size heuristics differ from native; measure, don't assume).
- [ ] Formalize the **determinism guarantee** as a cross-target CI test
      (native and wasm outputs compared bit-for-bit).

## Phase 4 — Threads, later and optional

`Par::Seq` is the wasm story until wasm threads (SharedArrayBuffer +
atomics) justify more. If demand appears: a `Par` backend over wasm threads
without `atomic-wait` (busy-wait or `memory.atomic.wait32` where available).
Not needed for any current consumer; keep as a recorded non-goal until it is.

## Boundary note — what does NOT live in this repo

The BLAS/LAPACK **ABI shim** (Fortran symbol names, column-major leading
dimensions, transpose/uplo flags, ILP64 widths, `info` codes, workspace-query
protocol) is consumer-side. Upstream already ships **`faer-ffi`** (a C-ABI
crate) — the natural foundation; anything *generic* added to `faer-ffi`
(routine coverage, C-ABI surface) upstreams here, while the Fortran-flavored
LBT layer stays with the consumer (for Ruju: inside its runtime, registered
in its internal ccall symbol table).

## Cadence

Track upstream releases; rebase the clone rather than diverge. A phase is
"done" when its PRs are merged upstream and the clone carries nothing — the
roadmap's end state is that this repo is unnecessary.
