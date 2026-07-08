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

**Prime directive: keep the carry thin.** Strategy settled 2026-07-08:
nothing is submitted upstream (Andy's decision — do not revisit
unprompted). We vendor the minimum patch set against a pinned upstream
base, re-verify on every faer release, and drop patches the moment a
release doesn't need them. New capability is built *alongside* faer
(companion crates / consumer shim over public APIs), not inside it.

## Phase 0 — Carry the enabler ✅ (maintenance mode)

- [x] The **4-line 32-bit fix**: `(n >> 32)` → `((n as u64) >> 32) as u32` in
      `operator/{eigen, self_adjoint_eigen, svd}`. Carried in
      `patches/0001-*.patch`; zero-`v0` regression tests exist in the
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

## Phase 1 — Wasm consumer ergonomics

- [ ] A documented **wasm recipe** (`docs/wasm.md` in this repo): feature
      set, `panic = "abort"`, `opt-level = "z"` + fat LTO, expected sizes
      per decomposition (the measured 51 KiB → 396 KiB table), the `no_std`
      path, and the determinism note (bit-identical to native).
- [ ] **Relaxed-SIMD route documented**: consumers reach pulp's
      `RelaxedSimd` via feature unification (verified working). FMA via
      `relaxed_madd` is the single biggest wasm perf lever (+~10% size).
- [ ] **Size regression tracking** in this repo's CI: the per-decomposition
      wasm sizes as a checked budget, so dependency creep is caught early.

## Phase 2 — Coverage growth (the LAPACK-parity tail)

Prioritized by what a LAPACK-replacing consumer hits first; each item is
built **alongside faer** (a companion crate or the consumer's shim, over
faer's public low-level modules) with accuracy tests against LAPACK
reference values. Verified-missing today:

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

Upstream is a resource, not an obligation: releases are evaluated and
adopted only when they advance us (then re-pin, re-apply `patches/`,
re-run the gate); when upstream deviates from our needs, we stay pinned.
The carry stays minimal — if an adopted release builds on 32-bit targets
without our patch, the patch is deleted. A phase is "done" when its
capability is available to consumers with the gate green.
