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

## Phase 3 — Wasm performance ✅ (2026-07-08; browser-run refinement open)

- [x] A **benchmark harness** (`bench/`): wasm-vs-native per decomposition
      across sizes under node, results in `docs/benchmarks-2026-07.md`;
      CI keeps it buildable. Browser run: follow-up refinement, on real
      hardware.
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
capability is available to consumers with the gate green.
