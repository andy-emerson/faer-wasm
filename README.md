# faer-wasm

Makes [faer](https://github.com/sarah-quinones/faer-rs) usable as a
first-class wasm32 dependency. **Thin-carry discipline**: we vendor the
smallest possible patch set, keep it clean against a pinned upstream base,
re-verify on every faer release, and drop patches the moment upstream
doesn't need them. Nothing is submitted upstream (by choice — see
ROADMAP.md); new capability is built alongside faer, not inside it.
Empirical basis in docs/.

## Contents

- `ROADMAP.md` — the phased plan. Phases 0, 1, 3 (the carried fix, CI
  gate, consumer recipe, size budgets, benchmarks + tuning) are done;
  Phase 2 (the LAPACK-parity tail) is underway — Schur + reordering
  landed first.
- `patches/` — the carried patch set (base commit in
  `patches/UPSTREAM-BASE.txt`):
  `0001-fix-32bit-usize-shift.patch`, the 4-line fix that makes faer
  build on wasm32 (and armv7/i686): `(n >> 32)` on 32-bit `usize` →
  `((n as u64) >> 32) as u32`, in `operator/{eigen,self_adjoint_eigen,svd}`;
  and `0002-expose-schur-kernels.patch`, 6 visibility-only lines
  (`pub(crate)` → `pub`, no behavior change) exposing the Schur kernels
  faer already has, so `schur/` can drive them from outside.
- `schur/` — **`faer-schur`**, the first Phase 2 companion crate:
  real + complex Schur decomposition (`gees`-shaped) and eigenvalue
  reordering (`trexc`/`trsen`-shaped) over faer's public API. `no_std`,
  wasm-gated in CI, accuracy-tested against backward error /
  orthogonality / faer's own EVD (`schur/tests/accuracy.rs`).
- `smoke-test/` — a zero-import `no_std` consumer crate that builds to
  `wasm32-unknown-unknown` (with the patch applied to a local faer checkout)
  and runs matmul / LU / QR / SVD / EVD under node, verified bit-identical
  to native x86-64.
- `upstream/` — **archived, shelved**: a complete contribution (fix +
  regression tests + wasm CI job, as `git am`-able patches with PR/issue
  text) that was prepared and then deliberately not submitted. Kept
  because the patches double as our own regression tests; don't extend it.
- `docs/wasm.md` — **the consumer recipe**: cargo setup, features that
  work (and `rayon`, which doesn't), the `no_std` zero-import pattern,
  sizes + budgets, the relaxed-SIMD (FMA) route, determinism guarantee.
- `bench/` + `docs/benchmarks-2026-07.md` — the wasm-vs-native benchmark
  harness and its first published numbers (opt-level ~1.75×, relaxed-SIMD
  ~11%, large matmul at 1.8–1.9× native, mid-size blocking cliffs
  identified).
- `docs/research-faer-wasm-2026-07.md` — the verification research:
  measured sizes (51 KiB matmul → ~396 KiB full suite), pulp simd128 status
  (already complete upstream), LinearAlgebra coverage matrix.

## Evidence grid

Per the working contract (CLAUDE.md): claims are graded on **strength**
(stated < built < observed < tested < proven) and **durability**
(by-hand < scripted < CI-enforced < cross-checked), and never above
their evidence.

| claim | strength | durability | evidence |
| - | - | - | - |
| faer + carried patch builds for wasm32 (`linalg`, `linalg,std`) | tested | CI-enforced | wasm gate, every push |
| patches apply cleanly on the pinned base | tested | CI-enforced | gate does clone → pin → apply |
| full dense suite runs under node, exact hand-verified values | tested | CI-enforced | `check.mjs`, 4 build variants |
| native ↔ wasm bit-identical, incl. relaxed-SIMD build | tested | CI-enforced | `determinism.mjs` (3 probes spanning matmul/LU/QR/SVD/EVD — not all inputs; known NOT to extend to 8×8 Schur raw doubles, see docs/wasm.md §5) |
| `.wasm` sizes 59→447 KB, within budgets | tested | CI-enforced | `size-budgets.json` |
| relaxed-SIMD emits real FMA (`relaxed_madd`) | observed | by-hand | 2026-07 disassembly counts |
| `rayon` cannot build on wasm32 | observed | by-hand | build probe at the pin |
| perf: matmul 1.9× native; opt-level z→3 ≈ 1.75×; relaxed-SIMD ≈ +11% | observed | scripted | `bench/`, min-of-2 reps, shared box |
| tuning: unblocked kernels win ≤ n=256 (LU 1.25–1.5× native, QR ≈ 0.9× native-default) | observed | scripted | `bench/tune.mjs` sweep |
| Schur (f64+c64) + reordering correct: backward error ~1e-15, orthogonality, eigenvalues match faer EVD, reorder invariants; n ≤ 150 incl. blocked/AED path | tested | CI-enforced | `schur/tests/accuracy.rs` + wasm property probes in all full variants |
| c64 compute broken under `+relaxed-simd` at this pin (incl. faer's own `.eigenvalues()`); f64 + c64 matmul unaffected | tested | CI-enforced | isolated 2026-07-08 (Hessenberg stage, ‖·‖²≈186); `check.mjs` canary asserts the broken state |
| shelved upstream patches recreate the branch byte-identically | observed | by-hand | `git am` round-trip, 2026-07-08 |
| SVD/EVD wasm overhead is untuned (~3.3×+) | observed | scripted | bench tables |
| blocked paths must win beyond some n | stated | — | untested past n=256 |
| runs in real browsers (only node/V8 exercised) | stated | — | browser run never performed |

## Quick start

The smoke test path-depends on **both** upstream clones sitting in the repo
root (gitignored): `faer-rs/` and `pulp/`. Commits are pinned in
`patches/UPSTREAM-BASE.txt`.

    git clone https://github.com/sarah-quinones/faer-rs
    git clone https://github.com/sarah-quinones/pulp
    git -C faer-rs checkout <faer commit in patches/UPSTREAM-BASE.txt>
    git -C pulp    checkout <pulp commit in patches/UPSTREAM-BASE.txt>
    for p in patches/*.patch; do git -C faer-rs apply "../$p"; done
    cd smoke-test && cargo build --lib --target wasm32-unknown-unknown --release --features full
    node check.mjs   # exact-value + size gate; run.mjs just prints
    # (--lib: the `native` bin is host-only, for the determinism cross-check)

Consumer-facing build recipe (features, sizes, SIMD, determinism):
`docs/wasm.md`.
