# faer-wasm

Makes [faer](https://github.com/sarah-quinones/faer-rs) usable as a
first-class wasm32 dependency. **Thin-carry discipline**: we vendor the
smallest possible patch set, keep it clean against a pinned upstream base,
re-verify on every faer release, and drop patches the moment upstream
doesn't need them. Nothing is submitted upstream (by choice — see
ROADMAP.md); new capability is built alongside faer, not inside it.
Empirical basis in docs/.

## Contents

- `ROADMAP.md` — the phased plan. Phases 0–1 (the carried fix, CI gate,
  consumer recipe, size budgets) are done; next are the LAPACK-parity
  tail and the benchmark harness.
- `patches/0001-fix-32bit-usize-shift.patch` — the 4-line fix that makes
  faer build on wasm32 (and armv7/i686): `(n >> 32)` on 32-bit `usize` →
  `((n as u64) >> 32) as u32`, in `operator/{eigen,self_adjoint_eigen,svd}`.
  Base commit in `patches/UPSTREAM-BASE.txt`.
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
