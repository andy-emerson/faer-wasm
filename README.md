# faer-wasm

Staging ground for making [faer](https://codeberg.org/sarah-quinones/faer-rs)
a first-class wasm32 citizen. **Thin-fork discipline**: nothing lives here
long-term — every change's destination is an upstream PR (Codeberg;
GitHub is a mirror). See ROADMAP.md; empirical basis in docs/.

## Contents

- `ROADMAP.md` — the phased plan (Phase 0: the 32-bit fix + wasm CI).
- `patches/0001-fix-32bit-usize-shift.patch` — the 4-line fix that makes
  faer build on wasm32 (and armv7/i686): `(n >> 32)` on 32-bit `usize` →
  `((n as u64) >> 32) as u32`, in `operator/{eigen,self_adjoint_eigen,svd}`.
  Base commit in `patches/UPSTREAM-BASE.txt`.
- `smoke-test/` — a zero-import `no_std` consumer crate that builds to
  `wasm32-unknown-unknown` (with the patch applied to a local faer checkout)
  and runs matmul / LU / QR / SVD / EVD under node, verified bit-identical
  to native x86-64.
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
    git -C faer-rs apply ../patches/0001-fix-32bit-usize-shift.patch
    cd smoke-test && cargo build --target wasm32-unknown-unknown --release
    node run.mjs
