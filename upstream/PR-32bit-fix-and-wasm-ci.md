# PR to file on Codeberg: 32-bit fix + wasm32 CI

Target: https://codeberg.org/sarah-quinones/faer-rs (canonical home; GitHub
is a mirror). Base: `main` at `0539947` (v0.24.4 era) — rebase if upstream
has moved; the diff is tiny and should apply cleanly.

The branch is prepared locally as `wasm32-32bit-fix` on the `faer-rs/`
clone (two commits). To recreate it from scratch:

    git checkout -b wasm32-32bit-fix
    git am upstream/0001-*.patch upstream/0002-*.patch

Then push to a Codeberg fork and open the PR with the title and body
below.

---

## Title

fix 32-bit usize overflow in matrix-free operators, add wasm32 CI

## Body

faer 0.24.4 does not compile for **any 32-bit target** —
`wasm32-unknown-unknown`, `armv7`, `i686` — with any feature set:

```
error: this arithmetic operation will overflow
   --> faer/src/./operator/eigen/mod.rs:319:13
    |
319 |             let n1 = (n >> 32) as u32;
    |                      ^^^^^^^^^ attempt to shift right by `32_i32`, which would overflow
```

The zero-`v0` fallback in the matrix-free partial eigen/svd operators
splits `n: usize` into two `u32` halves with `(n >> 32)`, which is a
compile-time overflow error when `usize` is 32 bits. Four sites, three
files: `operator/eigen/mod.rs` (real + cplx),
`operator/self_adjoint_eigen/mod.rs`, `operator/svd/mod.rs`.

### Commit 1 — the fix + regression tests

Widen before shifting: `(n >> 32)` → `((n as u64) >> 32) as u32`. This is
a no-op on 64-bit targets and well-defined on 32-bit ones.

The fallback only runs when the caller passes an all-zero initial vector —
a path none of the existing tests exercised (they all normalize a random
`v0`). This adds zero-`v0` regression tests covering all four sites
(`partial_eigen` real + cplx, `partial_self_adjoint_eigen_imp`,
`partial_svd_imp`), same style/assertions as the neighboring tests.

### Commit 2 — wasm32 CI

`.woodpecker/wasm.yaml`:

- `cargo check --target wasm32-unknown-unknown --no-default-features
  --features linalg` (and `linalg,std`) — this alone would have caught the
  overflow, and keeps 32-bit breakage from silently regressing;
- `faer-wasm-test/`, a tiny `no_std` cdylib consumer (excluded from the
  workspace, mirroring `faer-no-std-test`) that builds to a
  **zero-import** wasm module and runs matmul + partial-pivot LU solve
  under node 22, checked against hand-computed values (trace = 114,
  solution sum = 31/35).

The `rayon` feature is deliberately not checked: it can't build on wasm32
today (`atomic-wait` has no wasm port); `Par::Seq` is the wasm mode.

### Verification

With the fix applied, the full dense suite — matmul, partial-pivot LU
solve, QR, SVD, self-adjoint EVD, and general complex EVD — builds with
`default-features = false, features = ["linalg"]` and runs under node
**bit-identical to native x86-64** (rustc 1.94.1). Sizes with
`opt-level = "z"`, fat LTO, `panic = "abort"`, strip: 51 KiB matmul-only
up to ~396 KiB for the full suite (~435 KiB with pulp's `relaxed-simd`
FMA — pulp's simd128/relaxed-simd backend and the `gemm` crate's wasm
microkernels already work out of the box, with real `f64x2.mul` /
`f32x4.relaxed_madd` in the disassembly).

Context: there is real faer-on-wasm usage in the wild — #222 (sparse
Cholesky on wasm32 in Firefox), fixed in 2025. This makes the target a
checked configuration so it stays working.

### Side observation (not addressed here, happy to follow up)

The split converts `n` to the scalar type as
`from_f64(n0 as f64) + from_f64(n1 as f64)` — the high half `n1` is never
scaled by 2^32, so the sum equals `n` only for `n < 2^32`. On 64-bit
targets with `n >= 2^32` (a >4-billion-row operator, admittedly
theoretical) the normalization constant would be wrong. If the intent is
an exact conversion, `from_f64(n0 as f64) + from_f64(n1 as f64) *
from_f64(4294967296.0)` would do it; this PR keeps the minimal
compile-fix and doesn't change 64-bit behavior.
