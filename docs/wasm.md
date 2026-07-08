# faer on wasm32 — the consumer recipe

How to depend on faer from a `wasm32-unknown-unknown` crate. Everything
here is measured, not assumed; the evidence trail is
`research-faer-wasm-2026-07.md` and the enforcement is
`.github/workflows/wasm-gate.yml`. Toolchain of record: rustc 1.94.1,
node 22.

## 0. You need the carried patch

Plain faer 0.24.4 from crates.io does **not** compile for any 32-bit
target — `(n >> 32)` on a 32-bit `usize` is a compile error in
`operator/{eigen,self_adjoint_eigen,svd}`. Set up the pinned + patched
checkout first (repo README "Quick start"): clone `faer-rs` and `pulp`,
pin to `patches/UPSTREAM-BASE.txt`, apply `patches/*.patch`, then depend
by path.

## 1. Cargo setup

```toml
[dependencies]
faer = { path = "../faer-rs/faer", default-features = false, features = ["linalg"] }
# optional but recommended — unlocks real FMA, see §4:
pulp = { path = "../pulp/pulp", default-features = false, features = ["relaxed-simd"] }

[profile.release]
opt-level = "z"     # size; "3" trades ~size for speed if you prefer
lto = true
codegen-units = 1
panic = "abort"
strip = true
```

Feature facts (all verified):

- `linalg` alone builds and gives the full dense suite: matmul, LU
  (partial + full pivot), Cholesky family, QR ± column pivoting, SVD,
  self-adjoint and general EVD, generalized EVD, triangular solve/inverse,
  full complex support.
- `linalg,std` also builds on wasm32.
- `rayon` does **not** build (`atomic-wait` has no wasm port) and must
  stay off. `Par::Seq` is a first-class argument accepted by every compute
  routine — sequential is the wasm mode, not a degraded fallback.

## 2. no_std / zero-import modules

For a module that instantiates with an empty import object (no JS glue,
no wasm-bindgen), see `smoke-test/src/lib.rs`: `crate-type = ["cdylib"]`,
`#![no_std]`, a leak-only bump allocator over `memory.grow` seeded from
`__heap_base`, and a `panic_handler` that hits `unreachable`. The produced
module needs zero imports:

```js
const { instance } = await WebAssembly.instantiate(bytes, {});
```

## 3. Sizes (pre-wasm-opt, pre-gzip)

Measured 2026-07-08 on the smoke test (pulp `relaxed-simd` feature in the
tree; budgets enforced in CI via `smoke-test/size-budgets.json`):

| variant | bytes | budget |
| - | -: | -: |
| matmul only | 59,207 | 66,000 |
| + LU solve | 123,751 | 137,000 |
| + QR, SVD, both EVDs | 447,270 | 492,000 |
| same, `+simd128,+relaxed-simd` baked in | 440,441 | 485,000 |

The 2026-07 research build with a leaner staging (no pulp `relaxed-simd`
feature) measured 51.4 KiB / 106.4 KiB / 396.2 KiB — treat ~400 KiB as
the planning number for the full suite. `wasm-opt` and gzip/brotli
transport shrink it further (unmeasured here).

## 4. SIMD

pulp (faer's SIMD layer) ships a complete wasm backend; nothing needs to
be contributed or configured upstream.

- **Baseline:** simd128 code paths are compiled behind
  `#[target_feature]` and selected at runtime via a host-set flag
  (`pulp-wasm-simd-flag`), or selected at compile time by building with
  `-C target-feature=+simd128`.
- **FMA (the big lever):** depend on `pulp` with
  `features = ["relaxed-simd"]` (cargo feature unification does the rest)
  and build with:

  ```sh
  RUSTFLAGS='-C target-feature=+simd128,+relaxed-simd' cargo build --target wasm32-unknown-unknown --release
  ```

  This emits real `f64x2.relaxed_madd` (253 occurrences in the research
  disassembly vs 0 baseline) and runs in node 22 and any
  relaxed-SIMD-capable browser. The gate builds this variant on every
  push and its results are bit-identical to the plain build for the
  reference probes.

## 5. Determinism

The smoke-test probe values are **bit-identical** between native x86-64
and wasm (all variants, including relaxed-SIMD):

```
matmul_trace     = 114
lu_solve_sum     = 0.8857142857142857   (31/35)
qr_svd_evd_probe = 1.9483450492039642
```

CI compares exactly (`Object.is`, no tolerance). Treat any cross-target
difference as a bug, not noise.

## 6. What CI enforces (`wasm gate`)

On every push/PR: fetch both upstreams at the pinned commits → apply
`patches/` → build all four variants → run each under node with exact
value checks and size budgets. If a faer re-pin or a dependency bump
breaks the build, changes a result bit, or bloats a binary past budget,
the gate fails.
