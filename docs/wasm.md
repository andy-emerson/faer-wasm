# faer on wasm32 — the consumer recipe

How to depend on faer from a `wasm32-unknown-unknown` crate. Everything
here is measured, not assumed; the evidence trail is
`research-faer-wasm-2026-07.md` and the enforcement is
`.github/workflows/wasm-gate.yml`. Toolchain of record: rustc 1.94.1,
node 22.

## 0. You need the carried patches

Plain faer 0.24.4 from crates.io does **not** compile for any 32-bit
target — `(n >> 32)` on a 32-bit `usize` is a compile error in
`operator/{eigen,self_adjoint_eigen,svd}`; that's patch 0001 (4 lines).
Patch 0002 (6 lines, visibility-only) exposes the Schur kernels that §8's
companion crate drives. Set up the pinned + patched checkout first (repo
README "Quick start"): clone `faer-rs` and `pulp`, pin to
`patches/UPSTREAM-BASE.txt`, apply `patches/*.patch`, then depend by
path.

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
| + QR, SVD, both EVDs, Schur (f64 + c64) | 738,486 | 806,000 |
| same, `+simd128,+relaxed-simd` baked in | 726,178 | 793,000 |

The full variant grew from ~447 KB when `faer-schur` landed (§8): real
Schur + reordering is only ~+26 KB (its kernels were already in the
binary via the EVD), but the **c64 Schur adds ~+260 KB** — the whole
complex eigensolver stack, monomorphized for the first time (the earlier
"general EVD" probe fed a real matrix into the real kernels). Skip the
complex driver if you don't need it and the ~473 KB / ~400 KB planning
numbers still hold. `wasm-opt` and gzip/brotli transport shrink all of
these further (unmeasured here).

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

- **⚠ Relaxed-SIMD is real-valued-only at the current pin.** Building
  with `+relaxed-simd` makes faer's **complex (`c64`) kernels produce
  grossly wrong results** — found 2026-07-08 by the Schur gate: the c64
  Hessenberg reduction has O(1) backward error (‖A − QHQᴴ‖² ≈ 186 on a
  unit-scale 4×4), and even faer's public `.eigenvalues()` on a complex
  input returns garbage (eigenvalue sum off trace by ~3.6). Real `f64`
  paths and c64 *matmul* are unaffected; the plain `+simd128` build is
  correct for everything. Until this is root-caused (upstream faer/pulp
  at our pin), use relaxed-SIMD only for real-valued workloads. CI
  asserts the broken state on every push (`check.mjs` canary), so an
  upstream fix will trip the gate and retire this caveat.

## 5. Determinism — scope it carefully

The three original smoke-test probe values are **bit-identical** between
native x86-64 and wasm (all variants, including relaxed-SIMD):

```
matmul_trace     = 114
lu_solve_sum     = 0.8857142857142857   (31/35)
qr_svd_evd_probe = 1.9483450492039642
```

CI compares exactly (`Object.is`, no tolerance). For *those probes*,
treat any cross-target difference as a bug, not noise.

**Bit-identity does not generalize.** The first counterexample (found
2026-07-08): the raw doubles out of an 8×8 Schur decomposition differ
between native and wasm in the last ulps — pulp's reduction order
depends on SIMD width (AVX vs simd128), which the tiny 3×3 probes never
exposed — and under relaxed-SIMD FMA the QR iteration can take a
different path entirely, landing the (canonically unordered) eigenvalues
in a different diagonal order. All results are equally *correct* (same
backward error); they are not the same bits. The gate therefore scores
the Schur pipeline by integer-valued correctness properties
(`schur_probe` = 11, `schur_probe_cplx` = 3) and bit-compares the score.
Takeaway for consumers: expect bit-reproducibility per fixed
target+flags, not across targets, once matrices are big enough for SIMD
reductions to kick in.

## 6. What CI enforces (`wasm gate`)

On every push/PR: fetch both upstreams at the pinned commits → apply
`patches/` → run the `faer-schur` accuracy tests natively → build all
four variants → run each under node with exact value checks and size
budgets → cross-target determinism (bit-for-bit). If a faer re-pin or a
dependency bump breaks the build, changes a result bit, or bloats a
binary past budget, the gate fails. The full-relaxed variant also runs
the c64 canary: it *asserts* the known-broken c64×relaxed-simd state
(§4), so an upstream fix fails loudly instead of going unnoticed.

## 7. Blocking parameters on wasm (measured 2026-07-08)

faer's default blocking thresholds are tuned for native caches and
misfire badly on wasm — mid-size factorizations run up to ~10× slower
than necessary. Measured fix (details + tables in
`benchmarks-2026-07.md`): **prefer unblocked kernels through n≈256** by
calling the low-level factor APIs with explicit parameters. The
high-level solvers (`.partial_piv_lu()`, `.qr()`) use the untuned
defaults, so hot paths should call:

```rust
use faer::linalg::lu::partial_pivoting::factor as lu;
use faer::linalg::qr::no_pivoting::factor as qr;
use faer::{Auto, Spec};

// LU: stay on the unblocked kernel (1.25–1.5× native, vs 2.3–9.6× default)
let params = lu::PartialPivLuParams {
    recursion_threshold: n, // measured win up to n = 256
    ..Auto::<f64>::auto()
};
lu::lu_in_place(a, &mut perm, &mut perm_inv, par, stack, Spec::new(params));

// QR, n ≥ 64: classical Householder, panel width 1 (≈ native speed,
// vs ~8–10× slower with the default panel width)
let mut q_coeff = Mat::<f64>::zeros(1, n); // block size = row count = 1
qr::qr_in_place(a, q_coeff.as_mut(), par, stack, Spec::new(Auto::<f64>::auto()));
```

Unmeasured beyond n=256 — blocked paths must win eventually; re-run
`bench/tune.mjs` before extrapolating. SVD/EVD carry related (smaller)
overheads via their internal bidiag/tridiag stages; untuned for now.

## 8. Schur decomposition + eigenvalue reordering (`faer-schur`)

faer computes the Schur form internally (its EVD pipeline) but does not
expose it. The companion crate `schur/` in this repo provides it over
faer's public modules — `gees`/`trexc`/`trsen`-shaped, `no_std`, builds
for wasm32 (it's in the `full` smoke variant, gated in CI):

```rust
use faer_schur::real::{real_schur, real_schur_select, real_schur_move};
use faer_schur::complex::complex_schur;

let s = real_schur(a.as_ref(), Par::Seq)?;      // A = Z T Zᵀ (T quasi-triangular)
// reorder: eigenvalues with Re λ > 0 into the leading m×m block
let select: Vec<bool> = (0..n).map(|k| s.w_re[k] > 0.0).collect();
let (mut t, mut z) = (s.t, s.z);
let m = real_schur_select(t.as_mut(), Some(z.as_mut()), &select)?;
```

In-place, allocation-free variants (`real_schur_in_place` +
`real_schur_scratch`, ditto complex) take a `MemStack`. Requires
`patches/0002-expose-schur-kernels.patch` on the pinned faer checkout —
a 6-line visibility-only patch (no behavior change), dropped the moment
upstream exposes the kernels. Accuracy is tested in CI
(`schur/tests/accuracy.rs`): backward error ‖A − ZTZᵀ‖ ~1e-15,
orthogonality, quasi-triangular structure, eigenvalue agreement with
faer's EVD, and reorder invariants, at sizes through n=150 (covering
both the unblocked and the blocked/AED paths). Reminder from §4: the
c64 driver is correct on plain simd128 but broken under relaxed-SIMD at
this pin.
