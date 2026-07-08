# Instructions for Claude

This repo makes **faer** (pure-Rust linear algebra) usable as a first-class
`wasm32` dependency. It exists to serve a consumer (Ruju, a Rust
reimplementation of Julia for WebAssembly), but everything here is scoped
**language-agnostically** — work only on things any wasm-targeting consumer
of faer would want.

**Prime directive — thin carry.** Andy decided (2026-07-08) NOT to submit
anything upstream — do not prepare upstream PRs, issues, or contribution
material, and do not ask him to file anything. Instead we *carry* the
minimum: vendor the smallest possible patch set in `patches/`, keep it
`git am`-clean against the pinned base, and re-verify on every faer
release. If upstream ever fixes 32-bit builds independently, drop the
patch. New capability (Schur, Sylvester, …) is built **alongside** faer —
in companion crates or the consumer's shim over faer's public API — never
as patches to faer itself unless there is no other way.

Start by reading `README.md`, then `ROADMAP.md` (the phased plan — work the
lowest unfinished phase), then `docs/research-faer-wasm-2026-07.md` (the
empirical evidence behind the plan: measured sizes, pulp simd128 status,
the LinearAlgebra coverage matrix).

## Working setup

- The upstream clones live at `faer-rs/` and `pulp/` in the repo root
  (gitignored — never commit them). Pin them to the commits in
  `patches/UPSTREAM-BASE.txt` and apply `patches/*.patch` to `faer-rs/`.
- Install the target if missing: `rustup target add wasm32-unknown-unknown`.

## Verification (the gate for any change)

```sh
cd smoke-test
cargo build --lib --target wasm32-unknown-unknown --release --features full
node check.mjs   # exact values (matmul / LU / QR / SVD / EVD) + size budget
```

(`--lib` matters: the `native` bin is host-only — building it for wasm is
a duplicate-`panic_impl` error. Determinism cross-check:
`cargo run --release --features full --bin native > native-bits.txt &&
node determinism.mjs <wasm> native-bits.txt`.)

Results were bit-identical between native x86-64 and wasm in the 2026-07
verification — treat any cross-target difference as a bug, not noise. CI
(`.github/workflows/wasm-gate.yml`) runs the same gate across all four
build variants with size budgets from `smoke-test/size-budgets.json`;
current sizes are tabulated in `docs/wasm.md` §3.

## Upstream policy

**Shelved by decision, not oversight.** A complete Phase 0 contribution
(fix + regression tests + wasm CI job) was prepared and archived under
`upstream/`; Andy chose not to submit it. Leave it archived, don't extend
it, and don't revisit the decision unless he raises it.

**Release policy (Andy, 2026-07-08): upstream is a resource, not an
obligation.** Evaluate each faer release; adopt it (re-pin, re-apply
`patches/`, re-run the gate) only when it advances our cause. Slight
accommodations to upstream changes are fine. If upstream's direction
deviates from our needs, do NOT follow it — stay on the pinned base; the
pin means nothing upstream can break us without our consent. If an
adopted release builds on 32-bit without our patch, delete the patch and
note it in ROADMAP.md.

## Commits

Author every commit as `andy-emerson <emerson.andrew@gmail.com>`, credit
yourself with a `Co-Authored-By` trailer, develop on `main`, and verify
local and remote are in sync after pushing.
