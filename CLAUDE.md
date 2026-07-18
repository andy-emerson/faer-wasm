# Instructions for Claude

This repo makes **faer** (pure-Rust linear algebra) usable as a first-class
`wasm32` dependency. It exists to serve a consumer (Ruju, a Rust
reimplementation of Julia for WebAssembly), but everything here is scoped
**language-agnostically** — work only on things any wasm-targeting consumer
of faer would want.

**Prime directive — thin carry.** Upstreaming is **de-prioritized, not
forbidden** (Andy, 2026-07-11, revising the 2026-07-08 "nothing goes
upstream"): do not prepare upstream PRs, issues, or contribution material
now, but **record every upstream-worthy finding** in ROADMAP.md's
upstream ledger as it is discovered; when the project settles toward
completeness, the architect decides what moves back upstream. Meanwhile
we *carry* the minimum: vendor the smallest possible patch set in
`patches/`, keep it `git am`-clean against the pinned base, and re-verify
on every faer release. If upstream ever fixes 32-bit builds
independently, drop the patch. New capability (Schur, Sylvester, …) is
built **alongside** faer — in companion crates or the consumer's shim
over faer's public API — never as patches to faer itself unless there is
no other way.

**Identity — same problem, new environment (Andy, 2026-07-11).** We are
not copy-catting. LAPACK defines the *destination* (operation coverage,
semantics, accuracy); faer is the *point of origin* (the Rust code that
gets us onto wasm); the implementation may legitimately end up different
from both. Optimize **correctness and measured performance on the
target** — never fidelity to how either ancestor does things. Keep an
ancestor's convention only when it earns its place: interop contracts
(LAPACK `ipiv`/`dgeqrf` storage), proven numerics (convergence
machinery), or a shape that measurement shows compiles well on wasm.
When measurement disagrees with LAPACK's structure, faer's defaults, or
published research, measurement wins — the record already shows all
three happening (LU recursion disabled, faer routing overridden, the
SVD `recursion_threshold` recommendation refuted by sweep).

**End state and success metric (Andy, 2026-07-18; full record in
ROADMAP "Re-derived goals").** The end state is a self-contained,
wasm-native library — faer is scaffolding, retired one measured
campaign at a time, never removed as a motive in itself. **Success is
distance from the machine's ceiling** (streaming ops vs measured
memory bandwidth, multiply-class ops vs measured peak arithmetic);
beating scipy is the market comparison, not the goal. Structure:
the BLAS layer is being built first as its own finished product, the
LAPACK-layer kernels re-route onto it. No shared-memory threading
(COOP/COEP excluded by the architect); WebGPU-f32 and consumer-layer
batch Workers stay on the board.

Start by reading `README.md`, then `ROADMAP.md` (the phased plan — the
architect picks which phase to work; see the contract below), then
`docs/research-faer-wasm-2026-07.md` (the
empirical evidence behind the plan: measured sizes, pulp simd128 status,
the LinearAlgebra coverage matrix).

## Working contract (adopted 2026-07-08; modeled on Lua2D's AGENTS.md)

**Roles.** Andy is the **architect**: he decides *what* and *why* — scope,
priorities, trade-offs (e.g. upstreaming deferred to project maturity;
releases adopted on our terms). A session is the **engineer**: it
proposes *how*, executes after agreement, and must not cut the architect
out of decisions.

**Every pass, three steps.** (1) Report current state honestly,
uncertainties and failures included — the ROADMAP and the status artifact
are the medium. (2) The architect picks the next milestone; the engineer
may propose. (3) Agree the plan, then execute.

**Deviation routing.** Discoveries mid-milestone: unrelated → record in
ROADMAP and move on; blocking the milestone (or a small measurable win on
its path) → handle now; architectural → stop and ask the architect.

**Race the foundation (Andy session, 2026-07-18).** Measurement-
before-rewrite has a blind spot: candidates get raced against
incumbents, but incumbents never get raced against a null hypothesis
(faer's gemm lost 15–20% for two weeks as the presumed floor). Carried
dependencies are raced periodically like everything else.

**Claim grading.** Every claim in README/docs carries evidence graded on
two independent axes: **strength** (stated < built < observed < tested <
proven) and **durability** (by-hand < scripted < CI-enforced <
cross-checked). Never grade a claim above its evidence, on either axis.
The evidence grid in `README.md` is the ledger — update it whenever a
claim is added or its evidence changes tier.

**Phase close.** Every phase ends with a truth-seeking **doc pass** —
docs checked against evidence, the opposite orientation from goal-seeking
code review. (The 2026-07-08 sweep is the model: it caught a tautological
test and six stale claims.)

**Coverage rule (Andy, 2026-07-11: "why is there no consistency in our
testing and benchmarking?").** Every new kernel gets the full treatment
before its campaign closes: correctness test AND benchmark row, in every
number type it supports — or an explicit gap line in `STATUS.md` saying
why not. `STATUS.md` is the one-page plain-English scoreboard (what we
changed in faer, what we ship, how good it is); update it in the same
commit as the change it describes, in plain language — it is written for
the architect to keep the whole project in his head, not for engineers.

## Working setup

- The upstream clones live at `faer-rs/` and `pulp/` in the repo root
  (gitignored — never commit them). Pin them to the commits in
  `patches/UPSTREAM-BASE.txt`; apply `patches/faer-rs/*.patch` to
  `faer-rs/` (0001: the 4-line 32-bit fix; 0002: 6 visibility-only lines
  exposing the Schur kernels; 0004: 1-line fix for the no_std
  AED-deflation-window `log2(n/n)`=0 bug that exploded eigensolver
  iterations ~50-85× on wasm) and `patches/pulp/*.patch` to `pulp/`
  (0003: 4-line fix for the relaxed-simd complex-mul argument-order bug
  — this one is a *correctness* fix, not behavior-neutral). All dropped
  when upstream makes them unnecessary.
- `schur/` is the first Phase 2 companion crate (`faer-schur`: Schur +
  eigenvalue reordering over faer's public API). Its gate:
  `cd schur && cargo test --release` (CI runs it too).
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
verification — treat any cross-target difference as a bug, not noise
for the fixed-value probes (larger eigen-pipelines are property-scored
instead; docs/wasm.md §5). CI (`.github/workflows/wasm-gate.yml`) runs
the gate across all four build variants with size budgets from
`smoke-test/size-budgets.json`, plus: faer-schur accuracy tests, dense
f64+c64 foundation probes, the same probes in headless Chrome
(`browser-check.mjs`), and the bench efficiency gate (`bench/gate.mjs`).
Current sizes are tabulated in `docs/wasm.md` §3.

## Upstream policy

**Deferred by decision, not oversight** (revised 2026-07-11:
de-prioritized, no longer forbidden). A complete Phase 0 contribution
(fix + regression tests + wasm CI job) was prepared and archived under
`upstream/`. Leave it archived and don't extend it for now — it becomes
the submission template when the upstreaming window opens. Candidates
accumulate in ROADMAP.md's **upstream ledger**; the architect decides
when the project has settled enough to start sending things back.

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
