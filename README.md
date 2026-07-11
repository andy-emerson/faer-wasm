# faer-wasm

Makes [faer](https://github.com/sarah-quinones/faer-rs) usable as a
first-class wasm32 dependency. **Thin-carry discipline**: we vendor the
smallest possible patch set, keep it clean against a pinned upstream base,
re-verify on every faer release, and drop patches the moment upstream
doesn't need them. Upstreaming is de-prioritized, not forbidden —
candidates are tracked in ROADMAP.md's upstream ledger for when the
project settles; new capability is built alongside faer, not inside it.
Empirical basis in docs/.

## Contents

- `ROADMAP.md` — the phased plan. Phases 0, 1, 3 (the carried fix, CI
  gate, consumer recipe, size budgets, benchmarks + tuning) are done;
  Phase 2 (the LAPACK-parity tail) is underway — Schur + reordering
  landed first.
- `patches/` — the carried patch set (base commits in
  `patches/UPSTREAM-BASE.txt`), split by target repo:
  `faer-rs/0001-fix-32bit-usize-shift.patch`, the 4-line fix that makes
  faer build on wasm32 (and armv7/i686): `(n >> 32)` on 32-bit `usize` →
  `((n as u64) >> 32) as u32`, in `operator/{eigen,self_adjoint_eigen,svd}`;
  `faer-rs/0002-expose-schur-kernels.patch`, 6 visibility-only lines
  (`pub(crate)` → `pub`, no behavior change) exposing the Schur kernels
  faer already has, so `schur/` can drive them from outside; and
  `pulp/0003-fix-relaxed-simd-complex-mul-argument-order.patch`, a 4-line
  correctness fix: pulp's wasm RelaxedSimd backend passed NEON
  accumulator-first FMA arguments to the accumulator-last `relaxed_madd`
  in its complex `mul_add_e`/`mul_e` kernels, making all c64 compute
  wrong under `+relaxed-simd` (docs/wasm.md §4); and
  `faer-rs/0004-fix-no_std-deflation-window-log2.patch`, a 1-line fix to
  the `no_std` branch of the Schur AED deflation-window default, which
  computed `log2(n/n)` = 0 instead of `n/log2(n)` — on every `no_std`
  build (i.e. all typical wasm builds) the aggressive-early-deflation
  window degenerated to 2 for 150 ≤ n < 590, exploding eigensolver
  iteration counts ~50-85× (n=512: 1091 AED calls/852 sweeps → 26/22
  after the fix; docs/research-eig-wasm-2026-07.md).
- `schur/` — **`faer-schur`**, the first Phase 2 companion crate:
  real + complex Schur decomposition (`gees`-shaped) and eigenvalue
  reordering (`trexc`/`trsen`-shaped) over faer's public API. `no_std`,
  wasm-gated in CI, accuracy-tested against backward error /
  orthogonality / faer's own EVD (`schur/tests/accuracy.rs`).
- `smoke-test/` — a zero-import `no_std` consumer crate that builds to
  `wasm32-unknown-unknown` (with the patch applied to a local faer checkout)
  and runs matmul / LU / QR / SVD / EVD under node, verified bit-identical
  to native x86-64.
- `upstream/` — **archived, deferred**: a complete contribution (fix +
  regression tests + wasm CI job, as `git am`-able patches with PR/issue
  text) prepared 2026-07 and held back (upstreaming is de-prioritized
  until the project settles — see ROADMAP.md's upstream ledger). Kept
  because the patches double as our own regression tests and as the
  submission template; don't extend it.
- `docs/wasm.md` — **the consumer recipe**: cargo setup, features that
  work (and `rayon`, which doesn't), the `no_std` zero-import pattern,
  sizes + budgets, the relaxed-SIMD (FMA) route, determinism guarantee.
- `docs/benchmarks-vs-pyodide-2026-07.md` — the external head-to-head:
  faer-wasm vs Pyodide's scipy/numpy in the same V8 (manual
  `pyodide-bench` workflow). Chronological run log (Run 1: "Pyodide wins
  most of this suite", geomean 0.41×) with a current-status note on top;
  the wasm-shaped kernels since flipped matmul/QR/LU-solve/eigvals to
  wins — current tables live in `docs/research-eig-wasm-2026-07.md`.
- `bench/` + `docs/benchmarks-2026-07.md` — the wasm-vs-native benchmark
  harness (f64 + c64 ops) and its first published numbers (opt-level
  ~1.75×, relaxed-SIMD ~11%, large matmul at 1.8–1.9× native, mid-size
  blocking cliffs identified), plus `gate.mjs`: the CI efficiency gate
  (ratio bands, scaling windows, tuned-vs-default guards).
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
| `.wasm` sizes 59→922 KB, within budgets (c64 stacks dominate; real-only ~500 KB) | tested | CI-enforced | `size-budgets.json` |
| relaxed-SIMD emits real FMA (`relaxed_madd`) | observed | by-hand | 2026-07 disassembly counts |
| `rayon` cannot build on wasm32 | observed | by-hand | build probe at the pin |
| perf: matmul 1.9× native; opt-level z→3 ≈ 1.75×; relaxed-SIMD ≈ +11% | observed | scripted | `bench/`, min-of-2 reps, shared box |
| tuning: unblocked kernels win ≤ n=256 (LU 1.25–1.5× native, QR ≈ 0.9× native-default) | observed | scripted | `bench/tune.mjs` sweep |
| Schur (f64+c64) + reordering correct: backward error ~1e-15, orthogonality, eigenvalues match faer EVD, reorder invariants; n ≤ 150 incl. blocked/AED path | tested | CI-enforced | `schur/tests/accuracy.rs` + wasm property probes in all full variants |
| pulp relaxed-simd c64 bug (transposed FMA args in `mul_add_e`/`mul_e`) root-caused and fixed by carried `patches/pulp/0003` | tested | CI-enforced | isolated stage-by-stage 2026-07-08; `schur_probe_cplx` == 3 required on BOTH full variants guards the fix |
| shelved upstream patches recreate the branch byte-identically | observed | by-hand | `git am` round-trip, 2026-07-08 |
| blocked/multishift paths win beyond measured crossovers (LU: recursion NEVER wins to n=1024; eig: multishift from n=512, crossover ≈ 480) | observed | scripted | `lu-largen.mjs`, crossover grid run 29134291933 |
| foundation ops (LU/QR/LLT/SVD/EVD/eigenvalues) correct at n=33 (SIMD tails) + n=96 (blocked paths), f64 AND c64 | tested | CI-enforced | `dense_f64_probe`=26 / `dense_c64_probe`=24, identical on native, node, and Chrome, all full variants |
| runs in real browsers (headless Chrome, incl. relaxed-SIMD variant) | tested | CI-enforced | `browser-check.mjs` (raw CDP), exact values, every push |
| no cliff-class perf regressions: op/matmul ratios (×3 band), O(n³) scaling windows, tuned-kernels-still-win guards | tested | CI-enforced | `bench/gate.mjs` vs `bench/expected-ratios.json`; bands sized to the 3–10× cliffs actually observed |
| ops scale at their complexity class: fitted exponents in [1.8, 3.2], no step jumps > 4×(n₂/n₁)³ | tested | CI-enforced | `bench/complexity.mjs --gate` per push; full sweep tables in benchmarks doc |
| the 2026-07-09 "blocked Schur/EVD loses 2–13×" finding was upstream bug 0004 for n ≥ 150, not tuning; post-fix `faer-schur` routes per-n at the measured 480 crossover (`recommended_params(n)`) | tested | CI-enforced | root-caused via iteration counters (`run_eigvals_counters`); patch applied by every gate run; crossover re-validated on 3 runner instances |
| vs Pyodide, dated snapshot (Runs 1–4, 2026-07-09, pre-kernels): matmul 4–20×, tuned QR 1.3–1.7×, eigensolvers then losing 2.5–3×, geomean 0.57× — superseded by the kernel-era rows below | observed | scripted | Actions runs 1–4, `docs/benchmarks-vs-pyodide-2026-07.md` |
| `faer-wasm-kernels` LU (blocked + recursive) factors correctly: ‖PA−LU‖/solve gates, identical pivots between the two drivers, agreement with faer | tested | CI-enforced | `kernels/tests/lu.rs` across sizes 1–512 × block/crossover settings, per push |
| the wasm LU default is a lean flat simd128 panel (recursion off): fastest wasm LU measured, beats scipy at n ≤ 128 (1.1–1.5×), ~parity at 256, ~0.8× at 512. A large-n probe (to n=1024) proved recursion never wins on the runner, so it's disabled by default | observed | scripted | Runs 5–7; `lu-tune.yml` + `lu-largen.mjs` on the runner; gated (`gate.mjs`) |
| full Ax=b via the kernels (flat factor + our substitution) beats scipy's `np.linalg.solve` 1.4–1.7× at every size n=64–512 | observed | scripted | Run 8 (29061871875); solve kernel correctness-gated in `kernels/tests/lu.rs`, efficiency-gated in `gate.mjs` |
| Pyodide's scipy links OpenBLAS 0.3.28 built as generic C (`RISCV64_GENERIC`), no arch microkernels | tested | scripted | `show_config()` printed by every `pyodide-bench` run since Run 4 |
| Pyodide's QR is *doubly* un-optimized: OpenBLAS ships no QR routines, so scipy runs reference-netlib `dgeqrf` over generic-C BLAS — hence `qr_r_tuned` wins 1.3–1.7× structurally | proven | by-hand | `docs/research-qr-wasm-2026-07.md` (verified against LAPACK/OpenBLAS source; 3-vote panel pending credit reset) |
| wasm-shaped unblocked QR kernel (`kernels/src/qr.rs`) beats scipy 2.5–3.0× at every size n=64–512, and faer's own `block_size=1` path 2–3.5× | observed | scripted | Run 6 (29048492853) same-run three-way; correctness gated in `kernels/tests/qr.rs` (‖A−QR‖, ‖QᵀQ−I‖, |R| vs faer), efficiency gated in `gate.mjs` |
| recursive QR is contraindicated on wasm (ReLAPACK excludes it; `dgeqrt3` recurses to skinny gemms); the predicted Hessenberg lever was since built (as the flat unblocked kernel, not block-apply) | proven | by-hand | `docs/research-qr-wasm-2026-07.md`; outcome in `research-eig-wasm-2026-07.md` |

| upstream bug 0004 (`no_std` AED window = `log2(n/n)` = 0, 150 ≤ n < 590) root-caused and fixed by carried patch: iteration counts collapse ~50–85× (n=512: 1091 AED/852 sweeps → 26/22) | tested | CI-enforced | `docs/research-eig-wasm-2026-07.md`; counters measured on runner pre/post; patch applied + exact-value gates on every push |
| eigvals kernel pipeline (`eigvals_k3`: flat Hessenberg + hqr below 480, repaired multishift above) beats scipy 1.75×/2.05×/1.83×/1.24× at n=64/128/256/1024 with separated ranges; n=512 is parity (separated 1 of 3 runs) | observed | scripted | replication gate in `pyodide-vs-faer.mjs` (5 alternating rounds, WIN only on range separation), runs 29137919745 + 29140693223 |
| Hessenberg + hqr kernels correct: ‖AQ−QH‖/‖QᵀQ−I‖/eigenvalue-preservation, and hqr eigenvalues match faer at n=1–256 (+ conjugate-pair/trace invariants) | tested | CI-enforced | `kernels/tests/{hessenberg,schur_small}.rs`, per push |
| Hessenberg kernel is 3.2×/7.0× faster than faer's blocked reduction at n=512/1024, and faer's blocked path has a machine-sensitive cache cliff (7–95× across runner instances at n=1024) that the kernel avoids | observed | scripted | phase-split probe run 29136868733; cliff cross-checked on 3 machines |
| full dense SVD has no >1.5× wasm win: threshold knob refuted by sweep (default optimal both directions), Jacobi killed by measurement (12–15 sweeps → 0.1–0.2×), all algorithm replacements refuted or author-conceded; 0.5–0.8× ≈ ceiling | tested | cross-checked | runner sweeps 29070389762/29065493103 + 103-agent adversarial verification (23/25 claims, 3-vote), `docs/research-svd-wasm-2026-07.md` |
| kernels are generic over `WasmScalar` (f64x2/f32x4); f64 behavior unchanged by the refactor; f32 correctness gated at eps32 tolerances | tested | CI-enforced | exact-value smoke probes unchanged; `kernels/tests/f32.rs` per push |
| f32 column vs scipy float32: matmul 4.3–9.1× (n ≥ 128; 0.5× at 64 — small-n gemm overhead), LU-solve 2.4–3.0×, QR 3.7–5.1×, eigvals 2.0–4.3× — scipy's s-routines are ≈ no faster than its d-routines on wasm | observed | scripted | run 29140693223, both sides single precision |

## Quick start

The smoke test path-depends on **both** upstream clones sitting in the repo
root (gitignored): `faer-rs/` and `pulp/`. Commits are pinned in
`patches/UPSTREAM-BASE.txt`.

    git clone https://github.com/sarah-quinones/faer-rs
    git clone https://github.com/sarah-quinones/pulp
    git -C faer-rs checkout <faer commit in patches/UPSTREAM-BASE.txt>
    git -C pulp    checkout <pulp commit in patches/UPSTREAM-BASE.txt>
    for p in patches/faer-rs/*.patch; do git -C faer-rs apply "../$p"; done
    for p in patches/pulp/*.patch;    do git -C pulp    apply "../$p"; done
    cd smoke-test && cargo build --lib --target wasm32-unknown-unknown --release --features full
    node check.mjs   # exact-value + size gate; run.mjs just prints
    # (--lib: the `native` bin is host-only, for the determinism cross-check)

Consumer-facing build recipe (features, sizes, SIMD, determinism):
`docs/wasm.md`.
