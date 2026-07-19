# Engineer session notes (carry-over, 2026-07-19)

Working knowledge NOT in the other docs. Plan of record: ROADMAP "BLAS
campaign sequencing" (tuning in progress → f32/c64 → LAPACK). Campaign
evidence log: docs/blas-ab-2026-07.md steps 1–6. Scoreboard: STATUS §3.

## Git / session mechanics
- All work lives on `claude/repo-familiarization-rbsmi2`; main is at
  b54603b (PR #1 merge). The git proxy 403s direct pushes to main —
  merge via PR when Andy asks. Everything through commit 0d480b7
  (4-accumulator reductions) is pushed.
- Commit as author `andy-emerson <emerson.andrew@gmail.com>` with
  Co-Authored-By + Claude-Session trailers. Run `git add` from the
  repo ROOT — after `cd blas`/`cd bench`, relative pathspecs fail.
- The session token CANNOT edit `.github/workflows/*` (no workflow
  scope). Open item for Andy: add `cd blas && cargo test --release`
  gate line to CI.

## Runner measurement procedure (used for every verdict)
1. Insert a TEMPORARY routing block in `bench/pyodide-vs-faer.mjs`
   just before the `const SIZES = [64, 128, 256, 512];` anchor.
   Since the 2026-07-19 restructure the blas rooflines live in
   blas/bench (own wasm, builds in seconds): execSync
   `cd ../blas/bench && cargo build --release --target
   wasm32-unknown-unknown --lib`, then per level `cargo run --release
   --bin native lN-bits[-f32|-z|-c] > /tmp/bits` and `node
   lN-roofline.mjs target/.../blas_bench.wasm /tmp/bits
   [--f32|--c64|--c32]` (cwd blas/bench), then `process.exit(0)`. Commit as
   "TEMPORARY: ... (revert after draws)". NOTE: the implementation
   being measured must ALSO be committed/pushed before dispatch —
   the runner clones the branch head.
2. Dispatch 2× `actions_run_trigger` on `pyodide-bench.yml`, ref the
   branch. `git revert --no-edit <routing-sha>` immediately (queued
   runs keep their head SHA).
3. Poll: unauthenticated curl gets rate-limited ("?" results) — use
   the MCP `actions_list` (result overflows: python-json the saved
   file for id/status/head_sha). Then `list_workflow_jobs` for the
   job id, `get_job_logs` with `tail_lines` 60–105 (the table sits
   above ~45 lines of upload/cleanup noise).
- Verdict rules: ≥2 runner draws, same-machine interleaved variants,
  sub-1.3× margins are direction-only unless unanimous.

## blas crate invariants
- `#[target_feature(enable = "simd128")]` on the WHOLE call chain or
  intrinsics compile out-of-line (measured 6.4×; simd128 is NOT a
  default wasm32 feature). Pattern: safe pub wrapper → cfg_attr'd
  unsafe `imp` → lanes.rs methods (also annotated).
- lanes.rs native fallback emulates lane structure elementwise →
  reductions bit-identical native↔wasm BY CONSTRUCTION. Verified via
  `run_l{1,2,3}_probe` vs `native l{1,2,3}-bits` in the roofline
  scripts (4 + 8 + 9 probes, all green on container + every draw).
- Tuned variants are kept bit-identical to the plain shape by
  preserving the per-element rounding sequence (β first, then one α·b
  rounding + one mul-add per k, k ascending) — that's what makes
  dispatch invisible and testable with assert_eq on bits.
- Levels 2/3 are literal loops of Level 1/2 calls on column slices —
  one stream implementation per op (map: blas/src/README.md). Layout
  since the 2026-07-19 restructure: netlib naming, one file per
  routine per type (daxpy.rs/saxpy.rs...) under src/L{1,2,3}; tuned
  kernels (d/saxpy4, d/saxpy4in, d/saxpy_dot(4)) in src/kernels.rs;
  tests mirror it under tests/L{1,2,3}/ with main.rs+common.rs per
  level; the live scoreboard is blas/bench/README.md; the
  bit-identity results and expected probe patterns are
  blas/tests/README.md.

## bench harness map (bench/ = faer side; blas/bench = blas side)
- State: a, b (inputs), sym (SACRIFICIAL — triad destination and L2/L3
  mutation target), tri (a with diag = 2n+1, keeps solves bounded),
  rhs (n×1 scratch for L2 y / trmv/trsv x).
- Exports: run_l1_layer(0..8: copy,swap,scal,axpy,rot,dot,nrm2,asum,
  iamax), run_l2_layer(0..7: gemv,gemv_t,ger,symv,trmv,trsv,syr,syr2),
  run_l3_layer(0..7: gemm,symm_l,syrk,syr2k,trmm_l,trsm_l,trmm_r,
  trsm_r), run_l3_tuned_gemm (tiled), run_l3_col4_gemm, probes as
  above, run_ceiling_bw (pure triad v2), run_ceiling_flops(iters).
  c64 twins (state az/bz/symz/triz/rhsz): run_l1_layer_z(0..10:
  copy,swap,scal,dscal,axpy,rot,dotu,dotc,nrm2,asum,iamax),
  run_l2_layer_z(0..9: gemv,gemv_t,gemv_c,geru,gerc,hemv,trmv,trsv,
  her,her2), run_l3_layer_z(0..7: gemm,hemm_l,herk,her2k,trmm_l,
  trsm_l,trmm_r,trsm_r), probes run_l{1,2,3}_probe_z (5/10/9, native
  modes l{1,2,3}-bits-z); c64 L3 scores against the F64 flops
  ceiling with 4x FLOP accounting (complex MAC = 8 real FLOPs).
  c32 twins mirror all of it with _c suffixes over state
  ac/bc/symc/tric/rhsc (the c64 fills cast to f32); c32 L3 scores
  against the F32 flops ceiling (complex f32 MAC = 8 real f32
  FLOPs); roofline flag --c32.
- Scripts: blas/bench/l{1,2,3}-roofline.mjs (L1/L2 GB/s vs fastest
  same-run stream at n=2048; L3 GFLOP/s vs arithmetic peak at n=512;
  --f32 switches type) over the blas-bench wasm. bench/gemm-tune-ab.mjs
  (4-way faer race) takes TWO wasm paths since the restructure. Old
  race harnesses (run_blas_ab, run_l1_ab) still exported from
  bench-harness — run_blas_ab(4,0) = faer gemm reference. The layer
  rows/probes/ceilings themselves live in blas/bench/src/lib.rs.

## Tuning campaign state (task #22 — CLOSED 2026-07-19)
Six levers shipped, every verdict two runner draws (docs steps 6–9):
gemm dispatch (tiled4x4 ≤1.5 MB of A / col4), 4-accumulator
reductions (dot at the read ceiling), fan-out/fan-in blocked shapes
through gemv + all Level 3 (`blas/src/kernels.rs`), fused 4-column
symv (symm_left 84–86% of peak), blocked trmv/trsv. Fused iamax
raced and REFUTED (reverted, loss recorded in its module docs).
FMA variants DEFERRED — architect decision on the determinism
trade-off (relaxed-madd rounding is implementation-dependent); the
step-1 evidence and the deferral are recorded in ROADMAP. Next per
sequencing: clone the tuned layer into f32 and c64.

## Register (Andy)
Lead with the direct answer ("Faster."), plain adult language, no
jargon walls, lean tables with only requested columns, honest
uncertainty flagged inline. Architect decides scope; propose, then
execute on his go.
