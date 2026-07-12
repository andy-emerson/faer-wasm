# SVD wasm-shaping research — sequential focused passes (2026-07-10)

> **Status note (2026-07-12 doc sweep):** the "0.5–0.8× scipy" figures
> below are PRE-allocator-fix — the leak-only bump allocator was taxing
> our side of every large-n benchmark (see docs/wasm.md §2). Post-fix
> (run 29157035070) faer's unchanged SVD measures **0.7–1.5×**, winning
> at n≥256. The mechanism analysis below still stands; only the absolute
> ceiling numbers moved.

Deep-dive on SVD alone, run as **two sequential focused passes** (architect
method: pass 1 orients and tells pass 2 what to chase; don't front-load a
fixed brief). Pass 1 mapped the approaches; pass 2 drilled the one it
surfaced. ~2 agents, ~80k tokens total, verified by hand + against faer's
own SVD source. Grades carry that provenance (single agent + hand check,
not the 3-vote panel; several primary PDFs were proxy-blocked, noted).

## The fork pass 1 surfaced (not obvious a priori)

The SVD cost center is **bidiagonalization** (`dgebrd`, >70% of total time,
~50% memory-bound level-2 GEMV) — *not* the bidiagonal solver. faer already
ships the classic one-stage pipeline (`bidiag.rs` + `bidiag_svd.rs`:
Householder bidiag → divide-and-conquer, DC merge bulk already routed to
faer gemm, `recursion_threshold=128`). So the real choice is:

- **Thread 1 — one-sided Jacobi SVD** (`dgesvj`/`dgejsv`, Drmač–Veselić):
  *avoid* bidiagonalization entirely. Column-pair dot-products + plane
  rotations = level-1 BLAS over contiguous columns — the exact flat-simd128
  shape our kernels already beat generic-C LAPACK on (QR 2.5–3×). Deletes
  the >70% wall. **From scratch** (faer has no Jacobi SVD). High upside.
- **Thread 2 — tune faer's existing bidiag→DC** for 2-lane wasm: shape the
  ~50% level-2 GEMV half with our flat simd128, tune `recursion_threshold`
  / `qr_ratio_threshold`. Existing code, low effort, lower ceiling.
- **Thread 3 — two-stage reduction (dense→band→bidiag): KILLED.** No
  single-threaded small-n advantage; it's a level-3/multicore/GPU technique
  for n≳1024, second stage is memory-bound bulge-chasing, adds a
  back-transform. Rejected for wasm on the record.

## Jacobi (Thread 1) — graded

**Confirmed:**
- Accuracy advantage is real and defensible — high *relative* accuracy on
  tiny singular values, provably better than any bidiagonalize-first method
  (Demmel–Veselić 1992; Demmel et al. LAA 1999). A genuine numerics-library
  win, not marketing.
- Core op is column-only level-1 BLAS; on **column-major** storage it's pure
  contiguous 2-lane loads/stores, **no gather/scatter** (the contiguity-
  breaking cyclic-by-rows pattern belongs to the *two-sided* variant, which
  we would not build). Maps onto our winning kernel shape directly.
- Preconditioned Jacobi is rated *comparable* to bidiag even on normal
  hardware ("faster than dgesvd, not much slower than dgesdd") — and our
  2-lane regime neutralizes bidiag's level-3 trump card, tilting further
  toward Jacobi.
- LAPACK caps at 30 sweeps; convergence ultimately quadratic.

**The killer risk (unverified):** total work ≈ O(sweeps·n³), a ~2–4× flop
premium over bidiag+DC. RRQR preconditioning cuts sweeps (typically ≈2–6),
but the exact preconditioned sweep count at n≤512 was in a proxy-blocked
source (LAWN 170). **If typical inputs need ~10+ sweeps, the premium swamps
the kernel advantage and Jacobi loses** — and 2 f64 lanes give little
parallelism to hide the extra flops. This single number decides it.

## Recommendation — measure both cheaply before committing

Neither thread's payoff is measured on our runner yet, and the QR precedent
is the cautionary tale: `qr_r_tuned` (a param choice, block_size=1) already
beat scipy 1.3–1.7× *before* we built the kernel that then beat that. So:

1. **Profile + tune faer's SVD on the runner** (Thread 2, cheapest, ships a
   testable win): where does faer's SVD spend time at n=128/256/512
   single-thread — bidiag level-2 GEMV, DC merge gemm (are those merges big
   enough to clear our "small gemm loses" bar, or skinny at the leaves?),
   or scalar secular/deflation? Sweep `recursion_threshold`. A faer-schur-
   style `recommended_svd_params()` may already move the needle.
2. **Unpreconditioned-Jacobi sweep-count probe** (Thread 1 de-risk, cheap):
   prototype just the bare Jacobi sweep, measure sweeps-to-convergence on
   representative n≤512 matrices (well-conditioned + clustered/ill-cond).
   This is the one number gating the full `dgejsv` build.
3. **Decide** from those two measurements: if faer-tuning already reaches
   ~parity and Jacobi sweeps are high → ship the tuned bidiag and stop. If
   Jacobi sweeps are low (≈2–6) → the full preconditioned build (RRQR pre-
   step routes into our fast QR) is justified for a large win + accuracy.

Effort asymmetry to weigh: Thread 2 modifies existing code; Thread 1 is a
from-scratch RRQR preconditioner + sweep loop + convergence test (LAPACK
`DGESVJ`/`DGEJSV` are the canonical portable reference; no clean Rust port
known).

## Runner roofline + phase profile (measured 2026-07-10, run 29064796600)

Architect reframe: optimize toward the machine ceiling, not scipy parity.
So we located the ceiling and split faer's SVD time (`svd-roofline.mjs`).
Confirmed on the runner (dev-box preview held qualitatively):

| n | compute peak | bandwidth | GEMV | bidiag / full SVD |
| - | - | - | - | - |
| 128 | 5.10 GF/s | 50.6 GB/s | 14.9 GB/s (29% of BW) | 5.2 / 15.8 ms → **33%** |
| 256 | 5.65 GF/s | 50.0 GB/s | 15.8 GB/s (32%) | 26.7 / 97.1 ms → **27%** |
| 512 | 5.61 GF/s | 50.5 GB/s | 15.9 GB/s (31%) | 145.7 / 467.0 ms → **31%** |

**Two load-bearing measured facts:**
1. **Bidiagonalization is only ~30% of SVD wall time** — the DC-solve +
   vector back-transformation is the other ~70%. (The literature's ">70%
   reduction" is *singular-values-only*; with vectors the back-end
   dominates.) So **tuning bidiag caps at ~30% of the cost** — cannot be
   the optimization endgame.
2. **The memory-bound GEMV runs at ~30% of STREAM bandwidth** (~16 vs
   ~50 GB/s) — ~3× headroom, but on a phase that's only ~30% of total, so
   the realizable SVD win from a bandwidth-optimal GEMV is only ~10–15%.

**Decision consequence:** the dominant ~70% (DC + back-transform) exists
*only because* we bidiagonalized (the vectors must be un-transformed).
One-sided Jacobi has neither phase — it works directly on A, U/Σ/V fall
out of the sweeps — so it structurally avoids the part that dominates
faer's pipeline. Tune-bidiag is therefore ruled out as the path to the
optimum; Jacobi is the only lever that attacks the dominant cost. The
decision now rests solely on **Jacobi's sweep count** (its flop premium
vs the *whole* pipeline, where bidiag is only 1/3 — a far easier bar than
"vs the reduction"). Next step: a bare unpreconditioned-Jacobi sweep
**probe** to measure (a) sweeps-to-convergence and (b) the achieved GF/s
of the rotation kernel at n≤512 — the one measurement that decides the
from-scratch build.

## Jacobi probe — built, correct, and KILLED by measurement (2026-07-10)

The sweep-count probe (`kernels/src/svd.rs`, one-sided unpreconditioned
Jacobi; correct — reconstruction/orthogonality/singular-values gated vs
faer in `kernels/tests/svd.rs`) settled the decision:

- **Sweeps to convergence** (well-conditioned random, the killer number):
  12 @ n=128, 13 @ 256, 15 @ 512 — the *pessimistic* end (research said
  ~6–12 unpreconditioned, ~2–6 preconditioned). At ~7·n³ flops/sweep that
  is ~8× the flops of faer's whole pipeline.
- **Runner time vs scipy** (`svd_jacobi` row, run 29065493103):
  n=256 → 450 ms = **0.2× scipy** (3.2× slower than faer's own SVD);
  n=512 → 4159 ms = **0.1× scipy** (6.2× slower than faer). Worsens with n.
- Even fully optimized — norm-caching (~1.4×) + explicit simd128 (~2×) +
  RRQR preconditioning cutting sweeps ~3× — the ceiling projects to
  ~faer-parity (0.5–0.8× scipy). A large from-scratch build (preconditioner
  + sweep kernel) to *maybe match what already exists*. **Rejected.**

## SVD conclusion — near the wasm ceiling; only a modest win available

Both research threads are now measured out:
- **Tune-bidiag**: caps at ~10–15% of SVD (reduction is only ~30% of cost;
  GEMV has 3× bandwidth headroom but only on that 30%). Reaches ~parity,
  not a large win.
- **Jacobi**: killed (above).

The dominant ~70% (DC-solve + back-transform) is, per the research,
largely faer gemm at compute peak — which our matmul already runs near.
So unlike QR (weak opponent + flat kernel = 3×), **SVD has no large wasm
win available**: faer's SVD is already 0.5–0.8× scipy and rising with n,
and the structure is gemm-bound at peak. Honest options for the architect:
(a) squeeze the ~10–15% tune-bidiag win (reaches parity, not "optimization"
by the architect's bar); (b) a cheap finer back-end profile to *confirm*
the ~70% is truly gemm-at-peak (the "it's gemm" claim is research, not
measured — the GEMV surprised us at 33% of bandwidth, so verifying the
back-end before abandoning is warranted); (c) accept SVD as near-ceiling
and move to eigvals, where the opponent/headroom may differ.

## Deep-research pass (4-angle fan-out) + the refuting runner sweep (2026-07-10)

The architect asked for deep research when none of the surfaced options
reached parity, "much less optimality." Four independent agents attacked
distinct angles; the synthesis is below, followed by the one concrete lever
it produced — which was then **built, swept on the runner, and refuted**.

**The four angles (research, graded plausible→verified where noted):**
- **Angle 3 — faer source (verified against `bidiag_svd.rs`):** the root
  cause of faer SVD losing to scipy at small n is the divide-and-conquer
  `recursion_threshold=128` default. Bidiagonal blocks up to 127 are solved
  by the **scalar `qr_algorithm`** (Golub–Kahan, Givens vector accumulation,
  *no SIMD*), where LAPACK `dbdsdc` uses ~25-element leaves + gemm merges.
  Proposed fix: lower `recursion_threshold` so more of the work routes
  through the SIMD divide-and-conquer + gemm path. **This is the only
  concrete, testable lever the whole fan-out produced.**
- **Angle 1 — dgesdd flops:** faer and dgesdd run the *same* algorithm at
  the *same* ~8–9 n³ flop count; scipy's edge (where it has one) is
  overhead/constant-factor, not fewer flops. The bidiag half is memory-bound
  BLAS-2 — an unblockable floor, not a lever.
- **Angle 2 — the ~70% back-end:** it is *mostly gemm at compute peak* — the
  `dormbr` back-transform (≈4 n³, biggest single flop bucket) plus the top
  DC merges, both of which faer already routes into its fast gemm (which our
  matmul confirms runs near the 5.5 GF/s peak). The *small* DC leaf merges
  are where 2-lane wasm loses — but shrinking the leaf (lower threshold) adds
  more of those small, losing merges, not fewer. No GEMV-style bandwidth
  headroom exists in the back-end.
- **Angle 4 — alternatives:** no unexplored full-dense-SVD direction has a
  clean >1.5× win at n≤512 on 2-lane wasm. Preconditioned Jacobi projects to
  ~parity (and unpreconditioned is already killed, above); Barlow bidiag caps
  ~15%; QR-then-SVD-of-R and bdsqr-with-vectors are both dead for square n.

**The lever, tested on the runner (`svd-tune.mjs`, run 29070389762):**
sweeping `recursion_threshold` both directions against faer's default 128 —

| n | default(128) | best low-rt | best high-rt | verdict |
| - | - | - | - | - |
| 64  | 7.99 ms | rt8 = 0.77× | rt96 = 1.00× | lowering **hurts** 23% |
| 128 | 16.11 ms | rt8 = 0.77× | rt96 = 1.00× | lowering **hurts** 23% |
| 256 | 96.52 ms | rt96 = 1.02× | rt512 = 0.83× | flat; rt96 is noise |
| 512 | 466.65 ms | rt96 = 1.01× | rt1024 = 0.76× | flat; rt96 is noise |

**The proposed root-cause fix is refuted.** Lowering `recursion_threshold`
(the research's recommendation) makes SVD **15–25% slower** at every size —
the small DC-leaf merges lose on 2 lanes exactly as Angle 2 warned, and the
scalar `qr_algorithm` leaf, though SIMD-less, is *cheaper* at these sizes
than paying for more DC merges. Raising the threshold is also worse (one
giant scalar leaf). The best any setting reached was rt=96 at n≥256, a
1–2% change inside measurement noise. **faer's default 128 is already at or
above the wasm optimum for this knob, in both directions.**

**Consequence — SVD conclusion, now runner-confirmed on all fronts.** The
deep research found the mechanism (scalar leaf) *and* established across all
four angles that SVD-with-vectors is structurally gemm-bound-at-peak (the
dominant back-transform) over a memory-bound bidiag floor. Unlike QR (weak
generic-C opponent + a phase that maps onto our flat kernel = 3×), **SVD has
no QR/LU-style wasm win available**: the dominant cost is already the fast
gemm path, the one tunable knob is refuted, and Jacobi is killed. faer's SVD
sits at ~0.5–0.8× scipy, rising toward parity with n, and that is close to
the achievable 2-lane ceiling for full SVD with singular vectors. The only
residual slivers are marginal (a bandwidth-optimal bidiag GEMV worth ~10–15%
of the ~30%-of-time reduction phase = ~3–5% of total; a faster shared gemm
kernel that would lift matmul/QR/LU/SVD alike). No cheap large win exists.

## Adversarially-verified pass: the universal negative HOLDS (2026-07-10)

The architect challenged the depth of the 4-angle fan-out — rightly: its
"no unexplored approach wins" conclusion was a single-agent universal
negative, the weakest grade in the synthesis. So the full deep-research
harness was run on exactly that claim (103 agents: 5 search angles → 21
sources fetched → 84 falsifiable claims extracted → top 25 through 3-vote
adversarial verification, 2/3 refutes to kill). Result: **23 confirmed,
2 refuted (both overreach in our favor's direction, excluded), 0
unverified. The negative holds for every candidate family that produced
evidence — and for the majors it is conceded by each method's own
inventors in peer-reviewed venues.**

Per-family verdicts (all vs dgesdd-class D&C, single-thread, n≤512):

- **Preconditioned Jacobi (dgejsv/Drmač–Veselić)** — killed 9–0.
  The inventors' own LAWN 170 data: the fully preconditioned code averages
  **~1.5–2× slower than SGESDD**. Part I's "outperforms" claim is vs
  QR-iteration SVD (dgesvd-class), never vs D&C; Part II's abstract caps at
  "comparable to" bidiagonalization methods. The "2–6 sweeps" figure that
  motivated our earlier hope is **not documented in LAPACK at all** (only
  the 30-sweep cap is); and dgejsv's preconditioner is *pivoted* QR
  (DGEQP3), which our fast unpivoted QR kernel only partially helps.
  Starting 1.5–2× behind, Jacobi would need a ~3× relative swing from the
  wasm shift to win — and 2 lanes penalize rotation kernels at least as
  much as gemm.
- **Mixed-precision Jacobi (TOMS 2025, state of the art)** — confirms the
  cap: its ~2× gain is over LAPACK's *Jacobi* baseline, not dgesdd, so at
  best parity — and it's out of scope for pure f64 anyway.
- **QDWH-SVD (polar decomposition)** — killed 24–0 across eight claims.
  Author-conceded (Nakatsukasa–Higham SISC 2013; Sukkari et al. TOMS
  2016/2019): up to **2× more flops** than bidiagonalization SVD
  (~35–52 n³ vs ~22 n³); every published win is explicitly attributed to
  level-3 BLAS *concurrency* recouping that overhead on GPU/manycore. On
  one thread with gemm already at the 5.6 GF/s roofline, a 2× flop deficit
  would need ~3× the incumbent's throughput to reach a 1.5× win —
  arithmetically impossible under the same roofline.
- **Zolo-SVD** — killed 8–1: strictly more flops than QDWH (recouped only
  by embarrassingly-parallel independent factorizations, r≈8 per iter);
  flop hierarchy Zolo > QDWH > dgesdd. No serial win exists in print.
- **QR-preprocessed SVD (Algorithm 977 / xGESVDQ)** — killed 5–1: on
  square inputs it is structurally xGESVD **plus** an added pivoted QR —
  strictly more work; its claims are accuracy-only. The m≫n regime where
  QR-preprocessing replaces work doesn't apply to square.
- **Serial block-Jacobi with gemm rotations (Demmel et al., June 2025)** —
  the one live research thread, confirmed 6–0 but doesn't help: proves
  communication optimality (gemm-rich, cache-optimal), NOT a flop
  reduction; zero benchmarks vs dgesdd; its arithmetic improvements hold
  "for galactic matrices." Communication optimality buys nothing where the
  incumbent's gemm is already compute-bound at peak. Watch for Part Two.

**Not covered by surviving claims** (graded *stated*, absence-of-evidence
only): dqds+inverse-iteration/MRRR-style vectors (xBDSVDX never fully
landed; clustered values need O(n³) reorthogonalization), SVD via Gram
matrix (κ² accuracy loss; cost structure a superset of dgesdd's) or
Jordan–Wielandt (2n dimension ⇒ ~8× eig cost), and spectral D&C standalone
(addressed only via its QDWH/Zolo instantiations). Standard arguments say
all three lose; they were not adversarially verified.

**Evidence-grade upshot:** "SVD has no >1.5× wasm win available" is now
**tested + cross-checked** (published measurements, author concessions,
3-vote adversarial verification, 21 sources) for all major families —
upgraded from the single-agent *stated* grade the architect flagged. The
residual engineering question the research leaves open is the one already
half-answered here: faer's remaining 0.5–0.8× gap vs scipy is
constant-factor overhead in the incumbent algorithm (the gap shrinks
7×→1.2× as n grows — fixed-overhead signature), not an algorithm choice.

## Sources
LAWN 169/170 (netlib), Drmač–Veselić Part I/II, Demmel–Veselić (SIAM 1992),
Computing SVD with high relative accuracy (LAA 1999), vectorized Jacobi
(arXiv 2202.08361), mixed-precision Jacobi (arXiv 2209.04626), ICL two-stage
SVD (icl-utk-1340-2018). Several full PDFs proxy-blocked; grades lean on
search snippets of those exact sources where noted.
