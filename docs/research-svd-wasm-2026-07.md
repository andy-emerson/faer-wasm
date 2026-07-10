# SVD wasm-shaping research — sequential focused passes (2026-07-10)

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

## Sources
LAWN 169/170 (netlib), Drmač–Veselić Part I/II, Demmel–Veselić (SIAM 1992),
Computing SVD with high relative accuracy (LAA 1999), vectorized Jacobi
(arXiv 2202.08361), mixed-precision Jacobi (arXiv 2209.04626), ICL two-stage
SVD (icl-utk-1340-2018). Several full PDFs proxy-blocked; grades lean on
search snippets of those exact sources where noted.
