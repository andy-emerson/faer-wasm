# Schur (want_t + Z) wasm research — 2026-07-11

Deep-research pass ordered by the architect for the Schur campaign
(next-sessions plan item 1), scoped to the **cost delta** from the
already-swept eigenvalues-only pipeline to full Schur-with-vectors — the
shared pipeline (Hessenberg, multishift-vs-lahqr routing, sweep gemm)
was researched in `research-eig-wasm-2026-07.md` and is not re-litigated
here. Harness: 5-angle search fan-out → 18 sources fetched → 53
falsifiable claims → top 25 through 3-vote adversarial verification
(**21 confirmed, 4 refuted, 0 unverified**; run `wf_a18f11c8-af8`,
100 agents). Provenance caveat: netlib.org and arxiv.org returned 403
through the session proxy, so source-code claims were verified against
the GitHub Reference-LAPACK mirror (authoritative) and paper claims
against abstracts/secondary implementations (weaker — flagged per
claim). All dlaqr5 structural facts are LAPACK ≥ 3.10 (the Thijs Steel
rewrite), which matches the opponent's bundled 3.12.

## The headline mechanism (RQ1) — confirmed 3-0, line-verified

The eigvals→Schur delta in LAPACK's sweep kernel (`dlaqr5`) is **exactly
two mechanisms, nothing else**:

1. **`WANTT` is pure range-widening.** With `WANTT`, each bulge-chase
   update's row/column window is `JTOP=1 … JBOT=N` instead of clamping
   to the active block `[KTOP, KBOT]` (`dlaqr5.f` 381–387, 708–714,
   784–791). Same reflectors, wider application range.
2. **`WANTZ` adds Z-column updates** over `ILOZ..IHIZ` — the only other
   divergence.

There is no third cost. 2×2 standardization (`dlanv2`-shape) happens at
deflation time, not per sweep step, and no source quantified its share —
the fraction split (full-range T applies vs Z updates vs
standardization) **must come from our own instrumentation** (open
question 1).

## How LAPACK keeps the delta at ~1.26× (RQ2) — confirmed 3-0

With `KACC22 ∈ {1,2}`, `dlaqr5` accumulates each slab's reflections into
a small orthogonal factor `U` (order `KDU = 2·NSHFTS`, seeded to
identity) and applies **all** far-from-diagonal H-row updates
("horizontal multiply", `NH`-column panels), above-slab column updates
("vertical multiply", `NV`-row panels), **and the Z update** as DGEMMs
against `U`. **Z rides the accumulation machinery essentially for
free** — one extra `(IHIZ−ILOZ+1) × NU × NU` gemm band per slab step,
structurally identical to the vertical H update. With `KACC22 = 0`,
every reflector is applied as scalar loops over the widened ranges
instead.

**The payoff is bounded and is a single knob** (confirmed 3-0):

- The contracted (inner) gemm dimension is `NU ≤ KDU = 2·NSHFTS`;
  `NSHFTS` is hard-capped at `(N−3)/6` and forced even. Bounds: 42 at
  n=256, 170 at n=1024; IPARMQ's actual table gives NS ≈ 64 at n=1024,
  so `KDU ≈ 128`.
- `KACC22` is one ILAENV integer (ISPEC=16), clamped to 0..2 and passed
  straight through — accumulate-vs-direct flips with one integer.
- `KACC22 = 2` (the DTRMM variant exploiting U's 2×2 block structure)
  was **deliberately dropped in the LAPACK 3.10 rewrite** (now identical
  to 1; DTRMM declared but never called). Even reference LAPACK
  abandoned the structure-exploiting refinement.

**The central unanswered engineering question**: LAPACK's 1.26× is
achieved on hosts where gemm is ~10× scalar loops. On our target gemm is
only ~2× flat simd128 loops, and the inner dimension is small — *which
side of LAPACK's design point does the runner fall on?* No published
serial eigvals-vs-Schur cost split exists (searched; none survived).
Needs the head-to-head benchmark (open question 2). Note faer's
multishift already implements the accumulated path (wh/wv workspaces —
`research-eig-wasm-2026-07.md`), so above the crossover we inherit it;
the question bites for our below-crossover hand kernel and for whether
flat want_t/Z beats accumulation even at n ≥ 512 here.

## Tuning interface (confirmed 3-0)

`DLAQR0`/`ZLAQR0` pass a job string `JBCMPZ` ('S'/'E' × 'V'/'N') to
every ILAENV query (NMIN/NW/NIBBLE/NS/KACC22), so per-job tuning is
architecturally sanctioned — **but reference IPARMQ ignores the job
string**, so the opponent tunes both paths identically. We are free to
pick different crossovers/shift counts for Schur than for eigvals;
LAPACK's own hook anticipates it. (The 480 crossover was measured for
all three pipelines on run 29134291933, so the current routing already
respects this; re-sweep belongs to the global tuning pass.)

## Complex Schur (RQ4) — confirmed 3-0/2-1

- `ZLAQR0` is a **structural twin** of `DLAQR0`: same control loop, same
  JBCMPZ tuning, shifts forced even (a lone pair duplicated), `ZLAQR5`
  chases shift **pairs** with 3×1 reflectors and has a ZGEMM
  accumulated-U path.
- `ZLAHQR` (the small-matrix kernel) is **structurally different from
  dlahqr**: **single-shift with 2×1 reflectors** (`ZLARFG(2,…)`) — each
  bulge position touches only 2 rows / 2 columns / 2 Z-columns — and its
  Z accumulation is a plain flat per-reflector loop, no batching.
- A hand c64 kernel therefore faces a different want_t/Z cost model than
  the real double-shift case, and chooses between a zlahqr-shape
  (single-shift 2×1, cheaper per position, one shift per sweep) and a
  paired-shift 3×1 chase. No prior art found on packing one c64 per
  128-bit lane in a bulge chase; the ~4× complex:real folklore ratio was
  **neither confirmed nor refuted** (open question 4).

## Reordering (RQ5) — confirmed 3-0 / medium

**Kressner's block reordering** (LAWN 171 / TOMS 2006) beats LAPACK
`dtrexc`/`dtrsen` **"by up to a factor of four"** serially: delay the
orthogonal transformations from adjacent swaps inside a small diagonal
window, accumulate into an nw×nw factor, batch-apply to off-diagonal T
and Q. Critically for us, **level-3 application is optional per the
paper itself** — the windowing/delaying is a cache-locality win
separable from gemm routing (corroborated by three independent
implementations; the 4× number is single-paper sourced-unverified,
presumably gemm-strong-host). Reference LAPACK still lacks it, so it
does not help the opponent. This upgrades `faer-schur`'s per-swap
reordering as a future lever if reordering ever shows up hot.

## Post-2015 serial landscape (RQ6) — the QDWH-trap check

- **StarNEig / Algorithm 1019 (TOMS 2022): concurrency-only, confirmed
  3-0.** Its core is explicitly inherited Braman–Byers–Mathias
  multishift+AED; the novelty is task scheduling; perf claims are only
  vs *multi-threaded* LAPACK/ScaLAPACK. Inapplicable to no-threads wasm
  — the exact analog of the SVD QDWH/Zolo trap, as suspected.
- **RQR (Camps–Mach–Vandebril–Watkins, LAA 2025): genuinely serial**,
  reported ~17% (n<75) to ~29% (larger) faster than `ZLAHQR` with
  1.5–2× smaller backward error — **but the baseline is the unblocked
  single-shift ZLAHQR**, not the production ZLAQR0 path the opponent
  runs above n=75, and the paper says nothing about want_t/Z. Ledger
  note, not a build target. (Single-paper, sourced-unverified numbers.)
- Nothing found on cache-oblivious Hessenberg QR or fused bulge chains
  that beats dlaqr0–5 serially at n ≤ 1024.

## The opponent (RQ7) — confirmed 3-0, cross-checked durability

`scipy.linalg.schur` on the Pyodide OpenBLAS 0.3.28-generic build runs
**verbatim reference-netlib Fortran for the entire Schur path**:
`DGEHRD/DORGHR/DHSEQR/DLAQR0–5/DTREXC`. OpenBLAS's optimized-LAPACK
layer covers only the LU/Cholesky/triangular families (+`laed3`
post-0.3.28) — nothing Schur-shaped; the optional (default-off) ReLAPACK
layer likewise covers no Schur-path routine. Verified by directory
listing, GitHub code-search API, a blobless clone of v0.3.28, and source
diffs against Reference-LAPACK 3.12. Only the *internal BLAS calls*
(dlaqr5's DGEMMs, dgehrd/dorghr's DLARFB/DGEMM) hit optimized kernels.
**This is the same reference-LAPACK-over-generic-BLAS bar we beat for
QR** — our current 0.4–0.6× deficit is against reference Fortran logic,
so closing it is accumulation/range-widening engineering, not beating
hand-tuned assembly. Bonus finding for Phase 2: ReLAPACK ships recursive
`trsyl`/`tgsyl` Sylvester solvers (default-off in OpenBLAS) — relevant
context for the Sylvester item.

## Refuted (do not build on these)

- "LAPACK doesn't pack bulges as tightly as possible" (as extracted from
  the Karlsson–Kressner–Lang optimally-packed-chains framing) — **0-3**.
  Treat the paper's ~33% chase-flop-reduction claim as unestablished for
  the ≥3.10 code.
- "KACC22 exists identically in the complex path incl. the 2×2-structure
  exploitation" — **0-3** as stated (zlaqr5 has an accumulated ZGEMM
  path, but the claim's specifics overreached).
- "ZLAHQR's entire want_t delta is carried by the I1/I2 bounds as
  described" — **0-3**. **Secondhand descriptions of the small-kernel
  want_t mechanics are unreliable; derive the index ranges directly from
  `dlahqr.f`/`zlahqr.f` — or, for us, from faer's own `lahqr` (the
  pinned source our kernel was ported from), which implements want_t and
  Z.**
- "Zero 'laqr' hits in OpenBLAS kernel/" — **1-2** on search-scope
  grounds (the compiled-path conclusion stands via build-file
  verification; one orphan dead-code dlaqr5 copy exists).

## Open questions → the build's measurement plan

1. **Fraction split of the delta on our target** (full-range T applies
   vs Z updates vs 2×2 standardization): instrument the hand hqr kernel
   with independent want_t / Z toggles.
2. **Does accumulated-U gemm batching pay at gemm ≈ 2× flat, inner dim
   2·NS ≈ 32–128?** Head-to-head on the runner; the answer may differ
   for T updates vs Z updates.
3. **Q-formation strategy** (RQ3 — *no surviving claims at all*):
   dorghr-style backward accumulation (~4/3 n³, exploits the growing
   identity block) vs forward application to I (~2n³), blocked-WY vs
   unblocked flat. First-principles flop count says backward + flat;
   measure.
4. **c64 twins**: zlahqr-shape (single-shift 2×1) vs paired-shift 3×1 on
   2-lane SIMD, and the real c64:real cost ratio — decision point (e) of
   the campaign.

## Build plan implied (campaign steps, refined by the research)

(a) **Z-accumulating Hessenberg**: form Q from the kernel's stored
reflectors by backward accumulation, flat dot/axpy (RQ3 answer pending
measurement — it's the cheapest-flop shape and matches the 5/5 flat
precedent). (b) **hqr want_t + Z**: extend the hand kernel with the
widened ranges and Z applies ported from faer's own `lahqr` (not from
secondhand LAPACK descriptions — see refuted list), reinstating 2×2
standardization (`lahqr_schur22`-shape) which eigenvalues-only mode
deleted. (c) **Routing**: keep the provisional 480 crossover
(faer's accumulated multishift above it) pending the global tuning
pass. (d) **Benchmark**: wasm-vs-native + replication-gated vs
`scipy.linalg.schur` at n = 64–1024, plus the want_t/Z toggle
instrumentation of open question 1. (e) **c64 decision** afterward, per
open question 4.

## Sources

Reference-LAPACK master via GitHub raw (dlaqr0/dlaqr5/zlaqr0/zlaqr5/
zlahqr/iparmq — primary, line-verified), netlib explore-html snapshots,
OpenMathLib/OpenBLAS + Molcas GitLab mirror + HPAC/ReLAPACK (opponent
verification, cross-checked), LAWN 171 (Kressner block reordering,
abstract + secondary corroboration), arXiv 2007.03576 (Algorithm
1019/StarNEig), arXiv 2411.17671 (RQR), Karlsson–Kressner–Lang TOMS
2014 (optimally packed chains — claim refuted as extracted). Full
per-claim votes and evidence in the workflow journal
(`wf_a18f11c8-af8`).
