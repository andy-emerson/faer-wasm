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

## The build + runner verdicts (same day, 2026-07-11) — ⚠ tables in this section are PRE-allocator-fix; superseded by the CORRECTION section below

The campaign steps (a)–(d) were built and benchmarked (commit `eb98432`;
pyodide run 29146566266, wasm gate 29146577830 — both green, smoke
probes bit-unchanged). `schur_k` = kernel Hessenberg →
backward-accumulated Q (`hessenberg_form_q`) → hand `hqr` want_t+Z with
`dlanv2` standardization below the (frozen, provisional) 480 crossover;
kernel-Hessenberg front-end + faer's repaired multishift (want_t, Z
seeded with Q) above it. Fused simd128 `refl3`/`refl2` primitives carry
the Z updates and widened column applies (worth 12–15% end-to-end below
the crossover on the dev box).

**Replication-gated verdicts vs `scipy.linalg.schur`** (5 alternating
rounds, claim only on range separation; runner — pre-fix, kept as the
historical record):

| n | scipy med [range] | schur_k med [range] | verdict |
| - | - | - | - |
| 64 | 1.73 [1.72..1.74] | 1.32 [1.31..1.42] | **WIN 1.31×** |
| 128 | 14.52 [14.49..14.77] | 8.32 [8.17..8.57] | **WIN 1.75×** |
| 256 | 94.47 [94.46..95.95] | 89.66 [75.48..96.67] | OVERLAP 1.05× — no claim |
| 512 | 741.7 [740.9..742.8] | 1121.3 [1088..1135] | **LOSS 0.66×** |
| 1024 | 3601.2 [3598..3610] | 5120.4 [5102..5161] | **LOSS 0.70×** |

Against the shipping `faer-schur` baseline (same run, main grid):
9.83→1.31 ms @64 (7.5×), 24.9→8.1 @128 (3.1×), 223.8→75.1 @256 (3.0×),
1561.7→1124.4 @512 (1.39×). The scoreboard arc for Schur: 0.2×/0.4–0.6×
(baseline) → 1.31×/1.75× wins at 64/128, parity at 256, 0.66×/0.70×
at 512/1024 (pre-fix; the CORRECTION below revises the top sizes to
1.08×/1.10×/0.99×). Wasm-vs-native for `schur_k` (dev container, scripted):
1.43×/1.16×/1.10×/1.81×/1.40× at n=64–1024.

**Open question 1 answered (the delta's fraction split, runner):**

| n | eigvals | +T | +Z | T+Z | T share | Z share | total delta |
| -: | -: | -: | -: | -: | -: | -: | -: |
| 64 | 0.88 | 1.00 | 1.21 | 1.32 | 9% | 25% | 1.50× |
| 128 | 5.06 | 5.93 | 7.36 | 8.14 | 11% | 28% | 1.61× |
| 256 | 46.77 | 71.39 | 55.11 | 74.38 | 33%* | 11%* | 1.59× |
| 512 | 567.9 | 678.6 | 972.6 | 1079.6 | 10% | 37% | 1.90× |
| 1024 | 2527.0 | 2989.3 | 4682.6 | 5191.4 | 9% | 42% | 2.05× |

(*n=256 is min-of-2 with the same wide spread the replication row shows
— treat the 256 split as noisy.) **Z is the dominant delta cost and
grows with n**; T-widening is ~10%.

**Open question 2 partially answered.** Above the crossover the pipeline
rides faer's accumulated multishift (the KACC22-style path), and its
measured eigvals→Schur delta is **1.90–2.05×** — while scipy's reference
`dhseqr` on the same machine pays only 1.06–1.30× (1.73/1.63 @64 …
3601/3145 @1024). So LAPACK's accumulation discipline is NOT being
matched by faer's implementation of the same idea on wasm: at n=512 our
+Z alone costs ~405 ms (form_q ≈ 60–80 ms of it) where scipy's *entire*
T+Z+Q-formation delta is ~149 ms. The 512/1024 losses live exactly
there. Candidate levers, in evidence order: (i) route Z through our own
machinery — extend the hand `hqr`+Z past 480 (the 480 crossover was
measured against pre-kernel pipelines and never for the Schur+Z shapes;
known re-tune debt, frozen until the global pass); (ii) a wasm-shaped
Z-accumulation for the multishift sweep (the research's U-GEMM question,
now with a measured target); (iii) profile faer's want_t/Z multishift
internals for the same class of no_std/routing landmine that 0004 was.
Below the crossover the flat want_t/Z applies already deliver
LAPACK-grade deltas (1.50–1.61×) against reference dlahqr-class costs —
and the wins.

## The c64 twins — decision point (e), built + verdicts (2026-07-11)

Built same day on the architect's go (commits `d6a6636`/`6880b5a`):
`kernels/hessenberg_cplx` (zgehd2-shape reduction + zunghr-shape backward
Q), `kernels/schur_small_cplx` (`chqr_schur_in_place` — faer's
Givens-based single-shift complex lahqr shape with want_t/Z, `rotg`
ported verbatim except LAPACK `r = a` semantics on `b == 0`), flat
*scalar* complex loops per the Jacobi-probe discipline (bare-correct
first, simd where measurement says). Correctness gates all green
(`kernels/tests/schur_cplx.rs`: ‖A−ZTZᴴ‖, unitarity, exact triangular T,
eigenvalues vs faer, diag(T)=w).

**Replication verdicts vs `scipy.linalg.schur(ac, output='complex')`**
(run 29157035070, 5 alternating rounds, ranges separated at every size):

| n | scipy med [range] | schur_c64_k med [range] | verdict |
| - | - | - | - |
| 64 | 5.42 [5.42..5.44] | 3.92 [3.91..3.94] | **WIN 1.38×** |
| 128 | 36.71 [36.66..36.76] | 27.48 [27.08..27.64] | **WIN 1.34×** |
| 256 | 277.9 [277.7..278.3] | 310.0 [298.5..324.2] | **LOSS 0.90×** |
| 512 | 1674.7 [1673.3..1676.7] | 1528.9 [1528.4..1534.2] | **WIN 1.10×** |
| 1024 | 9923.7 [9916..9946] | 9672.5 [9670..9706] | **WIN 1.03×** |

The c64 arc: 0.4–0.9× baseline → wins at 4 of 5 sizes. The one loss has
a located mechanism: at 256 the pipeline is our `chqr`, whose rotation
applies are scalar, while faer's own complex lahqr (the `schur_c64`
baseline row: 253.7 ms vs our 310.3 at 256 on this machine) applies
rotations through pulp's SIMD — **the identified next lever is a
simd128 complex-rotation primitive** (one c64 per 128-bit lane, the
c64 twin of `refl3`/`refl2`), which should flip 256 and widen 64–128.
At 512+ both routes ride faer's complex multishift and our Hessenberg
front-end wins the difference.

**CORRECTION — the "machine class" read was wrong; it was the allocator
fix.** Run 29157035070 (post-fix) vs the earlier runs: scipy's times
barely moved (its Pyodide allocator was never affected — schur@512
742→673 ms), while our allocation-heavy rows dropped 1.6–1.8×
(eigvals_k3@512 586→364 ms, schur_k@512 1121→612 ms, suite geomean
0.93→1.70×). The leak-only bump allocator was taxing OUR side of every
large-n comparison — cold pages and `memory.grow` inside the timing
loop on every faer temp — so all pre-fix verdicts at n ≥ 512
understated us. **Post-fix scoreboard (run 29157035070, the new
reference): real `schur_k` WIN 1.24×/1.67×/1.08×/1.10× at n=64–512,
0.99× range-separated loss at 1024; `eigvals_k3` WIN at ALL five sizes
incl. 512 (1.52×) and 1024 (1.51×) — the former 512-parity verdict was
the allocator tax, not the kernel.** Post-fix mode split: T ~7–24%,
Z ~18–33%, total delta 1.49–1.97× — Z still the dominant lever.
Re-verification debt recorded on the watch list: every pre-fix
large-n measurement (incl. the blocked-Hessenberg "machine cliff"
magnitudes, the 480 crossover, and the LU large-n verdicts) carries
some allocator tax and should be re-read against post-fix runs before
being cited; within-run ratios are less contaminated than
cross-run/cross-build comparisons since both sides of each in-run pair
paid the same allocator.

**The n=1024 crash that run A exposed** (and run B, post-fix, survived):
faer's c64 matmul allocates per-call temporaries via GlobalAlloc — one
c64 multishift at n=600 measured at **15.4 GB cumulative / ~25K
allocations, peak live ~19 MB** (counting-allocator probe,
`kernels/tests/alloc_probe.rs`). Fatal on the leak-only bump allocator
the zero-import pattern used; fixed by a 3-line LIFO-rewind in `dealloc`
(+3,734 MB → +7 MB per call at n=600), documented in docs/wasm.md §2,
guarded by the alloc_probe peak-live assertion, ledgered upstream (also
a no_std perf hazard: 25K allocations per solve).

## Close-out (2026-07-12): crot verdict, f32 row, and the runner-drift rule

The campaign's last build half (commit `6c3fb49`) delivered the
predicted simd128 complex-rotation primitive (`kernels/cplx.rs`
`crot_streams`/`crot_row_pair`, one c64 per 128-bit lane) wired into
`chqr`'s three apply loops, plus the f32 Schur benchmark row.

**The cross-run reading was a trap.** The next pyodide-bench run
(29172715791) read c64@256 as 0.90×→**0.76×** — the crot change
apparently a 17% regression at exactly its target size. But rows whose
code did not change between the runs drifted the same direction:
`schur_k`@256 81.9→93.6 ms, `eigvals_k3`@256 42.2→48.7 ms (7–15% on
identical binaries; scipy's side moved ~7% too). CI hands out a
different machine each run. **Cross-run ratios cannot judge a code
change; only within-run, interleaved comparisons count.** That rule now
has a tool: `bench/ab-crot.mjs` builds two variants and times them
alternating on one machine, with an untouched op as a same-machine
control row.

**Same-machine verdicts** (run 29173699093 on a CI runner; control
`schur_k` read 0.99–1.00× OVERLAP at all sizes, validating the method):

| n | crot med [range] ms | scalar med [range] ms | verdict |
| - | - | - | - |
| 64 | 3.32 [3.30..3.67] | 3.92 [3.91..4.61] | **crot WINS 1.18×** |
| 128 | 23.23 [23.00..24.54] | 27.25 [27.15..27.66] | **crot WINS 1.17×** |
| 256 | 376.4 [306.2..378.2] | 315.5 [309.9..329.6] | OVERLAP — no claim |

At 256 the machine itself was noisy (the *control* swung 17–23% within
rounds there), so no claim either way; a second same-machine A/B in the
dev container separated cleanly in crot's favor at 256 (292.8 [286..307]
vs 367.5 [363..375], **1.25×**, median of 5). **Decision: keep crot** —
it wins 1.17–1.25× everywhere measurement separates and never
measurably loses. The RQ4 prediction half-held: it widened 64/128 as
predicted; it did not flip 256 — c64@256 stays the campaign's one
recorded residual vs scipy (0.76–0.90× depending on the machine drawn).

**Postscript (2026-07-12, run 29177564170):** a third runner machine
read c64@256 as a separated **1.21× WIN** — the "residual" is not a
fixed loss but machine-dependent near-parity. Generalized as the
verdict-stability rule (README evidence grid): rows with margins under
~1.3× flip WIN/LOSS with the CI machine drawn; margins ≥1.4× replicate
on every machine.

**f32 Schur row** (run 29172715791 main grid, closing the coverage
gap): 1.7× / 2.5× / 2.2× / 1.1× at n=64/128/256/512 vs
`scipy.linalg.schur(a32)` — the same shapes that win at f64 win at f32,
consistent with the R-ratio account (both sides of R scale together).

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
