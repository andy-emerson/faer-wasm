# STATUS — the one-page scoreboard

Plain-English answer to three questions: **what did we change in faer,
what do we ship, and how good is it?** Updated at the end of every
working session. Details and evidence live in `docs/` and the README
evidence grid; this page is the summary you can hold in your head.

Last updated: 2026-07-19.

## 1. What we changed in faer itself

**Four patches, ~19 lines total.** That's everything. They live in
`patches/` and get applied to a pinned copy of faer on every build.

| patch | size | what it does |
| - | - | - |
| 0001 | 4 lines | fixes the compile error that stopped faer building on wasm at all |
| 0002 | 6 lines | makes faer's internal Schur functions public so our code can call them |
| 0003 | 4 lines (to pulp, faer's SIMD library) | fixes a bug that made all complex math wrong in one build mode |
| 0004 | 1 line | fixes a typo'd formula that made eigenvalue solving ~50–85× slower |

Everything else in this repo is **our own code in our own folders**
(`kernels/`, `schur/`, `smoke-test/`, `bench/`). It calls faer; it does
not modify faer.

## 2. What we ship, and how good it is

"vs scipy" = the honest head-to-head on the CI runner, both sides
running in the browser engine. Numbers >1 mean we're faster.

### Double precision, real numbers (f64) — the main product

| operation | works? | tested? | benchmarked? | vs scipy |
| - | - | - | - | - |
| multiply (matmul) | ✓ (faer's, unmodified) | ✓ | ✓ | 3–20× faster |
| solve Ax=b (LU) | ✓ our kernel | ✓ | ✓ | 1.6–2× faster |
| QR | ✓ our kernel | ✓ | ✓ | ~3× faster |
| SVD | ✓ (faer's, unmodified) | ✓ | ✓ | 0.7–1.5× (near its ceiling — proven) |
| eigenvalues | ✓ our kernel | ✓ | ✓ | 1.5–2.1× faster, all sizes |
| Schur (T and Z) | ✓ our kernel | ✓ | ✓ | 1.1–1.7× faster up to n=512; ~tie at 1024 |
| eigenvectors (full eig) | ✓ our kernel (new 07-12) | ✓ | ✓ | 1.5–2× faster, all sizes incl. 1024 |

### Single precision, real (f32)

| operation | works? | tested? | benchmarked? | vs scipy |
| - | - | - | - | - |
| multiply, LU, QR, eigenvalues | ✓ same kernels as f64 | ✓ | ✓ | 2–9× faster |
| Schur | ✓ same kernel as f64 | ✓ | ✓ | 1.7–2.5× faster to n=256, 1.1× at 512 |
| eigenvectors (full eig) | ✓ same kernel as f64 | ✓ | ✓ | 2.9–4.4× faster |

### Double precision, complex (c64)

| operation | works? | tested? | benchmarked? | vs scipy |
| - | - | - | - | - |
| multiply, LU, QR, SVD, eigen | ✓ (faer's, unmodified) | ✓ | ✓ | mixed: matmul 4–5×, QR 2–4×, LU ~1× |
| Schur | ✓ our kernel + SIMD rotations | ✓ | ✓ | ~parity: verdicts at single sizes flip with the CI machine (0.9–1.6×) |
| eigenvectors (full eig) | ✓ our kernel (new 07-12) | ✓ | ✓ | 2.1–3.2× faster, all sizes — the widest margins in the project |

### Single precision, complex (c32)

Nothing of ours exists. faer's built-in c32 works but is untested and
unmeasured by us. Building c32 versions of our kernels is a known,
scoped job — queued behind the packaging decision.

## 3. The direction reset (2026-07-18) — read this first

Andy's benchmark experiment overturned a founding assumption: faer's
matrix multiply — the engine every kernel routes its heavy work to —
is actually **slower than a simple streaming loop** on the reference
machines (by 10–30% up to n=512; about even at 1024). That triggered a
re-derivation of the project goals. The decisions, in plain terms:

- **New yardstick**: success is now "how close to the machine's
  physical speed limits are we," not "how much faster than scipy are
  we." scipy stays on the scoreboard for marketing only.
- **New plan**: build our own complete BLAS layer first (the simple
  fast loops, properly named, tested, and benchmarked), then make our
  LAPACK-layer functions use it. Step 1 (done): the FMA build — a
  faster multiply instruction — doubles the machine's speed limit but
  does NOT rescue faer's multiply; it helps some of our loops
  (trmm/trsm/gemv) and actively hurts another (syrk), so each
  operation gets its variant picked by measurement. Step 2 (done):
  Andy asked whether the plan's "hand-SIMD buys nothing here" rows
  were ever tested — they weren't, we raced them, and all three
  assumptions were wrong (swap 1.2–1.3× faster with SIMD, asum
  3.5–4×, iamax 1.4–1.6×, on all three CI machines). The full build
  list with evidence per row is `docs/blas-layer-plan-2026-07.md`,
  and the layer has its home: the `blas/` crate — one folder per
  BLAS level, one file per function, the plan table in its README.
- **The f64 tuning pass is DONE — campaign closed 2026-07-19**:
  before cloning the layer into the other number types, the f64
  loops were made fast (Andy's revised sequencing — the clones
  inherit the speed for free). Four rounds, every verdict backed by
  two CI machines. (1) Matrix
  multiply got two tuned shapes — a register-tiled version for small
  matrices and a stream-friendly four-column version for large — and
  now **beats faer at every size we measure** (1.25–1.8×, confirmed
  on two CI machines); the right shape is picked automatically by
  size, invisibly, because both produce bit-identical results. (2)
  The summing functions (dot, absolute-sum, norm) now use four
  parallel accumulators — 1.4–2.2× faster on the container. (3) The
  same two blocked shapes were carried through matrix×vector and all
  six matrix–matrix functions: every one gained 10–30% on the
  container (the matrix–matrix family moved from 41–52% to 48–56% of
  the machine's arithmetic limit). Results stay bit-for-bit identical
  to the plain loops everywhere the math allows — locked by tests —
  with the two unavoidable reorderings documented and tested to their
  own fixed order. Rounds 2–3 are CI-machine confirmed on two draws:
  the dot product runs at the machine's memory read limit,
  matrix×vector at 1.75× its old speed, and the matrix–matrix family
  at 48–56% of the arithmetic limit (was 34–44%). A bonus finding:
  transpose-matrix×vector got 1.3–1.7× faster without being touched —
  it's built as a loop of dot products, so the dot improvement flowed
  through, which is exactly why the layer is structured as
  compositions. Round 4 (also two-draw confirmed): the symmetric
  multiply-vector now fuses both triangles' work into one pass over
  four columns at a time — 2× faster, and the symmetric
  matrix–matrix multiply that is built on it reached **84–86% of the
  machine's arithmetic limit**, the best matrix–matrix row on the
  board. One candidate lost its race: a single-pass version of
  find-largest-element measured slower than the shipped two-pass
  shape on both CI machines and was reverted — the loss is recorded
  so it isn't re-tried blind. Remaining lever: the
  faster-multiply-instruction (FMA) build — held for an architect
  decision, because wasm's relaxed FMA rounds differently on
  different hardware, which would trade away our identical-results-
  everywhere guarantee if shipped as the default.
- **The f64 BLAS layer is COMPLETE** (2026-07-18): Level 3's six
  matrix–matrix functions landed the same way — matrix multiply is
  literally "matrix × vector, once per column", and so on down; the
  whole 23-function layer is four loop shapes plus one scalar
  function. Tested (30 tests), identical bits native/wasm on every
  check (21 probes), and speed-scored: at build time the
  matrix–matrix ops ran at 34–44% of the machine's arithmetic speed
  limit — matching what the original experiment measured for the
  simple loop that beat faer, and the headroom the tuning pass has
  since claimed (see the bullet above). Two CI machines agreed within
  1% on every row. Andy's sequencing is the plan of record — revised
  same day to tune-first: tune and benchmark f64 (now done), then the
  other number types, and no LAPACK work until the BLAS layer is
  done.
- **Level 2 is built** (2026-07-18): all seven matrix–vector functions
  (multiply, transpose multiply, rank-1/2 updates, symmetric multiply,
  triangular multiply and solve), each one literally a loop of Level-1
  calls over matrix columns — the plan's classification became the
  code structure. Tested to the same standard (exact bit agreement
  where the math allows, independent cross-checks everywhere, solves
  verified by multiplying back), identical bits native/wasm on all
  checks. Speed on the CI machines: the rank-1 updates run at 83–100%
  of the machine's limit; the multiply-vector family sat near half of
  it at build time — the tuning pass (bullet above) has since roughly
  doubled it.
- **Level 1 is built** (2026-07-18): all ten vector functions, tested
  (12 tests: exact bit-agreement where the math allows, error-bounded
  elsewhere) and speed-scored on the CI machines — the in-place ops
  run at 81–100% of the machine's fastest memory stream, the rest sit
  at the read-bandwidth limit or within reach of it. Same code gives
  identical bits on native and wasm, checked on every run. Getting
  this honest caught two of our own measurement bugs: our first build
  was 6× slower than the raced prototype (a missing wasm build flag —
  now a documented rule), and the old bandwidth-ceiling probe was
  accidentally timing memory allocation, understating every machine's
  true speed limit by ~2× (old 12–18 GB/s readings; really ~32).
- **Threading: decided no** — browsers demand a server configuration
  (COOP/COEP) Andy excludes, and the honest payoff is small for our
  matrix sizes anyway. GPU (for f32) and batch parallelism remain
  future options.
- **Long-term**: faer-wasm is heading toward standing on its own.
  faer's remaining functions get replaced one measured campaign at a
  time — never on principle, only on evidence.

## 4. Known gaps and next levers (for the architect to pick from)

- **Schur near parity at larger sizes** — Schur rows with margins under
  ~1.3× flip between win and loss depending on which machine CI hands
  us (complex Schur at 256 read 0.89× on two machines and 1.21× on a
  third; real Schur at 512 read 1.10× and 0.95×). The honest claim is
  "about even". Rows with margins of 1.4× and up — all of LU/QR/
  matmul/eigenvalues/eig — replicate on every machine.
- **real Schur at n=1024** — a tie, not a win; the remaining cost is in
  faer's large-size path, levers documented in the research doc.
- **re-check old numbers** — on 2026-07-11 we found our old benchmark
  setup was unfairly slowing OUR side of every large-size comparison (an
  allocator problem, now fixed). Numbers published before then understate
  us; they're flagged, not yet all re-measured.
- **benchmark honesty (2026-07-12)** — CI hands us a different machine
  each run, and identical code drifted 7–15% between two runs. So a
  number from one run can't be compared with a number from another; only
  comparisons made back-to-back inside a single run count. Our
  scipy-vs-us verdicts already work that way; judging our own code
  changes now does too (`bench/ab-crot.mjs`).

## The consistency rule (adopted 2026-07-11)

Every new kernel gets the full treatment before its campaign closes:
correctness test **and** benchmark row, in **every** number type it
supports — or an explicit "gap" line in this file saying why not.
This page gets updated in the same commit.
