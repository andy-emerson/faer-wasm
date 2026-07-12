# STATUS — the one-page scoreboard

Plain-English answer to three questions: **what did we change in faer,
what do we ship, and how good is it?** Updated at the end of every
working session. Details and evidence live in `docs/` and the README
evidence grid; this page is the summary you can hold in your head.

Last updated: 2026-07-12.

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
| Schur | ✓ our kernel + SIMD rotations | ✓ | ✓ | 1.0–1.6× faster, except ~0.8–0.9× at n=256 |
| eigenvectors (full eig) | **✗ — gap** | — | — | needs the complex twin of the new eigenvector kernel; scoped, not built |

### Single precision, complex (c32)

Nothing of ours exists. faer's built-in c32 works but is untested and
unmeasured by us. Building c32 versions of our kernels is a known,
scoped job — queued behind the packaging decision.

## 3. Known gaps and next levers (for the architect to pick from)

- **complex Schur at n=256** — the one remaining loss (0.8–0.9×
  depending on which machine CI hands us). We built the SIMD rotation
  fix for it; measured fairly (same machine, alternating), it makes our
  kernel 1.2× faster at small sizes and never measurably hurts, so we
  kept it — but it didn't flip 256. Whatever scipy does better there is
  not yet located; parked as the campaign's recorded residual.
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
