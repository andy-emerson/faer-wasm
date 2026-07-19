# Correctness tests

One test file per level (`level1.rs`, `level2.rs`, `level3.rs`), each
test landing in the same commit as the implementation it covers — see
the testing contract in `../README.md` for the per-class standard
(bit-for-bit for elementwise streams, error-bounded vs a
higher-precision reference for reductions and Level 2/3, exact index
semantics for iamax).

Performance is NOT tested here: roofline scoring runs in the wasm
runtime on the reference CI machines via `../../bench/`.
