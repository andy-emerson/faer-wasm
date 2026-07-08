// Native side of the cross-target determinism gate: print the probe values
// as exact f64 bit patterns. determinism.mjs compares these against the same
// computations run in the wasm build — bit-for-bit, no tolerance.
fn main() {
    println!("matmul_trace={:016x}", consumer::matmul_trace().to_bits());
    println!("lu_solve_sum={:016x}", consumer::lu_solve_sum().to_bits());
    println!(
        "qr_svd_evd_probe={:016x}",
        consumer::qr_svd_evd_probe().to_bits()
    );
    println!("schur_probe={:016x}", consumer::schur_probe().to_bits());
    // schur_probe_cplx is intentionally NOT in the determinism list: the
    // full-relaxed build is known-broken for c64 at this pin (check.mjs
    // canary), so a native-vs-relaxed comparison would always fail.
}
