// Native side of the blas cross-target determinism gate: prints the
// probe results as hex bit patterns; the roofline scripts compare the
// wasm build against them. Modes: l{1,2,3}-bits, l{1,2,3}-bits-f32,
// l{1,2,3}-bits-z, and l{1,2,3}-bits-c.
fn main() {
    let mode = std::env::args().nth(1).unwrap_or_default();
    let dump = |vals: Vec<f64>| {
        for v in vals {
            println!("{:016x}", v.to_bits());
        }
    };
    match mode.as_str() {
        "l1-bits" => dump((0..4).map(|op| blas_bench::run_l1_probe(op)).collect()),
        "l2-bits" => dump((0..8).map(|op| blas_bench::run_l2_probe(op)).collect()),
        "l3-bits" => dump((0..9).map(|op| blas_bench::run_l3_probe(op)).collect()),
        "l1-bits-f32" => dump((0..4).map(|op| blas_bench::run_l1_probe_f32(op)).collect()),
        "l2-bits-f32" => dump((0..8).map(|op| blas_bench::run_l2_probe_f32(op)).collect()),
        "l3-bits-f32" => dump((0..9).map(|op| blas_bench::run_l3_probe_f32(op)).collect()),
        "l1-bits-z" => dump((0..5).map(|op| blas_bench::run_l1_probe_z(op)).collect()),
        "l2-bits-z" => dump((0..10).map(|op| blas_bench::run_l2_probe_z(op)).collect()),
        "l3-bits-z" => dump((0..9).map(|op| blas_bench::run_l3_probe_z(op)).collect()),
        "l1-bits-c" => dump((0..5).map(|op| blas_bench::run_l1_probe_c(op)).collect()),
        "l2-bits-c" => dump((0..10).map(|op| blas_bench::run_l2_probe_c(op)).collect()),
        "l3-bits-c" => dump((0..9).map(|op| blas_bench::run_l3_probe_c(op)).collect()),
        _ => {
            eprintln!("usage: native l{{1,2,3}}-bits[-f32|-z|-c]");
            std::process::exit(2);
        }
    }
}
