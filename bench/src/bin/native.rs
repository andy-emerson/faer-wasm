// Native side of the wasm-vs-native benchmark. Same ops, same sizes, same
// adaptive-iteration logic as bench.mjs; emits JSON lines:
//   {"target":"native-o3","op":"matmul","n":128,"ns":123456.7}
use std::time::Instant;

fn time_op(f: impl Fn() -> f64, n: usize) -> f64 {
    let mut sink = 0.0f64;
    sink += f(); // warmup + compile
    let t0 = Instant::now();
    sink += f();
    let per = t0.elapsed().as_secs_f64().max(1e-9);
    // ~150ms target, capped (mirrors bench.mjs, incl. its wasm leak cap)
    let leak_cap = (250e6 / (4.0 * 8.0 * (n * n) as f64)).floor() as usize;
    let iters = ((0.15 / per).ceil() as usize).clamp(3, 500.min(leak_cap.max(3)));
    let t0 = Instant::now();
    for _ in 0..iters {
        sink += f();
    }
    let ns = t0.elapsed().as_secs_f64() * 1e9 / iters as f64;
    assert!(sink.is_finite());
    ns
}

fn main() {
    let label = std::env::args().nth(1).unwrap_or_else(|| "native-o3".into());
    // (the blas determinism-bits modes live in the blas.wasm repo.)
    let ops: &[(&str, usize, extern "C" fn() -> f64)] = &[
        ("matmul", 256, bench_harness::run_matmul),
        ("lu_solve", 256, bench_harness::run_lu_solve),
        ("qr", 256, bench_harness::run_qr),
        ("svd", 256, bench_harness::run_svd),
        ("sa_evd", 256, bench_harness::run_sa_evd),
        ("gen_evd", 128, bench_harness::run_gen_evd),
        ("matmul_c64", 256, bench_harness::run_matmul_c64),
        ("lu_solve_c64", 256, bench_harness::run_lu_solve_c64),
        ("qr_c64", 256, bench_harness::run_qr_c64),
    ];
    for &n in &[32usize, 64, 128, 256] {
        bench_harness::setup(n);
        for &(name, max_n, f) in ops {
            if n > max_n {
                continue;
            }
            let ns = time_op(move || f(), n);
            println!(r#"{{"target":"{label}","op":"{name}","n":{n},"ns":{ns:.1}}}"#);
        }
        // factor-only baselines with library-default blocking, for the
        // tuning comparison (0 = default; mirrors tune.mjs)
        let ns = time_op(|| bench_harness::run_lu_factor_tuned(0, 0), n);
        println!(r#"{{"target":"{label}","op":"lu_factor","n":{n},"ns":{ns:.1}}}"#);
        let ns = time_op(|| bench_harness::run_qr_factor_tuned(0, 0), n);
        println!(r#"{{"target":"{label}","op":"qr_factor","n":{n},"ns":{ns:.1}}}"#);
    }
    // Schur campaign rows (2026-07-11): the kernel full-Schur pipeline and
    // the faer-schur baseline to n=1024 — the native side of
    // schur-largen.mjs's wasm-vs-native ratio
    for &n in &[64usize, 128, 256, 512, 1024] {
        bench_harness::setup(n);
        let ns = time_op(|| bench_harness::run_schur_k(), n);
        println!(r#"{{"target":"{label}","op":"schur_k","n":{n},"ns":{ns:.1}}}"#);
        let ns = time_op(|| bench_harness::run_schur(), n);
        println!(r#"{{"target":"{label}","op":"schur_faer","n":{n},"ns":{ns:.1}}}"#);
        let ns = time_op(|| bench_harness::run_schur_c64_k(), n);
        println!(r#"{{"target":"{label}","op":"schur_c64_k","n":{n},"ns":{ns:.1}}}"#);
        let ns = time_op(|| bench_harness::run_schur_c64(), n);
        println!(r#"{{"target":"{label}","op":"schur_c64_faer","n":{n},"ns":{ns:.1}}}"#);
    }
}
