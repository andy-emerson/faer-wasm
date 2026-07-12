//! Regression guard for the c64 large-n wasm memory hazard (found
//! 2026-07-11): faer's c64 matmul allocates per-call temporaries through
//! the global allocator, and one c64 multishift call at n=600 was measured
//! at 15.4 GB cumulative across ~25K allocations — fatal on a leak-only
//! bump allocator, harmless on any freeing one because PEAK LIVE stays
//! tiny (~19 MB). The wasm shims now use a LIFO-rewind bump; this test
//! guards the property that makes that fix sufficient: if a faer re-pin
//! ever starts HOLDING those temporaries (peak live grows), the wasm
//! hazard returns silently and this fails first.
use core::sync::atomic::{AtomicUsize, Ordering};
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::Mutex;

// The counters below are process-global, so the two tests MUST NOT run
// concurrently or their measurement windows pollute each other (found in
// the 2026-07-12 sweep: deltas swung 5 MB ↔ 300 MB with the harness's
// default parallel test threads). Each test holds this for its duration.
static SERIAL: Mutex<()> = Mutex::new(());

static TOTAL: AtomicUsize = AtomicUsize::new(0);
static COUNT: AtomicUsize = AtomicUsize::new(0);
static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);
static BIGGEST: AtomicUsize = AtomicUsize::new(0);

struct Counting;
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        TOTAL.fetch_add(l.size(), Ordering::Relaxed);
        COUNT.fetch_add(1, Ordering::Relaxed);
        let live = LIVE.fetch_add(l.size(), Ordering::Relaxed) + l.size();
        PEAK.fetch_max(live, Ordering::Relaxed);
        BIGGEST.fetch_max(l.size(), Ordering::Relaxed);
        System.alloc(l)
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        LIVE.fetch_sub(l.size(), Ordering::Relaxed);
        System.dealloc(p, l)
    }
}
#[global_allocator]
static A: Counting = Counting;

#[test]
fn count_complex_multishift_allocs() {
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    use faer::linalg::evd::schur::{self, complex_schur};
    use faer::dyn_stack::{MemBuffer, MemStack};
    use faer::{c64, Auto, Mat, Par};

    let n = 600usize;
    let mut s = 0x9E3779B97F4A7C15u64;
    let mut next = move || {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((s >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    };
    let a = Mat::from_fn(n, n, |_, _| c64::new(next(), next()));
    let mut h = a.clone();
    let mut tau = vec![c64::new(0.0, 0.0); n - 2];
    let mut work = vec![c64::new(0.0, 0.0); n];
    faer_wasm_kernels::hessenberg_cplx::hessenberg_cplx_factor_in_place(h.as_mut(), &mut tau, &mut work);
    for j in 0..n {
        for i in j + 2..n {
            h[(i, j)] = c64::new(0.0, 0.0);
        }
    }
    let params: schur::SchurParams = Auto::<c64>::auto();
    let mut w = faer::Col::<c64>::zeros(n);
    let mut mem = MemBuffer::new(schur::multishift_qr_scratch::<c64>(n, n, false, false, Par::Seq, params));

    let t0 = (TOTAL.load(Ordering::Relaxed), COUNT.load(Ordering::Relaxed));
    let (info, _, _) = complex_schur::multishift_qr::<c64>(
        false, h.as_mut(), None, w.as_mut(), 0, n, Par::Seq, MemStack::new(&mut mem), params,
    );
    let t1 = (TOTAL.load(Ordering::Relaxed), COUNT.load(Ordering::Relaxed));
    assert!(info == 0);
    let peak_mb = PEAK.load(Ordering::Relaxed) as f64 / 1048576.0;
    println!(
        "inside multishift: {:.1} MB total across {} allocations (biggest {:.1} MB, peak live {:.1} MB)",
        (t1.0 - t0.0) as f64 / 1048576.0,
        t1.1 - t0.1,
        BIGGEST.load(Ordering::Relaxed) as f64 / 1048576.0,
        peak_mb,
    );
    // the property the LIFO-rewind wasm allocator relies on: temporaries
    // are transient, so peak live memory stays far below the cumulative
    // traffic (n=600 measured ~19 MB live vs 15.4 GB cumulative)
    assert!(
        peak_mb < 256.0,
        "faer's c64 multishift now holds {peak_mb:.0} MB live — the wasm LIFO-rewind bump may no longer be sufficient"
    );
}

#[test]
fn count_f64_multishift_and_c64_matmul() {
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    use faer::linalg::evd::schur::{self, real_schur};
    use faer::linalg::matmul::matmul;
    use faer::dyn_stack::{MemBuffer, MemStack};
    use faer::{c64, Accum, Auto, Mat, Par};

    let n = 600usize;
    let mut s = 0x2545F4914F6CDD1Du64;
    let mut next = move || {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((s >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    };
    // f64 multishift (same shape as the c64 probe)
    let a = Mat::from_fn(n, n, |_, _| next());
    let mut h = a.clone();
    let mut tau = vec![0.0f64; n - 2];
    let mut work = vec![0.0f64; n];
    faer_wasm_kernels::hessenberg::hessenberg_factor_in_place(h.as_mut(), &mut tau, &mut work);
    for j in 0..n {
        for i in j + 2..n {
            h[(i, j)] = 0.0;
        }
    }
    let params: schur::SchurParams = Auto::<f64>::auto();
    let mut wr = faer::Col::<f64>::zeros(n);
    let mut wi = faer::Col::<f64>::zeros(n);
    let mut mem = MemBuffer::new(schur::multishift_qr_scratch::<f64>(n, n, false, false, Par::Seq, params));
    let t0 = (TOTAL.load(Ordering::Relaxed), COUNT.load(Ordering::Relaxed));
    let (info, _, _) = real_schur::multishift_qr::<f64>(
        false, h.as_mut(), None, wr.as_mut(), wi.as_mut(), 0, n, Par::Seq, MemStack::new(&mut mem), params,
    );
    let t1 = (TOTAL.load(Ordering::Relaxed), COUNT.load(Ordering::Relaxed));
    assert!(info == 0);
    let f64_mb = (t1.0 - t0.0) as f64 / 1048576.0;
    println!(
        "f64 multishift n=600: {f64_mb:.1} MB across {} allocations",
        t1.1 - t0.1
    );
    // regression guard (2026-07-12 sweep — this test previously only
    // printed, and the two tests raced on the global counters): measured
    // 5.2 MB / 903 allocations serialized; the c64 pathology class it
    // must catch was 15,400 MB. 64 MB splits them with 12× headroom.
    assert!(
        f64_mb < 64.0,
        "f64 multishift cumulative allocation grew to {f64_mb:.1} MB"
    );

    // one c64 matmul at a sweep-ish shape
    let ac = Mat::from_fn(96, 96, |_, _| c64::new(next(), next()));
    let bc = Mat::from_fn(96, 504, |_, _| c64::new(next(), next()));
    let mut cc = Mat::<c64>::zeros(96, 504);
    let t0 = (TOTAL.load(Ordering::Relaxed), COUNT.load(Ordering::Relaxed));
    matmul(cc.as_mut(), Accum::Replace, ac.as_ref(), bc.as_ref(), c64::new(1.0, 0.0), Par::Seq);
    let t1 = (TOTAL.load(Ordering::Relaxed), COUNT.load(Ordering::Relaxed));
    let c64_mb = (t1.0 - t0.0) as f64 / 1048576.0;
    println!(
        "one c64 matmul (96x96)x(96x504): {c64_mb:.3} MB across {} allocations",
        t1.1 - t0.1
    );
    // measured 0.023 MB / 2 allocations serialized; the LIFO-rewind shims
    // absorb per-call temps only while they stay bounded per call
    assert!(
        c64_mb < 16.0,
        "c64 matmul per-call allocation grew to {c64_mb:.3} MB"
    );
}
