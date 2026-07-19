//! blas-bench — the BLAS layer's own measurement harness: the roofline
//! rows, the cross-target determinism probes, and the machine-ceiling
//! probes, for both number types. Self-contained: depends only on
//! faer-wasm-blas (no faer), so it builds in seconds and the blas/
//! folder is the complete product — code (../src), correctness
//! (../tests), measurement (here). Scripts: `l{1,2,3}-roofline.mjs`
//! in this folder; scoreboard: ../tests/README.md.
//!
//! State is plain column-major Vecs with cs = n (no padding). The
//! determinism probes are self-contained (own LCG, own matrices) and
//! reproduce the bench-harness probe bit patterns exactly — verified
//! against the step-7/9/10 runner logs at the move. The faer market
//! races stay in ../../bench (they need faer): `gemm-tune-ab.mjs`
//! loads both wasm modules side by side.

struct State {
    n: usize,
    a: Vec<f64>,
    b: Vec<f64>,
    // sacrificial destination (triad target + L2/L3 mutation target)
    sym: Vec<f64>,
    // a's values with a dominant diagonal: solves stay bounded
    tri: Vec<f64>,
    rhs: Vec<f64>,
    a32: Vec<f32>,
    b32: Vec<f32>,
    sym32: Vec<f32>,
    tri32: Vec<f32>,
    rhs32: Vec<f32>,
    // c64 twins — own LCG fills (re/im interleaved draws), same roles
    az: Vec<C64>,
    bz: Vec<C64>,
    symz: Vec<C64>,
    triz: Vec<C64>,
    rhsz: Vec<C64>,
}

use faer_wasm_blas::C64;

struct StateCell(core::cell::UnsafeCell<Option<State>>);
unsafe impl Sync for StateCell {}
static STATE: StateCell = StateCell(core::cell::UnsafeCell::new(None));

fn state_mut() -> &'static mut State {
    unsafe { (*STATE.0.get()).as_mut().expect("call setup(n) first") }
}

// deterministic fill (splitmix-style LCG), values in [-1, 1] —
// same recipe and seeds as the bench-harness state
fn fill(n: usize, mut s: u64) -> Vec<f64> {
    let mut v = vec![0.0f64; n * n];
    for j in 0..n {
        for i in 0..n {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            v[j * n + i] = ((s >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0;
        }
    }
    v
}

/// Column-major n×n fill from a probe's LCG closure (the order faer's
/// `Mat::from_fn` filled in — probe bits depend on it).
fn alloc_mat_f64(n: usize, next: &mut impl FnMut() -> f64) -> Vec<f64> {
    let mut v = vec![0.0f64; n * n];
    for j in 0..n {
        for i in 0..n {
            v[j * n + i] = next();
        }
    }
    v
}

fn alloc_mat_f32(n: usize, next: &mut impl FnMut() -> f32) -> Vec<f32> {
    let mut v = vec![0.0f32; n * n];
    for j in 0..n {
        for i in 0..n {
            v[j * n + i] = next();
        }
    }
    v
}

#[no_mangle]
pub extern "C" fn setup(n: usize) {
    let a = fill(n, 0x9E3779B97F4A7C15);
    let b = fill(n, 0xD1B54A32D192ED03);
    // symmetric, diagonally dominant (same construction as the
    // bench-harness state: a + aᵀ with a boosted diagonal)
    let mut sym = vec![0.0f64; n * n];
    for j in 0..n {
        for i in 0..n {
            sym[j * n + i] = a[j * n + i] + a[i * n + j];
        }
    }
    for i in 0..n {
        sym[i * n + i] += 2.0 * n as f64;
    }
    let mut tri = a.clone();
    for i in 0..n {
        tri[i * n + i] = 2.0 * n as f64 + 1.0;
    }
    let mut rhs = vec![0.0f64; n.max(1)];
    {
        let mut s = 0x853C49E6748FEA9Bu64;
        for v in rhs.iter_mut().take(n) {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            *v = ((s >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0;
        }
    }
    let a32: Vec<f32> = a.iter().map(|&v| v as f32).collect();
    let b32: Vec<f32> = b.iter().map(|&v| v as f32).collect();
    let sym32: Vec<f32> = sym.iter().map(|&v| v as f32).collect();
    let tri32: Vec<f32> = tri.iter().map(|&v| v as f32).collect();
    let rhs32: Vec<f32> = rhs.iter().map(|&v| v as f32).collect();
    let az = fill_z(n * n, 0x1B873593CC9E2D51);
    let bz = fill_z(n * n, 0xE6546B64B432C925);
    let symz = bz.clone(); // sacrificial destination, like sym
    let mut triz = az.clone();
    for i in 0..n {
        triz[i * n + i] = C64::new(2.0 * n as f64 + 1.0, 0.7);
    }
    let rhsz = fill_z(n.max(1), 0x2545F4914F6CDD1D);
    unsafe {
        *STATE.0.get() = Some(State {
            n,
            a,
            b,
            sym,
            tri,
            rhs,
            a32,
            b32,
            sym32,
            tri32,
            rhs32,
            az,
            bz,
            symz,
            triz,
            rhsz,
        })
    }
}

/// Deterministic complex fill: re then im from one LCG stream.
fn fill_z(len: usize, mut s: u64) -> Vec<C64> {
    let mut next = move || {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((s >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    };
    (0..len)
        .map(|_| {
            let re = next();
            let im = next();
            C64::new(re, im)
        })
        .collect()
}

/// Column-major n×n complex fill from a probe's LCG closure (re then
/// im per element, column-major order — probe bits depend on it).
fn alloc_mat_c64(n: usize, next: &mut impl FnMut() -> f64) -> Vec<C64> {
    let mut v = vec![C64::ZERO; n * n];
    for j in 0..n {
        for i in 0..n {
            let re = next();
            let im = next();
            v[j * n + i] = C64::new(re, im);
        }
    }
    v
}

#[no_mangle]
pub extern "C" fn run_l1_layer(op: usize) -> f64 {
    use faer_wasm_blas::L1 as l1;
    let s = state_mut();
    let n = s.n;
    let acs = s.n;
    let bcs = s.n;
    let ap = s.a.as_mut_ptr();
    let bp = s.b.as_mut_ptr();
    let col = |p: *mut f64, cs: usize, j: usize| unsafe {
        core::slice::from_raw_parts_mut(p.add(j * cs), n)
    };
    let mut sink = 0.0f64;
    match op {
        0 => {
            for j in 0..n {
                l1::dcopy(col(ap, acs, j), col(bp, bcs, j));
            }
            sink += unsafe { *bp };
        }
        1 => {
            for j in 0..n {
                l1::dswap(col(ap, acs, j), col(bp, bcs, j));
            }
            sink += unsafe { *ap };
        }
        2 => {
            for j in 0..n {
                l1::dscal(-1.0, col(ap, acs, j));
            }
            sink += unsafe { *ap };
        }
        3 => {
            for j in 0..n {
                l1::daxpy(0.001, col(ap, acs, j), col(bp, bcs, j));
            }
            sink += unsafe { *bp };
        }
        4 => {
            for j in 0..n {
                l1::drot(col(ap, acs, j), col(bp, bcs, j), 0.8, 0.6);
            }
            sink += unsafe { *ap };
        }
        5 => {
            for j in 0..n {
                sink += l1::ddot(col(ap, acs, j), col(bp, bcs, j));
            }
        }
        6 => {
            for j in 0..n {
                sink += l1::dnrm2(col(ap, acs, j));
            }
        }
        7 => {
            for j in 0..n {
                sink += l1::dasum(col(ap, acs, j));
            }
        }
        8 => {
            for j in 0..n {
                sink += l1::idamax(col(ap, acs, j)) as f64;
            }
        }
        _ => return f64::NAN,
    }
    sink
}


/// Level-2 roofline rows over the persistent state. op: 0 gemv,
/// 1 gemv_t, 2 ger, 3 symv, 4 trmv, 5 trsv, 6 syr, 7 syr2. Mutating
/// targets and constants chosen so values stay bounded across bench
/// iterations (small alpha on accumulating updates; trmv/trsv copy a
/// fresh x from b's first column each call — 8n bytes, noise next to
/// the 4·n²–16·n² the matrix stream moves).
#[no_mangle]
pub extern "C" fn run_l2_layer(op: usize) -> f64 {
    use faer_wasm_blas::L1 as l1;
    use faer_wasm_blas::L2 as l2;
    let s = state_mut();
    let n = s.n;
    let acs = s.n;
    let scs = s.n;
    let tcs = s.n;
    let a_len = if n == 0 { 0 } else { acs * (n - 1) + n };
    let s_len = if n == 0 { 0 } else { scs * (n - 1) + n };
    let t_len = if n == 0 { 0 } else { tcs * (n - 1) + n };
    let a = unsafe { core::slice::from_raw_parts(s.a.as_ptr(), a_len) };
    let bx = unsafe { core::slice::from_raw_parts(s.b.as_ptr(), n) }; // b's first column
    let sym = unsafe { core::slice::from_raw_parts_mut(s.sym.as_mut_ptr(), s_len) };
    let tri = unsafe { core::slice::from_raw_parts(s.tri.as_ptr(), t_len) };
    let y = unsafe { core::slice::from_raw_parts_mut(s.rhs.as_mut_ptr(), n) };
    match op {
        0 => l2::dgemv(0.001, n, n, a, acs, bx, 0.5, y),
        1 => l2::dgemv_t(0.001, n, n, a, acs, bx, 0.5, y),
        2 => l2::dger(0.0001, n, n, sym, scs, bx, bx),
        3 => l2::dsymv(0.001, n, sym, scs, true, bx, 0.5, y),
        4 => {
            l1::dcopy(bx, y);
            l2::dtrmv(n, tri, tcs, false, false, y);
        }
        5 => {
            l1::dcopy(bx, y);
            l2::dtrsv(n, tri, tcs, false, false, y);
        }
        6 => l2::dsyr(0.0001, n, sym, scs, true, bx),
        7 => l2::dsyr2(0.0001, n, sym, scs, true, bx, bx),
        _ => return f64::NAN,
    }
    y[0] + sym[0] + y[n - 1]
}

/// Level-2 cross-target determinism probes: fixed LCG-filled 257×257
/// matrix (odd size — tails everywhere), one op each, folded to one
/// value with the layer's own asum. op: 0 gemv, 1 gemv_t, 2 ger,
/// 3 symv, 4 trmv, 5 trsv, 6 syr, 7 syr2.
#[no_mangle]
pub extern "C" fn run_l2_probe(op: usize) -> f64 {
    use faer_wasm_blas::L1 as l1;
    use faer_wasm_blas::L2 as l2;
    const N: usize = 257;
    let mut s = 7u64;
    let mut next = || {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((s >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    };
    let mut a = alloc_mat_f64(N, &mut next);
    for i in 0..N {
        a[i * N + i] = 2.0 * N as f64 + 1.0; // solve-safe diagonal
    }
    let cs = N;
    let a_len = cs * (N - 1) + N;
    let x: [f64; N] = core::array::from_fn(|_| next());
    let mut y: [f64; N] = core::array::from_fn(|_| next());
    let am = &mut a[..a_len];
    match op {
        0 => l2::dgemv(0.7, N, N, am, cs, &x, 0.3, &mut y),
        1 => l2::dgemv_t(0.7, N, N, am, cs, &x, 0.3, &mut y),
        2 => l2::dger(0.7, N, N, am, cs, &x, &x),
        3 => l2::dsymv(0.7, N, am, cs, true, &x, 0.3, &mut y),
        4 => {
            y.copy_from_slice(&x);
            l2::dtrmv(N, am, cs, false, false, &mut y);
        }
        5 => {
            y.copy_from_slice(&x);
            l2::dtrsv(N, am, cs, false, false, &mut y);
        }
        6 => l2::dsyr(0.7, N, am, cs, true, &x),
        7 => l2::dsyr2(0.7, N, am, cs, true, &x, &x),
        _ => return f64::NAN,
    }
    let a_probe = &a[..N];
    l1::dasum(&y) + y[0] + y[N - 1] + l1::dasum(a_probe)
}


/// Level-3 roofline rows over the persistent state. op: 0 gemm,
/// 1 symm_left, 2 syrk, 3 syr2k, 4 trmm_left, 5 trsm_left,
/// 6 trmm_right, 7 trsm_right. The in-place triangular ops re-copy B
/// fresh from `b` each call (an O(n²) copy against O(n³) work), so
/// values never grow across iterations; accumulating ops use small
/// alpha with beta = 0.5. `sym` is the sacrificial destination
/// (already true for the triad probe); `tri` (dominant diagonal) keeps
/// the solves bounded.
#[no_mangle]
pub extern "C" fn run_l3_layer(op: usize) -> f64 {
    use faer_wasm_blas::L1 as l1;
    use faer_wasm_blas::L3 as l3;
    let s = state_mut();
    let n = s.n;
    let acs = s.n;
    let bcs = s.n;
    let scs = s.n;
    let tcs = s.n;
    let len = |cs: usize| if n == 0 { 0 } else { cs * (n - 1) + n };
    let a = unsafe { core::slice::from_raw_parts(s.a.as_ptr(), len(acs)) };
    let b = unsafe { core::slice::from_raw_parts(s.b.as_ptr(), len(bcs)) };
    let tri = unsafe { core::slice::from_raw_parts(s.tri.as_ptr(), len(tcs)) };
    let sym = unsafe { core::slice::from_raw_parts_mut(s.sym.as_mut_ptr(), len(scs)) };
    let refresh = |dst: &mut [f64]| {
        for j in 0..n {
            l1::dcopy(&b[j * bcs..j * bcs + n], &mut dst[j * scs..j * scs + n]);
        }
    };
    match op {
        0 => l3::dgemm(0.001, n, n, n, a, acs, b, bcs, 0.5, sym, scs),
        1 => l3::dsymm_left(0.001, n, n, tri, tcs, true, b, bcs, 0.5, sym, scs),
        2 => l3::dsyrk(0.001, n, n, a, acs, 0.5, sym, scs, true),
        3 => l3::dsyr2k(0.001, n, n, a, acs, b, bcs, 0.5, sym, scs, true),
        4 => {
            refresh(sym);
            l3::dtrmm_left(1.0, n, n, tri, tcs, false, false, sym, scs);
        }
        5 => {
            refresh(sym);
            l3::dtrsm_left(1.0, n, n, tri, tcs, false, false, sym, scs);
        }
        6 => {
            refresh(sym);
            l3::dtrmm_right(1.0, n, n, tri, tcs, true, false, sym, scs);
        }
        7 => {
            refresh(sym);
            l3::dtrsm_right(1.0, n, n, tri, tcs, true, false, sym, scs);
        }
        _ => return f64::NAN,
    }
    sym[0] + sym[len(scs) - 1]
}

/// Tuning-campaign candidate row: the 4×4 register-tiled gemm on the
/// same state/constants as run_l3_layer(0) — bit-identical output,
/// raced for speed only.
#[no_mangle]
pub extern "C" fn run_l3_tuned_gemm() -> f64 {
    use faer_wasm_blas::L3 as l3;
    let s = state_mut();
    let n = s.n;
    let acs = s.n;
    let bcs = s.n;
    let scs = s.n;
    let len = |cs: usize| if n == 0 { 0 } else { cs * (n - 1) + n };
    let a = unsafe { core::slice::from_raw_parts(s.a.as_ptr(), len(acs)) };
    let b = unsafe { core::slice::from_raw_parts(s.b.as_ptr(), len(bcs)) };
    let sym = unsafe { core::slice::from_raw_parts_mut(s.sym.as_mut_ptr(), len(scs)) };
    l3::dgemm_tiled(0.001, n, n, n, a, acs, b, bcs, 0.5, sym, scs);
    sym[0] + sym[len(scs) - 1]
}

/// Same, for the 4-column fused candidate.
#[no_mangle]
pub extern "C" fn run_l3_col4_gemm() -> f64 {
    use faer_wasm_blas::L3 as l3;
    let s = state_mut();
    let n = s.n;
    let acs = s.n;
    let bcs = s.n;
    let scs = s.n;
    let len = |cs: usize| if n == 0 { 0 } else { cs * (n - 1) + n };
    let a = unsafe { core::slice::from_raw_parts(s.a.as_ptr(), len(acs)) };
    let b = unsafe { core::slice::from_raw_parts(s.b.as_ptr(), len(bcs)) };
    let sym = unsafe { core::slice::from_raw_parts_mut(s.sym.as_mut_ptr(), len(scs)) };
    l3::dgemm_col4(0.001, n, n, n, a, acs, b, bcs, 0.5, sym, scs);
    sym[0] + sym[len(scs) - 1]
}

/// Level-3 cross-target determinism probes: fixed LCG-filled 65×65
/// matrices (odd — tails everywhere), one op each, folded column-wise
/// with the layer's own asum. op order matches run_l3_layer plus
/// symm_right at 8.
#[no_mangle]
pub extern "C" fn run_l3_probe(op: usize) -> f64 {
    use faer_wasm_blas::L1 as l1;
    use faer_wasm_blas::L3 as l3;
    const N: usize = 65;
    const K: usize = 33;
    let mut st = 11u64;
    let mut next = || {
        st = st
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((st >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    };
    let mut a = alloc_mat_f64(N, &mut next);
    for i in 0..N {
        a[i * N + i] = 2.0 * N as f64 + 1.0;
    }
    let b = alloc_mat_f64(N, &mut next);
    let mut c = alloc_mat_f64(N, &mut next);
    let (acs, bcs, ccs) = (N, N, N);
    let len = |cs: usize| cs * (N - 1) + N;
    let av = &a[..len(acs)];
    let bv = &b[..len(bcs)];
    let cv = &mut c[..len(ccs)];
    match op {
        0 => l3::dgemm(0.7, N, K, N, av, acs, bv, bcs, 0.3, cv, ccs),
        1 => l3::dsymm_left(0.7, N, N, av, acs, true, bv, bcs, 0.3, cv, ccs),
        2 => l3::dsyrk(0.7, N, K, av, acs, 0.3, cv, ccs, true),
        3 => l3::dsyr2k(0.7, N, K, av, acs, bv, bcs, 0.3, cv, ccs, true),
        4 => l3::dtrmm_left(0.7, N, N, av, acs, false, false, cv, ccs),
        5 => l3::dtrsm_left(0.7, N, N, av, acs, false, false, cv, ccs),
        6 => l3::dtrmm_right(0.7, N, N, av, acs, true, false, cv, ccs),
        7 => l3::dtrsm_right(0.7, N, N, av, acs, true, false, cv, ccs),
        8 => l3::dsymm_right(0.7, N, N, av, acs, true, bv, bcs, 0.3, cv, ccs),
        _ => return f64::NAN,
    }
    let mut fold = 0.0;
    for j in 0..N {
        fold += l1::dasum(&cv[j * ccs..j * ccs + N]);
    }
    fold + cv[0] + cv[len(ccs) - 1]
}


/// Cross-target determinism probe: fixed LCG data (len 1001 — odd, so the
/// scalar tail runs too), one reduction per op. Native bin and wasm build
/// must return identical bits (the lane-emulation construction in
/// blas/src/lanes.rs). op: 0 dot, 1 asum, 2 nrm2, 3 iamax.
#[no_mangle]
pub extern "C" fn run_l1_probe(op: usize) -> f64 {
    use faer_wasm_blas::L1 as l1;
    const LEN: usize = 1001;
    let mut x = [0.0f64; LEN];
    let mut y = [0.0f64; LEN];
    let mut s = 42u64;
    let mut next = || {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((s >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    };
    for v in x.iter_mut() {
        *v = next();
    }
    for v in y.iter_mut() {
        *v = next();
    }
    match op {
        0 => l1::ddot(&x, &y),
        1 => l1::dasum(&x),
        2 => l1::dnrm2(&x),
        3 => l1::idamax(&x) as f64,
        _ => f64::NAN,
    }
}
/// Memory-bandwidth probe: triad c = a + 2.5·b over the n×n state
/// matrices. Bytes moved per call = 3·8·n² (read a, read b, write c).
#[no_mangle]
pub extern "C" fn run_ceiling_bw() -> f64 {
    // v2 (2026-07-18): the original version allocated + copied an n×n
    // matrix INSIDE the timed region (to_owned per call) and streamed c
    // twice — depressing the measured ceiling and miscounting traffic.
    // Now a pure single-pass triad sym ← a + 2.5·b over persistent
    // state: exactly 3·8·n² bytes per call, no allocation. (sym is
    // sacrificed as the destination — don't run symmetric-eigen benches
    // on the same instance after this probe.)
    let s = state_mut();
    let n = s.n;
    let acs = s.n;
    let bcs = s.n;
    let ccs = s.n;
    let ap = s.a.as_ptr();
    let bp = s.b.as_ptr();
    let cp = s.sym.as_mut_ptr();
    for j in 0..n {
        unsafe {
            triad(cp.add(j * ccs), ap.add(j * acs), bp.add(j * bcs), n);
        }
    }
    s.sym[0] + s.sym[(n - 1) * n + n - 1]
}

/// c ← a + 2.5·b, one pass, 2 lanes 2× unrolled.
#[cfg_attr(target_arch = "wasm32", target_feature(enable = "simd128"))]
unsafe fn triad(c: *mut f64, a: *const f64, b: *const f64, len: usize) {
    #[cfg(target_arch = "wasm32")]
    {
        use core::arch::wasm32::*;
        let k = f64x2_splat(2.5);
        let mut i = 0usize;
        while i + 4 <= len {
            let a0 = v128_load(a.add(i) as *const v128);
            let b0 = v128_load(b.add(i) as *const v128);
            let a1 = v128_load(a.add(i + 2) as *const v128);
            let b1 = v128_load(b.add(i + 2) as *const v128);
            v128_store(c.add(i) as *mut v128, f64x2_add(a0, f64x2_mul(b0, k)));
            v128_store(c.add(i + 2) as *mut v128, f64x2_add(a1, f64x2_mul(b1, k)));
            i += 4;
        }
        while i < len {
            *c.add(i) = *a.add(i) + *b.add(i) * 2.5;
            i += 1;
        }
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        for i in 0..len {
            *c.add(i) = *a.add(i) + *b.add(i) * 2.5;
        }
    }
}

/// Peak-arithmetic probe: register-resident mul+add chains, 8 independent
/// v128 accumulators × `iters` rounds. FLOPs per call = iters · 8 · 2 lanes
/// · 2 ops. Fused (one relaxed_madd = still 2 FLOPs) when the build has
/// relaxed-simd — so this probe measures the ceiling OF THE BUILD.
#[no_mangle]
pub extern "C" fn run_ceiling_flops(iters: usize) -> f64 {
    #[cfg(target_arch = "wasm32")]
    unsafe {
        ceiling_flops_imp(iters)
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        let _ = iters;
        f64::NAN
    }
}

#[cfg(target_arch = "wasm32")]
#[target_feature(enable = "simd128")]
unsafe fn ceiling_flops_imp(iters: usize) -> f64 {
    use core::arch::wasm32::*;
    let m = f64x2_splat(1.000000001);
    let a = f64x2_splat(1e-9);
    let mut acc = [f64x2_splat(1.0); 8];
    for _ in 0..iters {
        for k in 0..8 {
            #[cfg(target_feature = "relaxed-simd")]
            {
                acc[k] = f64x2_relaxed_madd(acc[k], m, a);
            }
            #[cfg(not(target_feature = "relaxed-simd"))]
            {
                acc[k] = f64x2_add(f64x2_mul(acc[k], m), a);
            }
        }
    }
    let mut s = 0.0;
    for k in 0..8 {
        s += f64x2_extract_lane::<0>(acc[k]) + f64x2_extract_lane::<1>(acc[k]);
    }
    s
}



// ============================================================
// f32 BLAS-layer rows and probes — one-for-one twins of the f64
// exports above (same ops, same constants, same probe recipes with
// values cast to f32), reading the *32 state fields. Extern returns
// stay f64 (the wasm ABI the scripts read); the f32 result is cast
// once at the end.
// ============================================================

#[no_mangle]
pub extern "C" fn run_l1_layer_f32(op: usize) -> f64 {
    use faer_wasm_blas::L1 as l1;
    let s = state_mut();
    let n = s.n;
    let acs = s.n;
    let bcs = s.n;
    let ap = s.a32.as_mut_ptr();
    let bp = s.b32.as_mut_ptr();
    let col = |p: *mut f32, cs: usize, j: usize| unsafe {
        core::slice::from_raw_parts_mut(p.add(j * cs), n)
    };
    let mut sink = 0.0f64;
    match op {
        0 => {
            for j in 0..n {
                l1::scopy(col(ap, acs, j), col(bp, bcs, j));
            }
            sink += unsafe { *bp } as f64;
        }
        1 => {
            for j in 0..n {
                l1::sswap(col(ap, acs, j), col(bp, bcs, j));
            }
            sink += unsafe { *ap } as f64;
        }
        2 => {
            for j in 0..n {
                l1::sscal(-1.0, col(ap, acs, j));
            }
            sink += unsafe { *ap } as f64;
        }
        3 => {
            for j in 0..n {
                l1::saxpy(0.001, col(ap, acs, j), col(bp, bcs, j));
            }
            sink += unsafe { *bp } as f64;
        }
        4 => {
            for j in 0..n {
                l1::srot(col(ap, acs, j), col(bp, bcs, j), 0.8, 0.6);
            }
            sink += unsafe { *ap } as f64;
        }
        5 => {
            for j in 0..n {
                sink += l1::sdot(col(ap, acs, j), col(bp, bcs, j)) as f64;
            }
        }
        6 => {
            for j in 0..n {
                sink += l1::snrm2(col(ap, acs, j)) as f64;
            }
        }
        7 => {
            for j in 0..n {
                sink += l1::sasum(col(ap, acs, j)) as f64;
            }
        }
        8 => {
            for j in 0..n {
                sink += l1::isamax(col(ap, acs, j)) as f64;
            }
        }
        _ => return f64::NAN,
    }
    sink
}

#[no_mangle]
pub extern "C" fn run_l2_layer_f32(op: usize) -> f64 {
    use faer_wasm_blas::L1 as l1;
    use faer_wasm_blas::L2 as l2;
    let s = state_mut();
    let n = s.n;
    let acs = s.n;
    let scs = s.n;
    let tcs = s.n;
    let a_len = if n == 0 { 0 } else { acs * (n - 1) + n };
    let s_len = if n == 0 { 0 } else { scs * (n - 1) + n };
    let t_len = if n == 0 { 0 } else { tcs * (n - 1) + n };
    let a = unsafe { core::slice::from_raw_parts(s.a32.as_ptr(), a_len) };
    let bx = unsafe { core::slice::from_raw_parts(s.b32.as_ptr(), n) };
    let sym = unsafe { core::slice::from_raw_parts_mut(s.sym32.as_mut_ptr(), s_len) };
    let tri = unsafe { core::slice::from_raw_parts(s.tri32.as_ptr(), t_len) };
    let y = unsafe { core::slice::from_raw_parts_mut(s.rhs32.as_mut_ptr(), n) };
    match op {
        0 => l2::sgemv(0.001, n, n, a, acs, bx, 0.5, y),
        1 => l2::sgemv_t(0.001, n, n, a, acs, bx, 0.5, y),
        2 => l2::sger(0.0001, n, n, sym, scs, bx, bx),
        3 => l2::ssymv(0.001, n, sym, scs, true, bx, 0.5, y),
        4 => {
            l1::scopy(bx, y);
            l2::strmv(n, tri, tcs, false, false, y);
        }
        5 => {
            l1::scopy(bx, y);
            l2::strsv(n, tri, tcs, false, false, y);
        }
        6 => l2::ssyr(0.0001, n, sym, scs, true, bx),
        7 => l2::ssyr2(0.0001, n, sym, scs, true, bx, bx),
        _ => return f64::NAN,
    }
    (y[0] + sym[0] + y[n - 1]) as f64
}

#[no_mangle]
pub extern "C" fn run_l3_layer_f32(op: usize) -> f64 {
    use faer_wasm_blas::L1 as l1;
    use faer_wasm_blas::L3 as l3;
    let s = state_mut();
    let n = s.n;
    let acs = s.n;
    let bcs = s.n;
    let scs = s.n;
    let tcs = s.n;
    let len = |cs: usize| if n == 0 { 0 } else { cs * (n - 1) + n };
    let a = unsafe { core::slice::from_raw_parts(s.a32.as_ptr(), len(acs)) };
    let b = unsafe { core::slice::from_raw_parts(s.b32.as_ptr(), len(bcs)) };
    let tri = unsafe { core::slice::from_raw_parts(s.tri32.as_ptr(), len(tcs)) };
    let sym = unsafe { core::slice::from_raw_parts_mut(s.sym32.as_mut_ptr(), len(scs)) };
    let refresh = |dst: &mut [f32]| {
        for j in 0..n {
            l1::scopy(&b[j * bcs..j * bcs + n], &mut dst[j * scs..j * scs + n]);
        }
    };
    match op {
        0 => l3::sgemm(0.001, n, n, n, a, acs, b, bcs, 0.5, sym, scs),
        1 => l3::ssymm_left(0.001, n, n, tri, tcs, true, b, bcs, 0.5, sym, scs),
        2 => l3::ssyrk(0.001, n, n, a, acs, 0.5, sym, scs, true),
        3 => l3::ssyr2k(0.001, n, n, a, acs, b, bcs, 0.5, sym, scs, true),
        4 => {
            refresh(sym);
            l3::strmm_left(1.0, n, n, tri, tcs, false, false, sym, scs);
        }
        5 => {
            refresh(sym);
            l3::strsm_left(1.0, n, n, tri, tcs, false, false, sym, scs);
        }
        6 => {
            refresh(sym);
            l3::strmm_right(1.0, n, n, tri, tcs, true, false, sym, scs);
        }
        7 => {
            refresh(sym);
            l3::strsm_right(1.0, n, n, tri, tcs, true, false, sym, scs);
        }
        _ => return f64::NAN,
    }
    (sym[0] + sym[len(scs) - 1]) as f64
}

/// f32 gemm dispatch-check rows (tiled vs col4 on the same state) —
/// the byte threshold is inherited from f64; these race it directly.
#[no_mangle]
pub extern "C" fn run_l3_tuned_gemm_f32() -> f64 {
    use faer_wasm_blas::L3 as l3;
    let s = state_mut();
    let n = s.n;
    let acs = s.n;
    let bcs = s.n;
    let scs = s.n;
    let len = |cs: usize| if n == 0 { 0 } else { cs * (n - 1) + n };
    let a = unsafe { core::slice::from_raw_parts(s.a32.as_ptr(), len(acs)) };
    let b = unsafe { core::slice::from_raw_parts(s.b32.as_ptr(), len(bcs)) };
    let sym = unsafe { core::slice::from_raw_parts_mut(s.sym32.as_mut_ptr(), len(scs)) };
    l3::sgemm_tiled(0.001, n, n, n, a, acs, b, bcs, 0.5, sym, scs);
    (sym[0] + sym[len(scs) - 1]) as f64
}

#[no_mangle]
pub extern "C" fn run_l3_col4_gemm_f32() -> f64 {
    use faer_wasm_blas::L3 as l3;
    let s = state_mut();
    let n = s.n;
    let acs = s.n;
    let bcs = s.n;
    let scs = s.n;
    let len = |cs: usize| if n == 0 { 0 } else { cs * (n - 1) + n };
    let a = unsafe { core::slice::from_raw_parts(s.a32.as_ptr(), len(acs)) };
    let b = unsafe { core::slice::from_raw_parts(s.b32.as_ptr(), len(bcs)) };
    let sym = unsafe { core::slice::from_raw_parts_mut(s.sym32.as_mut_ptr(), len(scs)) };
    l3::sgemm_col4(0.001, n, n, n, a, acs, b, bcs, 0.5, sym, scs);
    (sym[0] + sym[len(scs) - 1]) as f64
}

/// f32 L1 determinism probe — same LCG recipe as run_l1_probe with
/// values cast to f32. op: 0 dot, 1 asum, 2 nrm2, 3 iamax.
#[no_mangle]
pub extern "C" fn run_l1_probe_f32(op: usize) -> f64 {
    use faer_wasm_blas::L1 as l1;
    const LEN: usize = 1001;
    let mut x = [0.0f32; LEN];
    let mut y = [0.0f32; LEN];
    let mut s = 42u64;
    let mut next = || {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (((s >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0) as f32
    };
    for v in x.iter_mut() {
        *v = next();
    }
    for v in y.iter_mut() {
        *v = next();
    }
    match op {
        0 => l1::sdot(&x, &y) as f64,
        1 => l1::sasum(&x) as f64,
        2 => l1::snrm2(&x) as f64,
        3 => l1::isamax(&x) as f64,
        _ => f64::NAN,
    }
}

/// f32 L2 determinism probes — same 257×257 recipe cast to f32.
#[no_mangle]
pub extern "C" fn run_l2_probe_f32(op: usize) -> f64 {
    use faer_wasm_blas::L1 as l1;
    use faer_wasm_blas::L2 as l2;
    const N: usize = 257;
    let mut s = 7u64;
    let mut next = || {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (((s >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0) as f32
    };
    let mut a = alloc_mat_f32(N, &mut next);
    for i in 0..N {
        a[i * N + i] = 2.0 * N as f32 + 1.0;
    }
    let cs = N;
    let a_len = cs * (N - 1) + N;
    let x: [f32; N] = core::array::from_fn(|_| next());
    let mut y: [f32; N] = core::array::from_fn(|_| next());
    let am = &mut a[..a_len];
    match op {
        0 => l2::sgemv(0.7, N, N, am, cs, &x, 0.3, &mut y),
        1 => l2::sgemv_t(0.7, N, N, am, cs, &x, 0.3, &mut y),
        2 => l2::sger(0.7, N, N, am, cs, &x, &x),
        3 => l2::ssymv(0.7, N, am, cs, true, &x, 0.3, &mut y),
        4 => {
            y.copy_from_slice(&x);
            l2::strmv(N, am, cs, false, false, &mut y);
        }
        5 => {
            y.copy_from_slice(&x);
            l2::strsv(N, am, cs, false, false, &mut y);
        }
        6 => l2::ssyr(0.7, N, am, cs, true, &x),
        7 => l2::ssyr2(0.7, N, am, cs, true, &x, &x),
        _ => return f64::NAN,
    }
    let a_probe = &a[..N];
    (l1::sasum(&y) + y[0] + y[N - 1] + l1::sasum(a_probe)) as f64
}

/// f32 L3 determinism probes — same 65×65 recipe cast to f32.
#[no_mangle]
pub extern "C" fn run_l3_probe_f32(op: usize) -> f64 {
    use faer_wasm_blas::L1 as l1;
    use faer_wasm_blas::L3 as l3;
    const N: usize = 65;
    const K: usize = 33;
    let mut st = 11u64;
    let mut next = || {
        st = st
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (((st >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0) as f32
    };
    let mut a = alloc_mat_f32(N, &mut next);
    for i in 0..N {
        a[i * N + i] = 2.0 * N as f32 + 1.0;
    }
    let b = alloc_mat_f32(N, &mut next);
    let mut c = alloc_mat_f32(N, &mut next);
    let (acs, bcs, ccs) = (N, N, N);
    let len = |cs: usize| cs * (N - 1) + N;
    let av = &a[..len(acs)];
    let bv = &b[..len(bcs)];
    let cv = &mut c[..len(ccs)];
    match op {
        0 => l3::sgemm(0.7, N, K, N, av, acs, bv, bcs, 0.3, cv, ccs),
        1 => l3::ssymm_left(0.7, N, N, av, acs, true, bv, bcs, 0.3, cv, ccs),
        2 => l3::ssyrk(0.7, N, K, av, acs, 0.3, cv, ccs, true),
        3 => l3::ssyr2k(0.7, N, K, av, acs, bv, bcs, 0.3, cv, ccs, true),
        4 => l3::strmm_left(0.7, N, N, av, acs, false, false, cv, ccs),
        5 => l3::strsm_left(0.7, N, N, av, acs, false, false, cv, ccs),
        6 => l3::strmm_right(0.7, N, N, av, acs, true, false, cv, ccs),
        7 => l3::strsm_right(0.7, N, N, av, acs, true, false, cv, ccs),
        8 => l3::ssymm_right(0.7, N, N, av, acs, true, bv, bcs, 0.3, cv, ccs),
        _ => return f64::NAN,
    }
    let mut fold = 0.0f32;
    for j in 0..N {
        fold += l1::sasum(&cv[j * ccs..j * ccs + N]);
    }
    (fold + cv[0] + cv[len(ccs) - 1]) as f64
}

/// f32 peak-arithmetic probe: f32x4 twin of run_ceiling_flops.
/// FLOPs per call = iters · 8 · 4 lanes · 2 ops.
#[no_mangle]
pub extern "C" fn run_ceiling_flops_f32(iters: usize) -> f64 {
    #[cfg(target_arch = "wasm32")]
    unsafe {
        ceiling_flops_f32_imp(iters)
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        let _ = iters;
        f64::NAN
    }
}

#[cfg(target_arch = "wasm32")]
#[target_feature(enable = "simd128")]
unsafe fn ceiling_flops_f32_imp(iters: usize) -> f64 {
    use core::arch::wasm32::*;
    let m = f32x4_splat(1.0000001);
    let a = f32x4_splat(1e-7);
    let mut acc = [f32x4_splat(1.0); 8];
    for _ in 0..iters {
        for k in 0..8 {
            #[cfg(target_feature = "relaxed-simd")]
            {
                acc[k] = f32x4_relaxed_madd(acc[k], m, a);
            }
            #[cfg(not(target_feature = "relaxed-simd"))]
            {
                acc[k] = f32x4_add(f32x4_mul(acc[k], m), a);
            }
        }
    }
    let mut s = 0.0f32;
    for k in 0..8 {
        s += f32x4_extract_lane::<0>(acc[k])
            + f32x4_extract_lane::<1>(acc[k])
            + f32x4_extract_lane::<2>(acc[k])
            + f32x4_extract_lane::<3>(acc[k]);
    }
    s as f64
}

// ============================================================
// c64 BLAS-layer rows and probes — twins of the f64 exports above
// over the *z state fields (16-byte elements; complex constants pick
// the same magnitudes). Extern returns stay f64; complex results are
// folded re+im at the end.
// ============================================================

#[no_mangle]
pub extern "C" fn run_l1_layer_z(op: usize) -> f64 {
    use faer_wasm_blas::L1 as l1;
    let s = state_mut();
    let n = s.n;
    let ap = s.az.as_mut_ptr();
    let bp = s.bz.as_mut_ptr();
    let col = |p: *mut C64, j: usize| unsafe { core::slice::from_raw_parts_mut(p.add(j * n), n) };
    let mut sink = 0.0f64;
    match op {
        0 => {
            for j in 0..n {
                l1::zcopy(col(ap, j), col(bp, j));
            }
            sink += unsafe { (*bp).re };
        }
        1 => {
            for j in 0..n {
                l1::zswap(col(ap, j), col(bp, j));
            }
            sink += unsafe { (*ap).re };
        }
        2 => {
            for j in 0..n {
                l1::zscal(C64::new(-1.0, 0.0), col(ap, j));
            }
            sink += unsafe { (*ap).re };
        }
        3 => {
            for j in 0..n {
                l1::zdscal(-1.0, col(ap, j));
            }
            sink += unsafe { (*ap).re };
        }
        4 => {
            for j in 0..n {
                l1::zaxpy(C64::new(0.001, 0.0), col(ap, j), col(bp, j));
            }
            sink += unsafe { (*bp).re };
        }
        5 => {
            for j in 0..n {
                l1::zdrot(col(ap, j), col(bp, j), 0.8, 0.6);
            }
            sink += unsafe { (*ap).re };
        }
        6 => {
            for j in 0..n {
                let d = l1::zdotu(col(ap, j), col(bp, j));
                sink += d.re + d.im;
            }
        }
        7 => {
            for j in 0..n {
                let d = l1::zdotc(col(ap, j), col(bp, j));
                sink += d.re + d.im;
            }
        }
        8 => {
            for j in 0..n {
                sink += l1::dznrm2(col(ap, j));
            }
        }
        9 => {
            for j in 0..n {
                sink += l1::dzasum(col(ap, j));
            }
        }
        10 => {
            for j in 0..n {
                sink += l1::izamax(col(ap, j)) as f64;
            }
        }
        _ => return f64::NAN,
    }
    sink
}

/// Level-2 c64 roofline rows. op: 0 gemv, 1 gemv_t, 2 gemv_c, 3 geru,
/// 4 gerc, 5 hemv, 6 trmv, 7 trsv, 8 her, 9 her2. Same
/// bounded-value discipline as the f64 rows (small alpha on
/// accumulating updates; trmv/trsv re-copy x fresh each call).
#[no_mangle]
pub extern "C" fn run_l2_layer_z(op: usize) -> f64 {
    use faer_wasm_blas::L1 as l1;
    use faer_wasm_blas::L2 as l2;
    let s = state_mut();
    let n = s.n;
    let cs = s.n;
    let len = if n == 0 { 0 } else { cs * (n - 1) + n };
    let a = unsafe { core::slice::from_raw_parts(s.az.as_ptr(), len) };
    let bx = unsafe { core::slice::from_raw_parts(s.bz.as_ptr(), n) };
    let sym = unsafe { core::slice::from_raw_parts_mut(s.symz.as_mut_ptr(), len) };
    let tri = unsafe { core::slice::from_raw_parts(s.triz.as_ptr(), len) };
    let y = unsafe { core::slice::from_raw_parts_mut(s.rhsz.as_mut_ptr(), n) };
    let al = C64::new(0.001, 0.0);
    let al4 = C64::new(0.0001, 0.0);
    let be = C64::new(0.5, 0.0);
    match op {
        0 => l2::zgemv(al, n, n, a, cs, bx, be, y),
        1 => l2::zgemv_t(al, n, n, a, cs, bx, be, y),
        2 => l2::zgemv_c(al, n, n, a, cs, bx, be, y),
        3 => l2::zgeru(al4, n, n, sym, cs, bx, bx),
        4 => l2::zgerc(al4, n, n, sym, cs, bx, bx),
        5 => l2::zhemv(al, n, sym, cs, true, bx, be, y),
        6 => {
            l1::zcopy(bx, y);
            l2::ztrmv(n, tri, cs, false, false, y);
        }
        7 => {
            l1::zcopy(bx, y);
            l2::ztrsv(n, tri, cs, false, false, y);
        }
        8 => l2::zher(0.0001, n, sym, cs, true, bx),
        9 => l2::zher2(al4, n, sym, cs, true, bx, bx),
        _ => return f64::NAN,
    }
    y[0].re + y[0].im + sym[0].re + y[n - 1].re
}

/// Level-3 c64 roofline rows. op: 0 gemm, 1 hemm_left, 2 herk,
/// 3 her2k, 4 trmm_left, 5 trsm_left, 6 trmm_right, 7 trsm_right.
#[no_mangle]
pub extern "C" fn run_l3_layer_z(op: usize) -> f64 {
    use faer_wasm_blas::L1 as l1;
    use faer_wasm_blas::L3 as l3;
    let s = state_mut();
    let n = s.n;
    let cs = s.n;
    let len = if n == 0 { 0 } else { cs * (n - 1) + n };
    let a = unsafe { core::slice::from_raw_parts(s.az.as_ptr(), len) };
    let b = unsafe { core::slice::from_raw_parts(s.bz.as_ptr(), len) };
    let tri = unsafe { core::slice::from_raw_parts(s.triz.as_ptr(), len) };
    let sym = unsafe { core::slice::from_raw_parts_mut(s.symz.as_mut_ptr(), len) };
    let refresh = |dst: &mut [C64]| {
        for j in 0..n {
            l1::zcopy(&b[j * cs..j * cs + n], &mut dst[j * cs..j * cs + n]);
        }
    };
    let al = C64::new(0.001, 0.0);
    let be = C64::new(0.5, 0.0);
    let one = C64::ONE;
    match op {
        0 => l3::zgemm(al, n, n, n, a, cs, b, cs, be, sym, cs),
        1 => l3::zhemm_left(al, n, n, tri, cs, true, b, cs, be, sym, cs),
        2 => l3::zherk(0.001, n, n, a, cs, 0.5, sym, cs, true),
        3 => l3::zher2k(al, n, n, a, cs, b, cs, 0.5, sym, cs, true),
        4 => {
            refresh(sym);
            l3::ztrmm_left(one, n, n, tri, cs, false, false, sym, cs);
        }
        5 => {
            refresh(sym);
            l3::ztrsm_left(one, n, n, tri, cs, false, false, sym, cs);
        }
        6 => {
            refresh(sym);
            l3::ztrmm_right(one, n, n, tri, cs, true, false, sym, cs);
        }
        7 => {
            refresh(sym);
            l3::ztrsm_right(one, n, n, tri, cs, true, false, sym, cs);
        }
        _ => return f64::NAN,
    }
    sym[0].re + sym[0].im + sym[len - 1].re
}

/// c64 cross-target determinism probes, L1: fixed LCG data (len 1001 —
/// odd, so the scalar tails run), one reduction per op. op: 0 dotu,
/// 1 dotc, 2 nrm2, 3 asum, 4 iamax.
#[no_mangle]
pub extern "C" fn run_l1_probe_z(op: usize) -> f64 {
    use faer_wasm_blas::L1 as l1;
    const LEN: usize = 1001;
    let mut s = 43u64;
    let mut next = || {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((s >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    };
    let mut mk = |_: usize| -> Vec<C64> {
        (0..LEN)
            .map(|_| {
                let re = next();
                let im = next();
                C64::new(re, im)
            })
            .collect()
    };
    let x = mk(0);
    let y = mk(1);
    match op {
        0 => {
            let d = l1::zdotu(&x, &y);
            d.re + d.im
        }
        1 => {
            let d = l1::zdotc(&x, &y);
            d.re + d.im
        }
        2 => l1::dznrm2(&x),
        3 => l1::dzasum(&x),
        4 => l1::izamax(&x) as f64,
        _ => f64::NAN,
    }
}

/// c64 L2 determinism probes: fixed LCG-filled 257×257 complex matrix,
/// one op each, folded with the layer's own dzasum. op: 0 gemv,
/// 1 gemv_t, 2 gemv_c, 3 geru, 4 gerc, 5 hemv, 6 trmv, 7 trsv,
/// 8 her, 9 her2.
#[no_mangle]
pub extern "C" fn run_l2_probe_z(op: usize) -> f64 {
    use faer_wasm_blas::L1 as l1;
    use faer_wasm_blas::L2 as l2;
    const N: usize = 257;
    let mut s = 8u64;
    let mut next = || {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((s >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    };
    let mut a = alloc_mat_c64(N, &mut next);
    for i in 0..N {
        a[i * N + i] = C64::new(2.0 * N as f64 + 1.0, 0.7); // solve-safe diagonal
    }
    let cs = N;
    let a_len = cs * (N - 1) + N;
    let x: Vec<C64> = (0..N)
        .map(|_| {
            let re = next();
            let im = next();
            C64::new(re, im)
        })
        .collect();
    let mut y: Vec<C64> = (0..N)
        .map(|_| {
            let re = next();
            let im = next();
            C64::new(re, im)
        })
        .collect();
    let am = &mut a[..a_len];
    let al = C64::new(0.7, -0.2);
    let be = C64::new(0.3, 0.1);
    match op {
        0 => l2::zgemv(al, N, N, am, cs, &x, be, &mut y),
        1 => l2::zgemv_t(al, N, N, am, cs, &x, be, &mut y),
        2 => l2::zgemv_c(al, N, N, am, cs, &x, be, &mut y),
        3 => l2::zgeru(al, N, N, am, cs, &x, &x),
        4 => l2::zgerc(al, N, N, am, cs, &x, &x),
        5 => l2::zhemv(al, N, am, cs, true, &x, be, &mut y),
        6 => {
            l1::zcopy(&x, &mut y);
            l2::ztrmv(N, am, cs, false, false, &mut y);
        }
        7 => {
            l1::zcopy(&x, &mut y);
            l2::ztrsv(N, am, cs, false, false, &mut y);
        }
        8 => l2::zher(0.7, N, am, cs, true, &x),
        9 => l2::zher2(al, N, am, cs, true, &x, &x),
        _ => return f64::NAN,
    }
    let a_probe = &a[..N];
    l1::dzasum(&y) + y[0].re + y[0].im + y[N - 1].re + l1::dzasum(a_probe)
}

/// c64 L3 determinism probes: fixed LCG-filled 65×65 complex matrices
/// (odd — tails everywhere), one op each. op order matches
/// run_l3_layer_z plus hemm_right at 8.
#[no_mangle]
pub extern "C" fn run_l3_probe_z(op: usize) -> f64 {
    use faer_wasm_blas::L1 as l1;
    use faer_wasm_blas::L3 as l3;
    const N: usize = 65;
    const K: usize = 33;
    let mut st = 12u64;
    let mut next = || {
        st = st
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((st >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    };
    let mut a = alloc_mat_c64(N, &mut next);
    for i in 0..N {
        a[i * N + i] = C64::new(2.0 * N as f64 + 1.0, 0.7);
    }
    let b = alloc_mat_c64(N, &mut next);
    let mut c = alloc_mat_c64(N, &mut next);
    let cs = N;
    let len = cs * (N - 1) + N;
    let av = &a[..len];
    let bv = &b[..len];
    let cv = &mut c[..len];
    let al = C64::new(0.7, -0.2);
    let be = C64::new(0.3, 0.1);
    match op {
        0 => l3::zgemm(al, N, K, N, av, cs, bv, cs, be, cv, cs),
        1 => l3::zhemm_left(al, N, N, av, cs, true, bv, cs, be, cv, cs),
        2 => l3::zherk(0.7, N, K, av, cs, 0.3, cv, cs, true),
        3 => l3::zher2k(al, N, K, av, cs, bv, cs, 0.3, cv, cs, true),
        4 => l3::ztrmm_left(al, N, N, av, cs, false, false, cv, cs),
        5 => l3::ztrsm_left(al, N, N, av, cs, false, false, cv, cs),
        6 => l3::ztrmm_right(al, N, N, av, cs, true, false, cv, cs),
        7 => l3::ztrsm_right(al, N, N, av, cs, true, false, cv, cs),
        8 => l3::zhemm_right(al, N, N, av, cs, true, bv, cs, be, cv, cs),
        _ => return f64::NAN,
    }
    let mut fold = 0.0;
    for j in 0..N {
        fold += l1::dzasum(&cv[j * cs..j * cs + N]);
    }
    fold + cv[0].re + cv[0].im + cv[len - 1].re
}
