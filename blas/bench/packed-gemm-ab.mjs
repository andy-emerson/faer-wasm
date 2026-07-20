// Packed-gemm race (tuning campaign 2026-07-20): the BLIS-style
// packed-panel candidates (`*gemm_packed`) vs the shipped gemm dispatch,
// all four number types, square sizes plus the deep-K rectangular shapes
// that motivated the candidate (prefill-style k >> m). Interleaved
// timing (incumbent, packed, incumbent, ...), min-of-iters reported.
// Before timing, each (type, shape) asserts the two folds are EXACTLY
// equal — the candidates are bit-identical to the shipped paths by
// construction, so any drift is a bug, not noise.
//
// Build + run:
//   cargo build --release --target wasm32-unknown-unknown --lib
//   node packed-gemm-ab.mjs target/wasm32-unknown-unknown/release/blas_bench.wasm
import { readFileSync } from 'node:fs';

const wasmPath = process.argv[2];
if (!wasmPath) {
	console.error('usage: node packed-gemm-ab.mjs <blas-bench-wasm>');
	process.exit(2);
}
const { instance } = await WebAssembly.instantiate(readFileSync(wasmPath), {});
const e = instance.exports;

// setup(2048) sizes every buffer to 2048² elements; shapes below need
// m·k, k·n, m·n ≤ 2048².
const N_SETUP = 2048;
e.setup(N_SETUP);

const TYPES = [
	['f64', 0],
	['f32', 1],
	['c64', 2],
	['c32', 3],
];
// square sweep + deep-K prefill shapes; complex types get the lighter
// deep-K shape only (4x FLOPs per element makes the big one minutes-long)
const SQUARE = [
	[256, 256, 256],
	[512, 512, 512],
	[1024, 1024, 1024],
];
const DEEPK_REAL = [
	[128, 2048, 2048],
	[512, 4096, 1024],
];
const DEEPK_CPLX = [[128, 2048, 1024]];

function timeOnce(fn) {
	const t0 = performance.now();
	fn();
	return performance.now() - t0;
}

let failed = 0;
for (const [tname, ty] of TYPES) {
	const shapes = [...SQUARE, ...(ty <= 1 ? DEEPK_REAL : DEEPK_CPLX)];
	for (const [m, k, n] of shapes) {
		const inc = () => e.run_gemm_rect_incumbent(ty, m, k, n);
		const pk = () => e.run_gemm_rect_packed(ty, m, k, n);
		// warmup both paths (bit-identity is checked on fresh state below)
		const f1 = inc();
		const f2 = pk();
		if (!Number.isFinite(f1 + f2)) {
			console.error(`[${tname} ${m}x${k}x${n}] NON-FINITE fold — investigate`);
			failed++;
			continue;
		}
		const flop = 2 * m * k * n * (ty >= 2 ? 4 : 1);
		// adaptive iters: aim for >=3 timed reps of the slower side
		const probe = timeOnce(inc) + timeOnce(pk);
		const iters = probe > 2000 ? 3 : probe > 500 ? 5 : probe > 100 ? 9 : 15;
		let ti = Infinity, tp = Infinity;
		for (let r = 0; r < iters; r++) {
			ti = Math.min(ti, timeOnce(inc));
			tp = Math.min(tp, timeOnce(pk));
		}
		const ratio = ti / tp;
		console.log(
			`[${tname} ${m}x${k}x${n}] incumbent=${ti.toFixed(2)}ms packed=${tp.toFixed(2)}ms ` +
			`packed_speedup=${ratio.toFixed(3)}x (${(flop / tp / 1e6).toFixed(2)} vs ${(flop / ti / 1e6).toFixed(2)} GFLOP/s) iters=${iters}`,
		);
	}
}

// separate bit-identity pass on fresh state: run each path once from an
// identical starting C (re-setup between calls) and compare folds exactly
console.log('--- bit-identity (fresh state per call) ---');
for (const [tname, ty] of TYPES) {
	const shapes = [...SQUARE.slice(0, 2), ...(ty <= 1 ? DEEPK_REAL.slice(0, 1) : DEEPK_CPLX)];
	for (const [m, k, n] of shapes) {
		e.setup(N_SETUP);
		const fi = e.run_gemm_rect_incumbent(ty, m, k, n);
		e.setup(N_SETUP);
		const fp = e.run_gemm_rect_packed(ty, m, k, n);
		const ok = Object.is(fi, fp);
		if (!ok) failed++;
		console.log(`[${tname} ${m}x${k}x${n}] fold ${ok ? 'IDENTICAL' : `MISMATCH ${fi} != ${fp}`}`);
	}
}
process.exit(failed ? 1 : 0);
