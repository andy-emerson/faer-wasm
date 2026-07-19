// Level-3 roofline scoreboard for the shipped BLAS layer (blas/ crate):
// the multiply-class ops are scored in GFLOP/s against the machine's
// measured arithmetic peak (run_ceiling_flops) — the roofline axis that
// matters once an op does O(n³) work on O(n²) data. Also verifies the
// Level-3 cross-target determinism probes against native bit patterns
// (`cargo run --release --bin native l3-bits`).
//
// Lives beside the layer it measures (blas/bench). Build + run:
//   cargo build --release --target wasm32-unknown-unknown --lib
//   cargo run --release --bin native l3-bits[-f32|-z] > bits.txt
//   node l3-roofline.mjs target/wasm32-unknown-unknown/release/blas_bench.wasm bits.txt [--f32|--c64]
import { readFileSync } from 'node:fs';

const wasmPath = process.argv[2];
if (!wasmPath) {
	console.error('usage: node l3-roofline.mjs <blas-bench-wasm> [native-bits-file] [--f32]');
	process.exit(2);
}
const bytes = readFileSync(wasmPath);
const { instance } = await WebAssembly.instantiate(bytes, {});
const e = instance.exports;

// --f32 anywhere in argv: score the f32 layer (same recipes, *_f32
// exports, 4-byte elements; the bandwidth ceiling is bytes-agnostic).
const F32 = process.argv.includes('--f32');
// --c64 / --c32: score a complex layer (*_z / *_c exports; a complex
// multiply-add is 4x the real FLOPs, so the FLOP counts scale 4x; the
// ceiling is the same-precision REAL flops probe - complex arithmetic
// IS f64/f32 arithmetic).
const C64F = process.argv.includes('--c64');
const C32F = process.argv.includes('--c32');
const CPLX = C64F || C32F;
const sfx = C64F ? '_z' : C32F ? '_c' : F32 ? '_f32' : '';
const EB = C64F ? 16 : F32 ? 4 : 8;

// ---- determinism probes first
const probeNames = CPLX ? [
	'gemm',
	'hemm_left',
	'herk',
	'her2k',
	'trmm_left',
	'trsm_left',
	'trmm_right',
	'trsm_right',
	'hemm_right',
] : [
	'gemm',
	'symm_left',
	'syrk',
	'syr2k',
	'trmm_left',
	'trsm_left',
	'trmm_right',
	'trsm_right',
	'symm_right',
];
const wasmBits = probeNames.map((_, op) => {
	const buf = new DataView(new ArrayBuffer(8));
	buf.setFloat64(0, e['run_l3_probe' + sfx](op));
	return buf.getBigUint64(0).toString(16).padStart(16, '0');
});
console.log('## L3 determinism probes (LCG 65x65, bits)');
probeNames.forEach((name, i) => console.log(`${name}: ${wasmBits[i]}`));
const bitsFile = process.argv[3];
if (bitsFile) {
	const native = readFileSync(bitsFile, 'utf8').trim().split('\n');
	let ok = true;
	probeNames.forEach((name, i) => {
		if (native[i] !== wasmBits[i]) {
			console.error(`DETERMINISM FAIL ${name}: native ${native[i]} wasm ${wasmBits[i]}`);
			ok = false;
		}
	});
	if (!ok) process.exit(1);
	console.log(`native <-> wasm: bit-identical, all ${probeNames.length} probes`);
}

// ---- arithmetic ceiling (register-resident, same run)
e.setup(64); // any state works for the flops probe
const ceilSfx = F32 || C32F ? '_f32' : ''; // complex scores against its real peak
{
	e['run_ceiling_flops' + ceilSfx](1000); // compile warm
}
const LANES = F32 || C32F ? 4 : 2;
const flopsOnce = (iters) => {
	const t0 = performance.now();
	const s = e['run_ceiling_flops' + ceilSfx](iters);
	if (!Number.isFinite(s)) throw new Error('flops probe non-finite');
	return (iters * 8 * LANES * 2) / ((performance.now() - t0) / 1e3) / 1e9;
};
const peak = Math.max(flopsOnce(2_000_000), flopsOnce(2_000_000), flopsOnce(2_000_000));
console.log(`\narithmetic peak (register-resident, same run): ${peak.toFixed(1)} GFLOP/s`);

// ---- roofline rows (n=512: the shipping-regime size the A/B verdicts
// were rendered at; each op is O(n³))
const N = 512;
e.setup(N);
// op index -> [name, FLOPs per call]
const OPS = CPLX ? [
	['gemm', 8 * N * N * N],
	['hemm_left', 8 * N * N * N],
	['herk', 4 * N * N * (N + 1)],
	['her2k', 8 * N * N * (N + 1)],
	['trmm_left', 4 * N * N * (N + 1)],
	['trsm_left', 4 * N * N * (N + 1)],
	['trmm_right', 4 * N * N * (N + 1)],
	['trsm_right', 4 * N * N * (N + 1)],
] : [
	['gemm', 2 * N * N * N],
	['symm_left', 2 * N * N * N],
	['syrk', N * N * (N + 1)],
	['syr2k', 2 * N * N * (N + 1)],
	['trmm_left', N * N * (N + 1)],
	['trsm_left', N * N * (N + 1)],
	['trmm_right', N * N * (N + 1)],
	['trsm_right', N * N * (N + 1)],
];
console.log(`\n| op | ms/call | GFLOP/s | % of peak |`);
console.log('| - | -: | -: | -: |');
for (let op = 0; op < OPS.length; op++) {
	const [name, flops] = OPS[op];
	let sink = e['run_l3_layer' + sfx](op); // warm + compile
	if (!Number.isFinite(sink)) throw new Error(`${name}: non-finite`);
	let best = Infinity;
	for (let r = 0; r < 4; r++) {
		const it = 2;
		const t0 = performance.now();
		for (let i = 0; i < it; i++) sink += e['run_l3_layer' + sfx](op);
		best = Math.min(best, (performance.now() - t0) / it);
	}
	if (!Number.isFinite(sink)) throw new Error(`${name}: non-finite`);
	const gf = flops / (best / 1e3) / 1e9;
	console.log(
		`| ${name} | ${best.toFixed(3)} | ${gf.toFixed(2)} | ${((100 * gf) / peak).toFixed(0)}% |`,
	);
}
