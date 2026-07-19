// Level-2 roofline scoreboard for the shipped BLAS layer (blas/ crate):
// times each L2 function over the n×n state and scores achieved GB/s
// against the fastest same-run stream (see l1-roofline.mjs for why a
// single triad number under-caps read-modify-write mixes). Also
// verifies the Level-2 cross-target determinism probes against a file
// of native bit patterns (from `cargo run --release --bin native
// l2-bits`).
//
// Lives beside the layer it measures (blas/bench). Build + run:
//   cargo build --release --target wasm32-unknown-unknown --lib
//   cargo run --release --bin native l2-bits[-f32|-z] > bits.txt
//   node l2-roofline.mjs target/wasm32-unknown-unknown/release/blas_bench.wasm bits.txt [--f32|--c64]
import { readFileSync } from 'node:fs';

const wasmPath = process.argv[2];
if (!wasmPath) {
	console.error('usage: node l2-roofline.mjs <blas-bench-wasm> [native-bits-file] [--f32]');
	process.exit(2);
}
const bytes = readFileSync(wasmPath);
const { instance } = await WebAssembly.instantiate(bytes, {});
const e = instance.exports;

// --f32 anywhere in argv: score the f32 layer (same recipes, *_f32
// exports, 4-byte elements; the bandwidth ceiling is bytes-agnostic).
const F32 = process.argv.includes('--f32');
// --c64 / --c32: score a complex layer (*_z / *_c exports; gemv
// gains the conjugate-transpose row, ger splits u/c, symv -> hemv).
const C64F = process.argv.includes('--c64');
const C32F = process.argv.includes('--c32');
const CPLX = C64F || C32F;
const sfx = C64F ? '_z' : C32F ? '_c' : F32 ? '_f32' : '';
const EB = C64F ? 16 : F32 ? 4 : 8;

// ---- determinism probes first
const probeNames = CPLX
	? ['gemv', 'gemv_t', 'gemv_c', 'geru', 'gerc', 'hemv', 'trmv', 'trsv', 'her', 'her2']
	: ['gemv', 'gemv_t', 'ger', 'symv', 'trmv', 'trsv', 'syr', 'syr2'];
const wasmBits = probeNames.map((_, op) => {
	const buf = new DataView(new ArrayBuffer(8));
	buf.setFloat64(0, e['run_l2_probe' + sfx](op));
	return buf.getBigUint64(0).toString(16).padStart(16, '0');
});
console.log('## L2 determinism probes (LCG 257x257, bits)');
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

// ---- roofline rows (n=2048: matrices stream from DRAM)
const N = 2048;
e.setup(N);

// op index -> [name, bytes moved per call]
// gemv/gemv_t read all of A (8n²); symmetric/triangular ops touch half
// the matrix (4n² read; +4n² write-back for the rank updates).
const OPS = CPLX ? [
	['gemv', EB * N * N],
	['gemv_t', EB * N * N],
	['gemv_c', EB * N * N],
	['geru', 2 * EB * N * N],
	['gerc', 2 * EB * N * N],
	['hemv', (EB / 2) * N * N],
	['trmv', (EB / 2) * N * N],
	['trsv', (EB / 2) * N * N],
	['her', EB * N * N],
	['her2', EB * N * N],
] : [
	['gemv', EB * N * N],
	['gemv_t', EB * N * N],
	['ger', 2 * EB * N * N],
	['symv', (EB / 2) * N * N],
	['trmv', (EB / 2) * N * N],
	['trsv', (EB / 2) * N * N],
	['syr', EB * N * N],
	['syr2', EB * N * N],
];
const rows = [];
for (let op = 0; op < OPS.length; op++) {
	const [name, bytesMoved] = OPS[op];
	let sink = e['run_l2_layer' + sfx](op); // warm + compile
	if (!Number.isFinite(sink)) throw new Error(`${name}: non-finite`);
	let best = Infinity;
	for (let r = 0; r < 5; r++) {
		const it = 8;
		const t0 = performance.now();
		for (let i = 0; i < it; i++) sink += e['run_l2_layer' + sfx](op);
		best = Math.min(best, (performance.now() - t0) / it);
	}
	if (!Number.isFinite(sink)) throw new Error(`${name}: non-finite`);
	rows.push([name, best, bytesMoved / (best / 1e3) / 1e9]);
}

// triad AFTER the op rows: run_ceiling_bw sacrifices sym as its
// destination, and symv/ger/syr/syr2 want sym's real values first
const ceilOnce = () => {
	let s = e.run_ceiling_bw();
	const t0 = performance.now();
	const it = 8;
	for (let i = 0; i < it; i++) s += e.run_ceiling_bw();
	if (!Number.isFinite(s)) throw new Error('ceiling non-finite');
	return (3 * 8 * N * N * it) / ((performance.now() - t0) / 1e3) / 1e9;
};
const triadCeil = Math.max(ceilOnce(), ceilOnce(), ceilOnce());
console.log(`\ntriad bandwidth (same run): ${triadCeil.toFixed(1)} GB/s`);
const ceil = Math.max(triadCeil, ...rows.map((r) => r[2]));
console.log(`scoring ceiling (fastest same-run stream): ${ceil.toFixed(1)} GB/s\n`);
console.log('| op | ms/call | GB/s | % of ceiling |');
console.log('| - | -: | -: | -: |');
for (const [name, best, gbs] of rows) {
	console.log(
		`| ${name} | ${best.toFixed(3)} | ${gbs.toFixed(1)} | ${((100 * gbs) / ceil).toFixed(0)}% |`,
	);
}
