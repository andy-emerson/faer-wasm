// Level-1 roofline scoreboard for the shipped BLAS layer (blas/ crate):
// times each L1 function streaming the n×n state column-by-column and
// scores achieved GB/s against the SAME-RUN bandwidth ceiling (triad),
// per the re-derived success metric. Also verifies the cross-target
// determinism probes: pass a file of native bit patterns (from
// `cargo run --release --bin native l1-bits`) to require wasm bits to
// match exactly.
//
// Lives beside the layer it measures (blas/bench). Build + run:
//   cargo build --release --target wasm32-unknown-unknown --lib
//   cargo run --release --bin native l1-bits[-f32|-z] > bits.txt
//   node l1-roofline.mjs target/wasm32-unknown-unknown/release/blas_bench.wasm bits.txt [--f32|--c64]
import { readFileSync } from 'node:fs';

const wasmPath = process.argv[2];
if (!wasmPath) {
	console.error('usage: node l1-roofline.mjs <blas-bench-wasm> [native-bits-file] [--f32]');
	process.exit(2);
}
const bytes = readFileSync(wasmPath);
const { instance } = await WebAssembly.instantiate(bytes, {});
const e = instance.exports;

// --f32 anywhere in argv: score the f32 layer (same recipes, *_f32
// exports, 4-byte elements; the bandwidth ceiling is bytes-agnostic).
const F32 = process.argv.includes('--f32');
// --c64 / --c32: score a complex layer (*_z / *_c exports; dot
// splits into dotu/dotc, scal gains the real-alpha dscal row).
const C64F = process.argv.includes('--c64');
const C32F = process.argv.includes('--c32');
const CPLX = C64F || C32F;
const sfx = C64F ? '_z' : C32F ? '_c' : F32 ? '_f32' : '';
const EB = C64F ? 16 : F32 ? 4 : 8;

// ---- determinism probes first (cheap, and a failure should kill the run)
const probeNames = CPLX
	? ['dotu', 'dotc', 'nrm2', 'asum', 'iamax']
	: ['dot', 'asum', 'nrm2', 'iamax'];
const wasmBits = probeNames.map((_, op) => {
	const buf = new DataView(new ArrayBuffer(8));
	buf.setFloat64(0, e['run_l1_probe' + sfx](op));
	return buf.getBigUint64(0).toString(16).padStart(16, '0');
});
console.log('## determinism probes (LCG len=1001, bits)');
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

// ---- roofline rows
// n=2048: 32 MB per matrix, so every op streams from DRAM rather than
// last-level cache — the ceiling (a 96 MB triad) and the ops then live
// in the same memory regime.
const N = 2048;
e.setup(N);

// same-run bandwidth ceiling (triad over the state, 3*8*n^2 bytes/call)
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

// op index -> [name, bytes moved per call over the n^2 elements]
const OPS = CPLX ? [
	['copy', 2 * EB * N * N],
	['swap', 4 * EB * N * N],
	['scal', 2 * EB * N * N],
	['dscal', 2 * EB * N * N],
	['axpy', 3 * EB * N * N],
	['rot', 4 * EB * N * N],
	['dotu', 2 * EB * N * N],
	['dotc', 2 * EB * N * N],
	['nrm2', EB * N * N],
	['asum', EB * N * N],
	['iamax', EB * N * N],
] : [
	['copy', 2 * EB * N * N],
	['swap', 4 * EB * N * N],
	['scal', 2 * EB * N * N],
	['axpy', 3 * EB * N * N],
	['rot', 4 * EB * N * N],
	['dot', 2 * EB * N * N],
	['nrm2', EB * N * N],
	['asum', EB * N * N],
	// iamax reads the input twice (value pass + index rescan) but only
	// 8n^2 of it is mandatory — the score is deliberately conservative
	['iamax', EB * N * N],
];
const rows = [];
for (let op = 0; op < OPS.length; op++) {
	const [name, bytesMoved] = OPS[op];
	let sink = e['run_l1_layer' + sfx](op); // warm + compile
	if (!Number.isFinite(sink)) throw new Error(`${name}: non-finite`);
	let best = Infinity;
	for (let r = 0; r < 5; r++) {
		const it = 8;
		const t0 = performance.now();
		for (let i = 0; i < it; i++) sink += e['run_l1_layer' + sfx](op);
		best = Math.min(best, (performance.now() - t0) / it);
	}
	if (!Number.isFinite(sink)) throw new Error(`${name}: non-finite`);
	rows.push([name, best, bytesMoved / (best / 1e3) / 1e9]);
}

// The scoring ceiling is the fastest stream observed IN THIS RUN (triad
// or any op): read/write mixes overlap writebacks differently, so a
// single triad number under-caps read-modify-write ops — STREAM reports
// Copy/Scale/Add/Triad separately for the same reason. Self-calibrating
// per machine, same-run only.
const ceil = Math.max(triadCeil, ...rows.map((r) => r[2]));
console.log(`scoring ceiling (fastest same-run stream): ${ceil.toFixed(1)} GB/s\n`);
console.log('| op | ms/call | GB/s | % of ceiling |');
console.log('| - | -: | -: | -: |');
for (const [name, best, gbs] of rows) {
	console.log(
		`| ${name} | ${best.toFixed(3)} | ${gbs.toFixed(1)} | ${((100 * gbs) / ceil).toFixed(0)}% |`,
	);
}
