// Level-3 roofline scoreboard for the shipped BLAS layer (blas/ crate):
// the multiply-class ops are scored in GFLOP/s against the machine's
// measured arithmetic peak (run_ceiling_flops) — the roofline axis that
// matters once an op does O(n³) work on O(n²) data. Also verifies the
// Level-3 cross-target determinism probes against native bit patterns
// (`cargo run --release --bin native l3-bits`).
//
//   node l3-roofline.mjs <bench-wasm> [native-bits-file]
import { readFileSync } from 'node:fs';

const wasmPath = process.argv[2];
if (!wasmPath) {
	console.error('usage: node l3-roofline.mjs <bench-wasm> [native-bits-file]');
	process.exit(2);
}
const bytes = readFileSync(wasmPath);
const { instance } = await WebAssembly.instantiate(bytes, {});
const e = instance.exports;

// ---- determinism probes first
const probeNames = [
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
	buf.setFloat64(0, e.run_l3_probe(op));
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
	console.log('native <-> wasm: bit-identical, all 9 probes');
}

// ---- arithmetic ceiling (register-resident, same run)
e.setup(64); // any state works for the flops probe
{
	e.run_ceiling_flops(1000); // compile warm
}
const flopsOnce = (iters) => {
	const t0 = performance.now();
	const s = e.run_ceiling_flops(iters);
	if (!Number.isFinite(s)) throw new Error('flops probe non-finite');
	return (iters * 8 * 2 * 2) / ((performance.now() - t0) / 1e3) / 1e9;
};
const peak = Math.max(flopsOnce(2_000_000), flopsOnce(2_000_000), flopsOnce(2_000_000));
console.log(`\narithmetic peak (register-resident, same run): ${peak.toFixed(1)} GFLOP/s`);

// ---- roofline rows (n=512: the shipping-regime size the A/B verdicts
// were rendered at; each op is O(n³))
const N = 512;
e.setup(N);
// op index -> [name, FLOPs per call]
const OPS = [
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
	let sink = e.run_l3_layer(op); // warm + compile
	if (!Number.isFinite(sink)) throw new Error(`${name}: non-finite`);
	let best = Infinity;
	for (let r = 0; r < 4; r++) {
		const it = 2;
		const t0 = performance.now();
		for (let i = 0; i < it; i++) sink += e.run_l3_layer(op);
		best = Math.min(best, (performance.now() - t0) / it);
	}
	if (!Number.isFinite(sink)) throw new Error(`${name}: non-finite`);
	const gf = flops / (best / 1e3) / 1e9;
	console.log(
		`| ${name} | ${best.toFixed(3)} | ${gf.toFixed(2)} | ${((100 * gf) / peak).toFixed(0)}% |`,
	);
}
