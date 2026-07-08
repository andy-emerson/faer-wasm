// Blocking-parameter sweep for the wasm build (Phase 3, "measure, don't
// assume"). Times factor-only LU and QR with explicit blocking parameters
// against the library defaults (config 0 = default).
//   node tune.mjs <wasm-path>
// Emits JSON lines: {"op":"lu","n":64,"cfg":"rt=64,bs=64","ns":...}
import { readFileSync } from 'node:fs';

const wasmPath = process.argv[2];
if (!wasmPath) {
	console.error('usage: node tune.mjs <wasm-path>');
	process.exit(2);
}
const bytes = readFileSync(wasmPath);

const SIZES = [32, 64, 128, 256];

async function timeCfg(n, exportName, args) {
	// fresh instance per config: resets the leaked heap
	const { instance } = await WebAssembly.instantiate(bytes, {});
	const e = instance.exports;
	e.setup(n);
	const f = () => e[exportName](...args);
	let sink = f(); // warmup
	let t0 = performance.now();
	sink += f();
	const per = Math.max((performance.now() - t0) / 1e3, 1e-9);
	const leakCap = Math.floor(250e6 / (4 * 8 * n * n));
	const iters = Math.min(Math.max(Math.ceil(0.1 / per), 3), Math.min(300, Math.max(leakCap, 3)));
	t0 = performance.now();
	for (let i = 0; i < iters; i++) sink += f();
	if (!Number.isFinite(sink)) throw new Error(`${exportName}(${args}) non-finite`);
	return ((performance.now() - t0) * 1e6) / iters;
}

for (const n of SIZES) {
	// LU: recursion_threshold (unblocked below this) × block_size
	const luCfgs = [[0, 0]];
	for (const rt of [32, 64, 128, 256]) luCfgs.push([rt, 0]);
	for (const bs of [16, 32, 128]) luCfgs.push([0, bs]);
	for (const [rt, bs] of luCfgs) {
		const ns = await timeCfg(n, 'run_lu_factor_tuned', [rt, bs]);
		console.log(JSON.stringify({ op: 'lu', n, cfg: `rt=${rt || 'dflt'},bs=${bs || 'dflt'}`, ns: Math.round(ns * 10) / 10 }));
	}
	// QR: householder panel width × blocking threshold (1<<30 = never blocked)
	const qrCfgs = [[0, 0]];
	for (const bs of [1, 4, 8, 16, 24, 32, 48]) qrCfgs.push([bs, 0]);
	qrCfgs.push([0, 1 << 30]);
	qrCfgs.push([32, 1 << 30]);
	for (const [bs, bt] of qrCfgs) {
		const ns = await timeCfg(n, 'run_qr_factor_tuned', [bs, bt]);
		console.log(JSON.stringify({ op: 'qr', n, cfg: `bs=${bs || 'rec'},bt=${bt || 'dflt'}`, ns: Math.round(ns * 10) / 10 }));
	}
}
