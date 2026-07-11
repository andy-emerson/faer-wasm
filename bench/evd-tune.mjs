// EVD/Schur wasm crossover finder (fix-1 of the eigen plan). Post-patch-0004
// the multishift/AED path is repaired; the remaining cheap win is routing:
// scalar lahqr wins small n, multishift wins large n, and faer's default
// blocking_threshold=75 (native-tuned) is far below the wasm crossover.
// This measures multishift vs lahqr at a fine size grid for the three
// pipelines we ship (eigvals real, Schur+Z real, Schur+Z c64) so the
// companion-crate recommended params can pin the measured thresholds.
//   node evd-tune.mjs <bench-wasm>
import { readFileSync } from 'node:fs';

const wasmPath = process.argv[2];
if (!wasmPath) {
	console.error('usage: node evd-tune.mjs <bench-wasm>');
	process.exit(2);
}
const bytes = readFileSync(wasmPath);

async function time(exportName, n, args = []) {
	let best = Infinity;
	const reps = n >= 512 ? 3 : 4;
	for (let rep = 0; rep < reps; rep++) {
		const { instance } = await WebAssembly.instantiate(bytes, {});
		const e = instance.exports;
		e.setup(n);
		const f = () => e[exportName](...args);
		let sink = f();
		let t0 = performance.now();
		sink += f();
		const per = Math.max((performance.now() - t0) / 1e3, 1e-9);
		const iters = Math.min(Math.max(Math.ceil(0.15 / per), 3), 100);
		t0 = performance.now();
		for (let i = 0; i < iters; i++) sink += f();
		best = Math.min(best, ((performance.now() - t0) * 1e6) / iters); // ns
		if (!Number.isFinite(sink)) throw new Error(`${exportName}(n=${n}): non-finite`);
	}
	return best / 1e6; // ms
}

const LAHQR = 1 << 30; // blocking above any n => always the scalar kernel
// [pipeline label, export, extra fixed args]
const PIPELINES = [
	['eigvals (real, no Z)', 'run_eigvals_tuned', (b) => [b, 0, 0]],
	['schur+Z (real)', 'run_schur_tuned', (b) => [b]],
	['schur+Z (c64)', 'run_schur_c64_tuned', (b) => [b]],
];
const SIZES = [96, 128, 192, 256, 320, 384, 448, 512];

console.log('# EVD/Schur multishift-vs-lahqr crossover grid (on-runner, post-0004)');
console.log('# blocking=0 -> faer default 75 (multishift for all sizes below); LAHQR = pinned scalar');
for (const [label, exp, argf] of PIPELINES) {
	console.log(`\n## ${label}`);
	console.log('| n | multishift | lahqr | winner |');
	for (const n of SIZES) {
		const ms = await time(exp, n, argf(0));
		const lq = await time(exp, n, argf(LAHQR));
		const w = ms < lq ? `multishift ${(lq / ms).toFixed(2)}x` : `lahqr ${(ms / lq).toFixed(2)}x`;
		console.log(`| ${n} | ${ms.toFixed(2)} ms | ${lq.toFixed(2)} ms | ${w} |`);
	}
}
