// SVD roofline + phase profile (architect direction 2026-07-10): locate the
// machine ceiling and where faer's SVD time actually goes, so we optimize
// toward the hardware limit rather than toward scipy parity.
//   node svd-roofline.mjs <bench-wasm>
//
// Anchors: compute peak from matmul (near-peak gemm GF/s), bandwidth from a
// STREAM triad. Then place a raw GEMV against bandwidth (bidiag's ~50% is
// GEMV), and split faer's SVD into reduction (bidiag) vs solve (DC+backtransform).
import { readFileSync } from 'node:fs';

const wasmPath = process.argv[2];
if (!wasmPath) {
	console.error('usage: node svd-roofline.mjs <bench-wasm>');
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
		const iters = Math.min(Math.max(Math.ceil(0.15 / per), 5), 200);
		t0 = performance.now();
		for (let i = 0; i < iters; i++) sink += f();
		best = Math.min(best, ((performance.now() - t0) * 1e6) / iters); // ns
		if (!Number.isFinite(sink)) throw new Error(`${exportName}(n=${n}): non-finite`);
	}
	return best / 1e6; // ms
}

const SIZES = [128, 256, 512];
console.log('# SVD roofline + phase profile (on-runner)');
console.log('# GF/s = 10^9 flop/s, GB/s = 10^9 byte/s, min-of-N');
for (const n of SIZES) {
	const mm = await time('run_matmul', n); // 2n^3 flop, near compute peak
	const stream = await time('run_stream', n); // 24 n^2 bytes, bandwidth anchor
	const gemv = await time('run_gemv', n); // 8 n^2 bytes read, memory-bound
	const bidiag = await time('run_bidiag_only', n); // the reduction
	const svd = await time('run_svd', n); // full SVD

	const s = (ms) => ms / 1e3;
	const mmGF = (2 * n ** 3) / s(mm) / 1e9;
	const streamGB = (24 * n * n) / s(stream) / 1e9;
	const gemvGB = (8 * n * n) / s(gemv) / 1e9;
	const gemvGF = (2 * n * n) / s(gemv) / 1e9;
	const reductionPct = (bidiag / svd) * 100;

	console.log(`\nn=${n}:`);
	console.log(`  compute peak (matmul)   ${mmGF.toFixed(2)} GF/s   [${mm.toFixed(3)} ms]`);
	console.log(`  bandwidth  (STREAM triad) ${streamGB.toFixed(2)} GB/s [${stream.toFixed(4)} ms]`);
	console.log(`  GEMV                    ${gemvGB.toFixed(2)} GB/s  (${((gemvGB / streamGB) * 100).toFixed(0)}% of bandwidth), ${gemvGF.toFixed(2)} GF/s`);
	console.log(`  faer bidiag (reduction) ${bidiag.toFixed(3)} ms`);
	console.log(`  faer full SVD           ${svd.toFixed(3)} ms   -> reduction is ${reductionPct.toFixed(0)}% of SVD`);
}
