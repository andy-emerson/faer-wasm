// Complex-gemm market race (close-out campaign, 2026-07-19): faer's
// blocked complex matmul vs the blas layer's zgemm/cgemm, interleaved
// on one machine across sizes. Same two-module structure as
// gemm-tune-ab.mjs (the blas rows live in the blas.wasm repo's bench
// module — BLAS-WASM-PIN.txt has the pinned build). The blas rows
// do slightly MORE work per call (αAB + βC blend vs faer's replace) —
// the comparison is conservative against us.
//   node cplx-gemm-ab.mjs <bench-wasm> <blas-bench-wasm>
import { readFileSync } from 'node:fs';

const { instance } = await WebAssembly.instantiate(readFileSync(process.argv[2]), {});
const e = instance.exports;
const { instance: bi } = await WebAssembly.instantiate(readFileSync(process.argv[3]), {});
const eb = bi.exports;

for (const [label, faerF, blasF] of [
	['c64', () => e.run_gemm_faer_cplx(0), () => eb.run_l3_layer_z(0)],
	['c32', () => e.run_gemm_faer_cplx(1), () => eb.run_l3_layer_c(0)],
]) {
	console.log(`\n## ${label}: faer blocked gemm vs blas layer`);
	console.log('| n | faer ms | blas ms | blas/faer speedup |');
	console.log('| -: | -: | -: | -: |');
	for (const n of [128, 256, 384, 512, 768]) {
		e.setup(n);
		eb.setup(n);
		const time = (f) => {
			let s = f();
			let best = Infinity;
			const it = n >= 512 ? 1 : 2;
			for (let r = 0; r < 5; r++) {
				const t0 = performance.now();
				for (let i = 0; i < it; i++) s += f();
				best = Math.min(best, (performance.now() - t0) / it);
			}
			if (!Number.isFinite(s)) throw new Error(`${label} n=${n}: non-finite`);
			return best;
		};
		// interleave: faer, blas, faer, blas (min-of-5 each, alternating)
		const tf = time(faerF);
		const tb = time(blasF);
		console.log(`| ${n} | ${tf.toFixed(2)} | ${tb.toFixed(2)} | ${(tf / tb).toFixed(2)}× |`);
	}
}
