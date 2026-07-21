// Tuning-campaign race: three bit-identical gemm shapes (column-axpy /
// register tile / 4-column fused) + faer's blocked gemm, across
// sizes, interleaved on one machine — locates the tile↔col4 dispatch
// crossover on the reference class. Since the 2026-07-19 restructure
// the blas rows live in the blas.wasm repo's bench module (see
// BLAS-WASM-PIN.txt for the pinned build), so this loads
// TWO wasm files side by side — same process, same machine, still an
// interleaved race.
//   node gemm-tune-ab.mjs <bench-wasm> <blas-bench-wasm>
import { readFileSync } from 'node:fs';

const { instance } = await WebAssembly.instantiate(readFileSync(process.argv[2]), {});
const e = instance.exports;
const { instance: bi } = await WebAssembly.instantiate(readFileSync(process.argv[3]), {});
const eb = bi.exports;
const VARIANTS = [
	['faer', () => e.run_blas_ab(4, 0)],
	['colaxpy', () => eb.run_l3_layer(0)],
	['tiled', () => eb.run_l3_tuned_gemm()],
	['col4', () => eb.run_l3_col4_gemm()],
];
console.log('| n | faer | colaxpy | tiled4x4 | col4 | best |');
console.log('| -: | -: | -: | -: | -: | - |');
for (const n of [128, 192, 256, 384, 512, 768, 1024]) {
	e.setup(n);
	eb.setup(n);
	const times = [];
	for (const [name, f] of VARIANTS) {
		let s = f();
		let best = Infinity;
		const it = n >= 768 ? 1 : 2;
		for (let r = 0; r < 5; r++) {
			const t0 = performance.now();
			for (let i = 0; i < it; i++) s += f();
			best = Math.min(best, (performance.now() - t0) / it);
		}
		if (!Number.isFinite(s)) throw new Error(`${name} n=${n}: non-finite`);
		times.push([name, best]);
	}
	const best = times.reduce((a, b) => (b[1] < a[1] ? b : a));
	console.log(
		`| ${n} | ${times.map((t) => t[1].toFixed(2)).join(' | ')} | ${best[0]} |`,
	);
}
