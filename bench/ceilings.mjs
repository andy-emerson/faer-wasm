// Ceiling probes (roofline metric, 2026-07-18): the machine's memory
// bandwidth and peak SIMD arithmetic, measured in-harness so every
// benchmark row can be scored as a fraction of them. Also scores the
// streaming gemm and the triad-shaped ops directly.
//   node ceilings.mjs <bench-wasm>
import { readFileSync } from 'node:fs';

const bytes = readFileSync(process.argv[2]);
if (!bytes) process.exit(2);

async function inst(n) {
	const { instance } = await WebAssembly.instantiate(bytes, {});
	instance.exports.setup(n);
	return instance.exports;
}
function best(f, times = 7) {
	let b = Infinity;
	for (let i = 0; i < times; i++) {
		const t0 = performance.now();
		const s = f();
		const dt = performance.now() - t0;
		if (!Number.isFinite(s)) throw new Error('non-finite');
		b = Math.min(b, dt);
	}
	return b;
}

// memory bandwidth: triad over n=1024 (24 MB working set, far past cache)
{
	const e = await inst(1024);
	const bytesMoved = 3 * 8 * 1024 * 1024;
	e.run_ceiling_bw();
	const ms = best(() => e.run_ceiling_bw());
	console.log(`memory bandwidth (triad, 24 MB): ${(bytesMoved / (ms * 1e6)).toFixed(2)} GB/s`);
}

// peak arithmetic: register-resident chains (no memory traffic)
{
	const e = await inst(64);
	const iters = 4_000_000;
	const flops = iters * 8 * 2 * 2;
	e.run_ceiling_flops(1000);
	const ms = best(() => e.run_ceiling_flops(iters));
	console.log(`peak f64 SIMD arithmetic: ${(flops / (ms * 1e6)).toFixed(2)} GFLOP/s`);
	// score the streaming gemm against it at n=512
	const e2 = await inst(512);
	e2.run_blas_ab(4, 1);
	const gms = best(() => e2.run_blas_ab(4, 1), 3);
	const gflops = (2 * 512 ** 3) / (gms * 1e6);
	console.log(
		`streaming-loop gemm @512: ${gflops.toFixed(2)} GFLOP/s = ${((gflops * (ms * 1e6)) / flops * 100).toFixed(0)}% of peak`,
	);
}
