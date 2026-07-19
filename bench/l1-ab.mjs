// L1 assumption race (2026-07-18): swap/asum/iamax, plain scalar loop vs
// hand-SIMD, interleaved on one machine. ratio > 1 = SIMD faster.
//   node l1-ab.mjs <bench-wasm>
import { readFileSync } from 'node:fs';
const bytes = readFileSync(process.argv[2]);
const OPS = [[0, 'swap'], [1, 'asum'], [2, 'iamax']];
const SIZES = [256, 512, 1024]; // vector totals 65k–1M elements
const ROUNDS = 5;

async function timeOnce(op, variant, n) {
	const { instance } = await WebAssembly.instantiate(bytes, {});
	const e = instance.exports;
	e.setup(n);
	let sink = e.run_l1_ab(op, variant);
	if (!Number.isFinite(sink)) throw new Error('non-finite');
	const iters = n >= 1024 ? 20 : 60;
	const t0 = performance.now();
	for (let i = 0; i < iters; i++) sink += e.run_l1_ab(op, variant);
	if (!Number.isFinite(sink)) throw new Error('non-finite');
	return (performance.now() - t0) / iters;
}
const stats = (xs) => {
	const s = [...xs].sort((a, b) => a - b);
	return { med: s[Math.floor(s.length / 2)], lo: s[0], hi: s[s.length - 1] };
};
console.log('| op | n | plain ms | simd ms | simd/plain | verdict |');
console.log('| - | -: | -: | -: | -: | - |');
for (const [op, name] of OPS) {
	for (const n of SIZES) {
		const tp = [];
		const ts = [];
		for (let r = 0; r < ROUNDS; r++) {
			tp.push(await timeOnce(op, 0, n));
			ts.push(await timeOnce(op, 1, n));
		}
		const p = stats(tp);
		const sm = stats(ts);
		const sep = p.hi < sm.lo || sm.hi < p.lo;
		const ratio = p.med / sm.med;
		console.log(
			`| ${name} | ${n} | ${p.med.toFixed(3)} | ${sm.med.toFixed(3)} | ${ratio.toFixed(2)}× | ${!sep ? 'OVERLAP' : ratio > 1 ? 'SIMD WINS' : 'plain WINS'} |`,
		);
	}
}
