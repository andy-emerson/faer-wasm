// BLAS-layer A/B (architect-directed, 2026-07-13): streaming-loop variant
// vs the faer path for every "unchanged" BLAS-level op, interleaved on one
// machine per the verdict-stability rule. Verdicts require min..max range
// separation; ratio > 1 means the streaming loop is FASTER than faer.
//
//   node blas-ab.mjs <bench-wasm> [--fma]
// --fma: also time variant 2 (the fused-FMA streaming loop) — meaningful
// only on a build compiled with -C target-feature=+simd128,+relaxed-simd.
import { readFileSync } from 'node:fs';

const wasmPath = process.argv[2];
if (!wasmPath) {
	console.error('usage: node blas-ab.mjs <bench-wasm> [--fma]');
	process.exit(2);
}
const THREE_WAY = process.argv.includes('--fma');
const bytes = readFileSync(wasmPath);

const OPS = [
	[0, 'copy', 'L1'],
	[1, 'gemv', 'L2'],
	[2, 'ger', 'L2'],
	[3, 'trsv (1 rhs)', 'L2'],
	[4, 'gemm', 'L3'],
	[5, 'syrk', 'L3'],
	[6, 'trmm', 'L3'],
	[7, 'trsm', 'L3'],
];
const SIZES = [64, 128, 256, 512, 1024];
const ROUNDS = 5;

async function timeOnce(op, variant, n) {
	const { instance } = await WebAssembly.instantiate(bytes, {});
	const e = instance.exports;
	e.setup(n);
	let sink = e.run_blas_ab(op, variant); // warm + compile
	if (!Number.isFinite(sink)) throw new Error(`op=${op} v=${variant} n=${n}: non-finite`);
	// level-3 ops at large n are slow; O(n²) ops need iterations to time
	const heavy = op >= 4;
	const iters = heavy ? (n >= 512 ? 1 : 4) : n >= 512 ? 8 : 40;
	const t0 = performance.now();
	for (let i = 0; i < iters; i++) sink += e.run_blas_ab(op, variant);
	if (!Number.isFinite(sink)) throw new Error('non-finite');
	return (performance.now() - t0) / iters;
}

const stats = (xs) => {
	const s = [...xs].sort((a, b) => a - b);
	return { med: s[Math.floor(s.length / 2)], lo: s[0], hi: s[s.length - 1] };
};

// which ops have a fused variant 2 (gemv + the four L3 ops)
const HAS_FMA = new Set([1, 4, 5, 6, 7]);

if (!THREE_WAY) {
	console.log('| op | level | n | faer med ms | loop med ms | loop/faer | verdict |');
	console.log('| - | - | -: | -: | -: | -: | - |');
} else {
	console.log('| op | level | n | faer | loop | loopFMA | loop/faer | fma/faer | fma/loop |');
	console.log('| - | - | -: | -: | -: | -: | -: | -: | -: |');
}
for (const [op, name, level] of OPS) {
	for (const n of SIZES) {
		const tf = [];
		const tl = [];
		const tm = [];
		const doFma = THREE_WAY && HAS_FMA.has(op);
		for (let r = 0; r < ROUNDS; r++) {
			tf.push(await timeOnce(op, 0, n));
			tl.push(await timeOnce(op, 1, n));
			if (doFma) tm.push(await timeOnce(op, 2, n));
		}
		const f = stats(tf);
		const l = stats(tl);
		if (!THREE_WAY) {
			const sep = f.hi < l.lo || l.hi < f.lo;
			const ratio = f.med / l.med;
			const verdict = !sep ? 'OVERLAP' : ratio > 1 ? 'loop WINS' : 'faer WINS';
			console.log(
				`| ${name} | ${level} | ${n} | ${f.med.toFixed(3)} | ${l.med.toFixed(3)} | ${ratio.toFixed(2)}× | ${verdict} |`,
			);
		} else {
			const m = doFma ? stats(tm) : null;
			console.log(
				`| ${name} | ${level} | ${n} | ${f.med.toFixed(3)} | ${l.med.toFixed(3)} | ${m ? m.med.toFixed(3) : '—'} | ${(f.med / l.med).toFixed(2)}× | ${m ? (f.med / m.med).toFixed(2) + '×' : '—'} | ${m ? (l.med / m.med).toFixed(2) + '×' : '—'} |`,
			);
		}
	}
}
