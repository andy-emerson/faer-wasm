// CI gate, per build variant:
//   node check.mjs <wasm-path> <variant>     variant = matmul | lu | full | full-relaxed
//
// 1. Exact comparison against the hand-verified reference values
//    (docs/research-faer-wasm-2026-07.md §3). Results have been bit-identical
//    between native x86-64 and wasm since the 2026-07 verification — any
//    difference is a bug, not noise, so this intentionally checks exact
//    doubles, not tolerances. This applies to the relaxed-SIMD build too.
// 2. Size budget from size-budgets.json — catches dependency/codegen creep.
import { readFileSync } from 'node:fs';

const wasmPath = process.argv[2] ?? './target/wasm32-unknown-unknown/release/consumer.wasm';
const variant = process.argv[3] ?? 'full';

const reference = {
	matmul_trace: 114,
	lu_solve_sum: 0.8857142857142857,   // 31/35
	qr_svd_evd_probe: 1.9483450492039642,
	schur_probe: 11,        // faer-schur real f64 property score: 6 checks + m=5 (see smoke-test/src/lib.rs)
	schur_probe_cplx: 3,    // faer-schur c64 property score
};
const required = {
	'matmul': ['matmul_trace'],
	'lu': ['matmul_trace', 'lu_solve_sum'],
	'full': ['matmul_trace', 'lu_solve_sum', 'qr_svd_evd_probe', 'schur_probe', 'schur_probe_cplx'],
	'full-relaxed': ['matmul_trace', 'lu_solve_sum', 'qr_svd_evd_probe', 'schur_probe'],
}[variant];
if (!required) {
	console.error(`unknown variant "${variant}" (want matmul | lu | full | full-relaxed)`);
	process.exit(2);
}

const wasm = readFileSync(new URL(wasmPath, import.meta.url));
const { instance } = await WebAssembly.instantiate(wasm, {});
const e = instance.exports;

let failed = false;
for (const name of required) {
	if (typeof e[name] !== 'function') {
		console.log(`[${variant}] ${name}: MISSING export`);
		failed = true;
		continue;
	}
	const got = e[name]();
	const want = reference[name];
	const ok = Object.is(got, want);
	console.log(`[${variant}] ${name} = ${got} (want ${want}) ${ok ? 'ok' : 'FAIL'}`);
	failed ||= !ok;
}

// Canary: at the current pin, c64 compute is known-broken when built with
// +relaxed-simd (grossly wrong results from the complex Hessenberg on; real
// f64 paths are unaffected — see docs/wasm.md §4). The gate asserts the
// broken state so that an upstream fix trips CI: then update this, the
// reference table, and drop the docs caveat.
if (variant === 'full-relaxed') {
	const got = e.schur_probe_cplx();
	const fixed = Object.is(got, reference.schur_probe_cplx);
	console.log(`[${variant}] schur_probe_cplx = ${got} — ${fixed
		? 'UNEXPECTEDLY CORRECT: upstream fixed c64×relaxed-simd? update gate + docs/wasm.md §4'
		: 'known-broken under relaxed-simd (expected at this pin)'}`);
	failed ||= fixed;
}

const budgets = JSON.parse(readFileSync(new URL('./size-budgets.json', import.meta.url)));
const budget = budgets[variant];
const sizeOk = wasm.byteLength <= budget;
console.log(`[${variant}] size = ${wasm.byteLength} B (budget ${budget} B) ${sizeOk ? 'ok' : 'OVER BUDGET'}`);
failed ||= !sizeOk;

process.exit(failed ? 1 : 0);
