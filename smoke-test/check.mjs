// CI gate: exact comparison against the hand-verified reference values
// (docs/research-faer-wasm-2026-07.md §3). Results have been bit-identical
// between native x86-64 and wasm since the 2026-07 verification — any
// difference is a bug, not noise, so this intentionally checks exact
// doubles, not tolerances.
import { readFileSync } from 'node:fs';

const wasm = readFileSync(new URL(
	process.argv[2] ?? './target/wasm32-unknown-unknown/release/consumer.wasm',
	import.meta.url,
));
const { instance } = await WebAssembly.instantiate(wasm, {});
const e = instance.exports;

const expected = {
	matmul_trace: 114,
	lu_solve_sum: 0.8857142857142857,   // 31/35
	qr_svd_evd_probe: 1.9483450492039642,
};

let failed = false;
for (const [name, want] of Object.entries(expected)) {
	if (typeof e[name] !== 'function') {
		console.log(`${name}: MISSING export (build with --features full)`);
		failed = true;
		continue;
	}
	const got = e[name]();
	const ok = Object.is(got, want);
	console.log(`${name} = ${got} (want ${want}) ${ok ? 'ok' : 'FAIL'}`);
	failed ||= !ok;
}
process.exit(failed ? 1 : 0);
