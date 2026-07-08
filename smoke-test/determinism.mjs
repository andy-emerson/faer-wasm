// Cross-target determinism gate:
//   node determinism.mjs <wasm-path> <native-output-file>
// The native file comes from `cargo run --release --features full --bin native`
// and holds name=<16-hex-digit f64 bits> lines. Every probe the native bin
// printed is re-run in the wasm module and compared bit-for-bit; any
// difference fails. (schur_probe is an integer-valued property score — the
// raw Schur doubles are NOT cross-target bit-identical at 8x8, see
// smoke-test/src/lib.rs — so bit-comparing the score is exactly as strong as
// the gate can honestly be for that pipeline.)
import { readFileSync } from 'node:fs';

const [wasmPath, nativePath] = process.argv.slice(2);
if (!wasmPath || !nativePath) {
	console.error('usage: node determinism.mjs <wasm-path> <native-output-file>');
	process.exit(2);
}

const native = Object.fromEntries(
	readFileSync(nativePath, 'utf8')
		.trim()
		.split('\n')
		.map(line => line.split('=')),
);

const wasm = readFileSync(new URL(wasmPath, import.meta.url));
const { instance } = await WebAssembly.instantiate(wasm, {});
const e = instance.exports;

const bitsOf = (x) => {
	const dv = new DataView(new ArrayBuffer(8));
	dv.setFloat64(0, x);
	return dv.getBigUint64(0).toString(16).padStart(16, '0');
};

let failed = false;
for (const name of Object.keys(native)) {
	if (typeof e[name] !== 'function' || !native[name]) {
		console.log(`${name}: MISSING (wasm export or native line)`);
		failed = true;
		continue;
	}
	const w = bitsOf(e[name]());
	const n = native[name];
	const ok = w === n;
	console.log(`${name}: wasm ${w} vs native ${n} ${ok ? 'IDENTICAL' : 'DIFFER'}`);
	failed ||= !ok;
}
process.exit(failed ? 1 : 0);
