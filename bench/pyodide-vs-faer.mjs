// Head-to-head: faer-wasm vs Pyodide (numpy/scipy compiled to wasm), same
// problems, same V8, same process.
//   PYODIDE_PATH=<path/to/pyodide.mjs> node pyodide-vs-faer.mjs <bench-wasm>
//
// Pyodide is the incumbent scientific-computing-on-wasm stack (scipy's
// LAPACK compiled to wasm), which makes this the apples-to-apples "why
// faer-wasm" comparison — unlike a native-OpenBLAS baseline, both sides pay
// the same wasm tax. Each stack does its idiomatic call for the same
// mathematical problem; known asymmetries are footnoted in the report.
//
// NOTE the container this repo is developed in blocks the Pyodide package
// CDN; the pyodide-bench.yml workflow runs this on a GitHub runner (open
// egress). Results land in the job log and as a workflow artifact.
import { readFileSync, writeFileSync } from 'node:fs';

const wasmPath = process.argv[2];
if (!wasmPath) {
	console.error('usage: PYODIDE_PATH=<pyodide.mjs> node pyodide-vs-faer.mjs <bench-wasm>');
	process.exit(2);
}
const SIZES = [64, 128, 256, 512];
// [name, faer bench export, faer args (fixed), python lambda body]
// The *_tuned rows use the docs/wasm.md §7 parameters — the honest current
// best-case faer — against the closest python equivalent (factor-only vs
// scipy lu_factor; R-only QR on both sides).
const OPS = [
	['matmul', 'run_matmul', null, 'a @ b'],
	['lu_solve', 'run_lu_solve', null, 'np.linalg.solve(a, rhs)'],
	['lu_solve_wk', 'run_lu_solve_wk', [], 'np.linalg.solve(a, rhs)'],
	['lu_factor', 'run_lu_factor_tuned', [0, 0], 'sla.lu_factor(a)'],
	['lu_factor_tuned', 'run_lu_factor_tuned', [1 << 30, 0], 'sla.lu_factor(a)'],
	['lu_factor_wk', 'run_lu_factor_wk', [0], 'sla.lu_factor(a)'],
	['lu_factor_rec', 'run_lu_factor_rec', [0], 'sla.lu_factor(a)'],
	['qr_r', 'run_qr', null, "np.linalg.qr(a, mode='r')"],
	['qr_r_tuned', 'run_qr_factor_tuned', [1, 1 << 30], "np.linalg.qr(a, mode='r')"],
	['qr_r_wk', 'run_qr_factor_wk', [], "np.linalg.qr(a, mode='r')"],
	['svd', 'run_svd', null, 'np.linalg.svd(a)'],
	['eigvals', 'run_gen_evd', null, 'np.linalg.eigvals(a)'],
	['eigvals_k3', 'run_eigvals_k3', [], 'np.linalg.eigvals(a)'],
	['schur', 'run_schur', null, 'sla.schur(a)'],
	['schur_k', 'run_schur_k', [], 'sla.schur(a)'],
	['matmul_c64', 'run_matmul_c64', null, 'ac @ bc'],
	['lu_solve_c64', 'run_lu_solve_c64', null, 'np.linalg.solve(ac, rhsc)'],
	['qr_r_c64', 'run_qr_c64', null, "np.linalg.qr(ac, mode='r')"],
	['schur_c64', 'run_schur_c64', null, "sla.schur(ac, output='complex')"],
	['schur_c64_k', 'run_schur_c64_k', [], "sla.schur(ac, output='complex')"],
	// f32 rows (f32/c32 phase): both sides in single precision — numpy
	// dispatches the LAPACK s-routines, faer rows ride the generic kernels.
	['matmul_f32', 'run_matmul_f32', [], 'a32 @ b32'],
	['lu_solve_f32', 'run_lu_solve_wk_f32', [], 'np.linalg.solve(a32, rhs32)'],
	['qr_r_f32', 'run_qr_factor_wk_f32', [], "np.linalg.qr(a32, mode='r')"],
	['eigvals_f32', 'run_eigvals_k3_f32', [], 'np.linalg.eigvals(a32)'],
	['schur_f32', 'run_schur_k_f32', [], 'sla.schur(a32)'],
];

// ---- faer side (same adaptive min-of-3 protocol as gate.mjs)
const bytes = readFileSync(wasmPath);
async function timeFaer(exportName, n, args) {
	let best = Infinity;
	const reps = n >= 512 ? 2 : 3; // 512-sized eig pipelines are seconds-scale
	for (let rep = 0; rep < reps; rep++) {
		const { instance } = await WebAssembly.instantiate(bytes, {});
		const e = instance.exports;
		e.setup(n);
		const f = args ? () => e[exportName](...args) : e[exportName];
		let sink = f();
		let t0 = performance.now();
		sink += f();
		const per = Math.max((performance.now() - t0) / 1e3, 1e-9);
		const leakCap = Math.floor(150e6 / (8 * 8 * n * n));
		const iters = Math.min(Math.max(Math.ceil(0.15 / per), 3), Math.min(200, Math.max(leakCap, 3)));
		t0 = performance.now();
		for (let i = 0; i < iters; i++) sink += f();
		best = Math.min(best, ((performance.now() - t0) * 1e6) / iters);
		if (!Number.isFinite(sink)) throw new Error(`${exportName}(n=${n}): non-finite`);
	}
	return best;
}

// ---- pyodide side
const { loadPyodide } = await import(process.env.PYODIDE_PATH ?? 'pyodide');
const py = await loadPyodide();
await py.loadPackage(['numpy', 'scipy'], { messageCallback: () => {} });
const versions = await py.runPythonAsync(
	'import numpy, scipy; f"pyodide numpy {numpy.__version__}, scipy {scipy.__version__}"',
);
console.log(`# ${versions}; node ${process.version}`);

// what BLAS/LAPACK is scipy actually linked against in Pyodide? (research
// question from docs/research-lu-wasm-2026-07.md — claimed OpenBLAS with
// generic C kernels, unverified until printed here)
console.log('## scipy/numpy build config');
console.log(
	await py.runPythonAsync(`
import io, contextlib, numpy, scipy
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    print("--- numpy ---"); numpy.show_config()
    print("--- scipy ---"); scipy.show_config()
buf.getvalue()
`),
);

await py.runPythonAsync(`
import time
import numpy as np
import scipy.linalg as sla

def setup(n):
    global a, b, rhs, ac, bc, rhsc
    rng = np.random.default_rng(0x9E3779B9 ^ n)
    a = rng.uniform(-1, 1, (n, n))
    b = rng.uniform(-1, 1, (n, n))
    rhs = rng.uniform(-1, 1, (n,))
    ac = rng.uniform(-1, 1, (n, n)) + 1j * rng.uniform(-1, 1, (n, n))
    bc = rng.uniform(-1, 1, (n, n)) + 1j * rng.uniform(-1, 1, (n, n))
    rhsc = rng.uniform(-1, 1, (n,)) + 1j * rng.uniform(-1, 1, (n,))
    global a32, b32, rhs32
    a32 = a.astype(np.float32)
    b32 = b.astype(np.float32)
    rhs32 = rhs.astype(np.float32)

def bench(f, reps=3):
    best = float("inf")
    for _ in range(reps):
        f()
        t0 = time.perf_counter(); f()
        per = max(time.perf_counter() - t0, 1e-9)
        iters = min(max(int(0.15 / per) + 1, 3), 200)
        t0 = time.perf_counter()
        for _ in range(iters):
            f()
        best = min(best, (time.perf_counter() - t0) / iters * 1e9)
    return best

# correctness spot-check before timing anything
setup(32)
t, z = sla.schur(a)
assert abs(a - z @ t @ z.T).max() < 1e-12, "pyodide schur residual too large"
`);

// ---- run both, emit table
const rows = [];
for (const n of SIZES) {
	await py.runPythonAsync(`setup(${n})`);
	for (const [name, faerExport, faerArgs, pyBody] of OPS) {
		const pyNs = await py.runPythonAsync(`bench(lambda: ${pyBody})`);
		const faerNs = await timeFaer(faerExport, n, faerArgs);
		rows.push({ op: name, n, faer_ms: faerNs / 1e6, pyodide_ms: pyNs / 1e6, speedup: pyNs / faerNs });
		console.log(JSON.stringify({ op: name, n, faer_ns: Math.round(faerNs), pyodide_ns: Math.round(pyNs) }));
	}
}

console.log('\n| op | n | faer-wasm | pyodide | speedup |');
console.log('| - | -: | -: | -: | -: |');
for (const r of rows) {
	console.log(
		`| ${r.op} | ${r.n} | ${r.faer_ms.toFixed(2)} ms | ${r.pyodide_ms.toFixed(2)} ms | ${r.speedup.toFixed(1)}× |`,
	);
}
const geo = rows.reduce((s, r) => s + Math.log(r.speedup), 0) / rows.length;
console.log(`\ngeomean speedup: ${Math.exp(geo).toFixed(2)}×`);
writeFileSync('pyodide-vs-faer-results.json', JSON.stringify(rows, null, '\t'));

// ---- eig replication gate (architect challenge 2026-07-11: single-run
// eigvals margins were inside measured cross-run variance — same op@512
// spanned 598/836/854 ms across runner instances). 5 independent rounds,
// ALTERNATING faer and scipy so machine drift hits both sides; a WIN/LOSS
// verdict requires the min..max ranges to separate, otherwise OVERLAP.
const ROUNDS = 5;
// architect direction 2026-07-11: the eig scoreboard includes n=1024 (the
// main grid stays at <=512; the replication gate is the eig record).
// 1024 exceeds the measured crossover grid (which stopped at 512) — the
// routing sends it to multishift, and this measures that extrapolation.
const REP_SIZES = [...SIZES, 1024];
// each entry races its own scipy call: [key, faer export, python body]
const REP_OPS = [
	['eigvals_k3', 'run_eigvals_k3', 'np.linalg.eigvals(a)'],
	['schur_k', 'run_schur_k', 'sla.schur(a)'],
	['schur_c64_k', 'run_schur_c64_k', "sla.schur(ac, output='complex')"],
];
const stats = (xs) => {
	const s = [...xs].sort((x, y) => x - y);
	return { min: s[0], med: s[Math.floor(s.length / 2)], max: s[s.length - 1] };
};
const fmtR = (r) => `${r.med.toFixed(2)} [${r.min.toFixed(2)}..${r.max.toFixed(2)}]`;
console.log(`\n## replication gate (${ROUNDS} alternating rounds, median [min..max] ms)`);
for (const n of REP_SIZES) {
	await py.runPythonAsync(`setup(${n})`);
	const faer = Object.fromEntries(REP_OPS.map(([k]) => [k, []]));
	const scipy = Object.fromEntries(REP_OPS.map(([k]) => [k, []]));
	for (let r = 0; r < ROUNDS; r++) {
		for (const [key, exp, pyBody] of REP_OPS) {
			faer[key].push((await timeFaerOnce(exp, n)) / 1e6);
			scipy[key].push((await py.runPythonAsync(`bench(lambda: ${pyBody}, reps=1)`)) / 1e6);
		}
	}
	console.log(`\nn=${n}:`);
	for (const [key, , pyBody] of REP_OPS) {
		const sc = stats(scipy[key]);
		const st = stats(faer[key]);
		const verdict =
			st.max < sc.min
				? `WIN (ranges separate, ${(sc.med / st.med).toFixed(2)}x)`
				: st.min > sc.max
					? `LOSS (ranges separate, ${(sc.med / st.med).toFixed(2)}x)`
					: `OVERLAP (median ratio ${(sc.med / st.med).toFixed(2)}x) — no claim`;
		console.log(`  scipy ${pyBody.padEnd(22)} ${fmtR(sc)}`);
		console.log(`  ${key.padEnd(11)} ${fmtR(st)}  -> ${verdict}`);
	}
}

// ---- schur_k want_t/Z cost split (research-schur-wasm-2026-07.md open
// question 1): mode 0 = eigvals-only baseline, 1 = +T (range widening),
// 2 = +Z (Q formation + Z updates), 3 = full Schur. min-of-2 per cell.
console.log('\n## schur_k want_t/Z cost split (min-of-2, ms)');
console.log('| n | eigvals (0) | +T (1) | +Z (2) | T+Z (3) | T share | Z share |');
console.log('| -: | -: | -: | -: | -: | -: | -: |');
for (const n of REP_SIZES) {
	const ms = [];
	for (const mode of [0, 1, 2, 3]) {
		let best = Infinity;
		for (let rep = 0; rep < 2; rep++) {
			best = Math.min(best, await timeFaerOnce('run_schur_k_mode', n, [mode]));
		}
		ms.push(best / 1e6);
	}
	const tShare = ((ms[1] - ms[0]) / ms[3]) * 100;
	const zShare = ((ms[2] - ms[0]) / ms[3]) * 100;
	console.log(
		`| ${n} | ${ms[0].toFixed(2)} | ${ms[1].toFixed(2)} | ${ms[2].toFixed(2)} | ${ms[3].toFixed(2)} | ${tShare.toFixed(0)}% | ${zShare.toFixed(0)}% |`,
	);
}

// single-round faer timing (fresh instance, adaptive iters, NOT min-of-3 —
// each replication round is one independent sample)
async function timeFaerOnce(exportName, n, args = []) {
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
	if (!Number.isFinite(sink)) throw new Error(`${exportName}(n=${n}): non-finite`);
	return ((performance.now() - t0) * 1e6) / iters; // ns
}
