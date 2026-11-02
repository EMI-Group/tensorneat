# The IREE backend: can TensorNEAT run on it here?

**Answer: no — not for a single case in the matrix, and not for a reason the benchmark protocol can work around.**
Every `--backend iree` run dies inside `Pipeline.setup()` while lowering the generation-0 species table, before a single generation is timed.
The failure is deterministic (40/40 fresh worker processes, identical message) and is an evoxc **exporter/capability gate**, not an IREE compile crash.
The SIGSEGV the pipeline owner reported is real and does reproduce here, but it sits *downstream* of three gates TensorNEAT never gets past.
No iree row carries a number; the CPU contract stays on the numpy reference backend.

## Environment (measured, not assumed)

| Item | Value |
|---|---|
| Date (UTC) | 2026-11-02, runs 04:00–04:25 |
| CPU | `INTEL(R) XEON(R) PLATINUM 8568Y+`, QEMU/KVM guest, 1 socket, 1 thread/core, no SMT |
| `nproc` | 20 |
| Platform | `Linux-6.8.0-146-generic-x86_64-with-glibc2.39` |
| Python | 3.12.3 (`$TMPDIR/iree-venv`, separate from `/tmp/tn-venv`) |
| evoxc | 0.1.0 |
| numpy | 2.5.4 |
| iree-base-compiler / iree-base-runtime | 3.11.0 / 3.11.0 (what `evoxc[iree]` resolves) |
| sympy / networkx / matplotlib / pytest | 1.14.0 / 3.7 / 3.11.2 / 9.2.0 |

## What was installed

```
uv venv "$TMPDIR/iree-venv" --python 3.12
uv pip install --python "$TMPDIR/iree-venv/bin/python" "evoxc[iree]" numpy sympy networkx matplotlib pytest
```

Exit 0, `Installed 22 packages in 558ms`, no warnings beyond uv's hardlink/copy note, no failure text.
The extra pulled `iree-base-compiler==3.11.0` + `iree-base-runtime==3.11.0`; nothing else changed version.
`/tmp/tn-venv` was not touched (its `import iree` is a `ModuleNotFoundError`, which is why this venv exists).

The iree/llvm-cpu stack itself works on this box, so no failure below can be blamed on the install:

| Probe | Result |
|---|---|
| `iree_probe.py --mode sanity` (64x64 `matmul` graph) | compiled in 0.248–0.264 s, `max_abs_err` 3.8e-06 vs numpy, exit 0 |

## Harness runs with `--backend iree`

Command (the one in the task, verbatim):

```
$TMPDIR/iree-venv/bin/python benchmarks/run_benchmark.py --impl new --backend iree \
  --cases xor_p100,xor3d_p100,custom_pagie_p500,cartpole_p100 \
  --generations 3 --warmup 1 --repeats 1 --timeout 300 \
  --json $TMPDIR/iree.json --out $TMPDIR/iree.md
```

Result: 4/4 rows `status="error"`, `VERDICT: INCOMPLETE`, CLI exit code 1, JSON and Markdown written in full.

Per-case status, each case additionally re-run **5 times in 5 fresh worker processes** (`benchmarks/harness/worker.py --impl new --backend iree --generations 3 --warmup 1 --seed 42`, records in `$TMPDIR/reps/*.json`):

| Case | Pop | Build (`Pipeline.setup`) | Timed run | Status over 5 fresh processes | ms/gen | compile (s) | peak RSS |
|---|---|---|---|---|---|---|---|
| `xor_p100` | 100 | fail | not reached | error 5/5 (export gate) | — | — | — |
| `xor_p500` | 500 | fail | not reached | error 5/5 (export gate) | — | — | — |
| `xor_p1000` | 1000 | fail | not reached | error 5/5 (export gate) | — | — | — |
| `xor3d_p100` | 100 | fail | not reached | error 5/5 (export gate) | — | — | — |
| `xor3d_p500` | 500 | fail | not reached | error 5/5 (export gate) | — | — | — |
| `xor3d_p1000` | 1000 | fail | not reached | error 5/5 (export gate) | — | — | — |
| `custom_pagie_p500` | 500 | fail | not reached | error 5/5 (export gate) | — | — | — |
| `cartpole_p100` | 100 | fail | not reached | error 5/5 (export gate) | — | — | — |

All 40 runs produced `status="error"` with one identical signature; 0 crashes, 0 timeouts, 0 missing rows.
`compile_time_s`, `ms_per_generation`, `gens_per_second` and all three RSS fields are `null` in every record, because the run dies before the first timing sample starts.
The same venv runs the same case on the numpy backend fine (`--backend numpy --cases xor_p100 --generations 3`: `ok`, 58.06 ms/gen, build 2.93 s, final best −0.2500), so this is iree-specific, not a broken environment.

## Failure signature (deterministic, first gate)

```
evoxc.core.errors.BackendError: stablehlo export: op 'scatter' at src/tensorneat/algorithm/neat/species.py:295
on a rank-3 operand along axis 0 is not supported by the v1 exporter
(numpy put_along_axis broadcast scatter is expressible for rank-1 axis=0 and rank-2 row/column writes;
use the numpy backend for the general case)
```

Top of the traceback: `benchmarks/harness/adapters/new.py:70 setup` → `src/tensorneat/pipeline.py:576 setup` → `src/tensorneat/algorithm/neat/neat.py:318 evoxc.build` → `evoxc/pipeline.py:1599 build` → `evoxc/backends/compiler.py:369 lower` → `evoxc/backends/stablehlo/writer.py:1813 _emit_scatter`.
The offending construct is the generation-0 species table: `SpeciesController.setup` writes its centre rows with an axis-0 scatter into `(species_size, max_nodes, 4)`, `(species_size, max_conns, 2)` and `(species_size, max_conns, 1)` arrays (species.py:290–313); line 295 is `center_node_attrs`, the first rank-3 one.
A trace-time census of that kernel reports exactly three rank-3 scatters: `(20, 50, 4)`, `(20, 100, 2)`, `(20, 100, 1)` for `xor_p100`.
What fails is the **rank**, not the size: evoxc's v1 exporter supports rank-1 `axis=0` and rank-2 row/column writes only, and a `conn_keys`/`conn_attrs` centre table is rank 3 for every genome layout, so population (100/500/1000), problem (XOR/XOR3d/CustomFuncFit/CartPole) and fitness target make no difference — as the 40 identical records show.

## Two more deterministic gates behind it

Probe-only trace-time bypasses (`iree_probe.py --mode pipeline --case xor_p100 --bypass …`) show the next blockers, in order:

| Probe | First failure |
|---|---|
| no bypass | `BackendError: stablehlo export: op 'scatter' … rank-3 operand along axis 0` (species.py:295) |
| `--bypass scatter` | `BackendError: capability drift: the iree backend cannot execute dtype uint32 (op 'constant') — its declared dtypes are ['bool', 'float16', 'float32', 'float64', 'int16', 'int32', 'int64', 'int8']` |
| `--bypass scatter,uint32` | `BackendError: stablehlo export: op 'tril' at src/tensorneat/algorithm/neat/_kernels.py:137 is not supported in v1` |

`uint32` is the genome row hash (`src/tensorneat/genome/gene/base.py:121–127`), which is outside iree's declared dtype set by design; `tril` is `prefix_sum_exclusive`'s triangular `dot` (`_kernels.py:137`) in the species update.
Both are in the *step* kernel, so they are only reachable after the setup gate is lifted.
Those bypasses live in `benchmarks/spikes/iree_probe.py` only; nothing in `src/` was changed for this report.

## The reported SIGSEGV does reproduce — downstream, on TensorNEAT's own kernel

A `while_loop` with three or more loop-carried tensors kills `iree-compile` (the C++ process exits `-11`, i.e. SIGSEGV) in a fresh process, `select` in the body or not, and the outcome is a per-run lottery:

| Loop-carried tensors | ok / 5, `select` body | ok / 5, plain body |
|---|---|---|
| 2 | 5 | 5 |
| 3 | 0 | 0 |
| 4 | 0 | 2 |
| 5 | 1 | 1 |
| 6 | 1 | 2 |

(probe: 256-wide float32 vector, one scalar counter plus the data carries; one fresh process per cell, i.e. `iree_probe.py --mode while --carries N --body select|plain --repeat 5`.)

TensorNEAT's own `common/tensorization.py::node_levels` — the 6-carry `while_loop` with `select` in its body that sits inside every NEAT forward graph — reproduces it standalone:

| Kernel | Fresh processes | ok | SIGSEGV | Compile when ok |
|---|---|---|---|---|
| `node_levels` N=8 P=4 K=3 | 12 | 8 | 4 | 0.693–0.781 s, `equal_numpy=True` every time |
| `node_levels` N=50 P=100 K=4 (benchmark `max_nodes`/`pop`) | 5 | 3 | 2 | 0.817 s, 0.828 s, 0.925 s, `equal_numpy=True` |

Exact signature when it dies (Python survives; the compiler subprocess does not):

```
evoxc.core.errors.BackendError: iree-compile failed to compile the StableHLO program for the iree backend (targets: llvm-cpu):
Error invoking IREE compiler tool iree-compile
Error code: -11
Stack dump: … (unsymbolised) …
```

A crashing attempt is also slow and variable: 0.98 s at N=8, 6.6 s for the whole 5-run N=50 batch above, but 68.06 s of wall clock (measured with `/usr/bin/time`) for one earlier N=50/P=100 attempt and 347 s for that 5-run batch — the compiler burns time before the recursion dies.
Nothing in the TensorNEAT *pipeline* reaches this point, because gates 1–3 fire during `lower()`/`export` first.

## Compile cost: the "3–7 s per graph" claim is not reproduced

| Graph | Compile (s) | Outcome |
|---|---|---|
| 64x64 `matmul` | 0.248–0.264 | runs, 3.8e-06 max abs err |
| synthetic 6-carry `while` + `select` | 0.318 | runs, bit-exact vs numpy |
| TensorNEAT `node_levels` N=8 P=4 K=3 | 0.693–0.781 | runs, bit-exact |
| TensorNEAT `node_levels` N=50 P=100 K=4 | 0.817 / 0.828 / 0.925 | runs, bit-exact |
| TensorNEAT init kernel (probe patch: 3 rank-3 scatters flattened) | 1.193 | runs; the step kernel then hits gate 2 |
| full `Pipeline` (setup + step) | never reached | gate 1 |
| a *crashing* compile at N=50 P=100 K=4 | 68.06 | SIGSEGV after a minute of compiler work |

The largest TensorNEAT graph that can be compiled here took 1.19 s, so 3–7 s per graph is not what this environment shows; what can take a minute is a compile that ends in the SIGSEGV.
The full step graph's compile time is unmeasurable until gates 1–3 are lifted.

## Minimal reproducer (self-contained)

The first gate, with no TensorNEAT involved — 18 lines, numpy backend accepted it, iree rejects it 5/5 fresh processes:

```python
import numpy as np
import evoxc

@evoxc.defn
def seed_centre(dead, centre):
    # (species_size, max_nodes, width) <- (max_nodes, width) into row 0:
    # the species-centre table, species.py:295's construct.
    return evoxc.scatter(dead, 0, centre)

specs = [evoxc.TensorSpec((20, 50, 4), evoxc.float32),
         evoxc.TensorSpec((50, 4), evoxc.float32)]
dead = np.full((20, 50, 4), np.nan, np.float32)
centre = np.zeros((50, 4), np.float32)

ref = evoxc.run(evoxc.build(seed_centre, *specs, backend="numpy"), dead, centre)
print("numpy ok:", bool(np.all(np.asarray(ref.numpy())[0] == 0)))   # True
evoxc.build(seed_centre, *specs, backend="iree")                    # BackendError, 5/5
```

The downstream SIGSEGV, on TensorNEAT's real kernel, is `python benchmarks/spikes/iree_probe.py --mode node-levels --max-nodes 8 --pop 4 --window 3 --repeat 5`; the synthetic carry-count matrix is `--mode while --carries N --body select|plain --repeat 5`.
The real pipeline gates are `python benchmarks/spikes/iree_probe.py --mode pipeline --case xor_p100 [--bypass scatter[,uint32]]`.

## What correlates, and what was ruled out

- **Correlates:** the number of `while_loop` loop-carried tensors. Two carried values never crashed (10/10 fresh processes across both body shapes); three or more crashed in 60–100% of the 50 fresh processes in the matrix above.
- `select` in the loop body is **not** the trigger (a `select`-free body crashes just as reliably at 3–4 carries) and is **not** sufficient (a 2-carry loop with `select` is fine); it only raises the crash rate at 4–6 carries.
- **Graph size** changes the cost, not the existence, of the crash: N=8/P=4 dies in ~1 s, while one N=50/P=100 attempt burned 68 s before dying.
- **Ruled out** for the TensorNEAT failures: population size (identical at 100/500/1000), problem family (all eight cases), species/genome configuration (the failing rank is structural), and the environment (the same venv compiles a `matmul` graph and runs the numpy backend).
- **Ruled out** for the crash: a Python-level fault (the Python process stays alive and receives an exception; the `iree-compile` subprocess is what exits `-11`), and thread configuration (all runs used runtime defaults; `--threads` was never varied here).

## Determinism and harness behaviour

- 40 fresh worker processes (8 cases × 5) → 40 identical `status="error"` rows, worker exit code 1 each; no crash, no timeout, no missing row.
- A failure in one case is confined to that case's row: the 4-case CLI run above completed all four rows, wrote JSON and Markdown, and exited 1.
- Verified for a *hard* crash too, not just an exception: a test-only `sitecustomize.py` in `$TMPDIR` that SIGSEGVs only the worker process for `xor3d_p100` produced a synthesised `status="error"` row (`error: [crashshim] segfaulting worker for xor3d_p100`) while the other two cases still ran and the report was written.
  The harness confines a SIGSEGV correctly (one worker process per run); no harness bug was found, and `run_benchmark.py` was not modified.
- `benchmarks/RESULTS.md` was not regenerated or edited for this report; all records referenced here live in `$TMPDIR` (`iree.json`, `reps/*.json`, `confine.json`) and are reproducible with the commands above.

## Verdict

iree rows are usable for **no** case (X/Y = ∅): no TensorNEAT graph reaches IREE's compiler, so there is no measured ms/gen, compile time or RSS to report for any of `xor_p100`, `xor_p500`, `xor_p1000`, `xor3d_p100`, `xor3d_p500`, `xor3d_p1000`, `custom_pagie_p500`, `cartpole_p100`.
Those cases are unavailable because `Pipeline.setup()` cannot be lowered on iree — evoxc's stablehlo v1 exporter rejects the rank-3 species-centre scatter at `src/tensorneat/algorithm/neat/species.py:295` (reproducer above, 5/5 fresh processes, 40/40 harness workers), and the two gates behind it (`uint32` genome hash outside iree's declared dtype set, `tril` in the species prefix sum) would each still have to be lifted.
The pipeline owner's SIGSEGV is real and reproducible on TensorNEAT's own 6-carry `node_levels` while-loop (4/12 fresh processes at N=8, 2/5 at N=50/P=100), but it is the fourth blocker, not the first, and its "3–7 s per graph" compile figure was not reproduced (0.26–1.19 s for every graph that compiled; a crashing compile cost 1–68 s).
Until `Pipeline` can be lowered on iree at all, `RESULTS.md` must not carry iree rows, and the CPU contract remains the numpy reference backend.
