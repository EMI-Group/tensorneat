# The `iree` backend on IREE 3.13: it runs, and numpy is still ahead on this CPU

**Verdict for the tree and session measured here: `backend="iree"` compiles and runs TensorNEAT end to end and is 1.54× (pop 100), 5.47× (pop 500) and 6.63× (pop 1000) slower per generation than the numpy reference interpreter, measured as the median of three fresh-process repeats per population; each static shape pays one 8.0–9.6 s `iree-compile` before its first generation.**
The problem classes exercised on iree are `XOR`, `XOR3d`, `CustomFuncFit`, `CartPole`, `Pendulum` and `MountainCarContinuous`, plus the `RecurrentGenome` and `HyperNEAT` paths: every one of them ran a full generation on this session's tree (configurations and outputs under "Measured: numpy vs iree").
These are IREE **3.13.0** numbers (`iree-base-compiler`/`iree-base-runtime`, what `uv sync --extra iree` now resolves); the 3.11.0 session this file used to record is superseded and only git holds it.
`xla` had no plugin in this session: the evoxc xla adapter needs a PJRT C API plugin `.so` (`EVOXC_PJRT_PLUGIN`, unset here) and the session's search found none, so this file measures `iree` on **llvm-cpu** only and carries no accelerator row. That absence is no longer a property of this box — `jax-rocm7-pjrt`'s `jax_plugins/xla_rocm7/xla_rocm_plugin.so` is now installed and the `xla` leg is measured end to end on ROCm in `GPU.md` (`## IREE vs XLA on the same device`), which is a separate accelerator artifact of its own session, not a row of this file. `tvm` was not attempted.
The CPU performance contract therefore stays on the numpy backend (`RESULTS.md`, a generated pre-fix record — the provenance of every report in this directory is tabulated in `CONTEXT.md`).
The ROCm accelerator leg is a **different artifact**: `GPU.md` + `benchmarks/results/gpu_rocm_2026-11-05.json`. No GPU number is folded into this file.

## Provenance: which tree these numbers describe

Every number here comes from one session of `benchmarks/backend_measure.py`, one fresh process per row, run 2026-11-05 12:09–12:19 UTC on the box and versions in the environment table below.
Measured tree: **`6869a131bf6a6903792bd70a7753fb020cd6bb93`** (HEAD of this worktree), `git status --porcelain` empty before and after the session.
The code under test (`src/`) is identical to the IREE-3.13 pin commit `a5973c89bebf39aa2363a2abfae75d83070b919c`: the only commit after it rewrites `benchmarks/GPU.md` and adds `benchmarks/results/gpu_rocm_2026-11-05.json`, which this session did not touch.
One fresh process per row, because peak RSS and the one-off build are per-process; the timing rows ran strictly one at a time and nothing else of this session's ran beside them.
Measurement conditions are stated, not smoothed: the 1-minute load average was **1.45** (5/15-minute 3.12/3.03) when the first row started, and this session's own rows drive it to **8–13** while an `iree` row compiles and runs at ≈11 threads — the box is shared, so the per-repeat spread below is the honest uncertainty, not a single median.
What this session does **not** separate: the compiler move 3.11.0 → 3.13.0 and the tree move (`src/` has changed since the earlier session) both happened before this measurement, and no row here was re-measured on 3.11.0, so the new ratios describe "the current stack", not the compiler alone — the numpy reference leg moved too (pop-100 median 39.738 ms in that session, 50.452 ms here), so part of any ratio change is this tree, not the compiler.
The earlier 3.11.0 session (2026-11-04 19:36–19:59 UTC, a tree between `ce5bb88` and `90d97b4`) that this file recorded before is superseded by this one; it is not quoted anywhere below.

This file replaces an earlier version that recorded 8/8 cases dying in `Pipeline.setup()` with
`BackendError: stablehlo export: op 'scatter' ... rank-3 operand along axis 0`.
That failure was real and is fixed at its source in `src/` (section "The five gates, in the order they fired"), and the gate was reproduced again on this session's stack.
`benchmarks/results/iree.json` and `benchmarks/iree.md` are the **historical** record of that pre-fix attempt (8/8 `error`); they were not re-measured for this report.

## Environment (measured, not assumed)

| Item | Value |
|---|---|
| Date (UTC) | 2026-11-05: timing rows 12:09–12:11, coverage smoke 12:12–12:13, repeat blocks 12:15–12:19 |
| CPU | `INTEL(R) XEON(R) PLATINUM 8568Y+`, QEMU/KVM guest, 20 vCPU, 20 cores / 20 siblings (1 thread/core), 235 GiB RAM |
| `nproc` | 20 |
| Platform | `Linux-6.8.0-146-generic-x86_64-with-glibc2.39` |
| Python | 3.12.3 |
| evoxc | 0.1.0 (installed distribution metadata; `pyproject.toml` resolves it from the `v0.1.1` git tag via `[tool.uv.sources]`, so the metadata version is below the tag) |
| numpy | 2.5.4 |
| iree-base-compiler / iree-base-runtime | 3.13.0 / 3.13.0 (`importlib.metadata.version`; `iree.compiler.version.VERSION` = `3.13.0rc20261016`) |
| sympy / networkx / matplotlib / pytest | 1.14.0 / 3.7 / 3.11.2 / 9.2.0 |
| Load average (1/5/15 min) | 1.45/3.12/3.03 when the first row started; 12.25–12.59 at the busiest point (this session's own rows) |

## What runs

| backend | install | what runs | compile needed |
|---|---|---|---|
| `numpy` | always available | everything | 0.30–0.45 s per generation graph (the evoxc interpreter build) |
| `iree` (llvm-cpu) | `uv sync --extra iree` (without uv: `pip install "evoxc[iree] @ git+https://github.com/EMI-Group/evoxc@v0.1.1"`; `pyproject.toml`'s `iree` extra names the IREE wheels directly, `iree-base-compiler>=3.13,<3.14` + `iree-base-runtime>=3.13,<3.14`) | `XOR`, `XOR3d`, `CustomFuncFit`, `CartPole`, `Pendulum`, `MountainCarContinuous`, `RecurrentGenome`, `HyperNEAT` — one full generation each, verified (see the coverage note above) | 7.2–9.6 s per generation graph, one-off per static shape |
| `xla` | plugin `.so` required | nothing | not reached: `the xla adapter requires a PJRT C API plugin library (.so exporting GetPjRtApi) and none was found` (searched `/usr/local/lib`, `/usr/lib`, `/root/.local/lib`, `./`) |
| `tvm` | extra | not attempted | — |

Correctness across backends (same seed 42, XOR, `DefaultGenome`, this machine, identical on all three repeats):
the final `max fitness` is bit-identical at pop 100 (`-0.176966` on both) and pop 500 (`-0.147786` on both);
at pop 1000 the two trajectories separate at `1e-08`-scale and land on `-0.053339` (numpy) vs `-0.052824` (iree), Δ = 5.2e-04, which is float-reassociation drift in iree's reductions amplified by selection, not a semantic difference.
`test/backends/test_iree_pipeline.py` pins exactly that contract — int/key payloads `array_equal`, floats within `FLOAT_RTOL = 1e-5` / `FLOAT_ATOL = 1e-6` — and the suite is `13 passed` on this session's stack (`uv run --extra iree pytest -q test/backends`, 3 `iree` cases + 10 marker-guard cases).

## The five gates, in the order they fired

Each gate below was reproduced verbatim on this machine at the time it was found, then removed **at its source in `src/`**, never by editing evoxc (the `evoxc` checkout used for the investigation is a read-only foreign repo).
The offending op's location is `op.location` in the traced evoxc IR, which is why the errors name a TensorNEAT `file:line`.
All five constraints still hold in the installed evoxc (evoxc 0.1.0 metadata, `v0.1.1` tag) on IREE 3.13.0: gates 1, 2, 4 and 5 were re-fired this session with minimal graphs, and gate 3 was re-read from the installed `DEFERRED_OPS`/`status()`. Nothing in `src/` was changed by this session.

### Gate 1 — rank-3 `scatter` (15 sites, export-time)

```
evoxc.core.errors.BackendError: stablehlo export: op 'scatter' at .../src/tensorneat/algorithm/neat/species.py:295
on a rank-3 operand along axis 0 is not supported by the v1 exporter
(numpy put_along_axis broadcast scatter is expressible for rank-1 axis=0 and rank-2 row/column writes;
use the numpy backend for the general case)
```

Re-fired this session with a three-line graph (`evoxc.scatter(dead, 0, centre)` on `(20, 50, 4)`): same `BackendError`, same wording.
evoxc's StableHLO v1 exporter (`evoxc/backends/stablehlo/writer.py:1821 _emit_scatter`) still supports only rank-1 `axis=0`, rank-2 `axis=0` and rank-2 `axis=1`; the frontend and the numpy kernel accept any rank, so the graph runs on numpy and dies at lowering.
The scatters wrote whole gene rows into `(species, max_nodes, 4)` / `(species, max_conns, 2)` / `(species, max_conns, 1)` species-centre tables:
`species.py:295`, `:304`, `:309` (generation-0 centre tables) and `_kernels.py:665`, `:674`, `:683`, `:733`, `:734`, `:735` (species scan bodies).
Rewrites, both in `src/tensorneat/algorithm/neat/_kernels.py`:
- `first_centre_table(species_size, row, fill, dtype)` — a single-row write is `concatenate([reshape(row, (1, …)), constant fill block])`, no scatter at all (every row is written, so there is nothing to mask); `species_size == 1` short-circuits to the reshape.
- `scatter_row(table, slot, update)` — flattens the trailing axes to one row length and performs the identical write as a **rank-2 axis-0 row scatter** (`(rows, R)` operand, `(rows,)` indices, `(rows, R)` updates), then reshapes back.

### Gate 2 — `uint32` / `uint64` in the graph (capability gate, not export)

```
evoxc.core.errors.BackendError: capability drift: the iree backend cannot execute dtype uint32 (function signature)
— its declared dtypes are ['bool', 'float16', 'float32', 'float64', 'int16', 'int32', 'int64', 'int8']
```

Re-fired this session with `evoxc.cast(x, evoxc.uint32)`: same `BackendError`, same eight-entry dtype list (the only change in 3.13 is the position note `(function signature)` where the old text said `(op 'cast')`).
The iree adapter declares no unsigned dtype (`evoxc/backends/adapters/iree.py::_dtype_capabilities()` — the writer's own `DTYPE_MAP` does carry `ui8`…`ui64`, the *adapter* capability set does not), and this library used unsigned carriers in three places:
- genome row identity packing (`genome/operations/alignment.py`, `uint64`) — used as a *sort key*, hence unsigned on purpose;
- key-range tests (`genome/operations/distance/{consts,digest}.py`, `common/tensorization.py`, `uint32`) — one `<` comparison instead of two;
- a `uint32` constant created through `common/tools.py:159`.

Rewritten as **same-width signed carriers**: two's complement makes add/sub/shift/or/xor/cast bit-identical, so the packed values do not move.
Where unsigned *order* mattered (the `argsort`/comparison on the packed identity) it is restored by one bijective sign flip, `bitwise_xor(v, INT64_MIN)` (`alignment.py::_order_ready`) — equality preserved, signed order equals unsigned order.
The `uint32` range trick became `logical_and(greater_equal(keys, 0), less(keys, span))` (`common/tensorization.py::_key_indices`), i.e. the same two-comparison form the float branch already used.
NumPy results are byte-identical after all of it (per-node A/B dumps: 144 arrays at pop 100, 72 at pop 500, 72 at pop 1000 in `algorithm/neat`; 23 arrays across gene ops; 108 schedule fields in `common`).

### Gate 3 — `tril`, `flip`, `cumsum` (ops the v1 exporter defers)

```
evoxc.core.errors.BackendError: stablehlo export: op 'tril' at .../src/tensorneat/algorithm/neat/_kernels.py:140 is not supported in v1
```

Re-read this session from the installed evoxc: `tril`, `triu`, `flip`, `cumsum`, `cumprod`, `where`, `scan`, `convolution`, `solve`, `erf`, `gelu`, `runtime_call`, `external_call` all report `status(...) == "deferred"` (they are in `evoxc/backends/stablehlo/ops.py:235 DEFERRED_OPS` or absent from every mapping table).
Rewrites:
- `tril` (`_kernels.py:140`, species prefix sum) → a **static constant** triangular mask: the matrix is `(species_size, species_size)` and fully static, so it is one `constant` op instead of a deferred op (and cheaper to run).
- `flip` (`_kernels.py:295`) → `evoxc.gather` with a static reversed `int32` index vector.
- `cumsum` (`genome/operations/mutation/cycle.py:121`, a `(P, 214)` suffix count) → a log2-doubling (Hillis–Steele) fold: a `while` over `shift < total` adding a shifted slice, in `int8`/`int32`. Bit-identical, ~24 extra traced ops, +0.3 ms at pop 100.

### Gate 4 — `iree-compile` could not legalize a shared index block (`iree_linalg_ext.scatter`)

```
<stdin>:11:10: error: failed to legalize unresolved source materialization from ('tensor<32x1xi64>') to ('tensor<32x1xi32>') that remained live after conversion (no matching callback)
    %8 = stablehlo.reshape %7 : (tensor<32xi64>) -> tensor<32x1xi64>
<stdin>:12:11: note: see existing live user here: %9 = unrealized_conversion_cast %8 : tensor<32x1xi32> to tensor<32xi32>
    %11 = "stablehlo.scatter"(%arg0, %8, %arg2) ({
```

Re-fired this session on IREE 3.13.0 with two rank-1 `scatter`s that share **one** index input: `iree-compile` still fails (`exit 1`, wrapped by evoxc as `BackendError`), and the control with two *distinct* index tensors compiles and runs — so the rule below is still enforced by the compiler, not by luck. The 3.13 wording is a "source materialization" of the index tensor where 3.11 said "unresolved materialization"; the same rank-2→rank-1 collapse is the thing that must stay single-consumer, and the same failure reproduces for the rank-2 row-scatter form `src/` actually uses.
This one is IREE-specific and is **not** an export deferral: the op passes evoxc's exporter and even compiles fine in isolation.
The writer emits every flat write as a rank-1 `axis=0` scalar-window scatter with a rank-2 `(K,1) i64` index tensor (`index_vector_dim = 1`).
IREE lowers that to `iree_linalg_ext.scatter`, whose `ins` wants the index tensor **collapsed to rank 1 i32**, and performs the collapse as an `unrealized_conversion_cast`. That materialization only resolves if the rank-2 index value has **exactly one consumer**; when one prepared index block (or two identical index expressions that IREE's CSE merges) feeds a **second** scatter, the rank-2 value stays live and the pass aborts.
Operand sharing, widths, i32-vs-i64 indices, `while`/`if` regions, `sort`, and the rank-2 row-scatter alternative are all irrelevant — deleting either of the two shared-index scatters from the failing module makes it compile.

Fix, in the shared helper's caller (`src/tensorneat/common/tensorization.py::incoming_table`, `_window_pair`): the two payloads that shared one index block (`in_conn`, `in_src`) now go into **one** `(P, 2*(N*K+1))` buffer written by **one** scatter, with the second half's cells offset by one window span and its dropped writes routed to its own dump column.
The resulting rule is now documented in `common/tools.py` and `common/CONTEXT.md`: **one prepared index block feeds exactly one scatter** — merge payloads into a wider buffer instead.

### Gate 5 — a negative `concatenate` axis (HyperNEAT only, `iree-compile` reject)

```
<stdin>:3:10: error: 'stablehlo.concatenate' op attribute 'dimension' failed to satisfy constraint:
64-bit signless integer attribute whose value is non-negative
    %0 = "stablehlo.concatenate"(%arg0, %arg1) {dimension = -1 : i64} : (tensor<160x3xf32>, tensor<160x1xf32>) -> tensor<160x4xf32>
```

Re-fired this session with a two-operand `evoxc.concatenate([a, b], axis=-1)` graph on IREE 3.13.0: identical constraint error. evoxc still forwards the frontend axis **verbatim** (`writer.py:1292 _emit_concatenate` writes `int(axis)` with no normalization), and StableHLO requires it non-negative, so the `axis=-1` spelling — perfectly legal at the frontend and in numpy — only fails once a compiled backend sees it.
Reached only by the HyperNEAT path (the plain NEAT graphs never wrote a negative axis), so it was invisible until the compiled run got that far.
Fix: write the explicit non-negative last axis in graph code — `hyperneat.py` (bias column, attribute stack, connection-key marker columns) and `genome/gene/base.py::_key_array` (`axis=len(shape)`).
Rule: **axes in graph ops are non-negative static integers**; the sweep found no other negative `concatenate` axis in `src/`.
End-to-end after the fix on this session's tree: `HyperNEAT(FullSubstrate(4/3/1), activate_time=3)` + `XOR3d`, seed 42, `backend="iree"` → one 9.143 s compile, then one full generation, finite, max fitness `-0.326813` (see the coverage table).

## Measured: numpy vs iree

Protocol: `benchmarks/backend_measure.py`, one `(case, backend, pop, opt_level)` per **fresh process** (peak RSS is a process high-water mark).
`Pipeline.setup()` wall time is reported as `setup_s` (host-only plus NEAT's small operator graph); the expensive generation graph compiles on the **first** `step()`, reported as `build_s` (`Pipeline.build_seconds`, the `evoxc.build` alone) and `first_step_s` (build + one run). The build is charged to no generation.
Then **3 untimed warmup** generations and **20 timed** generations; ms/generation is wall clock per `step()` including host materialisation.
Seed 42, XOR, `NEAT(pop_size=P, species_size=20, survival_threshold=0.01)`, `DefaultGenome(2→1, max_nodes=50, max_conns=100, output_transform=ACT.sigmoid)`.
**Three repeats per `(backend, pop)` cell**, run back to back in the same session (`r1`/`r2`/`r3`); the machine-readable records are scratch files, so every cell below is the measured per-run value, and the repeat spread is reported rather than averaged away.
Measured 2026-11-05, 12:09–12:19 UTC, on the box in the environment table; the exact commands are in "Reproduce".
`gens/s` is `1000 / mean`, exactly as the CLI reports it.

| run | pop | backend | opt_level | setup (s) | build (s) | ms/gen mean | ms/gen median | min / max | gens/s | peak RSS (MiB) | final max fitness |
|---|---|---|---|---|---|---|---|---|---|---|---|
| r1 | 100 | numpy | — | 0.034 | 0.315 | 50.676 | **50.561** | 38.788 / 64.361 | 19.733 | 121.61 | -0.176966 |
| r2 | 100 | numpy | — | 0.015 | 0.297 | 48.909 | **50.191** | 39.296 / 59.342 | 20.446 | 114.55 | -0.176966 |
| r3 | 100 | numpy | — | 0.016 | 0.306 | 50.435 | **50.452** | 39.845 / 62.967 | 19.828 | 114.50 | -0.176966 |
| r1 | 100 | iree | default | 1.081 | 8.013 | 70.219 | **75.166** | 34.667 / 112.073 | 14.241 | 246.75 | -0.176966 |
| r2 | 100 | iree | default | 1.156 | 8.611 | 71.437 | **77.809** | 34.303 / 112.878 | 13.998 | 244.11 | -0.176966 |
| r3 | 100 | iree | default | 1.069 | 8.319 | 74.735 | **80.883** | 35.272 / 128.652 | 13.381 | 244.29 | -0.176966 |
| r1 | 100 | iree | O0 | 1.049 | 8.467 | 76.469 | **80.965** | 35.552 / 118.780 | 13.077 | 244.19 | -0.176966 |
| r1 | 100 | iree | O1 | 1.046 | 8.536 | 68.211 | **73.034** | 34.261 / 105.724 | 14.660 | 244.06 | -0.176966 |
| r1 | 500 | numpy | — | 0.016 | 0.349 | 185.125 | **183.451** | 176.873 / 202.442 | 5.402 | 235.12 | -0.147786 |
| r2 | 500 | numpy | — | 0.016 | 0.365 | 168.841 | **167.377** | 164.033 / 190.398 | 5.923 | 235.27 | -0.147786 |
| r3 | 500 | numpy | — | 0.020 | 0.421 | 177.463 | **175.643** | 168.187 / 200.446 | 5.635 | 234.91 | -0.147786 |
| r1 | 500 | iree | default | 1.024 | 8.413 | 979.880 | **930.899** | 873.147 / 1238.120 | 1.020 | 292.44 | -0.147786 |
| r2 | 500 | iree | default | 1.049 | 8.315 | 981.239 | **960.220** | 859.058 / 1143.003 | 1.019 | 290.77 | -0.147786 |
| r3 | 500 | iree | default | 1.102 | 9.178 | 984.964 | **977.079** | 940.764 / 1079.465 | 1.015 | 292.02 | -0.147786 |
| r1 | 1000 | numpy | — | 0.016 | 0.447 | 327.922 | **325.626** | 314.022 / 381.849 | 3.050 | 379.52 | -0.053339 |
| r2 | 1000 | numpy | — | 0.017 | 0.426 | 299.572 | **298.426** | 293.803 / 323.601 | 3.338 | 379.83 | -0.053339 |
| r3 | 1000 | numpy | — | 0.018 | 0.451 | 322.241 | **318.269** | 310.155 / 354.670 | 3.103 | 379.98 | -0.053339 |
| r1 | 1000 | iree | default | 1.158 | 8.888 | 2287.059 | **2228.054** | 2126.798 / 3425.638 | 0.437 | 346.90 | -0.052824 |
| r2 | 1000 | iree | default | 1.177 | 8.917 | 1836.339 | **1754.934** | 1639.174 / 2254.813 | 0.545 | 345.23 | -0.052824 |
| r3 | 1000 | iree | default | 1.143 | 9.646 | 2151.265 | **2109.376** | 1650.414 / 3677.170 | 0.465 | 343.99 | -0.052824 |

Repeat medians: numpy 50.452 / 175.643 / 318.269 ms at pop 100 / 500 / 1000; iree 77.809 / 960.220 / 2109.376 ms.
Per-repeat paired ratios (iree ÷ numpy, median over median): pop 100 **1.49× / 1.55× / 1.60×**, pop 500 **5.07× / 5.74× / 5.56×**, pop 1000 **6.84× / 5.88× / 6.63×**.

Coverage smoke for the problem classes and network paths beyond the XOR timing rows (one full generation each, `generation_limit=1`, `pop_size=20`, `species_size=4`, `survival_threshold=0.1`, seed 42, one fresh process per row; command in "Reproduce"):

| problem / path | config | build (s) | one generation |
|---|---|---|---|
| `XOR3d` | `DefaultGenome(3→1, 50/100)`, sigmoid read-out | 7.869 | finite, max fitness -0.284553 |
| `CustomFuncFit` | Pagie sample, `num_samples=50`, 2→1 | 7.761 | finite, max fitness -0.045864 |
| `CartPole` | `max_step=200`, `repeat_times=1`, 4→2 | 7.902 | finite, max fitness 56.0 |
| `Pendulum` | `max_step=200`, 2→1 with a 6-node hidden layer | 7.791 | finite, max fitness -1062.370850 |
| `MountainCarContinuous` | `max_step=999`, 2→1, no hidden layer | 7.677 | finite, max fitness -0.697044 |
| `RecurrentGenome` | `activate_time=3`, 3→1, 16 node slots, `max_in_degree=16` | 7.183 | finite, max fitness -0.284553 |
| `HyperNEAT` | `FullSubstrate(4/3/1)`, `activate_time=3` | 9.143 | finite, max fitness -0.326813 |

Reading it honestly:

- **numpy still wins every measured population, by less than it used to.** Slower by 1.54× at pop 100, 5.47× at pop 500, 6.63× at pop 1000 (median-of-repeats ratios); the per-repeat spread is 1.49–1.60×, 5.07–5.74× and 5.88–6.84×, so the iree leg is still behind everywhere but the pop-100 cell is now inside a factor a laptop user would call "a bit slower" rather than two orders of magnitude. The ratio still **jumps** between pop 100 and pop 500 (1.5× → 5.5×) and then grows slowly (5.5× → 6.6×), so there is still no measured population at which iree flips ahead.
- **The slowdown is not stationary inside a run, and at pop 100 iree starts ahead.** The 20 timed generations of the pop-100 cell drift upward on both backends, but far harder on iree: its first five samples average 35.4–36.4 ms against numpy's 40.6–41.1 ms (iree ≈ 14 % *faster*), and its last five average 104–108 ms against numpy's 57–61 ms (≈ 1.8× slower), with the crossover around generation 6–8. The median therefore understates iree at the start of an evolution and overstates it later; at pop 500 and pop 1000 the drift is mild (iree 956 → 1000 ms, numpy 172 → 182 ms at pop 500). The likely mechanism is data-dependent work (the `while`-loop trip counts and the gated speciation rows grow as the population diverges), not a leak.
- **Compile dominates the small-population row.** Using `compile ÷ (setup + build + 23 × ms/gen)` (the 3 warmup + 20 timed generations after the build), compile is **74 %** of that wall clock at pop 100, **26–28 %** at pop 500 and **15–18 %** at pop 1000; the one-off `iree-compile` is 8.0–9.6 s for the generation graph against 0.30–0.45 s for the numpy build.
- **Peak RSS is a compile-phase artefact, not a generation-loop cost.** Raw peaks: 244–247 vs 114.5–121.6 MiB (pop 100), 290.8–292.4 vs 234.9–235.3 MiB (pop 500), 344.0–346.9 vs 379.5–380.0 MiB (pop 1000) — so iree only wins the raw number at pop 1000 (-8.6 %). But the process peak includes the one-off `iree-compile`: a fresh process that imports numpy+evoxc and compiles one 64×64 `matmul` for iree already peaks at **186.59 MiB**, and the compile runs inside the measured row. Subtracting that floor, the generation loop itself costs iree ≈ 58 MiB at pop 100, ≈ 105 MiB at pop 500 and ≈ 159 MiB at pop 1000 against numpy's 115/235/380 MiB, i.e. the compiled loop is the lighter one at every population and its footprint barely grows with the padded payload while the interpreter's triples. The earlier session's lower iree peaks (111.91 MiB at pop 100) belong to its 3.11.0 compiler build and are not comparable with this one's.
- **`opt_level`: O0, O1 and the default are indistinguishable; O2 and O3 now crash the compiler.** Pop-100 medians: default 75.166 / 77.809 / 80.883 (three repeats), `O0` 80.965, `O1` 73.034 — all inside the repeat spread, so nothing here justifies a flag. `O2` (and `O3`) **SIGSEGV inside `iree-compile`**: `Error code: -11` with a 45-frame `libIREECompiler.so` stack dump, preceded by 8 `warning: failed to move consumer into region` diagnostics on `stablehlo.select`, forwarded verbatim and surfaced as
  `evoxc.core.errors.BackendError: iree-compile failed to compile the StableHLO program for the iree backend (targets: llvm-cpu): Error invoking IREE compiler tool iree-compile / Error code: -11`; nothing falls back to another backend.
- The earlier **`iree-compile` SIGSEGV (≥3 loop-carried tensors)** was a real bug, not a configuration artefact: `genome/operations/mutation/cycle.py` seeded a `while_loop`-carried value with a **boolean `scatter`**; the seed is now a one-hot `equal`/`logical_and`, and the SIGSEGV has not reproduced since that fix (`PERF_NUMPY.md`, "IREE stayed compilable"). The generation graph still contains a `while_loop`, and every successful `iree` row above (11 timing rows plus the 7 coverage rows) compiled and executed it. Rule: **never seed a `while`-carried value with a `scatter`**.

Per-generation cost is dominated by the graph's dense `gather`/`argsort`/`scatter` traffic, which llvm-cpu lowers to generic, non-vectorised loops — the reason a compiled backend still loses to a tight numpy interpreter on this workload, 3.13 or not. Making iree competitive is a kernel-formulation project (dense key tables → sparse/segment reductions), not a compiler-flag one.

## What a user must know to run `backend="iree"`

- `uv sync --extra iree` pulls `iree-base-compiler` + `iree-base-runtime` (3.13.x on this tree); then `Pipeline(..., backend="iree")` just works for the shipped problems.
  Without uv, install the same extra from the pinned tag: `pip install "evoxc[iree] @ git+https://github.com/EMI-Group/evoxc@v0.1.1"`, plus `pip install --no-deps .` for tensorneat itself.
  Plain `pip install "evoxc[iree]"` is not supported: PyPI only ships `evoxc` 0.1.0, while `pyproject.toml` resolves the git tag `v0.1.1` through `[tool.uv.sources]` (and its `evoxc >= 0.1.1` floor cannot be satisfied from PyPI).
- **Compile is one-off per static shape**: population, `max_nodes`, `max_conns` are baked into the graph, so a shape change re-compiles (~8–10 s here). `Pipeline` caches the executable per shape, and the build is charged to no generation.
- **Kernel rules the exporter and iree impose** (all enforced by construction in `src/` today, and documented in `src/tensorneat/common/CONTEXT.md`, `src/tensorneat/algorithm/neat/CONTEXT.md`, `src/tensorneat/algorithm/hyperneat/CONTEXT.md`, `src/tensorneat/genome/CONTEXT.md` and `src/tensorneat/genome/operations/CONTEXT.md`):
  1. `scatter` must be rank-1 `axis=0` or a rank-2 row/column write — no rank ≥ 3 operand.
  2. One prepared index block feeds exactly one `scatter` (gate 4) — merge payloads into a wider buffer instead.
  3. No unsigned dtype anywhere in a graph: use same-width signed carriers, plus an explicit sign flip wherever unsigned *order* is required.
  4. No v1-deferred op: `cumsum`, `cumprod`, `tril`, `triu`, `flip`, `erf`, `gelu`, `where`, `scan`, `convolution`, `solve`, `runtime_call`, `external_call`, and the `sparse_*` family. Use a static constant, a `gather` with a static index vector, or a log2-doubling fold instead.
  5. Static dims only, **non-negative axes only**, and `select`/`gather` (single axis)/`sort`/`argsort`/`where`-free masks/`cond`/`while_loop`/static-bound slices/`rng_bit_generator` all export.
- **Activation functions**: no `ACT` entry needs `erf`/`gelu` (the set is `scaled_sigmoid`, `sigmoid`, `scaled_tanh`, `tanh`, `sin`, `relu`, `lelu`, `identity`, `inv`, `log`, `exp`, `abs`); a *user-supplied* activation that calls `evoxc.erf`/`evoxc.gelu` would still die at export with `op 'erf' ... is not supported in v1`.
- `opt_level="O2"`/`"O3"` **crash `iree-compile`** (`Error code: -11`) for this graph size on IREE 3.13.0. Leave the default, or use `O0`/`O1`; the three are within the measured repeat spread.
- Do not expect an iree speed-up: on this CPU, numpy is 1.5–6.8× faster per generation across the three measured populations. iree's measured advantages are a lighter generation loop (≈ 58/105/159 MiB against 115/235/380 MiB once the compile floor is subtracted) and a foot in the door on an accelerator, where `GPU.md` owns the numbers.
- Peak RSS is a **process** high-water mark that includes the one-off compile: an `iree` process carries ≈ 187 MiB of compiler library even when the generation loop needs 60 MiB.

## Still deferred by the evoxc v1 exporter (would bite other users)

Straight from the installed evoxc (`v0.1.1` tag, 0.1.0 metadata), re-probed this session, not inferred:
- `evoxc/backends/stablehlo/ops.py:235 DEFERRED_OPS` — 30 names: `runtime_call`, `external_call`, `block_call`, `rank`, `world_size`, `erf`, `gelu`, `diagonal`, `cholesky`, `qr`, `matrix_rank`, `svd`, `matrix_exp`, `random_multinomial`, plus the 16 `sparse_*`/`dense_dot_sparse` ops (`sparse_from_dense`, `sparse_to_dense`, `sparse_coo_to_csr`, `sparse_csr_to_coo`, `sparse_coo_to_csc`, `sparse_csc_to_coo`, `sparse_negate`, `sparse_add`, `sparse_multiply`, `sparse_multiply_dense`, `sparse_reduce_sum`, `sparse_transpose`, `sparse_reshape`, `sparse_concatenate`, `sparse_dot_sparse`, `dense_dot_sparse`).
- Names with no v1 mapping at all (`status(...) == "deferred"` but absent from that set): `cumsum`, `cumprod`, `flip`, `tril`, `triu`, `where`, `scan`, `convolution`, `solve`.
- Collectives are the opposite case: every `evoxc.dist` op **exports** (`evoxc.dist.broadcast` is the op name `broadcast_collective`, mapped to `stablehlo.collective_broadcast`), but `iree-compile` cannot legalize `stablehlo.collective_broadcast` for llvm-cpu and the runtime wheels ship no channel provider, so the adapter declares `collectives=False` and `lower()` rejects them explicitly.
- `evoxc/backends/stablehlo/writer.py:1821 _emit_scatter`: rank ≥ 3 operands, and any rank-2 form other than a full row or column write; at most one consumer per prepared index block (IREE, gate 4).
- iree's declared dtype set has no unsigned integer and no complex dtype (`evoxc/backends/adapters/iree.py::_dtype_capabilities`: `bool, float16, float32, float64, int8, int16, int32, int64`).
- Dynamic (non-static) dims cannot be exported at all.

## Reproduce

```bash
export PATH=/root/.local/bin:$PATH      # uv on this box
uv sync --extra iree                    # resolves iree-base-compiler/-runtime 3.13.x

# one fresh process per row — peak RSS and the build are per-process
uv run --extra iree python benchmarks/backend_measure.py --backend numpy --case xor --pop 100
uv run --extra iree python benchmarks/backend_measure.py --backend numpy --case xor --pop 500
uv run --extra iree python benchmarks/backend_measure.py --backend numpy --case xor --pop 1000
uv run --extra iree python benchmarks/backend_measure.py --backend iree  --case xor --pop 100
uv run --extra iree python benchmarks/backend_measure.py --backend iree  --case xor --pop 500
uv run --extra iree python benchmarks/backend_measure.py --backend iree  --case xor --pop 1000
uv run --extra iree python benchmarks/backend_measure.py --backend iree  --case xor --pop 100 --opt-level O0
uv run --extra iree python benchmarks/backend_measure.py --backend iree  --case xor --pop 100 --opt-level O1
uv run --extra iree python benchmarks/backend_measure.py --backend iree  --case xor --pop 100 --opt-level O2  # SIGSEGV inside iree-compile
# repeat each of the six core rows twice more for the r1/r2/r3 spread reported above

# coverage smoke: one full generation per RL class, one fresh process each
for spec in CartPole:4:2::200 Pendulum:2:1:6:200 MountainCarContinuous:2:1::999; do
  IFS=: read -r CLASS INPUTS OUTPUTS HIDDEN MAX_STEP <<< "$spec"
  PROBLEM="$CLASS" INPUTS="$INPUTS" OUTPUTS="$OUTPUTS" HIDDEN="$HIDDEN" MAX_STEP="$MAX_STEP" uv run --extra iree python - <<'PY'
import os, numpy as np
from tensorneat import Pipeline, algorithm, genome, problem
from tensorneat.common import ACT, AGG, to_numpy
cls = os.environ["PROBLEM"]
hidden = tuple(int(v) for v in os.environ["HIDDEN"].split(",") if v)
kwargs = {"max_step": int(os.environ["MAX_STEP"])}
if cls == "CartPole":
    kwargs["repeat_times"] = 1
neat = algorithm.NEAT(pop_size=20, species_size=4, survival_threshold=0.1,
    genome=genome.DefaultGenome(num_inputs=int(os.environ["INPUTS"]),
        num_outputs=int(os.environ["OUTPUTS"]), init_hidden_layers=hidden,
        node_gene=genome.BiasNode(activation_options=ACT.tanh, aggregation_options=AGG.sum)))
p = Pipeline(algorithm=neat, problem=getattr(problem, cls)(**kwargs), seed=42,
             generation_limit=1, backend="iree", is_detailed=False)
state = p.setup(); state, best, fitness = p.step(state)
fitness = to_numpy(fitness)
print(cls, "build_s=", round(p.build_seconds, 3), "generation=", int(to_numpy(state.generation)),
      "finite=", bool(np.all(np.isfinite(fitness))), "max_fitness=", float(np.max(fitness)))
PY
done

# coverage smoke: XOR3d, CustomFuncFit, RecurrentGenome, HyperNEAT — one generation each
uv run --extra iree python - <<'PY'
import numpy as np
from tensorneat import Pipeline, algorithm, genome, problem
from tensorneat.common import ACT, to_numpy
from tensorneat.algorithm.hyperneat import FullSubstrate, HyperNEAT

def neat(genome_obj):
    return algorithm.NEAT(pop_size=20, species_size=4, survival_threshold=0.1,
                          genome=genome_obj)

def run(name, algo, prob):
    p = Pipeline(algorithm=algo, problem=prob, seed=42, generation_limit=1,
                 backend="iree", is_detailed=False)
    state = p.setup(); state, best, fitness = p.step(state)
    fitness = to_numpy(fitness)
    print(name, "build_s=", round(p.build_seconds, 3),
          "generation=", int(to_numpy(state.generation)),
          "finite=", bool(np.all(np.isfinite(fitness))),
          "max_fitness=", float(np.max(fitness)))

run("XOR3d", neat(genome.DefaultGenome(num_inputs=3, num_outputs=1,
        max_nodes=50, max_conns=100, output_transform=ACT.sigmoid)), problem.XOR3d())
run("CustomFuncFit", neat(genome.DefaultGenome(num_inputs=2, num_outputs=1,
        max_nodes=50, max_conns=100, output_transform=ACT.sigmoid)),
    problem.CustomFuncFit(func=lambda p: np.array(
        [1/(1+p[0]**-4)+1/(1+p[1]**-4)], dtype=np.float32),
        low_bounds=[-1.0, -1.0], upper_bounds=[1.0, 1.0], method="sample",
        num_samples=50, seed=42))
run("RecurrentGenome", neat(genome.RecurrentGenome(num_inputs=3, num_outputs=1,
        max_nodes=16, max_in_degree=16, activate_time=3,
        output_transform=ACT.sigmoid)), problem.XOR3d())
run("HyperNEAT", HyperNEAT(
        substrate=FullSubstrate(
            input_coors=((-1, -1), (-0.33, -1), (0.33, -1), (1, -1)),
            hidden_coors=((-1, 0), (0, 0), (1, 0)),
            output_coors=((0, 1),)),
        neat=neat(genome.DefaultGenome(num_inputs=4, num_outputs=1,
            max_nodes=50, max_conns=100, output_transform=ACT.tanh)),
        activation=ACT.tanh, activate_time=3, output_transform=ACT.sigmoid),
    problem.XOR3d())
PY
```

The end-to-end backend test (`test/backends/test_iree_pipeline.py`) skips cleanly without the extra and runs a real XOR generation with it:

```bash
uv run --extra iree pytest -q test/backends   # 13 passed (3 iree + 10 marker-guard), or 10 passed + 1 skipped when the extra is absent
uv run --extra iree pytest -q -m requires_iree test/backends
```

Measured session wall clock: the eight timing rows (six core rows plus the `O0` and `O2` probes) took 2 min 27 s (12:09:00 → 12:11:27 UTC), the coverage smoke 66 s (12:12:51 → 12:13:57), the pop-100 repeats plus the `opt_level` probes ~35 s, and the two extra repeat blocks 4 min 3 s (12:15:20 → 12:19:23) — ~10.5 min in total.
That is well inside the half-hour budget the 3.11 numbers implied (14.9 s and 29.9 s per generation then, 0.96 s and ~2.1 s now), which is itself the headline: the same rows now cost seconds, not minutes.
The `setup_s`, `build_s` and peak-RSS columns are per-process values and cannot be summed across rows; every ms/generation column is one process's 20 timed generations after 3 untimed ones.
A pop-1000 numpy/iree row also prints `UserWarning: Embedding a tensor of 1608000 bytes as a graph constant (threshold 1048576 bytes, set via EVOXC_LARGE_CONSTANT_BYTES)` once per process — evoxc's large-constant notice for the padded gene geometry, not a measurement failure.

The minimal reproducers for gates 1–5 are ~20 lines of `evoxc` each (a rank-3 scatter, a `uint32` cast, a deferred op, two scatters sharing one index input and their distinct-index control, and a `concatenate(axis=-1)`); the recipes (trace the graphs, walk `op.location`, export with `evoxc.lower(graph, backend="iree").text()`, compile the text with `iree.compiler.compile_str`) live in `src/tensorneat/common/CONTEXT.md` and `src/tensorneat/algorithm/neat/CONTEXT.md`.

## What this does not change

`RESULTS.md` is generated evidence for the old-vs-new CPU contract and was not regenerated or hand-edited for this report: the contract is measured on the numpy backend and that is still what it measures.
`benchmarks/results/iree.json` + `benchmarks/iree.md` remain the historical pre-fix record (8/8 `error`); the numbers above supersede them for the code as of this session, which is the tree the ratios in this file belong to.
`PERF_NUMPY.md` remains the numpy-interpreter performance record, and `GPU.md` + `benchmarks/results/gpu_rocm_2026-11-05.json` remain the accelerator record — neither was re-measured or edited here.
A GPU optimizer pass landing in `src/tensorneat` in parallel with this session may change these numbers afterwards: every row above describes the stamped tree `6869a131` and nothing later, so re-run the commands in "Reproduce" before quoting them for a newer tree.
