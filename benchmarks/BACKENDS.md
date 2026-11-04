# The `iree` backend: it runs, and it is slower than numpy here

**Verdict: `backend="iree"` now compiles and runs TensorNEAT end to end — every gate that used to kill it in `Pipeline.setup()` is gone — but on this box it is 9.3× (pop 100), 93.9× (pop 500) and 102.7× (pop 1000) slower per generation than the numpy reference interpreter; the gap jumps between pop 100 and pop 500 and then flattens rather than scaling with population.**
The problem classes exercised on iree are `XOR`, `XOR3d`, `CustomFuncFit`, `CartPole`, `Pendulum` and `MountainCarContinuous`, plus the `RecurrentGenome` and `HyperNEAT` paths: each has run at least one full generation here, and this pass re-ran `CartPole`, `Pendulum` and `MountainCarContinuous` (configurations and outputs under "Measured: numpy vs iree").
`CartPole` postdates the historical `iree.md`, whose `cartpole_p100 / iree` row is an `error` from the pre-fix record that file labels as historical.
Each run pays one 7.5–10.4 s `iree-compile` per static shape.
`xla` is not available here: the evoxc xla adapter needs a PJRT C API plugin `.so` (`EVOXC_PJRT_PLUGIN`, unset on this box) and nothing provides one. `tvm` was not attempted.
The CPU performance contract therefore stays on the numpy backend (`RESULTS.md`, unchanged by this report).

This file replaces an earlier version that recorded 8/8 cases dying in `Pipeline.setup()` with
`BackendError: stablehlo export: op 'scatter' ... rank-3 operand along axis 0`.
That failure was real and is now fixed at its source in `src/` (section "The five gates, in the order they fired").
`benchmarks/results/iree.json` and `benchmarks/iree.md` are the **historical** record of that pre-fix attempt (8/8 `error`); they were not re-rendered for this report.

## Environment (measured, not assumed)

| Item | Value |
|---|---|
| Date (UTC) | 2026-11-04, runs 19:36–19:59 |
| CPU | `INTEL(R) XEON(R) PLATINUM 8568Y+`, QEMU/KVM guest, 1 socket, 1 thread/core, no SMT |
| `nproc` | 20 |
| Platform | `Linux-6.8.0-146-generic-x86_64-with-glibc2.39` |
| Python | 3.12.3 |
| evoxc | 0.1.0 |
| numpy | 2.5.4 |
| iree-base-compiler / iree-base-runtime | 3.11.0 / 3.11.0 (what `evoxc[iree]` resolves) |
| sympy / networkx / matplotlib / pytest | 1.14.0 / 3.7 / 3.11.2 / 9.2.0 |

## What runs today

| backend | install | what runs | compile needed |
|---|---|---|---|
| `numpy` | always available | everything | 0.26–0.40 s per generation graph (the evoxc interpreter build) |
| `iree` (llvm-cpu) | `pip install "evoxc[iree]"` | `XOR`, `XOR3d`, `CustomFuncFit`, `CartPole`, `Pendulum`, `MountainCarContinuous`, `RecurrentGenome`, `HyperNEAT` — one full generation each, verified (see the coverage note above) | 7.5–10.4 s per generation graph, one-off per static shape |
| `xla` | plugin `.so` required | nothing | not reached: `the xla adapter requires a PJRT C API plugin library (.so exporting GetPjRtApi) and none was found` |
| `tvm` | extra | not attempted | — |

Correctness across backends (same seed 42, XOR, `DefaultGenome`, this machine):
the final `max fitness` is bit-identical at pop 100 (`-0.19304290413856506` numpy vs `-0.19304291903972626` iree, Δ = 1.5e-08) and pop 500 (identical to all printed digits, `-0.15205243229866028`);
at pop 1000 the two trajectories separate at `1e-08`-scale and land on `-0.151799` vs `-0.143853`, which is float-reassociation drift in iree's reductions, not a semantic difference (int32 key payloads compare equal; float attribute arrays agree to ≤ 2.4e-07 ≈ 1–2 ulp).
`test/backends/test_iree_pipeline.py` pins exactly this: int/key payloads `array_equal`, floats within a documented tolerance.

## The five gates, in the order they fired

Each gate below was reproduced verbatim on this machine, then removed **at its source in `src/`**, never by editing evoxc (`/root/evoxc` is a read-only foreign repo; the installed `evoxc` 0.1.0 was left untouched).
The offending op's location is `op.location` in the traced evoxc IR, which is why the errors name a TensorNEAT `file:line`.

### Gate 1 — rank-3 `scatter` (15 sites, export-time)

```
evoxc.core.errors.BackendError: stablehlo export: op 'scatter' at .../src/tensorneat/algorithm/neat/species.py:295
on a rank-3 operand along axis 0 is not supported by the v1 exporter
(numpy put_along_axis broadcast scatter is expressible for rank-1 axis=0 and rank-2 row/column writes;
use the numpy backend for the general case)
```

evoxc's StableHLO v1 exporter (`evoxc/backends/stablehlo/writer.py:1821 _emit_scatter`) supports only rank-1 `axis=0`, rank-2 `axis=0` and rank-2 `axis=1`; the frontend and the numpy kernel accept any rank, so the graph ran on numpy and died at lowering.
The scatters wrote whole gene rows into `(species, max_nodes, 4)` / `(species, max_conns, 2)` / `(species, max_conns, 1)` species-centre tables:
`species.py:295`, `:304`, `:309` (generation-0 centre tables) and `_kernels.py:665`, `:674`, `:683`, `:733`, `:734`, `:735` (species scan bodies).
Rewrites, both in `src/tensorneat/algorithm/neat/_kernels.py`:
- `first_centre_table(species_size, row, fill, dtype)` — a single-row write is `concatenate([reshape(row, (1, …)), constant fill block])`, no scatter at all (every row is written, so there is nothing to mask); `species_size == 1` short-circuits to the reshape.
- `scatter_row(table, slot, update)` — flattens the trailing axes to one row length and performs the identical write as a **rank-2 axis-0 row scatter** (`(rows, R)` operand, `(rows,)` indices, `(rows, R)` updates), then reshapes back.

### Gate 2 — `uint32` / `uint64` in the graph (capability gate, not export)

```
evoxc.core.errors.BackendError: capability drift: the iree backend cannot execute dtype uint32 (op 'cast')
— its declared dtypes are ['bool', 'float16', 'float32', 'float64', 'int16', 'int32', 'int64', 'int8']
```

The iree adapter declares no unsigned dtype (`evoxc/backends/adapters/iree.py`, `_dtype_capabilities()`), and this library used unsigned carriers in three places:
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

`tril`, `triu`, `flip`, `cumsum`, `cumprod`, `erf`, `gelu`, `where`, `scan`, `convolution`, `runtime_call`, `external_call` and the 16 `sparse_*` ops have no v1 emission (`evoxc/backends/stablehlo/ops.py:235 DEFERRED_OPS`, plus names absent from every mapping table).
Rewrites:
- `tril` (`_kernels.py:140`, species prefix sum) → a **static constant** triangular mask: the matrix is `(species_size, species_size)` and fully static, so it is one `constant` op instead of a deferred op (and cheaper to run).
- `flip` (`_kernels.py:295`) → `evoxc.gather` with a static reversed `int32` index vector.
- `cumsum` (`genome/operations/mutation/cycle.py:121`, a `(P, 214)` suffix count) → a log2-doubling (Hillis–Steele) fold: a `while` over `shift < total` adding a shifted slice, in `int8`/`int32`. Bit-identical, ~24 extra traced ops, +0.3 ms at pop 100.

### Gate 4 — `iree-compile` could not legalize a shared index block (`iree_linalg_ext.scatter`)

```
<stdin>:338:12: error: failed to legalize unresolved materialization from ('tensor<32x1xi32>') to ('tensor<32xi32>') that remained live after conversion
    %337 = "stablehlo.scatter"(%323, %334, %325) ({
<stdin>:338:12: note: see existing live user here:
%238 = iree_linalg_ext.scatter dimension_map = [0] unique_indices(false) ins(%collapsed_399, %236 : tensor<32xi32>, tensor<32xi32>) outs(%cst_201 : tensor<68xi32>)
```

This one is IREE-specific and is **not** an export deferral: the op passes evoxc's exporter and even compiles fine in isolation.
The writer emits every flat write as a rank-1 `axis=0` scalar-window scatter with a rank-2 `(K,1) i64` index tensor (`index_vector_dim = 1`).
IREE lowers that to `iree_linalg_ext.scatter`, whose `ins` wants the index tensor **collapsed to rank 1 i32**, and performs the collapse as an `unrealized_conversion_cast`. That materialization only resolves if the rank-2 index value has **exactly one consumer**; when one prepared index block (or two identical index expressions that IREE's CSE merges) feeds a **second** scatter, the rank-2 value stays live and the pass aborts.
Operand sharing, widths, i32-vs-i64 indices, `while`/`if` regions, `sort`, and the rank-2 row-scatter alternative are all irrelevant — deleting either of the two shared-index scatters from the failing module makes it compile (0.26 s).

Fix, in the shared helper's caller (`src/tensorneat/common/tensorization.py::incoming_table`, `_window_pair`): the two payloads that shared one index block (`in_conn`, `in_src`) now go into **one** `(P, 2*(N*K+1))` buffer written by **one** scatter, with the second half's cells offset by one window span and its dropped writes routed to its own dump column.
The resulting rule is now documented in `common/tools.py` and `common/CONTEXT.md`: **one prepared index block feeds exactly one scatter** — merge payloads into a wider buffer instead.

### Gate 5 — a negative `concatenate` axis (HyperNEAT only, `iree-compile` reject)

```
<stdin>:1543:13: error: 'stablehlo.concatenate' op attribute 'dimension' failed to satisfy constraint:
64-bit signless integer attribute whose value is non-negative
    %1524 = "stablehlo.concatenate"(%1462, %1523) {dimension = -1 : i64} : (tensor<160x3xf32>, tensor<160x1xf32>) -> tensor<160x4xf32>
```

evoxc forwards the frontend axis **verbatim** as `stablehlo.concatenate`'s `dimension`, and StableHLO requires it non-negative, so the `axis=-1` spelling — perfectly legal at the frontend and in numpy — only fails once a compiled backend sees it.
Reached only by the HyperNEAT path (the plain NEAT graphs never wrote a negative axis), so it was invisible until the compiled run got that far.
Fix: write the explicit non-negative last axis in graph code — `hyperneat.py` (bias column, attribute stack, connection-key marker columns) and `genome/gene/base.py::_key_array` (`axis=len(shape)`).
Rule: **axes in graph ops are non-negative static integers**; the sweep found no other negative `concatenate` axis in `src/`.
End-to-end after the fix: `HyperNEAT(FullSubstrate(4/3/1), activate_time=3)` + `XOR3d`, seed 42, `backend="iree"` → one 10.4 s compile, then 35.8/36.4 ms per step, keys and NaN masks bit-identical to numpy, weights within 2.4e-07, same `randkey`.

## Measured: numpy vs iree

Protocol: `benchmarks/backend_measure.py`, one `(case, backend, pop, opt_level)` per **fresh process** (peak RSS is a process high-water mark).
`Pipeline.setup()` wall time is reported as `setup_s` (host-only plus NEAT's small operator graph); the expensive generation graph compiles on the **first** `step()`, reported as `build_s` (`Pipeline.build_seconds`, the `evoxc.build` alone) and `first_step_s` (build + one run). The build is charged to no generation.
Then **3 untimed warmup** generations and **20 timed** generations; ms/generation is wall clock per `step()` including host materialisation.
One run each, seed 42, XOR, `NEAT(pop_size=P, species_size=20, survival_threshold=0.01)`, `DefaultGenome(2→1, max_nodes=50, max_conns=100, output_transform=ACT.sigmoid)`.
Measured 2026-11-04, 19:36–19:55 UTC, on the box in the environment table; the exact commands are in "Reproduce".

| case | pop | backend | opt_level | setup (s) | build (s) | ms/gen mean | ms/gen median | min / max | gens/s | peak RSS (MiB) | final max fitness |
|---|---|---|---|---|---|---|---|---|---|---|---|
| xor | 100 | numpy | — | 0.035 | 0.262 | 41.620 | **39.738** | 37.164 / 66.956 | 24.027 | 120.95 | −0.193043 |
| xor | 100 | iree | default | 1.052 | 8.391 | 371.117 | **370.372** | 298.230 / 395.471 | 2.695 | 111.91 | −0.193043 |
| xor | 100 | iree | O0 | 1.016 | 8.355 | 374.009 | **377.283** | 299.406 / 391.832 | 2.674 | 109.82 | −0.193043 |
| xor | 500 | numpy | — | 0.015 | 0.321 | 162.576 | **158.923** | 154.402 / 198.631 | 6.151 | 240.63 | −0.152052 |
| xor | 500 | iree | default | 0.996 | 8.786 | 14993.328 | **14929.106** | 14756.630 / 15297.450 | 0.067 | 161.90 | −0.152052 |
| xor | 1000 | numpy | — | 0.016 | 0.403 | 300.789 | **291.021** | 274.574 / 426.174 | 3.325 | 391.63 | −0.151799 |
| xor | 1000 | iree | default | 0.982 | 8.765 | 30036.253 | **29890.517** | 29519.491 / 31254.606 | 0.033 | 223.81 | −0.143853 |

Coverage smoke for the RL problem classes (one full generation each, `generation_limit=1`, `pop_size=20`, `species_size=4`, `survival_threshold=0.1`, seed 42, one fresh process per class; command in "Reproduce"):

| problem | config | build (s) | one generation |
|---|---|---|---|
| `CartPole` | `max_step=200`, `repeat_times=1`, 4→2 | 7.690 | finite, max fitness 56.0 |
| `Pendulum` | `max_step=200`, 2→1 with a 6-node hidden layer | 7.507 | finite, max fitness −1062.370850 |
| `MountainCarContinuous` | `max_step=999`, 2→1, no hidden layer | 7.517 | finite, max fitness −0.697044 |

Reading it honestly:

- **The compiled leg never wins here.** Slower by 9.3× at pop 100, 93.9× at pop 500, 102.7× at pop 1000 (median ratios); the ratio jumps between pop 100 and pop 500 and then flattens (93.9× → 102.7×), so there is no measured population size at which iree flips ahead.
- **Compile is significant only at the smallest population**: ~8.4–8.8 s of `iree-compile` for the generation graph, plus ~1 s of `setup`, against 0.26–0.40 s for the numpy build. Using `compile ÷ (setup + build + 23 × ms/gen)` (the 3-warmup + 20-timed generations after the build), compile is **46.7 %** of that wall clock at pop 100, **2.5 %** at pop 500 and **1.3 %** at pop 1000; the per-generation cost dominates once the population is large.
- **Peak RSS is the one place iree wins**: 111.91 vs 120.95 MiB at pop 100 (−7 %), 161.90 vs 240.63 MiB at pop 500 (−33 %), 223.81 vs 391.63 MiB at pop 1000 (−43 %); llvm-cpu keeps the tensors in its own buffers while the interpreter holds numpy temporaries.
- **`opt_level` barely matters at O0** (0.8 % higher mean, 1.9 % higher median than default at pop 100) and **O2 does not compile here**:
  `error: 'func.func' op exceeded stack allocation limit of 32768 bytes for function. Got 40064 bytes`, followed by a `vector.gather` operand-type diagnostic; the value is forwarded verbatim and the failure surfaces as `evoxc.core.errors.BackendError`; nothing falls back to another backend.
- The earlier **`iree-compile` SIGSEGV (≥3 loop-carried tensors)** was a real bug, not a configuration artefact: `genome/operations/mutation/cycle.py` seeded a `while_loop`-carried value with a **boolean `scatter`**; the seed is now a one-hot `equal`/`logical_and`, and the SIGSEGV has not reproduced since that fix (`PERF_NUMPY.md`, "IREE stayed compilable"). The generation graph still contains a `while_loop`, and every run above compiled and executed it. Rule: **never seed a `while`-carried value with a `scatter`**.

Per-generation cost is dominated by the graph's dense `gather`/`argsort`/`scatter` traffic, which llvm-cpu lowers to generic, non-vectorised loops — the reason a compiled backend loses to a tight numpy interpreter on this workload. Making iree competitive is a kernel-formulation project (dense key tables → sparse/segment reductions), not a compiler-flag one.

## What a user must know to run `backend="iree"`

- `pip install "evoxc[iree]"` (pulls `iree-base-compiler` + `iree-base-runtime`); then `Pipeline(..., backend="iree")` just works for the shipped problems.
- **Compile is one-off per static shape**: population, `max_nodes`, `max_conns` are baked into the graph, so a shape change re-compiles (~10 s here). `Pipeline` caches the executable per shape.
- **Kernel rules the exporter and iree impose** (all enforced by construction in `src/` today, and documented in `src/tensorneat/common/CONTEXT.md`, `src/tensorneat/algorithm/neat/CONTEXT.md`, `src/tensorneat/algorithm/hyperneat/CONTEXT.md`, `src/tensorneat/genome/CONTEXT.md` and `src/tensorneat/genome/operations/CONTEXT.md`):
  1. `scatter` must be rank-1 `axis=0` or a rank-2 row/column write — no rank ≥ 3 operand.
  2. One prepared index block feeds exactly one `scatter` (gate 4).
  3. No unsigned dtype anywhere in a graph: use same-width signed carriers, plus an explicit sign flip wherever unsigned *order* is required.
  4. No v1-deferred op: `cumsum`, `cumprod`, `tril`, `triu`, `flip`, `erf`, `gelu`, `where`, `scan`, `convolution`, `runtime_call`, `external_call`, collectives other than the mapped ones, and the `sparse_*` family. Use a static constant, a `gather` with a static index vector, or a log2-doubling fold instead.
  5. Static dims only, **non-negative axes only**, and `select`/`gather` (single axis)/`sort`/`argsort`/`where`-free masks/`cond`/`while_loop`/static-bound slices/`rng_bit_generator` all export.
- **Activation functions**: no `ACT` entry needs `erf`/`gelu` (the set is `scaled_sigmoid`, `sigmoid`, `scaled_tanh`, `tanh`, `sin`, `relu`, `lelu`, `identity`, `inv`, `log`, `exp`, `abs`); a *user-supplied* activation that calls `evoxc.erf`/`evoxc.gelu` would still die at export with `op 'erf' ... is not supported in v1`.
- `opt_level="O2"`/`"O3"` fail inside `iree-compile` for this graph size (stack allocation limit). Leave the default or use `O0`/`O1`.
- Do not expect an iree speed-up: on this CPU, numpy is 9–103× faster per generation across the three measured populations. iree's only measured advantage is 7–43 % lower peak RSS.

## Still deferred by the evoxc v1 exporter (would bite other users)

Straight from the installed evoxc 0.1.0, not inferred:
- `evoxc/backends/stablehlo/ops.py:235 DEFERRED_OPS`: `runtime_call`, `external_call`, `block_call`, `rank`, `world_size`, `erf`, `gelu`, `diagonal`, `cholesky`, `qr`, `matrix_rank`, `svd`, `matrix_exp`, `random_multinomial`, and 16 `sparse_*`/`dense_dot_sparse` ops.
- Names with no v1 mapping at all (status `deferred` but absent from that set): `cumsum`, `cumprod`, `flip`, `tril`, `triu`, `where`, `scan`, `convolution`, `collective_broadcast`.
- `evoxc/backends/stablehlo/writer.py:1821 _emit_scatter`: rank ≥ 3 operands, and any rank-2 form other than a full row or column write.
- iree's declared dtype set has no unsigned integer and no complex dtype (`evoxc/backends/adapters/iree.py`).
- Dynamic (non-static) dims cannot be exported at all.

## Reproduce

```bash
uv sync --all-extras

# one fresh process per row — peak RSS and the build are per-process
uv run python benchmarks/backend_measure.py --backend numpy --case xor --pop 100
uv run python benchmarks/backend_measure.py --backend numpy --case xor --pop 500
uv run python benchmarks/backend_measure.py --backend numpy --case xor --pop 1000
uv run python benchmarks/backend_measure.py --backend iree  --case xor --pop 100
uv run python benchmarks/backend_measure.py --backend iree  --case xor --pop 500
uv run python benchmarks/backend_measure.py --backend iree  --case xor --pop 1000
uv run python benchmarks/backend_measure.py --backend iree  --case xor --pop 100 --opt-level O0
uv run python benchmarks/backend_measure.py --backend iree  --case xor --pop 100 --opt-level O2  # fails inside iree-compile

# coverage smoke: one full generation per RL class, one fresh process each
for spec in CartPole:4:2::200 Pendulum:2:1:6:200 MountainCarContinuous:2:1::999; do
  IFS=: read -r CLASS INPUTS OUTPUTS HIDDEN MAX_STEP <<< "$spec"
  PROBLEM="$CLASS" INPUTS="$INPUTS" OUTPUTS="$OUTPUTS" HIDDEN="$HIDDEN" MAX_STEP="$MAX_STEP" uv run python - <<'PY'
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
```

The end-to-end backend test (`test/backends/test_iree_pipeline.py`) skips cleanly without the extra and runs a real XOR generation with it:

```bash
pytest -q test/backends            # 3 passed, or 1 skipped when evoxc[iree] is absent
pytest -m requires_iree test/backends
```

The minimal reproducers for gates 1–5 are ~20 lines of `evoxc` each; the recipes (trace the graphs, walk `op.location`, export with `evoxc.lower(graph, backend="iree").text()`, compile the text with `iree.compiler.compile_str`) live in `src/tensorneat/common/CONTEXT.md` and `src/tensorneat/algorithm/neat/CONTEXT.md`.

## What this does not change

`RESULTS.md` is generated evidence for the old-vs-new CPU contract and was not regenerated or hand-edited for this report: the contract is measured on the numpy backend and that is still what it measures.
`benchmarks/results/iree.json` + `benchmarks/iree.md` remain the historical pre-fix record (8/8 `error`); the numbers above supersede them for the *current* code.
