<h1 align="center">
  <a href="https://github.com/EMI-Group/evox">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="./imgs/evox_brand_light.svg">
    <source media="(prefers-color-scheme: light)" srcset="./imgs/evox_brand_dark.svg">
      <img alt="EvoX Logo" height="50" src="./imgs/evox_brand_light.svg">
  </picture>
  </a>
  <br>
</h1>

<p align="center">
  <b>TensorNEAT 0.2.0</b> — NEAT / HyperNEAT neuroevolution on the
  <a href="https://github.com/EMI-Group/evoxc">EvoX Compiler</a> (<code>evoxc</code>). No JAX, no PyTorch.
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2404.01817">
    <img src="https://img.shields.io/badge/paper-arxiv-red?style=for-the-badge" alt="TensorNEAT Paper on arXiv">
  </a>
  <img src="https://img.shields.io/badge/python-%E2%89%A53.11-blue" alt="Python >= 3.11">
  <img src="https://img.shields.io/badge/backends-numpy%20%7C%20iree-4c9a72" alt="Backends: numpy reference interpreter, iree">
  <img src="https://img.shields.io/badge/license-GPL--3.0-555566" alt="License: GPL-3.0">
</p>

## TensorNEAT @ GECCO 2024

TensorNEAT received the **[GECCO 2024 Best Paper Award](https://gecco-2024.sigevo.org/Best-Paper-Awards#GECH_NE_Track)** (NE track).
Many thanks to everyone who has been supporting TensorNEAT.

---

## What changed in 0.2.0

The computation layer is the **EvoX Compiler** (`evoxc`), not JAX.

| | 0.1.x | 0.2.0 (this tree) |
|---|---|---|
| Compute layer | `jax.jit` + `jax.vmap` | `evoxc` graphs |
| Execution | XLA, traced per call | `@evoxc.defn` graph built **once**, then `exe(...)` per generation on a switchable backend |
| Default backend | JAX (CPU/GPU) | `"numpy"` reference interpreter (always available); `"iree"`, `"xla"`, `"tvm"` opt-in |
| Runtime deps | `jax`, `brax`, `gymnax`, `jaxopt`, `optax`, `flax`, `mujoco`, `mujoco-mjx` | `evoxc`, `numpy`, `networkx`, `matplotlib`, `sympy` |
| `import tensorneat` | pulls in JAX | pulls in **no** `jax` / `torch` / `brax` / `gymnax` / `mujoco` (release-gate test) |

The NEAT idea is unchanged. Irregular network topologies are **tensorized** into fixed-shape arrays so a whole population of differently-shaped networks is evaluated at once, with the population on axis 0 of every tensor. What changed is the engine: a graph is traced with `@evoxc.defn`, handed to `evoxc.build(backend=..., **options)`, and the resulting executable is reused for every generation of a given shape, instead of being `jax.jit`/`vmap`-traced.

Everything else is preserved from the pre-rewrite library and pinned by `test/parity/` against the frozen JAX implementation in `benchmarks/legacy_jax`: the gene encoding, the NaN padding semantics (an unconnected node value is `NaN`, a `NaN` fitness becomes `-inf`), speciation and distance kernels, and deterministic per-seed runs (`(seed, config, shape)` reproduces a trajectory, because all randomness is `evoxc.random` stateless keys).

## Quickstart (60 seconds)

```bash
git clone https://github.com/EMI-Group/tensorneat.git
cd tensorneat
uv sync                              # resolves evoxc from the v0.1.1 git tag, see Install
uv run python examples/func_fit/xor.py
```

`uv run` creates `.venv/`, installs the package in editable mode and runs the script. If `uv` is not on your `PATH`, it is usually at `~/.local/bin/uv`.

Real output from one run of this checkout (first generation, one middle generation, and the report tail — the run prints one such block per generation; the compile time and the per-generation `Cost time` move by a few tens of ms between runs, the fitness and topology numbers do not):

```
compile finished, cost time: 0.285115s
Generation: 1, Cost time: 41.80ms
 	fitness: valid cnt: 200, max: -0.2503, min: -0.4789, mean: -0.3197, std: 0.0536
	node counts: max: 4, min: 3, mean: 3.06
 	conn counts: max: 4, min: 1, mean: 2.02
 	species: 1, [200]
...
Generation: 40, Cost time: 52.34ms
 	fitness: valid cnt: 200, max: -0.0434, min: -0.5115, mean: -0.2336, std: 0.1138
	node counts: max: 7, min: 3, mean: 5.18
 	conn counts: max: 10, min: 1, mean: 6.95
 	species: 8, [53, 38, 29, 26, 22, 17, 11, 4]
...
Generation: 67, Cost time: 47.06ms
 	fitness: valid cnt: 200, max: -0.0009, min: -0.5536, mean: -0.2148, std: 0.1719
	node counts: max: 7, min: 3, mean: 5.12
 	conn counts: max: 12, min: 2, mean: 7.80
 	species: 7, [56, 41, 35, 28, 20, 14, 6]
Fitness limit reached!
input: [0. 0.], target: [0.], predict: [0.02215577]
input: [0. 1.], target: [1.], predict: [0.9849813]
input: [1. 0.], target: [1.], predict: [0.9658409]
input: [1. 1.], target: [0.], predict: [0.03987563]
loss: 0.0008683378109708428
final loss: 8.68e-04 after 67 generations
```

That is the whole quickstart: seed 42 converges at generation 67 in 4.2-4.6 s on this box (two fresh runs), so the demo budget is pop 200 with at most 150 generations and `fitness_target=-1e-3` (`fitness_target` is in fitness space and fitness is maximised, so "error below 1e-3" is `-1e-3`).
The script itself is `examples/func_fit/xor.py`, 41 lines including comments; a real run raises `pop_size` into the thousands and the generation limit into the hundreds.`compile finished` is the one-off evoxc build of the generation graph: `Pipeline.setup()` only prepares host state, the build happens inside the first `step` (that is, inside `auto_run`), and the resulting executable is then reused for every generation of that shape.

After evolution the best individual is available host-side, with no graph involved (`pipeline` is the one built by `examples/func_fit/xor.py`):

```python
state = pipeline.setup()                 # host-only; no graph is built yet
state, best = pipeline.auto_run(state)   # builds the graph on the first step, then evolves
pipeline.show(state, best)               # prints the best network's I/O behaviour

# `best` is a ONE-INDIVIDUAL BATCH; network_dict takes it as a single argument.
network = pipeline.algorithm.genome.network_dict(state, best)
print(sorted(network))
print(len(network["nodes"]), "nodes,", len(network["conns"]), "connections")
```

```
['conns', 'nodes', 'topo_layers', 'topo_order', 'useful_nodes']
4 nodes, 5 connections
```

`network_dict(state, payload)`, `repr(state, payload)`, `sympy_func(state, network, ...)` and `visualize(network, ...)` are **host-only methods on the genome instance** (`pipeline.algorithm.genome`, a `DefaultGenome` here — `tensorneat.genome` has no module-level `visualize`/`sympy_func`): they read one individual's payload and never feed back into a graph. `tensorneat.__version__` is `"0.2.0"` in an installed checkout (`"0.0.0"` outside one).

## Architecture

One generation is **one evoxc graph**: `split randkey -> ask -> transform -> problem.evaluate -> tell`.

```
genome / algorithm / problem kernels        src/tensorneat/genome, algorithm, problem
        │  all tensor ops are written as batch-first graph code
        ▼
  @evoxc.defn  (or a plain closure)         traced once, with host config baked in as constants
        │
        ▼
  evoxc.build(fn, *TensorSpec, backend=..., **options)      ← the compile site
        │     backend="numpy"   reference interpreter, always available
        │     backend="iree"    StableHLO export + iree-compile (llvm-cpu)
        │     backend="xla"     StableHLO export + PJRT plugin (needs EVOXC_PJRT_PLUGIN)
        │     backend="tvm"     opt-in extra
        ▼
  evoxc.Executable  cached in Pipeline._executables[(pop_size, max_nodes, max_conns,
                                                       max_in_degree, max_depth, key_span)]
        │
        ▼
  exe(randkey, node_keys, node_attrs, conn_keys, conn_attrs, generation, *species_leaves)
        │  once per generation:
        │  Pipeline.step -> evoxc.run(executable, state.randkey, *state.pop,
        │                              state.generation, *evoxc.tree_leaves(state.species))
        │  returns [randkey, generation, *pop, *species, fitness, max_overflow]
        ▼
  host code: logging, checkpointing, early stop   (no eager mode: all numerics are inside the graph)
```

- **Batch-first.** Axis 0 is the population `P` for every kernel; datasets/episodes are tiled over that axis by the problem (`P * samples`, `P * repeats`) and the schedule is repeated to match.
- **Static shapes.** `pop_size`, `max_nodes` and `max_conns` are fixed when the graph is built, so an executable is valid for exactly one shape signature; changing any of them builds a second executable (`Pipeline` caches by signature, see `test/integration/test_shape_rebuild.py`). Shapes are never silently resized.
- **Split gene payload.** One individual is four arrays: `node_keys (P, N) int32`, `node_attrs (P, N, A) float32`, `conn_keys (P, E, K) int32`, `conn_attrs (P, E, W) float32`. Structural identity lives in the int32 keys (so it is exact), parameters live in the float32 attributes (so they are differentiable).
- **Host only orchestrates.** Configuration, RNG keys and the tensors live in the immutable `State` container; construction validates and prints nothing. No numerical work happens on the host — not even in logging (it goes through `tensorneat.common.tools.to_numpy`).
- **No fallback.** An unsupported backend, dtype, shape or op raises the backend's own error (`evoxc.core.errors.BackendError`, `TensorNEATError` for configuration); nothing silently switches to another backend.

### Portability rules every kernel respects

These come from the StableHLO v1 exporter and from what the compiled backends accept; they are enforced by construction in `src/`:

- `scatter` is rank-1 `axis=0` or a rank-2 full row/column write — never a rank >= 3 operand.
- **One prepared index block feeds exactly one `scatter`**; merge payloads into a wider buffer instead.
- No unsigned dtype in an exported generation graph: same-width signed carriers (two's complement keeps arithmetic bit-identical) plus one explicit sign flip (`bitwise_xor(v, INT64_MIN)`) wherever unsigned *order* matters. The only `uint32` a graph produces is the digest `Genome.hash` returns, from a standalone helper no generation graph calls, so it never reaches a compiled artifact.
- No op the StableHLO v1 exporter defers. The gate is `evoxc.backends.stablehlo.ops.status(op.name) != "deferred"`, which covers two disjoint sets: every name in `evoxc`'s `DEFERRED_OPS` (`runtime_call`, `external_call`, `block_call`, `rank`, `world_size`, `erf`, `gelu`, `diagonal`, `cholesky`, `qr`, `matrix_rank`, `svd`, `matrix_exp`, `random_multinomial`, plus the 16 `sparse_*`/`dense_dot_sparse` names, numpy-backend-only in v1) and every op the exporter has no mapping for at all (`call`, `tril`, `triu`, `cumsum`, `cumprod`, `flip`, `roll`, `nan_to_num`, `solve`, ...). Use a static constant, a static-index `gather`, or a log2-doubling fold. Spellings matter, because the deferral is keyed on the graph op name: evoxc spells `where` as `select` and convolution as `conv` (both v1), and `scan`/`while_loop` are frontend loop constructs that trace to a `while` region (v1), so the shipped kernels use them freely — `where`, `convolution` and `scan` are simply not evoxc op names.
- Static dims only, non-negative axes only (a negative `concatenate` axis is legal at the frontend and rejected by `iree-compile`).

The evidence for each rule, including the exact failures that forced it, is in `benchmarks/BACKENDS.md`; the per-module rules live in the `CONTEXT.md` chain under `src/tensorneat/common/`, `src/tensorneat/algorithm/neat/`, `src/tensorneat/algorithm/hyperneat/`, `src/tensorneat/genome/` and `src/tensorneat/genome/operations/`.

## Features that actually exist

Every item below is backed by a file in this tree and, where a test exists, by that test.

- **Batched topology inference** — a whole population of different topologies in one forward pass (`genome/step.py`, `genome/grad.py`; `test/unit/genome/test_forward.py`).
- **Gradient descent through evolved topologies** — `genome.BaseGenome.grad(...)` is a **graph factory** returning an `evoxc.Graph` that maps `(node_keys, node_attrs, conn_keys, conn_attrs, inputs)` to gradients w.r.t. the float attribute tensors; `loss_fn` is scalarised as `sum(loss_fn(outputs))` because `evoxc.grad` needs one scalar output (`src/tensorneat/genome/grad.py`). The differentiated pass is a dense-weight twin of the window forward, because evoxc transposes `gather` with *put* semantics and would keep only one cotangent where a node fans out. The host-side convenience wrapper is `genome.DefaultGenome.grad_from_payload(state, payload, inputs, loss_fn)` (the `BaseGenomeHost` mixin in `src/tensorneat/genome/host.py`): it derives the schedule on the host, builds and compiles the gradient graph and returns `(node_attr_grads, conn_attr_grads)` for one batch-first payload. Verified against central finite differences in `test/unit/genome/test_grad.py` and against the legacy JAX gradients in `test/parity/test_grad.py`; the runnable reference is `examples/interpret_visualize/grad_descent.py`, which differentiates one evolved topology and checks the result against finite differences.
- **HyperNEAT** with substrates — `algorithm.HyperNEAT` (recurrent, `activate_time` tick-unroll) and `HyperNEATFeedForward` over `MLPSubstrate`, plus `FullSubstrate` (`algorithm/hyperneat/`, examples `xor_hyperneat*.py`).
- **Recurrent genomes** — `genome.RecurrentGenome` with a static `activate_time` unroll (`genome/recurrent.py`).
- **Speciation and distance kernels** — compatibility distance with node/connection gene digests, species assignment, stagnation, spawn allocation (`algorithm/neat/species.py`, `genome/operations/distance/`).
- **Visualization** — `pipeline.algorithm.genome.visualize(network, save_path=...)` (a genome instance method) draws the decoded topology to SVG via matplotlib (`examples/interpret_visualize/visualize_genome.py`).
- **Sympy formula + Python code generation** — `pipeline.algorithm.genome.sympy_func(state, network, ...)` (again an instance method, and it needs the `State`) returns an 8-tuple consumed by `to_python_code` / `to_latex_code` from `tensorneat.common.sympy_tools` (`examples/interpret_visualize/genome_sympy.py`). `RecurrentGenome` refuses it by design: a cyclic network has no closed form.
- **Three RL problems, all simulator-free** — `problem.CartPole` (discrete, fixed `max_step` horizon, return as fitness), `problem.Pendulum` (continuous torque, the logit squashed inside the problem as `max_torque * tanh(logit)`, a hidden layer needed to learn quickly) and `problem.MountainCarContinuous` (continuous force at the Gym-faithful 999-step default, one-time goal bonus). Fitness is the return (mean over `repeat_times`) in all three. Two documented caveats: `MountainCarContinuous` has a bang-bang idle optimum — a stationary policy scores ≈ `−0.00` and never reaches the goal, so its demo budget cannot "solve" it — and `Pendulum.show` "runs its own graph of one individual ... and reads both the return and the final observation out of the rollout carry" from `state.randkey`, a different reset key from any training episode, so on seed 42 its printed return (`-595.12`) is a different measurement from the training fitness (`-271.30`), not a contradiction. `examples/rl/cartpole.py` reaches 200/200 steps at pop 200 on CPU.
- **Checkpointing** — `Pipeline(..., is_save=True, save_dir=...)` writes `config.txt`, `log.txt`, `genomes/<generation>.npz` per generation and `genomes/best_genome.npz` (verified: 2 generations → `genomes/1.npz`, `genomes/2.npz`, `genomes/best_genome.npz`). `test/integration/test_checkpoint.py` round-trips it.
- **Deterministic per-seed runs** — all randomness is `evoxc.random` keys created from `seed`; the same `(seed, config, shape)` reproduces a run (`test/integration/test_determinism.py`).
- **Per-genome activation/aggregation menus** — `node_gene=genome.BiasNode(activation_options=[...], aggregation_options=[...])` takes plain callables; the built-in registries are `ACT` (12 options: `scaled_sigmoid`, `sigmoid`, `scaled_tanh`, `tanh`, `sin`, `relu`, `lelu`, `identity`, `inv`, `log`, `exp`, `abs`) and `AGG` (`sum`, `product`, `max`, `min`, `maxabs`, `mean`). There is no global `ACT.add_func` registry any more.
- **Shape-signature rebuild** and **population chunking** — a new `(P, max_nodes, max_conns, ...)` builds a new executable; `NEAT(pop_batch_size=...)` runs crossover/mutation as `evoxc.scan` chunks to cut peak memory.

Not in the library any more (they required JAX): the Brax/Gymnax/mujoco RL environments, and the EvoX 1.x algorithm/monitor adaptor.

## Backends

`backend=` is forwarded verbatim to `evoxc.build`; nothing falls back.

| backend | install | status on this box | compile |
|---|---|---|---|
| `numpy` | always available (default) | everything runs; fastest CPU option measured here | 0.26–0.40 s to build the interpreter graph |
| `iree` (llvm-cpu) | `uv sync --extra iree` | runs end to end for every shipped problem family (recorded in [`benchmarks/BACKENDS.md`](./benchmarks/BACKENDS.md)); `test/backends` → **3 `iree` tests (+10 marker-guard) passed** here | 7.0–10.4 s of `iree-compile` per generation graph, one-off per static shape (8.4–8.8 s on the timed XOR rows, in the session BACKENDS.md records — see its provenance section) |
| `xla` | needs a PJRT C API plugin `.so` via `EVOXC_PJRT_PLUGIN` | not available here (no plugin) | — |
| `tvm` | `uv sync --extra tvm` | not attempted | — |

Getting `iree` to run took five exporter/compiler gates (rank-3 `scatter`, unsigned dtypes, v1-deferred ops, one shared index block `iree-compile` could not legalize, and a negative `concatenate` axis); each failure and the fix that removed it is recorded in [`benchmarks/BACKENDS.md`](./benchmarks/BACKENDS.md).

Measured XOR, `DefaultGenome(2 -> 1, max_nodes=50, max_conns=100, output_transform=ACT.sigmoid)`, `NEAT(pop_size=P, species_size=20, survival_threshold=0.01)`, seed 42, 3 warmup + 20 timed generations, one fresh process per row (`benchmarks/backend_measure.py`, recorded in [`benchmarks/BACKENDS.md`](./benchmarks/BACKENDS.md)):

| case | pop | backend | build (s) | ms/gen (median) | peak RSS (MiB) | final max fitness |
|---|---|---|---|---|---|---|
| xor | 100 | `numpy` | 0.262 | **39.738** | 120.95 | −0.193043 |
| xor | 100 | `iree` | 8.391 | **370.372** | 111.91 | −0.193043 |
| xor | 500 | `numpy` | 0.321 | **158.923** | 240.63 | −0.152052 |
| xor | 500 | `iree` | 8.786 | **14929.106** | 161.90 | −0.152052 |
| xor | 1000 | `numpy` | 0.403 | **291.021** | 391.63 | −0.151799 |
| xor | 1000 | `iree` | 8.765 | **29890.517** | 223.81 | −0.143853 |

Reading it honestly: the compiled leg is **9.3× (pop 100), 93.9× (pop 500) and 102.7× (pop 1000) slower per generation** than the numpy interpreter here. The ratio jumps between pop 100 and pop 500 and then flattens (93.9× → 102.7×) — it does not grow with population, and there is no measured population at which `iree` flips ahead. What `iree` does win is peak memory (7 % lower at pop 100, 33 % at pop 500 and 43 % at pop 1000: llvm-cpu keeps tensors in its own buffers while the interpreter materialises numpy temporaries) and code that is portable to accelerators. Compile is paid once per static shape and is 46.7 % / 2.5 % / 1.3 % of a 3-warmup + 20-timed-generation run at pop 100/500/1000; `opt_level="O0"` is within ~2 % of the default.

Practical notes if you want to run `backend="iree"`: `uv sync --extra iree`, leave `compile_options` at the default or use `O0`/`O1` (`opt_level="O2"` fails inside `iree-compile` — `'func.func' op exceeded stack allocation limit of 32768 bytes ... Got 40064 bytes` — surfaced as a `BackendError`), and expect a recompile on every shape change. `test/backends` skips cleanly without the extra.

## Performance

Honest state: on CPU, with the numpy reference interpreter, **the rewrite is not uniformly faster per generation than the frozen pre-rewrite JAX implementation**. The table is the committed harness record in [`benchmarks/RESULTS.md`](./benchmarks/RESULTS.md), rendered from `benchmarks/results/authoritative.json` at 2026-11-04T19:01Z, git `ebeaf1f`, 3 repeats at seeds 42/43/44, **JAX 0.10.2**, 150 timed generations after 3 warmup, one process per run, on a **shared** 20-core INTEL Xeon Platinum 8568Y+ (other jobs ran concurrently, so the per-repeat spread is the honest uncertainty — `0.59×`–`1.37×` on the pop-100 XOR row; a median alone would hide it).
That record postdates the op-count pass but predates the tiled-evaluate pass, the hoisted RL rollout and the search-quality pass, so every number in it describes the tree at `ebeaf1f`; the post-fix measurements live in [`benchmarks/PERF_NUMPY.md`](./benchmarks/PERF_NUMPY.md) and the post-fix search-quality record in [`benchmarks/results/xor3d_quality_2026-11-04.md`](./benchmarks/results/xor3d_quality_2026-11-04.md).

| case | pop | legacy JAX ms/gen | this library (`numpy`) ms/gen | speed-up (old / new) |
|---|---|---|---|---|
| XOR | 100 | 38.99 | 36.44 | 1.07× (new faster) |
| XOR | 500 | 295.33 | 161.73 | 1.83× (new faster) |
| XOR | 1000 | 807.30 | 279.92 | 2.88× (new faster) |
| XOR3d | 100 | 24.53 | 39.12 | 0.63× (new slower) |
| XOR3d | 500 | 107.59 | 171.16 | 0.63× (new slower) |
| XOR3d | 1000 | 255.41 | 300.17 | 0.85× (new slower) |
| Pagie (`CustomFuncFit`) | 500 | 496.45 | 3156.73 | 0.16× (new slower, pre-fix tree) |

The new side is 1.07×/1.83×/2.88× faster on XOR and 0.63×/0.63×/0.85×/0.16× of the old speed on XOR3d at pop 100/500/1000 and Pagie at pop 500.
The Pagie row is the one the merged tree moved most: the tiled func-fit evaluate brings `custom_pagie_p500` to 833.81 ms/gen and 1361.4 MiB in [`benchmarks/PERF_NUMPY.md`](./benchmarks/PERF_NUMPY.md) pass 2 — 3.8× faster and 3.4× smaller than this record's 3156.73 ms and 4577.1 MiB, and 0.65× of the old 539.2 ms/gen measured in that same session.

Build/compile is the one latency number that went the other way: the evoxc build is 0.30 / 0.53 / 0.80 s for XOR at pop 100/500/1000 against 2.94 / 2.93 / 2.96 s for `jax.jit(...).lower().compile()`. Peak RSS is lower at every XOR population (114.9 vs 676.7 MiB at pop 100, 241.3 vs 720.7 at pop 500, 392.2 vs 705.1 at pop 1000) but 5.4× higher on Pagie (4577.1 vs 853.3 MiB), where the interpreter's per-generation working set grows with `pop_size × max_nodes × max_conns` while the XLA side stays close to its runtime floor.
Both of those are record-only facts: the same post-fix session measures Pagie at 1361.4 MiB, not 4577.1 MiB.

So the stated CPU contract ("at least as fast as the frozen JAX baseline") is **not met** on the record's tree, whose verdict `benchmarks/RESULTS.md` writes as `FAIL`: four timing rows fail (XOR3d at pop 100, 500 and 1000, and Pagie at pop 500) and three quality rows fail (`xor3d_p100` −0.0894 vs −0.2500, `xor3d_p500` −0.1789 vs −0.2500, `custom_pagie_p500` −0.0064 vs −0.0219, the new side below the old). The numpy interpreter beats the frozen JAX side on the whole XOR family and loses on XOR3d and Pagie.
On the merged tree the search-quality pass closes the two XOR3d quality rows and the Pagie row (see the quality record), while `xor3d_p100` still misses, so the contract is still not met — the generated record is not the evidence for that either way. Where the rewrite does win:

- **No JAX, no torch, no CUDA toolchain.** `import tensorneat` pulls in neither, enforced by `test/release_gate_no_jax.py`; the runtime dependency set is `evoxc`, `numpy`, `networkx`, `matplotlib`, `sympy`.
- **The graph is data.** It is inspectable, cacheable and transformable (`evoxc.Graph`, `evoxc.lower(...)`), instead of being an opaque `jax.jit` dispatch; one executable per static shape is built once and reused.
- **Latency to first generation.** Build 0.30 s at pop 100 vs 2.94 s of XLA compilation, and the same order for the XOR quickstart graph above, whose one quoted run printed `compile finished, cost time: 0.285115s` (build time is 0.26-0.40 s across runs and populations; it is a one-off per static shape).
- **Switchable backends.** The same kernels run on the IREE compiler today (verified end to end), which is where a GPU or a large-batch workload would pay off — that win is **not measured here**; on this CPU `iree` is 9–103× slower (see Backends).
- **Memory at small scale and on the compiled leg.** Peak RSS is 1.8–5.9× lower than the frozen JAX side on the three XOR rows (114.9/241.3/392.2 vs 676.7/720.7/705.1 MiB) and 5.4× higher on Pagie; on the compiled leg `iree` uses 7–43 % less than numpy (see Backends).

The measured bottleneck on the numpy backend is `tell` (speciation), not the forward pass, and the figures come from the two hand-measured passes recorded in [`benchmarks/PERF_NUMPY.md`](./benchmarks/PERF_NUMPY.md). Profiling 20 generations at pop 500 *before* the first pass put `evoxc`'s interpreter per-op overhead (operand resolution, dtype/shape validation, `Tensor` construction, ~13 µs/op) at ~63 % of a generation against ~37 % of numpy kernel bodies and 0.1 % of host-side Python, and the two `speciate` scans at 9 160 of the 10 767 executed ops per generation (85 %). The op-count pass (`a697531` → tree `f619f46`) brought a pop-500 generation to 8 336 executed ops (−23 %; 8 340 at pop 1000), and at the post-tiling tree `90d97b4` a generation executes 8 263 ops for `xor_p500`, 11 490 for `custom_pagie_p500` and 31 604 for `cartpole_p100` (op count is a leading indicator of wall time, never a proxy: pass 1's `agg` fold *added* 16 ops and made `forward` 26 % faster). No committed artifact records the overhead or speciation share after either pass, and none measures the merged tree.

## Install

Requires **Python >= 3.11** (`.python-version` is 3.12).

### uv (recommended — the only path that resolves `evoxc >= 0.1.1` automatically)

```bash
git clone https://github.com/EMI-Group/tensorneat.git
cd tensorneat
uv sync                     # runtime deps + the default `dev` group (pytest); --no-dev skips it
uv run pytest               # or: uv run python examples/func_fit/xor.py
```

Extras:

| extra | installs | for |
|---|---|---|
| `bench` | `jax >= 0.4.28`, `pytest` | the frozen legacy JAX implementation: `benchmarks/` baseline and the `test/parity/` oracle |
| `iree` | `evoxc[iree]` (`iree-base-compiler`, `iree-base-runtime`) | `backend="iree"`, `test/backends` |
| `tvm` | `evoxc[tvm]` | `backend="tvm"` (untested here) |
| `compilers` | `evoxc[compilers]` | every compiled backend evoxc supports |
| `docs` | `sphinx`, `sphinx-rtd-theme` | building `docs/` |
| `test` | `pytest >= 8.0` | the test suite without the JAX oracle |

e.g. `uv sync --extra bench --extra iree`.

`evoxc` is resolved from the **git tag `v0.1.1`**, not from PyPI, by `[tool.uv.sources]` in `pyproject.toml`:

```toml
[tool.uv.sources]
evoxc = { git = "https://github.com/EMI-Group/evoxc", tag = "v0.1.1" }
```

The reason is written next to the dependency in `pyproject.toml`: PyPI only ships the 0.1.0 wheel, which predates the v0.1.1 frontend work (linear `ir.verify`, two redundant verify passes removed from `lower()`, batched constant-payload emission, frame-walk source-location capture, an xla deadlock fix). Note that the tag's own package metadata still says `version = "0.1.0"`, so `evoxc.__version__` prints `0.1.0` even on the correct checkout.

### pip (verified, but with two caveats)

```bash
# 1. runtime deps, with evoxc from the tag (PyPI evoxc is the slower 0.1.0 wheel)
pip install "evoxc @ git+https://github.com/EMI-Group/evoxc@v0.1.1" \
            numpy networkx matplotlib sympy
# 2. this package, from a checkout (or a git URL)
git clone https://github.com/EMI-Group/tensorneat.git
pip install --no-deps ./tensorneat
```

Caveats, both verified on 2026-11-04:

1. **`tensorneat` has no release on PyPI** (`https://pypi.org/pypi/tensorneat/json` → 404), so `pip install tensorneat` cannot work; install from the git checkout.
2. **`--no-deps` is required.** `pyproject.toml` declares `evoxc >= 0.1.1` and the `>= 0.1.1` floor is deliberate ("the `>= 0.1.1` floor keeps non-uv installers honest: they cannot silently resolve back to the slower 0.1.0 wheel"), but the `v0.1.1` git tag still carries `version = "0.1.0"` metadata while PyPI only has 0.1.0, so pip's resolver refuses:

   ```
   ERROR: Could not find a version that satisfies the requirement evoxc>=0.1.1 (from tensorneat)
   (from versions: 0.1.0)
   ```

   Installing the git-pinned `evoxc` first and then `pip install --no-deps` works: `import tensorneat` reports `0.2.0` and a real XOR generation runs.

## Examples

The ten pipeline scripts use the public API only and print a per-generation log; the three `interpret_visualize` scripts are host-only introspection and import `evoxc` directly for that. Wall times below are from this box, one fresh process per script, `uv run` startup included (0.6–20.6 s end to end).

| script | what it demonstrates | command | verified |
|---|---|---|---|
| `examples/func_fit/xor.py` | 2-input XOR, `DefaultGenome(2 -> 1, max_nodes=12)`, pop 200, 150 gens, target `-1e-3` | `uv run python examples/func_fit/xor.py` | ran, 4.3-4.4 s; **converges** at generation 67, loss 8.68e-04 |
| `examples/func_fit/xor3d.py` | 3-bit parity (`problem.XOR3d`), `max_nodes=16`, pop 400, 150 gens, target `-1e-3` | `uv run python examples/func_fit/xor3d.py` | ran, 20.0-20.6 s; **converges** at generation 144, loss 9.34e-04 |
| `examples/func_fit/xor_origin.py` | `genome.OriginNode`/`OriginConn` (historical markers, whole-gene crossover), pop 300, 250 gens, target `-1e-6` | `uv run python examples/func_fit/xor_origin.py` | ran, 5.8-6.2 s; **converges** at generation 89, loss 8.86e-07 |
| `examples/func_fit/xor_recurrent.py` | `RecurrentGenome(activate_time=10)` on XOR3d, pop 300, 200 gens, target `-1e-4` | `uv run python examples/func_fit/xor_recurrent.py` | ran, 10.6-11.2 s; **converges** at generation 72, loss 1.66e-10 |
| `examples/func_fit/custom_func_fit.py` | Pagie polynomial via `problem.CustomFuncFit`, pop 200, 30 gens | `uv run python examples/func_fit/custom_func_fit.py` | ran, 5.1-5.4 s; demo budget, stops at the generation limit, loss 0.0435 -> 0.0300 |
| `examples/func_fit/xor_hyperneat.py` | recurrent HyperNEAT over `FullSubstrate` (CPPN 4 coords -> 1 weight), pop 300, 40 gens | `uv run python examples/func_fit/xor_hyperneat.py` | ran, 5.7-6.1 s; demo budget, loss 0.25 -> 0.0165 |
| `examples/func_fit/xor_hyperneat_feedforward.py` | `HyperNEATFeedForward` + `MLPSubstrate([4, 6, 6, 1])`, pop 300, 40 gens | `uv run python examples/func_fit/xor_hyperneat_feedforward.py` | ran, 9.4-9.9 s; demo budget, loss 0.2232 -> 0.1876 |
| `examples/rl/cartpole.py` | `problem.CartPole(max_step=200)`, two action logits, no `output_transform`, pop 200, 20 gens | `uv run python examples/rl/cartpole.py` | ran, 1.4-1.6 s; **converges** at generation 2, mean return 200/200 |
| `examples/rl/pendulum.py` | `problem.Pendulum(max_step=200)`, `init_hidden_layers=(6,)`, pop 100, 20 gens, target `-300` | `uv run python examples/rl/pendulum.py` | ran, 2.7-3.0 s; **converges** at generation 5 (training max -271.30; `show` reports one episode at -595.12) |
| `examples/rl/mountaincar.py` | `problem.MountainCarContinuous()`, 999-step default, pop 100, 10 gens, `fitness_target=float("inf")` | `uv run python examples/rl/mountaincar.py` | ran, 11.9-12.5 s; no fitness target (never stops early), demo return -0.00, goal not reached |
| `examples/interpret_visualize/genome_sympy.py` | sympy formula + Python code + LaTeX of one genome (host only) | `uv run python examples/interpret_visualize/genome_sympy.py` | ran, 0.8-1.1 s |
| `examples/interpret_visualize/grad_descent.py` | exact gradients through one evolved topology, checked against finite differences | `uv run python examples/interpret_visualize/grad_descent.py` | ran, 0.62-0.65 s, prints PASS (worst discrepancy 8.33e-06 against the 1e-3 tolerance) |
| `examples/interpret_visualize/visualize_genome.py` | draw one genome as SVG (host only) | `uv run python examples/interpret_visualize/visualize_genome.py` | ran, 0.95-1.9 s (the first run in a session pays a matplotlib font-cache build); writes an untracked, git-ignored `network.svg` next to the script |

The ten pipeline scripts use demo budgets on `backend="numpy"` (pop 100–400, 10–250 generations): the seven `func_fit` scripts mark theirs with a `# Demo budget:` comment, `rl/cartpole.py` marks its reduced `max_step` with `# Small budget:`, and `rl/pendulum.py` / `rl/mountaincar.py` size theirs through `pop_size` and `generation_limit` alone. Six of them reach their fitness target on seed 42 — `xor.py`, `xor3d.py`, `xor_origin.py`, `xor_recurrent.py`, `cartpole.py`, `pendulum.py`; three (`custom_func_fit.py`, the two HyperNEAT scripts) stop at the generation limit with a visibly falling loss and claim no convergence, and `mountaincar.py` sets `fitness_target=float("inf")`, so it has no target to cross. A real run raises `pop_size` into the thousands and the generation limit into the hundreds. The three `interpret_visualize` scripts are host-only (no `Pipeline`, under 2 s). Fitness is maximised everywhere, so a function-fitting `fitness_target` is a negative error bound. The Brax/Gymnax/mujoco examples of 0.1.x no longer exist.
## Tutorials

- `tutorials/quickstart.ipynb` — the current-API notebook, executed top to bottom: build a genome/algorithm/problem, `Pipeline.setup`/`step`/`auto_run`, the batch-first `best` payload, backend *selection* (documented with an `iree` install probe; the notebook itself only ever builds and runs `backend="numpy"`), and inspecting the built `evoxc` executable.
- `tutorials/legacy_jax/` — the three pre-rewrite notebooks (`tutorial-00-functional programming and state.ipynb`, `tutorial-01-genome.ipynb`, `tutorial-02-Usage.ipynb`), kept for concepts only: their code is the old `jax`/`vmap` API and does not run against this tree.

## Project layout

| Path | Purpose |
|---|---|
| `src/tensorneat/pipeline.py` | the host compile site, generation loop, logging and checkpoint writer |
| `src/tensorneat/genome/` | gene payload, genetic operators, forward/grad algebra, host-only introspection |
| `src/tensorneat/algorithm/` | `NEAT` (speciation, spawn, breeding) and `HyperNEAT`/`HyperNEATFeedForward` + substrates |
| `src/tensorneat/problem/` | `XOR`, `XOR3d`, `FuncFit`/`CustomFuncFit`, `CartPole`, `Pendulum`, `MountainCarContinuous` |
| `src/tensorneat/common/` | `State`, error types, tensorization/schedule, ACT/AGG registries, sympy codegen |
| `benchmarks/` | old-vs-new harness, the frozen legacy JAX implementation, `RESULTS.md`, `BACKENDS.md` |
| `test/` | `unit/`, `parity/` (vs legacy JAX), `integration/`, `backends/` (iree), `support/` (shared fixtures), the two release gates |
| `examples/` | the runnable scripts above |
| `docs/` | Sphinx configuration and autodoc stubs |
| `tutorials/` | the quickstart notebook and the JAX-era notebooks |

## Testing

```bash
uv run pytest                          # whole tree
uv run pytest -m "not parity and not slow"   # numpy backend only, no optional dependency
uv run pytest -m parity                # needs: uv sync --extra bench (frozen JAX oracle)
uv run pytest -m slow                  # integration: evolution, determinism, rebuild, checkpoints
uv run pytest -m requires_iree         # needs: uv sync --extra iree
uv run pytest test/release_gate_no_jax.py test/test_release_gate_surface.py
```

Measured on this box with every extra installed (`uv sync --all-extras`):

```
$ uv run pytest
1370 passed, 3 warnings in 86.91s (0:01:26)
```

All 1370 collected tests pass: `test/unit` 1236, `test/parity` 60, `test/integration` 43, `test/backends` 13 (3 `iree` compiled-backend cases plus the 10 marker-guard cases in `test/backends/test_conftest.py`), `test/support` 9, and 9 release gates (4 in `test/release_gate_no_jax.py` + 5 in `test/test_release_gate_surface.py`). The parity tests need the `bench` extra (`jax`) and the backend tests need `--extra iree`; without those extras the parity modules and the `iree` cases skip at import instead of failing, and everything else runs on a bare `uv sync`.

| suite | tests | extra needed | proves |
|---|---|---|---|
| `test/unit/` | 1236 | none | kernels against numpy oracles: schedule derivation, forward/NaN rules, mutation/crossover invariants, speciation (both `GATED_ROWS` paths), distance, parent selection and spawn, `grad` vs finite differences, ACT/AGG tables, the three RL problems against host oracles, HyperNEAT substrates/weights/genes plus a two-generation end-to-end run, version metadata, and the benchmark harness's own protocol/report/fingerprint |
| `test/parity/` | 60 | `--extra bench` | the same kernels on identical payloads against `benchmarks/legacy_jax` (rtol 1e-5 forward, 1e-4 gradients), marker `parity`; `test_oracle_integrity.py` additionally pins that the oracle stays a separate implementation, that its four modules are actually collected, and that the directory still collects 60 tests |
| `test/integration/` | 43 | none | whole-pipeline XOR evolution, per-seed determinism, rebuild on shape change, checkpoint round-trip, `show(state, best)`, the `ask`/`tell` diagnostics hook, chunked evaluate and its guard, `auto_run` stopping, HyperNEAT end to end, RL golden returns, marker `slow` |
| `test/backends/` | 13 | `--extra iree` for 3 of them | `backend="iree"` end to end, its agreement with numpy, and an unknown compile option raising instead of falling back (3 cases, marker `requires_iree`, optional CI `iree` job, not part of the numpy gate), plus the marker-guard suite `test_conftest.py` (10 cases, no extra): `conftest.py`'s `implies_requires_iree` `-m` truth table and the announced `git+https://github.com/EMI-Group/evoxc@v0.1.1` install command |
| `test/support/` | 9 | none | `test_lazy.py` pins the lazy-import helper (`support/lazy.py`): it imports a suite's package when the fixture runs, propagates every error by default (`NotImplementedError` included — a pending implementation is a defect), and skips only an explicit `allow_pending=True` `NotImplementedError` raised inside `src/tensorneat`; the rest of the directory (`runner.py`, `payloads.py`, `oracle.py`, `functions_ref.py`) is imported by the other suites rather than collected |
| `test/release_gate_no_jax.py` + `test/test_release_gate_surface.py` | 4 + 5 | none | import hygiene — `import tensorneat` leaves `jax`, `jaxlib`, `torch`, `brax`, `gymnax`, `mujoco` out of `sys.modules`, even under an import hook that raises for them, and a tiny XOR run still evolves — plus the package surface: every module imports in a fresh subprocess, every `__all__` name resolves, and no `src/` file imports a forbidden framework |

## Citation

If you use TensorNEAT in your research, please cite:

```
@article{10.1145/3730406,
  author = {Wang, Lishuang and Zhao, Mengfei and Liu, Enyu and Sun, Kebin and Cheng, Ran},
  title = {TensorNEAT: A GPU-accelerated Library for NeuroEvolution of Augmenting Topologies},
  year = {2025},
  publisher = {Association for Computing Machinery},
  address = {New York, NY, USA},
  url = {https://doi.org/10.1145/3730406},
  doi = {10.1145/3730406},
  journal = {ACM Trans. Evol. Learn. Optim.},
  month = apr,
  keywords = {Neuroevolution, GPU Acceleration, Algorithm Library}
}
```

## License

TensorNEAT is released under the **GNU General Public License v3.0**; the full text is `LICENSE`.
`pyproject.toml` declares exactly that: `license = { file = "LICENSE" }` plus the classifier `License :: OSI Approved :: GNU General Public License v3 (GPLv3)`, so a built wheel carries the GPL-3.0 text in its `License:` field, `License-File: LICENSE`, and the same classifier.

## Acknowledgements

1. Thanks to Kenneth O. Stanley and Risto Miikkulainen for [the NEAT algorithm](https://ieeexplore.ieee.org/abstract/document/6790655), which has greatly advanced neuroevolution.
2. Thanks to [neat-python](https://github.com/CodeReclaimers/neat-python) and [pureples](https://github.com/ukuleleplayer/pureples) for their clear Python implementations of NEAT and HyperNEAT.
3. Thanks to the [EvoX Compiler](https://github.com/EMI-Group/evoxc) (`evoxc`), the graph compiler and switchable-backend runtime this library's computation layer is built on.
4. Thanks to [EvoX Genesis](https://github.com/EMI-Group/evox) for the framework and the multi-device/distributed direction this library follows.
5. The pre-rewrite JAX implementation is preserved under [`benchmarks/legacy_jax`](./benchmarks/legacy_jax) as the frozen timing baseline and numerical oracle; thanks to the JAX team for the engine that powered 0.1.x and still powers that oracle.
6. TensorNEAT received the [GECCO 2024 Best Paper Award](https://gecco-2024.sigevo.org/Best-Paper-Awards#GECH_NE_Track).

## Community & Support

- Questions, bugs and feature requests: [GitHub Issues](https://github.com/EMI-Group/tensorneat/issues).
- QQ group: 297969717.

## Roadmap

- **Close the CPU gap on the numpy interpreter.** The measured bottleneck is `tell`, dominated by speciation, plus `evoxc`'s interpreter per-op overhead: before the op-count pass the two `speciate` scans were 9 160 of the 10 767 executed ops per generation at pop 500 (85 %) and per-op overhead was ~63 % of a generation; that pass took the same generation to 8 336 executed ops and the post-tiling tree to 8 263 for `xor_p500`, but no committed record re-measures the overhead or speciation share after either pass (see [`benchmarks/PERF_NUMPY.md`](./benchmarks/PERF_NUMPY.md)). The forward pass is not the problem.
- **Make a compiled backend actually competitive.** The IREE leg works but loses on llvm-cpu because the dense key-table `gather`/`argsort`/`scatter` traffic lowers to generic loops; that is a kernel-formulation project (sparse/segment reductions), not a compiler-flag one. No compiled win is claimed until it is measured.
- **Lift the remaining `evoxc` v1 exporter deferrals** upstream, then re-evaluate the compiled backends and re-run the contract.
- **More NEAT-family algorithms and substrates** (e.g. ES-HyperNEAT), and more RL problems that do not need JAX.
- **Docs and tutorials**: keep `docs/` and `tutorials/quickstart.ipynb` in step with the API.

Pull requests are welcome.
