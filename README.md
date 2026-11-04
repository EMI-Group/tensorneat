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

Real output from this checkout (first generation, one middle generation, and the report tail — the run prints one such block per generation):

```
compile finished, cost time: 0.298339s
Generation: 1, Cost time: 43.95ms
 	fitness: valid cnt: 200, max: -0.2503, min: -0.4789, mean: -0.3197, std: 0.0536
	node counts: max: 4, min: 3, mean: 3.06
 	conn counts: max: 3, min: 1, mean: 1.90
 	species: 1, [200]
...
Generation: 40, Cost time: 61.40ms
 	fitness: valid cnt: 192, max: -0.0895, min: -0.4992, mean: -0.2483, std: 0.1126
	node counts: max: 7, min: 3, mean: 4.54
 	conn counts: max: 7, min: 0, mean: 4.58
 	species: 9, [51, 34, 30, 22, 21, 16, 12, 7, 7]
...
Generation: 75, Cost time: 52.27ms
 	fitness: valid cnt: 195, max: -0.0009, min: -0.7297, mean: -0.2144, std: 0.1845
	node counts: max: 7, min: 3, mean: 4.59
 	conn counts: max: 8, min: 1, mean: 5.46
 	species: 7, [56, 41, 35, 27, 21, 13, 7]
Fitness limit reached!
input: [0. 0.], target: [0.], predict: [0.04544412]
input: [0. 1.], target: [1.], predict: [0.9785589]
input: [1. 0.], target: [1.], predict: [0.9711594]
input: [1. 1.], target: [0.], predict: [0.01849632]
loss: 0.0009246956906281412
final loss: 9.25e-04 after 75 generations
```

That is the whole quickstart: seed 42 converges at generation 75 in 4.6 s on this box, so the demo budget is pop 200 with at most 150 generations and `fitness_target=-1e-3` (`fitness_target` is in fitness space and fitness is maximised, so "error below 1e-3" is `-1e-3`).
The script itself is `examples/func_fit/xor.py`, 40 lines including comments; a real run raises `pop_size` into the thousands and the generation limit into the hundreds.
`compile finished` is the one-off evoxc build of the generation graph: `Pipeline.setup()` only prepares host state, the build happens inside the first `step` (that is, inside `auto_run`), and the resulting executable is then reused for every generation of that shape.

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

`network_dict`, `repr`, `sympy_func` and `visualize` are **host-only** converters: they read one individual's payload and never feed back into a graph. `tensorneat.__version__` is `"0.2.0"` in an installed checkout (`"0.0.0"` outside one).

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
  exe(node_keys, node_attrs, conn_keys, conn_attrs, generation, species, randkey)
        │  once per generation inside Pipeline.step
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
- No unsigned dtype inside a graph: same-width signed carriers (two's complement keeps arithmetic bit-identical) plus one explicit sign flip (`bitwise_xor(v, INT64_MIN)`) wherever unsigned *order* matters.
- No v1-deferred op: `cumsum`, `cumprod`, `tril`, `triu`, `flip`, `erf`, `gelu`, `where`, `scan`, `convolution`, `runtime_call`, `external_call`, the `sparse_*` family. Use a static constant, a static-index `gather`, or a log2-doubling fold.
- Static dims only, non-negative axes only (a negative `concatenate` axis is legal at the frontend and rejected by `iree-compile`).

The evidence for each rule, including the exact failures that forced it, is in `benchmarks/BACKENDS.md`; the per-module rules live in the `CONTEXT.md` chain under `src/tensorneat/common/`, `src/tensorneat/algorithm/neat/`, `src/tensorneat/algorithm/hyperneat/`, `src/tensorneat/genome/` and `src/tensorneat/genome/operations/`.

## Features that actually exist

Every item below is backed by a file in this tree and, where a test exists, by that test.

- **Batched topology inference** — a whole population of different topologies in one forward pass (`genome/step.py`, `genome/grad.py`; `test/unit/genome/test_forward.py`).
- **Gradient descent through evolved topologies** — `genome.BaseGenome.grad(...)` is a **graph factory** returning an `evoxc.Graph` that maps `(node_keys, node_attrs, conn_keys, conn_attrs, inputs)` to gradients w.r.t. the float attribute tensors; `loss_fn` is scalarised as `sum(loss_fn(outputs))` because `evoxc.grad` needs one scalar output (`src/tensorneat/genome/grad.py`). The differentiated pass is a dense-weight twin of the window forward, because evoxc transposes `gather` with *put* semantics and would keep only one cotangent where a node fans out. Verified against central finite differences in `test/unit/genome/test_grad.py` and against the legacy JAX gradients in `test/parity/test_grad.py`. **There is no shipped example script for it** — the two test suites are the working reference.
- **HyperNEAT** with substrates — `algorithm.HyperNEAT` (recurrent, `activate_time` tick-unroll) and `HyperNEATFeedForward` over `MLPSubstrate`, plus `FullSubstrate` (`algorithm/hyperneat/`, examples `xor_hyperneat*.py`).
- **Recurrent genomes** — `genome.RecurrentGenome` with a static `activate_time` unroll (`genome/recurrent.py`).
- **Speciation and distance kernels** — compatibility distance with node/connection gene digests, species assignment, stagnation, spawn allocation (`algorithm/neat/species.py`, `genome/operations/distance/`).
- **Visualization** — `genome.visualize(network)` draws the decoded topology to SVG via matplotlib (`examples/interpret_visualize/visualize_genome.py`).
- **Sympy formula + Python code generation** — `genome.sympy_func(...)` returns an 8-tuple consumed by `to_python_code` / `to_latex_code` from `tensorneat.common.sympy_tools` (`examples/interpret_visualize/genome_sympy.py`). `RecurrentGenome` refuses it by design: a cyclic network has no closed form.
- **Three RL problems, all simulator-free** — `problem.CartPole` (discrete, fixed `max_step` horizon, return as fitness), `problem.Pendulum` (continuous torque, the logit squashed inside the problem as `max_torque * tanh(logit)`, a hidden layer needed to learn quickly) and `problem.MountainCarContinuous` (continuous force at the Gym-faithful 999-step default, one-time goal bonus). Fitness is the return (mean over `repeat_times`) in all three. Two documented caveats: `MountainCarContinuous` has a bang-bang idle optimum — a stationary policy scores ≈ `−0.00` and never reaches the goal, so its demo budget cannot "solve" it — and `Pendulum.show` "runs its own graph of one individual ... and reads both the return and the final observation out of the rollout carry" from `state.randkey`, a different reset key from any training episode, so its printed return (`−48.08`) is a different measurement from the training fitness (`−238.84`), not a contradiction. `examples/rl/cartpole.py` reaches 200/200 steps at pop 200 on CPU.
- **Checkpointing** — `Pipeline(..., is_save=True, save_dir=...)` writes `config.txt`, `log.txt`, `genomes/<generation>.npz` per generation and `genomes/best_genome.npz` (verified: 2 generations → `genomes/1.npz`, `genomes/2.npz`, `genomes/best_genome.npz`). `test/integration/test_checkpoint.py` round-trips it.
- **Deterministic per-seed runs** — all randomness is `evoxc.random` keys created from `seed`; the same `(seed, config, shape)` reproduces a run (`test/integration/test_determinism.py`).
- **Per-genome activation/aggregation menus** — `node_gene=genome.BiasNode(activation_options=[...], aggregation_options=[...])` takes plain callables; the built-in registries are `ACT` (12 options: `scaled_sigmoid`, `sigmoid`, `scaled_tanh`, `tanh`, `sin`, `relu`, `lelu`, `identity`, `inv`, `log`, `exp`, `abs`) and `AGG` (`sum`, `product`, `max`, `min`, `maxabs`, `mean`). There is no global `ACT.add_func` registry any more.
- **Shape-signature rebuild** and **population chunking** — a new `(P, max_nodes, max_conns, ...)` builds a new executable; `NEAT(pop_batch_size=...)` runs crossover/mutation as `evoxc.scan` chunks to cut peak memory.

Not in the library any more (they required JAX): the Brax/Gymnax/mujoco RL environments, and the EvoX 1.x algorithm/monitor adaptor.

## Backends

`backend=` is forwarded verbatim to `evoxc.build`; nothing falls back.

| backend | install | status on this box | compile |
|---|---|---|---|
| `numpy` | always available (default) | everything runs; fastest CPU option measured here | 0.26–0.40 s per generation graph (interpreter build) |
| `iree` (llvm-cpu) | `uv sync --extra iree` | runs end to end for every shipped problem family (recorded in [`benchmarks/BACKENDS.md`](./benchmarks/BACKENDS.md)); `test/backends` → **3 passed** here | 7.0–10.4 s of `iree-compile` per generation graph, one-off per static shape (8.4–8.8 s on the timed XOR rows) |
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

| case | pop | legacy JAX ms/gen | this library (`numpy`) ms/gen | speed-up (old / new) |
|---|---|---|---|---|
| XOR | 100 | 38.99 | 36.44 | 1.07× (new faster) |
| XOR | 500 | 295.33 | 161.73 | 1.83× (new faster) |
| XOR | 1000 | 807.30 | 279.92 | 2.88× (new faster) |
| XOR3d | 100 | 24.53 | 39.12 | 0.63× (new slower) |
| XOR3d | 500 | 107.59 | 171.16 | 0.63× (new slower) |
| XOR3d | 1000 | 255.41 | 300.17 | 0.85× (new slower) |
| Pagie (`CustomFuncFit`) | 500 | 496.45 | 3156.73 | 0.16× (new slower) |

The new side is 1.07×/1.83×/2.88× faster on XOR and 0.63×/0.63×/0.85×/0.16× of the old speed on XOR3d at pop 100/500/1000 and Pagie at pop 500.

Build/compile is the one latency number that went the other way: the evoxc build is 0.30 / 0.53 / 0.80 s for XOR at pop 100/500/1000 against 2.94 / 2.93 / 2.96 s for `jax.jit(...).lower().compile()`. Peak RSS is lower at every XOR population (114.9 vs 676.7 MiB at pop 100, 241.3 vs 720.7 at pop 500, 392.2 vs 705.1 at pop 1000) but 5.4× higher on Pagie (4577.1 vs 853.3 MiB), where the interpreter's per-generation working set grows with `pop_size × max_nodes × max_conns` while the XLA side stays close to its runtime floor.

So the stated CPU contract ("at least as fast as the frozen JAX baseline") is **not met today**, and `benchmarks/RESULTS.md` records the verdict as `FAIL`: four timing rows fail (XOR3d at pop 100, 500 and 1000, and Pagie at pop 500) and three quality rows fail (`xor3d_p100` −0.0894 vs −0.2500, `xor3d_p500` −0.1789 vs −0.2500, `custom_pagie_p500` −0.0064 vs −0.0219, the new side below the old). The numpy interpreter beats the frozen JAX side on the whole XOR family and loses on XOR3d and Pagie. Where the rewrite does win:

- **No JAX, no torch, no CUDA toolchain.** `import tensorneat` pulls in neither, enforced by `test/release_gate_no_jax.py`; the runtime dependency set is `evoxc`, `numpy`, `networkx`, `matplotlib`, `sympy`.
- **The graph is data.** It is inspectable, cacheable and transformable (`evoxc.Graph`, `evoxc.lower(...)`), instead of being an opaque `jax.jit` dispatch; one executable per static shape is built once and reused.
- **Latency to first generation.** Build 0.30 s at pop 100 vs 2.94 s of XLA compilation, and 0.30 s for the XOR quickstart graph above (the run prints `compile finished, cost time: 0.298339s`).
- **Switchable backends.** The same kernels run on the IREE compiler today (verified end to end), which is where a GPU or a large-batch workload would pay off — that win is **not measured here**; on this CPU `iree` is 9–103× slower (see Backends).
- **Memory at small scale and on the compiled leg.** Peak RSS is 1.8–5.9× lower than the frozen JAX side on the three XOR rows (114.9/241.3/392.2 vs 676.7/720.7/705.1 MiB) and 5.4× higher on Pagie; on the compiled leg `iree` uses 7–43 % less than numpy (see Backends).

The measured bottleneck on the numpy backend is `tell` (speciation), not the forward pass: at pop 500 two `speciate` scans were 9 160 of 10 767 executed ops per generation (85 %) and ~63 % of a generation is `evoxc` interpreter per-op overhead, both recorded in [`benchmarks/PERF_NUMPY.md`](./benchmarks/PERF_NUMPY.md).

## Install

Requires **Python >= 3.11** (`.python-version` is 3.12).

### uv (recommended — the only path that resolves `evoxc >= 0.1.1` automatically)

```bash
git clone https://github.com/EMI-Group/tensorneat.git
cd tensorneat
uv sync                     # runtime deps only
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

All scripts use the public API only and print a per-generation log. Wall times below are from this box (`uv run` startup included, ~1 s).

| script | what it demonstrates | command | verified |
|---|---|---|---|
| `examples/func_fit/xor.py` | 2-input XOR, `DefaultGenome(2 -> 1, max_nodes=12)`, pop 200, 150 gens, target `-1e-3` | `uv run python examples/func_fit/xor.py` | ran, 4.6 s; **converges** at generation 75, loss 9.25e-04 |
| `examples/func_fit/xor3d.py` | 3-bit parity (`problem.XOR3d`), `max_nodes=16`, pop 400, 150 gens, target `-1e-4` | `uv run python examples/func_fit/xor3d.py` | ran, 12.4 s; **converges** at generation 93, loss 6.88e-05 |
| `examples/func_fit/xor_origin.py` | `genome.OriginNode`/`OriginConn` (historical markers, whole-gene crossover), pop 300, 250 gens, target `-1e-6` | `uv run python examples/func_fit/xor_origin.py` | ran, 8.8 s; **converges** at generation 123, loss 7.48e-07 |
| `examples/func_fit/xor_recurrent.py` | `RecurrentGenome(activate_time=10)` on XOR3d, pop 300, 200 gens, target `-1e-4` | `uv run python examples/func_fit/xor_recurrent.py` | ran, 15.7 s; **converges** at generation 110, loss 1.88e-05 |
| `examples/func_fit/custom_func_fit.py` | Pagie polynomial via `problem.CustomFuncFit`, pop 200, 30 gens | `uv run python examples/func_fit/custom_func_fit.py` | ran, 5.3 s; demo budget, stops at the generation limit, final loss 0.0258 |
| `examples/func_fit/xor_hyperneat.py` | recurrent HyperNEAT over `FullSubstrate` (CPPN 4 coords -> 1 weight), pop 300, 40 gens | `uv run python examples/func_fit/xor_hyperneat.py` | ran, 5.3 s; demo budget, stops at the generation limit, final loss 0.0463 |
| `examples/func_fit/xor_hyperneat_feedforward.py` | `HyperNEATFeedForward` + `MLPSubstrate([4, 6, 6, 1])`, pop 300, 40 gens | `uv run python examples/func_fit/xor_hyperneat_feedforward.py` | ran, 7.5 s; demo budget, stops at the generation limit, final loss 0.1502 |
| `examples/rl/cartpole.py` | `problem.CartPole(max_step=200)`, two action logits, no `output_transform`, pop 200, 20 gens | `uv run python examples/rl/cartpole.py` | ran, 2.1 s; **converges** at generation 2, mean return 200.00 |
| `examples/rl/pendulum.py` | `problem.Pendulum(max_step=200)`, one torque logit squashed inside the problem, a 6-node hidden layer, pop 100, 20 gens | `uv run python examples/rl/pendulum.py` | ran, 5.0 s; **converges** at generation 8 (`show` reports mean return −48.08 on its own reset key) |
| `examples/rl/mountaincar.py` | `problem.MountainCarContinuous(max_step=999)`, one action logit, no hidden layer, pop 100, 10 gens | `uv run python examples/rl/mountaincar.py` | ran, 19.4 s; stops at the generation limit, mean return −0.00 — the idle optimum, `show` reports "goal not reached" |
| `examples/interpret_visualize/genome_sympy.py` | sympy formula + Python code + LaTeX of one genome | `uv run python examples/interpret_visualize/genome_sympy.py` | ran, 0.8 s |
| `examples/interpret_visualize/visualize_genome.py` | draw one genome as SVG | `uv run python examples/interpret_visualize/visualize_genome.py` | ran, 0.9 s; writes an untracked `network.svg` next to the script |

The ten pipeline scripts use demo budgets (pop 100–400, 10–250 generations, numpy backend; the `func_fit` scripts mark them with `# Demo budget:` in the source), and six of them reach their fitness target on seed 42 — `xor.py`, `xor3d.py`, `xor_origin.py`, `xor_recurrent.py`, `cartpole.py`, `pendulum.py`; the other four (`custom_func_fit.py`, the two HyperNEAT scripts, `mountaincar.py`) stop at the generation limit with a visibly falling loss and claim no convergence (`mountaincar.py` lands on the documented idle optimum). A real run raises `pop_size` into the thousands and the generation limit into the hundreds. The two `interpret_visualize` scripts do host-only introspection of one genome payload (no `Pipeline`, no evolution, under 1 s); each uses `evoxc.build`/`evoxc.run` host-side to draw that one individual. Fitness is maximised everywhere, so a function-fitting `fitness_target` is a negative error bound. The Brax/Gymnax/mujoco examples of 0.1.x no longer exist.

## Tutorials

- `tutorials/quickstart.ipynb` — the current-API notebook, executed top to bottom: build a genome/algorithm/problem, `Pipeline.setup`/`step`/`auto_run`, the batch-first `best` payload, switching `backend=`, and inspecting the built `evoxc` executable.
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
| `test/` | `unit/`, `parity/` (vs legacy JAX), `integration/`, `backends/` (iree), release gate |
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
uv run pytest test/release_gate_no_jax.py
```

Measured on this box with every extra installed (`uv sync --all-extras`):

```
$ uv run pytest -o addopts="" -q
659 passed, 3 warnings in 59.88s
```

(`-o addopts=""` is needed because `pyproject.toml` sets `addopts = "-q"` and a doubled `-q` suppresses the summary line.) The counts come apart as `test/unit` 578, `test/parity` 53, `test/integration` 21, `test/backends` 3 and `test/release_gate_no_jax.py` 4. The parity tests need the `bench` extra (`jax`) and the backend tests need `--extra iree`; without those extras the parity modules and `test/backends` skip at import instead of failing, and everything else runs on a bare `uv sync`.

| suite | tests | extra needed | proves |
|---|---|---|---|
| `test/unit/` | 578 | none | kernels against numpy oracles: schedule derivation, forward/NaN rules, mutation/crossover invariants, speciation bookkeeping, distance, `grad` vs finite differences, ACT/AGG tables, CartPole vs a host Euler rollout |
| `test/parity/` | 53 | `--extra bench` | the same kernels on identical payloads against `benchmarks/legacy_jax` (rtol 1e-5 forward, 1e-4 gradients), marker `parity` |
| `test/integration/` | 21 | none | whole-pipeline XOR evolution, per-seed determinism, rebuild on shape change, checkpoint round-trip, `show(state, best)`, marker `slow` |
| `test/backends/` | 3 | `--extra iree` | `backend="iree"` end to end and its agreement with numpy, marker `requires_iree` (in no gate) |
| `test/release_gate_no_jax.py` | 4 | none | `import tensorneat` leaves `jax`, `jaxlib`, `torch`, `brax`, `gymnax`, `mujoco` out of `sys.modules` — even under an import hook that raises for them — and a tiny XOR run still evolves |

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

- **Close the CPU gap on the numpy interpreter.** The measured bottleneck is `tell`, dominated by speciation (two `speciate` scans were 85 % of the executed ops per generation at pop 500, and `evoxc` interpreter per-op overhead is ~63 % of a generation; see [`benchmarks/PERF_NUMPY.md`](./benchmarks/PERF_NUMPY.md)); the forward pass is not the problem.
- **Make a compiled backend actually competitive.** The IREE leg works but loses on llvm-cpu because the dense key-table `gather`/`argsort`/`scatter` traffic lowers to generic loops; that is a kernel-formulation project (sparse/segment reductions), not a compiler-flag one. No compiled win is claimed until it is measured.
- **Lift the remaining `evoxc` v1 exporter deferrals** upstream, then re-evaluate the compiled backends and re-run the contract.
- **More NEAT-family algorithms and substrates** (e.g. ES-HyperNEAT), and more RL problems that do not need JAX.
- **Docs and tutorials**: keep `docs/` and `tutorials/quickstart.ipynb` in step with the API.

Pull requests are welcome.
