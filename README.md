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

Real output from this checkout (first generation, last generation and the report tail):

```
compile finished, cost time: 0.319147s
Generation: 1, Cost time: 73.17ms
 	fitness: valid cnt: 500, max: -0.2503, min: -0.4789, mean: -0.3212, std: 0.0514
	node counts: max: 4, min: 3, mean: 3.09
 	conn counts: max: 3, min: 1, mean: 1.96
 	species: 1, [500]
...
Generation: 40, Cost time: 76.80ms
 	fitness: valid cnt: 454, max: -0.2500, min: -0.3528, mean: -0.2557, std: 0.0135
	node counts: max: 6, min: 3, mean: 3.71
 	conn counts: max: 4, min: 0, mean: 2.02
 	species: 5, [204, 58, 115, 71, 52]
Generation limit reached!
input: [0. 0.], target: [0.], predict: [0.50017095]
input: [0. 1.], target: [1.], predict: [0.50017095]
input: [1. 0.], target: [1.], predict: [0.5000479]
input: [1. 1.], target: [0.], predict: [0.5000479]
loss: 0.25
```

That script is a deliberately tiny demo budget (pop 500, 40 generations, target error `< 1e-3`); on this box seed 42 does **not** cross the target inside 40 generations, so it prints `Generation limit reached!` — raise `generation_limit` or `pop_size` for a real run.

A complete, converging run (this exact script was executed; ~5 s on CPU):

```python
from tensorneat import Pipeline, algorithm, genome, problem
from tensorneat.common import ACT

pipeline = Pipeline(
    algorithm=algorithm.NEAT(
        pop_size=200,
        species_size=10,
        survival_threshold=0.01,
        genome=genome.DefaultGenome(
            num_inputs=2,
            num_outputs=1,
            max_nodes=12,
            output_transform=ACT.sigmoid,
        ),
    ),
    problem=problem.XOR(),
    seed=42,
    fitness_target=-1e-3,   # fitness is maximised: "error below 1e-3"
    generation_limit=200,
)
state = pipeline.setup()                 # host-only; no graph is built yet
state, best = pipeline.auto_run(state)   # builds the graph once, then evolves
pipeline.show(state, best)               # prints the best network's behaviour

# `best` is a ONE-INDIVIDUAL BATCH; network_dict takes it as a single argument.
network = pipeline.algorithm.genome.network_dict(state, best)
print(sorted(network))
print(len(network["nodes"]), "nodes,", len(network["conns"]), "connections")
```

```
Generation: 75, Cost time: 50.00ms
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
- **CartPole RL** — `problem.CartPole` is the only RL problem: a batched, evoxc-native Gym CartPole rollout (fixed horizon, return as fitness, `max_step` is the ceiling). `examples/rl/cartpole.py` reaches 200/200 steps in 2 generations at pop 200 on CPU.
- **Checkpointing** — `Pipeline(..., is_save=True, save_dir=...)` writes `config.txt`, `log.txt`, `genomes/<generation>.npz` per generation and `genomes/best_genome.npz` (verified: 2 generations → `genomes/1.npz`, `genomes/2.npz`, `genomes/best_genome.npz`). `test/integration/test_checkpoint.py` round-trips it.
- **Deterministic per-seed runs** — all randomness is `evoxc.random` keys created from `seed`; the same `(seed, config, shape)` reproduces a run (`test/integration/test_determinism.py`).
- **Per-genome activation/aggregation menus** — `node_gene=genome.BiasNode(activation_options=[...], aggregation_options=[...])` takes plain callables; the built-in registries are `ACT` (12 options: `scaled_sigmoid`, `sigmoid`, `scaled_tanh`, `tanh`, `sin`, `relu`, `lelu`, `identity`, `inv`, `log`, `exp`, `abs`) and `AGG` (`sum`, `product`, `max`, `min`, `maxabs`, `mean`). There is no global `ACT.add_func` registry any more.
- **Shape-signature rebuild** and **population chunking** — a new `(P, max_nodes, max_conns, ...)` builds a new executable; `NEAT(pop_batch_size=...)` runs crossover/mutation as `evoxc.scan` chunks to cut peak memory.

Not in the library any more (they required JAX): the Brax/Gymnax/mujoco RL environments, and the EvoX 1.x algorithm/monitor adaptor.

## Backends

`backend=` is forwarded verbatim to `evoxc.build`; nothing falls back.

| backend | install | status on this box | compile |
|---|---|---|---|
| `numpy` | always available (default) | everything runs; fastest CPU option measured here | 1.6–2.0 s per graph set (interpreter build) |
| `iree` (llvm-cpu) | `uv sync --extra iree` | runs end to end for every shipped problem family (recorded in [`benchmarks/BACKENDS.md`](./benchmarks/BACKENDS.md)); `test/backends` → **3 passed** here | 8–11 s of `iree-compile` per generation graph, one-off per static shape |
| `xla` | needs a PJRT C API plugin `.so` via `EVOXC_PJRT_PLUGIN` | not available here (no plugin) | — |
| `tvm` | `uv sync --extra tvm` | not attempted | — |

Measured XOR, `DefaultGenome(2 -> 1, max_nodes=12, max_conns=24)`, `species_size=20`, seed 42, 3 warmup + 20 timed generations, one fresh process per row (`benchmarks/backend_measure.py`, recorded in [`benchmarks/BACKENDS.md`](./benchmarks/BACKENDS.md)):

| case | pop | backend | build (s) | ms/gen (median) | peak RSS (MiB) | final max fitness |
|---|---|---|---|---|---|---|
| xor | 100 | `numpy` | 1.60 | **42.74** | 120.9 | −0.193043 |
| xor | 100 | `iree` | 9.35 | **213.12** | 108.6 | −0.193043 |
| xor | 500 | `numpy` | 1.62 | **262.80** | 263.6 | −0.152052 |
| xor | 500 | `iree` | 9.97 | **6366.20** | 153.0 | −0.152052 |
| xor | 1000 | `numpy` | 1.70 | **546.70** | 433.9 | −0.151799 |
| xor | 1000 | `iree` | 10.86 | **12765.81** | 210.2 | −0.143853 |

Re-run for this README on the same box, pop 100: `numpy` 43.42 ms/gen, `iree` 213.77 ms/gen after a 7.67 s build, peak RSS 108.6 vs 118.2 MiB, and the identical final max fitness `−0.193043` on both backends.

Reading it honestly: the compiled leg is **5.0× (pop 100), 24.2× (pop 500) and 23.4× (pop 1000) slower per generation** than the numpy interpreter here, and the gap grows with population size. What `iree` does win is peak memory (30–50% lower: llvm-cpu keeps tensors in its own buffers while the interpreter materialises numpy temporaries) and code that is portable to accelerators. Compile cost must be paid once per static shape, and at 20-generation budgets it is 35–50% of the wall clock.

Practical notes if you want to run `backend="iree"`: `uv sync --extra iree`, leave `compile_options` at the default (`opt_level="O2"`/`"O3"` exceed IREE's per-function stack-promotion limit for this graph and fail inside `iree-compile` with a `BackendError`), and expect a recompile on every shape change. `test/backends` skips cleanly without the extra.

## Performance

Honest state: on CPU, with the numpy reference interpreter, **this rewrite is slower per generation than the frozen pre-rewrite JAX implementation**. The table is the committed harness record in [`benchmarks/RESULTS.md`](./benchmarks/RESULTS.md) (150 timed generations, 3 repeats at seeds 42/43/44, one process per run, measured on a shared 20-core INTEL Xeon Platinum 8568Y+ box, Python 3.12, JAX 0.12.0.1, git `89f6200d6bba`; per-repeat spread and protocol in that file).

| case | pop | legacy JAX ms/gen | this library (`numpy`) ms/gen | JAX is |
|---|---|---|---|---|
| XOR | 100 | 31.35 | 58.02 | 1.85× faster |
| XOR | 500 | 157.01 | 357.25 | 2.28× faster |
| XOR | 1000 | 246.76 | 671.73 | 2.72× faster |
| XOR3d | 100 | 19.59 | 69.48 | 3.55× faster |
| XOR3d | 500 | 81.42 | 380.42 | 4.67× faster |
| XOR3d | 1000 | 141.18 | 698.15 | 4.94× faster |
| Pagie (`CustomFuncFit`) | 500 | 371.83 | 3443.44 | 9.26× faster |

Build/compile is the one latency number that went the other way: the evoxc build is 0.56 / 1.52 / 2.19 s for XOR at pop 100/500/1000 against 2.74 / 2.82 / 2.76 s for `jax.jit(...).lower().compile()`. Peak RSS is lower at small populations (145 vs 587 MiB at pop 100) and much higher at large ones (1992 vs 631 MiB at pop 1000) — the interpreter's per-generation working set grows with `pop_size × max_nodes × max_conns`, while the XLA side stays close to its runtime floor.

So the stated CPU contract ("at least as fast as the frozen JAX baseline") is **not met today**, and `benchmarks/RESULTS.md` records the verdict as `FAIL` on timing. Where the rewrite does win:

- **No JAX, no torch, no CUDA toolchain.** `import tensorneat` pulls in neither, enforced by `test/release_gate_no_jax.py`; the runtime dependency set is `evoxc`, `numpy`, `networkx`, `matplotlib`, `sympy`.
- **The graph is data.** It is inspectable, cacheable and transformable (`evoxc.Graph`, `evoxc.lower(...)`), instead of being an opaque `jax.jit` dispatch; one executable per static shape is built once and reused.
- **Latency to first generation.** Build 0.56 s at pop 100 vs 2.74 s of XLA compilation, and 0.32 s for a tiny genome in the quickstart run above.
- **Switchable backends.** The same kernels run on the IREE compiler today (verified end to end), which is where a GPU or a large-batch workload would pay off — that win is **not measured here**; on this CPU `iree` is 5–24× slower (see Backends).
- **Memory at small scale and on the compiled leg** (see above and `benchmarks/BACKENDS.md`).

The measured bottleneck on the numpy backend is `tell` (speciation: 4 000–40 000 genome-distance pairs per generation), not the forward pass; `src/tensorneat/pipeline.py` documents the per-phase breakdown.

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
| `examples/func_fit/xor.py` | 2-input XOR, `DefaultGenome(2 -> 1, max_nodes=12)`, pop 500, 40 gens, target `-1e-3` | `uv run python examples/func_fit/xor.py` | ran, 3.7 s; stops at the generation limit with max fitness `-0.2500` on seed 42 |
| `examples/func_fit/xor3d.py` | 3-bit parity (`problem.XOR3d`), `max_nodes=16`, pop 500, 50 gens, target `-1e-4` | `uv run python examples/func_fit/xor3d.py` | ran, 6.2 s; stops at the generation limit (`-0.2500`) |
| `examples/func_fit/xor_origin.py` | `genome.OriginNode`/`OriginConn` (historical markers, whole-gene crossover) | `uv run python examples/func_fit/xor_origin.py` | ran, 8.1 s; loss 0.25 at the demo budget |
| `examples/func_fit/xor_recurrent.py` | `RecurrentGenome(activate_time=10)` on XOR3d | `uv run python examples/func_fit/xor_recurrent.py` | ran, 5.9 s; loss 0.25 at the demo budget |
| `examples/func_fit/custom_func_fit.py` | Pagie polynomial via `problem.CustomFuncFit` | `uv run python examples/func_fit/custom_func_fit.py` | ran, 6.8 s; final loss 0.0258 |
| `examples/func_fit/xor_hyperneat.py` | recurrent HyperNEAT over `FullSubstrate` (CPPN 4 coords -> 1 weight) | `uv run python examples/func_fit/xor_hyperneat.py` | ran, 6.4 s; loss 0.0463 |
| `examples/func_fit/xor_hyperneat_feedforward.py` | `HyperNEATFeedForward` + `MLPSubstrate([4, 6, 6, 1])` | `uv run python examples/func_fit/xor_hyperneat_feedforward.py` | ran, 9.5 s; loss 0.1502 |
| `examples/rl/cartpole.py` | `problem.CartPole`, two action logits, no `output_transform` | `uv run python examples/rl/cartpole.py` | ran, 2.4 s; 200/200 steps at generation 2 |
| `examples/interpret_visualize/genome_sympy.py` | sympy formula + Python code + LaTeX of one genome (host only) | `uv run python examples/interpret_visualize/genome_sympy.py` | ran, 1.0 s |
| `examples/interpret_visualize/visualize_genome.py` | draw one genome as SVG (host only) | `uv run python examples/interpret_visualize/visualize_genome.py` | ran, 2.0 s; writes `network.svg` next to the script |

The eight pipeline scripts use demo budgets (`# Demo budget:` in the source: pop 200–500, 20–50 generations, numpy backend); a real run raises `pop_size` into the thousands and the generation limit into the hundreds. The two `interpret_visualize` scripts are host-only (no `Pipeline`, ~2 s). Fitness is maximised everywhere, so a function-fitting `fitness_target` is a negative error bound. The Brax/Gymnax/mujoco examples of 0.1.x no longer exist.

## Tutorials

- `tutorials/quickstart.ipynb` — the current-API notebook, executed top to bottom: build a genome/algorithm/problem, `Pipeline.setup`/`step`/`auto_run`, the batch-first `best` payload, switching `backend=`, and inspecting the built `evoxc` executable.
- `tutorials/legacy_jax/` — the three pre-rewrite notebooks (`tutorial-00-functional programming and state.ipynb`, `tutorial-01-genome.ipynb`, `tutorial-02-Usage.ipynb`), kept for concepts only: their code is the old `jax`/`vmap` API and does not run against this tree.

## Project layout

| Path | Purpose |
|---|---|
| `src/tensorneat/pipeline.py` | the host compile site, generation loop, logging and checkpoint writer |
| `src/tensorneat/genome/` | gene payload, genetic operators, forward/grad algebra, host-only introspection |
| `src/tensorneat/algorithm/` | `NEAT` (speciation, spawn, breeding) and `HyperNEAT`/`HyperNEATFeedForward` + substrates |
| `src/tensorneat/problem/` | `XOR`, `XOR3d`, `FuncFit`/`CustomFuncFit`, `CartPole` |
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

Measured on this box, no optional extras installed:

```
$ uv run pytest -o addopts="" --co -q
505 tests collected in 0.43s

$ uv run pytest -o addopts="" -q
505 passed, 5 skipped in 31.57s
```

(`-o addopts=""` is needed because `pyproject.toml` sets `addopts = "-q"` and a doubled `-q` suppresses the summary line.) The five skips are the four JAX parity modules and `test/backends`, which skip at import when their extra is missing:

```
SKIPPED [1] test/parity/test_distance.py:57: parity needs the `bench` extra (jax)
SKIPPED [1] test/parity/test_fitness.py:49: parity needs the `bench` extra (jax)
SKIPPED [1] test/parity/test_forward.py:50: parity needs the `bench` extra (jax)
SKIPPED [1] test/parity/test_grad.py:58: parity needs the `bench` extra (jax)
```

With the `iree` extra installed the backend suite runs too: `uv run --extra iree pytest -o addopts="" -q` → `508 passed, 4 skipped in 40.21s`, and `uv run --extra iree pytest test/backends` → `3 passed in 8.31s` (a real XOR generation on `backend="iree"` compared against numpy).

| suite | proves |
|---|---|
| `test/unit/` | kernels against numpy oracles: schedule derivation, forward/NaN rules, mutation/crossover invariants, speciation bookkeeping, distance, `grad` vs finite differences, ACT/AGG tables, CartPole vs a host Euler rollout |
| `test/parity/` | the same kernels on identical payloads against `benchmarks/legacy_jax` (rtol 1e-5 forward, 1e-4 gradients), marker `parity` |
| `test/integration/` | whole-pipeline XOR evolution, per-seed determinism, rebuild on shape change, checkpoint round-trip, `show(state, best)`, marker `slow` |
| `test/backends/` | `backend="iree"` end to end and its agreement with numpy, marker `requires_iree` (in no gate) |
| `test/release_gate_no_jax.py` | `import tensorneat` leaves `jax`, `jaxlib`, `torch`, `brax`, `gymnax`, `mujoco` out of `sys.modules` — even under an import hook that raises for them — and a tiny XOR run still evolves |

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

The repository's license text is `LICENSE`, which currently contains the **GNU General Public License v3.0**, and `pyproject.toml` points at it with `license = { file = "LICENSE" }`.
Note the inconsistency: `pyproject.toml` also carries the classifier `License :: OSI Approved :: BSD License`, which contradicts that file. Treat `LICENSE` as authoritative until the two are reconciled.

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

- **Close the CPU gap on the numpy interpreter.** The measured bottleneck is `tell`, dominated by speciation's genome-distance pairs (4 000–40 000 per generation at pop 100–1000); the forward pass is not the problem.
- **Make a compiled backend actually competitive.** The IREE leg works but loses on llvm-cpu because the dense key-table `gather`/`argsort`/`scatter` traffic lowers to generic loops; that is a kernel-formulation project (sparse/segment reductions), not a compiler-flag one. No compiled win is claimed until it is measured.
- **Lift the remaining `evoxc` v1 exporter deferrals** upstream, then re-evaluate the compiled backends and re-run the contract.
- **More NEAT-family algorithms and substrates** (e.g. ES-HyperNEAT), and more RL problems that do not need JAX.
- **Docs and tutorials**: keep `docs/` and `tutorials/quickstart.ipynb` in step with the API.

Pull requests are welcome.
