# examples — runnable scripts for the evoxc TensorNEAT API

## Intent
One small, copy-pasteable script per supported problem family, written against the public `tensorneat` API only.
Every pipeline script is the same four lines of orchestration: build `Pipeline(algorithm, problem, ...)`, `setup()`, `auto_run()`, `show()`.
Budgets are deliberately tiny and marked `# Demo budget:` in the code (pop 200–500, 20–50 generations, `backend="numpy"`), so each run finishes in seconds to well under a minute on CPU and needs no optional dependency.
A real run raises `pop_size` into the thousands and the generation limit into the hundreds; nothing else changes.
Nothing here imports `jax`, `torch`, `brax`, `gymnax` or `mujoco`; the simulators and the EvoX 1.x adaptor those examples used no longer exist.

## API Surface (how to run)
- From the repository root, with the package importable (`uv pip install -e .` or `PYTHONPATH=src`):
  - `python examples/func_fit/xor.py` — 2-input XOR, the smallest end-to-end run.
  - `python examples/func_fit/xor3d.py` — 3-input parity, 8 samples.
  - `python examples/func_fit/custom_func_fit.py` — Pagie polynomial through `CustomFuncFit`.
  - `python examples/func_fit/xor_recurrent.py` — `RecurrentGenome` (tick-unrolled forward).
  - `python examples/func_fit/xor_origin.py` — NEAT-original gene copy + historical marker.
  - `python examples/func_fit/xor_hyperneat.py` — recurrent HyperNEAT over `FullSubstrate`.
  - `python examples/func_fit/xor_hyperneat_feedforward.py` — `HyperNEATFeedForward` over `MLPSubstrate`.
  - `python examples/rl/cartpole.py` — evoxc-native `CartPole` rollout, two logits, no `output_transform`.
  - `python examples/interpret_visualize/genome_sympy.py` — sympy formula / Python / LaTeX of one genome (host only).
  - `python examples/interpret_visualize/visualize_genome.py` — SVG drawing of one genome (host only, writes `network.svg` next to itself).
- `backend="numpy"` is the reference interpreter and works everywhere; `"iree"` / `"xla"` are opt-in extras, compiled once per shape, and an unsupported backend raises rather than falling back.
- Fitness is maximised everywhere, so a `fitness_target` on a function-fitting run is a negative error bound (`-1e-3` ≈ MSE below 1e-3).

## Constraints
- Public API only: `from tensorneat import Pipeline, algorithm, genome, problem` plus `tensorneat.common.{ACT, AGG}`; `HyperNEATFeedForward` and the substrates come from `tensorneat.algorithm.hyperneat`.
- Construction-level configuration only; there is no eager compute mode, so a script never touches a traced tensor.
- `algorithm.num_inputs`/`num_outputs` come from the genome and must match `problem.input_shape`/`output_shape` (the `Pipeline` asserts it).
- Host-only introspection (`network_dict`, `repr`, `sympy_func`, `visualize`) never feeds back into a graph, and needs a one-individual (`P == 1`) payload.
- No argparse, no helper packages, no unused imports; a script stays ≲ 110 lines and is a single readable top-level run.

## Test Strategy
- Each script is run by hand on the numpy backend with its demo budget; a healthy run prints a per-generation log, ends with `pipeline.show(state, best)` and exits 0.
- `interpret_visualize/*.py` need only sympy and matplotlib, finish in ~2 s, and write `network.svg` next to themselves (a generated artifact, not tracked).
- No pytest target lives here: `test/` owns the automated suites, so these scripts are smoke demos rather than assertions.

## Routing Table
| Path | Purpose |
|---|---|
| `./func_fit` | Supervised function-fitting scripts (XOR, XOR3d, Pagie, recurrent, origin genes, both HyperNEAT variants). |
| `./rl` | `cartpole.py`: the evoxc-native CartPole problem, two-logit policy, backend-switch notes. |
| `./interpret_visualize` | Host-only genome introspection: sympy formula/Python/LaTeX and SVG topology drawing. |
