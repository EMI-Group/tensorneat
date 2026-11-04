# examples — runnable scripts for the evoxc TensorNEAT API

## Intent
One small, copy-pasteable script per supported problem family, written against the public `tensorneat` API (the two `interpret_visualize` demos also import `evoxc` directly).
Every pipeline script is the same four lines of orchestration: build `Pipeline(algorithm, problem, ...)`, `setup()`, `auto_run()`, `show()`.
`Pipeline.setup()` builds no graph: it is host-only (random key, initial population, task data).
The evoxc graph is built by the first `Pipeline.step()` inside `auto_run`, one-off per static shape, and `auto_run` prints `compile finished, cost time: ...s` once before generation 1.
Every demo runs on `backend="numpy"` and finishes in 0.8–19.6 s per script on this CPU, wall clock including `uv run` startup; the two `interpret_visualize` demos are ~1 s and `rl/mountaincar.py` is the slowest at ~19.6 s.
Budgets stay demo-sized and marked where they are set (`# Demo budget:` in every function-fitting script, `# Small budget:` in `rl/cartpole.py`); a real run raises `pop_size` into the thousands and the generation limit into the hundreds.
Nothing here imports `jax`, `torch`, `brax`, `gymnax` or `mujoco`.

## API Surface (how to run)
From the repository root, with the package importable (`uv pip install -e .` or `PYTHONPATH=src`), run `uv run python examples/<subdir>/<script>.py`.
- `examples/func_fit/` — supervised function fitting: `xor.py`, `xor3d.py`, `custom_func_fit.py`, `xor_recurrent.py`, `xor_origin.py`, `xor_hyperneat.py`, `xor_hyperneat_feedforward.py`.
- `examples/rl/` — simulator-free evoxc-native control: `cartpole.py` (discrete, two logits), `pendulum.py` (continuous torque, 6-node hidden layer) and `mountaincar.py` (continuous force, Gym-faithful 999-step horizon, `fitness_target=float("inf")` to run the full budget).
- `examples/interpret_visualize/` — host-only genome introspection: `genome_sympy.py`, `visualize_genome.py`.
`backend="numpy"` is the reference interpreter and works everywhere; `"iree"` / `"xla"` are opt-in extras, compiled once per shape, and an unsupported backend raises rather than falling back.
Fitness is maximised everywhere, so a `fitness_target` on a function-fitting run is a negative error bound (`-1e-3` ≈ MSE below 1e-3); on the RL problems it is a return (200 for CartPole, -300 for Pendulum) or `float("inf")` for MountainCarContinuous.

## Constraints
- The seven `func_fit` scripts and all three `rl` scripts use the public API only: `from tensorneat import Pipeline, algorithm, genome, problem` plus `tensorneat.common.{ACT, AGG}`, and the two HyperNEAT scripts additionally take `HyperNEAT`/`HyperNEATFeedForward` and the substrates from `tensorneat.algorithm.hyperneat`.
- The two `interpret_visualize` demos build one individual themselves: `evoxc.build(lambda key: g.initialize(state, key, 1), evoxc.TensorSpec((), evoxc.int64), backend="numpy")` then `evoxc.run(exe, evoxc.random.key(42))`; `genome_sympy.py` additionally imports `to_python_code`/`to_latex_code` from `tensorneat.common.sympy_tools`.
- Construction-level configuration only; there is no eager compute mode, so a script never touches a traced tensor.
- `algorithm.num_inputs`/`num_outputs` come from the genome and must match `problem.input_shape`/`output_shape` (the `Pipeline` asserts it).
- Host-only introspection (`network_dict`, `repr`, `sympy_func`, `visualize`) never feeds back into a graph, and needs a one-individual (`P == 1`) payload.
- `max_in_degree` is a genome static: a recurrent genome accumulates many incoming edges per node, so `xor_recurrent.py` sizes the window at 16; too small a window aborts the run with "the execution schedule dropped N connection(s)".
- No argparse, no helper packages, no unused imports; a script stays ≲ 110 lines and is a single readable top-level run.

## Test Strategy
Six of the ten pipeline demos stop on `Fitness limit reached!` at seed 42: `xor` (4.7 s), `xor3d` (12.8 s), `xor_origin` (7.7 s), `xor_recurrent` (16.1 s), `cartpole` (2.2 s) and `pendulum` (5.3 s, mean return -48.08).
The four remaining pipeline demos stop on `Generation limit reached!` and only reduce their loss (or stay idle) within the demo budget: `xor_hyperneat` (5.4 s, 0.25 → 0.046), `xor_hyperneat_feedforward` (7.9 s, 0.223 → 0.150), `custom_func_fit` (5.2 s, 0.0435 → 0.0258) and `mountaincar` (19.6 s, mean return -0.00, goal not reached).
The two `interpret_visualize` demos print host-side text and exit 0 in about a second.
A healthy pipeline run prints the compile line, a per-generation log, and ends with `pipeline.show(state, best)`.
No pytest target lives here: `test/` owns the automated suites, so these scripts are smoke demos rather than assertions.

## Routing Table
| Path | Purpose |
|---|---|
| `./func_fit` | Supervised function-fitting scripts (XOR, XOR3d, Pagie, recurrent, origin genes, both HyperNEAT variants). |
| `./rl` | `cartpole.py`, `pendulum.py`, `mountaincar.py`: the evoxc-native RL problems, their policy shapes and backend-switch notes. |
| `./interpret_visualize` | Host-only genome introspection: sympy formula/Python/LaTeX and SVG topology drawing. |

## Notes for Agents
On the parity problems the constant-0.5 predictor is the exact linear optimum (MSE 0.25); the `survival_threshold` / `species_size` escape hatch is documented in `./func_fit/CONTEXT.md`.
The four `Generation limit reached!` demos are bounded by their demo budget and must not be documented as solving anything: three only reduce their loss and MountainCarContinuous sits at the reward's idle optimum.
