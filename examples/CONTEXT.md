# examples — runnable scripts for the evoxc TensorNEAT API

## Intent
One small, copy-pasteable script per supported problem family, written against the public `tensorneat` API only.
Every pipeline script is the same four lines of orchestration: build `Pipeline(algorithm, problem, ...)`, `setup()`, `auto_run()`, `show()`.
`Pipeline.setup()` builds no graph: it is host-only (random key, initial population, task data).
The evoxc graph is built by the first `Pipeline.step()` inside `auto_run`, one-off per static shape, and `auto_run` prints `compile finished, cost time: ...s` once before generation 1.
Every demo runs on `backend="numpy"` and finishes in 2–20.5 s per script on this CPU (the two `interpret_visualize` demos are ~1 s).
Budgets stay demo-sized and marked `# Demo budget:` in the code; a real run raises `pop_size` into the thousands and the generation limit into the hundreds.
Nothing here imports `jax`, `torch`, `brax`, `gymnax` or `mujoco`.

## API Surface (how to run)
From the repository root, with the package importable (`uv pip install -e .` or `PYTHONPATH=src`), run `uv run python examples/<subdir>/<script>.py`.
- `examples/func_fit/` — supervised function fitting: `xor.py`, `xor3d.py`, `custom_func_fit.py`, `xor_recurrent.py`, `xor_origin.py`, `xor_hyperneat.py`, `xor_hyperneat_feedforward.py`.
- `examples/rl/cartpole.py` — evoxc-native `CartPole` rollout with a two-logit policy.
- `examples/interpret_visualize/` — host-only genome introspection: `genome_sympy.py`, `visualize_genome.py`.
`backend="numpy"` is the reference interpreter and works everywhere; `"iree"` / `"xla"` are opt-in extras, compiled once per shape, and an unsupported backend raises rather than falling back.
Fitness is maximised everywhere, so a `fitness_target` on a function-fitting run is a negative error bound (`-1e-3` ≈ MSE below 1e-3).

## Constraints
- Public API only: `from tensorneat import Pipeline, algorithm, genome, problem` plus `tensorneat.common.{ACT, AGG}`; `HyperNEATFeedForward` and the substrates come from `tensorneat.algorithm.hyperneat`.
- Construction-level configuration only; there is no eager compute mode, so a script never touches a traced tensor.
- `algorithm.num_inputs`/`num_outputs` come from the genome and must match `problem.input_shape`/`output_shape` (the `Pipeline` asserts it).
- Host-only introspection (`network_dict`, `repr`, `sympy_func`, `visualize`) never feeds back into a graph, and needs a one-individual (`P == 1`) payload.
- `max_in_degree` is a genome static: a recurrent genome accumulates many incoming edges per node, so its example sizes the window at 16; too small a window aborts the run with "the execution schedule dropped N connection(s)".
- No argparse, no helper packages, no unused imports; a script stays ≲ 110 lines and is a single readable top-level run.

## Test Strategy
Six of the ten pipeline demos cross their shipped `fitness_target` and stop on `Fitness limit reached!` at seed 42: `xor` (generation 67 of 150, loss 8.68e-04, 4.3 s), `xor3d` (144 of 150, loss 9.34e-04, 20.4 s), `xor_origin` (89 of 250, loss 8.86e-07, 5.8 s), `xor_recurrent` (72 of 200, loss 1.66e-10, 10.7 s), `cartpole` (2 of 20, return 200.00, 2.1 s) and `pendulum` (5 of 20, training max -271.30 against a -300 target, 3.8 s).
Three other pipeline demos stop on `Generation limit reached!` and only reduce their loss within the demo budget: `xor_hyperneat` (40 generations, 5.8 s, loss 0.25 → 0.0165), `xor_hyperneat_feedforward` (40 generations, 9.3 s, 0.223 → 0.188) and `custom_func_fit` (30 generations, 5.6 s, loss 0.0435 → 0.0300).
`mountaincar.py` sets `fitness_target=float("inf")`, so it has no target to cross: it always runs its full 10-generation budget (18.4 s) and ends at `MountainCarContinuous mean return: -0.00` with `goal not reached`.
The two `interpret_visualize` demos print host-side text and exit 0 in under a second, so these two plus all ten pipeline scripts exit 0.
A healthy pipeline run prints the compile line, a per-generation log, and ends with `pipeline.show(state, best)`.
No pytest target lives here: `test/` owns the automated suites, so these scripts are smoke demos rather than assertions.

## Routing Table
| Path | Purpose |
|---|---|
| `./func_fit` | Supervised function-fitting scripts (XOR, XOR3d, Pagie, recurrent, origin genes, both HyperNEAT variants). |
| `./rl` | `cartpole.py`: the evoxc-native CartPole problem, two-logit policy, backend-switch notes. |
| `./interpret_visualize` | Host-only genome introspection: sympy formula/Python/LaTeX and SVG topology drawing. |

## Notes for Agents
On the parity problems the constant-0.5 predictor is the exact linear optimum (MSE 0.25); the `survival_threshold` / `species_size` escape hatch is documented in `./func_fit/CONTEXT.md`.
The three `Generation limit reached!` demos are demo-budget loss reductions by design; do not make their docstrings claim to solve the problem.
