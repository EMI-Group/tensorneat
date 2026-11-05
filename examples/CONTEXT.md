# examples — runnable scripts for the evoxc TensorNEAT API

## Intent
One small, copy-pasteable script per supported problem family, written against the public `tensorneat` API (the two `interpret_visualize` demos also import `evoxc` directly).
Every pipeline script is the same four lines of orchestration: build `Pipeline(algorithm, problem, ...)`, `setup()`, `auto_run()`, `show()`.
`Pipeline.setup()` builds no graph: it is host-only (random key, initial population, task data).
The evoxc graph is built by the first `Pipeline.step()` inside `auto_run`, one-off per static shape, and `auto_run` prints `compile finished, cost time: ...s` once before generation 1.
Every demo runs on `backend="numpy"`; one fresh process per script takes 1.5–20.2 s end to end on this CPU including `uv run` startup, and the two `interpret_visualize` demos take 0.8–1.0 s.
Budgets stay demo-sized: the seven `func_fit` scripts mark theirs with a `# Demo budget:` comment, `rl/cartpole.py` marks its reduced `max_step` with `# Small budget:`, and `rl/pendulum.py` / `rl/mountaincar.py` set theirs through `pop_size` and `generation_limit` alone (`mountaincar.py` keeps the Gym-faithful 999-step horizon).
A real run raises `pop_size` into the thousands and the generation limit into the hundreds.
Nothing here imports `jax`, `torch`, `brax`, `gymnax` or `mujoco`.

## API Surface (how to run)
From the repository root, with the package importable (`uv pip install -e .` or `PYTHONPATH=src`), run `uv run python examples/<subdir>/<script>.py`.
- `examples/func_fit/` — supervised function fitting: `xor.py`, `xor3d.py`, `custom_func_fit.py`, `xor_recurrent.py`, `xor_origin.py`, `xor_hyperneat.py`, `xor_hyperneat_feedforward.py`.
- `examples/rl/` — evoxc-native rollouts: `cartpole.py` (two-logit policy), `pendulum.py` (one torque logit), `mountaincar.py` (one force logit, `fitness_target=float("inf")`).
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
Six of the ten pipeline demos cross their shipped `fitness_target` and stop on `Fitness limit reached!` at seed 42 (wall clock measured here, machine-specific): `xor` (generation 67 of 150, loss 8.68e-04, 4.3–4.4 s), `xor3d` (144 of 150, loss 9.34e-04, 20.0–20.6 s), `xor_origin` (89 of 250, loss 8.86e-07, 5.8–6.2 s), `xor_recurrent` (72 of 200, loss 1.66e-10, 10.6–10.9 s), `cartpole` (2 of 20, mean return 200.00, 1.4–1.6 s) and `pendulum` (5 of 20, training max -271.30 against a -300 target, 2.7–3.0 s; `show` reports the same genome at return -595.12).
Three other pipeline demos stop on `Generation limit reached!` and only reduce their loss within the demo budget: `xor_hyperneat` (40 generations, 5.7–6.0 s, loss 0.2500 → 0.0165), `xor_hyperneat_feedforward` (40 generations, 9.4–9.8 s, 0.2232 → 0.1876) and `custom_func_fit` (30 generations, 5.1–5.3 s, loss 0.0435 → 0.0300).
`mountaincar.py` sets `fitness_target=float("inf")`, so it has no target to cross: it always runs its full 10-generation budget (11.9–12.5 s) and ends at `MountainCarContinuous mean return: -0.00` with `goal not reached`.
The two `interpret_visualize` demos print host-side text and exit 0 (0.8–1.0 s each), so these two plus all ten pipeline scripts exit 0.
A healthy pipeline run prints the compile line, a per-generation log, and ends with `pipeline.show(state, best)`.
No pytest target lives here: `test/` owns the automated suites, so these scripts are smoke demos rather than assertions.

## Routing Table
| Path | Purpose |
|---|---|
| `./func_fit` | Supervised function-fitting scripts (XOR, XOR3d, Pagie, recurrent, origin genes, both HyperNEAT variants); six of the ten pipeline demos live here and four of them cross their target. |
| `./rl` | `cartpole.py`, `pendulum.py`, `mountaincar.py`: the evoxc-native RL problems, their policy shapes and backend-switch notes. |
| `./interpret_visualize` | Host-only genome introspection: sympy formula/Python/LaTeX and SVG topology drawing. |

## Notes for Agents
On the parity problems the constant-0.5 predictor is the exact linear optimum (MSE 0.25); the `survival_threshold` / `species_size` escape hatch is documented in `./func_fit/CONTEXT.md`.
The four `Generation limit reached!` demos are bounded by their demo budget and must not be documented as solving anything: three only reduce their loss and MountainCarContinuous sits at the reward's idle optimum.
Every generation, loss and return quoted in this subtree was measured by running the script at seed 42; a change to `pop_size`, `max_step`, `generation_limit` or a genome static invalidates the numbers in the script and in the child CONTEXT.md files together.
Wall-clock figures are this box's measurements and are labelled as machine-specific wherever they are quoted.
