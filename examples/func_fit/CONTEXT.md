# examples/func_fit

## Intent
Runnable, copy-pasteable supervised function-fitting examples for the evoxc TensorNEAT API.
Each script is a single top-level pipeline run: build the `Pipeline`, `setup()`, `auto_run()`, `show()`.
Demo budgets are deliberately tiny and every run uses `backend="numpy"`; wall clock is 4.7–16.1 s per script on this CPU.

## API Surface
Shared entry point is `from tensorneat import Pipeline, algorithm, genome, problem`, plus `from tensorneat.common import ACT` (`AGG` too in `custom_func_fit.py`).
The two HyperNEAT scripts additionally import their algorithm and substrate from `tensorneat.algorithm.hyperneat`.
Every script builds one `Pipeline(algorithm=..., problem=..., seed=42, generation_limit=..., fitness_target=...)` and ends with `pipeline.show(state, best)`.
Fitness is maximised, so `fitness_target` is a negative error bound.
Run any script from the repository root as `uv run python examples/func_fit/<script>.py`.

## Routing Table
| Script | Demonstrates |
|---|---|
| `xor.py` | 2-input XOR, `DefaultGenome(num_inputs=2, max_nodes=12, sigmoid)`, pop 200, `species_size=10`, target `-1e-3`, 150 generations; stops `Fitness limit reached!` at generation 75, final loss `9.25e-04`, 4.7 s. |
| `xor3d.py` | `XOR3d()` (3 inputs, 8 samples), `max_nodes=16`, pop 400, `species_size=20`, target `-1e-4`, 150 generations; stops `Fitness limit reached!` at generation 93, final loss `6.88e-05`, 12.8 s. |
| `xor_origin.py` | XOR3d with `OriginNode()` / `OriginConn()` (NEAT-paper whole-gene crossover + historical marker), pop 300, target `-1e-6`, 250 generations; stops `Fitness limit reached!` at generation 123, final loss `7.48e-07`, 7.7 s. |
| `xor_recurrent.py` | `RecurrentGenome(activate_time=10, max_in_degree=16)` on XOR3d, pop 300, `survival_threshold=0.3`, target `-1e-4`, 200 generations; stops `Fitness limit reached!` at generation 110, final loss `1.88e-05`, 16.1 s. |
| `xor_hyperneat.py` | Recurrent HyperNEAT: CPPN (4 coord inputs → 1 weight) over `FullSubstrate`, `activate_time=10`, pop 300, target `-1e-4`, 40 generations; stops `Generation limit reached!`, loss improves 0.25 → 0.046, 5.4 s. |
| `xor_hyperneat_feedforward.py` | Feedforward HyperNEAT: `HyperNEATFeedForward` + `MLPSubstrate(layers=[4, 6, 6, 1])`, no `activate_time`, pop 300, target `-1e-4`, 40 generations; stops `Generation limit reached!`, loss improves 0.223 → 0.150, 7.9 s. |
| `custom_func_fit.py` | Pagie polynomial via `CustomFuncFit` with numpy bounds sampled at `seed=42`, `BiasNode` activation/aggregation menus, pop 200, target `-1e-2`, 30 generations; stops `Generation limit reached!`, loss improves 0.0435 → 0.0258, 5.2 s. |

## Notes for Agents
On these parity problems the constant-0.5 predictor is the exact linear optimum, so its MSE is exactly 0.25; a run stuck at `loss: 0.25` sits on that plateau, not broken.
What escapes the plateau is breeding diversity: `survival_threshold=0.01` floors to one parent per species (mutation-only hill climbing), and `xor_origin.py` holds the -0.25 plateau through generation 61 before it climbs.
`xor_recurrent.py` therefore uses `survival_threshold=0.3`; `xor3d.py` instead raises `species_size` to 20.
`xor_hyperneat.py`, `xor_hyperneat_feedforward.py` and `custom_func_fit.py` stop on `Generation limit reached!` by design and only reduce their loss within the demo budget; do not "fix" them by claiming convergence in their docstrings.

## Constraints
- Imports only `tensorneat` and `numpy`; never `jax`, `torch`, `brax`, `gymnax`, `mujoco`.
- `CustomFuncFit.func` runs host-side with numpy, one 1-D point per call; it is never traced into a graph.
- Custom activation functions are plain callables in a per-genome `activation_options=[...]` list (no global registration).
- A recurrent substrate pairs with `HyperNEAT` and needs `activate_time`; `MLPSubstrate` pairs with `HyperNEATFeedForward` and takes none.
- Keep each script ≲ 110 lines, plain module-level literals, `if __name__ == "__main__":` guard, no argparse.
- Budgets stay demo-sized behind a `# Demo budget:` comment; a real run uses pop_size in the thousands with a few hundred generations.

## Test Strategy
Each script is run by hand with `uv run python examples/func_fit/<script>.py` on the numpy backend and exits 0.
A healthy run prints `compile finished, cost time: ...s` once, then a per-generation log, and ends with `pipeline.show(state, best)`.
No pytest target lives here: `test/` owns the automated suites, so these are smoke demos rather than assertions.
