# examples/func_fit

## Intent
Runnable, copy-pasteable supervised function-fitting examples for the evoxc TensorNEAT API.
Each script is a single top-level pipeline run: build the `Pipeline`, `setup()`, `auto_run()`, `show()`.
Demo budgets are deliberately tiny and every run uses `backend="numpy"`; wall clock is 4.2–19.8 s per script on this CPU, including `uv run` startup.
## API Surface
Shared entry point is `from tensorneat import Pipeline, algorithm, genome, problem`, plus `from tensorneat.common import ACT` (`AGG` too in `custom_func_fit.py`).
The two HyperNEAT scripts additionally import their algorithm and substrate from `tensorneat.algorithm.hyperneat`.
Every script builds one `Pipeline(algorithm=..., problem=..., seed=42, generation_limit=..., fitness_target=...)` and ends with `pipeline.show(state, best)`.
Fitness is maximised, so `fitness_target` is a negative error bound.
Run any script from the repository root as `uv run python examples/func_fit/<script>.py`.

## Routing Table
| Script | Demonstrates |
|---|---|
| `xor.py` | 2-input XOR, `DefaultGenome(num_inputs=2, max_nodes=12, sigmoid)`, pop 200, `species_size=10`, target `-1e-3`, 150 generations; stops `Fitness limit reached!` at generation 67, final loss `8.68e-04`, 4.2–4.6 s wall. |
| `xor3d.py` | `XOR3d()` (3 inputs, 8 samples), `max_nodes=16`, pop 400, `species_size=20`, target `-1e-3`, 150 generations; stops `Fitness limit reached!` at generation 144, final loss `9.34e-04`, 19.5–19.8 s wall. |
| `xor_origin.py` | XOR3d with `OriginNode()` / `OriginConn()` (NEAT-paper whole-gene crossover + historical marker), pop 300, target `-1e-6`, 250 generations; stops `Fitness limit reached!` at generation 89, final loss `8.86e-07`, 5.8 s wall. |
| `xor_recurrent.py` | `RecurrentGenome(activate_time=10, max_in_degree=16)` on XOR3d, pop 300, `survival_threshold=0.3`, target `-1e-4`, 200 generations; stops `Fitness limit reached!` at generation 72, final loss `1.66e-10`, 10.2–10.6 s wall. |
| `xor_hyperneat.py` | Recurrent HyperNEAT: CPPN (4 coord inputs → 1 weight) over `FullSubstrate`, `activate_time=10`, pop 300, target `-1e-4`, 40 generations; stops `Generation limit reached!`, loss improves 0.25 → 0.0165, 5.6–5.8 s wall. |
| `xor_hyperneat_feedforward.py` | Feedforward HyperNEAT: `HyperNEATFeedForward` + `MLPSubstrate(layers=[4, 6, 6, 1])`, no `activate_time`, pop 300, target `-1e-4`, 40 generations; stops `Generation limit reached!`, loss improves 0.223 → 0.188, 9.0–9.6 s wall. |
| `custom_func_fit.py` | Pagie polynomial via `CustomFuncFit` with numpy bounds sampled at `seed=42`, `BiasNode` activation/aggregation menus, pop 200, target `-1e-2`, 30 generations; stops `Generation limit reached!`, loss improves 0.0435 → 0.0300, 4.9–5.2 s wall. |
## Notes for Agents
On these parity problems the constant-0.5 predictor is the exact linear optimum, so its MSE is exactly 0.25; a run stuck at `loss: 0.25` sits on that plateau, not broken.
What escapes the plateau is breeding diversity: `survival_threshold=0.01` floors to one parent per species (mutation-only hill climbing), and `xor_origin.py` holds the -0.25 plateau through generation 61 before it climbs.
`xor_recurrent.py` therefore uses `survival_threshold=0.3`; `xor3d.py` instead raises `species_size` to 20.
Measured at this HEAD with seed 42: four of the seven scripts cross their shipped `fitness_target` and stop on `Fitness limit reached!` — `xor` (generation 67), `xor3d` (generation 144), `xor_origin` (89) and `xor_recurrent` (72).
`xor3d.py` ships `fitness_target=-1e-3`, not `-1e-4`: at this budget the full 150 generations end at loss `1.6e-04`, so a `-1e-4` bound would never trigger.
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
