# examples/func_fit

## Intent
Runnable, copy-pasteable supervised function-fitting examples for the evoxc-based TensorNEAT API.
Each script is a single top-level pipeline run: build the `Pipeline`, `setup()`, `auto_run()`, `show()`.
Budgets stay small (pop 1000, ≤ 150 generations, numpy backend) so a run finishes well under a minute on CPU.

## API Surface
Shared entry point is `from tensorneat import Pipeline, algorithm, genome, problem`, plus `from tensorneat.common import ACT`.
Every script builds one `Pipeline(algorithm=..., problem=..., seed=42, generation_limit=..., fitness_target=...)` and ends with `pipeline.show(state, best)`.

## Routing Table
| Script | Demonstrates |
|---|---|
| `xor.py` | 2-input XOR, `DefaultGenome(num_inputs=2, max_nodes=12, output_transform=ACT.sigmoid)`, target `-1e-3`, 100 generations. |
| `xor3d.py` | `problem.XOR3d()` (3 inputs, 8 samples), `max_nodes=16`, target `-1e-4`, 150 generations. |
| `custom_func_fit.py` | Pagie polynomial via `problem.CustomFuncFit` with numpy bounds sampled at `seed=42`, `BiasNode` activation/aggregation menus, target `-1e-2`. |
| `xor_recurrent.py` | `genome.RecurrentGenome(activate_time=10)` on XOR3d; `activate_time` is the tick-unroll cost knob. |
| `xor_origin.py` | XOR3d with `genome.OriginNode()` / `genome.OriginConn()` (NEAT-paper whole-gene crossover + historical marker). |
| `xor_hyperneat.py` | Recurrent HyperNEAT: CPPN genome (4 coord inputs → 1 weight) over `FullSubstrate`, `activate_time=10`, target `-1e-4`. |
| `xor_hyperneat_feedforward.py` | Feedforward HyperNEAT: `HyperNEATFeedForward` + `MLPSubstrate(layers=[4, 6, 6, 1])`, no `activate_time`, target `-1e-4`. |

## Constraints
- Imports only `tensorneat` and `numpy`; never `jax`, `torch`, `brax`, `gymnax`, `mujoco`.
- Fitness is MAXIMISED, so `fitness_target` is a negative error bound.
- `CustomFuncFit.func` runs host-side with numpy, one 1-D point per call; it is never traced into a graph.
- Custom activation functions are plain callables in a per-genome `activation_options=[...]` list (no global registration).
- A recurrent substrate pairs with `HyperNEAT` and needs `activate_time`; `MLPSubstrate` pairs with `HyperNEATFeedForward` and takes none.
- Keep each script ≲ 110 lines, plain module-level literals, `if __name__ == "__main__":` guard, no argparse.
