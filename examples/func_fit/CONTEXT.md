# examples/func_fit

## Intent
Runnable, copy-pasteable supervised function-fitting examples for the evoxc-based TensorNEAT API.
Each script is a single top-level pipeline run: build the `Pipeline`, `setup()`, `auto_run()`, `show()`.
Budgets stay small (pop 1000, ≤ 150 generations, numpy backend) so a run finishes well under a minute on CPU.

## API Surface
- `xor.py` — 2-input XOR, `DefaultGenome(num_inputs=2, max_nodes=12, output_transform=ACT.sigmoid)`, target `-1e-3`, 100 generations.
- `xor3d.py` — `problem.XOR3d()` (3 inputs, 8 samples), `max_nodes=16`, target `-1e-4`, 150 generations.
- `custom_func_fit.py` — Pagie polynomial via `problem.CustomFuncFit` with numpy bounds sampled at `seed=42`, `BiasNode` activation/aggregation menus, target `-1e-2`.
- `xor_recurrent.py` — `genome.RecurrentGenome(activate_time=10)` on XOR3d; `activate_time` is the tick-unroll cost knob.
- `xor_origin.py` — XOR3d with `genome.OriginNode()` / `genome.OriginConn()` (NEAT-paper whole-gene crossover + historical marker).

## Constraints
- Imports only `tensorneat` and `numpy`; never `jax`, `torch`, `brax`, `gymnax`, `mujoco`.
- Fitness is MAXIMISED, so `fitness_target` is a negative error bound.
- `CustomFuncFit.func` runs host-side with numpy, one 1-D point per call; it is never traced into a graph.
- Custom activation functions are plain callables in a per-genome `activation_options=[...]` list (no global registration).
- Keep each script ≲ 110 lines, plain module-level literals, `if __name__ == "__main__":` guard, no argparse.

## Known Gaps
- `xor_grad.py`, `xor_hyperneat.py`, `xor_hyperneat_feedforward.py` are still pre-rewrite (JAX-era) scripts and are not yet ported.
