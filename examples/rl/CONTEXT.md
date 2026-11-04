# examples/rl

## Intent
`cartpole.py` is a single, copy-pasteable example of the evoxc-native `CartPole` problem evolved with NEAT.
The rollout is a fixed-horizon loop, so evaluation cost grows linearly with `max_step`.
On the numpy reference backend it finishes in 2.3 s and reaches a full 200-step episode in 2 generations.

## API Surface
- Imports `from tensorneat import Pipeline, algorithm, genome, problem` plus `from tensorneat.common import ACT, AGG`.
- `algorithm.NEAT(pop_size=200, species_size=10, survival_threshold=0.1, genome=genome.DefaultGenome(num_inputs=4, num_outputs=2, init_hidden_layers=(), node_gene=genome.BiasNode(activation_options=ACT.tanh, aggregation_options=AGG.sum)))`.
- `problem.CartPole(max_step=200, repeat_times=1, fitness_target=200)`, `Pipeline(..., seed=42, generation_limit=20, fitness_target=200, backend="numpy")`.
- Run from the repository root as `uv run python examples/rl/cartpole.py`.

## Policy and cost
Two output logits: the problem pushes right iff `out[1] > out[0]`.
There is no `output_transform`, because an argmax-style transform would collapse the two logits into one value and destroy the ordering the environment needs.
`max_step` is both the fitness ceiling (a frozen episode's return is its survival time) and the cost knob.
The problem raises `TensorNEATError` if `fitness_target > max_step`.
Per-generation cost on the numpy reference backend is ~0.4–0.5 s at pop 200 / `max_step=200` and ~0.9 s at pop 100 / `max_step=500`; the ~0.3 s compile is negligible.
A compiled backend is not faster on CPU for this workload.

## Backend switch
`backend="numpy"` is the reference interpreter and always available.
`"iree"` / `"xla"` are opt-in extras, compiled once per static shape and measured slower on CPU.
An unsupported backend raises instead of silently falling back.

## Constraints
Public API only, construction-level configuration only; no `jax`, `torch`, `brax`, `gymnax` or `mujoco`.
Keep the script ≲ 110 lines with plain module-level literals, an `if __name__ == "__main__":` guard and no argparse.
`Pipeline.setup()` is host-only; the evoxc graph is built by the first `step()` inside `auto_run` and cached for every generation.

## Test Strategy
Run by hand on the numpy backend; a healthy run prints `compile finished, cost time: ...s`, then a per-generation log, and exits 0 after `pipeline.show(state, best)`.
With `seed=42` it prints `CartPole mean return: 200.00` and stops on `Fitness limit reached!` after 2 of the 20 allowed generations.
No pytest target lives here: `test/` owns the automated suites.

## Routing Table
| Script | Demonstrates |
|---|---|
| `cartpole.py` | evoxc-native `CartPole` rollout, two-logit policy without `output_transform`, `max_step` as the fitness ceiling and cost knob, backend-switch note. |
