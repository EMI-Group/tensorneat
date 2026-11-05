# examples/rl

## Intent
Three runnable demos, one per shipped RL environment: `cartpole.py`, `pendulum.py`, `mountaincar.py`.
Each is the same four lines of orchestration: build `Pipeline(algorithm, problem, ...)`, `setup()`, `auto_run()`, `show()`.
`Pipeline.setup()` builds no graph: the evoxc graph is built by the first `step()` inside `auto_run`, one-off per static shape, and prints `compile finished, cost time: ...s` once before generation 1.
Budgets stay demo-sized (`pop_size` 100–200, `generation_limit` 10–20) and `cartpole.py` marks its reduced `max_step` with `# Small budget:`; a real run raises `pop_size` into the thousands and the generation limit into the hundreds.
Nothing here imports `jax`, `torch`, `brax`, `gymnax` or `mujoco`.

## API Surface (how to run)
From the repository root, with the package importable (`uv sync --all-extras` or `uv run` after `uv sync`), run `uv run python examples/rl/<script>.py`.
- `cartpole.py`: `NEAT(pop_size=200, species_size=10, survival_threshold=0.1, genome=DefaultGenome(num_inputs=4, num_outputs=2, init_hidden_layers=(), node_gene=BiasNode(ACT.tanh, AGG.sum)))`, `problem.CartPole(max_step=200, repeat_times=1, fitness_target=200)`, `Pipeline(seed=42, generation_limit=20, fitness_target=200, backend="numpy")`.
- `pendulum.py`: `NEAT(pop_size=100, species_size=10, survival_threshold=0.1, genome=DefaultGenome(num_inputs=2, num_outputs=1, init_hidden_layers=(6,), node_gene=BiasNode(ACT.tanh, AGG.sum)))`, `problem.Pendulum(max_step=200, fitness_target=-300)`, `Pipeline(seed=42, generation_limit=20, fitness_target=-300, backend="numpy")`.
- `mountaincar.py`: the same `NEAT` recipe as Pendulum but with `init_hidden_layers=()`, `problem.MountainCarContinuous()` with no `max_step` override (so it runs the Gym-faithful 999-step default), `Pipeline(seed=42, generation_limit=10, fitness_target=float("inf"), backend="numpy")`.
No script sets `output_transform`, because all three problems own their action mapping or logit ordering.
`fitness_target` on `Pipeline` must be a number: it is compared as `max fitness >= fitness_target`, so `float("inf")` is the documented spelling of "never stop early" and `None` would raise a `TypeError`.
Import lines are `from tensorneat import Pipeline, algorithm, genome, problem` plus `from tensorneat.common import ACT, AGG` in every script.

## Policy and cost
### CartPole
Two output logits: the problem pushes right iff `out[1] > out[0]`.
There is no `output_transform`, because an argmax-style transform would collapse the two logits into one value and destroy the ordering the environment needs.
`max_step` is both the fitness ceiling (a frozen episode's return is its survival time) and the cost knob; the problem raises `TensorNEATError` if `fitness_target > max_step`.
Per-generation cost on the numpy reference backend is 0.21–0.28 s at pop 200 / `max_step=200` and ~0.38 s at pop 100 / `max_step=500`; the 0.28–0.30 s build is negligible.
### Pendulum
Two inputs `(theta, theta_dot)`, one output logit; the problem owns the squashing, `u = max_torque * tanh(logit)`, so `output_transform` must stay unset — a bounded transform would rescale the logit and shrink the torque range.
`theta = 0` is upright and every return is `-cost <= 0`, so `fitness_target` must be `<= 0` here.
The `init_hidden_layers=(6,)` layer is load-bearing: the same budget without it plateaus around -527 (per the problem node's notes), while with it the run reaches the target at generation 5.
Per-generation cost is 0.23–0.40 s at pop 100 / `max_step=200` with the hidden layer, plus a 0.28–0.30 s one-off build; the hidden layer roughly doubles the genome forward, which is the dominant term.
### MountainCarContinuous
Two inputs `(position, velocity)`, one output logit; the problem owns the squashing, `a = tanh(logit)`, so `output_transform` must stay unset.
Cost is linear in the horizon: 0.63–1.24 s per generation at pop 100 / `max_step=999` (≈10.4 s for the ten generations, which is why the budget is 10), plus a 0.28 s build; the same population at 200 steps costs ~0.15 s per generation.
This environment's reward has an idle optimum — `a = 0` returns exactly `0.0` while any action costs `0.1*a**2` per step — so evolution at pop 100 converges to idleness instead of solving.
Measured references for the script docstring: a hand-written `a = sign(velocity)` bang-bang policy returns `90.8` (min 89.1 over 8 episodes) at `max_step` 200 and 999, "always push right" returns exactly `-20` at 200 and `-99.9` at 999, and a solving episode nets about `+89` because the penalty accrues over the whole climb.
The script therefore demonstrates the integration (environment + one-logit policy shape + pipeline), not a solved control task, and its docstring says so.

## Backend switch
`backend="numpy"` is the reference interpreter and always available; all three demos use it.
`"iree"` / `"xla"` are opt-in extras, compiled once per static shape and measured slower end to end on CPU for these workloads.
An unsupported backend raises instead of silently falling back.

## Measured output (seed 42, `backend="numpy"`, this CPU box)
All three scripts exit 0 with the full per-generation log; nothing is trimmed here.
`cartpole.py`, wall clock 1.4–1.5 s: 2 of the 20 generations run (per-generation max fitness `169.00`, then `200.00`), `Fitness limit reached!`, `CartPole mean return: 200.00`.
`pendulum.py`, wall clock 2.7–2.9 s, of which the five generations cost 0.23–0.40 s each (≈1.7 s in total) and the build 0.28–0.30 s:
```
generation max fitness: -1004.6690, -866.0824, -767.3397, -447.8309, -271.2978Fitness limit reached!
Pendulum repeat 0: return -595.12
	final state: theta=-3.8919 rad, theta_dot=-0.7849
Pendulum mean return: -595.12
```
The per-generation `max` column tops out at -271.30 (generation 5), which is the training fitness; `show` builds its own one-episode executable on `state.randkey` and reports -595.12 for the same genome, so the two numbers are different measurements, not a contradiction.
`mountaincar.py`, wall clock 11.6–11.8 s, of which ~10.4 s is generations at 0.63–1.24 s each, 0.28 s is the build and ~1 s is interpreter/`uv run` startup:
```
generation max fitness: -0.0332, -0.0111, -0.0022, -0.0002, -0.0002, -0.0003, -0.0002, -0.0000, -0.0000, -0.0000Generation limit reached!
MountainCarContinuous repeat 0: return -0.00, goal not reached
	final state: position=-0.5118, velocity=-0.0004
MountainCarContinuous mean return: -0.00
```
The run stops on `Generation limit reached!` because `fitness_target=float("inf")` never triggers; the best mean return is `-0.00` (the idle policy) and `show` reports `goal not reached`, exactly the idle optimum the docstring warns about.

## Constraints
Public API only: `from tensorneat import Pipeline, algorithm, genome, problem` plus `tensorneat.common.{ACT, AGG}`.
Construction-level configuration only; there is no eager compute mode, so a script never touches a traced tensor.
`algorithm.num_inputs`/`num_outputs` come from the genome and must match `problem.input_shape`/`output_shape` (the `Pipeline` asserts it): 4→2 for CartPole, 2→1 for Pendulum and MountainCarContinuous.
Never set `output_transform` on these three problems: each one owns its action mapping (argmax ordering) or squashing (`tanh`).
Host-only introspection (`network_dict`, `repr`, `sympy_func`, `visualize`) never feeds back into a graph, and needs a one-individual (`P == 1`) payload.
`max_in_degree` is a genome static: a recurrent genome accumulates many incoming edges per node (the `func_fit` recurrent example sizes the window at 16); too small a window aborts the run with "the execution schedule dropped N connection(s)".
No argparse, no helper packages, no unused imports; a script stays ≲ 110 lines and is a single readable top-level run (`cartpole.py` 71, `pendulum.py` 82, `mountaincar.py` 88 lines).
Keep the whole-repo rule that nothing here imports `jax`, `torch`, `brax`, `gymnax` or `mujoco`.

## Test Strategy
Run by hand on the numpy backend; a healthy run prints `compile finished, cost time: ...s`, then a per-generation log, and exits 0 after `pipeline.show(state, best)`.
With `seed=42`: `cartpole.py` prints `CartPole mean return: 200.00` after 2 of 20 generations (1.4–1.5 s wall); `pendulum.py` prints `Pendulum mean return: -595.12` after 5 of 20 generations (2.7–2.9 s wall); `mountaincar.py` prints `MountainCarContinuous mean return: -0.00` after 10 of 10 generations (11.6–11.8 s wall).
All runs are deterministic per seed: repeated runs reproduce every fitness figure above to the last digit, only the `Cost time` column moves.
The three runs are the measured reference for the budgets; a change to `pop_size`, `max_step`, `generation_limit` or `init_hidden_layers` invalidates the numbers in the docstrings and in this file together.
No pytest target lives here: `test/` owns the automated suites, so these scripts are smoke demos rather than assertions.

## Routing Table
| Script | Demonstrates |
|---|---|
| `cartpole.py` | Discrete `CartPole` rollout, two-logit policy without `output_transform`, `max_step` as the fitness ceiling and cost knob, backend-switch note. |
| `pendulum.py` | Continuous `Pendulum` swing-up, one torque logit squashed inside the problem (`max_torque*tanh`), 6-node hidden layer as the learning enabler, negative return so `fitness_target <= 0`, measured 2.7–2.9 s demo. |
| `mountaincar.py` | Continuous `MountainCarContinuous` at the Gym-faithful 999-step default, one action logit, `fitness_target=float("inf")` to force the full budget, and the reward's idle optimum documented instead of a fake solve. |
