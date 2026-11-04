# examples/rl

## Intent
Three runnable demos, one per shipped RL environment: `cartpole.py`, `pendulum.py`, `mountaincar.py`.
Each is the same four lines of orchestration: build `Pipeline(algorithm, problem, ...)`, `setup()`, `auto_run()`, `show()`.
`Pipeline.setup()` builds no graph: the evoxc graph is built by the first `step()` inside `auto_run`, one-off per static shape, and prints `compile finished, cost time: ...s` once before generation 1.
Budgets stay demo-sized and are marked `# Demo budget:`-style in the code; a real run raises `pop_size` into the thousands and the generation limit into the hundreds.
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
Per-generation cost on the numpy reference backend is ~0.4–0.5 s at pop 200 / `max_step=200` and ~0.9 s at pop 100 / `max_step=500`; the ~0.3 s compile is negligible.
### Pendulum
Two inputs `(theta, theta_dot)`, one output logit; the problem owns the squashing, `u = max_torque * tanh(logit)`, so `output_transform` must stay unset — a bounded transform would rescale the logit and shrink the torque range.
`theta = 0` is upright and every return is `-cost <= 0`, so `fitness_target` must be `<= 0` here.
The `init_hidden_layers=(6,)` layer is load-bearing: the same budget without it plateaus around -527 (per the problem node's notes), while with it the run reaches the target at generation 8.
Per-generation cost is 0.35–0.52 s at pop 100 / `max_step=200` with the hidden layer, plus a 0.27 s one-off build; the hidden layer roughly doubles the genome forward, which is the dominant term (the rollout itself is ~46 ms).
### MountainCarContinuous
Two inputs `(position, velocity)`, one output logit; the problem owns the squashing, `a = tanh(logit)`, so `output_transform` must stay unset.
Cost is linear in the horizon: 1.07–1.73 s per generation at pop 100 / `max_step=999` (which is why the budget is 10 generations), plus a 0.27 s build; the same population at 200 steps is ~0.38 s per generation.
This environment's reward has an idle optimum — `a = 0` returns exactly `0.0` while any action costs `0.1*a**2` per step — so evolution at pop 100 converges to idleness instead of solving.
Measured references for the script docstring: a hand-written `a = sign(velocity)` bang-bang policy returns `90.8` at `max_step` 200 and 999, "always push right" returns `-20` at 200 and `-99.9` at 999, and a solving episode nets only ~`+89` because the penalty accrues over the whole climb.
The script therefore demonstrates the integration (environment + one-logit policy shape + pipeline), not a solved control task, and its docstring says so.

## Backend switch
`backend="numpy"` is the reference interpreter and always available; all three demos use it.
`"iree"` / `"xla"` are opt-in extras, compiled once per static shape and measured slower end to end on CPU for these workloads.
An unsupported backend raises instead of silently falling back.

## Measured output (seed 42, `backend="numpy"`, 20-core CPU box)
Both scripts exit 0 with the full generation log; nothing is trimmed here.
`pendulum.py`, wall clock 5.21 s (repeat runs: 5.12 s warm, 6.86 s cold including interpreter startup), of which generations cost 0.35–0.52 s each and the build 0.27 s:
```
compile finished, cost time: 0.273558s
Generation: 1, Cost time: 350.73ms
 	fitness: valid cnt: 100, max: -1004.6690, min: -1744.2228, mean: -1469.5670, std: 172.5686
	node counts: max: 10, min: 8, mean: 9.01
 	conn counts: max: 19, min: 15, mean: 17.83
 	species: 1, [100]
Generation: 2, Cost time: 423.94ms
 	fitness: valid cnt: 100, max: -866.0824, min: -1795.5364, mean: -1428.4888, std: 193.8810
	node counts: max: 11, min: 8, mean: 9.20
 	conn counts: max: 20, min: 15, mean: 17.97
 	species: 1, [100]
Generation: 3, Cost time: 519.87ms
 	fitness: valid cnt: 100, max: -846.6193, min: -1860.7457, mean: -1385.9426, std: 216.9156
	node counts: max: 11, min: 8, mean: 9.25
 	conn counts: max: 20, min: 14, mean: 17.43
 	species: 1, [100]
Generation: 4, Cost time: 485.15ms
 	fitness: valid cnt: 100, max: -721.2616, min: -1741.9164, mean: -1364.9036, std: 196.5529
	node counts: max: 12, min: 7, mean: 8.99
 	conn counts: max: 21, min: 12, mean: 16.89
 	species: 1, [100]
Generation: 5, Cost time: 482.12ms
 	fitness: valid cnt: 100, max: -722.5805, min: -1713.2578, mean: -1376.8560, std: 205.8871
	node counts: max: 11, min: 8, mean: 9.40
 	conn counts: max: 20, min: 14, mean: 17.59
 	species: 1, [100]
Generation: 6, Cost time: 436.80ms
 	fitness: valid cnt: 100, max: -800.5901, min: -1812.3959, mean: -1342.8043, std: 191.2333
	node counts: max: 11, min: 8, mean: 9.24
 	conn counts: max: 20, min: 13, mean: 17.34
 	species: 1, [100]
Generation: 7, Cost time: 483.65ms
 	fitness: valid cnt: 100, max: -811.0939, min: -1860.3704, mean: -1413.2347, std: 230.8492
	node counts: max: 11, min: 8, mean: 9.23
 	conn counts: max: 20, min: 13, mean: 17.10
 	species: 1, [100]
Generation: 8, Cost time: 508.42ms
 	fitness: valid cnt: 100, max: -238.8435, min: -1908.8149, mean: -1323.3953, std: 277.5597
	node counts: max: 11, min: 7, mean: 8.69
 	conn counts: max: 20, min: 12, mean: 16.28
 	species: 1, [100]
Fitness limit reached!
Pendulum repeat 0: return -48.08
	final state: theta=0.1148 rad, theta_dot=-0.0000
Pendulum mean return: -48.08
```
The per-generation `max` column tops out at -238.84 (generation 8), which is the training fitness; `show` builds its own one-episode executable on `state.randkey` and reports -48.08 for the same genome, so the two numbers are different measurements, not a contradiction.
`mountaincar.py`, wall clock 19.21 s (repeat runs: 19.14 s, 19.34 s, 19.64 s), of which ~15.5 s is generations at 1.07–1.73 s each, 0.27 s is the build and ~4 s is interpreter/`uv run` startup:
```
compile finished, cost time: 0.267169s
Generation: 1, Cost time: 1067.44ms
 	fitness: valid cnt: 100, max: -0.0332, min: -99.5781, mean: -43.7771, std: 33.8107
	node counts: max: 4, min: 3, mean: 3.07
 	conn counts: max: 3, min: 1, mean: 1.95
 	species: 1, [100]
Generation: 2, Cost time: 1362.32ms
 	fitness: valid cnt: 100, max: -0.0111, min: -55.9180, mean: -10.2901, std: 14.2163
	node counts: max: 4, min: 3, mean: 3.13
 	conn counts: max: 3, min: 1, mean: 2.00
 	species: 1, [100]
Generation: 3, Cost time: 1390.25ms
 	fitness: valid cnt: 100, max: -0.0008, min: -60.3265, mean: -6.3668, std: 12.3063
	node counts: max: 4, min: 3, mean: 3.28
 	conn counts: max: 4, min: 1, mean: 2.09
 	species: 2, [79, 21]
Generation: 4, Cost time: 1352.82ms
 	fitness: valid cnt: 100, max: -0.0004, min: -55.3304, mean: -3.0875, std: 7.8575
	node counts: max: 5, min: 3, mean: 3.46
 	conn counts: max: 4, min: 1, mean: 2.34
 	species: 2, [39, 61]
Generation: 5, Cost time: 1691.91ms
 	fitness: valid cnt: 100, max: -0.0003, min: -78.0870, mean: -2.8219, std: 9.5096
	node counts: max: 5, min: 3, mean: 3.41
 	conn counts: max: 4, min: 1, mean: 2.26
 	species: 2, [68, 32]
Generation: 6, Cost time: 1667.56ms
 	fitness: valid cnt: 100, max: -0.0001, min: -30.8505, mean: -2.0872, std: 4.5248
	node counts: max: 5, min: 3, mean: 3.57
 	conn counts: max: 4, min: 1, mean: 2.44
 	species: 2, [47, 53]
Generation: 7, Cost time: 1725.92ms
 	fitness: valid cnt: 100, max: -0.0003, min: -46.4085, mean: -3.1506, std: 7.4295
	node counts: max: 5, min: 3, mean: 3.63
 	conn counts: max: 4, min: 1, mean: 2.47
 	species: 2, [51, 49]
Generation: 8, Cost time: 1703.19ms
 	fitness: valid cnt: 100, max: -0.0008, min: -44.2178, mean: -4.7057, std: 10.2910
	node counts: max: 5, min: 3, mean: 3.73
 	conn counts: max: 4, min: 1, mean: 2.54
 	species: 3, [57, 35, 8]
Generation: 9, Cost time: 1669.63ms
 	fitness: valid cnt: 100, max: -0.0003, min: -72.1549, mean: -3.8615, std: 11.7545
	node counts: max: 5, min: 3, mean: 3.77
 	conn counts: max: 4, min: 1, mean: 2.61
 	species: 3, [50, 38, 12]
Generation: 10, Cost time: 1630.07ms
 	fitness: valid cnt: 100, max: -0.0000, min: -70.8499, mean: -4.1536, std: 11.3379
	node counts: max: 5, min: 3, mean: 3.67
 	conn counts: max: 4, min: 0, mean: 2.39
 	species: 3, [46, 41, 13]
Generation limit reached!
MountainCarContinuous repeat 0: return -0.00, goal not reached
	final state: position=-0.5138, velocity=-0.0016
MountainCarContinuous mean return: -0.00
```
The run stops on `Generation limit reached!` because `fitness_target=float("inf")` never triggers; the best mean return is `-0.00` (the idle policy) and `show` reports `goal not reached`, exactly the idle optimum the docstring warns about.
CartPole with the same seed still stops on `Fitness limit reached!` after 2 of 20 generations with `CartPole mean return: 200.00`.

## Constraints
Public API only: `from tensorneat import Pipeline, algorithm, genome, problem` plus `tensorneat.common.{ACT, AGG}`.
Construction-level configuration only; there is no eager compute mode, so a script never touches a traced tensor.
`algorithm.num_inputs`/`num_outputs` come from the genome and must match `problem.input_shape`/`output_shape` (the `Pipeline` asserts it): 4→2 for CartPole, 2→1 for Pendulum and MountainCarContinuous.
Never set `output_transform` on these three problems: each one owns its action mapping (argmax ordering) or squashing (`tanh`).
Host-only introspection (`network_dict`, `repr`, `sympy_func`, `visualize`) never feeds back into a graph, and needs a one-individual (`P == 1`) payload.
`max_in_degree` is a genome static: a recurrent genome accumulates many incoming edges per node, so a recurrent example sizes the window at 16; too small a window aborts the run with "the execution schedule dropped N connection(s)".
No argparse, no helper packages, no unused imports; a script stays ≲ 110 lines and is a single readable top-level run (`cartpole.py` 71, `pendulum.py` 81, `mountaincar.py` 87 lines).
Keep the whole-repo rule that nothing here imports `jax`, `torch`, `brax`, `gymnax` or `mujoco`.

## Test Strategy
Run by hand on the numpy backend; a healthy run prints `compile finished, cost time: ...s`, then a per-generation log, and exits 0 after `pipeline.show(state, best)`.
With `seed=42`: `cartpole.py` prints `CartPole mean return: 200.00` after 2 of 20 generations; `pendulum.py` prints `Pendulum mean return: -48.08` after 8 of 20 generations (5.2 s wall); `mountaincar.py` prints `MountainCarContinuous mean return: -0.00` after 10 of 10 generations (19.2 s wall).
All runs are deterministic per seed: repeated runs reproduce every fitness figure above to the last digit, only the `Cost time` column moves.
The three runs are the measured reference for the budgets; a change to `pop_size`, `max_step`, `generation_limit` or `init_hidden_layers` invalidates the numbers in the docstrings and in this file together.
No pytest target lives here: `test/` owns the automated suites, so these scripts are smoke demos rather than assertions.

## Routing Table
| Script | Demonstrates |
|---|---|
| `cartpole.py` | Discrete `CartPole` rollout, two-logit policy without `output_transform`, `max_step` as the fitness ceiling and cost knob, backend-switch note. |
| `pendulum.py` | Continuous `Pendulum` swing-up, one torque logit squashed inside the problem (`max_torque*tanh`), 6-node hidden layer as the learning enabler, negative return so `fitness_target <= 0`, measured 5.2 s demo. |
| `mountaincar.py` | Continuous `MountainCarContinuous` at the Gym-faithful 999-step default, one action logit, `fitness_target=float("inf")` to force the full budget, and the reward's idle optimum documented instead of a fake solve. |
