# rl — reinforcement-learning problems on evoxc

## Intent
Score a genome by the return of a batched episode rollout, with the environment dynamics written as evoxc ops instead of a JAX simulator.
`CartPole` is the shipped environment and doubles as the reference recipe for any environment a user wants to add: no registry, no simulator dependency, just a step kernel and one loop.

## API Surface
- `cartpole.py: CartPole(BaseProblem)` — `__init__(max_step=500, repeat_times=1, gravity=9.8, masscart=1.0, masspole=0.1, length=0.5, force_mag=10.0, dt=0.02, x_threshold=2.4, theta_threshold=12.0, reset_scale=0.05, fitness_target=None)`; every knob is validated on the host and raises `TensorNEATError`.
- `fitness_target` is validation-only: `None` skips the check, a value above `max_step` raises because a frozen episode's return can never reach it; the attribute is stored as given.
- `reset_scale = 0.0` is legal and pins every initial state to `(0, 0, 0, 0)`, which is what makes a rollout comparable with the host oracle.
- `input_shape = (4,)` (`x, x_dot, theta, theta_dot`), `output_shape = (2,)` (two logits; `out[1] > out[0]` pushes right).
- `evaluate(state, keys, act_fn, transformed) -> (P,)`: return in `[0, max_step]`, averaged over `repeat_times` initial states.
- `show(state, key, act_fn, transformed)`: host-side, one individual; builds its own graph over the single rank-0 key and prints one survival/return/state line per repeat plus the mean return.
- Trace-side helpers: `_rollout` (reset draw + loop → `(total, final_obs)`), `_loop_cond`, `_loop_body`, `_returns_of`, `_advance` (Euler step + death test in one observation split), `_death_mask` (the shared strict-inequality predicate), `_terminated` (rank-1 façade of it, kept for host-side introspection), `_force_from_logits`, `_nan_rows`, `_constants` (returns the `_CartPoleConstants` NamedTuple: float32 physics scalars plus the int32 loop scalars).
- `_to_numpy` is the host-side accessor `show` uses; `_require_positive_int`, `_require_positive`, `_require_non_negative` are the host-side validators.

## Algebra
One `evoxc.while_loop` over `max_step` timesteps, carry `(obs (B, 4), alive (B,), total (B,), step ())` with `B = P * repeat_times` (individual-major, so `P=4,R=1` and `P=2,R=2` draw the same four initial states).
The return accumulator **rides in the carry** and the loop emits no output: `evoxc.scan` would desugar to a `while` that carries a `(max_step, B)` stacked output and scatters one row per iteration, which is `B * max_step` worth of array traffic per timestep for a result the carry already holds.
Per step: forward the policy, pick the force, add `alive` to `total` **before** the transition (so the failing step scores, as in Gym), integrate, mark death on `|x| > 2.4 or |theta| > 12deg` (`>` strictly, radians in the trace), then freeze the state with `enp.where(enp.expand_dims(alive, 1), next_obs, obs)`.
The alive mask is accumulated (`logical_and` with `~dead`), not recomputed from the frozen state: a dead episode's frozen `x_dot`/`theta_dot` can carry it back inside the thresholds, and an episode must never resume scoring.
`total` is the episode return; `_returns_of` reshapes `(B,) -> (P, R)` and `evaluate` means over axis 1, so a mean is the only reduction after the loop.
Initial states: one `evoxc.random.uniform(key, (B, 4), low=-reset_scale, high=reset_scale)` — no key threading through the loop, no `np.random`; the same key reproduces a run bit-identically on one backend.
NaN policy logits act as push-left inside the rollout (`enp.greater` is false against NaN), but `_nan_rows` (`evoxc.isnan` of the row sum) selects a NaN **into the accumulator**, so any NaN logit at any step makes that episode's return NaN, the mean over repeats keeps it NaN, and `finalize_fitness` maps it to `-inf`; no separate poison leaf and no post-loop poison reduction exist.

## Constraints
- Every arithmetic constant is an in-trace `enp.full((), value)` (float32); scalar float literals become f64 on the stablehlo path and break iree concat/type unification.
- `_constants()` is built **once per rollout, outside the loop**: `enp.full` is a graph Constant op, and the reference interpreter pays per traced op, so a loop-invariant scalar (`4/3`, `-force_mag`, `-reset_scale`, NaN, the int32 `1` and `max_step`) belongs in the bundle, not in the step body.
- The trip count is static (`step < max_step`), never "while any episode is alive": a data-dependent horizon would make the cost depend on the population's behaviour and would let a NaN logit arriving after every episode died escape the poison.
- Use non-negative axes (`axis=1`) in `split`/`concatenate`, and broadcast the rank-1 alive mask with `expand_dims` (a traced tensor has no `__getitem__`).
- The state is frozen after death, so an episode's return equals its survival time and `max_step` is a hard fitness ceiling; every carry leaf keeps its dtype/shape across iterations (`while_loop` typing).
- `_returns_of` is the only place the `(P, R)` layout is assumed; adding an environment means a step helper + a `while_loop` + the same finish step, in one file per environment.
- Control flow makes this environment numpy / iree-llvm-cpu / xla only; TVM rejects `while`/`if`/gather/scatter and the failure must surface as `evoxc.BackendError`.
- `show` is host-side only (`evoxc.build` + `evoxc.run` on the key it is given), never called from a trace and never called with a schedule of more than one individual.
- Physics algebra is fixed op-for-op: precomputing reciprocals (`1/total_mass`, …) reassociates float32 arithmetic and changes which step a chaotic trajectory dies on, so returns would stop matching `test/unit/problem/test_cartpole.py`.

## Measured cost
Rollout component only (`CartPole.evaluate` traced alone, no genome forward, toy policy `(tanh(theta) - x, 2*theta + 3*theta_dot)`), one `evoxc.build` then 7 `evoxc.run`, build excluded, best median of 3 fresh processes; CPython 3.14, evoxc 0.1.0, numpy 2.5.4, iree-base-compiler 3.11 (llvm-cpu), one 20-core box.

| pop | max_step | R | numpy ms | iree ms | ratio |
| --- | --- | --- | --- | --- | --- |
| 100 | 200 | 1 | 69.6 | 8.2 | 8.5× |
| 1000 | 200 | 1 | 79.5 | 29.2 | 2.7× |
| 100 | 500 | 1 | 173.9 | 20.6 | 8.4× |
| 1000 | 500 | 1 | 202.3 | 73.4 | 2.8× |
| 100 | 500 | 4 | 189.4 | 44.9 | 4.2× |
| 1000 | 50 | 1 | 20.6 | 7.3 | 2.8× |

Build is one-off per shape: **0.06 s** on numpy, **0.87–1.24 s** on iree (flat in `pop`/`max_step`/`repeat`).
Cost is linear in the horizon on both backends (200 → 500 steps is 2.5× the cost at either population) and sub-linear in `pop` (10× the population costs ≈1.15× on numpy and ≈3.6× on iree).
No crossover was measured: iree wins every cell, by 2.7–8.5× on the rollout component; the box drifts ±30 % between processes, so the ratios are the load-bearing numbers and single-run medians can move by 2× under contention.

Pipeline level (`/tmp/tnscratch/bench.py --case cartpole … --share`, numpy, `pop_batch_size` unset, one generation per `step`): `step` median, the same `step` with `problem.evaluate` monkeypatched to zeros, and the difference (the evaluate share).

| pop | max_step | R | step ms | no-evaluate ms | evaluate share ms | rollout share of it | build s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 100 | 200 | 1 | 440.9 | 50.1 | 391 | 18 % | 0.52 |
| 1000 | 200 | 1 | 1848.9 | 599.8 | 1249 | 6 % | 1.15 |
| 100 | 500 | 1 | 1016.9 | 49.1 | 968 | 18 % | 0.52 |
| 1000 | 500 | 1 | 3803.9 | 603.1 | 3201 | 6 % | 1.15 |
| 100 | 500 | 4 | 1866.7 | 52.8 | 1814 | 10 % | 0.52 |

The dominant cost of a CartPole generation is therefore **not** this node: the genome `transform` + `forward` inside `act_fn` is 76–94 % of the evaluate share (≈343 ms of 391 ms at `P=100/max_step=200`, ≈2927 ms of 3201 ms at `P=1000/max_step=500`), and `tell`/`speciate` is the rest of the generation at large `P`.
That forward is also the part that cannot compile on iree today (see Known Issues), so the 2.7–8.5× rollout win is a component measurement, not an end-to-end pipeline claim.

Per-timestep traced-op count of the body (traced once, `Builder.create` census, toy policy included): **60 ops**, down from 66 with `evoxc.scan` — one observation split instead of two (the death test reuses the transition's own `x'`/`theta'`), a reduce-sum + `isnan` NaN census instead of two slices + `or` + `reshape`, no `negate`, and the loop-invariant constants moved out of the body (the bundle grew 11 → 17 Constants, paid once per run).
The reference interpreter costs ≈9–10 µs per traced op per iteration at `P = 100–1000`, so op count is the only per-step lever on numpy; `while_loop` also removes the scan's stack machinery, which alone was ≈25 µs/iteration at `P=100` and ≈240 µs/iteration at `P=1000/max_step=500`.
Kernel-change delta on the rollout component (numpy, median): 94.0 → 80.0 ms (`P=100/200`), 118.7 → 92.2 (`P=1000/200`), 246.5 → 200.5 (`P=100/500`), 424.2 → 229.4 (`P=1000/500`), i.e. 15–46 %; the pipeline-level evaluate share moves 4–11 % because the genome forward dominates it.

## Notes for Agents
- Recommended configuration, as measured: on `backend="numpy"` keep `max_step` small (cost is linear in the horizon, ≈0.35 ms per step per 100 individuals) and accept that a CartPole generation is genome-forward-bound, not rollout-bound; on `backend="iree"` this rollout is 2.7–8.5× cheaper with a ≈1 s one-off build per shape, but no end-to-end iree run is possible today, so the shipped default stays `numpy`.
- Cross-backend fitness is **not** bitwise reproducible: numpy and iree round float32 differently, so a chaotic trajectory can die one step apart (measured: identical for saturating/constant policies, 0.1 % of rows differ by ±1 step with the sensitive toy policy). `test/unit` pins the numpy side; there is no cross-backend trajectory contract.
- `evoxc.scan` is **not** export-blocked on iree 3.11 for this kernel — the earlier "rank-3 scatter" note applied to a rank-3 stacked output, and this kernel's output was rank-1. `while_loop` is nonetheless the right form here: it drops the stack, builds faster on iree (1.55 s → ≈1.0 s) and runs slightly faster.
- iree cannot build `Pipeline` at all today, independently of this node: `algorithm/neat/species.py` scatters a rank-3 operand along axis 0 in population init, `genome/gene/base.py` emits a `uint32` constant iree cannot execute, and `common/tensorization.py::derive_schedule`'s `flat_scatter` fails iree-compile legalisation. Those belong to the `algorithm`/`genome`/`common` nodes; until they are fixed there is no end-to-end CartPole iree number and `backend="iree"` on a full pipeline must keep failing loudly.
- `_terminated` is kept rank-1 for host-side and test use; the loop does not call it (it tests `x'`/`theta'` directly through `_death_mask`), so changing its shape breaks `test/unit/problem/test_cartpole.py` without any runtime reason.
- Insertion order in `_loop_body` is load-bearing for the NaN contract: `_nan_rows` selects the NaN into `total` **before** `alive` is updated, so a poison arriving on the step an episode dies still lands.

## Routing Table
| Path | Responsibility |
|---|---|
| `./cartpole.py` | Batched CartPole dynamics, fixed-horizon `while_loop` rollout with a carried return, return-based fitness, NaN-poison accumulator, host-side report. |
