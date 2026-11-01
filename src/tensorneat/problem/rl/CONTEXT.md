# rl — reinforcement-learning problems on evoxc

## Intent
Score a genome by the return of a batched episode rollout, with the environment dynamics written as evoxc ops instead of a JAX simulator.
`CartPole` is the shipped environment and doubles as the reference recipe for any environment a user wants to add: no registry, no simulator dependency, just a step kernel and a `scan`.

## API Surface
- `cartpole.py: CartPole(BaseProblem)` — `__init__(max_step=500, repeat_times=1, gravity=9.8, masscart=1.0, masspole=0.1, length=0.5, force_mag=10.0, dt=0.02, x_threshold=2.4, theta_threshold=12.0, reset_scale=0.05, fitness_target=None)`; every knob is validated on the host and raises `TensorNEATError`.
- `fitness_target` is validation-only: `None` skips the check, a value above `max_step` raises because a frozen episode's return can never reach it; the attribute is stored as given.
- `reset_scale = 0.0` is legal and pins every initial state to `(0, 0, 0, 0)`, which is what makes a rollout comparable with the host oracle.
- `input_shape = (4,)` (`x, x_dot, theta, theta_dot`), `output_shape = (2,)` (two logits; `out[1] > out[0]` pushes right).
- `evaluate(state, keys, act_fn, transformed) -> (P,)`: return in `[0, max_step]`, averaged over `repeat_times` initial states.
- `show(state, key, act_fn, transformed)`: host-side, one individual; builds its own graph over the single rank-0 key and prints one survival/return/state line per repeat plus the mean return.
- In-trace helpers: `_rollout` (reset draw + scan → `returns`, final `obs`, poison flags), `_scan_body`, `_euler_step`, `_terminated`, `_force_from_logits`, `_nan_rows`, `_any_repeat`, `_finish`, `_constants` (returns the `_CartPoleConstants` NamedTuple of float32 graph scalars); `_to_numpy` is the host-side accessor `show` uses.
- Host-side validators: `_require_positive_int`, `_require_positive`, `_require_non_negative`.

## Algebra
One `evoxc.scan` over `max_step` timesteps, carry `(obs (B, 4), alive (B,), poisoned (B,))` with `B = P * repeat_times` (individual-major, so `P=4,R=1` and `P=2,R=2` draw the same four initial states).
Per step: forward the policy, pick the force, reward `alive` **before** the transition (so the failing step scores, as in Gym), integrate, mark death on `|x| > 2.4 or |theta| > 12deg` (`>` strictly, radians in the trace), then freeze the state with `enp.where(enp.expand_dims(alive, 1), next_obs, obs)`, and emit the rank-1 `alive` mask as the scan output — a rank-3 stacked output would need a `scatter` the stablehlo exporter rejects.
`returns = enp.sum(rewards, axis=0)` is an integer in `[0, max_step]`; `_finish` reshapes `(P*R,) -> (P, R)` and means over axis 1.
Initial states: one `evoxc.random.uniform(key, (B, 4), low=-reset_scale, high=reset_scale)` — no key threading, no `np.random`; the same key reproduces a run bit-identically.
NaN policy logits act as push-left inside the rollout (`enp.greater` is false against NaN), but `_nan_rows` accumulates a per-episode poison flag in the carry, `_any_repeat` folds it per individual, and `evaluate` selects NaN before `finalize_fitness` — so any NaN logit poisons that individual to `-inf` (the explicit `evoxc.isnan` guard; the opposite docstring in `test/unit/problem/test_cartpole.py::test_nan_logits_fall_back_to_push_left_and_keep_the_fitness_finite` is a known doc conflict).

## Constraints
- Every arithmetic constant is an in-trace `enp.full((), value)` (float32); scalar float literals become f64 on the stablehlo path and break iree concat/type unification.
- Use non-negative axes (`axis=1`) in `split`/`concatenate`, keep the alive mask rank-1 and broadcast it with `expand_dims`.
- The state is frozen after death, so an episode's return equals its survival time and `max_step` is a hard fitness ceiling; every carry leaf keeps its dtype/shape across iterations (`while_loop` typing).
- `_finish`/`_any_repeat` are the only place the `(P, R)` layout is assumed; adding an environment means a step helper + a `scan` + the same finish step, in one file per environment.
- Control flow makes this environment numpy / iree-llvm-cpu / xla only; TVM rejects `while`/`if`/gather/scatter and the failure must surface as `evoxc.BackendError`.
- `show` is host-side only (`evoxc.build` + `evoxc.run` on the key it is given), never called from a trace and never called with a schedule of more than one individual.

## Measured cost (this machine, evoxc 0.1.0, CPython 3.14, 20 cores)
Setup: toy linear policy `(B, 4) -> (B, 2)` (one `enp.matmul` against a constant), `max_step = 500`, one `evoxc.build` then 5 `evoxc.run` calls, median reported (variation across repeats is ±5 %).
- numpy reference backend: **≈180 ms** per `evaluate` at `P = 100`, **≈315 ms** at `P = 1000` (build ≈85 ms / ≈70 ms).
- iree (llvm-cpu, `evoxc[iree]` 3.11): **≈24 ms** at `P = 100`, **≈310 ms** at `P = 1000`; build ≈1.3 s.
- The iree advantage shrinks with the horizon at large batch: at `max_step = 50` iree is 2.1 ms vs numpy 17.9 ms (`P=100`) and 7.2 ms vs 19.8 ms (`P=1000`); at `max_step = 100`, 4.1 vs 33.2 ms and 15.2 vs 41.2 ms; at `max_step = 500` the two backends tie at `P = 1000`.
Cost is interpreter-bound on numpy and horizon-bound on iree, so the pre-rewrite note ("≈165 ms numpy vs ≈30 ms iree at `P=100`") still holds in shape — numpy measures at ≈180 ms here, iree at ≈24 ms.
Practical guidance: use `backend="iree"` for anything beyond smoke tests at small-to-moderate populations (≈10x at `P=100`), measure before assuming a win at `P ≳ 1000` where iree currently ties numpy at `max_step=500`, and shrink `max_step` on numpy (cost is linear in the horizon: ≈18 ms for 50 steps at `P=100`).
`build` cost is paid once per `(P, max_nodes, max_conns, max_in_degree, key_span)` signature, so it disappears in a long run but dominates a single-shot `evoxc.evaluate` of this kernel (≈1.3 s on iree).

## Routing Table
| Path | Responsibility |
|---|---|
| `./cartpole.py` | Batched CartPole dynamics, fixed-horizon rollout, return-based fitness, NaN-poison guard, host-side report. |
