# rl — reinforcement-learning problems on evoxc

## Intent
Score a genome by the return of a batched episode rollout, with the environment dynamics written as evoxc ops instead of a JAX simulator.
`CartPole` is the shipped environment and doubles as the reference recipe for any environment a user wants to add: no registry, no simulator dependency, just a step kernel and a `scan`.

## API Surface
- `cartpole.py: CartPole(BaseProblem)` — `__init__(max_step=500, repeat_times=1, gravity=9.8, masscart=1.0, masspole=0.1, length=0.5, force_mag=10.0, dt=0.02, x_threshold=2.4, theta_threshold=12.0, reset_scale=0.05)`.
- `input_shape = (4,)` (`x, x_dot, theta, theta_dot`), `output_shape = (2,)` (two logits; `out[1] > out[0]` pushes right).
- `evaluate(state, keys, act_fn, transformed) -> (P,)`: return in `[0, max_step]`, averaged over `repeat_times` initial states.
- `show(state, key, act_fn, transformed)`: host-side, prints survival length, total return and the final state of one rollout.
- `cartpole_step(obs, force, constants)` is the plain step helper the rollout body uses (Euler integration of the Gym `CartPole-v1` equations).

## Algebra
One `evoxc.scan` over `max_step` timesteps, carry `(obs (B, 4), alive (B,))` with `B = P * repeat_times` (individual-major).
Per step: forward the policy, pick the force, integrate, mark death on `|x| > 2.4 or |theta| > 12deg`, reward `alive` before the transition (so the failing step scores, as in Gym), freeze the state once dead, and emit the rank-1 `alive` mask as the scan output — a rank-3 stacked output would need a `scatter` the stablehlo exporter rejects.
Initial states: one `evoxc.random.uniform(key, (B, 4), low=-0.05, high=0.05)` — no key threading, no `np.random`.

## Constraints
- Every arithmetic constant is an in-trace `enp.full((), value)` (float32); scalar float literals become f64 on the stablehlo path and break iree concat/type unification.
- Use non-negative axes (`axis=1`) in `split`/`concatenate`, keep the alive mask rank-1 and broadcast it with `expand_dims`.
- The state is frozen after death, so an episode's return equals its survival time and `max_step` is a hard fitness ceiling.
- Adding an environment means: a step helper + a `scan` + a `finish` step that reshapes `(P*repeat,)` returns to `(P, repeat)` and averages — keep it in one file per environment.
- Control flow makes this environment numpy / iree-llvm-cpu / xla only; TVM rejects `while`/`if`/gather/scatter and the failure must surface as `evoxc.BackendError`.

## Routing Table
| Path | Responsibility |
|---|---|
| `./cartpole.py` | Batched CartPole dynamics, fixed-horizon rollout, return-based fitness, host-side report. |
