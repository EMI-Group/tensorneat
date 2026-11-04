# ./unit/problem — the problem layer's algebra and protocol suites

## Intent
Prove that each problem under `../../../src/tensorneat/problem/` computes the fitness its docstring promises, on the numpy backend, with no optional dependency and no genome.
A problem is host-side configuration plus plain kernel helpers, so every case drives the real `evaluate`/`show` (or a documented in-trace helper) with a *genome-free* `act_fn` — a plain Python closure over `evoxc.numpy` ops, inlined into the graph by the shared runner — and asserts on the returned `(P,)` fitness or on the helper's values.
Assertions are closed forms, the independent numpy references of `../support/oracle.py`, or identities; never the implementation's intermediate state.

## Layout
`test_func_fit.py` — 18 cases from 13 tests: the individual-major tiling (`row p*S + s` is sample `s` of individual `p`, discriminated against the sample-major layout by a row-index predictor), the four reductions (`mse`/`rmse`/`mae`/`mape`) against closed forms and `support.oracle.func_fit_loss`, the both-axes reduction with `O = 2`, `mape` with a zero target ⇒ `-inf`, a NaN predictor ⇒ `-inf` (whole population and mixed populations), the `XOR`/`XOR3d` dataset literals re-derived from their definition, `CustomFuncFit`'s grid and seeded-sample datasets, `cartesian_product`, and the configuration errors.
`test_cartpole.py` — 26 cases from 11 tests: `evaluate` against `support.oracle.cartpole_rollout` (host-numpy Euler) for constant policies at `max_step` 20 and 50, the analytic 9-step survival, the logit→force rule (ties push left), death as a strict inequality at `|x| > 2.4` / `|theta| > 12deg` in radians, the frozen-state rule (an episode that dies at `k` scores `k` at every horizon ≥ `k`), `repeat_times`' individual-major `(P, R)` averaging, determinism per key and divergence across keys, the integral `[0, max_step]` return range, NaN-logit poisoning, the logit→force rule, and the constructor's domain validation.
`test_pendulum.py` — 35 cases from 8 tests: `evaluate` against the file's own float32 `_pendulum_rollout` (the batched-in-time dynamics transcribed host-side, bit-identical at batch 1) for idle/push/pull policies and for a state-dependent policy at `max_step` 20/60/200, a float64 Gym-semantics oracle on the same recovered `obs0` (`rtol 1e-5`), the documented `[-max_step*16.2736, 0]` return range, `repeat_times`' individual-major `(P, R)` averaging, key determinism and divergence, NaN-logit poisoning through the `0.001*u**2` penalty, the floor-mod angle wrap (through `_angle_normalize` and through a policy that spins the pole past twelve turns), the velocity clip applied *before* the angle integrates, and every constructor rejection plus in-domain storage.
`test_mountaincar.py` — 36 cases from 12 tests: `evaluate` against the file's float32 `_mountaincar_rollout` (bit-identical at batch 1) for constant policies at 20/60 and a custom track at 100, a float64 Gym-semantics oracle on the same `obs0` (`rtol 1e-5`), the reset drawn inside the valley at rest, `repeat_times`' `(P, R)` layout through a two-family policy, key determinism and divergence, live-NaN poisoning (and the post-goal NaN that the frozen episode masks away), the idle policy's exactly `0.0` return at 20/60/999, the `[-0.1*max_step, 100]` range with its saturated floor, the goal bonus paid exactly once (a bang-bang policy returns the same total at every horizon `>= 140`), the frozen post-goal state via two `_loop_body` calls, the two clips plus the left-wall velocity reset, the goal conjunction, and the constructor's domain validation.
`test_problem_api.py` — 18 cases from 10 tests: per-sample `input_shape`/`output_shape` (and `(S,) + shape` consistency with the dataset), arity pairing with `algorithm.NEAT.num_inputs`/`num_outputs`, `setup(state)` adding no keys, `static_batch_size`'s leading dimension and its four rejections, `finalize_fitness`'s exact six-entry mapping, host-side `show`, the `show_details` no-op, and the absent legacy `jitable` flag.
`conftest.py` — the resolved inputs (`problem`, `common`, `algorithm`, `payloads`, `oracle`, `runner`, `evoxc`, `enp`), the host `state`, the fake execution schedule and its factory, and the two graph runners (`run_evaluate`, `run_in_graph`).
`__init__.py` — namespace marker; the directory must stay an importable package (pytest's prepend import mode names a module by its basename).
133 collected cases, 0 skips, 0 xfail, ~5 s wall on the numpy backend; the whole directory belongs to the default gate `pytest -m "not parity and not slow"`.

## Oracles used (never the kernel under test)
`support.oracle.func_fit_loss` (the reduction over the sample *and* output axes) and `support.oracle.cartpole_rollout` + `cartpole_params` (the only CartPole oracle, since the legacy RL problems are gone).
`test_pendulum.py` and `test_mountaincar.py` carry their own host-numpy rollouts (`_pendulum_rollout`, `_mountaincar_rollout`, plus `_physics`/`_constants`-mirroring kwargs) instead of extending `../support/oracle.py`, which sits at its own size budget as a sibling node; the two transcribe different dynamics, so this is not duplication.
Each of those oracles exists twice in the same file: a float32 transcription that mirrors the kernel op for op (and is therefore bit-identical at batch 1) and a float64 Gym-semantics transcription checked at `rtol 1e-5`, which is what proves the float32 graph is the documented environment and not merely self-consistent.
Closed forms where one is cheap: `mse = 0.25`, `rmse = mae = 0.5` for a constant `0.5` predictor on XOR; `-2.5` for the row-index predictor's first individual; `mape = 1` for a zero predictor on a strictly positive dataset; the `k = 9` survival count of a constant policy from rest; `finalize_fitness`'s `[nan, -1.5, 0, 3, +inf] -> [-inf, -1.5, 0, 3, +inf]`; MountainCar's exactly `0.0` for the idle policy and `100 - 0.1*tanh(logit)**2` for one body step that reaches the goal; Pendulum's `-0.1*T` for a saturated `±max_torque` push.
Identities where no absolute value exists: `inputs.shape == (S,) + input_shape`, and the `B = P * repeat_times` identity in `test_repeat_times_averages_individual_major_rows`.

## How the suite runs the code
`evaluate` is a plain helper inlined into the caller's graph, so `run_evaluate` closes over the problem, the `State` and the `act_fn` and passes only tensors: `run_defn(lambda *arrays: problem.evaluate(state, evoxc.random.key(key_seed), act_fn, FakeSchedule(*arrays)), *fields)`.
The schedule is passed **field by field** and rebuilt inside the trace — the shared runner treats only top-level arrays as data, so a namedtuple argument would be baked as a static whose fields are numpy arrays.
The key is rebuilt from an integer `key_seed` inside the trace: a rank-0 `evoxc` key has no `__array__`, and a stateless key is a compile-time constant anyway.
`run_in_graph` is the escape hatch for the in-trace helpers that are not the whole protocol (`finalize_fitness`, `static_batch_size`, CartPole's `_force_from_logits`/`_terminated` with `_constants()`, Pendulum's `_angle_normalize`/`_reset`, MountainCar's `_advance`/`_terminated`/`_reset` and its `_loop_body` carry).
`static_batch_size` is also correct on a host numpy schedule, which is how its rejection paths are reached without a trace; the error also escapes `evoxc.build` unchanged when raised inside one, and both routes are asserted.
`act_fn` closures are pure `enp` — a traced tensor has no `__getitem__` and there is no eager mode — and the fake schedule is the only thing a problem may read from `transformed`.

## The fake schedule
A problem may read exactly one thing from `transformed`: `static_batch_size(transformed)`, the static leading dimension of its leaves.
`conftest.FakeSchedule` is therefore a two-leaf namedtuple (`weights (P, k) float32`, `scores (P,) int32`) with deliberately un-schedule-like field names, so a problem test cannot accidentally depend on the genome's schema and a problem failure stays distinguishable from a genome failure.
Degenerate schedules (leaves that disagree about `P`, rank-0 leaves, an empty schedule, a non-tensor leaf) are built inline where the shape under test is visible in the test body.

## Making CartPole comparable to a host oracle
`reset_scale = 0.0` is a legal configuration (only a negative scale is rejected) and makes every initial state exactly `(0, 0, 0, 0)`, so the numpy oracle can be fed the same `obs0` the graph draws; with the default `reset_scale` the draw comes from `evoxc.random` and is not reproducible host-side, so those tests assert relations between runs (determinism, divergence, averaging strictness) instead of absolute values.
A constant policy survives exactly 9 steps from rest in *both* directions (the dynamics are symmetric under `(x, theta, force) -> (-x, -theta, -force)`), so the `repeat_times` layout test cannot distinguish rows by direction: `_split_row_policy` uses rows that push constantly (9) and rows that balance the pole for the full horizon (20 at `max_step = 20`).
`B = P * repeat_times` depends only on the product, so `P = 4, R = 1` and `P = 2, R = 2` draw the same four initial states from the same key — that identity pins the individual-major `(P, R)` layout without any analytic return value.

## Recovering `obs0` for Pendulum and MountainCar
Neither environment has a `reset_scale` knob, so there is no configuration that pins its initial state, and the draw still has to come from `evoxc.random`.
`_reset_observations(run_in_graph, evoxc, env, batch, key_seed)` retraces the environment's own `_reset` in a one-op graph with the key rebuilt from the integer `key_seed` inside the trace, and takes the host result — the same `run_evaluate` rebuilds the identical key, so the oracle is fed exactly the states the graph starts from.
That is why the agreement is bit-exact in float32: the oracle runs the documented dynamics on the same numbers, in the same order.
A state-dependent policy needs the oracle to see the state it acted on, so the rollouts return `(total, states)` and the policy is evaluated per step from `states[:, t]`, mirroring what `act_fn` sees inside the loop.
Both environment oracles are written for batch 1 (`(1, 2)` observations, `(1, 1)` logits); `P > 1` cases assert relations (layout, determinism, divergence, bounds) rather than absolute values.

## Behaviour this suite pins (gotchas worth knowing before editing it)
`mape` divides by the target: a non-zero prediction on a zero target is `+inf` loss, while a *zero* prediction on a zero target is `0/0 = NaN`; both reach `-inf` fitness through different arithmetic, and each is asserted separately.
CartPole demotes NaN logits to `-inf` fitness through the explicit `evoxc.isnan` guard (`_nan_rows` → `_any_repeat` → select NaN before `finalize_fitness`), while the decision site still reads NaN as "push left"; one poisoned repeat demotes the whole individual.
CartPole's fitness is a mean of 0/1 rewards, so it is always finite outside the NaN case and lies in `[0, max_step]`; `max_step` is a hard ceiling and `fitness_target > max_step` is rejected at construction.
Pendulum's per-step cost is `angle_normalize(theta)**2 + 0.1*theta_dot**2 + 0.001*u**2` on the **pre-transition** state with `u = max_torque*tanh(logit)`, so the return of *any* policy lies in `[-max_step*(pi**2 + 0.1*max_speed**2 + 0.001*max_torque**2), 0]` — `-16.2736*max_step` at the defaults — and the environment never terminates.
Pendulum's wrap is `x - 2*pi*floor((x+pi)/(2*pi))` (a floor-mod into `[-pi, pi)`), not `evoxc.remainder`, which is fmod on stablehlo/iree and keeps the sign of the dividend; a policy that spins the pole past twelve turns still scores the *wrapped* angle, and its unwrapped cost is ~226x larger on the rows that spin (~12x on the populations tested here), which is how the test detects a regression to fmod.
Pendulum clips the velocity **before** the angle integrates, so `theta' = theta + clip(theta_dot + (...)*dt, +/-max_speed)*dt`; a spin started at `theta_dot = 7.9` advances the angle by exactly `8.0*dt`, not by the unclipped `8.19*dt`.
MountainCar pays the goal bonus exactly once: `done` is written into the loop's `alive` flag, the accumulator is masked by the **pre-step** alive value, and the state is frozen afterwards, so a policy that reaches the goal at step `k` returns the same total at every horizon `>= k` (the frozen state still satisfies `done`, so an unmasked bonus would keep growing).
MountainCar's reward is `(100 if done else 0) - 0.1*tanh(logit)**2`, so the idle policy (`logit = 0`) returns exactly `0.0` at every horizon — the deceptive optimum, and the reason the shipped example does not solve the task; the return of any policy lies in `[-0.1*max_step, 100]`.
MountainCar's penalty is charged on the *squashed* action `a = tanh(logit)`, and `clip(a, -1, 1)` is unobservable because `|tanh| <= 1` (it saturates to exactly `1.0` in float32 at `logit = 30`); the tests pin `a`, not the redundant clip.
Only the left wall zeroes the velocity (`position == min_position and velocity < 0`); the `0.6` clamp keeps the velocity, and `done` is the conjunction `position' >= goal_position and velocity' >= goal_velocity`, so reaching `0.45` while rolling backwards is not done.
MountainCar's `-0.1*max_step` floor is only approached, not attained, in float32 (`-99.89905` at `max_step = 999` against a `-99.9` bound), so the bound is asserted as an inequality and the exact value is left to the oracle.
Pendulum and MountainCar have no `reset_scale` knob and no NaN census: NaN logits reach `-inf` by propagating through the action penalty (`0.001*u**2` / `0.1*a**2`), and a NaN that only appears after the goal is masked away with the rest of the frozen episode.
`CustomFuncFit(method="grid")` builds the per-axis `np.arange(low, high, step)` and takes their row-major cartesian product with the first axis slowest, so `S` is the product of the grid lengths and is unrelated to `num_samples`.
`CustomFuncFit` validates `func(low_bounds)` and the box/method/step configuration before it reaches `FuncFit.__init__`, so a bad box wins over a bad `error_method`.
`cartesian_product` is not re-exported by `tensorneat.problem` or `tensorneat.problem.func_fit`; it is reached as `problem.func_fit.custom.cartesian_product`.
`algorithm.NEAT` rejects `pop_size < species_size * min_species_size` (default `10 * 1`), so the arity cross-check builds it at `pop_size = 10`; a smaller population raises `TensorNEATError` before the arity can be read.
`setup(state)` returns the same `State` and registers no keys; `setup(None)` starts from the same empty key set.

## Conventions
Nothing imports `tensorneat`, `support` or `evoxc` at module scope: `../conftest.py` prepends `src/`, `benchmarks/legacy_jax` and `test/` to `sys.path` at collection time, so every input is a `scope="module"` fixture that imports inside its body.
Those imports are plain statements — the suite must never convert a broken or regressed module into a skip, so a green run here really executed every kernel it claims to.
Every graph runs through `support.runner` only, through the fixtures in `conftest.py`, never a module-scope import.
Shapes stay small (`pop <= 8`, `S <= 8`; horizons `max_step <= 50` for CartPole/func-fit, `<= 200` for Pendulum, and `999` only for MountainCar at population 1) and the backend is always `"numpy"`; no `jax`/`torch`, no network, no GPU, no display.

## Constraints
A test may not edit `../../../src`, `../../../benchmarks` or `pyproject.toml`; a defect or a doc conflict found here goes into the report to the parent.
A src defect is never hidden: keep the real assertion and mark the test `@pytest.mark.xfail(strict=True, reason="src defect: …")`. No such xfail exists today.
Files stay under ~1000 lines: `test_func_fit.py` 811, `test_cartpole.py` 786, `test_pendulum.py` 599, `test_mountaincar.py` 599, `test_problem_api.py` 482, `conftest.py` 335.
Suite output must be deterministic — CartPole randomness is driven by explicit integer `key_seed`s and `CustomFuncFit` by explicit `seed`s, never by the global numpy state; Pendulum and MountainCar likewise only ever pass an integer `key_seed` (never a literal drawn state) so a re-run reproduces every number.
`pytest.mark.slow` and `pytest.mark.parity` are not used here: every case is a batched `while_loop` over at most a few hundred float32 steps on populations of at most eight individuals (~5 s for the whole directory).

## Routing Table
No subdirectories — this node is a leaf.
Read-only inputs: `../../../src/tensorneat/problem/` (code under test), `../../../src/tensorneat/common/` (`State`, `TensorNEATError`), `../../../src/tensorneat/algorithm/` (the arity cross-check), `../support/` (runner, oracle, payloads).
Sibling suites: `../common`, `../genome`, `../algorithm`, `../../parity`, `../../integration`.
