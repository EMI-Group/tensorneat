# ./unit/problem — the problem layer's algebra and protocol suites

## Intent
Prove that each problem under `../../../src/tensorneat/problem/` computes the fitness its docstring promises, on the numpy backend, with no optional dependency and no genome.
A problem is host-side configuration plus plain kernel helpers, so a test needs no `DefaultGenome`, no `transform` and no `forward`: every case drives the real `evaluate` with a *genome-free* `act_fn` (a plain Python closure over `evoxc.numpy` ops, inlined into the graph by the shared runner) and asserts on the returned `(P,)` fitness.
The assertions are closed forms and hand-written numpy references, never the implementation's intermediate state.
Every test that depends on source code that is still a design skeleton carries a real signature and docstring and a body of `pytest.skip("not implemented: <clause> (owner: <module>.<symbol>)")`, so the module is collectable and the skip text names the owner.

## Layout
`test_func_fit.py` — `FuncFit`'s tiling (`row p*S + s` is sample `s` of individual `p`, discriminating against the sample-major layout), the four reductions (`mse`/`rmse`/`mae`/`mape`) against closed forms and `support.oracle.func_fit_loss`, the both-axes reduction with `O = 2`, `mape` with a zero target ⇒ `-inf` fitness, a NaN predictor ⇒ `-inf` fitness, the `XOR`/`XOR3d` dataset literals, `CustomFuncFit`'s grid and seeded-sample datasets, `cartesian_product`, and the configuration errors (`error_method`, box/step/sample validation).
`test_cartpole.py` — `evaluate` against `support.oracle.cartpole_rollout` for constant policies at `max_step` 20 and 50, the analytic push-right survival count (9), death as a strict inequality at `|x| > 2.4` / `|theta| > 12deg` in radians, the frozen-state rule (an episode that dies at `k` scores `k` at any horizon ≥ `k`), `repeat_times`' individual-major `(P, R)` averaging, determinism per key and divergence across keys, the integral `[0, max_step]` return range, NaN logits, the logit→force rule (ties push left) and the constructor's domain validation.
`test_problem_api.py` — per-sample `input_shape`/`output_shape` (and `(S,) + shape` consistency with the dataset), the arity pairing with `algorithm.NEAT.num_inputs`/`num_outputs`, `setup(state)` adding no keys, `static_batch_size` on a fake schedule (leading dimension, disagreeing leaves, no leading dimension), `finalize_fitness` (NaN ⇒ `-inf`, `+inf` untouched), host-side `show`/`show_details`, and the absent legacy `jitable` flag.
`conftest.py` — the lazily resolved inputs (`problem`, `common`, `algorithm`, `payloads`, `oracle`, `runner`, `evoxc`, `enp`), the host `state`, the fake execution schedule and its factory, and the two graph runners (`run_evaluate`, `run_in_graph`).

## Oracles used (never the kernel under test)
`support.oracle.func_fit_loss` (the reduction over the sample **and** output axes) and `support.oracle.cartpole_rollout` + `cartpole_params` (the only CartPole oracle, since the legacy RL problems are gone).
Closed forms where one is cheap: `mse = 0.25`, `rmse = mae = 0.5` for a constant `0.5` predictor on XOR; `mape = 1` for a zero predictor on a strictly positive dataset; the `k = 9` survival count of a constant policy from rest; `finalize_fitness`'s six-entry mapping.
Identities where no absolute value exists: `inputs.shape == (S,) + input_shape`, and the `B = P * repeat_times` identity in `test_repeat_times_averages_individual_major_rows`.

## How the suite runs the code
`evaluate` is a plain helper inlined into the caller's graph, so `run_evaluate` closes over the problem, the `State` and the `act_fn` and passes only tensors: `run_defn(lambda *arrays: problem.evaluate(state, evoxc.random.key(key_seed), act_fn, FakeSchedule(*arrays)), *fields)`.
The schedule is passed **field by field** and rebuilt inside the trace — the shared runner treats only top-level arrays as data, so a namedtuple argument would be baked as a static whose fields are numpy arrays.
The key is rebuilt from an integer `key_seed` inside the trace: a rank-0 `evoxc` key has no `__array__`, and a stateless key is a compile-time constant anyway.
`run_in_graph` is the escape hatch for the in-trace helpers that are not the whole protocol (`finalize_fitness`, `static_batch_size`, `CartPole._terminated`/`_force_from_logits` with `_constants()`).

## The fake schedule
A problem may read exactly one thing from `transformed`: `static_batch_size(transformed)`, the static leading dimension of its leaves.
`conftest.FakeSchedule` is therefore a two-leaf namedtuple (`weights (P, k) float32`, `scores (P,) int32`) with deliberately un-schedule-like field names, so a problem test cannot accidentally depend on the genome's schema and a problem failure stays distinguishable from a genome failure.
Degenerate schedules (leaves that disagree about `P`, rank-0 leaves) are built inline from the `schedule_type` fixture, where the shape under test is visible in the test body.

## Making CartPole comparable to a host oracle
`reset_scale = 0.0` is a legal configuration (only a negative scale is rejected) and makes every initial state exactly `(0, 0, 0, 0)`, so the numpy oracle can be fed the same `obs0` the graph draws; with the default `reset_scale` the draw comes from `evoxc.random` and is not reproducible host-side, so those tests assert relations between runs (determinism, divergence, averaging strictness) instead of absolute values.
`B = P * repeat_times` depends only on the product, so `P = 4, R = 1` and `P = 2, R = 2` draw the same four initial states from the same key — that identity pins the individual-major `(P, R)` layout without any analytic return value.
Both constant policies survive exactly 9 steps from rest (the dynamics are symmetric under `(x, theta, force) -> (-x, -theta, -force)`), so a policy meant to differ from another must differ in more than its sign; the logit/force *sign* is checked on `_force_from_logits` rather than through a return.

## Known doc/source divergences this suite pins or survives
CartPole's fitness is a mean of 0/1 rewards, so it can never be `NaN`: a NaN-logit policy compares `False` (`enp.greater`), acts as "push left", and gets a **finite** fitness — unlike func-fit's "NaN prediction ⇒ `-inf`" rule. `test_nan_logits_fall_back_to_push_left_and_keep_the_fitness_finite` pins that behaviour and must be flipped if `src` adds an explicit NaN guard.
`max_step` is a hard fitness ceiling, so the docstring's "`fitness_target` must be `<= max_step`" is a caller-side rule with no validator in `src` today (`Pipeline` only compares `max(fitnesses)`); the suite asserts the reachable `[0, max_step]` range instead.
`../../../src/tensorneat/problem/rl/CONTEXT.md` names a module-level `cartpole_step(obs, force, constants)` helper that `cartpole.py` does not have (it is `CartPole._euler_step`); tests use the documented methods.

## Conventions
Nothing imports `tensorneat` or `support` at module scope: `import tensorneat` still raises `NotImplementedError` (a `State()` default argument in a design skeleton), so every input is a `scope="module"` fixture resolved through `support.lazy.require`, and a fixture body that calls an unfinished `src` constructor converts that same error into a skip.
Every graph runs through `support.runner` only, through the fixtures in `conftest.py`, never a module-scope import.
`act_fn` closures are pure `enp` (a traced tensor has no `__getitem__` and there is no eager mode), and the fake schedule is the only thing a problem is allowed to read from `transformed`.
Shapes stay small (`pop <= 4`, `max_step <= 50`, `S <= 8`) and the backend is always `"numpy"`; no `jax`/`torch`, no network, no GPU, no display.

## Constraints
A test may not edit `../../../src`, `../../../benchmarks` or `pyproject.toml`; a defect or a doc conflict found here goes into the report to the parent.
Files stay under ~1000 lines, one module per source module area, named `test_<module>.py`.
Suite output must be deterministic — CartPole randomness is driven by explicit integer `key_seed`s and `CustomFuncFit` by explicit `seed`s, never by the global numpy state.
`pytest.mark.slow` is not used here: every case is under 50 rollout steps on populations of at most four individuals, so the whole directory belongs to the default gate `pytest -m "not parity and not slow"`.

## Routing Table
No subdirectories — this node is a leaf.
Read-only inputs: `../../../src/tensorneat/problem/` (code under test), `../../../src/tensorneat/common/` (`State`, `TensorNEATError`), `../../support/` (runner, oracle, payloads).
Sibling suites: `../common`, `../genome`, `../algorithm`, `../../parity`, `../../integration`.
