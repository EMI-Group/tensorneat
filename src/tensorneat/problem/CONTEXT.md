# problem — task / fitness layer

## Intent
Turn a batch-first transformed genome into one scalar fitness per individual, entirely inside an evoxc trace.
Two families: supervised function fitting (`func_fit/`, dataset fixed at construction) and reinforcement learning (`rl/`, a batched CartPole rollout `scan`ned over timesteps).
Nothing here imports a simulator, `jax` or `torch`: problems own the *algebra* (tiling, reduction, dynamics), the backend owns execution.
A problem is pure host-side configuration (`StatefulBaseClass`) plus plain-Python kernel helpers that are inlined into the caller's graph.

## Protocol (`base.py`)
- `evaluate(self, state, keys, act_fn, transformed) -> fitness (P,)` — fitness is **maximized**; the body runs inside an active trace.
- `keys` is **one** evoxc key (rank-0 int64 from `evoxc.random.key(seed)`), never a `(P,)` batch: evoxc has no batched key type and random ops are not vmappable, so a population draws a whole `(P, ...)` block from one key.
- `act_fn(batch, *input_shape) -> (batch, *output_shape)` with population/batch on axis 0; `batch` may be `P * S` (func fit tiles the dataset) or `P * R` (rollout repeats), not necessarily `P`.
- `transformed` is the batch-first execution schedule from `Genome.transform`; the only thing a problem may read from it is `static_batch_size(transformed)` (schema-agnostic: the static leading dim of its leaves), because problems cannot otherwise know `P`.
- `input_shape` / `output_shape` are **per-sample** shapes (`XOR` → `(2,)` / `(1,)`), so `input_shape[-1]` equals `algorithm.num_inputs`; legacy problems returned dataset-shaped tuples — that convention is gone.
- `setup(self, state=None) -> State` is inherited from `StatefulBaseClass` (`common/stateful_class.py:29`: returns `state` unchanged, or a fresh `State()` when `None`); no shipped problem registers keys today, `state` exists for protocol uniformity (and for user problems that need normalization stats).
- `show(self, state, key, act_fn, transformed)` is **host-side** (never called from a trace): it evaluates a batch of one and prints a human-facing report.
- `show_details(self, state, key, act_fn, transformed)` is a host-side no-op hook called by `Pipeline.auto_run` when `show_problem_details=True`.
- `finalize_fitness(fitness)` maps NaN → `-inf` and is applied by every problem, so a network that poisons itself with NaN can never win selection.
- The legacy `jitable` flag is gone: a problem is always traced, and whether a kernel compiles is the backend's business (`evoxc.BackendError` propagates unchanged, never a silent host fallback).

## Function fitting algebra (`func_fit/`)
- Dataset: `inputs (S, I)` and `targets (S, O)` as host numpy arrays; they are genuinely static, so they enter the graph as `evoxc.constant(evoxc.from_numpy(...))` baked at trace time (no `State` threading, no per-call upload).
- A baked dataset travels inside the compiled artifact, so keep `S` modest: beyond ~1 MiB of payload evoxc warns about the constant, i.e. roughly `S * I > 250k` floats.
- Evaluation without vmap-over-samples: `broadcast_to(inputs, (P, S, I))` then `reshape` to `(P*S, I)`; row `p*S + s` belongs to individual `p`, then **one** `act_fn` call over the whole tiled batch, then `reshape` back to `(P, S, O)`.
- Error tensor: `err = pred - expand_dims(targets, 0)` broadcast over the population axis.
- Loss (per individual, reduced over sample *and* output axes, exactly like the legacy `jnp.mean` over all elements): `mse = mean(err**2)`, `rmse = sqrt(mse)`, `mae = mean(|err|)`, `mape = mean(|err / targets|)` — `mape` with a zero target yields inf, which is legacy behaviour and is kept.
- Fitness: `finalize_fitness(-loss)`; NaN predictions (an unconnected/dead topology) give a NaN loss and therefore `-inf`.
- `XOR` (S=4, I=2), `XOR3d` (S=8, I=3) are literals; `CustomFuncFit` samples the box (`method="sample"`, `np.random.default_rng(seed)`) or a cartesian grid (`method="grid"`, `step_size`), and builds targets host-side with `np.apply_along_axis(func, -1, inputs)`.
- `CustomFuncFit` samples from a seeded `Generator`, so a run is reproducible without touching the global `np.random` state (the legacy global-seed coupling is gone).

## CartPole algebra (`rl/cartpole.py`)
- Classic Gym `CartPole-v1` physics with Euler integration: `dt=0.02`, `g=9.8`, `masscart=1.0`, `masspole=0.1`, half-pole `length=0.5`, `force_mag=10`, dead when `|x| > 2.4` or `|theta| > 12deg` (`= 0.2095 rad`), reset `U(-0.05, 0.05)^4`.
- `input_shape = (4,)` = `(x, x_dot, theta, theta_dot)`; `output_shape = (2,)` = two continuous logits, action = "push right" iff `out[1] > out[0]` (a tie pushes left, matching `argmax` index 0) computed with `enp.split` + `enp.greater`, so no integer/bool control flow and no `argmax` dtype handling.
- A genome used with `CartPole` must emit two logits: do **not** set an `argmax`-style `output_transform` (the problem owns the discrete decision).
- One `evoxc.scan` over `max_step=500` timesteps: `xs = enp.zeros((max_step, 1))`, carry `(obs (B, 4), alive (B,), poisoned (B,))`, `B = P * repeat_times`, individual-major.
- Step body: `reward_t = alive` (the failing step still scores, matching Gym returns), `alive = alive & ~( |x'| > 2.4 | |theta'| > 12deg )`, state frozen after death via `enp.where(enp.expand_dims(alive, 1), next_obs, obs)` — no `cond`, no data-dependent trip count, every carried dtype/shape constant across iterations.
- A traced tensor has no `__getitem__`: broadcast with `enp.expand_dims`, split columns with `enp.split`, never `x[:, 0]`.
- Fitness = mean over the `repeat_times` axis of the summed reward (return in `[0, max_step]`), then `finalize_fitness`.
- Randomness: a single `evoxc.random.uniform(key, (B, 4), low=-0.05, high=0.05)`; no key threading through the scan, no `np.random`.
- Because episodes are frozen, an individual's return is its survival time: `max_step` is the fitness ceiling, so `fitness_target` must be `<= max_step`.
- `CartPole(..., fitness_target=None)` validates only when a target is passed (`TensorNEATError` above `max_step`); the documented parameter order is unchanged, so `benchmarks/cases.py` and the pipeline construct it without a target.
- NaN logits act as push-left inside the rollout (`enp.greater` is false against NaN) but poison that individual's fitness to `-inf` through an explicit `evoxc.isnan` guard (`_nan_rows` → `_any_repeat` → select NaN before `finalize_fitness`).

## Constraints
- Compute path uses only `evoxc.numpy as enp` plus top-level `evoxc.{isnan, select, from_numpy, constant}`; there is no eager mode, so never call `np.*` on a traced value.
- `enp` gaps to remember: no `isnan`/`isfinite` (`evoxc.isnan`), no `tile`/`take`/`take_along_axis`/`einsum`/`floor`/`isfinite`; compose from `broadcast_to`, `reshape`, `split`, `concatenate`, `where`, `select`.
- `enp.zeros/ones/full/empty` are **graph** constants (the concrete creators of the same name live at top level `evoxc.*`); create them inside the traced function, never at module scope.
- Scalar float literals become f64 constants on the stablehlo/iree path, so every arithmetic constant in a kernel is built with `enp.full((), v)` to stay float32 and to keep numpy/iree numerically aligned.
- Static-shape hygiene for compiler backends: use non-negative axes in `concatenate`/`split` (`axis=1`, not `axis=-1`) and keep `scan` outputs rank-1 — a rank-3 stacked output needs a rank-3 `scatter` that the v1 stablehlo exporter rejects.
- No `@evoxc.defn` may call another `@evoxc.defn`, and a Python callable is not a static value: `evaluate`/`show` are plain (undecorated) helpers inlined into `Pipeline.step`'s graph, and `act_fn` reaches them by closure.
- No python control flow over traced values: horizon, dataset size and population are static config; the only loop in this node is CartPole's `scan`.
- Errors: constructor/config validation raises `tensorneat.common.errors.TensorNEATError`; shape/dtype/backend errors from evoxc are never caught or re-wrapped.
- Files ≤ ~1000 lines; the tiling + reduction lives once in `func_fit/func_fit.py`, one file per concrete problem.

## Test Strategy
- The shipped `test/unit/problem/` bodies are `pytest.skip` placeholders that name their owner; the semantics they describe are what the kernels implement, and this node is verified against `test/support/oracle.py` (`func_fit_loss`, `cartpole_params`/`cartpole_rollout`) directly.
- `import tensorneat` fails while sibling nodes are skeletons, so exercising this node needs a synthetic parent package (`types.ModuleType("tensorneat")` with `__path__ = ["src/tensorneat"]` in `sys.modules`) and `state=None`.
- `test/unit/problem/` runs on the numpy backend with no optional dependency: `test_func_fit.py` (a dummy `act_fn` such as `lambda x: x[..., :1]` or a constant-output one; assert the all-zero predictor's XOR fitness equals `-0.5`, a NaN predictor gives `-inf`, tiling row order `p*S + s`, and each `error_method`), `test_cartpole.py` (a host numpy reference of the same Euler dynamics; equal returns for `max_step` 20/50, `|theta| >` limit death, frozen state after death, determinism for one key and divergence for another), `test_problem_api.py` (shapes, `TensorNEATError` on bad config, `static_batch_size` on a fake schedule).
- `test/parity/problem/` (needs the `bench` extra) compares the new func-fit kernel against `benchmarks/legacy_jax/tensorneat_legacy/problem/func_fit` for `XOR`, `XOR3d` and `CustomFuncFit(method="grid")` under a fixed random network, float32, `rtol≈1e-5`.
- There is no CartPole parity oracle (the legacy RL problems are deleted), so `test/unit/problem/test_cartpole.py`'s host numpy reference *is* the oracle.
- `test/integration/` owns end-to-end runs (XOR through the Pipeline); a CartPole smoke run uses `max_step=20`, a small population and two generations.

## Known Issues
- `show(state, key, act_fn, transformed)` builds its own one-individual graph, so the `act_fn` it receives must be a predictor over **traced** tensors (`act_fn(batch)` with the schedule bound inside the same trace).
  A `Pipeline.show` that passes a host-bound `functools.partial(algorithm.forward, state, transformed, payload)` therefore fails with `evoxc.TraceError: Concrete Tensor operands are not allowed in graph ops`; the pipeline must build the transform → forward graph itself or hand over a schedule-parametric predictor.
- `FuncFit.setup`/`CartPole.setup` still come from `common.stateful_class.StatefulBaseClass.setup`, which is a skeleton while `common` is implemented (no shipped problem registers keys, so `evaluate` never calls it).

## Notes for Agents
- CartPole cost is interpreter-bound on numpy and horizon-bound on iree: a measured `max_step=500` scan with a toy linear `(B,4)->(B,2)` policy costs ≈180 ms per `evaluate` at `P=100` and ≈315 ms at `P=1000` on the numpy reference backend, versus ≈24 ms at `P=100` and ≈310 ms at `P=1000` with `backend="iree"` (llvm-cpu, build ≈1.3 s per shape); a real genome forward is heavier per step than the toy policy.
- `max_step=50` costs ≈18 ms (numpy, `P=100`) / 2.1 ms (iree); cost is linear in the horizon, so shrinking `max_step` is the cheap lever on numpy.
- Practical guidance for RL runs: `backend="iree"` for anything beyond smoke tests at small-to-moderate populations, and measure before assuming a win at `P ≳ 1000`.
- TVM cannot compile this node's rollouts at all (it rejects `if`/`while`/gather/scatter), so `CartPole` is numpy / iree / xla only — a TVM build must fail loudly, never fall back.

## Routing Table
| Path | Responsibility |
|---|---|
| `./base.py` | `BaseProblem` protocol, `static_batch_size(transformed)`, `finalize_fitness(fitness)`. |
| `./func_fit` | Supervised function fitting: `FuncFit` tiling + loss reduction, `XOR`, `XOR3d`, `CustomFuncFit`. |
| `./rl` | Batched `CartPole` rollout (`scan` over timesteps) and the recipe for user-defined evoxc environments. |
| `../common/errors.py` | `TensorNEATError`, the single config-validation exception (sibling — read-only, escalate writes to parent). |
| `../genome/base.py` | Produces the batch-first `transformed` schedule this node consumes and the `forward` kernel used as `act_fn` (sibling — read-only). |
| `../../test/unit/problem` | Unit suites for this node (sibling — read-only, escalate writes to parent). |
