# problem — task / fitness layer

## Intent
Turn a batch-first transformed genome into one scalar fitness per individual, entirely inside an evoxc trace.
Two families: supervised function fitting (`func_fit/`, dataset fixed at construction) and reinforcement learning (`rl/`, a batched CartPole rollout `while_loop`ed over timesteps).
Nothing here imports a simulator, `jax` or `torch`: problems own the *algebra* (tiling, reduction, dynamics), the backend owns execution.
A problem is pure host-side configuration (`StatefulBaseClass`) plus plain-Python kernel helpers that are inlined into the caller's graph.

## Protocol (`base.py`)
- `evaluate(self, state, keys, act_fn, transformed) -> fitness (P,)` — fitness is **maximized**; the body runs inside an active trace.
- `keys` is **one** evoxc key (rank-0 int64 from `evoxc.random.key(seed)`), never a `(P,)` batch: evoxc has no batched key type and random ops are not vmappable, so a population draws a whole `(P, ...)` block from one key.
- `act_fn(batch, *input_shape) -> (batch, *output_shape)` with population/batch on axis 0; `batch` may be `P * S` (func fit tiles the dataset) or `P * R` (rollout repeats), not necessarily `P`.
- `transformed` is the batch-first execution schedule from `Genome.transform`; the only thing a problem may read from it is `static_batch_size(transformed)` (schema-agnostic: the static leading dim of its leaves), because problems cannot otherwise know `P`.
- `input_shape` / `output_shape` are **per-sample** shapes (`XOR` → `(2,)` / `(1,)`), so `input_shape[-1]` equals `algorithm.num_inputs`; legacy problems returned dataset-shaped tuples — that convention is gone.
- `setup(self, state=None) -> State` comes from `StatefulBaseClass`; `FuncFit` overrides it to validate its dataset budget first, `CartPole` inherits it unchanged (no shipped problem registers keys today).
- `show(self, state, key, act_fn, transformed)` is **host-side** (never called from a trace): it evaluates a batch of one and prints a human-facing report.
- The schedule container is opaque to a problem, so `show` rebuilds it *inside* its one-off trace with `evoxc.tree_unflatten(evoxc.tree_structure(transformed), list(arrays))` — the leaves travel as graph data arguments and the structure comes from the pytree, never from a positional `type(transformed)(*leaves)` (that broke on HyperNEAT's nested `SubstrateSchedule`: 17 leaves against 6 fields).
  A flat `Transformed` namedtuple and a nested batch-first container therefore both round-trip, at any static `P`.
- `show_details(self, state, key, act_fn, transformed)` is a host-side no-op hook called by `Pipeline.auto_run` when `show_problem_details=True`.
- `finalize_fitness(fitness)` maps NaN → `-inf` (3 ops: `evoxc.isnan`, a rank-0 `-inf` constant, `evoxc.select`) and is applied by every problem, so a network that poisons itself with NaN can never win selection.
- The legacy `jitable` flag is gone: a problem is always traced, and whether a kernel compiles is the backend's business (`evoxc.BackendError` propagates unchanged, never a silent host fallback).

## Function fitting algebra (`func_fit/`)
- Dataset: `inputs (S, I)` and `targets (S, O)` as host numpy arrays; they are genuinely static, so they enter the graph as `evoxc.constant(evoxc.from_numpy(...))` baked at trace time (no `State` threading, no per-call upload).
- Those are the **only** constants this node bakes, always `(S, I)` and `(S, O)` — never pre-tiled, never scaled with the population (XOR: 48 B, XOR3d: 128 B, Pagie 50 samples: 600 B).
- A baked dataset travels inside the compiled artifact, so its footprint is budgeted on the host: `FuncFit.MAX_DATASET_NBYTES = 1 << 20` (evoxc's `EVOXC_LARGE_CONSTANT_BYTES`), enforced by `validate_dataset()` at construction, in `setup()` and again at bake time; `dataset_nbytes` reports the exact figure.
  Exceeding it raises `TensorNEATError` (a configuration mistake, not a slow-but-valid run); a subclass may raise the attribute deliberately.
- **Amplification rule:** the problem hands the network a `(P * S, I)` batch, so every per-row allocation on the `act_fn` path costs `S`× more than at `P` rows — and `evoxc`'s `enp.zeros/ones/full/empty` materialise *full Constant ops*.
  `Genome.forward`'s `(P*S, max_nodes − num_inputs)` NaN seed is 1.92 MB at P=200/S=50 and grows linearly in both factors; that is a genome-side cost, and it is why the dataset must never be pre-tiled.
- Evaluation without vmap-over-samples: `tiled = evoxc.tile(inputs, (P, 1))` (numpy `tile` semantics, so row `p*S + s` is sample `s` of individual `p`) — one op, then **one** `act_fn` call over the whole tiled batch, then `reshape` back to `(P, S, O)`.
- Error tensor: `err = pred - targets`, the `(S, O)` targets broadcasting numpy-style over the population axis (no `expand_dims`).
- Loss (per individual, reduced over sample *and* output axes, exactly like the legacy `jnp.mean` over all elements): `mse = mean(err**2)`, `rmse = sqrt(mse)`, `mae = mean(|err|)`, `mape = mean(|err / targets|)` — `mape` with a zero target yields inf, which is legacy behaviour and is kept.
- Fitness: `finalize_fitness(-loss)`; NaN predictions (an unconnected/dead topology) give a NaN loss and therefore `-inf`.
- `XOR` (S=4, I=2), `XOR3d` (S=8, I=3) are literals; `CustomFuncFit` samples the box (`method="sample"`, `np.random.default_rng(seed)`) or a cartesian grid (`method="grid"`, `step_size`), and builds targets host-side with `np.apply_along_axis(func, -1, inputs)`.
- `CustomFuncFit` samples from a seeded `Generator`, so a run is reproducible without touching the global `np.random` state (the legacy global-seed coupling is gone).
- `evaluate` traces 14 IR ops in total; tiling + loss is ≈0.15 ms at pop 500 (XOR), i.e. a negligible share of the generation — the evaluate cost is `genome.forward` on the `P*S` batch.

## CartPole algebra (`rl/cartpole.py`)
- Classic Gym `CartPole-v1` physics with Euler integration: `dt=0.02`, `g=9.8`, `masscart=1.0`, `masspole=0.1`, half-pole `length=0.5`, `force_mag=10`, dead when `|x| > 2.4` or `|theta| > 12deg` (`= 0.2095 rad`), reset `U(-0.05, 0.05)^4`.
- `input_shape = (4,)` = `(x, x_dot, theta, theta_dot)`; `output_shape = (2,)` = two continuous logits, action = "push right" iff `out[1] > out[0]` (a tie pushes left, matching `argmax` index 0) computed with `enp.split` + `enp.greater`, so no integer/bool control flow and no `argmax` dtype handling.
- A genome used with `CartPole` must emit two logits: do **not** set an `argmax`-style `output_transform` (the problem owns the discrete decision).
- One `evoxc.while_loop` over `max_step` timesteps with carry `(obs (B,4), alive (B,), total (B,), step ())`, `B = P * repeat_times`, individual-major.
  The return accumulates in the carry: a `scan` would stack a `(max_step, B)` output and scatter one row per iteration, i.e. `B*max_step` traffic for a value the carry already holds.
- Step body: `total += alive` (the failing step still scores, matching Gym returns), `alive = alive & ~( |x'| > 2.4 | |theta'| > 12deg )` on the transition's own `x'`/`theta'`, state frozen after death via `enp.where(enp.expand_dims(alive, 1), next_obs, obs)` — no `cond`, no data-dependent trip count, every carried dtype/shape constant across iterations.
- Loop-invariant constants live in the `_CartPoleConstants` bundle, built once per rollout outside the loop (`four_thirds`, `neg_force_mag`, `neg_reset_scale`, `nan`, int32 `one_step`/`horizon`), so the interpreter pays one Constant per run instead of one per timestep; the body is 60 traced ops.
- Physics is fixed op-for-op: precomputing reciprocals or reassociating the float32 arithmetic changes which step a chaotic trajectory dies on, so returns would stop matching the host-numpy oracle in `test/unit/problem/test_cartpole.py`.
- A traced tensor has no `__getitem__`: broadcast with `enp.expand_dims`, split columns with `enp.split`, never `x[:, 0]`.
- `_returns_of` reshapes `(B,)` to `(P, R)` and means over axis 1 → fitness in `[0, max_step]`, then `finalize_fitness`.
- Randomness: a single `evoxc.random.uniform(key, (B, 4), low=-reset_scale, high=reset_scale)`; no key threading through the loop, no `np.random`.
- Because episodes are frozen, an individual's return is its survival time: `max_step` is the fitness ceiling, so `fitness_target` must be `<= max_step`.
- `CartPole(..., fitness_target=None)` validates only when a target is passed (`TensorNEATError` above `max_step`); the documented parameter order is unchanged, so `benchmarks/cases.py` and the pipeline construct it without a target.
- NaN logits act as push-left inside the rollout (`enp.greater` is false against NaN) but a NaN census (`evoxc.isnan(enp.reduce_sum(logits, axis=1))`) is accumulated into `total`, so that individual's fitness is NaN and `finalize_fitness` maps it to `-inf`.

## Measured Cost (this box, evoxc 0.1.0, numpy reference backend unless stated)
Function fitting (`XOR`/`XOR3d`, `DefaultGenome(max_nodes=50, max_conns=100)`, `NEAT(pop_size=500, species_size=20)`): a full `step` is ≈350 ms, of which the evaluate segment (transform + schedule repeat + `genome.forward` at `P*S` rows + tiling + loss) is only ≈15–30 ms; `tell`/speciation dominates.
`evaluate` alone with a trivial `act_fn`: 0.15 ms (XOR pop 500), 0.20 ms (XOR3d pop 500), 0.26 ms (CustomFuncFit pop 200 / 50 samples).
CartPole rollout component (`evaluate` traced alone, no genome forward): numpy 69.6 / 79.5 ms at `max_step=200` (P=100 / 1000) and 173.9 / 202.3 ms at `max_step=500`; iree-llvm-cpu 8.2 / 29.2 / 20.6 / 73.4 ms for the same cells, i.e. 2.7–8.5× cheaper, with a 0.87–1.24 s one-off build per shape versus 0.06 s on numpy.
Cost is linear in `max_step` on both backends and sub-linear in `pop`; no crossover was measured. The authoritative table and the recommended configuration live in `./rl/CONTEXT.md`.
At pipeline level the rollout is only 6–18 % of the CartPole evaluate share (the genome forward is the rest), and `Pipeline` cannot build on iree today for reasons owned by `algorithm`/`genome`/`common`, so the shipped default stays `numpy`.

## Constraints
- Compute path uses only `evoxc.numpy as enp` plus top-level `evoxc.{isnan, select, tile, from_numpy, constant}`; there is no eager mode, so never call `np.*` on a traced value.
- `enp` gaps to remember: no `isnan`/`isfinite` (`evoxc.isnan`), no `tile`/`take`/`take_along_axis`/`einsum`/`floor`/`isfinite` (tiling is `evoxc.tile`, the rest compose from `broadcast_to`, `reshape`, `split`, `concatenate`, `where`, `select`).
- `enp.zeros/ones/full/empty` are **graph** constants that materialise their whole array (the concrete creators of the same name live at top level `evoxc.*`); create them inside the traced function, never at module scope, and never with a shape that scales with the population or the horizon.
- Scalar float literals become f64 constants on the stablehlo/iree path, so every arithmetic constant in a kernel is built with `enp.full((), v)` to stay float32 and to keep numpy/iree numerically aligned.
- Static-shape hygiene for compiler backends: use non-negative axes in `concatenate`/`split` (`axis=1`, not `axis=-1`) and keep loop outputs rank-1 — a rank-3 stacked output needs a rank-3 `scatter` that the v1 stablehlo exporter rejects.
- No `@evoxc.defn` may call another `@evoxc.defn`, and a Python callable is not a static value: `evaluate`/`show` are plain (undecorated) helpers inlined into `Pipeline.step`'s graph, and `act_fn` reaches them by closure.
- `show`'s `act_fn` must be a predictor over **traced** tensors (the schedule it closes over is rebuilt in the same trace); `Pipeline._constant_act_fn` satisfies that, a host-bound `functools.partial(algorithm.forward, state, transformed, payload)` raises `evoxc.TraceError`.
- No python control flow over traced values: horizon, dataset size and population are static config; the only loop in this node is CartPole's `while_loop`.
- Errors: constructor/config validation raises `tensorneat.common.errors.TensorNEATError`; shape/dtype/backend errors from evoxc are never caught or re-wrapped.
- Files ≤ ~1000 lines; the tiling + reduction lives once in `func_fit/func_fit.py`, one file per concrete problem.

## Test Strategy
- `test/unit/problem/` (62 tests, numpy backend, no optional dependency) is the node's real suite: `test_func_fit.py` pins the tiling order (`p*S + s`), every `error_method` against a numpy oracle, the NaN → `-inf` mapping and the XOR/XOR3d literals; `test_cartpole.py` holds the host-numpy dynamics oracle and pins death at `|theta| > 12 deg`, frozen state after death, the fitness ceiling and per-key determinism; `test_problem_api.py` pins shapes, `static_batch_size` on a fake schedule and the `TensorNEATError` config paths.
- `test/parity/` (needs the `bench` extra) diffs the func-fit fitness of a fixed random network against `benchmarks/legacy_jax/tensorneat_legacy/problem/func_fit` for `XOR`, `XOR3d` and `CustomFuncFit(method="grid")`, float32, `rtol≈1e-5`; there is no CartPole parity oracle (the legacy RL problems were deleted), so the unit-test oracle *is* the reference.
- Verification beyond the suites: driving `Pipeline` end-to-end on the numpy backend (`setup`/`step`/`auto_run`/`show`) and the four examples (`examples/func_fit/{xor,xor3d,custom_func_fit}.py`, `examples/rl/cartpole.py`), plus `test/release_gate_no_jax.py`.
- There is no test for the dataset budget (`MAX_DATASET_NBYTES` / `validate_dataset`) or for the `evaluate` op count today; both are verified by hand.

## Known Issues
- `Pipeline.show` on **HyperNEAT** still fails before it ever reaches this node: `HyperNEAT._query_substrate` takes `pop_size = self.pop_size` instead of the payload's leading dimension, so the pipeline's one-individual schedule derivation raises `evoxc.core.errors.ShapeError: concatenate: dims must agree, got P*Q vs 1`.
  `problem.FuncFit.show` itself handles the nested `SubstrateSchedule` (verified at `P=1` and `P>1`); the algorithm node has to derive `P` from the payload it is handed.
- `CartPole` cannot build on TVM at all (it rejects `if`/`while`/gather/scatter), and a full `Pipeline` cannot build on iree today: `algorithm/neat/species.py` scatters a rank-3 operand along axis 0, `genome/gene/base.py` emits a `uint32` constant, and `common/tensorization.py::derive_schedule`'s `flat_scatter` fails iree-compile legalisation.
  Both must keep failing loudly, never fall back.
- Cross-backend CartPole fitness is not bitwise reproducible (numpy and iree round float32 differently, so a sensitive policy can differ by ±1 step); no test may assert cross-backend trajectory equality.

## Notes for Agents
- Cross-node cost findings (not editable from here):
  - `genome/default.py` bakes `enp.full((P*S, max_nodes − num_inputs), nan)` as the forward value seed — the largest constant in every func-fit graph (1.92 MB at P=200/50 samples, 20 MB at S=500, 80 MB-scale at larger geometry). A `(1, width)` constant plus a broadcast would make it population-independent.
  - `common/tensorization.py` bakes 104 population-scaled constants (3.43 MB at P=200, 8.56 MB at P=500, e.g. `(P, max_conns+1)` twice per window table).
  - `problem/base.py::finalize_fitness` costs 3 ops where `evoxc.nan_to_num(x, nan=-inf)` would cost 1; not taken (inf semantics differ from numpy's).
- On numpy, keep `max_step` small for CartPole (cost is linear in the horizon, ≈0.35 ms per step per 100 individuals) and expect a generation to be genome-forward-bound, not rollout-bound; on iree the rollout is 2.7–8.5× cheaper but no end-to-end iree run is possible today.
- Re-measure with the dev harness pattern (`problem.evaluate` traced alone with a trivial `act_fn` for this node's own cost; a `step` with `evaluate` monkeypatched to zeros for the evaluate share), never with a single-shot pipeline run: build (0.5–1.2 s) dominates one generation at small budgets.

## Routing Table
| Path | Responsibility |
|---|---|
| `./base.py` | `BaseProblem` protocol, `static_batch_size(transformed)`, `finalize_fitness(fitness)`. |
| `./func_fit` | Supervised function fitting: `FuncFit` tiling + loss reduction + dataset budget, `XOR`, `XOR3d`, `CustomFuncFit`. |
| `./rl` | Batched `CartPole` rollout (`while_loop` over timesteps), measured cost table, and the recipe for user-defined evoxc environments. |
| `../common/errors.py` | `TensorNEATError`, the single config-validation exception (sibling — read-only, escalate writes to parent). |
| `../genome/base.py` | Produces the batch-first `transformed` schedule this node consumes and the `forward` kernel used as `act_fn` (sibling — read-only). |
| `../../test/unit/problem` | Unit suites for this node (sibling — read-only, escalate writes to parent). |
