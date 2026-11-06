# func_fit — supervised function-fitting problems

## Intent
Fit a genome to a fixed dataset: each individual scores `-loss` of its predictions over every sample.
`FuncFit` owns the only non-trivial part — how a small dataset becomes a batched evaluation — and each concrete problem is just a dataset plus two shapes.
The dataset is baked into the compiled artifact, so its size is paid on every build and is budgeted on the host (`MAX_DATASET_NBYTES`).

## API Surface
- `func_fit.py: FuncFit(BaseProblem)` — `__init__(error_method: str = "mse")` validates `error_method in ERROR_METHODS == ("mse", "rmse", "mae", "mape")` (else `TensorNEATError`) and then checks the dataset budget if the dataset already exists.
- Subclass obligations: `inputs -> np.ndarray (S, I)`, `targets -> np.ndarray (S, O)`, `input_shape -> (I,)`, `output_shape -> (O,)`; the base bodies raise `NotImplementedError` naming the hook.
- `MAX_DATASET_NBYTES: int = 1 << 20` — compile budget for the two baked dataset constants (evoxc's `EVOXC_LARGE_CONSTANT_BYTES` threshold); a subclass may raise it deliberately.
- `dataset_nbytes -> int` — exact footprint of what gets baked (`inputs.nbytes + targets.nbytes` of the float32 arrays); never raises for size, `NotImplementedError` only while a subclass has no dataset yet.
- `validate_dataset() -> int` — same footprint, raising `TensorNEATError` (shapes, bytes, budget, remedy) when it exceeds `MAX_DATASET_NBYTES`.
- `setup(state=None) -> State` — validates the budget, then delegates to `StatefulBaseClass.setup`; func fit registers no keys.
- The budget is therefore enforced at construction, in `setup`, and again at bake time in `_dataset_constants()`, so a direct `evaluate` call (the parity harness) is guarded too; the `TensorNEATError` propagates unchanged through `evoxc.build`.
- `evaluate(state, keys, act_fn, transformed) -> (P,)`: tiling + one `act_fn` call + reduction + `finalize_fitness`; `keys` is accepted and unused (func fit is deterministic).
- `_predict_and_loss(act_fn, pop_size) -> (pred (P, S, O), loss (P,))` is the shared in-trace body behind `evaluate` and `show`.
- `_host_dataset() -> (inputs, targets)` is the single float32 coercion, shared by `dataset_nbytes`, `validate_dataset` and `_dataset_constants`, so the budget arithmetic and the baked constants can never disagree.
- `_dataset_footprint(inputs, targets)` is the byte sum and `_check_dataset_budget(inputs, targets)` the single raise (message carries the two shapes, the bytes, the budget and the remedy); `validate_dataset` and `_dataset_constants` both go through the latter.
- `_dataset_constants()` bakes `(S, I)`/`(S, O)` constants via `evoxc.constant(evoxc.from_numpy(...))`; `_loss(err, targets)` applies the configured reduction over axes 1 and 2.
- `show(state, key, act_fn, transformed)`: host-side, `key` unused (the problem is deterministic); threads the schedule leaves into a one-off `evoxc.evaluate` graph and rebuilds the container there with `evoxc.tree_unflatten(evoxc.tree_structure(transformed), list(arrays))` — never `type(transformed)(*leaves)`, which only holds for a flat one-level container and breaks on HyperNEAT's nested `SubstrateSchedule`; then prints `input | target | predict` per sample and one `loss:` line per individual, returning `None`.
  The one-off graph runs with `**build_options(state)` (`problem/base.py`): `backend`/`device`/`compile_options` come off the state the pipeline registered, so `Pipeline(backend="iree").show(...)` reports from the backend the run trained on, a state registering no backend reports from the numpy reference interpreter, and the backend is never silently substituted (an unsupported one raises from `evoxc`).
  The two graph results are read with `tensorneat.common.to_numpy` (the canonical host accessor), so this module owns no private array conversion.
- The reconstruction is structure-driven, so any batch-first pytree with tensors on its leaves works (flat `Transformed`, nested `SubstrateSchedule`) at any static `P`; the leaves must be concrete tensors, since `evoxc.evaluate` derives one `TensorSpec` per argument.
- `xor.py: XOR(error_method="mse")` — literals for XOR over `{0,1}^2`: `input_shape=(2,)`, `output_shape=(1,)`, S=4, `dataset_nbytes == 48`.
- `xor3d.py: XOR3d(error_method="mse")` — 3-bit parity: `input_shape=(3,)`, `output_shape=(1,)`, S=8, `dataset_nbytes == 128`.
- `custom.py: CustomFuncFit(func, low_bounds, upper_bounds, method="sample", num_samples=100, step_size=None, seed=None, error_method="mse")` stores `func`, float32 `low_bounds`/`upper_bounds`, `method`, `num_samples`, `step_size`, `seed` and the generated `_inputs`/`_targets`; it validates the box before generating anything (shape equality, every bound finite, every axis a proper interval `low < high`), so a box that cannot produce a non-empty sample or grid fails as a `TensorNEATError` at construction.
- `custom.py: cartesian_product(arr1, arr2)` — row-major column-wise product, first input slowest; rank-1 inputs are treated as single columns and produce `(n1*n2, 2)`; disagreeing ranks or a rank above 2 raise `TensorNEATError`.

## Algebra (the one thing to get right)
Tile, don't loop: `tiled = evoxc.tile(inputs, (P, 1))` → `pred = act_fn(tiled)` → `reshape((P, S, O))` → `err = pred - targets` → mean over axes 1 and 2 (sample **and** output), negated, then `finalize_fitness`.
`evoxc.tile` is numpy `tile` semantics, so repeating the `(S, I)` block `P` times on axis 0 gives row `p*S + s` = sample `s` of individual `p` (individual-major, the layout `act_fn` must see) in **one** op.
The `(S, O)` targets broadcast numpy-style against the `(P, S, O)` error tensor in both `enp.subtract` and `mape`'s `enp.divide`, so no `expand_dims` is needed anywhere in this node.
The population size comes from `base.static_batch_size(transformed)` (the only thing read from the schedule); the dataset comes from `evoxc.constant(evoxc.from_numpy(...))` (genuinely static data, baked at trace time).
`initial`-free reductions: `mse = mean(err**2)`, `rmse = sqrt(mse)`, `mae = mean(|err|)`, `mape = mean(|err / targets|)` (a zero target gives `inf`, a NaN prediction gives NaN → `-inf` fitness).
Each reduction is a true mean over **both** the sample and output axes (`enp.mean(..., axis=(1, 2))`, `func_fit.py::FuncFit._loss`), i.e. divided by `S*O`, so the normalization is by the element count, never by `S` alone.
Concrete anchors on `XOR3d` (`S=8, O=1`, four targets equal to 1): a constant-0 predictor gives `mse = 4/8 = 0.5` → fitness `-0.5`; a constant-0.5 predictor gives `mse = 8*0.25/8 = 0.25` → fitness `-0.25` (the same `-0.25` the unit suite pins for XOR's constant-0.5 predictor, `test/unit/problem/test_func_fit.py:393`).
Fitness is a pure negated loss, never an accuracy: the return for sigmoid-bounded outputs lies in `[-1, 0]` and has no `fitness_target` of its own (`XOR3d()` takes only `error_method`; the target lives on `Pipeline`, whose default of `1` is unreachable for this problem, so the run always goes to `generation_limit` unless the caller passes e.g. `-1e-4` as `examples/func_fit/xor3d.py:29` does).

## Constraints
- **Amplification rule.** The problem hands `act_fn` a `(P*S, I)` batch, so every per-row tensor on the activation path — the tiled inputs, the predictions, the per-level window ops — is `S`× taller than the population; the dataset constants must therefore stay `(S, I)`/`(S, O)` and must never be pre-tiled or scaled with `P`.
  The schedule and the payload are the one exception: they travel at their native `P` rows, the genome recovers the static `S = batch // P` (`NodeStepMixin._sample_count`) and expands its own sample-invariant operands once per forward (`genome/window.py::by_samples`), so nothing derived from them is re-derived at `P*S` rows.
- **Dataset budget.** The baked pair is bounded by `MAX_DATASET_NBYTES` (1 MiB) because evoxc warns at that size and, unlike a generic constant, the same `S` also multiplies every per-row tensor on the activation path; an oversized dataset is a configuration error, not a slow-but-valid run (justification in `validate_dataset`'s docstring).
- `method="sample"` draws one `(num_samples, I)` block with `np.random.default_rng(seed).uniform(low, high, ...)` on the host only, so `S == num_samples` (rows, not points per axis) and the global `np.random` state is never touched.
- `method="grid"` builds the cartesian product of `np.arange(low[i], upper[i], step[i])` per axis with the module helper, so the first axis moves slowest and `S` is the product of grid lengths; a fine step can exceed the budget and raises at construction.
- Targets are computed host-side with `np.apply_along_axis(func, -1, inputs)`; a scalar per row is promoted to `(S, 1)`, and `output_shape` is read back from the generated targets (`(O,)`).
- Constructor validation (all `TensorNEATError`): unknown `method`, `num_samples <= 0`, missing/wrong-shape/non-positive `step_size` for `"grid"`, `low_bounds.shape != upper_bounds.shape`, a non-finite bound, an axis where `low_bounds[i] >= upper_bounds[i]` (reported with the offending axis and bounds), `func(low_bounds)` raising, and an over-budget dataset; the shape check on the bounds runs first.
- Everything that touches a traced value is `enp` only (plus top-level `evoxc.tile`, since `enp` has no `tile`); the dataset, bounds and targets stay host numpy, and datasets are always coerced to float32 before baking.
- No python control flow over traced values and no `np.*` on a traced tensor; `show` and the budget helpers are the only host-side entry points.
- Fitness semantics are pinned: the four reductions, NaN → `-inf` (`finalize_fitness`) and `mape`'s zero-target `inf` all match the frozen legacy implementation.

## Measured cost
numpy reference backend, one config per fresh process, build excluded; the numbers come from a scratch harness (not shipped) that times `problem.evaluate` alone with a genome-free `act_fn`, so it is this node's tiling + reduction, not the genome forward (its `--problem-only` mode).
The harness records no tree, so read them as the order of magnitude — sub-millisecond — and never as a per-tree record:

| case | cost | command |
| --- | --- | --- |
| XOR, pop 500 | 0.15 ms | `--case xor --pop 500 --problem-only` |
| XOR3d, pop 500 | 0.19 ms | `--case xor3d --pop 500 --problem-only` |
| CustomFuncFit, pop 200 / 50 samples | 0.25 ms | `--case custom --pop 200 --samples 50 --problem-only` |

`FuncFit.evaluate` alone (genome-free `act_fn`) traces **14 IR ops** including the `return` — the two baked dataset constants, the rank-0 `-inf`, `tile`, `reshape`, the error/reduction chain and the `isnan`/`select` gate of `finalize_fitness` — plus `act_fn`'s own ops; `keys` is unused, so no key op appears.
The evaluate *segment* is not this node's cost: one profiled `custom_pagie_p500` generation of the pass-2 tree (`90d97b4`; pop 500, `max_nodes=50`, `max_in_degree=4`, `CustomFuncFit(method="sample", num_samples=500)`, so `S=500`) splits into pre-ask+transform 7 ms, tile-dataset **0.2 ms**, `by_samples` expansion 62 ms, forward setup (`_hit_conn_attrs` + NaN seed) 64 ms, forward level loop 470 ms, loss **0.6 ms** and tell 93 ms, so this node owns ≈0.8 ms (≈0.1 %) and the level loop owns the rest.
The committed pass-2 record for that cell is 833.81 ms per generation at 1361.4 MiB peak RSS and a 1.63 s build (`benchmarks/PERF_NUMPY.md`); this session measures 847.13 ms on numpy and 16.18 ms on iree+rocm (`../CONTEXT.md` holds the whole five-cell session table).
The tiled `(P*S, I)` batch is exactly what `act_fn` receives, but the schedule and the payload arrive at their native `P` rows: `host_payload._prepared_query` emits no repeat (it prepares one pass per group), and the genome derives its sample-invariant statics at `P` rows and expands them inside the forward (`genome/window.py::by_samples`).
The remaining regrouping is the chunk budget `host_payload.eval_chunk_rows(device_kind, max_nodes)`: `EVAL_CHUNK_ROWS = 12_500` tiled rows on `device=None`/`"cpu"` (25 individuals at `S=500`), and `max(12_500, DEVICE_EVAL_CHUNK_BYTES // (4 * max_nodes))` with `DEVICE_EVAL_CHUNK_BYTES = 1 << 30` on every other device kind — ≈5.37 M rows at `max_nodes=50`, never below the CPU budget.
That budget is read at trace time and unrolled (one prepared pass per group, group outputs concatenated in individual-major row order), so it is part of `Pipeline._shape_signature` and a CPU build is never reused on a device leg.
On CPU a 250 000-row block is 20 groups; on a non-CPU device the same block fits the window and is one unchunked pass, and `xor_p100`/`xor_p500`/`xor_p1000` (`S=4`, ≤4 000 tiled rows) never chunk on either leg.
A chunked group slices both trees through `host_payload._slice_batch`, which raises `TensorNEATError` on a non-scalar leaf whose axis 0 is not `P` instead of sharing or mis-slicing it.

## Graph-constant footprint
`evoxc`'s `enp.zeros/ones/full/empty` materialise a full `Constant` op, and this problem hands `act_fn` a `P*S` batch, so anything a kernel sizes by that batch lands in the artifact — which is why the dataset is the one thing this node bakes, always `(S, I)`/`(S, O)`.
This node bakes exactly three constants: `inputs`, `targets`, and the rank-0 `-inf` of `finalize_fitness` — 48 B for XOR (`inputs (4,2)` 32 B + `targets (4,1)` 16 B), 128 B for XOR3d, ≈1.2 kB for a 100-sample two-input `CustomFuncFit`.
The rest of the artifact's constant volume belongs to siblings: `common/tensorization.py`'s population-scaled tables (its `(P, ...)` zeros, index and level blocks) and the genome's per-pass fills, which are one-element constants broadcast to shape (`genome/window.py::fill`) and so do not grow with `S`.
So the dataset is never the constant problem here.

## Known Issues / Notes for Agents
- **Cross-node (problem/base):** `finalize_fitness` costs 3 ops (`isnan` + `full` + `select`); `evoxc.nan_to_num(x, nan=float("-inf"))` is one op with the same semantics (posinf/neginf untouched), so that helper has a 2-op headroom this node cannot take.
- `Pipeline.show` on HyperNEAT works through the algorithm layer (`HyperNEAT._query_substrate` derives its batch from the payload, not from `pop_size`) and `FuncFit.show` rebuilds the nested `SubstrateSchedule` with `tree_unflatten`; no test drives a `SubstrateSchedule` through `static_batch_size`/`evaluate`/`show`, so that path is hand-verified only.
- `dataset_nbytes` has no caller in `src/` (the budget path is `validate_dataset`/`_check_dataset_budget`); the budget itself is pinned by `test/unit/problem/test_func_fit.py` (the 48-byte footprint, the exact boundary at construction and `setup`, the bake-time re-check) and `test/unit/problem/test_custom.py` (an over-budget grid).
- The budget cannot guard the allocation it exists to bound: `CustomFuncFit.__init__` validates the box and then calls `generate_dataset()` before `super().__init__()`, so a huge `num_samples` (or a fine grid) is fully materialised, targets included, before `validate_dataset` raises.
- The scratch harness named above is a throwaway driver kept outside the repository and run from the worktree root, so it is not shipped and no path to it should be relied on; re-create it if you need to re-measure, calling `problem.evaluate` alone with a genome-free `act_fn`.
  Its `--problem-only` mode is the metric for this node, while its `--share` mode only bounds the segment, because the zeros arm feeds degenerate fitness into `tell` and the two arms speciate differently.

## Routing Table
| Path | Responsibility |
|---|---|
| `./func_fit.py` | `FuncFit`: dataset budget/baking, population tiling, loss reduction, `show`. |
| `./xor.py` | 2-input XOR literals. |
| `./xor3d.py` | 3-input parity literals. |
| `./custom.py` | User-supplied function sampling/gridding, `cartesian_product`. |
