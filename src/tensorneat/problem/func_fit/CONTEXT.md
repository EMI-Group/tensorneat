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
- `_dataset_constants()` bakes `(S, I)`/`(S, O)` constants via `evoxc.constant(evoxc.from_numpy(...))`; `_loss(err, targets)` applies the configured reduction over axes 1 and 2.
- `show(state, key, act_fn, transformed)`: host-side, no `state`/`key` use; threads the schedule leaves into a one-off `evoxc.evaluate` graph and rebuilds the container there with `evoxc.tree_unflatten(evoxc.tree_structure(transformed), list(arrays))` — never `type(transformed)(*leaves)`, which assumed a flat one-level namedtuple and crashed on HyperNEAT's nested `SubstrateSchedule`; then prints `input | target | predict` per sample and one `loss:` line per individual, returning `None`.
- The reconstruction is structure-driven, so any batch-first pytree with tensors on its leaves works (flat `Transformed`, nested `SubstrateSchedule`) at any static `P`; the leaves must be concrete tensors, since `evoxc.evaluate` derives one `TensorSpec` per argument.
- `xor.py: XOR(error_method="mse")` — literals for XOR over `{0,1}^2`: `input_shape=(2,)`, `output_shape=(1,)`, S=4, `dataset_nbytes == 48`.
- `xor3d.py: XOR3d(error_method="mse")` — 3-bit parity: `input_shape=(3,)`, `output_shape=(1,)`, S=8, `dataset_nbytes == 128`.
- `custom.py: CustomFuncFit(func, low_bounds, upper_bounds, method="sample", num_samples=100, step_size=None, seed=None, error_method="mse")` stores `func`, float32 `low_bounds`/`upper_bounds`, `method`, `num_samples`, `step_size`, `seed` and the generated `_inputs`/`_targets`.
- `custom.py: cartesian_product(arr1, arr2)` — row-major column-wise product, first input slowest; rank-1 inputs are treated as single columns and produce `(n1*n2, 2)`.

## Algebra (the one thing to get right)
Tile, don't loop: `tiled = evoxc.tile(inputs, (P, 1))` → `pred = act_fn(tiled)` → `reshape((P, S, O))` → `err = pred - targets` → mean over axes 1 and 2 (sample **and** output), negated, then `finalize_fitness`.
`evoxc.tile` is numpy `tile` semantics, so repeating the `(S, I)` block `P` times on axis 0 gives row `p*S + s` = sample `s` of individual `p` (individual-major, the layout `act_fn` must see) in **one** op — the former `expand_dims` + `broadcast_to` + `reshape` chain was three.
The `(S, O)` targets broadcast numpy-style against the `(P, S, O)` error tensor in both `enp.subtract` and `mape`'s `enp.divide`, so no `expand_dims` is needed anywhere in this node.
The population size comes from `base.static_batch_size(transformed)` (the only thing read from the schedule); the dataset comes from `evoxc.constant(evoxc.from_numpy(...))` (genuinely static data, baked at trace time).
`initial`-free reductions: `mse = mean(err**2)`, `rmse = sqrt(mse)`, `mae = mean(|err|)`, `mape = mean(|err / targets|)` (a zero target gives `inf`, a NaN prediction gives NaN → `-inf` fitness).

## Constraints
- **Amplification rule.** The problem hands `act_fn` a `(P*S, I)` batch, so every per-row allocation on the activation path costs `S`× more than the same allocation at population size; the dataset constants must therefore stay `(S, I)`/`(S, O)` and must never be pre-tiled or scaled with `P`.
- **Dataset budget.** The baked pair is bounded by `MAX_DATASET_NBYTES` (1 MiB) because evoxc warns at that size and, unlike a generic constant, the same `S` also multiplies the whole activation path; an oversized dataset is a configuration error, not a slow-but-valid run (justification in `validate_dataset`'s docstring).
- `method="sample"` draws one `(num_samples, I)` block with `np.random.default_rng(seed).uniform(low, high, ...)` on the host only; the global `np.random` state is never touched.
- `method="grid"` builds the cartesian product of `np.arange(low[i], upper[i], step[i])` per axis with the module helper, so the first axis moves slowest and `S` is the product of grid lengths; a fine step can exceed the budget and raises at construction.
- Targets are computed host-side with `np.apply_along_axis(func, -1, inputs)`; a scalar per row is promoted to `(S, 1)`, and `output_shape` is read back from the generated targets (`(O,)`).
- Constructor validation (all `TensorNEATError`): unknown `method`, `num_samples <= 0`, missing/wrong-shape/non-positive `step_size` for `"grid"`, `low_bounds.shape != upper_bounds.shape`, `func(low_bounds)` raising, and an over-budget dataset.
- Everything that touches a traced value is `enp` only (plus top-level `evoxc.tile`, since `enp` has no `tile`); the dataset, bounds and targets stay host numpy, and datasets are always coerced to float32 before baking.
- No python control flow over traced values and no `np.*` on a traced tensor; `show` and the budget helpers are the only host-side entry points.
- Fitness semantics are pinned: the four reductions, NaN → `-inf` (`finalize_fitness`) and `mape`'s zero-target `inf` all match the frozen legacy implementation.

## Measured cost
numpy reference backend, one config per fresh process, build excluded; dev harness `/tmp/tnscratch/bench.py --problem-only` (times `problem.evaluate` alone with a genome-free `act_fn`, so it is this node's tiling + reduction, not the genome forward):

| case | before (`04e64ea`) | after | command |
| --- | --- | --- | --- |
| XOR, pop 500 | 0.20 ms (a second run: 0.18) | **0.15 ms** | `--case xor --pop 500 --problem-only --reps 9` |
| XOR3d, pop 500 | 0.23 ms | **0.19 ms** | `--case xor3d --pop 500 --problem-only` |
| CustomFuncFit, pop 200 / 50 samples | 0.29 ms (a second run: 0.31) | **0.25 ms** | `--case custom --pop 200 --samples 50 --problem-only` |

Traced ops: `FuncFit.evaluate` alone (genome-free `act_fn`) is **14 IR ops** including the `return` — 4 constants (randkey, inputs, targets, `-inf`), `tile`, `reshape`, `subtract`, `square`, `reduce_mean`, `negate`, `not_equal`, `select`, plus `act_fn`'s own op — down from **17** (both `expand_dims` plus `broadcast_to`+`reshape`). The whole `Pipeline.step` graph drops 3986 → 3983 ops at pop 200.
The evaluate *segment* is not this node's cost: at pop 500 / XOR a standalone `transform` is ≈11.5 ms, `transform`+`forward` is ≈21.4 ms at `P` rows and ≈33.0 ms at `P*S` rows, so `genome.forward` on the tiled batch (158 traced ops incl. the level `while`) ≈ **21.6 ms**, against **0.15 ms** for this node's tiling+loss (≈1 % of the segment). The remaining segment cost is `_repeat_batch` + forward in siblings.
`bench.py --share` (step minus step-with-`evaluate`-replaced-by-zeros) reports 366 ms vs 337 ms at pop 500 (≈29 ms), consistent with that forward figure; treat the number as an upper bound, because the zeros arm feeds degenerate fitness into `tell`, so the two arms speciate differently and the difference is not a clean subtraction.

## Graph-constant footprint
`evoxc`'s `enp.zeros/ones/full/empty` materialise a full `Constant` op, and this problem hands `act_fn` a `P*S` batch, so anything sized by that batch lands in the artifact.
Measured with `/tmp/tnscratch/const_classify.py` (patches `evoxc.ops.constant` and attributes every constant to its creating frame), Pagie `CustomFuncFit` (2 inputs, `method="sample"`, seed 42, `max_nodes=50`, `max_conns=100`, `species_size=10`), build 0.69 s:

| config (P, S, max_nodes) | constants | total | func_fit's share | largest constant |
| --- | --- | --- | --- | --- |
| 200, 50, 50 | 748 | 6.162 MB | 3 consts ≈ 0.6 kB | `(10000, 48)` = 1.92 MB @ `genome/default.py:195` |
| 200, 100, 50 | 748 | 8.163 MB | 3 consts ≈ 1.2 kB | `(20000, 48)` = 3.84 MB @ `genome/default.py:195` |
| 200, 500, 50 | 748 | 24.168 MB | 3 consts ≈ 6.0 kB | `(100000, 48)` = 19.2 MB @ `genome/default.py:195` |
| 200, 50, 250 | 748 | 30.457 MB | 3 consts ≈ 0.6 kB | `(10000, 248)` = 9.92 MB @ `genome/default.py:195` |
| 1000, 100, 50 | 948 | 40.940 MB | 3 consts ≈ 1.2 kB | `(100000, 48)` = 19.2 MB @ `genome/default.py:195` |

XOR at pop 500 (`S=4`, `max_nodes=50`): 748 constants, **10.749 MB**, of which func_fit is 48 B (`inputs (4,2)` 32 B + `targets (4,1)` 16 B); the largest are `(500, 201)` ×4 = 0.402 MB each from `common/tensorization.py`, then `(500, 50, 4)` = 0.4 MB (also `tensorization`) and the 6 P-scaled constants of `genome/default.py` (0.384 MB together).
So the dataset is never the constant problem here: the growth is `4 · P · S · (max_nodes − num_inputs)` bytes for the genome's NaN value seed (`genome/default.py:195`) plus 104 `P`-scaled zeros/gather constants in `common/tensorization.py` (3.43 MB at P=200, 8.56 MB at P=500).
An ~80 MB constant needs e.g. `P=1000, S=100, max_nodes=202` — reachable from genome geometry and samples, never from the dataset.

## Known Issues / Notes for Agents
- **Cross-node (genome):** `genome/default.py:195` seeds activations with `enp.full((P*S, max_nodes − num_inputs), nan)`, a `P*S`-scaled graph constant that grows `S`× per generation of extra samples; it belongs to the genome owner. A `(1, width)` constant plus `evoxc.broadcast`/`evoxc.replicate_tensor` would keep it `P`-independent.
- **Cross-node (common):** `common/tensorization.py` creates 104 `P`-scaled constants (e.g. `(P, max_nodes + 1)` zeros ×4) — the second-largest constant group at every population.
- **Cross-node (problem/base):** `finalize_fitness` costs 3 ops (`not_equal` + `full` + `select`); `evoxc.nan_to_num(x, nan=float("-inf"))` is one op with the same semantics (posinf/neginf untouched), so that helper has a 2-op headroom this node cannot take.
- `Pipeline.show` on HyperNEAT still fails before reaching this node (`HyperNEAT._query_substrate` derives the batch from `self.pop_size`); `FuncFit.show` itself handles the nested `SubstrateSchedule`.
- The dev harnesses named above live in `/tmp/tnscratch/` (not shipped) and are driven from the worktree root; `bench.py --problem-only` is the metric for this node, `--share` only bounds the segment.

## Routing Table
| Path | Responsibility |
|---|---|
| `./func_fit.py` | `FuncFit`: dataset budget/baking, population tiling, loss reduction, `show`. |
| `./xor.py` | 2-input XOR literals. |
| `./xor3d.py` | 3-input parity literals. |
| `./custom.py` | User-supplied function sampling/gridding, `cartesian_product`. |
