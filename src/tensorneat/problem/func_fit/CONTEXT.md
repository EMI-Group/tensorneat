# func_fit — supervised function-fitting problems

## Intent
Fit a genome to a fixed dataset: each individual scores `-loss` of its predictions over every sample.
`FuncFit` owns the only non-trivial part — how a small dataset becomes a batched evaluation — and each concrete problem is just a dataset plus two shapes.

## API Surface
- `func_fit.py: FuncFit(BaseProblem)` — `__init__(error_method: str = "mse")` validates `error_method in ERROR_METHODS == ("mse", "rmse", "mae", "mape")` and raises `TensorNEATError` otherwise.
- Subclass obligations: `inputs -> np.ndarray (S, I)`, `targets -> np.ndarray (S, O)`, `input_shape -> (I,)`, `output_shape -> (O,)`; the base bodies raise `NotImplementedError` naming the hook.
- `evaluate(state, keys, act_fn, transformed) -> (P,)`: tiling + one `act_fn` call + reduction + `finalize_fitness`; `keys` is accepted and unused (func fit is deterministic).
- `_predict_and_loss(act_fn, pop_size) -> (pred (P, S, O), loss (P,))` is the shared in-trace body behind `evaluate` and `show`.
- `_dataset_constants()` bakes `np.asarray(..., float32)` copies of `inputs`/`targets` as `evoxc.constant(evoxc.from_numpy(...))`; `_loss(err, targets)` applies the configured reduction over axes 1 and 2.
- `show(state, key, act_fn, transformed)`: host-side, no `state`/`key` use; threads the schedule leaves into a one-off `evoxc.evaluate` graph (rebuilding the container inside the trace), then prints `input | target | predict` per sample and one `loss:` line per individual, returning `None`.
- `xor.py: XOR(error_method="mse")` — literals for XOR over `{0,1}^2`: `input_shape=(2,)`, `output_shape=(1,)`, S=4.
- `xor3d.py: XOR3d(error_method="mse")` — 3-bit parity: `input_shape=(3,)`, `output_shape=(1,)`, S=8.
- `custom.py: CustomFuncFit(func, low_bounds, upper_bounds, method="sample", num_samples=100, step_size=None, seed=None, error_method="mse")` stores `func`, float32 `low_bounds`/`upper_bounds`, `method`, `num_samples`, `step_size`, `seed` and the generated `_inputs`/`_targets`.
- `custom.py: cartesian_product(arr1, arr2)` — row-major column-wise product, first input slowest; rank-1 inputs are treated as single columns and produce `(n1*n2, 2)`.

## Algebra (the one thing to get right)
Tile, don't loop: `broadcast_to(expand_dims(inputs, 0), (P, S, I))` → `reshape((P*S, I))` → `pred = act_fn(tiled)` → `reshape((P, S, O))` → `err = pred - expand_dims(targets, 0)` → mean over axes 1 and 2 (sample **and** output), negated, then `finalize_fitness`.
The `expand_dims` is mandatory: `enp.broadcast_to` cannot introduce a new leading axis (`ShapeError` otherwise).
Row `p*S + s` of the tiled batch is sample `s` of individual `p`; the population size comes from `base.static_batch_size(transformed)` (the only thing read from the schedule), the dataset from `evoxc.constant(evoxc.from_numpy(...))` (genuinely static data, baked at trace time).
`initial`-free reductions: `mse = mean(err**2)`, `rmse = sqrt(mse)`, `mae = mean(|err|)`, `mape = mean(|err / targets|)` (a zero target gives `inf`, a NaN prediction gives NaN → `-inf` fitness).

## Constraints
- `method="sample"` draws one `(num_samples, I)` block with `np.random.default_rng(seed).uniform(low, high, ...)` on the host only; the global `np.random` state is never touched.
- `method="grid"` builds the cartesian product of `np.arange(low[i], upper[i], step[i])` per axis with the module helper, so the first axis moves slowest and `S` is the product of grid lengths.
- Targets are computed host-side with `np.apply_along_axis(func, -1, inputs)`; a scalar per row is promoted to `(S, 1)`, and `output_shape` is read back from the generated targets (`(O,)`).
- Constructor validation (all `TensorNEATError`): unknown `method`, `num_samples <= 0`, missing/wrong-shape/non-positive `step_size` for `"grid"`, `low_bounds.shape != upper_bounds.shape`, and `func(low_bounds)` raising.
- Dataset size is a compile-time constant: keep `S * I` under ~250k floats (evoxc warns above 1 MiB of constant payload).
- Everything that touches a traced value is `enp` only; the dataset, bounds and targets stay host numpy, and datasets are always coerced to float32 before baking.
- No python control flow over traced values and no `np.*` on a traced tensor; `show` is the only host-side entry point and it builds/run its own graph.

## Routing Table
| Path | Responsibility |
|---|---|
| `./func_fit.py` | `FuncFit`: dataset baking, population tiling, loss reduction, `show`. |
| `./xor.py` | 2-input XOR literals. |
| `./xor3d.py` | 3-input parity literals. |
| `./custom.py` | User-supplied function sampling/gridding, `cartesian_product`. |
