# func_fit — supervised function-fitting problems

## Intent
Fit a genome to a fixed dataset: each individual scores `-loss` of its predictions over every sample.
`FuncFit` owns the only non-trivial part — how a small dataset becomes a batched evaluation — and each concrete problem is just a dataset plus two shapes.

## API Surface
- `func_fit.py: FuncFit(BaseProblem)` — `__init__(error_method: str = "mse")` with `error_method in {"mse", "rmse", "mae", "mape"}` (validated with `TensorNEATError`).
- Abstract hooks implemented by subclasses: `inputs -> np.ndarray (S, I)`, `targets -> np.ndarray (S, O)`, `input_shape -> tuple[int, ...]`, `output_shape -> tuple[int, ...]`.
- `evaluate(state, keys, act_fn, transformed) -> (P,)`: dataset tiling + one `act_fn` call + reduction + `finalize_fitness`; `keys` is accepted and unused (func fit is deterministic).
- `show(state, key, act_fn, transformed)`: host-side, evaluates a batch of one and prints `input | target | predict` rows plus the loss.
- `xor.py: XOR()` — literals for XOR over `{0,1}^2`: `input_shape=(2,)`, `output_shape=(1,)`, S=4.
- `xor3d.py: XOR3d()` — 3-bit parity: `input_shape=(3,)`, `output_shape=(1,)`, S=8.
- `custom.py: CustomFuncFit(func, low_bounds, upper_bounds, method="sample", num_samples=100, step_size=None, seed=None, error_method="mse")` plus the module helper `cartesian_product(arr1, arr2)`.

## Algebra (the one thing to get right)
Tile, don't loop: `broadcast_to(inputs, (P, S, I))` → `reshape((P*S, I))` → `pred = act_fn(tiled)` → `reshape((P, S, O))` → `err = pred - expand_dims(targets, 0)` → reduce over axes 1 and 2.
Row `p*S + s` of the tiled batch is sample `s` of individual `p`; the population size comes from `base.static_batch_size(transformed)`, the dataset from `evoxc.constant(evoxc.from_numpy(...))` (genuinely static data, baked at trace time).

## Constraints
- `method="sample"` draws with `np.random.default_rng(seed)` on the host only; never seed the global `np.random`.
- Targets are computed host-side with `np.apply_along_axis(func, -1, inputs)`, so `func` maps a 1-D point of shape `(I,)` to a scalar or `(O,)` array.
- `method="grid"` requires `step_size` of the same shape as the bounds and strictly positive; `method="sample"` requires `num_samples > 0`.
- Dataset size is a compile-time constant: keep `S * I` under ~250k floats (evoxc warns above 1 MiB of constant payload).
- Everything that touches a traced value is `enp` only; the dataset, bounds and targets stay host numpy.

## Routing Table
| Path | Responsibility |
|---|---|
| `./func_fit.py` | `FuncFit`: dataset baking, population tiling, loss reduction, `show`. |
| `./xor.py` | 2-input XOR literals. |
| `./xor3d.py` | 3-input parity literals. |
| `./custom.py` | User-supplied function sampling/gridding, `cartesian_product`. |
