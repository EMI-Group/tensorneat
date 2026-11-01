# gene/conn — connection genes

## Intent
Turn one connection's `(attributes, source value)` pair into the scalar message that is aggregated at the destination node.
`DefaultConn` carries a single `weight`; `OriginConn` adds an integer `historical_marker` key column (NEAT-paper gene alignment) and inherits the `DefaultConn` weight machinery.

## API Surface
`BaseConn(BaseGene)`:
- `fixed_attrs = ("input_index", "output_index")` (two int32 key columns: source key, destination key).
- `forward(state, attrs, source_values) -> (P, max_nodes, K)`: elementwise in `(attrs, source_values)`, NaN-safe (`isnan(value) -> 0` before multiplying, exactly as the pre-rewrite gene did), returning one scalar message per edge slot.
- `get_historical_marker(state, key_row)` — present when the gene declares `historical_marker`; used by crossover alignment.
- `to_dict(state, key_row, attr_row) -> {"in", "out", "weight"[, "historical_marker"]}` and `sympy_func(state, conn_dict, inputs)` (returns `inputs * weight` and the symbol→value map) for host code generation.
`DefaultConn(weight_init_mean=0.0, weight_init_std=1.0, weight_mutate_power=0.15, weight_mutate_rate=0.8, weight_replace_rate=0.1, weight_lower_bound=None, weight_upper_bound=None)` — `attr_specs = (weight,)`.
`OriginConn(*args, **kwargs)` — `fixed_attrs = ("input_index", "output_index", "historical_marker")`, `attr_specs = (weight,)`, whole-gene crossover, and `get_historical_marker` returning key column 2.

## Constraints
- `forward` must be a pure elementwise function of `(attrs, source_values)`: connection messages never couple across edges, which is what keeps the per-level kernel a single vectorized op.
- `weight` lives in the float32 attr array; `historical_marker` lives in the int32 key array and is compared exactly.
- NaN source values are masked out before the message is aggregated; a gene must not rely on `NaN * 0` arithmetic.
- Additional key columns shift nothing in `attr_specs`; consumers read `attr_specs`/`fixed_attrs`, never hardcoded column indices.
- Files ≤ ~1000 lines.

## Routing Table
| Path | Responsibility |
|---|---|
| `./base.py` | `BaseConn`: key columns, `forward` contract, historical-marker hook. |
| `./default.py` | `DefaultConn`: weight spec, NaN-safe message, sympy/repr. |
| `./origin.py` | `OriginConn`: historical-marker key column, whole-gene crossover. |
