# Intent
Connection genes: one row of the `conns` array (`shape (max_conns, conn_gene.length)`, float32, NaN-padded) describes one directed edge between two node keys.
Every conn gene is identified by the fixed columns `[0:len(fixed_attrs)]`; `DefaultGenome`/`RecurrentGenome` default to `DefaultConn()`.

# API Surface
`BaseConn` (`base.py`) — `fixed_attrs = ["input_index", "output_index"]`; adds `new_zero_attrs(state)` (least-influence attrs, used by "add conn" mutation); `forward(state, attrs, inputs)` → a scalar contributed to the target node's aggregation; host-side `repr`, `to_dict(state, conn)` → `{"in", "out"}`, `sympy_func(state, conn_dict, inputs)` → `(expr, {symbol: value})`.
`DefaultConn` (`default.py`) — `custom_attrs = ["weight"]`, row length 3; ctor kwargs `weight_{init_mean=0.0, init_std=1.0, mutate_power=0.15, mutate_rate=0.2, replace_rate=0.015, lower_bound=-5.0, upper_bound=5.0}`.
- `new_random_attrs` = clipped normal draw; `new_identity_attrs` = `[1.0]`; `new_zero_attrs` = `[0.0]`; `mutate` = `mutate_float` + clip; `distance` = `|w1 - w2|`.
- `forward` is NaN-safe: `jnp.where(jnp.isnan(weight), 0, weight) * jnp.where(jnp.isnan(inputs), 0, inputs)`, i.e. `w * x`.
- `sympy_func` returns `inputs * Symbol("c_{in}_{out}_w")` with `{symbol: weight}`; signature has `precision=None` instead of `is_output_node`.
`OriginConn` (`origin.py`) — `DefaultConn` plus `fixed_attrs = ["input_index", "output_index", "historical_marker"]` (length 4), `crossover` overridden to a single scalar coin flip so all attrs of the chosen parent stay together, `get_historical_marker(state, row)` = `row[2]`, plus 4-column `repr`/`to_dict`.
"Origin" here means "as in the original NEAT paper" (https://github.com/EMI-Group/tensorneat/issues/11), not raw/un-activated values.

# Constraints
`forward` receives the *source node value* as `inputs` (a scalar, from a padded `(max_nodes,)` value vector), and the result is masked by `valid_mask` at the node's aggregation — a NaN weight or NaN source must therefore not poison the row.
`to_dict`/`repr` must not hit NaN rows (they `int()`/`float()` the columns on the host).
`historical_marker` at column 2 is written by `genome/base.py` and `algorithm/neat/neat.py` only when it appears in `fixed_attrs`, and is then auto-incremented per new connection; keep its position fixed.

# Known Issues
`DefaultConn.sympy_func`'s extra `precision=None` parameter makes it signature-incompatible with `BaseNode.sympy_func`'s `is_output_node`.

# Routing Table
No subdirectories.
