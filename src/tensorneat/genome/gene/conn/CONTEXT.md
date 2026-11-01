# gene/conn — connection genes

## Intent
Turn one connection's `(attributes, source value)` pair into the scalar message that is aggregated at the destination node.
`DefaultConn` carries a single `weight` attribute; `OriginConn` adds an integer `historical_marker` key column (NEAT-paper gene alignment) and inherits the whole `DefaultConn` weight machinery.

## Key columns (int32, exact comparison, `I_INF` on absent rows)
`BaseConn.fixed_attrs = ("input_index", "output_index")` — source node key, destination node key; `key_width = 2`.
`OriginConn.fixed_attrs = ("input_index", "output_index", "historical_marker")` — `key_width = 3`, the marker is column 2.
`BaseConn.labels = {"input_index": "in", "output_index": "out"}` makes the inherited `BaseGene.repr`/`to_dict` render `DefaultConn(in: 1, out: 4, weight: 0.5)` and `{"in": 1, "out": 4, "weight": 0.5}` with no family override.
`BaseConn.uses_historical_marker` is `True` iff `"historical_marker" in fixed_attrs`; `algorithm/neat` reads it to decide gene alignment.
`BaseConn.get_historical_marker` is the abstract hook (`NotImplementedError` for families without the column); `OriginConn` returns `key_column(keys, "historical_marker")`.

## Attribute columns (float32, NaN on absent rows)
`DefaultConn.attr_specs` is the single `weight` column, `attr_width = 1`.
The `__init__` knobs (`weight_init_mean/std`, `weight_mutate_power/rate`, `weight_replace_rate`, `weight_lower_bound/upper_bound`) are stored on the instance and rebuild the instance `attr_specs`; the class-level table is only the default declaration.
Both bounds given with `lower >= upper` raises `common.errors.ConfigError`; one-sided and unbounded configurations are legal.
`new_identity_attrs` → `(1,)` with `weight = 1.0` (add-node split reproduces its source; deliberately **not** clipped into the bounds, legacy parity).
`new_zero_attrs` → `(1,)` with `weight = 0.0` (a new connection disturbs nothing until mutation).
`OriginConn` inherits the weight column, both neutral rows and `sympy_func`.

## Forward contract
`forward(state, attrs, source_values) -> (P, max_nodes, K)`: `sanitize(source) * sanitize(weight)`, where `sanitize(x) = select(isnan(x), 0, x)` and `weight = self.attr_column(attrs, "weight")` on `attrs (P, max_nodes, K, 1)`.
Both operands are sanitized before the product, so dead slots carry exactly `0.0` and no `NaN * 0` cotangent can reach reverse mode.
The message stays a pure elementwise function of `(attrs, source_values)`: no cross-edge coupling, no reduction, no control flow.

## Split-payload assembly
`BaseConn.from_attrs(state, in_keys, out_keys, attrs=None, historical_marker=None) -> (conn_keys, conn_attrs)` builds the pair from pre-built columns — `conn_keys (*shape, key_width) int32`, `conn_attrs (*shape, attr_width) float32`; the payload is never one packed row.
`attrs` accepts `(*shape, attr_width)` rows, a `(*shape,)` per-connection scalar broadcast over the attribute axis (the dense-substrate weight table of `algorithm/hyperneat`), or `None` for zeros.
`historical_marker` is accepted only by families that declare the column; omitted (`OriginConn`) it is filled with the flat row index `0..prod(shape)-1` reshaped to `shape`, supplied it is cast to int32 and stacked as column 2.
`algorithm/hyperneat` builds its dense substrate payload with exactly this pair (`node_gene.from_attrs(keys)`, `conn_gene.from_attrs(in_keys, out_keys, weights)`).
Shape disagreements, an unknown marker, a wrong column count or an unusable attribute width raise `common.errors.ConfigError`.

## Genetic operators
`mutate`, `distance` and `hash` are inherited from `BaseGene` and act on the float attribute rows only; NaN padding rows stay NaN.
`BaseGene.crossover` (per-column 50/50 pick) is the `DefaultConn` behaviour.
`OriginConn.crossover` draws one normal per gene row (`(*shape, 1)`) and `select`s that whole attribute row from one parent, so columns of two parents never mix.

## Host-only code generation
`sympy_func` (implemented on `DefaultConn`, inherited by `OriginConn`) returns `inputs * c_{in}_{out}_w` plus `{symbol: conn_dict["weight"]}`.
`repr`/`to_dict` come from `BaseGene` through `labels` plus the declared column names, so `OriginConn` automatically includes `historical_marker`.

## Constraints
- Compute path: `evoxc` ops only (`evoxc.select/isnan/multiply/random`, `evoxc.numpy as enp`); no `jax`/`torch`, no eager NumPy, no `numpy.random`.
- `forward` is pure elementwise in the edge slot: never gather, reduce or branch on traced values inside a family.
- Additional key columns shift nothing in `attr_specs`; consumers read `attr_specs`/`fixed_attrs`, never hardcoded column indices.
- `sympy_func` names the weight symbol from `(in, out)` only, so two `OriginConn`s that differ solely in `historical_marker` share a symbol name.
- `BaseGene.column` is a shape-preserving slice (`slice(i, i+1)` + `evoxc.reshape`), so `DefaultConn.forward` is differentiable end to end through `evoxc.grad`; a dim-dropping integer index or `[..., i]` is rejected in a trace and must never be reintroduced.
- Files ≤ ~1000 lines.

## Routing Table
| Path | Responsibility |
|---|---|
| `./base.py` | `BaseConn`: key columns, labels, `uses_historical_marker`, `from_attrs` split-payload assembly, abstract `forward`/`get_historical_marker`/`sympy_func`. |
| `./default.py` | `DefaultConn`: weight spec from the constructor knobs, identity/zero rows, NaN-safe message, sympy/repr. |
| `./origin.py` | `OriginConn`: `historical_marker` key column, whole-gene crossover, marker read. |
