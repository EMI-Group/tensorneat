# gene/conn — connection genes

## Intent
Turn one connection's `(attributes, source value)` pair into the scalar message that is aggregated at the destination node.
`DefaultConn` carries a single `weight` attribute column (index 0); `OriginConn` adds an integer `historical_marker` key column (NEAT-paper gene alignment) and inherits the whole `DefaultConn` weight machinery.
Every kernel here is batch-first (population on axis 0) and elementwise in the edge slot: no cross-edge coupling, no reduction and no control flow inside a family.

## Key columns (int32, exact comparison, `I_INF` on absent rows)
`BaseConn.fixed_attrs = ("input_index", "output_index")` — source node key, destination node key; `key_width = 2`.
`OriginConn.fixed_attrs = ("input_index", "output_index", "historical_marker")` — `key_width = 3`, the marker is column 2.
`BaseConn.labels = {"input_index": "in", "output_index": "out"}` makes the inherited `BaseGene.repr`/`to_dict` render `DefaultConn(in: 1, out: 4, weight: 0.5)` and `{"in": 1, "out": 4, "weight": 0.5}` with no family override.
`BaseConn.uses_historical_marker` is `True` iff `"historical_marker" in fixed_attrs`; its readers are `genome/base.py` (initial markers = row index), `genome/step.py::max_conn_marker`, `genome/operations/mutation/base.py` (whether to write the marker column) and `algorithm/neat/neat.py::_new_gene_keys`.
`BaseConn.get_historical_marker` is the abstract hook (`NotImplementedError` for families without the column); `OriginConn` returns `key_column(keys, "historical_marker")`, and nothing in `src/`, `test/` or `examples/` calls it — `genome/step.py` reads key column `"historical_marker"` directly.
Marker identity **across genomes** is the allocator's responsibility, not this node's: the gene only reads the column (`origin.py`'s docstring holds the measured homology consequence).

## Attribute columns (float32, NaN on absent rows)
`DefaultConn.attr_specs` is the single `weight` column, `attr_width = 1`, shipped law `init_mean 0.0` / `init_std 1.0` / `mutate_power 0.15` / `mutate_rate 0.8` / `replace_rate 0.1` / unbounded.
The `__init__` knobs (`weight_init_mean/std`, `weight_mutate_power/rate`, `weight_replace_rate`, `weight_lower_bound/upper_bound`) are stored on the instance and rebuild the instance `attr_specs`; the class-level table is only the default declaration.
Both bounds given with `lower >= upper` raises `common.errors.ConfigError`; one-sided and unbounded configurations are legal, and a bounded family clips floats into its bounds in `mutate` while an `"int"` column is never clipped.
`new_identity_attrs` → one baked `(1,)` constant with `weight = 1.0` (the add-node split reproduces its source), deliberately **not** clipped into the bounds (legacy parity).
`new_zero_attrs` → one baked `(1,)` constant with `weight = 0.0` (a new connection disturbs nothing until mutation).
`OriginConn` inherits the weight column, both neutral rows and `sympy_func`.

## Forward contract
`BaseConn.prepare(attrs) -> ConnOperands` hoists the loop-invariant half of one pass: each declared column is read once (`column`, a `slice` + `reshape` pair) and its NaN replaced with `0` ("a NaN attribute means an absent, padding row") through one shared rank-0 `0` constant.
`prepare` raises `ConfigError` when `attrs.shape[-1] != attr_width` and deliberately takes **no** `state`: its only production caller, `genome/step.py::_hit_conn_attrs`, has no state to pass and the cleaning is pure column arithmetic.
`ConnOperands.columns` is one cleaned tensor per `attr_specs` column in table order, and the bundle's lifetime is one forward pass.
`column_of(attrs, name, default=None)` reads one named column of either a raw `(..., CA)` block (through `attr_column`) or a `ConnOperands` bundle (by spec index); `default` returns that Python float when the table declares no such column, which is the *only* route to the identity weight — every shipped family declares `weight`, so the `1.0` fallback is unreachable here, and `None` raises `ConfigError`.
`forward(state, attrs, source_values) -> (P, max_nodes, K)`: `multiply(source_values, weight)` with `weight = column_of(prepared, "weight", 1.0)`; `attrs` is the bundle or the raw `(P, N, K, attr_width)` block, which is prepared inline, and `None` is **not** accepted.
Only the **weight** is NaN-sanitised, once per pass in `prepare`; the *source* values are passed through uncleaned, so a dead or not-yet-computed slot carries `NaN` out of this kernel.
Those `NaN` messages never reach a reduction — the node gene's aggregation replaces every dead slot with its option's neutral value *before* reducing — so no `isnan` + `select` on the source is emitted per unrolled level (`default.py::forward` holds that rationale).
The message stays a pure elementwise function of `(attrs, source_values)`.

## Split-payload assembly
`BaseConn.from_attrs(state, in_keys, out_keys, attrs=None, historical_marker=None) -> (conn_keys, conn_attrs)` builds the pair from pre-built columns — `conn_keys (*shape, key_width) int32`, `conn_attrs (*shape, attr_width) float32`; the payload is never one packed row because evoxc has no bitcast.
Keys are cast to int32 and stacked by the inherited `gene/base.py::_key_array`, which concatenates on `axis=len(shape)` (never `-1`) because the StableHLO exporter forwards the axis as a non-negative dimension.
The attribute part goes through the inherited `_normalize_attrs`, which narrows to float32 and broadcasts rather than validating value ranges.
`attrs` accepts `(*shape, attr_width)` rows, a `(*shape,)` per-connection scalar broadcast over the attribute axis (the spelling a dense-substrate weight table would take), or `None` for zeros.
`historical_marker` is accepted only by families that declare the column; omitted (`OriginConn`) it is filled with the flat row index `0..prod(shape)-1` reshaped to `shape` (`_row_index_keys`), supplied it is cast to int32 and stacked as column 2.
Passing `historical_marker` to a family without that column, destination keys of a different shape than the source keys, or attribute rows of an unusable shape raises `common.errors.ConfigError`.
`algorithm/hyperneat` builds its dense substrate payload itself, not through this pair: `hyperneat.py` broadcasts the constant node keys with `enp.broadcast_to`, `_conn_keys` broadcasts the `(Q, 2)` endpoint table and appends its own `arange(Q) + 1` marker column, and `_attr_columns` stacks the `(P, rows)` attribute columns with the decoded weights painted into `"weight"`.
`from_attrs` therefore has no call site in `src/`, `test/`, `examples/`, `benchmarks/` or `docs/` — only docstrings name it as a consumer.

## Genetic operators
`mutate` and `distance` are inherited from `BaseGene` and act on the float attribute rows only.
`mutate` never revives padding: a perturbed NaN entry stays NaN and the re-draw mask is folded with `equal(attrs, attrs)`, so a padding row is never written.
`distance` compares columns left to right in `attr_specs` order; this family's table is a single float column, so it takes the one-column fast path `abs(subtract(attrs1, attrs2))` with no slice and no accumulation, and the multi-column rule (an `"int"` column via `not_equal`, a later int term added as a bool, the accumulator started as float32) lives in `gene/base.py::distance`.
`BaseGene.crossover` (per-column 50/50 pick from one `U[0, 1)` block draw) is the `DefaultConn` behaviour.
`OriginConn.crossover` draws one normal per gene row (`(*shape, 1)`) and `select`s that whole attribute row from one parent, so columns of two parents never mix; a family with no attribute column returns parent 1 without tracing a draw.

## Host-only code generation
`sympy_func` (implemented on `DefaultConn`, inherited by `OriginConn`) returns `inputs * c_{in}_{out}_w` plus `{symbol: conn_dict["weight"]}`; a `conn_dict` carrying no `weight` returns the source expression and an empty substitution map.
`sympy_func` names the weight symbol from `(in, out)` only, so two `OriginConn`s that differ solely in `historical_marker` share a symbol name.
`repr`/`to_dict` come from `BaseGene` through `labels` plus the declared column names, so `OriginConn` automatically includes `historical_marker`.

## Constraints
- Compute path: `evoxc` ops only (`evoxc.select/isnan/multiply/cast/random`, `evoxc.numpy as enp`); no `jax`/`torch`, no eager NumPy, no `numpy.random`.
- `forward` is pure elementwise in the edge slot: never gather, reduce or branch on traced values inside a family.
- Additional key columns shift nothing in `attr_specs`; consumers read `attr_specs`/`fixed_attrs`, never hardcoded column indices.
- `BaseGene.column` is a shape-preserving slice (`slice(i, i+1)` + `evoxc.reshape`), so `DefaultConn.forward` is differentiable end to end through `evoxc.grad`; a dim-dropping integer index or `[..., i]` is rejected in a trace and must never be reintroduced.
- Files stay under ~1000 lines: `base.py` 309, `default.py` 208, `origin.py` 90.

## Known Issues / Notes for Agents
`OriginConn.crossover` and `OriginConn.get_historical_marker` are never executed by a committed test: `OriginConn` appears there only as a three-column key layout, so a marker-carrying genome inside `NEAT.tell` is unguarded.
`BaseConn.get_historical_marker` has no caller in `src/`, `test/` or `examples/`.
The `attr_width == 0` branches (`column_of`'s `default`, `_attr_row`'s empty row, the draw-free `crossover`, the width-0 early return in `sympy_func`) are unreachable in this tree, because no shipped conn family declares an empty attribute table.
`ConnOperands` is defined here but not re-exported: `conn/__init__.py` exports exactly `BaseConn`, `DefaultConn`, `OriginConn`, and `genome/step.py`, `algorithm/hyperneat/genes.py` and `conn/default.py` import the bundle class from `.base` directly.
Coverage of the shipped path is indirect: `test/unit/genome/test_mutation_weight.py` pins the weight law, `test/unit/genome/test_payload.py` the marker column's presence/layout, and `DefaultConn.prepare` runs on every forward through `genome/step.py`; the `ConnOperands` spelling is asserted directly only for `HyperNEATConn` in `test/unit/algorithm/test_genes.py`.

## Routing Table
| Path | Responsibility |
|---|---|
| `./base.py` | `BaseConn` (309) and `ConnOperands`: the key-column contract and `labels`, `uses_historical_marker`, `from_attrs` split-payload assembly (key stack via `../base.py::_key_array`, attribute broadcast via `_normalize_attrs`, `_row_index_keys` for an omitted marker), the loop-invariant bundle (`prepare`), `column_of`, and the abstract `forward`/`get_historical_marker`/`sympy_func`. |
| `./default.py` | `DefaultConn` (208): weight spec from the constructor knobs, identity/zero rows, NaN-safe elementwise message, `sympy_func`. |
| `./origin.py` | `OriginConn` (90): `historical_marker` key column, whole-gene crossover, marker read. |
| `./__init__.py` | The three public names `BaseConn`, `DefaultConn`, `OriginConn`. |
