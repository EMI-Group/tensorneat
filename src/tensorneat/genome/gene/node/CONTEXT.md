# gene/node — node genes

## Intent
Turn one node's aggregated incoming-message window into that node's value, with the aggregation and activation choice stored in the gene row.
`DefaultNode` is the NEAT-python behaviour (`bias`, `response`, `aggregation`, `activation`); `BiasNode` drops `response`; `OriginNode` inherits `DefaultNode`'s table and changes only crossover to a whole-gene pick.
The forward path is graph code (`evoxc` ops only); `repr` / `to_dict` / `sympy_func` are host-only and never traced.
`algorithm/hyperneat/genes.py::HyperNEATNode` is a fourth `BaseNode` subclass (owned by `algorithm/`): two int option columns, no bias/response, `combine` returning the aggregate unchanged, so it consumes this node's `forward`/`prepare` contract with the width-fallback defaults.

## Public names
`gene/node/__init__.py` exports exactly `BaseNode`, `DefaultNode`, `BiasNode`, `OriginNode` — the same four names `gene/__init__.py` re-exports.
`base.py` additionally exposes `NodeOperands` (the prepared bundle `genome/step.py` builds) and uses the trace-time helpers `column` / `option_masks` / `select_cascade` from `../base.py`.

## Attr column layout (binding)
The key side is one `int32` column `index` (the node key) declared by `BaseNode.fixed_attrs`, rendered as `idx` through `BaseNode.labels`.
`DefaultNode` declares four float32 attribute columns in this exact order: `bias` (float), `response` (float), `aggregation` (int option index), `activation` (int option index); `attr_width == 4`.
`BiasNode` declares three: `bias` (float), `aggregation` (int option index), `activation` (int option index); `attr_width == 3`.
`OriginNode` declares no table of its own and inherits `DefaultNode`'s four columns verbatim.
The instance `attr_specs` is rebuilt in the constructor from the knobs, with `options=tuple(range(len(option_list)))` and `replace_rate` taken from the matching `*_replace_rate`; the class-level table only records the default single-option layout (bias `N(0, 1)`, response `N(1, 0)`, one option per option column, option `replace_rate` `0.1`).
Column order *is* the `*_attrs` column order, so `bias=0`, `response=1`, `aggregation=2`, `activation=3` (and `bias=0`, `aggregation=1`, `activation=2` for `BiasNode`) is what `test/support/oracle.default_node_attrs_of` reads.
`-1` in the `activation` column means identity (`common.functions.act.ACT_IDENTITY`) and is never an option index; aggregation has no `-1` sentinel.
An option index is gene data, never a static shape, so a cascade can never be narrowed to the options a running population happens to use.

## Identity / random / zero rows
`DefaultNode.new_identity_attrs` -> `(0, 1, aggregation_default, activation_default)` built by walking `attr_specs`, with the two floats clipped into their configured bounds.
`BiasNode.new_identity_attrs` -> `(0, aggregation_default, activation_default)`, always a real activation option and never `-1`, because a split node must keep the activation it had (the legacy `lax.switch(-1, ...)` clamped to the last option).
`new_random_attrs` is inherited from `../base.py::BaseGene` (a one-line delegation to `blocks.init_attrs`): the rank-0 key is split **once** into one stream per column, each non-degenerate column draws its own batched value (clipped Gaussian for a float, uniform option index for a multi-option int), and a column that degenerates to a constant traces **no** draw — `response_init_std` is `0.0` in `DefaultNode` and every option column of the shipped layout holds a single option, so the shipped table traces exactly one draw (`bias`).
`new_zero_attrs` is inherited too and sets **every** float column to `0` and every int column to its first option, so a node gene's zero row is `(0, 0, option 0, option 0)` — `response` becomes `0`, not the identity row's `1`; it has no production caller for a node gene (add-conn uses the conn gene's zero row).
Both rows are one `evoxc.constant` (`_attr_row` -> `blocks.static_row`, or the empty row at `attr_width == 0`), and `_attr_row` refuses a value count that disagrees with `attr_width`.
`aggregation_default` / `activation_default` are int indices resolved with `options.index(default)` (identity comparison, so a wrapper callable is not recognised); `None` means option 0.

## Hooks: implemented here vs inherited
`BaseNode` implements `fixed_attrs`/`labels`, `self_neutral_aggregations`, `setup`, `needs_has_valid`, `forward`, `prepare`, `_option_ids`, `column_of`, `aggregate`, `_host_column`, the host-only `_aggregation_name` / `_activation_name` / `_option_name`, and the generic label-driven `repr` / `to_dict`; `combine` and `sympy_func` stay abstract (`NotImplementedError`), because a base class without an attribute table cannot know its columns or its option names.
`forward(state, attrs, messages, is_output_node, valid_mask)` takes the raw `(P, N, NA)` attribute block **or** a `NodeOperands` bundle (a raw block is prepared inside the call), the `(P, N, K)` message window, the `(P, N)` output-row mask and the `(P, N, K)` liveness mask; `aggregate` takes `(state, attrs, messages, valid_mask)` and `combine` `(state, attrs, aggregated)`.
It computes `z = combine(aggregate(...))` and then `select_cascade(activation_masks, [fn(z) for fn in activation_options], z)`; there is no `select(is_output_node, z, activated)` step, because `prepare` writes `ACT_IDENTITY` into the activation id of every output row, where it matches no option and the cascade returns `z` unchanged.
`prepare(attrs, is_output_node)` takes **no `state`** — it is pure column arithmetic like `BaseGene.attr_column` — and it is called by `forward` when the caller hands a raw block and, in the pipeline, only by `genome/step.py::_StepStatics.node_operands`, which passes the `(P, N)` output-row mask, expands the returned bundle to the evaluated row count with `window.by_samples`, and raises when a later level/tick is handed a different attribute tensor.
`prepare` returns `NodeOperands(columns, aggregation_masks, activation_masks, zero)`: one `(P, N)` read per attribute column in table order, the per-option `ids == index` masks of both dispatch columns, and one rank-0 `float32` zero shared by every level's cascade — built once per forward pass, so no level re-reads an attribute or re-emits an `equal`.
A family with no attribute column at all gets one broadcast constant id (aggregation option `0`, activation `ACT_IDENTITY`) through `window.fill` instead of a population-scaled `enp.full`; the output-row override is emitted only when the table declares an `activation` column.
`prepare` assumes the caller already 0-padded the attributes of absent rows (`genome/step.py::_padding_safe_attrs`), so a NaN option id never reaches the `int32` cast.
`aggregate` is implemented **once** on `BaseNode` as `select_cascade(aggregation_masks, [fn(messages, valid_mask) for fn in aggregation_options], zero)` followed by the `has_valid` guard `select(any_slot(valid_mask), aggregated, zero)`; no family overrides it, and a family needing a different operator over the window would override it without copying the cascade.
`needs_has_valid` drops that guard when every configured option is in `self_neutral_aggregations` (= `(AGG.sum, AGG.mean)`), which is the identity-compared, stricter copy of `common/functions/agg.py`'s `zero_on_empty` set (that one also marks `AGG.maxabs`), so a `maxabs` configuration pays one `any_slot` + `select` per level for a value that is already `0.0` and a custom wrapper callable always keeps the guard.
`common.functions.apply_activation` / `apply_aggregation` are not called by this node any more — `option_masks`/`select_cascade` are the same cascade with the per-level `equal`/cast hoisted — and the option ids themselves stay local to `prepare`: only the derived `aggregation_masks`/`activation_masks` are carried in `NodeOperands`.
`DefaultNode` overrides `__init__`, `new_identity_attrs`, `combine` (`bias + response * aggregated`), `repr`, `to_dict`, `sympy_func`.
`BiasNode` overrides `__init__`, `new_identity_attrs`, `combine` (`bias + aggregated`), `repr`, `to_dict`, `sympy_func`, and shares `_option_list` / `_default_index` / `_option_spec` with `DefaultNode` instead of copying them; it deliberately does not chain the response-carrying constructor, because a stored response knob would advertise configuration that cannot reach the payload.
`OriginNode` overrides only `crossover`: one normal draw of shape `(*shape, 1)` per gene row (broadcast over the attribute axis, `(1,)` for a rank-0 row) and `select(draw > 0, attrs1, attrs2)`, so a child row is a whole parent row; its `if not self.attr_specs: return attrs1` guard is a width-0 spelling that ships unreachable.
`OriginNode` therefore defines neither `repr` nor `to_dict` and inherits `DefaultNode`'s, so all four shipped families render through a family override and `BaseNode`'s generic rendering has no caller.
`mutate`, the default per-column `crossover`, `distance`, `from_attrs` and every width helper are inherited from `gene/base.BaseGene` unchanged.

## Op count
`aggregate` evaluates every configured option on the whole `(P, N, K)` window — each option's reduce is its own static K-plane fold in `common/functions/agg.py` (`O(K)` ops, switching to the backend `reduce_*` only above that module's fold-width guard) — and then picks the row's option with one `select` per option; the `has_valid` guard, when emitted, adds `window.any_slot` (K `slice` + K-1 `logical_or` + one `reshape`) plus one `select`.
`combine` is 2 ops (`multiply`, `add`) for `DefaultNode`, 1 (`add`) for `BiasNode` and 0 for `HyperNEATNode`; the activation cascade starts from `z` (the identity branch) and emits one `select` per configured option over `(P, N)`.
A shipped single-option `DefaultNode` (`AGG.sum`, one activation) therefore spends per level one K-plane fold, 1 aggregation `select`, 3 combine-like ops and 1 activation `select`, and emits no `has_valid` guard.
The per-**pass** cost is the `width` attribute column reads (each a `slice` plus a differentiable `reshape`), the `int32` casts of the option columns, one `equal` per configured option, one output-row `select`, and the pooled rank-0 constants.
The budget here is dispatched ops plus the fold's element volume (`P × max_nodes × K` for the fold, `P × max_nodes` after it, never `max_nodes²`); `../../CONTEXT.md` holds the measured per-op laws (numpy interpreter ≈19–20 µs per dispatched op; a flat iree+rocm graph ≈63 µs fixed plus ≈2.2–2.5 µs per dispatch) and the column-read measurement that makes a read's two cheap ops the floor.

## Host-only API (siblings may rely on it)
`to_dict` returns `{"idx": int, "bias": float, "res": float, "agg": str, "act": str}` for `DefaultNode` / `OriginNode` and the same mapping without `"res"` for `BiasNode`; `agg` / `act` are registry names and `-1` renders as `"identity"`.
`repr` reuses `to_dict` and prints `TypeName(idx=..., bias=..., [response=...,] aggregation=<name>, activation=<name>)` with the floats rounded to `precision`.
`sympy_func(state, node_dict, inputs, is_output_node=False)` takes the incoming edge contributions (an iterable of sympy expressions, or one expression treated as a one-term list) and returns `(expression, {symbol: value})`; the symbols are `n_<idx>_b` / `n_<idx>_r` (`BiasNode` emits only `n_<idx>_b`), and the activation is applied only when `is_output_node` is false.
A custom option callable that is not in the global `ACT` / `AGG` registry is named by its own `__name__`; an option id outside `[0, len(options))` raises `ConfigError` instead of wrapping.
`genome/step.py::_node_step` calls `node_gene.forward(state, node_operands, messages, output_rows, live)` once per level/tick, where `live` is `present & (xv == xv)`; `genome/operations/mutation/` uses `new_identity_attrs` for the add-node split.

## Constraints
Output nodes are never activated here; the final activation belongs to the genome-level `output_transform`.
`messages` may carry `NaN` on a dead slot (`live` is the only liveness authority), and every aggregation option replaces the slots of `valid_mask` with its neutral element before reducing, so no gene hook's result depends on a dead slot.
`setup()` chains `BaseGene.setup` and rejects an empty option list (`ConfigError`); the constructor rejects an empty list, a non-callable option and a `*_default` outside its list.
No `jax` / `torch`, no `numpy.random`, no eager NumPy on the compute path; host helpers convert through `BaseGene._host_row` (`common.tools.to_numpy`).
Files stay well under ~1000 lines (`base.py` 611, `default.py` 444, `bias.py` 307, `origin.py` 61, `__init__.py` 18) and a new family adds declarations plus hooks, never a copy of the table/operator machinery.

## Coverage notes
The `attr_width == 0` branches (`column_of`'s default, `_option_ids`' constant id, `_attr_row`'s empty row, `OriginNode`'s empty-spec guard) are unreachable in-tree: every shipped node family declares 2, 3 or 4 columns.
No test asserts `BiasNode`'s `repr` / `to_dict` / `sympy_func` or its identity row; its shipped callers are the `examples/rl` and `examples/func_fit` forward paths plus `examples/interpret_visualize` for the host hooks.
`OriginNode.crossover` is never executed by a test: every crossover assertion runs the per-attribute `DefaultNode` path, and only `examples/func_fit/xor_origin.py` drives the whole-gene pick.

## Routing Table
| Path | Responsibility |
|---|---|
| `./base.py` | `BaseNode`: key column + `idx` label, `self_neutral_aggregations`/`needs_has_valid`, `setup` option validation, `forward`, `prepare`/`NodeOperands`, `_option_ids`, `column_of`, the single generic `aggregate`, abstract `combine`/`sympy_func`, host option-name helpers, generic repr/to_dict. |
| `./default.py` | `DefaultNode`: 4-column table, constructor knobs, identity row, `bias + response * agg`, repr/to_dict/sympy, shared `_option_list` / `_default_index` / `_option_spec`. |
| `./bias.py` | `BiasNode`: 3-column table, identity row on a real activation option, `bias + agg`, repr/to_dict/sympy without `res`. |
| `./origin.py` | `OriginNode`: `DefaultNode` attributes and host hooks, whole-gene crossover only. |
