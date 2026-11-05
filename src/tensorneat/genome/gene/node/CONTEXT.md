# gene/node — node genes

## Intent
Turn one node's aggregated incoming-message window into that node's value, with the aggregation and activation choice stored in the gene row.
`DefaultNode` is the NEAT-python behaviour (`bias`, `response`, `aggregation`, `activation`); `BiasNode` drops `response`; `OriginNode` keeps the `DefaultNode` attributes and only changes crossover to a whole-gene pick.
The forward path is graph code (`evoxc` ops only); `repr` / `to_dict` / `sympy_func` are host-only and never traced.

## Attr column layout (binding)
The key side is one `int32` column `index` (the node key) declared by `BaseNode.fixed_attrs`, rendered as `idx` through `BaseNode.labels`.
`DefaultNode` and `OriginNode` declare four float32 attribute columns in this exact order: `bias` (float), `response` (float), `aggregation` (int option index), `activation` (int option index); `attr_width == 4`.
`BiasNode` declares three: `bias` (float), `aggregation` (int option index), `activation` (int option index); `attr_width == 3`.
The instance `attr_specs` is rebuilt in the family constructor from the knobs, with `options=tuple(range(len(option_list)))` and `replace_rate` taken from the matching `*_replace_rate`; the class-level table only records the default single-option layout.
Column order *is* the `*_attrs` column order, so `bias=0`, `response=1`, `aggregation=2`, `activation=3` (and `bias=0`, `aggregation=1`, `activation=2` for `BiasNode`) is what `test/support/oracle.default_node_attrs_of` reads.
`-1` in the `activation` column means identity (`common.functions.act.ACT_IDENTITY`) and is never an option index; aggregation has no `-1` sentinel.

## Identity / random rows
`DefaultNode.new_identity_attrs` -> `(0, 1, aggregation_default, activation_default)`, the two floats clipped into their configured bounds.
`BiasNode.new_identity_attrs` -> `(0, aggregation_default, activation_default)`, always a real activation option and never `-1` (the legacy `lax.switch(-1, ...)` clamped to the last option, so a split node silently switched activation).
`new_random_attrs` and `new_zero_attrs` are inherited unchanged: one batched `evoxc.random` draw per spec (clipped Gaussian for floats, uniform option index for ints), and zeros / first option respectively.
`aggregation_default` / `activation_default` are int indices resolved with `options.index(default)`; `None` means option 0.

## Hooks: implemented here vs inherited
`BaseNode` implements `setup`, `forward`, `prepare` (the `NodeOperands` bundle), `aggregate`, `needs_has_valid`, `column_of`, `repr`, `to_dict` and the host-only `_aggregation_name` / `_activation_name` / `_option_name`.
`forward` is `z = combine(attrs, aggregate(...))` then `select_cascade(activation_masks, [fn(z) for fn in activation_options], z)`; there is no `select(is_output_node, z, activated)` step, because `prepare` writes `ACT_IDENTITY` into the activation id of every output row, where it matches no option and the cascade returns `z` unchanged.
`prepare` reads every attribute column once, casts the option columns to `int32`, forces `ACT_IDENTITY` on the output rows and builds the per-option `ids == index` masks, so no level re-reads an attribute or re-emits an `equal`; a family with no attribute column gets one broadcast constant id (aggregation option `0`, activation `ACT_IDENTITY`) instead of a population-scaled `enp.full`.
`aggregate` is implemented **once** on `BaseNode` as `select_cascade(aggregation_masks, [fn(messages, valid_mask) for fn in aggregation_options], zero)` followed by the `has_valid` guard `select(any_slot(valid_mask), aggregated, 0)`; `needs_has_valid` drops that guard when every configured option is in `self_neutral_aggregations` (`AGG.sum`, `AGG.mean`); no family overrides `aggregate`.
`common.functions.apply_activation` / `apply_aggregation` are not called by this node any more — `option_masks`/`select_cascade` are the same cascade with the per-level `equal`/cast hoisted — and `NodeOperands.aggregation_ids`/`activation_ids` are stored but never read.
`prepare` assumes the caller already 0-padded the attributes of absent rows (`genome/step.py::_padding_safe_attrs`, `default.py`/`recurrent.py` forward paths); a NaN option id would otherwise be cast to `int32` and only later discarded.
`combine` and `sympy_func` stay abstract on `BaseNode` (accurate `NotImplementedError`) because a base class without an attribute table cannot know its columns or option names.
`DefaultNode` overrides `__init__`, `new_identity_attrs`, `combine` (`bias + response * aggregated`), `repr`, `to_dict`, `sympy_func`.
`BiasNode` overrides `__init__`, `new_identity_attrs`, `combine` (`bias + aggregated`), `repr`, `to_dict`, `sympy_func`, and shares `_option_list` / `_default_index` / `_option_spec` with `DefaultNode` instead of copying them; it deliberately does not chain the response-carrying constructor.
`OriginNode` overrides only `crossover`: one normal draw of shape `(*shape, 1)` per gene row (broadcast over the attribute axis) and `select(draw > 0, attrs1, attrs2)`, so a child row is a whole parent row.
`mutate`, the default per-column `crossover`, `distance`, `hash`, `from_attrs` and every width helper are inherited from `gene/base.BaseGene` unchanged.

## Op count
`aggregate` evaluates every configured option on the whole `(P, N, K)` window (one masked reduce each) plus one `select` per option, then the `has_valid` guard when a configured option is not self-neutral; a single-option gene is 1 reduce + 1 select.
The activation cascade starts from `z` (the identity branch) and emits one `select` per configured option over `(P, N)`, so the default single-option configuration is a 1-select cascade.
The per-**pass** cost is the attribute column reads, the two `int32` casts and one `equal` per option (all in `prepare`); per **level** only the reduce/select dispatch and `combine` remain, and option ids are gene data rather than a static shape, so no static shortcut can reduce the cascade to the options a population happens to use.
A per-level node step therefore stays `O(max_nodes x K)` gathers plus a constant dispatch cost, never `max_nodes^2`.

## Host-only API (siblings may rely on it)
`to_dict` returns `{"idx": int, "bias": float, "res": float, "agg": str, "act": str}` for `DefaultNode` / `OriginNode` and the same mapping without `"res"` for `BiasNode`; `agg` / `act` are registry names and `-1` renders as `"identity"`.
`repr` reuses `to_dict` and prints `TypeName(idx=..., bias=..., [response=...,] aggregation=<name>, activation=<name>)`; `BaseNode.repr` / `BaseNode.to_dict` are the generic label-driven rendering for a family without named options.
`sympy_func(node_dict, inputs, is_output_node=False)` takes the incoming edge contributions (an iterable of sympy expressions, or one expression treated as a one-term list) and returns `(expression, {symbol: value})` with the `n_<idx>_b` / `n_<idx>_r` symbol names.
A custom option callable that is not in the global `ACT` / `AGG` registry is named by its own `__name__`; an option id outside `[0, len(options))` raises `ConfigError` instead of wrapping.
`genome/base.py` calls `node_gene.forward(state, node_attrs, msg, is_output_row, live)` once per level/tick; `genome/operations/mutation/` uses `new_identity_attrs` for add-node splits.

## Constraints
Output nodes are never activated here; the final activation belongs to the genome-level `output_transform`.
`messages` arrives NaN-sanitized (0 where the source value was NaN) and dead slots are neutral-filled per aggregation option, so no gene hook sees NaN.
`setup()` chains `BaseGene.setup` and rejects an empty option list (`ConfigError`); the constructor rejects an empty list, a non-callable option and a `*_default` outside its list.
No `jax` / `torch`, no `numpy.random`, no eager NumPy on the compute path; host helpers convert through `BaseGene._host_row` (`common.tools.to_numpy`).
Files stay well under ~1000 lines and a new family adds declarations plus hooks, never a copy of the table/operator machinery.

## Routing Table
| Path | Responsibility |
|---|---|
| `./base.py` | `BaseNode`: key column + `idx` label, `setup` option validation, `forward`, the single generic `aggregate`, abstract `combine`/`sympy_func`, host option-name helpers, generic repr/to_dict. |
| `./default.py` | `DefaultNode`: 4-column table, constructor knobs, identity row, `bias + response * agg`, repr/to_dict/sympy, shared `_option_list` / `_default_index` / `_option_spec`. |
| `./bias.py` | `BiasNode`: 3-column table, identity row on a real activation option, `bias + agg`, repr/to_dict/sympy without `res`. |
| `./origin.py` | `OriginNode`: `DefaultNode` attributes with whole-gene crossover. |
