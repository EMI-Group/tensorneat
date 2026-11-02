# gene — attr-table driven gene families

## Intent
Define what a gene *is* (its int32 key columns and its float32 attribute columns) and how it converts, mutates, crosses over, distances, hashes, and forwards, so that node and conn families are declarations over one shared mechanism instead of copy-paste.
`BaseGene` owns the table-driven plumbing; `node/` and `conn/` add only their structural hooks.
Nothing here imports `jax`/`torch`, and no hook on the compute path runs eager NumPy.

## API Surface
`BaseGene(StatefulBaseClass)`:
- class data: `fixed_attrs: tuple[str, ...]` (int32 key columns), `attr_specs: tuple[GeneAttr, ...]` (float32 attribute columns in row order), `labels: dict[str, str]` (short render names; `repr`/`to_dict` fall back to the declared name).
- `key_width` / `attr_width` (`len(fixed_attrs)` / `len(attr_specs)`), `length = key_width + attr_width` (display only).
- `setup(state=None)` validates the table (`_validate_specs`: non-empty names, unique names, wide-enough int option lists, `init_std >= 0`, `lower <= upper`) and registers the gene; violations raise `common.errors.ConfigError`.
- `attr_index(name)` / `key_index(name)` (host ints, `ConfigError` if unknown), `attr_column(attrs, name)` / `key_column(keys, name)` (trace-time column read through `column`).
- `new_random_attrs(state, key, shape) -> (*shape, attr_width)` — one batched `evoxc.random` draw per spec, clipped into the spec bounds; `shape=()` is supported (draws `(1,)` then reshapes).
- `new_identity_attrs(state)` — floats at their init mean, ints at their first option; `new_zero_attrs(state)` — zeros, ints at their first option. Both return `(attr_width,)` and are rebuilt inside the graph (`enp` constants, never eager `np`).
- `mutate(state, key, attrs)` — batched over leading axes: floats `mutate_float` noise `mutate_rate` of the time and `replace_rate` re-init from the spec distribution, ints re-drawn uniformly over the option list with `replace_rate`.
- `crossover(state, key, attrs1, attrs2)` — one rank-0 normal draw broadcast over the attribute columns, per-column 50/50 pick.
- `distance(state, attrs1, attrs2)` — sum over the attribute axis of `|Δ|` for floats and `!=` (as 0/1) for ints, with NaN padding treated as "absent" (a NaN row contributes 0).
- `hash(keys)` — `hash_int32` over the int32 key columns (xor-mix: `h = h ^ (k + HASH_GOLDEN + (h << 6) + (h >> 2))`), bit-identical to `common.tools.hash_rows`.
- `from_attrs(state, keys, attrs=None)` — split-payload assembly from pre-built columns; validation of a wrong column count / width raises `ConfigError`.
- Host-only: `repr`/`to_dict` (label-driven, `-1` activation renders `"identity"`), plus each family's `sympy_func`.

Module helpers (trace-time, used by every family):
- `column(tensor, index) -> tensor[..., index]` — shape-preserving slice `slice(index, index+1)` + `evoxc.reshape`, because a dim-dropping integer index (and `[..., i]`) is rejected in a trace and has no usable VJP; this form is differentiable (verified through `evoxc.grad`).
- `hash_int32(keys) -> (...,) uint32` — the hash body; `uint32` ops only, because there is no bitcast in evoxc.

`GeneAttr` is a dataclass: `name`, `kind` (`"float"`/`"int"`), `init_mean`/`init_std` (floats) or `options` (ints), `mutate_power`, `mutate_rate`, `replace_rate`, optional `lower`/`upper`.

Trace-time entry points of the families: `BaseNode.forward(state, attrs, messages, is_output_node, valid_mask) -> (P, N)` and `BaseConn.forward(state, attrs, source_values) -> (P, N, K)`.
Family dispatch calls the authoritative `common.functions` signatures: `apply_aggregation(values, mask, agg_ids, agg_fns)` and `apply_activation(z, act_ids, act_fns)`.

## Attribute column layout (binding for every consumer)
`DefaultNode` / `OriginNode`: `bias, response, aggregation, activation` (`attr_width = 4`, ints in columns 2 and 3).
`BiasNode`: `bias, aggregation, activation` (`attr_width = 3`).
`DefaultConn` / `OriginConn`: `weight` in column 0 (`attr_width = 1`).
`-1` in an activation column means identity and is never an option index; aggregation has no `-1`.
`test/support/oracle.py` (`default_node_attrs_of`, `default_conn_attr_of`) reads exactly this order.

## Constraints
- Attribute order in `attr_specs` *is* the column order of `*_attrs`; never reorder without changing every consumer, and consumers read `attr_specs`/`fixed_attrs`, never hardcoded indices.
- One batched draw per spec with the full output shape: never a per-row Python loop and never per-row key splits (evoxc keys are rank-0).
- Int attributes (option indices) live in the float32 attr array and are compared after an `int32` cast; `-1` activation identity is handled by `apply_activation`, not by an option-table index.
- No `jax`/`torch`, no `numpy.random`, no eager NumPy on the compute path; option branches are `evoxc.select` cascades over NaN-free branches (each aggregation option neutral-fills dead slots before its reduce).
- Keys are never floats, and padding sentinels never mix: `I_INF` in key arrays, NaN in attr arrays.
- Errors: configuration violations raise `common.errors.TensorNEATError` subclasses (`ConfigError`); evoxc shape/backend errors propagate unchanged.
- Files ≤ ~1000 lines; a new gene family adds declarations plus hooks, never a copy of the mutate/crossover/distance code.

## Hooks other nodes rely on
`genome/base.py`: `initialize` uses `new_random_attrs`, `new_identity_attrs` (add-node), `new_zero_attrs`; the forward step calls `conn_gene.forward` then `node_gene.forward`; `grad` differentiates only the float attr arrays, which works because `attr_column` is differentiable.
`genome/operations/**`: `mutate`, `crossover`, `distance`, `hash`, `attr_width`/`key_width`, `from_attrs`.
`algorithm/**`: `from_attrs` (payload assembly), `uses_historical_marker`/`get_historical_marker`, `node_gene.from_attrs(keys)` + `conn_gene.from_attrs(in_keys, out_keys, weights)` for the HyperNEAT substrate.
`BaseConn.from_attrs` accepts `attrs=None` (zeros), `(*shape,)` per-connection scalars broadcast over the attribute axis, or full `(*shape, attr_width)` rows.

## Origin families (verified behaviour)
`OriginNode`/`OriginConn` differ from the default families **only** in `crossover`; every other hook (attr table, mutate, distance, forward, historical marker) is inherited unchanged and the marker is read from key column 2, which the distance digest verifies row-for-row.
Their whole-gene pick is real: on a fixed `(P, M, A)` payload every output row equals one parent's row verbatim (`A=4` node, `A=1` conn), versus the per-column mix of `BaseGene.crossover` — and for the one-column conn gene the override is behaviourally identical to the inherited per-column pick.
Their crossover has **no observable effect** when the algorithm pairs identical parents: `algorithm/neat/_kernels.py::pair_parents` draws both parents from `randint(0, survive)` with `survive = clamp(floor(survival_threshold * member_count), 1, …)`, so a species with fewer than `2 / survival_threshold` members gets `survive == 1` and `winner == loser`, making the child equal to parent 1 for every gene family (`survival_threshold=0.01`, used by every example, needs 200 members, so `pop_size=100` can never activate crossover).
Verified on seeds 42/43/44 with `pop_size=100, species_size=20, survival_threshold=0.01, max_nodes=50, max_conns=100, XOR`: the instrumented `father != mother` slot count is `0` in all 1 200 generation-cases, the `DefaultNode` and `OriginNode` populations are bit-identical for every generation, and all four node/conn family combinations share one fitness history; with the `survive` floor raised to `min(2, member_count)` the fraction becomes 0.50 median and the families diverge.
The marker allocator (`algorithm/neat/neat.py::_new_gene_keys`) derives new markers per individual, so on an evolved `OriginConn` population only 83–97 % of shared `(in, out)` edges carry the same marker; full-tuple homology measured 7.5–14.6 pp below structural `(in, out)` homology at generation 100 (seeds 42–44), and the host tuple-equality check agrees exactly with the traced `find_homologous`.
A flat XOR MSE of exactly 0.25 (every output ≈ 0.5) is reproducible for *all* families at `max_nodes=16` or with `XOR3d` at `pop_size=100`; the best genome's read-out sits at pre-activation ≈0 because its only live path ignores one input, so it is a capacity/selection wall, not an origin-gene defect.

## Test Strategy
Covered by `test/unit/genome/` (payload layout, forward vs `test/support/oracle.py`, mutate/crossover/distance invariants) and `test/parity/genome/` (kernel parity against `benchmarks/legacy_jax`).
Verification used while implementing: attr-table checks (widths, sentinels, draw shapes/ranges, identity/zero rows, self-distance 0, per-column vs whole-row crossover), a hand-built chain payload driven through `common.tensorization.derive_schedule` and compared against `test/support/oracle.forward` — default options `max|diff| 4.5e-08`, mixed aggregation×activation (including `-1` identity and `max`/`mean` neutral slots) `max|diff| 8.0e-08` — and `evoxc.grad` on the node/conn attrs (`max|AD-FD| 4.8e-04`, int-column gradients exactly 0, `-inf`/NaN source slots 0 gradient).

## Routing Table
| Path | Responsibility |
|---|---|
| `./base.py` | `GeneAttr` and `BaseGene`: table-driven attrs, `column`/`hash_int32`, batched mutate/crossover/distance/hash, `from_attrs`, host `repr`/`to_dict`. |
| `./blocks.py` | Block-shaped (whole-attribute-block) genetic operators: `init_attrs`, `mutate_attrs`, `crossover_attrs`, plus the static-operand builders `static_row`/`column_constant`/`_parameter_table`; `BaseGene`'s `new_random_attrs`/`mutate`/`crossover`/`_attr_row` are one-line delegations to it (one draw per participating column, one gate/`select`/`clip` cascade over the whole `(*shape, attr_width)` block, one `U[0,1)` block draw for the crossover pick). |
| `./node/` | Node genes: key column, hit-window aggregation, combine, activation selection. |
| `./conn/` | Conn genes: key columns (optional historical marker), per-edge message from (attrs, source value). |
