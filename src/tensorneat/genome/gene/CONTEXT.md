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
- `attr_index(name)` / `key_index(name)` (host ints, `ConfigError` if unknown), `attr_column(attrs, name)` / `key_column(keys, name)` (trace-time column read through `column`), `has_attr(name)` (host bool, for the zero-width table).
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

## Zero attribute width (`attr_specs = ()`, `attr_width == 0`)
A gene may declare no float columns at all (a HyperNEAT/CPPN conn gene is the motivating case), so every `*_attrs` array is `(*shape, 0)` float32 and no attribute column can be read.
`BaseGene.has_attr(name)` is the host predicate; `attr_column`/`key_column`/`attr_index` still raise `ConfigError` for an unknown name, and `column_of(attrs, name, default)` returns `float(default)` when the column is absent and no default is given it raises that same `ConfigError` instead of fabricating a column.
The neutral fallbacks are fixed and shared by the traced and host paths: aggregation = option `0` (`AGG.sum`), activation = `ACT_IDENTITY`, node `bias = 0` / `response = 1`, conn `weight = 1.0`, `distance = 0`.
`_normalize_attrs` accepts only `(*shape, 0)` (or the bare `shape`) in this case and raises `ConfigError` for a non-empty attr array; `_attr_row` returns `static_row(())`; `blocks.crossover_attrs` early-returns `attrs1` because `evoxc.random.uniform` rejects a zero-numel shape.
`execute_mutation`/`execute_crossover` pass a `(P, max_conns, 0)` block through unchanged, and `from_attrs` accepts `None`, `(*shape, 0)` and the per-row `(*shape,)` spelling.
Still broken **outside** this node for `attr_width == 0`: `../operations/distance/digest.py::_gather_rows` and `../operations/distance/default.py::distance_from_alignment` build `enp.arange(..., step=count * width)` with `width == 0` (`ZeroDivisionError`), so `prepare_distance`/`measure_distance` — and therefore `Pipeline.step` with a zero-width gene — raise until those two guards land.

## Constraints
- **A column read is two cheap ops, and that is the floor.** `column` (`base.py`) is `slice` + `reshape`, both views: measured marginal cost ≈0.09 ms per read against a `(2000, 50, 4)` block and 0.87 ms for the whole eight-read `BaseGene.distance` kernel, while `(P, max_nodes, attr_width)` block fusion is 1.9-3.3× slower per call (4× the elementwise traffic plus an innermost-axis `reduce_sum`). `x[..., i]` lowers to `constant` + `gather` (still two ops) and its VJP raises `evoxc.ShapeError` in the numpy backend, so no one-op differentiable read exists in evoxc 0.1.0. Traced-op counts here (~250 ops per generation at `pop_size=500`, 94 % of them from `BaseGene.distance`) are not a cost measure — do not "optimize" them into block passes.
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
Their crossover has **no observable effect** when the algorithm pairs identical parents: `algorithm/neat/_kernels.py::pair_parents` draws both parents from `randint(0, survive)` with `survive = clamp(floor(survival_threshold * member_count), 1, …)`, so a species with fewer than `2 / survival_threshold` members gets `survive == 1` and `winner == loser`, making the child equal to parent 1 for every gene family.
`survive` is per species, so `survival_threshold=0.01` needs a species of ≥200 members: `pop_size=100` never has one (all four families bit-identical on seeds 42-44), but at `pop_size=500` generation 1 is a single species of 500 members (`survive=5`), so crossover runs and all three Origin variants diverge from the default (payload differs from generation 2; `OriginConn` earlier because its extra key column shifts the mutation draws).
Verified on seeds 42/43/44 with `pop_size=100, species_size=20, survival_threshold=0.01, max_nodes=50, max_conns=100, XOR`: the instrumented `father != mother` slot count is `0` in all 1 200 generation-cases, the `DefaultNode` and `OriginNode` populations are bit-identical for every generation, and all four node/conn family combinations share one fitness history; with the `survive` floor raised to `min(2, member_count)` the fraction becomes 0.50 median and the families diverge.
The marker allocator (`algorithm/neat/neat.py::_new_gene_keys`) derives new markers per individual, so on an evolved `OriginConn` population only 83–97 % of shared `(in, out)` edges carry the same marker; full-tuple homology measured 7.5–14.6 pp below structural `(in, out)` homology at generation 100 (seeds 42–44), and the host tuple-equality check agrees exactly with the traced `find_homologous`.
Acceptance measurement (pop 100, `survival_threshold=0.01`, seed 42, 150 generations, MSE = `-max_fitness`): default, node-only, conn-only and both reach 2.6e-11 at generation 100 and 1.1e-21 at generation 150 with a bit-identical per-generation series, reproduced bit-identically in a second fresh process, while seed 43 ends at 0.0189 in all four.
**Definitive reproduction of the "Origin genes learn nothing" report** (150 generations, MSE = `-max_fitness`, seed 42, values at generations 25/50/100/150).
XOR, `pop_size=500, species_size=20, max_nodes=16, survival_threshold=0.01`: default 0.192 / 0.0188 / 1.6e-5 / 1.6e-5, `OriginNode` 0.178 / 0.0118 / 0 / 0, `OriginConn` 0.159 / 0.0437 / 0.0189 / 0.0189, `OriginBoth` 0.195 / 0.0219 / 1.6e-5 / 0 -- no family stalls and none reaches 0.25.
Same config at `survival_threshold=0.1`: default 0.176 / 0.0793 / 0.0124 / 2.1e-5, `OriginNode` 0.158 / 0.0471 / 5e-8 / 0, `OriginConn` 0.184 / 0.0241 / 1.8e-5 / 1.6e-5, `OriginBoth` 0.145 / 0.0443 / 7.6e-5 / 1.6e-5.
`pop_size=100, max_nodes=50, survival_threshold=0.01`: all four bit-identical, 0.188 / 0.0078 / 0 / 0.
The literal example script (`XOR3d, species_size=10, pop_size=500, max_nodes=16, survival_threshold=0.01`): default 0.25 / 0.25 / 0.25 / 0.24999997, `OriginNode` 0.210 / 0.0152 / 8.7e-6 / 8.7e-6, `OriginConn` 0.25 / 0.243 / 0.128 / 0.115, `OriginBoth` 0.25 / 0.245 / 0.0687 / 4.6e-4.
So the reported symptom is **not reproducible in the claimed direction**: in the XOR config every family learns, and in the literal XOR3d config it is the *default* family (and `OriginBoth`) that sits at 0.25, while `OriginNode` and `OriginConn` learn.
Over seeds 42-49 the XOR config never produces a 0.25 plateau for any family: every run ends at ≤0.0189, the one-input-ignored local optimum (default 2/8 seeds, `OriginNode` 3/8, `OriginConn` 4/8, `OriginBoth` 2/8 -- no family effect).
Over seeds 42-45 in the literal XOR3d config the plateau hits default 2/4 and `OriginBoth` 3/4 but `OriginNode` 0/4 and `OriginConn` 0/4, so 0.25 is a config/seed attractor every family can fall into (the constant-0.5 read-out of the documented capacity/selection wall), not an Origin-family defect.
`OriginNode`-only is a strict no-op only while crossover is inert (pop 100); at pop 500 it diverges by design (whole-gene inheritance couples `bias/response/aggregation/activation`, which the default per-column pick does not), and `OriginConn`'s one-column whole-gene pick is distributionally identical to the default per-column pick, so its trajectory differences are pure RNG-stream differences.
The remaining reported symptoms trace to code outside this node: `algorithm/neat/_kernels.py::pair_parents` (survivor floor of 1 kills crossover when no species reaches `2 / survival_threshold` members), `algorithm/neat/neat.py::_new_gene_keys` (node keys grow past `key_span`, so an added node and its two split edges are silently unbound -- measured 17-30 % of live connection rows at generation 30+), and the per-individual marker allocator (homology, matters only once crossover is active: 3/3 seeds reach MSE < 0.01 with a population-wide marker id versus 2/3 without).
The families themselves need no behavioural change; their `crossover` is whole-gene, their `mutate`/`distance`/`forward` are the inherited and unit-tested ones.

## Test Strategy
Covered by `test/unit/genome/` (payload layout, forward vs `test/support/oracle.py`, mutate/crossover/distance invariants) and `test/parity/genome/` (kernel parity against `benchmarks/legacy_jax`).
`PYTHONPATH=src python -m pytest test/unit/genome test/parity test/release_gate_no_jax.py -q` reports 152 passed in ~31 s (95 unit/genome, 53 parity, 4 release gate), and `import tensorneat` loads none of `jax`/`torch`/`brax`/`gymnax`/`mujoco`.
Verification used while implementing: attr-table checks (widths, sentinels, draw shapes/ranges, identity/zero rows, self-distance 0, per-column vs whole-row crossover), a hand-built chain payload driven through `common.tensorization.derive_schedule` and compared against `test/support/oracle.forward` — default options `max|diff| 4.5e-08`, mixed aggregation×activation (including `-1` identity and `max`/`mean` neutral slots) `max|diff| 8.0e-08` — and `evoxc.grad` on the node/conn attrs (`max|AD-FD| 4.8e-04`, int-column gradients exactly 0, `-inf`/NaN source slots 0 gradient).

## Routing Table
| Path | Responsibility |
|---|---|
| `./base.py` | `GeneAttr` and `BaseGene`: table-driven attrs, `column`/`hash_int32`, batched mutate/crossover/distance/hash, `from_attrs`, host `repr`/`to_dict`. |
| `./blocks.py` | Block-shaped (whole-attribute-block) genetic operators: `init_attrs`, `mutate_attrs`, `crossover_attrs`, plus the static-operand builders `static_row`/`column_constant`/`_parameter_table`; `BaseGene`'s `new_random_attrs`/`mutate`/`crossover`/`_attr_row` are one-line delegations to it (one draw per participating column, one gate/`select`/`clip` cascade over the whole `(*shape, attr_width)` block, one `U[0,1)` block draw for the crossover pick). |
| `./node/` | Node genes: key column, hit-window aggregation, combine, activation selection. |
| `./conn/` | Conn genes: key columns (optional historical marker), per-edge message from (attrs, source value). |
