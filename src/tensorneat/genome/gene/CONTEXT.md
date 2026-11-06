# gene — attr-table driven gene families

## Intent
Define what a gene *is* (its int32 key columns and its float32 attribute columns) and how it converts, mutates, crosses over, distances and forwards, so that node and conn families are declarations over one shared mechanism instead of copy-paste.
`base.py` owns the table, the split-payload assembly and the generic operators; `blocks.py` owns the block-shaped operator kernels those operators delegate to; `node/` and `conn/` add only their structural hooks.
Nothing here imports `jax`/`torch`, and no hook on the compute path runs eager NumPy.

## Public names
`gene/__init__.py` exports exactly `BaseGene`, `GeneAttr`, `BaseNode`, `DefaultNode`, `BiasNode`, `OriginNode`, `BaseConn`, `DefaultConn`, `OriginConn` — the same nine names `genome/__init__.py` re-exports.
`base.py` additionally exposes the trace-time helpers `column`, `option_masks`, `select_cascade` (used by `node/base.py`); `algorithm/hyperneat/genes.py` imports `GeneAttr` from this module path.
`blocks.py` declares `__all__ = ["column_constant", "crossover_attrs", "init_attrs", "mutate_attrs", "static_row"]`, is imported only by `base.py`, and `column_constant`'s only caller is its own `_ColumnPool`.

## Gene payload (binding)
A gene row is **split, never packed**: `*_keys (*shape, key_width) int32` (`I_INF` on an absent row) and `*_attrs (*shape, attr_width) float32` (NaN on an absent row).
`evoxc` has no bitcast, so a mixed-dtype row is impossible: every operator twiddles int keys and float attrs separately.
`from_attrs` is the only assembly path and it **narrows rather than validates dtypes** (keys cast to `int32`, attrs to `float32`); a wrong key width, a wrong attribute shape or a key column of the wrong shape raises `ConfigError`.

## The attr table (`base.py`)
`BaseGene(StatefulBaseClass)` class data: `fixed_attrs: tuple[str, ...]` (int32 key columns), `attr_specs: tuple[GeneAttr, ...]` (float32 attribute columns in row order), `labels: dict[str, str]` (short render names; `repr`/`to_dict` fall back to the declared name).
`key_width`/`attr_width` are `len(fixed_attrs)`/`len(attr_specs)`; `length = key_width + attr_width` is a display/benchmark convenience with no compute-path reader, and its only caller is `benchmarks/spikes/forward_micro.py`.
`GeneAttr` is a frozen dataclass: `name`, `kind` (`"float"`/`"int"`), `init_mean`, `init_std`, `mutate_power`, `mutate_rate`, `replace_rate`, `lower_bound`, `upper_bound`, `options`.
- `setup(state=None)` runs `_validate_specs` (unique names across keys and attrs; a `"float"` spec carrying no `options`; an `"int"` spec carrying the contiguous list `0..n-1`; a known `kind`; `lower_bound <= upper_bound`; `mutate_rate`/`replace_rate` in `[0, 1]`) and returns the incoming state unchanged — a gene registers nothing and reads nothing back, so `node/base.py::BaseNode.setup` chains it to add its option-list check.
- `attr_index`/`key_index` are host lookups (`ConfigError` when unknown), `attr_column`/`key_column` are trace-time reads through `column`, and `has_attr(name)` is the host predicate a zero-width family needs.
- `new_random_attrs(state, key, shape)` is `blocks.init_attrs`: one draw per participating column (the rank-0 key is split once for all columns, never per row) and no draw at all for a column that degenerates to a constant (`init_std == 0`, a single-option `"int"`); `shape=()` yields the single `(attr_width,)` row, each column drawn at `(1,)`.
- `new_identity_attrs` is the "neutral by construction" row (a float at its init mean, an `"int"` at its first option) and `new_zero_attrs` the inert one (floats `0`, ints first option); both return `(attr_width,)` built as one graph constant, and a family overrides them where the neutral row is not the init distribution (`DefaultConn`'s identity weight is `1.0`, deliberately unclipped).
- `mutate(state, key, attrs)` → `blocks.mutate_attrs`, `crossover(state, key, attrs1, attrs2)` → `blocks.crossover_attrs` (one `U[0, 1)` block draw, `< 0.5` takes the first parent), and `distance(state, attrs1, attrs2)` → the per-column kernel below; all are batched over the leading axes.
- `mutate` is the only operator that validates the attribute width (`ConfigError`); `crossover`/`distance` leave a mismatched block to the evoxc op.
- Host-only: `repr`/`to_dict` (label-driven, `-1` activation renders `"identity"`) and each family's `sympy_func`.
- Every kernel takes `state` as its first parameter for the protocol and never reads it (`blocks.py` does not even take it).

## Module helpers (trace-time, used by every family)
`column(tensor, index)` is `_column_view` — one `slice(i, i+1)` view that keeps the width axis — plus one `reshape` that drops it: the shape-preserving read, because a dim-dropping integer index has a VJP that raises `evoxc.ShapeError` in the numpy backend while this form is differentiable through `evoxc.grad`.
`_column_view` alone is the slice half, for a kernel that reads several columns of one block and pays a single trailing `reshape`.
`option_masks(ids, count)` + `select_cascade(masks, branches, otherwise)` are the per-option `ids == index` masks and the `select` cascade a family builds once per forward pass, hoisting the per-level `equal`/cast into its `prepare`; an id outside the option list matches no mask and keeps `otherwise`, exactly as `common.functions.apply_activation`/`apply_aggregation` do.

## Attribute column layout (binding for every consumer)
`DefaultNode` / `OriginNode`: `bias, response, aggregation, activation` (`attr_width = 4`; ints in columns 2 and 3).
`BiasNode`: `bias, aggregation, activation` (`attr_width = 3`).
`DefaultConn` / `OriginConn`: `weight` in column 0 (`attr_width = 1`).
`-1` in an activation column means identity and is never an option index; aggregation has no `-1`.
`test/support/oracle.py` (`default_node_attrs_of`, `default_conn_attr_of`) reads exactly this order.

## Keys and sentinels
`I_INF` (`int32` max) in a key array means "no row"; NaN in an attribute array means "no value" and "unconnected node value". The two never mix and a key is never a float.
Keys are compared exactly and never by row position: hidden node keys are arbitrary and may move rows, so a consumer compares key values, never offsets.
`from_attrs` does not enforce key uniqueness (the alignment and distance kernels assume the first two key columns are unique within an individual) and does not bound a key to the schedule window — a key outside `[0, key_span)` is silently unmatched downstream, not rejected here.
`OriginConn.from_attrs` fills an omitted `historical_marker` with the flat row index `0..prod(shape)-1` reshaped to `shape`, so the markers it writes are distinct by construction.

## Zero attribute width (`attr_specs = ()`, `attr_width == 0`)
A gene may declare no float columns at all (a HyperNEAT/CPPN conn gene is the motivating case), so every `*_attrs` array is `(*shape, 0)` float32 and no attribute column can be read.
`has_attr(name)` is the host predicate; `attr_index`/`attr_column`/`key_column` raise `ConfigError` for an unknown name, and `column_of(attrs, name, default)` (declared by `node/base.py` and `conn/base.py`) returns `float(default)` when the column is absent while the no-default call raises that same `ConfigError` instead of fabricating a column.
The neutral fallbacks are fixed and shared by the traced and host paths: aggregation = option `0` (`AGG.sum`), activation = `ACT_IDENTITY`, node `bias = 0` / `response = 1`, conn `weight = 1.0`, `distance = 0`.
`init_attrs` returns `enp.zeros((*shape, 0))`, `mutate_attrs` returns its input unchanged, `crossover` returns parent 1 with no draw (`evoxc.random.uniform` rejects a zero-numel shape), `_normalize_attrs` accepts only `(*shape, 0)` or the bare `(*shape,)` spelling, and `_attr_row`/`static_row` return the empty row.
`../operations/` is width-aware for such a family: `distance/digest.py::_gather_rows` returns the empty `(P, M, 0)` block unchanged (raising `ConfigError` if it is not exactly that shape) and `distance/default.py::DefaultDistance.distance_from_alignment` skips the attribute gather and its row index, so `prepare_distance`/`measure_distance` and `Pipeline.step` score it on its structural disjoint/excess term alone.

## Formulation statics
Nothing in this node reads `kernel_mode`, `distance_centre` or `distance_formulation`, and this node carries no host|accelerator seam: the gene kernels are formulation-agnostic elementwise/random code with no sort, scatter or control flow, and the nearest seams live beside it in `../window.py` (seed splice) and `../operations/` (alignment, cycle predicate, distance).
Gene configuration lives on the instance, so no `State` entry — a formulation static included — can change a gene kernel's behaviour.

## Constraints
- **A column read is two cheap ops, and that is the floor.** This node's own measurement on the reference numpy interpreter puts a read's marginal cost at ≈0.09 ms against a `(2000, 50, 4)` block (the whole eight-read `BaseGene.distance` kernel: 0.87 ms), while `(P, max_nodes, attr_width)` block fusion is 1.9–3.3× slower per call (4× the elementwise traffic plus an innermost-axis `reduce_sum`). `x[..., i]` lowers to `constant` + `gather` (still two ops) and its VJP raises `evoxc.ShapeError`, so no one-op differentiable read exists in evoxc 0.1.0. Op count is not the cost measure for these reads — so do not "optimize" them into block passes.
- **The attribute term of `distance` is strided-read bound at the shapes it runs on.** This node's measurement at `(500, 50, 4)`, per-op medians: a view `slice` ≈9 µs, a contiguous `(P, R)` pass ≈36 µs, `abs` of a width-1 view ≈54 µs, `not_equal` of two width-1 views ≈48 µs, `add` of two width-1 views ≈65 µs, one 4-wide `subtract` ≈133 µs — so the shipped per-column form costs 0.42 ms against 0.45 (one wide `subtract`), 0.51 (one wide `abs`), 0.59 / 0.78 (a same-kind run as one block) and 1.3 (fully blocked `select` + last-axis `reduce_sum`). The rule: narrow the operands first (`subtract` produces the contiguous column), never widen a pass to save a dispatch.
- `distance` sums left to right in `attr_specs` order (the float32 order is observable): a float column is `abs(subtract(...))` of the two width-1 views, an `"int"` column is `not_equal` (never `|Δ|`: `inf - inf` is NaN where `not_equal(inf, inf)` is False), and a later `"int"` term is added **as a bool** (`evoxc.add` promotes bool to the float32 result dtype exactly, exporting a `stablehlo.convert`) while the first term is cast explicitly so the accumulator starts float32. A one-column table is its own width-1 view and takes a fast path with no slice and no accumulation.
- Attribute order in `attr_specs` *is* the column order of `*_attrs`; never reorder without changing every consumer, and consumers read `attr_specs`/`fixed_attrs`, never hardcoded indices.
- One batched draw per participating spec with the full output shape: never a per-row Python loop and never per-row key splits (evoxc keys are rank-0).
- Int attributes (option indices) live in the float32 attr array and are compared after an `int32` cast; the `-1` activation identity is resolved by the option-dispatch cascade, not by an option-table index.
- No `jax`/`torch`, no `numpy.random`, no eager NumPy on the compute path; option branches are `evoxc.select` cascades over NaN-free branches (each aggregation option neutral-fills dead slots before its reduce).
- Errors: configuration violations raise `common.errors.TensorNEATError` subclasses (`ConfigError`); evoxc shape/backend errors propagate unchanged.
- Files stay under ~1000 lines (`blocks.py` 873 and `base.py` 861 are the largest here); a new gene family adds declarations plus hooks, never a copy of the mutate/crossover/distance code — `bias.py` re-spells its response-free constructor and identity-row walk (while sharing `_option_list`/`_default_index`/`_option_spec`) because a response knob it never uses would advertise configuration that cannot reach the payload.

## Graph constants
Every population-scaled fill here is one element plus `../window.fill`'s `broadcast_to`, never a materialised `enp.full`: `column_constant` for a per-column constant, `_parameter_table` for a non-uniform per-column clip table (a uniform one is a pooled rank-0 constant), and the `_ScalarPool` rank-0 scalars.
A static `(1, N)` table, a rank-0 literal and the `(attr_width,)` identity/zero row keep their direct `evoxc.constant`/`enp.full` (`static_row` bakes one `evoxc.constant` of the row, or an empty `(0,)` row).
The choice is measured: at `P=1000`, `max_nodes=50`, `attr_width=4` the materialised block is 4.03 MB against 31 935 B baked by the broadcast form at the same per-generation time inside the box's ±30 % drift.

## Hooks other nodes rely on
`genome/base.py::initialize` uses `new_random_attrs` only; `genome/operations/mutation/base.py` uses `new_identity_attrs` (the add-node split and its identity edge) and `new_zero_attrs` (add-conn).
`genome/step.py` prepares the node and conn operands once per pass and then calls `conn_gene.forward` then `node_gene.forward` per level/tick; `grad` differentiates only the float attr arrays, which works because `column` is differentiable.
`genome/operations/**` uses `mutate`, `crossover`, `distance`, `attr_width`/`key_width` and `from_attrs`; `uses_historical_marker` is read by `mutation/base.py`, `step.py` and `algorithm/neat/neat.py`.
`algorithm/hyperneat` builds its dense substrate payload with `node_gene.from_attrs(keys)` and `conn_gene.from_attrs(in_keys, out_keys, weights)`, and reads `GeneAttr` from `gene/base.py`.
`BaseConn.from_attrs` accepts `attrs=None` (zeros), a `(*shape,)` per-connection scalar broadcast over the attribute axis, or full `(*shape, attr_width)` rows, plus an optional `historical_marker` key column.

## Origin families
`OriginNode`/`OriginConn` differ from the default families **only** in `crossover`; attr table, mutate, distance, forward and the historical marker are inherited unchanged, and the marker is key column 2 (verified row-for-row by the distance digest).
Their whole-gene pick is real: on a fixed `(P, M, A)` payload every output row equals one parent's row verbatim (`A=4` node, `A=1` conn), versus the per-column mix of `BaseGene.crossover`; for the one-column conn gene the override is distributionally identical to the inherited per-column pick, so its trajectory differences are RNG-stream differences only.
Their crossover has **no observable effect** while the algorithm pairs identical parents: `algorithm/neat/_kernels.py::pair_parents` draws both parents from `randint(0, survive)` with `survive = clamp(floor(survival_threshold * member_count), 1, …)`, so a species below `2 / survival_threshold` members gets `survive == 1` and a child equal to parent 1 for every gene family (`survival_threshold=0.01` therefore needs a species of ≥200 members, which `pop_size=100` never has; `pop_size=500`'s first generation is one 500-member species, where all three Origin variants diverge).
`OriginConn`'s marker is allocated per individual by `algorithm/neat/neat.py::_new_gene_keys`, so on an evolved population only 83–97 % of shared `(in, out)` edges carry the same marker and full-tuple homology measures 7.5–14.6 pp below `(in, out)` homology (generation 100, seeds 42–44); marker identity across genomes is the allocator's responsibility (also stated in `conn/origin.py`'s docstring).
A flat 0.25 MSE is a config/seed attractor every family can fall into (the constant-0.5 read-out wall), not an Origin-family defect; a `max_in_degree` overflow `TensorNEATError` on an Origin run belongs to `../operations/mutation`'s invariant, not here.
The families need no behavioural change.

## Unread and untested surface
`BaseConn.get_historical_marker` (the abstract raise plus `OriginConn`'s override) has no caller in `src/`, `test/` or `examples/` — `../step.py` reads `conn_gene.key_column(conn_keys, "historical_marker")` directly — so a grep hit there is not a use site.
`BaseNode.repr`/`to_dict` are pure `super()` pass-throughs.
Hashing is **host-only** (`genome/host.py::BaseGenomeHost.hash` folds `common.tools.hash_rows` over the numpy payload); this node carries no in-graph hash and emits no unsigned dtype.
`attr_width == 0` is unreachable in this tree: only `BaseGene` itself declares `attr_specs = ()` and `test/support/payloads.py` rejects a zero width, so every width-0 branch is untested as well as unused.
`_validate_specs` never checks `init_std`/`mutate_power`: a negative `init_std` reaches `evoxc.random.normal` (an opaque backend error where the table implies a `ConfigError`) while in the mutation path it only flips a sign, a negative `mutate_power` passes every check, and no test exercises any of its raise paths.
`BaseNode.self_neutral_aggregations = (AGG.sum, AGG.mean)` duplicates `common/functions/agg.py`'s `zero_on_empty` set, which also marks `AGG.maxabs`; the gene's copy is the conservative one, and the divergence costs one `any_slot` + `select` per level when `maxabs` is configured.
The class-level `attr_specs` of `DefaultNode`/`BiasNode`/`DefaultConn` is always shadowed by the instance table `__init__` builds (which repeats every constructor default as a literal); `algorithm/hyperneat/genes.py::HyperNEATConn` is the only family that reads its class table, because it never rebuilds one.
`blocks.mutate_attrs`' static-drop claim is narrower than its docstring: `_perturbs` requires `mutate_power != 0` but `perturb_live` is only `cutoff > 0` (= `mutate_rate > 0`), so a float column with `mutate_power == 0` and `mutate_rate > 0` keeps the whole perturb cascade (a zero column plus one block `select`) for a value-identical result, and `_replaces` is only `replace_rate > 0`, so a single-option `"int"` column still draws a `randint` plus a `U[0, 1)` gate it can never act on (both reachable only through a caller-supplied knob: no shipped column sets `mutate_power = 0`, and shipped option columns declare more than one option).
`replace_live` is a redundant re-derivation of `_replaces`: `cutoff > 0 and cutoff > perturb_cutoff` reduces to `replace_rate > 0` for both kinds (`"int"` columns get the `_NEVER = -1.0` perturb cutoff and so can never take the perturb branch).
`crossover_attrs` draws one **block-sized** `uniform` (`(*shape, attr_width)`) with unpooled Python-float bounds, the one place in `blocks.py` that breaks the module's own "every draw stays per column, every constant pooled" rule.
`blocks.py`'s module docstring and `_cutoffs`' still describe per-column parameters (rates, bounds) as `(*batch_shape, attr_width)` static tables broadcast over the block; the shipped `mutate_attrs` compares each column against a **rank-0** pooled scalar (`_mask_block`) and materialises a table only for clip bounds.

## Test Strategy
`test/unit/genome/` covers this node on the numpy backend with no optional extra: `test_payload.py`, `test_transform_validation.py`, `test_forward.py`, `test_recurrent.py`, `test_grad.py`, `test_mutation.py`, `test_mutation_weight.py`, `test_crossover.py`, `test_distance.py`, `test_distance_digest.py`, `test_host_helpers.py`, `test_level_order.py` (payload layout, the forward against the host oracle `test/support/oracle.py`, mutate/crossover/distance/digest invariants, grad against central differences).
`test/parity/` (`test_forward.py`, `test_grad.py`, `test_distance.py`, `test_convert.py`, `test_fitness.py`, `test_oracle_integrity.py`) compares the same kernels against `benchmarks/legacy_jax` and needs the `bench` extra.
What the unit suite pins: the attribute table (widths, sentinels, draw shapes and ranges, identity/zero rows, self-distance `0`, per-column versus whole-row crossover), the forward against `test/support/oracle.py` on hand-built chain payloads built through `common.tensorization.derive_schedule` (default options, and mixed aggregation×activation including `-1` identity and `max`/`mean` neutral slots, at `rtol` 1e-6–1e-5), and `evoxc.grad` against central differences at `h = 1e-2` with `atol` 1e-3, int-column and `-inf`/NaN source-slot gradients exactly `0`.
`test/unit/algorithm/test_genes.py` covers the HyperNEAT substrate genes (that node's concern) and the release gate `test/release_gate_no_jax.py` enforces the import hygiene this node must keep.

## Routing Table
| Path | Responsibility |
|---|---|
| `./base.py` | `GeneAttr` and `BaseGene`: table + validation, `column`/`_column_view`, `option_masks`/`select_cascade`, split-payload assembly (`from_attrs`), batched mutate/crossover/distance, identity/zero/random rows, host `repr`/`to_dict`. |
| `./blocks.py` | Block-shaped operator kernels and their static operands: `init_attrs`, `mutate_attrs`, `crossover_attrs`, `static_row`/`column_constant`/`_parameter_table`, the `_ScalarPool`/`_ColumnPool` constant pools, `_mask_block`/`_clip_block`; `BaseGene`'s `new_random_attrs`/`mutate`/`crossover`/`_attr_row` are one-line delegations to it. |
| `./node/` | Node genes: key column, hit-window aggregation, combine, activation selection. |
| `./conn/` | Conn genes: key columns (optional historical marker), per-edge message from (attrs, source value). |
| `./__init__.py` | The nine public names above. |
