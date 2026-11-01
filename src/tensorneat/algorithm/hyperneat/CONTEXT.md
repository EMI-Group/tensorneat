# HyperNEAT (`tensorneat.algorithm.hyperneat`)

## Intent
HyperNEAT evolves only a CPPN (an ordinary NEAT network from `../neat`) and derives a *substrate* network from it: the CPPN is queried once per substrate connection at the coordinates of that connection's two endpoints, and its single scalar output becomes that connection's weight.
The substrate topology is fixed at construction, so this node turns a whole CPPN population into one batch-first execution schedule of a dense, identical-shaped phenotype network — no padding, no NaN rows.
Three things live here: the decode (CPPN → weights → gene payload → schedule), the two concrete variants (`HyperNEAT` recurrent, `HyperNEATFeedForward` feedforward), and the two substrate gene kernels (`HyperNEATNode`, `HyperNEATConn`).
All evolution, speciation and `randkey` consumption stay in `../neat`; the substrate object is a host-side geometry table and never enters `State`.

## API Surface
`hyperneat.py`: `SubstrateSchedule`, `HyperNEAT(BaseAlgorithm)`, `HyperNEATNode(BaseNode)`, `HyperNEATConn(BaseConn)`.
`HyperNEAT.__init__(self, substrate, neat, weight_threshold=0.3, max_weight=5.0, aggregation=AGG.sum, activation=ACT.sigmoid, activate_time=10, output_transform=ACT.sigmoid)`.
`HyperNEAT` implements the full `../base.py` protocol — `setup`, `ask`, `transform`, `forward`, `tell`, `show_details`, `num_inputs`, `num_outputs`, `pop_size` — by delegating evolution to `self.neat` and decoding with `self.hyper_genome`.
`setup(state=None)` chains `self.neat.setup(state)` then `self.hyper_genome.setup(state)`; the substrate is stateless and has no `setup`.
`ask(state)` returns the CPPN payload untouched (`self.neat.ask`) at zero op cost; the phenotype is not part of the evolved population.
`tell`, `show_details` and `pop_size` delegate to `self.neat`; `pop_size` is a read-only property, not an instance attribute.
`num_inputs` = `substrate.num_inputs - 1` (trailing bias removed), `num_outputs` = `substrate.num_outputs`.
Subclass hooks used by the feedforward variant: class attrs `_genome_cls` (executor genome class) and `_required_connection_type`, plus `_genome_extra_kwargs()` (the recurrent `activate_time`).
`hyperneat_feedforward.py`: `HyperNEATFeedForward(HyperNEAT)` with `_genome_cls = DefaultGenome`, `_required_connection_type = "feedforward"`, `_genome_extra_kwargs() -> {}`, and an `__init__` that merely drops `activate_time` from the signature; every kernel and every protocol method is inherited.
`__init__.py` re-exports `HyperNEAT`, `HyperNEATFeedForward`, `SubstrateSchedule`, `HyperNEATNode`, `HyperNEATConn`, `BaseSubstrate`, `DefaultSubstrate`, `FullSubstrate`, `MLPSubstrate`.

## SubstrateSchedule (the object `transform` returns)
`SubstrateSchedule(NamedTuple)` = `(schedule, node_keys, node_attrs, conn_keys, conn_attrs)`: the substrate genome's `Transformed` **plus** the dense gene payload it executes.
It exists because `genome.forward` needs the payload while the algorithm protocol carries exactly one `transformed` object through `problem.evaluate`; the base protocol's `transform -> schedule` return alone cannot express it.
Every leaf of the container keeps the population on axis 0, so `problem.static_batch_size` and any `evoxc.tree_map`/`tree_leaves` walk still see a plain batch-first pytree.
Properties `level` and `overflow` delegate to `self.schedule`; `overflow` must stay, because the pipeline reads it to decide on an executable rebuild.

## CPPN → Substrate Decode Algebra
Static sizes, all fixed at construction: `P = neat.pop_size`, `Q = substrate.conns_cnt`, `N = substrate.nodes_cnt`, `I = substrate.num_inputs` (substrate inputs *including* the bias), `O = substrate.num_outputs`, `C = neat.num_inputs = substrate.query_coors.shape[1] = 2 * CD` (CPPN input width), `NA = node_gene.attr_width`, `CK = conn_gene.key_width`, `CA = conn_gene.attr_width`.
`I` (substrate input neurons) and `C` (CPPN query width) are different numbers and must never be interchanged: `I` sizes the bias column of `forward`, `C` sizes the coordinate rows of `query_coors`.
`HyperNEAT.transform(state, pop)` performs, in order:
1. `query_res (P, Q) = _query_substrate(state, pop)` — `neat.transform(state, pop)` gives the CPPN schedule, `_expand_schedule` repeats every population-axis leaf `Q` times, `_query_coors` builds the `(P, Q, C)` grid, the grid is flattened to `(P * Q, C)`, and one `neat.genome.forward` call returns `(P * Q, 1)` reshaped to `(P, Q)`.
2. `weights (P, Q) = _weights_from_query(query_res)` — dead zone, continuity shift and scaling into `[-max_weight, max_weight]`.
3. `payload = _assemble_substrate(weights)` — a `genome.utils.GenomeTensors` of the four split arrays: `node_keys (P, N)`, `node_attrs (P, N, NA)`, `conn_keys (P, Q, CK)`, `conn_attrs (P, Q, CA)`.
4. `schedule = hyper_genome.transform(state, payload.node_keys, payload.conn_keys)` — the substrate genome derives its own schedule; its layout is genome-owned and opaque to this node.
5. `return SubstrateSchedule(schedule, *payload)`.
`HyperNEAT.forward(state, transformed, inputs)` takes `(B, I - 1)`, appends a trailing column of ones (`B = inputs.shape[0]`, a static int — problems tile datasets over the population axis so `B` is `pop_size * samples`) and returns `(B, O)` from `hyper_genome.forward(state, schedule, node_keys, node_attrs, conn_keys, conn_attrs, biased)`.
The bias is the **last** substrate input, which is why `num_inputs` reports `substrate.num_inputs - 1` upward.
The expansion row order is `p * Q + q` (individual slowest) and matches the `(P, Q, C) -> (P * Q, C)` reshape; changing one without the other silently permutes weights across connections.
`neat.pop_batch_size` does not chunk this pass: the CPPN query is a single whole-population forward over the static `(P, Q, C)` grid, so peak memory scales with `pop_size * conns_cnt`.
A shape change (`pop_size`, `conns_cnt`, `nodes_cnt`) changes the traced graph, so the pipeline rebuilds the executable; nothing in this node caches per-shape state.

## Gene Contract (split payload, matching `genome/gene`)
`HyperNEATNode(BaseNode)` declares `fixed_attrs = ("index",)` and `attr_specs = ()` → `key_width == 1`, `attr_width == 0`: a substrate node carries no evolvable attributes because aggregation and activation are fixed at construction.
`HyperNEATNode.__init__(aggregation=AGG.sum, activation=ACT.sigmoid)` stores both callables; they are shared by every individual and never enter `State`.
`HyperNEATNode.forward(state, attrs, messages, is_output_node, valid_mask)` — the parameter order is `BaseNode.forward`'s, which is the order the genome calls positionally: `attrs` accepted and never read, `messages (..., K)` already-gathered incoming values, `valid_mask (..., K)` bool, `is_output_node` broadcastable bool.
The body is `z = evoxc.select(valid_mask, messages, 0.0)`, `aggregated = aggregation(z, valid_mask)`, `activated = activation(aggregated)`, `return evoxc.select(is_output_node, aggregated, activated)`: output nodes are aggregated but never activated (the substrate's output activation is `output_transform`), and a heterogeneous batch can never use `evoxc.cond`.
`HyperNEATConn(BaseConn)` inherits `fixed_attrs = ("input_index", "output_index")` and declares one `GeneAttr(name="weight", kind="float", init_mean=0.0, init_std=1.0, mutate_power=0.15, mutate_rate=0.8, replace_rate=0.1)` → `key_width == 2`, `attr_width == 1`.
`HyperNEATConn.__init__` resolves `self.weight_index` from the `attr_specs` names, never from an offset, so adding a spec cannot silently shift the weight column.
`HyperNEATConn.forward(state, attrs, source_values)` returns `source_values * weight`, and the genome hands it the **`ConnOperands` bundle** built once per forward pass by `BaseGenome._hit_conn_attrs` (`genome/base.py:1084` passes `hit_conn_attrs.conn`), i.e. a NamedTuple of already NaN-cleaned `(..., K)` columns rather than a raw `(..., K, CA)` tensor — the weight column must come from `attrs.columns[self.weight_index]`.
The real `BaseNode` contract also requires non-empty `aggregation_options`/`activation_options` class tuples (`genome/gene/node/base.py:111`) and `aggregation`/`activation` columns in `node_gene.attr_specs` (`genome/gene/node/base.py:204` via `prepare`), which `HyperNEATNode`'s empty `attr_specs` does not satisfy today (see Known Issues).
The conn kernel does no NaN guarding: dead slots are masked once by the node kernel's `valid_mask`, so masking has a single owner and the multiply stays one op.
Both genes implement the cheap host-only `repr`/`to_dict` hooks (`{"idx": …}` and `{"in","out","weight"}`) so `genome.network_dict`/`repr` work on a substrate genome.

## Substrate Protocol (host-only, pinned in `./substrate`)
`BaseSubstrate` is a plain host object — no `State`, no `setup`, never traced — exposing `connection_type` (`"feedforward"` / `"recurrent"`), `num_inputs` (input neurons *including* the trailing bias), `num_outputs`, `nodes_cnt`, `conns_cnt`, `query_coors (conns_cnt, 2*CD)` float32, `nodes (nodes_cnt,)` int32 and `conn_index (conns_cnt, 2)` int32.
Row `q` of `query_coors` must correspond 1:1 with row `q` of `conn_index`; the substrate's `connection_type` selects the executor genome (`recurrent` → `HyperNEAT`, `feedforward` → `HyperNEATFeedForward`).
Node keys must be the block order `inputs (0..I-1)`, `outputs (I..I+O-1)`, `hidden` — the genome's `is_output_node` test and `input_idx`/`output_idx` ranges depend on it, and `HyperNEAT._validate` re-checks it instead of letting it surface as a silently disabled activation.
This node reads only those tables: it never calls `make_nodes`/`make_conns`, never inspects coordinate ordering, and never lets the substrate layer know a gene column order.

## Decode helpers (implemented surface of `hyperneat.py`)
`_validate()` is pure host Python/NumPy over the configuration: CPPN query width (legacy wording "Query coors of Substrate should be equal to NEAT input size"), `0 <= weight_threshold < 1`, `max_weight > 0`, `substrate.num_inputs >= 2`, `activate_time >= 1`, `neat.num_outputs == 1`, `query_coors.shape[0] == conns_cnt`, `conn_index.shape == (conns_cnt, 2)`, `nodes.shape == (nodes_cnt,)` and a permutation of `0..nodes_cnt-1`, the node-key block order, and `_required_connection_type`.
`_make_hyper_genome()` builds `_genome_cls(num_inputs=substrate.num_inputs, num_outputs=substrate.num_outputs, max_nodes=substrate.nodes_cnt, max_conns=substrate.conns_cnt, node_gene=HyperNEATNode(aggregation, activation), conn_gene=HyperNEATConn(), output_transform=output_transform, **_genome_extra_kwargs())`; the mutation/crossover/distance operators fall back to the genome defaults and are never invoked, because the substrate payload is synthesised, not evolved, and no `max_in_degree`/`max_depth` static is passed (see Known Issues).
`_graph_constant(array)` is `evoxc.constant(evoxc.from_numpy(np.ascontiguousarray(array)))`; all substrate tables together are a few KiB, far under the 1 MiB `EVOXC_LARGE_CONSTANT_BYTES` warning threshold, and `__init__` never calls it (no graph work outside a trace).
`_query_coors()` reshapes the `(Q, C)` constant to `(1, Q, C)` and `enp.broadcast_to` it to `(P, Q, C)` — never a Python loop over `P` and never `evoxc.tile` (no batching rule).
`_expand_schedule(transformed, n_query)` is `evoxc.tree_map` over the CPPN schedule with `_expand_leaf`; `_expand_leaf` rewrites a leaf whose static axis 0 is `pop_size` into `(P * n_query, *leaf.shape[1:])` via `reshape → broadcast_to → reshape`, and returns any leaf without that axis unchanged (the docstring's "shared ordering table" case).
`_query_substrate` runs the query with `self.neat.genome.forward(state, expanded, *expanded_pop, queries)`; `*expanded_pop` relies on the `GenomeTensors` field order `(node_keys, node_attrs, conn_keys, conn_attrs)` matching `genome.forward`'s parameter order.
`_weights_from_query` is `sign(w) * max(|w| - weight_threshold, 0) * max_weight / (1 - weight_threshold)` built from `enp.abs`/`enp.greater`/`evoxc.select`/`enp.subtract`/`enp.sign`/`enp.multiply`; `sign(0) == 0` and `max(|w| - t, 0) == 0` for `|w| <= t`, so a tie with the threshold lands on `0` (a pruned edge), and NaN stays NaN by contract.
`_assemble_substrate(weights)` returns a `GenomeTensors`: `node_keys` is the broadcast `(N,)` constant, `node_attrs` has one column per `node_gene.attr_specs` entry (all zeros; `int` specs store their option index as float32 and index `0` is the first configured option) or an explicit `(P, N, 0)`, `conn_keys` takes columns 0/1 from the `(Q, 2)` `conn_index` constant and fills any further key column (`OriginConn`'s `historical_marker`) with the unique row id `enp.arange(1, Q + 1)`, and `conn_attrs` puts `weights` in the `"weight"` column with zeros elsewhere.
`_attr_columns` derives column positions from `attr_specs` names and raises `TensorNEATError` when a conn gene has no `"weight"` column (the decoded weights would be dropped).

## Divergences from the Legacy JAX Baseline (deliberate fixes)
1. `HyperNEATNode.forward` was signature-incompatible with the gene protocol (missing `valid_mask`, and the two legacy genomes passed it differently) — it now takes `(state, attrs, messages, is_output_node, valid_mask)` positionally, the `BaseNode.forward` order both genomes call.
2. `HyperNEATNode.forward` called `self.aggregation(inputs)` with one argument while every `AGG` member is `(z, mask)` — it now masks and aggregates through the two-argument contract, which also implements the missing invalid-source masking.
3. `jax.lax.cond(is_output_node, ...)` became `evoxc.select(is_output_node, aggregated, activated)`: `cond` needs a scalar predicate, the batch is heterogeneous, and evoxc forbids control flow inside vectorized contexts.
4. `vmap(self.neat.forward, in_axes=(None, None, 0))` over query rows became `_expand_schedule` + the flattened `(P * Q, C)` forward — evoxc has no `vmap` over region-bearing kernels.
5. `pop_size` was assigned as an instance attribute while the base protocol declares a property — it is now the property `neat.pop_size`.
6. Constructor `assert`s became `TensorNEATError` raises, and validation gained `0 <= weight_threshold < 1`, `max_weight > 0`, `substrate.num_inputs >= 2`, `activate_time >= 1`, `neat.num_outputs == 1`, the substrate table row counts and the node-key block order.
7. A multi-output CPPN used to misalign weights silently (`query_res.flatten()` onto one column) — it is now rejected at construction.
8. `HyperNEATFeedForward` no longer re-implements the constructor body or the substrate assertions: it overrides two class attributes, `_genome_extra_kwargs()` and the public `__init__` signature, so the two variants cannot drift.
9. The legacy single gene row `[in, out, weight]` cannot exist under evoxc (no bitcast), so the decode emits the four split payload arrays and the weight column is found by name.

## Dependencies (cross-node contracts)
`../neat`: `setup/ask/tell/show_details`, properties `pop_size`, `num_inputs`, `num_outputs`, `transform(state, pop)`, and the payload pytree `ask` returns (a `GenomeTensors`).
`../neat`'s **`.genome` attribute** is required by `_query_substrate` (see Known Issues).
`../../genome`: `RecurrentGenome(...)` (with `activate_time`) and `DefaultGenome(...)`, both constructed as `(num_inputs=substrate.num_inputs, num_outputs=substrate.num_outputs, max_nodes=substrate.nodes_cnt, max_conns=substrate.conns_cnt, node_gene=..., conn_gene=..., output_transform=..., [activate_time=...])`.
`../../genome`: `transform(state, node_keys, conn_keys) -> Transformed` and `forward(state, transformed, node_keys, node_attrs, conn_keys, conn_attrs, inputs (B, I)) -> (B, O)`, with `node_gene`/`conn_gene` attribute access and `GeneAttr`/`BaseNode`/`BaseConn` base classes.
`../../genome.utils.GenomeTensors`: the payload NamedTuple whose field order is the `*payload` unpacking contract.
`../../common`: `State`, `functions.{ACT, AGG}` (aggregation callables are `(z, mask)`), `errors.TensorNEATError`, `tools.to_numpy` (host-only gene `repr`/`to_dict`).
`../base.py`: `BaseAlgorithm`, including `show_details` as the only host-side member.
`problem`: `evaluate` reads only `static_batch_size(transformed)`, so the `SubstrateSchedule` container is legal as long as every leaf keeps axis 0 = population.
`problem`'s `FuncFit.show` does **not** honour that: it rebuilds the schedule with `schedule_type(*evoxc.tree_leaves(transformed))` (`problem/func_fit/func_fit.py:224`), which assumes a flat one-level NamedTuple and therefore breaks on the nested `SubstrateSchedule` (see Known Issues).
`pipeline.py`: creates `randkey`; this node seeds nothing and draws no randomness.

## Constraints
No `jax` / `torch` / `numpy.random` on the compute path; `numpy` is used only in `__init__`-time validation, for `np.ascontiguousarray` before `evoxc.from_numpy`, and in host-only `repr`/`to_dict`.
No (un)vectorization of any kind: no `vmap`, no `fori_loop`, no `switch`; this node contains no `cond`/`while_loop`/`scan` of its own and every kernel is region-free and batch-first.
No `.at[...]` updates, no `tile` on batched data (evoxc's `tile` has no batching rule), no `repeat`, no `max/sum(where=…)`, no `any`/`all`: mask with `evoxc.select`, reduce afterwards.
`evoxc` `getitem` is static-only and rejects `Ellipsis`, so column selection goes through `_attr_column`'s explicit `(slice(None),) * (ndim - 1) + (index,)` key.
Python `if`/`for` over constructor configuration is legal and unrolls at trace time; branching on traced values is a `TraceError`.
All shapes must be static: `P`, `Q`, `N`, `I`, `O`, `NA`, `CK`, `CA` come from configuration, never from a runtime tensor shape.
Nothing here mutates `State`, the substrate, the genome objects, or any array in place; the substrate geometry is read-only.

## Known Issues
**This node cannot run end-to-end against the real `genome`/`problem` today**; a minimal `Pipeline(HyperNEAT(FullSubstrate(...), NEAT(DefaultGenome(...), pop_size=8, species_size=2)), XOR3d()).setup()` fails in `hyper_genome.setup` before any graph is built.
1. `HyperNEATNode` (hyperneat-owned): `BaseNode.setup` rejects the empty `aggregation_options`/`activation_options` class tuples with `ConfigError` (`genome/gene/node/base.py:111`, first failure — `hyperneat.py:249` → `genome/base.py:632`).
2. `HyperNEATNode` (hyperneat-owned): with `setup` bypassed, the first forward trace dies in `BaseNode.prepare` — `attr_specs == ()` declares no `aggregation`/`activation` column (`genome/gene/node/base.py:204` → `genome/gene/base.py:278`), because `_node_step` calls `node_operands` → `node_gene.prepare(node_attrs, output_rows)` (`genome/base.py:1087,314`).
   The fix shape is `DefaultNode`'s: declare `aggregation`/`activation` `GeneAttr(kind="int", options=(0,))` columns plus one-element option tuples built from the constructor callables, since `_assemble_substrate` already paints every `attr_specs` column with `0` (option index 0).
3. `HyperNEATConn.forward` (hyperneat-owned): `_attr_column(attrs, ...)` assumes a tensor with a last axis, but the genome passes `ConnOperands` → `AttributeError: 'ConnOperands' object has no attribute 'shape'` (`hyperneat.py:955` vs `genome/base.py:1084`).
4. `_make_hyper_genome` (hyperneat-owned): it never passes `max_in_degree`, so a dense substrate inherits the genome default `max(initial in-degree, 4)` and the pipeline aborts with "the execution schedule dropped N connection(s)" (`pipeline.py:816`); `FullSubstrate` with 3 hidden neurons needs `K = 6`. Derive it from `bincount(substrate.conn_index[:, 1])`.
   `max_depth` is likewise unset (default 16) and is **not** covered by `Transformed.overflow`, so a feedforward substrate deeper than 16 silently leaves its tail `NaN`.
5. `problem.FuncFit.show` (problem-owned): `schedule_type(*evoxc.tree_leaves(transformed))` (`problem/func_fit/func_fit.py:224`) rebuilds a flat NamedTuple, so `Pipeline.show()` on a `SubstrateSchedule` raises `TypeError: SubstrateSchedule.__new__() takes 6 positional arguments but 17 were given`; the generic spelling is `evoxc.tree_unflatten(evoxc.tree_structure(transformed), list(arrays))`.
Inherited from `../neat`, not fixable here: `NEAT._new_gene_keys` reshapes the `(P,)`-returning `genome.max_node_key`/`max_conn_marker` to `()` (`neat.py:572-590`), which breaks `NEAT.tell` — and therefore `HyperNEAT.tell` — with `evoxc.ShapeError: reshape: element counts differ: 8 vs 1`; and `NEAT.show_details` calls `genome.node_exists(cfg, node_keys)`/`conn_exists(cfg, conn_keys)` with two arguments (`neat.py:752-753`), so `HyperNEAT.show_details` raises `TypeError` as well.
With those five blockers monkeypatched out, both variants complete `setup()` and two `step()` generations on the numpy backend (recurrent `FullSubstrate` and feedforward `MLPSubstrate`), so the decode algebra, the `(P·Q)` query expansion and the pipeline's tiled `(P*S)` batch are exercised and correct.
`NEAT.forward(state, transformed, inputs)` cannot serve the `(P * Q)` query batch because it reads the un-expanded population from `state`; the decode therefore calls `neat.genome.forward` directly and depends on the `NEAT.genome` attribute plus the `GenomeTensors` field order. If `../neat` renames the attribute or changes the payload arity, `_query_substrate` breaks first.
`hyperneat.py` is ~1015 lines, of which the majority is protocol docstring; the decode itself is ~370 lines. It is kept in one file because the decode algebra, the protocol delegation and the two gene kernels are one contract (the split row order); if it must shrink, move `HyperNEATNode`/`HyperNEATConn` to `genes.py` first.

## Test Strategy
Tests live in the sibling `./../../../test` tree (read-only from here) and mirror this node.
`test/unit/algorithm/hyperneat/test_weights.py`: the weight map against hand-computed values at `0`, `±weight_threshold`, `±(weight_threshold + ε)` and `±large`, plus NaN-in → NaN-out.
`test/unit/algorithm/hyperneat/test_node_gene.py`: masked aggregation with mixed `valid_mask` (absent edges contribute `0`), output nodes aggregated but not activated, and a `(P, N, K)` batch agreeing with a per-individual loop.
`test/unit/algorithm/hyperneat/test_conn_gene.py`: `source * weight` column selection under a batched `attrs (..., K, CA)`.
`test/unit/algorithm/hyperneat/test_assembly.py`: `_assemble_substrate` shapes and column alignment against `substrate.conn_index`/`substrate.nodes`, `_expand_leaf` row order `p * Q + q`, and the `(P, Q) -> (P * Q)` reshape agreement with `_query_coors`.
`test/unit/algorithm/hyperneat/test_validate.py`: every `TensorNEATError` condition, including the multi-output CPPN and a non-feedforward substrate for `HyperNEATFeedForward`.
`test/parity/algorithm/hyperneat/`: decode parity against `benchmarks/legacy_jax` for the same CPPN population (identical weights, node keys and connection endpoints), allowing only the documented divergences; the tests require the `bench` extra.
Integration: `examples/func_fit/xor_hyperneat.py` and `xor_hyperneat_feedforward.py` reach their fitness target with a fixed seed, deterministically, on the numpy backend.

## Notes for Agents
Every shape above is static: derive it from `self.neat`/`self.substrate`/`self.hyper_genome`, never from a traced tensor shape.
`_validate` runs before `_make_hyper_genome`, so a bad configuration raises before any genome is constructed; keep that order.
`_validate` checks `weight_threshold`/`max_weight`/`activate_time` with `_require_real` (rejecting `bool`, which is an `int` subclass) before the range checks, so a non-numeric value raises `TensorNEATError`, never `TypeError`; NumPy scalars pass.
The substrate tables are host data: read them at trace time through `_graph_constant`, never in `__init__`.
If a substrate ever needs a table over ~1 MiB, stop in-graph and escalate: the coordinates would have to become an explicit executable input, which is a `pipeline.py` change.

## Routing Table
| Path | Responsibility |
| --- | --- |
| `./hyperneat.py` | `SubstrateSchedule`, `HyperNEAT` (decode algebra + protocol delegation), the `_validate`/`_query_coors`/`_expand_schedule`/`_expand_leaf`/`_query_substrate`/`_weights_from_query`/`_assemble_substrate`/`_attr_columns`/`_conn_keys` helpers, `HyperNEATNode` and `HyperNEATConn`. |
| `./hyperneat_feedforward.py` | `HyperNEATFeedForward`: the feedforward variant delta only (`_genome_cls`, `_required_connection_type`, `_genome_extra_kwargs`, `__init__` signature). |
| `./substrate` | Host-only substrate geometry (`BaseSubstrate`, `DefaultSubstrate`, `FullSubstrate`, `MLPSubstrate`): coordinate → node/connection analysis, `connection_type` contract, `query_coors`/`conn_index` tables, shared geometry helpers. |
| `./__init__.py` | Re-exports the two algorithms, `SubstrateSchedule`, the two gene kernels and the four substrate classes. |
