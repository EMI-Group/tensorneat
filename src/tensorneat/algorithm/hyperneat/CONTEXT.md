# HyperNEAT (`tensorneat.algorithm.hyperneat`)

## Intent
HyperNEAT evolves only a CPPN (an ordinary network from `../neat`) and derives a *substrate* network from it: the CPPN is queried once per substrate connection at the coordinates of that connection's two endpoints, and its single scalar output becomes that connection's weight.
The substrate topology is fixed at construction, so this node turns a whole CPPN population into one batch-first execution schedule of a dense, identical-shaped phenotype network — no padding, no NaN rows.
Four things live here: the decode (CPPN → weights → gene payload → schedule), the two concrete variants (`HyperNEAT` recurrent, `HyperNEATFeedForward` feedforward), the two substrate gene kernels (`HyperNEATNode`, `HyperNEATConn`, in `./genes.py`), and the host-only substrate geometry package in `./substrate`.
All evolution, speciation and `randkey` consumption stay in `../neat`; the substrate object is host-side geometry that never enters `State`.

## API Surface
`hyperneat.py` (930 lines): `SubstrateSchedule`, `HyperNEAT(BaseAlgorithm)`, and the `HyperNEATNode`/`HyperNEATConn` re-export from `./genes.py`.
`genes.py` (297 lines): `HyperNEATNode(BaseNode)`, `HyperNEATConn(BaseConn)`, plus the host-only helpers only they use (`_key_int`, `_attr_float`, `_require_callable`, the `_NODE_OPTION_SPECS` table).
`hyperneat_feedforward.py` (113 lines): `HyperNEATFeedForward(HyperNEAT)`, a delta file with no kernels of its own.
`__init__.py` re-exports `HyperNEAT`, `HyperNEATFeedForward`, `SubstrateSchedule`, `HyperNEATNode`, `HyperNEATConn`, `BaseSubstrate`, `DefaultSubstrate`, `FullSubstrate`, `MLPSubstrate` and lists exactly those nine in `__all__`; `../__init__.py` re-exports only the two algorithms.
`HyperNEAT.__init__(self, substrate, neat, weight_threshold=0.3, max_weight=5.0, aggregation=AGG.sum, activation=ACT.sigmoid, activate_time=10, output_transform=ACT.sigmoid)`; `HyperNEATFeedForward.__init__` is the same signature minus `activate_time`.
`setup(state=None)` chains `self.neat.setup(state)` then `self.hyper_genome.setup(state)`; the substrate is stateless and has no `setup`.
`ask(state)` returns `self.neat.ask(state)` untouched (zero ops): the phenotype is not part of the evolved population.
`transform(state, pop) -> SubstrateSchedule`, `forward(state, transformed, inputs)`, `prepare_forward(state, transformed, pop=None, samples=1)`, `forward_prepared(state, transformed, prepared, inputs)` are the decode/execute members; `tell` and `show_details` delegate to `self.neat`; `pop_size` is the read-only property `neat.pop_size`, never an instance attribute.
`num_inputs` = `substrate.num_inputs - 1` (trailing bias removed), `num_outputs` = `substrate.num_outputs`.
Subclass hooks: class attrs `_genome_cls` (executor genome class) and `_required_connection_type`, plus `_genome_extra_kwargs()` (the recurrent `activate_time`).
`HyperNEATFeedForward` sets `_genome_cls = DefaultGenome`, `_required_connection_type = "feedforward"`, `_genome_extra_kwargs() -> {}`, and an `__init__` that only drops `activate_time`; every kernel and protocol method is inherited, so the two variants cannot drift.
The feedforward variant still carries the inherited `activate_time = 10` attribute (the base `__init__` always assigns it): its genome ignores it, but `_substrate_signature` reads it.

## SubstrateSchedule (the object `transform` returns)
`SubstrateSchedule(NamedTuple)` = `(schedule, node_keys, node_attrs, conn_keys, conn_attrs)`: the substrate genome's `Transformed` **plus** the dense gene payload it executes.
It exists because `genome.forward` needs the payload while the base protocol carries exactly one `transformed` object from `algorithm.transform` through `problem.evaluate` into `algorithm.forward`/`forward_prepared`.
Every leaf keeps the population on axis 0, including the leaves of the nested `schedule`, so `problem.static_batch_size` and any `evoxc.tree_map`/`tree_leaves`/`_slice_batch` walk still sees a plain batch-first pytree (this is what lets the chunked func-fit evaluate slice a `SubstrateSchedule`).
Properties `level` and `overflow` delegate to `self.schedule`; nothing in the tree reads them — `host_payload._schedule_overflow` takes `edge_overflow` directly and then the nested `getattr(transformed.schedule, "edge_overflow")` fallback — so they are convenience aliases for direct callers only.

## CPPN → Substrate Decode Algebra
Static sizes: `P` = the **payload's** leading dimension (the CPPN's `pop_size` during an evolution step, `1` for a `Pipeline.show` decode, any static batch is legal); `Q = substrate.conns_cnt`; `N = substrate.nodes_cnt`; `I = substrate.num_inputs` (substrate inputs *including* the bias); `O = substrate.num_outputs`; `C = neat.num_inputs = substrate.query_coors.shape[1] = 2 * CD`; `NA = node_gene.attr_width = 2`; `CK = conn_gene.key_width = 2`; `CA = conn_gene.attr_width = 1`, all fixed at construction.
`I` (substrate input neurons) and `C` (CPPN query width) are different numbers and must never be interchanged: `I` sizes the bias column of `forward`, `C` sizes the coordinate rows of `query_coors`.
Every decode helper that tiles a batch takes `pop_size` as a static argument (`_query_coors(pop_size)`, `_expand_schedule(transformed, n_query, pop_size)`, `_expand_leaf(leaf, n_query, pop_size)`) and `_assemble_substrate` reads it from `weights.shape[0]`; none of them may fall back to `self.pop_size`.
`HyperNEAT.transform(state, pop)` performs, in order:
1. `query_res (P, Q) = _query_substrate(state, pop)` — `neat.transform(state, pop)` gives the CPPN schedule, `_expand_schedule` repeats every population-axis leaf `Q` times, `_query_coors` builds the `(P, Q, C)` grid, the grid is flattened to `(P * Q, C)`, and one `neat.genome.forward` call returns `(P * Q, 1)` reshaped to `(P, Q)`.
2. `weights (P, Q) = _weights_from_query(query_res)` — dead zone, continuity shift and scaling by `max_weight / (1 - weight_threshold)`; the map is unclipped, so `max_weight` is reached at `|query| == 1` and a larger CPPN output grows proportionally beyond it.
3. `payload = _assemble_substrate(weights)` — a `genome.utils.GenomeTensors` of the four split arrays: `node_keys (P, N)`, `node_attrs (P, N, 2)`, `conn_keys (P, Q, 2)`, `conn_attrs (P, Q, 1)`.
4. `schedule = hyper_genome.transform(state, payload.node_keys, payload.conn_keys, input_keys=tuple(range(I)), output_keys=tuple(range(I, I + O)))` — only the two key arrays go in, and the substrate genome derives its schedule from the substrate's *actual* key vectors, never from its default layout; the rest of the schedule's layout is genome-owned and opaque to this node.
5. `return SubstrateSchedule(schedule, *payload)`.
**Read-out law.** `schedule.output_slot` is the authority for the phenotype's outputs: passing `output_keys=I..I+O-1` makes it name the substrate's output neurons, and `genome/step.py` gathers the network outputs at `output_slot` and builds the per-row output-activation mask from it — never "the last `O` payload rows", which are hidden rows whenever the substrate has hidden neurons.
`input_keys=0..I-1` is the first payload row range, exactly the rows `genome.forward` seeds with the network inputs, so input injection keeps its invariant; `input_slot` only pins the levels.
`HyperNEAT.forward(state, transformed, inputs)` takes `(B, I - 1)`, appends a trailing column of ones (`B = int(inputs.shape[0])`, normally `pop_size * samples`) and returns `(B, O)` from `hyper_genome.forward(state, transformed.schedule, *payload rows, biased)`.
The bias is the **last** substrate input, which is why `num_inputs` reports `substrate.num_inputs - 1` upward; the ones column is sized from the query's own row count, never from `pop_size`, because the genome expands the payload's population axis to `B` internally.
The expansion row order is `p * Q + q` (individual slowest) and matches the `(P, Q, C) -> (P * Q, C)` reshape; changing one without the other silently permutes weights across connections.
`neat.pop_batch_size` does not chunk this pass: the CPPN query is a single whole-population forward over the static `(P, Q, C)` grid, so peak memory scales with `P * Q`.

## Prepared-Pass Pair
`prepare_forward(state, transformed, pop=None, samples=1)` ignores `pop` (the substrate payload travels inside `transformed`) and prepares `hyper_genome.prepare_pass(state, transformed.schedule, *payload rows, samples)` — `samples` is forwarded unchanged, because the substrate schedule sits at `P` rows and a tiled query is the substrate genome's own sample axis.
`forward_prepared(state, transformed, prepared, inputs)` is `forward` minus the prelude: the same trailing bias column, then `hyper_genome.forward_prepared`.
`forward` is exactly those two back to back, bit for bit, because `genome.forward` *is* `prepare_pass` + `forward_prepared`; `test/unit/algorithm/test_hyperneat.py` pins the equality.
The signature is uniform with `NEAT`'s (`pop=None`, `samples`), so `host_payload._NetworkActFn` needs no arity adapter even though `NEAT` takes the payload and `HyperNEAT` does not.
`host_payload._NetworkActFn` is the pair's only consumer: its `__call__` goes through `prepare`, so every generation runs the prepared pair and `HyperNEAT.forward` has no in-tree call site (it stays protocol API for direct callers).
The tiled func-fit path (`host_payload._prepared_query`) slices both the `SubstrateSchedule` and its payload per chunk group and prepares one substrate pass per group at that group's `samples`; `_slice_batch` enforces population-major leaves, so a `SubstrateSchedule` whose nested schedule keeps `P` on every axis 0 is required — `test/integration/test_chunking_guard.py` pins a forced chunk budget to bit-identical fitness.

## Gene Contract (split payload, matching `genome/gene`)
`HyperNEATNode(BaseNode)` declares `fixed_attrs = ("index",)` → `key_width == 1`, and `attr_specs = (GeneAttr("aggregation", kind="int", replace_rate=0.1, options=(0,)), GeneAttr("activation", kind="int", replace_rate=0.1, options=(0,)))` → `attr_width == 2`; the class-level `attr_specs` is that table and `__init__` reinstalls the same module constant `_NODE_OPTION_SPECS` per instance.
`HyperNEATNode.__init__(aggregation=AGG.sum, activation=ACT.sigmoid)` stores the two callables, rejects a non-callable with `ConfigError`, and sets `aggregation_options = (aggregation,)` / `activation_options = (activation,)`: a substrate node's aggregation and activation are fixed at construction, so each option column has exactly one legal index, `0`.
A substrate node carries no evolvable attribute but must still declare the two `int` option columns `BaseNode.prepare` reads; an empty table would raise `ConfigError` from `BaseNode.setup`/`prepare`, and `_assemble_substrate` paints every node option column with `0`, which selects the constructor's callables in every substrate node of every individual.
`HyperNEATNode.combine(state, attrs, aggregated) -> aggregated` is the identity (a substrate node has no bias/response affine map); `forward` is inherited unchanged from `BaseNode` (`apply_activation(combine(aggregate(...)))`, dead slots neutralised per aggregation option, output rows left unactivated because `prepare` writes `ACT_IDENTITY` into their activation id).
`HyperNEATNode.repr`/`to_dict` are host-only and render the two option ids as registry names (e.g. `{"idx": 3, "agg": "sum", "act": "sigmoid"}`), resolving the `-1` sentinel to `"identity"`.
`HyperNEATConn(BaseConn)` inherits `fixed_attrs = ("input_index", "output_index")` and declares one `GeneAttr(name="weight", kind="float", init_mean=0.0, init_std=1.0, mutate_power=0.15, mutate_rate=0.8, replace_rate=0.1)` → `key_width == 2`, `attr_width == 1`; `__init__` resolves `self.weight_index` from the `attr_specs` names, never from an offset, and only `repr`/`to_dict` use it.
`HyperNEATConn.forward(state, attrs, source_values)` returns `source_values * self.column_of(attrs, "weight")`, accepting both raw `(..., K, CA)` rows and the `ConnOperands` bundle the genome actually passes; the weight column comes from the bundle's NaN-cleaned columns.
The conn kernel does no NaN guarding: dead slots are masked once through the node gene's `valid_mask`, so masking has one owner and the multiply stays one op.
Both genes implement the cheap host-only `repr`/`to_dict` hooks (`{"idx", "agg", "act"}` and `{"in", "out", "weight"}`) so `genome.network_dict`/`repr` work on a substrate genome.

## Substrate Protocol (host-only, pinned in `./substrate`)
`BaseSubstrate` is a plain host object — no `State`, no `setup`, never traced — exposing `connection_type` (`"feedforward"` / `"recurrent"`), `num_inputs` (input neurons *including* the trailing bias), `num_outputs`, `nodes_cnt`, `conns_cnt`, `query_coors (conns_cnt, 2*CD)` float32, `nodes (nodes_cnt,)` int32 and `conn_index (conns_cnt, 2)` int32.
Row `q` of `query_coors` must correspond 1:1 with row `q` of `conn_index`; the substrate's `connection_type` selects the executor genome (`recurrent` → `HyperNEAT`, `feedforward` → `HyperNEATFeedForward`).
Node keys must be the block order inputs `0..I-1`, outputs `I..I+O-1`, hidden — `_validate` re-checks it instead of letting it surface as a silently disabled activation, and it is the same invariant that makes the `output_keys`/`input_keys` pinning above correct.
This node reads only those tables: it never calls a `make_nodes`/`make_conns` builder, never inspects coordinate ordering, and never lets the substrate layer know a gene column order.

## Decode helpers (implemented surface of `hyperneat.py`)
`_validate()` is pure host Python/NumPy over the configuration and raises `TensorNEATError` for each of: `query_coors` not rank 2 or `query_coors.shape[1] != neat.num_inputs` (legacy wording "Query coors of Substrate should be equal to NEAT input size"), `weight_threshold` not a real number (`bool` rejected) or outside `[0, 1)`, `max_weight` not real or `<= 0`, `substrate.num_inputs < 2`, `activate_time` not real or `< 1`, `neat.num_outputs != 1`, `query_coors.shape[0] != conns_cnt`, `conn_index.shape != (conns_cnt, 2)`, `nodes.shape != (nodes_cnt,)`, `nodes` not a permutation of `0..nodes_cnt-1`, a node-key prefix that is not `0..I+O-1`, and a `connection_type` mismatch when `_required_connection_type` is set.
`_require_real` runs before each range check, so a non-numeric threshold/weight/horizon raises `TensorNEATError`, never `TypeError`; NumPy scalars pass.
`_validate` runs before `_make_hyper_genome` (which calls `_static_bounds`), so a bad configuration raises before any genome is constructed.
`_static_bounds()` returns `{"max_in_degree": K}` plus `{"max_depth": D}` for the feedforward executor only.
`K` is the exact maximum in-degree `bincount(conn_index[:, 1], minlength=nodes_cnt).max()`, floored at `1` and at `substrate.num_inputs`; the floor exists because the genome constructor validates `max_in_degree` against its (never used) dense initial `num_inputs -> num_outputs` network and would otherwise reject the bound.
The in-degree bound is *per destination node*, so for a full/MLP substrate it is the number of lower-layer neurons wired into one neuron: `FullSubstrate` with 4 inputs and 3 fully interconnected hidden neurons needs `K = 7` (self-loop included), and `MLPSubstrate(layers=[4, 5, 5, 5, 1])` needs `K = 5`.
`D` is the longest path length from the substrate inputs, `max(host_graph.node_depth(nodes, conn_index, range(num_inputs)).values())` floored at `1`, derived only when `_required_connection_type == "feedforward"`.
`node_depth` raises `GraphError` (a `TensorNEATError`) for a cyclic or input-unreachable substrate; that error propagates unchanged.
A too-small `K` is caught by `Transformed.edge_overflow` (the pipeline raises), but a too-small `D` would silently leave the tail of a deep substrate at `NaN`, which is why the feedforward bound exists.
`_make_hyper_genome()` builds `_genome_cls(num_inputs=substrate.num_inputs, num_outputs=substrate.num_outputs, max_nodes=substrate.nodes_cnt, max_conns=substrate.conns_cnt, node_gene=HyperNEATNode(aggregation, activation), conn_gene=HyperNEATConn(), output_transform=output_transform, **_static_bounds(), **_genome_extra_kwargs())`; the mutation/crossover/distance operators fall back to genome defaults and are never invoked, because the substrate payload is synthesised.
`_graph_constant(array)` is `evoxc.constant(evoxc.from_numpy(np.ascontiguousarray(array)))`; all substrate tables together are a few KiB, far under the 1 MiB `EVOXC_LARGE_CONSTANT_BYTES` warning threshold, and `__init__` never calls it (no graph work outside a trace).
`_query_coors(pop_size)` reshapes the `(Q, C)` constant to `(1, Q, C)` and `enp.broadcast_to` it to `(P, Q, C)` — never a Python loop over `P` and never `evoxc.tile` (no batching rule).
`_expand_schedule(transformed, n_query, pop_size)` is `evoxc.tree_map` over the CPPN schedule with `_expand_leaf`; `_expand_leaf` rewrites a leaf whose static axis 0 is the decoded batch `pop_size` into `(P * n_query, *leaf.shape[1:])` via `reshape → broadcast_to → reshape`, and returns any leaf without that axis unchanged.
`_query_substrate` runs the query with `self.neat.genome.forward(state, expanded, *expanded_pop, queries)`; `*expanded_pop` relies on the `GenomeTensors` field order `(node_keys, node_attrs, conn_keys, conn_attrs)` matching `genome.forward`'s parameter order.
`_weights_from_query` is `sign(w) * max(|w| - weight_threshold, 0) * max_weight / (1 - weight_threshold)` built from `enp.abs`/`enp.greater`/`evoxc.select`/`enp.subtract`/`enp.sign`/`enp.multiply`; the dead zone is strict, so `sign(0) == 0` and a tie with the threshold lands on `0` (a pruned edge), and NaN stays NaN by contract (`sign(NaN)` carries it through the final multiply even though the dead-zone `select` picks `0`).
`_assemble_substrate(weights)` returns a `GenomeTensors`: `node_keys` is the broadcast `(N,)` constant; `node_attrs` has one zero column per `node_gene.attr_specs` entry (an `int` spec stores its option index `0` as float32); `conn_keys` takes columns 0/1 from the `(Q, 2)` `conn_index` constant and fills any further key column (`OriginConn`'s `historical_marker`) with the unique row id `enp.arange(1, Q + 1)`; `conn_attrs` puts `weights` in the `"weight"` column with zeros elsewhere.
`_attr_columns(pop_size, rows, specs, weights)` derives column positions from `attr_specs` names, returns an explicit `(P, rows, 0)` when `specs` is empty (never `enp.stack([])`), and raises `TensorNEATError` when `weights` is given but no spec is named `"weight"` (the decoded weights would be dropped); the stack axis is the explicit `axis=2`, never `-1`.
`_conn_keys(pop_size, conns_cnt, key_width)` broadcasts the endpoint table and concatenates any extra key columns with the explicit `axis=2`, the extra count being `key_width - 2`.

## Formulation statics (`kernel_mode` / `distance_centre` / `distance_formulation`)
No module under `algorithm/hyperneat/` names or imports any of the three statics (`kernel_mode`, `distance_centre`, `distance_formulation`, `is_accelerator`, `schedule_accelerator`): the decode itself is formulation-independent, since it only reads substrate tables, reshapes weights and appends a bias column.
They still reach this node through two delegates: `genome/base.py::transform` forwards `kernel_mode(state)` into `derive_schedule`, so both the CPPN schedule (`neat.transform` inside `_query_substrate`) and the substrate phenotype schedule (`hyper_genome.transform` inside `transform`) take the accelerator formulation on a GPU-class device; and `tell`/`show_details` delegate to `neat.tell`, whose speciation consumes `distance_centre`/`distance_formulation` plus `kernel_mode`.
`Pipeline._shape_signature` appends `host_payload._formulation_signature(state)`'s triple for every algorithm including this one, so changing the statics forces exactly one rebuild here as elsewhere.
Nothing in this node may read the statics directly: a kernel with a device-specific formulation belongs in `../neat`/`../../genome`, not in the decode.

## Executable Cache Key (what a substrate change does)
`Pipeline._shape_signature` is the CPPN payload half `(pop_size, max_nodes, max_conns, max_in_degree, max_depth, key_span)` of `_genome()` (which resolves `algorithm.neat.genome` for a wrapper), then `host_payload._substrate_signature(algorithm)`, then `_formulation_signature(state)`, then the device's evaluate chunk budget.
`_substrate_signature` is `(nodes_cnt, conns_cnt, num_inputs, num_outputs, activate_time, nodes.shape, conn_index.shape, query_coors.shape, weight_threshold, max_weight)` and `()` for an algorithm with no `substrate` attribute, with `-1`/`()` fallbacks for a substrate that lacks an attribute.
Those entries are exactly what the graph bakes: the substrate counts become the executor's static sizes, its tables become graph constants, and the weight map's two numbers become float32 constants.
Replacing the substrate on a live pipeline therefore forces exactly one new build and the rebuilt pass is index-for-index equal to a fresh pipeline built on the same geometry (`test/integration/test_hyperneat_integration.py`, which also pins that the CPPN half of the signature is unchanged).

## Backend Behaviour (iree)
The whole generation graph exports to StableHLO v1 and runs end to end on `backend="iree"` llvm-cpu for both variants; the measured per-step rows, the compiled-vs-numpy tolerance table and the 10.4 s one-off compile for that case live in `benchmarks/BACKENDS.md`, which is the record for that backend.
The one exporter rule this node owns is the negative-axis ban: the bias `concatenate` (`axis=1`), the attribute stack (`axis=2`) and the connection-key marker `concatenate` (`axis=2`) must all be written as explicit non-negative static axes, because the exporter forwards the axis verbatim as `stablehlo.concatenate`'s `dimension` and `iree-compile` rejects a negative one.
`test/backends` has no HyperNEAT cell (it covers a tiny XOR `NEAT` pipeline only), so iree coverage of this node remains a hand-run record.

## Constraints
No `jax` / `torch` / `numpy.random` on the compute path; `numpy` is used only in `__init__`/`_validate`/`_static_bounds`, for `np.ascontiguousarray` before `evoxc.from_numpy`, and in host-only `repr`/`to_dict`.
No (un)vectorization of any kind: no `vmap`, no `fori_loop`, no `switch`; this node contains no `cond`/`while_loop`/`scan` of its own and every kernel is region-free and batch-first.
Axes in graph ops are written as non-negative static integers (`axis=1`, `axis=2`, never `-1`): `evoxc.stack` normalises a negative axis internally, every other op does not, so the rule is uniform here.
No `.at[...]` updates, no `tile` on batched data (evoxc's `tile` has no batching rule), no `repeat`, no `max/sum(where=…)`, no `any`/`all`: mask with `evoxc.select`, reduce afterwards.
`evoxc` `getitem` is static-only and rejects `Ellipsis`, so column selection goes through the genes' `column_of` rather than a literal last-axis index.
Python `if`/`for` over constructor configuration is legal and unrolls at trace time; branching on traced values is a `TraceError`.
All shapes must be static: `Q`, `N`, `I`, `O`, `NA`, `CK`, `CA` come from configuration, and `P` from the payload argument's static axis-0 length.
Nothing here mutates `State`, the substrate, the genome objects, or any array in place; the substrate geometry is read-only and already frozen as graph constants.
User-facing validation raises `TensorNEATError`/`ConfigError`, never `assert`; backend/shape errors propagate unchanged.
Files stay under ~1000 lines (`hyperneat.py` 930, `genes.py` 297, `hyperneat_feedforward.py` 113).

## Known Issues
`NEAT.forward(state, transformed, inputs)` cannot serve the `(P * Q)` query batch because it reads the un-expanded population from `state`; the decode therefore calls `neat.genome.forward` directly and depends on the `NEAT.genome` attribute plus the `GenomeTensors` field order — if `../neat` renames the attribute or changes the payload arity, `_query_substrate` breaks first.
`_expand_leaf` decides "population-indexed" from the shape alone (`if not leaf.shape or int(leaf.shape[0]) != pop_size: return leaf`), not from a declared axis, so a future non-population leaf whose first dimension happens to equal `P` would be silently expanded; every leaf the two call sites pass today is population-leading, and the unchanged-leaf branch is what the tests pin.
`_static_bounds` passes `max_in_degree`/`max_depth` into the executor genome while `benchmarks/legacy_jax` constructs its legacy genome with the library defaults, so a decode-parity test could not be exact on a substrate whose in-degree or depth binds; no `test/parity/` case exists for this node, so the divergence is unpinned.
`_static_bounds` covers in-degree and feedforward depth only; a recurrent substrate has no depth bound at all (`activate_time` ticks replace it), so one needing more than `activate_time` propagation steps must raise `activate_time` explicitly.
The feedforward variant's inherited `activate_time = 10` enters `_substrate_signature` although its genome ignores the value, so two `HyperNEATFeedForward` instances that differ only in that dead attribute would rebuild; harmless, but it means the attribute is not free to repurpose.

## Test Strategy
Tests live in the sibling `./../../../test` tree (read-only from here) and cover this node behaviourally, with no marker and no `xfail`.
`test/unit/algorithm/test_hyperneat.py`: the decode helpers (`_query_coors`, `_expand_leaf`/`_expand_schedule` including the unchanged-leaf case, `_query_substrate` row order, `_assemble_substrate`, `_conn_keys`), `_static_bounds` on the shipped substrates, `_make_hyper_genome`, every `_validate` raise path, protocol arities/delegation, the `SubstrateSchedule.level`/`overflow` aliases, single-individual vs population-row decode equality, the bias column, `forward` == `prepare_forward`+`forward_prepared`, tiled individual-major batching, and `test_forward_reads_the_substrate_output_neurons` (the `output_slot` read-out law).
`test/unit/algorithm/test_hyperneat_weights.py`: the weight map at `0`, the threshold tie, just above it, large magnitudes, NaN in/out, batching and constructor wiring.
`test/unit/algorithm/test_genes.py`: `HyperNEATNode`'s option table, non-callable rejection, identity `combine`, masked aggregation, the all-dead window (`0.0`), the output-row rule, mean/sum agreement and the `ConnOperands` spelling; `HyperNEATConn`'s key/attr widths and weight column.
`test/unit/algorithm/test_hyperneat_feedforward.py`: the variant hooks, the `__init__` signature delta, the `DefaultGenome` executor with the substrate depth, the recurrent-substrate rejection, decode equality with the recurrent variant, the bias column, `forward` == prepared split on a deep substrate, `activate_time` independence, and one full generation step.
`test/unit/algorithm/test_hyperneat_e2e.py`: two-generation runs of both variants (finite, repeatable), `best_genome`/`network_dict`/report, and the feedforward phenotype sized by the MLP substrate.
`test/integration/test_hyperneat_integration.py`: `show_config` reports the inner `NEAT` species law, a substrate geometry change forces exactly one rebuild without a stale phenotype, and `auto_run` checkpoint plus CPPN `network_dict`.
`test/integration/test_chunking_guard.py`: a forced small chunk budget is bit-identical for a `HyperNEAT` run (the nested `SubstrateSchedule` slices correctly).
`test/unit/algorithm/test_substrate.py` covers the child subtree's geometry; `test/release_gate_no_jax.py` resolves `tensorneat.algorithm.HyperNEAT`/`HyperNEATFeedForward` as public names.
There is no `test/parity/algorithm/hyperneat/`: the decode has no frozen-oracle comparison.
Every kernel must be deterministic given `(seed, state)`: no dict ordering, no host RNG, no float-to-int conversion without an explicit mask.

## Notes for Agents
Every shape above is static: `Q`/`N`/`I`/`O`/`NA`/`CK`/`CA` come from `self.substrate`/`self.hyper_genome`, and the decode batch `P` from the payload's static axis-0 length, never from a value only known at run time.
`transform` must accept any static batch, not just `neat.pop_size`: `Pipeline.show` derives a one-individual schedule through `_constant_act_fn`, which calls `algorithm.transform(state, one)` and then the prepared pair, and the decode of `P = 1` is value-identical to that individual's full-population row.
`_validate` runs before `_make_hyper_genome`, and both run in `__init__`: never move a graph op or an `evoxc.constant` into construction, and keep the substrate tables as trace-time reads.
The substrate tables are host data: read them through `_graph_constant` at trace time, never in `__init__` (only `_static_bounds`/`_validate` read them on the host).
If a substrate ever needs a table over ~1 MiB, stop in-graph and escalate: the coordinates would have to become an explicit executable input, which is a `pipeline.py` change.

## Routing Table
| Path | Responsibility |
| --- | --- |
| `./hyperneat.py` | `SubstrateSchedule`, `HyperNEAT` (decode algebra + protocol delegation), the `_validate`/`_static_bounds`/`_make_hyper_genome`/`_genome_extra_kwargs`/`_graph_constant`/`_query_coors`/`_expand_schedule`/`_expand_leaf`/`_query_substrate`/`_weights_from_query`/`_assemble_substrate`/`_attr_columns`/`_conn_keys` helpers, and the `HyperNEATNode`/`HyperNEATConn` re-export from `./genes.py`. |
| `./genes.py` | `HyperNEATNode` (fixed single-option aggregation/activation columns, identity `combine`) and `HyperNEATConn` (one multiply by the decoded weight via `column_of`), plus their host-only helpers. |
| `./hyperneat_feedforward.py` | `HyperNEATFeedForward`: the feedforward variant delta only (`_genome_cls`, `_required_connection_type`, `_genome_extra_kwargs`, `__init__` signature). |
| `./substrate` | Host-only substrate geometry (`BaseSubstrate`, `DefaultSubstrate`, `FullSubstrate`, `MLPSubstrate`): coordinate → node/connection analysis, `connection_type` contract, `query_coors`/`conn_index` tables, shared geometry helpers. |
| `./__init__.py` | Re-exports the two algorithms, `SubstrateSchedule`, the two gene kernels and the four substrate classes. |
