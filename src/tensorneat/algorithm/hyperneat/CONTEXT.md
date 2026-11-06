# HyperNEAT (`tensorneat.algorithm.hyperneat`)

## Intent
HyperNEAT evolves only a CPPN (an ordinary `../neat` network) and derives a *substrate* network from it: the CPPN is queried once per substrate connection at the coordinates of that connection's two endpoints, and its single scalar output becomes that connection's weight.
The substrate topology is fixed at construction, so a whole CPPN population decodes into one batch-first execution schedule of a dense, identical-shaped phenotype network — no padding, no NaN rows.
This node owns the decode (CPPN → weights → gene payload → schedule), the two variants (`HyperNEAT` recurrent, `HyperNEATFeedForward` feedforward), the two substrate gene kernels (`./genes.py`) and the host-only substrate geometry (`./substrate`).
All evolution, speciation and `randkey` consumption stay in `../neat`; the substrate object is host-side geometry that never enters `State`.

## API Surface
`hyperneat.py` (930 lines): `SubstrateSchedule`, `HyperNEAT(BaseAlgorithm)` and the decode helpers; `genes.py` (297): the two gene kernels and their host-only helpers; `hyperneat_feedforward.py` (113): `HyperNEATFeedForward`, a delta file with no kernels of its own.
`__init__.py` re-exports `HyperNEAT`, `HyperNEATFeedForward`, `SubstrateSchedule`, `HyperNEATNode`, `HyperNEATConn` and the four substrate classes; `../__init__.py` re-exports only the two algorithms.
`HyperNEAT.__init__(substrate, neat, weight_threshold=0.3, max_weight=5.0, aggregation=AGG.sum, activation=ACT.sigmoid, activate_time=10, output_transform=ACT.sigmoid)`; `HyperNEATFeedForward.__init__` is that signature minus `activate_time`.
`setup(state=None)` chains `neat.setup` then `hyper_genome.setup` (the substrate is stateless and has no `setup`), and `ask` returns `neat.ask(state)` untouched at zero ops.
Protocol members: `transform(state, pop) -> SubstrateSchedule`, `forward`, `prepare_forward(state, transformed, pop=None, samples=1)`, `forward_prepared`; `tell`/`show_details` delegate to `neat`, and `pop_size` is the read-only property `neat.pop_size`.
`num_inputs` = `substrate.num_inputs - 1` (trailing bias removed), `num_outputs` = `substrate.num_outputs`.
Subclass hooks: `_genome_cls`, `_required_connection_type` and `_genome_extra_kwargs()` (the recurrent `activate_time`); `HyperNEATFeedForward` overrides only those three plus its `__init__` signature, so the variants cannot drift.

## Decode Algebra
Static sizes: `P` = the payload's leading dimension (any static batch — the CPPN's `pop_size` in an evolution step, `1` for a `Pipeline.show` decode); `Q = substrate.conns_cnt`; `N = substrate.nodes_cnt`; `I = substrate.num_inputs` (inputs *including* the bias); `O = substrate.num_outputs`; `C = neat.num_inputs = substrate.query_coors.shape[1] = 2 * CD`; `NA`/`CK`/`CA` are the node and conn gene key/attr widths (`2`/`2`/`1` for the shipped genes).
`I` and `C` are different numbers and must never be interchanged: `I` sizes the bias column of `forward`, `C` sizes the coordinate rows of `query_coors`.
Every tiling helper takes `pop_size` as a static argument (`_query_coors`, `_expand_schedule`, `_expand_leaf`) and `_assemble_substrate` reads it from `weights.shape[0]`; none may fall back to `self.pop_size`.
`transform(state, pop)`, in order:
1. `_query_substrate(state, pop) -> (P, Q)` — `neat.transform` gives the CPPN schedule, `_expand_schedule` repeats every population-axis leaf `Q` times, `_query_coors` builds the `(P, Q, C)` grid, the grid flattens to `(P * Q, C)`, and one `neat.genome.forward` returns `(P * Q, 1)` reshaped to `(P, Q)`.
2. `_weights_from_query` — dead zone, continuity shift and scale `max_weight / (1 - weight_threshold)`; unclipped, so `max_weight` is reached at `|query| == 1` and a larger CPPN output grows proportionally beyond it.
3. `_assemble_substrate` — the four-array `GenomeTensors` payload of the dense substrate.
4. `hyper_genome.transform(state, payload.node_keys, payload.conn_keys, input_keys=tuple(range(I)), output_keys=tuple(range(I, I + O)))` — only the two key arrays go in, and the schedule is derived from the substrate's *actual* key vectors, never from the executor's default layout.
5. `return SubstrateSchedule(schedule, node_keys, node_attrs, conn_keys, conn_attrs)` — the pair travels together because the protocol forwards exactly one `transformed` object.
**Read-out law.** `schedule.output_slot` is the authority for the phenotype's outputs: the `output_keys=I..I+O-1` pinning makes it name the substrate output neurons, and `genome/step.py` gathers the network outputs at `output_slot` and derives the per-row output-activation mask from it — never "the last `O` payload rows", which are hidden rows whenever the substrate has hidden neurons.
`input_keys=0..I-1` is the first payload row range, exactly the rows `genome.forward` seeds with the inputs, so input injection keeps its invariant (`input_slot` only pins the levels).
`forward(state, transformed, inputs)` takes `(B, I - 1)`, appends a trailing ones column (`B = int(inputs.shape[0])`, normally `pop_size * samples`) and returns `(B, O)` from `hyper_genome.forward(state, transformed.schedule, *payload rows, biased)`; the bias is the **last** substrate input, hence `num_inputs = substrate.num_inputs - 1`.
The ones column is sized from the query's own row count, never from `pop_size`, because the genome expands the payload's population axis to `B` internally.
`SubstrateSchedule` is a `NamedTuple`; every leaf, including the nested `schedule`'s, keeps the population on axis 0, so `problem.static_batch_size`, `evoxc.tree_map` and the chunked-evaluate `_slice_batch` all see a plain batch-first pytree, and its `level`/`overflow` properties delegate to `self.schedule` (nothing in the tree reads them: `host_payload._schedule_overflow` reads `edge_overflow`, then the nested `transformed.schedule.edge_overflow` fallback).
The expansion row order is `p * Q + q` (individual slowest) and matches the `(P, Q, C) -> (P * Q, C)` reshape; changing one without the other silently permutes weights across connections, and `neat.pop_batch_size` does not chunk this pass (one whole-population forward over the static grid, so peak memory scales with `P * Q`).

## Prepared-Pass Pair
`prepare_forward(state, transformed, pop=None, samples=1)` ignores `pop` — the substrate payload travels inside `transformed` — and prepares `hyper_genome.prepare_pass(state, transformed.schedule, *payload rows, samples)`, forwarding `samples` unchanged because the substrate schedule sits at `P` rows and a tiled query is the substrate genome's own sample axis.
`forward_prepared(state, transformed, prepared, inputs)` is `forward` minus the prelude: the same trailing bias column, then `hyper_genome.forward_prepared`.
`forward` is exactly those two back to back, bit for bit, because `genome.forward` *is* `prepare_pass` + `forward_prepared`; `test_hyperneat.py` pins the equality.
The signature is uniform with `NEAT`'s (`pop=None`, `samples`), so `host_payload._NetworkActFn` needs no arity adapter even though `NEAT` takes the payload and `HyperNEAT` does not.
That class is the pair's only consumer and its `__call__` goes through `prepare`, so every generation runs the prepared pair and `HyperNEAT.forward` has no in-tree call site (it stays protocol API for direct callers).
The tiled func-fit path (`host_payload._prepared_query`) slices both the `SubstrateSchedule` and its payload per chunk group and prepares one substrate pass per group at that group's `samples`, which is why every leaf must be population-major; `test/integration/test_chunking_guard.py` pins a forced chunk budget to bit-identical fitness.

## Gene Contract
`HyperNEATNode`: `key_width == 1` (the `index` key alone) and `attr_width == 2` — the two `int` option columns `aggregation` then `activation`, each declaring the single option index `0` that selects the callables fixed at construction.
`HyperNEATNode.__init__(aggregation=AGG.sum, activation=ACT.sigmoid)` rejects a non-callable with `ConfigError`; a substrate node carries no evolvable attribute but must still declare both option columns `BaseNode.prepare` reads (an empty table raises `ConfigError`), and `_assemble_substrate` paints them with `0`.
`combine(state, attrs, aggregated) -> aggregated` is the identity (no bias/response affine map), and `forward` is inherited unchanged (`apply_activation(combine(aggregate(...)))`, dead slots neutralised per option, output rows unactivated through the `ACT_IDENTITY` id `prepare` writes); `repr`/`to_dict` are host-only and render the option ids as registry names.
`HyperNEATConn`: `key_width == 2` (the endpoint keys) and `attr_width == 1` (the `weight` spec), with `self.weight_index` resolved from the spec names rather than an offset; `forward(attrs, source_values) = source_values * self.column_of(attrs, "weight")` accepts raw rows and the genome's `ConnOperands` bundle, and does no NaN guarding because dead slots are masked once through the node gene's `valid_mask`.

## Substrate Protocol (host-only, pinned in `./substrate`)
`BaseSubstrate` is a plain host object — no `State`, no `setup`, never traced — exposing `connection_type` (`"feedforward"` / `"recurrent"`), `num_inputs` (inputs *including* the trailing bias), `num_outputs`, `nodes_cnt`, `conns_cnt`, `query_coors (conns_cnt, 2*CD)` float32, `nodes (nodes_cnt,)` int32 and `conn_index (conns_cnt, 2)` int32.
Row `q` of `query_coors` must correspond 1:1 with row `q` of `conn_index`, and `connection_type` selects the executor genome (`recurrent` → `HyperNEAT`, `feedforward` → `HyperNEATFeedForward`).
Node keys must be the block order inputs `0..I-1`, outputs `I..I+O-1`, hidden: `_validate` re-checks it, and that order is what makes the `input_keys`/`output_keys` pinning above correct.
This node reads only those tables — never a `make_nodes`/`make_conns` builder, never a coordinate ordering, never a gene column order.

## Validation and Static Bounds
`_validate()` is pure host Python/NumPy over the configuration and raises `TensorNEATError` for each of: `query_coors` not rank 2 or `query_coors.shape[1] != neat.num_inputs` (legacy wording "Query coors of Substrate should be equal to NEAT input size"), `weight_threshold` not a real number (`bool` rejected) or outside `[0, 1)`, `max_weight` not real or `<= 0`, `substrate.num_inputs < 2`, `activate_time` not real or `< 1`, `neat.num_outputs != 1`, `query_coors.shape[0] != conns_cnt`, `conn_index.shape != (conns_cnt, 2)`, `nodes.shape != (nodes_cnt,)`, `nodes` not a permutation of `0..nodes_cnt-1`, a node-key prefix that is not `0..I+O-1`, and a `connection_type` mismatch when `_required_connection_type` is set.
`_require_real` runs before each range check, so a non-numeric threshold/weight/horizon raises `TensorNEATError`, never `TypeError`; NumPy scalars pass.
`_validate` runs before `_make_hyper_genome` (which calls `_static_bounds`), so a bad configuration raises before any genome is constructed.
`_static_bounds()` returns `{"max_in_degree": K}` plus `{"max_depth": D}` for the feedforward executor only.
`K` is the exact maximum in-degree `bincount(conn_index[:, 1], minlength=nodes_cnt).max()`, floored at `1` and at `substrate.num_inputs` because the genome constructor validates `max_in_degree` against its (never used) dense initial `num_inputs -> num_outputs` network.
The bound is *per destination node*, so for a dense substrate it counts the lower-layer neurons wired into one neuron: `FullSubstrate` with 4 inputs and 3 fully interconnected hidden neurons needs `K = 7` (self-loop included), `MLPSubstrate(layers=[4, 5, 5, 5, 1])` needs `K = 5`.
`D` is `max(host_graph.node_depth(nodes, conn_index, range(num_inputs)).values())` floored at `1`; a too-small `K` is caught by `Transformed.edge_overflow` (the pipeline raises), but a too-small `D` would silently leave the tail of a deep substrate at `NaN`, and `node_depth`'s `GraphError` for a cyclic or input-unreachable substrate propagates unchanged.
`_make_hyper_genome()` builds `_genome_cls(num_inputs=substrate.num_inputs, num_outputs=substrate.num_outputs, max_nodes=substrate.nodes_cnt, max_conns=substrate.conns_cnt, node_gene=HyperNEATNode(aggregation, activation), conn_gene=HyperNEATConn(), output_transform=output_transform, **_static_bounds(), **_genome_extra_kwargs())`; its mutation/crossover/distance operators fall back to genome defaults and are never invoked, because the payload is synthesised.

## Payload Assembly Helpers
`_weights_from_query` is `sign(w) * max(|w| - weight_threshold, 0) * max_weight / (1 - weight_threshold)`; the dead zone is strict, so `sign(0) == 0` and a tie with the threshold lands on `0` (a pruned edge), while NaN stays NaN by contract.
`_assemble_substrate(weights)` returns a `GenomeTensors` whose columns are read from the gene *tables*, never a hardcoded index: `node_keys` broadcasts the `(N,)` constant, `node_attrs` carries one zero column per `node_gene.attr_specs` entry (an `int` spec stores option index `0` as float32), `conn_keys` takes columns 0/1 from the `(Q, 2)` `conn_index` constant and fills any further key column (`OriginConn`'s `historical_marker`) with the unique row id `enp.arange(1, Q + 1)`, and `conn_attrs` puts `weights` in the `"weight"` column.
`_attr_columns(pop_size, rows, specs, weights)` returns an explicit `(P, rows, 0)` when `specs` is empty (never `enp.stack([])`) and raises `TensorNEATError` when `weights` is given but no spec is named `"weight"` (the decoded weights would be dropped).
`_query_coors(pop_size)` reshapes the `(Q, C)` constant to `(1, Q, C)` and `enp.broadcast_to` it to `(P, Q, C)` — never a Python loop over `P`, never `evoxc.tile` (no batching rule in evoxc); `_expand_schedule` is `evoxc.tree_map` with `_expand_leaf`.
`_query_substrate` runs `self.neat.genome.forward(state, expanded, *expanded_pop, queries)`, so `*expanded_pop` relies on the `GenomeTensors` field order matching `genome.forward`'s parameter order; `_graph_constant(array)` is `evoxc.constant(evoxc.from_numpy(np.ascontiguousarray(array)))`, cheap here because all substrate tables together are a few KiB against the 1 MiB `EVOXC_LARGE_CONSTANT_BYTES` warning threshold, and `__init__` never calls it.

## Formulation statics (`kernel_mode` / `distance_centre` / `distance_formulation`)
No module under `algorithm/hyperneat/` names or imports any of the three statics (nor `is_accelerator`/`schedule_accelerator`): the decode is formulation-independent, since it only reads substrate tables, reshapes weights and appends a bias column.
They nonetheless reach this node through two delegates: `genome/base.py::transform` forwards `kernel_mode(state)` into `derive_schedule`, so both the CPPN schedule (`neat.transform` inside `_query_substrate`) and the substrate phenotype schedule (`hyper_genome.transform` inside `transform`) take the accelerator formulation on a GPU-class device; and `tell`/`show_details` delegate to `neat.tell`, whose speciation consumes `distance_centre`/`distance_formulation` plus `kernel_mode`.
`Pipeline._shape_signature` appends `host_payload._formulation_signature(state)`'s triple for every algorithm including this one, so changing the statics forces exactly one rebuild here as elsewhere.
Nothing here may read the statics directly: a device-specific formulation belongs in `../neat` or `../../genome`, never in the decode.

## Executable Cache Key
`Pipeline._shape_signature` is the CPPN payload half `(pop_size, max_nodes, max_conns, max_in_degree, max_depth, key_span)` of `_genome()` (which resolves `algorithm.neat.genome` for a wrapper), then `host_payload._substrate_signature(algorithm)`, then `_formulation_signature(state)`, then the device's evaluate chunk budget.
`_substrate_signature` is `(nodes_cnt, conns_cnt, num_inputs, num_outputs, activate_time, nodes.shape, conn_index.shape, query_coors.shape, weight_threshold, max_weight)`, `()` for an algorithm with no `substrate` attribute, with `-1`/`()` fallbacks for a substrate that lacks an attribute.
Those entries are exactly what the graph bakes: the substrate counts become the executor's static sizes, its tables become graph constants, and the weight map's two numbers become float32 constants.
Replacing the substrate on a live pipeline therefore forces exactly one new build, and the rebuilt pass is index-for-index equal to a fresh pipeline on the same geometry (`test/integration/test_hyperneat_integration.py`, which also pins that the CPPN half of the signature is unchanged).

## Backend Behaviour (iree)
The whole generation graph exports to StableHLO v1 and runs end to end on `backend="iree"` llvm-cpu for both variants; the measured per-step rows, the cross-backend tolerance table and the one-off compile cost for that case live in `benchmarks/BACKENDS.md`.
The exporter rule this node owns is the negative-axis ban: the bias `concatenate` (`axis=1`), the attribute stack (`axis=2`) and the connection-key marker `concatenate` (`axis=2`) must all use explicit non-negative static axes, because the exporter forwards the axis verbatim as `stablehlo.concatenate`'s `dimension` and `iree-compile` rejects a negative one.
`test/backends` has no HyperNEAT cell (it covers a tiny XOR `NEAT` pipeline only), so iree coverage of this node remains a hand-run record.

## Constraints
No `jax` / `torch` / `numpy.random` on the compute path; `numpy` is used only in `__init__`/`_validate`/`_static_bounds`, for `np.ascontiguousarray` before `evoxc.from_numpy`, and in host-only `repr`/`to_dict`.
No `vmap`/`fori_loop`/`switch` and no `cond`/`while_loop`/`scan` of this node's own: every kernel here is region-free and batch-first, with non-negative static axes (never `-1`) in every graph op.
Every shape is static (`Q`/`N`/`I`/`O`/`NA`/`CK`/`CA` from configuration, `P` from the payload's static axis 0); no `.at[...]` updates, no `tile` on batched data, no `repeat`, no `max/sum(where=…)`, no `any`/`all` — mask with `evoxc.select` and reduce afterwards.
`evoxc` `getitem` is static-only and rejects `Ellipsis`, so column selection goes through `column_of`; Python `if`/`for` over constructor configuration unrolls at trace time, while branching on a traced value is a `TraceError`.
Nothing here mutates `State`, the substrate, the genome objects or any array in place, and user-facing validation raises `TensorNEATError`/`ConfigError` (never `assert`) while backend/shape errors propagate unchanged.

## Known Issues
`NEAT.forward(state, transformed, inputs)` cannot serve the `(P * Q)` query batch because it reads the un-expanded population from `state`; the decode therefore calls `neat.genome.forward` directly and depends on the `NEAT.genome` attribute plus the `GenomeTensors` field order, so a rename or an arity change in `../neat` breaks `_query_substrate` first.
`_expand_leaf` decides "population-indexed" from the shape alone (`if not leaf.shape or int(leaf.shape[0]) != pop_size: return leaf`), not from a declared axis, so a future non-population leaf whose first dimension happens to equal `P` would be silently expanded; every leaf the two call sites pass today is population-leading, and only the unchanged-leaf branch is pinned by a test.
`_static_bounds` passes `max_in_degree`/`max_depth` into the executor genome while `benchmarks/legacy_jax` constructs its legacy genome with the library defaults, so a decode-parity test could not be exact on a substrate whose in-degree or depth binds; no `test/parity/` case exists for this node, so that divergence is unpinned.
`_static_bounds` covers in-degree and feedforward depth only; a recurrent substrate has no depth bound at all (`activate_time` ticks replace it), so one needing more propagation steps than `activate_time` must raise `activate_time` explicitly.
`HyperNEATFeedForward` inherits `activate_time = 10` although its genome ignores it, and the value enters `_substrate_signature`, so two instances differing only in that dead attribute would rebuild.

## Test Strategy
Tests live in the sibling `./../../../test` tree (read-only from here) and cover this node behaviourally, with no marker and no `xfail`.
`test/unit/algorithm/test_hyperneat.py` covers the decode helpers (`_query_coors`, `_expand_leaf`/`_expand_schedule` including the unchanged-leaf case, `_query_substrate` row order, `_assemble_substrate`, `_conn_keys`), `_static_bounds`, `_make_hyper_genome`, every `_validate` raise path, protocol arities and delegation, `P = 1` vs population-row decode equality, the bias column, `forward` = `prepare_forward`+`forward_prepared` and `test_forward_reads_the_substrate_output_neurons` (the read-out law).
`test_hyperneat_weights.py` covers the weight map (zero, the threshold tie, just above it, large magnitudes, NaN, batching); `test_genes.py` both genes' widths and option tables plus the `ConfigError`, identity `combine`, masked aggregation, all-dead window and `ConnOperands` rules; `test_hyperneat_feedforward.py` the variant delta, substrate depth, recurrent-substrate rejection and the prepared split; `test_hyperneat_e2e.py` two-generation runs of both variants.
`test/integration/test_hyperneat_integration.py` covers `show_config`, the substrate-change rebuild and the checkpoint round-trip, and `test/integration/test_chunking_guard.py` pins a forced chunk budget to bit-identical fitness.
`test/unit/algorithm/test_substrate.py` covers the child subtree and `test/release_gate_no_jax.py` resolves both algorithm names; no `test/parity/algorithm/hyperneat/` exists, so the decode has no frozen-oracle comparison.
Every kernel must be deterministic given `(seed, state)`: no dict ordering, no host RNG, no float-to-int conversion without an explicit mask.

## Notes for Agents
`transform` must accept any static batch, not just `neat.pop_size`: `Pipeline.show` derives a one-individual schedule through `_constant_act_fn`, which calls `algorithm.transform(state, one)` and then the prepared pair, and the `P = 1` decode is value-identical to that individual's full-population row.
The substrate tables are host data: read them through `_graph_constant` at trace time, never in `__init__` (only `_validate`/`_static_bounds` read them on the host), and keep `_validate` before `_make_hyper_genome`.
If a substrate ever needs a table over ~1 MiB, stop in-graph and escalate: the coordinates would have to become an explicit executable input, which is a `pipeline.py` change.

## Routing Table
| Path | Responsibility |
| --- | --- |
| `./hyperneat.py` | `SubstrateSchedule`, `HyperNEAT` (decode algebra + protocol delegation) and the `_validate`/`_static_bounds`/`_make_hyper_genome`/`_genome_extra_kwargs`/`_graph_constant`/`_query_coors`/`_expand_schedule`/`_expand_leaf`/`_query_substrate`/`_weights_from_query`/`_assemble_substrate`/`_attr_columns`/`_conn_keys` helpers. |
| `./genes.py` | `HyperNEATNode` (fixed single-option aggregation/activation columns, identity `combine`) and `HyperNEATConn` (one multiply by the decoded weight via `column_of`), plus their host-only helpers. |
| `./hyperneat_feedforward.py` | `HyperNEATFeedForward`: the feedforward variant delta only (`_genome_cls`, `_required_connection_type`, `_genome_extra_kwargs`, `__init__` signature). |
| `./substrate` | Host-only substrate geometry (`BaseSubstrate`, `DefaultSubstrate`, `FullSubstrate`, `MLPSubstrate`): coordinate → node/connection analysis, `connection_type` contract, `query_coors`/`conn_index` tables, shared geometry helpers. |
| `./__init__.py` | Re-exports the two algorithms, `SubstrateSchedule`, the two gene kernels and the four substrate classes. |
