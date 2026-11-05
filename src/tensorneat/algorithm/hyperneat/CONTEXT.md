# HyperNEAT (`tensorneat.algorithm.hyperneat`)

## Intent
HyperNEAT evolves only a CPPN (an ordinary NEAT network from `../neat`) and derives a *substrate* network from it: the CPPN is queried once per substrate connection at the coordinates of that connection's two endpoints, and its single scalar output becomes that connection's weight.
The substrate topology is fixed at construction, so this node turns a whole CPPN population into one batch-first execution schedule of a dense, identical-shaped phenotype network — no padding, no NaN rows.
Four things live here: the decode (CPPN → weights → gene payload → schedule), the two concrete variants (`HyperNEAT` recurrent, `HyperNEATFeedForward` feedforward), the two substrate gene kernels (`HyperNEATNode`, `HyperNEATConn`, in `./genes.py`), and the host-only substrate geometry package.
All evolution, speciation and `randkey` consumption stay in `../neat`; the substrate object is a host-side geometry table and never enters `State`.

## API Surface
`hyperneat.py`: `SubstrateSchedule`, `HyperNEAT(BaseAlgorithm)`, plus the re-exported `HyperNEATNode` / `HyperNEATConn` it imports from `./genes.py`.
`genes.py`: `HyperNEATNode(BaseNode)`, `HyperNEATConn(BaseConn)`, and the host-only helpers only they use (`_key_int`, `_attr_float`, `_require_callable`, the `_NODE_OPTION_SPECS` option table).
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
Properties `level` and `overflow` delegate to `self.schedule`, but nothing in the tree reads them: `host_payload._schedule_overflow` takes the nested `getattr(transformed.schedule, "edge_overflow")` fallback, so both are convenience aliases for direct callers only.

## CPPN → Substrate Decode Algebra
Static sizes: `P` = the **payload's** leading dimension — `neat.pop_size` during an evolution step, but any static batch is legal (`Pipeline.show` decodes one individual), because `transform` reads `P` off `pop.node_keys.shape[0]` rather than off the config; `Q = substrate.conns_cnt`, `N = substrate.nodes_cnt`, `I = substrate.num_inputs` (substrate inputs *including* the bias), `O = substrate.num_outputs`, `C = neat.num_inputs = substrate.query_coors.shape[1] = 2 * CD` (CPPN input width), `NA = node_gene.attr_width = 2` (the `aggregation` / `activation` option columns), `CK = conn_gene.key_width`, `CA = conn_gene.attr_width`, all fixed at construction.
Every decode helper that tiles a batch takes `pop_size` as a static argument (`_query_coors(pop_size)`, `_expand_schedule(transformed, n_query, pop_size)`, `_expand_leaf(leaf, n_query, pop_size)`) and `_assemble_substrate` reads it from `weights.shape[0]`; none of them may fall back to `self.pop_size`, whose only remaining job is the base-protocol read-only property.
`I` (substrate input neurons) and `C` (CPPN query width) are different numbers and must never be interchanged: `I` sizes the bias column of `forward`, `C` sizes the coordinate rows of `query_coors`.
`HyperNEAT.transform(state, pop)` performs, in order:
1. `query_res (P, Q) = _query_substrate(state, pop)` — `neat.transform(state, pop)` gives the CPPN schedule, `_expand_schedule` repeats every population-axis leaf `Q` times, `_query_coors` builds the `(P, Q, C)` grid, the grid is flattened to `(P * Q, C)`, and one `neat.genome.forward` call returns `(P * Q, 1)` reshaped to `(P, Q)`.
2. `weights (P, Q) = _weights_from_query(query_res)` — dead zone, continuity shift and scaling into `[-max_weight, max_weight]`.
3. `payload = _assemble_substrate(weights)` — a `genome.utils.GenomeTensors` of the four split arrays: `node_keys (P, N)`, `node_attrs (P, N, 2)`, `conn_keys (P, Q, CK)`, `conn_attrs (P, Q, CA)`.
4. `schedule = hyper_genome.transform(state, payload.node_keys, payload.conn_keys)` — the substrate genome derives its own schedule; its layout is genome-owned and opaque to this node.
5. `return SubstrateSchedule(schedule, *payload)`.
`HyperNEAT.forward(state, transformed, inputs)` takes `(B, I - 1)`, appends a trailing column of ones (`B = inputs.shape[0]`, a static int — problems tile datasets over the population axis so `B` is `pop_size * samples`) and returns `(B, O)` from `hyper_genome.forward(state, schedule, node_keys, node_attrs, conn_keys, conn_attrs, biased)`.
The bias is the **last** substrate input, which is why `num_inputs` reports `substrate.num_inputs - 1` upward.
`HyperNEAT.prepare_forward(state, transformed, pop=None, samples=1)` ignores `pop` (the substrate payload travels inside `transformed`) and prepares `hyper_genome.prepare_pass` from that payload, forwarding `samples`; `HyperNEAT.forward_prepared` is `forward` minus the prelude — the same bias column, then `hyper_genome.forward_prepared`.
`HyperNEAT.forward` has no in-tree call site: `Pipeline.step` hands `algorithm.forward_prepared`/`algorithm.prepare_forward` to `host_payload._NetworkActFn`, whose `__call__` goes through `prepare`, so every generation runs the prepared pair and `forward` stays protocol API for direct callers.
The substrate schedule is at `P` rows, so a tiled `B = P * samples` query prepares the substrate genome's own sample axis; the CPPN query expansion (`_expand_schedule`, `samples=1`) happens once inside `HyperNEAT.transform`, not in either prepared method.
The expansion row order is `p * Q + q` (individual slowest) and matches the `(P, Q, C) -> (P * Q, C)` reshape; changing one without the other silently permutes weights across connections.
`neat.pop_batch_size` does not chunk this pass: the CPPN query is a single whole-population forward over the static `(P, Q, C)` grid, so peak memory scales with `P * conns_cnt`.
A change to `pop_size`, `conns_cnt`/`nodes_cnt` or a `_static_bounds` value changes the traced graph, so the executable must be rebuilt; nothing in this node caches per-shape state, but `pipeline._shape_signature` keys only the CPPN's genome statics (`pop_size`, `max_nodes`, `max_conns`, `max_in_degree`, `max_depth`, `key_span`) and nothing substrate-derived, so a pipeline given a different substrate would silently reuse the stale executable — construct a new `Pipeline` after changing the substrate.

## Gene Contract (split payload, matching `genome/gene`)
`HyperNEATNode(BaseNode)` declares `fixed_attrs = ("index",)` → `key_width == 1`, and `attr_specs = (GeneAttr("aggregation", kind="int", replace_rate=0.1, options=(0,)), GeneAttr("activation", kind="int", replace_rate=0.1, options=(0,)))` → `attr_width == 2`.
The class-level `attr_specs` is that default table; `__init__` reinstalls exactly the same tuple per instance, and a one-element `_NODE_OPTION_SPECS` module constant is the single source of the column layout.
`HyperNEATNode.__init__(aggregation=AGG.sum, activation=ACT.sigmoid)` stores the two callables, rejects a non-callable with `ConfigError`, and sets `self.aggregation_options = (aggregation,)` / `self.activation_options = (activation,)`: the aggregation and activation of a substrate node are fixed at construction, so each option column has exactly one legal index, `0`.
A substrate node therefore carries no evolvable attribute, but it must still declare the two `int` option columns `BaseNode.prepare` reads (`genome/gene/node/base.py:197-210`); an empty table would raise `ConfigError` from `BaseNode.setup` and `BaseNode.prepare`.
`_assemble_substrate` paints every `node_gene.attr_specs` column with `0`, so payload row option index `0` selects the constructor's callables in every substrate node of every individual.
`HyperNEATNode` implements `combine(state, attrs, aggregated) -> aggregated` — a substrate node has no bias/response affine map, and `BaseNode.combine` is abstract.
`HyperNEATNode` inherits `BaseNode.forward(state, attrs, messages, is_output_node, valid_mask)` unchanged: the body is `apply_activation(combine(aggregate(...)))`, dead slots are neutralised per aggregation option inside `apply_aggregation` (a node with no live slot gets exactly `0.0`), and output rows are left unactivated because `BaseNode.prepare` writes `ACT_IDENTITY` into their activation id.
`HyperNEATNode.repr`/`to_dict` are host-only and render the two option ids as registry names via the inherited `_aggregation_name` / `_activation_name` helpers, e.g. `{"idx": 3, "agg": "sum", "act": "sigmoid"}`.
`HyperNEATConn(BaseConn)` inherits `fixed_attrs = ("input_index", "output_index")` and declares one `GeneAttr(name="weight", kind="float", init_mean=0.0, init_std=1.0, mutate_power=0.15, mutate_rate=0.8, replace_rate=0.1)` → `key_width == 2`, `attr_width == 1`.
`HyperNEATConn.__init__` resolves `self.weight_index` from the `attr_specs` names, never from an offset, so adding a spec cannot silently shift the weight column; the index is used by the host-only `repr`/`to_dict` only.
`HyperNEATConn.forward(state, attrs, source_values)` returns `source_values * self.column_of(attrs, "weight")`, so it accepts both the raw `(..., K, CA)` rows and the `ConnOperands` bundle the genome actually passes (`genome/base.py:1084` passes `hit_conn_attrs.conn`, built once per forward pass by `BaseGenome._hit_conn_attrs`); the weight column comes from the bundle's NaN-cleaned `columns[attr_index("weight")]`.
The conn kernel does no NaN guarding: dead slots are masked once through the node gene's `valid_mask`, so masking has a single owner and the multiply stays one op.
Both genes implement the cheap host-only `repr`/`to_dict` hooks (`{"idx", "agg", "act"}` and `{"in", "out", "weight"}`) so `genome.network_dict`/`repr` work on a substrate genome.

## Substrate Protocol (host-only, pinned in `./substrate`)
`BaseSubstrate` is a plain host object — no `State`, no `setup`, never traced — exposing `connection_type` (`"feedforward"` / `"recurrent"`), `num_inputs` (input neurons *including* the trailing bias), `num_outputs`, `nodes_cnt`, `conns_cnt`, `query_coors (conns_cnt, 2*CD)` float32, `nodes (nodes_cnt,)` int32 and `conn_index (conns_cnt, 2)` int32.
Row `q` of `query_coors` must correspond 1:1 with row `q` of `conn_index`; the substrate's `connection_type` selects the executor genome (`recurrent` → `HyperNEAT`, `feedforward` → `HyperNEATFeedForward`).
Node keys must be the block order `inputs (0..I-1)`, `outputs (I..I+O-1)`, `hidden` — the genome's `is_output_node` test and `input_idx`/`output_idx` ranges depend on it, and `HyperNEAT._validate` re-checks it instead of letting it surface as a silently disabled activation.
This node reads only those tables: it never calls `make_nodes`/`make_conns`, never inspects coordinate ordering, and never lets the substrate layer know a gene column order.

## Decode helpers (implemented surface of `hyperneat.py`)
`_validate()` is pure host Python/NumPy over the configuration: CPPN query width (legacy wording "Query coors of Substrate should be equal to NEAT input size"), `0 <= weight_threshold < 1`, `max_weight > 0`, `substrate.num_inputs >= 2`, `activate_time >= 1`, `neat.num_outputs == 1`, `query_coors.shape[0] == conns_cnt`, `conn_index.shape == (conns_cnt, 2)`, `nodes.shape == (nodes_cnt,)` and a permutation of `0..nodes_cnt-1`, the node-key block order, and `_required_connection_type`.
`_static_bounds()` derives the executor genome's static bounds from the substrate and returns `{"max_in_degree": K}` plus `{"max_depth": D}` for the feedforward executor only.
`max_in_degree` is the exact maximum in-degree `bincount(conn_index[:, 1], minlength=nodes_cnt).max()`, floored at `1` and at `substrate.num_inputs`; the `num_inputs` floor exists because the genome constructor validates `max_in_degree` against its (never used) dense initial `num_inputs -> num_outputs` network (`genome/base.py:475-484`) and would otherwise reject the bound.
The in-degree bound is *per destination node*, so for a full/MLP substrate it is the number of lower-layer neurons wired into one neuron: a `FullSubstrate` with 4 inputs and 3 fully interconnected hidden neurons needs `K = 7` (4 inputs + 3 hidden, self-loop included), and `MLPSubstrate(layers=[4, 5, 5, 5, 1])` needs `K = 5`.
`max_depth` is derived only when `_required_connection_type == "feedforward"`: it is the longest path length from the substrate inputs, `max(host_graph.node_depth(nodes, conn_index, range(num_inputs)).values())` floored at `1`, and a deep feedforward substrate can no longer leave its tail at `NaN` (`Transformed.overflow` covers only the in-degree window, never depth).
`node_depth` raises `GraphError` (a `TensorNEATError`) for a cyclic or input-unreachable substrate; that error propagates unchanged, never caught here.
The recurrent executor (`RecurrentGenome`) ignores `max_depth` — it ticks a fixed `activate_time` — so the recurrent variant passes no `max_depth` at all.
`_make_hyper_genome()` builds `_genome_cls(num_inputs=substrate.num_inputs, num_outputs=substrate.num_outputs, max_nodes=substrate.nodes_cnt, max_conns=substrate.conns_cnt, node_gene=HyperNEATNode(aggregation, activation), conn_gene=HyperNEATConn(), output_transform=output_transform, **_static_bounds(), **_genome_extra_kwargs())`; the mutation/crossover/distance operators fall back to the genome defaults and are never invoked, because the substrate payload is synthesised, not evolved.
`_graph_constant(array)` is `evoxc.constant(evoxc.from_numpy(np.ascontiguousarray(array)))`; all substrate tables together are a few KiB, far under the 1 MiB `EVOXC_LARGE_CONSTANT_BYTES` warning threshold, and `__init__` never calls it (no graph work outside a trace).
`_query_coors()` reshapes the `(Q, C)` constant to `(1, Q, C)` and `enp.broadcast_to` it to `(P, Q, C)` — never a Python loop over `P` and never `evoxc.tile` (no batching rule).
`_expand_schedule(transformed, n_query, pop_size)` is `evoxc.tree_map` over the CPPN schedule with `_expand_leaf`; `_expand_leaf` rewrites a leaf whose static axis 0 is the decoded batch `pop_size` into `(P * n_query, *leaf.shape[1:])` via `reshape → broadcast_to → reshape`, and returns any leaf without that axis unchanged (the docstring's "shared ordering table" case).
`_query_substrate` runs the query with `self.neat.genome.forward(state, expanded, *expanded_pop, queries)`; `*expanded_pop` relies on the `GenomeTensors` field order `(node_keys, node_attrs, conn_keys, conn_attrs)` matching `genome.forward`'s parameter order.
`_weights_from_query` is `sign(w) * max(|w| - weight_threshold, 0) * max_weight / (1 - weight_threshold)` built from `enp.abs`/`enp.greater`/`evoxc.select`/`enp.subtract`/`enp.sign`/`enp.multiply`; `sign(0) == 0` and `max(|w| - t, 0) == 0` for `|w| <= t`, so a tie with the threshold lands on `0` (a pruned edge), and NaN stays NaN by contract.
`_assemble_substrate(weights)` returns a `GenomeTensors`: `node_keys` is the broadcast `(N,)` constant, `node_attrs` has one zero column per `node_gene.attr_specs` entry (two for `HyperNEATNode`; an `int` spec stores its option index as float32, and index `0` is the first — here the only — configured option) or an explicit `(P, N, 0)`, `conn_keys` takes columns 0/1 from the `(Q, 2)` `conn_index` constant and fills any further key column (`OriginConn`'s `historical_marker`) with the unique row id `enp.arange(1, Q + 1)`, and `conn_attrs` puts `weights` in the `"weight"` column with zeros elsewhere.
`_attr_columns` derives column positions from `attr_specs` names and raises `TensorNEATError` when a conn gene has no `"weight"` column (the decoded weights would be dropped).

## Divergences from the Legacy JAX Baseline (deliberate fixes)
1. `HyperNEATNode` used to have no attribute table and an option-empty aggregation/activation pair; it now declares the two `int` option columns with one option each and inherits `BaseNode.forward`, so the node algebra (masked aggregation, output-node rule) has a single implementation.
2. The legacy one-argument `aggregation(inputs)` call is gone: every `AGG` member is `(z, mask)`, and dead slots are neutralised per option inside `apply_aggregation`.
3. `jax.lax.cond(is_output_node, ...)` became `ACT_IDENTITY` written into the output rows' activation id inside `BaseNode.prepare`: `cond` needs a scalar predicate, and the batch is heterogeneous.
4. `vmap(self.neat.forward, in_axes=(None, None, 0))` over query rows became `_expand_schedule` + the flattened `(P * Q, C)` forward — evoxc has no `vmap` over region-bearing kernels.
5. `pop_size` was assigned as an instance attribute while the base protocol declares a property — it is now the property `neat.pop_size`.
6. Constructor `assert`s became `TensorNEATError` raises, and validation gained `0 <= weight_threshold < 1`, `max_weight > 0`, `substrate.num_inputs >= 2`, `activate_time >= 1`, `neat.num_outputs == 1`, the substrate table row counts and the node-key block order.
7. A multi-output CPPN used to misalign weights silently (`query_res.flatten()` onto one column) — it is now rejected at construction.
8. `HyperNEATFeedForward` no longer re-implements the constructor body or the substrate assertions: it overrides two class attributes, `_genome_extra_kwargs()` and the public `__init__` signature, so the two variants cannot drift.
9. The legacy single gene row `[in, out, weight]` cannot exist under evoxc (no bitcast), so the decode emits the four split payload arrays and the weight column is found by name.
10. The executor genome used to inherit the default `max_in_degree = max(initial in-degree, 4)` and a fixed `max_depth = 16`; both now come from the substrate via `_static_bounds()`, so a dense substrate no longer overflows its window and a deep feedforward substrate no longer truncates.

## Dependencies (cross-node contracts)
`../neat`: `setup/ask/tell/show_details`, properties `pop_size`, `num_inputs`, `num_outputs`, `transform(state, pop)`, and the payload pytree `ask` returns (a `GenomeTensors`).
`../neat`'s **`.genome` attribute** is required by `_query_substrate` (see Known Issues).
`../../genome`: `RecurrentGenome(...)` (with `activate_time`) and `DefaultGenome(...)`, both constructed as `(num_inputs=substrate.num_inputs, num_outputs=substrate.num_outputs, max_nodes=substrate.nodes_cnt, max_conns=substrate.conns_cnt, node_gene=..., conn_gene=..., output_transform=..., [activate_time=...])` plus the `max_in_degree` / `max_depth` bounds of `_static_bounds()`.
`../../genome`: `transform(state, node_keys, conn_keys) -> Transformed` and `forward(state, transformed, node_keys, node_attrs, conn_keys, conn_attrs, inputs (B, I)) -> (B, O)`, with `node_gene`/`conn_gene` attribute access, `BaseNode.combine`/`aggregate`/`prepare`, `BaseConn.column_of` and the `GeneAttr`/`BaseNode`/`BaseConn` base classes.
`../../genome.utils.GenomeTensors`: the payload NamedTuple whose field order is the `*payload` unpacking contract.
`../../common`: `State`, `functions.{ACT, AGG}` (aggregation callables are `(z, mask)`), `errors.{TensorNEATError, ConfigError}`, `tools.to_numpy` (host-only gene `repr`/`to_dict`), and `host_graph.node_depth` (imported from the module, not from `common.__init__`).
`../base.py`: `BaseAlgorithm`, including `show_details` as the only host-side member.
`problem`: `evaluate` reads only `static_batch_size(transformed)`, so the `SubstrateSchedule` container is legal as long as every leaf keeps axis 0 = population.
`problem`'s `FuncFit.show` rebuilds the nested schedule with `evoxc.tree_unflatten`, so a `SubstrateSchedule` survives the round trip and `Pipeline.show` reports a decoded substrate.
`pipeline.py`: creates `randkey`; this node seeds nothing and draws no randomness.

## Backend Behaviour (`iree`)
`backend="iree"` (iree-base-compiler 3.11.0, llvm-cpu) compiles and runs this node's whole generation graph end to end, so HyperNEAT is no longer the holdout among the shipped cases.
The graph needs nothing beyond `broadcast_to`/`reshape`/`concatenate`/`gather`-class static ops from the CPPN decode, and the one exporter blocker it had was the negative `concatenate` axis documented under `## Constraints`.
Reference measurement, `HyperNEAT(FullSubstrate(4 inputs, 3 hidden, 1 output), NEAT(pop_size=20, species_size=5, survival_threshold=0.1, genome=DefaultGenome(num_inputs=4, num_outputs=1, output_transform=ACT.tanh)), activation=ACT.tanh, activate_time=3, output_transform=ACT.sigmoid)` + `XOR3d()`, seed 42, `setup()` + 3 `step()`s:
`iree` — one `evoxc.build` inside the first `step` at `build_seconds == 10.32` (`builds == 1`), then 35.8 / 36.4 ms per step; max fitness −0.375114 / −0.375004 / −0.375004.
`numpy` — `build_seconds == 1.96`, 32.8 / 33.1 ms per step, the same max fitness.
Cross-backend the `node_keys`/`conn_keys` payloads are exactly equal, the Node/conn attr NaN padding patterns are exactly equal (3600/3600 and 1920/1920 NaN cells at step 0), the finite attr cells and the fitness differ by ≤ 7.2e-7 (float32 reassociation), and `randkey`/`generation` are bit-identical — the decode is weight-identical across backends, only the last bits of the reductions move.

## Constraints
No `jax` / `torch` / `numpy.random` on the compute path; `numpy` is used only in `__init__`/`_static_bounds` validation, for `np.ascontiguousarray` before `evoxc.from_numpy`, and in host-only `repr`/`to_dict`.
No (un)vectorization of any kind: no `vmap`, no `fori_loop`, no `switch`; this node contains no `cond`/`while_loop`/`scan` of its own and every kernel is region-free and batch-first.
Axes in graph ops are written as non-negative static integers: `forward` concatenates the bias column with `axis=1`, `_conn_keys` with `axis=2` and `_attr_columns` stacks with `axis=2`, never `-1`, because the StableHLO exporter forwards a `concatenate` axis verbatim as `stablehlo.concatenate`'s `dimension`, which StableHLO requires to be non-negative — a `-1` axis makes `iree-compile` reject the whole generation graph.
Do not reintroduce a negative axis in any other graph op either (`gather`/`scatter`/`slice`/`broadcast`/reduction `axis=`): `evoxc.stack` normalises a negative axis internally, every other op does not, so the rule is uniform here.
No `.at[...]` updates, no `tile` on batched data (evoxc's `tile` has no batching rule), no `repeat`, no `max/sum(where=…)`, no `any`/`all`: mask with `evoxc.select`, reduce afterwards.
`evoxc` `getitem` is static-only and rejects `Ellipsis`, so column selection goes through the genes' `column_of` (which wraps `genome.gene.base.column`'s explicit `(slice(None),) * (ndim - 1) + (index,)` key) rather than a literal last-axis index.
Python `if`/`for` over constructor configuration is legal and unrolls at trace time; branching on traced values is a `TraceError`.
All shapes must be static: `Q`, `N`, `I`, `O`, `NA`, `CK`, `CA` come from configuration, and `P` from the payload argument's static axis-0 length — never from a value only known at run time.
Nothing here mutates `State`, the substrate, the genome objects, or any array in place; the substrate geometry is read-only.
User-facing validation raises `TensorNEATError`/`ConfigError`, never `assert`; backend/shape errors propagate unchanged.
Files stay under ~1000 lines (`hyperneat.py` ≈ 820, `genes.py` ≈ 300).

## Known Issues
`NEAT.forward(state, transformed, inputs)` cannot serve the `(P * Q)` query batch because it reads the un-expanded population from `state`; the decode therefore calls `neat.genome.forward` directly and depends on the `NEAT.genome` attribute plus the `GenomeTensors` field order. If `../neat` renames the attribute or changes the payload arity, `_query_substrate` breaks first.
No test file for this node exists: `test/unit/algorithm/` holds only `test_neat_tell.py`/`test_parent_selection.py`/`test_spawn.py`/`test_speciation.py` (all hard-coded skips, 71 skipped), `test/parity/algorithm/` does not exist, and `test/integration/` has no HyperNEAT case; the only reference is the import-name check in `test/release_gate_no_jax.py`.
`_static_bounds` covers in-degree and feedforward depth only; a recurrent substrate has no depth bound at all (`activate_time` ticks replace it), so one that needs more than `activate_time` propagation steps must raise `activate_time` explicitly.

## Test Strategy
Tests live in the sibling `./../../../test` tree (read-only from here) and mirror this node; the coverage below is the specification the missing `test/unit/algorithm/hyperneat/` directory must satisfy.
`test/unit/algorithm/hyperneat/test_weights.py`: the weight map against hand-computed values at `0`, `±weight_threshold`, `±(weight_threshold + ε)` and `±large`, plus NaN-in → NaN-out.
`test/unit/algorithm/hyperneat/test_node_gene.py`: masked aggregation with mixed `valid_mask` (absent edges contribute `0`), output nodes aggregated but not activated, a `(P, N, K)` batch agreeing with a per-individual loop, and `HyperNEATNode`'s declared `attr_width == 2` / single-option tables.
`test/unit/algorithm/hyperneat/test_conn_gene.py`: `source * weight` column selection from a `ConnOperands` bundle (the spelling the genome actually passes) and from raw rows.
`test/unit/algorithm/hyperneat/test_bounds.py`: `_static_bounds()` on a `FullSubstrate` (`K = inputs + hidden`), an `MLPSubstrate` (`K` = widest lower layer, `D` = layer count) and the recurrent no-`max_depth` case.
`test/unit/algorithm/hyperneat/test_assembly.py`: `_assemble_substrate` shapes and column alignment against `substrate.conn_index`/`substrate.nodes`, `_expand_leaf` row order `p * Q + q`, and the `(P, Q) -> (P * Q)` reshape agreement with `_query_coors`.
`test/unit/algorithm/hyperneat/test_validate.py`: every `TensorNEATError` condition, including the multi-output CPPN and a non-feedforward substrate for `HyperNEATFeedForward`.
`test/parity/algorithm/hyperneat/`: decode parity against `benchmarks/legacy_jax` for the same CPPN population (identical weights, node keys and connection endpoints), allowing only the documented divergences; the tests require the `bench` extra.
Integration: `examples/func_fit/xor_hyperneat.py` and `xor_hyperneat_feedforward.py` must reach their fitness target with a fixed seed, deterministically, on the numpy backend.

## Notes for Agents
Every shape above is static: `Q`/`N`/`I`/`O`/`NA`/`CK`/`CA` come from `self.substrate`/`self.hyper_genome`, and the decode batch `P` from the payload's static axis-0 length, never from a value only known at run time.
`transform` must accept any static batch, not just `neat.pop_size`: `Pipeline.show` derives a one-individual schedule, and the decode is value-identical to the same individual's rows of a full-population forward (verified for batches 1…8).
`_validate` runs before `_make_hyper_genome` (which calls `_static_bounds`), so a bad configuration raises before any genome is constructed; keep that order.
`_validate` checks `weight_threshold`/`max_weight`/`activate_time` with `_require_real` (rejecting `bool`, which is an `int` subclass) before the range checks, so a non-numeric value raises `TensorNEATError`, never `TypeError`; NumPy scalars pass.
`HyperNEATNode.__init__` checks its two options with `_require_callable`, so a non-callable aggregation/activation raises `ConfigError` at construction instead of inside a later trace.
The substrate tables are host data: read them at trace time through `_graph_constant`, never in `__init__` (only `_static_bounds` reads them on the host).
If a substrate ever needs a table over ~1 MiB, stop in-graph and escalate: the coordinates would have to become an explicit executable input, which is a `pipeline.py` change.

## Routing Table
| Path | Responsibility |
| --- | --- |
| `./hyperneat.py` | `SubstrateSchedule`, `HyperNEAT` (decode algebra + protocol delegation), the `_validate`/`_static_bounds`/`_make_hyper_genome`/`_genome_extra_kwargs`/`_graph_constant`/`_query_coors`/`_expand_schedule`/`_expand_leaf`/`_query_substrate`/`_weights_from_query`/`_assemble_substrate`/`_attr_columns`/`_conn_keys` helpers, and the `HyperNEATNode`/`HyperNEATConn` re-export from `./genes.py`. |
| `./genes.py` | `HyperNEATNode` (fixed single-option aggregation/activation columns, identity `combine`) and `HyperNEATConn` (one multiply by the decoded weight via `column_of`), plus their host-only helpers. |
| `./hyperneat_feedforward.py` | `HyperNEATFeedForward`: the feedforward variant delta only (`_genome_cls`, `_required_connection_type`, `_genome_extra_kwargs`, `__init__` signature). |
| `./substrate` | Host-only substrate geometry (`BaseSubstrate`, `DefaultSubstrate`, `FullSubstrate`, `MLPSubstrate`): coordinate → node/connection analysis, `connection_type` contract, `query_coors`/`conn_index` tables, shared geometry helpers. |
| `./__init__.py` | Re-exports the two algorithms, `SubstrateSchedule`, the two gene kernels and the four substrate classes. |
