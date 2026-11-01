# HyperNEAT (`tensorneat.algorithm.hyperneat`)

## Intent
HyperNEAT evolves only a CPPN (an ordinary NEAT network from `../neat`) and derives a *substrate* network from it: the CPPN is queried once per substrate connection at the coordinates of that connection's two endpoints, and its single scalar output becomes that connection's weight.
The substrate topology is fixed at construction, so this node turns a whole CPPN population into one batch-first execution schedule of a dense, identical-shaped phenotype network — no padding, no NaN rows.
Three things live here: the decode (CPPN → weights → gene payload → schedule), the two concrete variants (`HyperNEAT` recurrent, `HyperNEATFeedForward` feedforward), and the two substrate gene kernels (`HyperNEATNode`, `HyperNEATConn`).
All evolution, speciation and `randkey` consumption stay in `../neat`; the substrate object is a host-side geometry table and never enters `State`.

## API Surface
`hyperneat.py`: `HyperNEAT(BaseAlgorithm)`, `HyperNEATNode(BaseNode)`, `HyperNEATConn(BaseConn)`.
`HyperNEAT.__init__(self, substrate, neat, weight_threshold=0.3, max_weight=5.0, aggregation=AGG.sum, activation=ACT.sigmoid, activate_time=10, output_transform=ACT.sigmoid)`.
`HyperNEAT` implements the full `../../algorithm/base.py` protocol — `setup`, `ask`, `transform`, `forward`, `tell`, `show_details`, `num_inputs`, `num_outputs`, `pop_size` — by delegating evolution to `self.neat` and decoding with `self.hyper_genome`.
`setup(state)` chains `self.neat.setup(state)` then `self.hyper_genome.setup(state)`; the substrate is stateless and has no `setup`.
`ask(state)` returns the CPPN population untouched (`self.neat.ask`), at zero op cost: the phenotype is not part of the evolved population.
`tell(state, fitness)`, `show_details(state, fitness)` and `pop_size` delegate to `self.neat`; `pop_size` is a read-only property, not an instance attribute.
`num_inputs` = `substrate.num_inputs - 1` (the trailing bias removed) and `num_outputs` = `substrate.num_outputs`; `Pipeline` asserts the former against `problem.input_shape[-1]`.
`forward(state, transformed, inputs)` takes `(P, substrate.num_inputs - 1)`, appends a trailing column of ones and returns `(P, substrate.num_outputs)`.
Subclass hooks used by the feedforward variant: class attrs `_genome_cls` (executor genome class) and `_required_connection_type`, plus `_genome_extra_kwargs()` (the recurrent `activate_time`).
`hyperneat_feedforward.py`: `HyperNEATFeedForward(HyperNEAT)` with `_genome_cls = DefaultGenome`, `_required_connection_type = "feedforward"`, `_genome_extra_kwargs() -> {}`, and an `__init__` that merely drops `activate_time` from the signature; every kernel and every protocol method is inherited.
`__init__.py` re-exports `HyperNEAT`, `HyperNEATFeedForward`, `HyperNEATNode`, `HyperNEATConn`, `BaseSubstrate`, `DefaultSubstrate`, `FullSubstrate`, `MLPSubstrate`.

## CPPN → Substrate Decode Algebra
Static sizes, all fixed at construction: `P = neat.pop_size`, `Q = substrate.conns_cnt`, `N = substrate.nodes_cnt`, `I = substrate.num_inputs` (substrate inputs *including* the bias), `O = substrate.num_outputs`, `C = neat.num_inputs = substrate.query_coors.shape[1] = 2 * CD` (CPPN input width), `node_len = hyper_genome.node_gene.length`, `conn_len = hyper_genome.conn_gene.length`.
`I` (substrate input neurons) and `C` (CPPN query width) are different numbers and must never be interchanged: `I` sizes the bias column of `forward`, `C` sizes the coordinate rows of `query_coors`.
`HyperNEAT.transform(state, pop)` performs, in order:
1. `transformed_cppn = self.neat.transform(state, pop)` — batch-first CPPN schedule, individual on axis 0 of every leaf.
2. `_query_coors()` → `(P, Q, C)`: `substrate.query_coors (Q, C)` (row `q` = `[source_coor, target_coor]`) becomes a graph constant and is repeated per individual with `enp.broadcast_to` — never a Python loop over `P`.
3. `_expand_schedule(transformed_cppn, Q)`: every leaf carrying the population axis becomes `(P * Q, *leaf.shape[1:])` in the flat order `p * Q + q`; leaves without a population axis are left untouched and broadcast inside the genome kernels. This is the evoxc-native replacement for the legacy `vmap(neat.forward, in_axes=(None, None, 0))`.
4. `_query_substrate(state, pop)` → `query_res (P, Q)`: `self.neat.forward(state, tiled_schedule, queries (P * Q, C))` returns `(P * Q, 1)` and is reshaped `(P * Q,) → (P, Q)`.
5. `_weights_from_query(query_res)` → `weights (P, Q)` in `[-max_weight, max_weight]`: `sign(w) * max(|w| - weight_threshold, 0) * max_weight / (1 - weight_threshold)` — exactly the legacy three-`where` chain (dead-zone zero, `±weight_threshold` continuity shift, scale) collapsed into two elementwise ops.
6. `_assemble_substrate(weights)` → `h_nodes (P, N, node_len)`, `h_conns (P, Q, conn_len)`: node rows come from the static `substrate.nodes (N,)` keys (broadcast along axis 0), connection rows come from the static `substrate.conn_index (Q, 2)` endpoint keys plus the per-individual weight column.
7. `return self.hyper_genome.transform(state, h_nodes, h_conns)` — the returned `Transformed` pytree is genome-owned and opaque to this node.

`_graph_constant(array)` isolates the trace-time constant idiom `evoxc.constant(evoxc.from_numpy(np.ascontiguousarray(array)))`; all substrate tables together are a few KiB, far under the 1 MiB `EVOXC_LARGE_CONSTANT_BYTES` warning threshold, and `__init__` must never call it (no graph work outside a trace).
`neat.pop_batch_size` does not chunk this pass: the CPPN query is a single whole-population forward, so peak memory scales with `pop_size * conns_cnt` and with the tiled schedule — small in practice, but it is why the schedule tiling is `broadcast_to`-based rather than a materialised `Q`-fold copy in host code.
A shape change (`pop_size`, `conns_cnt`, `nodes_cnt`) changes the traced graph, so the pipeline rebuilds the executable; nothing in this node caches per-shape state.

## Substrate Protocol (host-only, pinned in `./substrate`)
`BaseSubstrate` is a plain host object — no `State`, no `setup`, never traced — exposing `connection_type` (`"feedforward"` / `"recurrent"`), `num_inputs` (input neurons *including* the trailing bias), `num_outputs`, `nodes_cnt`, `conns_cnt`, `query_coors (conns_cnt, 2*CD)` float32, `nodes (nodes_cnt,)` int32 and `conn_index (conns_cnt, 2)` int32.
Row `q` of `query_coors` must correspond 1:1 with row `q` of `conn_index`; the substrate's `connection_type` selects the executor genome (`recurrent` → `HyperNEAT`, `feedforward` → `HyperNEATFeedForward`).
Node keys must be the block order `inputs (0..I-1)`, `outputs (I..I+O-1)`, `hidden` — the genome's `is_output_node` test and `input_idx`/`output_idx` ranges depend on it, and `__init__` validates it instead of letting it surface as a silently disabled activation.
This node reads only those tables: it never calls `make_nodes`/`make_conns`, never inspects coordinate ordering, and never lets the substrate layer know a gene column order.

## Gene Contract
`HyperNEATNode(BaseNode)` declares `fixed_attrs = ["index"]`, `custom_attrs = []` → `node_len == 1`: a substrate node carries no evolvable attributes because aggregation and activation are fixed at construction.
`HyperNEATNode.__init__(aggregation=AGG.sum, activation=ACT.sigmoid)`; `forward(state, attrs, ins, valid_mask, is_output_node)` is batch-first over arbitrary leading axes with `attrs (..., node_len)`, `ins (..., max_sources)`, `valid_mask (..., max_sources)` bool, `is_output_node` broadcastable bool.
`HyperNEATNode.forward` masks with `evoxc.select(valid_mask, ins, 0.0)`, aggregates with the two-argument callable `self.aggregation(z, valid_mask)`, and applies `activation` through one `evoxc.select(is_output_node, z, activated)` — output nodes are aggregated but never activated (the substrate's output activation is `output_transform`), and a heterogeneous batch can never use `evoxc.cond`.
`HyperNEATConn(BaseConn)` declares `fixed_attrs = ["input_index", "output_index"]`, `custom_attrs = ["weight"]` → `conn_len == 3`, rows `[input_index, output_index, weight]` with the weight column repainted on every `transform`.
`HyperNEATConn.forward(state, attrs, inputs)` returns `inputs * attrs[..., len(self.fixed_attrs)]` over `attrs (..., max_sources, conn_len)` and `inputs (..., max_sources)` — the weight index is derived from `fixed_attrs`, never a literal `-1`.
The conn kernel does no NaN guarding: absent edges are zeroed once by the node kernel's `valid_mask`, so masking has a single owner and the multiply stays one op.

## Divergences from the Legacy JAX Baseline (deliberate fixes)
1. `HyperNEATNode.forward` was signature-incompatible with the gene protocol (missing `valid_mask`, called as keyword by one genome and as a 5th positional by the other) — it now takes `(state, attrs, ins, valid_mask, is_output_node)` positionally.
2. `HyperNEATNode.forward` called `self.aggregation(inputs)` with one argument while every `AGG` member is `(z, mask)` — it now masks and aggregates through the two-argument contract, which also implements the missing invalid-source masking.
3. `jax.lax.cond(is_output_node, ...)` became `evoxc.select(is_output_node, z, activated)`: `cond` needs a scalar predicate, the batch is heterogeneous, and evoxc forbids control flow inside vectorized contexts.
4. `vmap(self.neat.forward, in_axes=(None, None, 0))` over query rows became `_expand_schedule` + a single flattened `(P * Q, C)` forward — evoxc has no `vmap` over region-bearing kernels.
5. `pop_size` was assigned as an instance attribute while the base protocol declares a property — it is now the property `neat.pop_size`.
6. Constructor `assert`s became `TensorNEATError` raises, and validation gained `0 <= weight_threshold < 1`, `max_weight > 0`, `substrate.num_inputs >= 2`, `activate_time >= 1`, `neat.num_outputs == 1`, `query_coors.shape[0] == conn_index.shape[0] == conns_cnt`, `nodes.shape[0] == nodes_cnt` and the node-key block order.
7. A multi-output CPPN used to misalign weights silently (`query_res.flatten()` onto one column) — it is now rejected at construction.
8. `HyperNEATFeedForward` no longer re-implements the constructor body or the substrate assertions: it overrides two class attributes, `_genome_extra_kwargs()` and the public `__init__` signature, so the two variants cannot drift.
9. `HyperNEAT.transform` no longer returns a variant-dependent 3-tuple vs 4-tuple; it returns whatever `hyper_genome.transform` returns, so the schedule layout is the genome's contract, not this node's.

## Dependencies (cross-node contracts)
`../neat`: `NEAT.setup/ask/tell/show_details`, static `pop_size`, `num_inputs`, `num_outputs`, and the batch-first `transform(state, pop)` / `forward(state, transformed, inputs (P, *in)) -> (P, *out)` pair this node re-drives at a `(P, Q, C)` query grid.
`../../genome`: `RecurrentGenome(...)` (with `activate_time`) and `DefaultGenome(...)` — both constructed as `(num_inputs=substrate.num_inputs, num_outputs=substrate.num_outputs, max_nodes=substrate.nodes_cnt, max_conns=substrate.conns_cnt, node_gene=..., conn_gene=..., output_transform=..., [activate_time=...])` — plus the batch-first `transform(state, nodes (P, N, node_len), conns (P, Q, conn_len))` and `forward(state, transformed, inputs (P, I))`.
`../../genome/gene`: the payload builders `node_gene.from_attrs(keys (N,) int32)` → `(N, node_len)` and `conn_gene.from_attrs(in_keys (Q,), out_keys (Q,), weights (P, Q))` → `(P, Q, conn_len)`, broadcasting the structural columns and encoding the traced float weights into the weight slot.
The payload builders must be traceable over traced float attributes and must not rely on a bitcast: **evoxc has no reinterpret/bitcast op**, so the float→row encoding is entirely genome-owned (see the package CONTEXT note on gene payload encoding).
`../../common`: `State`, `functions.{ACT, AGG}` (callables with signature `(z, mask)` for aggregation), `errors.TensorNEATError`.
`../base.py`: `BaseAlgorithm` — the protocol this node implements, including `show_details` as the only host-side member.
`pipeline.py`: creates `randkey` and the epoch; this node seeds nothing and draws no randomness.

## Constraints
No `jax` / `torch` / `numpy.random` on the compute path; `numpy` is used only in `__init__`-time validation and to coerce host substrate tables before `evoxc.from_numpy`.
No (un)vectorization of any kind: no `vmap`, no `fori_loop`, no `switch`; this node contains no `cond`/`while_loop`/`scan` of its own and every kernel is region-free and batch-first.
No `.at[...]` updates, no `tile` on batched data (evoxc's `tile` has no batching rule), no `repeat`, no `max/sum(where=…)`, no `any`/`all`: mask with `evoxc.select`, reduce afterwards.
Python `if`/`for` over constructor configuration is legal and unrolls at trace time; branching on traced values is a `TraceError`.
All shapes must be static: `P`, `Q`, `N`, `I`, `O`, `node_len`, `conn_len` come from configuration, never from a runtime tensor shape.
Files stay under ~1000 lines: `hyperneat.py` holds the algorithm and its decode helpers plus the two gene kernels (~450 lines), `hyperneat_feedforward.py` holds only the three-variant delta (~100 lines).
Nothing here mutates `State`, the substrate, the genome objects, or any array in place; the substrate geometry is read-only.

## Test Strategy
Tests live in the sibling `./../../../test` tree (read-only from here) and mirror this node.
`test/unit/algorithm/hyperneat/test_weights.py`: the weight map against hand-computed values at `0`, `±weight_threshold`, `±(weight_threshold + ε)` and `±large` (range `[-max_weight, max_weight]`), plus NaN-in → NaN-out.
`test/unit/algorithm/hyperneat/test_node_gene.py`: masked aggregation with mixed `valid_mask` (absent edges must contribute `0`), output nodes aggregated but not activated, and the `(P, N, S)` batch agreeing with a per-individual loop.
`test/unit/algorithm/hyperneat/test_conn_gene.py`: `inputs * weight` column selection under a batched `attrs (..., S, 3)`.
`test/unit/algorithm/hyperneat/test_assembly.py`: `_assemble_substrate` row/column alignment against `substrate.conn_index` and `substrate.nodes`, `broadcast_to`-identical node rows across `P`, and the `(P * Q) → (P, Q)` ordering of the query reshape (row `p * Q + q` uses individual `p`, query `q`).
`test/unit/algorithm/hyperneat/test_validation.py`: every `TensorNEATError` condition, including the multi-output CPPN and a non-feedforward substrate for `HyperNEATFeedForward`.
`test/parity/algorithm/hyperneat/`: decode parity against `benchmarks/legacy_jax` for the same CPPN population (identical weights, node rows and connection rows for a `FullSubstrate`/`MLPSubstrate`), allowing only the documented divergences above; the tests require the `bench` extra.
Integration: `examples/func_fit/xor_hyperneat.py` and `xor_hyperneat_feedforward.py` reach their fitness target with a fixed seed, deterministically, on the numpy backend.

## Notes for Agents
Every shape above is static: derive it from `self.neat` and `self.substrate` at construction, never from a traced tensor shape.
When touching the decode, keep step ordering intact — the `(P * Q)` flat row order of `_expand_schedule` and the `(P, Q, C)` reshape of `_query_coors` are one contract; changing one without the other silently permutes weights across connections.
The substrate tables are host data: read them at trace time through `_graph_constant`, never in `__init__`.
If a substrate ever needs a table over ~1 MiB, stop in-graph and escalate: the coordinates would have to become an explicit executable input, which is a `pipeline.py` change.

## Routing Table
| Path | Responsibility |
| --- | --- |
| `./hyperneat.py` | `HyperNEAT` (decode algebra + protocol delegation), the `_query_coors`/`_expand_schedule`/`_query_substrate`/`_weights_from_query`/`_assemble_substrate` helpers, `HyperNEATNode` and `HyperNEATConn`. |
| `./hyperneat_feedforward.py` | `HyperNEATFeedForward`: the feedforward variant delta only (`_genome_cls`, `_required_connection_type`, `_genome_extra_kwargs`, `__init__` signature). |
| `./substrate` | Host-only substrate geometry (`BaseSubstrate`, `DefaultSubstrate`, `FullSubstrate`, `MLPSubstrate`): coordinate → node/connection analysis, `connection_type` contract, `query_coors`/`conn_index` tables. |
| `./__init__.py` | Re-exports the two algorithms, the two gene kernels and the four substrate classes. |
