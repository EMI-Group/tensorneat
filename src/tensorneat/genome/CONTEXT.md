# genome — gene payloads, genetic operators, and the network forward/grad algebra

## Intent
Turn a NEAT individual into fixed-shape `int32`/`float32` evoxc tensors, and evaluate a whole population of differently-shaped networks in one batch-first kernel.
This node owns exactly four things: (1) the **gene payload** (key columns in `int32`, attribute columns in `float32`, padded rows), (2) the **genetic operators** (mutation / crossover / distance) as batched kernels, (3) the **forward algebra** over a precomputed execution schedule, and (4) host-only introspection (`network_dict`, `sympy_func`, `visualize`, `repr`, `hash`).
It does not own the schedule derivation (`../common/tensorization.py`) nor `ACT`/`AGG` (`../common/functions/`); it consumes both.

## Gene payload (binding)
Two arrays per gene type, batch-first, static shapes, no NaN in integers:
- `node_keys (P, max_nodes) int32` — node key per row; `I_INF` marks an absent row.
- `node_attrs (P, max_nodes, node_gene.attr_width) float32` — custom attrs; rows for absent nodes are NaN.
- `conn_keys (P, max_conns, conn_gene.key_width) int32` — `(in_key, out_key)` plus `historical_marker` when the conn gene declares it; `I_INF` marks an absent row.
- `conn_attrs (P, max_conns, conn_gene.attr_width) float32` — e.g. `weight`; absent rows are NaN.
`fixed_attrs` (key columns) are integers and stay integers; `custom_attrs` are floats and stay floats.
There is **no bitcast** float↔int in evoxc, so a single mixed-dtype row is impossible: the payload is split into a key array and an attr array, and this split is why genetic operators are integer-index twiddling over `*_keys` while only `*_attrs` does float arithmetic.
Key/row invariant: `initialize` writes nodes in key order, so input keys are rows `0..num_inputs-1` and output keys are rows `max_nodes-num_outputs..max_nodes-1`; those rows are never deleted or moved, so `get_input_idx()`/`get_output_idx()` are usable as *row indices*.
Hidden node keys are arbitrary and may move rows; always compare keys (`node_keys`), never row positions, for graph identity.
Padding sentinels never mix: `I_INF` (`int32` max) for "no index", float NaN for "no value" and for "unconnected node value".

## Genome API (batch-first, population on axis 0)
`BaseGenome` holds host-side configuration (`num_inputs`, `num_outputs`, `max_nodes`, `max_conns`, `max_in_degree`, `max_depth`, transforms, `init_hidden_layers`) and the gene/operator objects; nothing on the compute path reads `State`.
- `initialize(state, key, pop_size) -> GenomeTensors` — `key` is a rank-0 `evoxc.random` key and randomness is drawn once per tensor with the batched output shape, so individual `i` gets an independent stream from one call.
- `transform(state, node_keys, conn_keys) -> Transformed` — schedule derivation, delegated to `../common/tensorization.py`; pure tensor ops, no host round-trip.
- `forward(state, transformed, node_keys, node_attrs, conn_keys, conn_attrs, inputs) -> (P, num_outputs)` — trace-time kernel.
- `grad(state, transformed, node_*, conn_*, inputs, loss_fn) -> evoxc.Graph` — **graph factory** (like `evoxc.grad`), not a trace-time kernel: it traces its own graph and returns it for the Pipeline to cache/compile/run.
- `execute_mutation(state, key, genome, new_node_keys (P,), new_conn_keys (P,3)) -> GenomeTensors`
- `execute_crossover(state, key, genome1, genome2) -> GenomeTensors`
- `execute_distance(state, genome1, genome2) -> (P,)` — pairwise, batched over pairs; the algorithm forms the pairs.
- Host-only (NumPy in, Python out, never traced): `network_dict`, `sympy_func`, `visualize`, `repr`, `hash`, `get_input_idx`, `get_output_idx`.
`GenomeTensors` is a namedtuple-like pytree of exactly the four payload arrays; every kernel takes and returns it by field, never by column offsets.
Structural mutation keys (`new_node_keys`, `new_conn_keys`) are **allocated by the caller** (globally unique across the population) and passed in as int32 tensors — they are not RNG draws.

## Execution schedule (`Transformed`, owned by `../common/tensorization.py`)
Per-population, batch-first, static shapes; field names are a binding contract with `common`:
- `src (P, max_conns) int32`, `dst (P, max_conns) int32` — source/destination node slot of each connection row (`I_INF` = absent row or unmatched key); `edge_valid (P, max_conns) bool` — real edge with both endpoints matched.
- `in_conn (P, max_nodes, K) int32` — connection-row index of the k-th incoming edge of node slot `i`; `I_INF` padding.
- `in_src (P, max_nodes, K) int32` — source **node slot** of that same edge, a per-individual index in `[0, max_nodes)` (not a flat `P*N` index); `I_INF` in the same positions as `in_conn`.
- `in_valid (P, max_nodes, K) bool` — True where the slot holds a live edge; equals `in_conn != I_INF`.
- `node_level (P, max_nodes) int32` — longest-path depth (inputs = 0); `I_INF` for absent, unreachable or cyclic nodes.
- `node_exists (P, max_nodes) bool` — row validity; `topo_key (P, max_nodes) int32` — `node_level * max_nodes + slot`.
- `input_slot (P, num_inputs) int32`, `output_slot (P, num_outputs) int32` — node slot holding each input/output key, in key order; `I_INF` when unmatched.
- `edge_overflow (P,) int32` — valid edges dropped because their in-group rank reached `K`; the `overflow` property is `edge_overflow > 0` (there is no `overflow` field).
- No Python statics live in the pytree: `pop_size`, `max_nodes`, `max_conns`, `max_in_degree`, `key_span` are build-time arguments of `derive_schedule` and part of the executable cache key.
`K` is the genome's `max_in_degree`; slots are filled in increasing connection-row order so a node's window ordering is canonical and stable across runs.

## Forward algebra (feedforward)
`values (P, max_nodes) float32`, all NaN except the input rows, which hold `input_transform(inputs)`.
Unroll depth in **Python** (`for l in range(1, max_depth+1)`) — never `scan`/`while_loop`/`cond`, which evoxc cannot batch or differentiate.
Per level:
1. `xv = values[arange(P)[:, None, None], in_src] -> (P, max_nodes, K)` (a batch-wise gather along axis 1 — `in_src` is a per-individual node slot, never a flat `P*N` index); `live = in_valid & ~isnan(xv)`; `xs = select(live, xv, 0)` — the NaN→0 sanitization is mandatory before any differentiable op, because a zeroed cotangent multiplied by a NaN local derivative still yields NaN.
2. `hit_attrs = conn_attrs[arange(P)[:, None, None], in_conn] -> (P, max_nodes, K, CA)` is computed **once per generation** (attributes are constant during a forward pass); `msg = conn_gene.forward(state, hit_attrs, xs) -> (P, max_nodes, K)`, one scalar message per edge slot.
3. `agg = apply_aggregation(agg_index, msg, live, aggregation_options) -> (P, max_nodes)` — each option reduces the last axis after replacing dead slots with that option's neutral (`0`, `1`, `±inf`), so every branch is NaN-free; options are selected with `select` chains because `select` is the only batched selection op (`cond` is a region op).
4. `z = bias + response * agg` (node gene `combine`); `z = act(z)` unless the node is an output node — output nodes are never activated by the node gene, the final activation belongs to `output_transform`.
5. `update = (node_level == l) & node_exists & (valid_count > 0)`; `values = select(update, z, values)`. A node with no live incoming edge keeps NaN, and its descendants lose their only live source and stay NaN — this reproduces the legacy NaN poisoning exactly.
Read out `gather(values, output_idx, axis=1) -> (P, num_outputs)`, then `output_transform`.
Cost is `O(depth × max_nodes × K)` with only elementwise/reduce/gather ops: work scales with real depth and real in-degree, never `max_nodes²`.
Measured budget (numpy backend, `P=1000`, `max_nodes=50`, `K=2`, ~20 ops per level): ≈1.8 ms per unrolled level, i.e. ≈14 ms for a depth-8 forward against the 157 ms legacy generation — so **op count per level**, not element count, is what the implementation must keep thin.

## Forward algebra (recurrent)
Same window machinery, no levels: unroll `activate_time` ticks in Python.
Each tick: re-inject `input_transform`-free inputs into the input rows, compute every node from the previous tick's values (Jacobi/synchronous), then `values = select(node_exists & valid_count > 0, z, values)`.
`input_transform` is **not** applied by `RecurrentGenome` (legacy parity; HyperNEAT uses it as a CPPN).

## Gradient algebra
No control-flow (region) op exists in the forward, so reverse-mode works; but three evoxc rules shape the API:
- `evoxc.grad` requires a **single scalar output**, so the loss is scalarized as `sum(loss_fn(outputs))`; per-individual gradients are unaffected because the population axis is independent.
- Only float inputs are differentiable: `node_attrs` and `conn_attrs` are the differentiated arguments; keys and the schedule are non-differentiated int inputs (an int32 *differentiated* argument is a hard `TransformError`).
- `sort`/`argsort` have no AD rule, so the schedule derivation must **not** be inside the differentiated graph — this is why `grad` takes an already-derived `transformed` and returns its own graph.
Gradients are cleaned with `select(isnan(g), 0, g)` and by masking absent rows, so padding never carries NaN.

## Static invariants (performance knobs, never silent)
- `max_in_degree` (`K`, default `max(initial in-degree, 4)`): enforced by `execute_mutation`, which refuses an added connection that would push a node past `K`; `initialize` rejects an `init_hidden_layers` layout that needs more.
- `max_depth` (`D`, default 16): a network deeper than `D` leaves the unreachable nodes NaN (they cannot win) and raises `overflow` for host-side diagnostics.
- `activate_time` (default 10) is the recurrent tick count.
`Transformed.overflow` is the only host-visible diagnostic; nothing silently drops a connection or a node.

## Constraints
- No `jax`/`torch` import anywhere in this node; all tensor math is evoxc ops (`enp`, `evoxc.gather/scatter/select/reduce_*/dot`, `evoxc.random`).
- No eager NumPy on the compute path; the host helpers (`network_dict`, `sympy_func`, `visualize`, `repr`, `hash`) convert inputs with `Tensor.numpy()` and never feed results back into a graph.
- Every kernel is already batched: there is no per-individual Python loop, no `vmap`/`vectorize` over these kernels, and no `scan`/`while_loop`/`cond`.
- Shapes are static at trace time: `pop_size`, `max_nodes`, `max_conns`, `K`, `D` are Python ints; a change forces a rebuild (the Pipeline caches per shape signature).
- Integer arithmetic: an `int32` tensor plus a Python literal promotes to `int64` in evoxc, so index math stays in int32 tensors.
- Errors: configuration violations raise `../common/errors.TensorNEATError`; tensor/backend errors from evoxc propagate unchanged — never caught and re-wrapped.
- Files ≤ ~1000 lines; shared machinery (one window-reduce node step) lives in `base.py` and is parameterized by the update mask, never duplicated between `default.py` and `recurrent.py`.

## Deviations from the pre-rewrite library (intentional)
- `activation == -1` means **identity** everywhere; the legacy `lax.switch(-1, ...)` clamped to the last option, so `BiasNode` split nodes silently used the wrong activation.
- Activation/aggregation options are evaluated for **all** configured options and selected with `select`, so a non-total option (division by zero, `log`/`sqrt` of a possibly negative value) can poison gradients even where it is not selected; single-option configurations emit no selection op at all.
- Random draws differ from the JAX build by construction (`evoxc.random` vs `jax.random`), so parity with `benchmarks/legacy_jax` is asserted on kernels for given genomes (forward, grad, distance), not on RNG-dependent operator output.

## Test Strategy
`test/unit/genome/` — numpy backend, no optional dependency: payload layouts and padding sentinels, `initialize` row/key invariant, forward on hand-built XOR networks (including unconnected-node NaN poisoning), `-1` activation identity, in-degree overflow handling, mutation/crossover/distance invariants (counts, key uniqueness, capacity).
`test/parity/genome/` — kernels vs the frozen `benchmarks/legacy_jax` implementation on identical genomes: forward values (`allclose`) and NaN patterns, `grad` vs `jax.grad` on float attributes, and `execute_distance` (deterministic, so it must match exactly).
`test/integration/` — evolve XOR to target with the numpy backend and assert determinism for a fixed seed; the release-gate test asserts no `jax`/`torch` import.

## Routing Table
| Path | Responsibility |
|---|---|
| `./gene/` | `BaseGene` attr-table machinery plus node/conn gene families. |
| `./gene/node/` | `BaseNode`, `DefaultNode`, `BiasNode`, `OriginNode`: key/attr layout, `combine`, activation choice, identity/zero attrs. |
| `./gene/conn/` | `BaseConn`, `DefaultConn`, `OriginConn`: per-edge message, optional `historical_marker`. |
| `./operations/` | Batched mutation / crossover / distance contracts and the in-degree invariant. |
| `./operations/mutation/` | Structural (add/delete node, add/delete conn) and value mutation, `DefaultMutation` and `RecurrentMutation`. |
| `./operations/crossover/` | Key-alignment crossover (`DefaultCrossover`). |
| `./operations/distance/` | Compatibility distance (disjoint/excess + attribute distance). |
| `./base.py` | `BaseGenome`: configuration, `initialize`, the shared window-reduce node step, `grad` factory, host-only helpers. |
| `./default.py` | `DefaultGenome`: depth-unrolled level-synchronous forward. |
| `./recurrent.py` | `RecurrentGenome`: tick-unrolled Jacobi forward. |
| `./utils.py` | Batched genome-array editing helpers (`add_node`, `add_conn`, `delete_*`, `first_free_row`, `fetch_random_index`) and host-only display helpers. |

Out of node but required: `../common/tensorization.py` (`derive_schedule(node_keys, conn_in, conn_out, input_keys, output_keys, pop_size, max_nodes, max_conns, max_in_degree, key_span)` → `Transformed`), `../common/functions/` (`ACT`/`AGG`, `apply_activation`, `apply_aggregation` with `-1` = identity), `../common/tools.py` (`I_INF`), `../common/state.py` (`State`), `../common/errors.py` (`TensorNEATError`).
Consumers: `../algorithm/neat/` (`ask`/`transform`/`forward`/`tell`), `../algorithm/hyperneat/` (`RecurrentGenome` as CPPN), `../pipeline.py` (compile site, per-shape executable cache).
