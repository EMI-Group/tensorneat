# genome — gene payloads, genetic operators, and the network forward/grad algebra

## Intent
Turn a NEAT individual into fixed-shape `int32`/`float32` evoxc tensors, and evaluate a whole population of differently-shaped networks in one batch-first kernel.
This node owns exactly four things: (1) the **gene payload** (key columns in `int32`, attribute columns in `float32`, padded rows), (2) the **genetic operators** (mutation / crossover / distance) as batched kernels, (3) the **forward algebra** over a precomputed execution schedule plus the exact gradient path beside it (`grad.py`), and (4) host-only introspection (`network_dict`, `sympy_func`, `visualize`, `repr`, `get_input_idx`/`get_output_idx`) beside the in-graph `hash`.
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
`BaseGenome` holds host-side configuration (`num_inputs`, `num_outputs`, `max_nodes`, `max_conns`, `max_in_degree`, `max_depth`, `key_span`, transforms, `init_hidden_layers`) and the gene/operator objects; nothing on the compute path reads `State`.
`key_span` (`S`, default `max(max_nodes, num_inputs + num_outputs) + max_conns`) bounds every node key: keys must lie in `[0, S)`, a caller allocating larger keys must raise it, and `S` is part of the Pipeline's shape signature.
A key outside `[0, S)` is *silently unmatched* by the schedule's dense key table: the edge or node becomes invisible without setting `edge_overflow` (which only reports in-degree > `K`), which is this node's one silent-failure mode — a caller that allocates structural keys (the algorithm) must keep them below `S` and raise `key_span` in the shape signature instead.
`DefaultGenome`/`RecurrentGenome` accept `key_span=` explicitly (last keyword before super) so `algorithm/` can raise it past the default for its own structural-key allocation.
`node_exists`, `conn_exists`, `max_node_key`, `max_conn_marker` are the thin in-graph helpers `algorithm/` calls; `execute_mutation`/`execute_crossover`/`execute_distance`/`transform` accept both the `GenomeTensors` payload and the positional four-array spelling.
The shared window step is `_hit_conn_attrs` + `_node_step` + `_output_values`, and `_gather_slots` is the flat-index gather both use (`evoxc.gather` has `np.take` semantics, so the index must be batch-wide and one-dimensional).
`_hit_conn_attrs` returns a `_StepStatics` bundle — gathered+NaN-cleaned conn attributes (`ConnOperands`), the flat value-window index, the present-slot mask, the output-row mask, and the `NodeOperands` built on the first step — because node and conn attributes are constant during a forward pass, so every loop-invariant operand is derived once and a level pays only the value-dependent ops.
- `initialize(state, key, pop_size) -> GenomeTensors` — `key` is a rank-0 `evoxc.random` key and randomness is drawn once per tensor with the batched output shape, so individual `i` gets an independent stream from one call.
- `transform(state, *arrays) -> Transformed` — schedule derivation, delegated to `../common/tensorization.py`; pure tensor ops, no host round-trip; accepts a `GenomeTensors`, `(node_keys, conn_keys)` or the full four-array payload (attributes ignored).
- `forward(state, transformed, node_keys, node_attrs, conn_keys, conn_attrs, inputs) -> (P, num_outputs)` — trace-time kernel.
- `grad(state, transformed, node_*, conn_*, inputs, loss_fn) -> evoxc.Graph` — **graph factory** (like `evoxc.grad`), not a trace-time kernel: it traces its own graph and returns it for the Pipeline to cache/compile/run, and its body is `grad.build_grad(self, ...)`.
- `execute_mutation(state, key, genome, new_node_keys (P,), new_conn_keys (P,3)) -> GenomeTensors`
- `execute_crossover(state, key, genome1, genome2) -> GenomeTensors`
- `execute_distance(state, genome1, genome2) -> (P,)` — pairwise, batched over pairs; the algorithm forms the pairs.
- Host-only, NumPy in / Python out, never traced: `network_dict(state, genome, whether_re_cound_idx=True)`, `repr(state, genome, precision=2)`, `get_input_idx()`, `get_output_idx()` (and the family-implemented `sympy_func(state, network, ...)` / `visualize(network, ...)`), which live in `host.py::BaseGenomeHost`; each payload-taking one needs a `GenomeTensors` holding **one** individual as a `P == 1` batch, because `_split_payload` rejects a `(N,)` row and `_individual` then raises `ConfigError` (a `TensorNEATError`) telling the caller to slice first: `GenomeTensors(*(array[p : p + 1] for array in payload))` (there is no `payload.of(i)`).
- `hash(genome) -> (P,) uint32` is **not** host-only despite living in `host.py`: it is an in-graph kernel over the whole population (`evoxc` ops require an active trace, so an eager call raises `TraceError`).
`GenomeTensors` is a namedtuple-like pytree of exactly the four payload arrays; every kernel takes and returns it by field, never by column offsets.
Structural mutation keys (`new_node_keys`, `new_conn_keys`) are **allocated by the caller** (globally unique across the population) and passed in as int32 tensors — they are not RNG draws.

## Execution schedule (`Transformed`, owned by `../common/tensorization.py`)
Per-population, batch-first, static shapes; field names are a binding contract with `common`:
- `in_src (P, max_nodes, K) int32` — flat index into `reshape(values, (P*max_nodes,))` of the source row of the k-th incoming edge of each node row; in range (0) where the slot is empty.
- `in_conn (P, max_nodes, K) int32` — flat index into `reshape(conn_attrs, (P*max_conns,))` of the same slot.
- `in_valid (P, max_nodes, K) bool` — True where that slot holds a real connection.
- `node_level (P, max_nodes) int32` — depth (longest path from an input node; inputs are 0), `I_INF` for absent rows; zeros in the recurrent variant.
- `node_exists (P, max_nodes) bool`, `overflow (P,) bool` — row validity, and "some node's in-degree exceeded `K`".
- static metadata `K`, `max_nodes`, `max_conns` as Python ints.
`K` is the genome's `max_in_degree`; slots are filled in increasing connection-row order so a node's window ordering is canonical and stable across runs.

## Forward algebra (feedforward)
`values (P, max_nodes) float32`, all NaN except the input rows, which hold `input_transform(inputs)`.
Step 0 (both genomes): `node_attrs = self._padding_safe_attrs(node_attrs, node_keys)` replaces the attrs of **absent** rows with `0` via `select`, so the option-index columns (`activation`, `aggregation`) never reach the int cast as NaN (that cast emitted `RuntimeWarning: invalid value encountered in cast`); a NaN *value* column of a live row is untouched, so real corruption still poisons as documented.
Unroll depth in **Python** (`for l in range(1, max_depth+1)`) — never `scan`/`while_loop`/`cond`, which evoxc cannot batch or differentiate.
Per level:
1. `xv = gather(values_flat, in_src_flat) -> (P, max_nodes, K)`; `live = in_valid & ~isnan(xv)`; `xs = select(live, xv, 0)` — the NaN→0 sanitization is mandatory before any differentiable op, because a zeroed cotangent multiplied by a NaN local derivative still yields NaN.
2. `hit_attrs = gather(conn_attrs_flat, in_conn_flat)` is computed **once per generation** (attributes are constant during a forward pass); `msg = conn_gene.forward(hit_attrs, xs) -> (P, max_nodes, K)`, one scalar message per edge slot.
3. `agg = apply_aggregation(agg_index, msg, live, aggregation_options) -> (P, max_nodes)` — each option reduces the last axis after replacing dead slots with that option's neutral (`0`, `1`, `±inf`), so every branch is NaN-free; options are selected with `select` chains because `select` is the only batched selection op (`cond` is a region op).
4. `z = bias + response * agg` (node gene `combine`); `z = act(z)` unless the node is an output node — output nodes are never activated by the node gene, the final activation belongs to `output_transform`.
5. `update = (node_level == l) & node_exists & (valid_count > 0)`; `values = select(update, z, values)`. A node with no live incoming edge keeps NaN, and its descendants lose their only live source and stay NaN — this reproduces the legacy NaN poisoning exactly.
Read out `gather(values, output_idx, axis=1) -> (P, num_outputs)`, then `output_transform`.
Cost is `O(depth × max_nodes × K)` with only elementwise/reduce/gather ops: work scales with real depth and real in-degree, never `max_nodes²`.
**Op count per level, not element count, is the budget** — the numpy backend pays ≈80 µs per traced op at `P=1000`, `max_nodes=50`, `K=2` regardless of the element count, so every op removed from the unrolled step is worth ≈0.08 ms per level.
Measured on the numpy backend (build excluded, `evoxc.run` median of 7, single-threaded; `max_nodes=50`, `max_conns=100`, `K=2`, depth 8, depth 1 = the per-pass overhead):

| shape | depth 1 | depth 8 | per level |
|---|---|---|---|
| `P=100` | 1.03 ms | 4.20 ms | 0.53 ms |
| `P=500` | 2.42 ms | 13.24 ms | 1.66 ms |
| `P=1000` | 3.78 ms | 24.75 ms | 3.09 ms |

Build (trace + lower + compile) costs 50–180 ms per shape.
Schedule + forward per generation, measured on the same box (`P=1000`, `max_nodes=50`, `max_conns=100`, 96 live edges, depth 8, build excluded, median of 7 `evoxc.run`): `K=2` → `transform` 40.6 ms + `forward` 24.8 ms = 65.4 ms; `K=4` (the default `max_in_degree` of a 2-input genome) → 47.6 + 30.4 = 78.0 ms; `P=100` → 8.2 + 4.8 ms at `K=2`.
The forward scales linearly in `max_depth`, so the default `max_depth=16` roughly doubles the figures above for a genome whose real depth is smaller — `max_depth` is a genuine cost knob the Pipeline owner may lower per problem.
The step now emits **35 traced ops per level** (was 44): the in-scope share is 13 — flatten+gather+reshape of the value window (3), `isnan`+`logical_and` liveness (2), conn-gene `isnan`/`select`/`multiply` (3), node `combine` (2), and `logical_and`+`reduce_max`+`select` for the update (3) — leaving ≈22 ops per level in `../common/functions/` and `default.py`'s level mask, which is why the ≈14 ms target is out of this node's reach.
Remaining hot spots, all outside this node: `apply_activation`'s per-option `equal`+`select` cascade and its `cast` of ids (≈7 ops, `common/functions/act.py`), `apply_aggregation`'s per-option `equal`+`select` plus its `has_valid` guard and ids cast (≈8 ops, `common/functions/agg.py`), and `default.py`'s per-level `node_level == l` mask (3 ops).

## Forward algebra (recurrent)
Same window machinery, no levels: unroll `activate_time` ticks in Python.
Each tick: re-inject `input_transform`-free inputs into the input rows, compute every node from the previous tick's values (Jacobi/synchronous), then `values = select(node_exists & valid_count > 0, z, values)`.
`input_transform` is **not** applied by `RecurrentGenome` (legacy parity; HyperNEAT uses it as a CPPN).
`RecurrentGenome.sympy_func`/`visualize` are documented refusals: a cyclic network has no finite closed form and the layered layout assumes a depth-ordered DAG, so both raise `TensorNEATError` naming that reason and pointing at `network_dict()` / `DefaultGenome` (never a bare `NotImplementedError` stub).

## Gradient algebra
No control-flow (region) op exists in the forward, so reverse-mode works; but three evoxc rules shape the API:
- `evoxc.grad` requires a **single scalar output**, so the loss is scalarized as `sum(loss_fn(outputs))`; per-individual gradients are unaffected because the population axis is independent.
- Only float inputs are differentiable: `node_attrs` and `conn_attrs` are the differentiated arguments; keys and the schedule are non-differentiated int inputs (an int32 *differentiated* argument is a hard `TransformError`).
- `sort`/`argsort` have no AD rule, so the schedule derivation must **not** be inside the differentiated graph — this is why `grad` takes an already-derived `transformed` and returns its own graph.
Attribute gradients are masked at the graph input (`select(exists & ~isnan(a), a, 0)`), so padding rows gradient to exactly `0`.
`grad` does **not** differentiate the plain forward: evoxc 0.1.0 derives a `gather` VJP with *put* semantics, so the fan-out gathers of the window step under-accumulate and the gradient is wrong wherever a node feeds several edges.
Instead `grad` differentiates a mathematically identical **dense** pass built beside the plain one in `grad.py`: a per-individual `W (P, N, N)` weight matrix scattered from the conn-gene `weight` column (unique `(dst, src)` cells, `0` elsewhere), `agg = dot(W, xs)` per level, and the live-slot gate `live_count = dot(live_matrix, ~isnan(values))` — `dot`'s VJP is exact.
`grad.dense_forward` produces the same values as `BaseGenome.forward` (checked to 1e-6, measured exactly equal on float32), so the gradients are the gradients of the plain forward even though the ops differ: sum aggregation is `dot`, mean aggregation is the same `dot` divided by `maximum(live_count, 1)`.
`grad` still returns an `evoxc.Graph` for the Pipeline to cache/compile/run; the plan (`W` cells, live matrix, level masks) is host-computed once per graph inside `grad.dense_grad_plan`.

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
- File-size outcome: `base.py` (1375 lines) is one job — configuration, payload, schedule glue and the plain window forward — after its two side responsibilities moved out: NumPy-in/Python-out introspection went to `host.py` (`BaseGenomeHost`, inherited by `BaseGenome`), and the exact gradient path went to `grad.py` (`BaseGenome.grad` delegates to `grad.build_grad`; every public name unchanged). `grad.py` (498), `gene/base.py` (801), `utils.py` (640) and the mutation files (≤573) are under budget.

## Deviations from the pre-rewrite library (intentional)
- `activation == -1` means **identity** everywhere; the legacy `lax.switch(-1, ...)` clamped to the last option, so `BiasNode` split nodes silently used the wrong activation.
- Activation/aggregation options are evaluated for **all** configured options and selected with `select`, so a non-total option (division by zero, `log`/`sqrt` of a possibly negative value) can poison gradients even where it is not selected; single-option configurations emit no selection op at all.
- Random draws differ from the JAX build by construction (`evoxc.random` vs `jax.random`), so parity with `benchmarks/legacy_jax` is asserted on kernels for given genomes (forward, grad, distance), not on RNG-dependent operator output.

## Known Issues / Notes for Agents
**evoxc 0.1.0 `gather` VJP has put semantics** (`evoxc/transforms/rules.py`: `g_x = scatter(zeros_like(x), idx, ct)`), so any gather with duplicate indices silently keeps one contribution instead of accumulating them.
That defect hits both the window gather of `values` in the plain forward and `gather(conn_attrs, in_conn)`, so no gradient may be taken through the window formulation until evoxc ships a scatter-add; the defect is unfixed here on purpose.
`grad.dense_grad_plan` therefore never gathers attributes: it reads the edge list (`conn_attrs[:, :, weight]`, `conn_exists`, `in_src`, `dst`) and scatters into unique cells.
All scatter indices in `grad.dense_grad_plan` are collision-free across the population burst because the flattened cell index carries a per-individual stride (`p * (N*N + 1)`) plus one dump cell (`N*N`) for unused rows, which is sliced off after the scatter.
Only those float writes can collide-free-scatter; a plain `(P, N, N)` scatter would let different individuals write the same flattened cell.
`grad.dense_grad_plan` is host-side NumPy over `Transformed` only (schedule is static), so it costs one Python pass per compiled graph and nothing per generation.
`_node_step`, `_hit_conn_attrs` and `_gather_slots` remain the only plain-forward path and must keep their per-level op budget; the exact path in `grad.py` is used exclusively by `BaseGenome.grad`.
`grad` refuses at graph-build time what the dense formulation cannot represent: a non-`"feedforward"` genome, `aggregation_options` other than a single `AGG.sum`/`AGG.mean` (the `ConfigError` names the offending option), a conn gene without a `weight` column, duplicate `(dst, src)` pairs, and edges with no incoming-window slot (`ShapeBoundError` — a `max_in_degree` rebuild signal, never a silent drop); all five refusals live in `grad.dense_grad_plan`.
Attribute gradients are masked at the graph input (`select(exists & ~isnan(a), a, 0)`) and cleaned at the output (`select(isnan(g), 0, g)`), so padding rows, absent rows and non-weight attribute columns (option ids, historical markers) gradient to exactly `0`.
Finite-difference check of the current path (2-input → 3-hidden → 1-output XOR genome, `max_nodes=8`, `max_conns=12`, `DefaultNode`/`DefaultConn`, `key_span=64`, central differences at `h=1e-2`): the analytic gradient matches to max absolute difference ≈4.7e-5 on the node attributes and ≈5.3e-5 on the connection weight — float32 FD noise, identical before and after the `grad.py` split; padding rows gradient to exactly `0` and no gradient entry is NaN/inf.
Cost of the exact path on the numpy backend (`P=1000`, `max_nodes=50`, `max_conns=100`, 9 unrolled levels): ≈13 ms for the dense forward without AD and ≈82 ms per compiled `grad` run, against ≈0.9 s to trace+lower+compile the graph once per shape signature.

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
| `./base.py` | `BaseGenome`: configuration, `initialize`, `transform`, the shared window-reduce node step (`_hit_conn_attrs`/`_node_step`), the `grad` entry point that delegates to `./grad.py`, `_padding_safe_attrs`, `_output_row_mask`/`_is_output_row` (the output-row helper the forward spec names), abstract `sympy_func`/`visualize` hooks (1375 lines). |
| `./grad.py` | The exact gradient path and why it exists: `build_grad` (the `evoxc.grad` graph factory), `dense_grad_plan` (host statics), `dense_forward`, `dense_node_step`, `_DenseGradPlan`, `_sanitize_attrs`/`_graph_constant`/`_tensor_spec` — the dense-weight twin of the window forward that sidesteps evoxc 0.1.0's put-semantics `gather` VJP (498 lines). |
| `./host.py` | `BaseGenomeHost` mixin: host-only helpers (`network_dict`, `_get_node_dict`, `_get_conn_dict`, `repr`, `hash`, `_row_hashes`, `get_input_idx`, `get_output_idx`, `_individual`), `HASH_QUANT_SCALE`, `_row_slice` (348 lines). |
| `./default.py` | `DefaultGenome`: depth-unrolled level-synchronous forward. |
| `./recurrent.py` | `RecurrentGenome`: tick-unrolled Jacobi forward. |
| `./utils.py` | (640 lines) Batched genome-array editing helpers (`add_node`, `add_conn`, `delete_*`, `first_free_row`, `fetch_random_index`) and host-only display helpers. |

Out of node but required: `../common/tensorization.py` (`derive_schedule` → `Transformed`), `../common/functions/` (`ACT`/`AGG`, `apply_activation(z, act_ids, act_fns)`, `apply_aggregation(values, mask, agg_ids, agg_fns)` with `-1` = identity), `../common/tools.py` (`I_INF`, flat gather/scatter), `../common/state.py` (`State`), `../common/errors.py` (`TensorNEATError`), `../common/host_graph.py` (host-only topological walks), `../common/sympy_tools.py` (the 8-tuple codegen).
Consumers: `../algorithm/neat/` (`ask`/`transform`/`forward`/`tell`), `../algorithm/hyperneat/` (`RecurrentGenome` as CPPN), `../pipeline.py` (compile site, per-shape executable cache).
