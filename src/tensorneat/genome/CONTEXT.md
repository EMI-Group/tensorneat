# genome — gene payloads, genetic operators, and the network forward/grad algebra

## Intent
Turn a NEAT individual into fixed-shape `int32`/`float32` evoxc tensors, and evaluate a whole population of differently-shaped networks in one batch-first kernel.
This node owns exactly four things: (1) the **gene payload** (key columns in `int32`, attribute columns in `float32`, padded rows), (2) the **genetic operators** (mutation / crossover / distance) as batched kernels, (3) the **forward algebra** over a precomputed execution schedule plus the exact gradient path beside it (`grad.py`), and (4) host-only introspection (`network_dict`, `sympy_func`, `visualize`, `repr`, `hash`).
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
The shared window step is `_hit_conn_attrs` + `_node_step` + `_output_values`; `_hit_conn_attrs` returns a `_StepStatics` bundle (gathered conn attributes as `ConnOperands`, the block-shaped value-window gather index, the present mask, the rank-0 NaN fill and output-row mask, and the `NodeOperands` built on the first step) because node/conn attributes are constant during a forward pass, so every loop-invariant operand is derived once and a level pays only value-dependent ops.
- `initialize(state, key, pop_size) -> GenomeTensors` — `key` is a rank-0 `evoxc.random` key and randomness is drawn once per tensor with the batched output shape, so individual `i` gets an independent stream from one call.
- `transform(state, node_keys, conn_keys) -> Transformed` — schedule derivation, delegated to `../common/tensorization.py`; pure tensor ops, no host round-trip.
- `forward(state, transformed, node_keys, node_attrs, conn_keys, conn_attrs, inputs) -> (P, num_outputs)` — trace-time kernel.
- `grad(state, transformed, node_*, conn_*, inputs, loss_fn) -> evoxc.Graph` — **graph factory** (like `evoxc.grad`), not a trace-time kernel: it traces its own graph and returns it for the Pipeline to cache/compile/run, and its body is `grad.build_grad(self, ...)`.
- `execute_mutation(state, key, genome, new_node_keys (P,), new_conn_keys (P,3)) -> GenomeTensors`
- `execute_crossover(state, key, genome1, genome2) -> GenomeTensors`
- `execute_distance(state, genome1, genome2) -> (P,)` — pairwise, batched over pairs; the algorithm forms the pairs.
- Host-only (NumPy in, Python out, never traced): `network_dict`, `sympy_func`, `visualize`, `repr`, `hash`, `get_input_idx`, `get_output_idx` — the non-`sympy_func`/`visualize` ones live in `host.py::BaseGenomeHost`, so each takes **one** individual: `_individual` raises `TensorNEATError` telling the caller to slice first, e.g. `GenomeTensors(*(a[i:i+1] for a in payload))` (there is no `payload.of(i)`).
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

## Cost law (measured, numpy reference backend)
Two rules govern every kernel here; both were measured, not assumed.
1. **An ordinary traced op costs ≈80 µs at `P=1000` regardless of element count**, so op count is the budget for elementwise/gather/select/reshape work.
2. **An innermost-axis reduction over a `(P, N, K)` tensor is 10–20× an ordinary op and does scale with elements**: `np.max((1000,50,4) int32, axis=2)` = 1.00 ms and bool = 0.74 ms, versus 0.093 ms for the same data with the window on axis 1, versus 0.167 ms for an unrolled `maximum`/`logical_or` fold of the 4 slices; `np.sum` = 1.50 ms versus 1.06 ms for the slice-add fold (and 6.37 vs 3.49 ms at the tiled `4P=4000` batch).
Hence `window.any_slot` (`K` slices + `K-1` logical ops, `_WINDOW_FOLD_LIMIT = 8`) replaces `reduce_max(mask, -1)` wherever "any live slot" is asked — more traced ops, far less time.
The same law applies to `common/functions/agg.py`'s `reduce_sum`/`count` over the window axis: the remaining per-level cost in the forward.

## Forward algebra (feedforward)
`values (P, max_nodes) float32`, all NaN except the input rows, which hold `input_transform(inputs)`.
Step 0 (both genomes): `node_attrs = self._padding_safe_attrs(node_attrs, node_keys)` replaces the attrs of **absent** rows with `0` via `select`, so the option-index columns (`activation`, `aggregation`) never reach the int cast as NaN; a NaN *value* column of a live row is untouched, so real corruption still poisons.
Step 1: `hit_conn_attrs = self._hit_conn_attrs(transformed, conn_attrs)` once (attributes are constant during a pass); the value-window gather index it builds is **block-shaped** `(P, N, K)` so `evoxc.gather` splices it directly (no reshape back).
Step 2: one `evoxc.while_loop` over levels, `_depth_limit(transformed)` = `min(max_depth, max over existing rows of node_level, 1)` — `node_level` is `I_INF` on absent/unreachable/cyclic rows so it is masked and clamped first.
Loop body (`level` carried, starts at 1) is one `_node_step` with `update_mask = node_level == level` (the schedule's `I_INF` padding already excludes absent rows, so no `node_exists` term):
1. `xv = _gather_prepared(values, flat_index, present, nan)` — one flatten, one flat gather, one `select` that overwrites every statically dead slot with **NaN**.
2. `live = xv == xv` — one op; IEEE makes `x == x` False exactly on NaN, so liveness needs no `isnan` and no present-mask AND.
3. `msg = conn_gene.forward(statics.conn, xv)` — one `multiply` by the weight column; the gene needs no sanitisation because dead slots are neutral-filled by the aggregation before any reduce, and live slots pass through exactly.
4. `z = node_gene.forward(...)`: per-option aggregation (dead slots neutral-filled, dispatch over masks precomputed in `NodeOperands`), `combine` (`bias + response * agg`), then the activation — id `-1` (identity) on output rows, because the final activation belongs to `output_transform`.
5. `update = update_mask & any_slot(live)`; `values = select(update, z, values)` — a node with no live incoming edge keeps NaN, and its descendants lose their only live source: the legacy NaN poisoning, reproduced exactly.
Read out `gather(values, output_idx, axis=1) -> (P, num_outputs)`, then `output_transform`.
Cost is `O((depth+1) × max_nodes × K)` with only elementwise/reduce/gather ops: work scales with real depth and real in-degree, never `max_depth` and never `max_nodes²`.
**Graph shape (P=100, `max_depth=16`, `K=4`, `max_nodes=50`)**: 79 entry ops + a 31-op body + a 2-op condition; the body runs `depth+1` times — **4 / 2 / 3** trips on the populations after 8 generations at P=100/500/1000, where the per-level work dominates and the header is amortised.

| measured (build excluded, median of 7, single-threaded) | P=100 | P=500 | P=1000 |
|---|---|---|---|
| `transform` (schedule) | 6.4 ms | 12.4 ms | 28.1 ms |
| `forward(P)` | 2.0 ms | 3.0 ms | 7.9 ms |
| `forward(4P)` — the tiled XOR batch the pipeline evaluates | 4.5 ms | 9.9 ms | 26.8 ms |
| `Pipeline.step` end-to-end | 255 ms | 975 ms | 1849 ms |
The forward is no longer the genome node's bottleneck (`transform` is, at `P≥500`), and neither is the genome node in `Pipeline.step` — `algorithm`'s ask/tell dominates it (≈97 % at `P=1000`).
`transform` is 391 traced ops of which `../common/tensorization.py::node_levels` is 64 % (a 55-op `while_loop` body relaxing `level`/`reach`, 3 innermost-axis `reduce_max` per iteration, 4–5 iterations) — see "Outside this node".

## Forward algebra (recurrent)
Same window machinery, no levels: unroll `activate_time` ticks in **Python** (the recurrent path keeps its tick loop; the level `while_loop` is feedforward-only).
Each tick: re-inject `input_transform`-free inputs into the input rows, compute every node from the previous tick's values (Jacobi/synchronous), then `values = select(node_exists & valid_count > 0, z, values)`.
`input_transform` is **not** applied by `RecurrentGenome` (legacy parity; HyperNEAT uses it as a CPPN).
`RecurrentGenome.sympy_func`/`visualize` are documented refusals: a cyclic network has no finite closed form and the layered layout assumes a depth-ordered DAG, so both raise `TensorNEATError` naming that reason and pointing at `network_dict()` / `DefaultGenome` (never a bare `NotImplementedError` stub).

## Gradient algebra
`grad` never differentiates the plain forward, so the forward's `while_loop` region and the schedule's `argsort`/`sort` never meet reverse-mode. Three evoxc rules shape the API:
- `evoxc.grad` requires a **single scalar output**, so the loss is scalarized as `sum(loss_fn(outputs))`; per-individual gradients are unaffected because the population axis is independent.
- Only float inputs are differentiable: `node_attrs` and `conn_attrs` are the differentiated arguments; keys and the schedule are non-differentiated int inputs (an int32 *differentiated* argument is a hard `TransformError`).
- `sort`/`argsort` have no AD rule, so the schedule derivation must **not** be inside the differentiated graph — this is why `grad` takes an already-derived `transformed` and returns its own graph.
Attribute gradients are masked at the graph input (`select(exists & ~isnan(a), a, 0)`), so padding rows gradient to exactly `0`.
Why a second formulation exists: evoxc 0.1.0 derives a `gather` VJP with *put* semantics, so the fan-out gathers of the window step under-accumulate and the gradient is wrong wherever a node feeds several edges.
`grad` therefore differentiates a mathematically identical **dense** pass built beside the plain one in `grad.py`: a per-individual `W (P, N, N)` weight matrix scattered from the conn-gene `weight` column (unique `(dst, src)` cells, `0` elsewhere), `agg = dot(W, xs)` per level, and the live-slot gate `live_count = dot(live_matrix, ~isnan(values))` — `dot`'s VJP is exact.
`grad.dense_forward` produces the same values as `BaseGenome.forward` (checked to 1e-6, measured exactly equal on float32), so the gradients are the gradients of the plain forward even though the ops differ: sum aggregation is `dot`, mean aggregation is the same `dot` divided by `maximum(live_count, 1)`.
`grad` still returns an `evoxc.Graph` for the Pipeline to cache/compile/run; the plan (`W` cells, live matrix, level masks) is host-computed once per graph inside `grad.dense_grad_plan`.

## Static invariants (performance knobs, never silent)
- `max_in_degree` (`K`, default `max(initial in-degree, 4)`): enforced by `execute_mutation`, which refuses an added connection that would push a node past `K`; `initialize` rejects an `init_hidden_layers` layout that needs more.
- `max_depth` (`D`, default 16) is a **ceiling on the while-loop trip count**, not a cost multiplier: a network deeper than `D` leaves the unreachable nodes NaN (they cannot win) and raises `overflow` for host-side diagnostics.
- `activate_time` (default 10) is the recurrent tick count.
`Transformed.overflow` is the only host-visible diagnostic; nothing silently drops a connection or a node.

## Constraints
- No `jax`/`torch` import anywhere in this node; all tensor math is evoxc ops (`enp`, `evoxc.gather/scatter/select/slice/reduce_*/dot`, `evoxc.random`).
- No eager NumPy on the compute path; the host helpers (`network_dict`, `sympy_func`, `visualize`, `repr`, `hash`) convert inputs with `Tensor.numpy()` and never feed results back into a graph.
- Every kernel is already batched: no per-individual Python loop and no `vmap`/`vectorize` over these kernels.
- The **only** region op in this node is the feedforward forward's level `while_loop` (`default.py`); it may not be differentiated (it is not — `grad.py` owns the differentiable path), and anything added inside it pays the region-overhead too, so loop-invariant work belongs in `_StepStatics`/`_hit_conn_attrs`.
- Shapes are static at trace time: `pop_size`, `max_nodes`, `max_conns`, `K`, `D` are Python ints; a change forces a rebuild (the Pipeline caches per shape signature).
- Integer arithmetic: an `int32` tensor plus a Python literal promotes to `int64` in evoxc, so index math stays in int32 tensors, and every sentinel/level index is a **rank-0** constant (`window._iinf`, `enp.full((), ...)`) — a full-size `I_INF` materialisation is a large graph constant and an extra op per use.
- Errors: configuration violations raise `../common/errors.TensorNEATError`; tensor/backend errors from evoxc propagate unchanged — never caught and re-wrapped.
- Shared machinery (one window-reduce node step) lives in `base.py` and is parameterized by the update mask, never duplicated between `default.py` and `recurrent.py`.
- File sizes: `base.py` (1293) is one job — configuration, payload, schedule glue, the shared window step and the `grad` entry point; the per-pass window helpers (`_slot_index`, `_column_index`, `_column_mask`, `_gather_prepared`, `any_slot`, `_iinf`, `_row_offsets`) live in `window.py` (211), introspection in `host.py` (348), the exact gradient path in `grad.py` (498), the gene dispatch/mask helpers in `gene/base.py` (853).

## Deviations from the pre-rewrite library (intentional)
- `activation == -1` means **identity** everywhere; the legacy `lax.switch(-1, ...)` clamped to the last option, so `BiasNode` split nodes silently used the wrong activation.
- Activation/aggregation options are evaluated for **all** configured options and selected with `select`, over masks precomputed once per pass in `NodeOperands` (`gene/base.py::option_masks`/`select_cascade`); a non-total option (division by zero, `log/sqrt` of a possibly negative value) can therefore poison gradients even where it is not selected.
- Random draws differ from the JAX build by construction (`evoxc.random` vs `jax.random`), so parity with `benchmarks/legacy_jax` is asserted on kernels for given genomes (forward, grad, distance), not on RNG-dependent operator output.
- The forward's per-level `any_slot` fold changes the *order* of the `reduce_max`-equivalent for bool masks only (it is a pure "any", so the result is identical); any future innermost-axis fold of a **float** reduce must check the non-numpy backends' summation order first.

## Known Issues / Notes for Agents
**evoxc 0.1.0 `gather` VJP has put semantics** (`evoxc/transforms/rules.py`: `g_x = scatter(zeros_like(x), idx, ct)`), so any gather with duplicate indices silently keeps one contribution instead of accumulating them.
That defect hits both the window gather of `values` in the plain forward and `gather(conn_attrs, in_conn)`, so no gradient may be taken through the window formulation until evoxc ships a scatter-add; the defect is unfixed here on purpose.
`grad.dense_grad_plan` therefore never gathers attributes: it reads the edge list (`conn_attrs[:, :, weight]`, `conn_exists`, `in_src`, `dst`) and scatters into unique cells.
All scatter indices in `grad.dense_grad_plan` are collision-free across the population burst because the flattened cell index carries a per-individual stride (`p * (N*N + 1)`) plus one dump cell (`N*N`) for unused rows, which is sliced off after the scatter.
`grad.dense_grad_plan` is host-side NumPy over `Transformed` only (schedule is static), so it costs one Python pass per compiled graph and nothing per generation.
`_node_step`, `_hit_conn_attrs` and `_gather_prepared` remain the only plain-forward path and must keep their per-level op budget; the exact path in `grad.py` is used exclusively by `BaseGenome.grad`.
`grad` refuses at graph-build time what the dense formulation cannot represent: a non-`"feedforward"` genome, `aggregation_options` other than a single `AGG.sum`/`AGG.mean` (the `ConfigError` names the offending option), a conn gene without a `weight` column, duplicate `(dst, src)` pairs, and edges with no incoming-window slot (`ShapeBoundError` — a `max_in_degree` rebuild signal, never a silent drop); all five refusals live in `grad.dense_grad_plan`.
Attribute gradients are masked at the graph input (`select(exists & ~isnan(a), a, 0)`) and cleaned at the output (`select(isnan(g), 0, g)`), so padding rows, absent rows and non-weight attribute columns (option ids, historical markers) gradient to exactly `0`.
Finite-difference check of the current path (2-input → 3-hidden → 1-output XOR genome, `max_nodes=8`, `max_conns=12`, `DefaultNode`/`DefaultConn`, `key_span=64`, central differences at `h=1e-2`): the analytic gradient matches to max absolute difference ≈4.7e-5 on the node attributes and ≈5.3e-5 on the connection weight — float32 FD noise; padding rows gradient to exactly `0` and no gradient entry is NaN/inf.
`genome/utils.py:209 key_rows` still does `reduce_max(match, axes=-1)` over a `(P, E, N)` bool — an innermost-axis reduce by the cost law above, but it is per-mutation rather than per-level, and a slice fold there would need `max_nodes` slices, so it wants a different fix (e.g. a transposed layout) before anyone touches it.
Behavioural gate: `test/unit/genome` contains only `pytest.skip(...)` placeholders (true at HEAD), so a change to this node is verified by a **bit-exact forward/transform comparison on a fixed payload** (12 `Transformed` fields + `forward` output + the tiled-4× output + a 5-generation seed-42 `fitness_history`), not by the unit suite.

## Outside this node (measured, for the owner of `common/`)
`transform` is dominated by `../common/tensorization.py::node_levels` — 64 % of its 28 ms at `P=1000`, a `while_loop` whose body costs 3 innermost-axis `reduce_max` per iteration. Folding those the way `window.any_slot` does is the single largest remaining win; dropping the `reach` (cycle) relaxation for a feedforward genome measured −27 % of the whole transform (needs a static `check_cycles=False`), one sort instead of two for `rank`+`cc` −3.8 %, and fusing the four `slot_lookup`s −1.5 % (both bit-identical, verified).
`../common/functions/agg.py`'s `sum`/`mean`/`max`/`min`/`product` kernels and the `count` reduce are innermost-axis reduces over `(P, N, K)`: a `K`-slice fold measured −20 % of `forward(P)` (bit-identical on numpy, order-of-operations risk on other backends).
`../common/functions/act.py` still emits the sigmoid body plus one `select` per configured option per level.
Configuration lever outside this node: `max_nodes`/`max_conns` are the dominant cost multiplier because every kernel runs on the padded shapes — `50/100 → 12/24` measured `transform` 27.2 → 10.5 ms and `forward` 43.6 → 13.5 ms at `P=1000` (not bit-identical: payload shapes shift the mutation RNG draws).

## Test Strategy
`test/unit/genome/` — numpy backend: currently **placeholder modules whose bodies are `pytest.skip`**, so they pass without exercising anything; the real gate for this node is the bit-exact forward/transform comparison described under "Known Issues".
`test/parity/genome/` — kernels vs the frozen `benchmarks/legacy_jax` implementation on identical genomes: forward values (`allclose`) and NaN patterns, `grad` vs `jax.grad` on float attributes, and `execute_distance` (deterministic, so it must match exactly).
`test/integration/` — evolve XOR to target with the numpy backend and assert determinism for a fixed seed; `test/release_gate_no_jax.py` asserts `import tensorneat` pulls in no `jax`/`torch`.
`RecurrentGenome` and the `grad` dense path are not covered by the level-loop machinery, so a change to `_node_step`/`window.py` must re-check both (they share the window step).

## Routing Table
| Path | Responsibility |
|---|---|
| `./gene/` | `BaseGene` attr-table machinery, the option mask/dispatch helpers (`option_masks`, `select_cascade`) plus node/conn gene families. |
| `./gene/node/` | `BaseNode`, `DefaultNode`, `BiasNode`, `OriginNode`: key/attr layout, `NodeOperands` (per-pass option masks, shared rank-0 zero, `needs_has_valid`), `combine`, activation choice, identity/zero attrs. |
| `./gene/conn/` | `BaseConn`, `DefaultConn`, `OriginConn`: per-edge message (a bare `multiply` on pre-sanitised values), `ConnOperands`, optional `historical_marker`. |
| `./operations/` | Batched mutation / crossover / distance contracts and the in-degree invariant. |
| `./operations/mutation/` | Structural (add/delete node, add/delete conn) and value mutation, `DefaultMutation` and `RecurrentMutation`. |
| `./operations/crossover/` | Key-alignment crossover (`DefaultCrossover`). |
| `./operations/distance/` | Compatibility distance (disjoint/excess + attribute distance). |
| `./window.py` | Per-pass window machinery: `_slot_index` (block-shaped flat index), `_row_offsets`, `_column_index`/`_column_mask` (conn-attribute gather), `_gather_prepared` (flatten + flat gather + NaN fill), `any_slot` (the "any live slot" fold + `_WINDOW_FOLD_LIMIT`), `_iinf` (rank-0 sentinel), `_i32` (211 lines). |
| `./base.py` | `BaseGenome`: configuration, `initialize`, `transform`, `_hit_conn_attrs`/`_node_step`/`_StepStatics`, `_padding_safe_attrs`, `_output_row_mask`/`_is_output_row`/`_output_values`, the `grad` entry point that delegates to `./grad.py`, abstract `sympy_func`/`visualize` hooks (1293 lines). |
| `./grad.py` | The exact gradient path and why it exists: `build_grad` (the `evoxc.grad` graph factory), `dense_grad_plan` (host statics), `dense_forward`, `dense_node_step`, `_DenseGradPlan`, `_sanitize_attrs`/`_graph_constant`/`_tensor_spec` — the dense-weight twin of the window forward that sidesteps evoxc 0.1.0's put-semantics `gather` VJP (498 lines). |
| `./host.py` | `BaseGenomeHost` mixin: host-only helpers (`network_dict`, `_get_node_dict`, `_get_conn_dict`, `repr`, `hash`, `_row_hashes`, `get_input_idx`, `get_output_idx`, `_individual`), `HASH_QUANT_SCALE`, `_row_slice` (348 lines). |
| `./default.py` | `DefaultGenome`: depth-adaptive `while_loop` level-synchronous forward (`_depth_limit`), host-side `network_dict`/`sympy_func`/`visualize` extensions. |
| `./recurrent.py` | `RecurrentGenome`: tick-unrolled (Python) Jacobi forward. |
| `./utils.py` | (640 lines) Batched genome-array editing helpers (`add_node`, `add_conn`, `delete_*`, `first_free_row`, `fetch_random_index`) and host-only display helpers. |

Out of node but required: `../common/tensorization.py` (`derive_schedule` → `Transformed`), `../common/functions/` (`ACT`/`AGG`, `apply_activation(z, act_ids, act_fns)`, `apply_aggregation(values, mask, agg_ids, agg_fns)` with `-1` = identity), `../common/tools.py` (`I_INF`, flat gather/scatter), `../common/state.py` (`State`), `../common/errors.py` (`TensorNEATError`), `../common/host_graph.py` (host-only topological walks), `../common/sympy_tools.py` (the 8-tuple codegen).
Consumers: `../algorithm/neat/` (`ask`/`transform`/`forward`/`tell`), `../algorithm/hyperneat/` (`RecurrentGenome` as CPPN), `../pipeline.py` (compile site, per-shape executable cache).
