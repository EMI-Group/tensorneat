# tensorneat/common — host primitives, batched tensor tools, the tensorization (execution-schedule) engine, ACT/AGG registries, sympy codegen

## Intent
Framework-wide foundation for the evoxc rewrite: the immutable `State` container, the error hierarchy, the batched gather/scatter/mutation helpers every kernel is built from, the **execution schedule** that turns a ragged evolved topology into static-shape arrays, the activation/aggregation registries, and host-side sympy code generation.
Nothing here imports `genome`, `algorithm`, `problem`, or `pipeline` — this node is the bottom of the package dependency DAG.
The whole node is JAX-free: graph code uses `evoxc` ops and `evoxc.numpy as enp`; host-only code (sympy codegen, `State` persistence, dict/network export) may use plain Python, `numpy`, `sympy`.

## Dtype & sentinel contract (binding for genome / algorithm / problem)
The gene payload is **split by dtype**: `node_keys (P, max_nodes) int32`, `node_attrs (P, max_nodes, A) float32`, `conn_keys (P, max_conns, 2..3) int32` (`(in_key, out_key)` first, an optional historical marker third), `conn_attrs (P, max_conns, W) float32`.
Absent key rows are `I_INF`; absent attribute values are `NaN` — the pre-rewrite NaN padding semantics survive in the attribute arrays (`isnan(attrs[..., 0])` still means "row unused"), while key arrays use the integer sentinel.
Keys are `int32` in `[0, KEY_MAX)` with `KEY_MAX = 2**24`; a key outside that range, or two nodes with the same key, is a caller bug (see `ConfigError`).
One integer sentinel only: `I_INF = 2147483647` (`int32` max) = "no index / unmatched / unavailable"; it lives in integer arrays only, never mixed with `NaN` in the same array.
Rationale: evoxc has **no bitcast op** (`cast` is numeric only), so a mixed-dtype gene row is impossible and the payload is split; keys stay `int32` so absence is an exact comparison and `key_span` really is a key-magnitude bound.
Consequences: an absent row is `keys == I_INF` in the key arrays and `isnan` in the attribute arrays; int-valued genes (activation/aggregation option positions) live in the float attribute columns, stay integral-valued, and are cast with `cast(..., int32)` before use; in-graph gene identity uses the **key columns** with `equal`, never a hash of the row (see Known Gaps).
`tensorization.derive_schedule` accepts **either** spelling for its key arguments: `int32` with `I_INF` padding (canonical — what the payload and every kernel pass) or `float32` with `NaN` padding (the pre-rewrite/parity spelling `test/support/oracle.py` still uses).
That dispatch is a trace-time `if` on a static dtype, never a runtime branch: `present = keys != I_INF` for integer keys, `present = isfinite(keys) & (keys < KEY_MAX)` for float keys.

## Execution schedule contract (`Transformed`)
`tensorization.derive_schedule(...)` is one `@evoxc.defn` graph; its output is a host-visible pytree of arrays only (**no Python statics inside** — every static fact is implied by a shape or a dtype).

Runtime inputs (batch-first, population on axis 0); the five statics are **positional-or-keyword** (evoxc passes statics positionally into `build`, and `functools.partial(fn, **statics)` also works — no call site needs a `partial` dance):
- `node_keys (P, max_nodes)` `int32` (or `float32`) — node key column; `I_INF`/`NaN` marks an absent row.
- `conn_in (P, max_conns)`, `conn_out (P, max_conns)` — connection endpoint keys, same dtype as `node_keys`.
- `input_keys (I,)`, `output_keys (O,)` — keys of the input/output nodes, shared by every genome (no batch axis).
- statics `pop_size P`, `max_nodes N`, `max_conns C`, `max_in_degree K`, `key_span S` — the Pipeline's shape signature, and therefore the cache key of the compiled executable.

Fields, all batch-first, in `Transformed` (a `NamedTuple`):

| field | shape | dtype | meaning |
|---|---|---|---|
| `src` | `(P, C)` | int32 | source node slot of each connection row; `I_INF` = absent row or unmatched key |
| `dst` | `(P, C)` | int32 | destination node slot of each connection row; `I_INF` likewise |
| `edge_valid` | `(P, C)` | bool | real conn row AND both endpoints matched |
| `in_conn` | `(P, N, K)` | int32 | conn row index of the k-th incoming edge of node `i`; `I_INF` padding |
| `in_src` | `(P, N, K)` | int32 | source slot of that same edge; `I_INF` padding |
| `in_valid` | `(P, N, K)` | bool | that window slot holds a live edge (`in_conn != I_INF`) |
| `node_level` | `(P, N)` | int32 | longest-path depth from the input nodes (`input` nodes = 0); `I_INF` = absent node or a node on a cycle |
| `node_exists` | `(P, N)` | bool | the node row is present (`node_keys != I_INF` / not `NaN`) |
| `topo_key` | `(P, N)` | int32 | `node_level * N + slot`, a cheap total-order key for a topological sweep; `I_INF` when `node_level` is |
| `input_slot` | `(P, I)` | int32 | node slot holding input `j`; `I_INF` if unmatched |
| `output_slot` | `(P, O)` | int32 | node slot holding output `j`; `I_INF` if unmatched |
| `edge_overflow` | `(P,)` | int32 | number of valid edges dropped because their in-group rank reached `K` |

`level` and `overflow` exist as read-only properties (`level` → `node_level`, `overflow` → `edge_overflow > 0`) for the pre-rewrite host callers and the numpy oracle; new kernels use the stored field names.

Invariances the consumers may rely on:
- `in_conn`, `in_src` and `in_valid` mark the same positions; a column is padded iff it is past that node's in-degree, so columns `0..d-1` of a node are dense.
- For every `(p, c)` with `edge_valid[p, c]`, exactly one `(p, i, k)` has `in_conn[p, i, k] == c` and `in_src[p, i, k] == src[p, c]`; dropped edges (rank `>= K`) are the only exceptions and are counted in `edge_overflow`.
- A node with `in_valid[p, i, 0] == False` has no incoming edge (input nodes, or isolated nodes).
- `node_level[p, i]` is `I_INF` iff node `i` is absent or unreachable-from-inputs / on a cycle; `reduce_max(node_level)` bounds the execution loop.
- The schedule never reflects population member identity: `src/dst` are *slots within one genome's own arrays*, never innovation numbers.
- Executing level 0, then 1, ... is a valid topological order; within one level, nodes are independent.
- `edge_overflow > 0` means the schedule is lossy for that genome — host code (Pipeline/algorithm) must rebuild with a larger `max_in_degree` (or raise `ShapeBoundError`); kernels never silently drop edges.

Derivation recipe (validated against a numpy oracle, keep the shape of it):
1. dense key→slot table `(P, S+1) int32` (trailing column = write dump), one flat scatter over node keys.
2. `src`, `dst` = flat gather from that table with sanitized keys (`absent -> 0`, clamp to `[0, S-1]`), then `edge_valid = row present & table hit`, and `node_exists` from the node key column.
3. `rank` = global rank of each edge in the `(dst, conn_index)` order: stable `argsort` of `dst_sorted * C + index`, then one flat scatter of `arange(C)`.
4. `cc[i]` = number of edges with `dst <= i`: one stable `argsort` of the combined `(edge dsts * 2, node slots * 2 + 1)` array, one flat scatter for its inverse; `cc[i] = inv[C + i] - i`.
5. `start[i] = cc[i-1]` (0 for `i = 0`), `col = rank - start[dst]`.
6. `in_conn`/`in_src` = flat scatter of conn ids/src into `(P, N*K + 1)` at `dst*K + col`; edges with `col >= K` (or invalid) go to the dump slot; `edge_overflow = reduce_sum(col >= K & edge_valid)`; `in_valid = in_conn != I_INF`.
7. `node_level` = `while_loop` longest-path relaxation over the `(P, N, K)` table, inputs pinned to 0, capped at `2N` iterations (a snapshot at `N` distinguishes a live cycle from a long acyclic path); unresolved present nodes are cyclic.
8. `topo_key`, `input_slot`, `output_slot` are then free.

Cost: `O(P * (S + C log C + N*K*D))` with `D` = max depth, i.e. edge-proportional except for the `N*K` padding and the one-off `key_span` table; **no `cumsum`, no scatter-with-reduce, no `max_nodes**2` matrix**.
`transform` runs once per generation, so its cost is amortized over the whole generation.

## Measured cost
`derive_schedule` on the numpy reference backend, `N=50 C=200 K=8 S=64`, one build per shape then 15 timed runs (median / min, ms), single-threaded CPU (`x86_64`, Linux, CPython 3.14, numpy 2.5.4, evoxc 0.1.0 from `/root/evoxc`):

| population | P=100 | P=500 | P=1000 |
|---|---|---|---|
| random, cyclic (worst case: the `2N` relaxation cap is hit) | 79.8 / 77.5 | 286.7 / 279.4 | 535.4 / 525.7 |
| feedforward, acyclic (leaves the loop after `depth` iterations) | 21.0 / 20.7 | 75.6 / 74.8 | 140.0 / 138.2 |

The cyclic row is the worst case; a real generation's population is mostly acyclic, and the schedule is one shared traversal per population, not per network.
`incoming_table` and `node_levels` embed `(P, N*K+1)` int32 buffers as graph constants, so a build at `P=1000` crosses evoxc's 1 MiB large-constant threshold (1.6 MB, warned and ignored) — a build-time cost, not a per-run one.
Reproduction (the synthetic `tensorneat` package is needed only while `src/tensorneat/__init__.py` still imports the pre-merge `genome`/`pipeline` files):

```
PYTHONPATH=src:test .venv/bin/python - <<'PY'
import sys, types, time, statistics, warnings
sys.path.insert(0, "src")
pkg = types.ModuleType("tensorneat"); pkg.__path__ = ["src/tensorneat"]; sys.modules["tensorneat"] = pkg
import numpy as np, evoxc
from tensorneat.common import tensorization, tools
from support.runner import build_defn, tensor
N, C, K, S = 50, 200, 8, 64
for P in (100, 500, 1000):
    rng = np.random.default_rng(1000 + P)
    nk = np.full((P, N), tools.I_INF, np.int32); nk[:, :N - 4] = np.arange(N - 4, dtype=np.int32)
    ci = np.full((P, C), tools.I_INF, np.int32); co = np.full((P, C), tools.I_INF, np.int32)
    for p in range(P):
        present = nk[p][nk[p] != tools.I_INF]
        ci[p] = rng.choice(present, size=C); co[p] = rng.choice(present, size=C)
    args = (nk, ci, co, np.arange(2, dtype=np.int32),
            np.arange(N - 6, N - 4, dtype=np.int32), P, N, C, K, S)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        exe = build_defn(tensorization.derive_schedule, *args)
    run_args = tuple(tensor(a) if isinstance(a, np.ndarray) else a for a in args)
    evoxc.run(exe, *run_args)
    ts = []
    for _ in range(15):
        t0 = time.perf_counter(); evoxc.run(exe, *run_args); ts.append((time.perf_counter() - t0) * 1000)
    print(P, round(statistics.median(ts), 1), "ms")
PY
```

## Forward-side contract (consumed by `genome`)
Per level `l = 0 .. reduce_max(node_level)`:
- `mask = in_valid & ~isnan(gathered source value)` — the legacy `conn_exists & src_valid` mask.
- gather each conn gene attribute with `tools.masked_flat_gather` from the flattened `(P, C*attrs)` payload using `in_conn` (indices `(P, N, K)`).
- gather source values with `tools.masked_flat_gather(vals, in_src)` (`(P, N, K)`).
- `z = functions.agg.apply_aggregation(contribs, mask, agg_ids)` — masked reduce over the `K` axis, neutral-padded, dispatching on the per-node aggregation id.
- `new = functions.act.apply_activation(z, act_ids)`; `vals = select((node_level == l) & valid_count > 0, new, vals)` with `valid_count = reduce_sum(in_valid, -1)`.
The per-node `valid_count` guard matters: a node whose incoming edges are all invalid keeps `NaN` (legacy behaviour relied on `lax.cond(has_valid, ...)`).
Kernels must never index outside `[0, size)` — evoxc's numpy backend **raises** on out-of-range gather/scatter where JAX clamped, so clamping plus masking is mandatory, not an optimisation.

## API Surface
### `state.py`
`State(**kwargs)` — immutable host container over a dict; `register/update/remove` return new `State`s (raise on duplicate/missing keys); `__getattr__` raises `KeyError` for a missing key; `save(file_name)`/`load(file_name)` pickle it; `tree_flatten`/`tree_unflatten` + `evoxc.register_pytree_node` registration so it can cross a graph boundary.
`StatefulBaseClass` in `stateful_class.py` — `setup(state) -> State`, `show_config(registered_objects=None) -> dict` for pipeline logging (host-only).

### `errors.py`
`TensorNEATError(Exception)` root; subclasses `ConfigError`, `ShapeBoundError` (a static schedule bound was exceeded: `max_in_degree`, `key_span`, `max_conns`), `GraphError` (cycle where feedforward is required, missing input/output node), `BackendMismatchError`. Backend/shape errors from evoxc are **never** re-wrapped; only validation raises these.

### `tools.py`
Sentinels `I_INF`, `KEY_MAX`; `batch_offsets(p, r)`, `flat_gather(x, idx)`, `masked_flat_gather(x, idx, mask=None, fill=nan)`, `flat_scatter(x, idx, upd)`, `masked_flat_scatter(x, idx, upd, mask)` (dump-slot routing); `fetch_first`, `fetch_random`, `rank_elements`, `argmin_with_mask`; mutation kernels `mutate_float`, `mutate_int(key, val, options, replace_rate)` (no `random.choice` in evoxc: `gather(options, randint)`); `to_numpy(tensor)` — the host-side array accessor every logging/checkpoint path uses, re-exported by `common.__init__`; `hash_rows` — **host-only** row hash for `genome.hash()` (numpy `view(np.uint32)`), never called inside a graph.

### `tensorization.py`
`Transformed` `NamedTuple`; `derive_schedule` (`@evoxc.defn`); the internal steps as plain helper functions (`key_slot_table`, `slot_lookup`, `edge_group_rank`, `cum_in_degree`, `incoming_table`, `node_levels`) so they are individually testable; `validate_schedule(transformed)` host-side invariant checker used by tests.

### `functions/`
Exports exactly `ACT`, `AGG`, `FunctionManager`, `apply_activation(z, act_ids, act_fns=ACT_FUNCS)`, `apply_aggregation(values, mask, agg_ids, agg_fns=AGG_FUNCS)`: `ACT`/`AGG` are `FunctionManager`s over 12 activations and 6 aggregations in a fixed, parity-critical order; activations are elementwise `(z) -> z'`, aggregations reduce the last axis of a `(P, N, K)` contribution tensor with neutral padding and are `NaN`-aware through the `mask`. `ACT.name_of(func)`/`AGG.name_of(func)` are the reverse registry lookups. See `functions/CONTEXT.md`.

### `host_graph.py`
Host-only, pure Python (stdlib + `errors.GraphError`), never traced: `decode_graph(nodes, conns, node_attrs=1, conn_attrs=2)` → `(dict[int, tuple[float, ...]], set[tuple[int, int]])`; `topological_sort(nodes, conns)` → `(topo_order, topo_layers)` (Kahn, layers ascending, `GraphError` on a cycle); `find_useful_nodes(nodes, conns, output_idx)` → `set[int]`; `check_cycles(nodes, conns, from_idx, to_idx)` → `bool`; `node_depth(nodes, conns, input_idx)` → `dict[int, int]`, which must equal `Transformed.node_level` for a well-formed feedforward genome.
`common.__init__` re-exports `decode_graph`, `topological_sort` and `find_useful_nodes`.

### `sympy_tools.py`
Host-only (`sympy`, `numpy`, regex): `analysis_nodes_exprs`, `round_expr`, `replace_variable_names`, `to_latex_code`, `to_python_code` — the 8-tuple contract of `Genome.sympy_func` is unchanged from the pre-rewrite library.
It is deliberately not re-exported by `common.__init__` (its only consumer is `Genome.sympy_func`, which imports `..common.sympy_tools` directly), so the package surface stays what the contract lists.

## evoxc idioms and gotchas (validated on this box)
- `@evoxc.defn` + `evoxc.build(fn, *TensorSpec, backend=..., device=...)`; the numpy reference interpreter is the default and is enough for `test/unit`.
- `evoxc.numpy as enp` for in-graph constants (`enp.zeros/ones/full/arange`), `evoxc.constant(evoxc.tensor(ndarray))` for host data; dtypes come from `evoxc.int32/float32/...`, **not** from `enp`.
- `gather(x, idx, axis)` / `scatter(x, idx, upd, axis)` are **not** batched `take_along_axis`: the index tensor replaces the axis, so per-row indexing is impossible unless the source is flattened. Hence `cpu`-side kernels always go through `tools.flat_gather` / `tools.flat_scatter`, which flatten to 1-D (`flat = row * R + col`) and reshape back.
- `scatter` is set-only (no add/max modes) and `reductions` have no `where=`/`initial=`; masked reduction is done by padding with the neutral element (`0`, `-inf`, `+inf`, `1`) before `reduce_*`.
- `argsort` returns int64; its inverse permutation is a `flat_scatter` of `arange(n)`; `stable=True` is required for tie-breaking (group ranks).
- There is no `broadcast_to` on `evoxc` (use `enp.broadcast_to`), no `searchsorted`, no `choice`, no `bitcast`, no `where` (it is `evoxc.select(pred, on_true, on_false)`).
- Control flow (`evoxc.while_loop`/`cond`/`scan`) is traced into regions: loop bodies must keep shapes/dtypes fixed, conditions must be scalar tensors, and no `vmap` may be applied over them.
- `evoxc.slice(x, start, lengths)` takes per-axis tuples; `evoxc.reshape`/`transpose` are the only structural ops needed here.
- An `@evoxc.defn` object is not callable: call `.fn` for a plain function (the tests bind `.fn` with `functools.partial`), or `evoxc.build` the `Defn` itself.

## Constraints
- No `jax`/`torch`/`numpy.random` in graph code; randomness is `evoxc.random` stateless keys (`key(seed)`, `split`, `randint(key, shape, low, high)`, `uniform`, `normal`, `permutation`).
- Files ≤ ~1000 lines; shared math lives here, never duplicated in `genome`/`algorithm`.
- Nothing in this node imports from `genome`, `algorithm`, `problem`, `pipeline`, `benchmarks`, or `test`.
- Every public function carries a napoleon docstring stating shapes, dtypes, sentinels, and the invariances it relies on.
- Errors: validation raises `errors.TensorNEATError` subclasses on the host, at configuration/build time, before a graph is traced; in-graph overflow is reported as data (`edge_overflow`), never silenced.

## Known Gaps / Open Decisions
- The pre-rewrite helper names are gone and must not be re-introduced by consumers: `attach_with_inf` → `masked_flat_gather`; `hash_array` → `hash_rows` (host-only).
- Host-side graph walks (visualisation, `network_dict`, sympy codegen) use `host_graph.py`, which owns the graph algorithms for interpretability and export.
- The in-graph `(N, N)` adjacency-matrix topological sort of the pre-rewrite library has no replacement, because execution order comes from `node_level`/`topo_key`.
- **No bitcast op in evoxc** (v1): the payload is split into int32 keys and float32 attributes; `genome.hash()` is host-only bit-hashing; in-graph gene identity (crossover matching) compares key columns with `equal`. Revisit if evoxc grows `bitcast`/`reinterpret`.
- **`cumsum` is a StableHLO export deferral** in evoxc (works on the numpy interpreter only), which is why the schedule avoids it (see the `cc` trick in the derivation recipe). `argsort`/`sort`/`scatter`/`gather` have no such caveat.
- No scatter-with-reduce ⇒ per-node aggregation uses the padded `(P, N, K)` table; `K` is a static bound, so in-degree growth forces a rebuild (reported via `edge_overflow`).
- No per-row batched gather/scatter in the evoxc frontend (shape rules mimic `take`); the flat-index helpers are the only sanctioned workaround, and they require `P*R` to stay inside int32.
- `key_span S` is a static bound on key magnitude: keys must stay compact (hidden-node key compaction on mutation) or the table grows; `S` is part of the shape signature and rebuilds on overflow.

## Notes for Agents
- `test/unit/common/*` currently holds one `pytest.skip("not implemented: …, owner: common/…")` per test body: the acceptance suite is a prose specification, not runnable assertions, and it is owned outside this node. Verify changes here with an equivalence script against a numpy reference (see `test/support/oracle.py` for the schedule, `test/support/functions_ref.py` for ACT/AGG) kept out of the repo, or by running the test bodies once their owner fills them in.
- `import tensorneat` does not work yet: `src/tensorneat/genome/base.py` imports the removed name `build_schedule`, and `src/tensorneat/pipeline.py` is still the JAX implementation. Both are sibling nodes' files, so until they land `tensorneat.common` is imported through a synthetic package — `pkg = types.ModuleType("tensorneat"); pkg.__path__ = ["src/tensorneat"]; sys.modules["tensorneat"] = pkg` — exactly as the measured-cost snippet does; the submodules themselves are framework-free (no `jax`/`torch`/`brax`/`gymnax`/`mujoco` enters `sys.modules`).
- `test/support/oracle.py::schedule` is stale against the merged `Transformed`: it builds the node's pre-merge field set (`level=...`, no `node_exists`/`in_valid`), so a comparison script has to substitute a compatible output container for the oracle (the oracle's algorithm itself is unchanged and agrees field by field).
- `evoxc` in a bare checkout: `uv venv .venv && uv pip install --python .venv/bin/python numpy sympy networkx matplotlib pytest /root/evoxc`; `pytest` finds `src/` and `test/support` through `test/conftest.py`, so a run is `PYTHONPATH=src:.venv/… pytest test/unit/common`.
