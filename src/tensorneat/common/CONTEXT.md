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
Depth-qualified, on the numpy reference backend, shape signature `N=50 C=200 K=8 S=64`, 46 present node rows (4 padded), 2 inputs, 2 outputs; every figure is the **median of 3 campaigns × 15 timed `evoxc.run` calls** with the build cached (one `evoxc.build` per shape, timed once and reported separately), single-threaded (1 BLAS/OpenMP thread), INTEL Xeon Platinum 8568Y+, Linux 6.8 x86_64/glibc 2.39, CPython 3.14.8, numpy 2.5.4, evoxc 0.1.0 from `/root/evoxc`, ms:

| topology | live edges | depth | P=100 run (of which `node_levels`) | P=500 | P=1000 |
|---|---|---|---|---|---|
| feed-forward DAG, depth 5 (payload filled to `C`) | 200 | 5 | 9.0 (4.8) | 40.7 (17.0) | 74.2 (32.7) |
| feed-forward DAG, depth 18 (adjacent levels only → short fill) | 111 | 18 | 18.5 (14.8) | 60.8 (52.6) | 117.4 (98.3) |
| feed-forward DAG, depth 18, payload filled to `C` (forward jumps) | 200 | 18 | 18.5 (15.1) | 63.8 (51.7) | 124.0 (98.2) |
| random, cyclic (worst case: the relaxation hits the `2N` cap) | 200 | — | 79.5 (75.1) | 280.4 (268.6) | 547.9 (507.7) |

One-off `build` (cached per shape by the Pipeline, so not paid per generation): P=100 86–103 ms, P=500 270–330 ms, P=1000 595–630 ms.

Interpretation — the numbers behave like the `O(P · (C log C + N·K·D))` formula:

- **Linear in P.** Depth-5 run: 9.0 → 40.7 → 74.2 ms for P = 100 → 500 → 1000 (×4.6 then ×1.8), i.e. ≈ `1.7 ms + 0.073 ms` per member at depth 5; per-member cost falls from 0.090 to 0.074 ms only because the fixed per-run prologue/epilogue is amortised. `node_levels` alone is 4.8 → 17.0 → 32.7 ms (×3.5, ×1.9): linear.
- **Linear in depth, and depth is the dominant factor.** With the shapes *and* the live-edge count held fixed (row 1 vs row 3), P=1000 goes 74.2 → 124.0 ms while `node_levels` alone goes 32.7 → 98.2 ms (×3.00) — the relaxation's trip count is `D + 1` = 6 → 19 iterations (×3.17), i.e. ≈ 5.2–5.5 ms per iteration at P=1000 over the `(P, N, K)` window with its two relaxations. The remainder (total − `node_levels`, ≈19–42 ms at P=1000) is depth-independent: the two stable `argsort`s (`edge_group_rank`, `cum_in_degree`, `O(C log C)`) plus the `(P, N·K)` window zero-fill/scatter, i.e. the `C log C` and `N·K` terms. It is data- and layout-sensitive — numpy's stable sort is much faster on run-rich and `I_INF`-padded columns, so the 111-edge payload leaves a 19 ms remainder where the 200-edge depth-5 population leaves 41 ms — hence a range, not a constant.
- **Cyclic worst case.** 100 relaxation iterations instead of 6: `node_levels` is ×15.5 the depth-5 figure (trip-count ratio 16.7) and 93–96 % of the whole run. A population without a reachable cycle never pays this — in a real run most members are acyclic and the schedule is one shared traversal per population, not per network.
- **Lever, not implemented.** `node_levels` carries *two* max-plus relaxations in one loop body: `level` (the answer) and `reach` (only there to tell a live cycle from a long acyclic path). Skipping `reach` for a caller that can guarantee acyclicity would roughly halve the trip cost — the dominant term — but it needs a static "acyclic" flag in the Pipeline's shape signature (and therefore in the executable cache key). Not wired up.
- `incoming_table` (through `_window_table`) and `node_levels` embed `(P, N·K+1)`/`(P, N·K)` int32 buffers as graph constants: at P=1000 that is 9 constants above evoxc's 1 MiB threshold, ≈14.4 MB total (warned and ignored). P ≤ 500 stays below the threshold (800 KB). A build-time cost, not a per-run one.
Reproduction (build **once**, then time `evoxc.run` only — `run_defn` per iteration would time the build; the synthetic `tensorneat` package is needed only while `src/tensorneat/__init__.py` still imports the stale sibling modules):

```
PYTHONPATH=src:test .venv/bin/python - <<'PY'
import os, statistics, sys, time, types, warnings
for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(var, "1")
pkg = types.ModuleType("tensorneat"); pkg.__path__ = ["src/tensorneat"]; sys.modules["tensorneat"] = pkg
import numpy as np, evoxc
from tensorneat.common import tensorization, tools
from support.runner import build_defn, tensor

N, C, K, S = 50, 200, 8, 64
INPUTS, OUTPUTS, PRESENT = [0, 1], [N - 6, N - 5], list(range(N - 4))

def layered_dag(depth, rng, dense=False):
    hidden = [k for k in PRESENT if k not in INPUTS and k not in OUTPUTS]
    level = {k: 0 for k in INPUTS} | {k: depth for k in OUTPUTS}
    level |= {k: 1 + i % (depth - 1) for i, k in enumerate(hidden)}
    by_level = {}
    for k, lvl in level.items():
        by_level.setdefault(lvl, []).append(k)
    edges = set()
    for k in sorted(level, key=lambda key: level[key]):      # 2 preds on the previous level
        if level[k]:
            preds = by_level[level[k] - 1]
            for _ in range(2):
                edges.add((int(preds[rng.integers(len(preds))]), k))
    in_deg = {}
    for _u, v in edges:
        in_deg[v] = in_deg.get(v, 0) + 1
    if dense:                                                # any forward jump: all C rows live
        cand = [(u, v) for u in PRESENT for v in PRESENT
                if level[u] < level[v] and (u, v) not in edges]
        cand.sort(key=lambda uv: level[uv[1]] - level[uv[0]]); rng.shuffle(cand)
        cand.sort(key=lambda uv: level[uv[1]] - level[uv[0]])
    else:                                                    # adjacent levels only
        cand = [(u, v) for lvl in range(depth) for u in by_level.get(lvl, [])
                for v in by_level.get(lvl + 1, []) if (u, v) not in edges]
        rng.shuffle(cand)
    for u, v in cand:                                        # fill up to C edges, in-degree <= K
        if len(edges) >= C:
            break
        if in_deg.get(v, 0) < K:
            edges.add((u, v)); in_deg[v] = in_deg.get(v, 0) + 1
    return sorted(edges)

def cyclic(rng):
    edges = set()
    while len(edges) < C:
        edges.add((int(PRESENT[rng.integers(len(PRESENT))]), int(PRESENT[rng.integers(len(PRESENT))])))
    return sorted(edges)

def payload(pop_size, edges, rng):
    nk = np.full((pop_size, N), tools.I_INF, np.int32); nk[:, :N - 4] = np.arange(N - 4)
    ci = np.full((pop_size, C), tools.I_INF, np.int32); co = np.full((pop_size, C), tools.I_INF, np.int32)
    for p in range(pop_size):
        rows = list(edges); rng.shuffle(rows)                 # mutation order, not sorted
        ci[p, :len(rows)] = [u for u, _ in rows]; co[p, :len(rows)] = [v for _, v in rows]
    return (nk, ci, co, np.asarray(INPUTS, np.int32), np.asarray(OUTPUTS, np.int32),
            pop_size, N, C, K, S)

def timed(exe, run_args, repeat=15):
    evoxc.run(exe, *run_args)
    out = []
    for _ in range(repeat):
        t0 = time.perf_counter(); evoxc.run(exe, *run_args); out.append(time.perf_counter() - t0)
    return statistics.median(out) * 1e3

for label, depth, dense in (("dag_d5", 5, False), ("dag_d18", 18, False),
                            ("dag_d18d", 18, True), ("cyclic", 0, False)):
    for pop_size in (100, 500, 1000):
        rng = np.random.default_rng(2024 + pop_size + depth)
        edges = cyclic(rng) if depth == 0 else layered_dag(depth, rng, dense)
        args = payload(pop_size, edges, rng)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            exe = build_defn(tensorization.derive_schedule, *args)
        run_args = tuple(tensor(a) if isinstance(a, np.ndarray) else a for a in args)
        total = timed(exe, run_args)          # time the schedule before building anything else
        out = evoxc.run(exe, *run_args)       # its own inputs feed the node_levels measurement
        lvl = (np.asarray(out.in_conn.numpy()), np.asarray(out.in_src.numpy()),
               np.asarray(out.input_slot.numpy()), N)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lvl_exe = build_defn(tensorization.node_levels, *lvl)
        lvl = tuple(tensor(a) if isinstance(a, np.ndarray) else a for a in lvl)
        print(f"{label:8s} P={pop_size:5d} edges={len(edges):3d} "
              f"run={total:7.2f}ms node_levels={timed(lvl_exe, lvl):7.2f}ms")
PY
```

Keeping the measurement order fixed matters: building the `node_levels` executable between the two timings changes the heap layout and the same payload then measures ~25 % faster (the depth-5 cells above would read 8.8 / 29.9 / 58.1 ms). Time the full schedule first.

## Forward-side contract (consumed by `genome`)
Per level `l = 0 .. reduce_max(node_level)`:
- `mask = in_valid & ~isnan(gathered source value)` — the legacy `conn_exists & src_valid` mask.
- gather each conn gene attribute with `tools.masked_flat_gather` from the flattened `(P, C*attrs)` payload using `in_conn` (indices `(P, N, K)`).
- gather source values with `tools.masked_flat_gather(vals, in_src)` (`(P, N, K)`).
- `z = functions.agg.apply_aggregation(contribs, mask, agg_ids)` — masked reduce over the `K` axis, neutral-padded, dispatching on the per-node aggregation id.
- `new = functions.act.apply_activation(z, act_ids)`; `vals = select((node_level == l) & valid_count > 0, new, vals)` with `valid_count = reduce_sum(in_valid, -1)`.
The per-node `valid_count` guard matters: a node whose incoming edges are all invalid keeps `NaN` (legacy behaviour relied on `lax.cond(has_valid, ...)`).
Kernels must never index outside `[0, size)` — evoxc's numpy backend **raises** on out-of-range gather/scatter where JAX clamped, so clamping plus masking is mandatory, not an optimisation.
Cost of one level on the numpy interpreter at `(P, N, K) = (1000, 50, 8)` with the executable built once and `evoxc.run` timed (median of 15): 2.68 ms with one activation and one aggregation option, 17.8 ms with the canonical option lists.
About 54 % of the one-option figure is the mandatory `select(mask, z, 0.0)` pass and 34 % the `reduce_sum`, so the dispatch machinery around them is ~0.3 ms per level; the `(P, N, K)` aggregation passes, not the dispatch, are the term to attack next.

## API Surface
### `state.py`
`State(**kwargs)` — immutable host container over a dict; `register/update/remove` return new `State`s (raise on duplicate/missing keys); `__getattr__` raises `KeyError` for a missing key; `save(file_name)`/`load(file_name)` pickle it; `tree_flatten`/`tree_unflatten` + `evoxc.register_pytree_node` registration so it can cross a graph boundary.
`StatefulBaseClass` in `stateful_class.py` — `setup(state) -> State`, `show_config(registered_objects=None) -> dict` for pipeline logging (host-only).

### `errors.py`
`TensorNEATError(Exception)` root; subclasses `ConfigError`, `ShapeBoundError` (a static schedule bound was exceeded: `max_in_degree`, `key_span`, `max_conns`), `GraphError` (cycle where feedforward is required, missing input/output node), `BackendMismatchError`. Backend/shape errors from evoxc are **never** re-wrapped; only validation raises these.

### `tools.py`
Sentinels `I_INF`, `KEY_MAX`; `batch_offsets(p, r)`, `flat_gather(x, idx)`, `masked_flat_gather(x, idx, mask=None, fill=nan)`, `flat_scatter(x, idx, upd)`, `masked_flat_scatter(x, idx, upd, mask)` (dump-slot routing); `fetch_first`, `fetch_random`, `rank_elements`, `argmin_with_mask`; mutation kernels `mutate_float`, `mutate_int(key, val, options, replace_rate)` (no `random.choice` in evoxc: `gather(options, randint)`); `to_numpy(tensor)` — the host-side array accessor every logging/checkpoint path uses, re-exported by `common.__init__`; `hash_rows` — **host-only** row hash for `genome.hash()` (numpy `view(np.uint32)`), never called inside a graph.

### `tensorization.py`
`Transformed` `NamedTuple`; `derive_schedule` (`@evoxc.defn`); the internal steps as plain helper functions (`key_slot_table`, `slot_lookup`, `edge_group_rank`, `cum_in_degree`, `incoming_table`, `node_levels`) so they are individually testable, plus the two private bodies `_window_table` (the single copy of `incoming_table`'s masked-scatter → slice → reshape pass, used for `in_conn` and `in_src`) and `_host_array` (which converts through `tools.to_numpy`); `validate_schedule(transformed)` host-side invariant checker used by tests.

### `functions/`
Exports exactly `ACT`, `AGG`, `FunctionManager`, `apply_activation(z, act_ids, act_fns=ACT_FUNCS)`, `apply_aggregation(values, mask, agg_ids, agg_fns=AGG_FUNCS)`: `ACT`/`AGG` are `FunctionManager`s over 12 activations and 6 aggregations in a fixed, parity-critical order; activations are elementwise `(z) -> z'`, aggregations reduce the last axis of a `(P, N, K)` contribution tensor with neutral padding and are `NaN`-aware through the `mask`. `ACT.name_of(func)`/`AGG.name_of(func)` are the reverse registry lookups. See `functions/CONTEXT.md`.
Both dispatchers run once per unrolled depth level, so they emit only what the id column needs, and every choice is made at trace time.
No ids `cast` when the id column is already `int32` (the node gene's `prepare` casts once per forward pass); a `float32`/`int64` gene column is still cast inside the dispatcher.
No `equal`/`select` pair for an option whose kernel is the identity (the `identity` kernel carries the static `is_identity` flag), because selecting it would rewrite the accumulator with itself.
`apply_aggregation` shares the neutral-padded window (`select(mask, z, neutral)`) and the live-edge count between its options through one `_Window` memo, and emits the `has_valid` guard only when some option is not `zero_on_empty` (`sum`/`mean`/`maxabs` are; `product`/`max`/`min` and unannotated custom kernels are not).
Measured graph ops (`evoxc.trace`, numpy backend, `(P, N, K) = (100, 50, 8)`): `apply_activation` 10 with one option / 74 with the canonical 12, `apply_aggregation` 8 / 62 with the canonical 6, one agg∘act level 17 / 135, and a whole `DefaultGenome.forward` (16 unrolled levels, `max_nodes=50`, `max_conns=100`) 621 / 2509.
The `equal`+`select` of a one-option dispatch is not removable: the `-1` identity id on the output rows (plus the documented out-of-range → identity / → `0.0` fallbacks) needs a predicate, so a one-option level is the kernel call plus two selector ops.

### `host_graph.py`
Host-only, pure Python (stdlib + `errors.GraphError`), never traced: `decode_graph(nodes, conns, node_attrs=1, conn_attrs=2)` → `(dict[int, tuple[float, ...]], set[tuple[int, int]])`; `topological_sort(nodes, conns)` → `(topo_order, topo_layers)` (Kahn, layers ascending, `GraphError` on a cycle); `find_useful_nodes(nodes, conns, output_idx)` → `set[int]`; `check_cycles(nodes, conns, from_idx, to_idx)` → `bool`; `node_depth(nodes, conns, input_idx)` → `dict[int, int]`, which must equal `Transformed.node_level` for a well-formed feedforward genome.
`common.__init__` re-exports `decode_graph`, `topological_sort` and `find_useful_nodes`.

### `sympy_tools.py`
Host-only (`re` + `sympy`, no numpy, no evoxc): `analysis_nodes_exprs`, `round_expr`, `replace_variable_names`, `to_latex_code`, `to_python_code` — the 8-tuple contract of `Genome.sympy_func` is unchanged from the pre-rewrite library, and `to_python_code` emits the legacy line-by-line source (allocate `h`/`o`, then assignment lines) rather than a `def`.
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
- `tensorization.py` sits at ~1000 lines on purpose: it is one responsibility (derive the schedule) with a long derivation docstring per step, and splitting it would separate the recipe from the kernels that implement it.
- Inside this node one rule has one owner: `tools` owns the positive-static validator (`_positive_static`, used by `batch_offsets` and by every static in `tensorization`), the rank-0 scalar-constant builder (`_scalar_constant`) and the host-side array accessor (`to_numpy`, which `tensorization._host_array` converts through); those three are imported, never re-written. `host_graph` keeps its own `_I_INF` literal on purpose — `tools` pulls in `evoxc` and that module is deliberately backend-free.
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
- The ACT/AGG dispatch is at its op floor for the documented id contract, and the remaining cost is data movement: `sum`/`mean`/`maxabs` still need one `select(mask, z, 0.0)` over `(P, N, K)` per level, so the forward pass would gain only from a caller that guarantees dead message slots are already exactly `0.0` (today the genome only NaN-sanitises them, and it owns that contract).
- evoxc does not CSE equal literals, so each dispatch emits one `constant` per distinct literal (3 for a one-option level, 18 for the canonical aggregation list); sharing them across the unrolled levels needs a graph-level constant cache, which evoxc does not expose.
- `agg.maxabs` pays the guarded `tools.flat_gather`: `_flat_index_block` emits `less`/`greater_equal`/`logical_or`/`select` plus a `(P, 1)` `arange` constant (4 KB at P=1000) even though an `argmax` result is provably inside `[0, K)`; no trusted, guard-free gather variant exists. `maxabs` is not an option of the default genome, so the default XOR path never pays it (3.19 vs 2.34 ms per level at P=1000).
- The default genome's forward pass unrolls `max_depth = 16` level steps for every population, even a three-node network, so a per-level saving is multiplied by 16; unrolling on the real depth (or a level `while_loop`) is a genome-side change (`genome/` owns it).

## Notes for Agents
- `test/unit/common/*` currently holds one `pytest.skip("not implemented: …, owner: common/…")` per test body: the acceptance suite is a prose specification, not runnable assertions, and it is owned outside this node. Verify changes here with an equivalence script against a numpy reference (see `test/support/oracle.py` for the schedule, `test/support/functions_ref.py` for ACT/AGG) kept out of the repo, or by running the test bodies once their owner fills them in.
- `import tensorneat` does not work yet: `src/tensorneat/genome/base.py` imports the removed name `build_schedule`, and `src/tensorneat/pipeline.py` is still the JAX implementation. Both are sibling nodes' files, so until they land `tensorneat.common` is imported through a synthetic package — `pkg = types.ModuleType("tensorneat"); pkg.__path__ = ["src/tensorneat"]; sys.modules["tensorneat"] = pkg` — exactly as the measured-cost snippet does; the submodules themselves are framework-free (no `jax`/`torch`/`brax`/`gymnax`/`mujoco` enters `sys.modules`).
- `test/support/oracle.py::schedule` is stale against the merged `Transformed`: it builds the node's pre-merge field set (`level=...`, no `node_exists`/`in_valid`), so a comparison script has to substitute a compatible output container for the oracle (the oracle's algorithm itself is unchanged and agrees field by field).
- `evoxc` in a bare checkout: `uv venv .venv && uv pip install --python .venv/bin/python numpy sympy networkx matplotlib pytest /root/evoxc`; `pytest` finds `src/` and `test/support` through `test/conftest.py`, so a run is `PYTHONPATH=src:.venv/… pytest test/unit/common`.
