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
**Graph dtypes are signed only**: no `uint8/uint16/uint32/uint64` may appear in any operand, result or constant of a graph this package builds, because the iree backend's declared set is float16/32/64, int8/16/32/64 and bool — an unsigned value lowers nowhere but the numpy interpreter.
Use a same-width **signed** carrier instead, and where *unsigned order* genuinely matters (wrap-around comparisons, modular hashing) flip the sign explicitly rather than casting to an unsigned dtype.
`tools.hash_rows` is the one deliberate exception: it is numpy host code, never traced.
Consequence for range tests: `key ∈ [0, key_span)` is two signed comparisons in the key's own dtype (`greater_equal(keys, 0)` and `less(keys, key_span)`), which costs one op more than the `cast(keys, uint32) < key_span` trick that cannot be lowered to iree.

## Execution schedule contract (`Transformed`)
`tensorization.derive_schedule(...)` is one `@evoxc.defn` graph; its output is a host-visible pytree of arrays only (**no Python statics inside** — every static fact is implied by a shape or a dtype).

Runtime inputs (batch-first, population on axis 0); the five statics are **positional-or-keyword** (evoxc passes statics positionally into `build`, and `functools.partial(fn, **statics)` also works — no call site needs a `partial` dance):
- `node_keys (P, max_nodes)` `int32` (or `float32`) — node key column; `I_INF`/`NaN` marks an absent row.
- `conn_in (P, max_conns)`, `conn_out (P, max_conns)` — connection endpoint keys, same dtype as `node_keys`.
- `input_keys (I,)`, `output_keys (O,)` — keys of the input/output nodes, shared by every genome (no batch axis).
- statics `pop_size P`, `max_nodes N`, `max_conns C`, `max_in_degree K`, `key_span S` — the Pipeline's shape signature, and therefore the cache key of the compiled executable — plus a trailing `acyclic=False`, a caller assertion that the population contains no cycle (a build-time knob, not part of the shape signature).

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
`Transformed` is a plain `typing.NamedTuple` with **no `evoxc.register_pytree_node` call** (only `State` registers, in `state.py`): evoxc's namedtuple branch (`evoxc/core/tree.py::_flatten_into`, checked before plain tuple) flattens it to exactly those 12 leaves in field order via `_fields` and rebuilds with `spec.type(*children)`, so `evoxc.tree_map`/`tree_leaves`/`host_payload._slice_batch` never visit the `level`/`overflow` properties.
Tests pin the order and dtypes: `test/unit/common/test_tensorization.py::DOCUMENTED_FIELDS` / `DOCUMENTED_DTYPES` and `test_level_and_overflow_are_compatibility_aliases`.

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
3. `edge_order` — **one** stable `argsort` of the combined `(edge key 2*dst, node key 2*slot + 1)` array, built in the narrowest integer dtype that holds its `2N + 2` sentinel (`int8` up to N=62, `int16` up to N=16382, else `int32`); its inverse permutation `inv` yields **both** `cc[i] = inv[C + i] - i` (edges with `dst <= i`) and each edge's global rank `rank[c] = inv[c] - dst[c]` (the node entries below `dst[c]` precede it, and edges of one `dst` keep connection order).
4. `start[i] = cc[i-1]` (0 for `i = 0`), `col = rank - start[dst]`.
5. `in_conn`/`in_src` = flat scatter of conn ids/src into `(P, N*K + 1)` at `dst*K + col`; edges with `col >= K` (or invalid) go to the dump slot; `edge_overflow = reduce_sum(col >= K & edge_valid)`; `in_valid = in_conn != I_INF`.
6. `node_level` = `while_loop` longest-path relaxation over the `(P, N, K)` table, inputs pinned to 0, in the narrowest carrier dtype. The default runs the level-only relaxation (cap `N`) and then decides **in-graph** with `evoxc.cond` whether any valid edge violates `level[dst] > level[src]`: no violation ⇔ acyclic (a cycle always leaves one violation), and the level-only answer is then exact; a violation runs the two-relaxation machinery (`level` + `reach`, cap `2N`, snapshot at `N`) that reports every node on/after a live cycle as `I_INF`. `acyclic=True` returns the level-only answer directly.
7. `topo_key`, `input_slot`, `output_slot` are then free.

Cost: `O(P * (S + C log C + N*K*D))` with `D` = max depth, i.e. edge-proportional except for the `N*K` padding and the one-off `key_span` table; **no `cumsum`, no unsigned dtype, no scatter-with-reduce, no `max_nodes**2` matrix**.
`transform` runs once per generation, so its cost is amortized over the whole generation.

## Measured cost of `derive_schedule` (node-local)
Depth-qualified, on the numpy reference backend, shape signature `N=50 C=200 K=8 S=64`, 46 present node rows (4 padded), 2 inputs, 2 outputs, a layered feed-forward DAG population; every figure is the **median of 15 timed `evoxc.run` calls** with the build cached, single-threaded (1 BLAS/OpenMP thread), ms, and the snippet at the end of this section regenerates it.
Every figure in this section is a node-local schedule measurement; the per-generation pipeline cost is recorded in `benchmarks/PERF_NUMPY.md`, whose milliseconds are not restated here.
"before" is the two-sort / two-relaxation derivation used as the measurement baseline, "after" is the shipped one; the single-cell cells were measured interleaved in one session and the box drifts ±30 % between and within sessions, so the *ratio* is the load-bearing number:

| topology | P=100 | P=500 | P=1000 | speed-up at P=1000 |
|---|---|---|---|---|
| depth 5, 200 live edges | 10.7 → 5.9 | 34.4 → 16.6 | 83.2 → 30.5 | 2.7× |
| depth 8, 200 live edges | 12.7 → 5.6 | 40.7 → 13.3 | 97.1 → 33.1 | 2.9× |
| depth 18, 111 live edges | 20.7 → 8.5 | 66.9 → 20.2 | 144.5 → 33.8 | 4.3× |
| random cyclic, 200 live edges | 92.6 → 67.7 | 311.1 → 149.9 | 607.6 → 245.9 | 2.5× |

Single-cell interleaved protocol (three alternating rounds of 15 runs, P=1000, depth 8): before 103.5 / 102.4 / 105.2 ms, after 33.6 / 33.4 / 33.5 ms, `acyclic=True` 30.2 / 32.1 / 30.2 ms — ~3.1×, and the cycle-safe default is within ~10 % of the explicit acyclic assertion.
One-off `build` (cached per shape by the Pipeline, so not paid per generation): P=100 ~65 ms, P=500 ~140 ms, P=1000 ~280 ms; the default traces both `cond` branches (+2–6 % over `acyclic=True`).

Where the time goes (nesting-aware exclusive per-op profile, P=1000, depth 8, default path, 580 dispatches): `gather` 16 ops 7.3 ms · `select` 47 ops 6.6 ms · `scatter` 4 ops 4.0 ms (≈1 µs per 200 k-element scatter — the most expensive op per element) · `add` 32 ops 2.6 ms · `maximum` 72 ops 1.7 ms (the `K` fold: 8 slices per relaxation step) · `while` own overhead 1.5 ms · `reduce_max` 12 ops 1.2 ms · `argsort` 1 op 1.2 ms · `cast` 18 ops 1.1 ms · `reshape` 91 ops 0.9 ms; every other op kind under 1 ms.

Interpretation:

- **Per-op interpreter overhead is a first-class term.** 580 dispatches at ~10–15 µs of numpy-call/dispatch cost is 6–9 ms of the ~33 ms, so op *count* matters as much as op size: one removed `(P, N)` select is worth ~50 µs, and the tail is many small `(P, C)`/`(P, N)` ops rather than one big pass.
- **Depth does not dominate.** Rows depth 5 and depth 18 with the same live-edge count differ by ~10 % (30.5 vs 33.8 ms at P=1000) although the relaxation trip count is 6 vs 19: the loop is cheap (one `(P, N, K)` gather plus a `K` fold per iteration in a narrow carrier), and the depth-independent window build (the `in_conn`/`in_src` scatters), the one sort and the key tables hold the remaining cost.
- **The sort is not a term.** The derivation pays one `int16` stable `argsort` (1.2 ms): numpy's stable sort has a radix fast path for `int8`/`int16` and a slow general path for `int32`/`int64`, which is why the merged sort is keyed in the narrowest dtype (see the idioms list).
- **A cyclic population pays ~2.5× an acyclic one** (245.9 vs 33.1 ms at depth 8): the default's level-only campaign runs first and the `cond` fallback then runs the full two-relaxation machinery. Cycles are the RecurrentGenome's business; a feed-forward population never pays it.
- **Share of a generation.** The schedule is ~5 % of one `Pipeline.step` at pop 1000 (XOR, `DefaultGenome`, `NEAT`, build excluded, same session as the table above), so further schedule work cannot move the generation figure; `speciate`/`breed`/mutation (in `algorithm/`) own it and `benchmarks/PERF_NUMPY.md` owns the per-generation milliseconds.

Rejected levers (measured in this node on the numpy interpreter; do not re-derive):

- **Dense `(N, N)` adjacency / max-plus matrix power** for the level relaxation: `P·N·N` = 2.5 M element-ops per iteration against `P·N·K` = 400 k for the window relaxation, and max-plus has no matmul form (it would need `N` elementwise `maximum`s), so it is ≥ 6× worse at `K=8`.
- **One-hot / einsum counting for `rank` and `cc`**: the prefix count per edge is a `(P, C, C)` (40 M at P=1000) or `(P, C, N)` (10 M) tensor against a 250 k-element sort — the sort wins by an order of magnitude at these sizes.
- **Static Python unrolling of the relaxation** to a `max_depth` static: per-iteration work is unchanged, the loop overhead it removes is ~1.5 ms in total, and the unroll bound must be `≥ depth` for every member, so a `max_depth = 16` unroll runs more iterations than the 6–19 the convergent loop actually needs.
- **One gather over a stacked `(P, 2, N, K)` `level`/`reach` carrier** instead of two: same copied volume, one dispatch less — measured slower.
- **Fusing the two cycle-violation gathers** (`src`/`dst` interleaved) and **casting the violation mask to `int8`**: both inside noise (36.4 vs 35.8 ms), rejected.
- **Per-member `cond`** (one branch per population member): `vmap` over control flow is deferred in evoxc and the batch axis must be explicit — not expressible.
- **`reduce_max` over the `K` axis** instead of the pairwise `maximum` fold: 1.26 vs 0.30 ms per level at P=1000 (numpy's `np.max` over a trailing 8-axis is slower than seven `(P, N)` `maximum`s).
- **A `-2**30` sentinel instead of the `select(greater(best, -1), best + 1, -1)` guard**: an unsettled predecessor would settle its child at `0`; caught by the field-by-field diff.
- **A trusted, guard-free index block everywhere** (the escape test is what makes the backend *raise* instead of reading a neighbouring row): only offered as an opt-in `trusted=True` and only used where the index range is provably inside the table.

### The derivation tail is at its op floor (a 17-op trim is rejected; do not re-derive)
A trim of the derivation to 17 fewer executed ops (339 against 356 in the measured module: `_key_indices` drops the dtype-identical `cast` and uses a 0-d zero, `edge_valid` becomes `max(src, dst) < max_nodes`, `_slot_of` drops an already-redundant `present` redirect, `incoming_table` stops sanitising the cells `_window_pair` dumps) is **rejected**: it is value- and work-identical yet costs a third of a generation.
Measured in this node in one session (XOR, seed 42, fresh process, 8-10 timed generations, 2-3 interleaved rounds; the paired ratio is the result, never the absolute milliseconds): `Pipeline.step` 317-323 ms with the trim reverted vs 428-442 ms with it, i.e. **+34 % at pop 1000** and ~+5 % at pop 500, so the effect is non-linear in `P`.
It is not a data or layout change: the `Transformed` fields are bit-identical with the same dtypes, shapes and contiguity; per-step runtime op counts differ by exactly the 17 removed ops; operand/result dtype signatures are identical; peak RSS is the same (~805 MB); `derive_schedule` alone is *faster* (10.4 vs 10.9 ms).
It is not attributable to one hunk: reverting any single one of the four alone still measures 431-448 ms; only restoring the whole derivation returns 317-323 ms.
That leaves a whole-graph execution effect inside the numpy interpreter (the step graph's op/constant mix, not any value this module produces), which no local edit here can control.
**Reasoning rule: executed-op count is not a proxy for time at this level.** A trim that is value-identical and measurably faster on `transform` in isolation (6.66 -> 6.47 ms) can still cost 34 % of a generation once inlined into the step graph; measure `Pipeline.step` end-to-end at pop 1000 in fresh interleaved processes before landing a "free" removal.

### `node_levels` is at its op floor (measured in this node; do not re-derive)
Where the time goes at the real window: on a real feed-forward XOR population (seed 42, generation 7, `DefaultGenome(num_inputs=2, num_outputs=1, max_nodes=50, max_conns=100)`, `NEAT(pop_size=500, species_size=20)`, so `(P, N, K) = (500, 50, 4)` and depth ≈ 4), `node_levels(acyclic=True)` is 2.79–3.15 ms of a 6.9–7.2 ms `derive_schedule` (the box drifts; the ratio is the result, ~40–45 %).
Its fast path is ~16 executed ops per loop trip over `depth + 1` trips, i.e. ~35–40 µs per op at this size, so the loop is bound by *executed op count* and every lever below was measured as neutral or worse (base = 100 %, interleaved rounds of 15 `evoxc.run`, build excluded, medians of 5 alternating rounds).
Value identity of every candidate except the two-sub-step one was checked field by field against the unmodified `node_levels` on the real feed-forward population **and** a synthetic cyclic one, with `acyclic` both `True` and `False`.

- **`reduce_max` over the `K` axis instead of the pairwise fold**: 4.81 vs 3.15 ms (1.52× worse) — the K=8 / P=1000 verdict above holds at the narrower K=4 window, so the fold stays.
- **Dropping the per-trip `(P, N)` dump-column `concatenate`** (gather from `values` with `min(in_src, N - 1)` and one precomputed `(P, N, K)` window mask instead of a `-2` dump cell): 3.26 vs 2.79 ms (1.17× worse) — one `(P, N, K)` `select` costs ~0.5 ms more than the `(P, N)` concat it replaces, so the dump cell stays.
- **Two relaxation sub-steps per loop trip** (halves the trip count, keeps the pairwise fold): 3.67 vs 3.04 ms (1.21× worse); it also raises the fast-path level of a *cyclic* payload handed to an `acyclic=True` genome to `2N` instead of `N`, so it is not value-identical there either.
- **Shrinking the loop state from three leaves to two** (jump `step` to `cap` inside the body instead of carrying the `changed` leaf and folding it into the condition): 1.00× — the per-trip state copy is not a term.
- **Folding `pinned` out of the fast path's exit blank** (`blank = level < 0` instead of `~pinned & (level < 0)`): provably identical, because a pinned node is `0` in every trip of both campaigns so `level < 0` already excludes it, and it removes two `(P, N)` ops per campaign — but it measures 0.98–1.00×, under the ±2 % noise floor, so it was not landed (take it as a free cleanup only alongside a change that pays for its own re-measurement).
- **Static unrolling, dense `(N, N)` max-plus, one-hot prefix counting, per-member `cond`**: still rejected for the reasons in the list above.
A variant is cheapest tested without touching the file: `exec` `textwrap.dedent(inspect.getsource(tensorization.node_levels))` with one textual substitution into `dict(vars(tensorization))`, then `build_defn(variant, in_conn, in_src, input_slot, max_nodes, acyclic, (src, dst, edge_valid))` (arrays from a real `Transformed`, `edges` as the tuple) and compare against the unmodified function on the same arrays — both built in one process, one `evoxc.run` per sample, variants interleaved.
The ceiling here is small by construction: `transform` is ~3 % of one `Pipeline.step` at pop 500 (XOR, seed 42, measured in this node), so even a 2× `node_levels` would move a generation by ~1 %; the per-generation milliseconds themselves are `benchmarks/PERF_NUMPY.md`'s.

Reproduction (build **once**, then time `evoxc.run` only — `run_defn` per iteration would time the build; the synthetic `tensorneat` package lets the snippet run from `src/` without installing the package):

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

for label, depth in (("dag_d5", 5), ("dag_d8", 8), ("dag_d18", 18), ("cyclic", 0)):
    for pop_size in (100, 500, 1000):
        rng = np.random.default_rng(2024 + pop_size + depth)
        edges = cyclic(rng) if depth == 0 else layered_dag(depth, rng)
        base = payload(pop_size, edges, rng)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            exe = build_defn(tensorization.derive_schedule, *base)          # cycle-safe default
            acy = build_defn(tensorization.derive_schedule, *base, True)    # acyclic=True
        run_args = tuple(tensor(a) if isinstance(a, np.ndarray) else a for a in base)
        print(f"{label:8s} P={pop_size:5d} edges={len(edges):3d} "
              f"run={timed(exe, run_args):7.2f}ms  "
              f"acyclic={timed(acy, run_args + (True,)):7.2f}ms", flush=True)
PY
```

Measure a cell you intend to compare in a fresh process: building several executables in one process changes the heap layout and later cells then read up to 25 % faster.
`evoxc.build` takes a static either positionally (as above) or through `functools.partial(derive_schedule.fn, ...)`; the `@evoxc.defn` object itself is not callable.
The nesting-aware per-op profile of the same graph is a one-file monkeypatch of `evoxc.backends.numpy.kernels.dispatch` (time each kernel, subtract the nested time of the ops it runs, report exclusive time per op name); re-derive it whenever a change has to be attributed to an op.

## Forward-side contract (consumed by `genome`)
Per level `l = 0 .. reduce_max(node_level)`:
- `mask = in_valid & ~isnan(gathered source value)` — the legacy `conn_exists & src_valid` mask.
- gather each conn gene attribute with `tools.masked_flat_gather` from the flattened `(P, C*attrs)` payload using `in_conn` (indices `(P, N, K)`).
- gather source values with `tools.masked_flat_gather(vals, in_src)` (`(P, N, K)`).
- `z = functions.agg.apply_aggregation(contribs, mask, agg_ids)` — masked reduction over the `K` axis as a static `K`-plane fold (not a `reduce_*`; `functions/CONTEXT.md` owns the recipe and its `K <= 128` bound), neutral-padded, dispatching on the per-node aggregation id.
- `new = functions.act.apply_activation(z, act_ids)`; `vals = select((node_level == l) & valid_count > 0, new, vals)` with `valid_count = reduce_sum(in_valid, -1)`.
The per-node `valid_count` guard matters: a node whose incoming edges are all invalid keeps `NaN` (legacy behaviour relied on `lax.cond(has_valid, ...)`).
Kernels must never index outside `[0, size)` — evoxc's numpy backend **raises** on out-of-range gather/scatter where JAX clamped, so clamping plus masking is mandatory, not an optimisation.
Cost of one level, measured in this node on the numpy interpreter at `(P, N, K) = (1000, 50, 4)`, one aggregation and one activation option, executable built once and `evoxc.run` timed (median of 15): ~1.1 ms; 16.6 ms with the canonical six-aggregation / twelve-activation option lists, which is the per-option `select` cascade. `functions/CONTEXT.md` owns the per-option numbers and their per-op split.
The mandatory `select(mask, z, 0.0)` on `(P, N, K)` is the single most expensive op of a level now that the `K` axis is folded, and it cannot become a masked multiply (a dead slot may hold `NaN` or `±inf`, whose product with `0` is `NaN`), so the aggregation passes — not the dispatch — remain the term to attack.

## API Surface
### `state.py`
`State(**kwargs)` — immutable host container over a dict; `register/update/remove` return new `State`s (raise on duplicate/missing keys); `__getattr__` raises `KeyError` for a missing key; `save(file_name)`/`load(file_name)` pickle it; `tree_flatten`/`tree_unflatten` + `evoxc.register_pytree_node` registration so it can cross a graph boundary.
`StatefulBaseClass` in `stateful_class.py` — `setup(state) -> State`, `show_config(registered_objects=None) -> dict` for pipeline logging (host-only).

### `errors.py`
`TensorNEATError(Exception)` root; subclasses `ConfigError`, `ShapeBoundError` (a static schedule bound was exceeded: `max_in_degree`, `key_span`, `max_conns`), `GraphError` (cycle where feedforward is required, missing input/output node). Backend/shape errors from evoxc are **never** re-wrapped; only validation raises these.

### `tools.py`
Sentinels `I_INF`, `KEY_MAX`; `batch_offsets(p, r)`, `prepare_flat_index(x, idx, present=None, fallback=0)` (the flat `(P, n)` index block of a batched gather/scatter, reusable across calls), `flat_gather_prepared(x, flat_idx, shape)`, `flat_scatter_prepared(x, flat_idx, updates)`, `flat_gather(x, idx)`, `masked_flat_gather(x, idx, mask=None, fill=nan)`, `flat_scatter(x, idx, upd)`, `masked_flat_scatter(x, idx, upd, mask)` (dump-slot routing); `fetch_first`, `fetch_random`, `rank_elements`, `argmin_with_mask`; mutation kernels `mutate_float`, `mutate_int(key, val, options, replace_rate)` — standalone public kernels, not the genome path (attribute mutation is `genome/gene/blocks.py::mutate_attrs`), and no `random.choice` in evoxc: `gather(options, randint)`; `to_numpy(tensor)` — the host-side array accessor every logging/checkpoint path uses, re-exported by `common.__init__`: it returns a `numpy.ndarray` unchanged, else calls a callable `.numpy()`, else falls back to `np.asarray`, and `evoxc.Tensor` deliberately has **no** `__array__` (so `np.asarray` would leave it untouched); `hash_rows` — **host-only** numpy xor-mix over a row's bytes (`view(np.uint32)`), the host twin of `genome/gene/base.py::hash_int32`; `genome.hash()` itself is in-graph (`enp.zeros(..., uint32)` + `bitwise_xor`) and never calls it, so `hash_rows` has **no production caller** (tests + `common.__all__` only).
Every flat-index helper takes `trusted=False`: the default emits the escape guard (`select` + `less`/`greater_equal`/`logical_or` + redirect select) that makes the backend **raise** on an out-of-range index instead of silently reading a neighbouring row, and `trusted=True` skips it for a caller that has *proved* every index lies inside `[0, R)` (the schedule uses it for `key_slot_table`, `slot_lookup`, the `(P, N*K)` window scatters and both inverse-permutation scatters). Never pass `trusted=True` on data you have not bounded: the raise is the contract that replaces silent corruption.

### `tensorization.py`
`Transformed` `NamedTuple`; `derive_schedule(node_keys, conn_in, conn_out, input_keys, output_keys, pop_size, max_nodes, max_conns, max_in_degree, key_span, acyclic=False)` (`@evoxc.defn`, statics positional-or-keyword with `acyclic` trailing); `edge_order(dst, edge_valid, max_nodes, max_conns) -> (rank, cc)` — the one merged sort `derive_schedule` uses; the internal steps as plain helper functions (`key_slot_table`, `slot_lookup`, `edge_order`, `incoming_table`, `node_levels`) so they are individually testable and a caller pays for what it asks for; `node_levels(in_conn, in_src, input_slot, max_nodes, acyclic=False, edges=None)` where `edges` is the `(src, dst, edge_valid)` triple the cycle test needs (`ConfigError` when `acyclic=False` and it is missing); the two private bodies `_window_pair` (the single merged masked scatter that writes `in_conn` and `in_src` into a doubled `(P, 2*(N*K+1))` buffer — one index block, one scatter) and `_host_array` (which converts through `tools.to_numpy`); `validate_schedule(transformed)` host-side invariant checker used by tests.
The narrow integer carriers are a contract detail, not a knob: the level relaxation and both sort keys are built in the smallest of `int8`/`int16`/`int32` that provably holds every reachable value *and* its "after everything" sentinel (a real value inside the dtype, never `I_INF`), with the thresholds documented next to each builder; results are cast back to `int32` on the way out.

### `functions/`
Exports exactly `ACT`, `AGG`, `FunctionManager`, `apply_activation(z, act_ids, act_fns=ACT_FUNCS)`, `apply_aggregation(values, mask, agg_ids, agg_fns=AGG_FUNCS)`: `ACT`/`AGG` are `FunctionManager`s over 12 activations and 6 aggregations in a fixed, parity-critical order; activations are elementwise `(z) -> z'`, aggregations reduce the last axis of a `(P, N, K)` contribution tensor with neutral padding and are `NaN`-aware through the `mask`. `ACT.name_of(func)`/`AGG.name_of(func)` are the reverse registry lookups. See `functions/CONTEXT.md`.
Both dispatchers run once per unrolled depth level, so they emit only what the id column needs, and every choice is made at trace time.
No ids `cast` when the id column is already `int32` (the node gene's `prepare` casts once per forward pass); a `float32`/`int64` gene column is still cast inside the dispatcher.
No `equal`/`select` pair for an option whose kernel is the identity (the `identity` kernel carries the static `is_identity` flag), because selecting it would rewrite the accumulator with itself.
`apply_aggregation` shares the neutral-padded window (`select(mask, z, neutral)`) and the live-edge count between its options through one `_Window` memo, and emits the `has_valid` guard only when some option is not `zero_on_empty` (`sum`/`mean`/`maxabs` are; `product`/`max`/`min` and unannotated custom kernels are not).
Per-dispatch executed-op counts live in `functions/CONTEXT.md`'s table; do not copy them here. The fold means a level costs `K` slices + `K - 1` adds beyond the kernels themselves.
`_Window` additionally memoises the total (the `K`-plane fold of `padded(0.0)`, shared by `sum` and `mean`, one fold instead of two) and one 0-d `float32` `constant` per distinct literal (evoxc does not CSE duplicate constants; op-count-only, ~5 µs per op).
The `equal`+`select` of a one-option dispatch is not removable: the `-1` identity id on the output rows (plus the documented out-of-range → identity / → `0.0` fallbacks) needs a predicate, so a one-option level is the kernel call plus two selector ops.

### `host_graph.py`
Host-only, pure Python (stdlib + `errors.GraphError`), never traced: `decode_graph(nodes, conns, node_attrs=1, conn_attrs=2)` → `(dict[int, tuple[float, ...]], set[tuple[int, int]])`; `topological_sort(nodes, conns)` → `(topo_order, topo_layers)` (Kahn, layers ascending, `GraphError` on a cycle); `find_useful_nodes(nodes, conns, output_idx)` → `set[int]`; `check_cycles(nodes, conns, from_idx, to_idx)` → `bool`; `node_depth(nodes, conns, input_idx)` → `dict[int, int]`, which must equal `Transformed.node_level` for a well-formed feedforward genome.
`common.__init__` re-exports `decode_graph`, `topological_sort` and `find_useful_nodes`.

### `sympy_tools.py`
Host-only (`re` + `sympy`, no numpy, no evoxc): `analysis_nodes_exprs`, `round_expr`, `replace_variable_names`, `to_latex_code`, `to_python_code` — the 8-tuple contract of `Genome.sympy_func` is unchanged from the pre-rewrite library, and `to_python_code` emits the legacy line-by-line source (allocate `h`/`o`, then assignment lines) rather than a `def`.
It is deliberately not re-exported by `common.__init__` (its only consumer is `Genome.sympy_func`, which imports `..common.sympy_tools` directly), so the package surface stays what the contract lists.
`to_latex_code` and `to_python_code` take the 8-tuple **unpacked** (`to_latex_code(*genome.sympy_func(state, network))`), i.e. eight positional arguments `(symbols, args_symbols, input_symbols, nodes_exprs, output_exprs, forward_func, topo_order, useful_nodes)`; `symbols` and `topo_order` are keyed by the genome's node **keys**, while `nodes_exprs`/`output_exprs` are keyed by **sympy Symbols**, and `forward_func` is accepted for signature compatibility only (never called).
`analysis_nodes_exprs` counts `input_cnt` over keys *starting with* `i`, and `genome.sympy_func` names only the raw input symbol `i<k>` that way (`norm<k>` is matched by the trailing `norm` branch), so `input_cnt == I` and both renderers' `topo_order[input_cnt:]` slice is correct for a standard genome whose inputs sort first in the topological order.
`to_latex_code`/`to_python_code` ignore `input_symbols`, `output_exprs` and `forward_func` (only the latter is documented as accepted-for-compatibility): the emitted lines are per-node assignments, so the closed-form outputs are re-derived from `nodes_exprs`.
Both renderers are legacy-verbatim (`benchmarks/legacy_jax/tensorneat_legacy/common/sympy_tools.py`) and have no unit test of their own beyond `test/unit/common/test_functions.py::test_to_python_code_emits_runnable_source_for_a_hand_built_expression`, whose hand-built `nodes_exprs` carries one symbol per input and only `exp`.
That test does not pin the general runnability the docstring claims: the emitted source needs numpy's names injected (`np`, `exp`, `log`, `sin`) and is *not* executable for any option that sympy renders as `Max(...)`/`Min(...)`/`Abs(...)`/`Piecewise(...)` (`relu`, `lelu`, `inv` and the `max`/`min`/`maxabs` aggregations).

## evoxc idioms and gotchas (validated on this box)
- `@evoxc.defn` + `evoxc.build(fn, *TensorSpec, backend=..., device=...)`; the numpy reference interpreter is the default and is enough for `test/unit`.
- `evoxc.numpy as enp` for in-graph constants (`enp.zeros/ones/full/arange`), `evoxc.constant(evoxc.tensor(ndarray))` for host data; dtypes come from `evoxc.int32/float32/...`, **not** from `enp`.
- `gather(x, idx, axis)` / `scatter(x, idx, upd, axis)` are **not** batched `take_along_axis`: the index tensor replaces the axis, so per-row indexing is impossible unless the source is flattened. Hence `cpu`-side kernels always go through `tools.flat_gather` / `tools.flat_scatter`, which flatten to 1-D (`flat = row * R + col`) and reshape back.
- `scatter` is set-only (no add/max modes) and `reductions` have no `where=`/`initial=`; masked reduction is done by padding with the neutral element (`0`, `-inf`, `+inf`, `1`) before `reduce_*`.
- `argsort` returns int64; its inverse permutation is a `flat_scatter` of `arange(n)`; `stable=True` is required for tie-breaking (group ranks).
- **`argsort(kind="stable")` has a radix fast path for `int8`/`int16` and a slow general path for `int32`/`int64`**: `(1000, 250)` costs 7.4 ms as `int32` but 1.0 ms as `int16` and 0.62 ms as `int8`. Anything sorted here must be keyed in the narrowest integer dtype that fits its range *including* the sentinel (unstable `kind` is not an option: the schedule needs in-group order).
- `evoxc.cond` accepts a `while_loop` in each branch on the numpy backend (both branches are traced, one is executed), so a cheap "fast path + expensive fallback" decision is expressible in-graph.
- No unsigned dtype in a graph (see the dtype contract): a non-negative range test is two signed comparisons (`>= 0` and `< span`) in the key's own dtype, not `cast(keys, uint32) < span`, which saves an op but cannot be lowered to iree.
- There is no `broadcast_to` on `evoxc` (use `enp.broadcast_to`), no `searchsorted`, no `choice`, no `bitcast`, no `where` (it is `evoxc.select(pred, on_true, on_false)`).
- Control flow (`evoxc.while_loop`/`cond`/`scan`) is traced into regions: loop bodies must keep shapes/dtypes fixed, conditions must be scalar tensors, and no `vmap` may be applied over them.
- `evoxc.slice(x, start, lengths)` takes per-axis tuples; `evoxc.reshape`/`transpose` are the only structural ops needed here.
- An `@evoxc.defn` object is not callable: call `.fn` for a plain function (the tests bind `.fn` with `functools.partial`), or `evoxc.build` the `Defn` itself.

## Scatter portability on `backend="iree"` (measured, IREE 3.11.0)
**One index value feeds exactly one scatter.** That is the whole rule; everything below is why.
`evoxc`'s StableHLO writer emits a flat write (`tools.flat_scatter_prepared`, `tools.masked_flat_scatter`) as a rank-1 axis-0 *scalar-window* scatter: operand `tensor<Rxi32>`, indices `tensor<nx1xi64>` (`convert` from the int32 block, then the range-normalize `compare`/`add`/`select`), updates `tensor<nxi32>`, `#stablehlo.scatter<update_window_dims = [], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>`.
IREE lowers that to `iree_linalg_ext.scatter dimension_map = [0]` whose `ins` wants the index tensor **collapsed to rank 1** (`tensor<nxi32>`), and it performs the collapse inside `iree-stablehlo-to-iree-input` as an `unrealized_conversion_cast` rather than a `tensor.collapse_shape`.
When the same index SSA value (an explicitly shared block *or* two expressions IREE's CSE merges) is consumed by a **second** scatter, that materialization stays live and the pass aborts: `failed to legalize unresolved materialization from ('tensor<nx1xi32>') to ('tensor<nxi32>') that remained live after conversion`, reported at the source location of the second scatter.
The emitted form itself is fine — a single scatter with the same shapes compiles in ~0.25 s at any size — so this is contextual, not a bad op spelling.
Alternatives tried and rejected: the rank-2 axis-0 *row* form (operand `(R, 1)`, indices `(n, 1)`, `update_window_dims = [1]`) fails identically; the shape of the operand, the width, i32-vs-i64 indices and the presence of `while`/`if` regions or `sort` are irrelevant (bisected by deleting ops from the failing module — removing *either* of the two shared-index scatters makes the whole 7 164-line module compile, and so does giving one of them a distinct index expression).
Consequences for this node: `incoming_table` writes `in_conn` and `in_src` through **one** scatter over a doubled `(P, 2*(N*K+1))` buffer (`tensorization._window_pair`); any caller that wants several payloads in the same cells must do the same, never `flat_scatter_prepared` twice with one block (the current `genome/` sites each scatter through their own index expression, so they are unaffected — keep it that way).
Reproduction loop (seconds per iteration, no pipeline rebuild):
1. monkeypatch `evoxc.build` with `g = evoxc.trace(fn, *specs); text = evoxc.lower(g, backend="iree").text()` and also call the original so the run continues;
2. `iree.compiler.compile_str(text, target_backends=["llvm-cpu"], input_type="stablehlo")` — a failure reproduces from the text alone, and `str(exc)` carries the `iree-compile` diagnostics;
3. to attribute a failure to an op, delete ops from the dumped module (replace each result with a fresh `func.func` argument of the same type) and recompile: a module that still fails is still a reproducer.
Minimal reproducer (two scatters, one block): `P, R, N = 4, 68, 32`; `block = tools.prepare_flat_index(x, idx, trusted=True)`; `tools.flat_scatter_prepared(x, block, u1)` **and** `tools.flat_scatter_prepared(x, block, u2)` → `failed to legalize unresolved materialization from ('tensor<128x1xi64>') to ('tensor<128x1xi32>')`; with only the first call it compiles.
End-to-end on this box (`DefaultGenome(num_inputs=2, num_outputs=1, ..., output_transform=ACT.sigmoid)` + `NEAT` + `XOR`, seed 42): pop 4 / species 2 / `max_nodes=4` / `max_conns=8` compiles the step graph in 7.4 s and pop 100 / species 10 / `max_nodes=12` / `max_conns=24` in 8.4 s; `setup` (its own ask graph) compiles in ~0.9 s; three generations of both configs reproduce the numpy backend bit-for-bit.

## Constraints
- No `jax`/`torch`/`numpy.random` in graph code; randomness is `evoxc.random` stateless keys (`key(seed)`, `split`, `randint(key, shape, low, high)`, `uniform`, `normal`, `permutation`).
- Files ≤ ~1000 lines; shared math lives here, never duplicated in `genome`/`algorithm`.
- `tensorization.py` (998 lines) is one responsibility — derive the schedule — with the derivation recipe, the contract and the measured-cost/rejected-lever record next to the kernels, and it sits at the top of the line budget because splitting it would separate the recipe from the code that implements it; condense prose before adding code.
- Inside this node one rule has one owner: `tools` owns the positive-static validator (`_positive_static`, used by `batch_offsets` and by every static in `tensorization`), the rank-0 scalar-constant builder (`_scalar_constant`) and the host-side array accessor (`to_numpy`, which `tensorization._host_array` converts through); those three are imported, never re-written. `host_graph` keeps its own `_I_INF` literal on purpose — `tools` pulls in `evoxc` and that module is deliberately backend-free.
- Nothing in this node imports from `genome`, `algorithm`, `problem`, `pipeline`, `benchmarks`, or `test`.
- Every public function carries a napoleon docstring stating shapes, dtypes, sentinels, and the invariances it relies on.
- Errors: validation raises `errors.TensorNEATError` subclasses on the host, at configuration/build time, before a graph is traced; in-graph overflow is reported as data (`edge_overflow`), never silenced.

## Known Gaps / Open Decisions
- The pre-rewrite helper names are gone and must not be re-introduced by consumers: `attach_with_inf` → `masked_flat_gather`; `hash_array` → `hash_rows` (host-only).
- Host-side graph walks (visualisation, `network_dict`, sympy codegen) use `host_graph.py`, which owns the graph algorithms for interpretability and export.
- The in-graph `(N, N)` adjacency-matrix topological sort of the pre-rewrite library has no replacement, because execution order comes from `node_level`/`topo_key`.
- **No bitcast op in evoxc** (v1): the payload is split into int32 keys and float32 attributes; `genome.hash()` is in-graph (`genome/gene/base.py::hash_int32`, uint32 xor-mix over the key columns) and `tools.hash_rows` is its host twin, not the other way round; in-graph gene identity (crossover matching) compares key columns with `equal`. Revisit if evoxc grows `bitcast`/`reinterpret`.
- **The v1 StableHLO op table is the portability gate.** `stablehlo.ops.status` reports `cumsum`, `erf`, `gelu`, `tril` and `flip` as `deferred` (numpy interpreter only — hence the `cc` counting trick instead of `cumsum`), and although `scatter` itself is v1 an operand of rank ≥ 3 is rejected by the exporter, which is why every write here goes through the rank-1/2 flat-index helpers. Together with the signed-dtype rule this is the whole checklist: the `setup` and `step` graphs a real pipeline builds contain no unsigned dtype and no deferred op, and both lower on `backend="iree"`. `argsort`/`sort`/`gather`/`select`/`while`/`if`/`pad`/`tile`/`dot`/`random_*` have no such caveat.
- No scatter-with-reduce ⇒ per-node aggregation uses the padded `(P, N, K)` table; `K` is a static bound, so in-degree growth forces a rebuild (reported via `edge_overflow`).
- No per-row batched gather/scatter in the evoxc frontend (shape rules mimic `take`); the flat-index helpers are the only sanctioned workaround, and they require `P*R` to stay inside int32.
- `key_span S` is a static bound on key magnitude: keys must stay compact (hidden-node key compaction on mutation) or the table grows; `S` is part of the shape signature and rebuilds on overflow.
- The ACT/AGG dispatch is at its op floor for the documented id contract, and the remaining cost is data movement: `sum`/`mean`/`maxabs` still need one `select(mask, z, 0.0)` over `(P, N, K)` per level, so the forward pass would gain only from a caller that guarantees dead message slots are already exactly `0.0` (today the genome only NaN-sanitises them, and it owns that contract).
- evoxc does not CSE equal literals, so each dispatch emits one `constant` per distinct literal (3 for a one-option level, 18 for the canonical aggregation list); sharing them across the unrolled levels needs a graph-level constant cache, which evoxc does not expose.
- `agg.maxabs` pays `tools.flat_gather`'s guard chain (`less`/`greater_equal`/`logical_or`/`select` plus a `(P, 1)` `arange` constant) even though an `argmax` result is provably inside `[0, K)`: the guard-free path exists as `trusted=True` but `functions/` has not adopted it, because `flat_gather`'s caller there cannot bound `K` itself. `maxabs` is not an option of the default genome, so the default XOR path never pays it (3.19 vs 2.34 ms per level at P=1000, measured in this node).
- The remaining `derive_schedule` cost is the `(P, N, K)` window build (two scatters, ~1 µs per 200 k-element scatter), the ~47 `(P, C)`/`(P, N)` selects/`add`s of the tail and ~580 dispatches of interpreter overhead; a further 2× would need the window representation itself to change, which the `Transformed` contract fixes.
- The default genome's forward pass unrolls `max_depth = 16` level steps for every population, even a three-node network, so a per-level saving is multiplied by 16; unrolling on the real depth (or a level `while_loop`) is a genome-side change (`genome/` owns it).
- **The step graph of a pop-1000 pipeline embeds a population-sized constant and says so at build time.** `Pipeline.setup()` plus the first `step` at pop 1000 (`DefaultGenome(num_inputs=2, num_outputs=1, max_nodes=50, max_conns=100)` + `NEAT` + `XOR`) emits exactly one `UserWarning: Embedding a tensor of 1608000 bytes as a graph constant (threshold 1048576 bytes, set via EVOXC_LARGE_CONSTANT_BYTES); prefer passing it as an explicit input instead.` — and a build run under `-W error::UserWarning` fails there instead.
- The emitter is `tensorization._window_pair` (the `enp.full` seed of its scratch buffer), which fills a doubled `(P, 2 * (N * K + 1))` buffer with `I_INF`: the fill value is constant but the shape scales with `pop_size` (1000 × 402 × 4 B = the reported 1 608 000 bytes at `N=50 K=4`), so evoxc materialises the whole array as a graph constant instead of broadcasting a small one.
- Pop 500 stays below the threshold (804 000 B, no warning).
- The cost is build time and artifact size, not wall time: the one-off step build is ≈1.5 s at pop 1000 on this box (the `derive_schedule` build alone is ≈0.3 s) with peak RSS ≈395 MiB, and per-generation time is unaffected.
- Fix direction for whoever takes it: emit a rank-0 (or `(1, 2 * (N * K + 1))`) `I_INF` constant and `enp.broadcast_to` it to the target shape, which also shrinks the compiled artifact and the build — then measure `Pipeline.step` end to end at pop 1000, because a value-identical "free" op removal in this node has already cost +34 % per generation.

## Notes for Agents
- `test/unit/common/*` is the runnable kernel suite here, with `test/support/oracle.py` the numpy schedule oracle and `test/support/functions_ref.py` the ACT/AGG oracle; measured on this tree: `uv run pytest -o addopts="" -q test/unit/common` → 155 passed (numpy backend, no optional dependency) and `uv run pytest -o addopts="" -q -m parity` → 53 passed (needs `uv sync --extra bench` for the jax oracle). A performance change here still needs a field-by-field diff against the implementation it replaces (feed-forward, cyclic, in-degree > `K`, absent nodes, unmatched keys, both key spellings, and both `acyclic` settings) because the suites do not pin bit-exactness of every `Transformed` field.
- `import tensorneat` works from an installed checkout (`uv pip install --python .venv/bin/python -e . --no-deps`) and pulls in no `jax`/`torch`/`brax`/`gymnax`/`mujoco`; the measured-cost snippet's synthetic `tensorneat` package is only a shortcut for timing `tensorneat.common` straight from `src/` without installing.
- `test/support/oracle.py::schedule` rebuilds the shipped `Transformed` field set (`node_level`, `node_exists`, `in_valid` included) from an independent numpy derivation, so it is the field-by-field reference a schedule change is diffed against.
- `evoxc` in a bare checkout: `uv venv .venv && uv pip install --python .venv/bin/python evoxc numpy sympy networkx matplotlib pytest && uv pip install --python .venv/bin/python -e . --no-deps`; `pytest` finds `src/` and `test/support` through `test/conftest.py`, so a run is `.venv/bin/python -m pytest test/unit/common`.
- Timing anything here: build once and time `evoxc.run`, one cell per fresh process (the numpy allocator makes a second executable in the same process skew later cells up to 25 %), and treat the ratio against the previous implementation as the result — absolute milliseconds move ~30 % between sessions on this box.
