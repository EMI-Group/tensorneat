# tensorneat/common — host primitives, batched tensor tools, the tensorization (execution-schedule) engine, ACT/AGG registries, sympy codegen

## Intent
Framework-wide foundation: the immutable `State` container, the error hierarchy, the batched gather/scatter/mutation helpers every kernel is built from, the **execution schedule** turning a ragged evolved topology into static-shape arrays, the ACT/AGG registries and host-side sympy code generation.
Nothing here imports `genome`, `algorithm`, `problem`, or `pipeline` — this node is the bottom of the package dependency DAG.

## Dtype & sentinel contract (binding for genome / algorithm / problem)
The gene payload is **split by dtype**: `node_keys (P, max_nodes) int32`, `node_attrs (P, max_nodes, A) float32`, `conn_keys (P, max_conns, 2..3) int32` (`(in_key, out_key)` first, an optional historical marker third), `conn_attrs (P, max_conns, W) float32`.
Absent key rows are `I_INF`; absent attribute values are `NaN` — the pre-rewrite NaN padding semantics survive in the attribute arrays (`isnan(attrs[..., 0])` still means "row unused"), while key arrays use the integer sentinel.
Keys are `int32` in `[0, KEY_MAX)` with `KEY_MAX = 2**24`; a key outside that range, or two nodes with the same key, is a caller bug (see `ConfigError`).
One integer sentinel only: `I_INF = 2147483647` (`int32` max) = "no index / unmatched / unavailable", in integer arrays only, never mixed with `NaN` in one array.
Rationale: evoxc has **no bitcast op** (`cast` is numeric only), so a mixed-dtype gene row is impossible and the payload is split; keys stay `int32` so absence is an exact comparison and `key_span` really is a key-magnitude bound.
Consequences: an absent row is `keys == I_INF` in the key arrays and `isnan` in the attribute arrays; int-valued genes (activation/aggregation option positions) live in the float attribute columns, stay integral-valued and are cast with `cast(..., int32)` before use; in-graph gene identity uses the **key columns** with `equal`, never a hash of the row (see the Rationale above).
`tensorization.derive_schedule` accepts **either** spelling for its key arguments: `int32` with `I_INF` padding (canonical — what the payload and every kernel pass) or `float32` with `NaN` padding (the pre-rewrite/parity spelling `test/support/oracle.py` still uses); the dispatch is a trace-time `if` on a static dtype, never a runtime branch — `present = keys != I_INF` for integer keys, `present = isfinite(keys) & (keys < KEY_MAX)` for float keys.
**Graph dtypes are signed only**: no `uint8/uint16/uint32/uint64` may appear in any operand, result or constant of a graph this package builds, because the iree backend's declared set is float16/32/64, int8/16/32/64 and bool — an unsigned value lowers nowhere but the numpy interpreter.
Use a same-width **signed** carrier instead, and where *unsigned order* genuinely matters (wrap-around comparisons, modular hashing) flip the sign explicitly rather than casting to an unsigned dtype; `tools.hash_rows` is the one deliberate exception, being numpy host code that is never traced.
Consequence for range tests: `key ∈ [0, key_span)` is two signed comparisons in the key's own dtype (`greater_equal(keys, 0)` and `less(keys, key_span)`), one op more than the `cast(keys, uint32) < key_span` trick that cannot be lowered to iree.

## Execution schedule contract (`Transformed`)
`tensorization.derive_schedule(...)` is one `@evoxc.defn` graph; its output is a host-visible pytree of arrays only (**no Python statics inside** — every static fact is implied by a shape or a dtype).

Runtime inputs (batch-first, population on axis 0); the five statics are **positional-or-keyword** (evoxc passes statics positionally into `build`, and `functools.partial(fn, **statics)` also works):
- `node_keys (P, max_nodes)` `int32` (or `float32`) — node key column; `I_INF`/`NaN` marks an absent row.
- `conn_in (P, max_conns)`, `conn_out (P, max_conns)` — connection endpoint keys, same dtype as `node_keys`.
- `input_keys (I,)`, `output_keys (O,)` — keys of the input/output nodes, shared by every genome (no batch axis).
- statics `pop_size P`, `max_nodes N`, `max_conns C`, `max_in_degree K`, `key_span S` — the Pipeline's shape signature and therefore the cache key of the compiled executable — plus a trailing `acyclic=False`, a caller assertion that the population contains no cycle (a build-time knob, not part of the shape signature).

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
1. dense key→slot table `(P, S+1) int32` (trailing column = write dump), one flat scatter over node keys; the accelerator formulation never builds it (`key_template` answers the same lookups by equality).
2. `src`, `dst` = flat gather from that table with sanitized keys (`absent -> 0`, clamp to `[0, S-1]`), then `edge_valid = row present & table hit`, and `node_exists` from the node key column.
3. `edge_order` — **one** stable `argsort` of the combined `(edge key 2*dst, node key 2*slot + 1)` array, in the narrowest integer dtype that holds its `2N + 2` sentinel (`int8` up to N=62, `int16` up to N=16382, else `int32`); the argsort **output** cast once to `int32` is returned as `order` (position -> entry id, entries `< C` being connection rows in `(dst, connection index)` order), and its inverse permutation `inv` (one scatter) yields `cc[i] = inv[C + i] - i`, i.e. `inv[C + i] = cc[i] + i`; the accelerator formulation counts that same predicate directly (`cumulative_indegree`) instead of inverting the permutation, and the `argsort` itself stays in both.
4. `start[i] = cc[i-1]` (0 for `i = 0`), `deg[i] = cc[i] - start[i]`; node `i`'s in-edges sit at sorted positions `start[i] + i + k` for `k < deg[i]`.
5. `in_conn`/`in_src` = two gathers at that `(P, N, K)` index block (`in_conn = order[start[i] + i + k]`, `in_src = src[in_conn]`), masked to `I_INF` where `k >= deg[i]`; `edge_overflow = reduce_sum(max(deg - K, 0))`; `in_valid = in_conn != I_INF`.
6. `node_level` = `while_loop` longest-path relaxation over the `(P, N, K)` table, inputs pinned to 0, in the narrowest carrier dtype: the default runs the level-only relaxation (cap `N`), then decides **in-graph** with `evoxc.cond` whether any valid edge violates `level[dst] > level[src]` — no violation ⇔ acyclic, and the level-only answer is then exact; a violation runs the two-relaxation machinery (`level` + `reach`, cap `2N`, snapshot at `N`) that reports every node on/after a live cycle as `I_INF`. `acyclic=True` returns the level-only answer directly.
7. `topo_key`, `input_slot`, `output_slot` are then free.

Cost: `O(P * (S + C log C + N*K*D))` with `D` = max depth, i.e. edge-proportional except for the `N*K` padding and the one-off `key_span` table; **no `cumsum`, no unsigned dtype, no scatter-with-reduce, no `max_nodes**2` matrix**.
`transform` runs once per generation, so its cost is amortized over the whole generation.

## Measured cost of `derive_schedule` (node-local)
Depth-qualified, on the numpy reference backend, shape signature `N=50 C=200 K=8 S=64`, 46 present node rows (4 padded), 2 inputs, 2 outputs, a layered feed-forward DAG population; every figure is the **median of 15 timed `evoxc.run` calls** with the build cached, single-threaded (1 BLAS/OpenMP thread), ms, and the snippet at the end of this section regenerates it.
Every figure here is node-local; the per-generation pipeline cost lives in `benchmarks/PERF_NUMPY.md` and is not restated here.
"before" is the two-sort / two-relaxation derivation used as the measurement baseline, "after" is the shipped one; the single-cell cells were measured interleaved in one session and the box drifts ±30 % between and within sessions, so the *ratio* is the load-bearing number:

| topology | P=100 | P=500 | P=1000 | speed-up at P=1000 |
|---|---|---|---|---|
| depth 5, 200 live edges | 10.7 → 5.9 | 34.4 → 16.6 | 83.2 → 30.5 | 2.7× |
| depth 8, 200 live edges | 12.7 → 5.6 | 40.7 → 13.3 | 97.1 → 33.1 | 2.9× |
| depth 18, 111 live edges | 20.7 → 8.5 | 66.9 → 20.2 | 144.5 → 33.8 | 4.3× |
| random cyclic, 200 live edges | 92.6 → 67.7 | 311.1 → 149.9 | 607.6 → 245.9 | 2.5× |

Single-cell interleaved protocol (three alternating rounds of 15 runs, P=1000, depth 8): before 103.5 / 102.4 / 105.2 ms, after 33.6 / 33.4 / 33.5 ms, `acyclic=True` 30.2 / 32.1 / 30.2 ms — ~3.1×, the cycle-safe default within ~10 % of the explicit acyclic assertion.
One-off `build` (cached per shape by the Pipeline, so not paid per generation): P=100 ~65 ms, P=500 ~140 ms, P=1000 ~280 ms; the default traces both `cond` branches (+2–6 % over `acyclic=True`).

Where the time goes (nesting-aware exclusive per-op profile, P=1000, depth 8, default path, 580 dispatches — taken on the pre-gather derivation, so the `scatter` row is the 4 ops of that tree, one of which the window gathers replaced; read `gather`/`select` as lower bounds and the ROCm/numpy subsections below for the split the shipped code measures): `gather` 16 ops 7.3 ms · `select` 47 ops 6.6 ms · `scatter` 4 ops 4.0 ms (≈1 µs per 200 k-element scatter — the most expensive op per element) · `add` 32 ops 2.6 ms · `maximum` 72 ops 1.7 ms (the `K` fold: 8 slices per relaxation step) · `while` own overhead 1.5 ms · `reduce_max` 12 ops 1.2 ms · `argsort` 1 op 1.2 ms · `cast` 18 ops 1.1 ms · `reshape` 91 ops 0.9 ms; every other op kind under 1 ms.

Interpretation:

- **Per-op interpreter overhead is a first-class term**: 580 dispatches at ~10–15 µs each is 6–9 ms of the ~33 ms, so op *count* matters as much as op size (one removed `(P, N)` select ≈ 50 µs; the tail is many small `(P, C)`/`(P, N)` ops, not one big pass).
- **Depth does not dominate**: rows depth 5 and depth 18 with the same live-edge count differ by ~10 % (30.5 vs 33.8 ms at P=1000) although the relaxation runs 6 vs 19 trips, because the loop is cheap (one `(P, N, K)` gather plus a `K` fold per trip in a narrow carrier) and the depth-independent window build, the one sort and the key tables hold the rest.
- **A cyclic population pays ~2.5× an acyclic one** (245.9 vs 33.1 ms at depth 8): the level-only campaign runs first and the `cond` fallback then runs the full two-relaxation machinery; cycles are the RecurrentGenome's business, so a feed-forward population never pays it.
- **Share of a generation**: ~5 % of one `Pipeline.step` at pop 1000 (XOR, `DefaultGenome`, `NEAT`, build excluded, same session as the table above), so `speciate`/`breed`/mutation (in `algorithm/`) own the generation figure.

Rejected levers (measured in this node on the numpy interpreter; do not re-derive):

- **Dense `(N, N)` adjacency / max-plus matrix power** for the relaxation: `P·N·N` = 2.5 M element-ops per iteration against `P·N·K` = 400 k, and max-plus has no matmul form (it needs `N` elementwise `maximum`s), so it is ≥ 6× worse at `K=8`.
- **One-hot / einsum counting for `rank` and `cc`**: a `(P, C, C)` (40 M at P=1000) or `(P, C, N)` (10 M) tensor against a 250 k-element sort — the sort wins by an order of magnitude.
- **Static Python unrolling to a `max_depth` static**: per-iteration work is unchanged, the removed loop overhead is ~1.5 ms in total, and the unroll bound must be `≥ depth` for every member, so `max_depth = 16` runs more iterations than the 6–19 a convergent loop needs.
- **One gather over a stacked `(P, 2, N, K)` `level`/`reach` carrier**: same copied volume, one dispatch less — measured slower.
- **Fusing the two cycle-violation gathers** (`src`/`dst` interleaved) and **casting the violation mask to `int8`**: both inside noise (36.4 vs 35.8 ms), rejected.
- **Per-member `cond`**: `vmap` over control flow is deferred in evoxc and the batch axis must be explicit — not expressible.
- **`reduce_max` over the `K` axis** instead of the pairwise `maximum` fold: 1.26 vs 0.30 ms per level at P=1000 (numpy's `np.max` over a trailing 8-axis is slower than seven `(P, N)` `maximum`s).
- **A `-2**30` sentinel** instead of the `select(greater(best, -1), best + 1, -1)` guard: an unsettled predecessor would settle its child at `0`; caught by the field-by-field diff.
- **A trusted, guard-free index block everywhere**: the escape test is what makes the backend *raise* instead of reading a neighbouring row, so `trusted=True` stays opt-in and only where the index range is provably inside the table.

### The derivation tail is at its op floor (a 17-op trim is rejected; do not re-derive)
A trim to 17 fewer executed ops (339 against 356: `_key_indices` drops the dtype-identical `cast` and uses a 0-d zero, `edge_valid` becomes `max(src, dst) < max_nodes`, `_slot_of` drops a redundant `present` redirect, `incoming_table` stops sanitising the dump cells of its then-scatter) is **rejected**: value- and work-identical, yet it costs a third of a generation.
One session (XOR, seed 42, fresh process, 8-10 timed generations, 2-3 interleaved rounds; the paired ratio is the result, never the absolute milliseconds): `Pipeline.step` 317-323 ms with the trim reverted vs 428-442 ms with it, i.e. **+34 % at pop 1000** and ~+5 % at pop 500, non-linear in `P`.
Not a data or layout change: `Transformed` is bit-identical (dtypes, shapes, contiguity), the runtime op counts differ by exactly the 17 ops, the dtype signatures match, peak RSS is the same (~805 MB) and `derive_schedule` alone is *faster* (10.4 vs 10.9 ms).
Not attributable to one hunk either: reverting any single one of the four still measures 431-448 ms; only restoring the whole derivation returns 317-323 ms, which leaves a whole-graph execution effect inside the numpy interpreter (the step graph's op/constant mix) that no local edit here can control.
**Reasoning rule: executed-op count is not a proxy for time at this level.** A trim that is value-identical and faster on `transform` in isolation (6.66 -> 6.47 ms) can cost 34 % of a generation once inlined; measure `Pipeline.step` end-to-end at pop 1000 in fresh interleaved processes before landing a "free" removal.

### `node_levels` is at its op floor (measured in this node; do not re-derive)
Where the time goes at the real window: on a real feed-forward XOR population (seed 42, generation 7, `DefaultGenome(num_inputs=2, num_outputs=1, max_nodes=50, max_conns=100)`, `NEAT(pop_size=500, species_size=20)`, so `(P, N, K) = (500, 50, 4)` and depth ≈ 4), `node_levels(acyclic=True)` is 2.79–3.15 ms of a 6.9–7.2 ms `derive_schedule` (the box drifts; the ratio, ~40–45 %, is the result).
Its fast path is ~16 executed ops per loop trip over `depth + 1` trips, i.e. ~35–40 µs per op, so the loop is bound by *executed op count* and every lever below measured neutral or worse (base = 100 %, interleaved rounds of 15 `evoxc.run`, build excluded, medians of 5 alternating rounds).
Value identity of every candidate except the two-sub-step one was checked field by field against the unmodified `node_levels` on the real feed-forward population **and** a synthetic cyclic one, with `acyclic` both `True` and `False`.
- **`reduce_max` over the `K` axis instead of the pairwise fold**: 4.81 vs 3.15 ms (1.52× worse), so the K=8 / P=1000 verdict holds at the narrower K=4 window.
- **Dropping the per-trip `(P, N)` dump-column `concatenate`** (gather with `min(in_src, N - 1)` plus one `(P, N, K)` window mask instead of a `-2` dump cell): 3.26 vs 2.79 ms (1.17× worse) — one `(P, N, K)` `select` costs ~0.5 ms more than the `(P, N)` concat.
- **Two relaxation sub-steps per loop trip**: 3.67 vs 3.04 ms (1.21× worse), and it raises the fast-path level of a *cyclic* payload handed to an `acyclic=True` genome to `2N` instead of `N`, so it is not value-identical there either.
- **Shrinking the loop state from three leaves to two** (jump `step` to `cap` instead of carrying `changed`): 1.00× — the per-trip state copy is not a term.
- **Folding `pinned` out of the fast path's exit blank** (`blank = level < 0` instead of `~pinned & (level < 0)`): provably identical (a pinned node is `0` in every trip, so `level < 0` already excludes it) and two `(P, N)` ops cheaper per campaign, but 0.98–1.00×, under the ±2 % noise floor — take it only alongside a change that pays for its own re-measurement.
- **Static unrolling, dense `(N, N)` max-plus, one-hot prefix counting, per-member `cond`**: rejected for the reasons above.
A variant is cheapest tested without touching the file: `exec` `textwrap.dedent(inspect.getsource(tensorization.node_levels))` with one textual substitution into `dict(vars(tensorization))`, then diff `build_defn` output against the unmodified function on arrays from a real `Transformed`, variants interleaved.
The ceiling is small by construction: `transform` is ~3 % of one `Pipeline.step` at pop 500 (XOR, seed 42, measured in this node), so even a 2× `node_levels` moves a generation by ~1 %.

### ROCm cost of the schedule's table writes (measured on MI300X, gfx942)
On a GPU the schedule's *writes* are the expensive part, and the reason is a lowering fact rather than a size: every `evoxc` scatter becomes `#iree_gpu.pipeline<Distribute>` with `workgroup_size = [1,1,1]` and grid `workgroups([%c1,%c1,%c1])`, i.e. one workgroup of one thread, while `gather` lowers to a parallel `TileAndFuse` dispatch.
Measured cost law, one-kernel graphs at the `xor_p500` geometry (`P=500, N=50, C=100, K=4, S=150`), session 2026-11-05 on the box's MI300X VF (gfx942, IREE 3.13, evoxc v0.1.1), median of 10 `evoxc.run` minus the 65 µs per-graph floor: 25 000 update cells → 540 µs, 75 000 → 1 351 µs, 100 000 → 1 664 µs, i.e. **16.6–21.6 ns per updated cell**.
Cost therefore tracks the number of *update elements*, not the destination width: `key_slot_table`'s 25 000 writes into 75 500 cells (21.6 ns/cell) and `edge_order`'s 75 000 writes into 75 000 cells (18.1) and the deleted window scatter's 100 000 into 201 000 (16.6) all sit on the same line, and a masked scatter that iterates its whole update tensor is not cheaper for writing fewer cells.
Narrowing the carrier is a **loss**: a 16-bit inverse permutation measures 27.8 ns/cell against 17.1 for `int32`, and `int8` wraps for a 150-entry permutation; an `enp.full(I_INF)` base instead of `enp.zeros` is noise (1 360 vs 1 348 µs). Do not re-derive these.
`incoming_table` is the one write this node removed: it scattered 100 000 cells into the doubled `(P, 2*(N*K+1))` buffer and now gathers the window from `edge_order`'s permutation (`order`, `cc`), with `Transformed` bit-identical — the whole pair measures 3 035 → 1 463 µs of device time and one warm `xor_p500` generation 84.88 → 83.29 ms (**−1.59 ms, −1.87 %**, 7 timed runs, build 8.51 → 8.87 s; the generation graph drops 11 dispatch sites / 10 exports, the 1-thread `scatter_201000xi32` replaced by one `elementwise_broadcast_500x50x4_i32` on a 3 250 × 64 grid performing both gathers).
The standing budget after it: 41 scatter dispatches per generation, every one still a single workgroup of a single thread, 254 657 update cells ≈ 4.35 ms of device time at 17.1 ns/cell.
The largest survivors are `edge_order`'s inverse permutation (75 000 cells, 1 351 µs, 88 % of the remaining pair — it is the `cumsum`-free prefix count, so replacing it means a Hillis-Steele prefix over the `(P, C+N)` position mask, cheap on GPU and dearer on numpy: unmeasured, next pass) and `key_slot_table`'s dense table (25 000 cells, 540 µs, no lever found — the `(P, key_span+1) int32` shape with `I_INF` misses is the tested contract and one write per node row is the minimum).
Rejected variant, kept as the numpy-side option: pack the pair into one payload (`in_conn * 64 + in_src` into `(P, N*K + 1)`, unpacked with `>> 6`/`& 63` — `//`/`%` is unusable, evoxc has no `floor_divide` and `divide` returns f64) halves the scatter to 50 000 cells; it measures 3 035 → 2 429 µs for the pair (**−606 µs, −1.4 %** of a generation) and is the only variant that also helps numpy (chain 3 590 → 2 252 µs), but it captures 2.6× less of the ROCm win and keeps a serialized scatter.
A node-only `cc` scatter into `(P, N+1)` is not cheaper (1 457 vs 1 421 µs) for the same reason.
Artifacts shrink with the deleted buffer: StableHLO 10.0 → 6.8 MB, baked constant payload 4.21 → 3.00 MB, HSACO 715 → 695 kB, and the pop-1000 large-constant warning disappears.

### numpy cost of the gather formulation (same benchmark case)
On the numpy interpreter the trade is the other way round and small: on a real evolved payload (`xor_p500` seed 42, 5 generations, `derive_schedule` built once and `evoxc.run` median of 20) `derive_schedule` reads 11.44 → 11.57 ms and at `xor_p1000` 18.57 → 18.32 ms, because `incoming_table`'s two gathers (1.18 → 1.37 ms, and 2.22 → 2.45) cost slightly more than the scatter they replaced while `edge_order` loses `rank` (1.08 → 0.99, 1.88 → 1.69).
End to end, 18 alternating fresh-process rounds per cell: `xor_p500` 119.48 → 119.65 ms per generation (raw median +0.14 %, paired ratio −0.40 %, paired 95 % CI −2.4 %…+1.6 %) and `xor_p1000` 182.70 → 181.97 ms (paired −0.42 %), with −17 dispatches per generation (8 210.6 → 8 193.6 and 8 263.75 → 8 246.75), peak RSS −3.1 % / −3.9 % and build −5.3 % / −8.5 %; the search trajectory is bit-identical (final best and `sha256(state.pop)` match on every round).

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

def layered_dag(depth, rng):                             # adjacent levels only, in-degree <= K
    hidden = [k for k in PRESENT if k not in INPUTS and k not in OUTPUTS]
    level = {k: 0 for k in INPUTS} | {k: depth for k in OUTPUTS}
    level |= {k: 1 + i % (depth - 1) for i, k in enumerate(hidden)}
    by_level = {}
    for k, lvl in level.items(): by_level.setdefault(lvl, []).append(k)
    edges = set()
    for k in sorted(level, key=level.get):               # 2 preds on the previous level
        if level[k]:
            preds = by_level[level[k] - 1]
            for _ in range(2): edges.add((int(preds[rng.integers(len(preds))]), k))
    in_deg = {}
    for _u, v in edges: in_deg[v] = in_deg.get(v, 0) + 1
    cand = [(u, v) for lvl in range(depth) for u in by_level.get(lvl, [])
            for v in by_level.get(lvl + 1, []) if (u, v) not in edges]
    rng.shuffle(cand)
    for u, v in cand:                                    # fill up to C edges, in-degree <= K
        if len(edges) >= C: break
        if in_deg.get(v, 0) < K: edges.add((u, v)); in_deg[v] = in_deg.get(v, 0) + 1
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
        rows = list(edges); rng.shuffle(rows)            # mutation order, not sorted
        ci[p, :len(rows)] = [u for u, _ in rows]; co[p, :len(rows)] = [v for _, v in rows]
    return (nk, ci, co, np.asarray(INPUTS, np.int32), np.asarray(OUTPUTS, np.int32), pop_size, N, C, K, S)

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
              f"run={timed(exe, run_args):7.2f}ms  acyclic={timed(acy, run_args + (True,)):7.2f}ms", flush=True)
PY
```

Measure a cell you intend to compare in a fresh process: building several executables in one process changes the heap layout and later cells then read up to 25 % faster.
The nesting-aware per-op profile of the same graph is a one-file monkeypatch of `evoxc.backends.numpy.kernels.dispatch` (time each kernel, subtract the nested time of the ops it runs, report exclusive time per op name); re-derive it whenever a change has to be attributed to an op.

## Forward-side contract (consumed by `genome`)
Per level `l = 0 .. reduce_max(node_level)`:
- `mask = in_valid & ~isnan(gathered source value)` — the legacy `conn_exists & src_valid` mask.
- gather each conn gene attribute with `tools.masked_flat_gather` from the flattened `(P, C*attrs)` payload using `in_conn` (indices `(P, N, K)`), and the source values with `tools.masked_flat_gather(vals, in_src)`.
- `z = functions.agg.apply_aggregation(contribs, mask, agg_ids)` — masked reduction over the `K` axis as a static `K`-plane fold (not a `reduce_*`; `functions/CONTEXT.md` owns the recipe and its `K <= 128` bound), neutral-padded, dispatching on the per-node aggregation id.
- `new = functions.act.apply_activation(z, act_ids)`; `vals = select((node_level == l) & valid_count > 0, new, vals)` with `valid_count = reduce_sum(in_valid, -1)`.
The per-node `valid_count` guard matters: a node whose incoming edges are all invalid keeps `NaN` (legacy behaviour relied on `lax.cond(has_valid, ...)`).
Kernels must never index outside `[0, size)` — evoxc's numpy backend **raises** on out-of-range gather/scatter where JAX clamped, so clamping plus masking is mandatory, not an optimisation.
One level costs ~1.1 ms on the numpy interpreter at `(P, N, K) = (1000, 50, 4)` with one aggregation and one activation option (executable built once, `evoxc.run` median of 15), and 16.6 ms with the canonical six-aggregation / twelve-activation lists — the per-option `select` cascade that `functions/CONTEXT.md` measures.
The mandatory `select(mask, z, 0.0)` on `(P, N, K)` is the single most expensive op of a level now that the `K` axis is folded and cannot become a masked multiply (a dead slot may hold `NaN` or `±inf`, whose product with `0` is `NaN`), so the aggregation passes — not the dispatch — remain the term to attack.

## API Surface
### `state.py`
`State(**kwargs)` — immutable host container over a dict; `register/update/remove` return new `State`s (raising on duplicate/missing keys); `__getattr__` raises `StateKeyError`, `__getitem__` the plain `KeyError`; `save`/`load` pickle it; `tree_flatten`/`tree_unflatten` + `evoxc.register_pytree_node` registration let it cross a graph boundary.
`StateKeyError(AttributeError, KeyError)` is the one dual-base exception in the package: `AttributeError` is what `copy.deepcopy` needs (it probes `__deepcopy__` on the *instance* with `getattr` and absorbs only `AttributeError`, which is where a bare `KeyError` used to escape), what makes `hasattr`/`getattr(state, name, default)` report absent, and what `Pipeline._shape_signature`'s `getattr(state, "pop", None)` probe needs to reach its documented `TensorNEATError` instead of a `KeyError`; `KeyError` keeps the legacy `except KeyError` around `state.pop_size` in `genome/base.py::_resolve_pop_size` raising its own `TensorNEATError` without a cross-node edit.
`copy.copy` and `pickle` never needed the change (CPython resolves `__copy__`/`__reduce_ex__` on the type, not the instance, and both already worked with the old bare-`KeyError` `__getattr__`).
`StatefulBaseClass` in `stateful_class.py` — `setup(state) -> State`, plus a generic `show_config(registered_objects=None) -> dict` that nothing calls (`Pipeline.show_config` renders through `host_payload._object_config`).
The price of the dual base is that `hasattr(state, "typo")` answers `False` instead of propagating, so a typo is loud only at a bare `state.typo` read, an explicit `except`, or `state["typo"]` (plain `KeyError`); writing a new key is still refused by `__setattr__` with `AttributeError("State is immutable")`.

### `errors.py`
`TensorNEATError(Exception)` root; subclasses `ConfigError`, `ShapeBoundError` (a static schedule bound was exceeded: `max_in_degree`, `key_span`, `max_conns`), `GraphError` (cycle where feedforward is required, missing input/output node). Backend/shape errors from evoxc are **never** re-wrapped; only validation raises these.

### `tools.py`
Sentinels `I_INF`, `KEY_MAX`; `batch_offsets(p, r)`, `prepare_flat_index(x, idx, present=None, fallback=0)` (the flat `(P, n)` index block of a batched gather/scatter, reusable across calls), `flat_gather_prepared(x, flat_idx, shape)`, `flat_scatter_prepared(x, flat_idx, updates)`, `flat_gather(x, idx)`, `masked_flat_gather(x, idx, mask=None, fill=nan)`, `flat_scatter(x, idx, upd)`, `masked_flat_scatter(x, idx, upd, mask)` (dump-slot routing); `fetch_first(mask, default=I_INF)`, `fetch_random`, `rank_elements`, `argmin_with_mask(arr, mask, *, filler=None, default=None)` — the first and last take their sentinels as hoistable graph constants (see "Hoistable sentinels" below); standalone mutation kernels `mutate_float` and `mutate_int(key, val, options, replace_rate)` (the genome path is `genome/gene/blocks.py::mutate_attrs`, and evoxc has no `random.choice`: `gather(options, randint)`); `to_numpy(tensor)` — the host-side array accessor every logging/checkpoint path uses and the read-back of `Pipeline._place_state`'s cross-device host hop, re-exported by `common.__init__`: a `numpy.ndarray` comes back unchanged, a device tensor (`.device.kind != "cpu"`) is first transferred with `Tensor.to(Device("cpu", 0))` (the single explicit device-to-host step; evoxc never transfers implicitly), then a callable `.numpy()` is called, else `np.asarray`, and `evoxc.Tensor` deliberately has **no** `__array__`; the CPU path never copies; `hash_rows` — **host-only** numpy xor-mix over a row's bytes (`view(np.uint32)`), the host twin of `genome/gene/base.py::hash_int32`.
Every flat-index helper takes `trusted=False`: the default emits the escape guard (`select` + `less`/`greater_equal`/`logical_or` + redirect select) that makes the backend **raise** on an out-of-range index instead of silently reading a neighbouring row, and `trusted=True` skips it only for a caller that has *proved* every index lies inside `[0, R)` (the schedule uses it for `key_slot_table`, `slot_lookup`, the `(P, N*K)` window gathers and both inverse-permutation scatters); never pass it on data you have not bounded.

### Hoistable sentinels (`tools._sentinel_constant`)
Every sentinel a helper emits is a rank-0 `constant` **op**, not a bare Python scalar, because a bare scalar is weakly promoted and changes the `select`'s result dtype (`_scalar_constant` owns that rationale).
One such op is charged per call, so a helper called from inside a `scan`/`cond` body pays it once per iteration; `tools._sentinel_constant(value, fallback, dtype)` is the single resolver that lets a caller hoist it: `None` builds `fallback`, a Python/NumPy number builds a rank-0 constant of `dtype`, and any other object is returned **verbatim** (no cast, no reshape).
A supplied constant whose dtype differs from `dtype` raises `ConfigError` at trace time — `evoxc.select` promotes silently instead of rejecting, so the pinning must be enforced here (`ConfigError`, never a silent widening).
`fetch_first(mask, default=I_INF)` accepts a pre-built `int32` constant for `default`; `argmin_with_mask(arr, mask, *, filler=None, default=None)` accepts them for the masking `+inf` (of `arr.dtype`) and for the `I_INF` "no candidate" index — `None` keeps the one-shot spelling op for op, and the documented semantics ("unset" and "no candidate" read the same `I_INF`) are unchanged.
Measured on this box (numpy backend, one small `scan` body calling both helpers, `evoxc.backends.numpy.kernels.dispatch` counted): the one-shot spelling costs 3 `constant` dispatches per iteration and the hoisted spelling 0, with bit-identical outputs; in a flat graph the two `I_INF` sentinels are emitted twice because evoxc has no constant CSE.
The intended hoister is `algorithm/neat/_kernels.py::speciate`, whose two `species_size`-long scan bodies each call one of these helpers (≈60 dispatches per generation at `species_size=20`); `tools.py` cannot fix that from here.
Helpers that still build a constant per call and have **no** hoistable spelling: `prepare_flat_index` (up to three: the `fallback`, the `pop_size * row_len` redirect and `batch_offsets`' `enp.arange`), hence `masked_flat_gather`/`flat_gather`/`flat_scatter`/`masked_flat_scatter` built on it, and `fetch_random` (two `int32`/`float32` sentinels, no production caller).
`batch_offsets(pop_size, row_len)` is the cheapest to hoist in a loop (one `enp.arange` plus `reshape`, caller-side, by building it once outside and adding it in), and it is already called both ways by `genome/operations/alignment.py` and `algorithm/neat/_keys.py`.

### `tensorization.py`
`Transformed` `NamedTuple`; `derive_schedule(node_keys, conn_in, conn_out, input_keys, output_keys, pop_size, max_nodes, max_conns, max_in_degree, key_span, acyclic=False, kernel_mode="host")` (`@evoxc.defn`, statics positional-or-keyword with `acyclic` then `kernel_mode` trailing); `edge_order(dst, edge_valid, max_nodes, max_conns, kernel_mode="host") -> (order, cc)`; the private `_key_resolver(node_keys, key_span, kernel_mode)` is the one selection point of the key-resolution formulation, and both of its routes are addressed by `_key_indices`'s sanitized `int32` cells so they cannot disagree about which keys are visible; the internal steps as plain, individually testable helpers (`key_slot_table`, `slot_lookup`, `edge_order`, `incoming_table`, `node_levels`); `incoming_table(order, cc, src, max_nodes, max_in_degree) -> (in_conn, in_src, edge_overflow)` (two gathers at the `start[i] + i + k` position block, `trusted=True` bounded by `inv[C + i] - 1 <= C + N - 1`); `node_levels(in_conn, in_src, input_slot, max_nodes, acyclic=False, edges=None)` where `edges` is the `(src, dst, edge_valid)` triple the cycle test needs (`ConfigError` when `acyclic=False` and it is missing); the private body `_host_array` (which converts through `tools.to_numpy`); `validate_schedule(transformed)` host-side invariant checker called by the unit tests and by `genome/grad.py::build_grad`.
The narrow integer carriers are a contract detail, not a knob: the level relaxation and both sort keys are built in the smallest of `int8`/`int16`/`int32` that provably holds every reachable value *and* its "after everything" sentinel (a real value inside the dtype, never `I_INF`), with the thresholds documented next to each builder; results are cast back to `int32` on the way out.

### `schedule_accelerator.py`
Exports `KERNEL_MODE_KEY = "kernel_mode"`, `KERNEL_MODES = ("host", "accelerator")`, `_DEFAULT_KERNEL_MODE = "host"`, `kernel_mode(state) -> str` (the state read every kernel uses), `resolve_mode(mode) -> str`, `is_accelerator(mode) -> bool`, `key_template(node_cells, node_visible, rows) -> resolve(query_cells, query_visible) -> slots (P, L) int32`, and `cumulative_indegree(dst, edge_valid, max_nodes) -> cc (P, N) int32`; `common.__init__` re-exports `kernel_mode`.
It is the second spelling of the schedule's two writes and returns byte-identical `Transformed` fields, so the schedule a caller sees never depends on the mode it was traced with.
`resolve_mode` is the one validation: `None`/absent → `"host"`, an unknown value raises `ConfigError` at trace time (before a graph exists); `Pipeline.setup` registers `kernel_mode` from the device kind via `host_payload._formulation_statics`, so `derive_schedule`'s default of `"host"` is what a caller that omits the static gets.
`key_template` is the equality template the distance and mutation layers already use: one broadcast `equal` of the `(P, L, 1)` query cells against the `(P, 1, N)` node cells, `select(match, slot_of_row, -1)`, `reduce_max` over the node axis, invisible rows → `_NO_NODE` (-2), invisible queries → `_NO_QUERY` (-1), miss → `I_INF`.
`reduce_max` is what makes it bit-identical rather than merely equivalent: the dense table's flat write leaves the **highest** slot at a cell two node rows name, and so does the maximum (this also covers two `float32` keys truncating to the same integer cell).
`cumulative_indegree` counts `edge_valid & (dst <= iota)` over a `(P, C, N)` table and `reduce_sum`s the edge axis — the same predicate the host form reads out of `inv[C + i] - i`, so `cc`, `edge_overflow` and every window position agree.
Both are read-only, so the accelerator mode materialises neither the `(P, key_span + 1)` table nor the inverse permutation; the `argsort` stays, because `order` is what `incoming_table` gathers its window out of.
Neither formulation needs a `cumsum` (deferred), an unsigned dtype, or a scatter at all, and the mode is a host-side Python string reaching the graph through the closure like every other formulation static.

### `functions/`
Exports exactly `ACT`, `AGG`, `FunctionManager`, `apply_activation(z, act_ids, act_fns=ACT_FUNCS)`, `apply_aggregation(values, mask, agg_ids, agg_fns=AGG_FUNCS)`: `ACT`/`AGG` are `FunctionManager`s over 12 activations and 6 aggregations in a fixed, parity-critical order; activations are elementwise `(z) -> z'`, aggregations reduce the last axis of a `(P, N, K)` contribution tensor with neutral padding and are `NaN`-aware through the `mask`; `ACT.name_of(func)`/`AGG.name_of(func)` are the reverse lookups. See `functions/CONTEXT.md`.
Both dispatchers run once per unrolled depth level, so they emit only what the id column needs and every choice is made at trace time: no `cast` when the id column is already `int32` (the node gene's `prepare` casts once per pass), no `equal`/`select` pair for an `is_identity` option, and one `_Window` memo for the neutral-padded window, the `sum`/`mean` total and the live-edge count, with the `has_valid` guard emitted only when some option is not `zero_on_empty` (`sum`/`mean`/`maxabs` are; `product`/`max`/`min` and unannotated custom kernels are not).
`_Window` also memoises one 0-d `float32` `constant` per distinct literal (evoxc does not CSE duplicate constants; op-count-only, ~5 µs per op).
The `equal`+`select` of a one-option dispatch is not removable: the `-1` identity id on the output rows (plus the documented out-of-range → identity / → `0.0` fallbacks) needs a predicate, so a one-option level is the kernel call plus two selector ops.

### `host_graph.py`
Host-only, pure Python (stdlib + `errors.GraphError`), never traced: `decode_graph(nodes, conns, node_attrs=1, conn_attrs=2)` → `(dict[int, tuple[float, ...]], set[tuple[int, int]])`; `topological_sort(nodes, conns)` → `(topo_order, topo_layers)` (Kahn, layers ascending, `GraphError` on a cycle); `find_useful_nodes(nodes, conns, output_idx)` → `set[int]`; `check_cycles(nodes, conns, from_idx, to_idx)` → `bool`; `node_depth(nodes, conns, input_idx)` → `dict[int, int]`, which must equal `Transformed.node_level` for a well-formed feedforward genome.
`common.__init__` re-exports `decode_graph`, `topological_sort` and `find_useful_nodes`; `host_graph.__all__` also lists `check_cycles` and `node_depth` (the in-graph mutation cycle test is `genome/operations/mutation/cycle.py::closes_cycle`).

### `sympy_tools.py`
Host-only (`re` + `sympy` + a `numpy` import deferred to the printer, no evoxc): `analysis_nodes_exprs`, `round_expr`, `replace_variable_names`, `to_latex_code`, `to_python_code` — the 8-tuple contract of `Genome.sympy_func` is unchanged from the pre-rewrite library.
`to_python_code` emits module-level source, never a `def`: `import numpy as np`, a `#     <symbol> -> i[<position>]` binding header for the caller-bound array `i`, `h = np.zeros(hidden_cnt)` (only when hidden nodes exist), `o = np.zeros(len(output_exprs) or output_cnt)`, one assignment line per useful node and a trailing `# o[k] = ...` closed-form block.
`_NumPySourcePrinter` (a `NumPyPrinter` subclass) is what makes that source genuinely NumPy-callable: `Max`/`Min` fold to right-nested `np.maximum`/`np.minimum` (never `functools.reduce`), `Abs` to `np.abs`, `Piecewise` to a right-nested eager `np.where` chain (a missing `True` branch nests on `np.nan`; eager evaluation is correct because every shipped branch is total), and any unregistered `sp.Function` by class name (`SympyClip` -> `np.clip`, a leading `Sympy` stripped) with `ConfigError` when NumPy has no such callable; the string is normalised `numpy.` -> `np.` because sympy keeps its known-function/known-constant tables on the class, so `exp`/`log`/`sin`/`tanh` still print with the `numpy.` prefix.
Every shipped ACT/AGG option renders executable source — `relu`, `lelu`, `inv`, `log`, `exp`, `abs`, the `max`/`min`/`maxabs` aggregations included — verified per option against `ACT.evoxc_fns`/`AGG.evoxc_fns` and against `DefaultGenome.forward`.
An independent audit execs all 18 per-option snippets and matches their kernel to ≤2.5e-7 absolute over inputs spanning negative/zero/positive (the whole [-1e3, 1e3] range for the other 17), the one divergence being `exp`: the emitted `np.clip(z, -10, 10)` (`SympyExp`) differs from the unclipped kernel by `e^-10`/`e^10` beyond ±10, exactly the divergence `functions/CONTEXT.md` records.
A `DefaultGenome` snippet (payload at `P = 1`, `exec`ed once per input row) matches `sympy_func`'s `forward_func` and the graph forward to the `precision=6` rounding of every weight — up to ~3e-5 absolute / 5e-4 relative on an `inv`-dominated genome, float32 round-off (~1e-7) when `round_expr` is bypassed — so the residual is rendering precision, not a codegen defect.
`round_expr` never collapses a real constant to zero: an atom that rounding would erase keeps its exact value, so `LOG_EPS`/`INV_EPS` (1e-7) survive as `1.0e-7` in the Python (`np.maximum(1.0e-7, ...)`) and LaTeX (`1.0 \cdot 10^{-7}`) rendering while `123456789/10**9` still renders `0.123457` at `precision=6`.
Not re-exported by `common.__init__` (its only consumer is `Genome.sympy_func`, which imports it directly).
`to_latex_code` and `to_python_code` take the 8-tuple **unpacked** (`to_latex_code(*genome.sympy_func(state, network))`): `symbols`/`topo_order` key by node **keys**, `nodes_exprs`/`output_exprs` by **sympy Symbols**, and `forward_func` is accepted for signature compatibility only.
`analysis_nodes_exprs` counts `input_cnt` over keys *starting with* `i`, and `genome.sympy_func` names only the raw input symbol `i<k>` that way (`norm<k>` takes the trailing `norm` branch), so `input_cnt == I` and both renderers' `topo_order[input_cnt:]` slice is correct for a standard genome whose inputs sort first.
`to_python_code` reads `input_symbols` (the binding header; empty means the `i<k>` symbols the emitted expressions reference, ordered by `k`) and `output_exprs` (output count/order and the closed-form block, one inlined line per output; empty means the `o<k>` key count and no block); `to_latex_code` reads neither.
That fallback header comes from the *pre-substitution* node/output expressions, so a hand-built 8-tuple whose node expressions carry only argument symbols gets no header although the body references `i[k]`; `Genome.sympy_func` always passes a non-empty `input_symbols`, so production never takes that path.
Both renderers are legacy-derived (`benchmarks/legacy_jax/tensorneat_legacy/common/sympy_tools.py`) and have one unit test, `test/unit/common/test_functions.py::test_to_python_code_emits_runnable_source_for_a_hand_built_expression`, which pins (`h = np.zeros(2)`, `o = np.zeros(1)`, `h[0]`, no `h[1]`, no `nan`, `0.123457`) and execs the source; per-option runnability and the `round_expr` guard are hand-verified only.

## evoxc idioms and gotchas (validated on this box)
- `@evoxc.defn` + `evoxc.build(fn, *TensorSpec, backend=..., device=...)`; the numpy reference interpreter is the default and is enough for `test/unit`.
- `evoxc.numpy as enp` for in-graph constants (`enp.zeros/ones/full/arange`), `evoxc.constant(evoxc.tensor(ndarray))` for host data; dtypes come from `evoxc.int32/float32/...`, **not** from `enp`.
- `gather(x, idx, axis)` / `scatter(x, idx, upd, axis)` are **not** batched `take_along_axis`: the index tensor replaces the axis, so per-row indexing needs a flattened source — `cpu`-side kernels go through `tools.flat_gather`/`tools.flat_scatter` (`flat = row * R + col`) and reshape back, so `P*R` must stay inside int32.
- `scatter` is set-only (no add/max modes) and `reductions` have no `where=`/`initial=`; masked reduction is done by padding with the neutral element (`0`, `-inf`, `+inf`, `1`) before `reduce_*`.
- `argsort` returns int64; its inverse permutation is a `flat_scatter` of `arange(n)`; `stable=True` is required for tie-breaking (group ranks).
- **`argsort(kind="stable")` has a radix fast path for `int8`/`int16` and a slow general path for `int32`/`int64`**: `(1000, 250)` costs 7.4 ms as `int32` but 1.0 ms as `int16` and 0.62 ms as `int8`. Anything sorted here must be keyed in the narrowest integer dtype that fits its range *including* the sentinel (unstable `kind` is not an option: the schedule needs in-group order).
- `evoxc.cond` accepts a `while_loop` in each branch on the numpy backend (both traced, one executed), so a "fast path + expensive fallback" decision is expressible in-graph.
- There is no `broadcast_to` on `evoxc` (use `enp.broadcast_to`), no `searchsorted`, no `choice`, no `bitcast`, no `where` (it is `evoxc.select(pred, on_true, on_false)`).
- Control flow (`evoxc.while_loop`/`cond`/`scan`) is traced into regions: bodies keep fixed shapes/dtypes, conditions are scalar tensors, and no `vmap` may be applied over them.
- `evoxc.slice(x, start, lengths)` takes per-axis tuples; `reshape`/`transpose` are the only other structural ops needed here.
- An `@evoxc.defn` object is not callable: use `.fn` (the tests bind it with `functools.partial`) or `evoxc.build` the `Defn` itself.

## Scatter portability on `backend="iree"` (measured, IREE 3.11.0)
**One index value feeds exactly one scatter.** That is the whole rule; everything below is why.
`evoxc`'s StableHLO writer emits a flat write (`tools.flat_scatter_prepared`, `tools.masked_flat_scatter`) as a rank-1 axis-0 *scalar-window* scatter: operand `tensor<Rxi32>`, indices `tensor<nx1xi64>` (`convert` from the int32 block, then range normalization), updates `tensor<nxi32>`.
IREE lowers it to `iree_linalg_ext.scatter dimension_map = [0]`, whose `ins` wants the index tensor **collapsed to rank 1** (`tensor<nxi32>`), and performs that collapse inside `iree-stablehlo-to-iree-input` as an `unrealized_conversion_cast` rather than a `tensor.collapse_shape`.
When the same index SSA value (an explicitly shared block *or* two expressions IREE's CSE merges) is consumed by a **second** scatter, that materialization stays live and the pass aborts: `failed to legalize unresolved materialization from ('tensor<nx1xi32>') to ('tensor<nxi32>') that remained live after conversion`, reported at the source location of the second scatter.
The emitted form is fine — one scatter with the same shapes compiles in ~0.25 s at any size — so this is contextual, not a bad op spelling.
Alternatives tried and rejected: the rank-2 axis-0 *row* form (operand `(R, 1)`, indices `(n, 1)`, `update_window_dims = [1]`) fails identically, and operand shape, width, i32-vs-i64 indices and `while`/`if`/`sort` regions are irrelevant (bisected by deleting ops — removing *either* shared-index scatter compiles the whole 7 164-line module, as does a distinct index expression).
Consequences for this node: the **host** formulation of the schedule needs exactly **two** scatters — `key_slot_table`'s dense table and `edge_order`'s inverse permutation — each through its own index expression, and the **accelerator** formulation needs none (it writes nothing, see `schedule_accelerator.py`); `incoming_table` deliberately writes nothing in either mode (it gathers the window out of the permutation, see below), and any caller that wants several payloads in the same cells must use one call over a buffer wide enough to hold both, never `flat_scatter_prepared` twice with one block.
On ROCm a scatter is also *slow*, not just fragile: it lowers to a single-workgroup, single-thread kernel costing ~16-21 ns per updated cell, while `gather` lowers to a parallel multi-workgroup kernel — the reason `incoming_table` derives the `(P, N, K)` window by reading the sort permutation instead of scattering `dst * K + col` cells (bit-identical `Transformed`, and ~1.6 ms of a 42.6 ms `xor_p500` generation removed).
Minimal reproducer (two scatters, one block): `P, R, N = 4, 68, 32`; `block = tools.prepare_flat_index(x, idx, trusted=True)`; `tools.flat_scatter_prepared(x, block, u1)` **and** `tools.flat_scatter_prepared(x, block, u2)` → `failed to legalize unresolved materialization from ('tensor<128x1xi64>') to ('tensor<128x1xi32>')`; with only the first call it compiles.
End-to-end on this box (`DefaultGenome(num_inputs=2, num_outputs=1, ..., output_transform=ACT.sigmoid)` + `NEAT` + `XOR`, seed 42): pop 4 / species 2 / `max_nodes=4` / `max_conns=8` compiles the step graph in 7.4 s and pop 100 / species 10 / `max_nodes=12` / `max_conns=24` in 8.4 s; `setup` (its own ask graph) compiles in ~0.9 s; three generations of both configs reproduce the numpy backend bit-for-bit.

## Constraints
- No `jax`/`torch`/`numpy.random` in graph code; randomness is `evoxc.random` stateless keys (`key(seed)`, `split`, `randint(key, shape, low, high)`, `uniform`, `normal`, `permutation`).
- Files ≤ ~1000 lines; shared math lives here, never duplicated in `genome`/`algorithm`.
- `tensorization.py` (975 lines) is one responsibility — derive the schedule — with the recipe, the contract and the measured-cost/rejected-lever record next to the kernels, because splitting it would separate the recipe from the code; condense prose before adding code.
- Inside this node one rule has one owner: `tools` owns `_positive_static` (every static in `batch_offsets`/`tensorization`), `_scalar_constant`, `_sentinel_constant` (the hoistable-sentinel resolver) and `to_numpy` (which `tensorization._host_array` converts through), imported and never re-written; `host_graph` keeps its own `_I_INF` literal on purpose, because `tools` pulls in `evoxc` and that module is deliberately backend-free.
- Nothing in this node imports from `genome`, `algorithm`, `problem`, `pipeline`, `benchmarks`, or `test`.
- Every public function carries a napoleon docstring stating shapes, dtypes, sentinels, and the invariances it relies on.
- Errors: validation raises `errors.TensorNEATError` subclasses on the host, at configuration/build time, before a graph is traced; in-graph overflow is reported as data (`edge_overflow`), never silenced.

## Known Gaps / Open Decisions
- Host-side graph walks use `host_graph.py`: `topological_sort`/`find_useful_nodes` are what `genome/default.py::network_dict` calls and `node_depth` what `algorithm/hyperneat` calls.
- The pre-rewrite helper names are gone and must not be re-introduced by consumers: `attach_with_inf` → `masked_flat_gather`; `hash_array` → `hash_rows` (host-only).
- **The v1 StableHLO op table is the portability gate.** `stablehlo.ops.status` reports `cumsum`, `erf`, `gelu`, `tril` and `flip` as `deferred` (numpy interpreter only — hence the `cc` counting trick instead of `cumsum`), and `scatter` is v1 but rejects a rank ≥ 3 operand, which is why every write goes through the rank-1/2 flat-index helpers; `argsort`/`sort`/`gather`/`select`/`while`/`if`/`pad`/`tile`/`dot`/`random_*` have no such caveat. With the signed-dtype rule that is the whole checklist: the `setup` and `step` graphs a real pipeline builds carry no unsigned dtype and no deferred op, and both lower on `backend="iree"`.
- `key_span S` is a static bound on key magnitude: keys must stay compact (hidden-node key compaction on mutation) or the table grows; `S` is part of the shape signature and rebuilds on overflow.
- `agg.maxabs` pays `tools.flat_gather`'s guard chain (`less`/`greater_equal`/`logical_or`/`select` plus a `(P, 1)` `arange` constant) even though an `argmax` result is provably inside `[0, K)`: `trusted=True` exists but `functions/` cannot bound `K` itself, and `maxabs` is not an option of the default genome, so the default XOR path never pays it (3.19 vs 2.34 ms per level at P=1000, measured in this node).
- The rest of `derive_schedule`'s cost is the `(P, N, K)` window gathers, the `(P, C)`/`(P, N)` selects/`add`s of the tail and the dispatch count — **647** executed ops on a `xor_p500` evolved payload (2026-11-05, `kernels.dispatch` monkeypatched; the pre-gather derivation was 664); a further 2× would need the window representation itself to change, which the `Transformed` contract fixes.
- The default genome's forward pass unrolls `max_depth = 16` level steps for every population, even a three-node network, so a per-level saving is multiplied by 16; unrolling on the real depth (or a level `while_loop`) is a genome-side change (`genome/` owns it).
- The step graph of a pop-1000 pipeline emits no large-constant warning: the largest baked constant is `edge_order`'s `(P, C + N)` `int32` zeros seed for the inverse permutation (600 000 B at `N=50 C=100 P=1000`), below evoxc's 1 048 576 B threshold, and the window's padding is a rank-0 `I_INF` constant broadcast by `select` rather than a population-shaped `enp.full`.
- `incoming_table` is the one place where numpy pays for ROCm: the two `(P, N, K)` gathers cost slightly more than the scatter they replaced (`incoming_table` 1.18 → 1.37 ms at pop 500, 2.22 → 2.45 at pop 1000, on a real evolved payload), cancelled by `edge_order` no longer deriving `rank`; net `derive_schedule` is ±0.15 ms and the end-to-end paired A/B is −0.4 %. See "numpy cost of the gather formulation" above for the protocol; do not quote the earlier single-payload figures for this pair.

## Known Issues
`test/unit/common/test_state.py::test_missing_entry_raises_key_error_and_escapes_hasattr` FAILS on this tree: it pins the defect that a missing `State` entry must escape `hasattr`/`getattr(state, name, default)` as a `KeyError`, which `StateKeyError(AttributeError, KeyError)` deliberately cannot do (see `### state.py`).
That test belongs to the `test` node (read-only from here) and needs: keep `pytest.raises(KeyError)` around the bare `state.typo` read, then assert `hasattr(state, "typo") is False`, `getattr(state, "typo", None) is None`, `isinstance(raised.value, AttributeError)`, and `pytest.raises(KeyError)` around `state["typo"]` with `not isinstance(..., AttributeError)`.
`genome.DefaultGenome.sympy_func` on a genome whose aggregation is `AGG.maxabs` is explosively slow: `functions/agg_sympy.py::SympyMaxabs` returns a `Piecewise` whose conditions are `Eq(Abs(term), Max(Abs(...)))` per node, and the closed-form substitution multiplies those branch sets layer by layer (≈27 s at 2 hidden layers, >110 s at 3, against 0.02 s at 1) — interpretability only, no generation or fitness path is affected.

## Public names with no production caller (audited at HEAD; tests + `__all__` only)
`StatefulBaseClass.show_config`/`_render_value` (the Pipeline renders through `host_payload._object_config`), `State.save`/`State.load` (checkpointing writes `config.txt`/`log.txt`/`*.npz`, never a pickle), `tools.hash_rows`, `tools.fetch_random` (the genome has its own `(row, hit)` primitive), `tools.mutate_float`/`mutate_int` (superseded by `genome/gene/blocks.py::mutate_attrs`), `host_graph.decode_graph`/`check_cycles`, and `FunctionManager.random_index`/`get_evoxc_func`/`index2name` (ids are drawn per attribute column with `evoxc.random.randint`, names come from `ACT.name_of`/`obtain_sympy`).
Not on that list: `tensorization.validate_schedule` has a source caller (`genome/grad.py::build_grad`).
They are exported on purpose (legacy parity, a standalone kernel surface), but none is exercised end-to-end by a shipped run — do not cite one as evidence that a path works.

## Notes for Agents
Every one of the 12 ACT and 6 AGG options is a real evoxc kernel with a validated sympy counterpart (`FunctionManager` checks the pairing at import) — no placeholder, no `NotImplementedError`, and `clip` is absent from both tables on purpose; the only surface a caller loses is `FunctionManager.add_func`/`update_sympy` (custom options are per-genome tuples now).
An empty population is *rejected*, not handled: `_positive_static` raises `ConfigError` for `pop_size == 0` in `derive_schedule`/`batch_offsets`, so `P >= 1` is the supported range and `P == 1` (the checkpoint/best-genome spelling) goes through every helper unchanged.
`test/unit/common/*` is the runnable kernel suite here, with `test/support/oracle.py` the numpy schedule oracle and `test/support/functions_ref.py` the ACT/AGG oracle; measured on this tree: `uv run pytest -o addopts="" -q test/unit/common` → 246 passed (numpy backend, no optional dependency) and `uv run pytest -o addopts="" -q -m parity` → 60 passed (needs `uv sync --extra bench` for the jax oracle).
A performance change here still needs a field-by-field diff against the implementation it replaces (feed-forward, cyclic, in-degree > `K`, absent nodes, unmatched keys, both key spellings, both `acyclic` settings), because the suites do not pin bit-exactness of every `Transformed` field.
`import tensorneat` works from an installed checkout (`uv pip install --python .venv/bin/python -e . --no-deps`) and pulls in no `jax`/`torch`/`brax`/`gymnax`/`mujoco`.
`test/support/oracle.py::schedule` rebuilds the shipped `Transformed` field set (`node_level`, `node_exists`, `in_valid` included) from an independent numpy derivation, so it is the field-by-field reference a schedule change is diffed against.
`evoxc` in a bare checkout: `uv venv .venv && uv pip install --python .venv/bin/python evoxc numpy sympy networkx matplotlib pytest && uv pip install --python .venv/bin/python -e . --no-deps`; `pytest` finds `src/` and `test/support` through `test/conftest.py`, so a run is `.venv/bin/python -m pytest test/unit/common`.
Timing anything here: build once and time `evoxc.run`, one cell per fresh process (a second executable in the same process skews later cells up to 25 %), and report the ratio against the previous implementation — absolute milliseconds move ~30 % between sessions on this box.
