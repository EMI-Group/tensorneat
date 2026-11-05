# distance — compatibility distance: pair-wise, digest merge, index-free resolution, dense single-centre, grouped k-centre

## Intent
Answer one question per parent pair, batch-first: how different are these two genomes?
The result feeds speciation, so it must be symmetric, exactly `0` at identity, `0` (never NaN) for an empty pair, and normalised by the larger gene count so pairs of different sizes stay comparable.
Speciation measures centre genomes against the whole population twice per generation, so the population side is packed and sorted **once** (the digest) and only the centre side is per-call work.
This node offers the same arithmetic in five shapes, all ending in one scalar tail (`DefaultDistance._scalar_distance`), which is what makes them bit-identical rather than merely close:
- the pair-wise `__call__` (contract unchanged; used by parity tests and by callers holding two explicit parents);
- the two-phase `prepare` / `measure` **merge**, whose population half is prepared once per generation;
- the two-phase `prepare` / `measure` **index-free** route (`index_free.py`), which resolves each centre gene against the digest with an `equal`/`reduce_min` table instead of two sorts — same `AlignedGenes` record, same tail, selected by `state.distance_formulation == "index_free"` (see `## Formulation seam`);
- inside `measure`, a **dense key-table fast path for a batch-1 centre** (`dense.py`), resolving the alignment with gathers instead of a sort;
- a **grouped `k`-centre** call against `group_digest(digest, k)`: `k * P` centre rows measured in one call, bit-identical (`tobytes()`) to `k` separate `P`-row calls — verified but wired into no production path (see `## Grouped route`).

## API Surface
`BaseDistance.__call__(state, genome, node_keys1, node_attrs1, conn_keys1, conn_attrs1, node_keys2, node_attrs2, conn_keys2, conn_attrs2, *, consts=None) -> (P,) float32` — validates both parents field by field (matching shapes, key rank 2 or 3, attribute rank 3, keys and attributes agreeing on `(P, M)`; a violation raises `ConfigError`) and returns `gene_distance(node family) + gene_distance(conn family)`.
`BaseDistance.prepare(state, genome, node_keys, node_attrs, conn_keys, conn_attrs, *, consts=None) -> DistanceDigest` — one pass over the population, returns the reusable digest.
`BaseDistance.measure(state, genome, centre..., digest, *, consts=None) -> (R,) float32` — one or more centre genomes against every digest row; the centre batch is `1` (broadcast, dense fast path when enabled) or `P` on a plain digest, and exactly `group * P` on a grouped one.
`grouped.group_digest(digest, group) -> GroupedDigest` — tiles a plain digest `group` times (inside the active trace, once per digest a caller reuses); `group` must be a positive `int` (`ConfigError` naming it otherwise).
`grouped.GroupedDigest` — `(node_keys, node_attrs, node_count, node_key_cols, conn_keys, conn_attrs, conn_count, conn_key_cols, node_statics, conn_statics, pop_size, group)`; the ten tensor/statics field names are `DistanceDigest`'s, so `measure`'s merge body reads either digest type unchanged, and `pop_size`/`group` are static ints (trace-local metadata, never a graph-boundary value).
`grouped.validate_centre(name, payload, digest) -> rows` and `grouped.validate_family(...)` are the host-only shape checks of the grouped route — the plain route calls the same `validate_family` from `base._validate_digest`, so the two wordings cannot drift.
`BaseDistance._single_centre_enabled(genome) -> bool` is the host-side twin of the hook's static gate: `False` in the base class, `dense_single_centre and dense.enabled(node gene) and dense.enabled(conn gene)` in `DefaultDistance`.
`BaseDistance._measure_single_centre(pool, state, genome, payload, digest)` is the hook: `None` in the base class, the dense route in `DefaultDistance` while `dense_single_centre` is `True` **and** both gene families pass the static gate.
`BaseDistance._merge_both(pool, state, genome, payload, digest)` is the one spelling of the row-wise route (both families, one `evoxc.add`), shared by the plain and grouped paths.
`index_free.formulation_mode(state) -> str` reads `state.distance_formulation` with a host `getattr` (default `"merge"` when the entry is missing, exactly like a bare `State()`) and validates it against `DISTANCE_FORMULATION_MODES = ("merge", "index_free")`, raising `TensorNEATError` naming the value and both accepted spellings; it emits no op.
`index_free.selected(state) -> bool` is `formulation_mode(state) == "index_free"` and is the only switch `measure` consults.
`index_free.resolve(gene, keys, digest_keys, digest_key_cols, *, statics, consts=None) -> AlignedGenes` is the index-free twin of `digest.measure_alignment` (same arguments, same record); `index_free.alignment(state, gene, keys, digest_keys, digest_key_cols, *, statics, consts=None)` is the host branch `_measure_family` calls — `resolve` when `selected(state)`, `digest.measure_alignment` otherwise — so the route choice is one `if` on a host string and never a traced value.
`BaseDistance.gene_measure(...)` and the private `_measure_family(pool, ...)` are the per-family drivers: `digest.measure_alignment` then `distance_from_alignment`. `gene_measure` is public with **no in-library caller today** (it starts its own `ConstantPool`).
`DefaultDistance(compatibility_disjoint=1.0, compatibility_weight=0.4, dense_single_centre=True)` — both weights finite and non-negative (`ConfigError` at construction); `dense_single_centre=False` forces the row-wise merge everywhere.
`DefaultDistance._scalar_distance(cnt1, matched, cnt2, attribute_distance, pool)` is the single scalar tail: `(non_homologous * compatibility_disjoint + attribute_distance * compatibility_weight) / max(cnt1, max(cnt2, 1))`, the denominator floored at `1.0` so an empty pair is `0.0` with no second `select`.
`DistanceDigest(...)` (19 fields, node family first) and `FamilyDigest(...)` (9 fields) live in `digest.py`; `field_names` reports `_fields`, and `test/unit/genome/test_distance_digest.py` pins that literal order, so a new digest concern belongs in a **new** class (`GroupedDigest`), never a field.
`digest.prepare_family(state, gene, keys, attrs, *, key_span, consts=None, dense=False, tail=None)`, `digest.measure_alignment(gene, keys, digest_keys, digest_key_cols, *, statics=None, consts=None) -> AlignedGenes` and `digest.match_rows(...)` are the per-family entry points; `layout.dense_width`/`dense_slot`/`dense_rows` are the table geometry.
`digest.match_rows` and `layout.dense_slot` are kept-public entry points with **no in-library caller today** (`__all__` members documenting the alternatives `measure_alignment`/`dense_rows`, which the live route consumes).
`dense.enabled(gene, genome) -> bool` and `dense.measure_centre(pool, state, genome, distance, payload, digest) -> (P,) float32 | None` are the fast path, whose per-generation statics travel in `digest.<family>_layout` (`DenseLayout`) and whose `cnt2` pair travels in `digest.tail_counts (2, P) int32`.
`AlignedGenes(rows, homologous, count, matched)` is the merge's result: `rows (R, M) int32` redirected to `0` where a gene has no homologue, `homologous (R, M) bool`, `count (R,)` the centre's real genes, `matched (R,)` its homologue count.
`consts` is an optional `ConstantPool`; pass one per top-level kernel entry so identical rank-0 constants cost one op instead of one per use site.
`prepare` and `measure` are **plain traced helpers**, never `@evoxc.defn` (a defn cannot call a defn), so the caller's graph owns them; a digest is a pytree and may be captured in a closure across a trace.

## Merge algebra (`digest.measure_alignment`)
A gene row folds into one `int64` composite, positionally and most significant first in base `_KEY_BASE = 2**31`: a node key is itself, an `(in_key, out_key)` connection key is `in * 2**31 + out`.
`2**31` is the whole non-negative `int32` key range, so the fold is injective for **every** legal key, and it never consults `key_span`.
`key_span` (`S`) bounds only the dense table (and the execution schedule); it must never be used as an injectivity bound.
Digits are not gated per column: `2**31` bounds every legal digit, so no fold can overflow, and an absent row needs no per-digit special case because the payload contract makes *every* key column `I_INF`, which folds to the side's padding sentinel by construction.
The fold is unrolled as `multiply`/`add` steps rather than a weighted `reduce_sum` over the key axis: an innermost-axis integer reduction costs an order of magnitude more than the elementwise pair it replaces.
The **centre side of a multi-column key family folds with a single `dot`**: `_pack_centre` contracts the key axis of `cast(keys, int64) (R, M, 2)` against `fold_i64 (2, 1)` `[2**32, 2]`, so `k0 * 2**32 + k1 * 2` — exactly the positional fold with the tag scale applied — lands in 4 traced ops (cast, dot, reshape, `not_equal`) instead of 11.
The cast in front of that `dot` **stays**: a mixed-dtype `dot` (`int32 × int64`) is accepted by the trace and by the numpy interpreter but is **not exportable** — the StableHLO writer forwards `dot` operand dtypes verbatim (`backends/stablehlo/writer.py::_emit_dot`) and emits an invalid `stablehlo.dot_general`, with no `lower()`-time error, so only an actual export reveals it.
`multiply` is the one op where mixed dtypes are safe: the writer equalises elementwise operands with a `stablehlo.convert`, which is why the node family's single key column folds as `multiply(int32_keys, tag_scale_i64)` in **one** op (see `## Cost`).
Two folded columns are all an `int64` composite can hold (`2 * 31 = 62` bits), so a gene with more columns (`OriginConn`'s `historical_marker`) folds its first two (one `slice` to `_prefix`) and **verifies** the rest row for row (`_key_row_matches` against `digest_key_cols`), keeping homology equal to full-tuple key-row equality.
The low bit of the sort key is the side tag (population `1`, centre `0`); each side's padding rows collapse onto their own sentinel: `_PAD_CENTRE` — the fold of an all-`I_INF` row — and `_PAD_POPULATION`.
Defining the centre sentinel *as* the fold of an absent row is what lets the `dot` path skip its `select`: an absent row already lands on `_PAD_CENTRE`, and the mask is read off the folded value.
It must not be `int64` max: max equals the next composite after a padding row and would manufacture a false homologue at the last merged position.
Injectivity plus the tag make a homologous pair exactly an adjacent `(centre, population)` pair whose composites differ by one; the parity is what proves the neighbour is a real population gene, so no separate tag test is needed.
The merge concatenates the **digest side first**, the centre second and one **trailing column** of `_PAD_CENTRE` third: the merged index of a population gene *is* its digest row, the tail reads its row table straight off the successor, and every centre gene has a successor position in range.
Sequence per family: `concatenate` to `(R, 2M + 1)`, one **stable** `argsort`, its inverse via a second **stable** `argsort` of the permutation, then two centre-width gathers (the successor's merged index and the composite it carries).
Nothing later is `2M`-wide: the homology test, the row table and both counts stay at the centre's own width `M`.
The inverse permutation is the measured formulation: `argsort(order)` is two ops while scattering the ranks costs four to eight, and the stable `argsort` is also 1.05 → 0.29 ms at pop 1000.
The trailing column is why the successor gather needs **no** `minimum` clamp and no in-range mask: `_PAD_CENTRE` is the merged row's maximum, so no centre gene sits at the last position.
`evoxc.slice(x, start, limit)` takes **lengths**, not end indices, which is why the centre slice is `((0, count), (pop_size, count))`.

## Formulation seam: merge versus index-free (`index_free.py`)
`state.distance_formulation` picks how a centre gene finds its homologue in the already-prepared digest: `"merge"` (the shipped sort-based `digest.measure_alignment`) or `"index_free"` (an `equal` table, no `argsort` at all).
`Pipeline.setup` registers the pair `distance_centre`/`distance_formulation` from the resolved device kind (`host_payload._formulation_statics`): a GPU-class kind gets `"tiled"`/`"index_free"`, every other kind and `None` get `"single"`/`"merge"`, and the pair is part of the executable-cache signature, so changing it forces exactly one rebuild.
The kernels default a **missing** entry to `"merge"` — a bare `State()` yields the legacy contract, which is what every direct kernel call and every test outside the pipeline sees.
The read is a host `getattr` on `state` (the `algorithm/neat/_validation.py::distance_centre_mode` pattern for the sibling static), never a graph-boundary value: the mode reaches the graph through the closure, and an unknown value raises `TensorNEATError` naming it and both accepted spellings (no silent fallback).
`measure` also **skips the dense single-centre branch** when `index_free.selected(state)` holds: that route is scatter-based and a device-time regression on the accelerator target, and the formulation switch must never take it.
The index-free body, per gene family, produces the same three quantities the merge produces and then reuses the same tail, which is what makes the routes bit-identical by construction rather than by measurement:
- `centre` composites carry the **population** tag (`pack * 2 + 1`): rank-2 keys fold with `_pack(..., scale=tag_scale_i64)` + `_tagged(..., _TAG_POPULATION, pad_centre_i64, None, one_i64)` in 4 ops, rank-3 keys with width `>= _FOLDED_KEYS` fold with `_pack_centre` + `select(valid, add(composite, one_i64), pad_centre_i64)`.
  An absent centre row therefore carries `_PAD_CENTRE` (even), which equals no population composite: real composites of both sides are **odd** (`pack * 2 + 1`), while the two padding sentinels are even and two apart, so the population side needs **no** presence mask and `prepare`/`build_statics` gained no constant, no table and no op.
- `table = equal(centre[:, :, None], digest_keys[:, None, :])` → `(R, M, M) bool` (evoxc's `equal` broadcasts), then the first matching population column and the "has a homologue" mask come from one `select` + `reduce_min` over that table — **never from `argmax`/`iota`**, because an operand-less `stablehlo.iota` dispatch does not compile (see `## Constraints` and `## Portability census`):
  `columns = enp.arange(0, count, dtype=int32)`, `no_match = pool.i32(count)`, `col = reduce_min(select(table, columns, no_match), axes=2)`, `homologous = less(col, no_match)`.
  `col` is therefore in range wherever the mask is true and redirected to `statics.zero_i32` anyway, so the tail's `select` never reads a phantom column.
- a key width past `_FOLDED_KEYS` ANDs in exactly the merge's confirmation, `digest._key_row_matches(keys, digest_key_cols, col, homologous, width, statics, pool)`, so homology is full-tuple equality and never the folded prefix alone.
- `rows = select(homologous, col, statics.zero_i32)`; the record is `AlignedGenes(rows, homologous, reduce_sum(valid, axes=1), reduce_sum(homologous, axes=1))` — same shapes, same dtypes (`count`/`matched` are int64, as `reduce_sum` on a bool returns), so `distance_from_alignment` → `_scalar_distance` consumes it unchanged.
Everything is row-local, so a `GroupedDigest` (tiled rows) and a broadcast batch-1 centre keep working with no grouped-specific code path.
Measured on this tree (numpy reference interpreter, seed-42 XOR payload, `DefaultGenome(2→1, max_nodes=50, max_conns=100)`):

| cell | merge | index_free |
| --- | --- | --- |
| traced ops, one `measure` call at a `P`-row centre (measure-only graph, digest at the boundary) | **94** | **85** |
| traced ops, one `measure` call at a batch-1 centre (the live CPU shape; merge = dense route) | **109** | **85** |
| of which `argsort` / `gather` / `constant`, `P`-row body | 4 / 6 / 0 | 0 / 2 / 0 |
| `prepare` ops / `constant` ops (unchanged by the switch) | 163 / 48 | 163 / 48 |
| standalone prepare+aligner, node family / conn family | 183 / 184 | 176 / 178 |
| ms per call, pop 500, tiled centre (8 rounds × 15 reps, interleaved A/B, build excluded) | **0.531** | 1.352 |
| ms per call, pop 500, batch-1 centre | **0.686** | 1.487 |
| ms per call, pop 1000, tiled centre | **1.368** | 2.920 |
| ms per call, pop 1000, batch-1 centre | **1.399** | 2.975 |

The `P`-row body's op count is the figure the merge's own table reports (93 op nodes / 94 with terminators, same counting rule); the index-free body is **85** ops after the `argmax` → `reduce_min(select(...))` rewrite, i.e. **80 + 5**: the five ops the portability fix costs are one `arange` `(M,)` constant-free column index, one `select` over the `(R, M, M)` table, the `reduce_min` (which replaced the `argmax`), the `less` that reads the mask back off the column, and the `constant`-pool slot for `no_match` (`argmax` was one op and emitted 23 ops of count-mode sort machinery on a compiled backend, so the swap is a clear net win there).
So the index-free route is **−9 ops and −4 argsorts** per `P`-row call but **2.1–2.6× the wall clock** on the numpy interpreter: the `(R, M, M)` `equal`/`select`/`reduce_min` chain is an `M²` element pass (5 M cells for the conn family at pop 500, three passes over it) where the merge sorts `2M + 1` columns, and the interpreter charges by element volume as much as by op count.
That is the intended trade, not a regression to fix: the route exists for a dispatch-bound backend, where the per-op charge dominates and the count-mode `argsort` fan is the whole cost — see `## Portability census` for what each body actually dispatches.
**Duplicate key rows inside one genome are the one measurable divergence**, and they are outside the payload contract (`initialize` and `execute_mutation` keep key rows unique, homology is by key tuple): with a centre carrying key `7` three times against one digest row, the merge's successor test marks only the **last** duplicate homologous (`homologous [T,T,F,F,T,F]`, `matched 3`, rows `[0,1,0,0,2,0]`, distance 15.74) while the index-free table marks **every** duplicate (`[T,T,T,T,T,F]`, `matched 5`, all pointing at the first matching column 2, distance 15.34).
Duplicate rows on the **population** side behave identically in both routes.
Matching the merge exactly would cost an extra per-call op on the hot path, so the divergence is documented rather than chased and `measure_alignment`/`prepare_family` were not changed to accommodate it.

## Portability census: index-free vs merge (measured 2026-11-05, IREE 3.13.0rc20261016)
Protocol: numpy backend, `DefaultGenome(num_inputs=2, num_outputs=1, max_nodes=50, max_conns=100, output_transform=ACT.sigmoid)` + `NEAT(pop_size=500, species_size=20, survival_threshold=0.01)` + `XOR()`, seed 42, `Pipeline.setup()` + 3 `step`s, then either (a) the **full generation graph** the pipeline builds (captured by monkeypatching `evoxc.build`), or (b) a **measure-only graph** — `dist.measure(state, genome, *centre, digest)` traced with the digest as graph-boundary tensors, so `prepare` is outside it and the module *is* the per-call body.
Both are exported to StableHLO with `evoxc.lower(graph, backend="iree").payload["mlir_text"]` and compiled with `uv run iree-compile <mlir> --iree-input-type=stablehlo --iree-hal-target-backends=<be> --compile-to=executable-configurations` (`llvm-cpu`, and `rocm --iree-rocm-target=gfx942`).
`distance_formulation="index_free"` is set on the state after `setup()` (`state = state.update(distance_formulation="index_free")`) with a fresh `Pipeline`, because the pair is part of the executable-cache signature.

### The blocker this pass found and fixed (`index_free.py`)
The first export of the index-free route **compiled on neither target**. `iree-compile merge/index_free_gen.mlir` exited 1 with:
```
index_free_gen.mlir:4761:13: error: 'stream.async.dispatch' op : streamable op expected to be in an execution region
index_free_gen.mlir:4761:13: error: 'stream.async.dispatch' op illegal for this phase of lowering in the stream dialect; expected to have been converted or removed
  ...pass statistics: iree-stream-verify-lowering-to-async
index_free_gen.mlir:4869:13: ... (same pair of errors)
```
Both sites are `stream.async.dispatch` ops carrying a **bare `stablehlo.iota`** as their body — `%4710 = stablehlo.iota dim = 2 : tensor<500x50x50xi64>` (`@main_dispatch_280_elementwise_broadcast_1250000_i64`) and `%4816 = stablehlo.iota dim = 2 : tensor<500x100x100xi64>` (`@main_dispatch_291_elementwise_broadcast_5000000_i64`), the `argmax(...)` the body used to take over the `(P, M, M)` table.
The `iota` has no operands, so the dispatch is dropped into the stream dialect without an execution region and the verifier rejects it.
So a **`(P, M, M)` `argmax` is not a portable op in this tree**, even though `ops.status("argmax")` reports `v1` and `lower()` sees nothing wrong — only an actual `iree-compile` reveals it.
The smallest correct fix, and the one this node ships: keep the comparison table, replace the index pick with an **operand-bearing** reduce —
`columns = enp.arange(0, count, dtype=vertex int32)`, `no_match = count`, `col = reduce_min(select(table, columns, no_match), axes=2)`, `homologous = less(col, no_match)` — which lowers to `select` + `reduce_min` dispatches over the table and emits no `iota` at all.
Both targets then exit 0, and the difference is measurable in the census below (`reduce_min` ×9 and `dot` ×5 in the generation graph, `argmax` untouched at 13 — those belong to `algorithm/`'s speciation scans and are present in the merge graph too, so they are outside this node but on the same shared graph and would block a fully compiled generation graph for either formulation; they are **not** introduced here).
Everything else the route emits is v1-portable: no unsigned dtype, no deferred op, every `scatter` rank ≤ 2 with an index expression of its own, and no `sort`/`argsort` dispatch at any centre width.

### Full generation graph
| cell | merge | index_free |
| --- | --- | --- |
| StableHLO text | 10 001 618 B, 1 915 op uses, 12 distinct op names | 9 983 042 B, 1 843 op uses, 12 distinct op names |
| `stablehlo.sort` / `dot_general` / `scatter` / `gather` uses | 5 / 1 / 42 / 129 | 5 / 5 / 26 / 129 |
| unsigned dtypes | 0 | 0 |
| deferred ops | 0 | 0 |
| `iree-compile` llvm-cpu | exit 0, **4.47 s** | exit 0, **4.75 s** |
| `iree-compile` rocm gfx942 | exit 0, **5.19 s** | exit 0, **5.03 s** |
| exports / unique symbols / unique dispatch base names / dispatch call sites (rocm text) | **330 / 330 / 206 / 458** | **322 / 322 / 198 / 431** |
| single-workgroup dispatches (`pipeline<Distribute>`, `workgroup_size=[1,1,1]`) | 89 | 83 |
The 5 `sort` uses are `algorithm/`'s crossover/digest sorts, not a per-call body's; the index-free route adds 4 more `dot_general` uses (the `_pack_centre` matvec per family per call site) and removes 16 `scatter` uses.
The traced-op-name gate over the same two graphs (`evoxc.trace`, then `evoxc.backends.stablehlo.ops.status(<evoxc op name>)` per op): merge **2 260 traced ops / 51 distinct names**, index_free **2 180 / 51**, **no deferred op in either** (the only unmapped name is `return`, the block terminator).
The per-name delta is `reduce_min ×9` (from 1), `dot ×5` (from 1), `scatter ×26` (from 42), `reshape ×351` (from 411).
### Per-call body, four cells
The live pipeline on CPU registers `distance_centre="single"`, so the speciation pass hands `measure` a **batch-1** centre: instrumenting `prepare`/`measure`/`_measure_family` over one `xor_p500` generation (gen 0, where the graph is traced) counts **1 `prepare` + 4 `measure` calls**, every one at `centre_rows=1` — not the ~40 calls the `distance_centre="tiled"` path records.
The merge takes the dense single-centre branch there (0 `_measure_family` calls); index-free skips that branch by contract and runs 8 `_measure_family` calls (2 per `measure`).
So both centre widths are reported, because they are different bodies:

| cell | exports / symbols / base names / call sites | single-wg (rocm) | sorts | count-mode argsort families | scatters | matvec | largest block |
| --- | --- | --- | --- | --- | --- | --- | --- |
| merge, batch-1 centre (dense route, live CPU shape) | 25 / 25 / 22 / 25 | **15** | 0 | 0 | 4 (`scatter_152x5xf32`, `scatter_51xf32`, `scatter_22502x2xf32`, `scatter_101xf32`) | 0 | `(22502, 2)` f32 = 45 004 elem |
| index_free, batch-1 centre | **13 / 13 / 12 / 13** | 6 | 0 | 0 | **0** | 1 (`matvec_100x2_i32xi64xi64`) | `(22502, 2)` / `(152, 5)` digest gathers, no wide intermediate |
| merge, `P`-row centre (sort route) | 35 / 35 / 25 / 36 | 6 | 0 | **8** (`reduction_500x101x101_{i64,i32xi32xi64,i32xi32xi64xi64xi64}`, the same three at `500x201x201`, `reduction_500x50x101_i64`, `reduction_500x100x201_i64`) | 0 | 1 (`matvec_50000x2_i32xi64xi64`) | `(500, 201)` int64 = 100 500 elem (804 KB) |
| index_free, `P`-row centre | **12 / 12 / 12 / 13** | 0 | 0 | 2 `(P,M,M)` reduce families (`reduction_500x50x50_i64xi64xi32xi32`, `reduction_500x100x100_i64xi64xi32xi32`) | 0 | 1 | `(500, 100)`-wide gather / `(50000, 2)` matvec |
Totals of the **live (batch-1)** per-call body: merge dark launches 25 dispatches with 15 of them serialised into a single workgroup (three `slow_memcpy` at `[128,1,1]`, the four `scatter` stores, and eight rank-0/small reductions), index-free dark launches **13** with 6 single-workgroup (its two `matvec`-free rank-0 reductions and its narrow gathers) and **no store into a wide table** — i.e. **−12 dispatches, −9 single-workgroup dispatches, 0 scatters**, at the price of one `matvec` per family.
At the `P`-row width the comparison is starker: merge 36 call sites with 8 count-mode `argsort` reduction families (each of which expands to 8 `(P,M,M)` blocks plus 4 wide reduces on a compiled backend) and a 804 KB `int64` block; index-free 13 call sites with **one** `(P,M,M)` reduce family per gene family and no sort at all.
The two `(P,M,M)` families in the index-free `P`-row body are the new `select`+`reduce_min` chain, not sort machinery — one dispatch per family (`M = 50` node, `M = 100` conn) where the merge's 8 families are 4 sorts and 4 reductions apiece.
### Element-volume arithmetic (per `measure` call, `R = 500`, `M = 100` conn / `50` node, `E = 2M + 1` = 201 / 101)
- **merge, `P`-row body**: two `argsort`s per family over the `(R, E)` merged composite, each expanded by the count-mode emission into `≈ 8 · E · R · E` element touches — node `8 × 101 × 500 × 101 ≈ 40.8 M`, conn `8 × 201 × 500 × 201 ≈ 161.6 M`, so `2 × (40.8 + 161.6) ≈ 405 M` element touches per call, on `int64`/`int32` operands.
- **index-free body**: one `(R, M, M)` equality table per family, `R · M · M` = `500 × 50 × 50 = 1.25 M` (node) and `500 × 100 × 100 = 5.0 M` (conn) bools, then a `select` and a `reduce_min` over the same extent: `3 × 6.25 M ≈ 18.8 M` element touches per call, on narrower dtypes.
  Ratio: **≈ 21.5×** fewer element touches per call, and **0** sort dispatches instead of 16.
- Per generation (4 `measure` calls on the live CPU path, ~40 on the `"tiled"` path): merge `≈ 1.6 G` / `≈ 16 G`, index-free `≈ 75 M` / `≈ 750 M` element touches.
That comparison is exactly why the route is switched on by device kind: on the numpy interpreter the `(R, M, M)` table costs 2.1–2.6× the wall clock of the sort (`## Formulation seam`), while on a dispatch-bound target the merge pays 8 reduction families per family per call where index-free pays one.
### Live-route note
The batch-1 merge body above is the **dense** route, so on CPU the index-free switch replaces a scatter-based dense body (4 scatters, 25 dispatches) with a scatter-free 13-dispatch body — not with the sort route the CPU default uses at a `P`-row centre.
Both formulations return bit-identical fitness at both centre widths: on `xor_p500` (P = 100/500) three generations of fitness history and payloads are `tobytes()`-equal between the formulations, and the standalone alignment agrees with `execute_distance` to `max |Δ| = 0.0` over 200/200 non-zero distances, for `DefaultNode`+`DefaultConn` (2 key columns) and `OriginNode`+`OriginConn` (3 columns, the marker-confirmation branch).

## Traced op census (static, `xor_p500` generation 0, `evoxc.trace` of the generation kernel)
`measure_distance` on the merge route traces to **exactly 6 `reduce_sum` and 4 `argsort` per call** (no `reduce_max`, no `scatter`, no `constant`), and `prepare_distance` to 4 `reduce_sum`, 2 `argsort` and 50 `constant`s.
On the live CPU path the speciation pass makes **4 `measure` calls**, so a generation executes **18 reduces** (4 from `prepare` + 4 × 2 counted reduces + the attribute terms' reduces) and **the `4 × argsort` of nothing**: the batch-1 centre takes the dense route there, which emits 4 scatters and no sort.
On the `distance_centre="tiled"` path the call count is ~40 and the sort/reduce counts scale with it; that path's numbers are in `## Compiled-backend dispatch census` below.
The 6 reduces of a merge call are, per gene family, `cnt1 = reduce_sum(valid)` (`digest.py:760`) and `matched = reduce_sum(homologous)` (`:761`), both `(R, M) bool → (R,) int64` at `axes=(1,)`, plus the attribute term `reduce_sum(select(homologous, |Δattr|, 0))` (`default.py:388`) as `(R, M) float32 → (R,) float32`; `cnt2` rides in `digest.tail_counts` and costs no reduce.
Their width is the **gene axis `M`, never the merge width**, so an accelerator profile's `reduction_500x201x201_*` / `reduction_500x101x101_*` families are **not** these `reduce_sum`s (they are the count-mode `argsort` expansion).
The index-free body replaces the 4 `argsort`s with `equal` + `select` + `reduce_min` + `less` per family (`reduce_min` over the `(R, M, M)` table, `axes=2`), and keeps the same 6 `reduce_sum`s, so the record is value-identical and op-lighter.
Casting the two bool reduces' operand to `int32` before the sum is value-identical (counts ≤ `max_nodes`/`max_conns`) and halves their element bytes on a backend whose int64 reductions dominate.

## Sort semantics (`stable=True` is a performance rule, not a tie rule)
Every `argsort` in the merge and the digest passes `stable=True`, and on the reference interpreter that is where the time goes: numpy's stable kind is timsort, which detects the runs the data already contains.
The digest side of the merge is ascending by construction, so the merged sort sees one sorted run of `M` beside the centre's `M` shuffled values: **1.43 → 0.28 ms** per `(1000, 200)` `int64` block at `M = 100`, and the inverse permutation **1.05 → 0.29 ms**.
`prepare_family` gains for the same reason: 1.51 → 1.25 ms (node) and 1.43 → 1.22 ms (conn) at `P = 1000`.
This is safe rather than lucky because none of these sorts has a *meaningful* tie: a homologous pair differs by one, real composites are unique inside a genome, and the only equal values are padding rows, whose attributes are `NaN` on both sides and which can never pass the homology test.
Stability also pins tie order to input order and matches a compiled backend's count-mode `argsort`.
Sort cost is **superlinear in row width, not in element count**: one `(500, 301)` `int64` `argsort` is dearer than `(500, 101)` + `(500, 201)` together, which is why fusing the two families' merges loses and why the grouped route is a wash in milliseconds.
The index-free route is the only body here that emits **no** sort at either centre width, which is what it exists for.

## Compiled-backend dispatch census (`xor_p500`, `--iree-hal-target-backends=rocm --iree-rocm-target=gfx942`, IREE 3.13.0)
The merge generation graph compiles to **330 `hal.executable.export`s / 330 unique symbols / 206 unique dispatch base names / 458 dispatch call sites** (89 single-workgroup) and the index-free one to **322 / 322 / 198 / 431** (83 single-workgroup) — see `## Portability census` for the per-body split and the text gates.
On this node's shipped CPU path (`distance_centre="single"`) the per-call body is the **dense** one (22–25 dispatches, 4 scatters, no sort), so the count-mode `argsort` fan below belongs to the `"tiled"` centre path (`distance_centre="tiled"`, the GPU default) and to `prepare`.
Every count-mode argsort on `(P, M)` emits 8 `(P,M,M)` blocks and 4 wide reduces — two `add dims=[0]` over `(M,P,M)` → `(P,M)` (the stable rank) and two `minimum dims=[2]` over `(P,M,M)` → `(P,M)` (the inverse permutation of the matched index, init `M`, and of the key column, init `INT64_MAX`) — plus one `(P, M, 2M+1)` successor fold, and `digest.py:720`'s int32 `argsort(order)` repeats all four.
At `P=500` these land as `reduction_500x101x101_*` and `reduction_500x201x201_*` (3 `_i64` + 1 `_i32xi32xi64` + 1 `_i32xi32xi64xi64xi64` each), `reduction_500x50x101_i64` / `reduction_500x100x201_i64` (the successor fold) and, once per generation from `prepare`, `reduction_500x50x50_i64` / `reduction_500x100x100_i64` (3 each); all are `iree_gpu.pipeline<TileAndFuse>` with `workgroup_size = [64,1,1]`, i.e. the reduction axis is tiled across a 64-thread workgroup and **not** serialised into one workgroup.
Root cause is `evoxc`'s count-mode sort emission, and **it cannot be switched off from a call site**: `backends/stablehlo/writer.py::_count_argsort_active` returns True whenever `_region_depth > 0` **or** `_select_inline_depth > 0`, whatever `sort_emission` says, and `_count_sort_active` applies the same rule to the values-sort path ("one rule for every sort emission site"); only a flat top-level block follows the mode.
Every live `measure` call except the caller's `scan` iteration 0 is emitted **inside the `evoxc.scan` body**, so `sort_emission="pair"` reaches at most 1 of 20 calls, and the region rule is a **correctness** workaround rather than an optimisation: iree 3.11.0 mis-executes a 2-operand `stablehlo.sort` whose index operand is not a region argument (a loop-carried value sorted in a body and then gathered with the returned index yields the UNSORTED input, measured upstream on a minimal `argsort` + `gather` loop body), and the count composition emits no `sort` op at all.
So the O(M²) fan is removable only by removing the **sort** from the per-call body — which is what the dense route does for a batch-1 centre and what the **index-free** route does at any width: its body emits one `(P,M,M)` `reduce_min` per family and no sort dispatch at all, which is exactly the trade the GPU device kind selects.
