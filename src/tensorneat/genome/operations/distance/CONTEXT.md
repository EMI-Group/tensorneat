# distance — compatibility distance, pair-wise, two-phase and dense-single-centre

## Intent
Answer one question per parent pair, batch-first: how different are these two genomes?
The result feeds speciation, so it must be symmetric, exactly `0` at identity, `0` (never NaN) for an empty pair, and normalised by the larger gene count so pairs of different sizes stay comparable.
Speciation measures **one centre genome against the whole population** twice per generation, in wide calls of `CENTRES_PER_CALL` centre rows, so a per-call kernel that re-aligns both sides would re-sort (or re-merge) the same population payload on every call.
This node therefore offers the same arithmetic in three shapes, all ending in one scalar tail:
- the pair-wise `__call__` (contract unchanged, used by parity tests and by callers holding two explicit parents);
- the two-phase `prepare` / `measure` pair, where the population side is sorted **once per generation**;
- inside `measure`, a **dense key-table fast path for a batch-1 centre** (`dense.py`), which resolves the alignment with gathers instead of a sort.

## API Surface
`BaseDistance.__call__(state, genome, node_keys1, node_attrs1, conn_keys1, conn_attrs1, node_keys2, node_attrs2, conn_keys2, conn_attrs2, *, consts=None) -> (P,) float32` — validates both parents field by field (matching shapes, key rank 2 or 3, attribute rank 3, keys and attributes agreeing on `(P, M)`; a violation raises `ConfigError`) and returns `gene_distance(node family) + gene_distance(conn family)`.
`BaseDistance.prepare(state, genome, node_keys, node_attrs, conn_keys, conn_attrs, *, consts=None) -> DistanceDigest` — one pass over the population, returns the reusable digest.
`BaseDistance.measure(state, genome, node_keys, node_attrs, conn_keys, conn_attrs, digest, *, consts=None) -> (P,) float32` — one centre genome against every digest row; the centre batch is `1` (dense fast path) or `P` (row-wise merge, where row `i` of the centre is paired with digest row `i`).
`BaseDistance._measure_single_centre(pool, state, genome, payload, digest)` is the hook: `None` in the base class, the dense route in `DefaultDistance` (only while `DefaultDistance.dense_single_centre` is `True` **and** both gene families pass the static gate).
`BaseDistance.gene_measure(...)` and the private `_measure_family(pool, ...)` are the per-family drivers of the row-wise route: `digest.measure_alignment` then `DefaultDistance.distance_from_alignment`.
`DefaultDistance(compatibility_disjoint=1.0, compatibility_weight=0.4, dense_single_centre=True)` — both weights must be finite and non-negative (`ConfigError` at construction); `dense_single_centre=False` forces the row-wise merge everywhere.
`DefaultDistance._scalar_distance(cnt1, matched, cnt2, attribute_distance, pool)` is the single scalar tail (`dense_single_centre` irrelevant): the pair-wise route, the row-wise digest route and the dense route all call it, which is what makes them bit-identical rather than merely close.
`DistanceDigest(node_keys, node_attrs, node_count, node_key_cols, conn_keys, conn_attrs, conn_count, conn_key_cols, node_dense, node_real, node_oor, conn_dense, conn_real, conn_oor, node_layout, conn_layout, tail_counts)` and `FamilyDigest(keys, attrs, count, key_cols, dense, real, oor, layout)` live in `digest.py` (pytrees of seventeen / eight leaves, `field_names` property).
`digest.prepare_family(state, gene, keys, attrs, *, key_span, consts=None, dense=False, tail=None)`, `digest.measure_alignment(gene, keys, digest_keys, digest_key_cols, *, consts=None) -> AlignedGenes`, `digest.dense_width(gene, key_span)`, `digest.dense_slot(gene, key_span, keys, pool) -> (slot, in_range)` and `digest.match_rows(...)` are the per-family entry points.
`dense.enabled(gene, genome) -> bool` and `dense.measure_centre(pool, state, genome, distance, payload, digest) -> (P,) float32 | None` are the fast path, whose per-generation statics travel in `digest.<family>_layout` (`DenseLayout`) and whose `cnt2` pair travels in `digest.tail_counts (2, P) int32`.
`AlignedGenes(rows, homologous, count, matched)` is what the merge returns: `rows (P, M) int32` already redirected to `0` where a gene has no homologue, `homologous (P, M) bool`, `count (P,)` the centre's real genes, `matched (P,)` its homologue count.
`consts` is an optional `ConstantPool`: pass one per top-level kernel entry so identical rank-0 constants cost one op instead of one per use site.
Both `prepare` and `measure` are **plain traced helpers**, never `@evoxc.defn` (a defn cannot call a defn), so the caller's own graph owns them; a digest is a pytree and may be captured in a closure across a trace.

## Rationale for `dense_single_centre=True` being the default
`measure`'s only high-volume caller is speciation, which passes a **batch-1** centre (`DistanceRows.row`), so the fast path is the shipped route.
Measured seed-42 XOR runs are **bit-identical** with the switch on and off: the five-generation `fitness_history` (per-generation max and mean) and the final population's SHA-256 match at pop 100, 500 and 1000, and `Pipeline.step` is 341 ms at pop 1000 with the fast path against 373 ms without it in the same five-generation run.
Behaviour can differ from the row-wise merge only for payloads that carry gene keys **outside `[0, genome.key_span)`** (`algorithm/`'s structural-key allocator can produce them, see its own `CONTEXT.md`); the gate below measures that case explicitly.
`dense_single_centre=False` is the one-word escape hatch for a caller that sees that divergence.

## Dense single-centre route (`dense.py`)
A genome is a sparse table of `(key, gene)` rows and NEAT keys are small integers, so a **centre** can be scattered into a table indexed by the key itself and **gathered** at the population's own dense index:
- node family — slot `= node_key`, width `S + 1`;
- connection family — slot `= in_key * S + out_key`, width `S * S + 1`;
with `S = genome.key_span`; the trailing slot (`D - 1`) is the **dump slot** that every absent or out-of-range row reads and writes.
Per family per call: `dense_slot` (2–8 ops) → scatter the centre's `owner` row index into `(D,)` → one gather at the digest's `dense` block → `matched = (owner_at < count) & pop_real` (+ exact marker equality when the gene has columns past `_FOLDED_KEYS`) → scatter the centre's attribute rows into `(D, A)` → one gather at `(P, M, A)` → `gene.distance` → reorder the per-population-gene distances into the centre's own row order (one flat scatter at `owner_at`, or at `select(matched, owner_at, count)` when a marker column exists) → `reduce_sum`.
That reorder is the bit-identity requirement: the tail sums over the *centre's* rows, in the centre's row order, exactly like the merge, so the float summation order is unchanged.
The marker-column `select` is a correctness requirement, not a nicety: a `(in_key, out_key)` slot says nothing about `historical_marker`, so a population edge that shares the endpoints but carries a different marker is *not* homologous and must not write its `|Δweight|` into that centre row.
Without the redirect the dense route over-counted the attribute term by `compatibility_weight · |Δweight| / max(count)` per shared-but-unmatched edge and `OriginConn` payloads diverged from `execute_distance` even with every key in range (measured before the fix: 20 / 3662 / 1471 mismatched rows at pop 100 / 500 / 1000; after: 0).
Counts: `matched` from the in-range mask, `cnt1` from the centre's **present** rows (`key != I_INF`, the same predicate the digest's `count` uses), `cnt2` from the digest's `count`.
Duplicate key rows inside one genome are outside the payload contract (`alignment` documents the same); the dump redirect makes them a last-writer-wins scatter instead of a crash.

## Static gate (never a runtime branch)
`dense.enabled(gene, genome)` is `True` while the family's centre tables fit `_DENSE_MAX_BYTES = 8 MiB` of float32 (`width * 4 * (1 + attr_width + marker_columns)`) and the width stays inside `int32`.
`S = 150` (`max_nodes = 50`, `max_conns = 100`) gives `22 501` connection cells (≈0.2 MB); `S = 500` (≈2 MB) still qualifies; `S = 1000` (8 MB) is borderline and larger spans fall back to the merge — which is exactly right, because a dense table grows with `S²` while the merge grows with the gene axis `M`.
**Both** families must qualify, or neither does: a partially dense `measure` would have to broadcast the centre payload for the other family, and the row-wise merge is then the honest route.

## Precondition: gene keys inside `[0, key_span)`, reported as `node_oor`/`conn_oor`
The dense table cannot address a key outside `[0, S)`, so such a gene is **unmatched** — exactly like `common.tensorization` drops it from the execution schedule — while the row-wise merge and `execute_distance` would still match it against the same out-of-range key in the other genome.
The two routes therefore agree on every payload whose keys respect the bound and can differ (by up to one gene pair per family) otherwise; there is no hash approximation and no runtime branch, because a table of `S²` cells cannot represent `2^62` key pairs.
The divergence is **not silent**: `prepare_family` sums `present & ~real` over the gene axis (both masks are already in hand) into `FamilyDigest.oor`, and `DistanceDigest` carries it as `node_oor` / `conn_oor` (`(P,) int64`, `int64` because `evoxc.reduce_sum` returns it and the cast would be a third op).
Cost: **two traced ops per family, four per generation** (`reduce_sum` + `subtract`), nothing per `measure` call, no signature change, no raise, and no traced value ever branches.
The caller owns the assertion: a caller that relies on the fast path checks the leaves on the host (the `pipeline.py` `max_overflow` pattern) or keeps `DefaultDistance(dense_single_centre=False)`.
`oor` is computed whether or not `dense.enabled` holds, so the diagnostic never depends on the static gate.
Measured (`algorithm/`'s live allocator, seed 42, XOR): at pop 1000 after 10 generations the allocator leaves present genes outside `[0, 150)` in most genomes and only those centres diverge — the equivalence matrix below pairs every mismatch with a non-zero count and finds **zero** mismatches without one.
The durable fix belongs to `algorithm/`: keep structurally allocated keys below `genome.key_span` (or raise `key_span` in the shape signature).

## Digest fields (batch-first, static shapes)
`node_keys (P, N) int64` / `conn_keys (P, E) int64` — **injective** packed composite per gene row, ascending with the padding sentinel last.
`node_attrs (P, N, NA) float32` / `conn_attrs (P, E, CA) float32` — the attribute rows permuted into that same sorted order, so the merge gathers them positionally; absent rows keep their NaN attributes.
A gene family with no attribute column (`attr_specs = ()`, e.g. an unweighted HyperNEAT/CPPN connection gene) contributes an empty `(P, M, 0)` block here: `_gather_rows` returns it unchanged (a zero-width row has exactly one permutation, itself) instead of tracing a `0`-step index block.
`node_count (P,) int32` / `conn_count (P,) int32` — real (present) genes per individual, i.e. the length of the valid prefix.
`node_key_cols (P, N, NK) int32` / `conn_key_cols (P, E, CK) int32` — key columns permuted into the same order, the exact-match oracle for a gene whose key width exceeds `_FOLDED_KEYS`; for a narrower key the fold is already injective, so the field keeps the *unpermuted* key table (`key_rows`) and the merge reads it not at all.
`node_dense (P, N) int32` / `conn_dense (P, E) int32` — dense key-table slot of each gene, **permuted into the digest's order** so the fast path pairs it with `attrs` positionally; `node_real` / `conn_real (P, M) bool` — the in-range mask; `node_oor` / `conn_oor (P,) int64` — present genes the dense table cannot address, the precondition diagnostic above.
`prepare` writes no RNG and touches no host memory, so a digest is valid for one generation and cacheable for the whole speciate pass.
The centre is *never* prepared: `measure` packs the centre's key rows itself (`_pack`, or `dense_slot` on the fast path).

## Per-trace constant pool (`consts.py`)
`ConstantPool` memoises the rank-0 `int32`/`int64`/`float32` constants a kernel entry needs (`i32`/`i64`/`f32`, keyed by `repr` so `-0.0` and `0.0` stay distinct), because `enp.full((), v, dtype=...)` emits one `constant` op per call and evoxc never deduplicates them.
One pool belongs to **one trace**: a `SymbolicTensor` belongs to one builder, so a pool must never be module-global or cached across `evoxc.build` calls; every entry point starts a private pool when the caller passes none.
Reusing a pooled tensor as several operands is legal and is exactly what collapses the duplicates: the distance files dropped from 503 `constant` ops per generation to 163.
The pool carries **signed dtypes only** — no unsigned helper exists, and none may be added (see `## Constraints`).
The remaining constants are `int32` sort/index sentinels (`I_INF`, `1`, `merged_width - 1`, `_KEY_BASE`, the dump slot), so the next lever is a pool threaded from the generation entry (`algorithm/neat/_distance.py`), worth ~150 ops per generation and not reachable from this node.

## Merge algebra (`digest.measure_alignment`)
A gene row folds into one `int64` composite, positionally and most significant first in base `_KEY_BASE = 2**31`: a node key is itself, an `(in_key, out_key)` connection key is `in * 2**31 + out`.
`2**31` is the whole non-negative `int32` key range, so the fold is injective for **every** key a payload can legally hold, and it never consults `key_span`.
That is the binding rule of the merge: `key_span` (`S`) bounds only the dense table (and the execution schedule), and it must never be used as an injectivity bound.
Digits are not gated per column: `2**31` bounds every legal digit, so no fold can overflow, and an absent row's digits are irrelevant because the caller replaces its whole composite with the side's sentinel.
The fold is unrolled as `multiply`/`add` steps rather than a weighted `reduce_sum` over the key axis: an innermost-axis integer reduction costs an order of magnitude more than the pair of elementwise ops it replaces.
A rank-2 node key is its own leading column, so `_pack` emits no slice and no reshape for it (two ops: `not_equal` for validity, one `cast`); a rank-3 connection key reads each column with one kept-axis slice plus one `reshape`.
Two folded columns are all an `int64` composite can hold (`2 * 31 = 62` bits), so a gene with more columns (`OriginConn`'s `historical_marker`) folds its first two and **verifies** the remaining columns row-for-row (`_key_rows_equal` against `digest_key_cols`), keeping homology equal to full-tuple key-row equality.
The low bit of the sort key is the side tag (population `1`, centre `0`) and each side's padding rows collapse onto their own sentinel (`_PAD_CENTRE = 2**63 - 2`, `_PAD_POPULATION = 2**63 - 4`): above every real composite, even, and two apart.
The centre's tag is `0`, so `_tagged` skips its `add` and costs one `multiply` plus one `select`.
Injectivity plus the tag make a homologous pair exactly an adjacent `(centre, population)` pair whose composites differ by one; the parity is what proves the neighbour is a real population gene, so no separate tag test is needed.
The merge concatenates the **digest side first** and the centre second: then the merged index of a population gene *is* its digest row, and the tail reads the row table straight off the successor with no offset arithmetic.
Sequence per family: `concatenate` to `(P, 2M)`, one **stable** `argsort`, its inverse via a second **stable** `argsort` of the permutation, then two centre-width gathers — the successor's merged index and the composite it carries.
Nothing later is `2M`-wide: the homology test, the row table and both counts stay at the centre's own width `M`.
The inverse permutation is the one measured formulation trade-off: `argsort(order)` is two ops, while scattering the ranks costs four to eight, and with both sorts stable the `argsort` is also 1.05 → 0.29 ms at pop 1000, so the scatter is behind on both counts.
The successor position is clamped to `2 * M - 1` rather than range-checked and masked: the last merged position always holds a padded row (both sentinels sit above every real composite), and no sentinel equals a real `composite + 1`, so the clamped read cannot pass the homology test.
`evoxc.slice(x, start, limit)` takes **lengths**, not end indices (`start + limit` is the exclusive end), which is why the centre slice is `((0, count), (pop_size, count))`.

## Sort semantics (`stable=True` is a performance rule, not a tie rule)
Every `argsort` in the merge and the digest passes `stable=True`, and on the numpy reference interpreter that is where the time went: numpy's stable kind is timsort, which detects the runs the data already contains.
The digest side of the merge is ascending by construction, so the merged sort sees one sorted run of `M` beside the centre's `M` shuffled values: **1.43 → 0.28 ms** per `(1000, 200)` `int64` block at `M = 100`, and the inverse permutation **1.05 → 0.29 ms**.
`prepare_family` gains for the same reason (an evolved payload is a real-gene run followed by one long padding run): 1.51 → 1.25 ms (node) and 1.43 → 1.22 ms (conn) at `P = 1000`.
This is safe rather than lucky because none of these sorts has a *meaningful* tie: a homologous pair differs by one, real composites are unique inside a genome, and the only equal values are padding rows, whose attributes are `NaN` on both sides and which can never pass the homology test.
Stability also *pins* tie order to input order and matches a compiled backend's count-mode `argsort`, so the reference interpreter and a compiled backend agree on ties as well as on values.

## Cost (measured, numpy reference backend, seed-42 XOR population after 5 generations, build excluded, 1 warmup + 5 timed runs, one fresh process per pop)

| kernel | P=100 | P=500 | P=1000 |
|---|---|---|---|
| `prepare` (dense fields + `oor` included) | 137 ops | 137 | 137 |
| `prepare` + `measure` (batch-1 centre, **dense**) | 252 ops / 187 788 B constants | 252 / 203 788 B | 252 / 223 788 B |
| `prepare` + `measure` (batch-`P` centre, merge) | 287 ops / 189 460 B | 287 / 211 860 B | 287 / 239 860 B |
| `execute_distance` (unchanged) | 218 ops | 218 | 218 |
| `measure`'s own marginal constants | 16 B (4 rank-0 int32) | 16 B | 16 B |

Baked constant bytes of the `prepare` + `measure(batch 1)` graph, `enp.zeros` → `window.fill` (pop 1000): **839 772 → 223 788 B** (−615 984 B), largest single constant **404 000 → 90 008 B**; at pop 100 `249 372 → 187 788 B`, pop 500 `511 772 → 203 788 B`.
What is left is irreducible: the two `key_span`-scaled `(width + 1,)` int32 tables (22 502 cells = 90 008 B each, `S = 150`) plus a handful of small seeds.
`measure` itself adds only four rank-0 constants; every avoidable byte was the `DenseLayout` built inside `prepare`.

Round-2 same-session before → after: `measure` **141 → 111 ops**, wall clock inside drift at pop 1000 (2.43 → 2.25 ms median, minima 2.20), i.e. **2.45 → 2.25 µs/pair**; `prepare` 102 → **128 ops** (+26, once per generation against ~40 calls).
Per-phase op attribution of the old 141 (recursive op walk + per-phase graphs, pop 1000): `dense_slot`/`slot` 8, owner scatter+gather+matched 3, per-family counts 4, attribute table pair 3, `gene.distance` **27 (node) / 6 (conn)**, reorder 5 + slice 1 + reduce 1, scalar tail **15 per family**, final `add` 1.
The round-2 shape removes the owner redirect (−1/family), flattens the centre key table once (−1 conn), and stacks the two families' tail inputs on a leading axis so the tail is traced **once** (15+15+1 → 11+9 stacking ops+1 combine), landing at 111.

### The 1.0 µs/pair target is out of reach on this backend, and the arithmetic says why
`measure(batch 1)` needs ≤ ~50 traced ops for ≤1.0 ms at an ~20 µs mean marginal op cost (pop 1000).
The measured floor of this formulation is ~100: `gene.distance` alone is 33 ops (20 of the 111 are its column reads — `gene/base.py::column` is a `slice`+`reshape` pair per side per column, 5 attribute columns over two families), the two slot/owner/count blocks are ~25, the attribute table pair + reorder + reduce is ~20, and the stacked tail is 21.
Element volume is already minimal (`P × M` per family, the alignment is a scatter/gather, never a sort), so the remaining levers are cross-node: a one-op column read in `gene/base.py::column` (~0.3 ms/family) and a centre-batched `measure` API in `algorithm/` (40 calls/generation each pay the full tail and slot machinery).
Round-2 rejects, measured or arithmetically excluded: fusing the per-column arithmetic into one `(P, M, A)` block pass (`gene/base.py` measured it 3.3× slower — element volume, not op count, is the currency on block ops); a gather-based inverse permutation instead of the reorder scatter (one extra 3-D index build, no op saved); folding `abs`/`not_equal` into one masked block pair (block ops cost ~3× a column op, and the left-to-right column chain must stay byte-exact).
Remaining hot spot: `gene.distance` (33 ops) plus the two attribute block gathers; the wall clock still scales with `P × M × attr_width`, not with real gene counts.

## Constraints
- **No unsigned dtypes in a graph — `iree`'s declared dtype set has none** (`float16/float32/float64/int8/int16/int32/int64/bool`, `evoxc/backends/adapters/iree.py::_dtype_capabilities`), so a `uint32`/`uint64` operand, result, constant or graph-boundary value makes that backend refuse the whole graph (`BackendError: capability drift`). Use **same-width signed carriers**: a range test is `logical_and(greater_equal(key, 0), less(key, S))` on `int32` (`+1` op for a one-column key, `+2` for two columns, against the unsigned one-comparison form), which is the same truth value for every `int32` value while `0 <= S < 2**31` — guarded statically in `digest.dense_width`, the one place `key_span` becomes a table geometry. The `>= 0` half is not decoration: a negative structural key must land in the dump slot exactly as `I_INF` does.
- Trace-time kernels only: no `jax`/`torch`/`numpy.random`, no `vmap`/`scan`/`while_loop`/`cond`, no Python loop over individuals or over pairs, no eager NumPy. Nothing here is differentiable and nothing draws RNG, so repeated calls are bit-identical.
- `prepare`/`measure` must stay plain helpers (not `@evoxc.defn`) — they are called inside the caller's graph, and a defn cannot call a defn. `DistanceDigest` must stay a pytree of tensors so it can be captured in a closure inside a trace.
- The fast/slow decision is **static** (gene geometry, `key_span`, the `dense_single_centre` switch). No traced value ever drives Python control flow, and a partially dense `measure` is impossible by construction.
- Never use `evoxc.gather(..., axis=1)`: it costs 105 ms at `(P=1000, M=100)`. Every row permutation here is a flat gather with a `p * R + col` index block (`common.tools.flat_gather` would add escape checks that a permutation cannot need), and the one scatter that reorders values into centre order is a raw `evoxc.scatter` on a flat index because every index is already inside `[0, M]` (`masked_flat_scatter`'s generic redirect cost 18 ops against 7).
- `evoxc.scatter` takes **rank-1 indices per axis** (`updates` must have shape `x.shape[:axis] + indices.shape + x.shape[axis+1:]`): a `(M,)` index scatters a `(M, A)` update block along axis 0, and a per-row index needs the flat spelling.
- Never build an `O(M^2)` per-pair comparison matrix: `(P, 100, 1) < (P, 1, 100)` is 3.3 ms and a `(P, 100, 100)` reduce is 4.4 ms.
- Never derive a composite radix from `key_span`: the merge's composite uses `2**31` for injectivity, and the dense table derives its *slot* from `key_span` only because it is an address, not an identity.
- Every `int64` block is element volume that shows up in the wall clock: `(P, 2M)` int64 is 6.4 MB at `4P = 4000`, `max_conns = 100`, and the merge holds two of them plus the `int32` permutation — which is why the connection family costs twice the node family at the same op count, and why the dense route wins on exactly that family.
- `measure` holds one `(P, M, A)` temporary per call (plus one `(P, M+1)` on the dense route) and the digest four arrays per family plus the two dense blocks; peak RSS at pop 1000 is ~2.1 GiB (pop 500: ~1.1 GiB), so a new full-size temporary in this path is a real regression even when it saves ops.
- The dense constants (`enp.full((width,), …)`, `enp.zeros((width, A))`) are graph constants of `width` cells — ~0.2 MB per call at `S = 150`, which is why `_DENSE_MAX_BYTES` is a hard cap and not a soft warning.
- **Population-scaled seeds go through `genome.window.fill`, never `enp.zeros`/`enp.full`**: `DenseLayout.base` `(P·(M+1),)` and `DenseLayout.zeros` `(P,)` were the last such constants in this node (608 000 B for both families at `P = 1000`); only genuinely static tables (`key_span`-scaled `(width,)`, index `arange`s) keep `enp.zeros`/`enp.full`.
- The attribute distance is masked/redirected so it never reads a non-homologous slot (a `NaN * 0` product would poison a sum), and int option columns contribute `!=` (`1` per changed option) rather than `|id1 - id2|` — that rule lives in `BaseGene.distance`.
- `attr_width == 0` (a gene family with `attr_specs = ()`) is a supported configuration, not an error case: `_gather_rows` returns the empty `(P, M, 0)` block unchanged, `distance_from_alignment` and `dense._family_distance` skip the attribute term (`attribute_distance=None` in the shared tail), so such a family's distance is its structural disjoint/excess term alone.
  Every row stride built from an attribute width must be behind a static width branch — `enp.arange(0, n, 0)` is a `ZeroDivisionError` on the reference backend, and `common.tools.batch_offsets` rejects a `0` row length with a `ConfigError`.
- Disjoint and excess genes are counted by one term (`cnt1 + cnt2 - 2h`).
- The non-empty guard rides on the denominator: `maximum(float32(max(cnt1, cnt2)), 1.0)`, so an empty pair yields `0.0 / 1.0` with no second `select`.
- Key tables are accepted as `(P, M)` or `(P, M, K)` via `alignment.key_rows`; rank-0 integer constants come from the pool (`enp.full((), v, dtype=...)` underneath) because a Python literal would widen an `int32` tensor to `int64`.
- Errors: a parent shape mismatch, a wrong key width, a digest built for another gene axis, or a centre batch that is neither `1` nor `P` raises `ConfigError` (`TensorNEATError`); `evoxc` errors propagate unchanged.
- Files ≤ ~1000 lines (`digest.py` 924, `base.py` 708, `default.py` 491, `dense.py` 343, `layout.py` 195, `consts.py` 124).

## Bit-identity argument (three routes, one tail)
`__call__`/`gene_distance`, `prepare`+`measure` (merge) and `prepare`+`measure` (dense) differ only in how the homologue table is produced — `alignment.find_homologous`, `digest.measure_alignment`, or the dense key table — and all three feed `_scalar_distance`.
For the dense route: `matched` and the per-gene attribute distances are the *same multiset* as the merge's (a slot is shared exactly when the key tuples are equal), the per-gene values are written back into the centre's own row order before the reduction, and the same `gene.distance` runs on the same rows, so even the float summation order is identical — which is required because float addition is not associative.
The only escape is a key the dense table cannot address (`>= key_span`), documented above as the route's precondition.

## Test Strategy
`test/unit/genome/test_distance.py` is the acceptance spec (its bodies are `pytest.skip` placeholders): zero at identity (including an all-padded pair), symmetry within one ulp for unequal sizes, strict monotonicity in one added non-homologous gene, the `|delta| * compatibility_weight` attribute term with a `1 : 2` ratio, int columns contributing `1`, empty pairs `0` and never NaN, batched call equal to per-pair calls, and determinism.
`test/unit/genome` runs **95** passing tests and `test/parity` **53**; `test/release_gate_no_jax.py` passes **4**.
Verified evidence until those bodies land (scripts live in `$TMPDIR`, not in the repo):
- **Equivalence matrix** (dense batch-1 `measure` vs `execute_distance` on the centre tiled to `P`, row-wise `array_equal`; 5 families × pop 100/500/1000 × generations 1–10 × 10 centres = 1 500 centres, 750 000 rows per table, seed 42, live `Pipeline` payloads):
  (a) **keys remapped into `[0, key_span)`** — `DefaultNode+DefaultConn`, `BiasNode+DefaultConn`, `OriginNode+OriginConn` (key width 3, marker verification), zero-attribute conn gene (`attr_specs = ()`), `RecurrentGenome`: **0 mismatched rows in all 15 family/pop cells**; batch-`P` `measure` vs `execute_distance` also **0**.
  (b) **the live allocator's payloads** — mismatched rows only where `*_oor > 0`: default 3 / 3 084 / 7 445, bias 2 / 6 081 / 1 969, origin 3 / 2 501 / 111, zero-attribute 3 / 3 084 / 7 445, recurrent 3 / 808 / 5 635 (pop 100 / 500 / 1000); **0 mismatches without a non-zero count** in any cell, and the graph's `node_oor`/`conn_oor` equal a host recount of the payload on every payload checked.
  Batch-`P` `measure` is 0 mismatches on the live table too, i.e. the merge route is unaffected by the precondition.
- **Determinism / behaviour neutrality**: five-generation seed-42 `Pipeline` XOR runs with `dense_single_centre=True` and `False` produce identical per-generation `(max, mean)` fitness and an identical final-population SHA-256 at pop 100, 500 and 1000.
- **Byte-identity of the shared tail**: `measure` (both shapes) and `execute_distance` are `tobytes()`-equal (never `np.array_equal` — NaN is not equal to itself), and a 5-generation seed-42 pop-100 `fitness_history` matches the pre-change snapshot.
Zero-attribute families are not shipped as genes, so the `attr_width == 0` path is verified by subclassing a shipped family with `attr_specs = ()` (an unweighted connection gene) and driving `Pipeline` XOR on the numpy backend, plus `prepare`/`measure`/`execute_distance` traced and run inside one graph.
A regression case for the dense route (and one for `attr_width == 0`) belongs in `test/unit/genome/test_distance.py` (a read-only node for `distance/`).
Trace a bound method through a closure (`lambda *a: obj.method(state, *a)`) and build rank-0 RNG keys *inside* the trace (`evoxc.random.key(seed)`), because an `evoxc` key cannot cross the graph boundary as data.

## Rejected (measured, so nobody re-derives them)
A **galloping / `log2 M`-step binary search** of each centre composite in the digest's sorted block: reads ~4 × `(P, M)` int64 instead of one `(P, 2M)` sort, but needs ~36 steps × ~4 ops at `M = 100` (81 ops against the merge's 41) and measured **14.06 ms vs 8.60 ms** per wide conn `measure` at pop 1000.
A **scatter-derived inverse permutation** instead of `argsort(order)`: behind on both ops (7 vs 2) and ms (1.05 vs 0.29).
**`masked_flat_scatter` for the dense route's attribute reorder**: 18 ops of out-of-range redirect against 7 for the raw flat `evoxc.scatter` (the indices are already in `[0, M]`).
**A `key_span`-derived composite radix**: aliases genes the moment `algorithm/` allocates a key outside the span (`2**31` folding is injective for the whole `int32` range and costs the same).
**Hashing an out-of-range key into the dense table** (e.g. `key % width`): trades a documented precondition for silent collisions, which would make two different genes homologous.

## Cross-node needs (measured here, owned elsewhere)
`algorithm/neat/_distance.py` (`DistanceRows`, 188 generation ops) and `genome/base.py::prepare_distance/measure_distance` should thread **one `ConstantPool` per generation** into `prepare`/`measure`; the pool already accepts it, and the ~163 constants the distance files still emit per generation would collapse to the handful of distinct values (~150 ops).
`algorithm/` should keep structurally allocated keys below `genome.key_span` (or carry a raised span in the shape signature): that is what makes the dense route exact for every payload, and it is the same fix that stops the execution schedule from silently dropping evolved genes.
`algorithm/neat/_distance.py::DistanceRows.row` must keep passing a **batch-1 centre** to `measure_distance` to get the fast path; a population-shaped centre is legal but always takes the row-wise merge (`DistanceRows.matrix` does that on purpose, one digest row per centre).
`genome/gene/base.py:50` reads attribute columns with one kept-axis slice plus one reshape per column: 132 + 132 traced ops per generation at pop 500/1000, and the per-column attribute arithmetic of `BaseGene.distance` is now the largest remaining block inside `measure`.
`operations/alignment.py` still embeds full-size graph constants and a private `_scalar`; its `find_homologous` is only on the pair-wise route now, so it inherits nothing from this node.

## Routing Table
| File | Responsibility |
|---|---|
| `./base.py` | `BaseDistance`: pair-wise `__call__`, the two-phase `prepare` / `measure` drivers, the `_measure_single_centre` hook, `_measure_family`, `gene_measure`, the `distance_from_alignment` adapter, and payload/digest validation (`_validate_single`, `_validate_digest`, `_broadcast_payload`). |
| `./default.py` | `DefaultDistance`: weights, the `dense_single_centre` switch, `_measure_single_centre` (delegates to `./dense.py`), `gene_distance` (pair-wise alignment hook), `distance_from_alignment` (the aligned attribute gather), `distance_from_rows` (its row-table spelling) and `_scalar_distance` (the one scalar tail all three routes share). |
| `./dense.py` | The single-centre fast path: `enabled` (the static `_DENSE_MAX_BYTES`/`int32` gate), `measure_centre` (stacks both families' tail inputs and calls the shared tail once) and `_family_terms` (`FamilyTerms`: centre count, homologue count, summed attribute distance), plus the marker table, the marker-aware attribute-reorder column and `_stack`. |
| `./layout.py` | `DenseLayout` — the fast path's per-generation statics (owner/attribute table bases, centre row index, flat reorder offsets and buffer, count/zeros/padding/dump `int32`/`float32` scalars, the tail constants) built once in `prepare` with `genome.window.fill` for the population-scaled seeds, and `reorder` (the flat scatter that writes per-population-gene values into centre order). |
| `./digest.py` | `FamilyDigest`/`DistanceDigest`, the injective `2**31` composite fold (`_pack`, `_tagged`, tags, padding sentinels), the per-column/permutation helpers (`_column`, `_gather_rows`, `_permute`), `dense_width`/`dense_slot` (key → table slot, dump slot for absent or out-of-range rows), `prepare_family` (which also counts the out-of-range genes the dense table cannot address), `measure_alignment`/`AlignedGenes` (the merge, both sorts `stable=True`) and `match_rows`. |
| `./consts.py` | `ConstantPool`: per-trace rank-0 constant memoisation (`i32`/`i64`/`f32`), the reason constants are not deduplicated by evoxc, and why the pool holds no unsigned dtype. |
| `./__init__.py` | Re-exports `BaseDistance`, `DefaultDistance`, `DistanceDigest`, `FamilyDigest`. |
