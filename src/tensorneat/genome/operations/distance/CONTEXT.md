# distance — compatibility distance, pair-wise and two-phase

## Intent
Answer one question per parent pair, batch-first: how different are these two genomes?
The result feeds speciation, so it must be symmetric, exactly `0` at identity, `0` (never NaN) for an empty pair, and normalised by the larger gene count so pairs of different sizes stay comparable.
Speciation measures **one centre genome against the whole population** twice per generation, in wide calls of `CENTRES_PER_CALL` centre rows, so a per-call kernel that re-aligns both sides would re-sort the same population payload on every call.
This node therefore offers the same arithmetic in two shapes: the pair-wise `__call__` (contract unchanged, still used by parity tests and by callers holding two explicit parents) and the two-phase `prepare` / `measure` pair, where the population side is sorted **once per generation**.
Both routes end in the *same* scalar tail (`DefaultDistance.distance_from_alignment`), which is what makes them bit-identical rather than merely close.

## API Surface
`BaseDistance.__call__(state, genome, node_keys1, node_attrs1, conn_keys1, conn_attrs1, node_keys2, node_attrs2, conn_keys2, conn_attrs2, *, consts=None) -> (P,) float32` — validates both parents field by field (matching shapes, key rank 2 or 3, attribute rank 3, keys and attributes agreeing on `(P, M)`; a violation raises `ConfigError`) and returns `gene_distance(node family) + gene_distance(conn family)`.
`BaseDistance.prepare(state, genome, node_keys, node_attrs, conn_keys, conn_attrs, *, consts=None) -> DistanceDigest` — one pass over the population, returns the reusable digest.
`BaseDistance.measure(state, genome, node_keys, node_attrs, conn_keys, conn_attrs, digest, *, consts=None) -> (P,) float32` — one centre genome against every digest row; the centre batch is `1` (broadcast internally) or `P`.
`consts` is an optional `ConstantPool`: pass one per top-level kernel entry so identical rank-0 constants cost one op instead of one per use site.
Both `prepare` and `measure` are **plain traced helpers**, never `@evoxc.defn` (a defn cannot call a defn), so the caller's own graph owns them; the digest is a pytree of eight tensors and may be captured in a closure across a trace.
`BaseDistance.gene_measure(...)` and the private `_measure_family(pool, ...)` are the per-family drivers: `digest.measure_alignment` then `DefaultDistance.distance_from_alignment`.
`BaseDistance.distance_from_alignment(...)` is the adapter a subclass overrides; the default re-spells the aligned rows as an `I_INF` table and defers to `distance_from_rows`.
`DefaultDistance.distance_from_rows(state, gene, attrs1, attrs2, rows, cnt1, cnt2, *, consts=None)` is the documented row-table spelling of the tail; it derives the homology mask from `rows` and delegates, so the arithmetic is stated once.
`DefaultDistance(compatibility_disjoint=1.0, compatibility_weight=0.4)` — both weights must be finite and non-negative (`ConfigError` at construction).
`DistanceDigest(node_keys, node_attrs, node_count, node_key_cols, conn_keys, conn_attrs, conn_count, conn_key_cols)` and `FamilyDigest(keys, attrs, count, key_cols)` live in `digest.py` (pytrees, `field_names` property), together with `prepare_family(state, gene, keys, attrs, *, consts=None)`, `measure_alignment(gene, keys, digest_keys, digest_key_cols, *, consts=None) -> AlignedGenes` and `match_rows(...)` (the row-table adapter, `select(homologous, rows, I_INF)` plus an `int32` cast).
`AlignedGenes(rows, homologous, count, matched)` is what the alignment returns: `rows (P, M) int32` already redirected to `0` where a gene has no homologue, `homologous (P, M) bool`, `count (P,)` the centre's real genes, `matched (P,)` its homologue count — so no consumer re-derives a mask or a count.

## Digest fields (batch-first, static shapes)
`node_keys (P, N) int64` / `conn_keys (P, E) int64` — **injective** packed composite per gene row, ascending with the padding sentinel last.
`node_attrs (P, N, NA) float32` / `conn_attrs (P, E, CA) float32` — the attribute rows permuted into that same sorted order, so the merge gathers them positionally; absent rows keep their NaN attributes.
A gene family with no attribute column (`attr_specs = ()`, e.g. an unweighted HyperNEAT/CPPN connection gene) contributes an empty `(P, M, 0)` block here: `_gather_rows` returns it unchanged (a zero-width row has exactly one permutation, itself) instead of tracing a `0`-step index block.
`node_count (P,) int32` / `conn_count (P,) int32` — real genes per individual, i.e. the length of the valid prefix.
`node_key_cols (P, N, NK) int32` / `conn_key_cols (P, E, CK) int32` — key columns permuted into the same order, the exact-match oracle for a gene whose key width exceeds `_FOLDED_KEYS`; for a narrower key the fold is already injective, so the field keeps the *unpermuted* key table (`key_rows`) and nothing reads it.
`prepare` writes no RNG and touches no host memory, so a digest is valid for one generation and cacheable for the whole speciate pass.
The centre is *never* prepared: `measure` packs the centre's key rows itself (`_pack`) and merges them against the digest.

## Per-trace constant pool (`consts.py`)
`ConstantPool` memoises the rank-0 `int32`/`int64`/`float32` constants a kernel entry needs (`i32`/`i64`/`f32`, keyed by `repr` so `-0.0` and `0.0` stay distinct), because `enp.full((), v, dtype=...)` emits one `constant` op per call and evoxc never deduplicates them.
One pool belongs to **one trace**: a `SymbolicTensor` belongs to one builder, so a pool must never be module-global or cached across `evoxc.build` calls; every entry point starts a private pool when the caller passes none.
Reusing a pooled tensor as several operands is legal and is exactly what collapses the duplicates: the distance files dropped from 503 `constant` ops per generation to 163.
The remaining constants are `int32` sort/index sentinels (`I_INF`, `1`, `merged_width - 1`, `_KEY_BASE`), so the next lever is a pool threaded from the generation entry (`algorithm/neat/_distance.py`), worth ~150 ops per generation and not reachable from this node.

## Merge algebra (`digest.measure_alignment`)
A gene row folds into one `int64` composite, positionally and most significant first in base `_KEY_BASE = 2**31`: a node key is itself, an `(in_key, out_key)` connection key is `in * 2**31 + out`.
`2**31` is the whole non-negative `int32` key range, so the fold is injective for **every** key a payload can legally hold, and it never consults `key_span`.
That is the binding rule of this module: `key_span` (`S`) bounds only the execution schedule's dense key table, and `algorithm/` allocates structural keys per individual that can exceed it, so `key_span` is **not** an injectivity bound and must never be used as one.
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
The inverse permutation is the one measured formulation trade-off: `argsort(order)` is two ops, while scattering the ranks costs four to eight, and with both sorts stable the `argsort` is also 1.05 → 0.29 ms at pop 1000, so the scatter is now behind on both counts.
The successor position is clamped to `2 * M - 1` rather than range-checked and masked: the last merged position always holds a padded row (both sentinels sit above every real composite), and no sentinel equals a real `composite + 1`, so the clamped read cannot pass the homology test.
`evoxc.slice(x, start, limit)` takes **lengths**, not end indices (`start + limit` is the exclusive end), which is why the centre slice is `((0, count), (pop_size, count))`.

## Sort semantics (`stable=True` is a performance rule, not a tie rule)
Every `argsort` in this module passes `stable=True`, and on the numpy reference interpreter that is where the time went: numpy's stable kind is timsort, which detects the runs the data already contains.
The digest side of the merge is ascending by construction, so the merged sort sees one sorted run of `M` beside the centre's `M` shuffled values: **1.43 → 0.28 ms** per `(1000, 200)` `int64` block at `M = 100`, and the inverse permutation **1.05 → 0.29 ms**.
`prepare_family` gains for the same reason (an evolved payload is a real-gene run followed by one long padding run): 1.51 → 1.25 ms (node) and 1.43 → 1.22 ms (conn) at `P = 1000`.
This is safe rather than lucky because none of these sorts has a *meaningful* tie: a homologous pair differs by one, real composites are unique inside a genome, and the only equal values are padding rows, whose attributes are `NaN` on both sides and which can never pass the homology test — so tie order changes no emitted number.
Stability also *pins* tie order to input order, which a comparison sort does not, and it is the tie semantics a compiled backend's count-mode `argsort` already has (rank sums plus an iota tie-break), so the reference interpreter and a compiled backend now agree on ties as well as on values.
Rule for future work: before reaching for a smaller `O(M log M)` formulation, check whether `stable=True` (or `sort` in place of `argsort`) already removed the element volume — here it did, for zero traced ops.

## Cost (measured, numpy reference backend, seed-42 XOR population, build excluded)
Per-call traced ops (before → after this node's op diet; the pair-wise column is `execute_distance`, measured on the same payloads):

| kernel | before | after |
|---|---|---|
| `measure_distance` | 232 | **151** |
| `prepare_distance` | 114 | **64** |
| `execute_distance` | 495 | 477 |
| `match_node` / `match_conn` (inside `measure`) | 49 / 54 | **35 / 43** |
| `tail_node` / `tail_conn` | 70 / 49 | **61 / 40** |

The stable-sort change added and removed nothing: `measure` is 151 and `prepare` 64 ops before and after it, and the whole-generation trace is 3785 ops either way.

Whole-generation attribution (one trace of `pipeline._generation_kernel`, ops by `op.location.file`): **4860 → 3838** traced ops at pop 500/1000 (2852 → 2478 at pop 100).
The distance files are **2236 → 1422** at pop 500/1000 (776 → 526 at pop 100) — `digest.py` 687, `default.py` 552, `consts.py` 163, `base.py` 20.
Milliseconds, head-to-head in one process set (round 2: the pre-`stable=True` tree vs this one, median of two interleaved rounds, 2 rounds each, fresh process per config):

| kernel | P=100 | P=500 | P=1000 |
|---|---|---|---|
| `measure_distance` | 1.79 → 1.41 | 4.67 → 3.36 | 8.59 → 5.58 |
| `prepare_distance` | 0.70 → 0.58 | 1.75 → 1.49 | 3.04 → 2.29 |
| `match_node` | 0.38 → 0.32 | 1.08 → 0.79 | 2.03 → 1.26 |
| `match_conn` | 0.68 → 0.45 | 2.42 → 1.34 | 4.69 → 2.42 |
| `tail_node` | 0.48 → 0.50 | 0.96 → 0.98 | 1.54 → 1.56 |
| `tail_conn` | 0.29 → 0.31 | 0.51 → 0.53 | 0.76 → 0.77 |
| `Pipeline.step` | 48 → 44 | 321 → 265 | 556 → 484 |

Cumulative against the pre-op-diet tree (`b18716e`), measured in the same interleaving discipline: `measure` **2.18 → 1.41 / 5.37 → 3.36 / 8.95 → 5.58 ms** (−35 / −37 / −38 %), `prepare` **1.06 → 0.58 / 2.40 → 1.49 / 4.30 → 2.29** (−45 / −38 / −47 %), `Pipeline.step` **49 → 44 / 387 → 265 / 624 → 484** (−11 / −32 / −22 %).
The op count is the deterministic result; the milliseconds are not.
Round 1 (the op diet) removed 35 % of `measure`'s traced ops and moved its wall clock not at all, which is what identified round 2: at pop ≥500 the reference interpreter was **array-traffic bound**, and the traffic was the `(P, 2M)` `int64` sort that the two-op inverse forces.
Stability removed that traffic without touching the arithmetic or the op count, so `measure` is now back to dispatch bound — 5.58 ms at pop 1000 is `match_node` 1.26 + `match_conn` 2.42 + the two tails 2.33, and the merge's remaining ~74 ops cost ~80 µs each.
The box drifts ±30 % between sessions, so only the before/after pairs above (measured back to back) are comparable.

## Constraints
- Trace-time kernels only: no `jax`/`torch`/`numpy.random`, no `vmap`/`scan`/`while_loop`/`cond`, no Python loop over individuals or over pairs, no eager NumPy. Nothing here is differentiable and nothing draws RNG, so repeated calls are bit-identical.
- `prepare`/`measure` must stay plain helpers (not `@evoxc.defn`) — they are called inside the caller's graph, and a defn cannot call a defn. `DistanceDigest` must stay a pytree of tensors so it can be captured in a closure inside a trace.
- Never use `evoxc.gather(..., axis=1)`: it costs 105 ms at `(P=1000, M=100)`. Every row permutation here is a flat gather with a `p * R + col` index block (`common.tools.flat_gather` would add escape checks that a permutation cannot need).
- Never build an `O(M^2)` per-pair comparison matrix: `(P, 100, 1) < (P, 1, 100)` is 3.3 ms and a `(P, 100, 100)` reduce is 4.4 ms.
- Never derive a composite radix from `key_span`: structural keys are allocated per individual by `algorithm/` and can exceed `S`, so a `key_span`-based radix silently aliases genes (that is the failure the current base `2**31` fold removes).
- Every `int64` block is element volume that shows up in the wall clock: `(P, 2M)` int64 is 6.4 MB at `4P = 4000`, `max_conns = 100`, and the merge holds two of them plus the `int32` permutation — which is why the connection family costs twice the node family at the same op count, and why its sorts are the first thing to make cheap.
- `stable=True` on every `argsort` is a performance constraint, not a taste: the reference interpreter's stable kind detects the runs this data already has (the digest half is ascending, an evolved payload is a real-gene run plus a padding run), and it costs zero traced ops. Do not drop the flag while "simplifying", and check it before replacing a sort with a smaller-but-more-op formulation — the `log2 M` galloping search was measured and lost to it (14.06 vs 8.60 ms per wide conn `measure` at pop 1000, 81 vs 41 ops).
- `measure` holds one `(P, M, A)` and one `(P, 2M)` temporary per call and the digest holds four arrays per family; peak RSS at pop 1000 is 2132 MiB (pop 500: 1129 MiB), so a new full-size temporary in this path is a real regression even when it saves ops.
- The attribute distance is masked to `0` **before** the `reduce_sum` (a `NaN * 0` product would poison the sum), and int option columns contribute `!=` (`1` per changed option) rather than `|id1 - id2|` — that rule lives in `BaseGene.distance`.
- Disjoint and excess genes are counted by one term (`cnt1 + cnt2 - 2h`).
- The non-empty guard rides on the denominator: `maximum(float32(max(cnt1, cnt2)), 1.0)`, so an empty pair yields `0.0 / 1.0` with no second `select`.
- Key tables are accepted as `(P, M)` or `(P, M, K)` via `alignment.key_rows`; rank-0 integer constants come from the pool (`enp.full((), v, dtype=...)` underneath) because a Python literal would widen an `int32` tensor to `int64`.
- Errors: a parent shape mismatch, a wrong key width, a digest built for another gene axis, or a centre batch that is neither `1` nor `P` raises `ConfigError` (`TensorNEATError`); `evoxc` errors propagate unchanged.
- Files ≤ ~1000 lines (`digest.py` 672, `base.py` 588, `default.py` 385, `consts.py` 101).

## Bit-identity argument
`__call__`/`gene_distance` and `prepare`+`measure` differ only in how the homologue table is produced (`alignment.find_homologous` vs `digest.measure_alignment`), and both feed the *same* tail.
`rows` only ever selects attribute rows, and both producers index a block whose row `j` holds the *same* gene: the pair-wise path indexes parent 2's payload rows, the digest path its sorted positions.
Because composites are injective and the side tags differ, a homologous pair is exactly an adjacent `(centre, population)` pair differing by one, so every row agrees in "is there a homologue"; the attached row number is free to differ because only attribute *values* flow through it.
The reduction runs in parent-1 row order in both routes, so even the float summation order is identical, which is required because float addition is not associative.

## Test Strategy
`test/unit/genome/test_distance.py` is the acceptance spec (its bodies are `pytest.skip` placeholders): zero at identity (including an all-padded pair), symmetry within one ulp for unequal sizes, strict monotonicity in one added non-homologous gene, the `|delta| * compatibility_weight` attribute term with a `1 : 2` ratio, int columns contributing `1`, empty pairs `0` and never NaN, batched call equal to per-pair calls, and determinism.
Verified evidence until those bodies land:
- Byte-identity against the frozen pre-change tree on a real evolved seed-42 population: the eight digest leaves (same order, dtypes and shapes), `measure_distance` and `execute_distance` all `tobytes()`-equal (never `np.array_equal` — NaN is not equal to itself), together with `execute_crossover` and both mutations.
- Wide-path equivalence at pop 100/500/1000: `DistanceRows.matrix` (4-centre blocks, `K * P` rows, digest tiled by `concatenate` of whole population blocks — `np.repeat` tiles it wrongly and shows 395 mismatches) and `DistanceRows.row` (batch-1 broadcast) both equal the pair-wise `execute_distance` result for every centre against every population row — 8000 pairs, `maxdiff = 0.0`, zero bytewise mismatches.
- Determinism (two same-seed pipelines give identical fitness arrays at pop 100 and 1000) and a 5-generation seed-42 pop-100 `fitness_history` byte-identical to the pre-change snapshot.
- `pytest test/parity` 53 passed; `pytest test/unit` keeps its five pre-existing failures (four stale `xfail(strict=True)` in `test/unit/algorithm/test_neat_tell.py` and `test_speciation.py::test_centre_genes_keep_the_split_payload_contract`), unchanged by this node.
The stable-sort round re-ran all four gates unchanged (`FAILURES: none`, 0 mismatches at the three pops, snapshot identical, same pytest counts), which is the evidence for the claim that stability changes no emitted number — the sorts' only ties are padding rows whose attributes are `NaN` on both sides.

## Remaining hot spots
`match_conn` is still the largest piece of `measure` (2.42 ms of 5.58 ms at pop 1000) but it is no longer traffic bound: with both sorts stable the merge's `int64` blocks move 4.5× faster, so what is left is its ~74 traced ops (≈80 µs each) plus the two `2M`-wide flat gathers, whose cost is close to independent of the op count.
`tail_node` (1.56 ms, 28 % of `measure`) is second only because `gene.distance` emits per-attribute-column arithmetic (four node columns against one connection column) — that arithmetic lives in `../../gene/base.py`, not here.
A merge that searched instead of sorted was the leading idea and it **lost**: a `log2 M`-step galloping search of each centre composite in the digest's ascending block reads ~4 × `(P, M)` int64 instead of one `(P, 2M)` sort, but at `M = 100` it needs ~36 steps of ~4 ops each (81 ops vs 41 for the whole merge) and measured **14.06 ms vs 8.60 ms** per wide conn `measure` at pop 1000 — the op dispatch it adds costs more than the bytes it saves once the sort is stable.
The next levers here are therefore op count (`_pack`'s per-column slice+reshape, the two `reduce_sum` counts, both of which the `AlignedGenes` contract needs) and padded shapes: `M` is `max_nodes`/`max_conns` (50/100 in the benchmark) while an evolved genome holds 3–5 nodes and 0–4 connections.

## Cross-node needs (measured here, owned elsewhere)
`algorithm/neat/_distance.py` (`DistanceRows`, 188 generation ops) and `genome/base.py::prepare_distance/measure_distance` should thread **one `ConstantPool` per generation** into `prepare`/`measure`; the pool already accepts it, and the ~163 constants the distance files still emit per generation would collapse to the handful of distinct values (~150 ops).
`genome/gene/base.py:50` reads attribute columns with one kept-axis slice plus one reshape per column: 132 + 132 traced ops per generation at pop 500/1000 (264 of the 485 that file contributes), which a single rank-3 pipeline would remove without touching the arithmetic.
`operations/alignment.py` still embeds full-size graph constants and a private `_scalar`; its `find_homologous` is only on the pair-wise route now, so it inherits nothing from this node.

## Routing Table
| File | Responsibility |
|---|---|
| `./base.py` | `BaseDistance`: pair-wise `__call__`, the two-phase `prepare` / `measure` drivers, `_measure_family`, `gene_measure`, the `distance_from_alignment` adapter, and payload/digest validation (`_validate_single`, `_validate_digest`, `_broadcast_payload`). |
| `./default.py` | `DefaultDistance`: weights and their validation, `gene_distance` (pair-wise alignment hook), `distance_from_alignment` (the scalar tail both routes share) and `distance_from_rows` (its row-table spelling). |
| `./digest.py` | `FamilyDigest`/`DistanceDigest`, the injective `2**31` composite fold (`_pack`, `_tagged`, tags, padding sentinels), the per-column/permutation helpers (`_column`, `_gather_rows`, `_permute`), `prepare_family` (sort once per generation), `measure_alignment`/`AlignedGenes` (the merge, both sorts `stable=True`) and `match_rows` (row-table adapter). |
| `./consts.py` | `ConstantPool`: per-trace rank-0 constant memoisation (`i32`/`i64`/`f32`), the reason constants are not deduplicated by evoxc. |
| `./__init__.py` | Re-exports `BaseDistance`, `DefaultDistance`, `DistanceDigest`, `FamilyDigest`. |
