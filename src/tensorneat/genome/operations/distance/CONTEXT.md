# distance — compatibility distance, pair-wise and two-phase

## Intent
Answer one question per parent pair, batch-first: how different are these two genomes?
The result feeds speciation, so it must be symmetric, exactly `0` at identity, `0` (never NaN) for an empty pair, and normalised by the larger gene count so pairs of different sizes stay comparable.
Speciation measures **one centre genome against the whole population** twice per generation (`2 * species_size` calls), so a per-call kernel that re-aligns both sides re-sorts the same population payload 40 times at pop 1000.
This node therefore offers the same arithmetic in two shapes: the pair-wise `__call__` (contract unchanged, still used by parity tests and by callers that hold two explicit parents) and the two-phase `prepare` / `measure` pair, where the population side is aligned **once per generation**.
Both routes end in the *same* function `DefaultDistance.distance_from_rows`, which is what makes them bit-identical rather than merely close.

## API Surface
`BaseDistance.__call__(state, genome, node_keys1, node_attrs1, conn_keys1, conn_attrs1, node_keys2, node_attrs2, conn_keys2, conn_attrs2) -> (P,) float32` — validates both parents field by field (matching shapes, key rank 2 or 3, attribute rank 3, keys and attributes agreeing on `(P, M)`; a violation raises `ConfigError`) and returns `gene_distance(node family) + gene_distance(conn family)` using `genome.node_gene` / `genome.conn_gene`.
`BaseDistance.prepare(state, genome, node_keys, node_attrs, conn_keys, conn_attrs) -> DistanceDigest` — one pass over the population, returns the reusable digest.
`BaseDistance.measure(state, genome, node_keys, node_attrs, conn_keys, conn_attrs, digest) -> (P,) float32` — one centre genome against every digest row; the centre batch is `1` (broadcast internally) or `P`.
Both are **plain traced helpers**, never `@evoxc.defn` (a defn cannot call a defn), so the caller's own graph owns them; the digest is a pytree of eight tensors and may be captured in a closure across a trace.
`BaseDistance.gene_measure(state, gene, keys, attrs, digest_keys, digest_attrs, digest_count, digest_key_cols)` is the per-family driver of `measure`, mirroring `gene_distance` for `__call__`: alignment (`digest.match_rows`) then the shared tail.
`DefaultDistance.distance_from_rows(state, gene, attrs1, attrs2, rows, cnt1, cnt2)` is the single scalar tail both routes call: aligned attribute gather → `gene.distance` → `select`-masked `reduce_sum` over parent-1 row order → `non_homologous = cnt1 + cnt2 - 2h` → `max` denominator → non-empty `select` guard.
`DefaultDistance(compatibility_disjoint=1.0, compatibility_weight=0.4)` — both weights must be finite and non-negative (`ConfigError` at construction).
`DistanceDigest(node_keys, node_attrs, node_count, node_key_cols, conn_keys, conn_attrs, conn_count, conn_key_cols)` and `FamilyDigest(keys, attrs, count, key_cols)` live in `digest.py` (pytrees, `field_names` property), together with `prepare_family(state, gene, keys, attrs)` and `match_rows(gene, keys, digest_keys, digest_key_cols)`.
`DefaultDistance.gene_distance(...)` keeps its legacy signature and reduces to `find_homologous` + `distance_from_rows`, so a subclass states the arithmetic once.

## Digest fields (batch-first, static shapes)
`node_keys (P, N) int64` / `conn_keys (P, E) int64` — **injective** packed composite per gene row, ascending with the padding sentinel last.
`node_attrs (P, N, NA) float32` / `conn_attrs (P, E, CA) float32` — the attribute rows permuted into that same sorted order, so the merge gathers them positionally; absent rows keep their NaN attributes.
`node_count (P,) int32` / `conn_count (P,) int32` — real genes per individual, i.e. the length of the valid prefix.
`node_key_cols (P, N, NK) int32` / `conn_key_cols (P, E, CK) int32` — key columns permuted into the same order, the exact-match oracle for a gene whose key width exceeds `_FOLDED_KEYS`; for a narrower key the fold is already injective, so the field keeps the *unpermuted* key table and nothing reads it.
`prepare` writes no RNG and touches no host memory, so a digest is valid for one generation and cacheable for the whole speciate pass.
The centre is *never* prepared: `measure` packs the centre's key rows itself (`_pack`) and merges them against the digest.

## Merge algebra (`digest.match_rows`)
A gene row folds into one `int64` composite, positionally and most significant first in base `_KEY_BASE = 2**31`: a node key is itself, an `(in_key, out_key)` connection key is `in * 2**31 + out`.
`2**31` is the whole non-negative `int32` key range, so the fold is injective for **every** key a payload can legally hold, and it never consults `key_span`.
That is the binding rule of this module: `key_span` (`S`) bounds only the execution schedule's dense key table, and `algorithm/` allocates structural keys per individual that can exceed it, so `key_span` is **not** an injectivity bound and must never be used as one.
Digits are not gated per column: `2**31` bounds every legal digit, so no fold can overflow, and an absent row's digits are irrelevant because the caller replaces its whole composite with the side's sentinel.
The fold is unrolled as `multiply`/`add` steps rather than a weighted `reduce_sum` over the key axis: an innermost-axis integer reduction costs an order of magnitude more than the pair of elementwise ops it replaces.
Two folded columns are all an `int64` composite can hold (`2 * 31 = 62` bits), so a gene with more columns (`OriginConn`'s `historical_marker`) folds its first two and **verifies** the remaining columns row-for-row against `digest_key_cols` (`_key_rows_equal`), keeping homology equal to full-tuple key-row equality.
The low bit of the sort key is the side tag (population `1`, centre `0`) and each side's padding rows collapse onto their own sentinel (`_PAD_CENTRE = 2**63 - 2`, `_PAD_POPULATION = 2**63 - 4`): above every real composite, even, and two apart.
Injectivity plus the tag make a homologous pair exactly an adjacent `(centre, population)` pair whose composites differ by one; the parity of the right neighbour is checked explicitly, because a `(population, centre)` adjacency of two *different* genes can also differ by one.
The merge is one `concatenate(axis=1)` to `(P, 2M)`, one comparison `argsort` (not stable — the cheap one), one flat gather of the permutation in place of a second `sort`, the adjacency and parity compares, then a `flat_scatter` of the matched digest columns back into the **centre's own row order** — the reduction therefore runs in payload order exactly as `__call__` does, which is required because float addition is not associative.

## Cost (measured, numpy reference backend, XOR population after 3 generations, build excluded, fresh process per config)
Per distance call, centre broadcast to `P`:

| P | old `__call__` | `prepare` (once per generation) | `measure` (per centre) | modelled speciate `prepare + 40 * measure` |
|---|---|---|---|---|
| 100 | 5.09 ms | ~0.8 ms | 2.93 ms | ~118 ms (old 204 ms) |
| 500 | 13.80 ms | ~2.2 ms | 7.24 ms | ~292 ms (old 610 ms) |
| 1000 | 25.4 ms | ~2.5 ms | 14.5 ms | ~486 ms (old 1103 ms) |

Isolated speciate-graph timing (`bench.py`, old `execute_distance` vs new digest, median of 3 fresh processes): **pop 1000 1103 → 486 ms (2.27×)**, **pop 500 610 → 289 ms (2.12×)**.
Per-call `measure` at pop 1000 is 14.5 ms (min 14.46), pop 500 7.24 (min 7.0), pop 100 2.93; `Pipeline.step` end-to-end is 1927 → 747 ms at pop 1000 and 1040 → 443 ms at pop 500.
Traced ops: old `547` → `measure 309` + `prepare 116` once per generation.
At pop 1000 an op costs ~46 µs, so the op count *is* the result: 309 measure ops are ~196 of shared tail (`distance_from_rows`, twice) and ~113 of alignment (`match_rows`, twice, of which ~26 are rank-0 constants).
The per-call figure is box-load dependent (the same 309-op kernel measured 10.8 ms in an earlier, quieter window; the old pair-wise path moved from 25.4 to 27.6 ms per call over the same period), so compare the old/new ratio rather than an absolute millisecond target.

## Constraints
- Trace-time kernels only: no `jax`/`torch`/`numpy.random`, no `vmap`/`scan`/`while_loop`/`cond`, no Python loop over individuals or over pairs, no eager NumPy. Nothing here is differentiable and nothing draws RNG, so repeated calls are bit-identical.
- `prepare`/`measure` must stay plain helpers (not `@evoxc.defn`) — they are called inside the caller's graph, and a defn cannot call a defn. `DistanceDigest` must stay a pytree of tensors so it can be captured in a closure inside a trace.
- Never use `evoxc.gather(..., axis=1)`: it costs 105 ms at `(P=1000, M=100)`. Every row permutation here is `common.tools.flat_gather`/`flat_scatter` with a `p * R + col` index block.
- Never build an `O(M^2)` per-pair comparison matrix: `(P, 100, 1) < (P, 1, 100)` is 3.3 ms and a `(P, 100, 100)` reduce is 4.4 ms.
- Never derive a composite radix from `key_span`: structural keys are allocated per individual by `algorithm/` and can exceed `S`, so a `key_span`-based radix silently aliases genes (that is the failure the current base `2**31` fold removes).
- `flat_scatter`/`masked_flat_gather` indices are always masked or clamped before they reach the backend (the numpy backend raises on an out-of-range index); the merge's dropped writes land in one trailing dump column.
- The attribute distance is masked to `0` **before** the `reduce_sum` (a `NaN * 0` product would poison the sum), and int option columns contribute `!=` (`1` per changed option) rather than `|id1 - id2|` — that rule lives in `BaseGene.distance`.
- Disjoint and excess genes are counted by one term (`cnt1 + cnt2 - 2h`).
- Key tables are accepted as `(P, M)` or `(P, M, K)` via `alignment.key_rows`; rank-0 integer constants are built with `enp.full((), v, dtype=...)` because a Python literal would widen an `int32` tensor to `int64`.
- Errors: a parent shape mismatch, a wrong key width, a digest built for another gene axis, or a centre batch that is neither `1` nor `P` raises `ConfigError` (`TensorNEATError`); `evoxc` errors propagate unchanged.
- Files ≤ ~1000 lines (`digest.py` 486, `base.py` 434, `default.py` 304).

## Bit-identity argument
`__call__` and `prepare`+`measure` differ only in how the homologue table `rows` is produced (`alignment.find_homologous` vs `digest.match_rows`), and both feed the same `distance_from_rows` with the same `cnt1`/`cnt2`.
`rows` is compared nowhere except by `not_equal(rows, I_INF)` and by the gather that reads parent-2 attribute rows, and both producers index a block whose row `j` holds the *same* gene: the pair-wise path indexes parent 2's payload rows, the digest path its sorted positions.
Because composites are injective, a homologous pair is exactly an adjacent `(centre, population)` pair differing by one, so every `rows` entry agrees in "is there a homologue"; the attached row number is free to differ because only attribute *values* flow through it.
The per-gene terms are then scattered back to parent-1 row order before the `reduce_sum`, so even the float summation order is identical.

## Test Strategy
`test/unit/genome/test_distance.py` is the acceptance spec (its bodies are `pytest.skip` placeholders): zero at identity (including an all-padded pair), symmetry within one ulp for unequal sizes, strict monotonicity in one added non-homologous gene, the `|delta| * compatibility_weight` attribute term with a `1 : 2` ratio, int columns contributing `1`, empty pairs `0` and never NaN, batched call equal to per-pair calls, and determinism.
Verified evidence until those bodies land: the bit-identity matrix below, plus a host-formula reference with shuffled parent-2 rows (max deviation ~1e-8), `0 → 0.25 → 0.4` monotonicity, and exact bit-equality between one `P`-pair call and `P` single-pair calls.
Bit-identity matrix (`np.array_equal`, never `allclose`): real evolved XOR payloads at pop 100/500/1000 × gene variants `default` / `BiasNode` / `OriginNode` / `OriginConn` / both × centre kinds (a population row, all-padding, no-connections, no-nodes, batch-1) — every combination returned `maxdiff = 0.000e+00`; and in one graph at pop 1000, five row rotations × 1000 rows compare `measure` against `execute_distance`, with the centre spelled both as a materialised gather and as `broadcast_to` of one row: 5000 pairs, 0 mismatches, and the two spellings agree bit-for-bit.
Scenario-level A/B (seed 42, `SpeciesController.speciate` monkeypatched to prepare once and pass a `measure` closure — the integration `algorithm/` will use): pop 1000 × 3 generations, pop 500 × 4 and pop 100 × 6 give byte-identical payload arrays, identical per-generation fitness vectors and identical `species_keys` / `best_fitness` / `last_improved` / `member_count` / `idx2species` / `next_species_key` on every generation.
Speciate-only A/B on the recorded gen-0/1/2 inputs also compares the ten `SpeciesState` leaves of `execute_distance` against `measure` inside one graph: equal at every generation.

## Remaining hot spots
`distance_from_rows` is the dominant cost of a `measure` call: ~98 traced ops per family (~196 of the 309), against ~56 per family for the whole merge — the alignment is not the bottleneck, the shared tail is.
Inside it, `gene.distance` emits per-attribute-column arithmetic (4 node columns versus 1 connection column, which is why the node family is the more expensive half) and `masked_flat_gather` costs a mask broadcast plus a NaN fill that only re-establishes what the `select` around the `reduce_sum` already enforces.
The other lever is the number of calls, not the kernel: one `measure` per species centre means the same 309-op graph runs `2 * species_size` times per generation, while batching the live centres into one wider call would pay those ops once.
Padded shapes still decide the bill: `M` is `max_nodes`/`max_conns` (50/100 in the benchmark) while an evolved genome holds 3-5 nodes and 0-4 connections, so shrinking `max_nodes`/`max_conns` shrinks every number above proportionally.

## Routing Table
| File | Responsibility |
|---|---|
| `./base.py` | `BaseDistance`: pair-wise `__call__`, the two-phase `prepare` / `measure` drivers, `gene_measure`, and payload/digest validation (`_validate_single`, `_validate_digest`, `_broadcast_payload`). |
| `./default.py` | `DefaultDistance`: weights and their validation, `gene_distance` (pair-wise alignment hook), `distance_from_rows` (the shared scalar tail). |
| `./digest.py` | `FamilyDigest`/`DistanceDigest`, the injective `2**31` composite fold (`_pack`, `_tagged`, tags, padding sentinels, `_key_rows_equal`), `prepare_family` (sort once) and `match_rows` (merge one centre). |
| `./__init__.py` | Re-exports `BaseDistance`, `DefaultDistance`, `DistanceDigest`, `FamilyDigest`. |
