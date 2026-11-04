# crossover — outcome-preserving key alignment and attribute recombination

## Intent
Recombine a batch of parent pairs into a batch of offspring without changing structure: parent 1 (the fitter parent) supplies `node_keys`/`conn_keys` verbatim, and only float attributes may come from parent 2.
A gene is homologous when its whole key-column tuple matches, so alignment is on keys, never on row positions — parent 2's rows may sit anywhere.
Nothing here is random per individual and nothing resizes the payload: a row that is padded in parent 1 stays padded, so the offspring has exactly parent 1's shape and dtypes.
The operator is order-sensitive by contract: `(genome1, genome2)` = (winner, loser), and swapping them yields parent 2's structure.

## API Surface
`BaseCrossover.__call__(state, genome, key, node_keys1, node_attrs1, conn_keys1, conn_attrs1, node_keys2, node_attrs2, conn_keys2, conn_attrs2) -> GenomeTensors` — the generic driver: validates that all four arrays of both parents have identical shapes (`ConfigError`), `split_n(key, 2)` into one key per gene family, then per family `self.find_homologous(...)` + `self.crossover_gene_attrs(...)`, returning `GenomeTensors(node_keys1, recombined_node_attrs, conn_keys1, recombined_conn_attrs)`.
`DefaultCrossover.__call__(...)` keeps the same signature and delegates to the base driver.
`DefaultCrossover.find_homologous(keys1, keys2) -> (P, M) int32` delegates to `../alignment.find_homologous` (row into parent 2 per gene of parent 1, `I_INF` where absent or padded); it is the method the acceptance test traces, hence it takes no `state`.
`DefaultCrossover.align_gene_rows(keys1, keys2) -> ((P, M) int32 rows, (P, M) bool present)` delegates to `../alignment.homologous_pair` and is what the driver calls; the row is already redirected to `0` where the mask is False, so the driver never recomputes an `I_INF` comparison.
`DefaultCrossover.crossover_gene_attrs(state, gene, key, attrs1, aligned, attrs2) -> (P, M, A) float32` makes ONE `../alignment.gather_attr_rows` call for the whole `(P, M, A)` block, ONE batched `gene.crossover` call over that block, and ONE rank-3 `select` of `attrs1` against the `present` mask — the mask arrives from the alignment, so it costs no `not_equal` and no redirect `select` here.
Key tables are accepted as `(P, N)` (rank 2, the payload's node-key layout) or `(P, N, 1)`; `..alignment.key_rows` normalises them to `(P, N, K)` and any other rank raises `ConfigError`.

## Alignment (`../alignment.py`, the rank merge shared with distance)
This node no longer owns an aligner: `../alignment.py` holds the one implementation both crossover and compatibility distance consume, so the two operators cannot drift apart on what a "key row" is or on how a homologue is found.
`find_homologous`/`homologous_pair` = two sorts per gene family: pack each key row into one `int64` identity (sign-flipped, see `../CONTEXT.md`), concat both parents' identities, `argsort` once, invert that permutation with one encoded `int32` `sort` (`_successor_ranks`), and read the successor (`rank + 1`, clamped) through two row gathers — the successor's identity must equal the gene's own identity with the parent-2 tag set.
The rank inversion encodes `order[s] * 2**K + min(s + 1, 2M - 1)` (`K` = bit length of `2M - 1`, the two fields never overlap) and sorts it: `order` is a bijection onto `[0, 2M)`, so the sorted position *is* the row and the low `K` bits are its clamped successor position — no scatter, no second `argsort`, and the `+1`/clamp ride in the low field as one static `(1, 2M)` table.
Measured: that `int32` sort costs 0.05 ms (nodes, `M = 50`) and 0.12 ms (conns, `M = 100`) at pop 500 against 0.42 ms for the two `scatter` writes it replaced, and it removes the `(P, 2M)` blank/update index blocks — 5 ops instead of 12, and the merge now emits no `scatter` at all.
The parent tag lives in the low `_TAG_BITS` of the sort key (parent 1 = `0`), so inside a group of equal identities parent 1 sorts first and the ordering never depends on the sort being stable — which is why the sort runs unstable (4× faster on the reference interpreter).
The successor index is clamped to the last position; the clamped gene is its own successor and its tag differs, so the clamp cannot invent a homologue.
`row_identity` packs the first `_PACKED_KEY_WIDTH = 2` key columns side by side (31 bits each + 2 tag bits = 64, injective over the whole legal non-negative `int32` key domain); columns beyond that are confirmed exactly by `_confirm_rows`, one `equal` per spare column — never an innermost-axis reduce.
The one precondition of the merge is the payload's own uniqueness invariant: the first two key columns are unique within one parent (a node gene has a single key column; a connection gene's first two columns are its unique `(in_key, out_key)` pair, which `execute_mutation` refuses to duplicate and `initialize` builds once).
`_pad_identity(width)` is the host mirror of `row_identity` on a padded tuple, so one `not_equal` separates parent 1's padding from a real gene.

## Constraints
- Trace-time kernel only: no `jax`/`torch`/`numpy.random`, no `vmap`/`scan`/`while_loop`/`cond`, no Python loop over individuals, no eager NumPy.
- Alignment is `O(M log M)` per individual and never an `O(M^2)` comparison or a row-index alignment.
- Op count *is* the budget on the reference interpreter (~80 µs per traced op, more on wide blocks): every mask is one op over the whole `(P, M)` block, one gather reads a whole attribute **row** rather than one per column, and `_flat_row_index` folds the per-individual stride into the row term so only one of its ops touches the `(P, M, A)` block.
- A one-column attribute family (`DefaultConn`) never builds a `(P, M, 1)` index block: `gather_attr_rows` collapses to one `(P, M)` row gather plus a `reshape` back.
- The merge emits no `scatter`: the rank inversion is a `sort` of an encoded `int32` block, so the portability rule "at most one scatter per prepared index block" is not spent here at all.
- `I_INF` rows are redirected by `select` before any index arithmetic reaches the backend; a flat gather index is always in range by construction (a clamped rank, a redirected row).
- A gene without a homologue keeps parent 1's attributes exactly, and `gene.attr_width == 0` returns `attrs1` unchanged (a zero-width row would trip `batch_offsets`' positive-length validation).
- Crossover never changes structure, never fills a padded row from parent 2's row at the same index, and never mixes an `I_INF` into a float array.
- Errors: a parent shape mismatch raises `ConfigError` (`TensorNEATError`); `evoxc` errors propagate unchanged.
- Files ≤ ~1000 lines (`base.py` 133, `default.py` 209; the aligner lives in `../alignment.py`, 669 lines).

## Measured Cost (numpy reference backend, build excluded)
`DefaultGenome(num_inputs=2, num_outputs=1, max_nodes=50, max_conns=100, output_transform=ACT.sigmoid)` + `NEAT(pop 500/1000)` + `XOR()`, seed 42, parents = the generation-5 payload: **142 traced ops**, **5.4 ms at pop 500** and **10.3 ms at pop 1000**.
Same-session interleaved A/B on the identical payload (11 rounds each, medians): 6.85 ms before this path's rewrite, 5.67 ms after — **-17 %**, children bit-identical (all four payload arrays).
Per generation the kernel executes **117 `alignment.py` ops + 6 `crossover` ops** (was 119 + 14), i.e. 1.4 % of the 8 614 IR ops a pop-500 `Pipeline.step` runs.
Per-op split at pop 500 (instrumented, ms): merge `argsort` 1.09 (2 calls) > `gene.crossover`'s uniform draw 0.80 + select 0.69 (out of this node) > final mask `select` 0.23 > `sort` 0.18 > gathers 0.36 > adds 0.30 > `bitwise_*` 0.49 > casts 0.18.
The merge is **sort-bound**: one `argsort` of the `(P, M)`/`(P, 2M)` `int64` identities per family is 20 % of the kernel and cannot be removed (a dense key table would need `key_span`-bounded keys, see `../../CONTEXT.md`), and `gene.crossover` (`../gene/blocks.py`) is another ~27 % with its per-entry `U[0,1)` block draw.

## Test Strategy
`test/unit/genome/test_crossover.py` is the acceptance spec: parent-1 structure kept, per-attribute 50/50 for homologous genes, parent-1 attributes for non-homologous genes, padding preserved, self-crossover identity, and `find_homologous` exactness on shuffled parent-2 rows.
The distance suites (`test_distance.py`, `test_distance_digest.py`) pin the shared alignment and its op budgets, so any change here must leave `measure_distance`/`execute_distance` bit-identical.
Bit-exactness of a crossover change is additionally checked by hashing the child payload and ten seed-42 generation hashes (payload + fitness) of the `Pipeline`.
A numpy prototype check of the encoded rank inversion against the `scatter` one covers heavy-tie inputs (all-padded parents) as well as the permutation case.

## Known Issues
- `gene.crossover` (`../gene/blocks.py::crossover_attrs`, ~13 traced ops) is the single largest remaining item after the merge `argsort`: its per-entry `U[0,1)` block draw alone is 0.80 ms of the 5.4 ms pop-500 kernel and its `select` 0.69 ms. It is out of this node's write scope.
- The merge `argsort` (1.09 ms at pop 500, 2 calls) is the second largest and is structural: matching two key sets needs a sort unless keys are bounded by `key_span` (the distance path's dense table).
- Refuted here: dropping the two `int64`→`int32` casts by building offsets in `int64` (no measurable change), a single combined sort for both gene families (the packed identities need all 64 bits, so families cannot be given disjoint ranges), and per-column row gathers for the 4-column node family (≈18 ops vs 4, element traffic decides).
- A gene family with more than two key columns (`OriginConn`'s `historical_marker`) pays one gather plus one `equal` per spare column in `../alignment._confirm_rows`; not exercised by the benchmark config (`DefaultConn` has two key columns).

## Routing Table
| File | Responsibility |
|---|---|
| `./base.py` | `BaseCrossover`: shape validation, per-family key split, generic recombination driver, `key_rows` misuse guard. |
| `./default.py` | `DefaultCrossover`: `find_homologous` delegation and `crossover_gene_attrs` (homologue gather + single batched gene crossover). |
