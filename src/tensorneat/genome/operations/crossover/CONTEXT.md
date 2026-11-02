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
`DefaultCrossover.crossover_gene_attrs(state, gene, key, attrs1, homologous_rows, attrs2) -> (P, M, A) float32` redirects absent homologues to row `0` (`select`), makes ONE `../alignment.gather_attr_rows` call for the whole `(P, M, A)` block, makes ONE batched `gene.crossover` call over that block, and selects `attrs1` where the homologue is `I_INF` (one rank-3 mask, so no per-column select).
Key tables are accepted as `(P, N)` (rank 2, the payload's node-key layout) or `(P, N, 1)`; `..alignment.key_rows` normalises them to `(P, N, K)` and any other rank raises `ConfigError`.

## Alignment (`../alignment.py`, the rank merge shared with distance)
This node no longer owns an aligner: `../alignment.py` holds the one implementation both crossover and compatibility distance consume, so the two operators cannot drift apart on what a "key row" is or on how a homologue is found.
`find_homologous` = one `argsort` per gene family: pack each key row into one `uint64` identity, concat both parents' identities, `argsort` once, take each gene's rank as the inverse permutation, and read the successor (`rank + 1`) through two row gathers — the successor's identity must equal the gene's own identity with the parent-2 tag set.
The parent tag lives in the low `_TAG_BITS` of the sort key (parent 1 = `0`), so inside a group of equal identities parent 1 sorts first and the ordering never depends on the sort being stable — which is why the sort runs unstable (4× faster on the reference interpreter).
The successor index is clamped to the last position; the clamped gene is its own successor and its tag differs, so the clamp cannot invent a homologue.
`row_identity` packs the first `_PACKED_KEY_WIDTH = 2` key columns side by side (31 bits each + 2 tag bits = 64, injective over the whole legal non-negative `int32` key domain); columns beyond that are confirmed exactly by `_confirm_rows`, one `equal` per spare column — never an innermost-axis reduce.
The one precondition of the merge is the payload's own uniqueness invariant: the first two key columns are unique within one parent (a node gene has a single key column; a connection gene's first two columns are its unique `(in_key, out_key)` pair, which `execute_mutation` refuses to duplicate and `initialize` builds once).
`_pad_identity(width)` is the host mirror of `row_identity` on a padded tuple, so one `not_equal` separates parent 1's padding from a real gene.

## Constraints
- Trace-time kernel only: no `jax`/`torch`/`numpy.random`, no `vmap`/`scan`/`while_loop`/`cond`, no Python loop over individuals, no eager NumPy.
- Alignment is `O(M log M)` per individual and never an `O(M^2)` comparison or a row-index alignment.
- Op count *is* the budget on the reference interpreter (~80 µs per traced op, more on wide blocks): every mask is one op over the whole `(P, M)` block, one gather reads a whole attribute **row** rather than one per column, and `_flat_row_index` folds the per-individual stride into the row term so only one of its ops touches the `(P, M, A)` block.
- `I_INF` rows are redirected by `select` before any index arithmetic reaches the backend; a flat gather index is always in range by construction (a clamped rank, a redirected row).
- A gene without a homologue keeps parent 1's attributes exactly, and `gene.attr_width == 0` returns `attrs1` unchanged (a zero-width row would trip `batch_offsets`' positive-length validation).
- Crossover never changes structure, never fills a padded row from parent 2's row at the same index, and never mixes an `I_INF` into a float array.
- Errors: a parent shape mismatch raises `ConfigError` (`TensorNEATError`); `evoxc` errors propagate unchanged.
- Files ≤ ~1000 lines (`base.py` ~130, `default.py` ~190, `align.py` ~390).

## Measured Cost (numpy reference backend, build excluded, median of 5)
`DefaultGenome(num_inputs=2, num_outputs=1, max_nodes=50, max_conns=100, output_transform=ACT.sigmoid)` + `DefaultCrossover()`: **145 traced ops**, `4.7 / 16.6 / 34.7 ms` at pop 100 / 500 / 1000.
The 433-op baseline measured in the same session on the same box: `8.9 / 31.0 / 64.8 ms` — **1.87×** faster, and a bit-identical 6-generation seed-42 evolution trajectory (same 435→145-op operand, same `fitness_history` at pop 100).
Per-component at pop 1000: `find_homologous` node 39 ops / 4.4 ms, `find_homologous` conn 53 ops / 11.4 ms, `gather_attr_rows` 12 ops / 0.9 and 3.7 ms.
Inside `find_homologous` the load is element volume, not op count: the merge blocks are `(P, 2M)` `uint64` (1.6 MB at the conn family), two `argsort`s (1.8 ms each at `(P,200)`) plus one derived gather carry ~half its time.

## Test Strategy
`test/unit/genome/test_crossover.py` is the acceptance spec (its bodies are `pytest.skip` placeholders): parent-1 structure kept, per-attribute 50/50 for homologous genes, parent-1 attributes for non-homologous genes, padding preserved, self-crossover identity, and `find_homologous` exactness on shuffled parent-2 rows.
Verified evidence until those bodies land: a scratch oracle (`{key tuple: row}` in plain Python, 1200 random cases across key widths 1/2/3 and all-padded/empty/single-match parents) agrees with `find_homologous` on every row, and the crossover kernel reproduces the pre-rewrite 433-op implementation's `fitness_history` bit-for-bit over 6 generations at pop 100, seed 42.

## Known Issues
- `gene.crossover` (`../gene/base.py`) is now the crossover kernel's dominant cost: 7 traced ops but **7.3 ms (node) + 9.0 ms (conn) at pop 1000** — ~46 % of the kernel's runtime, element-bound on `(P, M, A)` float blocks (its per-column uniform draw, comparison and select). It is out of this node's write scope; the alignment is no longer the bottleneck.
- `../alignment.py::find_homologous` (167/185 traced ops) is still the older hash+tagged-argsort+`masked_flat_scatter` merge and is what `distance/default.py` uses; porting `align.py`'s rank merge there needs a write scope above this node.
- A gene family with more than two key columns (`OriginConn`'s `historical_marker`) pays one gather plus one `equal` per spare column in `_confirm_rows`; not exercised by the benchmark config (`DefaultConn` has two key columns).

## Routing Table
| File | Responsibility |
|---|---|
| `./base.py` | `BaseCrossover`: shape validation, per-family key split, generic recombination driver, `key_rows` misuse guard. |
| `./default.py` | `DefaultCrossover`: `find_homologous` delegation and `crossover_gene_attrs` (homologue gather + single batched gene crossover). |
| `./align.py` | The rank merge: `row_identity` (packed key prefix), `find_homologous`, `_confirm_rows`, `gather_attr_rows`, `_gather_rows`, `_flat_row_index`, `_pad_identity`. |
