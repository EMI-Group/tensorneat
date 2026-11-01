# crossover — outcome-preserving key alignment and attribute recombination

## Intent
Recombine a batch of parent pairs into a batch of offspring without changing structure: parent 1 (the fitter parent) supplies `node_keys`/`conn_keys` verbatim, and only float attributes may come from parent 2.
A gene is homologous when its whole key-column tuple matches, so alignment is on keys, never on row positions — parent 2's rows may sit anywhere.
Nothing here is random per individual and nothing resizes the payload: a row that is padded in parent 1 stays padded, so the offspring has exactly parent 1's shape and dtypes.

## API Surface
`BaseCrossover.__call__(state, genome, key, node_keys1, node_attrs1, conn_keys1, conn_attrs1, node_keys2, node_attrs2, conn_keys2, conn_attrs2) -> GenomeTensors` — the generic driver: validates that all four arrays of both parents have identical shapes (`ConfigError`), `split_n(key, 2)` into one key per gene family, then per family `self.find_homologous(...)` + `self.crossover_gene_attrs(...)`, returning `GenomeTensors(node_keys1, recombined_node_attrs, conn_keys1, recombined_conn_attrs)`.
`DefaultCrossover.__call__(...)` keeps the same signature and delegates to the base driver.
`DefaultCrossover.find_homologous(keys1, keys2) -> (P, M) int32` delegates to `..alignment.find_homologous` (row into parent 2 per gene of parent 1, `I_INF` where absent or padded); it is the method the acceptance test traces, hence it takes no `state`.
`DefaultCrossover.crossover_gene_attrs(state, gene, key, attrs1, homologous_rows, attrs2) -> (P, M, A) float32` gathers parent 2's homologous rows (`masked_flat_gather` over the `(P, M*A)` view with index `row * A + arange(A)`, `fill = NaN`), makes ONE batched `gene.crossover` call over the whole block, and selects `attrs1` where the homologue is `I_INF`.
Key tables are accepted as `(P, N)` (rank 2, the payload's node-key layout) or `(P, N, 1)`; `..alignment.key_rows` normalises them to `(P, N, K)` and any other rank raises `ConfigError`.

## Constraints
- Trace-time kernel only: no `jax`/`torch`/`numpy.random`, no `vmap`/`scan`/`while_loop`/`cond`, no Python loop over individuals, no eager NumPy.
- Alignment is `O(M log M)` per individual — hash → tagged `argsort` merge → full key-row confirmation in `../alignment.py`. Never an `O(M^2)` comparison, never a row-index alignment.
- `I_INF` rows are redirected to a real row before any `row * width` arithmetic (an `int32` overflow would otherwise produce an escaping index) and the gather mask still blocks the read; `evoxc.select` needs the `(P, M)` homologue mask expanded to `(P, M, A)` itself.
- A gene without a homologue keeps parent 1's attributes exactly, and `gene.attr_width == 0` returns `attrs1` unchanged (a zero-width row would trip `batch_offsets`' positive-length validation).
- Crossover never changes structure, never fills a padded row from parent 2's row at the same index, and never mixes an `I_INF` into a float array.
- Errors: a parent shape mismatch raises `ConfigError` (`TensorNEATError`); `evoxc` errors propagate unchanged.
- Files ≤ ~1000 lines (`base.py` ~130, `default.py` ~186).

## Test Strategy
`test/unit/genome/test_crossover.py` is the acceptance spec (its bodies are `pytest.skip` placeholders): parent-1 structure kept, per-attribute 50/50 for homologous genes, parent-1 attributes for non-homologous genes, padding preserved, self-crossover identity, and `find_homologous` exactness on shuffled parent-2 rows.
Verified evidence until those bodies land: scratch kernels through `test/support/runner.run_defn` against a host `{key tuple: row}` oracle with shuffled rows, plus `(P, N, 1)` vs `(P, N)` equivalence, `OriginConn` three-column alignment (a differing `historical_marker` makes the gene non-homologous) and all-padded/empty parents.

## Routing Table
| File | Responsibility |
|---|---|
| `./base.py` | `BaseCrossover`: shape validation, per-family key split, generic recombination driver, `key_rows` misuse guard. |
| `./default.py` | `DefaultCrossover`: `find_homologous` delegation and `crossover_gene_attrs` (homologue gather + single batched gene crossover). |
