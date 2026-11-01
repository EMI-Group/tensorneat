# distance — compatibility distance between pairs of genomes

## Intent
Answer one question per parent pair, batch-first: how different are these two genomes?
The result feeds speciation, so it must be symmetric, exactly `0` at identity, `0` (never NaN) for an empty pair, and normalised by the larger gene count so pairs of different sizes stay comparable.
The algorithm forms the pairs; this node only computes one scalar per pair on axis 0.

## API Surface
`BaseDistance.__call__(state, genome, node_keys1, node_attrs1, conn_keys1, conn_attrs1, node_keys2, node_attrs2, conn_keys2, conn_attrs2) -> (P,) float32` — validates both parents field by field (matching shapes, key rank 2 or 3, attribute rank 3, keys and attributes agreeing on `(P, M)`; a violation raises `ConfigError`) and returns `gene_distance(node family) + gene_distance(conn family)` using `genome.node_gene` / `genome.conn_gene`.
`DefaultDistance(compatibility_disjoint=1.0, compatibility_weight=0.4)` — both weights must be finite and non-negative (`ConfigError` at construction).
`DefaultDistance.__call__(...)` keeps the same signature and delegates to the base driver.
`DefaultDistance.gene_distance(state, gene, keys1, attrs1, keys2, attrs2) -> (P,) float32` is the subclass hook: `family = (non_homologous_count * compatibility_disjoint + homologous_attribute_distance * compatibility_weight) / max(cnt1, cnt2)`, guarded to `0` when `max(cnt1, cnt2) == 0`.
Order of operations inside `gene_distance`: `rows = alignment.find_homologous(keys1, keys2)` → `homologous = rows != I_INF` → `cnt1`/`cnt2` from `utils.valid_count`, `h` from `valid_count(rows)` → `non_homologous = cnt1 + cnt2 - 2h` → parent 2's homologous attribute rows gathered from the `(P, M*A)` view (`masked_flat_gather`, per-row offsets `row * A + arange(A)`, `mask = homologous`, `fill = NaN`) → `gene.distance(...)` → mask non-homologous slots to `0` → `reduce_sum` → divide by the float32 `max(cnt1, cnt2)` behind a `select`.

## Constraints
- Trace-time kernel only: no `jax`/`torch`/`numpy.random`, no `vmap`/`scan`/`while_loop`/`cond`, no Python loop over individuals, no eager NumPy. Nothing here is differentiable and nothing draws RNG, so repeated calls are bit-identical.
- The attribute distance is masked to `0` **before** the `reduce_sum`: a `NaN * 0` product would poison the sum, and the same holds for `inf`.
- Disjoint and excess genes are counted by one term (`cnt1 + cnt2 - 2h`), and int option columns contribute `!=` (`1` per changed option) rather than `|id1 - id2|` — that rule lives in `BaseGene.distance`, not here.
- Key tables are accepted as `(P, M)` or `(P, M, K)` via `alignment.key_rows`; rank-0 integer constants are built with `enp.full((), v, dtype=...)` because a Python literal would widen an `int32` tensor.
- Errors: a parent shape mismatch raises `ConfigError` (`TensorNEATError`); `evoxc` errors propagate unchanged.
- Files ≤ ~1000 lines (`base.py` ~147, `default.py` ~262).

## Test Strategy
`test/unit/genome/test_distance.py` is the acceptance spec (its bodies are `pytest.skip` placeholders): zero at identity (including an all-padded pair), symmetry within one ulp for unequal sizes, strict monotonicity in one added non-homologous gene, the `|delta| * compatibility_weight` attribute term with a `1 : 2` ratio, int columns contributing `1`, empty pairs `0` and never NaN, batched call equal to per-pair calls, and determinism.
Verified evidence until those bodies land: scratch kernels through `test/support/runner.run_defn` against a host formula reference with shuffled parent-2 rows (max deviation ~1e-8), `0 → 0.25 → 0.4` monotonicity, and exact bit-equality between one `P`-pair call and `P` single-pair calls.

## Routing Table
| File | Responsibility |
|---|---|
| `./base.py` | `BaseDistance`: payload validation (`_validate_payload`), per-family dispatch, the additive node + connection result. |
| `./default.py` | `DefaultDistance`: weights and their validation, `gene_distance` (homologue alignment, disjoint/excess count, attribute term, normaliser). |
