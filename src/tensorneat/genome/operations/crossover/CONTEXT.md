# crossover — outcome-preserving key alignment and attribute recombination

## Intent
Recombine a batch of parent pairs into a batch of offspring without changing structure: parent 1 (the fitter parent) supplies `node_keys`/`conn_keys` verbatim, and only its float attribute rows may come from parent 2.
A gene is homologous when its whole key-column tuple matches, so alignment is on keys, never on row positions — parent 2's rows may sit anywhere.
Nothing resizes and no key is allocated: a row that is padded in parent 1 stays padded, so the offspring has exactly parent 1's shapes and dtypes.
The operator is order-sensitive by contract: `(genome1, genome2)` = `(winner, loser)`, and swapping them yields parent 2's structure.
The payload carries no `enabled` flag (the frozen legacy library has none either), so there is no disabled-gene re-enable rule — a gene present in both parents is always recombined per attribute.
Alignment is deterministic; the only randomness is each gene hook's per-entry `U[0, 1)` block draw, taken from one split key per family in `(node, conn)` order, so the RNG consumption of a call is determined by the shapes and never by the homology pattern.

## API Surface
`BaseCrossover.__call__(state, genome, key, node_keys1, node_attrs1, conn_keys1, conn_attrs1, node_keys2, node_attrs2, conn_keys2, conn_attrs2) -> GenomeTensors` — the driver: validates field by field that parent 2's array shape equals parent 1's (`ConfigError`), `evoxc.random.split_n(key, 2)` (node key first, connection key second), then per family `self.align_gene_rows(key_rows(keys1), key_rows(keys2), state=state)` + `self.crossover_gene_attrs(state, gene, family_key, attrs1, aligned, attrs2)`, returning `GenomeTensors(node_keys1, node_attrs, conn_keys1, conn_attrs)` — parent 1's key arrays untouched.
`BaseCrossover` defines neither hook, so a subclass supplies both: `align_gene_rows(keys1, keys2, *, state=None) -> (rows, present)` and `crossover_gene_attrs(state, gene, key, attrs1, aligned, attrs2) -> attrs`.
`DefaultCrossover.__call__(...)` keeps the base signature and forwards every argument to the base driver, with no behaviour of its own.
`DefaultCrossover.align_gene_rows(keys1, keys2, *, state=None) -> ((P, M) int32 rows, (P, M) bool present)` delegates to `../alignment.homologous_pair` — the live path and the only place `state.kernel_mode` can switch — and `rows` arrives already redirected to `0` where `present` is `False`, so an unmasked gather cannot escape.
`DefaultCrossover.find_homologous(keys1, keys2, *, state=None) -> (P, M) int32` delegates to `../alignment.find_homologous`, the `I_INF`-marked reference spelling, and forwards `state` too; no `src/` caller exists — the acceptance tests pin it, while `__call__` runs `align_gene_rows`.
`DefaultCrossover.crossover_gene_attrs(state, gene, key, attrs1, aligned, attrs2) -> (P, M, A) float32` returns `attrs1` unchanged when `gene.attr_width == 0`, and otherwise pays one `../alignment.gather_attr_rows` for the whole block, one batched `gene.crossover` call, and one `reshape` of `present` plus one rank-3 `select` back to `attrs1`.
Key tables are accepted as `(P, N)` (the payload's node-key layout) or `(P, N, 1)`; `key_rows` normalises them to `(P, N, K)` and any other rank raises `ConfigError` (owned by `../alignment.py`).

## Alignment (shared; the implementation lives in `../alignment.py`)
This node owns no aligner: `../alignment.py` holds the one homologue implementation both crossover and compatibility distance consume, so the two operators cannot drift on what a key row is or on how a homologue is found.
`homologous_pair`/`find_homologous` take a keyword-only `state=None` and branch **once** on the static `state.kernel_mode`; `BaseCrossover.__call__` forwards its `state` into both `align_gene_rows` calls, the symmetric spelling of `crossover_gene_attrs`, so a traced graph carries exactly one formulation.
`"host"` (the numpy default, and what `state=None` resolves to) is the two-sort rank merge and the only route the numpy interpreter takes; `"accelerator"` is one `(P, M, M)` equality table with a `reduce_min` over a masked `arange`, selected on a GPU-class device kind where a compiled backend would otherwise lower each sort to a bitonic one-workgroup-per-row sort — the parent node holds the formulation ledger, its selection rule and its measurements.
Either way the table is the same answer: one row per gene of parent 1, `present` computed before the redirect, parent 1's own padded rows excluded (their identity is the pad sentinel), and `find_homologous`'s `I_INF` spelling is that same table re-spelled.
The precondition this operator inherits is the payload's own uniqueness invariant: the first two key columns are unique within one parent (a node gene has a single key column; a connection gene's first two columns are its unique `(in_key, out_key)` pair, which `execute_mutation` refuses to duplicate), and a gene with more key columns pays one gather plus one `equal` per spare column.

## Constraints
- Trace-time kernel only: no `jax`/`torch`/`numpy.random`, no `vmap`/`scan`/`while_loop`/`cond`, no Python loop over individuals, no eager NumPy.
- Crossover never changes structure, never fills a padded row from parent 2's row at the same index, never allocates a key, and never mixes an `I_INF` into a float array.
- Op budget on the reference interpreter (≈19–20 µs per dispatched op): every mask is one op over the whole `(P, M)` block, one gather reads a whole attribute **row** rather than one per column, and the absent-row mask costs exactly one closing `select` because the alignment already redirected the row (no `not_equal`, no redirect `select` here).
- A one-column attribute family (`DefaultConn`) takes `gather_attr_rows`' fast path — a row gather of the collapsed `(P, M)` view plus one `reshape` back — instead of a `(P, M, 1)` index block.
- Alignment is `O(M log M)` per individual on the host formulation; the accelerator formulation is a `(P, M, M)` table by design and is never selected on the numpy default.
- A gene without a homologue keeps parent 1's attributes bit-for-bit, and `attr_width == 0` returns `attrs1` without tracing a draw — in `crossover_gene_attrs` and again inside `gene.crossover` (whose `crossover_attrs` early-returns on an empty attribute axis).
- Errors: a parent shape mismatch and a bad key-table rank raise `ConfigError` (`TensorNEATError`); `evoxc` errors propagate unchanged.
- Files ≤ ~1000 lines: `base.py` 136, `default.py` 219, `__init__.py` 9; the aligner is `../alignment.py` (723).

## Measured Cost (numpy reference backend, build excluded)
`DefaultGenome(num_inputs=2, num_outputs=1, max_nodes=50, max_conns=100, output_transform=ACT.sigmoid)` + `NEAT(pop 500/1000)` + `XOR()`, seed 42, parents = the generation-5 payload: **143 executed ops**, counted with `test/unit/ir_ops.count_graph_ops` on the traced `execute_crossover` with the `return` terminator excluded (the count is population-independent).
Per generation the kernel dispatches the two families' `alignment.py` bodies plus 6 `crossover` ops, which makes crossover the smallest structural block of a generation.
The merge is **sort-bound**: one `argsort` over the two parents' concatenated `(P, 2M)` `int64` identities per family is the largest single item and cannot be removed on the host route (a dense key table would need `key_span`-bounded keys, see `../../CONTEXT.md`), and `gene.crossover` (`../gene/blocks.py::crossover_attrs`, 3 traced ops — draw, `less`, `select`) is the other large item, with its element-volume draw over the whole `(P, M, A)` block.

## Test Strategy
`test/unit/genome/test_crossover.py` is the acceptance spec: parent-1 structure kept, offspring keys ⊆ parent 1, homologous attributes one of the two parents' per entry (never a blend, no neighbour-row gather), non-homologous genes keep parent-1 attributes, padding preserved, shapes/dtypes, self-crossover bit-identity, `find_homologous` exact on the key tuple (`historical_marker` included) and `I_INF` for absent and padded genes, plus `gene/blocks.py::crossover_attrs` driven directly.
Every test call uses the two-argument spelling, so the suite pins `state=None` and the `"host"` formulation; the accelerator formulation's crossover side is covered only by the run-level digests, and the test that would close the gap passes `kernel_mode="accelerator"` through `align_gene_rows`/`find_homologous` and compares `rows`/`present` to the `state=None` run (`./test` is a sibling node).
There is no `test/parity` module for crossover — unlike `execute_distance`, the operator's law is pinned only by the unit suite plus the run-level bit-exactness gate — but the distance suites (`test_distance.py`, `test_distance_digest.py`) do pin the shared alignment, so a change here must leave `measure_distance`/`execute_distance` bit-identical.
Bit-exactness of a crossover change: hash the child payload, then ten seed-42 generation hashes (the four payload arrays + fitness) of the `Pipeline`, against the reference digests in `../CONTEXT.md`.

## Known Issues
- The merge `argsort` is structural on the host route; the accelerator route's `(P, M, M)` table is the trade offered against it and stays unwired on numpy, where its element volume loses.
- Refuted here: dropping the two `int64`→`int32` casts by building offsets in `int64` (no measurable change), a single combined sort for both gene families (the packed identities need all 64 bits, so families cannot be given disjoint ranges), and per-column row gathers for the 4-column node family (≈18 ops vs 4, element traffic decides).
- The two hooks are duck-typed on `DefaultCrossover`: `BaseCrossover.__call__` has no fallback, so a subclass that omits `align_gene_rows` fails with an `AttributeError` at trace time rather than a `TensorNEATError`.

## Routing Table
| File | Responsibility |
|---|---|
| `./base.py` | `BaseCrossover` (136): per-field parent-shape validation, the `split_n(key, 2)` family split and the generic driver over the two subclass hooks. |
| `./default.py` | `DefaultCrossover` (219): base delegation, the `state`-aware `align_gene_rows`/`find_homologous` delegation, and `crossover_gene_attrs` (row gather + one batched gene crossover + mask select). |
| `./__init__.py` | Re-exports `BaseCrossover` and `DefaultCrossover` as `__all__`. |
