# operations — batched genetic operators

## Intent
Mutation, crossover and compatibility distance as batch-first kernels over the split gene payload (`node_keys`, `node_attrs`, `conn_keys`, `conn_attrs`), population on axis 0, every shape static at build time.
This node also owns the structural invariants that keep the execution schedule valid: `max_nodes`, `max_conns` and `genome.max_in_degree` are hard caps and a refused operation is a no-op for that individual, never a silent repair.
Operators never allocate RNG keys or structural keys: the algorithm draws `key` and allocates `new_node_keys`/`new_conn_keys`, so a `(seed, config, shape)` triple reproduces a run.

## Call contract (binding for `genome/base.py` and `algorithm/`)
- `BaseMutation.__call__(state, genome, key, node_keys, node_attrs, conn_keys, conn_attrs, new_node_keys (P,), new_conn_keys (P, 3) -> GenomeTensors` — validates the two key shapes (`ConfigError`), then `split_n(key, 2)` and runs the structural stage before the value stage.
- `new_conn_keys` columns: `0` = add-node identity edge, `1` = add-node copy edge, `2` = plain add-connection edge; each becomes the `historical_marker` when the conn gene declares one.
- `BaseCrossover.__call__(state, genome, key, node_keys1, node_attrs1, conn_keys1, conn_attrs1, node_keys2, node_attrs2, conn_keys2, conn_attrs2) -> GenomeTensors` — `genome1` is the fitter parent and keeps its keys; identical parent shapes are required (`ConfigError`).
- `BaseDistance.__call__(state, genome, node_keys1, node_attrs1, conn_keys1, conn_attrs1, node_keys2, node_attrs2, conn_keys2, conn_attrs2) -> (P,) float32` — the algorithm forms the pairs; distances are `0` for an empty pair and never NaN.
- Payload key ranks: node keys are `(P, N)` as `GenomeTensors` allocates them; `(P, N, 1)` is also accepted and returned verbatim. Connection keys are `(P, E, CK)`.
- `base.py` forwards as `self.mutation(state, self, key, *payload, new_node_keys, new_conn_keys)`, `self.crossover(state, self, key, *genome1, *genome2)` and `self.distance(state, self, *genome1, *genome2)`.
- The mutation operator reads only host statics off `genome`: `num_inputs`, `num_outputs`, `max_nodes`, `max_conns`, `max_in_degree`, `network_type`, `node_gene`, `conn_gene`, and the predicate `check_cycles(node_keys, conn_keys, src_keys, dst_keys)`.

## Mutation semantics
One `evoxc.random.uniform(key, (4, P))` draw decides the structural operation per individual, rows in the order `(node_add, node_delete, conn_add, conn_delete)`: the first row whose draw is below its probability wins and row `4` means "nothing".
At most one structural change therefore applies per individual per generation (the four helpers are mutually exclusive, not independent), and value mutation always runs afterwards.
Each private helper takes a trailing `(P,) bool` gate and is a batch-wide no-op where it is false; `select` returns the untouched row bit-for-bit, `-0.0` and NaN included, so a probability of `0` is the identity.
Add-node is the only output-preserving structural change: the chosen connection is deleted, the new node is written at `first_free_row(node_keys)` with `node_gene.new_identity_attrs`, an identity edge (`conn_gene.new_identity_attrs`, weight `1`) runs old source → new node and a copy edge (recorded original attributes) runs new node → old destination. It is refused when fewer than two connection rows are free *before* the delete.
Add-connection draws its two endpoints with one draw each and refuses an endpoint of `I_INF`, a duplicate `(in_key, out_key)` pair, a cycle when `network_type == "feedforward"` (`genome.check_cycles`), and a destination already at `max_in_degree` incoming edges.
Delete-node never names an input row (`0..num_inputs-1`) or an output row (`N-num_outputs..N-1`) and detaches every touching connection with `delete_conns_by_key`; delete-conn clears exactly one row.
`choose_node_key`/`choose_connection_key` take one draw per call and return the key *at the drawn row* together with that row, so key and row always agree, and `I_INF` everywhere when nothing is eligible.
`RecurrentMutation` replaces only the add-connection step: one `(P, max_conn_tries, 2)` `randint` tableau plus one `(P, max_conn_tries)` `uniform` `p_recur` decision, the first accepted attempt chosen with a reduce (no `cond`), and a `p_recur` hit makes "closes a cycle" a hard requirement; `p_recur=0` draws nothing and degenerates to `DefaultMutation`.

## Crossover semantics
`DefaultCrossover` always keeps parent 1's structure and recombines attributes only: a gene whose exact key-column tuple has a homologue in parent 2 is crossed per attribute by `gene.crossover`, one with no homologue keeps parent 1's attributes, and padded rows stay padded.
`alignment.find_homologous(keys1, keys2)` is the only aligner and serves both crossover and distance: one packed `uint64` identity per key row (`row_identity`) → one `argsort` of the concatenated `2M` identities → each gene's rank as `argsort(order)` → successor test "the element right after me is a parent-2 row with my identity" → `_confirm_rows` for key columns past the packed prefix. `O(M log M)`, never an `O(M^2)` scan, one batched kernel.

## Distance semantics
Two shapes of the same arithmetic: the pair-wise `__call__` (fitter/first parent pair per row) and the two-phase `prepare`/`measure` pair speciation uses, where the population side is sorted **once per generation** and each wide `measure` call only merges the centres against that digest.
Both routes end in the *same* scalar tail (`DefaultDistance.distance_from_alignment`, which re-spells the aligned rows as an `I_INF` row table and defers to the documented `distance_from_rows`), so they are bit-identical rather than merely close.
`gene_distance` (pair-wise) takes its homologue table from `alignment.find_homologous`; `measure` takes it from `digest.measure_alignment`, which returns `AlignedGenes(rows, homologous, count, matched)` so no consumer re-derives a mask or a count.
`execute_distance` therefore measures **221 traced ops** (was 495) at `2.0 / 5.1 / 9.8 ms` for pop 100 / 500 / 1000 (was `4.7 / 13.7 / 25.5`).
Per-call two-phase cost (medians, build excluded, interleaved with the pre-change tree): `prepare` **114 → 64 ops** at `1.0 → 0.6 / 2.2 → 1.6 / 4.0 → 2.9 ms`; `measure` **232 → 151 ops** at `2.1 → 1.6 / 4.4 → 4.5 / 8.1 → 7.7 ms`.
`measure`'s op count fell 35 % while its wall clock is flat at pop ≥ 500: at that size the reference interpreter is array-traffic bound, and the removed ops were scalar/constant dispatches.
The generation's distance files are **2236 → 1422 traced ops** (pop 500/1000; `776 → 526` at pop 100) out of a whole generation of `4860 → 3785`, and `Pipeline.step` is `44.6 → 41.9 / 335 → 303 / 593 → 529 ms`.
Inside `measure` the merge dominates, not the model: `measure_alignment` is ~65-77 % of its ms and the aligned-attribute tail ~28 %, and the merge's `(P, 2M)` `int64` sort is the element-volume centre of the whole path.
The merge algebra (injective `2**31` fold, digest-side-first concatenation so a population gene's merged index *is* its digest row, `argsort(order)` inverse, clamped-successor test against `composite + 1`, `_confirm_rows` for a third key column) is stated in `./distance/CONTEXT.md`.
`ConstantPool` (`./distance/consts.py`) exists because `enp.full((), v, dtype)` emits one `constant` op per call and evoxc never deduplicates: one pool per trace per entry point took the distance files from **503 to 163** constant ops per generation, and a pool threaded from the generation entry would collapse the rest (~150 ops, cross-node).
Non-homologous slots are zeroed **before** the attribute `reduce_sum` (a `NaN * 0` product would poison the sum), the attribute term uses `BaseGene.distance` (`|delta|` for float columns, `!=` for int option columns), and the empty-pair guard folds into `maximum(float32(max(cnt1, cnt2)), 1.0)` with one `divide` so no pair can be `inf`/NaN.

## Constraints
- No `jax`/`torch`/`numpy.random`; no `vmap`/`scan` over individuals; no Python loop over individuals — operators are already batched and never sit inside the `grad` graph, so `argsort` is legal here.
- `while_loop` is allowed only where it replaces a statically unbounded unroll of *identical* work: the cycle check uses one region, and its per-hop frontier stays flat (`(P*(N+1),)` with one trailing dump column) so a hop is four `O(E)` ops.
- Static Python `if` is used only for host configuration (probabilities, `network_type`, key width); every per-individual decision is a `select`/mask.
- evoxc's numpy backend raises on an out-of-range gather/scatter index, so every index is masked or clamped before it reaches the backend (`masked_flat_gather`/`masked_flat_scatter` from `common.tools`, or a caller-provided *in-range* index where the code already guarantees it).
- Rank-0 constants cost one traced op each and evoxc never deduplicates them: build each distinct value **once per trace** (a `ConstantPool`-style bundle threaded through the kernel, or the entry point's own bundle) rather than at every use site; a bare Python literal additionally promotes `int32`/`float32` to `int64`/`float64`, so the typed constant is still required.
- Configuration violations raise `ConfigError` (a `TensorNEATError` subclass) at construction time; `evoxc` errors are never caught or re-wrapped.
- Files ≤ ~1000 lines (`mutation/base.py` 738 is the largest, then `distance/digest.py` 654, `distance/base.py` 588, `operation/alignment.py` 424).
- Traced-op budget is the lever here, but element volume is close behind and it is what the clock pays at pop ≥ 500: the merge's `(P, 2M)` **int64** blocks are 6.4 MB each at `4P`/`max_conns=100`, which is why the conn family costs twice the node family at the same op count.
- Element-volume rules are layout-specific and measured, not guessed: an innermost-axis reduce over the `(P, N, K)` incoming window is 10-20× an ordinary op (fold the `K` slices), while for the `(P, max_conn_tries, E)` slabs of the recurrent add-connection the **contiguous innermost** axis is the cheap one (`axes=2` 0.82 ms vs transposed 1.65 ms).
- `measure` holds one `(P, M, A)` and one `(P, 2M)` temporary per call and the digest four arrays per family; peak RSS at pop 1000 is ~2.1 GiB (pop 500: ~1.1 GiB), so a new full-size temporary in this path is a real regression even when it saves ops.

## Deviations from the pre-rewrite library (intentional)
- The four structural operations are mutually exclusive through one shared draw; the legacy build gated each of them with an independent uniform and could apply several per individual.
- `choose_node_key` takes the static row counts `num_inputs`/`num_outputs` (the legacy signature passed `input_idx`/`output_idx` lists); the four private mutation helpers take the trailing gate.
- Alignment is the packed-identity rank merge over `row_identity` (`_COLLISION_WINDOW`/`searchsorted`/hash-and-confirm no longer exist in any form): the parent tag lives in the low 2 bits of the `uint64` sort key, so parent 1 sorts first inside a group of equal identities and the single `argsort` may run unstable; only key columns past the packed prefix are confirmed by `_confirm_rows`.
- Non-finite distance weights are rejected as well as negative ones, and `compatibility_disjoint`/`compatibility_weight` are validated at construction.

## Test Strategy
`test/unit/genome/test_mutation.py`, `test_crossover.py`, `test_distance.py` are real tests today (95 passing in `test/unit/genome`), and they are the acceptance specs for the contracts above.
`test/parity` (53 passing) holds the kernels against the frozen legacy implementation on identical genomes.
The behaviour-neutral gates used for every op-diet change in this node are fixed-input oracles rather than assertions: a real evolved seed-42 XOR population (pop 100, 2 steps) replayed through `execute_crossover`, `execute_distance`, `prepare_distance`, `measure_distance`, `execute_mutation` and `RecurrentGenome.execute_mutation` on captured input payloads, compared bytewise (NaN-aware for padding rows) against a frozen copy at `commit b18716e`; plus the wide `DistanceRows` path (4 tiled centres × P rows) against batch-1 `measure`, and the 5-generation seed-42 `fitness_history` against the same frozen run.
A mutation or alignment change must be trajectory-identical, so the fingerprint is the run, not the code: four payload arrays + the ten `SpeciesState` leaves + `randkey` per generation.
`find_homologous` is also checked against a pure-Python `{key tuple: row}` oracle at key widths 1/2/3, with shuffled parent-2 rows and an adversarial case whose packed prefixes match while the third column differs (`_confirm_rows` must reject it).
Trace a bound method through a closure (`lambda *a: obj.method(state, *a)`) and build rank-0 RNG keys *inside* the trace (`evoxc.random.key(seed)`), because an `evoxc` key cannot cross the graph boundary as data.

## Known Issues
- Five `test/unit/algorithm/` tests fail at `b18716e` and still do (algorithm node, not this one): four stale `xfail(strict=True)` cases in `test_neat_tell.py` and `test_speciation.py::test_centre_genes_keep_the_split_payload_contract` (which wants population-shaped centre payloads while `DistanceRows` sends batch-1 rows).
- **Add-node is not output-preserving**: `gene/node/default.py::new_identity_attrs` keeps `aggregation_default`/`activation_default`, so the split node re-aggregates and re-activates (worst |Δoutput| 0.236, identical at `b18716e`). The fix belongs to the gene node (identity aggregation + identity activation).
- **Structural keys can exceed `genome.key_span`**: `algorithm/` allocates node keys as `max_node_key + 1 + arange(P)`, so a real pop-100 run reaches keys 225–249 while `key_span` is 150; `common.tensorization.derive_schedule`'s dense key table then silently drops those genes without setting `edge_overflow`. Either the algorithm keeps keys below `key_span` or the pipeline's shape signature carries a raised one — genome/algorithm owners.
- `closes_cycle` is traced unconditionally although exactly one configuration makes it statically dead (`network_type == "recurrent"` with `p_recur == 0`); eliding that region is ~7.6 ms per generation at pop 1000 (`mutation/`)
- Remaining op blocks outside this node, measured: `genome/gene/base.py:50` reads each attribute column as kept-axis slice + reshape (**264 traced ops per generation**), `gene/*/mutate` ~65 ops, and `common/tools.py::fetch_random`/`fetch_first` still pay two sentinel `select`s each (`mutation/arrays.py`'s `(row, hit)` pair is the cheaper primitive).
- `test/support/oracle.py::schedule` still passes a `level=` keyword that `common.tensorization.Transformed` no longer declares (the field is `node_level`), so oracle-based forward checks cannot run until the test-support owner updates it.
- Hot spot: `distance/digest.py::measure_alignment` — the `(P, 2M)` `int64` sort plus its inverse is ~4.8 ms of a ~7.7 ms `measure` call at pop 1000 for the conn family (6.4 MB per int64 block at `4P`). The measured alternative to the merge is a `log2 M`-step gather binary search of each centre composite in the digest's sorted block (fewer bytes moved, a handful more ops); the scatter inverse was measured and rejected (faster in ms, +6 ops/family, over the op budget).

## Routing Table
| Path | Responsibility |
|---|---|
| `./alignment.py` | The single shared homologue implementation: `find_homologous(keys1, keys2)` (packed-identity rank merge, consumed by crossover and distance alike), `row_identity`, `gather_attr_rows`, `key_rows` (rank-2/rank-3 key-table normaliser) and the private `_confirm_rows`/`_gather_rows`/`_flat_row_index`/`_pad_identity` helpers. |
| `./mutation/` | Structural (add/delete node, add/delete conn) and value mutation: `BaseMutation`, `DefaultMutation`, `RecurrentMutation`. |
| `./crossover/` | Outcome-preserving key alignment and attribute recombination: `BaseCrossover`, `DefaultCrossover`. |
| `./distance/` | Compatibility distance (disjoint/excess + attribute term): `BaseDistance`, `DefaultDistance`. |
