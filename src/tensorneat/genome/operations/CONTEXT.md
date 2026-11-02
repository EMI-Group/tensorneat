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
`alignment.find_homologous(keys1, keys2)` is the only aligner: hash → tagged `argsort` over the concatenation of both parents' hashes → adjacent `(parent 1, parent 2)` pair → full key-row confirmation. `O(M log M)`, never an `O(M^2)` scan, one batched kernel.

## Distance semantics
`DefaultDistance.gene_distance` = homologue table + `distance_from_rows`; the two-phase `BaseDistance.prepare`/`measure` build the same table through `digest.match_rows` and share the identical tail, so speciation runs `prepare` once and `measure` per centre at the same bits.
Cost at pop 1000, per `measure` call over `4P = 4000` pairs: **232 traced ops, 31.3 ms → 7.8 µs per pair** (was 310 ops / 43.2 ms / 10.8 µs); `prepare` 114 ops, 3.9 ms; `Pipeline.step` 813 → 654 ms.
`match_rows` (49 ops node / 55 conn, 6.7 / 14.9 ms) folds the key columns into injective `2**31`-radix int64 composites (never `key_span`-derived), concatenates the centre's block with the digest's **sorted** block, `argsort`s the `2M` merge, and turns that permutation into its inverse with one `scatter`; every later step then stays at the centre's own width `M` — two `flat_gather`s read the successor in merged order and the composite it carries, and a homologue is exactly "the successor is the population composite `composite + 1`", with no key-row gather and no `reduce_min`.
That `+1` test needs no side-tag check: a centre composite is always even (tag `0`, and so is `_PAD_CENTRE`) and `_PAD_POPULATION` is even, so equality with an odd value already proves the neighbour is a real population gene.
`distance_from_rows` (70 ops node / 49 conn, 5.1 / 2.5 ms) fetches the aligned attribute rows with **one unmasked** `(P, M, A)` gather — `rows` is redirected to row `0` where it is absent, so no `I_INF` index reaches the backend and no `NaN` fill is needed — and counts homologous genes as `reduce_sum` of the mask itself instead of a second `valid_count` pass over the keys.
Non-homologous slots are zeroed **before** the attribute `reduce_sum` (a `NaN * 0` product would poison the sum), the attribute term uses `BaseGene.distance` (`|delta|` for float columns, `!=` for int option columns) and folds from the first column's term instead of an all-zero block (no `add`, no full-size `zeros` graph constant), and the quotient is `select`-guarded so no pair can be `inf`/NaN.

## Constraints
- No `jax`/`torch`/`numpy.random`; no `vmap`/`scan`/`while_loop`/`cond`; no Python loop over individuals — operators are already batched and never sit inside the `grad` graph, so `argsort` is legal here.
- Static Python `if` is used only for host configuration (probabilities, `network_type`, key width); every per-individual decision is a `select`/mask.
- evoxc's numpy backend raises on an out-of-range gather/scatter index, so every index is masked or clamped before it reaches the backend (`masked_flat_gather`/`masked_flat_scatter` from `common.tools`).
- Rank-0 integer constants are built with `enp.full((), value, dtype=evoxc.int32)` — an `int32` tensor plus a Python literal promotes to `int64`.
- Configuration violations raise `ConfigError` (a `TensorNEATError` subclass) at construction time; `evoxc` errors are never caught or re-wrapped.
- Files ≤ ~1000 lines (`mutation/default.py` is the largest at ~570).
- Traced-op budget is the lever here, but element volume is close behind: `match_rows` is `(P, 2M)` **int64** blocks (6.4 MB each at `4P`, `max_conns=100`), which is why the conn family costs twice the node family at the same op count.
- `measure` holds one `(P, M, A)` and one `(P, 2M)` temporary per call and the digest holds four arrays per family; peak RSS at pop 1000 is 2132 MiB (pop 500: 1129 MiB), so a new full-size temporary in this path is a real regression even when it saves ops.

## Deviations from the pre-rewrite library (intentional)
- The four structural operations are mutually exclusive through one shared draw; the legacy build gated each of them with an independent uniform and could apply several per individual.
- `choose_node_key` takes the static row counts `num_inputs`/`num_outputs` (the legacy signature passed `input_idx`/`output_idx` lists); the four private mutation helpers take the trailing gate.
- Alignment is the tagged-`argsort` merge over the concatenated key hashes (`_COLLISION_WINDOW`/`searchsorted` no longer exist); the collision window is closed by comparing the full key rows of the surviving pairs.
- Non-finite distance weights are rejected as well as negative ones, and `compatibility_disjoint`/`compatibility_weight` are validated at construction.

## Test Strategy
`test/unit/genome/test_mutation.py`, `test_crossover.py`, `test_distance.py` are the acceptance specs (their bodies are still `pytest.skip` placeholders, so they document the invariants rather than assert them).
Until those bodies land, the verified evidence is scratch-kernel runs through `test/support/runner.run_defn` with hand-built payloads and a duck-typed stand-in genome: shapes/dtypes, `I_INF`⇔NaN padding pairing, key uniqueness, caps over repeated generations, add-node split output preservation, duplicate/cycle/in-degree refusal, crossover key alignment against a host `{key tuple: row}` oracle, and distance identity/symmetry/monotonicity/batching/determinism.
Trace a bound method through a closure (`lambda *a: obj.method(state, *a)`) and build rank-0 RNG keys *inside* the trace (`evoxc.random.key(seed)`), because the runner cannot carry an `evoxc` key as data.

## Known Issues
- `pytest test/unit/genome` collects and reports all-skips: the module bodies never trace an operator, so no change in this node is gated by them — the evidence is a bit-exact A/B over `fitness_history`, the four payload arrays, the ten `SpeciesState` leaves and `randkey` per generation (method pointers are not stable hashes, so the run is the fingerprint).
- `test/support/oracle.py::schedule` still passes a `level=` keyword that `common.tensorization.Transformed` no longer declares (the field is `node_level`), so oracle-based forward checks cannot run until the test-support owner updates it.
- `BaseGenome.check_cycles` and the genome statics the mutation operator reads are implemented in `genome/base.py`; this node only calls them, so a mutation kernel traced against a real genome needs that node finished.
- Traced-op distribution of `execute_mutation` at pop 1000 (**1360 ops**, 13.9 ms at pop 100 / 36.5 ms at pop 500 / 65.7 ms at pop 1000): `mutate_structure` 1227 + `mutate_values` 132; inside the structural stage `utils.check_cycles` is **358** (49 hops × 5 ops) and `_mutate_add_conn` 553, against `_mutate_add_node` 391, `_mutate_delete_conn` 128, `_mutate_delete_node` 126, `_structural_gates` 28, `node_gene.mutate` 100. `execute_crossover` is 433 ops / 30.5 ms.
- `utils.check_cycles` keeps its frontier **flat** (`padded (P*(N+1),)` with one trailing dump column, the `edge_src`/`edge_dst`/dump flat index blocks built once), so one hop is four `O(E)` ops: `live = gather(padded, src_flat) & valid`, `padded = scatter(padded, select(live, dst_flat, dump_flat), live)` — never `masked_flat_gather`/`masked_flat_scatter`, whose escape checks and fill masks its indices cannot need (`_node_rows` already answers an in-range row, and a refused edge writes the dump column rather than clearing a real node).
- The remaining cost in this node is now `_mutate_add_node`'s 391 ops (`add_node` 52 + `delete_conn_by_pos` 59 + `fetch_random_index` 11 + two `new_identity_attrs` + gates) and `mutate_values`' 132 (`node_gene.mutate` 100 for two attribute columns), both dispatch-and-constant bound rather than work bound.

## Routing Table
| Path | Responsibility |
|---|---|
| `./alignment.py` | Shared exact-key homologue table: `find_homologous(keys1, keys2)` (hash + tagged `argsort` merge) and `key_rows(keys)` (rank-2/rank-3 key-table normaliser). |
| `./mutation/` | Structural (add/delete node, add/delete conn) and value mutation: `BaseMutation`, `DefaultMutation`, `RecurrentMutation`. |
| `./crossover/` | Outcome-preserving key alignment and attribute recombination: `BaseCrossover`, `DefaultCrossover`. |
| `./distance/` | Compatibility distance (disjoint/excess + attribute term): `BaseDistance`, `DefaultDistance`. |
