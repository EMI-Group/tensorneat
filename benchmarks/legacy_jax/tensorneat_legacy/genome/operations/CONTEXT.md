# operations — genome-level evolutionary operators

## Intent
Mutation, crossover and compatibility-distance operators that act on whole genomes (node/conn arrays) rather than on single genes.
Each operator is a `StatefulBaseClass` called by `genome/base.py` (`execute_mutation`, `execute_crossover`, `execute_distance`) and vmapped over the population by `algorithm/neat/neat.py`.
All operators are written to be jittable/vmappable: branching is `jax.lax.cond`, per-row work is `vmap`, and NaN rows act as empty slots.

## Genome array conventions (shared by every operator here)
`nodes`: shape `(genome.max_nodes, node_gene.length)`, row `[0]` is the node index key, empty slots are all-NaN rows.
`conns`: shape `(genome.max_conns, conn_gene.length)`, row `[0]=input_index`, `[1]=output_index`, then `[2]=historical_marker` only when that gene declares it, then custom attrs (e.g. weight).
`I_INF` (int32 max) is the "no index found" sentinel returned by `fetch_first`/`fetch_random`; comparisons use `idx == I_INF`.
Custom attrs are the slice after `gene.fixed_attrs`, accessed via `genome.utils.extract_gene_attrs` / `set_gene_attrs`.
Empty slots are never compacted by mutation or crossover — deletions leave holes and insertion reuses the first NaN row.

## API Surface

### mutation/base.py — `BaseMutation`
`__call__(self, state, genome, randkey, nodes, conns, new_node_key, new_conn_key)` raises `NotImplementedError`.
Returns `(nodes, conns)` with the same shapes as the inputs.
`new_node_key` is a scalar `()` because at most one node is created per call.
`new_conn_key` is shape `(3,)` because at most three connections are created per call (two from add-node, one from add-conn).

### crossover/base.py — `BaseCrossover`
`__call__(self, state, genome, randkey, nodes1, nodes2, conns1, conns2)` raises `NotImplementedError`.
Returns `(new_nodes, new_conns)` with the winner's shapes `(max_nodes, node_gene.length)` / `(max_conns, conn_gene.length)`.

### distance/base.py — `BaseDistance`
`__call__(self, state, genome, nodes1, nodes2, conns1, conns2)` (no `randkey`) raises `NotImplementedError`.
Returns a scalar float32 compatibility distance.

### mutation/default.py — `DefaultMutation(BaseMutation)`
Constructor kwargs and defaults: `conn_add=0.2`, `conn_delete=0.2`, `node_add=0.1`, `node_delete=0.1` (positional or keyword).
`__call__` asserts `new_node_key.shape == ()` and `new_conn_key.shape == (3,)`, splits the key in two, and runs `mutate_structure` then `mutate_values`.
`mutate_structure` applies the four structural mutations independently and in the fixed order add-node → delete-node → add-conn → delete-conn, each guarded by `jax.lax.cond(r_i < p_i, ...)` so a single call may perform up to four structural changes.
The `if self.node_add > 0:` / `if self.conn_delete > 0:` guards are static Python branches (parameters are Python floats, not traced).
`mutate_add_node` picks a random existing connection, deletes it, appends one node (first NaN row) carrying `genome.node_gene.new_identity_attrs(state)`, and appends two connections (first NaN rows) `i_key→new_node_key` with `conn_gene.new_identity_attrs` and `new_node_key→o_key` with the deleted connection's original attrs (so the subnetwork is function-preserving).
`mutate_add_node` is skipped when no connection exists, or when free node space `< 1` or free conn space `< 2`; it consumes `new_conn_key[0]` and `new_conn_key[1]` as historical markers when the conn gene has them.
`mutate_delete_node` samples a node excluding input and output keys, NaNs its row, and NaNs every connection whose input or output equals that key (`jnp.where` over a broadcast mask).
`mutate_add_conn` samples an input key from all nodes (`allow_input_keys=True, allow_output_keys=True`) and an output key from hidden+output nodes (`allow_input_keys=False, allow_output_keys=True`), then rejects the add if the pair already exists (`fetch_first` on the (input,output) match) or conn space is exhausted.
For `genome.network_type == "feedforward"` `mutate_add_conn` additionally builds the `(N, N)` index matrix with `genome.utils.unflatten_conns` and rejects the pair when `common.check_cycles` reports a cycle; for `"recurrent"` no cycle check is done; any other value raises `ValueError` at trace time.
A successful add uses `conn_gene.new_zero_attrs(state)` (weight 0) and `new_conn_key[2]` as historical marker when applicable.
`mutate_delete_conn` samples one connection and NaNs its row (no-op when `idx == I_INF`).
`mutate_values` is the weight/bias step: it splits keys into `max_nodes` and `max_conns` subkeys, `vmap`s `node_gene.mutate` / `conn_gene.mutate` over every row, `vmap`s the attrs back with `set_gene_attrs`, then restores NaN rows with `jnp.where(jnp.isnan(...), jnp.nan, ...)`.
`choose_node_key(key, nodes, input_idx, output_idx, allow_input_keys=False, allow_output_keys=False)` returns `(key, idx)` using `fetch_random` over a validity mask, returning `I_INF`/NaN when nothing is selectable.
`choose_connection_key(key, conns)` returns `(i_key, o_key, idx)` from `fetch_random` over non-NaN rows.

### mutation/recurrent.py — `RecurrentMutation(DefaultMutation)`
Inherits `__call__`, `mutate_values`, `choose_node_key`, `choose_connection_key` unchanged; only `mutate_structure` is overridden.
Constructor is keyword-only (`def __init__(self, *, p_recur: float = 0.1, max_conn_tries: int = 20, **kwargs)`) and forwards `conn_add`/`conn_delete`/`node_add`/`node_delete` to `DefaultMutation`; it validates `0 <= p_recur <= 1` and `max_conn_tries >= 1`.
Add-node, delete-node and delete-conn are re-implemented but semantically identical to `DefaultMutation`.
Add-conn is replaced by `_mutate_add_conn_recurrent_optimized`: it computes `unflatten_conns`/`conns_exist_matrix` once outside the loop, then `jax.vmap`s an `attempt` closure over `max_conn_tries` split keys and takes `fetch_first(accept_flags)`.
Each attempt draws `(i_key, o_key)` plus `k_recur`; with probability `p_recur` the pair must form a cycle (`check_cycles`, via `force_recur & forms_cy`), otherwise any non-duplicate pair is accepted — so recurrence is biased, not required, and `network_type` is not inspected at all.
No candidate found within `max_conn_tries` leaves the genome unchanged.
Scheduling uses a separate key `k_schedule` (5-way split) for the four trigger uniforms, unlike `DefaultMutation` which draws triggers from the same subkeys it hands to the operators.
Not exported from `mutation/__init__.py` and not used by `RecurrentGenome` (which defaults to `DefaultMutation` with `network_type = "recurrent"`); it must be imported from this module explicitly.

### crossover/default.py — `DefaultCrossover(BaseCrossover)`
`__call__(state, genome, randkey, nodes1, conns1, nodes2, conns2)` where `genome1` is assumed to be the fitter parent.
Splits `randkey` into `max_nodes` / `max_conns` per-row keys and maps the node stage and the connection stage identically.
Homology is decided by the *whole* fixed-attr prefix (`nodes[:, :len(node_gene.fixed_attrs)]`, `conns[:, :len(conn_gene.fixed_attrs)]`), so with `OriginConn` the alignment key is `(input_index, output_index, historical_marker)`.
`_find_homologous_indices(keys1, keys2)` hashes each fixed-attr row with `common.hash_array`, `argsort`s genome2's hashes, `searchsorted`s genome1's hashes and scans a fixed `_COLLISION_WINDOW = 4` offsets for an exact `jnp.all(keys1 == keys2[orig], axis=1)` match; returns `(M,)` int32 with `I_INF` for no match, i.e. O(M log M) instead of the previous O(M²) scan.
`_crossover_with_index` applies `gene.crossover` on homologous pairs and keeps the winner's attrs verbatim when `homologous_idx == I_INF`.
Result is assembled with `set_gene_attrs` on top of `nodes1`/`conns1`, so gene sets present only in parent 2 (disjoint/excess genes of the loser) are dropped and array shapes stay the winner's.
Inheritance granularity comes from the gene: `DefaultNode`/`DefaultConn` do per-attribute uniform mixing (`jnp.where(normal>0, attrs1, attrs2)`), `OriginConn` draws one number and keeps one parent's whole attr vector.

### distance/default.py — `DefaultDistance(BaseDistance)`
Constructor kwargs: `compatibility_disjoint=1.0`, `compatibility_weight=0.4`.
`__call__(state, genome, nodes1, conns1, nodes2, conns2)` returns `gene_distance(nodes) + gene_distance(conns)` (node and conn distances are simply summed).
`gene_distance(state, gene, genes1, genes2)` concatenates both gene sets into `(2*max_rows, gene.length)`, lexsorts by fixed attrs (`jnp.lexsort(identifiers.T[::-1])`), appends one NaN row, and compares consecutive rows `fr, sr` to flag homologous pairs (`intersect_mask`, also requiring `~isnan(fr[:,0])`).
`non_homologous_cnt = cnt1 + cnt2 - 2 * sum(intersect_mask)` lumps excess and disjoint genes together — there is no separate excess/disjoint coefficient.
Homologous distance is `sum(gene.distance(...) * intersect_mask)` with NaN contributions zeroed (`jnp.where(jnp.isnan(hgd), 0, hgd)`); for conns `gene.distance` is `abs(w1 - w2)`, for nodes it is |Δbias| + |Δresponse| + agg/act mismatches.
Result is `(non_homologous_cnt * compatibility_disjoint + homologous_distance * compatibility_weight) / max(cnt1, cnt2)`, and exactly `0` when both sets are empty.

### "Origin" variants
There are no origin-specific operation classes in this directory; `OriginNode`/`OriginConn` genes are plugged into `DefaultMutation`/`DefaultCrossover`/`DefaultDistance` (see `test/origin_operations_test.py`).
The only behavioural switch is `"historical_marker" in genome.conn_gene.fixed_attrs`, checked inline in add-node and add-conn to decide whether to append the marker column, plus the marker allocation performed upstream in `algorithm/neat/neat.py`.

## JAX primitives
`jax.lax.cond` for every "do / do nothing" and success/failure branch; `jax.vmap` for per-row value mutation, per-row attrs extraction, per-gene crossover, per-pair gene distance and per-attempt recurrent sampling; `jax.random.split` for all key derivation; `jnp.argsort`/`jnp.searchsorted`/`jnp.lexsort` in the homology and distance alignments.
No `lax.fori_loop` or `lax.scan` lives in this directory — `fori_loop` appears in the imported helpers (`common.tools.hash_array`, `common.graph.check_cycles`) and `lax.scan`-style batching is done outside, by `NEAT._vmap_or_scan`.
Nothing in this directory is non-jittable; the non-jittable helper `genome/utils.py:re_cound_idx` (Python loops + `jax.device_get`) lives in the parent node and is not called from these operators.

## Constraints & Gotchas
The declared base signatures are stale relative to the implementations: `BaseCrossover` and `BaseDistance` document `... nodes1, nodes2, conns1, conns2` while `DefaultCrossover.__call__`, `DefaultDistance.__call__` and `genome/base.py:execute_crossover/execute_distance` all pass `... nodes1, conns1, nodes2, conns2` — a custom subclass copied from the base signature silently receives node/conn arrays swapped.
`DefaultMutation` reuses each of the four subkeys both to draw its trigger uniform (`jax.random.uniform(k1, shape=(4,))`) and as the operator's own `randkey`, so trigger and operator randomness are correlated; `RecurrentMutation` avoids this with a dedicated `k_schedule` key.
Insertion relies on `fetch_first(jnp.isnan(...))` finding a free row, so every add helper must be pre-guarded on remaining space or the last free row is overwritten.
`_find_homologous_indices` only probes 4 positions after each `searchsorted` hit, so more than 4 hash collisions at one bucket can lose a homologous gene.
Crossover can only shrink or preserve gene sets (winner-based assembly), never import loser-only genes.
The `"historical_marker"` checks are string-tuple membership tests on `genome.conn_gene.fixed_attrs`, evaluated at trace time — adding markers requires a gene class with that fixed attr, not an operator flag.
`RecurrentMutation` is importable but unexported and unused in-tree; the default recurrent path is `DefaultMutation` on `RecurrentGenome` (`network_type = "recurrent"` skips the cycle check).

## Routing Table
| Path | Contents |
| --- | --- |
| `./mutation/` | `BaseMutation` plus `DefaultMutation` (all four structural mutations and value mutation) and the `RecurrentMutation` variant. |
| `./crossover/` | `BaseCrossover` and `DefaultCrossover` (hash-based homology alignment, winner-only inheritance). |
| `./distance/` | `BaseDistance` and `DefaultDistance` (sorted-alignment compatibility distance). |

## Related (outside this node)
`genome/utils.py` provides `add_node`, `add_conn`, `delete_node_by_pos`, `delete_conn_by_pos`, `unflatten_conns`, `extract_gene_attrs`, `set_gene_attrs`.
`genome/gene/` defines the gene-level `mutate`/`crossover`/`distance`/`new_identity_attrs`/`new_zero_attrs` contracts consumed here.
`common/tools.py` provides `fetch_first`, `fetch_random`, `hash_array`, `I_INF`; `common/graph.py` provides `check_cycles`.
`algorithm/neat/neat.py` generates `new_node_keys` (`(pop_size,)`) and `new_conn_markers` (`(pop_size, 3)`) and vmaps `execute_mutation` / `execute_crossover` over the population.
