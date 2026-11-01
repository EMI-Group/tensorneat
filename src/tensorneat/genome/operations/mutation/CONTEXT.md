# mutation — structural and value mutation, batch-first

## Intent
Turn one population-wide RNG key into an edited gene payload: at most one structural change per individual (add/delete node, add/delete connection) gated by the operator's probabilities, followed by attribute mutation of every existing gene.
The stage split exists because the structural stage is integer-key twiddling over the payload while the value stage is float arithmetic delegated to the gene objects; `BaseMutation.__call__` runs both from one `split_n(key, 2)`.
Nothing here allocates keys: `new_node_keys (P,)` and `new_conn_keys (P, 3)` arrive caller-allocated and population-unique.

## API Surface
`BaseMutation.__call__(state, genome, key, node_keys, node_attrs, conn_keys, conn_attrs, new_node_keys, new_conn_keys) -> GenomeTensors` — shape-validates the key tensors (`ConfigError`), then `mutate_structure` → `mutate_values`.
`BaseMutation.mutate_structure(...)` / `mutate_values(...)` are public stages; the value stage equals calling `genome.node_gene.mutate` / `genome.conn_gene.mutate` once each on the full attribute tables and re-masking the rows that were NaN before.
`BaseMutation.structural_probabilities -> (node_add, node_delete, conn_add, conn_delete)` is the gate order a subclass must supply.
`DefaultMutation(conn_add=0.2, conn_delete=0.2, node_add=0.1, node_delete=0.1)` — each probability in `[0, 1]` else `ConfigError`; `RecurrentMutation(*, p_recur=0.1, max_conn_tries=20, **kwargs)` adds the cycle-directed add-connection.
`choose_node_key(key, node_keys, num_inputs, num_outputs, allow_input_keys=False, allow_output_keys=False) -> (keys, rows)` and `choose_connection_key(key, conn_keys) -> (in_keys, out_keys, rows)` return `I_INF` in every output when nothing is eligible; the returned key is the key at the drawn row.
The four private helpers (`_mutate_add_node`, `_mutate_delete_node`, `_mutate_add_conn`, `_mutate_delete_conn`) take a trailing `(P,) bool` gate and return `GenomeTensors`.

## Gate semantics (the load-bearing rule)
One `evoxc.random.uniform(key, (4, P))` draw per `mutate_structure` call; a row is a hit where the draw is below its probability, the first hitting row wins (`argmax` guarded by a `reduce_max` hit test, index `4` = "nothing"), and each helper receives the resulting one-hot gate.
So the operations are mutually exclusive per individual and generation, at most one structural change applies, and no individual can both delete a node and delete a connection.
Where a gate is false the helper writes nothing: `genome.utils`' writers are unconditional at a row, so every helper composes its result as `select(gate, written, original)`, which returns the original row bit-for-bit (`-0.0`, NaN included).
A probability of `0` produces an all-False row and is therefore a true identity; `network_type` outside `{"feedforward", "recurrent"}` raises `ConfigError`.

## Structural rules
- Add-node preserves the individual's output: delete the chosen connection, write the new node at `first_free_row(node_keys)` with `node_gene.new_identity_attrs`, add the identity edge (old source → new node, `conn_gene.new_identity_attrs`) and the copy edge (new node → old destination, recorded original attributes) at two successive free connection rows (the second `first_free_row` runs on the array after the first write, so the rows differ). Refused when there is no candidate connection, no free node row, or fewer than two free connection rows *before* the delete — a half-splittable genome is left untouched.
- Add-conn draws a source (inputs and outputs allowed) and a destination (inputs refused) from two split keys, then refuses an `I_INF` endpoint, a duplicate `(in_key, out_key)` pair, a cycle when `network_type == "feedforward"` via `genome.check_cycles`, and a destination already at `genome.max_in_degree` incoming edges.
- Delete-node never names an input row or an output row and detaches every touching connection with `delete_conns_by_key`; delete-conn clears exactly one row.
- Key layout: new rows always go to `first_free_row` for keys and attributes alike, key rows are built as `(P, CK) int32` with `enp.full((), v, dtype=evoxc.int32)` constants, and the `historical_marker` column is written only when `genome.conn_gene.uses_historical_marker`.
- `RecurrentMutation` overrides only add-conn: one `randint(key, (P, max_conn_tries, 2), 0, N)` tableau plus one `uniform(key, (P, max_conn_tries))` `p_recur` decision, acceptance = present endpoints & not duplicate & in-degree ok & (must-cycle ⇒ `check_cycles`), first accepted attempt via a reduce over the attempt axis, edge written with `conn_gene.new_zero_attrs`. `p_recur == 0` draws no `p_recur` uniform and degenerates to `DefaultMutation`.

## Constraints
- Trace-time kernel only: no `jax`/`torch`/`numpy.random`, no `vmap`/`scan`/`while_loop`/`cond`, no Python loop over individuals, no eager NumPy; `argsort` is unnecessary here and the mutations never enter the `grad` graph.
- Randomness is one rank-0 key per call, split with `evoxc.random.split_n`; draws are batched over the population, never per individual.
- Padded rows keep their sentinels in both arrays: `I_INF` keys, NaN attributes; no operator may pair an `I_INF` key with finite attributes or a finite key with NaN attributes.
- Capacity invariants are hard: `max_nodes`, `max_conns`, `max_in_degree`. A refused operation is a no-op for that individual, never a partially applied write.
- Errors: configuration problems raise `ConfigError` (`TensorNEATError`); `evoxc` shape/backend errors propagate unchanged.
- Files ≤ ~1000 lines: `base.py` (~410) owns the key split, gate draw and value stage, `default.py` (~570) the four helpers plus the two `choose_*_key` draws, `recurrent.py` (~300) the recurrent add-conn.

## Routing Table
| File | Responsibility |
|---|---|
| `./base.py` | `BaseMutation`: key split, shared `(4, P)` gate draw (`_structural_gates`), stage ordering, batched value mutation, `_gated`/`_broadcast_row`/`_int32` helpers. |
| `./default.py` | `DefaultMutation`: the four gated structural helpers, `choose_node_key`, `choose_connection_key`, probability validation, `_edge_keys`/`_attribute_row` row builders. |
| `./recurrent.py` | `RecurrentMutation`: cycle-directed add-connection (`p_recur`, `max_conn_tries`) and its attempt-tableau acceptance reduce. |
