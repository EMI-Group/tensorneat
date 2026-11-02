# mutation — plan-based structural and value mutation, batch-first

## Intent
Turn one population-wide RNG key into an edited gene payload: at most one structural change per individual (add/delete node, add/delete connection) gated by the operator's probabilities, followed by attribute mutation of every existing gene.
The stage split exists because the structural stage is integer-key twiddling over the payload while the value stage is float arithmetic delegated to the gene objects; `BaseMutation.__call__` runs both from one `split_n(key, 2)`.
Nothing here allocates keys: `new_node_keys (P,)` and `new_conn_keys (P, 3)` arrive caller-allocated and population-unique.

## Two-phase structural stage (the design that sets the op budget)
The stage **plans, then writes once**: `_edit_context` builds the shared masks/index blocks, `_plan_operations` resolves the four candidates independently (each a `NamedTuple` in `plan.py`, never a payload rewrite), `_merge_plan` folds the four mutually-exclusive plans into one `Edit`, and `plan.apply` performs exactly **four `scatter`s plus one clearing `select`** over the payload.
That structure is what makes the traced-op count proportional to *distinct accesses* instead of to intermediate whole-array rewrites; under the evoxc reference interpreter an op costs ~80 µs regardless of element count, so op count is the whole budget.

## API Surface
`BaseMutation.__call__(state, genome, key, node_keys, node_attrs, conn_keys, conn_attrs, new_node_keys, new_conn_keys) -> GenomeTensors` — shape-validates the key tensors (`ConfigError`), then `mutate_structure` → `mutate_values`.
`BaseMutation.mutate_structure(...)` / `mutate_values(...)` are the public stages; the value stage equals calling `genome.node_gene.mutate` / `genome.conn_gene.mutate` once each on the full attribute tables and re-masking the rows that were NaN before.
`BaseMutation.structural_probabilities -> (node_add, node_delete, conn_add, conn_delete)` is the gate order a subclass must supply, and the extension point is the four plans: `plan_add_node(state, genome, key, context, node_keys, conn_keys, conn_attrs, new_node_keys, gate)`, `plan_delete_node(state, genome, key, context, node_keys, conn_keys, gate)`, `plan_add_conn(state, genome, key, context, node_keys, conn_keys, new_conn_keys, gate)`, `plan_delete_conn(state, genome, key, context, conn_keys, gate)`.
`DefaultMutation(conn_add=0.2, conn_delete=0.2, node_add=0.1, node_delete=0.1)` — each probability in `[0, 1]` else `ConfigError`; `RecurrentMutation(*, p_recur=0.1, max_conn_tries=20, **kwargs)` overrides `plan_add_conn` only.
Each plan reports its own `ok` flag **and** its candidate rows/values; it never writes the payload, so a refused operation is a batch-wide no-op by construction.

## Gate semantics (the load-bearing rule)
One `evoxc.random.uniform(key, (4, P))` draw per `mutate_structure` call; a row is a hit where the draw is below its probability, the first hitting row wins (`argmax` guarded by a `reduce_max` hit test, index `4` = "nothing"), and each plan receives one `(P,) bool` gate.
So the operations are mutually exclusive per individual and generation, at most one structural change applies, and no individual can both delete a node and delete a connection.
`_merge_plan` relies on that exclusivity: the per-array slot values are filled by priority-encoded `select` chains, and an individual whose gate is off writes the dump row, so its arrays come out bit-for-bit identical (`-0.0`, NaN included).
A probability of `0` is a true identity — the plan is elided at trace time, keeping even its cycle predicate out of the graph; `network_type` outside `{"feedforward", "recurrent"}` raises `ConfigError`.

## Structural rules
- Add-node preserves the split topology: the consumed connection becomes `old_src -> new -> old_dst`, the new node takes `node_gene.new_identity_attrs`, the identity edge `conn_gene.new_identity_attrs` (weight `1`), the copy edge the consumed connection's recorded attributes, and both new edges take the *first two rows that are free before the delete*. It is refused when there is no candidate connection, no free node row, or fewer than two free connection rows (`copy_row` has no hit) — a half-splittable genome is left untouched.
- Add-conn draws a source (inputs and outputs allowed) and a destination (inputs refused) from one `(P, 2, N)` mask block and one `argmax`, then refuses an absent endpoint, a duplicate `(in_key, out_key)` pair, a destination already at `genome.max_in_degree` incoming edges, and — for `network_type == "feedforward"` — a cycle (`cycle.closes_cycle`).
- Delete-node never names an input row or an output row (`context.deletable_node`) and detaches every touching connection with one `(P, E)` mask applied after the scatter; delete-conn clears exactly one row.
- New rows always go to the first free row of the matching array; the `historical_marker` column is written only when `genome.conn_gene.uses_historical_marker`.
- `RecurrentMutation.plan_add_conn`: one `(P, max_conn_tries, 2)` `randint` tableau plus one `(P, max_conn_tries)` `uniform` `p_recur` decision; acceptance = present endpoints & destination is not an input & not a duplicate & in-degree ok; the first accepted attempt wins by a reduce over the attempt axis (never `cond`); **one** `closes_cycle` call on the chosen edge then enforces the `p_recur` requirement. `p_recur == 0` draws no `p_recur` uniform and degenerates to `DefaultMutation`. `max_conn_tries` bounds the tableau's element volume, not its op count.

## Constraints
- Trace-time kernel only: no `jax`/`torch`/`numpy.random`, no `vmap`/`scan`/`cond`, no Python loop over individuals, no eager NumPy; `argsort`/`sort` are unnecessary here and the mutations never enter the `grad` graph.
- Randomness is one rank-0 key per call, split with `evoxc.random.split_n`; draws are batched over the population, never per individual. A row fetch returns `(row, hit)` rather than an `I_INF` index (`arrays.first_row`/`arrays.random_row`), so a gate is a plain `logical_and` chain and the empty-row index is `0` — never a sentinel `select` plus a rank-0 constant.
- Padded rows keep their sentinels in both arrays: `I_INF` keys, NaN attributes; no operator may pair an `I_INF` key with finite attributes or a finite key with NaN attributes.
- Capacity invariants are hard: `max_nodes`, `max_conns`, `max_in_degree`. A refused operation is a no-op for that individual, never a partially applied write.
- Rank-0 constants and static index blocks are built **once** per stage (in `EditContext`, or as a folded `graph_constant`) and shared: a per-use `enp.full((), v, int32)` is one traced op each.
- Scatter semantics: `evoxc.scatter` resolves duplicate indices last-write-wins, which is how `apply` orders its per-array slots (delete, then identity, then copy) and why the trailing dump row can absorb "no write".
- Errors: configuration problems raise `ConfigError` (`TensorNEATError`); `evoxc` shape/backend errors propagate unchanged.
- Files ≤ ~1000 lines: `base.py` (741) owns the key split, the gate draw, the context, the plan merge and the value stage; `default.py` (379) the four `DefaultMutation` plans; `recurrent.py` (232) the recurrent add-conn; `arrays.py` (260) the flat-index primitives; `plan.py` (220) the plan datatypes and the single writer; `cycle.py` (198) the reachability predicate.

## Measured cost (numpy reference interpreter, `DefaultGenome(num_inputs=2, num_outputs=1, max_nodes=50, max_conns=100)`, build excluded, one warmup + 5 timed runs)
| variant | pop | traced ops (was) | ms (was) |
| --- | --- | --- | --- |
| `DefaultMutation` | 100 | **514** (1360) | 10.3 (14.7) |
| `DefaultMutation` | 500 | **514** (1360) | 30.5 (53.0) |
| `DefaultMutation` | 1000 | **514** (1360) | 48.8–63.5 (99.0) |
| `RecurrentMutation` | 1000 | **616** (1141) | 72.5 (39.0) |
The `DefaultMutation` figure is a 2.65× op cut; the box drifts ±30 %, so the op counts are the load-bearing numbers.
Remaining distribution of the 514: `mutate_values` 134 (of which the gene-side `node_gene.mutate` ≈100 and `conn_gene.mutate` 27 — see "Outside this node"), `plan_add_conn` ≈132 (**68 of them the cycle predicate**), `_merge_plan`+`apply` ≈100, `plan_add_node` ≈52, `_edit_context` ≈24, `_structural_gates` 19, `plan_delete_node` ≈21, `plan_delete_conn` ≈12.
`RecurrentMutation` is cheaper in ops but *slower* at pop ≥ 500 than at HEAD: its `(P, max_conn_tries, E)` duplicate/in-degree blocks are 2 M elements each (20 tries), so element volume, not op count, dominates there — a `max_conn_tries` reduction is the lever, not more fusion.

## Known Issues
- **Add-node does not preserve the network output** — measured worst |Δoutput| 2.36e-01 on a fixed payload, and the *same* 2.36e-01 at HEAD, so it is a property of the gene-side identity row (`gene/node/default.py::new_identity_attrs` keeps `aggregation_default`/`activation_default`, which are not identity), not of this node. Owner of `gene/`: to make the split output-preserving the inserted node needs the identity aggregation and an identity activation.
- The value stage mutates the rows the structural stage just wrote (both edges of a split get fresh noise). That is the stage order at HEAD too, so it is not a regression — but it means "split preserves output" can only hold for `mutate_structure` in isolation.
- `RecurrentMutation`'s `closes_cycle` runs on the *chosen* attempt only (one 68-op predicate) rather than gating the choice per attempt: a must-cycle attempt that does not close a cycle is rejected instead of being skipped in favour of a later attempt, which biases sampling slightly (fewer accepted recurrent edges) to keep `max_conn_tries` out of the op budget.

## Outside this node (measured, for the owners)
- `gene/node/default.py::mutate` is ≈100 traced ops — the largest single block left in `execute_mutation` (two attribute columns, each a draw → clip → `select` chain). `gene/conn/default.py::mutate` is 27. Together they are 25 % of the kernel.
- `gene/node/default.py::new_identity_attrs` emits 10 ops and `gene/conn/default.py::new_identity_attrs`/`new_zero_attrs` 4 each (rank-0 constants per column); folding those rows into one `enp.stack`/`graph_constant` would save ~10 more.
- `common/tools.py::fetch_random`/`fetch_first` cost 11/6 ops because of the sentinel `select`; `arrays.py` carries the cheaper `(row, hit)` pair used here, which the rest of `genome/` could adopt.
- `_structural_gates` (19) spends 6 ops on rank-0 constants; a shared constant bundle in `common` would shave a few ops from every kernel that compares against `I_INF`.

## Test Strategy
`test/unit/genome/test_mutation.py` (15 real tests) and `test_crossover.py` (9) pass against this implementation, as does `test/parity/` (53): `pytest test/unit/genome test/parity` = 148 passed.
Beyond the suites, the acceptance evidence for a change here is three throwaway scripts run on the numpy backend: (a) 8 seeds × 64 individuals of invariant checks — padding agreement, `max_nodes`/`max_conns` caps, unique node keys and `(in_key, out_key)` pairs, in-degree ≤ `max_in_degree`, input/output rows never moved or cleared, at most one structural change per individual (0 violations); (b) a `Pipeline` smoke run (6 generations, deterministic across two pipelines with the same seed); (c) an op-count and timing harness whose counts are the numbers in "Measured cost".
