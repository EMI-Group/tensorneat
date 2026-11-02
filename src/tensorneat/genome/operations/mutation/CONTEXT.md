# mutation — plan-based structural and value mutation, batch-first

## Intent
Turn one population-wide RNG key into an edited gene payload: at most one structural change per individual (add/delete node, add/delete connection) gated by the operator's probabilities, followed by attribute mutation of every existing gene.
The stage split exists because the structural stage is integer-key twiddling over the payload while the value stage is float arithmetic delegated to the gene objects; `BaseMutation.__call__` runs both from one `split_n(key, 2)`.
Nothing here allocates keys: `new_node_keys (P,)` and `new_conn_keys (P, 3)` arrive caller-allocated.
That allocation is **not** guaranteed fresh: `algorithm/neat/neat.py::_new_gene_keys` bases it on the pre-crossover parent population while the payload mutated is the crossover child, so a child can already own the key it is handed (see Known Issues); `plan_add_node` therefore validates the allocation it receives.

## Cost model (drives every design choice here)
The backend is the evoxc reference numpy interpreter, so a traced op costs **dispatch (~10–15 µs) plus array traffic (~0.2 µs per 1 000 elements at pop 1000)**: both op count and element volume are budget.
Measured consequences that shape this node:
- Rank-0 constants are real ops: a per-use `i32(...)`/`enp.full((), ...)` is ~10 µs each, so every repeated constant is hoisted into `EditContext` and shared per stage.
- A full-payload op on `(P, E, CK)` (300 k elements) costs ~60–80 µs at pop 1000, so a helper that flattens/reshapes the same array four times is four times the traffic.
- For the recurrent `(P, tries, E)` slabs the **contiguous innermost reduce is the cheaper one** on this box (2 M elements: `axes=2` 0.82 ms vs the transposed `axes=1` 1.65 ms; `equal` 0.68 vs 1.11 ms) — the opposite of the incoming-window rule in `../../CONTEXT.md`, so do not transpose these slabs.
- A broadcast key match against the node table is the one formulation to avoid: `(P, 2E+2, N)` is 10.1 M elements at pop 1000 (plus an int32 cast and `argmax`), which cost 21 of the 23 ms the cycle predicate used to take.
- Guarding one plan is cheap when the reduce is over the node or connection axis (tens of elements per row): the add-node validation below is `(P, N)` + `(P, E)` equality/reduce pairs, 11 traced ops, unmeasurable in wall time.

## Two-phase structural stage
The stage **plans, then writes once**: `_edit_context` builds the shared masks/index blocks, `_plan_operations` resolves the four candidates independently (each a `NamedTuple` in `plan.py`, never a payload rewrite), `_merge_plan` folds the four mutually-exclusive plans into one `Edit`, and `plan.apply` performs one `select` per write slot plus one clearing `select` per array.
A slot's "no write" row is the array's own row count, which matches no row of the shared row-index constant, so no dump row, no flat index block and no per-call column constant is needed for the write path (`arrays.write_rows`).
The connection key and attribute arrays share their row layout, so `apply` builds one `(P, E, 1)` row mask per slot and both arrays consume it.
Every write slot is gated on its plan's `ok` (`_merge_plan`), so folding a validation into `ok` is a bit-exact no-op: nothing in the payload is rewritten and the value stage still runs for that individual.

## API Surface
`BaseMutation.__call__(state, genome, key, node_keys, node_attrs, conn_keys, conn_attrs, new_node_keys, new_conn_keys) -> GenomeTensors` — shape-validates the key tensors (`ConfigError`), then `mutate_structure` → `mutate_values`.
`BaseMutation.mutate_structure(...)` / `mutate_values(...)` are the public stages; the value stage equals calling `genome.node_gene.mutate` / `genome.conn_gene.mutate` once each on the full attribute tables and re-masking the rows that were NaN before.
`BaseMutation.structural_probabilities -> (node_add, node_delete, conn_add, conn_delete)` is the gate order a subclass must supply, and the extension point is the four plans: `plan_add_node(state, genome, key, context, node_keys, conn_keys, conn_attrs, new_node_keys, gate)`, `plan_delete_node(state, genome, key, context, node_keys, conn_keys, gate)`, `plan_add_conn(state, genome, key, context, node_keys, conn_keys, new_conn_keys, gate)`, `plan_delete_conn(state, genome, key, context, conn_keys, gate)`.
`DefaultMutation(conn_add=0.2, conn_delete=0.2, node_add=0.1, node_delete=0.1)` — each probability in `[0, 1]` else `ConfigError`; `RecurrentMutation(*, p_recur=0.1, max_conn_tries=20, **kwargs)` overrides `plan_add_conn` only, so it inherits `plan_add_node` and therefore the split validation.
Each plan reports its own `ok` flag **and** its candidate rows/values; it never writes the payload, so a refused operation is a batch-wide no-op by construction.
`arrays.py` is the flat-index toolkit: `first_row`/`random_row`/`masked_random_row` return a `(row, hit)` pair (no `I_INF` sentinel and no sentinel `select`), `take_rows`/`take_block`/`take_block_at` gather row blocks from a static base index, `write_rows`/`scatter_rows` write them, and `column_base`/`row_offsets`/`graph_constant`/`i32` build the static blocks. `recurrent.py` imports `first_row`, `i32`, `row_offsets`, `take_rows` from it — keep those names and signatures.
`degree.py` is the **diagnostic audit**, exported from the package as `in_degree(node_keys, conn_keys) -> (P, N) int32`, `max_in_degree(...) -> (P,) int32` and `violation(node_keys, conn_keys, limit) -> (P,) int32` (node rows strictly over `limit`).
It is trace-time only (call it from a graph, never as eager NumPy), it reads presence the way the write path does (`src key != I_INF`), it reports duplicate node keys as a union (both rows count every edge into that key — the same fold the schedule performs), and it is **never called by `execute_mutation`**: its `(P, N, E)` intermediate is 15 traced ops / 0.3–2.1 ms at pop 100–1000, i.e. diagnostic budget, not generation budget.

## Gate semantics (the load-bearing rule)
One `evoxc.random.uniform(key, (4, P))` draw per `mutate_structure` call; a row is a hit where the draw is below its probability, the first hitting row wins (`argmax` guarded by a `reduce_max` hit test, index `4` = "nothing"), and each plan receives one `(P,) bool` gate.
So the operations are mutually exclusive per individual and generation, at most one structural change applies, and no individual can both delete a node and delete a connection.
`_merge_plan` relies on that exclusivity: the per-array slot values are filled by priority-encoded `select` chains, and an individual whose gate is off writes the "no write" row, so its arrays come out bit-for-bit identical (`-0.0`, NaN included).
A probability of `0` is a true identity — the plan is elided at trace time, keeping even its cycle predicate out of the graph; `network_type` outside `{"feedforward", "recurrent"}` raises `ConfigError`.

## Structural rules
- Add-node preserves the split topology: the consumed connection becomes `old_src -> new -> old_dst`, the new node takes `node_gene.new_identity_attrs`, the identity edge `conn_gene.new_identity_attrs` (weight `1`), the copy edge the consumed connection's recorded attributes, and both new edges take the *first two rows that are free before the delete*. It is refused when there is no candidate connection, no free node row, or fewer than two free connection rows (`copy_row` has no hit) — a half-splittable genome is left untouched.
- Add-node **validates the caller's allocation**, because a split is the only write that can grow a destination's in-degree: it refuses the split when `new_node_keys` already equals a present node key (a duplicate key would fold two node rows onto one schedule slot) and when the consumed edge's destination is already at `genome.max_in_degree` incoming rows.
  The destination's post-write in-degree equals its pre-write count (the consumed row `in_key -> out_key` is replaced by the copy edge `new -> out_key`), so that second test is a no-op on every legal payload and only refuses to add an edge to a destination that is already over `K`; the new node itself always ends with in-degree `1`.
  Both tests fold into the plan's `ok` (`(P, N)` and `(P, E)` equality + reduce, 11 traced ops total, no new rank-0 constant), so a refused split leaves the payload bit-identical apart from the value stage.
- Add-conn draws a source (inputs and outputs allowed) and a destination (inputs refused) from one `(P, 2, N)` mask block, `masked_random_row`'s single `argmax`, and an "every slice has a candidate" test (`_fetch_two` = `reduce_max` over the node axis, then `reduce_min` over the two slices); a `reduce_min` over both axes would ask whether *every node row* is eligible and would refuse add-conn population-wide.
- Add-conn refuses an absent endpoint, a duplicate `(in_key, out_key)` pair, a destination already at `genome.max_in_degree` incoming edges, and — for `network_type == "feedforward"` — a cycle (`cycle.closes_cycle`).
- Delete-node never names an input row or an output row (`context.deletable_node`) and detaches every touching connection with one `(P, E)` mask applied after the write slots; delete-conn clears exactly one row.
- New rows always go to the first free row of the matching array; the `historical_marker` column is written only when `genome.conn_gene.uses_historical_marker`.
- `RecurrentMutation.plan_add_conn`: one `(P, max_conn_tries, 2)` `randint` tableau plus one `(P, max_conn_tries)` `uniform` `p_recur` decision; acceptance = present endpoints & destination is not an input & not a duplicate & in-degree ok; the first accepted attempt wins by a reduce over the attempt axis (never `cond`); **one** `closes_cycle` call on the chosen edge then enforces the `p_recur` requirement. `p_recur == 0` draws no `p_recur` uniform, degenerates to `DefaultMutation`, and elides the cycle predicate (next section). `max_conn_tries` bounds the tableau's element volume, not its op count.
- Padding never enters the recurrent confirmation slabs: a padded connection row is `I_INF` in *every* key column while an accepted candidate has two finite keys, so key equality alone excludes padded rows and the `present_conn` factor is redundant there (the `exists` term still gates a candidate whose endpoint *is* `I_INF`).

## Cycle predicate (`cycle.py`)
`closes_cycle(node_keys, conn_keys, src_keys, dst_keys)` answers "would `src -> dst` close a cycle?" as `src` reachable from `dst`, by one `while_loop` monotone frontier growth on a **flat** frontier with one trailing dump column (four `O(P·E)` ops per hop, once per real hop).
Key resolution is a **merge join**: every query and every node folds its key and payload into one injective `int64` composite, the `(P, N + Q)` merge (`Q = 2E + 2`) is sorted once, `cumsum` of the node tags gives the rank of the first node at or after each position, and one flat `gather` reads the row; the low field keeps queries in `[0, Q)` and nodes in `[Q, Q + N)` in ascending row order, so that first node is the *smallest* row carrying the key — the same row the previous `argmax(argmax(match))` produced.
No dense key→row table is possible: the algorithm allocates node keys as `max_node_key + 1 + arange(P)`, so a real pop-100 run reaches keys 225–249 with `key_span=150` and keeps growing; keys must be treated as arbitrary `int32`.
Result: 106 traced ops (up from 64) but 7.6 ms at pop 1000 (from 23.0), 4.2 ms at 500 and 1.4 ms at 100 — the join trades ~42 cheap ops for a 40× volume cut.
The call site is **statically gated** on `genome.network_type == "feedforward" or self.p_recur > 0.0`: those are the only two consumers (the acyclicity veto for a feedforward genome, and the `p_recur` requirement built only when `p_recur > 0`), so a recurrent genome with `p_recur == 0` traces neither the region nor its ops (`execute_mutation` 350 → 241 traced ops, 34.1 → 16.7 ms at pop 1000); `p_recur > 0` (361 ops) and feedforward (325) are unchanged.
A feedforward genome's add-conn can close a cycle, so that branch keeps the predicate unconditionally.

## Constraints
- Trace-time kernel only: no `jax`/`torch`/`numpy.random`, no `vmap`/`scan`/`cond`, no Python loop over individuals, no eager NumPy; `sort`/`argsort`/`cumsum` are legal (the mutation never enters the `grad` graph).
- Randomness is one rank-0 key per call, split with `evoxc.random.split_n`; draws are batched over the population, never per individual, and the draw order is part of the contract (see the bit-identity gate).
- A row fetch returns `(row, hit)` rather than an `I_INF` index (`arrays.first_row`/`random_row`), so a gate is a plain `logical_and` chain and the empty-row index is `0` — never a sentinel `select` plus a rank-0 constant.
- Padded rows keep their sentinels in both arrays: `I_INF` keys, NaN attributes; no operator may pair an `I_INF` key with finite attributes or a finite key with NaN attributes.
- Capacity invariants are hard: `max_nodes`, `max_conns`, `max_in_degree`. A refused operation is a no-op for that individual, never a partially applied write.
  `max_in_degree` is enforced at the write that could grow it: add-conn compares the destination's present row count against the limit before writing, add-node refuses a colliding allocation and a destination already over the limit, and the delete operations can only remove rows.
- Rank-0 constants and static index blocks are built **once** per stage (in `EditContext`, or as a folded `graph_constant`) and shared: a per-use `enp.full((), v, int32)` is one traced op each.
- Zero-width attribute tables (`attr_width == 0`) work: the write path uses row-index selects, so nothing builds a `0`-stride arange.
- Errors: configuration problems raise `ConfigError` (`TensorNEATError`); `evoxc` shape/backend errors propagate unchanged.
- Files ≤ ~1000 lines: `base.py` (738) owns the key split, the gate draw, the context, the plan merge and the value stage; `default.py` (411) the four `DefaultMutation` plans, `_fetch_two` and the split validation; `recurrent.py` (279) the recurrent add-conn; `cycle.py` (291) the merge-join reachability predicate; `plan.py` (266) the plan datatypes, `EditContext` and the single writer; `arrays.py` (381) the flat-index primitives; `degree.py` (118) the diagnostic in-degree audit.

## Measured cost (numpy reference interpreter, `DefaultGenome(num_inputs=2, num_outputs=1, max_nodes=50, max_conns=100)`, build excluded, one warmup + median of 5, RNG key created inside the trace)
| kernel | pop | traced ops | ms |
| --- | --- | --- | --- |
| `DefaultMutation` `execute_mutation` | 100 / 500 / 1000 | **462** | 5.6 / 13.6 / 24.0 |
| `RecurrentMutation` `execute_mutation` (`p_recur=0.1`) | 100 / 500 / 1000 | **498** | 6.0 / 15.2 / 28.3 |
| `RecurrentMutation` `execute_mutation` (`p_recur=0.0`, cycle elided) | 1000 | **241** (was 350) | 16.7 (was 34.1) |
| `degree.in_degree` (diagnostic) | 100 / 500 / 1000 | 15 | 0.29 / 1.08 / 2.11 |
The box drifts ±10–30 %, so op counts and same-process A/B ratios are the load-bearing numbers.
The add-node validation is +11 traced ops (406 → 417 at the reproducer's shapes, `max_nodes=16`): median-of-5 run time unchanged within noise at pop 100/500/1000 (5.9 / 17.1 / 29.8 ms).
Per-file op ledger (`mutledger2.py`, pop 100): default 462 = `cycle.py` 106 + `base.py` 74 + `arrays.py` 66 + `default.py` 54 + `plan.py` 17 + ~145 gene/tools-side; recurrent 498 = `cycle.py` 106 + `arrays.py` 76 + `base.py` 74 + `recurrent.py` 59 + `default.py` 21 + `plan.py` 17 + the same ~145.
Remaining reducible blocks, in order: `closes_cycle`'s 106 (its own 7.6 ms at pop 1000; inside it the int64 `sort` of the `(P, 252)` merge ~1.7 ms, the `cumsum` rank ~3.1 ms, the transpose-back `scatter` ~1.5 ms), the gene-side `node_gene.mutate`/`conn_gene.mutate` (~65 ops, owner `../gene/`), and `arrays.py`'s remaining static constants.
Whole-generation context: `Pipeline.step` at pop 100/500/1000 is 45 / 336 / 603 ms, of which mutation is now ~5.6 / 13.6 / 24 ms — `speciate` dominates a generation, so a mutation-only cut moves `step` by a few percent.
Element volume, recurrent add-conn at pop 1000: two `(P, max_conn_tries, E)` bool slabs (2 M elements each) and nothing else of that size — the int32 cast of the in-degree slab, the two `present_conn` factors and the second output-key equality are gone, and endpoint/attempt gathers run on the `(P·tries)` blocks.

## Known Issues
- **The caller can hand out a colliding node key** (outside this node): `algorithm/neat/neat.py::_new_gene_keys` (~601-674, the allocation at ~647-657) computes `max_node_key(state, state.pop.node_keys) + 1 + slot` from the **pre-crossover parent population**, while `_create_next_generation` mutates the **crossover child**, whose structure comes from a possibly different parent with a larger maximum key.
  Reproduced end to end at `DefaultGenome(num_inputs=3, num_outputs=1, max_nodes=16, max_conns=100)` + `OriginNode`/`OriginConn` + `NEAT(pop_size=500, species_size=10, survival_threshold=0.01)` + `XOR3d`, seed 6: generation 53, slot 65, allocation 112 while the child already owns key 112 → a duplicate node row, an identity edge into a destination already at `max_in_degree=4` → `edge_overflow=1` → `Pipeline.step` raised.
  Fuzz on the captured payload: add-node with the stale allocation violates 200/200 RNG keys and 0/200 with a child-based allocation; the other three operations are clean either way.
  The local guard (above) converts this into one skipped split per collision; the source fix is to base the allocation on the crossed payload (or a population-wide maximum).
- **Allocated keys outrun `key_span`** (outside this node): `key_span` defaults to `max(max_nodes, num_inputs + num_outputs) + max_conns` (116 here, 150 at `max_nodes=50`) while the allocator reaches 1 974–8 117 over 250 generations — at generation 53, 494 of 500 allocated keys are outside the span, and `common/tensorization.py::key_slot_table` treats an out-of-span key as a *miss* without setting `edge_overflow`, so those node rows and every edge touching them silently vanish from the evaluated phenotype.
  Genome/algorithm owners: either allocate below the span or carry a raised span in the shape signature (`tensorneat/genome/base.py` default, `algorithm/` allocation).
- Add-node does not preserve the network output — measured worst |Δoutput| 2.36e-01 on a fixed payload. It is a property of the gene-side identity row (`../../gene/node/default.py::new_identity_attrs` keeps `aggregation_default`/`activation_default`, which are not identity), not of this node; owner of `../gene/`: the inserted node needs the identity aggregation and an identity activation.
- The value stage mutates the rows the structural stage just wrote (both edges of a split get fresh noise), so "split preserves output" can only hold for `mutate_structure` in isolation.
- The cycle predicate is still traced and executed on every generation where `conn_add` can fire, even when no individual's add-conn gate is set: the gate is a traced value and cannot drive Python control flow. Only the statically dead case (`recurrent` + `p_recur == 0`) is elided.
- `RecurrentMutation`'s `closes_cycle` runs on the *chosen* attempt only rather than gating the choice per attempt: a must-cycle attempt that does not close a cycle is rejected instead of being skipped in favour of a later attempt, which biases sampling slightly (fewer accepted recurrent edges) to keep `max_conn_tries` out of the op budget.
- `RecurrentMutation`'s confirmation slabs are `O(P · max_conn_tries · max_conns)` bools; `max_conn_tries` (default 20) is the volume lever at pop ≥ 500 (tries 20 → 1 is ~1 ms at pop 1000 now), not an op lever.
- `p_recur == 0` still splits the key into `attempt_key`/`recur_key` although no `p_recur` draw happens; removing the split would change `attempt_key` and therefore the draws, so it is not behaviour-neutral.

## Test Strategy
`test/unit/genome/test_mutation.py` (15 tests) and `test_crossover.py` (9) are the acceptance spec and pass (`pytest test/unit/genome`, 95 passed), together with `test/parity/` (53 passed).
A change to this node is additionally gated by **bit-identical mutation output on fixed inputs**: `execute_mutation` for `DefaultGenome`+`DefaultMutation` and `RecurrentGenome`+{`DefaultMutation`, `RecurrentMutation`} on a real seed-42 evolved payload with `evoxc.random.key(12345)`, `new_node_keys = arange(P) + 1000`, `new_conn_keys = arange(3P).reshape(P, 3) + 5000`, compared NaN-aware per leaf over ≥ 40 RNG keys and over configurations (each structural probability 0/1 in turn, saturated node and connection tables, `historical_marker` 3-column keys, `attr_width == 0` genes).
The **capacity gate** is a soak: drive `Pipeline.step` on the reproducer family (`OriginNode`/`OriginConn`, pop 500, `max_nodes` 16 and 50, `XOR3d`) for ≥ 250 generations over ≥ 20 seeds and assert, host-side on the evaluated payload after every generation, that no destination key has more than `max_in_degree` present incoming rows — `degree.in_degree`/`violation` express the same check in-graph for a future test.
The last soak (46 runs × 250 generations: seeds 0–19 at `max_nodes=16`, 100–109 at `max_nodes=50`, plus fully-key-visible variants with a raised `key_span`) reports zero violations and zero raises, and seed 6 (which raised at generation 53 before the guard) completes 250 generations.
The trajectory gate is the 5-generation seed-42 XOR fitness history at pop 100 plus two same-seed pipelines producing identical fitness, and the four payload arrays / `SpeciesState` leaves / `randkey` per generation.
`pytest test/unit` still reports 5 pre-existing failures in `test/unit/algorithm/` (`test_neat_tell.py` ×4, `test_speciation.py` ×1), and `pytest test` adds 3 `test/integration/test_xor_run.py` failures; all eight fail identically on `b18716e` and are outside this node.

## Routing Table
| Path | Responsibility |
|---|---|
| `./base.py` | Gate draw, `EditContext` construction, plan dispatch/elision, plan merge, value stage, `__call__`. |
| `./default.py` | `DefaultMutation`: the four plans, `_fetch_two`, and the add-node allocation/in-degree validation. |
| `./recurrent.py` | `RecurrentMutation.plan_add_conn`: attempt tableau, `p_recur` requirement, statically gated cycle predicate. |
| `./cycle.py` | `closes_cycle`: one merge-join + monotone frontier `while_loop` acyclicity predicate. |
| `./plan.py` | Plan `NamedTuple`s, `EditContext`, `Edit` and the single `apply` writer. |
| `./arrays.py` | Flat-index toolkit: row fetches, block gathers, row writes, static index blocks, `i32`. |
| `./degree.py` | Diagnostic in-degree audit of a post-write payload (never in the hot path). |
