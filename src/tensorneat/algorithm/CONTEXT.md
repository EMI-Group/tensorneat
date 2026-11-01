# Algorithm (`tensorneat.algorithm`)

## Intent
The evolutionary layer: it owns the population, the ask → transform → forward → tell generation cycle, speciation, parent selection, the crossover/mutation dispatch, and the HyperNEAT CPPN → substrate decode.
Everything here runs **inside an evoxc trace** — `pipeline.py` builds one graph per shape signature covering a whole generation — except the explicitly host-side helpers (`__init__` validation, `show_details`, substrate construction).
This layer owns no network mathematics: genomes own `transform/forward/grad`, problems own `evaluate`, and the algorithm only shuffles whole populations with static shapes.
Cost profile: O(species_size × pop_size) gene-row distance work + O(pop_size) crossover/mutation per generation; no `max_nodes × max_nodes` tensor ever appears here.

## Kernel Contract (binding for this subtree)
Batch-first: population is axis 0, always; a kernel is written once over the whole batch.
`vmap` is never used — evoxc refuses to vectorize region-bearing control flow, so per-individual work would have to be rewritten anyway; the batch axis is spelled out explicitly instead.
Control flow (`evoxc.scan` / `evoxc.while_loop` / `evoxc.cond`) appears only at the outermost level of a kernel; every operation inside a loop body is a whole-batch tensor op.
Species loops are fixed-length `evoxc.scan`s over the static `species_size` axis with per-slot masks; no loop trip count is data-dependent.
`pop_batch_size` memory batching is an `evoxc.scan` over the batch axis of the mutation kernel only.
evoxc has no `.at[]`, no in-place mutation and no `cumsum`/`max(where=…)`/`sum(where=…)`/`any`/`all`/`take`/`searchsorted`/`repeat`: mask with `evoxc.select` before reducing, prefix-sum with `dot` against a static triangular ones matrix, and move data with flattened `evoxc.gather` / `evoxc.scatter`.
`evoxc.scatter` is replacement-only — an accumulating update is recomputed as "reduce first, scatter once".

## BaseAlgorithm Protocol (`base.py`)
`setup(state) -> State` registers the algorithm's tensors; it runs after the genome's and the problem's `setup` on the same threaded `State`.
`ask(state) -> pop` hands out the population to evaluate; for NEAT `pop = (pop_nodes (P, max_nodes, node_len), pop_conns (P, max_conns, conn_len))` and it costs zero ops.
`transform(state, pop) -> Transformed` decodes the **whole** population into one batch-first execution schedule (axis 0 = individual); the Pipeline never maps it per genome.
`forward(state, transformed, inputs) -> outputs` runs the decoded batch with `inputs (P, *input_shape)` and returns `(P, *output_shape)`.
`tell(state, fitness) -> State` consumes a `(P,)` fitness array (fitness is maximised) and returns the next-generation state.
`show_details(state, fitness) -> None` is the only host-side method: it converts with `common.tools.to_numpy` and prints (evoxc has no `io_callback`, and `runtime_call` exists on the numpy backend only).
`num_inputs`, `num_outputs`, `pop_size` are static Python ints; `Pipeline` asserts the first two against `problem.input_shape[-1]` / `output_shape[-1]`.

## State Layout
The algorithm registers exactly five keys on the shared `State`: `pop_size` (static Python int, so the batch-first `genome.initialize` can size axis 0), `pop_nodes`, `pop_conns`, `generation` (0-d float32 — kept float so it flows through `last_improved`), and `species`.
`randkey` is created by `Pipeline.setup` with `evoxc.random.key(seed)` and consumed here; the algorithm never seeds itself, and a `(seed, config, shape)` triple reproduces a run bit-for-bit.
`species` is a frozen dataclass `SpeciesState` (owned by `./neat`): `(species_size,)` float32 `species_keys`, `best_fitness`, `last_improved`, `member_count` (NaN = dead slot), `(pop_size,)` float32 `idx2species`, `(species_size, max_nodes, node_len)` int `center_nodes`, `(species_size, max_conns, conn_len)` int `center_conns`, and 0-d `next_species_key`.
Dead species slots are addressed solely by the `isnan(species_keys)` mask; the center gene rows of a dead slot hold stale integers that are never read, because the integer gene payload cannot carry NaN.
`generation` is incremented at the *start* of `tell` (legacy order), so stagnation is evaluated against the generation that produced the fitness being consumed.

## NEAT Generation Algebra (`tell`)
1. Sanitize fitness: NaN and ±inf → `-inf`, so a broken topology can never win selection.
2. Species fitness = per-species maximum of member fitness, computed as a masked reduce over the `(species_size, pop_size)` membership comparison `idx2species[None, :] == species_keys[:, None]`.
3. Stagnation: `stagnant = (f_species <= best_fitness) & (generation - last_improved > max_stagnation)`, disabled for the top `species_elitism` ranked species; a killed species is written to NaN / `-inf`.
4. Species sort by species fitness descending, applied to every `(species_size, …)` field with one gather over a permutation.
5. Spawn allocation: linear-rank (`species_number_calculate_by="rank"`) or fitness-proportional (`"fitness"`), damped by `spawn_number_change_rate`, floored at `min_species_size`, integerised, then error-corrected so `sum(spawn) == pop_size`.
6. Parent selection per species: members sorted by fitness descending, survivors = `clamp(floor(survival_threshold * members), 1, members)`, two parents drawn uniformly from the survivor prefix, elites (`genome_elitism`) pinned to the species' best member.
7. Block flattening: population slot `p` maps to species `loc(p) = #{i : spawn[i] > 0 and exclusive_prefix[i] <= p} - 1` with `idx_in_block(p) = p - exclusive_prefix[loc(p)]`, using a triangular-matmul prefix sum plus gather — never a host loop over species.
8. Winner/loser decided by fitness comparison; `genome.execute_crossover` then runs batch-first over the `(P, …)` pair.
9. Mutation gets per-genome unique `new_node_keys = max_node_key + 1 + arange(P)` and `new_conn_markers (P, 3)` (zeros when the conn gene carries no `historical_marker`), then runs batch-first, scan-chunked by `pop_batch_size`.
10. Elite slots keep the unmutated crossover children; `speciate` finally rebuilds species membership for the new population.

## Speciation Without vmap-over-Control-Flow (`./neat/species.py`)
`speciate` is two `evoxc.scan`s over `evoxc.numpy.arange(species_size)`, each carrying `(idx2species, center_nodes, center_conns, species_keys, o2c, next_species_key, ...)`:
- phase 1 (center reservation): for each live slot `i`, compute `d_i = genome.execute_distance(center_i, pop)` `(P,)`, take `closest = argmin(select(unassigned, d_i, +inf))`, assign `idx2species[closest] = species_keys[i]`, write `pop[closest]` as the new center and set `o2c[closest] = 0`.
- phase 2 (member assignment): a live slot assigns every genome with `d < compatibility_threshold` and `unassigned or d < o2c`; a dead slot creates a species keyed `next_species_key` seeded by the first unassigned genome, then assigns the same way.
- both phases are guarded: an iteration with no unassigned genome, or with a dead slot in phase 1, performs no writes (mask first, `scatter` once), so a fixed-length scan is equivalent to the legacy early-terminating `while_loop`.
- post-pass: still-unassigned genomes join the last species key, newly created slots get `best_fitness = -inf` / `last_improved = generation`, and `member_count` is a masked compare-sum `(species_size, pop_size)` reduce.
Rationale for scans instead of the legacy data-dependent `while_loop`: evoxc de-sugars `scan` to `while` + gather/scatter, so trip counts must stay static to codegen on compiler backends, and iree aborts on wide `while` carries (`select` plus ≥3–4 carried values).

## Randomness Budget
Per generation the algorithm performs exactly two `evoxc.random.split`s, yielding three subkeys consumed in this order: parent selection, crossover, mutation — the advanced parent key is stored back into `state.randkey`.
When `pop_batch_size` chunks a population kernel into a `scan`, the algorithm pre-splits `n_batches` keys with `evoxc.random.split_n` and passes the stacked `(n_batches,)` int64 array as the scan's `xs`, so each step receives a valid rank-0 key.
Key-form is whatever `splitmix64` produces (rank-0 int64); nothing here assumes a JAX `PRNGKey`.
There is no `random.choice` and no probability vector: uniform parent rank slots come from `evoxc.random.randint` with a broadcast symbolic `high`, which is also the fix for the legacy zero-survivor division.
No random draw happens inside a control-flow body, so keys never need to travel through a loop carry.

## Divergences from the Legacy JAX Baseline (deliberate fixes)
1. `survive_size == 0` (a 1-member species) produced NaN probabilities and killed `random.choice`; now `high = max(survive_size, 1)` yields two identical parents (self-crossover is the identity).
2. `species_number_calculate_by="fitness"` poisoned `min`/`sum` with the NaN `best_fitness` of dead slots; every reduction now masks dead slots first.
3. `HyperNEATNode.forward` disagreed with the node-gene protocol (missing `valid_mask`, `aggregation` called with one argument); the corrected contract is `forward(state, attrs, ins, valid_mask, is_output_node)` batch-first with `aggregation(ins, valid_mask)`. See `./hyperneat`.
4. `spawn_number` relied on NaN → int32 cast semantics; dead slots are explicitly zeroed before the cast.
5. `weight_threshold` was unvalidated while `1 - weight_threshold` divides; `__init__` now raises `TensorNEATError` for `weight_threshold ∉ [0, 1)`, `max_weight <= 0`, `species_size * min_species_size > pop_size`, a bad `species_number_calculate_by`, or a non-dividing `pop_batch_size`.
6. Sort tie-order stays bit-compatible with legacy by using a *stable ascending* `argsort` followed by an index reversal gather (evoxc `argsort(descending=True)` breaks ties the other way).

## Dependencies (cross-node contracts this subtree relies on)
`genome`: batch-first kernels `initialize(state, key)`, `transform(state, nodes, conns)`, `forward(state, transformed, inputs (P, …))`, `execute_crossover(state, key, w_nodes, w_conns, l_nodes, l_conns)`, `execute_mutation(state, key, nodes, conns, new_node_keys (P,), new_conn_markers (P, 3))`, `execute_distance(state, center_nodes, center_conns, pop_nodes, pop_conns) -> (P,)`.
`genome` RNG: every kernel takes **one** `evoxc.random` key and draws population-shaped randomness inside (`random.uniform(key, (P, n_attrs))`), because an evoxc key is a rank-0/2/4 scalar and cannot be indexed dynamically — this supersedes the plural `keys` wording in the package CONTEXT.
`genome`: static attributes `num_inputs`, `num_outputs`, `max_nodes`, `max_conns`, `node_gene.length`, `conn_gene.length`, `conn_gene.uses_historical_marker`, plus key/validity helpers `node_exists`, `conn_exists`, `max_node_key(state, pop_nodes)`, `max_conn_marker(state, pop_conns)` — the algorithm never inspects the padding encoding directly.
`genome/gene`: a payload builder that turns structural keys plus float attributes into gene rows (`BaseNode.from_attrs` / `BaseConn.from_attrs`), required by `./hyperneat/substrate` to synthesise the fixed substrate network.
`common`: `State` (immutable, evoxc-descended — a frozen dataclass or mapping so `evoxc.build` can flatten it), `errors.TensorNEATError`, `functions.{ACT, AGG, apply_activation, apply_aggregation}`, `tools.{rank_elements, fetch_first, argmin_with_mask, to_numpy}` in evoxc form.
`pipeline`: creates `randkey`, calls `setup`, drives `ask → transform → forward/evaluate → tell`, and calls `show_details` on host-readable values.
`problem`: `evaluate(state, keys, act_fn, transformed)` where `act_fn` is `algorithm.forward` partially applied — the problem owns the dataset tiling into the population axis.

## Constraints
No `jax` / `torch` import anywhere in this subtree; `import tensorneat` must stay free of them (release gate).
No Python branching, `bool()`, `len()` or host reads on traced values; static Python `if`/`for` over constructor configuration is fine and unrolls at trace time.
Constructor validation raises `tensorneat.common.errors.TensorNEATError`; trace/shape/backend errors propagate unchanged and are never re-wrapped or silently fallen back.
Nothing here reads `state` fields it did not register, except the documented genome/problem keys above.
Files stay under ~1000 lines; `species.py` carries speciation plus spawn/selection algebra, `neat.py` carries the population cycle, `hyperneat/` carries the CPPN→substrate decode.

## Test Strategy
Tests live in the sibling `./../../test` tree (read-only from here), mirroring this subtree: `test/unit/algorithm/` for the spawn/selection/stagnation algebra on fixed keys, `test/parity/algorithm/` for comparison against `benchmarks/legacy_jax` (documented divergences above are the only allowed mismatches), `test/integration/` for a full XOR run reaching its fitness target with a reproducible seed.
Every kernel here must be deterministic given `(seed, state)`: no dict ordering, no host-side RNG, no float-to-int conversion without an explicit mask.

## Routing Table
| Path | Responsibility |
| --- | --- |
| `./base.py` | `BaseAlgorithm` batch-first protocol: `setup`/`ask`/`transform`/`forward`/`tell`/`show_details` plus `num_inputs`/`num_outputs`/`pop_size`. |
| `./neat` | `NEAT` population cycle (ask/tell, elite handling, `pop_batch_size` scan batching) and `SpeciesController` + `SpeciesState` (speciation scans, stagnation, spawn allocation, parent pairing). |
| `./hyperneat` | `HyperNEAT` / `HyperNEATFeedForward` batch CPPN→substrate decode, `HyperNEATNode` / `HyperNEATConn` gene kernels, and the `substrate` package of host-only coordinate geometry. |
