# NEAT (`tensorneat.algorithm.neat`) — population cycle + species bookkeeping

## Intent
Everything in a NEAT generation that is neither network mathematics (genome) nor task evaluation (problem): the population payload, the generation counter, speciation, stagnation, spawn allocation, parent pairing, and the crossover/mutation dispatch.
`NEAT` (`neat.py`) is the algorithm the pipeline drives; `SpeciesController` + `SpeciesState` (`species.py`) own the species algebra.
Every kernel here is batch-first over `P = pop_size` and compiled by `pipeline.py` into the single per-generation evoxc graph, so no function may branch on a traced value or read the host.
Measured on the numpy reference backend with the real `DefaultGenome(max_nodes=50, max_conns=100)` and `XOR`, `species_size=20`: the whole `tell` runs in ~77 ms/generation at `P=100`, ~653 ms at `P=500` and ~1176 ms at `P=1000`, of which `speciate` is 60 % / 88 % / 88 % and breeding 37 % / 12 % / 12 % (see `## Measured Cost`); the interpreter, not the data, dominates.
The layer's own work is `O(species_size × pop_size)` distance work in `speciate` plus `O(pop_size)` selection/crossover/mutation, and it never materialises a `max_nodes × max_nodes` tensor; the measured cost is the genome's per-pair distance kernel — one `(pop_size,)` row per species slot per phase, not a tensor size.

## Layout (four files, each under 1000 lines)
`species.py` — `SpeciesState` plus `SpeciesController`: the documented, traced-facing methods and the species algebra, each delegating its tensor body to `_kernels.py`.
`_kernels.py` — plain graph helpers (no `@evoxc.defn`, no state): spawn allocation, block layout, parent pairing, membership, sort/prefix-sum primitives, `gated` (the static lazy-row gate) and the two speciation scans.
`_validation.py` — host-side configuration validators shared with `neat.py` (`positive_int`, `non_negative_int`, `unit_interval`, `positive_float`), each raising `TensorNEATError` with the parameter name.
`neat.py` — `_sanitize_fitness` and `NEAT`: setup, ask/transform/forward, the `tell` algebra, `_batched` chunking, host-side `show_details`.

## API Surface
`__init__.py` re-exports `NEAT`, `SpeciesController`, `SpeciesState`.

`NEAT(BaseAlgorithm)` — `neat.py`:
`__init__(genome, pop_size, species_size=10, max_stagnation=15, species_elitism=2, spawn_number_change_rate=0.5, genome_elitism=2, survival_threshold=0.1, min_species_size=1, compatibility_threshold=2.0, species_fitness_func=evoxc.max, species_number_calculate_by="rank", pop_batch_size=None)` — constructor kwargs are API-compatible with the pre-rewrite library.
`setup(state=None) -> State` requires `state.randkey` (`TensorNEATError` otherwise), registers `pop_size` (static int), `pop` (`GenomeTensors`, axis 0 = individual), `generation` (0-d float32) and `species`, and advances `randkey`.
`setup` is **host-side**: it builds and runs one small init graph (`split(randkey)` → `genome.initialize` → species seeding) and cannot be called inside an active trace.
`ask(state) -> GenomeTensors` (zero ops, returns `state.pop`); `transform(state, pop=None)` / `forward(state, transformed, inputs, pop=None)` delegate to the genome; `tell(state, fitness) -> State` runs the algebra below.
`show_details(state, fitness) -> None` is the only host-side member; `num_inputs`/`num_outputs`/`pop_size` are read-only properties (`pop_size` backed by `_pop_size`).
Private members: `_sanitize_fitness` (module function), `_seed_species(self, cfg, first)`, `_static_state(state)`, `_build_options(state)`, `_create_next_generation(state, winner, loser, elite_mask)`, `_new_gene_keys(state)`, `_batched(kernel, key, *args)`, `_gene_counts(state, payload)`, `_gene_count_executable(state, payload)`.

`SpeciesController(StatefulBaseClass)` — `species.py`:
`setup(state, first) -> State` where `first` is **one** `GenomeTensors` individual (no population axis); slot 0 becomes the single live species keyed `0`, `member_count[0] = pop_size`, centre = `first`, `best_fitness = -inf`, `next_species_key = 1.0`.
`update_species(state, fitness) -> (state, winner (P,), loser (P,), elite_mask (P,))`; it owns the generation's first `evoxc.random.split` and stores the advanced key back into `state.randkey`.
`speciate(state, distance_fn, pop=None) -> State`; the payload comes from `pop`, else from the `"pop"` state entry, else from the four split payload entries, else `TensorNEATError`.
Private: `_species_fitness`, `_stagnation`, `_cal_spawn_numbers_by_rank`, `_cal_spawn_numbers_by_fitness`, `_pair_parents(species_state, selection_key, spawn_number, fitness)`, `_block_layout(spawn_number) -> (loc, idx_in_block)`, `_membership`, static `_population(state, pop=None)`, static `_gather_species(species_state, order)`, static `_sort_desc_legacy(x, axis=-1)`, static `_prefix_sum_exclusive(values)`.
Module constant `SPAWN_MODES = ("rank", "fitness")`.

`SpeciesState` — frozen dataclass, one leaf per field, stored under `state.species`:
`species_keys (S,) f32` (NaN = dead slot), `best_fitness (S,) f32`, `last_improved (S,) f32`, `member_count (S,) f32`, `idx2species (P,) f32` (NaN = unassigned, only inside `speciate`), `center_node_keys (S, N) i32`, `center_node_attrs (S, N, NA) f32`, `center_conn_keys (S, E, CK) i32`, `center_conn_attrs (S, E, CA) f32`, `next_species_key` (0-d f32).
Centres store the **four split payload arrays** (payload reconciliation #2), not packed gene rows, so `distance_fn` receives real `GenomeTensors`.
Every scalar field is float32 so NaN encodes a dead slot and `generation` writes into `last_improved` without a dtype change inside a scan carry.

## Generation Algebra (`NEAT.tell`)
1. Sanitise fitness: finite entries pass through untouched, NaN and both infinities become `-inf`.
2. `generation += 1` before the species update (legacy order): stagnation is judged against the generation that produced the fitness being consumed.
3. `SpeciesController.update_species` → species fitness, stagnation, descending sort, spawn allocation, parent pairs.
4. `_new_gene_keys` → per-genome unique `new_node_keys (P,)` and `new_conn_markers (P, 3)`.
5. Crossover: one `gather` per payload field for winner and loser, then batch-first `genome.execute_crossover(state, key, winners, losers)` (winner first — part of the contract).
6. Mutation: batch-first `genome.execute_mutation(state, key, crossed, new_node_keys, new_conn_markers)`, scan-chunked by `pop_batch_size`.
7. Elitism: `select(elite_mask[:, None, …], crossover_children, mutated_children)` per payload field.
8. `SpeciesController.speciate(state, genome.execute_distance)` rebuilds membership for the new population.
9. Commit `pop`, `randkey`, `generation`, `species`.

`_sanitize_fitness(fitness, pop_size)` implements the guard as `finite = ~isnan(f - f)` plus `select` (exact for NaN and both infinities, no `isfinite` op), and raises `TensorNEATError` unless `fitness.shape == (pop_size,)`.

## Species Algebra
Membership is one primitive reused everywhere: `M[i, p] = idx2species[p] == species_keys[i]`, shape `(S, P)`; NaN compares False, so dead slots and unassigned rows never match.
Species fitness: `select(M[i], fitness, -inf)` then `species_fitness_func(..., axes=1)` — masking before the reduce keeps a member-less slot at `-inf` instead of NaN.
Stagnation: `improved = f_s > best_fitness`; `stagnant = (f_s <= best_fitness) & (generation - last_improved > max_stagnation)`, disabled when the species' fitness rank is below `species_elitism`; a killed slot writes NaN into `species_keys`/`best_fitness`/`last_improved`/`member_count` and `-inf` into its species fitness.
Sort: stable ascending `argsort` plus the reversal permutation `n - 1 - arange(n)` (`_sort_desc_legacy`), then one `gather` per field (`_gather_species`); after the sort the slot index *is* the fitness rank.
Spawn (rank mode): `n = sum(valid)`, `rank_score = n - arange(S)`, `denominator = max((n+1)*n/2, 1)`, `target = floor(rank_score/denominator * pop_size)`.
Spawn (fitness mode): `f' = f_live - min_live(f_live) + 1`, `rate = f' / sum_live(f')`, `target = floor(rate * pop_size)`; the `-inf` best fitness of a freshly created slot is `nan_to_num`-ed to weight `0`.
Spawn (both modes): `damped = member_count + (target - member_count) * spawn_number_change_rate`, floored at `min_species_size`, dead/NaN slots forced to `0.0`, cast to int32, then the residual added to the **first live slot** by one `scatter` so `sum(spawn) == pop_size` exactly.
Parent pairing: members ranked by fitness descending, `survive_size = clamp(floor(survival_threshold * member_count), 1, min(member_count, pop_size))`, one `randint(key, (S, 2, P), 0, survive[:, None, None])` draw for father and mother, `flat_gather` of the ranking, winner decided in species space (ties to the father, `>=`), elite slots (`idx_in_block < genome_elitism`) then pinned to their species' best member.
Block flattening: with exclusive prefix `E` of `spawn`, `loc(p)` is the **last** `i` with `spawn[i] > 0 and E[i] <= p` (reversed `argmax` of the mask) and `idx_in_block(p) = p - E[loc(p)]`; parents are one `gather` at `flat = loc * P + idx_in_block` into the row-major `(S, P)` pair arrays.
Speciation, scan 1 (centre reservation): per live slot `i`, `d_i` is the `(P,)` row `distance(centre_i, pop)` (computed lazily behind a `cond` below the gate, see `## Lazy Distance Rows`), `closest = argmin(select(unassigned, d_i, +inf))`, then assign `idx2species[closest] = species_keys[i]`, re-elect `pop[closest]` as the centre and set its distance to `0`; a dead slot or an empty unassigned set writes nothing.
Speciation, scan 2 (member assignment): a live slot assigns every genome with `d < compatibility_threshold` and `(unassigned | d < current distance)`; a dead slot with anything still unassigned creates a species keyed `next_species_key` seeded by the first unassigned genome, then assigns the same way; an iteration with nothing to do writes nothing.
Speciation post-pass: leftovers adopt the last species key, slots created this generation get `best_fitness = -inf` / `last_improved = generation`, and `member_count` is one masked compare-sum over `(S, P)`.

## Lazy Distance Rows (`GATED_ROWS`, one `cond` per scan iteration)
Both phases need one `(P,)` compatibility row per species slot, and the row of a slot is known before the iteration reads it: phase 1's row is `distance(pre-scan centre slot, pop)` (scan 1 reads only the row it inherited and writes only its own), and phase 2's row is `distance(post-phase-1 centre slot, pop)` except for a slot that *creates*, whose centre is `pop[seed]` with `seed` the first unassigned genome.
`_kernels.speciate` computes each row lazily, one slot at a time, behind exactly one `evoxc.cond` per scan iteration — `gated_row(active, compute_row, operands)`, where `compute_row` builds the payload and calls `distance_fn` and the false branch is `enp.zeros((pop_size,), float32)`.
The predicate is exactly the mask that lets the iteration write — scan 1 `live & any_pending`, scan 2 `live | create` — so a skipped iteration's zero row is discarded by its own `claim` / `take` mask and the substitution is bit-identical.
Scan 2 selects its centre *before* the `cond` (`select(create, pop[seed], carry row)` per gene array), so a created species needs no branch of its own and only two compatibility-kernel copies exist in the whole graph.
The gate is static: `gated(species_size, pop_size)` is `species_size * pop_size <= GATED_ROWS` (6000), a plain Python predicate on static ints evaluated at trace time, never a traced branch.
Below the gate the lazy `cond` wins because the species table is sparse — 1–3 live slots of 20 at `P=8/100`, 15 at `P=300` — so 14–19 of the 20 rows per phase never run.
Above the gate both scans call `distance_fn(state, centre_payload(...), population)` unconditionally: the pre-gate behaviour, with no `cond` and no zero row, because a skipped slot would still leave its `if` region and carries in the graph and at `P=500` that region overhead costs more than the rows it skips would save (measured 26 % of the whole generation), while from `P=500` up every slot is live and nothing is skippable anyway.
`GATED_ROWS` is a module constant so the public `NEAT` constructor is unchanged.

## Scan Patterns (why fixed-length scans are exact)
`evoxc` de-sugars `scan` into `while` + `gather`/`scatter`, so a trip count must be static and the carry must keep identical dtypes and shapes: both speciation phases are `scan(f, carry, enp.arange(species_size))` with per-slot masks, never a data-dependent `while_loop`.
The legacy early-exit conditions survive as masks, and because every write is mask-then-single-`scatter` the extra iterations are provably no-ops.
Carries are kept narrow and packed (`idx2species` and the running distance as two `(P,)` arrays; the species keys plus `next_species_key` as one `(S+1,)` array) because wide `select`-bearing carries are a known iree codegen hazard.
No random draw happens inside a scan body — keys never travel through a carry.

## Gene Keys & Padding
`max_node_key(state, node_keys)` and `max_conn_marker(state, conn_keys)` return one maximum **per individual** (`(P,) int32`, `0` for an individual with no real row), so `_new_gene_keys` builds a per-individual base: `new_node_keys = base + 1 + arange(P)` (`(P,) int32`, no reshape) and `new_conn_markers = reshape(base_marker, (P, 1)) + 1 + arange(P * 3).reshape(P, 3)` when `conn_gene.uses_historical_marker`, else zeros.
Uniqueness is therefore guaranteed **within one individual's own keys** (keys are namespaced per individual; a key handed to one genome may coincide with a key held by another, which is legal because the mutation kernel only inspects an individual's own rows): each allocated key is strictly above that individual's own maximum, and the three markers of one individual are distinct from each other.
Node keys and connection markers live in different key columns, so a numerical coincidence between an allocated node key and an allocated marker is meaningless, and gene alignment compares full key rows (endpoints plus marker), so a marker that two individuals share never yields a false homologue.
The width `3` is a genome-level convention: one mutation call may add at most three connections per genome.
The `+1` is an int32 constant tensor on purpose: an `evoxc` addition against a Python int promotes the whole chain to int64, and the gene key columns are int32.
Validity is never inferred from packaging: this subtree calls `node_exists(node_keys)`, `conn_exists(conn_keys)`, `max_node_key` and `max_conn_marker` on the genome — the two validity helpers take only the key array and derive the padding sentinel themselves.
Dead species slots are recognised by `isnan(species_keys)` alone; their centre rows hold stale integers because int32 cannot carry NaN, and no kernel reads a centre gated by that mask.

## Divergences from the Pre-Rewrite Baseline (deliberate fixes)
1. Zero-survivor division: the old `(p_idx < survive_size) / survive_size` probability vector divided by zero for a one-member species; `randint` with `high = max(survive_size, 1)` yields two identical parents instead (self-crossover is the identity).
2. NaN-poisoned reductions: `"fitness"`-mode `min`/`sum` mixed in the NaN `best_fitness` of dead slots; every reduction now masks dead slots with `select` first.
3. NaN → int32 cast for `spawn_number`: dead and NaN slots are explicitly written to `0.0` before the cast instead of relying on cast semantics.
4. Configuration errors raise `TensorNEATError` instead of `assert`/`ValueError`.
5. Sort tie order is reproduced by a stable ascending `argsort` plus a reversal gather, because `argsort(descending=True)` breaks ties the other way.
6. Random streams differ by construction: one key per population kernel (drawing `(P, …)` randomness internally) instead of one key per genome, so parity tests compare kernels with fixed inputs and structural outcomes, not RNG bit-patterns.
7. The spawn residual goes to the first **live** slot instead of literally slot `0`: identical after the descending sort (live slots precede dead ones), but a hand-built unsorted table can no longer hand population slots to a dead species.

## Randomness Budget
`NEAT.setup` consumes one `evoxc.random.split` to separate the initialisation key from the key carried into generation 0 (setup only, once).
Per generation there are exactly two splits: `selection_key, rest = split(state.randkey)` inside `update_species` (parent selection only), then `cross_key, mut_key, next_randkey = split_n(rest, 3)` in `_create_next_generation` (crossover, mutation, next `randkey`).
`pop_batch_size` chunking pre-splits the mutation key with `split_n(key, n_batches)` and passes the stacked `(n_batches,)` int64 array as the scan `xs`, so each step receives a valid rank-0 key.
Keys are rank-0 scalars, so a batched kernel takes one key and draws its own population-shaped randomness; chunked and unchunked runs therefore draw different streams and `pop_batch_size` is part of the reproducibility config.

## Dependencies (cross-node contracts this subtree relies on)
`genome` (batch-first, one rank-0 key each): `setup(state)`, `initialize(state, key, pop_size) -> GenomeTensors`, `transform(state, node_keys, conn_keys)`, `forward(state, transformed, node_keys, node_attrs, conn_keys, conn_attrs, inputs)`, `execute_crossover(state, key, winners, losers)`, `execute_mutation(state, key, genome, new_node_keys, new_conn_markers)`, `execute_distance(state, centre_batch, pop_batch) -> (P,)` — the distance kernel is strictly row-independent and accepts **any** common batch width, never asserting `batch == pop_size` (speciation passes one `(pop_size,)` row per call through a lazy `cond`).
`genome` statics: `num_inputs`, `num_outputs`, `max_nodes`, `max_conns`, `node_gene.length`, `conn_gene.length`, `conn_gene.uses_historical_marker`; helpers `node_exists`, `conn_exists`, `max_node_key`, `max_conn_marker`.
`genome.utils.GenomeTensors` is the payload namedtuple — the four fields in declaration order.
`common`: `State` (immutable, `register`/`update`/`registered_keys`), `StatefulBaseClass`, `errors.TensorNEATError`, `tools.{rank_elements, fetch_first, argmin_with_mask, flat_gather, to_numpy}`.
`pipeline`: creates `randkey` with `evoxc.random.key(seed)`, calls `setup` on the host, drives `ask → transform → forward/evaluate → tell`, and calls `show_details` on host-readable values.

## Measured Cost (numpy backend, real `DefaultGenome(max_nodes=50, max_conns=100)`, XOR, `species_size=20`, this repo)
Whole `Pipeline.step` (one graph, build cached by the pipeline), one configuration per fresh process, median of 5 steps after 5 setup steps and a warmup, alternating the pre-change tree and this one on the same box:
`P=100` 240 / 240 -> 107 / 107 ms (2.25x), `P=300` 559 / 554 -> 456 / 452 ms (1.22x), `P=500` 830 / 834 and 806 / 806 -> 791 / 806 ms (parity), `P=1000` 1475 / 1811 -> 1471 / 1778 ms (parity; the spread inside each pair is run-to-run drift of this box, not a tree effect).
A generation-by-generation trace at `P=500` shows the two trees step-for-step identical (800–830 ms per step, identical live-node, live-connection, live-species and max-fitness trajectories).
Graph build (`pipeline.build_seconds`): 3.45 -> 3.96 s at `P=100`, 4.03 -> 4.23 s at `P=300`, 4.28 -> 4.15 s at `P=500`, 4.60 -> 4.65 s at `P=1000` — the gated path adds one kernel copy plus its region plumbing, ~15 % at the smallest population.
Phase split at the same configuration (separate executables, so the sum over-counts one generation): `evaluate` (`transform` + `forward` + `evaluate`) 30 / 135 / 280 ms, `update_species` (species fitness, stagnation, sort, spawn, pairing) 1.1 / 1.5 / 1.9 ms, crossover + mutation + elite select 29 / 76 / 142 ms, `speciate` 46 / 576 / 1032 ms at `P=100` / `500` / `1000`.
`tell` is 77 / 653 / 1176 ms and `speciate` is 60 % / 88 % / 88 % of it; at `P=100` the breeding kernel (37 % of `tell`, 29 ms) is now the second hot spot, from `P=500` up it is the 2 x 20 compatibility rows.
Live species slots in the measured states: 3 / 15 / 20 / 20 at `P=100 / 300 / 500 / 1000` — a small population's species table is mostly dead slots, which is why the lazy rows pay off there and nowhere else.
Per-row cost of `genome.execute_distance` (real payload shapes, standalone executable): ~2.5 ms of interpreter dispatch per call plus ~48 us per pair, flat from 100 to 20 000 rows, so in situ a row costs ~5.7 ms at `P=100` and ~33 ms at `P=1000`.
The distance kernel is genome-owned: the per-pair cost is the batch-wide key alignment (`find_homologous` sorts `2·M` key rows for every pair), which this node cannot change.
`NEAT.setup` (init kernel build + run): 40 / 38 / 47 ms at `P=100` / `500` / `1000`.
Against the declared legacy JAX baseline (51 / 109 / 157 ms per generation, measured in an earlier session) the cycle is ~2.1x / ~7.4x / ~9.4x over budget at `P=100` / `500` / `1000`.

## Known Issues
`next_species_key` must stay strictly greater than every live key; `setup` seeds it at `1.0` (slot 0 uses `0.0`) and each created species increments it, so the invariant holds as long as the counter is never rebound by hand — a colliding counter silently merges two species' `member_count`.
`test/unit/algorithm/test_spawn.py` encodes the brief's `loc(p) = #{i : spawn[i] > 0 and E[i] <= p} - 1`; that count is wrong for a live zero-spawn species in the middle of the table, so the oracle needs the "last qualifying block" form to agree with `block_layout` (cross-node item, `test/` is not writable from here).
`test/unit/algorithm/**` bodies are hard-coded `pytest.skip`, so the suite verifies nothing even against a working tree; this node is currently verified by driving `Pipeline` end-to-end on the numpy backend with the real `DefaultGenome`/`XOR` (pop 8/100/500/1000, `rank` and `fitness` spawn modes, `max_stagnation` kills, `OriginConn`'s `historical_marker` branch, `pop_batch_size` chunking, `RecurrentGenome`, XOR3d) plus bit-identical repeat runs.
`speciate`'s pair count is `2 · species_size · pop_size` and cannot be reduced without changing the algebra: each phase needs its own `(species_size, pop_size)` set of genuinely different centres, and a created species takes the first unassigned genome as its centre, so its row depends on every earlier assignment.
The remaining cost is therefore per-pair, not per-call: `DefaultDistance` re-sorts `2·M` key rows for every pair (`find_homologous`), i.e. the same genome is re-sorted `species_size` times per phase — a genome-side sorted-row reuse is the next lever and lies outside this node.
Liveness gating only pays while the species table is sparse, so it is confined below `GATED_ROWS`; a large population cannot exploit a sparse table even early in a run, when its species slots are still filling.
Above `GATED_ROWS` the graph must stay free of the `if` regions: an unconditional `cond` around the rows costs 26 % of the generation at `P=500` while skipping nothing, and chunked tables were measured slower still (one 20 000-row call at `P=1000` loses to 20 rows of `P`).
`_new_gene_keys` is per-individual by contract (see `## Gene Keys & Padding`), so the genome must not assume keys are globally unique across the population.

## Constraints
No `jax`/`torch`/`numpy.random` import; `import tensorneat` must stay free of them (release gate).
No branching, `bool()`, `len()` or host reads on traced values — static Python `if`/`for` over constructor configuration is fine and unrolls at trace time.
No `.at[]`, no `cumsum` on the compute path (prefix sums are triangular `dot`), no `where=`/`initial=`/`any`/`all`/`take`/`searchsorted`/`repeat`, no `vmap` over anything region-bearing, no `fori_loop`/`switch`.
Shapes are static: `P`, `S`, `max_nodes`, `max_conns` are fixed at build time; a change forces a rebuild by the pipeline.
Errors: configuration validation raises `common.errors.TensorNEATError` with a named message; trace/shape/backend errors propagate unchanged and are never re-wrapped or silently fallen back.
State keys touched: `randkey`, `pop_size`, `pop`, `generation`, `species`, plus the genome/problem keys documented above — nothing else.
Files stay under 1000 lines: `species.py` the species algebra, `_kernels.py` the graph helpers, `_validation.py` the host validators, `neat.py` the population cycle.

## Known Risks
`evoxc.random.randint` needs a symbolic `high` that stays ≥ 1 at runtime — guaranteed by `max(survive_size, 1)`, which also removes the legacy NaN case.
`evoxc.scatter` is replacement-only, so every "accumulate" update is restructured as reduce-then-scatter-once (spawn residual, stale-write elimination in the speciation scans).
Wide `while`/`scan` carries plus `select` can crash iree codegen (`AffinityAnalysis`), hence the packed-carry convention; the numpy reference backend is unaffected.
Reading a dead slot's centre rows would return stale integers, not NaN — always gate centre reads behind `isnan(species_keys)`.

## Test Strategy
Tests live in the sibling `test/` tree (read-only from here): `test/unit/algorithm/` covers the spawn/selection/stagnation algebra, the block layout and the speciation scans on fixed payloads.
`test/parity/algorithm/neat/` compares against `benchmarks/legacy_jax` on fixed gene payloads — identical selection/speciation outcomes for the deterministic steps, and only the divergences listed above allowed elsewhere.
`test/integration/` evolves XOR to its fitness target with a fixed seed and asserts run-to-run determinism for a `(seed, config, shape)` triple.
Every kernel here must be a pure function of `(state, key)`: no dict-order dependence, no host RNG, no float→int conversion without an explicit mask.
The lazy distance rows are verified by bit-identity of a whole run: `fitness_history` element-for-element plus sha256 of `node_keys`, `node_attrs`, `conn_keys`, `conn_attrs`, `species_keys`, `best_fitness`, `last_improved`, `member_count`, `idx2species`, the four centre arrays, `next_species_key` and `randkey`, compared against the pre-change tree at pop 100 and pop 8 (gated path) and pop 500 (ungated path) — `common.tools.to_numpy` first, because hashing an `evoxc.Tensor` directly hashes the object pointer rather than the data.

## Routing Table
| Path | Responsibility |
| --- | --- |
| `./neat.py` | `NEAT`: population cycle (`setup`/`ask`/`transform`/`forward`/`tell`), fitness sanitising, next-generation creation, `pop_batch_size` scan chunking, host-side `show_details`. |
| `./species.py` | `SpeciesState` (frozen dataclass) and `SpeciesController`: species fitness, stagnation, descending sort, spawn allocation, parent pairing, block layout, fixed-length speciation scans. |
| `./_kernels.py` | Plain graph helpers behind `species.py`: constants, payload helpers, sort/prefix-sum primitives, membership, spawn allocation, block layout, parent pairing, `GATED_ROWS`/`gated` (the static lazy-row gate) and the two speciation scans, payload reading. |
| `./_validation.py` | Host-side constructor validators shared by `species.py` and `neat.py`. |
| `./__init__.py` | Re-exports `NEAT`, `SpeciesController`, `SpeciesState`. |
