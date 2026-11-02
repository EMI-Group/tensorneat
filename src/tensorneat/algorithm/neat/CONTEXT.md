# NEAT (`tensorneat.algorithm.neat`) — population cycle + species bookkeeping

## Intent
Everything in a NEAT generation that is neither network mathematics (genome) nor task evaluation (problem): the population payload, the generation counter, speciation, stagnation, spawn allocation, parent pairing, and the crossover/mutation dispatch.
`NEAT` (`neat.py`) is the algorithm the pipeline drives; `SpeciesController` + `SpeciesState` (`species.py`) own the species algebra; `_distance.py` owns the compatibility-distance rows over one generation's population digest.
Every kernel here is batch-first over `P = pop_size` and compiled by `pipeline.py` into the single per-generation evoxc graph, so no function may branch on a traced value or read the host.
Measured on the numpy reference backend with the real `DefaultGenome(max_nodes=50, max_conns=100)` and `XOR`, `species_size=20`: `tell` runs in ~58 / ~393 / ~732 ms per generation at `P=100 / 500 / 1000`, of which `speciate` is 46 % / 78 % / 79 % and breeding 52 % / 21 % / 20 % (see `## Measured Cost`); the interpreter, not the data, dominates.
The layer's own work is one population digest plus `2 · ceil(species_size / 4)` distance calls in `speciate` and `O(pop_size)` selection/crossover/mutation; it never materialises a `max_nodes × max_nodes` tensor.

## Layout (five files, each under 1000 lines)
`species.py` — `SpeciesState` plus `SpeciesController`: the documented, traced-facing methods and the species algebra, each delegating its tensor body to `_kernels.py`.
`_kernels.py` — plain graph helpers (no `@evoxc.defn`, no state): spawn allocation, block layout, parent pairing, membership, sort/prefix-sum primitives, `gated` (the static lazy-row gate) and the two speciation scans.
`_distance.py` — `DistanceRows` (the digest-backed, population-shaped `distance_fn` `speciate` consumes) and the `repeat_leading` tiling primitive.
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
Private members: `_sanitize_fitness` and `_finite_bound` (module functions), `_seed_species(self, cfg, first)`, `_static_state(state)`, `_build_options(state)`, `_create_next_generation(state, winner, loser, elite_mask)`, `_new_gene_keys(state)`, `_batched(kernel, key, *args)`, `_gene_counts(state, payload)`, `_gene_count_executable(state, payload)`, `_distance_rows(digest, population)`.

`SpeciesController(StatefulBaseClass)` — `species.py`:
`setup(state, first) -> State` where `first` is **one** `GenomeTensors` individual (no population axis); slot 0 becomes the single live species keyed `0`, `member_count[0] = pop_size`, centre = `first`, `best_fitness = -inf`, `next_species_key = 1.0`.
`update_species(state, fitness) -> (state, winner (P,), loser (P,), elite_mask (P,))`; it owns the generation's first `evoxc.random.split` and stores the advanced key back into `state.randkey`.
`speciate(state, distance_fn, pop=None) -> State`; `distance_fn` is any callable `(state, centre_payload, pop) -> (P,)` or, for a centre payload of `species_size` rows, `(S, P)` — `_distance.DistanceRows` provides both forms; the population comes from `pop`, else from the `"pop"` state entry, else from the four split payload entries, else `TensorNEATError`.
Private: `_species_fitness`, `_stagnation`, `_cal_spawn_numbers_by_rank`, `_cal_spawn_numbers_by_fitness`, `_pair_parents(species_state, selection_key, spawn_number, fitness)`, `_block_layout(spawn_number) -> (loc, idx_in_block)`, `_membership`, static `_population(state, pop=None)`, static `_gather_species(species_state, order)`, static `_sort_desc_legacy(x, axis=-1)`, static `_prefix_sum_exclusive(values)`.
Module constant `SPAWN_MODES = ("rank", "fitness")`.

`DistanceRows(genome, digest, population, pop_size, species_size)` — `_distance.py`, created per generation inside `NEAT.tell`:
`__call__(state, centre, pop=None)` returns `(P,)` for a batch-1 or batch-`P` centre and `(S, P)` for a batch-`S` one; `row(state, centre)` and `matrix(state, centres)` are the explicit forms.
It validates that the digest's population axis and the measured population's static shapes match `pop_size` (`TensorNEATError` otherwise) and memoises its tiled digest per block width.
`tile_leading(x, blocks)` repeats the *population* (row `s*P + p` is `x[p]`) and `repeat_leading(x, repeats)` repeats each *centre* row (`k*repeats .. k*repeats+repeats-1` are `x[k]`), the two row orders `measure_distance` needs to see the pairs aligned.

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
7. Elitism: `select(elite_mask[:, None, …], crossover_children, mutated_children)` per payload field, with one reshaped mask per distinct field rank.
8. Speciation against the new population: `digest = genome.prepare_distance(state, population)` (once per generation), then `SpeciesController.speciate(state, self._distance_rows(digest, population), population)`.
9. Commit `pop`, `randkey`, `generation`, `species`.

`_sanitize_fitness(fitness, pop_size)` implements the guard as `finite = abs(f) <= _finite_bound(dtype)` plus `select`, and raises `TensorNEATError` unless `fitness.shape == (pop_size,)`.
`_finite_bound(dtype)` is `numpy.finfo(dtype).max` read at trace time (a Python float, never an array), so the predicate needs no invalid arithmetic: NaN compares false, both infinities compare false, every finite float32 is `<= FLT_MAX`.
The earlier `~isnan(f - f)` spelling computed `inf - inf` and made the numpy backend emit one `RuntimeWarning: invalid value encountered in subtract` per generation; the `abs` form is bit-identical on every float32 input and silent.

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
Speciation, scan 1 (centre reservation): per live slot `i`, `d_i` is the `(P,)` row `distance(centre_i, pop)` (computed behind one `cond` per iteration when the table is sparse, unconditionally when dense — see `## Distance Rows`), `closest = argmin(select(unassigned, d_i, +inf))`, then assign `idx2species[closest] = species_keys[i]`, re-elect `pop[closest]` as the centre and set its distance to `0`; a dead slot or an empty unassigned set writes nothing.
Speciation, scan 2 (member assignment): a live slot assigns every genome with `d < compatibility_threshold` and `(unassigned | d < current distance)`; a dead slot with anything still unassigned creates a species keyed `next_species_key` seeded by the first unassigned genome, then assigns the same way; an iteration with nothing to do writes nothing.
Speciation post-pass: leftovers adopt the last species key, slots created this generation get `best_fitness = -inf` / `last_improved = generation`, and `member_count` is one masked compare-sum over `(S, P)`.

## Distance Rows (one digest per generation; wide matrix when dense, lazy when sparse)
Both phases need one `(P,)` compatibility row per species slot, and the same population is measured `2 · species_size` times, so `NEAT.tell` digests it once: `genome.prepare_distance(state, pop)` sorts each gene family of the population into a `DistanceDigest` (8 leaves, batch `P`) and `DistanceRows` turns centres into rows against it without re-sorting the population.
The row of a slot is known before the iteration reads it, because every `scatter` in both scans writes its own slot only: scan 1's row for slot `i` is `distance(pre-scan centre i, pop)`, and scan 2's row for a **live** slot is `distance(post-scan-1 centre i, pop)`. Only a slot that *creates* a species has a carry-dependent centre (`pop[seed]`, `seed` = first unassigned genome).
**Dense table** (`gate` false, `species_size * pop_size > GATED_ROWS`): `_kernels.speciate` measures the whole centre table before scan 1 (`rows1`) and the post-scan-1 carry centres between the scans (`rows2`), and every iteration reads `gather(rows, slot, axis=0)`.
`DistanceRows.matrix` turns `S` centres into `(S, P)` with `ceil(S / CENTRES_PER_CALL)` wide `measure_distance` calls: both sides are tiled to `K * P` rows (`tile_leading` the population, `repeat_leading` the centres) with the aligned row order `s * P + p`, the tiled digest is memoised per block width (so one phase tiles the population once and the graph reuses those values), and the result is reshaped back to `(K, P)`.
The tiled digest is memoised *per width*, so a generation tiles the population twice (once per phase, same width) plus once per distinct width if the two phases differ — never once per call.
Scan 2 keeps exactly one `evoxc.cond` per iteration in the dense path, whose predicate is `create`: a creating slot's centre is `pop[seed]` and only the true branch pays the row; a live slot's false branch returns the pre-measured `gather(rows2, slot)`.
**Sparse table** (`gate` true, `species_size * pop_size <= GATED_ROWS`): the lazy per-slot path is unchanged, now routed through the digest — one `cond` per scan iteration, `gated_row(active, compute_row, operands)`, predicate `live & any_pending` (scan 1) or `live | create` (scan 2), false branch `enp.zeros((pop_size,), float32)` discarded by the iteration's own `claim` / `take` mask.
Forcing the wide matrix on a sparse table loses measurably (30.1 ms vs 55.6 ms of `speciate` at `P=100`, 152.2 vs 211.0 ms at `P=300`, gated vs forced-wide), because 1–3 of the 20 slots are live there; above the gate every slot is live, so nothing is skippable and the wide path wins.
`CENTRES_PER_CALL = 4` bounds the transient footprint of one call. Measured `speciate` at `P=1000 / 500` by block size: 20 → 726 / 496 ms, 8 → 763 / 340, **4 → 568 / 302**, 2 → 588 / 351, 1 → 598 / 343 — 4 is both the fastest and the cheapest to build; peak RSS is block-independent (see `## Known Issues`), so the block size is a speed choice only.
Bit-identity is a property of the genome kernel, not of this module: the compatibility kernel is row-independent (elementwise per-pair ops plus per-row reductions over the gene axis, one sorted alignment per row), so one `(k * P)` batch returns exactly the floats `k` batches of `P` do. It is verified end-to-end — see `## Test Strategy`.
Dispatch budget per generation, dense: `1` `prepare_distance` + `2 · ceil(20 / 4) = 10` `measure_distance` calls (11 total, plus one `cond` branch per creating slot), against `2 · 20 = 40` `execute_distance` calls before the digest existed.
`GATED_ROWS` and `CENTRES_PER_CALL` are module constants so the public `NEAT` constructor is unchanged.

## Scan Patterns (why fixed-length scans are exact)
`evoxc` de-sugars `scan` into `while` + `gather`/`scatter`, so a trip count must be static and the carry must keep identical dtypes and shapes: both speciation phases are `scan(f, carry, enp.arange(species_size))` with per-slot masks, never a data-dependent `while_loop`.
The legacy early-exit conditions survive as masks, and because every write is mask-then-single-`scatter` the extra iterations are provably no-ops.
Carries are kept narrow and packed (`idx2species` and the running distance as two `(P,)` arrays; the species keys plus `next_species_key` as one `(S+1,)` array) because wide `select`-bearing carries are a known iree codegen hazard.
No random draw happens inside a scan body — keys never travel through a carry.

## Gene Keys & Padding
`max_node_key(state, node_keys)` and `max_conn_marker(state, conn_keys)` return one maximum **per individual** (`(P,) int32`, `0` for an individual with no real row), so `_new_gene_keys` builds a per-individual base: `new_node_keys = base + arange(1, P + 1)` and `new_conn_markers = reshape(base_marker, (P, 1)) + arange(1, 3P + 1).reshape(P, 3)` when `conn_gene.uses_historical_marker`, else zeros.
The `+1` is folded into the int32 `arange` (`arange(1, P + 1)` is int32 and cannot promote the chain to int64 the way `base + 1 + arange(P)` did), and the `cast` on the genome maximum is emitted only when its traced dtype is not already int32.
Uniqueness is therefore guaranteed **within one individual's own keys** (keys are namespaced per individual; a key handed to one genome may coincide with a key held by another, which is legal because the mutation kernel only inspects an individual's own rows): each allocated key is strictly above that individual's own maximum, and the three markers of one individual are distinct from each other.
Node keys and connection markers live in different key columns, so a numerical coincidence between an allocated node key and an allocated marker is meaningless, and gene alignment compares full key rows (endpoints plus marker), so a marker that two individuals share never yields a false homologue.
The width `3` is a genome-level convention: one mutation call may add at most three connections per genome.
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
Speciation and the distance digest draw no randomness at all, so the digest/batching change cannot move an RNG stream.
`pop_batch_size` chunking pre-splits the mutation key with `split_n(key, n_batches)` and passes the stacked `(n_batches,)` int64 array as the scan `xs`, so each step receives a valid rank-0 key.
Keys are rank-0 scalars, so a batched kernel takes one key and draws its own population-shaped randomness; chunked and unchunked runs therefore draw different streams and `pop_batch_size` is part of the reproducibility config.

## Dependencies (cross-node contracts this subtree relies on)
`genome` (batch-first, one rank-0 key each): `setup(state)`, `initialize(state, key, pop_size) -> GenomeTensors`, `transform(state, node_keys, conn_keys)`, `forward(state, transformed, node_keys, node_attrs, conn_keys, conn_attrs, inputs)`, `execute_crossover(state, key, winners, losers)`, `execute_mutation(state, key, genome, new_node_keys, new_conn_markers)`, `prepare_distance(state, pop) -> DistanceDigest`, `measure_distance(state, centre, digest) -> (P,)` with a centre batch of `1` (broadcast internally) or `P` (row-wise).
`measure_distance` rejects a centre batch that is neither `1` nor the digest batch, which is exactly why `DistanceRows` tiles both sides to `K * P` rows instead of asking for a per-centre batch; the kernel is strictly row-independent, so any common batch width returns the same floats.
`execute_distance` remains the documented single-pair contract and is no longer called from here.
`genome` statics: `num_inputs`, `num_outputs`, `max_nodes`, `max_conns`, `node_gene.length`, `conn_gene.length`, `conn_gene.uses_historical_marker`; helpers `node_exists`, `conn_exists`, `max_node_key`, `max_conn_marker`.
`genome.utils.GenomeTensors` is the payload namedtuple — the four fields in declaration order.
`common`: `State` (immutable, `register`/`update`/`registered_keys`), `StatefulBaseClass`, `errors.TensorNEATError`, `tools.{rank_elements, fetch_first, argmin_with_mask, flat_gather, to_numpy}`.
`pipeline`: creates `randkey` with `evoxc.random.key(seed)`, calls `setup` on the host, drives `ask → transform → forward/evaluate → tell`, and calls `show_details` on host-readable values.

## Measured Cost (numpy backend, real `DefaultGenome(max_nodes=50, max_conns=100)`, XOR, `species_size=20`, seed 42)
Protocol: one configuration per fresh process, 5 setup steps, then one warmup and 5 timed runs; phases timed as *separate executables* traced on one snapshot state (`step` is the pipeline's own graph), so a generation's phase sum slightly over-counts. Interleaved A/B against the pre-change tree (`git worktree` at the parent commit, its own venv), same box, medians (minimums in parentheses), ms:

| P | evaluate | update_species | breed | speciate | tell | `Pipeline.step` | peak RSS |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 100 | 10.2 (10.0) | 1.20 (1.18) | 31.2 (30.4) → 30.3 (30.0) | 39.4 (38.6) → **26.9 (26.7)** | 71.8 (70.2) → 58.4 (57.8) | 90.4 (84.8) → **73.7 (71.6)** | 242 → 216 MiB |
| 500 | 24.8 (24.5) | 1.57 (1.36) | 77.0 (71.9) → 83.9 (76.3) | 577.0 (569.3) → **307.1 (311.4)** | 655.5 (642.6) → 392.5 (389.1) | 956.1 (938.9) → **443.7 (440.8)** | 638 → 1562 MiB |
| 1000 | 45.4 (44.6) | 2.07 (2.04) | 143.4 (140.1) → 148.5 (141.9) | 1095.5 (1088.4) → **581.3 (577.1)** | 1240.8 (1230.4) → 731.8 (721.1) | 1813.2 (1800.7) → **833.8 (789.7)** | 1118 → 2958 MiB |

`speciate` −32 % / −47 % / −47 % and the whole step −18 % / −54 % / −54 %; the breed column is unchanged within this box's ±7 % run-to-run drift (the algorithm-owned part of breed is ~0.6 % of its op count — see below).
`tell` is now `speciate` 46 % / 78 % / 79 %, breeding 52 % / 21 % / 20 %, species update ~2 %.
Against the declared legacy JAX baseline (51 / 109 / 157 ms per generation) the cycle is ~1.4x / ~4.1x / ~5.3x over budget at `P=100 / 500 / 1000` (was ~1.8x / ~8.8x / ~11.6x).
Live species slots in the measured states: 3 / 20 / 20 at `P=100 / 500 / 1000` — the sparse gate is what keeps `P=100` on the lazy path.
Graph build: `build_tell` 2.7 / 4.3 / 4.8 s and the pipeline's one-off build 3.2 / 3.9 / 4.5 s (before: 3.3 / 4.4 / 4.7 s); grouping the wide calls kept the build flat.
`genome.measure_distance` costs 0.8 / 6.5 / 11.3 ms per call (7.9 / 13.0 / 11.3 µs per pair) against `execute_distance`'s earlier ~25–48 µs per pair, which is the digest win this node now consumes.
Breeding split at `P=100` (separate executables): mutation kernel 24.8 ms, crossover kernel 5.1 ms, `_new_gene_keys` 0.08 ms, elite select 0.13 ms, parent gathers 0.09 ms.
`breed`'s graph (`update_species` + `_create_next_generation`) is 2014 IR ops at every population size, of which `genome.execute_mutation` is 1359 (67 %), `genome.execute_crossover` 432 (21 %), this node's `update_species` 189 (9 %, of which `_kernels.pair_parents` 38, `spawn_from_target` 26, `_stagnation` 19, `block_layout` 16, `const` 15, `spawn_numbers_by_rank` 15) and `neat.py` itself 29 (1.4 %: `_create_next_generation` 17 = 8 parent gathers + 3 key-split + 4 elite `select` + 2 rank reshapes, `_new_gene_keys` 9, `_sanitize_fitness` 5).
Timed as separate executables: `update_species` 1.31 / 1.83 / 2.23 ms and `_create_next_generation` 21.95 / 55.06 / 105.73 ms at `P=100 / 500 / 1000`, while the algorithm-owned half of the latter (the 8 gathers plus the elite `select`s, genome calls removed) is 0.19 / 0.45 / 0.75 ms — i.e. under 1 % of breed, so the breed cost is genome-kernel bound and no algorithm-owned op removal can move it.
`tell` increments `generation` through `_as_scalar_f32`, which emits the `cast`/`reshape` only when the static dtype/rank need one, so the 0-d `float32` counter costs one `add` and no no-op op.

## Known Issues
Peak RSS grows 2.4x–2.6x at `P >= 500` (638 → 1562 MiB at `P=500`, 1118 → 2958 MiB at `P=1000`) and the growth is **not** the tiling: measuring the same tree with `CENTRES_PER_CALL` forced to 1 / 2 / 4 / 20 gives 2681 / 2881 / 2793 / 2906 MiB peak, i.e. one centre row per call — the shape the pre-change tree used — already costs the full increase. The retained temporaries live in the genome's `prepare_distance`/`measure_distance` (a `(rows, 2 * max_conns, key_width)` masked gather plus the digest itself), so the fix belongs in `genome/operations/distance`; this node reports it rather than working around it. The sparse path is unaffected (242 → 216 MiB at `P=100`).
`speciate`'s pair count is still `2 · species_size · pop_size`; the batching cuts dispatches (40 → 11) and per-pair overhead, not the number of pairs. Each phase needs its own set of genuinely different centres and a created species takes the first unassigned genome as its centre, so its row depends on every earlier assignment.
The remaining per-pair cost is genome-owned: `DefaultDistance`'s `find_homologous` sorts `2·M` key rows for every pair, so the same population genome is re-sorted once per centre; a "prepare once, measure many" API for the *centre* side (or a cheaper homologue merge) is the next real lever and lies outside this node.
Breeding is the second hot spot at `P=100` (30 ms of a 58 ms `tell`) and the graph is 67 % `genome.execute_mutation` / 21 % `genome.execute_crossover`; the algorithm-owned part of `_create_next_generation` (8 parent gathers + 4 elite `select`s) measures 0.19 / 0.45 / 0.75 ms at `P=100 / 500 / 1000`, under 1 % of breed, so no algorithm-owned win is left in this subtree.
`_new_gene_keys` is correct as specified: `new_node_keys (P,) int32` / `new_conn_markers (P, 3) int32`, each strictly above that individual's own maximum (verified for `DefaultConn` at `P=100` and `OriginConn`'s `historical_marker` path at `P=1000`), `arange` offsets baked into int32 constants with no int64 promotion.
`show_details` is correct on the current genome contract (`node_exists(node_keys)` / `conn_exists(conn_keys)`, key array only); the reported `TypeError: BaseGenome.node_exists() takes 2 positional arguments but 3 were given` came from passing a `cfg` state at those two call sites and is gone — verified by driving real runs (DefaultGenome, RecurrentGenome, OriginConn, `pop_batch_size` chunking, `P=2`) and calling `show_details` on host values with no try/except.
`next_species_key` must stay strictly greater than every live key; `setup` seeds it at `1.0` (slot 0 uses `0.0`) and each created species increments it, so the invariant holds as long as the counter is never rebound by hand — a colliding counter silently merges two species' `member_count`.
`test/unit/algorithm/test_spawn.py` encodes the brief's `loc(p) = #{i : spawn[i] > 0 and E[i] <= p} - 1`; that count is wrong for a live zero-spawn species in the middle of the table, so the oracle needs the "last qualifying block" form to agree with `block_layout` (cross-node item, `test/` is not writable from here).
`test/unit/algorithm/**` now runs (139 passed, 5 failed): the five failures are four `[XPASS(strict)]` markers in `test_neat_tell.py` left over from the fixed `_new_gene_keys` reshape defect (`test_tell_increments_the_generation_before_the_species_update`, `test_tell_preserves_the_population_size_and_payload_shapes`, `test_tell_is_deterministic_for_the_same_state_and_fitness`, `test_tell_advances_the_randkey`) plus `test_speciation.py::test_centre_genes_keep_the_split_payload_contract`, which expects a population-shaped centre payload while `_kernels.speciate` now hands `distance_fn` a batch-1 centre (cross-node item, `test/` and `_kernels.py` are not writable from here); this node is additionally verified by driving `Pipeline` end-to-end on the numpy backend with the real `DefaultGenome`/`XOR` (pop 8/100/500/1000, `rank` and `fitness` spawn modes, `max_stagnation` kills, `OriginConn`'s `historical_marker` branch, `pop_batch_size` chunking, `RecurrentGenome`, XOR3d) plus bit-identical repeat runs.
`_new_gene_keys` is per-individual by contract (see `## Gene Keys & Padding`), so the genome must not assume keys are globally unique across the population.

## Constraints
No `jax`/`torch`/`numpy.random` import; `import tensorneat` must stay free of them (release gate). `numpy` appears only as `numpy.finfo(dtype).max` in `_finite_bound`, a host-side scalar read.
No branching, `bool()`, `len()` or host reads on traced values — static Python `if`/`for` over constructor configuration is fine and unrolls at trace time.
No `.at[]`, no `cumsum` on the compute path (prefix sums are triangular `dot`), no `where=`/`initial=`/`any`/`all`/`take`/`searchsorted`/`repeat`, no `vmap` over anything region-bearing, no `fori_loop`/`switch`.
Shapes are static: `P`, `S`, `max_nodes`, `max_conns` are fixed at build time; a change forces a rebuild by the pipeline.
Errors: configuration validation raises `common.errors.TensorNEATError` with a named message; trace/shape/backend errors propagate unchanged and are never re-wrapped or silently fallen back.
State keys touched: `randkey`, `pop_size`, `pop`, `generation`, `species`, plus the genome/problem keys documented above — nothing else.
Files stay under 1000 lines: `species.py` (966) the species algebra, `_kernels.py` (898) the graph helpers, `_distance.py` (315) the digest-backed rows, `_validation.py` (128) the host validators, `neat.py` (905) the population cycle.

## Known Risks
`evoxc.random.randint` needs a symbolic `high` that stays ≥ 1 at runtime — guaranteed by `max(survive_size, 1)`, which also removes the legacy NaN case.
`evoxc.scatter` is replacement-only, so every "accumulate" update is restructured as reduce-then-scatter-once (spawn residual, stale-write elimination in the speciation scans).
Wide `while`/`scan` carries plus `select` can crash iree codegen (`AffinityAnalysis`), hence the packed-carry convention; the numpy reference backend is unaffected.
Reading a dead slot's centre rows would return stale integers, not NaN — always gate centre reads behind `isnan(species_keys)`.
`DistanceRows`' tiling is a `broadcast_to` + `reshape`, i.e. a real copy: at `P=1000` one block of 4 centres materialises ~4k rows per digest leaf, and the memo keeps each width's tiled digest alive for the rest of the trace.

## Test Strategy
Tests live in the sibling `test/` tree (read-only from here): `test/unit/algorithm/` covers the spawn/selection/stagnation algebra, the block layout and the speciation scans on fixed payloads.
`test/parity/algorithm/neat/` compares against `benchmarks/legacy_jax` on fixed gene payloads — identical selection/speciation outcomes for the deterministic steps, and only the divergences listed above allowed elsewhere.
`test/integration/` evolves XOR to its fitness target with a fixed seed and asserts run-to-run determinism for a `(seed, config, shape)` triple.
Every kernel here must be a pure function of `(state, key)`: no dict-order dependence, no host RNG, no float→int conversion without an explicit mask.
Distance rows are verified by whole-run bit-identity: a `npz` dump of `fitness_history`, the four payload arrays, all ten `SpeciesState` leaves, `randkey` and `generation` per generation, compared with `tobytes()` equality (padding NaNs make `np.array_equal` report self-vs-self differences) against the pre-change tree at pop 100 (6 generations, gated path), pop 500 (4) and pop 1000 (3, dense path) — `common.tools.to_numpy` first, because hashing an `evoxc.Tensor` directly hashes the object pointer rather than the data.
The sparse/dense switch, the block size and the `cond` placement are all static trace-time choices, so a shape or gate regression shows up as a build error in `test/release_gate_no_jax.py`'s two-generation XOR run rather than as a silent numerical drift.

## Routing Table
| Path | Responsibility |
| --- | --- |
| `./neat.py` | `NEAT`: population cycle (`setup`/`ask`/`transform`/`forward`/`tell`), fitness sanitising, next-generation creation, `pop_batch_size` scan chunking, the per-generation distance digest wiring, host-side `show_details`. |
| `./species.py` | `SpeciesState` (frozen dataclass) and `SpeciesController`: species fitness, stagnation, descending sort, spawn allocation, parent pairing, block layout, fixed-length speciation scans. |
| `./_kernels.py` | Plain graph helpers behind `species.py`: constants, payload helpers, sort/prefix-sum primitives, membership, spawn allocation, block layout, parent pairing, `GATED_ROWS`/`gated` (the static lazy-row gate) and the two speciation scans, payload reading. |
| `./_distance.py` | `DistanceRows` (digest-backed `distance_fn`, batch-1/`P` row and batch-`S` matrix forms), `CENTRES_PER_CALL`, `tile_leading`/`repeat_leading`. |
| `./_validation.py` | Host-side constructor validators shared by `species.py` and `neat.py`. |
| `./__init__.py` | Re-exports `NEAT`, `SpeciesController`, `SpeciesState`. |
