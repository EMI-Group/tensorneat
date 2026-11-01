# NEAT (`tensorneat.algorithm.neat`) — population cycle + species bookkeeping

## Intent
Everything in a NEAT generation that is neither network mathematics (genome) nor task evaluation (problem): the population tensors, the generation counter, speciation, stagnation, spawn allocation, parent pairing, and the crossover/mutation dispatch.
`NEAT` (`neat.py`) is the algorithm the pipeline drives; `SpeciesController` + `SpeciesState` (`species.py`) own the species algebra.
Every kernel here is batch-first over `P = pop_size` and compiled by `pipeline.py` into the single per-generation evoxc graph, so no function may branch on a traced value or read the host.
The layer is deliberately cheap: `O(species_size × pop_size)` gene-row distance work in `speciate`, `O(pop_size)` selection/crossover/mutation, and never a `max_nodes × max_nodes` tensor.

## API Surface
`__init__.py` re-exports `NEAT`, `SpeciesController`, `SpeciesState`.

`NEAT(BaseAlgorithm)` — `neat.py`:
`__init__(genome, pop_size, species_size=10, max_stagnation=15, species_elitism=2, spawn_number_change_rate=0.5, genome_elitism=2, survival_threshold=0.1, min_species_size=1, compatibility_threshold=2.0, species_fitness_func=evoxc.max, species_number_calculate_by="rank", pop_batch_size=None)` — constructor kwargs are API-compatible with the pre-rewrite library.
`setup(state) -> State` registers `pop_size` (static int), `pop_nodes (P, max_nodes, node_len)`, `pop_conns (P, max_conns, conn_len)`, `generation` (0-d float32) and `state.species`, then advances `randkey`.
`ask(state) -> (pop_nodes, pop_conns)`; `transform(state, pop)`/`forward(state, transformed, inputs)` delegate to the genome; `tell(state, fitness) -> State` runs the nine-step algebra below.
`show_details(state, fitness) -> None` is the only host-side member; `num_inputs`/`num_outputs`/`pop_size` are read-only properties (`pop_size` is backed by `_pop_size` because the base class property has no setter).
Private members: `_sanitize_fitness` (module function), `_create_next_generation`, `_new_gene_keys`, `_batched`.

`SpeciesController(StatefulBaseClass)` — `species.py`:
`setup(state, first_nodes, first_conns) -> State`, `update_species(state, fitness) -> (State, winner (P,), loser (P,), elite_mask (P,))`, `speciate(state, distance_fn) -> State`.
Private: `_species_fitness`, `_stagnation`, `_cal_spawn_numbers_by_rank`, `_cal_spawn_numbers_by_fitness`, `_pair_parents`, `_block_layout`, static `_sort_desc_legacy`, static `_prefix_sum_exclusive`.

`SpeciesState` — frozen dataclass, one leaf per field, stored under `state.species`:
`species_keys (S,) f32` (NaN = dead slot), `best_fitness (S,) f32`, `last_improved (S,) f32`, `member_count (S,) f32`, `idx2species (P,) f32` (NaN = unassigned, only inside `speciate`), `center_nodes (S, max_nodes, node_len) int`, `center_conns (S, max_conns, conn_len) int`, `next_species_key` (0-d f32).
`S = species_size`, `P = pop_size`. Every scalar field is float32 so NaN can encode a dead slot and so `generation` writes into `last_improved` without a dtype change inside a scan carry.

## Generation Algebra (`NEAT.tell`)
1. Sanitise fitness: `finite = ~isnan(f - f)`, `clean = select(finite, f, -inf)` — exact for NaN and both infinities without an `isfinite` op.
2. `generation += 1` before the species update (legacy order): stagnation is judged against the generation that produced the fitness being consumed.
3. `SpeciesController.update_species` → species fitness, stagnation, species sort, spawn allocation, parent pairs.
4. `_new_gene_keys` → per-genome unique `new_node_keys (P,)` and `new_conn_markers (P, 3)`.
5. Crossover: `gather(pop, winner, axis=0)` / `gather(pop, loser, axis=0)`, then batch-first `genome.execute_crossover(state, key, w_nodes, w_conns, l_nodes, l_conns)`; argument order is part of the contract (winner first).
6. Mutation: batch-first `genome.execute_mutation(state, key, nodes, conns, new_node_keys, new_conn_markers)`, scan-chunked by `pop_batch_size`.
7. Elitism: `select(elite_mask[:, None, None], crossover_children, mutated_children)`.
8. `SpeciesController.speciate(state, genome.execute_distance)` rebuilds membership for the new population.
9. Commit `pop_nodes`, `pop_conns`, `randkey`, `generation`, `species`.

## Species Algebra
Membership is one primitive reused everywhere: `M[i, p] = idx2species[p] == species_keys[i]`, shape `(S, P)`; NaN compares False, so dead slots and unassigned rows never match.
Species fitness: `select(M[i], fitness, -inf)` then `species_fitness_func(..., axes=1)` — masking before the reduce keeps a member-less slot at `-inf` instead of NaN.
Stagnation: `improved = f_s > best_fitness`; `stagnant = (f_s <= best_fitness) & (generation - last_improved > max_stagnation)`, disabled when `rank_elements(f_s) < species_elitism`; a killed slot writes NaN into `species_keys`/`best_fitness`/`last_improved`/`member_count` and `-inf` into its species fitness (so the descending sort pushes it last).
Sort: `perm = _sort_desc_legacy(f_s)` — stable ascending `argsort` plus the reversal permutation `n - 1 - enp.arange(n)` — then one `gather` per `(S, …)` field; after the sort the slot index *is* the fitness rank.
Spawn (rank mode): `n = sum(valid)`, `rank_score = n - arange(S)`, `denominator = (n+1)*n/2`, `target = floor(rank_score/denominator * pop_size)`.
Spawn (fitness mode): `f' = f_live - min_live(f_live) + 1`, `rate = f' / sum_live(f')`, `target = floor(rate * pop_size)` — every reduction masks dead slots first.
Spawn (both modes): `damped = member_count + (target - member_count) * spawn_number_change_rate`, floored at `min_species_size`, dead/NaN slots forced to `0.0`, cast to int32, then `spawn[0] += pop_size - sum(spawn)` via a single `scatter` so `sum(spawn) == pop_size` exactly.
Parent pairing: members ranked by fitness descending (masked with `-inf` first), `survive_size = floor(survival_threshold * member_count)`, `high = max(survive_size, 1)`, one `evoxc.random.randint(key, (S, 2, P), low=0, high=high[:, None])` draw for father **and** mother, `gather(sorted_members, slots, axis=1)`, elites (`p_idx < genome_elitism`) pinned to `sorted_members[:, 0]`.
Block flattening: with exclusive prefix `E` of `spawn`, `loc(p) = #{i : spawn[i] > 0 and E[i] <= p} - 1` and `idx_in_block(p) = p - E[loc(p)]`; parents are then one gather at `flat = loc * pop_size + idx_in_block` into the row-major `(S, P)` parent arrays.
Winner/loser: `first` wins ties (`fitness[first] >= fitness[second]`), matching the legacy comparison.
Speciation, scan 1 (centre reservation): per live slot `i`, `d = distance_fn(state, center_nodes[i], center_conns[i], pop_nodes, pop_conns)`, `closest = argmin(select(isnan(idx2species), d, +inf))`, then assign `idx2species[closest] = species_keys[i]`, `center_[nodes|conns][i] = pop[closest]`, `o2c[closest] = 0`; a dead slot writes nothing.
Speciation, scan 2 (member assignment): a live slot assigns every genome with `d < compatibility_threshold` and `(unassigned | d < o2c)`; a dead slot creates a species keyed `next_species_key` seeded by `fetch_first(isnan(idx2species))`, then assigns against the new centre; an iteration with nothing unassigned writes nothing.
Speciation post-pass: leftovers adopt the last species key, slots created this generation get `best_fitness = -inf` / `last_improved = generation`, and `member_count` is one masked compare-sum over `(S, P)`.

## Scan Patterns (why fixed-length scans are exact)
`evoxc` de-sugars `scan` into `while` + `gather`/`scatter`, so a trip count must be static and the carry must keep identical dtypes and shapes: both speciation phases are `scan(f, carry, enp.arange(species_size))` with per-slot masks, never a data-dependent `while_loop`.
The legacy early-exit conditions survive as masks: scan 1 stops writing at the first dead slot, scan 2 stops writing once nothing is unassigned, and because every write is a mask-then-single-`scatter`, the extra iterations are provably no-ops.
`scan` requires `xs` with a rank ≥ 1 leaf and a `y` output with ≥ 1 symbolic leaf, so the index vector is the `xs` and the updated `idx2species` is reported as `y` (unused downstream).
No random draw happens inside any scan body — keys never have to travel through a carry.
Scan carries are kept narrow and packed (e.g. `idx2species` and `o2c` as one `(2, P)` tensor, `species_keys` plus `next_species_key` as one `(S+1,)` tensor) because wide carries with `select` are a known iree codegen hazard.

## Gene Keys & Padding
`new_node_keys = max_node_key(state, pop_nodes) + 1 + enp.arange(P)`; `new_conn_markers = max_conn_marker(state, pop_conns) + 1 + enp.arange(P * 3).reshape(P, 3)`, or `enp.zeros((P, 3))` when `conn_gene.uses_historical_marker` is False.
The width `3` is a genome-level convention: one mutation call may add at most three connections per genome.
Validity is never inferred from packaging: this subtree calls `node_exists`, `conn_exists`, `max_node_key`, `max_conn_marker` on the genome.
Dead species slots are recognised by `isnan(species_keys)` alone; their centre rows hold stale integers because int32 cannot carry NaN, and no kernel reads a centre gated by that mask.

## Divergences from the Pre-Rewrite Baseline (deliberate fixes)
1. Zero-survivor division: the old `(p_idx < survive_size) / survive_size` probability vector divided by zero for a one-member species; `randint` with `high = max(survive_size, 1)` yields two identical parents instead (self-crossover is the identity).
2. NaN-poisoned reductions: `"fitness"`-mode `min`/`sum` in the old code mixed in the NaN `best_fitness` of dead slots; every reduction now masks dead slots with `select` first.
3. NaN → int32 cast for `spawn_number`: dead and NaN slots are explicitly written to `0.0` before the cast instead of relying on cast semantics.
4. `species_number_calculate_by`/`weight_threshold`-style configuration errors raise `TensorNEATError` instead of `assert`/`ValueError`.
5. Sort tie order is reproduced exactly by a stable ascending `argsort` plus a reversal gather, because `argsort(descending=True)` breaks ties in the opposite direction.
6. Random streams differ from the baseline by construction: one key per population kernel (drawing `(P, …)` randomness internally) instead of one key per genome, so parity tests compare kernels with fixed inputs and structural outcomes, not RNG bit-patterns.

## Randomness Budget
Exactly two `evoxc.random` splits per generation: `selection_key, rest = split(state.randkey)` then `cross_key, mut_key, next_randkey = split_n(rest, 3)` — three consumed subkeys in the order parent selection, crossover, mutation, with `next_randkey` carried in `state.randkey`.
`pop_batch_size` chunking pre-splits the kernel key with `split_n(key, n_batches)` and passes the stacked `(n_batches,)` int64 array as the scan `xs`, so each step receives a valid rank-0 key.
There is no `choice`, no `fold_in` and no probability vector; uniform parent ranks come from `randint` with a broadcast symbolic `high`.
Keys are rank-0 scalars, so a batched kernel takes one key and draws its own population-shaped randomness.

## Dependencies (cross-node contracts)
`genome` (batch-first, one rank-0 key each): `setup(state)`, `initialize(state, key) -> (pop_nodes (P, max_nodes, node_len), pop_conns (P, max_conns, conn_len))` reading `P` from `state.pop_size`, `transform(state, nodes, conns)`, `forward(state, transformed, inputs)`, `execute_crossover(state, key, w_nodes, w_conns, l_nodes, l_conns)`, `execute_mutation(state, key, nodes, conns, new_node_keys (P,), new_conn_markers (P, 3))`, `execute_distance(state, center_nodes_i, center_conns_i, pop_nodes, pop_conns) -> (P,)`.
`genome` statics: `num_inputs`, `num_outputs`, `max_nodes`, `max_conns`, `node_gene.length`, `conn_gene.length`, `conn_gene.uses_historical_marker`; helpers `node_exists`, `conn_exists`, `max_node_key`, `max_conn_marker`.
`common`: `State` (immutable, evoxc-flattenable), `StatefulBaseClass`, `errors.TensorNEATError`, `tools.{rank_elements, fetch_first, argmin_with_mask, to_numpy}`.
`pipeline`: creates `randkey`, calls `setup`, drives `ask → transform → forward/evaluate → tell`, calls `show_details` on host-readable values.
`problem`: `evaluate(state, keys, act_fn, transformed)` with `act_fn` = `algorithm.forward` partially applied; the problem owns dataset tiling.

## Constraints
No `jax`/`torch` import; `import tensorneat` must stay free of them (release gate).
No branching, `bool()`, `len()` or host reads on traced values — static Python `if`/`for` over constructor configuration is fine and unrolls at trace time.
No `.at[]`, no `cumsum` on the compute path (prefix sums are triangular `dot`), no `where=`/`initial=`/`any`/`all`/`take`/`searchsorted`/`repeat`, no `vmap` over anything region-bearing, no `fori_loop`/`switch`.
Shapes are static: `P`, `S`, `max_nodes`, `max_conns` are fixed at build time; a change forces a rebuild by the pipeline.
Errors: configuration validation raises `common.errors.TensorNEATError` with a named message; trace/shape/backend errors propagate unchanged and are never re-wrapped or silently fallen back.
State keys touched: `randkey`, `pop_size`, `pop_nodes`, `pop_conns`, `generation`, `species`, plus the genome/problem keys documented above — nothing else.
Files stay under ~1000 lines: `neat.py` is the population cycle, `species.py` the species algebra; shared math lives in `common/`.

## Known Risks
`evoxc.random.randint` needs a symbolic `high` that stays ≥ 1 at runtime — guaranteed here by `max(survive_size, 1)`, which also removes the legacy NaN case.
`evoxc.scatter` is replacement-only, so every "accumulate" update is restructured as reduce-then-scatter-once (spawn residual, stale-write elimination in the speciation scans).
Wide `while`/`scan` carries plus `select` can crash iree codegen (`AffinityAnalysis`), hence the packed-carry convention; the numpy reference backend is unaffected.
Reading a dead slot's centre rows would return stale integers, not NaN — always gate centre reads behind `isnan(species_keys)`.
Chunked (`pop_batch_size`) and unchunked runs draw different random streams; `pop_batch_size` is part of the reproducibility config.

## Test Strategy
Tests live in the sibling `test/` tree (read-only from here): `test/unit/algorithm/neat/` covers the spawn/selection/stagnation algebra and the block layout on fixed keys with the numpy backend.
`test/parity/algorithm/neat/` compares against `benchmarks/legacy_jax` on fixed gene payloads — identical selection/speciation outcomes for the deterministic steps, and only the divergences listed above allowed elsewhere.
`test/integration/` evolves XOR to its fitness target with a fixed seed and asserts run-to-run determinism for a `(seed, config, shape)` triple.
Every kernel here must be a pure function of `(state, key)`: no dict-order dependence, no host RNG, no float→int conversion without an explicit mask.

## Routing Table
| Path | Responsibility |
| --- | --- |
| `./neat.py` | `NEAT`: population cycle (`setup`/`ask`/`transform`/`forward`/`tell`), fitness sanitising, next-generation creation, `pop_batch_size` scan chunking, host-side `show_details`. |
| `./species.py` | `SpeciesState` (frozen dataclass) and `SpeciesController`: species fitness, stagnation, descending sort, spawn allocation, parent pairing, block layout, fixed-length speciation scans, shared sort/prefix-sum utilities. |
| `./__init__.py` | Re-exports `NEAT`, `SpeciesController`, `SpeciesState`. |
