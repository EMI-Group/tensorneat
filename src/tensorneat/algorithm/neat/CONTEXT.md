# NEAT (`tensorneat.algorithm.neat`) — population cycle + species bookkeeping

## Intent
Everything in a NEAT generation that is neither network mathematics (genome) nor task evaluation (problem): the population payload, the generation counter, speciation, stagnation, spawn allocation, parent pairing, and the crossover/mutation dispatch.
`NEAT` (`neat.py`) is the algorithm the pipeline drives; `SpeciesController` + `SpeciesState` (`species.py`) own the species algebra; `_distance.py` owns the compatibility-distance rows over one generation's population digest.
Every kernel here is batch-first over `P = pop_size` and compiled by `pipeline.py` into the single per-generation evoxc graph, so no function may branch on a traced value or read the host.
Measured on the numpy reference backend with the real `DefaultGenome(max_nodes=50, max_conns=100)` and `XOR`, `species_size=20`: `Pipeline.step` runs in ~59 / ~335 / ~608 ms per generation at `P=100 / 500 / 1000`, of which `speciate` is 57 % / 78 % / 76 % and breeding 43 % / 21 % / 23 % (see `## Measured Cost`); the interpreter, not the data, dominates, and the padded gene axis is what multiplies every per-pair op.
The layer's own work is one population digest plus `2 · ceil(species_size / 4)` distance calls in `speciate` and `O(pop_size)` selection/crossover/mutation; it never materialises a `max_nodes × max_nodes` tensor.

## Layout (five files, each under 1000 lines)
`species.py` — `SpeciesState` plus `SpeciesController`: the documented, traced-facing methods and the species algebra, each delegating its tensor body to `_kernels.py`.
`_kernels.py` — plain graph helpers (no `@evoxc.defn`, no state): spawn allocation, block layout, parent pairing, membership, sort/prefix-sum primitives, the export-safe write/constant primitives (`scatter_row`, `first_centre_table`, `static_i32`/`static_f32`, `reversed_index`), `gated` (the static lazy-row gate) and the two speciation scans.
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
`_kernels.first_centre_table(species_size, row, fill, dtype)` (imported by `species.py` as `_first_centre_table`) builds a generation-0 centre table as `concatenate([reshape(row, (1,) + row.shape), constant fill block])`.
Module constant `SPAWN_MODES = ("rank", "fitness")`.

`DistanceRows(genome, digest, population, pop_size)` — `_distance.py`, created per generation inside `NEAT.tell`:
`__call__(state, centre, pop=None)` accepts a centre batch of exactly `pop_size` (measured as it stands) or `1` (tiled by `row`) and returns `(P,)`; any other batch raises `TensorNEATError` naming it.
`row(state, centre)` is the single-row form and `matrix(state, centres)` the explicit by-name `(S, P)` whole-table form — `__call__` never dispatches to `matrix`, because a `species_size`-row table and a population-shaped centre are indistinguishable when `pop_size == species_size`.
It validates that the digest's population axis and the measured population's static shapes match `pop_size` (`TensorNEATError` otherwise).
`repeat_leading(x, repeats)` repeats each *centre* row (`k*repeats .. k*repeats+repeats-1` are `x[k]`), which is the row order `measure_distance`'s row-wise pairing needs.

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
Scan 2 carries `assign`, `dist`, `keys_ext` plus two rank-1 per-slot tables — `seed_table (S,) int32` and `created_mask (S,) float32`, one single-element rank-1 `axis=0` `scatter` each — and **not** the four centre gene tables: its centre writes were the identity on every non-creating slot, so only a created row (`pop[seed_table[slot]]`) ever differs from the scan-1 table.
The four centre tables are rebuilt once after the scan as `select(created[:, None…], gather(population.<field>, seed_table, axis=0), scan1_table)` (four gathers + four selects at top level, one reshaped mask per field rank), which leaves the dead rows' stale integers untouched and the result bit-identical.
Scan 1 still carries the centre tables (it re-elects centres for live slots), so a scan-1 slot's centre write stays an unconditional `scatter_row`.
Both scans close over their graph-time constants (`I_INF`, `0`, `0.0`, `iota_i32(pop_size)`, the compatibility threshold and the gated false-branch zero row) instead of rebuilding them per iteration — one `constant` op each at the top level instead of one per slot.
Speciation post-pass: leftovers adopt the last species key, slots created this generation get `best_fitness = -inf` / `last_improved = generation`, and `member_count` is one masked compare-sum over `(S, P)`.
Scan-2 body size: 234 → 221 traced ops, `while` carry 9 → 7 values; scan-1 body 59 → 55 ops; whole-generation static graph 5 384 → 5 371.
Executed ops per generation (dispatch-level count at `P=500` and `P=1000`) 10 671 → 10 352.


## Distance Rows (one digest per generation; one population-shaped call per centre)
Both phases need one `(P,)` compatibility row per species slot, and the same population is measured `2 · species_size` times, so `NEAT.tell` digests it once: `genome.prepare_distance(state, pop)` sorts each gene family of the population into a `DistanceDigest` (8 leaves, batch `P`) and `DistanceRows` turns centres into rows against it without re-sorting the population.
Call shape: the `distance_fn` contract is population-shaped — `(state, centre, pop) -> (P,)` with **both** payloads batched `P` — so `DistanceRows.row` tiles its one centre row to `P` rows with `repeat_leading` and makes exactly one `genome.measure_distance` call; `matrix` is a static Python loop of `S` such rows concatenated on axis 0.
The explicit tile is what makes one call per centre legal for every gene variant: `measure_distance` accepts a centre batch of `1` (internal broadcast) or `P` (row-wise) and rejects anything else, so a centre that is not already population-shaped is tiled rather than handed over as one row.
`__call__` is single-valued: a batch of `P` is measured as it stands, a batch of `1` is tiled, and anything else raises — `speciate` always passes a population-shaped centre, so a whole centre table must be measured through `matrix` by name.
The row of a slot is known before the iteration reads it, because every `scatter` in both scans writes its own slot only: scan 1's row for slot `i` is `distance(pre-scan centre i, pop)`, and scan 2's row for a **live** slot is `distance(post-scan-1 centre i, pop)`. Only a slot that *creates* a species has a carry-dependent centre (`pop[seed]`, `seed` = first unassigned genome).
The dense path pre-measures **scan 1** outside the scan: `rows1 = distance_fn.matrix(state, centre_payload)` runs once before `evoxc.scan(reserve, …)` and each iteration reads `evoxc.gather(rows1, slot, axis=0)`.
That is exactly the row the iteration would have measured, because scan 1's centre for slot `i` is the pre-scan carry row `i` (every `scatter` in `reserve` writes its own slot and `arange(S)` visits each once).
The win is the loop body, not the arithmetic: it drops from 228 to 59 ops per iteration (static graph 2381 → 5384, since the kernel is now traced once per centre), worth about 13 % of `speciate` at `P=500`.
Scan 2 keeps its in-scan measurement on both paths, because a slot that creates a species measures the carry-dependent `pop[seed]`; pre-measuring it needs a per-iteration `cond` and measured neutral-to-slower with ~50 % more peak RSS (832 vs 556 MiB at `P=500`).
The dense path therefore requires a `distance_fn` with a `matrix` method and raises `TensorNEATError` naming it otherwise; the gated path keeps working with any callable.
**Sparse table** (`gate` true, `species_size * pop_size <= GATED_ROWS`): one `cond` per scan iteration wraps that row — `gated_row(active, compute_row, operands)`, predicate `live & any_pending` (scan 1) or `live | create` (scan 2), false branch `enp.zeros((pop_size,), float32)` discarded by the iteration's own `claim` / `take` mask.
**Dense table** (`gate` false): no `cond`, the row is computed unconditionally, because above the gate every slot is live and the `if` region would cost more than it skips (30.1 vs 55.6 ms of `speciate` at `P=100`, 152.2 vs 211.0 ms at `P=300`).
Grouping centres into wider calls does not pay: measured at `P=1000`, one `measure_distance` call per centre is 13.0 µs/pair at batch `P` (520 ms for the generation's 40 000 pairs), 2 centres per call 10.3, **4 centres per call 9.7**, 10 centres 10.3 and the whole `(S, P)` table in one call 13.8 (552 ms) — batching buys dispatch at the cost of array traffic, and the whole-table call also re-materialises an `(S · P, …)` tiled digest.
Re-measured end to end at **2 centres per call** (`matrix` tiling every batch-first digest leaf twice on axis `0`, `repeat_leading` block order, memoised per width): the isolated win does not survive — 8 paired alternating rounds at `P=500` give a median ratio of **1.17** (255.8 vs 227.0 ms), i.e. 17 % *slower*, with a bimodal per-round split (211–222 ms on four rounds, 265–294 on the other four) while the one-centre form stays at 224–233 ms. Block 2 is therefore rejected together with block 4/10/20; the wide batches double the transient size of every kernel op, which is the same mechanism that sank block 4 earlier.

Bit-identity is a property of the genome kernel, not of this module: the compatibility kernel is row-independent (elementwise per-pair ops plus per-row reductions over the gene axis, one sorted alignment per row), so a `(k · P)` batch returns exactly the floats `k` batches of `P` do; verified end-to-end — see `## Test Strategy`.
Dispatch budget per generation: `1` `prepare_distance` + `2 · species_size` population-shaped `measure_distance` calls, against `2 · species_size` `execute_distance` calls before the digest existed.
`GATED_ROWS` is a module constant so the public `NEAT` constructor is unchanged.

## Scan Patterns (why fixed-length scans are exact)
`evoxc` de-sugars `scan` into `while` + `gather`/`scatter`, so a trip count must be static and the carry must keep identical dtypes and shapes: both speciation phases are `scan(f, carry, enp.arange(species_size))` with per-slot masks, never a data-dependent `while_loop`.
The legacy early-exit conditions survive as masks, and because every write is mask-then-single-`scatter` (`scatter_row` for the rank-3 gene tables) the extra iterations are provably no-ops.
Carries are kept narrow and packed (`idx2species` and the running distance as two `(P,)` arrays; the species keys plus `next_species_key` as one `(S+1,)` array) because wide `select`-bearing carries are a known iree codegen hazard.
Scan 1 carries those three plus the four centre gene tables (9 values); scan 2 carries the same three plus the rank-1 `seed_table`/`created_mask` (7 values), and the created centre rows are rebuilt after it — see `## Species Algebra`.
No random draw happens inside a scan body — keys never travel through a carry.

## StableHLO Export Rules (binding for every kernel here)
Every graph this node contributes must survive `evoxc`'s StableHLO v1 exporter: `backend=` is forwarded verbatim by the pipeline, so `"iree"`/`"tvm"`/`"xla"` users lower these kernels and a rejected op is a hard `evoxc.BackendError`, never a fallback.
The v1 exporter accepts only the spellings below — a new kernel has to pick one of them:
- `scatter`: rank-1 `axis=0`, or a rank-2 row/column write (`axis` 0 or 1); any wider operand raises `BackendError` naming the rank.
  A per-slot table write goes through `_kernels.scatter_row(table, slot, update)`: it flattens the trailing axes into one row length `R`, writes exactly the cells of a rank-2 `axis=0` row scatter and reshapes back (a rank-2 table short-circuits to a plain `evoxc.scatter`).
  Both speciation scans use it for their `(S, N)`/`(S, N, NA)`/`(S, E, CK)`/`(S, E, CA)` table writes — scan 1 currently writes the four centre rows, and scan 2 no longer does (it writes the rank-1 `seed_table`/`created_mask` with plain `evoxc.scatter` and the created rows are rebuilt with `gather`+`select` after the scan); `keys_ext` (rank-1) and the `node_keys (S, N)` table stay on plain `evoxc.scatter`.
- No `tril`/`triu`, no `flip`, no `cumsum`, no `iota` — deferred/unmapped in v1.
  Static tables are baked as graph constants with `static_i32`/`static_f32` (one `constant` op each, bit-identical because the entries are exact index/`0`/`1` values): the gate matrix of `prefix_sum_exclusive` is `static_f32(np.tril(np.ones((S, S)), -1))`, and the reversal of `block_layout` is `gather(x, reversed_index(S), axis=0)`.
- Supported and in use: `gather` (single axis, static indices), `argsort`, `select`, `concatenate`, `constant`, `reshape`, `dot`, `argmax`/`argmin`, the `random_*` family, and `while`/`if` regions.
- A table whose rows are all written anyway is not a scatter: the generation-0 centre tables are `concatenate([reshape(row, (1,) + row.shape), constant fill block])` (`_kernels.first_centre_table`, used by `SpeciesController.setup`), which also exports at rank 3.
- `scan` stacks its per-step *outputs* along a new leading axis, so returning a whole `(S, …)` table from a scan body becomes a rank-4 scatter inside the desugared `while`; both speciation scans therefore return only their small per-step value (`assign` in scan 1, `create` in scan 2).
- Unsigned integer dtypes (`uint32`/`uint64`) are outside iree's declared dtype set — a graph is expected to use `int32`/`int64` only (the sibling `genome`/`common` nodes own their remaining occurrences).
The numpy reference interpreter accepts every rank, so a violation is invisible until `evoxc.lower(graph, backend="iree")`; the audit in `## Test Strategy` is part of verification, not an optional extra.

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
Protocol: one configuration per fresh process, 4 setup generations, then one warmup and 4 timed runs; phases timed as *separate executables* traced on one snapshot state (`step` is the pipeline's own graph), so a generation's phase sum slightly over-counts. Medians, ms, this box (it drifts ±30 % between sessions, so read ratios, not absolute milliseconds):
**Cumulative effect of the two kept speciation changes** (scan-1 row pre-measurement + narrow scan-2 carry): same-session paired A/B against the pre-change tree, 6 alternating rounds of `benchmarks/backend_measure.py --backend numpy --pop 500 --generations 20 --warmup 3`, medians per run — pre-change 247.9–333.3 ms, current 225.5–243.3 ms; median of the six per-round ratios **0.839 (−16 %)**, median-of-medians 271.2 → 231.6 (−14.6 %), mean 282.6 → 234.5 (−17.0 %), faster in 6 of 6 rounds. `P=100` −3.8 % (41.5 vs 43.4 ms); `P=1000` parity within this box's bimodality.
Executed ops per generation `P=500`/`P=1000` 11 159 → 10 352 (−7 %).
The wall-clock win exceeds the op win because the numpy interpreter charges ~13 µs per executed op (operand resolution + `_validate_result` + `Tensor` construction, ~63 % of a generation), so removing an op from a 20-iteration loop body is worth 20× removing a top-level one — and the biggest single win is not an op removal at all but moving the 20 compatibility kernels out of a `while` region (see `## Distance Rows`).
A generation is ~10.7 k executed ops, so every candidate change is judged by `executed_op_count × 13 µs`, never by IR lines alone; the count is obtained by walking `op.regions` of the built graph and weighting each `while` body by its trip count (`species_size`).

| P | evaluate | update_species | breed (`update_species` + `_create_next_generation`) | speciate | tell | `Pipeline.step` |
| --- | --- | --- | --- | --- | --- | --- |
| 100 | 8.5 | 1.3 | 21.9 | 28.6 (26.8 before the wave) | 50.3 (50.5) | 58.9 (58.9) |
| 500 | 41.1 | 1.8 | 60.5 | **230.0** (237.1) | 294.1 (304.4) | 335.2 (354.2) |
| 1000 | 96.3 | 2.2 | 116.7 | **388.1** (439.6) | 511.8 (562.5) | 608.1 (618.5) |

`tell` is `speciate` 57 % / 78 % / 76 % and breeding 43 % / 21 % / 23 %; the species update is ~1 %.
The pipeline's own graph (`benchmarks/run_benchmark.py --impl new --backend numpy --cases xor_p100,xor_p500,xor_p1000 --generations 20`, one repeat, warmup 3) reports **63.6 / 334.2 / 571.6 ms per generation**, i.e. the same numbers as `Pipeline.step`.
Against the declared legacy JAX baseline (51 / 109 / 157 ms per generation) the cycle is ~1.2x / ~3.1x / ~3.9x over budget at `P=100 / 500 / 1000`.
Live species slots in the measured states: 3 / 20 / 20 at `P=100 / 500 / 1000` — the sparse gate is what keeps `P=100` on the `cond` path.
The generation's distance work is `2 · species_size · pop_size` = 4 000 / 20 000 / 40 000 pairs at 6.7 / 11.5 / 9.7 µs per pair; that product, not the dispatch count, is what `speciate` measures.
Kernel width scaling at `P=1000` (`prepare_distance` + one `measure_distance` call, tiled digest and tiled centres, per-pair µs): 1 000 rows/call 13.0, 2 000 10.3, 4 000 **9.7**, 10 000 10.3, 20 000 13.8 — the 4-centre optimum and the 13.0 µs of the current one-centre-per-call shape bracket `speciate`'s 388 ms, and the whole-table call is the slowest option.
Dressed static gene geometry is the dominant cost driver at `P=1000` (whole `Pipeline.step`, same algorithm, largest real genome 6 nodes / 5 connections): `max_nodes/max_conns` 50/100 → 593.6 ms, 20/40 → 279.9, 10/16 → 178.3, 6/8 → 147.2 — i.e. the padded gene axis (`max_conns` wide in `measure_distance`, `max_nodes` wide in the node family) multiplies every per-pair op, and a 2.4x smaller static geometry is worth 3.3x end to end.
Graph build: the pipeline's one-off whole-generation build is 0.51 / 0.67 / 0.83 s at `P=100 / 500 / 1000` (`builds == 1` per shape signature); the speciate graph alone is 1 377 IR ops and the breed graph 2 014.
`breed`'s 2 014 IR ops at every population size: `genome.execute_mutation` 1 359 (67 %), `genome.execute_crossover` 432 (21 %), this node's `update_species` 189 (9 %: `_kernels.pair_parents` 38, `spawn_from_target` 26, `_stagnation` 19, `block_layout` 16, `const` 15, `spawn_numbers_by_rank` 15), `neat.py` itself 29 (1.4 %: `_create_next_generation` 17 = 8 parent gathers + 3 key-split + 4 elite `select` + 2 rank reshapes, `_new_gene_keys` 9, `_sanitize_fitness` 5).
Timed as separate executables: `update_species` 1.31 / 1.83 / 2.23 ms and `_create_next_generation` 21.95 / 55.06 / 105.73 ms at `P=100 / 500 / 1000`, while the algorithm-owned half of the latter (the 8 gathers plus the elite `select`s, genome calls removed) is 0.19 / 0.45 / 0.75 ms — under 1 % of breed, so breed is genome-kernel bound and no algorithm-owned op removal can move it.
`tell` increments `generation` through `_as_scalar_f32`, which emits the `cast`/`reshape` only when the static dtype/rank need one, so the 0-d `float32` counter costs one `add` and no no-op op.
**Measured and rejected: unrolling the speciation loops** (paired A/B inside one session, HEAD and the modified tree alternating, `benchmarks/backend_measure.py --backend numpy --pop P --generations 20|40 --warmup 3|5`, medians per run).
Both phases visit the static `species_size` axis, so both *can* be unrolled into static Python loops with `slot` an `int32` constant; the measurements say not to.
Both phases unrolled (with `distance_fn.matrix` pre-measurement dropped and loop-invariant constants hoisted to one op each): `P=500` **+7 %** (222.3–232.0 → 240.4–251.3 ms), `P=1000` −7 % (462–487 → 421–439), `P=100` −3 % (42.6–44.6 → 41.3–42.7), build 0.69 → 1.35 s, static graph 5 384 → 10 437 ops and **peak RSS +55 %** (539 → 825 MiB at `P=500`, 977 → 1 519 MiB at `P=1000`).
Executed ops are unchanged — wrapping `evoxc.backends.numpy.kernels.dispatch` counts 10 671 → 10 524 per generation — so the `while` region costs ~1 % of a generation; what unrolling changes is that every kernel temporary of every slot stays live in the interpreter's environment dict until the graph returns, which is both the RSS growth and the `P=500` regression (its absolute work is smallest, so it pays the cache pressure without the offsetting carry-copy saving that shows up at `P=1000`).
Phase 1 unrolled with phase 2 left a scan: parity everywhere (`P=500` 233.3 vs 233.3 ms, `P=1000` 463.7 vs 464.1, `P=100` 44.6 vs 45.1 — means of paired 40-generation runs; 20-generation runs read `P=1000` as −3…−5 % and `P=100` as −2 %), peak RSS back to the scan form's, build +0.12 s.
A 0–2 % effect inside this box's ±10 % drift does not pay for the restructure, so both phases stay `evoxc.scan`s.
**Kept: narrow scan-2 carry + hoisted body constants.** `P=500` median ratio 0.998, `P=1000` 1.002 (8 paired alternating rounds of 6–10 generations each, medians of per-generation times, `min` statistic 211.9 vs 220.9 ms at `P=500` under the benchmark's own 20-generation protocol), `P=100` **0.962** (41.5 vs 43.4 ms, the same direction in 8 of 8 rounds) — the gated path is where the win lands, because there every skipped slot duplicates the body's constant ops inside an `if` region while the genome kernel runs only for live slots.
Executed ops per generation 10 671 → 10 352 (−3.0 %), static graph 5 384 → 5 371, scan-2 body 234 → 221 ops, scan-2 `while` carry 9 → 7 values; build time unchanged (0.69 / 0.77 s at `P=500 / 1000`), peak RSS unchanged.
The dense-path cells (500/1000) sit inside this box's drift, so the change is reported as *neutral* there and *clearly positive* on the gated path, never as a win at large `P`.

## Known Issues
This node contributes **no** deferred op and no rank>=3 `scatter` to a generation graph: an audit of the full `Pipeline` graphs (`setup` plus one `step`, pop 100 and pop 1000, i.e. both the gated and the dense speciation path) reports 0 unsupported ops, and the `setup` graph lowers with `evoxc.lower(graph, backend="iree")`.
The `step` graph still does not lower, on foreign nodes only: `cumsum` at `genome/operations/mutation/cycle.py:121`, and `uint32`/`uint64` arithmetic at `common/tensorization.py:179-180`, `genome/operations/distance/digest.py:469/475/476`, `genome/operations/alignment.py:112/180-187/236-237/396/408-419` and the `uint32` constant `common/tools.py:159` (iree's declared dtype set is float16/32/64, int8/16/32/64, bool — no unsigned).
The repo-level performance contract is still missed, now by ~0.8x / ~2.1x / ~3.0x at `P=100 / 500 / 1000` (~42 / ~232 / ~480 ms against the legacy JAX 51 / 109 / 157), and this subtree is at its own floor: `speciate` is 40 000 pairs at 9.7 µs each at `P=1000` (388 ms), and every grouping of those pairs measured within ±10 % of that (one call per centre 13.0 µs/pair, 2 / 4 / 10 / 20 centres per call all slower end to end, whole-table call 13.8).
The pair count `2 · species_size · pop_size` is irreducible under NEAT's assignment semantics: each phase needs its own genuinely different centres and a created species takes the first unassigned genome as its centre, so its row depends on every earlier assignment.
The per-pair cost is genome-owned: one `measure_distance` call is ~309 traced ops over `(rows, 2 · max_conns)`-wide gene arrays, and the padded `max_conns`/`max_nodes` axes are what multiply it — shrinking the statics from 50/100 to 10/16 cuts the whole `Pipeline.step` 3.3x at `P=1000` (593.6 → 178.3 ms) with identical semantics. `DefaultDistance`'s homologue merge also sorts `2·M` key rows per pair, so the same population genome is re-sorted once per centre; an edge-proportional or centre-prepared distance kernel in `genome/operations/distance` is the next real lever and lies outside this node.
Peak RSS at `P >= 500` growth is likewise genome-owned: the retained temporaries are `prepare_distance`/`measure_distance`'s `(rows, 2 * max_conns, key_width)` masked gathers plus the digest itself, and forcing one centre per call already costs the full increase.
Breeding is the second hot spot (21.9 ms of a 50.3 ms `tell` at `P=100`, 116.7 of 511.8 at `P=1000`) and is 67 % `genome.execute_mutation` / 21 % `genome.execute_crossover`; the algorithm-owned part measures 0.19 / 0.45 / 0.75 ms (under 1 % of breed), so no algorithm-owned win is left in this subtree.
`_new_gene_keys` is correct as specified: `new_node_keys (P,) int32` / `new_conn_markers (P, 3) int32`, each strictly above that individual's own maximum (verified for `DefaultConn` at `P=100` and `OriginConn`'s `historical_marker` path at `P=1000`), `arange` offsets baked into int32 constants with no int64 promotion.
`show_details` is correct on the current genome contract (`node_exists(node_keys)` / `conn_exists(conn_keys)`, key array only); the reported `TypeError: BaseGenome.node_exists() takes 2 positional arguments but 3 were given` came from passing a `cfg` state at those two call sites and is gone — verified by driving real runs (DefaultGenome, RecurrentGenome, OriginConn, `pop_batch_size` chunking, `P=2`) and calling `show_details` on host values with no try/except. `Pipeline._details` is what used to print "…failed (…); continuing the run"; it lives in `pipeline.py` and no longer has anything to swallow.
`next_species_key` must stay strictly greater than every live key; `setup` seeds it at `1.0` (slot 0 uses `0.0`) and each created species increments it, so the invariant holds as long as the counter is never rebound by hand — a colliding counter silently merges two species' `member_count`.
`DistanceRows` must keep handing `measure_distance` a **population-shaped** centre: the batch-1 form reaches the genome's dense single-centre route (`DefaultDistance(dense_single_centre=True)`), whose documented key-range precondition this subtree's per-individual key allocator violates once `pop_size > genome.key_span` (`max_node_keys = base + arange(1, P + 1)` with `key_span = 150` at `max_nodes/max_conns = 50/100`).
Measured: swapping `single_payload` to a batch-1 payload (dropping the `repeat_leading` tile) keeps the bit-identity oracle green at `P=100` and breaks it at generation 1 of `P=500` and `P=1000` — 20 mismatching arrays (the four centre tables, `idx2species`, `member_count`, `species_keys`).
The tile is therefore load-bearing, not a workaround: it selects the row-wise merge, the only route that matches an out-of-range key against itself instead of dropping it (genome-side hazard, reported upward).
`test/unit/algorithm/test_spawn.py` encodes the brief's `loc(p) = #{i : spawn[i] > 0 and E[i] <= p} - 1`; that count is wrong for a live zero-spawn species in the middle of the table, so the oracle needs the "last qualifying block" form to agree with `block_layout` (cross-node item, `test/` is not writable from here).
`test/unit/algorithm` is 139 passed and 4 failed; the four failures are the `[XPASS(strict)]` markers in `test_neat_tell.py` that pin the fixed `_new_gene_keys` reshape defect (`test_tell_increments_the_generation_before_the_species_update`, `test_tell_preserves_the_population_size_and_payload_shapes`, `test_tell_is_deterministic_for_the_same_state_and_fitness`, `test_tell_advances_the_randkey`) — a test-owner action, not a src defect.
`test/integration/test_xor_run.py` fails all three tests on a pipeline/test contract disagreement, not in this subtree: `Pipeline.auto_run` returns `best_genome` as one individual **without** a population axis (what `src/tensorneat/CONTEXT.md` documents), while the tests require a one-individual *batch* `(1, max_nodes)`; two of them then feed that flat payload to `genome.transform` and hit `ValueError: not enough values to unpack (expected 2, got 1)` at `genome/base.py:654`. `Pipeline.show` itself works on this tree (`pipeline.py::_one_individual` normalises the payload).
This node is additionally verified by driving `Pipeline` end-to-end on the numpy backend with the real `DefaultGenome`/`XOR` (pop 8/100/300/301/500/1000, `rank` and `fitness` spawn modes, `max_stagnation` kills, `OriginConn`'s `historical_marker` branch, `pop_batch_size` chunking, `RecurrentGenome`, XOR3d) plus bit-identical repeat runs.
`_new_gene_keys` is per-individual by contract (see `## Gene Keys & Padding`), so the genome must not assume keys are globally unique across the population.

## Constraints
No `jax`/`torch`/`numpy.random` import; `import tensorneat` must stay free of them (release gate). `numpy` appears only as `numpy.finfo(dtype).max` in `_finite_bound`, a host-side scalar read.
No branching, `bool()`, `len()` or host reads on traced values — static Python `if`/`for` over constructor configuration is fine and unrolls at trace time.
No `.at[]`, no `cumsum` on the compute path (prefix sums are triangular `dot`), no `tril`/`triu`/`flip`/`iota`, no rank>=3 `scatter` (use `scatter_row` or a static constant — see `## StableHLO Export Rules`), no `where=`/`initial=`/`any`/`all`/`take`/`searchsorted`/`repeat`, no `vmap` over anything region-bearing, no `fori_loop`/`switch`.
Shapes are static: `P`, `S`, `max_nodes`, `max_conns` are fixed at build time; a change forces a rebuild by the pipeline.
Errors: configuration validation raises `common.errors.TensorNEATError` with a named message; trace/shape/backend errors propagate unchanged and are never re-wrapped or silently fallen back.
State keys touched: `randkey`, `pop_size`, `pop`, `generation`, `species`, plus the genome/problem keys documented above — nothing else.
Files stay under 1000 lines: `species.py` (948) the species algebra, `neat.py` (961) the population cycle, `_kernels.py` (942) the graph helpers, `_distance.py` (299) the digest-backed rows, `_validation.py` (128) the host validators.

## Known Risks
`evoxc.random.randint` needs a symbolic `high` that stays ≥ 1 at runtime — guaranteed by `max(survive_size, 1)`, which also removes the legacy NaN case.
`evoxc.scatter` is replacement-only, so every "accumulate" update is restructured as reduce-then-scatter-once (spawn residual, stale-write elimination in the speciation scans).
Wide `while`/`scan` carries plus `select` can crash iree codegen (`AffinityAnalysis`), hence the packed-carry convention; the numpy reference backend is unaffected.
Reading a dead slot's centre rows would return stale integers, not NaN — always gate centre reads behind `isnan(species_keys)`.
`DistanceRows`' tiling is a `broadcast_to` + `reshape`, i.e. a real copy: one population-shaped row materialises `P` copies of that centre across every digest leaf (`P=1000` × 8 leaves × the padded gene width), and it is the only avoidable piece of array traffic left in `speciate` at `P <= 500`.

## Test Strategy
Tests live in the sibling `test/` tree (read-only from here): `test/unit/algorithm/` covers the spawn/selection/stagnation algebra, the block layout and the speciation scans on fixed payloads.
`test/parity/algorithm/neat/` compares against `benchmarks/legacy_jax` on fixed gene payloads — identical selection/speciation outcomes for the deterministic steps, and only the divergences listed above allowed elsewhere.
`test/integration/` evolves XOR to its fitness target with a fixed seed and asserts run-to-run determinism for a `(seed, config, shape)` triple.
Every kernel here must be a pure function of `(state, key)`: no dict-order dependence, no host RNG, no float→int conversion without an explicit mask.
Distance rows are verified by whole-run bit-identity: a `npz` dump of `fitness_history`, the four payload arrays, all ten `SpeciesState` leaves, `randkey` and `generation` per generation, compared with `tobytes()` equality (padding NaNs make `np.array_equal` report self-vs-self differences) against the pre-change tree at pop 100 (6 generations, gated path), pop 500 (4) and pop 1000 (3, dense path) — `common.tools.to_numpy` first, because hashing an `evoxc.Tensor` directly hashes the object pointer rather than the data.
The sparse/dense switch, the block size and the `cond` placement are all static trace-time choices, so a shape or gate regression shows up as a build error in `test/release_gate_no_jax.py`'s two-generation XOR run rather than as a silent numerical drift.

## Export Audit (how the StableHLO rules are verified)
Recipe, cheap and per-op: monkeypatch `evoxc.build` to also keep `evoxc.trace(fn, *specs)` for every graph TensorNEAT builds, then walk `graph.module.functions[*].region.blocks` recursively (`op.regions` included, skipping the result-less `return` op) and classify each op with `evoxc.backends.stablehlo.ops.status(op.name)`; for `scatter` additionally check the operand rank and `op.attributes["axis"]` against the rank-1/rank-2 rule. `op.location` gives the exact `file:line` of an offender, so a regression is attributable without reading diffs.
Finish with `evoxc.lower(graph, backend="iree")`, which fails fast on the first unsupported op and does not need a working end-to-end backend.
The three rewritten primitives were also lowered, compiled and **run** on iree (`prefix_sum_exclusive`, `block_layout`, `scatter_row`, plus a `scan` that writes its table each step): every output is numpy-identical except a float32 scan-accumulated sum, which differs by ~3e-05 because iree reassociates the reduction — the written table itself is bit-identical.
Bit-identity evidence for the rewrite on numpy: `fitness_history`, the four payload arrays, all ten `SpeciesState` leaves, `randkey` and `generation` dumped per generation and compared with `tobytes()` equality against the pre-change tree at pop 100 (4 generations, gated) and pop 500 (2) in both spawn modes and at pop 1000 (2, dense path) — 144/72/72 arrays, zero mismatches. Degenerate shapes smoke-tested on both trees: `species_size` 1, 2, 3, 8.

## Routing Table
| Path | Responsibility |
| --- | --- |
| `./neat.py` | `NEAT`: population cycle (`setup`/`ask`/`transform`/`forward`/`tell`), fitness sanitising, next-generation creation, `pop_batch_size` scan chunking, the per-generation distance digest wiring, host-side `show_details`. |
| `./species.py` | `SpeciesState` (frozen dataclass) and `SpeciesController`: species fitness, stagnation, descending sort, spawn allocation, parent pairing, block layout, fixed-length speciation scans. |
| `./_kernels.py` | Plain graph helpers behind `species.py`: constants, payload helpers, sort/prefix-sum primitives, membership, spawn allocation, block layout, parent pairing, `GATED_ROWS`/`gated` (the static lazy-row gate) and the two speciation scans, payload reading. |
| `./_distance.py` | `DistanceRows` (digest-backed `distance_fn`; population-shaped `(P,)` row and static-loop `(S, P)` matrix forms) and `repeat_leading`. |
| `./_validation.py` | Host-side constructor validators shared by `species.py` and `neat.py`. |
| `./__init__.py` | Re-exports `NEAT`, `SpeciesController`, `SpeciesState`. |
