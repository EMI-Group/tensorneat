# NEAT (`tensorneat.algorithm.neat`) — population cycle + species bookkeeping

## Intent
Everything in a NEAT generation that is neither network mathematics (genome) nor task evaluation (problem): the population payload, the generation counter, speciation, stagnation, spawn allocation, parent pairing, and the crossover/mutation dispatch.
`NEAT` (`neat.py`) is the algorithm the pipeline drives; `SpeciesController` + `SpeciesState` (`species.py`) own the species algebra; `_distance.py` owns the compatibility-distance rows over one generation's population digest.
Every kernel here is batch-first over `P = pop_size` and compiled by `pipeline.py` into the single per-generation evoxc graph, so no function may branch on a traced value or read the host.
Measured on the numpy reference backend with the real `DefaultGenome(max_nodes=50, max_conns=100)` and `XOR`, `species_size=20`: `Pipeline.step` runs in ~43 / ~170–185 / ~311–320 ms per generation at `P=100 / 500 / 1000` (see `## Measured Cost`); the interpreter, not the data, dominates, and the padded gene axis is what multiplies every per-pair op.
The dense distance table costs ~49 ms of the `P=500` generation and ~90 ms of the `P=1000` one, measured as the gap to the same tree with the table replaced by zeros (~115 / ~225 ms) — the rest is evaluation and breeding.
The layer's own work is one population digest plus `2 · species_size` distance calls in `speciate` and `O(pop_size)` selection/crossover/mutation; it never materialises a `max_nodes × max_nodes` tensor.

## Layout (five files, each under 1000 lines)
`species.py` — `SpeciesState` plus `SpeciesController`: the documented, traced-facing methods and the species algebra, each delegating its tensor body to `_kernels.py`.
`_kernels.py` — plain graph helpers (no `@evoxc.defn`, no state): spawn allocation, block layout, parent pairing, membership, sort/prefix-sum primitives, the export-safe write/constant primitives (`scatter_row`, `first_centre_table`, `static_i32`/`static_f32`, `reversed_index`), `gated` (the static lazy-row gate) and the two speciation scans.
`_distance.py` — `DistanceRows` (the digest-backed, population-shaped `distance_fn` `speciate` consumes; its `matrix` form measures a whole species table inside one fixed-length scan) and the `repeat_leading` tiling primitive.
`_validation.py` — host-side configuration validators shared with `neat.py` (`positive_int`, `non_negative_int`, `unit_interval`, `positive_float`), each raising `TensorNEATError` with the parameter name.
`neat.py` — `_sanitize_fitness` and `NEAT`: setup, ask/transform/forward, the `tell` algebra, `_batched` chunking, host-side `show_details`.

## API Surface
`__init__.py` re-exports `NEAT`, `SpeciesController`, `SpeciesState`.

`NEAT(BaseAlgorithm)` — `neat.py`:
`__init__(genome, pop_size, species_size=10, max_stagnation=15, species_elitism=2, spawn_number_change_rate=0.5, genome_elitism=2, survival_threshold=0.1, min_species_size=1, compatibility_threshold=2.0, species_fitness_func=evoxc.max, species_number_calculate_by="rank", pop_batch_size=None)` — constructor kwargs are API-compatible with the pre-rewrite library.
`setup(state=None) -> State` requires `state.randkey` (`TensorNEATError` otherwise), registers `pop_size` (static int), `pop` (`GenomeTensors`, axis 0 = individual), `generation` (0-d float32) and `species`, and advances `randkey`.
`setup` is **host-side**: it builds and runs one small init graph (`split(randkey)` → `genome.initialize` → species seeding) and cannot be called inside an active trace.
`ask(state) -> GenomeTensors` (zero ops, returns `state.pop`); `transform(state, pop=None)` / `forward(state, transformed, inputs, pop=None)` delegate to the genome; `tell(state, fitness) -> State` runs the algebra below.
`prepare_forward(state, transformed, pop=None, samples=1)` / `forward_prepared(state, transformed, prepared, inputs)` are the same delegation split in two, calling `genome.prepare_pass` / `genome.forward_prepared` with the payload (`pop`, else `state.pop`) and `samples`; `forward` is their composition.
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
`matrix` runs one fixed-length `evoxc.scan` over the `S` centre rows and measures one population-shaped `genome.measure_distance` call per iteration; it returns the stacked `(S, P)` table, so its callers and its `(K, P)` contract are unchanged.
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
Speciation, scan 1 (centre reservation): per live slot `i`, `d_i` is the `(P,)` row `distance(centre_i, pop)` — computed behind one `cond` per iteration when the table is sparse, and on the dense path read by `gather` from the rows `distance_fn.matrix` measured *before* the scan (see `## Distance Rows`) —, `closest = argmin(select(unassigned, d_i, +inf))`, then assign `idx2species[closest] = species_keys[i]`, re-elect `pop[closest]` as the centre and set its distance to `0`; a dead slot or an empty unassigned set writes nothing.
Speciation, scan 2 (member assignment): a live slot assigns every genome with `d < compatibility_threshold` and `(unassigned | d < current distance)`; a dead slot with anything still unassigned creates a species keyed `next_species_key` seeded by the first unassigned genome, then assigns the same way; an iteration with nothing to do writes nothing.
Scan 2 carries `assign`, `dist`, `keys_ext` plus two rank-1 per-slot tables — `seed_table (S,) int32` and `created_mask (S,) float32`, one single-element rank-1 `axis=0` `scatter` each — and **not** the four centre gene tables: its centre writes were the identity on every non-creating slot, so only a created row (`pop[seed_table[slot]]`) ever differs from the scan-1 table.
The four centre tables are rebuilt once after the scan as `select(created[:, None…], gather(population.<field>, seed_table, axis=0), scan1_table)` (four gathers + four selects at top level, one reshaped mask per field rank), which leaves the dead rows' stale integers untouched and the result bit-identical.
Scan 1 still carries the centre tables (it re-elects centres for live slots), so a scan-1 slot's centre write stays an unconditional `scatter_row`.
Both scans close over their graph-time constants (`I_INF`, `0`, `0.0`, `iota_i32(pop_size)`, the compatibility threshold and the gated false-branch zero row) instead of rebuilding them per iteration — one `constant` op each at the top level instead of one per slot.
Speciation post-pass: leftovers adopt the last species key, slots created this generation get `best_fitness = -inf` / `last_improved = generation`, and `member_count` is one masked compare-sum over `(S, P)`.
Body sizes: scan-1 body 55 traced ops, scan-2 body 221 traced ops with a 7-value `while` carry.

## Distance Rows (one digest per generation; one population-shaped call per centre)
Both phases need one `(P,)` compatibility row per species slot, and the same population is measured `2 · species_size` times, so `NEAT.tell` digests it once: `genome.prepare_distance(state, pop)` sorts each gene family of the population into a `DistanceDigest` (batch `P` on every population-indexed leaf) and `DistanceRows` turns centres into rows against it without re-sorting the population.
Call shape: the `distance_fn` contract is population-shaped — `(state, centre, pop) -> (P,)` with **both** payloads batched `P` — so `DistanceRows.row` tiles its one centre row to `P` rows with `repeat_leading` and makes exactly one `genome.measure_distance` call.
The explicit tile is what makes one call per centre legal for every gene variant: `measure_distance` accepts a centre batch of `1` (internal broadcast) or `P` (row-wise) and rejects anything else, so a centre that is not already population-shaped is tiled rather than handed over as one row.
`__call__` is single-valued: a batch of `P` is measured as it stands, a batch of `1` is tiled, and anything else raises — `speciate` always passes a population-shaped centre, so a whole centre table must be measured through `matrix` by name.
The row of a slot is known before the iteration reads it, because every `scatter` in both scans writes its own slot only: scan 1's row for slot `i` is `distance(pre-scan centre i, pop)`, and scan 2's row for a **live** slot is `distance(post-scan-1 centre i, pop)`. Only a slot that *creates* a species has a carry-dependent centre (`pop[seed]`, `seed` = first unassigned genome).
The dense path pre-measures **scan 1** outside the scan: `rows1 = distance_fn.matrix(state, centre_payload)` runs once before `evoxc.scan(reserve, …)` and each iteration reads `evoxc.gather(rows1, slot, axis=0)`.
That is exactly the row the iteration would have measured, because scan 1's centre for slot `i` is the pre-scan carry row `i` (every `scatter` in `reserve` writes its own slot and `arange(S)` visits each once).
`matrix` measures those rows inside **one fixed-length `evoxc.scan`** (body: gather centre row `slot`, tile it with `repeat_leading`, one `genome.measure_distance` call; output stacked to `(S, P)`), not as `S` top-level calls.
The scan is a memory scope, not an arithmetic change: `evoxc`'s numpy interpreter pops a region's environment per iteration, so only one centre's transients are alive at a time, and the kernel still stays out of the reservation `while` body (the loop body keeps its 55 ops).
Measured consequence at `P=500 / 1000`: peak RSS 240.5 / 391.9 MiB against 454.4 / 810.0 MiB for the top-level spelling and 228.2 / 368.2 MiB for a table that returns zeros, i.e. the residual cost of the table is one call's transients (~12 / ~24 MiB) instead of `S` of them (~226 / ~441 MiB, ~11 / ~22 MiB per hoisted call).
It is also faster than the top-level spelling once the digest-backed kernel is cheap: 170.0–185.4 vs 188.2–203.6 ms at `P=500` and 319.6–311.0 vs 344.9–337.7 ms at `P=1000` (paired rounds, `## Measured Cost`), because the interpreter's per-op cost is lower at the top level only in so far as it avoids region entry, while the top-level form pays for a much larger live set.
Scan 2 keeps its in-scan measurement on both paths, because a slot that creates a species measures the carry-dependent `pop[seed]`; pre-measuring it needs a per-iteration `cond` and measured neutral-to-slower with ~50 % more peak RSS (832 vs 556 MiB at `P=500`).
The dense path therefore requires a `distance_fn` with a `matrix` method and raises `TensorNEATError` naming it otherwise; the gated path keeps working with any callable.
**Sparse table** (`gate` true, `species_size * pop_size <= GATED_ROWS`): one `cond` per scan iteration wraps that row — `gated_row(active, compute_row, operands)`, predicate `live & any_pending` (scan 1) or `live | create` (scan 2), false branch `enp.zeros((pop_size,), float32)` discarded by the iteration's own `claim` / `take` mask; the gate exists because an unconditional in-loop call costs 55.6 vs 30.1 ms of `speciate` at `P=100` and 211.0 vs 152.2 ms at `P=300` (ungated vs gated), i.e. the `if` regions pay off exactly while the table is sparse.
**Dense table** (`gate` false): scan 1 carries no kernel — it reads a pre-measured `matrix` row by `gather` — and scan 2 measures its `selected_centre` in-loop on both paths (`pop[seed]` for a creating slot, the closed-over scan-1 centre row otherwise), because a creating slot's centre depends on the scan's own assignments.
Grouping several centres into one `measure_distance` call is not used, for two independent reasons: the digest's `MeasureStatics` bake `pop_size` into population-sized constants (`pad_column_i64 (P, 1)`, the merge's row offsets, the attribute-gather base), so a `k · P` centre batch is rejected (`ShapeError` in `measure_alignment`) unless a **second digest is prepared from a `k`-tiled population** — an extra population sort per generation; and grouping does not reduce the retained bytes, because the sum of one call's transients over `S` calls is the same ~`S · P`-proportional total however the calls are cut. At `P=500` the whole table is ~49 ms of a 164–185 ms generation, so the ceiling on any grouping win is a few percent against a prepare that costs a few percent.
Bit-identity is a property of the genome kernel, not of this module: the compatibility kernel is row-independent (elementwise per-pair ops plus per-row reductions over the gene axis, one sorted alignment per row), so the in-scan rows are the same floats the top-level rows were — verified end-to-end (63 arrays per generation, zero mismatches; see `## Test Strategy`).
Dispatch budget per generation: `1` `prepare_distance` + `2 · species_size` population-shaped `measure_distance` calls, against `2 · species_size` `execute_distance` calls before the digest existed.
`GATED_ROWS` is a module constant so the public `NEAT` constructor is unchanged.

## Scan Patterns (why fixed-length scans are exact)
`evoxc` de-sugars `scan` into `while` + `gather`/`scatter`, so a trip count must be static and the carry must keep identical dtypes and shapes: both speciation phases are `scan(f, carry, enp.arange(species_size))` with per-slot masks, never a data-dependent `while_loop`.
The legacy early-exit conditions survive as masks, and because every write is mask-then-single-`scatter` (`scatter_row` for the rank-3 gene tables) the extra iterations are provably no-ops.
Carries are kept narrow and packed (`idx2species` and the running distance as two `(P,)` arrays; the species keys plus `next_species_key` as one `(S+1,)` array), which keeps the carry list short and every per-step op small.
Scan 1 carries those three plus the four centre gene tables (9 values); scan 2 carries the same three plus the rank-1 `seed_table`/`created_mask` (7 values), and the created centre rows are rebuilt after it — see `## Species Algebra`.
`DistanceRows.matrix` is a third, unrelated kind of scan: its trip count is `species_size` and its carry is a one-element dummy, so it writes no table at all — only its stacked per-iteration output `(S, P)`. It exists purely to bound the interpreter's live set (see `## Distance Rows`); a static Python loop over the same rows is the faster-but-hungrier spelling.
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
- `scan` stacks its per-step *outputs* along a new leading axis, so returning a whole `(S, …)` table from a scan body becomes a rank-4 scatter inside the desugared `while`; both speciation scans therefore return only their small per-step value (`assign` in scan 1, `create` in scan 2), and `matrix`'s scan returns one `(P,)` row per step (stacked to `(S, P)` by the scan, no write at all).
- Unsigned integer dtypes (`uint32`/`uint64`) are outside iree's declared dtype set — a graph is expected to use `int32`/`int64` only (the sibling `genome`/`common` nodes own their remaining occurrences).
The numpy reference interpreter accepts every rank, so a violation is invisible until `evoxc.lower(graph, backend="iree")`; the audit in `## Export Audit` is part of verification, not an optional extra.

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
Speciation and the distance digest draw no randomness at all, so neither the digest nor the scan that consumes it can move an RNG stream.
`pop_batch_size` chunking pre-splits the mutation key with `split_n(key, n_batches)` and passes the stacked `(n_batches,)` int64 array as the scan `xs`, so each step receives a valid rank-0 key.
Keys are rank-0 scalars, so a batched kernel takes one key and draws its own population-shaped randomness; chunked and unchunked runs therefore draw different streams and `pop_batch_size` is part of the reproducibility config.

## Dependencies (cross-node contracts this subtree relies on)
`genome` (batch-first, one rank-0 key each): `setup(state)`, `initialize(state, key, pop_size) -> GenomeTensors`, `transform(state, node_keys, conn_keys)`, `forward(state, transformed, node_keys, node_attrs, conn_keys, conn_attrs, inputs)`, `execute_crossover(state, key, winners, losers)`, `execute_mutation(state, key, genome, new_node_keys, new_conn_markers)`, `prepare_distance(state, pop) -> DistanceDigest`, `measure_distance(state, centre, digest) -> (P,)` with a centre batch of `1` (broadcast internally) or `P` (row-wise).
`measure_distance` rejects a centre batch that is neither `1` nor the digest batch, and its digest carries population-sized statics (`MeasureStatics`: the merge row offsets, `pad_column_i64`, the attribute-gather base), so only `1` and `P` centre batches are addressable — a `k · P` batch would need a digest prepared from a `k`-tiled population (see `## Distance Rows`).
The kernel is strictly row-independent, so any legal batch width returns the same floats.
`execute_distance` remains the documented single-pair contract and is no longer called from here.
`genome` statics: `num_inputs`, `num_outputs`, `max_nodes`, `max_conns`, `node_gene.length`, `conn_gene.length`, `conn_gene.uses_historical_marker`; helpers `node_exists`, `conn_exists`, `max_node_key`, `max_conn_marker`.
`genome.utils.GenomeTensors` is the payload namedtuple — the four fields in declaration order.
`common`: `State` (immutable, `register`/`update`/`registered_keys`), `StatefulBaseClass`, `errors.TensorNEATError`, `tools.{rank_elements, fetch_first, argmin_with_mask, flat_gather, to_numpy}`.
`pipeline`: creates `randkey` with `evoxc.random.key(seed)`, calls `setup` on the host, drives `ask → transform → forward/evaluate → tell`, and calls `show_details` on host-readable values`.

## Measured Cost (numpy backend, real `DefaultGenome(max_nodes=50, max_conns=100)`, XOR, `species_size=20`, seed 42)
Protocol: one configuration per fresh process, `benchmarks/backend_measure.py --backend numpy --pop P --generations 20 --warmup 3` (that harness's numbers are `Pipeline.step`), rounds run one at a time and interleaved between trees so box drift cancels; the harness reports the process's lifetime peak RSS.
`base` is commit a697531 (before the digest-backed `matrix`), `prev` is dff1643 (the table measured as `S` top-level calls), `current` is this tree (the table measured inside one `evoxc.scan`):

| P | base `step` mean (median) | prev `step` mean (median) | current `step` mean (median) | base / prev / current peak RSS | base / prev / current build |
| --- | --- | --- | --- | --- | --- |
| 100 | 43.7 (43.5) | 41.0 (39.0) | 42.5 (41.6) | 118.1 / 114.3 / 114.5 MiB | 0.29 / 0.25 / 0.26 s |
| 500 | 230.7–266.3 (215.9–279.8) | 188.2–203.6 (184.5–206.0) | **170.0–185.4 (166.8–184.6)** | 260.6 / 454.4 / **240.5 MiB** | 0.35 / 0.55 / **0.32–0.34 s** |
| 1000 | 484.0–464.6 (489.7–505.3) | 344.9–337.7 (341.0–336.2) | **319.6–311.0 (307.5–298.0)** | 431.5 / 810.0 / **391.9 MiB** | 0.44 / 0.62 / **0.41 s** |

Against `prev` the in-scan table is −10 % / −8 % wall and −47 % / −52 % peak RSS at `P=500 / 1000`, and −16 % / −20 % of build (one kernel traced instead of `species_size` copies: static generation graph 4116 → 2137 traced ops, 12 `while` regions).
Against `base` it is −26…−30 % wall at `P=500`, −34 % at `P=1000`, with peak RSS 8 % / 9 % below base rather than 2.1× above it; the gated path (`P=100`) is unchanged by construction.
Against the declared legacy JAX baseline (51 / 109 / 157 ms per generation) the cycle is now ~0.83x / ~1.6–1.7x / ~2.0x at `P=100 / 500 / 1000`.
**Retention rule (measured, not inferred).** A tree whose `matrix` returns zeros instead of measuring peaks at 228.2 MiB (`P=500`) and 368.2 MiB (`P=1000`), i.e. every other phase of the generation together fits in that; a table measured as `S` top-level calls peaks at 454.4 / 810.0 MiB, exactly `S = 20` times 11.3 / 22.1 MiB per call; a table measured inside one scan peaks at 240.5 / 391.9 MiB, i.e. one call's transients (~12 / ~24 MiB).
So the retained term is *one `measure_distance` call's* live `(P, 2 · max_conns)` intermediates (~14 wide `int64` blocks at `P=500`), and a kernel's live set is charged once per enclosing block: at the entry block it is charged for the rest of the generation, in a scan body once per iteration.
**Rule: never hoist a kernel out of a `while`/`scan` without measuring peak RSS; here the hoist is both slower (188–204 vs 170–185 ms at `P=500`) and 1.9× hungrier, so memory scoping and wall time point the same way.**
The `setup` graph is 74 traced ops; the generation graph is 2137 (2 `while` regions for the speciation scans plus the mutation chunk scan and the forward level loop).
`breed` remains genome-bound: `genome.execute_mutation` dominates the breed graph (67 % of its IR ops), `genome.execute_crossover` 21 %, this node's `update_species` 9 % and `neat.py` 1.4 %, so no algorithm-owned op removal can move it.
Dressed static gene geometry is the dominant cost driver at `P=1000` (whole `Pipeline.step`, same algorithm, largest real genome 6 nodes / 5 connections): `max_nodes/max_conns` 50/100 → 506.0 ms, 20/40 → 192.2, 10/16 → 126.2 — the padded gene axis (`max_conns` wide in `measure_distance`, `max_nodes` wide in the node family) multiplies every per-pair op.
Measured and rejected, still valid under the current kernel: unrolling either speciation phase into a static Python loop (0–2 % wall inside this box's ±10 % drift, build +0.12 s, peak RSS back to the hoist's level for both phases unrolled); pre-measuring scan 2's rows (needs a per-iteration `cond` for `pop[seed]`, +50 % peak RSS for no timed gain); 2/4/10/20 centres per `measure_distance` call (wider batches grow every transient, and the digest statics block it outright — see `## Distance Rows`).
The dense-path cells sit inside this box's ±30 % session drift, so read the paired ratios above, never absolute milliseconds.

## Known Issues
This node contributes **no** deferred op and no rank>=3 `scatter` to a generation graph: an audit of the full `Pipeline` graphs (`setup` plus one `step`, `P=100` and `P=1000`, i.e. both the gated and the dense speciation path) reports 0 `deferred` ops, `scatter` operand ranks `{(1, axis 0), (2, axis 0)}` only, and 12 `while` regions in the step graph.
Both graphs lower: `evoxc.lower(graph, backend="iree")` accepts `setup`, and `evoxc.build(..., backend="iree")` compiles and runs a whole `step` for every shipped case — 5x (pop 100) / 24x (pop 500) / 23x (pop 1000) slower per generation than the numpy interpreter on this CPU, at ~9-11 s of `iree-compile` per static shape (`benchmarks/BACKENDS.md` has the numbers, the protocol and the per-gate fix record).
The repo-level performance contract is still missed, by ~0.76x / 1.47x / 1.72x at `P=100 / 500 / 1000` (38.6 / 159.5 / 277.2 ms median against the legacy JAX 51 / 109 / 157; the per-cell protocol and the before/after table are `benchmarks/PERF_NUMPY.md`), and what remains is not in this node's distance table: the table is ~49 ms of the `P=500` generation and ~90 ms of the `P=1000` one.
This subtree is at its own floor: `speciate` is 40 000 pairs at 9.7 µs each at `P=1000`, and every grouping of those pairs measured within ±10 % of that (one call per centre 13.0 µs/pair, whole-table call 13.8).
The pair count `2 · species_size · pop_size` is irreducible under NEAT's assignment semantics: each phase needs its own genuinely different centres and a created species takes the first unassigned genome as its centre, so its row depends on every earlier assignment.
The per-pair cost is genome-owned: one `measure_distance` call is ~309 traced ops over `(rows, 2 · max_conns)`-wide gene arrays, and the padded `max_conns`/`max_nodes` axes are what multiply it — shrinking the statics from 50/100 to 10/16 cuts the whole `Pipeline.step` 4.0x at `P=1000` (506.0 → 126.2 ms) with identical semantics. `DefaultDistance`'s homologue merge also sorts `2·M` key rows per pair, so the same population genome is re-sorted once per centre; an edge-proportional or centre-prepared distance kernel in `genome/operations/distance` is the next real lever and lies outside this node.
Peak RSS at `P >= 500` is no longer this node's problem: the table's residual is one `measure_distance` call's transients (12 / 24 MiB over the 228 / 368 MiB floor), and the floor itself is evaluation plus breeding plus the `prepare_distance` digest, so any further reduction has to come from those phases or from the padded gene geometry, not from the table.
Breeding is the second hot spot and is 67 % `genome.execute_mutation` / 21 % `genome.execute_crossover`; the algorithm-owned part of it is under 1 %.
`_new_gene_keys` is correct as specified: `new_node_keys (P,) int32` / `new_conn_markers (P, 3) int32`, each strictly above that individual's own maximum (verified for `DefaultConn` at `P=100` and `OriginConn`'s `historical_marker` path at `P=1000`), `arange` offsets baked into int32 constants with no int64 promotion.
`show_details` is correct on the current genome contract (`node_exists(node_keys)` / `conn_exists(conn_keys)`, key array only), verified by driving real runs (DefaultGenome, RecurrentGenome, OriginConn, `pop_batch_size` chunking, `P=2`) and calling it on host values without a try/except; `pipeline.py` no longer has a swallow path around it.
`next_species_key` must stay strictly greater than every live key; `setup` seeds it at `1.0` (slot 0 uses `0.0`) and each created species increments it, so the invariant holds as long as the counter is never rebound by hand — a colliding counter silently merges two species' `member_count`.
`DistanceRows` must keep handing `measure_distance` a **population-shaped** centre: the batch-1 form reaches the genome's dense single-centre route (`DefaultDistance(dense_single_centre=True)`), whose documented key-range precondition this subtree's per-individual key allocator violates once `pop_size > genome.key_span` (`max_node_keys = base + arange(1, P + 1)` with `key_span = 150` at `max_nodes/max_conns = 50/100`).
Measured: swapping `single_payload` to a batch-1 payload (dropping the `repeat_leading` tile) keeps the bit-identity oracle green at `P=100` and breaks it at generation 1 of `P=500` and `P=1000` — 20 mismatching arrays (the four centre tables, `idx2species`, `member_count`, `species_keys`).
The tile is therefore load-bearing, not a workaround: it selects the row-wise merge, the only route that matches an out-of-range key against itself instead of dropping it (genome-side hazard, reported upward); `matrix`'s scan tiles one centre per iteration, so the copy is per row, not per table.
`test/unit/algorithm/test_spawn.py` encodes the brief's `loc(p) = #{i : spawn[i] > 0 and E[i] <= p} - 1`; that count is wrong for a live zero-spawn species in the middle of the table, so the oracle needs the "last qualifying block" form to agree with `block_layout` (cross-node item, `test/` is not writable from here).
The whole suite is green on this tree: `pytest` reports 551 passed / 0 failed (with the `bench`/`iree` extras) and `pytest -m parity` 53 passed, so `test/unit/algorithm` and `test/integration/` (the `Pipeline.auto_run` batch-first best-genome contract) both pass, and `test/backends` proves the `iree` portability claim above.
This node is additionally verified by driving `Pipeline` end-to-end on the numpy backend with the real `DefaultGenome`/`XOR` (pop 8/100/300/301/500/1000, `rank` and `fitness` spawn modes, `max_stagnation` kills, `OriginConn`'s `historical_marker` branch, `pop_batch_size` chunking, `RecurrentGenome`, XOR3d) plus bit-identical repeat runs.
`_new_gene_keys` is per-individual by contract (see `## Gene Keys & Padding`), so the genome must not assume keys are globally unique across the population.

## Constraints
No `jax`/`torch`/`numpy.random` import; `import tensorneat` must stay free of them (release gate). `numpy` appears only as `numpy.finfo(dtype).max` in `_finite_bound`, a host-side scalar read.
No branching, `bool()`, `len()` or host reads on traced values — static Python `if`/`for` over constructor configuration is fine and unrolls at trace time.
No `.at[]`, no `cumsum` on the compute path (prefix sums are triangular `dot`), no `tril`/`triu`/`flip`/`iota`, no rank>=3 `scatter` (use `scatter_row` or a static constant — see `## StableHLO Export Rules`), no `where=`/`initial=`/`any`/`all`/`take`/`searchsorted`/`repeat`, no `vmap` over anything region-bearing, no `fori_loop`/`switch`.
Shapes are static: `P`, `S`, `max_nodes`, `max_conns` are fixed at build time; a change forces a rebuild by the pipeline.
Errors: configuration validation raises `common.errors.TensorNEATError` with a named message; trace/shape/backend errors propagate unchanged and are never re-wrapped or silently fallen back.
State keys touched: `randkey`, `pop_size`, `pop`, `generation`, `species`, plus the genome/problem keys documented above — nothing else.
Files stay under 1000 lines: `_kernels.py` (992) the graph helpers, `neat.py` (966) the population cycle, `species.py` (950) the species algebra, `_distance.py` (306) the digest-backed rows, `_validation.py` (128) the host validators.

## Known Risks
`evoxc.random.randint` needs a symbolic `high` that stays ≥ 1 at runtime — guaranteed by `max(survive_size, 1)`, which also removes the legacy NaN case.
`evoxc.scatter` is replacement-only, so every "accumulate" update is restructured as reduce-then-scatter-once (spawn residual, stale-write elimination in the speciation scans).
The packed-carry convention stays even though the wide-carry codegen hazard no longer reproduces (9 loop-carried values compile in 6/6 fresh `evoxc.build` runs): packing is a shape-size decision (narrow `(P,)`/`(S+1,)` carries), not a workaround for a live fault.
The one real iree hazard found since is seeding a `while`-carried value with a `scatter`: that SIGSEGVs `iree-compile` 3.11 in the assembled generation graph while exporting and single-kernel compiles stay clean — seed with a one-hot `equal`/`logical_and` instead, and only `test/backends` proves portability.
`evoxc.scan` pops a region's environment per iteration, which is the only reason the in-scan table is cheap: moving those calls back to the top level (or unrolling the scan) re-charges every transient to the entry block for the rest of the generation.
Reading a dead slot's centre rows would return stale integers, not NaN — always gate centre reads behind `isnan(species_keys)`.
`DistanceRows`' tiling is a `broadcast_to` + `reshape`, i.e. a real copy: one population-shaped row materialises `P` copies of that centre across every digest leaf (`P=1000` × the padded gene width). `matrix` pays one such copy per scan iteration (freed with that iteration's region), which is part of the ~12 / ~24 MiB residual.

## Test Strategy
Tests live in the sibling `test/` tree (read-only from here): `test/unit/algorithm/` covers the spawn/selection/stagnation algebra, the block layout and the speciation scans on fixed payloads.
`test/parity/algorithm/neat/` compares against `benchmarks/legacy_jax` on fixed gene payloads — identical selection/speciation outcomes for the deterministic steps, and only the divergences listed above allowed elsewhere.
`test/integration/` evolves XOR to its fitness target with a fixed seed and asserts run-to-run determinism for a `(seed, config, shape)` triple.
Every kernel here must be a pure function of `(state, key)`: no dict-order dependence, no host RNG, no float→int conversion without an explicit mask.
Distance rows are verified by whole-run bit-identity: an `npz` dump of the fitness vector, the four payload arrays of the produced and the next population, all ten `SpeciesState` leaves, `randkey` and `generation` per generation, compared with `tobytes()` equality (padding NaNs make `np.array_equal` report self-vs-self differences) against the pre-change tree — 63 arrays per run at `P=500` and `P=1000` (3 generations each, i.e. the dense path), zero mismatches, taking `common.tools.to_numpy` first because hashing an `evoxc.Tensor` directly hashes the object pointer rather than the data.
The sparse/dense switch and the `cond` placement are static trace-time choices, so a shape or gate regression shows up as a build error in `test/release_gate_no_jax.py`'s two-generation XOR run rather than as a silent numerical drift.

## Export Audit (how the StableHLO rules are verified)
Recipe, cheap and per-op: monkeypatch `evoxc.build` to also keep `evoxc.trace(fn, *specs)` for every graph TensorNEAT builds, then walk `graph.module.functions[*].region.blocks` recursively (`op.regions` included, skipping the result-less `return` op) and classify each op with `evoxc.backends.stablehlo.ops.status(op.name)` — it returns `"v1"` (direct emitter), `"decompose"` (emitted as an expansion) or `"deferred"` (the writer raises); only `"deferred"` is a violation. For `scatter` additionally check the operand rank and `op.attributes["axis"]` against the rank-1/rank-2 rule, and count the `while` regions to see the graph's shape. `op.location` gives the exact `file:line` of an offender, so a regression is attributable without reading diffs.
Finish with `evoxc.lower(graph, backend="iree")` on a box that has the iree packages; it fails fast on the first unsupported op and does not need a working end-to-end backend.
Current result at `P=500` and `P=1000`: 0 deferred ops, `scatter` shapes `{(1, axis 0), (2, axis 0)}`, step graph 2137 ops / 12 `while` regions, `setup` graph 74 ops.
The same oracle runs in both spawn modes, and degenerate shapes are smoke-tested: `species_size` 1, 2, 3, 8.

## Routing Table
| Path | Responsibility |
| --- | --- |
| `./neat.py` | `NEAT`: population cycle (`setup`/`ask`/`transform`/`forward`/`tell`), fitness sanitising, next-generation creation, `pop_batch_size` scan chunking, the per-generation distance digest wiring, host-side `show_details`. |
| `./species.py` | `SpeciesState` (frozen dataclass) and `SpeciesController`: species fitness, stagnation, descending sort, spawn allocation, parent pairing, block layout, fixed-length speciation scans. |
| `./_kernels.py` | Plain graph helpers behind `species.py`: constants, payload helpers, sort/prefix-sum primitives, membership, spawn allocation, block layout, parent pairing, `GATED_ROWS`/`gated` (the static lazy-row gate) and the two speciation scans, payload reading. |
| `./_distance.py` | `DistanceRows` (digest-backed `distance_fn`; population-shaped `(P,)` row and the `(S, P)` table measured inside one fixed-length scan) and `repeat_leading`. |
| `./_validation.py` | Host-side constructor validators shared by `species.py` and `neat.py`. |
| `./__init__.py` | Re-exports `NEAT`, `SpeciesController`, `SpeciesState`. |
