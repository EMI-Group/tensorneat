# Algorithm (`tensorneat.algorithm`)

## Intent
The evolutionary layer: it owns the population, the ask → transform → forward → tell generation cycle, speciation, parent selection, the crossover/mutation dispatch, and the HyperNEAT CPPN → substrate decode.
Everything here runs **inside an evoxc trace** — `pipeline.py` builds one graph per shape signature covering a whole generation — except the explicitly host-side helpers (`__init__` validation, `setup` on the host, `show_details`, substrate construction).
This layer owns no network mathematics: genomes own `transform/forward/grad/distance`, problems own `evaluate`, and the algorithm only shuffles whole populations with static shapes.
Cost profile: one population digest plus `2 · species_size` population-shaped compatibility rows in `speciate` (scan 1's whole table measured up front on the dense path inside one fixed-length `evoxc.scan`, scan 2's measured in-loop) and `O(pop_size)` selection/crossover/mutation per generation; no `max_nodes × max_nodes` tensor ever appears here.
Measured with the real `DefaultGenome`/`XOR` (see `./neat/CONTEXT.md` `## Measured Cost`): `Pipeline.step` runs in ~44 / ~195 / ~355 ms per generation at pop 100 / 500 / 1000 (`species_size=20`), with peak RSS at or below the pre-digest tree in every cell.
The bounded structural-key allocator costs ~+4 / ~+8 / ~+33 ms of that (`+10 %` / `+5 %` / `+12 %`), almost entirely the `(pop_size, key_span + 1)` free-key table it needs for its collision fallback.
`speciate` is still the top term above pop 300 and it is per-pair bound: the pair count `2 · species_size · pop_size` is irreducible, and the cost of one pair is the genome's per-pair homologue sort.
`__init__.py` is the package surface: it re-exports `BaseAlgorithm`, `NEAT`, `HyperNEAT` and `HyperNEATFeedForward` (the last one from `./hyperneat`), and `__all__` lists exactly those four names.

## Kernel Contract (binding for this subtree)
Batch-first: population is axis 0, always; a kernel is written once over the whole batch.
`vmap` is never used — evoxc refuses to vectorize region-bearing control flow, so per-individual work would have to be rewritten anyway; the batch axis is spelled out explicitly instead.
Control flow (`evoxc.scan` / `evoxc.while_loop` / `evoxc.cond`) appears only at the outermost level of a kernel; every operation inside a loop body is a whole-batch tensor op.
Species loops are fixed-length `evoxc.scan`s over the static `species_size` axis with per-slot masks; no loop trip count is data-dependent.
`pop_batch_size` memory batching is an `evoxc.scan` over the batch axis of the mutation kernel only.
evoxc has no `.at[]`, no in-place mutation and no `cumsum`/`max(where=…)`/`sum(where=…)`/`any`/`all`/`take`/`searchsorted`/`repeat`: mask with `evoxc.select` before reducing, prefix-sum with `dot` against a static triangular ones matrix, and move data with flattened `evoxc.gather` / `evoxc.scatter`.
`evoxc.scatter` is replacement-only — an accumulating update is recomputed as "reduce first, scatter once".

## BaseAlgorithm Protocol (`base.py`)
`setup(self, state=None) -> State` registers the algorithm's tensors; `None` builds `State()` in the body because a `State()` default argument would be evaluated at import time and break `import tensorneat`.
`setup` runs after the genome's and the problem's `setup` on the same threaded `State`, and it is host-side: `NEAT.setup` builds and runs one `genome.initialize` graph, so it must never be called from inside an active trace.
`ask(state) -> pop` hands out the population to evaluate; for NEAT `pop` is the `genome.utils.GenomeTensors` payload (`node_keys (P, max_nodes)`, `node_attrs (P, max_nodes, NA)`, `conn_keys (P, max_conns, CK)`, `conn_attrs (P, max_conns, CA)`) and it costs zero ops.
`transform(state, pop) -> Transformed` decodes the **whole** population into one batch-first execution schedule (axis 0 = individual); the Pipeline never maps it per genome.
`forward(state, transformed, inputs) -> outputs` runs the decoded batch with `inputs (P, *input_shape)` and returns `(P, *output_shape)`.
`tell(state, fitness) -> State` consumes a `(P,)` fitness array (fitness is maximised) and returns the next-generation state.
`show_details(state, fitness) -> None` is the only other host-side method: it converts with `common.tools.to_numpy` and prints (evoxc has no `io_callback`, and `runtime_call` exists on the numpy backend only).
`num_inputs`, `num_outputs`, `pop_size` are static Python ints; `Pipeline` asserts the first two against `problem.input_shape[-1]` / `output_shape[-1]`.
`NEAT.transform`/`NEAT.forward` take an optional trailing `pop=` so a caller can forward a payload other than the one in `state`; `HyperNEAT` uses that for its `(P·Q)` CPPN query batch.

## State Layout
The algorithm registers exactly four keys on the shared `State`: `pop_size` (static Python int, so the batch-first `genome.initialize` can size axis 0), `pop` (one `genome.utils.GenomeTensors` payload, axis 0 = individual), `generation` (0-d float32 — kept float so it flows through `last_improved`), and `species`.
`randkey` is never registered here: `setup` only advances it, so `Pipeline.setup` must create it first with `evoxc.random.key(seed)` (`TensorNEATError` otherwise); a `(seed, config, shape)` triple then reproduces a run bit-for-bit.
`species` is a frozen dataclass `SpeciesState` (owned by `./neat`) with ten fields: `(species_size,)` float32 `species_keys`, `best_fitness`, `last_improved`, `member_count` (NaN = dead slot), `(pop_size,)` float32 `idx2species`, the four split payload arrays `center_node_keys (S, max_nodes) int`, `center_node_attrs (S, max_nodes, NA) float`, `center_conn_keys (S, max_conns, CK) int`, `center_conn_attrs (S, max_conns, CA) float`, and 0-d float32 `next_species_key`.
Dead species slots are addressed solely by the `isnan(species_keys)` mask; the center gene rows of a dead slot hold stale integers that are never read, because the integer gene payload cannot carry NaN.
`generation` is incremented at the *start* of `tell` (legacy order), so stagnation is evaluated against the generation that produced the fitness being consumed.

## NEAT Generation Algebra (`tell`)
1. Sanitize fitness: NaN and ±inf → `-inf`, so a broken topology can never win selection (`finite = abs(f) <= finfo(dtype).max`, no invalid arithmetic, hence no `RuntimeWarning`).
2. Species fitness = per-species maximum of member fitness, computed as a masked reduce over the `(species_size, pop_size)` membership comparison `idx2species[None, :] == species_keys[:, None]`.
3. Stagnation: `stagnant = (f_species <= best_fitness) & (generation - last_improved > max_stagnation)`, disabled for the top `species_elitism` ranked species; a killed species is written to NaN / `-inf`.
4. Species sort by species fitness descending, applied to every `(species_size, …)` field with one gather over a permutation.
5. Spawn allocation: linear-rank (`species_number_calculate_by="rank"`) or fitness-proportional (`"fitness"`), damped by `spawn_number_change_rate`, floored at `min_species_size`, integerised, then error-corrected so `sum(spawn) == pop_size` exactly — the residual is added to the **first live slot** by one `scatter`.
6. Parent selection per species: members sorted by fitness descending, survivors = `clamp(floor(survival_threshold * members), 1, min(members, pop_size))`, two parents drawn uniformly from the survivor prefix, elite slot `k` (`idx_in_block == k < genome_elitism`) pinned to its species' `k`-th ranked member clamped at `live_members - 1` — the legacy elementwise pin, so the top `genome_elitism` **distinct** members survive unmutated — winner on ties (`fitness[first] >= fitness[second]`).
7. Block flattening: population slot `p` maps to species `loc(p)` = the **last** slot with `spawn[i] > 0 and exclusive_prefix[i] <= p`, with `idx_in_block(p) = p - exclusive_prefix[loc(p)]`, using a triangular-matmul prefix sum plus gather — never a host loop over species.
8. Winner/loser decided by fitness comparison; `genome.execute_crossover` then runs batch-first over the `(P, …)` pair, winner payload first.
9. Mutation gets `new_node_keys (P,)` from `_keys.node_keys_for_split`: the wrapped per-individual spread `max_nodes + (max_node_key + (1 + slot)) mod (key_span - max_nodes)`, replaced by the lowest free key in `[0, key_span)` on the rows where that candidate is already held — so an allocated key is always inside the schedule's dense key table and never a key the individual owns.
   `new_conn_markers (P, 3)` keeps the pre-change monotone rule (zeros when the conn gene carries no `historical_marker`) and is bounded by nothing, because a marker is an exactly-matched identity and never a table index.
   The mutation then runs batch-first, scan-chunked by `pop_batch_size`, with the allocation taken from the **crossover child** (`crossed.node_keys`) rather than the pre-crossover parents; see `./neat/CONTEXT.md` `## Gene Keys & Padding`.
10. Elite slots keep the unmutated crossover children.
11. Speciation: one `genome.prepare_distance(state, pop)` digest for the new population, then `speciate` rebuilds species membership through `_distance.DistanceRows`; on the dense path scan 1's whole centre table is measured before the scan (`DistanceRows.matrix`, one population-shaped `measure_distance` per centre run inside the scan's own fixed-length loop) and each iteration reads its row with `gather`, while scan 2 measures its selected centre per iteration.

## Speciation Without vmap-over-Control-Flow (`./neat/species.py`)
`speciate(state, distance_fn, pop=None)` is two `evoxc.scan`s over `evoxc.numpy.arange(species_size)`; scan 1 carries `idx2species`, the four centre payload arrays, `species_keys`, `o2c` and `next_species_key`, while scan 2 carries `idx2species`, `o2c`, `species_keys` plus two rank-1 per-slot tables `seed_table (S,) int32` / `created_mask (S,) float32` and not the centre payload (its centre writes are the identity except on a creating slot, so the created rows are rebuilt once after the scan):
- phase 1 (center reservation): for each live slot `i`, take `d_i` `(P,)` (dense: `gather(rows1, i, axis=0)` from the table measured before the scan; gated: one population-shaped call behind a per-iteration `cond`), then `closest = argmin(select(unassigned, d_i, +inf))`, assign `idx2species[closest] = species_keys[i]`, write `pop[closest]` as the new center and set `o2c[closest] = 0`.
- phase 2 (member assignment): a live slot assigns every genome with `d < compatibility_threshold` and `unassigned or d < o2c`; a dead slot creates a species keyed `next_species_key` seeded by the first unassigned genome, then assigns the same way.
- both phases are guarded: an iteration with no unassigned genome, or with a dead slot in phase 1, performs no writes (mask first, `scatter` once), so a fixed-length scan is equivalent to the legacy early-terminating `while_loop`.
- post-pass: still-unassigned genomes join the last species key, newly created slots get `best_fitness = -inf` / `last_improved = generation`, and `member_count` is a masked compare-sum `(species_size, pop_size)` reduce.
Rationale for scans instead of the legacy data-dependent `while_loop`: evoxc de-sugars `scan` to `while` + gather/scatter, so trip counts must stay static to codegen on compiler backends, and iree aborts on wide `while` carries (`select` plus ≥3–4 carried values).
The rows themselves come from `_distance.DistanceRows`, a per-generation digest-backed `distance_fn`: because every `scatter` writes only its own slot, phase 1's centre for slot `i` is the pre-scan carry row `i`, so the dense path measures the whole phase-1 table with `matrix` before the scan and each iteration reads `gather(rows1, slot, axis=0)`; phase 2's live-slot centres are the post-scan-1 rows (also known up front), but a slot that *creates* a species has a carry-dependent centre (`pop[seed]`), so phase 2 keeps its in-loop measurement (`selected_centre`) on both paths.
Below `GATED_ROWS = 6000` (`species_size * pop_size`) both scans keep the lazy per-slot rows behind `evoxc.cond` (predicate = the mask that permits the iteration to write), which is measurably cheaper there because most slots are dead; above the gate scan 1's pre-measured table wins because every slot is live and the table's own `evoxc.scan` keeps the compatibility kernel out of the reservation `while` body at the cost of one centre's transients (see `./neat/CONTEXT.md` `## Distance Rows`).

## Randomness Budget
Per generation the algorithm performs exactly two `evoxc.random.split`s, yielding three subkeys consumed in this order: parent selection, crossover, mutation — the advanced parent key is stored back into `state.randkey`.
Speciation (the digest, the matrix, the scans) draws no randomness, so batching the distance rows cannot move an RNG stream.
When `pop_batch_size` chunks a population kernel into a `scan`, the algorithm pre-splits `n_batches` keys with `evoxc.random.split_n` and passes the stacked `(n_batches,)` int64 array as the scan's `xs`, so each step receives a valid rank-0 key.
Key-form is whatever `splitmix64` produces (rank-0 int64); nothing here assumes a JAX `PRNGKey`.
There is no `random.choice` and no probability vector: uniform parent rank slots come from `evoxc.random.randint` with a broadcast symbolic `high`, which is also the fix for the legacy zero-survivor division.
Keys never travel through a loop carry: when `pop_batch_size` chunks a population kernel, the pre-split rank-0 keys travel as the scan's `xs`, and each step draws its own population-shaped randomness from its step key inside the loop body.

## Divergences from the Legacy JAX Baseline (deliberate fixes)
1. `survive_size == 0` (a 1-member species) produced NaN probabilities and killed `random.choice`; now `high = max(survive_size, 1)` yields two identical parents (self-crossover is the identity).
2. `species_number_calculate_by="fitness"` poisoned `min`/`sum` with the NaN `best_fitness` of dead slots; every reduction now masks dead slots first.
3. `HyperNEATNode` disagreed with the node-gene protocol (missing `valid_mask`, `aggregation` called with one argument, no declared aggregation/activation columns); it now declares the two `int` option columns and an identity `combine`, so it runs through `BaseNode.forward` unchanged. See `./hyperneat`.
4. `spawn_number` relied on NaN → int32 cast semantics; dead slots are explicitly zeroed before the cast.
5. `weight_threshold` was unvalidated while `1 - weight_threshold` divides; `__init__` now raises `TensorNEATError` for `weight_threshold ∉ [0, 1)`, `max_weight <= 0`, `species_size * min_species_size > pop_size`, a bad `species_number_calculate_by`, or a non-dividing `pop_batch_size`.
6. Sort tie-order stays bit-compatible with legacy by using a *stable ascending* `argsort` followed by an index reversal gather (evoxc `argsort(descending=True)` breaks ties the other way).
7. The spawn residual lands on the first **live** slot instead of literally slot `0`: identical after the descending species sort (live slots precede dead ones), but a hand-built unsorted table can no longer hand population slots to a dead species.
8. Random streams differ from the baseline by construction: one key per population kernel (drawing `(P, …)` randomness inside) instead of one key per genome, so parity compares kernels on fixed inputs, never RNG bit-patterns.

## Search Quality (XOR / XOR3d, numpy backend, measured)
Configuration as in `benchmarks/cases.py`: `DefaultGenome(num_inputs=3|2, num_outputs=1, max_nodes=50, max_conns=100, output_transform=ACT.sigmoid)`, `NEAT(pop_size=P, species_size=20, survival_threshold=0.01)`, `XOR3d()` / `XOR()`, 150 generations, seed 42 unless stated.
XOR3d pop 500 seed 42 reaches its `-0.05` target at generation 61 and ends at `-8.74e-06`; XOR3d pop 1000 seed 42 reaches it at generation 37 and ends at `-8.73e-06`.
XOR3d pop 100 is a per-seed lottery at this budget (150 generations, measured on this tree): seeds 42/43/44/45/46/47 end at `-0.250000 / -0.188224 / -0.250000 / -0.250000 / -0.000385 / -0.250000`, seed 46 reaching the target at generation 91.
The frozen legacy JAX tree is the same lottery on a seed sweep (`-0.0397 / -0.1350 / -0.0922 / -0.250000` on seeds 42-45, its target missed on 43 and 44), so the residual pop-100 plateau is not a rewrite artefact.
XOR pop 100 seed 42 reaches its `-0.1` target at generation 56 (final `-0.0`) and XOR pop 500 seed 42 at generation 31 (final `-0.0`).
Live species at XOR3d pop 100: 1 at generation 1, 2 at generation 4, 4–5 by generations 12–20 (frozen tree: 1, then 2 at generation 3, 3 at generation 7, 4 at generation 12); at pop 500 the table fills to 20 live slots by generation 6 (1 / 2 / 18 / 18 / 20 / 20) and at pop 1000 by generation 4 (1 / 3 / 14 / 20 / 20 / 20).
Structural keys are bounded by construction: over those 150 generations at pop 100, 500 and 1000 there are **0** node keys and **0** connection endpoints outside `[0, key_span)` (the pre-change allocator left 1 313 / 20 762 / 321 571 node keys and 2 622 / 30 227 / 512 089 endpoints outside it), and the largest live key is `key_span - 1`.
The compatibility distance is not the gate: a fresh random population already spans 0.22–2.05 (median 0.93) against `compatibility_threshold = 2.0`, the same distribution as the frozen tree (median 0.90), and the two distance kernels agree to `3e-7` on identical converted payloads.
The escape from the `-0.25` attractor is a rare structural event — one mutation must build a hidden node with a usable weight — so it is budget-limited at pop 100 and reliable from pop 500 up.
Measured and rejected as fixes: a survivor floor of `min(2, members)` (no effect on any XOR3d pop-100 seed and it departs from the legacy `max(1, floor(thr*members))` count) and the legacy connection-gene mutation defaults (`weight_mutate_rate=0.2`, `weight_replace_rate=0.015`, clip ±5), which keep every pop-100 seed on the plateau and move the pop-500 win from generation 50 to ~230.
The remaining measured difference from the frozen tree is genome-side structural multiplicity: 16.7 % of legacy individuals take two or more structural changes per generation while this tree's first-hit-wins gate allows at most one (`../genome/operations/mutation`).

## Dependencies (cross-node contracts this subtree relies on)
`genome`: batch-first kernels `setup(state)`, `initialize(state, key, pop_size) -> GenomeTensors`, `transform(state, node_keys, conn_keys)`, `forward(state, transformed, node_keys, node_attrs, conn_keys, conn_attrs, inputs (P, …))`, `execute_crossover(state, key, winners, losers)` (winner payload first), `execute_mutation(state, key, payload, new_node_keys (P,), new_conn_markers (P, 3))`.
`genome` distance, two-phase: `prepare_distance(state, pop) -> DistanceDigest` (once per generation; an 8-leaf batch-first pytree that survives `State.register`), `measure_distance(state, centre, digest) -> (P,)` where `centre` may be batch-1 (broadcast internally) or batch-`P` (row-wise) — any *other* centre batch is rejected, which is why the wide form tiles both sides to `K·P` rows.
`genome.execute_distance` is still the documented single-pair contract but is no longer called from here.
`genome.utils.GenomeTensors` is the four-field payload namedtuple in declaration order `(node_keys, node_attrs, conn_keys, conn_attrs)`; `HyperNEAT` relies on that order for its `*payload` unpacking.
`genome` RNG: every kernel takes **one** `evoxc.random` key and draws population-shaped randomness inside (`random.uniform(key, (P, n_attrs))`), because an evoxc key is a rank-0/2/4 scalar and cannot be indexed dynamically — this supersedes the plural `keys` wording in the package CONTEXT.
`genome`: static attributes `num_inputs`, `num_outputs`, `max_nodes`, `max_conns`, `node_gene.length`, `conn_gene.length`, `conn_gene.uses_historical_marker`, plus key/validity helpers `node_exists(node_keys)` and `conn_exists(conn_keys)` (key array only — the padding sentinel is genome-owned) and `max_node_key(state, pop_nodes)` / `max_conn_marker(state, pop_conns)`, which return one maximum **per individual**, `(P,) int32` — the algorithm never inspects the padding encoding directly.
`genome`: `NEAT.genome` is read directly by `HyperNEAT._query_substrate`, which calls `genome.forward` on an expanded `(P·Q)` batch; `genome.forward` must therefore not assert `batch == pop_size`, and the decode takes `P` from the payload's own axis 0 so a one-individual `Pipeline.show` decode works.
`genome/gene`: `./hyperneat` synthesises the fixed substrate payload itself from the gene specs (`node_gene.key_width`/`attr_specs`, `conn_gene` spec) — no `from_attrs`-style builder is required from the genome.
`genome/gene/conn`: `conn_gene.forward` receives a `ConnOperands` bundle (not a bare attribute tensor) from `BaseGenome`; genes read columns through `column_of(operands, name)`. See `./hyperneat`.
Gene payload encoding is genome-owned and opaque here — in particular evoxc has **no bitcast/reinterpret op**, so how a float gene attribute travels inside an integer slot cannot be decided in this subtree.
`common`: `State` (immutable, evoxc-flattenable), `errors.TensorNEATError`, `StatefulBaseClass` (extended by `SpeciesController`), `functions.{ACT, AGG, apply_activation, apply_aggregation}` (`AGG` members take `(z, mask)`), `tools.{rank_elements, fetch_first, argmin_with_mask, flat_gather, to_numpy}` in evoxc form.
`pipeline`: creates `randkey`, calls `setup` on the host, drives `ask → transform → forward/evaluate → tell`, and calls `show_details` on host-readable values.
`problem`: `evaluate(state, keys, act_fn, transformed)` where `act_fn` is `algorithm.forward` partially applied — the problem owns the dataset tiling into the population axis.

## Measured Cost (numpy reference backend, real `DefaultGenome` + `XOR`)
`DefaultGenome(num_inputs=2, num_outputs=1, max_nodes=50, max_conns=100, output_transform=ACT.sigmoid)`, `NEAT(pop_size=P, species_size=20, survival_threshold=0.01)`, `XOR()`, seed 42, `backend="numpy"`; one configuration per fresh process, `benchmarks/backend_measure.py --backend numpy --pop P --generations 20 --warmup 3` (its numbers are `Pipeline.step`), rounds run one at a time and interleaved between trees so box drift cancels; peak RSS is the process's lifetime high-water mark.
`base` is commit a697531 (rows measured inside the reservation `while`), `prev` is dff1643 (the table measured as `species_size` top-level calls), `current` is this tree (the table measured inside one fixed-length `evoxc.scan`).
Legacy JAX baseline on the same box (earlier session): 51 / 109 / 157 ms per generation at pop 100 / 500 / 1000.

| pop | base `step` mean (median) | prev `step` mean (median) | current `step` mean (median) | base / prev / current peak RSS | base / prev / current build |
| --- | --- | --- | --- | --- | --- |
| 100 | 43.4–45.2 (43.8–44.0) | 41.0 (39.0) | 42.5 (41.6) | 118.2 / 114.3 / 114.5 MiB | 0.29 / 0.25 / 0.26 s |
| 500 | 286.3–286.4 (287.8–291.8) | 188.2–203.6 (184.5–206.0) | **165.5–179.5 (164.0–177.0)** | 260.9 / 454.4 / **240.5 MiB** | 0.34–0.36 / 0.55 / **0.32–0.34 s** |
| 1000 | 497.9–508.8 (496.3–567.4) | 344.9–337.7 (341.0–336.2) | **294.4–315.7 (290.0–309.7)** | 431.4 / 810.0 / **391.6 MiB** | 0.43 / 0.62 / **0.42–0.43 s** |

The current tree is 1.6–1.7x faster than `base` at pop 500 and 1.6–1.7x at pop 1000 while using 8–9 % *less* peak RSS and the same build.
Against `prev` the in-scan table is −10 % / −8 % wall and −47 % / −52 % peak RSS at pop 500 / 1000 for −16 % / −20 % build: one kernel traced instead of `species_size` copies (static generation graph 4116 → 2137 IR ops, 10 → 12 `while` regions), and executed dispatch calls per generation only 8 094 → 8 258 (+2 %) against 10 710 on `base`.
Mechanism: `evoxc`'s numpy interpreter keeps every SSA result of a block in that block's environment until the block returns but pops a nested region's environment per iteration, so a kernel's live set is charged once per enclosing block — once for the whole generation at the top level, once per iteration inside a `scan`.
`matrix` therefore runs its `species_size` population-shaped `measure_distance` calls inside a fixed-length scan: exactly one centre's `(P, 2·max_conns, …)` transients are alive at a time, where the top-level loop kept all `species_size` of them (accounted: 454.4 MiB − 228.2 MiB zeros-table floor = 20 × 11.3 MiB per hoisted call at pop 500, versus 240.5 − 228.2 = one call inside the scan).
Rule: measure peak RSS and build next to wall time for every hoist, and put a kernel inside a loop region rather than duplicating it at the top level whenever the loop is static.
The dense table is now ~49 ms of the pop-500 generation and ~90 ms of the pop-1000 one (gap to the same tree with the table body returning zeros, which runs 115 / 225 ms); the rest is evaluation and breeding.
The generation's distance work is `2 · species_size · pop_size` = 4 000 / 20 000 / 40 000 pairs.
Dispatch budget per generation: 1 `prepare_distance` + `2 · species_size` population-shaped `measure_distance` calls (scan 1's `species_size` inside `matrix`'s scan, scan 2's `species_size` in-loop) against `2 · species_size` `execute_distance` calls before the digest existed.
`genome.execute_mutation` is 1 359 of breed's 2 014 IR ops (67 %), `genome.execute_crossover` 432 (21 %), this subtree's `update_species` 189 (9 %) and `neat.py` 29 (1.4 %), so breed is genome-kernel bound.
Against the declared legacy JAX baseline the cycle is ~0.83x / ~1.5–1.65x / ~1.9–2.0x at pop 100 / 500 / 1000.

## Cost of the Bounded Key Allocator (interleaved A/B, same protocol)
`xor pop 100` / `xor3d pop 500` / `xor3d pop 1000`, three rounds each, the pre-change tree (`de943ca`, unbounded `max_node_key + 1 + slot`) against this one, one process per cell and the trees alternating so box drift cancels; a cell shows its per-round medians (and means).

| cell | pre-change `step` ms | bounded allocator `step` ms | delta | peak RSS |
| --- | --- | --- | --- | --- |
| xor p100 | 40.2 / 39.5 (41.8 / 41.7) | 44.1 / 44.5 (44.3 / 44.2) | +10 % | 114.4 → 114.8 MiB |
| xor3d p500 | 184.3 / 187.1 / 186.5 (187.2 / 188.5 / 186.9) | 200.3 / 192.7 / 195.3 (200.5 / 193.8 / 196.9) | +4–8 % | 264.9 → 266.0 MiB |
| xor3d p1000 | 313.3 / 316.8 / 319.0 (314.7 / 317.2 / 321.5) | 368.0 / 350.1 / 359.7 (365.6 / 351.1 / 356.5) | +12–14 % | 440.3 → 442.5 MiB |

The allocation is 35 traced ops (the generation graph 2 173 → 2 193 at pop 100 and 2 185 → 2 205 at pop 500, no new `while`/`if` region), but the interpreter charges element work too, and the cost sits in the fallback's `(pop_size, key_span + 1)` occupancy table: at pop 1000 a table-only policy measures 338 ms against 354 ms for the full allocator and 322 ms for the pre-change monotone rule on the same tree, so the free-key table, not the spread, is the bulk of the delta.
Build time is unchanged (0.27 / 0.33 / 0.42 s) and no key work leaves the graph.

## Known Issues
The repo-level performance contract is still missed, by ~0.83x / ~1.5–1.65x / ~1.9–2.0x at pop 100 / 500 / 1000 (see `## Measured Cost`).
From pop ≈ 300 upward `speciate` is the dominant term of `tell` and it is per-pair bound: the pair count `2 · species_size · pop_size` is irreducible (each phase needs its own genuinely different centres, and a created species takes the first unassigned genome as its centre, so its row depends on every earlier assignment), and `DefaultDistance`'s `find_homologous` still sorts `2·M` key rows per pair, so the same population genome is re-sorted once per centre — a genome-side centre-side prepare/sorted-row reuse is the next real lever and lies outside this subtree.
Peak RSS is not a dense-path penalty any more: 240.5 MiB at pop 500 and 391.6 MiB at pop 1000 are 8–9 % below the pre-digest tree, because the table's kernel runs inside a scan region and only one centre's transients are live.
`genome/base.py::_iinf` materialises a full-size `I_INF` constant, so tracing warns `Embedding a tensor of … bytes as a graph constant` from pop 500 (1.6 MB) upward (genome-owned; a rank-0 sentinel plus broadcasting would avoid it).
`test/unit/algorithm/**` runs real bodies (no skips) and covers the spawn/selection/stagnation algebra and the speciation scans; the only skip in the suite is `test/backends/test_iree_pipeline.py` without the `evoxc[iree]` extra, and `test/parity/` needs the `bench` extra (JAX).
`test/unit/algorithm/test_spawn.py` encodes `loc(p) = #{i : spawn[i] > 0 and E[i] <= p} - 1`; that count disagrees with `block_layout` for a live zero-spawn species in the middle of a table, so the oracle needs the "last qualifying block" form (cross-node item, `test/` is not writable from here).
`next_species_key` must stay strictly greater than every live key (`setup` seeds it at `1.0`, slot 0 uses `0.0`); a colliding counter silently merges two species' `member_count`.
`NEAT.setup` runs one `genome.initialize` graph on the host — it must be called outside any trace, and it advances `randkey` once, so the pipeline's per-generation key budget is unaffected.
The XOR3d pop-100 plateau is seed-dependent in both this tree and the frozen baseline (see `## Search Quality`), and no knob in this subtree lifts it: `survival_threshold` 0.1/0.5 and a `min(2, members)` survivor floor only change which seeds escape, so the legacy count formula stays.
The one measured lever left for pop 100 is genome-side: the mutation chain applies at most one structural change per individual per generation while the frozen tree's four independent operations give 16.7 % of individuals two or more.
The distance rows still tile a batch-1 centre to `pop_size` rows before `measure_distance`, which takes the row-wise merge route; with every gene key now inside `[0, genome.key_span)` the genome's dense single-centre route is reachable, and passing the batch-1 centre there is worth ~480 executed ops and ~50 ms per generation at pop 500 (~120 ms at pop 1000) for bit-identical results.
The structural-key allocator's fallback builds a `(pop_size, key_span + 1)` occupancy table every generation although fewer than 6 % of rows ever need it, which is the whole reason the allocator costs +4 / +8 / +33 ms at pop 100 / 500 / 1000.
Crossover is live only for species with more than `1 / survival_threshold` members (200 at the example `0.01`): below that the survivor prefix is a single genome and every non-elite child is a mutated copy of it.

## Known Risks (evoxc)
`evoxc.scan` de-sugars to `while` + gather/scatter, so a wide carry trips iree's affinity analysis on CUDA/ROCm; the numpy backend is the reference and packed carries are the mitigation.
Data-dependent `while_loop` trip counts work on numpy and iree/llvm-cpu but not on iree CUDA/ROCm or tvm — the reason every loop in this subtree is fixed-length.
`cumsum` does not export on compiler backends, hence prefix sums are triangular matmuls.
`evoxc.constant` warns above 1 MiB, which bounds how large a substrate coordinate table may be before it must become an explicit pipeline input.
`evoxc` has no `io_callback`/print op, so all logging stays in `show_details` on the host side.
`DistanceRows`' tiling is a real copy (`broadcast_to` + `reshape`), so one population-shaped row materialises `pop_size` copies of that centre across every digest leaf — inside `matrix`'s scan this happens once per iteration and is released with the region; `matrix` itself is a fixed-length `evoxc.scan` over the `species_size` centre rows, which is what bounds its live set to one centre (see `## Measured Cost`).

## Constraints
No `jax` / `torch` / `numpy.random` import anywhere in this subtree; `import tensorneat` must stay free of them (release gate).
No Python branching, `bool()`, `len()` or host reads on traced values; static Python `if`/`for` over constructor configuration is fine and unrolls at trace time.
Constructor validation raises `tensorneat.common.errors.TensorNEATError`; trace/shape/backend errors propagate unchanged and are never re-wrapped or silently fallen back.
Nothing here reads `state` fields it did not register, except the documented genome/problem keys above.
Files stay under ~1000 lines: the largest are `neat/_kernels.py` at 1004 (documented exception: shared graph primitives), `neat/neat.py` at 999, `neat/species.py` at 948 and `hyperneat/hyperneat.py` at 823; the bounded structural-key allocator lives in `neat/_keys.py` (325), the digest-backed distance rows and their tiling in `neat/_distance.py` (318), the gene kernels in `hyperneat/genes.py` (297), shared algebra helpers in `neat/_kernels.py`, host validators in `neat/_validation.py`.

## Test Strategy
Tests live in the sibling `./../../test` tree (read-only from here), mirroring this subtree: `test/unit/algorithm/` for the spawn/selection/stagnation algebra on fixed keys, `test/parity/algorithm/` for comparison against `benchmarks/legacy_jax` (documented divergences above are the only allowed mismatches), `test/integration/` for a full XOR run reaching its fitness target with a reproducible seed.
Every kernel here must be deterministic given `(seed, state)`: no dict ordering, no host-side RNG, no float-to-int conversion without an explicit mask.
End-to-end verification today is the real pipeline: `test/release_gate_no_jax.py` evolves `NEAT + DefaultGenome + XOR` for two generations, and `HyperNEAT + DefaultSubstrate/FullSubstrate/MLPSubstrate + XOR3d` runs the same way, so a shape or rank regression surfaces there as well as in `test/unit/algorithm/`.
The speciate rewrite is verified by whole-run bit-identity: per-generation `fitness_history`, the four payload arrays, all ten `SpeciesState` leaves, `randkey` and `generation` dumped to `npz` and compared with `tobytes()` equality (padding NaNs defeat `np.array_equal`) against the pre-change tree at pop 100 (4 generations, lazy path), pop 500 (3) and pop 1000 (2, dense path) — 144 arrays, zero mismatches.
Kernels are additionally verified by building each one through `evoxc.build` on the numpy backend against hand-built tables: `speciate` full assignment, `sum(spawn) == pop_size` in both modes, survivor-prefix/elite pinning, stagnation kills, 5-generation cycles at pop 100/500/1000, and bit-identical repeat runs.

## Routing Table
| Path | Responsibility |
| --- | --- |
| `./__init__.py` | Package surface only: re-exports `BaseAlgorithm`, `NEAT`, `HyperNEAT`, `HyperNEATFeedForward` and lists them in `__all__`. |
| `./base.py` | `BaseAlgorithm` batch-first protocol: `setup(state=None)`/`ask`/`transform`/`forward`/`tell`/`show_details` plus `num_inputs`/`num_outputs`/`pop_size`. |
| `./neat` | `NEAT` population cycle (ask/tell, elite handling, `pop_batch_size` scan batching), `SpeciesController` + `SpeciesState`, the `_kernels.py` graph helpers, the `_distance.py` digest-backed distance rows (`DistanceRows`, dense `matrix` table), the `_keys.py` bounded structural-key allocator and the `_validation.py` host validators. |
| `./hyperneat` | `HyperNEAT` / `HyperNEATFeedForward` batch CPPN→substrate decode (`SubstrateSchedule` payload wrapper), the `genes.py` `HyperNEATNode`/`HyperNEATConn` kernels, and the `./hyperneat/substrate` package of host-only coordinate geometry (`DefaultSubstrate`/`FullSubstrate`/`MLPSubstrate`). |
