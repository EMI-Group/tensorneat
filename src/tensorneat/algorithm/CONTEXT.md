# Algorithm (`tensorneat.algorithm`)

## Intent
The evolutionary layer: it owns the population, the ask → transform → forward → tell generation cycle, speciation, parent selection, the crossover/mutation dispatch, and the HyperNEAT CPPN → substrate decode.
Everything here runs **inside an evoxc trace** — `pipeline.py` builds one graph per shape signature covering a whole generation — except the explicitly host-side helpers (`__init__` validation, `setup` on the host, `show_details`, substrate construction).
This layer owns no network mathematics: genomes own `transform/forward/grad/distance`, problems own `evaluate`, and the algorithm only shuffles whole populations with static shapes.
Cost profile: one population digest plus `2 · ceil(species_size / 4)` compatibility calls in `speciate` and `O(pop_size)` selection/crossover/mutation per generation; no `max_nodes × max_nodes` tensor ever appears here.
Measured with the real `DefaultGenome`/`XOR` (see `## Measured Cost`): `tell` runs in ~58 / ~393 / ~732 ms per generation at pop 100 / 500 / 1000 (`species_size=20`), of which `speciate` is 46 / 78 / 79 % and breeding 52 / 21 / 20 %.
`speciate` is still the top term above pop 300 and it is per-pair bound: the pair count `2 · species_size · pop_size` is irreducible, and the cost of one pair is the genome's per-pair homologue sort.

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
6. Parent selection per species: members sorted by fitness descending, survivors = `clamp(floor(survival_threshold * members), 1, min(members, pop_size))`, two parents drawn uniformly from the survivor prefix, elites (`genome_elitism`) pinned to the species' best member, winner on ties (`fitness[first] >= fitness[second]`).
7. Block flattening: population slot `p` maps to species `loc(p)` = the **last** slot with `spawn[i] > 0 and exclusive_prefix[i] <= p`, with `idx_in_block(p) = p - exclusive_prefix[loc(p)]`, using a triangular-matmul prefix sum plus gather — never a host loop over species.
8. Winner/loser decided by fitness comparison; `genome.execute_crossover` then runs batch-first over the `(P, …)` pair, winner payload first.
9. Mutation gets `new_node_keys = max_node_key(state, node_keys) + 1 + arange(P)` and `new_conn_markers (P, 3)` (zeros when the conn gene carries no `historical_marker`), then runs batch-first, scan-chunked by `pop_batch_size`; the genome maxima are per-individual `(P,) int32`, so an allocated key is unique only *within* an individual — see `./neat/CONTEXT.md` `## Gene Keys & Padding`.
10. Elite slots keep the unmutated crossover children.
11. Speciation: one `genome.prepare_distance(state, pop)` digest for the new population, then `speciate` rebuilds species membership through `_distance.DistanceRows`, which measures the whole species centre table of a phase in one `(S, P)` matrix (blocked by `CENTRES_PER_CALL = 4`).

## Speciation Without vmap-over-Control-Flow (`./neat/species.py`)
`speciate(state, distance_fn, pop=None)` is two `evoxc.scan`s over `evoxc.numpy.arange(species_size)`, each carrying `idx2species`, the four centre payload arrays, `species_keys`, `o2c` and `next_species_key`:
- phase 1 (center reservation): for each live slot `i`, take `d_i = distance_fn(centre_i, pop)` `(P,)`, then `closest = argmin(select(unassigned, d_i, +inf))`, assign `idx2species[closest] = species_keys[i]`, write `pop[closest]` as the new center and set `o2c[closest] = 0`.
- phase 2 (member assignment): a live slot assigns every genome with `d < compatibility_threshold` and `unassigned or d < o2c`; a dead slot creates a species keyed `next_species_key` seeded by the first unassigned genome, then assigns the same way.
- both phases are guarded: an iteration with no unassigned genome, or with a dead slot in phase 1, performs no writes (mask first, `scatter` once), so a fixed-length scan is equivalent to the legacy early-terminating `while_loop`.
- post-pass: still-unassigned genomes join the last species key, newly created slots get `best_fitness = -inf` / `last_improved = generation`, and `member_count` is a masked compare-sum `(species_size, pop_size)` reduce.
Rationale for scans instead of the legacy data-dependent `while_loop`: evoxc de-sugars `scan` to `while` + gather/scatter, so trip counts must stay static to codegen on compiler backends, and iree aborts on wide `while` carries (`select` plus ≥3–4 carried values).
The rows themselves come from `_distance.DistanceRows`, a per-generation digest-backed `distance_fn`: because every `scatter` writes only its own slot, phase 1's centres and phase 2's live-slot centres are the rows the table was handed, so a dense table measures them all in one matrix per phase and each iteration reads `gather(rows, slot, axis=0)`; a slot that *creates* a species has a carry-dependent centre (`pop[seed]`) and keeps one `evoxc.cond` per iteration.
Below `GATED_ROWS = 6000` (`species_size * pop_size`) the sparse table keeps the previous lazy per-slot rows behind `evoxc.cond` (predicate = the mask that permits the iteration to write), which is measurably cheaper there (30.1 vs 55.6 ms of `speciate` at pop 100); above the gate the wide matrix wins because every slot is live (see `./neat/CONTEXT.md` `## Distance Rows`).

## Randomness Budget
Per generation the algorithm performs exactly two `evoxc.random.split`s, yielding three subkeys consumed in this order: parent selection, crossover, mutation — the advanced parent key is stored back into `state.randkey`.
Speciation (the digest, the matrix, the scans) draws no randomness, so batching the distance rows cannot move an RNG stream.
When `pop_batch_size` chunks a population kernel into a `scan`, the algorithm pre-splits `n_batches` keys with `evoxc.random.split_n` and passes the stacked `(n_batches,)` int64 array as the scan's `xs`, so each step receives a valid rank-0 key.
Key-form is whatever `splitmix64` produces (rank-0 int64); nothing here assumes a JAX `PRNGKey`.
There is no `random.choice` and no probability vector: uniform parent rank slots come from `evoxc.random.randint` with a broadcast symbolic `high`, which is also the fix for the legacy zero-survivor division.
No random draw happens inside a control-flow body, so keys never need to travel through a loop carry.

## Divergences from the Legacy JAX Baseline (deliberate fixes)
1. `survive_size == 0` (a 1-member species) produced NaN probabilities and killed `random.choice`; now `high = max(survive_size, 1)` yields two identical parents (self-crossover is the identity).
2. `species_number_calculate_by="fitness"` poisoned `min`/`sum` with the NaN `best_fitness` of dead slots; every reduction now masks dead slots first.
3. `HyperNEATNode` disagreed with the node-gene protocol (missing `valid_mask`, `aggregation` called with one argument, no declared aggregation/activation columns); it now declares the two `int` option columns and an identity `combine`, so it runs through `BaseNode.forward` unchanged. See `./hyperneat`.
4. `spawn_number` relied on NaN → int32 cast semantics; dead slots are explicitly zeroed before the cast.
5. `weight_threshold` was unvalidated while `1 - weight_threshold` divides; `__init__` now raises `TensorNEATError` for `weight_threshold ∉ [0, 1)`, `max_weight <= 0`, `species_size * min_species_size > pop_size`, a bad `species_number_calculate_by`, or a non-dividing `pop_batch_size`.
6. Sort tie-order stays bit-compatible with legacy by using a *stable ascending* `argsort` followed by an index reversal gather (evoxc `argsort(descending=True)` breaks ties the other way).
7. The spawn residual lands on the first **live** slot instead of literally slot `0`: identical after the descending species sort (live slots precede dead ones), but a hand-built unsorted table can no longer hand population slots to a dead species.
8. Random streams differ from the baseline by construction: one key per population kernel (drawing `(P, …)` randomness inside) instead of one key per genome, so parity compares kernels on fixed inputs, never RNG bit-patterns.

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
`DefaultGenome(num_inputs=2, num_outputs=1, max_nodes=50, max_conns=100, output_transform=ACT.sigmoid)`, `NEAT(pop_size=P, species_size=20, survival_threshold=0.01)`, `XOR()`, seed 42, `backend="numpy"`; one configuration per fresh process, 5 setup steps, one warmup plus five timed runs; phases timed as separate executables traced on one snapshot state (`step` is the pipeline's own graph), interleaved A/B against the pre-change tree, medians (minimums in parentheses), ms.
Legacy JAX baseline on the same box (earlier session): 51 / 109 / 157 ms per generation at pop 100 / 500 / 1000.

| pop | evaluate | update_species | breed | speciate | tell | `Pipeline.step` | peak RSS |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 100 | 10.2 (10.0) | 1.20 (1.18) | 31.2 (30.4) → 30.3 (30.0) | 39.4 (38.6) → **26.9 (26.7)** | 71.8 (70.2) → 58.4 (57.8) | 90.4 (84.8) → **73.7 (71.6)** | 242 → 216 MiB |
| 500 | 24.8 (24.5) | 1.57 (1.36) | 77.0 (71.9) → 83.9 (76.3) | 577.0 (569.3) → **307.1 (311.4)** | 655.5 (642.6) → 392.5 (389.1) | 956.1 (938.9) → **443.7 (440.8)** | 638 → 1562 MiB |
| 1000 | 45.4 (44.6) | 2.07 (2.04) | 143.4 (140.1) → 148.5 (141.9) | 1095.5 (1088.4) → **581.3 (577.1)** | 1240.8 (1230.4) → 731.8 (721.1) | 1813.2 (1800.7) → **833.8 (789.7)** | 1118 → 2958 MiB |

`speciate` −32 / −47 / −47 %, whole step −18 / −54 / −54 %; breed is unchanged within this box's ±7 % drift.
The first step of a run costs 4–5 s because it includes the one-off `evoxc.build`; `NEAT.setup` is ~40 ms.
Live species slots in the measured states: 3 / 20 / 20 at pop 100 / 500 / 1000, which is why only the smallest configuration stays on the lazy path.
Graph build (`build_tell` as a standalone executable): 2.7 / 4.3 / 4.8 s; the pipeline's one-off whole-generation build 3.2 / 3.9 / 4.5 s.
`genome.measure_distance` costs 0.8 / 6.5 / 11.3 ms per call (7.9 / 13.0 / 11.3 µs per pair) against `execute_distance`'s earlier ~25–48 µs per pair; `prepare_distance` adds 0.8 / 2.4 / 3.5 ms once per generation.
Dense dispatch budget per generation: 1 `prepare_distance` + `2 · ceil(20 / 4) = 10` `measure_distance` calls (11, plus one `cond` branch per creating species slot) instead of `2 · 20 = 40` `execute_distance` calls.
Breeding split at pop 100: mutation kernel 24.8 ms, crossover 5.1 ms, keygen 0.08 ms, elite select 0.13 ms, parent gathers 0.09 ms; its graph is 3572 IR ops of which `genome.execute_mutation` is 3116 (82 %), `genome.execute_crossover` 433 (17 %) and everything this subtree owns ~23 (0.6 %).
Against the declared legacy baseline the cycle is ~1.4x / ~4.1x / ~5.3x over budget at pop 100 / 500 / 1000 (was ~1.8x / ~8.8x / ~11.6x).

## Known Issues
The repo-level performance contract is still missed, now by ~1.4x / ~4.1x / ~5.3x at pop 100 / 500 / 1000 (see `## Measured Cost`).
From pop ≈ 300 upward `speciate` is ~78 % of `tell` and it is per-pair bound: the pair count `2 · species_size · pop_size` is irreducible (each phase needs its own genuinely different centres, and a created species takes the first unassigned genome as its centre, so its row depends on every earlier assignment), and `DefaultDistance`'s `find_homologous` still sorts `2·M` key rows per pair, so the same population genome is re-sorted once per centre — a genome-side centre-side prepare/sorted-row reuse is the next real lever and lies outside this subtree.
Peak RSS grows 2.4x–2.6x above pop 300 (638 → 1562 MiB at pop 500, 1118 → 2958 MiB at pop 1000) and it is **not** the centre batching: forcing `CENTRES_PER_CALL` to 1 / 2 / 4 / 20 gives 2681 / 2881 / 2793 / 2906 MiB peak, i.e. the pre-change call shape through the digest already costs the full increase, so the retained temporaries belong to `genome`'s `prepare_distance`/`measure_distance` (`(rows, 2·max_conns, key_width)` masked gathers plus the digest). Genome-side item; the sparse path is unaffected (242 → 216 MiB at pop 100).
`genome/base.py::_iinf` materialises a full-size `I_INF` constant, so tracing warns `Embedding a tensor of … bytes as a graph constant` from pop 500 (1.6 MB) upward (genome-owned; a rank-0 sentinel plus broadcasting would avoid it).
`test/unit/algorithm/**` is a spec written ahead of the implementation — its bodies are hard-coded `pytest.skip` (71 skipped), so it verifies nothing even against a working tree, and there is no `test/unit/algorithm/hyperneat/` directory at all; `test/parity/` needs the `bench` extra (JAX).
`test/unit/algorithm/test_spawn.py` encodes `loc(p) = #{i : spawn[i] > 0 and E[i] <= p} - 1`; that count disagrees with `block_layout` for a live zero-spawn species in the middle of a table, so the oracle needs the "last qualifying block" form (cross-node item, `test/` is not writable from here).
`next_species_key` must stay strictly greater than every live key (`setup` seeds it at `1.0`, slot 0 uses `0.0`); a colliding counter silently merges two species' `member_count`.
`NEAT.setup` runs one `genome.initialize` graph on the host — it must be called outside any trace, and it advances `randkey` once, so the pipeline's per-generation key budget is unaffected.

## Known Risks (evoxc)
`evoxc.scan` de-sugars to `while` + gather/scatter, so a wide carry trips iree's affinity analysis on CUDA/ROCm; the numpy backend is the reference and packed carries are the mitigation.
Data-dependent `while_loop` trip counts work on numpy and iree/llvm-cpu but not on iree CUDA/ROCm or tvm — the reason every loop in this subtree is fixed-length.
`cumsum` does not export on compiler backends, hence prefix sums are triangular matmuls.
`evoxc.constant` warns above 1 MiB, which bounds how large a substrate coordinate table may be before it must become an explicit pipeline input.
`evoxc` has no `io_callback`/print op, so all logging stays in `show_details` on the host side.
`DistanceRows`' tiling is a real copy (`broadcast_to` + `reshape`), so the wide path's footprint grows with `centres · pop_size` per digest leaf; `CENTRES_PER_CALL` is the knob that bounds it, and the memo keeps one tiled digest per width alive for the rest of the trace.

## Constraints
No `jax` / `torch` / `numpy.random` import anywhere in this subtree; `import tensorneat` must stay free of them (release gate).
No Python branching, `bool()`, `len()` or host reads on traced values; static Python `if`/`for` over constructor configuration is fine and unrolls at trace time.
Constructor validation raises `tensorneat.common.errors.TensorNEATError`; trace/shape/backend errors propagate unchanged and are never re-wrapped or silently fallen back.
Nothing here reads `state` fields it did not register, except the documented genome/problem keys above.
Files stay under ~1000 lines: the largest are `neat/species.py` at 966, `neat/neat.py` at 905, `neat/_kernels.py` at 898 and `hyperneat/hyperneat.py` at 823; the digest-backed distance rows and their tiling live in `neat/_distance.py` (315), the gene kernels in `hyperneat/genes.py` (297), shared algebra helpers in `neat/_kernels.py`, host validators in `neat/_validation.py`.

## Test Strategy
Tests live in the sibling `./../../test` tree (read-only from here), mirroring this subtree: `test/unit/algorithm/` for the spawn/selection/stagnation algebra on fixed keys, `test/parity/algorithm/` for comparison against `benchmarks/legacy_jax` (documented divergences above are the only allowed mismatches), `test/integration/` for a full XOR run reaching its fitness target with a reproducible seed.
Every kernel here must be deterministic given `(seed, state)`: no dict ordering, no host-side RNG, no float-to-int conversion without an explicit mask.
End-to-end verification today is the real pipeline: `test/release_gate_no_jax.py` evolves `NEAT + DefaultGenome + XOR` for two generations, and `HyperNEAT + DefaultSubstrate/FullSubstrate/MLPSubstrate + XOR3d` runs the same way, so a shape or rank regression surfaces there even though the unit bodies skip.
The speciate rewrite is verified by whole-run bit-identity: per-generation `fitness_history`, the four payload arrays, all ten `SpeciesState` leaves, `randkey` and `generation` dumped to `npz` and compared with `tobytes()` equality (padding NaNs defeat `np.array_equal`) against the pre-change tree at pop 100 (6 generations, lazy path), pop 500 (4) and pop 1000 (3, wide path), where the same run reaches `fitness_target=-0.1` at generation 55 (step 56, max fitness −0.062289) on both trees.
Absent runnable unit tests, this subtree is also verified by building each kernel through `evoxc.build` on the numpy backend against hand-built tables: `speciate` full assignment, `sum(spawn) == pop_size` in both modes, survivor-prefix/elite pinning, stagnation kills, 5-generation cycles at pop 100/500/1000, and bit-identical repeat runs.

## Routing Table
| Path | Responsibility |
| --- | --- |
| `./base.py` | `BaseAlgorithm` batch-first protocol: `setup(state=None)`/`ask`/`transform`/`forward`/`tell`/`show_details` plus `num_inputs`/`num_outputs`/`pop_size`. |
| `./neat` | `NEAT` population cycle (ask/tell, elite handling, `pop_batch_size` scan batching), `SpeciesController` + `SpeciesState`, the `_kernels.py` graph helpers, the `_distance.py` digest-backed distance rows (`DistanceRows`, `CENTRES_PER_CALL`) and the `_validation.py` host validators. |
| `./hyperneat` | `HyperNEAT` / `HyperNEATFeedForward` batch CPPN→substrate decode (`SubstrateSchedule` payload wrapper), the `genes.py` `HyperNEATNode`/`HyperNEATConn` kernels, and the `./hyperneat/substrate` package of host-only coordinate geometry (`DefaultSubstrate`/`FullSubstrate`/`MLPSubstrate`). |
