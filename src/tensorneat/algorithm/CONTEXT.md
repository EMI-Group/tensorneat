# Algorithm (`tensorneat.algorithm`)

## Intent
The evolutionary layer: it owns the population, the ask → transform → forward → tell cycle, speciation, parent selection, the crossover/mutation dispatch and the HyperNEAT CPPN → substrate decode.
Everything here runs **inside an evoxc trace** — one graph per shape signature covers a whole generation — except the host-side helpers (`__init__`/`_validate` checks, `setup`, `show_details`, substrate construction).
This layer owns no network mathematics: genomes own `transform/forward/grad/distance`, problems own `evaluate`, and the algorithm only shuffles whole populations at static shapes.
`__init__.py` re-exports `BaseAlgorithm`, `NEAT`, `HyperNEAT`, `HyperNEATFeedForward`; `__all__` lists exactly those four names.
`NEAT(genome, pop_size, species_size=10, max_stagnation=15, species_elitism=2, spawn_number_change_rate=0.5, genome_elitism=2, survival_threshold=0.1, min_species_size=1, compatibility_threshold=2.0, species_fitness_func=evoxc.max, species_number_calculate_by="rank", pop_batch_size=None)` validates host-side only (`neat/_validation.py`, `_keys.validate_key_space` for `key_span > max_nodes`, the `pop_batch_size` divides-`pop_size` check, `SpeciesController.__init__` for `survival_threshold ∈ (0, 1]` and `species_number_calculate_by ∈ ("rank", "fitness")`).
`NEAT` accepts **no** mutation-rate parameter: every structural/value probability belongs to the genome's `mutation` and gene objects.
`Pipeline`'s compiled cycle drives `ask → transform →` the `prepare_forward`/`forward_prepared` pair `→ tell`; `ask` runs inside the generation graph, every shipped `ask` returns `state.pop` unchanged (zero ops), and `algorithm.forward` exists on the contract with no in-tree call site.
Cost profile: one population digest plus `2 · species_size` compatibility rows in `speciate`, `O(pop_size)` selection/crossover/mutation, no `max_nodes × max_nodes` tensor anywhere; `speciate` is the top term above pop 300 and per-pair bound.

## Kernel Contract (binding for this subtree)
Batch-first: population is axis 0, always; a kernel is written once over the whole batch, and `vmap` is never used (evoxc refuses to vectorize region-bearing control flow) — the batch axis is spelled out.
Control flow (`evoxc.scan` / `evoxc.while_loop` / `evoxc.cond`) runs over static axes only: a static trip count and a statically built branch set.
An `evoxc.cond` may nest inside an `evoxc.scan` body when the predicate is a traced mask over branches that are both built statically, the nested form has been proven to compile on `iree`-llvm-cpu, and numpy stays the reference.
The shipped instance is `_kernels.speciate`'s gated path: `gated_row` builds one `cond` per scan iteration inside both spellings' step bodies whenever `species_size * pop_size <= GATED_ROWS = 6 000`.
Species loops are fixed-length `evoxc.scan`s over `arange(species_size)` with per-slot masks; `pop_batch_size` batching is an `evoxc.scan` over the batch axis of the mutation kernel only.
evoxc has no `.at[]`, no in-place mutation and no `cumsum`/`max(where=…)`/`sum(where=…)`/`any`/`all`/`take`/`searchsorted`/`repeat`: mask with `evoxc.select` before reducing, prefix-sum with `dot` against a static triangular ones matrix, move data with flattened `evoxc.gather`/`evoxc.scatter` (replacement-only — an accumulating update is "reduce first, scatter once").

## Device & Backend Placement
`NEAT._build_options(state)` is this subtree's only reader of `backend`/`device`/`compile_options`: `{"backend": state.backend}` when registered (else `"numpy"`), `device` only when registered and not `None`, then `compile_options` expanded last, so a `compile_options["device"]` entry overrides the registered device.
`NEAT.setup` is the subtree's only setup-time `evoxc.build`/`evoxc.run`: a plain closure `_init_kernel` (no `@evoxc.defn` here) built with `TensorSpec((), evoxc.int64)` and run on `state.randkey`; `Pipeline.setup` registers `device=None` meanwhile, so it runs on CPU.
`evoxc.run` moves nothing implicitly, so both run boundaries rely on their caller (`randkey` is cpu:0 by construction; the `state.pop` leaves in `_gene_counts` must already live on `executable.device`), and `show_details` is the only host-materialising surface (`to_numpy`), called by the pipeline on a `device=None` host view.

## Formulation Statics (device-derived, graph-visible configuration)
`Pipeline.setup` derives three trace-time statics from the resolved device kind through `host_payload._formulation_statics` and registers them as plain strings: `distance_centre` (`"tiled"`|`"single"`), `distance_formulation` (`"merge"`|`"index_free"`) and `kernel_mode` (`"host"`|`"accelerator"`).
A GPU-class kind (`rocm`, `hip`, `hips`, `cuda`, `vulkan`, `gpu`, case-insensitive) gets `("tiled", "index_free", "accelerator")`; every other kind and `None` gets `("single", "merge", "host")` — the numpy leg, value- and op-identical to a state carrying no statics.
They are never graph-boundary tensors: `Pipeline._static_config` drops the `_RUNTIME_KEYS` names (`pop`, `generation`, `species`, `randkey`) and every remaining `evoxc.Tensor`, so the strings are captured by the generation kernel's closure.
The setup-time initialisation graph gets the same treatment through `NEAT._static_state`, which drops every `evoxc.Tensor` and forces `pop_size`; there is no separate `NEAT._static_config`.
Both passes read the **state**, so updating any of the three entries forces exactly one rebuild and leaves the other signature entries untouched.
They are part of `Pipeline._shape_signature` (`host_payload._formulation_signature`, defaulted to `"tiled"`/`"merge"`/`"host"` for a state no `setup` produced) next to the payload geometry, the substrate statics and the tiled-evaluate chunk budget; `show_config()["pipeline"]` renders all three.
A missing entry is never an error — the kernels default to `"tiled"`/`"merge"`/`"host"` (`_validation.distance_centre_mode`, `common.schedule_accelerator.kernel_mode`, the genome's `distance_formulation` reader), which is what a bare `State()` and every committed test get.
`distance_centre` sets the width of every speciation centre payload here; `kernel_mode` selects the speciation body set (`_kernels.speciate` → `_scan_accelerator`), the distance grouping (`grouped.group_size`) and the schedule's scatter-free formulation; `distance_formulation` is read inside `genome/operations/distance`, not here.
Every accelerator formulation is bit-identical to the host route, generation for generation; no committed test selects the statics, so this subtree's evidence is hand-run verification.

## BaseAlgorithm Protocol (`base.py`)
`setup(self, state=None) -> State` registers the algorithm's tensors and is host-side, running after the genome's and the problem's on the same threaded `State`: `NEAT.setup` builds and runs one `genome.initialize` graph, so it must never be called inside an active trace.
`ask(state) -> pop` hands out the population to evaluate (the `GenomeTensors` payload, zero ops); `transform(state, pop) -> Transformed` decodes the **whole** population into one batch-first schedule; `forward(state, transformed, inputs) -> outputs` runs it at `inputs (P, *input_shape)`.
`prepare_forward(state, transformed, pop=None, samples=1) -> prepared` derives one pass's loop-invariant operands and `forward_prepared(state, transformed, prepared, inputs) -> outputs` runs one query from them; `forward` is exactly the two back to back, bit for bit.
That pair's signature is **uniform** across algorithms (`pop=None`, `samples` = the static tiling factor of the `P * samples` queried rows), so the pipeline needs no arity adapter, unlike `forward`.
`NEAT.prepare_forward` takes the payload (`None` = `state.pop`); `HyperNEAT.prepare_forward` ignores `pop` (its payload is the substrate schedule inside `transformed`); both forward `samples` into `genome.prepare_pass` / `hyper_genome.prepare_pass`, because the substrate schedule sits at `P` rows and a tiled query is the substrate genome's own sample axis.
`host_payload._NetworkActFn` is the pair's only consumer and the problem layer queries through `act_fn.prepare(batch_rows)`: an RL rollout asks once per generation (`problem/rl/_common.py`), the tiled func-fit evaluate once per chunk group (`host_payload._prepared_query`), so the prelude is paid once per batch shape rather than once per query.
`tell(state, fitness) -> State` consumes a `(P,)` maximised fitness array and returns the next-generation state; `show_details(state, fitness) -> None` is the only other host-side method (converts with `common.tools.to_numpy` and prints — evoxc has no `io_callback`).
`num_inputs`, `num_outputs`, `pop_size` are static Python ints, the first two asserted against the problem's declared shapes; `NEAT.transform`/`NEAT.forward` take an optional trailing `pop=`, which `HyperNEAT` uses for its `(P·Q)` CPPN query batch.

## State Layout
The algorithm registers exactly four keys: `pop_size` (static int, sizes the batch-first `genome.initialize`), `pop` (one `GenomeTensors` payload), `generation` (0-d float32 — kept float so it flows through `last_improved`) and `species`.
`randkey` is never registered here: `setup` only advances it, so `Pipeline.setup` must create it first with `evoxc.random.key(seed)` (`TensorNEATError` otherwise); a `(seed, config, shape)` triple then reproduces a run bit-for-bit.
`species` is a frozen dataclass `SpeciesState` (owned by `./neat`): `(species_size,)` float32 `species_keys`, `best_fitness`, `last_improved`, `member_count` (NaN = dead slot), `(pop_size,)` float32 `idx2species`, the four centre payload arrays at `(S, …)` and 0-d float32 `next_species_key`.
Dead slots are addressed solely by the `isnan(species_keys)` mask; a dead slot's centre gene rows hold stale integers that are never read, because the integer payload cannot carry NaN.
`generation` is incremented at the *start* of `tell`, so stagnation is judged against the generation that produced the fitness just consumed.

## NEAT Generation Algebra (`tell`)
1. Sanitize fitness: NaN and ±inf → `-inf` (`abs(f) <= finfo(dtype).max`, so no invalid arithmetic and no `RuntimeWarning`).
2. Species fitness = per-species maximum of member fitness (a masked reduce over `idx2species[None, :] == species_keys[:, None]`), then species sort by fitness descending with one gather per `(species_size, …)` field.
3. Stagnation: `(f_species <= best_fitness) & (generation - last_improved > max_stagnation)`, disabled for the top `species_elitism` ranked species; a killed species is written to NaN / `-inf`.
4. Spawn allocation: rank or fitness-proportional, damped by `spawn_number_change_rate`, floored at `min_species_size`, integerised, then error-corrected so `sum(spawn) == pop_size` exactly — the residual goes to the **first live slot** with one `scatter`.
5. Parent selection per species: survivors = `clamp(floor(survival_threshold * members), 1, min(members, pop_size))`, two parents drawn uniformly from the survivor prefix, elite slot `k` (`idx_in_block == k < genome_elitism`) pinned to its species' `k`-th ranked member clamped at `live_members - 1`, so the top `genome_elitism` **distinct** members survive unmutated.
6. Block flattening: slot `p` maps to species `loc(p)` = the **last** slot with `spawn[i] > 0 and exclusive_prefix[i] <= p`, with `idx_in_block(p) = p - exclusive_prefix[loc(p)]`, via triangular-matmul prefix sum plus gather — never a host loop over species.
7. Winner/loser by fitness comparison; `genome.execute_crossover` runs batch-first, winner payload first.
8. Mutation takes `new_node_keys (P,)` from `_keys.node_keys_for_split`: the wrapped spread `max_nodes + (max_node_key + (1 + slot)) mod (key_span - max_nodes)`, replaced by the lowest free key in `[0, key_span)` where that candidate is already held, so a key is always inside the dense key table and never one the individual owns; `new_conn_markers (P, 3)` keeps the monotone rule (zeros without `historical_marker`) and is unbounded, because a marker is an exactly-matched identity. Allocation comes from the **crossover child** (`crossed.node_keys`) — see `./neat/CONTEXT.md` `## Gene Keys & Padding`.
9. Elite slots keep the unmutated crossover children.
10. Speciation: one `genome.prepare_distance` digest, then `speciate` rebuilds membership through `_distance.DistanceRows`; on the dense path scan 1's whole centre table is measured before the scan (`DistanceRows.matrix`, one centre per call on the host leg, `k = 2` on the accelerator leg) and each iteration gathers its row, while scan 2 measures its centre per iteration.

## Speciation Without vmap-over-Control-Flow (`./neat/_kernels.py`, `./neat/_scan_accelerator.py`, wrapped by `./neat/species.py`)
`SpeciesController.speciate` is the host-facing wrapper (docstring, `pop` resolution, `SpeciesState` rebuild) and delegates to `_kernels.speciate`, which selects the body set from `state.kernel_mode` at trace time.
`speciate(state, distance_fn, pop=None)` is two fixed-length `evoxc.scan`s over `arange(species_size)` with per-slot masks, reproducing the legacy early-terminating `while_loop`; neither carries a centre gene table, because a slot writes only its own row, so the four tables are rebuilt once after each scan from a recorded `(source row, did-write)` pair.
The host spelling keeps per-slot tables (`closest_table`/`claim_table`, `seed_table`/`created_mask`, `keys_ext`); the accelerator spelling carries none — each step returns its record as the scan's stacked `y`, plus a rank-0 `remaining` counter in scan 1's gated path and a rank-0 `next_species_key` counter in scan 2.
Both phases are guarded (mask first, `scatter` once); the post-pass gives still-unassigned genomes the last species key, seeds new slots with `best_fitness = -inf` / `last_improved = generation`, and computes `member_count` as a masked compare-sum.
The rows come from `_distance.DistanceRows`, a per-generation digest-backed `distance_fn`; a *creating* slot's centre is `pop[seed]`, a function of the earlier slots' rows, so scan 2 cannot be grouped and only scan 1's table is.
Below `GATED_ROWS = 6 000` both scans keep the lazy per-slot rows behind `evoxc.cond` (cheaper there, most slots being dead); above it scan 1's pre-measured table wins, keeping the compatibility kernel out of the reservation body.
Phase 1 (centre reservation): per live slot take `d_i (P,)` (dense `gather(rows1, i, axis=0)`, gated one population-shaped call behind a `cond`), pick `closest` = lowest index attaining `min(select(unassigned, d_i, +inf))` (`argmin_with_mask` host, two `reduce_min`s accelerator, `I_INF` when nothing is left), assign `idx2species[closest] = species_keys[i]`, write `pop[closest]` as the new centre and set `o2c[closest] = 0`.
Phase 2 (member assignment): a live slot assigns every genome with `d < compatibility_threshold` and `unassigned or d < o2c`; a dead slot creates a species keyed `next_species_key` seeded by `seed = reduce_min(select(isnan(assign), pop_index, I_INF))` (`fetch_first` host), then assigns the same way.
Both phases are guarded (mask first, `scatter` once), so an idle iteration performs no writes; the post-pass gives still-unassigned genomes the last species key, seeds new slots with `best_fitness = -inf` / `last_improved = generation`, and computes `member_count` as a masked compare-sum.
The rows come from `_distance.DistanceRows`, a per-generation digest-backed `distance_fn`; a *creating* slot's centre is `pop[seed]`, a function of the earlier slots' rows, which is why scan 2 cannot be grouped and only scan 1's table does.
Below `GATED_ROWS = 6 000` both scans keep the lazy per-slot rows behind `evoxc.cond` (cheaper there, most slots being dead); above it scan 1's pre-measured table wins because its own `evoxc.scan` keeps the compatibility kernel out of the reservation body at the cost of one centre's transients.
Compile hazard: a population-shaped `reduce_max(pop)` as a scan's `cond` predicate SIGSEGVs `iree-compile` on llvm-cpu (`GenericVectorization`); the rank-0 counter is the exact substitute, because scan 1 claims at most one row per iteration.
Unrolling the slot axis into straight-line SSA is rejected (a marginal device-time win against a tripled static graph and 2× `iree-compile`), and the gated path is excluded twice over because twenty top-level `cond`s exceed the ROCm/HIP 16-binding cap — see `./neat/CONTEXT.md` `### The slot axis is not unrolled`.

## Randomness Budget
Per generation the algorithm performs exactly two `evoxc.random.split`s yielding three subkeys consumed in order — parent selection, crossover, mutation — and stores the advanced parent key into `state.randkey`.
Speciation (digest, matrix, scans) draws no randomness, so batching distance rows cannot move an RNG stream.
`pop_batch_size` chunking pre-splits `n_batches` keys with `evoxc.random.split_n` and passes the stacked int64 array as the scan's `xs`, so each step gets a rank-0 key and draws its own population-shaped randomness in the body.
Key-form is whatever `splitmix64` produces (rank-0 int64); nothing assumes a JAX `PRNGKey`, and there is no `random.choice` or probability vector — uniform parent rank slots come from `evoxc.random.randint` with a broadcast symbolic `high`, which also fixes the legacy zero-survivor division.

## Deliberate Divergences From the Legacy Oracle
1. `survive_size == 0` (a 1-member species) produced NaN probabilities; `high = max(survive_size, 1)` yields two identical parents, and self-crossover is the identity.
2. `species_number_calculate_by="fitness"` poisoned `min`/`sum` with the NaN `best_fitness` of dead slots; every reduction masks dead slots first.
3. `HyperNEATNode` declares its two `int` option columns and an identity `combine`, so the node-gene protocol (`valid_mask`, two-argument `aggregation`) runs it unchanged. See `./hyperneat`.
4. `spawn_number` relied on NaN → int32 cast semantics; dead slots are explicitly zeroed before the cast.
5. `HyperNEAT.__init__` validates `weight_threshold ∈ [0, 1)`, `max_weight > 0`, `activate_time >= 1`, the substrate's `inputs, outputs, hidden` key block order and the CPPN arity, raising instead of dividing by zero.
6. Sort tie-order stays bit-compatible by using a *stable ascending* `argsort` plus an index reversal gather (evoxc `argsort(descending=True)` breaks ties the other way).
7. The spawn residual lands on the first **live** slot rather than literally slot `0`: identical after the descending sort, but a hand-built unsorted table can no longer hand slots to a dead species.
8. Random streams differ by construction — one key per population kernel drawing `(P, …)` randomness inside instead of one key per genome — so parity compares kernels on fixed inputs, never RNG bit-patterns.

## Search Quality (numpy backend, measured)
The quality contract is `benchmarks/cases.py`: `DefaultGenome(num_inputs=3|2, num_outputs=1, max_nodes=50, max_conns=100, output_transform=ACT.sigmoid)`, `NEAT(pop_size=P, species_size=20, survival_threshold=0.01)`, `XOR3d()`/`XOR()`, 150 generations, no early stop, fitness = −mse.
"Escape" is some generation's best above −0.24 (off the −0.25 constant-0.5 optimum of a sigmoid read-out for these non-linearly-separable tasks); "target" is some generation's best at or above −0.05.
The post-fix record is `benchmarks/results/xor3d_quality_2026-11-04.md`; the seeds 0-19 XOR3d sweep is in `benchmarks/CONTEXT.md`.
XOR3d pop 100 (seeds 0-19) escapes **13/20** and reaches the target **11/20**, median escape generation 75; the frozen oracle on the same seeds escapes 4/20 and reaches the target 2/20 (Fisher exact two-sided p = 0.0095).
XOR3d pop 500 seeds 42-47 reaches the target 6/6 and pop 1000 seeds 42/43/44 3/3 (finals in `../CONTEXT.md`); XOR pop 100 seed 42 reaches its −0.1 target at generation 41 and pop 500 seed 42 at generation 33.
The pop-100 lever is genome-side and shipped: `DefaultGenome`/`RecurrentGenome` build `DefaultMutation(conn_add=0.4, conn_delete=0.05)` when no operator is passed, while an explicit `DefaultMutation()` keeps NEAT-python's 0.2/0.2.
Mechanism: at equal gates an add-conn is accepted about half as often as a delete-conn (duplicate, self-connection, cycle and `max_in_degree` refusals), so an equal-gate steady state of −0.14…−0.22 connections per individual per generation leaves populations under-wired; 0.4/0.05 moves it to +0.15 and buys wiring, not nodes, at an unchanged node count, and the escape hazard tracks the number of *structured* individuals.
Rejected in the same sweep (XOR3d pop 100, seeds 0-19): a same-generation add-conn value-mutation exemption (4/20 escape, 0/20 target), a milder 0.3/0.1 rebalance (7/20), that exemption stacked on it (7/20), `survival_threshold` 0.1/0.5 and a `min(2, members)` survivor floor (`../genome/CONTEXT.md` `## Rejected`).
Structural keys are bounded by construction: over 150-generation runs at pop 100, 500 and 1000 there are **0** node keys and **0** connection endpoints outside `[0, key_span)`.
Neither the compatibility distance (a fresh random population spans 0.22–2.05, median 0.93, against `compatibility_threshold = 2.0`) nor evaluation (hand-built payloads score bit-identically through `Pipeline.step` and a host numpy oracle) is the gate.

## Dependencies (cross-node contracts this subtree relies on)
`genome`: batch-first `setup(state)`, `initialize(state, key, pop_size)`, `transform`, `forward`, `prepare_pass`/`forward_prepared`, `execute_crossover(state, key, winners, losers)` (winner payload first), `execute_mutation(state, key, payload, new_node_keys (P,), new_conn_markers (P, 3))`.
`genome` distance, two-phase: `prepare_distance(state, pop) -> DistanceDigest` (once per generation, a batch-first pytree that survives `State.register`) and `measure_distance(state, centre, digest) -> (P,)`, where `centre` may be batch-1 (the dense single-centre route under `distance_centre="single"`, tiled to `pop_size` rows under `"tiled"`) or batch-`P` (row-wise); any other width is rejected, and a `k · P` batch only against a `group_digest`-tiled digest, which `DistanceRows.matrix` builds through `grouped.for_state` on the accelerator leg.
`genome.utils.GenomeTensors` is the four-field payload namedtuple in declaration order `(node_keys, node_attrs, conn_keys, conn_attrs)`; `HyperNEAT` relies on that order for its `*payload` unpacking.
`genome` RNG: every kernel takes **one** `evoxc.random` key and draws population-shaped randomness inside, because a key is rank-0/2/4 and cannot be indexed dynamically.
`genome`: statics `num_inputs`, `num_outputs`, `max_nodes`, `max_conns`, `key_span`, `node_gene.length`, `conn_gene.length`, `conn_gene.uses_historical_marker`, plus `node_exists`, `conn_exists`, `max_node_key` and `max_conn_marker` (one maximum **per individual**, `(P,) int32`) — the algorithm never inspects the padding encoding.
`genome`: `NEAT.genome` is read directly by `HyperNEAT._query_substrate` on a `(P·Q)` batch, so `genome.forward` must not assert `batch == pop_size` and the decode takes `P` from the payload's axis 0.
The gene payload encoding stays genome-owned: evoxc has **no bitcast**, so how a float attribute travels inside an integer slot is not decidable here; `./hyperneat` synthesises its own substrate payload from the gene specs, and `conn_gene.forward` receives a `ConnOperands` bundle whose columns genes read through `column_of`.
`common`: `State`, `errors`, `StatefulBaseClass`, `functions.{ACT, AGG}` (`AGG` members take `(z, mask)`), `schedule_accelerator.{kernel_mode, is_accelerator}`, `tools.{rank_elements, fetch_first, argmin_with_mask, flat_gather, to_numpy}`.
`problem`: `evaluate(state, key, act_fn, transformed)` where `key` is one rank-0 evoxc key and `act_fn` the prepared-pass pair; the problem owns the dataset tiling into the population axis.

## Measured Cost
The committed numpy record is `benchmarks/PERF_NUMPY.md`; this section keeps the numbers that decide kernel shape and the laws behind them.
This tree, this session, numpy backend, 150 generations with 3 warmup, medians: **33.04** ms per generation on `xor_p100`, **112.68** on `xor_p500`, **177.43** on `xor_p1000`, **847.13** on `custom_pagie_p500`, **248.70** on `cartpole_p100`.
The same five cells on `backend="iree"`, `device="rocm"` cost **17.42 / 13.24 / 16.29 / 16.18 / 150.41** ms per generation in that order, at a 7.2–8.0 s one-off build per static shape and ≈1.9 GiB peak RSS.
Executed dispatches per generation on the numpy interpreter: `xor_p500` **7 592**, `cartpole_p100` **30 994** (count by wrapping `evoxc.backends.numpy.kernels.dispatch`; region ops run their body once per iteration and `evoxc.scan` peels step 0 at the enclosing level, so compare counts at the same generation).
The numpy interpreter charges ≈**19–20 µs per dispatched op**, so at population-sized tensors dispatch count is the lever; on a workload that tiles the population (func fit, a rollout) element volume is the lever, and padded gene geometry (`max_nodes`/`max_conns`) is the largest single term.
On `iree`+rocm a generation costs **max(host submission, device)**: a flat graph pays ≈63 µs fixed plus ≈2.2–2.5 µs per dispatch (kernel count is the lever), one `while` iteration costs ≈50 µs because the loop condition is read back to the host, and `Tensor.to(cpu)` costs ≈47–60 µs per leaf regardless of size — hence the pipeline packs its per-generation host return into one leaf.
Dispatch budget: `2 · species_size · pop_size` pairs as 1 `prepare_distance` plus `2 · species_size` `measure_distance` calls on the host leg and `ceil(species_size / k) + species_size` on the accelerator leg (`30` at `S = 20`, `k = 2`).
A kernel's live set is charged once per enclosing block, so `matrix` measures inside a fixed-length scan — one centre's transients rather than `species_size` of them; measure peak RSS and build next to wall time for every hoist.
Grouping is the accelerator leg's route only: on the interpreter it cuts dispatches but the 2× wider `(2 · pop_size, 2 · max_conns)` intermediates make the cell bimodally slower, so `grouped.group_size` answers `k = 1` for every non-accelerator state (capped at `ACCELERATOR_GROUP = 2`, `GROUP_ROW_BUDGET // pop_size` and the slot count).
The bounded structural-key allocator costs ~+4 / ~+8 / ~+33 ms per generation (+10 % / +5 % / +12 %) at pop 100 / 500 / 1000, almost entirely the `(pop_size, key_span + 1)` free-key table its collision fallback builds although fewer than 6 % of rows need it.

## Known Issues
`test/unit/algorithm/test_spawn.py` encodes `loc(p) = #{i : spawn[i] > 0 and E[i] <= p} - 1`, which disagrees with `block_layout` for a live zero-spawn species in the middle of a table — the oracle needs the "last qualifying block" form (cross-node; `test/` is not writable here).
The XOR3d pop-100 cell is the algorithm's tightest and no algorithm-side knob gates it (`max_in_degree = 4` refuses only 0.00–0.04 of fired add-conn gates, `max_depth = 16` sits above an observed maximum node level of 6–9; species kill, spawn starvation, elite displacement and the output row are non-binding).
Crossover is live only for species with more than `1 / survival_threshold` members (200 at the example 0.01): below that the survivor prefix is one genome and every non-elite child is a mutated copy of it.
`SpeciesController._block_layout`/`_prefix_sum_exclusive` are maintained **test seams** (only caller `test/unit/algorithm/conftest.py::raw_kernel`); production goes through `_kernels.block_layout`/`prefix_sum_exclusive`, so a behavioural change belongs in `_kernels.py`.
`evoxc.scan` peels step 0 and traces its body again, so each scan body is traced twice, and `DistanceRows.__call__`'s batch-`pop_size` branch, its batch-width rejection, `_check_population`'s payload rejection and the idle centre mode's tiling branch are unexercised inside the package.

## Known Risks (evoxc)
`evoxc.scan` de-sugars to `while` + gather/scatter, so a wide carry trips iree's affinity analysis on CUDA/ROCm; the numpy backend is the reference and packed carries are the mitigation.
Data-dependent `while_loop` trip counts work on numpy and iree/llvm-cpu but not on iree CUDA/ROCm or tvm — the reason every loop here is fixed-length.
`cumsum` does not export on compiler backends, hence triangular-matmul prefix sums.
`evoxc` has no `io_callback`/print op, so all logging stays in `show_details` on the host side.
`DistanceRows`' tiling is a real copy, so one population-shaped row materialises `pop_size` copies of that centre; inside `matrix`'s scan it is released with the region — one centre live on the host leg, `k` on the grouped accelerator leg.

## Constraints
No `jax` / `torch` / `numpy.random` import anywhere in this subtree; `import tensorneat` must stay free of them (release gate).
No Python branching, `bool()`, `len()` or host reads on traced values; static Python `if`/`for` over constructor configuration unrolls at trace time.
Constructor validation raises `tensorneat.common.errors.TensorNEATError`; trace/shape/backend errors propagate unchanged and are never re-wrapped or silently fallen back.
Nothing here reads `state` fields it did not register, except the documented genome/problem keys and the three formulation statics.
One documented exception to the package's ~1000-line rule is kept un-split on purpose: `neat/neat.py` at 1026 (one generation's `tell`/breed algebra; the cohesive split, if taken, is its host diagnostics into `_details.py`).
The others sit at or inside the baseline with no room left: `neat/_kernels.py` exactly 1000, `neat/species.py` 979, `hyperneat/hyperneat.py` 930, `neat/_distance.py` 427, `neat/_scan_accelerator.py` 416, `neat/_keys.py` 325, `hyperneat/genes.py` 297, `neat/_validation.py` 165 — a new speciation helper belongs in `_scan_accelerator.py`/`_distance.py`.

## Test Strategy
Tests live in the sibling `./../../test` tree (read-only from here): `test/unit/algorithm/` for the spawn/selection/stagnation algebra on fixed keys and the speciation scans, `test/parity/algorithm/` against `benchmarks/legacy_jax` (the divergences above are the only allowed mismatches), `test/integration/` for a full XOR run reaching its target with a reproducible seed.
Every kernel must be deterministic given `(seed, state)`: no dict ordering, no host RNG, no float-to-int conversion without an explicit mask.
The accelerator spellings have no committed test — their bit-identity to the host route is a hand run, and a `test/backends` cell building a gated pop-100 graph in that mode is the missing regression guard.
The speciate rewrite is verified by whole-run bit-identity (`fitness_history`, the four payload arrays, all ten `SpeciesState` leaves, `randkey`, `generation`, compared with `tobytes()` equality because padding NaNs defeat `np.array_equal`) at pop 100, 500 and 1000 — 144 arrays, zero mismatches.

## Routing Table
| Path | Responsibility |
| --- | --- |
| `./__init__.py` | Package surface only: re-exports `BaseAlgorithm`, `NEAT`, `HyperNEAT`, `HyperNEATFeedForward` and lists them in `__all__`. |
| `./base.py` | `BaseAlgorithm` batch-first protocol: `setup(state=None)`/`ask`/`transform`/`forward`/`prepare_forward`/`forward_prepared`/`tell`/`show_details` plus `num_inputs`/`num_outputs`/`pop_size`. |
| `./neat` | `NEAT` population cycle (ask/tell, elite handling, `pop_batch_size` scan batching), `SpeciesController` + `SpeciesState`, the `_kernels.py` host-spelling graph helpers, the `_scan_accelerator.py` accelerator-spelling scan bodies, the `_distance.py` digest-backed distance rows (`DistanceRows`, dense `matrix`, accelerator grouping), the `_keys.py` bounded structural-key allocator and the `_validation.py` host validators (including the `distance_centre` reader). |
| `./hyperneat` | `HyperNEAT`/`HyperNEATFeedForward` batch CPPN→substrate decode (`SubstrateSchedule` payload wrapper with its own `prepare_pass`), the `genes.py` `HyperNEATNode`/`HyperNEATConn` kernels, and the `./hyperneat/substrate` package of host-only coordinate geometry (`DefaultSubstrate`/`FullSubstrate`/`MLPSubstrate`). |
