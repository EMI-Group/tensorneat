# ./algorithm — the NEAT generation algebra, verified on fixed tables and fixed keys

## Intent
Prove the algebra that sits *between* two evaluated generations: speciation, stagnation, spawn allocation, parent pairing and the `NEAT.tell` cycle.
Every assertion is a structural invariant on a hand-written population — a conservation law (`sum(spawn) == pop_size`), a mask (`isnan(species_keys)` is the only dead-slot test), a bound (parents only from the survivor prefix) or an identity (two runs from one state agree bit for bit).
The oracles are formulas re-derived from the documentation (numpy loops and closed forms), never a second copy of the kernel's gather/mask data movement.
The suite exists because these steps are the only ones rewritten from a data-dependent `evoxc.while_loop` into fixed-length `evoxc.scan`s: a mask that writes one slot too many, a sort tie-order or a residual correction changes every seeded run while the forward pass still looks correct.
Comparison against the pre-rewrite operator is `../../parity`'s job, not this directory's.
The package surface is pinned too: `test_exports.py` checks that `__all__` is exactly the documented set, that every entry is a callable defined inside the package, and that each documented `tensorneat.algorithm.*` spelling resolves to the class its defining submodule exposes.
The distance seam is pinned by `test_distance_dense.py`: the gated and dense `speciate` paths must produce one generation bit for bit, and the dense branch must be the branch a forced gate actually reaches.

## Layout and counts
`test_spawn.py` (69) — both allocators and the two spawn building blocks.
`test_speciation.py` (16) — the two speciation scans, dead-slot handling, the leftover and counter rules, the split-centre shape contract.
`test_parent_selection.py` (22) — survivor prefix, elite pinning, father-wins-tie, per-block draws, `elite_mask`.
`test_neat_tell.py` (37) — `_sanitize_fitness`, `tell` order/determinism/shape preservation, stagnation, `ask`/`transform`/`forward` wiring, `_batched` chunking, constructor validation.
`test_exports.py` (4) — the `tensorneat.algorithm` public surface: `__all__` is exactly the four documented spellings, every entry is a callable defined inside `tensorneat.algorithm`, and each spelling resolves to the class its defining submodule exposes.
`test_distance_dense.py` (7) — the gated/dense seam of `speciate`: a whole `NEAT.tell` driven twice from one state with `GATED_ROWS` forced to `0` and `10**9` (plus a `DistanceRows.matrix` call-count spy), `_kernels.speciate` on two hand-written distance tables with `gate` passed explicitly, and `DistanceRows.matrix` against its own per-centre spelling.
`conftest.py` — module fixtures, the `SpeciesState`/`State` table factories, and the two graph runners.
Whole directory: 155 passed, 0 xfailed, 0 failed, 0 skipped in ~5 s (`pytest test/unit/algorithm`), inside the default gate `pytest -m "not parity and not slow"`.

## Fixtures (`conftest.py`)
Module fixtures return modules only: `species` (`tensorneat.algorithm.neat.species`), `neat_module` (`neat.py`), `common`, `errors`, `payloads` (`support.payloads`).
`controller(**overrides)` → `SpeciesController` over `P = 8`, `S = 4`, `max_stagnation = 15`, `species_elitism = 2`, `spawn_number_change_rate = 0.5`, `genome_elitism = 2`, `survival_threshold = 0.1`, `min_species_size = 1`, `compatibility_threshold = 2.0`.
`species_state(**overrides)` → the canonical table: slots `0`/`1` live (keys `0.0`/`1.0`), slots `2`/`3` dead (NaN), `idx2species = [0,0,0,0,0,1,1,1]`, `next_species_key = 2.0`.
The four centre fields are the **split payload arrays** (`center_node_keys (S,N) int32`, `center_node_attrs (S,N,4) f32`, `center_conn_keys (S,E,2) i32`, `center_conn_attrs (S,E,1) f32`) — never packed integer gene rows; they are zero-filled because only `speciate` reads them.
`algorithm_state(bookkeeping, *, pop=None, generation=0.0, randkey=None)` → a `common.State` registering the payload under the four split `pop_*` entries when `pop` is a 4-field `GenomeTensors` (or under `pop` otherwise).
`neat(**overrides)` → `NEAT` over `payloads.default_genome()`; constructing it is cheap, `NEAT.setup` is the workload.
`species_kernel(method_name, bookkeeping, *extra, controller=None)` runs a `SpeciesController` method whose second parameter is a `SpeciesState`; the ten dataclass fields travel as graph inputs in `dataclasses.fields` order and a returned `SpeciesState` comes back as a name→numpy `dict`.
`raw_kernel(method_name, *arrays, controller=None)` runs a helper with no `SpeciesState` (`_prefix_sum_exclusive`, `_sort_desc_legacy`, and `_block_layout`, which is an instance method despite living next to the two statics).
Both runners look the method up on the class and pass the controller only for a plain instance method (`inspect.getattr_static` decides), so the controller stays a host-side value baked into the trace.
Module-scope imports of `tensorneat`, `numpy` and `support.*` are the convention here; a `SpeciesController` is never passed where a `SpeciesState` is expected (the runner's first positional after the method name is the table).

## Oracles with teeth
`expected_spawn` transcribes the spawn algebra (rank/linear target, `f' = f - min_live(f) + 1`, damping against `member_count`, `min_species_size` floor, dead slots to `0`, cast, residual) in float32 so a floor boundary cannot differ by an ULP.
`expected_block_layout` (a local copy lives in `test_parent_selection.py`) uses the implementation's authoritative rule: `loc(p)` is the **last** `i` with `spawn[i] > 0` and `E[i] <= p`, and `idx_in_block(p) = p - E[loc(p)]`; the *count* of qualifying blocks is wrong whenever a live zero-spawn species sits mid-table (`[3,0,5,0]` is the regression).
The residual lands on the first **live** slot (slot `0` only after the descending sort), so fixtures keep slot `0` live; the residual may also be negative when the `min_species_size` floor overflows the budget.
`assign_by_hand` re-derives both speciation scans with Python loops over a controlled `(P, P)` distance table keyed by each genome's identity, so the threshold/steal/`argmin` cases are testable without the real compatibility metric.
`test_distance_dense.py`'s `SpeciationCase` writes the whole `(P + S, P)` distance table by hand, row `r` belonging to the genome tagged `r` (population rows `0..P-1`, initial centre rows `P..P+S-1`); a fake `distance_fn` serves it through `matrix` (the initial centre rows, for the dense scan 1) and through `__call__` (a gather by the centre's gene tag, for scan 2). The expected `idx2species`, keys, `member_count`, centre-row identity and created-slot bookkeeping are derived from the documented scan rules, so the oracle never runs a second kernel.
`stagnation_reference` and `sanitize_reference` are the documented closed forms.
The father-wins-tie rule is pinned by *reproducing* the draw: the test runs the documented `evoxc.random.randint(key, (S,2,P), 0, survive)` plus `flat_gather` in its own one-op graph, then asserts `winner` is the father and `loser` the mother on an all-tied table.

## Gotchas for agents
pytest **hangs** (no output, needs a kill) when a test lets an exception escape an evoxc graph under the default `--tb=auto`; use `--tb=short` while debugging a deliberately broken kernel.
The suite imports `tensorneat` without `jax`: `src/tensorneat/pipeline.py` is evoxc-native and the per-test guard in `../conftest.py` keeps any optional framework out of `sys.modules`.
A `generation` value registered as a host 0-d numpy constant cannot be consumed by `_kernels.scalar_f32` inside a trace; send it as a positional graph input instead.
`evoxc.Tensor` has no `__array__`, so `np.asarray` on a kernel result silently yields a 0-d object array — unwrap with `support.runner.to_numpy` (or `.numpy()`).
Shapes stay small (`pop ≤ 8`, `max_nodes ≤ 8`) except where a test exists specifically to trip a static bound; `test_neat_tell.py` shares one `NEAT.setup` per test, which is the slowest thing in the directory (~0.5 s).
Files stay under ~1000 lines (`test_neat_tell.py` sits at 993, `test_speciation.py` at 1000); one module per `SpeciesController`/`NEAT` responsibility, not one per source file.

## Known documentation drift in `src/` (reported upward, not asserted here)
`_kernels.pair_parents` caps the survivor count as `max(min(member_count, pop_size), 1)`, which differs from the documented `min(member_count, pop_size)` only for a live species with `member_count == 0`; such a table never receives a block, so the case is unreachable through the public path.
`_kernels.block_layout`'s docstring says the `spawn > 0` mask term is what keeps a zero-spawn species from owning a slot; with the last-qualifying block rule that term is redundant (a `>= 0` mask gives identical arrays on every valid table), so no test can distinguish it.
`HyperNEAT`'s `weight_threshold`/`max_weight` validation is **not** here: those are `HyperNEAT.__init__` arguments and belong to `../hyperneat/`.
`test/release_gate_no_jax.py`'s resolved-surface list names `tensorneat.algorithm.NEAT`/`HyperNEAT` only, while `src/tensorneat/CONTEXT.md` documents `tensorneat.algorithm.HyperNEATFeedForward` as well; the package now re-exports it and `./test_exports.py` pins it, and the release-gate list is test-owned (reported upward).

## Coverage gaps in this directory
`../../../src/tensorneat/algorithm/hyperneat/**` has no behavioural test anywhere in the tree: `HyperNEAT`, `HyperNEATFeedForward`, `HyperNEATNode`, `HyperNEATConn`, `SubstrateSchedule` and the four substrate classes are touched only by `./test_exports.py` (identity and `__all__`) and by `../../release_gate_no_jax.py` (an attribute name), so their setup/transform/forward/tell and the substrate geometry are exercised only by `../../../examples/func_fit/xor_hyperneat.py` and `xor_hyperneat_feedforward.py`, which assert nothing.
`_distance.DistanceRows` is constructed by `test_distance_dense.py`: `matrix` is asserted against one `__call__` per centre row on four different genomes, and the batch-1 tiling of `__call__` runs there too. What stays unverified is `DistanceRows.__init__`'s two `TensorNEATError` rejections (digest population axis and payload batch) and `_check_population`'s shape rejection, plus `_kernels.speciate`'s "dense needs a digest-backed `distance_fn`" rejection (every forced-dense test hands it a table with `matrix`).
The dense `speciate` branch is exercised by `test_distance_dense.py` with `GATED_ROWS` forced to `0` and with `gate` passed explicitly; the rest of the tree still tops out at `S * P = 32`, so nothing else reaches it.
`_keys.node_keys_for_split`/`first_free_keys`/`validate_key_space` have no assertion anywhere: the allocator runs inside `NEAT.tell`, but no test checks that offspring node keys are unique, inside `[0, key_span)` or produced by the fallback.
`SpeciesController.update_species` is exercised only through `tell`, and `_gather_species`'s reorder of `species_keys` and the four centre arrays is never asserted; both spawn allocators and the block layout depend on that sorted slot order.
`_sort_desc_legacy` is never run directly (no test calls it through `raw_kernel`): its reverse-index tie order is only pinned indirectly, through `_pair_parents`' population-axis ranking in `./test_parent_selection.py`.
`NEAT.prepare_forward`/`forward_prepared` are not called here (the genome-level `prepare_pass` split lives in `../genome`), and `NEAT.show_details` is unasserted.
`_validation`'s four helpers and `SpeciesController.__init__`'s own rejections are reached only through `NEAT.__init__` in `./test_neat_tell.py`.

## Constraints
Backend is `"numpy"`; the per-test guard in `../conftest.py` fails any test that leaves `jax`/`torch`/`brax`/`gymnax`/`mujoco` in `sys.modules`.
No test body may be a `pytest.skip` placeholder, and an assertion is never weakened to get green — a real source defect becomes a strict `xfail` with `reason="src defect: …"` instead.
A test may not edit `../../../src`, `../../../benchmarks` or `pyproject.toml`; a discovered defect is reported upward, not fixed here.
Randomness is always an explicit `evoxc.random` key (`evoxc_key` from `../../conftest.py`, seeded from the test id); a test asserts a bound or a repeat, never a specific draw.
Deterministic and fast: no timing, no network, no GPU, no `tmp_path` persistence.

## Routing Table
| Path | Purpose |
|---|---|
| `./test_spawn.py` | spawn allocation (`rank`/`fitness`), residual correction, damping, floor, integerisation, prefix sum, block layout. |
| `./test_speciation.py` | the two speciation scans, dead slots, leftover rule, counter advance, split-centre shape contract. |
| `./test_parent_selection.py` | survivor prefix, one-member species, elite pinning and `elite_mask`, father-wins-tie, per-block draws, RNG hygiene. |
| `./test_neat_tell.py` | `_sanitize_fitness`, `tell` order/shape/determinism, stagnation, `ask`/`transform`/`forward`, chunking, constructor validation. |
| `./test_exports.py` | `tensorneat.algorithm.__all__` is exactly the documented spellings, every entry is a callable defined inside the package, and each spelling is the class its defining submodule exposes. |
| `./test_distance_dense.py` | the gated/dense seam: whole-`tell` equivalence under `GATED_ROWS = 0` vs `10**9` plus a `DistanceRows.matrix` call-count spy, direct `_kernels.speciate` on hand-written distance tables with `gate` explicit, and `DistanceRows.matrix` vs per-centre `__call__`. |
| `./conftest.py` | module fixtures, table factories, `species_kernel`/`raw_kernel`. |
| `../../../src/tensorneat/algorithm/neat` | code under test (`NEAT`, `SpeciesController`, `_kernels`, `_validation`) — read-only, escalate changes upward. |
