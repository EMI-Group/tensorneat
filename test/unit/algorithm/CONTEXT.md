# ./algorithm — the NEAT generation algebra, verified on fixed tables and fixed keys

## Intent
Prove the algebra that sits *between* two evaluated generations: speciation, stagnation, spawn allocation, parent pairing and the `NEAT.tell` cycle.
Every assertion is a structural invariant on a hand-written population — a conservation law (`sum(spawn) == pop_size`), a mask (`isnan(species_keys)` is the only dead-slot test), a bound (parents only from the survivor prefix) or an identity (two runs from one state agree bit for bit).
The oracles are formulas re-derived from the documentation (numpy loops and closed forms), never a second copy of the kernel's gather/mask data movement.
The suite exists because these steps are the only ones rewritten from a data-dependent `evoxc.while_loop` into fixed-length `evoxc.scan`s: a mask that writes one slot too many, a sort tie-order or a residual correction changes every seeded run while the forward pass still looks correct.
Comparison against the pre-rewrite operator is `../../parity`'s job, not this directory's.

## Layout and counts
`test_spawn.py` (69) — both allocators and the two spawn building blocks.
`test_speciation.py` (16) — the two speciation scans, dead-slot handling, the leftover and counter rules, the split-centre shape contract.
`test_parent_selection.py` (22) — survivor prefix, elite pinning, father-wins-tie, per-block draws, `elite_mask`.
`test_neat_tell.py` (33 + 4 strict xfail) — `_sanitize_fitness`, `tell` order/determinism/shape preservation, stagnation, `ask`/`transform`/`forward` wiring, `_batched` chunking, constructor validation.
`conftest.py` — module fixtures, the `SpeciesState`/`State` table factories, and the two graph runners.
Whole directory: 140 passed, 4 xfailed, 0 failed, 0 skipped in ~3.4 s (`pytest test/unit/algorithm`), inside the default gate `pytest -m "not parity and not slow"`.

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
`stagnation_reference` and `sanitize_reference` are the documented closed forms.
The father-wins-tie rule is pinned by *reproducing* the draw: the test runs the documented `evoxc.random.randint(key, (S,2,P), 0, survive)` plus `flat_gather` in its own one-op graph, then asserts `winner` is the father and `loser` the mother on an all-tied table.

## Pinned source defect (4 strict xfails in `test_neat_tell.py`)
`src/tensorneat/algorithm/neat/neat.py::NEAT._new_gene_keys` reshapes the `(P,)` result of `genome.max_node_key`/`max_conn_marker` to `()`, so `NEAT.tell` raises `evoxc.ShapeError: reshape: element counts differ: 8 vs 1` for any `pop_size > 1`.
Reproducer: `NEAT(genome=payloads.default_genome(), pop_size=8, species_size=4).setup(State(randkey=evoxc.random.key(42)))`, then trace `tell` on a `(8,)` fitness vector; `pop_size == 1` works.
The four tests that drive a real `tell` carry `@pytest.mark.xfail(strict=True, reason=TELL_DEFECT)` and flip to XPASS (a failure) the moment `src/` is fixed; the rest of the `tell` coverage runs through the species-level kernels and a `pop_size == 1` state.

## Gotchas for agents
pytest **hangs** (no output, needs a kill) when a test lets an exception escape an evoxc graph under the default `--tb=auto`; use `--tb=short` while debugging a deliberately broken kernel.
`import tensorneat` still loads the legacy JAX `src/tensorneat/pipeline.py` through `tensorneat/__init__`, so this suite needs `jax` installed until that file becomes evoxc-native; no test imports or uses jax, and the per-test guard in `../conftest.py` stays honest because the import happens at collection time.
A `generation` value registered as a host 0-d numpy constant cannot be consumed by `_kernels.scalar_f32` inside a trace; send it as a positional graph input instead.
`evoxc.Tensor` has no `__array__`, so `np.asarray` on a kernel result silently yields a 0-d object array — unwrap with `support.runner.to_numpy` (or `.numpy()`).
Shapes stay small (`pop ≤ 8`, `max_nodes ≤ 8`) except where a test exists specifically to trip a static bound; `test_neat_tell.py` shares one `NEAT.setup` per test, which is the slowest thing in the directory (~0.5 s).
Files stay under ~1000 lines (`test_neat_tell.py` sits at 1005, `test_speciation.py` at 1000); one module per `SpeciesController`/`NEAT` responsibility, not one per source file.

## Known documentation drift in `src/` (reported upward, not asserted here)
`_kernels.pair_parents` caps the survivor count as `max(min(member_count, pop_size), 1)`, which differs from the documented `min(member_count, pop_size)` only for a live species with `member_count == 0`; such a table never receives a block, so the case is unreachable through the public path.
`_kernels.block_layout`'s docstring says the `spawn > 0` mask term is what keeps a zero-spawn species from owning a slot; with the last-qualifying block rule that term is redundant (a `>= 0` mask gives identical arrays on every valid table), so no test can distinguish it.
`HyperNEAT`'s `weight_threshold`/`max_weight` validation is **not** here: those are `HyperNEAT.__init__` arguments and belong to `../hyperneat/`.

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
| `./conftest.py` | module fixtures, table factories, `species_kernel`/`raw_kernel`. |
| `../../../src/tensorneat/algorithm/neat` | code under test (`NEAT`, `SpeciesController`, `_kernels`, `_validation`) — read-only, escalate changes upward. |
