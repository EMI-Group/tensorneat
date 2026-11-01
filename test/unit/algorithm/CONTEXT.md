# ./algorithm — the NEAT generation algebra, on fixed tables and fixed keys

## Intent
Prove the algebra that sits *between* two evaluated generations: speciation, stagnation, spawn allocation, parent pairing and the `NEAT.tell` cycle.
Every assertion is a structural invariant on a hand-written population — a conservation law (`sum(spawn) == pop_size`), a mask (`isnan(species_keys)` is the only dead-slot test), a bound (parents come only from the survivor prefix) or identity (two runs from one state agree bit for bit) — never a comparison against the pre-rewrite operator's output (that is `../../parity`).
The suite exists because these steps are the only ones rewritten from a data-dependent `while_loop` into fixed-length `evoxc.scan`s: a mask that writes one slot too many, a sort tie-order or a residual correction changes every seeded run while the forward pass still looks correct.

## Layout
`test_speciation.py` — the two speciation scans: full assignment to live keys, the strict `compatibility_threshold`, `d < o2c` stealing, the one-member count, untouched dead-slot centres, one key per created species, determinism and the permutation law.
`test_spawn.py` — both allocators (`rank`/`fitness`): `sum(spawn) == pop_size`, dead slots at zero, the `min_species_size` floor, the `spawn_number_change_rate` damping, monotone linear ranking, NaN-masked fitness reductions, and the non-negative `int32` integerisation; plus the two building blocks `_prefix_sum_exclusive` and `_block_layout`.
`test_parent_selection.py` — `_pair_parents`: the survivor clamp, the one-member self-crossover, elite pinning to the species' best member, the fitter-parent rule and the per-block parent draw, with the numpy global RNG proven irrelevant.
`test_neat_tell.py` — `_sanitize_fitness`, the generation counter, the stagnation predicate, population shape/size preservation, `ask` idempotence, `tell` determinism, `randkey` advance, `pop_batch_size` chunking and the constructor-rejection table.
`conftest.py` — the lazily resolved modules and the shared host fixtures (see below).

## Shared fixtures (`conftest.py`)
`species`, `neat_module`, `common`, `errors` and `payloads` are `scope="module"` fixtures that return **modules** and build nothing; each is resolved with `support.lazy.require`, so the pending `NotImplementedError` of the `src/` skeletons is a named skip rather than a collection error.
`controller(**overrides)`, `species_state(**overrides)`, `algorithm_state(...)`, `neat(**overrides)` and `species_kernel(name, controller, bookkeeping, *extra)` are function-scoped factories; their first argument is a module fixture, so a skeleton import skips before a constructor can raise.
`species_kernel` is the only way a suite calls a `SpeciesController` method: it passes the `SpeciesState` fields as graph inputs in `dataclasses.fields` order, rebuilds the dataclass inside the traced closure and returns host numpy.
The default tables are `P = 8`, `S = 4` with slots `0`/`1` live and `2`/`3` dead, `idx2species = [0,0,0,0,0,1,1,1]`, `next_species_key = 2.0`; every test overrides exactly the field it is about.

## Conventions
Oracles are formulas, not implementations: `expected_spawn` transcribes the documented spawn algebra, `assign_by_hand` re-derives the two speciation scans with Python loops, `stagnation_reference`/`sanitize_reference` are the closed forms, `species_ranking`/`survivor_count`/`expected_block_layout` are the documented definitions — none of them calls the kernel it checks.
Distances are controlled, never computed: `test_speciation.table_distance` returns rows of a static `(P, P)` host matrix indexed by an identity key stored in the payload's hidden node row, because `speciate` only consumes a `distance_fn` and the real compatibility metric would make the threshold/steal/argmin cases untestable.
Randomness is always an explicit `evoxc.random` key (`evoxc_key` from `../../conftest.py`, seeded from the test id); a test that draws asserts a bound or a repeat, never a specific draw, and no test touches the numpy global RNG except to prove it has no influence.
Every test function has a real signature and docstring and a body of `pytest.skip("not implemented: … (owner: algorithm/neat/…)")`, so the suite is collectable before the implementation wave and the skip names the method it is waiting for.
Backend is `"numpy"`; the per-test guard in `../conftest.py` fails any test that leaves `jax`/`torch`/`brax`/`gymnax`/`mujoco` in `sys.modules`.

## Constraints
A test may not edit `../../../src`, `../../../benchmarks` or `pyproject.toml`; a discovered defect is reported upward, not fixed here.
`import tensorneat` raises today (`algorithm/base.py` evaluates `State()` in a default argument), so nothing here imports `tensorneat` at module scope — only `support.runner` (which imports `evoxc` alone) is a module-scope import.
`HyperNEAT`'s `weight_threshold`/`max_weight` validation is **not** here: those are `HyperNEAT.__init__` arguments and belong to `./hyperneat/test_validation.py`.
Chunked (`pop_batch_size`) and unchunked runs draw different random streams by design, so no test asserts numerical equality between them; the chunking test uses a deterministic kernel instead.
Files stay under ~1000 lines; one module per `SpeciesController`/`NEAT` responsibility, not one per source file.

## Routing Table
| Path | Purpose |
|---|---|
| `./test_speciation.py` | the two speciation scans, dead-slot handling and the leftover rule. |
| `./test_spawn.py` | spawn allocation (`rank`/`fitness`), damping, integerisation, prefix sum, block layout. |
| `./test_parent_selection.py` | survivor prefix, elite pinning, winner/loser, per-block draws. |
| `./test_neat_tell.py` | fitness sanitising, generation counter, stagnation, shape preservation, chunking, constructor validation. |
| `./conftest.py` | lazily resolved modules plus the controller/state/genome/kernel factories. |
| `../../../src/tensorneat/algorithm/neat` | code under test (`NEAT`, `SpeciesController`) — read-only, escalate changes upward. |
