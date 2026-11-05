# ./algorithm — the NEAT generation algebra and the whole HyperNEAT subtree

## Intent
Prove two things: the algebra that sits *between* two evaluated generations (speciation, stagnation, spawn allocation, parent pairing, the `NEAT.tell` cycle), and the behaviour of `algorithm/hyperneat/` (substrate geometry, the two substrate gene kernels, the CPPN→substrate decode, both algorithm variants and small end-to-end runs).
Every assertion compares against something independent of the implementation: a closed form, a hand-written Python loop, a hand-built table, or a structural invariant (a conservation law like `sum(spawn) == pop_size`, a mask, a bound, an identity such as two branches agreeing bit for bit).
The oracles are re-derived from the documentation (numpy loops, closed forms, hand `(P, S)` distance tables), never a second copy of the kernel's gather/mask data movement.
The generation steps matter because they were rewritten from data-dependent `evoxc.while_loop`s into fixed-length `evoxc.scan`s: a mask that writes one slot too many, a sort tie-order or a residual correction changes every seeded run while the forward pass still looks correct.
The HyperNEAT decode matters because a permuted query row or a shifted payload column silently reorders weights while every shape stays legal.
Comparison against the pre-rewrite operator is `../../parity`'s job, not this directory's.
The package surface is pinned too: `test_exports.py` checks that `__all__` is exactly the documented set, that every entry is a callable defined inside the package, and that each documented `tensorneat.algorithm.*` spelling resolves to the class its defining submodule exposes.
The distance seam is pinned by `test_distance_dense.py`: the gated and dense `speciate` paths must produce one generation bit for bit, and the dense branch must be the branch a forced gate actually reaches.

## Layout and counts
`test_spawn.py` (69) — both allocators and the two spawn building blocks.
`test_speciation.py` (16) — the two speciation scans, dead-slot handling, the leftover and counter rules, the split-centre shape contract.
`test_parent_selection.py` (22) — survivor prefix, elite pinning, father-wins-tie, per-block draws, `elite_mask`.
`test_neat_tell.py` (37) — `_sanitize_fitness`, `tell` order/determinism/shape preservation, stagnation, `ask`/`transform`/`forward` wiring, `_batched` chunking, constructor validation.
`test_exports.py` (4) — the `tensorneat.algorithm` public surface: `__all__` equals the documented set, every entry is a callable defined inside the package, and each documented spelling resolves to the class its defining submodule exposes.
`test_distance_dense.py` (7) — the `GATED_ROWS` seam: one `NEAT.tell` through both branches from one state, a counting spy proving which branch ran, a hand-computed `(P + S, P)` distance table driven through both gates, and `DistanceRows.matrix` against `__call__` centre by centre.
`test_substrate.py` (123) — the HyperNEAT substrate geometry and its validation matrix, on plain host NumPy: no graph, no runner, no `State`.
`test_genes.py` (18) — `HyperNEATNode` (option tables, `combine`, masked aggregation, output rows unactivated) and `HyperNEATConn` (weight column by name, raw rows and the prepared `ConnOperands` bundle, no kernel-side NaN guard).
`test_hyperneat_weights.py` (10) — the CPPN→weight map at `0`, the threshold tie, just-above-threshold, large magnitude and NaN, plus the batch/shape contract.
`test_hyperneat.py` (52) — the decode algebra (`_query_coors`, `_expand_leaf`/`_expand_schedule` row order `p*Q+q`, `_assemble_substrate`, `_static_bounds`, the whole `_validate` matrix), the protocol surface, `transform`'s batch independence, and `forward`/`prepare_forward`/`forward_prepared`.
`test_hyperneat_feedforward.py` (11) — the variant delta, its rejection of a recurrent substrate, payload equality with the base decode, `forward` on a depth-1 substrate where the single tick is the same arithmetic, and the `max_depth` it passes.
`test_hyperneat_e2e.py` (4) — pop-4 two-generation `Pipeline` runs of both variants over `CustomFuncFit`, determinism per seed, and a `network_dict`/`show` report on the best CPPN.
`hyperneat_kernel.py` — the shared trace harness (not a test module, not collected): `make_hyper`, `make_state`, `cppn_payload`, `make_substrate`, `make_cppn_genome`, `make_neat`, `run_method`, and the small default geometry constants.
`conftest.py` — module fixtures, the `SpeciesState`/`State` table factories, and the two graph runners.
Whole directory: 373 collected — 373 passed, 0 failed, 0 skipped in ~10 s (`pytest test/unit/algorithm`), inside the default gate `pytest -m "not parity and not slow"`.
Coverage of `src/tensorneat/algorithm/hyperneat/` from this directory alone: 99 % (618/620 statements) — `genes.py`, `hyperneat_feedforward.py`, both `__init__.py`, `substrate/base.py`, `substrate/default.py`, `substrate/full.py` and `substrate/mlp.py` at 100 %, `hyperneat.py` missing only `_attr_columns`' two unreachable rejections (`hyperneat.py:842`, `:848`).

## Fixtures and the trace harness
`conftest.py` module fixtures return modules only: `species` (`tensorneat.algorithm.neat.species`), `neat_module` (`neat.py`), `common`, `errors`, `payloads` (`support.payloads`).
`controller(**overrides)` → `SpeciesController` over `P = 8`, `S = 4`, `max_stagnation = 15`, `species_elitism = 2`, `spawn_number_change_rate = 0.5`, `genome_elitism = 2`, `survival_threshold = 0.1`, `min_species_size = 1`, `compatibility_threshold = 2.0`.
`species_state(**overrides)` → the canonical table: slots `0`/`1` live (keys `0.0`/`1.0`), slots `2`/`3` dead (NaN), `idx2species = [0,0,0,0,0,1,1,1]`, `next_species_key = 2.0`.
The four centre fields are the **split payload arrays** (`center_node_keys (S,N) int32`, `center_node_attrs (S,N,4) f32`, `center_conn_keys (S,E,2) i32`, `center_conn_attrs (S,E,1) f32`) — never packed integer gene rows; they are zero-filled because only `speciate` reads them.
`algorithm_state(bookkeeping, *, pop=None, generation=0.0, randkey=None)` → a `common.State` registering the payload under the four split `pop_*` entries when `pop` is a 4-field `GenomeTensors` (or under `pop` otherwise).
`neat(**overrides)` → `NEAT` over `payloads.default_genome()`; constructing it is cheap, `NEAT.setup` is the workload.
`species_kernel(method_name, bookkeeping, *extra, controller=None)` runs a `SpeciesController` method whose second parameter is a `SpeciesState`; the ten dataclass fields travel as graph inputs in `dataclasses.fields` order and a returned `SpeciesState` comes back as a name→numpy `dict`.
`raw_kernel(method_name, *arrays, controller=None)` runs a helper with no `SpeciesState` (`_prefix_sum_exclusive`, `_sort_desc_legacy`, and `_block_layout`, which is an instance method despite living next to the two statics).
Both runners look the method up on the class and pass the controller only for a plain instance method (`inspect.getattr_static` decides), so the controller stays a host-side value baked into the trace.
`hyperneat_kernel.py` is the same idea for `HyperNEAT` and the two genes: `run_method(obj, "method", *args, state=None)` closes over the host `State`, passes numpy arrays as tensors and Python scalars as statics, flattens a nested namedtuple (`Transformed`, `SubstrateSchedule`, `GenomeTensors`) and rebuilds it inside the trace, and returns host numpy; the default geometry is 2 CPPN inputs incl. the bias, 1 hidden neuron, 1 output, `P = 4`, `Q = 4`, `N = 4`, CPPN query width `C = 4`.
A hyperneat suite imports it as `from .hyperneat_kernel import ...`; nothing in it is collected by pytest.
Module-scope imports of `tensorneat`, `numpy` and `support.*` are the convention here; a `SpeciesController` is never passed where a `SpeciesState` is expected (the runner's first positional after the method name is the table).

## Oracles with teeth
`expected_spawn` transcribes the spawn algebra (rank/linear target, `f' = f - min_live(f) + 1`, damping against `member_count`, `min_species_size` floor, dead slots to `0`, cast, residual) in float32 so a floor boundary cannot differ by an ULP.
`expected_block_layout` (a local copy lives in `test_parent_selection.py`) uses the implementation's authoritative rule: `loc(p)` is the **last** `i` with `spawn[i] > 0` and `E[i] <= p`, and `idx_in_block(p) = p - E[loc(p)]`; the *count* of qualifying blocks is wrong whenever a live zero-spawn species sits mid-table (`[3,0,5,0]` is the regression).
The residual lands on the first **live** slot (slot `0` only after the descending sort), so fixtures keep slot `0` live; the residual may also be negative when the `min_species_size` floor overflows the budget.
`assign_by_hand` re-derives both speciation scans with Python loops over a controlled `(P, P)` distance table keyed by each genome's identity, so the threshold/steal/`argmin` cases are testable without the real compatibility metric.
`test_distance_dense.py`'s `SpeciationCase` writes the whole `(P + S, P)` distance table by hand, row `r` belonging to the genome tagged `r` (population rows `0..P-1`, initial centre rows `P..P+S-1`); a fake `distance_fn` serves it through `matrix` (the initial centre rows, for the dense scan 1) and through `__call__` (a gather by the centre's gene tag, for scan 2). The expected `idx2species`, keys, `member_count`, centre-row identity and created-slot bookkeeping are derived from the documented scan rules, so the oracle never runs a second kernel.
`stagnation_reference` and `sanitize_reference` are the documented closed forms.
The father-wins-tie rule is pinned by *reproducing* the draw: the test runs the documented `evoxc.random.randint(key, (S,2,P), 0, survive)` plus `flat_gather` in its own one-op graph, then asserts `winner` is the father and `loser` the mother on an all-tied table.
The substrate oracles are literal arrays hand-computed from the documented row order (`[src, tgt]` coordinate pairs, source varying slowest) and from the key blocks (inputs `0..I-1`, outputs `I..I+O-1`, hidden after), plus endpoint alignment re-derived with a Python loop over `conn_index`.
`test_hyperneat_weights.py` writes both a float32 transcription of the documented map (so a comparison can be exact) and the float64 closed form.
The node/conn gene oracles are left-to-right float32 folds (`+0.0` start, `1/(1+exp(-z))`) and per-individual Python loops over a `(P, N, K)` batch.
The `test_hyperneat.py` forward oracle is a hand phenotype evaluation on hand-set `conn_attrs` weights and all-zero option columns, with the bias column appended last.

## Read-out contract (source fixed, pinned by a passing test)
`test_hyperneat.py::test_forward_reads_the_substrate_output_neurons` pins the read-out on a substrate whose node order is inputs, outputs, hidden: `Transformed.output_slot` must name the output block's node (harness key `2`), and `forward` must return that neuron's documented `sigmoid(w3 * tanh(w0*a + w1 + w2*tanh(w0*a + w1)))` value instead of a hidden row's unactivated self-loop sum.
That is what the tree does now — the executor genome reads `transformed.output_slot` — so the test passes and the module carries no `xfail`; the defect it used to pin (outputs read from the last `num_outputs` node rows, which with a hidden block is a hidden neuron) must not come back silently.
## Known issues (reported, not fixed here)
`src/` documentation drift reported upward: `_weights_from_query`'s docstring claims the weights land in `[-max_weight, max_weight]`, which only holds for `|w| <= 1` (an unbounded CPPN output scales past `max_weight`); `_kernels.pair_parents` caps the survivor count as `max(min(member_count, pop_size), 1)`, which differs from the documented `min(...)` only for an unreachable live species with `member_count == 0`; `_kernels.block_layout`'s `spawn > 0` mask term is redundant under the last-qualifying-block rule.
`test/release_gate_no_jax.py`'s resolved-surface list names `tensorneat.algorithm.NEAT`/`HyperNEAT`/`HyperNEATFeedForward`, matching this directory's `__all__` pin in `./test_exports.py`.

## Gotchas for agents
pytest **hangs** (no output, needs a kill) when a test lets an exception escape an evoxc graph under the default `--tb=auto`; use `--tb=short` while debugging a deliberately broken kernel.
`evoxc`'s static allowlist rejects a hand-made gene object as a static argument, and a graph output must be a `SymbolicTensor`: `prepare_forward` (which returns a `_PreparedPass` holding host statics) therefore cannot be its own traced graph — run both halves inside one closure. The same restriction makes `NEAT.tell` inexpressible through `hyperneat_kernel.run_method` (its `State` carries concrete tensors), so the one-generation proof goes through `Pipeline.step`.
`np.array_equal` is wrong on a gene payload without `equal_nan=True`: `node_attrs`/`conn_attrs` are NaN-padded, so a payload compared with itself reports "differs".
`tensorneat.common.tools.to_numpy` is a one-*leaf* accessor and stacks a `GenomeTensors` into one array; unwrap a container with `support.runner.to_numpy` instead.
The suite imports `tensorneat` without `jax`: `src/tensorneat/pipeline.py` is evoxc-native and the per-test guard in `../conftest.py` keeps any optional framework out of `sys.modules`.
A `generation` value registered as a host 0-d numpy constant cannot be consumed by `_kernels.scalar_f32` inside a trace; send it as a positional graph input instead.
`evoxc.Tensor` has no `__array__`, so `np.asarray` on a kernel result silently yields a 0-d object array — unwrap with `support.runner.to_numpy` (or `.numpy()`).
Coverage is measured with `uv run --with coverage coverage run --source=src/tensorneat/algorithm/hyperneat -m pytest test/unit/algorithm`, then `coverage report -m`; `pytest --cov` is not installed, and the `--with pytest-cov` route breaks on `test/unit/conftest.py`'s import recorder ("cannot load module more than once per process").
The `output_transform=ACT.sigmoid` wiring of the e2e runs emits `RuntimeWarning: overflow encountered in exp` on saturating weights; the suite deliberately does not silence it.
Shapes stay small (`pop ≤ 8`, `max_nodes ≤ 8`) except where a test exists specifically to trip a static bound; `test_neat_tell.py` and the hyperneat graph builders each pay one `setup`/build per shape (~0.2–1 s), which dominates the directory's wall clock.
Files stay under ~1000 lines (`test_speciation.py` and `test_hyperneat_feedforward.py` are at 1000, `test_hyperneat.py` 997, `test_neat_tell.py` 993, `test_substrate.py` 990, `test_genes.py` 983); split by responsibility, not to keep adding to a full file.

## Remaining coverage gaps in this directory
`_distance.DistanceRows.__init__`'s two `TensorNEATError` rejections (digest/population batch mismatch), `_check_population`'s shape rejection, and `_kernels.speciate`'s "dense needs a digest-backed `distance_fn`" rejection are unverified.
`HyperNEAT._attr_columns`' two rejections (no `weight` spec, empty spec table) are unreachable through the public surface: they need a hand-made gene object evoxc's static allowlist refuses.
`_keys.node_keys_for_split`/`first_free_keys`/`validate_key_space` have no assertion anywhere: the allocator runs inside `NEAT.tell`, but no test checks that offspring node keys are unique, inside `[0, key_span)` or produced by the fallback.
`SpeciesController.update_species` is exercised only through `tell`, and `_gather_species`'s reorder of `species_keys` and the four centre arrays is never asserted; both spawn allocators and the block layout depend on that sorted slot order.
`_sort_desc_legacy` is never run directly (no test calls it through `raw_kernel`): its reverse-index tie order is only pinned indirectly, through `_pair_parents`' population-axis ranking in `./test_parent_selection.py`.
`NEAT.prepare_forward`/`forward_prepared` are not called here (the genome-level `prepare_pass` split lives in `../genome`), and `NEAT.show_details` is unasserted.
`_validation`'s four helpers and `SpeciesController.__init__`'s own rejections are reached only through `NEAT.__init__` in `./test_neat_tell.py`.
HyperNEAT is also exercised end to end outside this directory: `../../integration/test_hyperneat_integration.py` (3 cases) drives both variants through a whole `Pipeline` — config fallback, substrate rebuild, checkpoint and CPPN decode — alongside this directory's `./test_hyperneat_e2e.py`; `../../parity/algorithm/` still does not exist, so the decode is compared only against the documented algebra.

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
| `./test_distance_dense.py` | the `GATED_ROWS` seam: dense vs gated `tell` equivalence, the `matrix` spy, a hand-written distance table, `DistanceRows.matrix` vs `__call__`. |
| `./test_exports.py` | `tensorneat.algorithm.__all__` and the documented spellings (`BaseAlgorithm`, `NEAT`, `HyperNEAT`, `HyperNEATFeedForward`), including identity with the defining submodule. |
| `./test_substrate.py` | HyperNEAT substrate geometry on plain host NumPy: `node_key_blocks`/`cartesian_product`/`layer_coors`/`as_coor_array`/`check_same_coordinate_dim`, `DefaultSubstrate` re-exposure, `validate()` re-checks and `__repr__`, `FullSubstrate`/`MLPSubstrate` row blocks plus both `analysis_substrate` functions, and the whole validation matrix. |
| `./test_genes.py` | `HyperNEATNode` option tables/`combine`/masked aggregation/output-row rule and `HyperNEATConn` weight-column multiply on raw rows and the `ConnOperands` bundle. |
| `./test_hyperneat_weights.py` | the CPPN→weight map against a float32 transcription and the float64 closed form. |
| `./test_hyperneat.py` | the decode algebra (`_query_coors`, `_expand_*`, `_assemble_substrate`, `_static_bounds`, `_validate`), the protocol surface, `transform` batch independence, `forward`/prepared split. |
| `./test_hyperneat_feedforward.py` | the feedforward variant delta, its recurrent-substrate rejection, payload equality with the base decode, single-tick `forward` equality, `max_depth`. |
| `./test_hyperneat_e2e.py` | pop-4 two-generation `Pipeline` runs of both variants over `CustomFuncFit`, per-seed determinism, `network_dict`/`show`. |
| `./hyperneat_kernel.py` | shared HyperNEAT trace harness (`run_method`, `make_hyper`, `make_state`, `cppn_payload`, default geometry) — not collected. |
| `./conftest.py` | module fixtures, table factories, `species_kernel`/`raw_kernel`. |
| `../../../src/tensorneat/algorithm/neat` | code under test (`NEAT`, `SpeciesController`, `_kernels`, `_distance`, `_validation`) — read-only, escalate changes upward. |
| `../../../src/tensorneat/algorithm/hyperneat` | code under test (`HyperNEAT`, `HyperNEATFeedForward`, `genes.py`, `substrate/**`) — read-only, escalate changes upward. |
