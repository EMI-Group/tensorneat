# ./unit/genome — payload, forward/grad and operator suites for the genome subtree

## Intent
Prove that each genome kernel computes what its docstring promises, on the reference numpy backend, with no optional dependency.
The assertions are about *invariants* and hand-written oracles, never about the implementation's own intermediate state: closed-form values on analytically chosen networks, `support.oracle` rollouts on the tedious topologies, and set/count algebra for the stochastic operators.
Nothing skips and nothing is `xfail`: the suite is a verifier, not a spec (`git log` has the spec era).
Two modules pin a guard that a naive implementation would pass by accident: `test_forward.py` activates hidden rows but not output rows, and `test_recurrent.py` contrasts a live sibling against a starved one.

## Layout
99 test functions, 108 collected cases, one module per source module: `test_payload.py` 15, `test_forward.py` 11, `test_recurrent.py` 10 (19 cases — the one parametrized module: `activate_time` ticks and the invalid-tick rejections), `test_grad.py` 9, `test_mutation.py` 15, `test_crossover.py` 9, `test_distance.py` 9, `test_distance_digest.py` 8, `test_host_helpers.py` 8, `test_transform_validation.py` 5.
`test_payload.py` — `initialize` four-array layout/dtypes/sentinels, the input/output row-key invariant, `setup` chaining order and state immutability, the three constructor rejections, `transform` equal to `support.oracle.schedule` field by field and accepted by `common.validate_schedule`, and `edge_overflow` on a too-small `max_in_degree`.
`test_forward.py` — closed-form level composition on a 13-row chain (every intermediate read out through a probe output row), output rows unactivated while hidden rows are, activation `-1` = identity, both transforms by composition, bypassed node inert, unconnected node + descendants NaN, padded rows inert, population axis a pure batch axis, random DAGs vs the oracle, the `max_depth` tail bound.
`test_recurrent.py` — Jacobi ticks via closed-form orbits, input re-injection per tick, `input_transform` never applied, `output_transform` once after the last tick, NaN starvation across ticks, ring vs the oracle, `activate_time` = tick count, `activate_time` validation.
`test_grad.py` — `grad` returns an `evoxc.Graph` whose loss was traced (not called on an ndarray), determinism, shapes/dtypes, conn-weight and node-attribute gradients vs central finite differences (max error ≈5e-5 at `h=1e-2`, tolerance 1e-3), closed-form bias-only network, `NaN`/padding never poisons, padded rows gradient to exactly 0, `sum(loss)` scalarization leaves per-individual gradients unchanged.
`test_mutation.py` — probability validation, identity at zero, capacity (`max_nodes`/`max_conns`/`max_in_degree`), caller-allocated `new_node_keys`/`new_conn_keys` honoured, key uniqueness and sentinels intact, untouched rows bit-identical, one change per operation type on four independent per-individual gates (one individual may both gain and lose a node, the four stages composing sequentially), no duplicate edge, no cycle in a feedforward genome, `add_node` splits one edge, `delete_node` never touches input/output rows.
`test_crossover.py` — parent 1 keeps the structure, offspring keys ⊆ parent 1, aligned attributes are one of the two parents' (never a blend), non-homologous genes keep parent-1 attributes, padding preserved, shapes/dtypes, self-crossover bit-identity, `find_homologous` exact on the key tuple (including `historical_marker`) and `I_INF` for absent genes.
`test_distance.py` — configuration validation, zero at identity, symmetry, monotonicity in non-homologous genes, attribute weighting exact, one per changed int option, empty pair `0` not NaN, batched == pairwise, determinism.
`test_distance_digest.py` — on a seed-42 pop-100 XOR payload two `step`s deep, `execute_distance` (the pair kernel) is reproduced bit for bit (`tobytes()`, never `array_equal`) by the shipped single-centre digest, the batch-1 dense-centre fast path and `DefaultDistance(dense_single_centre=False)`'s row-wise merge; zero at identity and for an empty pair, two independently traced digests measuring identically, the digest's static contract plus `ConfigError` on a centre whose shapes disagree, `I_INF` in every key column of a padding row, and the traced-IR ceilings `MEASURE_OP_CEILING` 120 (marginal over `prepare`) / `PREPARE_OP_CEILING` 210.
`test_host_helpers.py` — `network_dict` structure/renumbering and single-individual refusal, `repr` exact rendering, `hash` in-graph, `sympy_func`'s 8-tuple plus one exact formula, `visualize` under `Agg` + `tmp_path` with a `networkx.draw` spy, and the recurrent `sympy_func`/`visualize` refusal being `TensorNEATError`.
`test_transform_validation.py` — `transform` rejects a per-individual node row, a `conn_keys` without a population axis and mismatched population sizes with `TensorNEATError`, and accepts both batched spellings.
`conftest.py` — the five lazily resolved modules (`genome`, `common`, `payloads`, `oracle`, `runner`), the three graph-glue fixtures (`run_kernel`, `run_transform`, `run_forward`) and the module-scoped `evoxc.Executable` cache they share.

## Conventions
Inputs are `scope="module"` fixtures resolved through `support.lazy.require`, so no module-scope `tensorneat`/`support` import; `pytest`, `numpy` and (in `test_payload.py`) `evoxc.random` are the only module-scope imports.
Every graph runs through `support.runner`'s fixtures; `runner.run_defn(closure, *args)` is the escape hatch for a kernel that takes no `state` (`g.hash`, `find_homologous`).
`run_kernel`/`run_transform`/`run_forward` reuse one compiled `evoxc.Executable` per `(genome identity, state identity, kernel or graph name, argument shapes and dtypes)`; a different genome/state object, a different static value or a different shape/dtype rebuilds.
The key keeps the genome and state objects alive (`_Identity`, never a bare `id()`) and keys frozen scalars by value, so an address is never recycled into a false hit and `pop_size=3` never reuses the `pop_size=4` graph.
`genome.grad` graphs are never cached — `runner.run_graph` retraces per call — and a call the conftest cannot classify (a mutable-container static, a callee that is not a bound method) rebuilds every time.
`genome.transform` is the only source of a `Transformed`: tests never name a schedule field, so a rename cannot silently pass.
A `Transformed` is passed field by field and rebuilt inside the traced closure — the shared runner treats only top-level array leaves as data, so a namedtuple argument would be baked as a static (see `conftest.py`).
A payload's `max_nodes`/`max_conns` must match the genome's statics: `output_slot`/`input_slot` come from the genome's key lists while the read-out takes the last `num_outputs` rows, so a capacity-comparison test builds one genome per capacity.
A node with no live incoming edge is *starved*, not contagious: the aggregation neutral-fills dead slots, so NaN spreads only by a consumer losing its every live source (this is what `test_recurrent.py`'s control chain pins).
Shapes stay small (`pop <= 4`, `max_nodes <= 13`) except in `test_distance_digest.py`, whose fixtures evolve a pop-100, `max_nodes` 50, `max_conns` 100 XOR population for two steps so the gene rows carry mutation-produced structure; the backend is always `"numpy"` and there is no `jax`/`torch`, no network, no GPU.
Randomness comes only from the repo-level `rng`/`evoxc_key` fixtures (seeded from the node id), never the global numpy state; a stochastic operator is pinned with probabilities of exactly `0.0`/`1.0`, not with a lucky draw.

## Constraints
A test may not edit `../../../src`, `../../../benchmarks` or `pyproject.toml`; a defect or a doc conflict found here goes into the report to the parent.
Whole node: 108 passed in ≈10 s on the numpy backend (`--durations=10`), at ≈225 `evoxc.build` calls per run, which is the floor: each forward/recurrent test legitimately builds a fresh genome+state and `grad` graphs are never cached.
`test_mutation.py` is ≈1000 lines for 15 invariant tests; it is at the concern threshold and should be split by operation family if it grows.
`test_host_helpers.py`'s `visualize` test asserts `matplotlib`/`networkx` are absent from `sys.modules` before the call, so it is order-sensitive within one process; pytest runs each module once, so this never bites in a normal run.
The guard in `test/unit/conftest.py` is the test-side counterpart of `test/release_gate_no_jax.py`: it freezes a session baseline at session start and records every import statement of `jax`/`jaxlib`/`torch`/`brax`/`gymnax`/`mujoco`, attributed to the file that executed it.
It therefore catches any import of an optional framework executed from `test/unit/**`, whether at module scope or inside a test body.
It excuses the imports originating in `src/` or in `pytest.importorskip`.
`pytest.mark.slow` is reserved for the oversized-population cases; the default gate is `pytest -m "not parity and not slow"`.

## Notes for Agents
`hash` is an **in-graph** kernel (no `state`, returns a symbolic `(P,) uint32`, traced via `runner.run_defn`), although `src/tensorneat/genome/CONTEXT.md` still lists it among the host-only helpers — trust the code.
`network_dict`/`repr`/`sympy_func`/`visualize` take `state` and ONE individual; `_individual` raises `ConfigError` on a `P > 1` batch, and `GenomeTensors` has no per-individual accessor, so a batch is sliced by rebuilding it: `GenomeTensors(*(array[i:i + 1] for array in payload))`.
`grad`'s loss should be written `residual * residual`: evoxc's `power` pullback emits `log(base)` regardless of exponent, so `residual ** 2` warns on negative residuals.
Two support-layer changes would be welcome and are out of this node's scope: `runner.run_built(executable, *args)` (the run-only entry point the executable cache needs, so the cache can move into `test/support/` and serve the other unit suites) and public `runner.is_data`/`runner.split_args` (the cache currently mirrors the private split so its key can never be coarser than the `TensorSpec` that reaches `build`).

## Routing Table
No subdirectories — this node is a leaf.
Read-only inputs: `../../../src/tensorneat/genome/` (code under test), `../../../src/tensorneat/common/tensorization.py` (`Transformed`), `../../support/` (payloads, oracle, runner).
Sibling suites: `../common`, `../algorithm`, `../problem`, `../../parity`, `../../integration`.
