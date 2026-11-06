# ./unit — numpy-backend verifier suite for the evoxc kernels

## Intent
Prove that every evoxc kernel in `../../src/tensorneat` computes what its docstring promises, on the reference numpy interpreter, with no optional framework imported by the tests themselves.
Each assertion compares against something independent of the implementation under test: a closed-form value, a plain-numpy oracle, a hand-built payload, or a structural invariant (`sum(spawn) == pop_size`, "no NaN survives `tell`", a hand-derived substrate geometry).
This directory is the whole content of the default gate; `./parity` (legacy JAX oracle) and `./integration` (whole `Pipeline`) are the two suites that go beyond it.

## Counts
`./common` 272, `./genome` 172, `./algorithm` 389, `./problem` 367, `./benchmarks` 82 — 1282 passing cases in the subdirectories, plus four modules at this node's root (`./test_host_payload.py` 12, `./test_pipeline_device.py` 13, `./test_ir_ops.py` 4, `./test_version.py` 2 = 31): **1313 passed, 0 failed, 0 skipped**.
`pytest -m "not parity and not slow"` from the repo root: **1376 passed, 121 deselected** in ~54 s after `uv sync --all-extras`, and **1371 passed, 6 skipped, 85 deselected** in ~37 s in a bare `uv sync` (the six skips are the four `jax` parity modules and the two `../backends` modules; `--ignore` is no longer needed, see `../backends/conftest.py`).
`./algorithm` carries no `xfail` (the HyperNEAT read-out test passes); nothing in this node is skipped.
`./algorithm` and `./genome` dominate the wall clock (one traced executable per graph), then `./common`, `./problem` and `./benchmarks`.

## How to run
`uv run pytest test/unit -o addopts="" -q` from the repo root (plain `pytest` also collects it, but the doubled `-q` from `addopts` hides the summary line) — numpy backend, no optional dependency, **1313 passed in ~41 s** after `uv sync --all-extras` (~36 s in a bare `uv sync`).
`pytest test/unit/<sub> -k <expr>` narrows the same gate while keeping the leak guard active.
Per node: `./algorithm` 13 s, `./genome` 11 s, `./common` 5 s, `./problem` 5 s, `./benchmarks` 0.4 s and the 31 root-level cases 1.3 s; no module here needs a network, a GPU or parallelism.

## Layout
`./common` (10 test modules) — schedule (`tensorization`) derivation/validation and its seams (`acyclic`, `trusted`, every refusal), the accelerator formulation of that same schedule and its resolver (`kernel_mode`), flat gather/scatter/mutation helpers, ACT/AGG dispatch, host graph walks, `sympy_tools` code generation (`to_python_code`/`to_latex_code` for all 12 ACT and 6 AGG ids), `State` including its copy/deepcopy/pickle round-trips.
`./genome` (14 test modules) — gene payloads, forward/recurrent/grad kernels, the accelerator spellings of the forward level sweep, `closes_cycle` and `homologous_pair`, the level-order-vs-row-order relabel proof, mutation/crossover/distance invariants, host-only introspection.
`./algorithm` (14 test modules + the `hyperneat_kernel.py` trace harness) — NEAT `tell` algebra (speciation, stagnation, spawn allocation, parent selection), the dense-vs-gated species-distance equivalence, the accelerator speciation gate and the three device-gated distance statics, and the whole HyperNEAT subtree: substrate geometry for all four substrates, `HyperNEATNode`/`HyperNEATConn`, the CPPN→weight map, `HyperNEAT`/`HyperNEATFeedForward`, and a pop-4 two-generation end-to-end run.
`./problem` (8 modules) — func-fit losses (`mse`/`rmse`/`mae`/`mape`), dataset budget and `CustomFuncFit` bounds, the `problem` protocol (`build_options`, backend-honouring `show`), the RL knob validation matrix (`require_finite` and friends), and all three RL environments: CartPole, plus Pendulum and MountainCar with an in-file host-numpy oracle each and hand-derived one-step values at non-zero state.
`./benchmarks` (6 modules) — the harness fingerprint (`git_report`, porcelain parsing), plus the protocol the benchmarks owner recorded: `aggregate` target/repeat bookkeeping, `report_format.reach` renderings, `speedup_bounds` best/worst, the case table's `gens_to_target` flags, and `--persist-note` on a `tmp_path` copy.
`./ir_ops.py` — the one traced-IR walker every op-count assertion in this node imports: `count_ops(block)`, `count_graph_ops(graph)`, `op_names(graph, *, include_terminators=False)` and `TERMINATOR = "return"`, descending `op.regions` → `region.blocks` → `block.ops` and counting no terminator unless asked; the route proofs and op budgets of `./common/test_schedule_accelerator.py`, `./genome/test_accelerator_equivalence.py`, `./genome/test_forward_accelerator.py`, `./genome/test_distance_digest.py` and `./algorithm/test_scan_accelerator.py` all come out of it.
`./test_ir_ops.py` — the walker's own 4 cases on hand-written graphs: a region-free block counted op by op, a nested `while_loop` descended two levels, the `region.blocks`-not-`region.ops` descent (a walk that hands a `Region` back raises `AttributeError`, the regression guarded here), and terminator accounting per reached block.
`./test_pipeline_device.py` — the `Pipeline(device=...)` placement layer, all 13 cases: `_placement_device`'s resolver spellings, `_host_build_options` stripping exactly the device-targeting keys, `setup()` handing the host phase stripped options and restoring the caller's on the state, `_place_state`/`_host_view` on `cpu` and a registered fake kind (cross-kind upload through host memory, a rank-0 `randkey` kept rank-0, configuration leaves left on the host), `to_numpy` as the one explicit D2H step, and `device="cpu"` bit-identical to `device=None` with one executable per shape.
`./test_version.py` — the package's `__version__` metadata and its `PackageNotFoundError` fallback.
`./test_host_payload.py` — the `host_payload._one_individual` contract `Pipeline.show` depends on: the one-individual batch passes through, the bare per-individual rows are promoted to it, a `P > 1` payload and a missing (`None`) payload are `common.TensorNEATError` naming both accepted spellings, and every pre-existing malformed-input message (arity, rank-0, impossible rank, population-axis disagreement) is unchanged.
Every directory here carries `__init__.py` (pytest's prepend import mode names a module by basename, so un-namespaced directories collide with `./parity`/`./integration`).

## Oracles
`../support/oracle.py` holds the numpy reference implementations the suites compare against: the schedule derivation, a NaN-aware forward pass, the four func-fit losses, the CartPole Euler rollout.
`./problem/test_pendulum.py` and `./problem/test_mountaincar.py` keep their own float32 op-for-op and float64 Gym-semantics rollouts in-file, plus a hand-written `_one_step_by_hand` that spells the published transition out in numeric literals at `theta != 0`, so a coefficient typo moves the kernel off a number that never touched the implementation.
`../support/functions_ref.py` holds the 12 ACT and 6 AGG reference kernels the dispatch tests loop over; `./common/test_sympy_tools.py` additionally `exec`s the generated NumPy against the real graph for every option.
`../support/payloads.py` builds every `int32`/`float32` gene payload by hand (`node_keys`/`node_attrs`/`conn_keys`/`conn_attrs`, `I_INF`/`NaN` padding) plus the canonical small genomes, so no test draws a topology it could have written.
`./genome/test_level_order.py` builds an isomorphic relabel whose edges point from a higher row to a lower one, so level-order execution is distinguished from row order.
`./algorithm/hyperneat_kernel.py` is the HyperNEAT analogue of the shared runner: one helper that traces a single host method (`run_method`) and rebuilds `Transformed`/`SubstrateSchedule`/`GenomeTensors` inside the trace; the substrate oracles are literal arrays hand-computed from the documented row order.
A test that needs none of those compares against a closed form written in the module itself (`np.where(fitness, -inf)`, the sigmoid chain, the species-count algebra).
Kernels are built only through `../support/runner.py` (`run_defn`/`run_graph`/`to_numpy`), and `./genome/conftest.py` caches one `evoxc.Executable` per graph and replays it with `evoxc.run` because tracing costs about ten times a run.

## The optional-framework guard (`conftest.py`)
`pytest_sessionstart` freezes the session baseline — the forbidden modules already in `sys.modules` — and a session-scoped autouse fixture does it when the conftest is loaded only at collection time; the freeze is idempotent, so it happens exactly once.
A session-wide recorder wraps `builtins.__import__` and notes every import *statement* whose root module is in `("jax", "jaxlib", "torch", "brax", "gymnax", "mujoco")` together with the file that executed it.
The autouse per-test fixture fails a test whose body or module let a forbidden module in, reporting the offending module and the importing file.
The recorder is what gives the guard teeth: it attributes a forbidden import to the file that executed it, including an import performed by a module under test during a test body.
It also supplies the allowances the guard needs: an import executed from `src/`, from `pytest.importorskip`, or from another suite (parity imports `jax` while it is collected, so `pytest test/unit test/parity` must not fail here) is not this suite's leak; one executed from a file below this directory is.
Each leak is reported once and forgotten, so one bad import does not turn every later test red.
Not covered: a dynamic `importlib.import_module` of an already-loaded framework, and anything a subprocess does — that is `../release_gate_no_jax.py`'s job, which checks the package itself in a clean interpreter.

## Known items
No `xfail` left in this node: `./algorithm/test_hyperneat.py::test_forward_reads_the_substrate_output_neurons` now passes, because the read-out follows `Transformed.output_slot` (the substrate's output keys `I .. I+O-1`) instead of the last `num_outputs` node rows; the test pins both the slot and the hand-computed value.
The source fix that made it pass is `src/`'s, not this suite's: the test was never weakened, only un-marked once the behaviour arrived.
The whole-node run emits 2 `RuntimeWarning: overflow encountered in exp` from the HyperNEAT end-to-end runs' `output_transform=ACT.sigmoid` on saturating weights; the suite deliberately does not silence them.
Declared size exceptions (each one cohesive concern whose module-local helpers would be duplicated by a split): `./genome/test_forward.py` 1136, `./genome/test_grad.py` 1063, `./genome/test_mutation.py` 1050 — split trigger 1200 lines; `./algorithm/test_substrate.py`, `test_genes.py`, `test_hyperneat.py` and `test_hyperneat_feedforward.py` sit at 983–1000 and split if they grow.
The device-gated accelerator formulations are covered on the numpy leg only, each against the host leg and each route-checked on the traced IR rather than on equal numbers: the schedule formulation and its resolver (`./common/test_schedule_accelerator.py`), `closes_cycle` and `homologous_pair` (`./genome/test_accelerator_equivalence.py`), the forward level sweep (`./genome/test_forward_accelerator.py`), the speciate gate's zero-`cond` contract (`./algorithm/test_scan_accelerator.py`) and the three distance statics (`./algorithm/test_distance_accelerator.py`).
No accelerator formulation has a compiled-backend case in this node: `xla`, `tvm` and a real GPU are untouched, so the accelerator work is numpy-verified only — the scatter-free and zero-`cond` arguments are exactly the ones a compiled leg would have to confirm.
`distance_formulation="index_free"` has no whole-`tell` case here: it is compared at the `(S, P)` matrix level only.
Remaining gaps, per node: `RecurrentMutation` and the `OriginNode`/`BiasNode` host hooks (`./genome`), `pipeline.py` (its device layer is `./test_pipeline_device.py`, its generations still only `./integration`), the substrate `_attr_columns` rejections that evoxc's static allowlist makes unreachable, and the untested `--list`/`--dry-run` CLI paths (`./benchmarks`); the child `CONTEXT.md` files carry the detail.

## Conventions
No test body is a placeholder: every test either asserts on a real kernel or, for a genuine source defect, is a strict `xfail(strict=True, reason="src defect: …")`.
One module per source module, named `test_<module>.py`; the directory tree mirrors `../../src/tensorneat/`, and a concern that would push a module past ~1000 lines gets its own focused module (`test_mutation_weight.py`, `test_level_order.py`, `test_tensorization_seams.py`, `test_distance_dense.py`, `test_sympy_tools.py`).
Shapes stay small (pop ≤ 16, `max_nodes` ≤ 16) except where a test exists specifically to trip a static bound or to run an end-to-end generation (pop 4–8).
Backend is always `"numpy"`; a test that needs `evoxc[iree]` is marked `requires_iree` and belongs to no gate.
An assertion must be able to fail and names the independent oracle it disagrees with, so a red test says which side of the comparison is wrong.
`jax` is imported only inside `./parity` modules; the guard above enforces it here.

## Constraints
No network, no GPU, no wall-clock or ordering dependence; `tmp_path` appears only in `./common/test_state.py` (the `State` save/load round-trip), `./genome/test_host_helpers.py` (the `visualize` rendering), `./benchmarks` (a temporary git repository and a `tmp_path` copy of a results record) and the e2e HyperNEAT/pipeline runs.
A test may not edit `../../src`, `../../benchmarks` or `pyproject.toml`; a discovered defect is reported upward (pinned here only as a strict `xfail` while it stands).
`../conftest.py` puts `src/`, `benchmarks/legacy_jax` and `test/` on `sys.path`; `./benchmarks/conftest.py` additionally appends `benchmarks/` so the harness's bare `harness` sibling import resolves; `conftest.py` here has one process-wide side effect, the import recorder, and it uses nothing outside the standard library and pytest.

## Routing Table
| Path | Purpose |
|---|---|
| `./common` | tensorization schedule derivation, its accelerator formulation and resolver, its seams (`acyclic`/`trusted`/refusals) and validation, flat gather/scatter/mutation helpers, ACT/AGG dispatch, host graph algorithms, `sympy_tools` code generation for every option, `State`. |
| `./genome` | payload layout, forward/recurrent/grad kernels, the accelerator forward sweep and the two accelerator kernel spellings, the level-order relabel proof, mutation/crossover/distance invariants, host introspection, `hash`. |
| `./algorithm` | NEAT `tell` algebra, the dense-vs-gated species-distance equivalence, the accelerator speciation gate and the device-gated distance statics, and the whole HyperNEAT subtree including a pop-4 end-to-end run. |
| `./problem` | func-fit losses and dataset budget, `CustomFuncFit` bounds, the `problem` protocol, the RL knob matrix, and the CartPole / Pendulum / MountainCar rollouts against host-numpy and hand-derived oracles. |
| `./benchmarks` | harness fingerprint plus the reporting protocol: `aggregate`, `report_format.reach`, `speedup_bounds`, the case table and `--persist-note`. |
| `./ir_ops.py` | the one traced-IR op walker (`count_ops`, `count_graph_ops`, `op_names`, `TERMINATOR`) every route proof and op budget in this node counts with — not collected. |
| `./test_ir_ops.py` | the walker pinned on hand-written graphs: region-free counting, a nested `while` body, `region.blocks`-not-`region.ops`, terminator accounting. |
| `./test_pipeline_device.py` | `Pipeline(device=...)`: resolver spellings, stripped host build options, `_place_state`/`_host_view`, the `to_numpy` D2H step, and cpu-vs-default bit-identity. |
| `./test_version.py` | `tensorneat.__version__` against the distribution metadata, plus the `PackageNotFoundError` fallback. |
| `./test_host_payload.py` | `host_payload._one_individual`: one-individual batch vs bare rows, the `P > 1` and `None` rejections naming both spellings, and the unchanged malformed-input messages. |
| `./conftest.py` | session baseline + import recorder: the per-test optional-framework guard. |
| `../support` | shared runner, hand-built payloads and numpy oracles used by every suite here. |
| `../release_gate_no_jax.py` | import-hygiene gate on the package in a clean subprocess (test root, not this node). |
