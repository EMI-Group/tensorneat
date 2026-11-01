# ./unit/genome — payload, forward/grad and operator suites for the genome subtree

## Intent
Prove that each genome kernel computes what its docstring promises, on the reference numpy backend, with no optional dependency.
The assertions are about *invariants* and hand-written oracles, never about the implementation's own intermediate state: closed-form values on analytically chosen networks, `support.oracle` rollouts on the tedious topologies, and set/count algebra for the stochastic operators.
Every test that depends on source code that is still a design skeleton carries a real signature and docstring and a body of `pytest.skip("not implemented: <clause> (owner: <module>.<symbol>)")`, so the module is collectable and the skip text names the owner.

## Layout
`test_payload.py` — `BaseGenome.initialize` layout and padding, the `get_input_idx`/`get_output_idx` row invariant, `setup` chaining, constructor rejection of impossible shapes, `transform` accepted by `common.validate_schedule` and equal to `support.oracle.schedule`.
`test_forward.py` — `DefaultGenome.forward`: closed-form level composition, output rows unactivated, activation `-1` = identity, the two transforms, skipped/unconnected/padded rows, batch independence, random DAGs vs the oracle, the `max_depth` bound.
`test_recurrent.py` — `RecurrentGenome.forward`: Jacobi ticks, raw input re-injection without `input_transform`, tick-count recurrence, `NaN` poisoning, ring vs the oracle, `activate_time` validation.
`test_grad.py` — `BaseGenome.grad` as a graph factory: finite differences on both attribute tensors, closed-form bias-only network, `NaN`/padding gradients are zero, deterministic random-free graph, `sum(loss)` scalarization.
`test_mutation.py` — capacity (`max_nodes`/`max_conns`/`max_in_degree`), one structural change per individual, caller-allocated keys, untouched rows bit-identical, per-operation structural correctness.
`test_crossover.py` — parent 1 keeps the structure, per-attribute 50/50 where keys align, parent 1 attributes where they do not, padding, self-crossover identity, `find_homologous` exactness.
`test_distance.py` — zero at identity, symmetry, monotonicity in non-homologous genes and in `|delta|`, empty pair is `0` not `NaN`, batched equals pairwise, determinism.
`test_host_helpers.py` — `network_dict`/`repr`/`hash`/`sympy_func` (8-tuple + one exact formula)/`visualize` (lazy `matplotlib`, `tmp_path`, `Agg`), and the recurrent refusal.
`conftest.py` — the five lazily resolved modules (`genome`, `common`, `payloads`, `oracle`, `runner`) and the three graph-glue fixtures (`run_kernel`, `run_transform`, `run_forward`).

## Conventions
Nothing imports `tensorneat` or `support` at module scope: `import tensorneat` still raises `NotImplementedError` (a `State()` default argument in a design skeleton), so every input is a `scope="module"` fixture resolved through `support.lazy.require`.
Every graph runs through `support.runner` only, and through `conftest.py`'s fixtures rather than a module-scope import.
`genome.transform` is the only source of a `Transformed`: tests never name a schedule field, so a rename cannot silently pass.
A `Transformed` is passed field by field and rebuilt inside the traced closure — the shared runner treats only top-level array leaves as data, so a namedtuple argument would be baked as a static (see `conftest.py`).
Payloads come from `support.payloads` and expectations from `support.oracle`; neither is re-implemented here.
Shapes stay small (`pop <= 4`, `max_nodes <= 8`) and the backend is always `"numpy"`; no `jax`/`torch`, no network, no GPU, no display.

## Constraints
A test may not edit `../../../src`, `../../../benchmarks` or `pyproject.toml`; a defect or a doc conflict found here goes into the report to the parent.
Files stay under ~1000 lines and one module per source module, named `test_<module>.py`.
Suite output must be deterministic — every random draw goes through the repo-level `rng`/`evoxc_key` fixtures (seeded from the node id), never the global numpy state.
`pytest.mark.slow` is reserved for the oversized-population cases; the default gate is `pytest -m "not parity and not slow"`.

## Routing Table
No subdirectories — this node is a leaf.
Read-only inputs: `../../../src/tensorneat/genome/` (code under test), `../../../src/tensorneat/common/tensorization.py` (`Transformed`), `../../support/` (payloads, oracle, runner).
Sibling suites: `../common`, `../../parity`, `../../integration`.
