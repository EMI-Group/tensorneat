# ./test/parity — kernel equality against the frozen legacy JAX library

## Intent
Prove that the evoxc rewrite computes the *same numbers* as `../benchmarks/legacy_jax/tensorneat_legacy` (imported as `tensorneat_legacy`) wherever the legacy semantics are well defined.
Parity compares **kernels on identical inputs only**: one hand-built genome, one hand-built input batch, two implementations.
It never compares RNG draws, initial populations, mutations, crossovers or `genome.hash()` — those differ by construction (`evoxc.random` splitmix64 vs `jax.random`, float32 row hash vs the legacy encoding) and a test that compared them would be asserting on the wrong thing.
The legacy library is the oracle; this directory is the only place in the repo allowed to import `jax`.
All 60 tests here are real and green on the numpy backend: 18 `test_convert` + 13 `test_forward` + 5 `test_distance` + 13 `test_fitness` + 5 `test_grad` + 6 `test_oracle_integrity`; with the `bench` extra the directory must report **zero skips**.
`test_oracle_integrity.py` guards the evidence itself: that the oracle is genuinely separate code, that the four `importorskip` modules are actually collected when `jax` is installed, and that the directory's collected count never silently shrinks.

## How to run
- `pytest -m parity` from the repo root (needs the `bench` extra: `pip install -e ".[bench]"`); about 20 s, dominated by evoxc build/compile in `test_forward` and `test_grad`.
- `pytest test/parity -q` — the same suite by path; the four oracle modules call `pytest.importorskip("jax")` at module scope, so a bare environment skips rather than errors.
- A bare `uv sync` environment (no `bench` extra) reports **22 passed, 6 skipped**: the 18 `test_convert` tests plus the 4 always-run independence tests of `test_oracle_integrity.py` pass, the four oracle modules skip at import, and the two guards that need a live oracle (`test_the_four_oracle_modules_import`, `test_collected_parity_tests_are_ratcheted`) skip with a reason naming the four uncollected modules; exit status is 0 and nothing fails.
- `pytest test/parity/test_convert.py -q` — the only module that needs neither `jax` nor `tensorneat`: `convert.py` is plain host Python.
- `pytest test/parity/test_oracle_integrity.py -q` — the independence checks run in any environment; the two collection guards need `jax`.

## Layout
```
test/parity/
  __init__.py       package marker — see Constraints (pytest prepend-mode basename collisions)
  conftest.py       lazy legacy/new imports + the fixture set that builds one genome in both libraries
  convert.py        the two-way column-layout conversion and the max_in_degree / hidden-layer rules
  test_convert.py   real tests of convert.py (18)
  test_oracle_integrity.py  oracle independence + collection guard + collected-count ratchet (6)
  test_forward.py   forward values + NaN pattern: chain, skipped node, isolated node, 5 random DAGs, 3-individual batch, matching output keys, input/output transform, lossless-schedule guard (13)
  test_distance.py  execute_distance: exact 0, 0.125, attribute-only 0.4/0.45, batch-vs-pairwise, empty pair (5)
  test_fitness.py   problem.XOR / XOR3d fitness, all four error_method values, mape denominator, the NaN divergence (13)
  test_grad.py      gradient values against jax.grad of the same loss, single-option ACT/AGG (5)
```

## What each oracle module pins
- `test_forward.py` — values within `rtol=1e-5, atol=1e-6` **and** identical NaN positions, on payloads one edge set wide of the legacy dense column; the skipped-node perturbation leaves both read-outs bit-identical; the 3-individual batch proves the population axis is a pure batch axis (`P` rows in one call == `P` legacy calls); `pair.new_genome.get_output_idx() == pair.legacy_genome.get_output_idx() == list(range(max_nodes - num_outputs, max_nodes))` on the default layout; `input_transform`/`output_transform` are shown to be applied (they match the legacy run *and* differ from / relate to the untransformed one).
- `test_oracle_integrity.py` — the differential evidence itself: the frozen tree `../../benchmarks/legacy_jax/tensorneat_legacy` exists with `__init__.py` and at least 20 `.py` files; an AST scan of every `.py` there finds real `import jax` / `from jax… import` statements (jax is never imported to prove it); no file in that tree imports the top name `tensorneat` and no `benchmarks/legacy_jax/tensorneat` directory shadows it; `importlib.util.find_spec("tensorneat")` resolves under `../../src/tensorneat` and outside the oracle tree; each of the four `importorskip` modules imports cleanly while `jax` is installed (a module-scope skip then fails naming the module and reason); and a `pytest --collect-only` subprocess over this directory reports exactly the documented number of collected tests.
- `test_distance.py` — identical payloads give exactly `0` with `==` (what speciation depends on); one extra connection gives `0.125`; doubled weights give `0.4`; a node-attribute shift adds `1.0*0.4/8 = 0.05`; the four-value batch is pairwise distinct so a cross-population reduction cannot pass; two empty genomes give `0`, not `NaN`.
- `test_fitness.py` — `mse`/`rmse`/`mae`/`mape` on XOR and XOR3d match the legacy problem layer; `mape` divides by the target so a zero target is `-inf` on both sides (asserted with `==`); the rewrite's NaN → `-inf` mapping is pinned as the one deliberate divergence.
- `test_grad.py` — the differentiated loss matches at forward tolerance; `sum(conn_grad, axis=0)[:, 0]` matches the legacy weight column and `sum(node_grad, axis=0)[:, :2]` the legacy bias/response columns (`rtol=1e-4, atol=1e-5`); integer id columns and the legacy key column are exactly `0`; absent rows carry exactly `0`, never `NaN`; the two edges into a bypassed node carry `0`.

## The conversion contract (`convert.py`, the column-layout spec)
Legacy payload — one float row per gene, absent row = all-`NaN` (`benchmarks/legacy_jax/tensorneat_legacy/genome/gene/base.py`):
`nodes (N, 1 + NA)` with column `0` = node key and columns `1:` = `(bias, response, aggregation, activation)`; `conns (E, CK + CA)` with columns `:CK` = `(input_index, output_index[, historical_marker])` and the rest = `(weight,)`.
Rewrite payload — keys and attributes split (`src/tensorneat/genome/gene/base.py`): `node_keys (P, N) int32`, `node_attrs (P, N, NA) float32`, `conn_keys (P, E, CK) int32`, `conn_attrs (P, E, CA) float32`, absent keys = `I_INF = 2147483647`, absent attributes = `NaN`.
`to_legacy(payload) -> (nodes, conns)` translates the key column(s) into integer-valued `float32` (`I_INF -> NaN`) and copies attribute columns verbatim, `NaN` included — a `NaN` attribute is *data* (the legacy NaN poisoning of an unconnected node), not padding to clean up.
`from_legacy(nodes, conns, *, conn_key_width=2, payload_factory=None) -> Payload` is the exact inverse (`NaN -> I_INF`, one individual, `pop_size == 1`); `conn_key_width=3` names `OriginConn`'s `historical_marker`, since the legacy matrix stores no width metadata.
Both directions reject what they cannot interpret (`TypeError`/`ValueError`): a batch where one individual is expected, a non-integer or out-of-range key, a partially absent legacy row, a width that leaves no attribute column.
`I_INF` and `KEY_MAX` are re-declared in `convert.py` on purpose: importing `tensorneat` there would make collection depend on the package, and `test_convert.py` asserts the duplicates against the package constants.

## The `max_in_degree` rule (non-negotiable)
The legacy forward aggregates over a full `max_nodes`-wide column (`u_conns[:, i]` of a dense `(max_nodes, max_nodes)` matrix) and truncates nothing.
The rewrite keeps a `(P, max_nodes, K)` incoming window and reports what does not fit in `Transformed.edge_overflow`.
So a parity genome is built with `max_in_degree = convert.max_in_degree_for(max_nodes) == max_nodes`, and every comparison first asserts `edge_overflow == 0` (the `new_schedule` fixture does it in the one place a schedule is produced).
A payload that needs a wider window is a bug in the payload, never a reason for a looser tolerance: a test that silently drops an edge must fail, not drift.
`genome_pair.build(..., max_in_degree=...)` exists only for the constructive half of `test_forward`'s guard, which shows that a narrower window reports `edge_overflow > 0`; such a pair is never compared value by value.
The narrowest legal window there is 5, not 7: `BaseGenome.__init__` raises `TensorNEATError` below the dense initial in-degree (the 5-wide hidden layer feeding the output), so the lossy pair uses `max_in_degree=5` with a 7-way fan-in into the output row and asserts `edge_overflow == 2`.

## The hidden-layer rule (why the two genomes are the same network)
The legacy `output_idx` is the last dense layer of `init_hidden_layers`; the rewrite pins output keys to rows `max_nodes - num_outputs .. max_nodes - 1`.
`convert.legacy_init_hidden_layers(max_nodes, num_inputs, num_outputs)` returns `(max_nodes - num_inputs - num_outputs,)`, the one width that makes the two agree, and `convert.legacy_initial_edge_count(...)` the dense edge count the legacy constructor requires `max_conns` to hold (15 for 8/2/1, 16 for 8/3/1).
Verified against the real legacy code: with that layer list the legacy forward on `chain_payload` reproduces the hand-computed sigmoid chain (`0.65958464` for inputs `[1, 1]`), and `get_output_idx() == [7]`.

## Fixtures (`conftest.py`) — the API the test modules use
`legacy` (module-scoped) — `pytest.importorskip("jax")` then `pytest.importorskip("tensorneat_legacy")`, both *inside the fixture body*, so a numpy-only environment reports a named skip instead of a collection error.
`new` — `support.lazy.require("tensorneat")`, which converts a pending-implementation `NotImplementedError` into a skip and lets an `ImportError` surface.
`payloads`, `runner` — the `support.payloads` / `support.runner` modules through the same `require`; `runner` is the only path to `evoxc.build`/`evoxc.run` in this directory.
`convert` — this directory's `convert.py`.
`genome_pair` — `build(num_inputs=2, num_outputs=1, max_nodes=8, max_conns=16, *, max_in_degree=None, activation="sigmoid", aggregation="sum", input_transform=None, output_transform=None) -> GenomePair`, resolving ACT/AGG **by name** against each library's registry, as one-element lists, and the sizes from the rules above (`max_in_degree` defaults to `max_nodes`).
`new_schedule` — `run(pair, payload) -> Transformed` via the public `genome.transform`, asserting `edge_overflow == 0` first.
`new_forward` / `legacy_forward` — the same payload + input batch through each `forward`; the rewrite is batch-first `(P, num_inputs) -> (P, num_outputs)`, the legacy one takes a single `(num_inputs,)` row because a legacy genome is one individual; `legacy_transformed` is exposed separately for the modules that need the legacy schedule itself.
`tile_payload` — `tile(payload, repeat)`, `np.repeat` along axis 0 (individual-major: row `p * repeat + s` is individual `p`'s sample `s`), the layout `FuncFit.evaluate` documents.
`new_distance` / `legacy_distance` — the rewrite takes aligned `P`-batches, the legacy operator one parent pair.
`new_fitness` / `legacy_fitness` — the rewrite returns `(P,)`, the legacy problem one scalar.
`new_fitness` supplies the `act_fn` contract `FuncFit.evaluate` documents: it reads `P` from `static_batch_size(transformed)`, tiles the dataset to `(P*S, I)`, and the fixture's `act_fn` reshapes that back to `(P, S, I)` and calls `genome.forward` once per sample (a static Python loop, no `vmap`) before stacking to `(P*S, O)`.
`new_grad` — returns `(loss, (node_grad, conn_grad))`: `genome.grad` is a graph factory whose inputs are the five tensors `(node_keys, node_attrs, conn_keys, conn_attrs, inputs)` with the schedule specialised as a constant, and `evoxc.grad` yields gradients only, so the loss is read from a forward pass of the same payload and `loss_fn`.
`legacy_grad` — `(loss, (grad_nodes, grad_conns))` from `jax.value_and_grad` over one shared gene matrix, hence the axis-0 sum rule in `test_grad`.
Legacy arrays are wrapped with `jnp.asarray` inside the fixtures: under JAX 0.12 the legacy forward indexes with a traced `lax.fori_loop` counter and raises `TracerArrayConversionError` on numpy input.
`Transformed` / `GenomeTensors` namedtuples are passed **field by field** and rebuilt inside the traced closure, because the shared runner classifies only top-level numpy leaves as data.

## Known coverage gaps (what parity cannot detect)
Every usable payload has edges that point from a lower row/key to a higher one, so topological order coincides with row order and a schedule that used the row index as the level would pass every value comparison; no acyclic descending edge (`source key > target key`) is ever compared (chain/skip/isolated in `../../support/payloads.py`, and `./test_forward.py:233-247` draws ascending targets only; `ring_payload`'s `3->2` is cyclic and used for the overflow counter alone).
`./test_fitness.py` only scores one-individual payloads (`P == 1`), so `FuncFit.evaluate`'s individual-major `(P * S, I)` tiling order is unobservable there — a transposed tiling would pass.
Only `activation="sigmoid"` / `aggregation="sum"` are ever configured (`./conftest.py:86-87`), i.e. 1 of 12 ACTs and 1 of 6 AGGs, and `mape` is only ever compared as `-inf` because both datasets contain zero targets (`./test_fitness.py:86-89`), so a wrong finite `mape` magnitude has no oracle.
`_assert_matches` compares values only where both sides are finite (`./test_forward.py:81-82`) and the five random-DAG cases never assert finiteness (`./test_forward.py:258`), so a case whose output row is unreachable compares NaN patterns only.
Never parity-compared at all: `initialize`, mutation, crossover, `NEAT.tell`/speciation, the `Pipeline`, `RecurrentGenome`, HyperNEAT, the three RL problems, `network_dict`/`sympy_func`/`visualize`; `execute_distance` is the only genetic-operator kernel with an oracle.
Parity runs on the numpy interpreter only (`../../support/runner.py` `DEFAULT_BACKEND`); a compiled backend's agreement with the oracle is inferred from its agreement with numpy (`../backends`).
The four oracle modules still skip at module scope when `jax` is absent (`./test_forward.py:50`, `./test_distance.py:57`, `./test_fitness.py:49`, `./test_grad.py:58`), so that environment compares no kernel; `./test_oracle_integrity.py` makes that state explicit (a skip reason naming the four uncollected modules) instead of silent, and fails when a module skips although `jax` is installed.

## Tolerances
`forward` values — `np.allclose(rtol=1e-5, atol=1e-6)` on `float32`, **plus** `np.isnan(new) == np.isnan(old)` elementwise: `NaN` is data here, so a matching `NaN` is as binding as a matching number.
`execute_distance` — `atol=1e-6` against the legacy kernel and against hand-computed closed forms; the identical-genome and empty-pair cases are asserted with exact `==` (`0` is the value speciation depends on).
Problem fitness — `np.isclose(rtol=1e-5, atol=1e-6)`; non-finite fitness is compared with `==`.
Gradients — `rtol=1e-4, atol=1e-5` (a gradient carries a few more ulp than a forward value); the differentiated loss itself uses the forward tolerance.
Tolerances are constants at the top of each module and must not be widened to hide a mismatch; a genuine disagreement is an `xfail(strict=True, reason="src defect: ...")` plus a report, never a looser assertion.

## Known divergences and shared rules encoded here
`genome.hash()` is never compared.
Activation id `-1` (identity in the rewrite, clamped to the last option by the legacy `lax.switch`) is excluded: parity genomes use option `0` of a one-element list and no attribute row is written with `-1`.
All configured ACT/AGG options are evaluated and selected, so a non-total option could poison a branch it was not selected into: every parity genome configures exactly one option on each side.
NaN fitness is the one *deliberate* divergence — `problem.base.finalize_fitness` maps `NaN -> -inf`, the legacy `FuncFit.evaluate` leaves the `NaN` for the algorithm to sanitise — and `test_fitness.py` pins it instead of asserting equality.
A `NaN` **source slot is dropped** from an aggregation when the destination also has a live source, on both sides (`live = ~isnan(values)` in the rewrite, `src_valid` in the legacy forward), so an unconnected node poisons a read-out only when nothing else feeds it; `test_forward` and `test_fitness` assert both halves of that rule rather than the naive "any `NaN` edge poisons".
`evoxc`'s power rule is `y*b/a`, so differentiating `x ** 2` at `x == 0` yields `NaN` where `jax` yields `0` (`evoxc.grad(sum(x**2))(x=[0., 3.]) -> [nan, 6.]`): `test_grad`'s padding case gives the output row `bias=0.5` so the loss is differentiated at a differentiable point; this is a foreign-library AD corner, not a `tensorneat` divergence.
RNG, `initialize`, mutation and crossover are never compared, at any tolerance.
`CartPole` has no parity oracle (the legacy RL problems are gone); `../unit/problem`'s numpy reference is its only oracle.

## Conventions
`pytestmark = [pytest.mark.parity]` at module scope in every module here, so `-m parity` selects the directory and `-m "not parity"` excludes it.
The four oracle modules call `pytest.importorskip("jax", reason=...)` at module scope immediately after `pytestmark`; `jax` is otherwise imported only inside fixture/test bodies, always after that guard.
Every test here is a real assertion: no `pytest.skip` body remains, and a module that unexpectedly skips is a defect in the module.
Docstrings state invariant / oracle / assertion for each test and are kept free of history wording.
No network, no GPU, `backend="numpy"` everywhere (`support.runner`'s default).
Shapes stay small (`max_nodes == 8`, `max_conns == 16`, `pop <= 8`).
Only `test_forward`'s random-DAG cases draw random numbers, from the `rng` fixture seeded by test id, so the suite is deterministic.

## Constraints
`pytest`'s prepend import mode names a test module by its bare basename, so `test_forward.py`/`test_distance.py`/`test_grad.py` here collide with `../unit/genome/'s` modules of the same names and abort the whole session with `import file mismatch`; `__init__.py` makes pytest resolve this directory through its package path instead, and must not be deleted.
`../src/tensorneat` and `../benchmarks/legacy_jax` are read-only inputs: a needed change there goes into the report to the parent, never into this directory's diff.
`./convert.py` must stay plain Python — no `evoxc`, no `jax`, no tracing — so test collection never depends on either library.
`../pyproject.toml` (markers, `testpaths`) is read-only; `parity` is already declared there.
Suite output is deterministic: the only randomness is the `rng` fixture, seeded from the test id.
Files stay under ~1000 lines.

## Routing Table
| Path | Purpose |
|---|---|
| `./conftest.py` | Lazy legacy/new imports and the fixture set that builds one genome in both libraries. |
| `./convert.py` | Column-layout conversion, `max_in_degree` and hidden-layer rules (plain Python, no libraries). |
| `./test_convert.py` | Real tests of `convert.py`: layouts, round trip, rejections, layout rules (18). |
| `./test_forward.py` | `forward` parity: values + NaN pattern on chains, skipped/isolated nodes, random DAGs, transforms, the lossless-schedule guard (12). |
| `./test_distance.py` | `execute_distance` parity: exact `0`, `0.125`, `0.4`/`0.45`, batch-vs-pairwise, empty pair (5). |
| `./test_fitness.py` | `problem.XOR`/`XOR3d` parity for the four `error_method` values + the NaN divergence (13). |
| `./test_grad.py` | `grad` parity against `jax.grad` of the same loss, single-option ACT/AGG (5). |
| `../unit` | numpy-backend kernel tests (the correctness oracle for the rewrite when no legacy oracle exists). |
| `../support` | `runner` (the only `evoxc` entry point), `payloads` (hand-built genomes), `lazy.require`. |
| `../../benchmarks/legacy_jax` | The frozen legacy implementation (sibling node) — read-only oracle for this directory; a change there is escalated to the parent. |
