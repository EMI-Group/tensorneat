# ./test/parity — kernel equality against the frozen legacy JAX library

## Intent
Prove that the evoxc rewrite computes the *same numbers* as `../benchmarks/legacy_jax/tensorneat_legacy` (imported as `tensorneat_legacy`) wherever the legacy semantics are well defined.
Parity compares **kernels on identical inputs only**: one hand-built genome, one hand-built input batch, two implementations.
It never compares RNG draws, initial populations, mutations, crossovers or `genome.hash()` — those differ by construction (`evoxc.random` splitmix64 vs `jax.random`, float32 row hash vs the legacy encoding) and a test that compared them would be asserting on the wrong thing.
The legacy library is the oracle; this directory is the only place in the repo allowed to import `jax`.

## How to run
- `pytest -m parity` from the repo root (needs the `bench` extra: `pip install -e ".[bench]"`).
- `pytest test/parity -q` — the same suite by path; `jax`/`tensorneat_legacy` are skipped-not-errored when absent, and the stub bodies report `not implemented: <owner>`.
- `pytest test/parity/test_convert.py -q` — the only module that runs green *today*: `convert.py` is plain host Python and does not need either library.

## Layout
```
test/parity/
  __init__.py       package marker — see Constraints (pytest prepend-mode basename collisions)
  conftest.py       lazy legacy/new imports + the fixture set that builds one genome in both libraries
  convert.py        the two-way column-layout conversion and the max_in_degree / hidden-layer rules
  test_convert.py   real tests of convert.py (the only non-stub module here)
  test_forward.py   forward values + NaN pattern, dense chain / skipped / isolated / random DAGs / transforms
  test_distance.py  execute_distance: identical (0), one extra connection (0.125), attribute only (0.4), batch
  test_fitness.py   problem.XOR / XOR3d fitness, all four error_method values, per problem
  test_grad.py      gradient values against jax.grad of the same loss, single-option ACT/AGG
```

## The conversion contract (`convert.py`, the column-layout spec)
Legacy payload — one float row per gene, absent row = all-`NaN` (`benchmarks/legacy_jax/tensorneat_legacy/genome/gene/base.py`):
`nodes (N, 1 + NA)` with column `0` = node key and columns `1:` = `(bias, response, aggregation, activation)`; `conns (E, CK + CA)` with columns `:CK` = `(input_index, output_index[, historical_marker])` and the rest = `(weight,)`.
Rewrite payload — keys and attributes split (`src/tensorneat/genome/gene/base.py`): `node_keys (P, N) int32`, `node_attrs (P, N, NA) float32`, `conn_keys (P, E, CK) int32`, `conn_attrs (P, E, CA) float32`, absent keys = `I_INF = 2147483647`, absent attributes = `NaN`.
`to_legacy(payload) -> (nodes, conns)` translates the key column(s) into integer-valued `float32` (`I_INF -> NaN`) and copies attribute columns verbatim, `NaN` included — a `NaN` attribute is *data* (the legacy NaN poisoning of an unconnected node), not padding to clean up.
`from_legacy(nodes, conns, *, conn_key_width=2, payload_factory=None) -> Payload` is the exact inverse (`NaN -> I_INF`, one individual, `pop_size == 1`); `conn_key_width=3` names `OriginConn`'s `historical_marker`, since the legacy matrix stores no width metadata.
Both directions reject what they cannot interpret (`TypeError`/`ValueError`): a batch where one individual is expected, a non-integer or out-of-range key, a partially absent legacy row, a width that leaves no attribute column.
`I_INF` and `KEY_MAX` are re-declared in `convert.py` on purpose: importing `tensorneat` there would break collection while `src/` is a skeleton, and `test_convert.py` asserts the duplicates against the package constants as soon as it imports.

## The `max_in_degree` rule (non-negotiable)
The legacy forward aggregates over a full `max_nodes`-wide column (`u_conns[:, i]` of a dense `(max_nodes, max_nodes)` matrix) and truncates nothing.
The rewrite keeps a `(P, max_nodes, K)` incoming window and reports what does not fit in `Transformed.edge_overflow`.
So a parity genome is built with `max_in_degree = convert.max_in_degree_for(max_nodes) == max_nodes`, and every comparison first asserts `edge_overflow == 0` (the `new_schedule` fixture does it in the one place a schedule is produced).
A payload that needs a wider window is a bug in the payload, never a reason for a looser tolerance: a test that silently drops an edge must fail, not drift.

## The hidden-layer rule (why the two genomes are the same network)
The legacy `output_idx` is the last dense layer of `init_hidden_layers`; the rewrite pins output keys to rows `max_nodes - num_outputs .. max_nodes - 1`.
`convert.legacy_init_hidden_layers(max_nodes, num_inputs, num_outputs)` returns `(max_nodes - num_inputs - num_outputs,)`, the one width that makes the two agree, and `convert.legacy_initial_edge_count(...)` the dense edge count the legacy constructor requires `max_conns` to hold (15 for 8/2/1, 16 for 8/3/1).
Verified against the real legacy code: with that layer list the legacy forward on `chain_payload` reproduces the hand-computed sigmoid chain (`0.65958464` for inputs `[1, 1]`), and `get_output_idx() == [7]`.

## Fixtures (`conftest.py`) — the API the test modules use
`legacy` (module-scoped) — `pytest.importorskip("jax")` then `pytest.importorskip("tensorneat_legacy")`, both *inside the fixture body*, so a numpy-only environment reports a named skip instead of a collection error.
`new` — `support.lazy.require("tensorneat")`, which converts only the pending-implementation `NotImplementedError` into a skip; an `ImportError` still surfaces.
`payloads`, `runner` — the `support.payloads` / `support.runner` modules through the same `require`; `runner` is the only path to `evoxc` in this directory.
`convert` — this directory's `convert.py`.
`genome_pair` — `build(num_inputs=2, num_outputs=1, max_nodes=8, max_conns=16, *, activation="sigmoid", aggregation="sum", input_transform=None, output_transform=None) -> GenomePair`, resolving ACT/AGG **by name** against each library's registry, as one-element lists, and both sizes from the two rules above.
`new_schedule` — `run(pair, payload) -> Transformed` via the public `genome.transform`, asserting `edge_overflow == 0` first.
`new_forward` / `legacy_forward` — the same payload + input batch through each `forward`; `legacy_transformed` is exposed separately for the modules that need the legacy schedule itself.
`tile_payload` — `tile(payload, repeat)`, `np.repeat` along axis 0 (individual-major: row `p * repeat + s` is individual `p`'s sample `s`), the layout `FuncFit.evaluate` documents.
`new_distance` / `legacy_distance`, `new_fitness` / `legacy_fitness`, `new_grad` / `legacy_grad` — one fixture per kernel per side.
Legacy arrays are wrapped with `jnp.asarray` inside the fixtures: under JAX 0.12 the legacy forward indexes with a traced `lax.fori_loop` counter and raises `TracerArrayConversionError` on numpy input.
`Transformed` / `GenomeTensors` namedtuples are passed **field by field** and rebuilt inside the traced closure, because the shared runner classifies only top-level numpy leaves as data.

## Tolerances
`forward` values — `np.allclose(rtol=1e-5, atol=1e-6)` on `float32`, **plus** `np.isnan(new) == np.isnan(old)` elementwise: `NaN` is data here, so a matching `NaN` is as binding as a matching number.
`execute_distance` — `atol=1e-6` against the legacy kernel and against hand-computed closed forms; the identical-genome and empty-pair cases are asserted with exact `==` (`0` is the value speciation depends on).
Problem fitness — `np.isclose(rtol=1e-5, atol=1e-6)`; non-finite fitness is compared with `==`.
Gradients — `rtol=1e-4, atol=1e-5` (a gradient carries a few more ulp than a forward value); the differentiated loss itself uses the forward tolerance.

## Known divergences encoded here
`genome.hash()` is never compared.
Activation id `-1` (identity in the rewrite, clamped to the last option by the legacy `lax.switch`) is excluded: parity genomes use option `0` of a one-element list and no attribute row is written with `-1`.
All configured ACT/AGG options are evaluated and selected, so a non-total option could poison a branch it was not selected into: every parity genome configures exactly one option on each side.
NaN fitness is the one *deliberate* divergence — `problem.base.finalize_fitness` maps `NaN -> -inf`, the legacy `FuncFit.evaluate` leaves the `NaN` for the algorithm to sanitise — and `test_fitness.py` pins it instead of asserting equality.
RNG, `initialize`, mutation and crossover are never compared, at any tolerance.
`CartPole` has no parity oracle (the legacy RL problems are gone); `../unit/problem`'s numpy reference is its only oracle.

## Conventions
`pytestmark = [pytest.mark.parity]` at module scope in every module here, so `-m parity` selects the directory and `-m "not parity"` excludes it.
Every test that depends on unimplemented source has a real signature, a docstring stating invariant / oracle / assertion / owner, and a body of `pytest.skip("not implemented: <what> (owner: <path>)")` — the file stays importable and collectable before the implementation wave.
No `jax` import at module scope anywhere: it is imported inside fixture or test bodies, always after `pytest.importorskip("jax")`.
No network, no GPU, `backend="numpy"` everywhere (`support.runner`'s default).
Shapes stay small (`max_nodes == 8`, `max_conns == 16`, `pop <= 8`).

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
| `./test_convert.py` | Real tests of `convert.py`: layouts, round trip, rejections, layout rules. |
| `./test_forward.py` | `forward` parity: values + NaN pattern on chains, skipped/isolated nodes, random DAGs, transforms. |
| `./test_distance.py` | `execute_distance` parity: exact `0`, `0.125`, `0.4`, batch-vs-pairwise, empty pair. |
| `./test_fitness.py` | `problem.XOR`/`XOR3d` parity for the four `error_method` values + the NaN divergence. |
| `./test_grad.py` | `grad` parity against `jax.grad` of the same loss, single-option ACT/AGG. |
| `../unit` | numpy-backend kernel tests (the correctness oracle for the rewrite when no legacy oracle exists). |
| `../support` | `runner` (the only `evoxc` entry point), `payloads` (hand-built genomes), `lazy.require`. |
| `../benchmarks/legacy_jax` | The frozen legacy implementation — read-only oracle for this directory. |
