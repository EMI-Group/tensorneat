# ./test/integration — whole-`Pipeline` runs on the numpy backend

## Intent
Prove the *assembled* library rather than its kernels: a `Pipeline` built from the documented public API evolves XOR to its fitness target, reproduces a run from `(seed, config, shapes)`, survives a shape-signature change, and leaves a loadable checkpoint.
Every test here drives the real `Pipeline` (`setup` → generation loop → `auto_run`) with the numpy reference interpreter, and asserts on what the pipeline returns or writes — never on an intermediate kernel.
No `jax`/`torch`, no network, no GPU, no wall-clock assertions: a run is judged by generations, fitness and payload shapes.
The whole directory is `pytest.mark.slow` (module-level `pytestmark`), so `pytest -m slow` selects it and `pytest -m "not parity and not slow"` leaves it out.

## How a test constructs a `Pipeline`
`./conftest.py` owns the single configuration, `TINY_CONFIG`, grouped by the object that owns each knob: genome `num_inputs=2, num_outputs=1, max_nodes=8, max_conns=16, max_in_degree=4, max_depth=8, output_transform="sigmoid"`, `NEAT(pop_size=16, species_size=4)`, `XOR(error_method="mse")`, `Pipeline(seed=0, fitness_target=-0.01, generation_limit=60, backend="numpy", device=None, compile_options=None, is_save=False, save_dir=None, is_detailed=True, show_problem_details=False)`.
`tiny_pipeline(**overrides)` returns that `Pipeline` **configured but not run**; an unknown knob raises `TypeError` (a silently ignored override would make a test assert a configuration it never built), and `output_transform` is either an `ACT` registry name (`"sigmoid"`), a callable, or `None`.
`fitness_target=-0.01` on XOR's negated `mse` means "mean squared error ≤ 0.01"; tests that need a fixed-length run pass `fitness_target=1.0`, which XOR can never reach.
`run_pipeline(pipeline)` calls `setup()` + `auto_run()` and returns `(state, best, history)`, `history` being per-generation `(generation, max_fitness)` pairs.
`evaluate_population(pipeline, state, pop)` re-scores any genome payload on the host through the documented kernel triple `algorithm.transform` → `algorithm.forward` (partial-applied into the problem's `act_fn`) → `problem.evaluate`, and is the only `evoxc` path besides `Pipeline` itself (`support.runner`).
Nothing imports `tensorneat` or `support` at module scope: `tn`, `payloads` and `runner` are `support.lazy.require` fixtures, so a design-skeleton `src/` yields named skips instead of collection errors.

## Layout
```
test/integration/
  __init__.py           package marker (basenames are unique here too; kept for parity with ./parity)
  conftest.py           TINY_CONFIG, tiny_pipeline, run_pipeline, evaluate_population, fitness_history
  test_xor_run.py       auto_run solves XOR inside the budget, returns (state, best), and the reported best re-scores to it
  test_determinism.py   same seed = same trajectory/population/randkey, other seed differs, global numpy state is irrelevant
  test_shape_rebuild.py pop_size / genome-size changes rebuild and still run; a mismatched shape raises; same shape reuses
  test_checkpoint.py    is_save writes config.txt, log.txt, genomes/*.npz, and the npz round-trips to a scorable payload
```

## Runtime budget
Target: the whole directory under ~20 s on the numpy reference interpreter, which is why every module pins its own generation budget instead of relying on the default 60.
`test_xor_run` is the only multi-generation module: `pop_size=16` × `generation_limit=60`, budgeted at ~10–20 s (one compile per shape plus 60 generation steps).
`test_determinism` runs `generation_limit=10` at most twice per test, `test_shape_rebuild` one generation per run, `test_checkpoint` two generations per run.
Each module's docstring records its measured wall time in Phase 2 (once `pipeline.py` exists); no test asserts on it.

## Conventions
`pytestmark = [pytest.mark.slow]` at module scope in every test module; `slow` is declared in `../pyproject.toml`.
Every test has a real signature and a docstring stating invariant / oracle / assertion / budget / owner, and a body of `pytest.skip("not implemented: <what> (owner: <module>.<symbol>)")`.
Shapes stay small (`pop_size` 8 or 16, `max_nodes` ≤ 16) and files stay under ~1000 lines.
Checkpoint tests always pass `save_dir=tmp_path`: the documented default is a timestamped directory, which the suite must never create in the repository.
`log.txt` is parsed with the documented column list (`Generation,Max,Min,Mean,Std,Cost Time`); the `Cost Time` column is read but never asserted on.

## Constraints
`pytest`'s prepend import mode names a test module by bare basename, so `./__init__.py` namespaces this directory as `integration.*`; do not delete it, and keep new module basenames unique across the tree.
`../src/tensorneat`, `../../benchmarks` and `../../pyproject.toml` are read-only inputs: a defect or a contract gap found from here goes into the report to the parent, never into this directory's diff.
Two documented behaviours are underspecified and are handled explicitly rather than guessed: the per-generation trajectory accessor (`fitness_history` prefers `pipeline.fitness_history` and falls back to `<save_dir>/log.txt`, raising if neither exists) and the executable-cache attribute (`test_the_same_shape_reuses_the_cached_executable` is written against the behaviour and skipped, naming `pipeline.py` as its owner).
A mismatched-shape assertion deliberately accepts any exception (`pytest.raises(Exception)`): the contract is "never silently reuse", and pinning one `evoxc` class would couple the suite to an internals name.

## Routing Table
| Path | Purpose |
|---|---|
| `./conftest.py` | The one tiny `Pipeline` configuration, the run helper, the host re-scoring helper, the lazy imports. |
| `./test_xor_run.py` | End-to-end XOR: target reached in budget, documented return pair, reported best re-scores. |
| `./test_determinism.py` | Reproducibility from `(seed, config, shapes)`; global numpy RNG neither read nor written; `randkey` advance. |
| `./test_shape_rebuild.py` | Static-shape contract: rebuild on signature change, refusal on mismatch, cache reuse. |
| `./test_checkpoint.py` | `is_save` artefacts: `config.txt`, `log.txt`, `genomes/*.npz`, and the npz round trip. |
| `../support` | `runner` (only `evoxc` entry point), `payloads` (hand-built payloads), `lazy.require`. |
| `../unit` | Kernel-level correctness suite (the oracle this directory builds on, never re-tests). |
| `../../src/tensorneat` | Code under test — read-only; `pipeline.py` owns the compile site and the cache. |
