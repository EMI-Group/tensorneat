# ./test

## Intent
Scratch/issue-driven verification scripts, benchmarks and notebooks for TensorNEAT — not a maintained unit-test suite.
Only `test_grad.py` is a genuine pytest module; the rest are runnable scripts, ad-hoc experiments, or stale pre-refactor code.
Files are kept as debugging artifacts referenced from GitHub issues (#11, #15, #38, #40, #41), not as CI-gated tests.

## API Surface (files)
`test_grad.py` — the only real pytest module: classes `TestGradFinite`, `TestJitGrad`, `TestGradDescentReducesLoss`, `TestBatchGrad`, `TestNaNPropagation`, `TestBiasNodeGrad`, `TestRecurrentGrad`, `TestNumericalGradCheck`; needs `jax.grad`/`jit`/`vmap` and NaN-padding semantics; run `pytest test/test_grad.py -v`.
`benchmark_optimization.py` — `argparse` benchmark for issues #38 (unflatten_conns) / #40 (crossover); flags `--save-baseline` and `--check`; baseline data goes to gitignored `test/benchmark_baseline/`.
`profile_memory_for_issue.py` — GPU/CPU peak-memory + compile-time + step-time profiler for `pipeline.step`; no CLI args; spawns one isolated subprocess per config.
`origin_operations_test.py` — print-only script (`mutation_default/origin`, `distance_default`, `distance_origin_case1/2`, `crossover_origin_case1/2/3`) illustrating origin-node/conn mutation, distance and crossover; no asserts.
`crossover_mutation.py` — 50-generation `create_next_generation` loop over pop 1000 checking outputs are not NaN.
`mlp_substrate.py` — 7-line script printing `analysis_substrate([3,4,2], (-1,1,-1,1))` coordinates; valid against current src.
`ranknet.py` — RankNet-as-NEAT-problem experiment: custom `BaseProblem` over 100×768 random data, pop 100, 150 generations.
`ranknet_neat.py` — RankNet variant with pairwise NaN masking and accuracy reporting; pop 10, 5 generations; header says it throws a `ValueError`.
`test.py` — module-level script comparing `jnp.where` vs boolean-indexing pairwise BCE loss; not collected by pytest (does not match `test_*.py`).
`test_genome.py` — single `test_output_work()` (ends in `assert True`); imports the removed `algorithm.neat` package.
`test_nan_fitness.py` / `nan_fitness.py` — near-duplicate NaN-genome debugging scripts importing `algorithm.neat` and loading `../examples/brax/nan_*.npy` (files absent).
`test_record_episode.py` — 0-byte placeholder.

## Notebooks
`test_compile_time.ipynb` — compares `vmap(func)` inside a 1000-iteration Python loop vs. hoisting `vmap_func = vmap(func)` out of the loop; measures `jit(...).lower().compile()` time.
`test_flatten.ipynb` — `genome.transform` / `genome.restore` round-trip and vmap'd `unflatten_conns`/`flatten_conns` shapes (imports `utils.tools`, pre-refactor).
`test_update_by_batch.ipynb` — `genome.update_by_batch` on 10000 inputs with a `NormalizedNode`: shows layer-output means before/after batch normalization and that `forward(vmap)` on the updated transform matches.
`test_efficient_b_spline.ipynb` — `%timeit` comparison of `BSplineConn` vs `CacheBSplineConn` (grid_cnt=6, order=3, cache_num=1e6) batch forward, plus plots of both curves.
`test_b_spline_conn.ipynb` — `BSplineConn` behavior: random attrs, `distance`, scalar `forward`, `update_by_batch` on 1000 samples, plots before/after grid update.
`test_kan.ipynb` — `DefaultGenome` with `KANNode` + `BSplineConn`: setup, initialize, transform, forward (prints NaN output).
`test_record_episode.ipynb` — `BraxEnv("ant")` / `GymNaxEnv("CartPole-v1")` evaluate with and without `record_episode=True`, single vs. vmapped batch, plus timing loops.
`test.ipynb` — ranknet pairwise-label/BCE scratchpad.

## Constraints
No pytest config anywhere: no `pytest.ini`, `conftest.py`, `setup.cfg` or `tox.ini` — pytest runs with defaults (`test_*.py`, `*_test.py`, `Test*`, `test*`).
CI never runs tests: `.github/workflows/pypi_publish.yaml` only builds and uploads to PyPI on release.
`python test/origin_operations_test.py` (and `benchmark_optimization.py`) execute genome setup at import time, so they need a working JAX backend before anything else.
`benchmark_optimization.py` and `profile_memory_for_issue.py` prepend `<repo>/src` to `sys.path`, so they run from any cwd without installing the package.
`profile_memory_for_issue.py` hardcodes `conda run -n jax_env python ...` for its subprocesses, so it only works inside that exact conda env.

## Known Issues
`test_genome.py`, `test_nan_fitness.py`, `nan_fitness.py` and `crossover_mutation.py` import `from algorithm.neat import *`, a package path that no longer exists (now `tensorneat.algorithm`) — they fail at import.
`test_nan_fitness.py` and `nan_fitness.py` also load `../examples/brax/nan_node.npy` and `nan_conn.npy`, which are not in the repo.
`test_nan_fitness.py`, `origin_operations_test.py` and `test_record_episode.py` are collected by pytest but contain zero `test_*` functions.
`benchmark_optimization.py` prints a theoretical memory figure for the pre-optimization implementation and reads peak memory from `memory_stats()` on GPU or `ru_maxrss` elsewhere, so numbers are not comparable across devices.

## Routing Table
No subdirectories — this node is a leaf.
Related code lives in `../src/tensorneat/genome/utils.py` (`unflatten_conns`), `../src/tensorneat/genome/operations/`, `../src/tensorneat/algorithm/hyperneat/substrate/mlp.py` and `../src/tensorneat/problem/`.
