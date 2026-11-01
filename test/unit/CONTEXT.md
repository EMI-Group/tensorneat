# ./unit — numpy-backend kernel and invariant suites

## Intent
Prove that every evoxc kernel in `../../src/tensorneat` computes what its docstring promises, on the reference numpy interpreter, with no optional dependency and no `jax`/`torch` in the process.
Each assertion compares against something *independent* of the implementation it under test: a closed-form value, a plain-numpy oracle from `../support/oracle.py`, or a structural invariant (`sum(spawn) == pop_size`, "no NaN survives `tell`").
The suite runs in the default gate: `pytest -m "not parity and not slow"`.

## Layout
`./common` — schedule (`tensorization`), flat tensor helpers, ACT/AGG dispatchers, host graph walks, `State`.
`./genome` — gene payloads, forward/grad on hand-built genomes, genetic operator invariants, host-only introspection.
`./algorithm` — speciation, stagnation, spawn allocation and parent selection algebra.
`./problem` — func-fit losses and CartPole dynamics against host-numpy references.

## Conventions
One module per source module, named `test_<module>.py`; the directory tree mirrors `../../src/tensorneat/`.
Shapes stay small (pop ≤ 16, `max_nodes ≤ 16`) except where a test exists specifically to trip a static bound.
Populations, payloads and keys come from `../support/` — a test never draws a topology it could build by hand.
`conftest.py` here fails any test that leaves `jax`/`torch`/`brax`/`gymnax`/`mujoco` in `sys.modules`, so the "no optional dependency" claim is enforced per test, not only by the release gate.
Every test that needs unimplemented source code carries a real signature and docstring with a body of `pytest.skip("not implemented: …")` naming the owner.

## Constraints
Backend is always `"numpy"`; a test needing `evoxc[iree]` is marked `requires_iree` and excluded from every gate.
No network, no GPU, no `tmp_path`-dependent persistence except in the integration suite.
A test may not edit `../../src`, `../../benchmarks` or `pyproject.toml`; a discovered defect is reported upward.

## Routing Table
| Path | Purpose |
|---|---|
| `./common` | tensorization schedule invariants, flat gather/scatter/mutation helpers, ACT/AGG dispatch, host graph algorithms, `State`. |
| `./genome` | payload layout, forward/recurrent/grad kernels, mutation/crossover/distance invariants, host introspection. |
| `./algorithm` | NEAT `tell` algebra: speciation, stagnation, spawn allocation, parent selection. |
| `./problem` | func-fit losses (`mse`/`rmse`/`mae`/`mape`) and the CartPole rollout. |
