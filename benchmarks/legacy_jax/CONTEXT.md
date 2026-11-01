# Benchmarks — legacy JAX baseline

## Intent
Frozen copy of the **pre-rewrite** TensorNEAT implementation (JAX), kept only so the new evoxc implementation can be timed against it and diffed against it numerically.
This directory is not part of the shipped distribution (packaging finds packages under `src/` only) and must never be imported by `src/tensorneat`.

## API Surface
`import tensorneat_legacy` after putting `benchmarks/legacy_jax` on `sys.path` — the package is a rename of the original `tensorneat` (all internal imports rewritten), so every legacy module path works with the `_legacy` suffix:
`tensorneat_legacy.pipeline.Pipeline`, `tensorneat_legacy.algorithm.neat.NEAT`, `tensorneat_legacy.genome.DefaultGenome`, `tensorneat_legacy.problem.func_fit.XOR`, `tensorneat_legacy.common.{ACT,AGG,State}`.

## Provenance
Provenance and the source commit are recorded in `PROVENANCE.md`; treat this tree as read-only.

## Constraints
- FROZEN: never edit, refactor, or format code here, and never "fix" its bugs — a moving baseline makes the benchmark meaningless.
- Requires `jax` (the `bench` extra); nothing else in the repo may depend on it.
- Runs are spawned in subprocesses by the harness so JAX threads/allocators cannot perturb the evoxc side of a measurement.
