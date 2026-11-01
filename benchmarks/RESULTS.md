# TensorNEAT benchmark — evoxc rewrite vs frozen JAX baseline

> **Template.** No measurement has been recorded in this checkout yet.
> This file is **generated**: run `python benchmarks/run_benchmark.py` and it is rewritten
> in place from `benchmarks/results/latest.json`. Do not hand-edit numbers here.

## Environment

| Item | Value |
|---|---|
| Date (UTC) | _filled by the generator_ |
| Python | _filled by the generator_ |
| Platform / CPU | _filled by the generator_ |
| numpy / jax / evoxc | _filled by the generator_ |

## Protocol

| Parameter | Value |
|---|---|
| Backend(s) of the new implementation | _filled by the generator_ |
| Warmup generations (untimed) | _filled by the generator_ |
| Timed generations | _filled by the generator_ |
| Repeats (seed + r) | _filled by the generator_ |
| Thread pinning | _filled by the generator_ |

Metrics: `compile_time_s` (one-off build), `ms_per_generation` (median wall clock per generation
after warmup, including host sync), `gens_per_second`, `peak_rss_mb`, and quality
(`final_best_fitness`, `generations_to_target`, best-fitness curve).
Definitions, fairness rules and verdict rules live in `benchmarks/CONTEXT.md`.

## Case matrix

| Case | Problem | Pop | Generations | Fitness target | Old | New |
|---|---|---|---|---|---|---|
| _filled by the generator_ | | | | | | |

## Results per case

_per case: one row per (implementation, backend) with compile time, ms/generation, generations/s,
peak RSS, final best fitness and generations-to-target._

## Old vs new

_per case and new backend: speed-up, delta ms/generation, quality delta, timing verdict, quality verdict._

## Verdict

`INCOMPLETE` — no runs recorded yet; run the CLI to produce the evidence.

## Quality curves

_best fitness at generation 1 / 10 / 25% / 50% / 100% and final, per implementation._

## Reference numbers (declared baseline)

The performance contract in `../CONTEXT.md` declares legacy CPU numbers measured on a
1-thread-bound box (51 ms/gen at pop 100, 109 ms/gen at pop 500, 157 ms/gen at pop 1000).
The generator prints those next to the old row measured on this box; the verdict uses the
measured row.

## What is not compared

Different PRNGs (JAX `PRNGKey` vs evoxc splitmix64) mean a shared seed identifies the same
*workload*, not the same search trajectory; numerical fidelity is covered by `test/parity/`.
Accelerator numbers are informational. CartPole is new-only (the legacy library has no
equivalent problem).

## How to reproduce

```bash
python benchmarks/run_benchmark.py --list
python benchmarks/run_benchmark.py            # both impls, numpy backend, all cases
```

The generator appends the exact command line, case list, generations, repeats and seeds that
produced the tables above.
