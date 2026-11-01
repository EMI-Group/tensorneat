# TensorNEAT benchmark — evoxc rewrite vs frozen JAX baseline

> **GENERATED FILE.** Produced by `python benchmarks/run_benchmark.py`, rendered from
> `/root/tensorneat/.genesis/workers/worker_T1_A117/benchmarks/results/latest.json` at 2026-11-01T18:08:34+00:00. It must not be hand-edited:
> rerun the CLI instead, or the numbers below stop matching the recorded evidence.

## Environment

| Item | Value |
|---|---|
| Date (UTC) | 2026-11-01T18:08:34+00:00 |
| Python | 3.12.3 |
| Platform / CPU | Linux-6.8.0-146-generic-x86_64-with-glibc2.39 · x86_64 · 20 cores |
| numpy / jax / evoxc / tensorneat | numpy 2.5.4 / jax 0.12.0.1 / evoxc 0.1.0 / tensorneat n/a |

## Protocol

| Parameter | Value |
|---|---|
| Backend(s) of the new implementation | `numpy` |
| Warmup generations (untimed) | 3 |
| Timed generations | 25 |
| Repeats (seed + r) | 1 (`seed + r`, seeds 42) |
| Thread pinning | `--threads 0` — runtime defaults (as users run it) |
| Scheduling | one worker process per run, strictly sequential — the two implementations never share a process |

Metrics: `compile_time_s` (one-off build), `ms_per_generation` (median wall clock per generation
after warmup, including host sync), `gens_per_second`, `peak_rss_mb`, and quality
(`final_best_fitness`, `generations_to_target`, best-fitness curve).
Definitions, fairness rules and verdict rules live in `benchmarks/CONTEXT.md`.

## Case matrix

| Case | Problem | Pop | Generations | Fitness target | Old | New |
|---|---|---|---|---|---|---|
| xor_p100 | XOR | 100 | 150 | -0.1 | yes | yes |
| xor_p500 | XOR | 500 | 150 | -0.1 | yes | yes |

| Case | Family | Notes |
|---|---|---|
| xor_p100 | func_fit | — |
| xor_p500 | func_fit | — |

## Results per case

### xor_p100

| Implementation | Backend | Compile (s) | ms/generation | generations/s | peak RSS (MiB) | Final best | Gens to target | Repeats |
|---|---|---|---|---|---|---|---|---|
| `old` tensorneat-legacy (JAX) | `jax` | 2.718 | 31.20 | 32.05 | 570.0 | -0.2257 | >28 | 1 |
| `new` tensorneat (evoxc) — error | `numpy` | n/a | n/a | n/a | n/a | n/a | n/a | 1 |

- `new` / `numpy`: **error** — NotImplementedError: store kwargs in an internal dict without exposing mutators

### xor_p500

| Implementation | Backend | Compile (s) | ms/generation | generations/s | peak RSS (MiB) | Final best | Gens to target | Repeats |
|---|---|---|---|---|---|---|---|---|
| `old` tensorneat-legacy (JAX) | `jax` | 2.795 | 182.94 | 5.47 | 615.9 | -0.1774 | >28 | 1 |
| `new` tensorneat (evoxc) — error | `numpy` | n/a | n/a | n/a | n/a | n/a | n/a | 1 |

- `new` / `numpy`: **error** — NotImplementedError: store kwargs in an internal dict without exposing mutators

## Old vs new

| Case | Backend | Old ms/gen | New ms/gen | Speed-up | Delta ms/gen | Old final best | New final best | Quality delta | Timing | Quality |
|---|---|---|---|---|---|---|---|---|---|---|
| xor_p100 | `numpy` | 31.20 | n/a | n/a | n/a | -0.2257 | n/a | n/a | n/a | n/a |
| xor_p500 | `numpy` | 182.94 | n/a | n/a | n/a | -0.1774 | n/a | n/a | n/a | n/a |

Timing PASS means new ms/generation <= old; quality PASS means the new final best stays within 10% of the old final best.

## Verdict

**INCOMPLETE**

- xor_p100/new/numpy: error — NotImplementedError: store kwargs in an internal dict without exposing mutators
- xor_p500/new/numpy: error — NotImplementedError: store kwargs in an internal dict without exposing mutators
- timing: no verdict for xor_p100/numpy (both sides need status ok and a measured ms/generation)
- timing: no verdict for xor_p500/numpy (both sides need status ok and a measured ms/generation)
- quality: no verdict for xor_p100/numpy (final best fitness missing on at least one side)
- quality: no verdict for xor_p500/numpy (final best fitness missing on at least one side)
- 2 run(s) not ok — cannot finalise the verdict (timing=INCOMPLETE, quality=n/a)

Sub-verdicts: timing `INCOMPLETE`, quality `n/a` — timing compares measured ms/generation, quality compares final best fitness at a 10% tolerance.

## Quality curves

### xor_p100

| Implementation | Backend | Seed | Best @1 | Best @10 | Best @25% | Best @50% | Best @100% | Final | Gens to target |
|---|---|---|---|---|---|---|---|---|---|
| `old` tensorneat-legacy (JAX) | `jax` | 42 | -0.2507 | -0.2500 | -0.2500 | -0.2495 | -0.2257 | -0.2257 | >28 |

### xor_p500

| Implementation | Backend | Seed | Best @1 | Best @10 | Best @25% | Best @50% | Best @100% | Final | Gens to target |
|---|---|---|---|---|---|---|---|---|---|
| `old` tensorneat-legacy (JAX) | `jax` | 42 | -0.2506 | -0.2477 | -0.2498 | -0.2388 | -0.1774 | -0.1774 | >28 |

Percent columns are generation `round(fraction × N)` of the N recorded generations (warmup + timed — the budget `generations_to_target` is indexed against), and `>N` means the case's fitness target was not reached within those N recorded generations. Each row is the first ok run recorded for that case/implementation/backend.

## Reference numbers (declared baseline)

| Case | Pop | Declared legacy ms/gen | Measured old ms/gen | Ratio (measured/declared) |
|---|---|---|---|---|
| xor_p100 | 100 | 51.00 | 31.20 | 0.61x |
| xor_p500 | 500 | 109.00 | 182.94 | 1.68x |

The performance contract in `../CONTEXT.md` declares legacy CPU numbers measured on a
1-thread-bound box (51 ms/gen at pop 100, 109 ms/gen at pop 500, 157 ms/gen at pop 1000).
The generator prints those next to the old row measured on this box; the verdict uses the
measured row.

## What is not compared

- Different PRNGs (JAX `PRNGKey` vs evoxc splitmix64) mean a shared seed identifies the same *workload*, not the same search trajectory; numerical fidelity is covered by `test/parity/`.
- Accelerator numbers are informational.
- CartPole is new-only (the legacy library has no equivalent problem).

## How to reproduce

```bash
/tmp/bench-venv/bin/python benchmarks/run_benchmark.py --impl both --cases xor_p100,xor_p500 --generations 25 --warmup 3 --repeats 1
python benchmarks/run_benchmark.py --list   # print the case matrix
python benchmarks/run_benchmark.py --dry-run  # print the planned worker commands
```

Cases: xor_p100, xor_p500. Timed generations: 25, warmup generations (untimed): 3, repeats: 1 (seeds 42), threads: runtime defaults (`--threads 0`). Backend(s) of the new side: numpy; the old side always ran on JAX, once per case.

Generated from `/root/tensorneat/.genesis/workers/worker_T1_A117/benchmarks/results/latest.json` at 2026-11-01T18:08:34+00:00 — do not hand-edit.
