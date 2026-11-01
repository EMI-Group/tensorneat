# TensorNEAT benchmark — evoxc rewrite vs frozen JAX baseline

> **GENERATED FILE.** Produced by `python benchmarks/run_benchmark.py`, rendered from
> `benchmarks/results/latest.json` at 2026-11-01T18:19:16+00:00. It must not be hand-edited:
> rerun the CLI instead, or the numbers below stop matching the recorded evidence.

## Environment

| Item | Value |
|---|---|
| Date (UTC) | 2026-11-01T18:19:16+00:00 |
| Python | 3.12.3 |
| Platform / CPU | Linux-6.8.0-146-generic-x86_64-with-glibc2.39 · x86_64 · 20 cores |
| numpy / jax / evoxc / tensorneat | numpy 2.5.4 / jax 0.12.0.1 / evoxc n/a / tensorneat n/a |

## Protocol

| Parameter | Value |
|---|---|
| Backend(s) of the new implementation | `numpy` |
| Warmup generations (untimed) | 3 |
| Timed generations | 150 |
| Repeats (seed + r) | 2 (`seed + r`, seeds 42, 43) |
| Thread pinning | `--threads 0` — runtime defaults (as users run it) |
| Scheduling | one worker process per run, strictly sequential — the two implementations never share a process |

Metrics: `compile_time_s` (one-off build), `ms_per_generation` (median wall clock per generation
after warmup, including host sync), `gens_per_second`, `peak_rss_mb`, and quality
(`final_best_fitness`, `generations_to_target`, best-fitness curve).
Definitions, fairness rules and verdict rules live in `benchmarks/CONTEXT.md`.

## Case matrix

| Case | Family | Problem | Pop | Timed generations | Fitness target | Old | New | Notes |
|---|---|---|---|---|---|---|---|---|
| xor_p100 | func_fit | XOR | 100 | 150 | -0.1 | yes | not run | — |
| xor_p500 | func_fit | XOR | 500 | 150 | -0.1 | yes | not run | — |
| xor_p1000 | func_fit | XOR | 1000 | 150 | -0.1 | yes | not run | — |
| xor3d_p100 | func_fit | XOR3d | 100 | 150 | -0.5 | yes | not run | — |
| xor3d_p500 | func_fit | XOR3d | 500 | 150 | -0.5 | yes | not run | — |
| xor3d_p1000 | func_fit | XOR3d | 1000 | 150 | -0.5 | yes | not run | — |
| custom_pagie_p500 | func_fit | CustomFuncFit | 500 | 150 | -0.05 | yes | not run | per-impl divergence: the legacy side seeds the global numpy RNG and vmaps a jax-typed func; the new side samples from default_rng(seed) with a numpy func — same distribution, different draws |
| cartpole_p100 | rl | CartPole | 100 | 150 | 180 | n/a | not run | new-only: the legacy library has no equivalent problem |

## Results per case

### xor_p100

| Implementation | Backend | Compile (s) | ms/generation | generations/s | peak RSS (MiB) | Final best | Gens to target | Repeats |
|---|---|---|---|---|---|---|---|---|
| `old` tensorneat-legacy (JAX) | `jax` | 2.774 | 36.09 | 27.71 | 581.8 | -0.0099 | 77 | 2 |

### xor_p500

| Implementation | Backend | Compile (s) | ms/generation | generations/s | peak RSS (MiB) | Final best | Gens to target | Repeats |
|---|---|---|---|---|---|---|---|---|
| `old` tensorneat-legacy (JAX) | `jax` | 2.686 | 128.40 | 7.79 | 626.9 | -0.0000 | 39 | 2 |

### xor_p1000

| Implementation | Backend | Compile (s) | ms/generation | generations/s | peak RSS (MiB) | Final best | Gens to target | Repeats |
|---|---|---|---|---|---|---|---|---|
| `old` tensorneat-legacy (JAX) | `jax` | 2.659 | 243.54 | 4.11 | 627.6 | -0.0000 | 32 | 2 |

### xor3d_p100

| Implementation | Backend | Compile (s) | ms/generation | generations/s | peak RSS (MiB) | Final best | Gens to target | Repeats |
|---|---|---|---|---|---|---|---|---|
| `old` tensorneat-legacy (JAX) | `jax` | 2.732 | 19.06 | 52.47 | 592.1 | -0.1389 | 1 | 2 |

### xor3d_p500

| Implementation | Backend | Compile (s) | ms/generation | generations/s | peak RSS (MiB) | Final best | Gens to target | Repeats |
|---|---|---|---|---|---|---|---|---|
| `old` tensorneat-legacy (JAX) | `jax` | 2.727 | 68.27 | 14.65 | 622.9 | -0.1576 | 1 | 2 |

### xor3d_p1000

| Implementation | Backend | Compile (s) | ms/generation | generations/s | peak RSS (MiB) | Final best | Gens to target | Repeats |
|---|---|---|---|---|---|---|---|---|
| `old` tensorneat-legacy (JAX) | `jax` | 2.690 | 181.03 | 5.52 | 642.9 | -0.1250 | 1 | 2 |

### custom_pagie_p500

| Implementation | Backend | Compile (s) | ms/generation | generations/s | peak RSS (MiB) | Final best | Gens to target | Repeats |
|---|---|---|---|---|---|---|---|---|
| `old` tensorneat-legacy (JAX) | `jax` | 2.756 | 362.46 | 2.76 | 791.8 | -0.0052 | 1 | 2 |

### cartpole_p100

no runs recorded

## Old vs new

no old-vs-new comparison available.

## Verdict

**INCOMPLETE**

- no old-vs-new comparison available

Sub-verdicts: timing `n/a`, quality `n/a` — timing compares measured ms/generation, quality compares final best fitness at a 10% tolerance.

## Quality curves

### xor_p100

| Implementation | Backend | Seed | Best @1 | Best @10 | Best @25% | Best @50% | Best @100% | Final | Gens to target |
|---|---|---|---|---|---|---|---|---|---|
| `old` tensorneat-legacy (JAX) | `jax` | 42 | -0.2507 | -0.2500 | -0.1929 | -0.0542 | -0.0000 | -0.0000 | 62 |

### xor_p500

| Implementation | Backend | Seed | Best @1 | Best @10 | Best @25% | Best @50% | Best @100% | Final | Gens to target |
|---|---|---|---|---|---|---|---|---|---|
| `old` tensorneat-legacy (JAX) | `jax` | 42 | -0.2506 | -0.2477 | -0.1152 | -0.0055 | -0.0000 | -0.0000 | 41 |

### xor_p1000

| Implementation | Backend | Seed | Best @1 | Best @10 | Best @25% | Best @50% | Best @100% | Final | Gens to target |
|---|---|---|---|---|---|---|---|---|---|
| `old` tensorneat-legacy (JAX) | `jax` | 42 | -0.2506 | -0.2427 | -0.0780 | -0.0004 | -0.0000 | -0.0000 | 35 |

### xor3d_p100

| Implementation | Backend | Seed | Best @1 | Best @10 | Best @25% | Best @50% | Best @100% | Final | Gens to target |
|---|---|---|---|---|---|---|---|---|---|
| `old` tensorneat-legacy (JAX) | `jax` | 42 | -0.2524 | -0.2500 | -0.2500 | -0.2447 | -0.0278 | -0.0278 | 1 |

### xor3d_p500

| Implementation | Backend | Seed | Best @1 | Best @10 | Best @25% | Best @50% | Best @100% | Final | Gens to target |
|---|---|---|---|---|---|---|---|---|---|
| `old` tensorneat-legacy (JAX) | `jax` | 42 | -0.2524 | -0.2500 | -0.2500 | -0.2500 | -0.2500 | -0.2500 | 1 |

### xor3d_p1000

| Implementation | Backend | Seed | Best @1 | Best @10 | Best @25% | Best @50% | Best @100% | Final | Gens to target |
|---|---|---|---|---|---|---|---|---|---|
| `old` tensorneat-legacy (JAX) | `jax` | 42 | -0.2524 | -0.2500 | -0.1850 | -0.0192 | -0.0000 | -0.0000 | 1 |

### custom_pagie_p500

| Implementation | Backend | Seed | Best @1 | Best @10 | Best @25% | Best @50% | Best @100% | Final | Gens to target |
|---|---|---|---|---|---|---|---|---|---|
| `old` tensorneat-legacy (JAX) | `jax` | 42 | -0.0490 | -0.0473 | -0.0347 | -0.0233 | -0.0064 | -0.0064 | 1 |

Percent columns are generation `round(fraction × N)` of the N recorded generations (warmup + timed — the budget `generations_to_target` is indexed against), and `>N` means the case's fitness target was not reached within those N recorded generations. Each row is the first ok run recorded for that case/implementation/backend.

## Reference numbers (declared baseline)

| Case | Pop | Declared legacy ms/gen | Measured old ms/gen | Ratio (measured/declared) |
|---|---|---|---|---|
| xor_p100 | 100 | 51.00 | 36.09 | 0.71x |
| xor_p500 | 500 | 109.00 | 128.40 | 1.18x |
| xor_p1000 | 1000 | 157.00 | 243.54 | 1.55x |

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
python benchmarks/run_benchmark.py --impl old --cases all --generations 150 --warmup 3 --repeats 2
python benchmarks/run_benchmark.py --list   # print the case matrix
python benchmarks/run_benchmark.py --dry-run  # print the planned worker commands
```

Interpreter used: /tmp/genesis/task_d7be275380ce6fc0/bench-venv/bin/python

Cases: xor_p100, xor_p500, xor_p1000, xor3d_p100, xor3d_p500, xor3d_p1000, custom_pagie_p500, cartpole_p100. Timed generations: 150, warmup generations (untimed): 3, repeats: 2 (seeds 42, 43), threads: runtime defaults (`--threads 0`). Backend(s) of the new side: numpy; the old side always ran on JAX, once per case.

Generated from `benchmarks/results/latest.json` into `benchmarks/RESULTS.md` at 2026-11-01T18:19:16+00:00 — do not hand-edit.
