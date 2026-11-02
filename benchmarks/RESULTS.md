# TensorNEAT benchmark — evoxc rewrite vs frozen JAX baseline

> **GENERATED FILE.** Produced by `python benchmarks/run_benchmark.py`, rendered from
> `benchmarks/results/latest.json` at 2026-11-02T05:00:13+00:00. It must not be hand-edited:
> rerun the CLI instead, or the numbers below stop matching the recorded evidence.

## Headline: old vs new

| Case | Pop | Backend | Old ms/gen | New ms/gen | Speed-up | Old compile (s) | New compile (s) | Old peak RSS (MiB) | New peak RSS (MiB) | Old final best | New final best | Gens to target (old) | Gens to target (new) | Timing | Quality |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| xor_p100 | 100 | `numpy` | 36.19 | 58.26 | 0.62x | 2.838 | 0.562 | 584.7 | 145.2 | -0.0099 | -0.0303 | 77 | 97 | FAIL | FAIL |
| xor_p500 | 500 | `numpy` | 165.54 | 360.44 | 0.46x | 2.865 | 1.504 | 642.4 | 1049.6 | -0.0000 | -0.0000 | 39 | 30 | FAIL | PASS |
| xor_p1000 | 1000 | `numpy` | 301.21 | 676.49 | 0.45x | 2.865 | 2.210 | 655.8 | 1991.9 | -0.0000 | -0.0000 | 32 | 29 | FAIL | FAIL |
| xor3d_p100 | 100 | `numpy` | 20.38 | 68.57 | 0.30x | 2.860 | 0.587 | 589.5 | 151.8 | -0.1389 | -0.2429 | 149 | >152 | FAIL | FAIL |
| xor3d_p500 | 500 | `numpy` | 74.32 | 360.38 | 0.21x | 2.834 | 1.527 | 644.0 | 1081.7 | -0.1576 | -0.1286 | >153 | 140 | FAIL | PASS |
| xor3d_p1000 | 1000 | `numpy` | 187.24 | 736.41 | 0.25x | 2.781 | 2.282 | 656.0 | 2056.2 | -0.1250 | -0.0089 | 62 | 83 | FAIL | PASS |
| custom_pagie_p500 | 500 | `numpy` | 443.96 | 3298.32 | 0.13x | 2.989 | 4.822 | 758.8 | 6174.3 | -0.0052 | -0.0139 | 121 | 77 | FAIL | FAIL |
| cartpole_p100 | 100 | `numpy` | n/a (not run) | 449.86 | n/a (needs both sides) | n/a (not run) | 0.887 | n/a (not run) | 137.1 | n/a (not run) | 200.0000 | n/a (not run) | 1 | n/a | n/a |

One row per case and new-side backend (the old side always runs on JAX). Speed-up is `old / new` per generation: above 1.00x the new side is faster. Timing PASS means new ms/generation is at or below the old one; quality PASS means the new final best stays within 10% of the old one. Every cell comes from the aggregated JSON records of this run; `n/a (reason)` marks a cell that could not be reached (not run, skipped, error, or not recorded).

## Environment

| Item | Value |
|---|---|
| Date (UTC) | 2026-11-02T05:00:13+00:00 |
| Python | 3.12.3 |
| Platform | Linux-6.8.0-146-generic-x86_64-with-glibc2.39 |
| CPU model | INTEL(R) XEON(R) PLATINUM 8568Y+ |
| CPU cores (`os.cpu_count`) | 20 |
| Memory | 241611 MiB |
| numpy / jax / evoxc / tensorneat (installed metadata) | numpy 2.5.4 / jax 0.12.0.1 / evoxc 0.1.0 / tensorneat n/a |
| Git commit | `1537f795312c` (clean) — `benchmarks/legacy_jax` and `src/` both come from this checkout |
| Thread pinning (`--threads`) | `--threads 0` — runtime defaults (as users run it), nothing exported |
| Thread env (set/unset) | OMP_NUM_THREADS unset · MKL_NUM_THREADS unset · OPENBLAS_NUM_THREADS unset · NUMEXPR_NUM_THREADS unset · XLA_FLAGS unset (CLI process, inherited unchanged by every worker — `--threads 0` exports nothing) |

## Protocol

| Parameter | Value |
|---|---|
| Backend(s) of the new implementation | `numpy` |
| Warmup generations (untimed) | 3 |
| Timed generations | 150 |
| Repeats (seed + r) | 2 (`seed + r`, seeds 42, 43) |
| Thread pinning | `--threads 0` — runtime defaults (as users run it), nothing exported |
| New-side build | inside the first `Pipeline.step` call (the evoxc build); the generation it consumes is recorded as `generations_consumed_before_timing` and excluded from the timed samples and from the quality curve — the untimed warmup is shortened by it so both sides time and record the same generation index |
| Compile time | reported separately as `compile_time_s` (old: `jax.jit(pipeline.step).lower(state).compile()`; new: the first `Pipeline.step` build) and never charged to a generation |
| Scheduling | one worker process per run, strictly sequential — the two implementations never share a process |

Metrics: `compile_time_s` (one-off build), `ms_per_generation` (median wall clock per generation
after warmup, including host sync), `gens_per_second`, `peak_rss_mb`, and quality
(`final_best_fitness`, `generations_to_target`, best-fitness curve).
The new side's build generation is excluded from both the timing and the quality curve, and compile/build seconds are never charged to a generation.
Definitions, fairness rules and verdict rules live in `benchmarks/CONTEXT.md`.

## Case matrix

| Case | Family | Problem | Pop | Timed generations | Fitness target | Old | New | Notes |
|---|---|---|---|---|---|---|---|---|
| xor_p100 | func_fit | XOR | 100 | 150 | -0.1 | yes | yes | — |
| xor_p500 | func_fit | XOR | 500 | 150 | -0.1 | yes | yes | — |
| xor_p1000 | func_fit | XOR | 1000 | 150 | -0.1 | yes | yes | — |
| xor3d_p100 | func_fit | XOR3d | 100 | 150 | -0.05 | yes | yes | — |
| xor3d_p500 | func_fit | XOR3d | 500 | 150 | -0.05 | yes | yes | — |
| xor3d_p1000 | func_fit | XOR3d | 1000 | 150 | -0.05 | yes | yes | — |
| custom_pagie_p500 | func_fit | CustomFuncFit | 500 | 150 | -0.01 | yes | yes | per-impl divergence: the legacy side seeds the global numpy RNG and vmaps a jax-typed func; the new side samples from default_rng(seed) with a numpy func — same distribution, different draws |
| cartpole_p100 | rl | CartPole | 100 | 150 | 180 | n/a | yes | new-only: the legacy library has no equivalent problem |

## Results per case

### xor_p100

| Implementation | Backend | Compile (s) | ms/generation | generations/s | peak RSS (MiB) | Final best | Gens to target | Repeats |
|---|---|---|---|---|---|---|---|---|
| `old` tensorneat-legacy (JAX) | `jax` | 2.838 | 36.19 | 27.63 | 584.7 | -0.0099 | 77 | 2 |
| `new` tensorneat (evoxc) | `numpy` | 0.562 | 58.26 | 17.16 | 145.2 | -0.0303 | 97 | 2 |

### xor_p500

| Implementation | Backend | Compile (s) | ms/generation | generations/s | peak RSS (MiB) | Final best | Gens to target | Repeats |
|---|---|---|---|---|---|---|---|---|
| `old` tensorneat-legacy (JAX) | `jax` | 2.865 | 165.54 | 6.04 | 642.4 | -0.0000 | 39 | 2 |
| `new` tensorneat (evoxc) | `numpy` | 1.504 | 360.44 | 2.77 | 1049.6 | -0.0000 | 30 | 2 |

### xor_p1000

| Implementation | Backend | Compile (s) | ms/generation | generations/s | peak RSS (MiB) | Final best | Gens to target | Repeats |
|---|---|---|---|---|---|---|---|---|
| `old` tensorneat-legacy (JAX) | `jax` | 2.865 | 301.21 | 3.32 | 655.8 | -0.0000 | 32 | 2 |
| `new` tensorneat (evoxc) | `numpy` | 2.210 | 676.49 | 1.48 | 1991.9 | -0.0000 | 29 | 2 |

### xor3d_p100

| Implementation | Backend | Compile (s) | ms/generation | generations/s | peak RSS (MiB) | Final best | Gens to target | Repeats |
|---|---|---|---|---|---|---|---|---|
| `old` tensorneat-legacy (JAX) | `jax` | 2.860 | 20.38 | 49.06 | 589.5 | -0.1389 | 149 | 2 |
| `new` tensorneat (evoxc) | `numpy` | 0.587 | 68.57 | 14.58 | 151.8 | -0.2429 | >152 | 2 |

### xor3d_p500

| Implementation | Backend | Compile (s) | ms/generation | generations/s | peak RSS (MiB) | Final best | Gens to target | Repeats |
|---|---|---|---|---|---|---|---|---|
| `old` tensorneat-legacy (JAX) | `jax` | 2.834 | 74.32 | 13.46 | 644.0 | -0.1576 | >153 | 2 |
| `new` tensorneat (evoxc) | `numpy` | 1.527 | 360.38 | 2.77 | 1081.7 | -0.1286 | 140 | 2 |

### xor3d_p1000

| Implementation | Backend | Compile (s) | ms/generation | generations/s | peak RSS (MiB) | Final best | Gens to target | Repeats |
|---|---|---|---|---|---|---|---|---|
| `old` tensorneat-legacy (JAX) | `jax` | 2.781 | 187.24 | 5.34 | 656.0 | -0.1250 | 62 | 2 |
| `new` tensorneat (evoxc) | `numpy` | 2.282 | 736.41 | 1.36 | 2056.2 | -0.0089 | 83 | 2 |

### custom_pagie_p500

| Implementation | Backend | Compile (s) | ms/generation | generations/s | peak RSS (MiB) | Final best | Gens to target | Repeats |
|---|---|---|---|---|---|---|---|---|
| `old` tensorneat-legacy (JAX) | `jax` | 2.989 | 443.96 | 2.25 | 758.8 | -0.0052 | 121 | 2 |
| `new` tensorneat (evoxc) | `numpy` | 4.822 | 3298.32 | 0.30 | 6174.3 | -0.0139 | 77 | 2 |

### cartpole_p100

| Implementation | Backend | Compile (s) | ms/generation | generations/s | peak RSS (MiB) | Final best | Gens to target | Repeats |
|---|---|---|---|---|---|---|---|---|
| `new` tensorneat (evoxc) | `numpy` | 0.887 | 449.86 | 2.22 | 137.1 | 200.0000 | 1 | 2 |

## Old vs new

| Case | Backend | Old ms/gen | New ms/gen | Speed-up | Delta ms/gen | Old final best | New final best | Quality delta | Timing | Quality |
|---|---|---|---|---|---|---|---|---|---|---|
| xor_p100 | `numpy` | 36.19 | 58.26 | 0.62x | +22.07 | -0.0099 | -0.0303 | -0.0204 | FAIL | FAIL |
| xor_p500 | `numpy` | 165.54 | 360.44 | 0.46x | +194.90 | -0.0000 | -0.0000 | +0.0000 | FAIL | PASS |
| xor_p1000 | `numpy` | 301.21 | 676.49 | 0.45x | +375.28 | -0.0000 | -0.0000 | -0.0000 | FAIL | FAIL |
| xor3d_p100 | `numpy` | 20.38 | 68.57 | 0.30x | +48.19 | -0.1389 | -0.2429 | -0.1040 | FAIL | FAIL |
| xor3d_p500 | `numpy` | 74.32 | 360.38 | 0.21x | +286.06 | -0.1576 | -0.1286 | +0.0290 | FAIL | PASS |
| xor3d_p1000 | `numpy` | 187.24 | 736.41 | 0.25x | +549.16 | -0.1250 | -0.0089 | +0.1161 | FAIL | PASS |
| custom_pagie_p500 | `numpy` | 443.96 | 3298.32 | 0.13x | +2854.36 | -0.0052 | -0.0139 | -0.0087 | FAIL | FAIL |

Timing PASS means new ms/generation <= old; quality PASS means the new final best stays within 10% of the old final best.

## Verdict

**FAIL**

- timing: xor_p100/numpy new 58.262 ms/gen > old 36.188 ms/gen (speed-up 0.621x)
- timing: xor_p500/numpy new 360.438 ms/gen > old 165.540 ms/gen (speed-up 0.459x)
- timing: xor_p1000/numpy new 676.493 ms/gen > old 301.209 ms/gen (speed-up 0.445x)
- timing: xor3d_p100/numpy new 68.570 ms/gen > old 20.383 ms/gen (speed-up 0.297x)
- timing: xor3d_p500/numpy new 360.376 ms/gen > old 74.318 ms/gen (speed-up 0.206x)
- timing: xor3d_p1000/numpy new 736.407 ms/gen > old 187.243 ms/gen (speed-up 0.254x)
- timing: custom_pagie_p500/numpy new 3298.323 ms/gen > old 443.960 ms/gen (speed-up 0.135x)
- quality: xor_p100/numpy new final best -0.0303 < threshold -0.0109 (old -0.0099 - 10% * \|old\|)
- quality: xor_p1000/numpy new final best -0.0000 < threshold -0.0000 (old -0.0000 - 10% * \|old\|)
- quality: xor3d_p100/numpy new final best -0.2429 < threshold -0.1528 (old -0.1389 - 10% * \|old\|)
- quality: custom_pagie_p500/numpy new final best -0.0139 < threshold -0.0057 (old -0.0052 - 10% * \|old\|)

Sub-verdicts: timing `FAIL`, quality `FAIL` — timing compares measured ms/generation, quality compares final best fitness at a 10% tolerance.

## Why the new side wins and loses

Per case, derived from the measured `old / new` speed-up:

- `xor_p100` (pop 100, family func_fit, backend `numpy`): **new loses** — 0.62x of the old speed (old 36.19 ms/gen → new 58.26 ms/gen; the old side is 1.61x faster).
- `xor_p500` (pop 500, family func_fit, backend `numpy`): **new loses** — 0.46x of the old speed (old 165.54 ms/gen → new 360.44 ms/gen; the old side is 2.18x faster).
- `xor_p1000` (pop 1000, family func_fit, backend `numpy`): **new loses** — 0.45x of the old speed (old 301.21 ms/gen → new 676.49 ms/gen; the old side is 2.25x faster).
- `xor3d_p100` (pop 100, family func_fit, backend `numpy`): **new loses** — 0.30x of the old speed (old 20.38 ms/gen → new 68.57 ms/gen; the old side is 3.36x faster).
- `xor3d_p500` (pop 500, family func_fit, backend `numpy`): **new loses** — 0.21x of the old speed (old 74.32 ms/gen → new 360.38 ms/gen; the old side is 4.85x faster).
- `xor3d_p1000` (pop 1000, family func_fit, backend `numpy`): **new loses** — 0.25x of the old speed (old 187.24 ms/gen → new 736.41 ms/gen; the old side is 3.93x faster).
- `custom_pagie_p500` (pop 500, family func_fit, backend `numpy`): **new loses** — 0.13x of the old speed (old 443.96 ms/gen → new 3298.32 ms/gen; the old side is 7.43x faster).

By population size:

- pop 100: **new loses** on all 2 comparable run(s) — 0.30x–0.62x of the old speed (the old side is 1.61x–3.36x faster).
- pop 500: **new loses** on all 3 comparable run(s) — 0.13x–0.46x of the old speed (the old side is 2.18x–7.43x faster).
- pop 1000: **new loses** on all 2 comparable run(s) — 0.25x–0.45x of the old speed (the old side is 2.25x–3.93x faster).

**Where the new side loses:** at the populations and edge counts above, the same per-op Python-loop dispatch is no longer amortised by edge-proportional work, while the old side's XLA program fuses the whole generation into compiled kernels and pays that dispatch once.

This section is written from the same `speedup` values as the headline and old-vs-new tables; it does not assume the new side is the winner. A rerun that changes those numbers changes this text with them.

Where the new side lands relative to the declared contract:

- Contract at pop 100 (`xor_p100`): 51.00 ms/gen declared; measured old 36.19 ms/gen (0.71x declared); measured new 58.26 ms/gen (1.14x declared, i.e. the new side is slower than the contract here).
- Contract at pop 500 (`xor_p500`): 109.00 ms/gen declared; measured old 165.54 ms/gen (1.52x declared); measured new 360.44 ms/gen (3.31x declared, i.e. the new side is slower than the contract here).
- Contract at pop 1000 (`xor_p1000`): 157.00 ms/gen declared; measured old 301.21 ms/gen (1.92x declared); measured new 676.49 ms/gen (4.31x declared, i.e. the new side is slower than the contract here).

The declared contract (51/109/157 ms/gen at pop 100/500/1000) was measured on a 1-thread-bound box; this run used `--threads 0`, so the comparison to the contract is informational and the measured old row on this box is what the verdict uses.

## Quality curves

### xor_p100

| Implementation | Backend | Seed | Best @1 | Best @10 | Best @25% | Best @50% | Best @100% | Final | Gens to target |
|---|---|---|---|---|---|---|---|---|---|
| `old` tensorneat-legacy (JAX) | `jax` | 42 | -0.2507 | -0.2500 | -0.1929 | -0.0542 | -0.0000 | -0.0000 | 62 |
| `new` tensorneat (evoxc) | `numpy` | 42 | -0.2502 | -0.2500 | -0.2205 | -0.0067 | -0.0000 | -0.0000 | 55 |

### xor_p500

| Implementation | Backend | Seed | Best @1 | Best @10 | Best @25% | Best @50% | Best @100% | Final | Gens to target |
|---|---|---|---|---|---|---|---|---|---|
| `old` tensorneat-legacy (JAX) | `jax` | 42 | -0.2506 | -0.2477 | -0.1152 | -0.0055 | -0.0000 | -0.0000 | 41 |
| `new` tensorneat (evoxc) | `numpy` | 42 | -0.2500 | -0.2421 | -0.0235 | -0.0000 | -0.0000 | -0.0000 | 28 |

### xor_p1000

| Implementation | Backend | Seed | Best @1 | Best @10 | Best @25% | Best @50% | Best @100% | Final | Gens to target |
|---|---|---|---|---|---|---|---|---|---|
| `old` tensorneat-legacy (JAX) | `jax` | 42 | -0.2506 | -0.2427 | -0.0780 | -0.0004 | -0.0000 | -0.0000 | 35 |
| `new` tensorneat (evoxc) | `numpy` | 42 | -0.2501 | -0.2416 | -0.0391 | -0.0000 | -0.0000 | -0.0000 | 31 |

### xor3d_p100

| Implementation | Backend | Seed | Best @1 | Best @10 | Best @25% | Best @50% | Best @100% | Final | Gens to target |
|---|---|---|---|---|---|---|---|---|---|
| `old` tensorneat-legacy (JAX) | `jax` | 42 | -0.2524 | -0.2500 | -0.2500 | -0.2447 | -0.0278 | -0.0278 | 149 |
| `new` tensorneat (evoxc) | `numpy` | 42 | -0.2515 | -0.2500 | -0.2500 | -0.2500 | -0.2358 | -0.2358 | >152 |

### xor3d_p500

| Implementation | Backend | Seed | Best @1 | Best @10 | Best @25% | Best @50% | Best @100% | Final | Gens to target |
|---|---|---|---|---|---|---|---|---|---|
| `old` tensorneat-legacy (JAX) | `jax` | 42 | -0.2524 | -0.2500 | -0.2500 | -0.2500 | -0.2500 | -0.2500 | >153 |
| `new` tensorneat (evoxc) | `numpy` | 42 | -0.2502 | -0.2500 | -0.2500 | -0.2500 | -0.0072 | -0.0072 | 140 |

### xor3d_p1000

| Implementation | Backend | Seed | Best @1 | Best @10 | Best @25% | Best @50% | Best @100% | Final | Gens to target |
|---|---|---|---|---|---|---|---|---|---|
| `old` tensorneat-legacy (JAX) | `jax` | 42 | -0.2524 | -0.2500 | -0.1850 | -0.0192 | -0.0000 | -0.0000 | 62 |
| `new` tensorneat (evoxc) | `numpy` | 42 | -0.2501 | -0.2500 | -0.1789 | -0.0195 | -0.0179 | -0.0179 | 59 |

### custom_pagie_p500

| Implementation | Backend | Seed | Best @1 | Best @10 | Best @25% | Best @50% | Best @100% | Final | Gens to target |
|---|---|---|---|---|---|---|---|---|---|
| `old` tensorneat-legacy (JAX) | `jax` | 42 | -0.0490 | -0.0473 | -0.0347 | -0.0233 | -0.0064 | -0.0064 | 125 |
| `new` tensorneat (evoxc) | `numpy` | 42 | -0.0461 | -0.0459 | -0.0332 | -0.0233 | -0.0221 | -0.0221 | >152 |

### cartpole_p100

| Implementation | Backend | Seed | Best @1 | Best @10 | Best @25% | Best @50% | Best @100% | Final | Gens to target |
|---|---|---|---|---|---|---|---|---|---|
| `new` tensorneat (evoxc) | `numpy` | 42 | 200.0000 | 200.0000 | 200.0000 | 200.0000 | 200.0000 | 200.0000 | 1 |

Percent columns are generation `round(fraction × N)` of the N recorded generations (warmup + timed — the budget `generations_to_target` is indexed against), and `>N` means the case's fitness target was not reached within those N recorded generations. Each row is the first ok run recorded for that case/implementation/backend.

## Reference numbers (declared baseline)

| Case | Pop | Declared legacy ms/gen | Measured old ms/gen | Ratio (measured/declared) |
|---|---|---|---|---|
| xor_p100 | 100 | 51.00 | 36.19 | 0.71x |
| xor_p500 | 500 | 109.00 | 165.54 | 1.52x |
| xor_p1000 | 1000 | 157.00 | 301.21 | 1.92x |

The performance contract in `../CONTEXT.md` declares legacy CPU numbers measured on a
1-thread-bound box (51 ms/gen at pop 100, 109 ms/gen at pop 500, 157 ms/gen at pop 1000).
The generator prints those next to the old row measured on this box; the verdict uses the
measured row.

## What is not compared

- **Same workload, not the same search.** The sides use different PRNGs (JAX `PRNGKey` vs evoxc splitmix64), so one seed identifies the same *workload*, not the same trajectory or a bitwise-equal population. Numerical fidelity is `test/parity/`'s job: it compares kernels against `benchmarks/legacy_jax`, and `pytest test/parity --collect-only -q` collects 53 tests on this box.
- **Accelerator numbers are informational.** The contract is CPU; `iree`/`xla`/`tvm` rows are reported when their runtime is installed and are never part of the old-vs-new verdict.
- **CartPole has no old-side row.** The frozen library's RL problems need gym/brax/mujoco, so `cartpole_p100` is new-only and marked `n/a` for old.
- **Import time, visualization and code generation are not timed.** Only `rss_after_import_mb` is recorded for the import phase; `Pipeline.show`, network drawing and sympy codegen sit outside every timed region.
- **No trajectory equality is expected.** The two sides evolve different populations from the same workload seed; parity of kernels (not of search) is what `test/parity/` verifies.

## How to reproduce

```bash
# regenerate the committed report: the literal command of this run
python benchmarks/run_benchmark.py --impl both --backend numpy --cases all --generations 150 --warmup 3 --repeats 2

# full matrix, both implementations on the primary backend
python benchmarks/run_benchmark.py --impl both --backend numpy --cases all --generations 150 --warmup 3 --repeats 2 --threads 0

# baseline only (frozen JAX)
python benchmarks/run_benchmark.py --impl old --cases all --generations 150 --warmup 3 --repeats 2 --threads 0

# candidate only, one backend at a time
python benchmarks/run_benchmark.py --impl new --backend numpy --cases all --generations 150 --warmup 3 --repeats 2 --threads 0
python benchmarks/run_benchmark.py --impl new --backend iree --cases xor_p500 --generations 150 --warmup 3 --repeats 2 --threads 0   # informational; errors cleanly when the IREE runtime is absent

# inspect the matrix and the planned worker commands (no run, no report)
python benchmarks/run_benchmark.py --list
python benchmarks/run_benchmark.py --dry-run
```

Interpreter used: /tmp/tn-venv/bin/python

Cases: xor_p100, xor_p500, xor_p1000, xor3d_p100, xor3d_p500, xor3d_p1000, custom_pagie_p500, cartpole_p100. Timed generations: 150, warmup generations (untimed): 3, repeats: 2 (seeds 42, 43), threads: runtime defaults (`--threads 0`). Backend(s) of the new side: numpy; the old side always runs on JAX, once per case. `--out` defaults to `benchmarks/RESULTS.md` and `--json` to `benchmarks/results/latest.json` (git-ignored); the report is regenerated in place from that JSON.

Generated from `benchmarks/results/latest.json` into `benchmarks/RESULTS.md` at 2026-11-02T05:00:13+00:00 — do not hand-edit.
