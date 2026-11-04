# TensorNEAT benchmark — evoxc rewrite vs frozen JAX baseline

> **GENERATED FILE.** Produced by `python benchmarks/run_benchmark.py`, rendered from
> `benchmarks/results/authoritative.json` at 2026-11-04T17:43:10+00:00. It must not be hand-edited:
> rerun the CLI instead, or the numbers below stop matching the recorded evidence.

## Headline: old vs new

| Case | Pop | Backend | Old ms/gen | New ms/gen | Speed-up | Speed-up range (best–worst) | Quality overlap | Old compile (s) | New compile (s) | Old peak RSS (MiB) | New peak RSS (MiB) | Old final best | New final best | Gens to target (old) | Gens to target (new) | Timing | Quality |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| xor_p100 | 100 | `numpy` | 43.51 | 37.73 | 1.15x | 1.02x–1.30x | inside | 2.916 | 0.303 | 677.6 | 115.1 | -0.0017 | -0.0095 | 78 | 51 | PASS | FAIL |
| xor_p500 | 500 | `numpy` | 268.12 | 159.16 | 1.68x | 1.48x–1.89x | new worse | 2.892 | 0.520 | 717.8 | 241.0 | -0.0000 | -0.0000 | 47 | 36 | PASS | PASS |
| xor_p1000 | 1000 | `numpy` | 816.69 | 292.33 | 2.79x | 2.71x–2.88x | new worse | 2.899 | 0.810 | 714.7 | 392.3 | -0.0000 | -0.0000 | 32 | 30 | PASS | PASS |
| xor3d_p100 | 100 | `numpy` | 24.58 | 40.60 | 0.61x | 0.59x–0.62x | new worse | 3.127 | 0.313 | 685.3 | 119.7 | -0.0751 | -0.2500 | 149 | >152 | FAIL | FAIL |
| xor3d_p500 | 500 | `numpy` | 99.83 | 169.92 | 0.59x | 0.51x–0.67x | inside | 3.061 | 0.536 | 746.2 | 265.5 | -0.2144 | -0.2500 | >153 | >152 | FAIL | FAIL |
| xor3d_p1000 | 1000 | `numpy` | 282.54 | 320.74 | 0.88x | 0.55x–1.24x | new worse | 2.982 | 0.785 | 743.1 | 441.2 | -0.0000 | -0.0001 | 75 | 113 | FAIL | PASS |
| custom_pagie_p500 | 500 | `numpy` | 526.71 | 3148.63 | 0.17x | 0.15x–0.18x | new worse | 2.990 | 2.221 | 861.1 | 4576.9 | -0.0052 | -0.0219 | 121 | >152 | FAIL | FAIL |
| cartpole_p100 | 100 | `numpy` | n/a (no old-side equivalent) | 386.07 | n/a (needs both sides) | n/a (needs both sides) | n/a (needs both sides) | n/a (no old-side equivalent) | 0.558 | n/a (no old-side equivalent) | 110.3 | n/a (no old-side equivalent) | 200.0000 | n/a (no old-side equivalent) | 2 | n/a | n/a |

One row per case and new-side backend (the old side always runs on JAX). Speed-up is `old / new` per generation: above 1.00x the new side is faster. Speed-up range is `old_min/new_max` … `old_max/new_min` over the per-repeat medians, i.e. the best and worst case over the repeats. Quality overlap states where the new side's per-repeat final-best range sits against the old side's: `inside` when the ranges intersect (every new repeat inside the old range, or the ranges partially overlap), `new worse` when every new repeat is below the old range, `new better` when every new repeat is above it — an observation, never a replacement for the verdict. Timing PASS means new ms/generation is at or below the old one; quality PASS means the new final best stays within `max(10% × |old|, 0.001)` of the old one. Every cell comes from the aggregated JSON records of this run; `n/a (reason)` marks a cell that could not be reached (not run, skipped, error, or not recorded).

## Environment

| Item | Value |
|---|---|
| Date (UTC) | 2026-11-04T17:43:10+00:00 |
| Python | 3.12.3 |
| Platform | Linux-6.8.0-146-generic-x86_64-with-glibc2.39 |
| CPU model | INTEL(R) XEON(R) PLATINUM 8568Y+ |
| CPU cores (`os.cpu_count`) | 20 |
| Memory | 241611 MiB |
| Load average (1/5/15 min) | 2.85 / 1.96 / 1.95 (1/5/15 min, document build); workers 1.17–4.73 at start, 1.17–4.73 at end (1-min, 30 run(s)) |
| numpy / jax / evoxc / tensorneat (installed metadata) | numpy 2.5.4 / jax 0.10.2 / evoxc 0.1.0 / tensorneat 0.2.0 |
| Git commit | `1f626e02b46c` (clean — generated report artifacts are not counted) — `benchmarks/legacy_jax` and `src/` both come from this checkout |
| Thread pinning (`--threads`) | `--threads 0` — runtime defaults (as users run it), nothing exported |
| Thread env (set/unset) | OMP_NUM_THREADS unset · MKL_NUM_THREADS unset · OPENBLAS_NUM_THREADS unset · NUMEXPR_NUM_THREADS unset · XLA_FLAGS unset (CLI process, inherited unchanged by every worker — `--threads 0` exports nothing) |

**Shared box.** These numbers were measured on a machine that was not exclusively ours: other jobs ran on it while the workers timed generations, so the per-repeat spread (the headline's speed-up range and the per-repeat quality-curve rows) is the honest uncertainty measure — a median alone would hide it.

## Protocol

| Parameter | Value |
|---|---|
| Backend(s) of the new implementation | `numpy` |
| Warmup generations (untimed) | 3 |
| Timed generations | 150 |
| Repeats (seed + r) | 2 (`seed + r`, seeds 42, 43) |
| Thread pinning | `--threads 0` — runtime defaults (as users run it), nothing exported |
| New-side build | inside the first `Pipeline.step` call (the evoxc build); the generation it consumes is recorded as `generations_consumed_before_timing`, excluded from the timed samples and subtracted from the untimed warmup, so both sides' timing samples cover the same generation indices |
| Compile time | reported separately as `compile_time_s` (old: `jax.jit(pipeline.step).lower(state).compile()`; new: the first `Pipeline.step` build) and never charged to a generation |
| Scheduling | one worker process per run, strictly sequential — the two implementations never share a process |

Metrics: `compile_time_s` (one-off build), `ms_per_generation` (median wall clock per generation
after warmup, including host sync), `gens_per_second`, `rss_final_mb`, and quality
(`final_best_fitness`, `generations_to_target`, best-fitness curve).
The build generation is also consumed from the new side's quality curve, so that curve starts one generation later than the old side's and records 152 generations against the old side's 153 — which is why an unreached target reads `>152` for new and `>153` for old. Compile/build seconds are never charged to a generation.
Definitions, fairness rules and verdict rules live in `benchmarks/CONTEXT.md`.
A thread-bound sensitivity run (`--threads 1`, XOR family, same CLI) is recorded in `benchmarks/THREADS.md` next to this report.

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
| `old` tensorneat-legacy (JAX) | `jax` | 2.916 | 43.51 | 22.99 | 677.6 | -0.0017 | 78 | 2 |
| `new` tensorneat (evoxc) | `numpy` | 0.303 | 37.73 | 26.50 | 115.1 | -0.0095 | 51 | 2 |

### xor_p500

| Implementation | Backend | Compile (s) | ms/generation | generations/s | peak RSS (MiB) | Final best | Gens to target | Repeats |
|---|---|---|---|---|---|---|---|---|
| `old` tensorneat-legacy (JAX) | `jax` | 2.892 | 268.12 | 3.73 | 717.8 | -0.0000 | 47 | 2 |
| `new` tensorneat (evoxc) | `numpy` | 0.520 | 159.16 | 6.28 | 241.0 | -0.0000 | 36 | 2 |

### xor_p1000

| Implementation | Backend | Compile (s) | ms/generation | generations/s | peak RSS (MiB) | Final best | Gens to target | Repeats |
|---|---|---|---|---|---|---|---|---|
| `old` tensorneat-legacy (JAX) | `jax` | 2.899 | 816.69 | 1.22 | 714.7 | -0.0000 | 32 | 2 |
| `new` tensorneat (evoxc) | `numpy` | 0.810 | 292.33 | 3.42 | 392.3 | -0.0000 | 30 | 2 |

### xor3d_p100

| Implementation | Backend | Compile (s) | ms/generation | generations/s | peak RSS (MiB) | Final best | Gens to target | Repeats |
|---|---|---|---|---|---|---|---|---|
| `old` tensorneat-legacy (JAX) | `jax` | 3.127 | 24.58 | 40.68 | 685.3 | -0.0751 | 149 | 2 |
| `new` tensorneat (evoxc) | `numpy` | 0.313 | 40.60 | 24.63 | 119.7 | -0.2500 | >152 | 2 |

### xor3d_p500

| Implementation | Backend | Compile (s) | ms/generation | generations/s | peak RSS (MiB) | Final best | Gens to target | Repeats |
|---|---|---|---|---|---|---|---|---|
| `old` tensorneat-legacy (JAX) | `jax` | 3.061 | 99.83 | 10.02 | 746.2 | -0.2144 | >153 | 2 |
| `new` tensorneat (evoxc) | `numpy` | 0.536 | 169.92 | 5.89 | 265.5 | -0.2500 | >152 | 2 |

### xor3d_p1000

| Implementation | Backend | Compile (s) | ms/generation | generations/s | peak RSS (MiB) | Final best | Gens to target | Repeats |
|---|---|---|---|---|---|---|---|---|
| `old` tensorneat-legacy (JAX) | `jax` | 2.982 | 282.54 | 3.54 | 743.1 | -0.0000 | 75 | 2 |
| `new` tensorneat (evoxc) | `numpy` | 0.785 | 320.74 | 3.12 | 441.2 | -0.0001 | 113 | 2 |

### custom_pagie_p500

| Implementation | Backend | Compile (s) | ms/generation | generations/s | peak RSS (MiB) | Final best | Gens to target | Repeats |
|---|---|---|---|---|---|---|---|---|
| `old` tensorneat-legacy (JAX) | `jax` | 2.990 | 526.71 | 1.90 | 861.1 | -0.0052 | 121 | 2 |
| `new` tensorneat (evoxc) | `numpy` | 2.221 | 3148.63 | 0.32 | 4576.9 | -0.0219 | >152 | 2 |

### cartpole_p100

| Implementation | Backend | Compile (s) | ms/generation | generations/s | peak RSS (MiB) | Final best | Gens to target | Repeats |
|---|---|---|---|---|---|---|---|---|
| `new` tensorneat (evoxc) | `numpy` | 0.558 | 386.07 | 2.59 | 110.3 | 200.0000 | 2 | 2 |

## Old vs new

| Case | Backend | Old ms/gen | New ms/gen | Speed-up | Delta ms/gen | Old final best | New final best | Quality delta | Timing | Quality |
|---|---|---|---|---|---|---|---|---|---|---|
| xor_p100 | `numpy` | 43.51 | 37.73 | 1.15x | -5.77 | -0.0017 | -0.0095 | -0.0078 | PASS | FAIL |
| xor_p500 | `numpy` | 268.12 | 159.16 | 1.68x | -108.97 | -0.0000 | -0.0000 | -0.0000 | PASS | PASS |
| xor_p1000 | `numpy` | 816.69 | 292.33 | 2.79x | -524.36 | -0.0000 | -0.0000 | -0.0000 | PASS | PASS |
| xor3d_p100 | `numpy` | 24.58 | 40.60 | 0.61x | +16.01 | -0.0751 | -0.2500 | -0.1749 | FAIL | FAIL |
| xor3d_p500 | `numpy` | 99.83 | 169.92 | 0.59x | +70.09 | -0.2144 | -0.2500 | -0.0356 | FAIL | FAIL |
| xor3d_p1000 | `numpy` | 282.54 | 320.74 | 0.88x | +38.20 | -0.0000 | -0.0001 | -0.0001 | FAIL | PASS |
| custom_pagie_p500 | `numpy` | 526.71 | 3148.63 | 0.17x | +2621.91 | -0.0052 | -0.0219 | -0.0166 | FAIL | FAIL |

Timing PASS means new ms/generation <= old; quality PASS means the new final best stays within the active tolerance `max(10% × |old|, 0.001)` of the old final best (the absolute floor keeps two runs that are both at the optimum from failing on float noise).

## Verdict

**FAIL**

- timing: xor3d_p100/numpy new 40.598 ms/gen > old 24.583 ms/gen (speed-up 0.606x)
- timing: xor3d_p500/numpy new 169.923 ms/gen > old 99.834 ms/gen (speed-up 0.588x)
- timing: xor3d_p1000/numpy new 320.738 ms/gen > old 282.542 ms/gen (speed-up 0.881x)
- timing: custom_pagie_p500/numpy new 3148.627 ms/gen > old 526.713 ms/gen (speed-up 0.167x)
- quality: xor_p100/numpy new final best -0.0095 < threshold -0.0027 (old -0.0017 - tolerance 0.0010, i.e. max(10% × \|old\|, 0.001 absolute floor))
- quality: xor3d_p100/numpy new final best -0.2500 < threshold -0.0826 (old -0.0751 - tolerance 0.0075, i.e. max(10% × \|old\|, 0.001 absolute floor))
- quality: xor3d_p500/numpy new final best -0.2500 < threshold -0.2359 (old -0.2144 - tolerance 0.0214, i.e. max(10% × \|old\|, 0.001 absolute floor))
- quality: custom_pagie_p500/numpy new final best -0.0219 < threshold -0.0062 (old -0.0052 - tolerance 0.0010, i.e. max(10% × \|old\|, 0.001 absolute floor))

Sub-verdicts: timing `FAIL`, quality `FAIL` — timing compares measured ms/generation, quality passes when `new >= old - max(10% × |old|, 0.001)`.

## Why the new side wins and loses

Per case, derived from the measured `old / new` speed-up:

- `xor_p100` (pop 100, family func_fit, backend `numpy`): **new wins** — 1.15x faster than the old side (old 43.51 ms/gen → new 37.73 ms/gen).
- `xor_p500` (pop 500, family func_fit, backend `numpy`): **new wins** — 1.68x faster than the old side (old 268.12 ms/gen → new 159.16 ms/gen).
- `xor_p1000` (pop 1000, family func_fit, backend `numpy`): **new wins** — 2.79x faster than the old side (old 816.69 ms/gen → new 292.33 ms/gen).
- `xor3d_p100` (pop 100, family func_fit, backend `numpy`): **new loses** — 0.61x of the old speed (old 24.58 ms/gen → new 40.60 ms/gen; the old side is 1.65x faster).
- `xor3d_p500` (pop 500, family func_fit, backend `numpy`): **new loses** — 0.59x of the old speed (old 99.83 ms/gen → new 169.92 ms/gen; the old side is 1.70x faster).
- `xor3d_p1000` (pop 1000, family func_fit, backend `numpy`): **new loses** — 0.88x of the old speed (old 282.54 ms/gen → new 320.74 ms/gen; the old side is 1.14x faster).
- `custom_pagie_p500` (pop 500, family func_fit, backend `numpy`): **new loses** — 0.17x of the old speed (old 526.71 ms/gen → new 3148.63 ms/gen; the old side is 5.98x faster).

By population size:

- pop 100: **mixed** — new wins 1 of 2 comparable run(s) — speed-up range 0.61x–1.15x.
- pop 500: **mixed** — new wins 1 of 3 comparable run(s) — speed-up range 0.17x–1.68x.
- pop 1000: **mixed** — new wins 1 of 2 comparable run(s) — speed-up range 0.88x–2.79x.

**Where the new side wins:** the evoxc graph is edge-proportional and depth-parallel — work scales with the real connections and the network depth instead of `max_nodes × max_nodes` padding — and on the numpy reference interpreter that useful work amortises its per-op Python-loop dispatch, while XLA's compiled whole-generation program carries more fixed overhead than it saves at small populations and low edge counts.

**Where the new side loses:** at the populations and edge counts above, the same per-op Python-loop dispatch is no longer amortised by edge-proportional work, while the old side's XLA program fuses the whole generation into compiled kernels and pays that dispatch once.

**Memory.** Peak RSS (`rss_final_mb`, the timed worker's high-water mark, taken as the max over the repeats) from the same runs, new vs old:

- `custom_pagie_p500` (pop 500, `numpy`): 4576.9 MiB vs 861.1 MiB — 5.32x (the new side uses more memory).
- `xor3d_p1000` (pop 1000, `numpy`): 441.2 MiB vs 743.1 MiB — 0.59x (the new side uses less memory).
- `xor_p1000` (pop 1000, `numpy`): 392.3 MiB vs 714.7 MiB — 0.55x (the new side uses less memory).
- `xor3d_p500` (pop 500, `numpy`): 265.5 MiB vs 746.2 MiB — 0.36x (the new side uses less memory).
- `xor_p500` (pop 500, `numpy`): 241.0 MiB vs 717.8 MiB — 0.34x (the new side uses less memory).
- `xor3d_p100` (pop 100, `numpy`): 119.7 MiB vs 685.3 MiB — 0.17x (the new side uses less memory).
- `xor_p100` (pop 100, `numpy`): 115.1 MiB vs 677.6 MiB — 0.17x (the new side uses less memory).

Worst case: `custom_pagie_p500` at 5.32x (4576.9 vs 861.1 MiB old, +3716 MiB).
The new side's peak grows with the population in the XOR family (115.1 → 241.0 → 392.3 MiB at pop 100, 500, 1000), while the old row moves only from 677.6 to 717.8 MiB.

Mechanism, as far as these runs support one: the old side's floor is the JAX/XLA runtime itself (182.2–182.3 MiB right after import); the new side starts from 73.6–73.8 MiB. Beyond those floors the new peak tracks the population while the old peak stays close to its runtime floor, so the working set the numpy interpreter materialises per generation is what grows with `pop_size × max_nodes × max_conns`; these runs do not attribute that growth to a single dominant term.

This section is written from the same `speedup` values as the headline and old-vs-new tables; it does not assume the new side is the winner. A rerun that changes those numbers changes this text with them.

Where the new side lands relative to the declared contract:

- Contract at pop 100 (`xor_p100`): 51.00 ms/gen declared; measured old 43.51 ms/gen (0.85x declared); measured new 37.73 ms/gen (0.74x declared).
- Contract at pop 500 (`xor_p500`): 109.00 ms/gen declared; measured old 268.12 ms/gen (2.46x declared); measured new 159.16 ms/gen (1.46x declared, i.e. the new side is slower than the contract here).
- Contract at pop 1000 (`xor_p1000`): 157.00 ms/gen declared; measured old 816.69 ms/gen (5.20x declared); measured new 292.33 ms/gen (1.86x declared, i.e. the new side is slower than the contract here).

The declared contract (51/109/157 ms/gen at pop 100/500/1000) was measured on a 1-thread-bound box; this run used `--threads 0`, so the comparison to the contract is informational and the measured old row on this box is what the verdict uses.

## Quality curves

### xor_p100

| Implementation | Backend | Seed | Best @1 | Best @10 | Best @25% | Best @50% | Best @100% | Final | Gens to target |
|---|---|---|---|---|---|---|---|---|---|
| `old` tensorneat-legacy (JAX) | `jax` | 42 | -0.2507 | -0.2500 | -0.1929 | -0.0542 | -0.0000 | -0.0000 | 62 |
| `old` tensorneat-legacy (JAX) | `jax` | 43 | -0.2519 | -0.2500 | -0.2364 | -0.1515 | -0.0034 | -0.0034 | 93 |
| `old` tensorneat-legacy (JAX) — aggregate | `jax` | 42, 43 | -0.2513 | -0.2500 | -0.2147 | -0.1028 | -0.0017 | -0.0017 | 78 |
| `new` tensorneat (evoxc) | `numpy` | 42 | -0.2508 | -0.2482 | -0.0428 | -0.0000 | -0.0000 | -0.0000 | 33 |
| `new` tensorneat (evoxc) | `numpy` | 43 | -0.2503 | -0.2500 | -0.2500 | -0.0596 | -0.0189 | -0.0189 | 68 |
| `new` tensorneat (evoxc) — aggregate | `numpy` | 42, 43 | -0.2505 | -0.2491 | -0.1464 | -0.0298 | -0.0095 | -0.0095 | 51 |

### xor_p500

| Implementation | Backend | Seed | Best @1 | Best @10 | Best @25% | Best @50% | Best @100% | Final | Gens to target |
|---|---|---|---|---|---|---|---|---|---|
| `old` tensorneat-legacy (JAX) | `jax` | 42 | -0.2506 | -0.2477 | -0.1203 | -0.0199 | -0.0000 | -0.0000 | 42 |
| `old` tensorneat-legacy (JAX) | `jax` | 43 | -0.2505 | -0.2453 | -0.1461 | -0.0002 | -0.0000 | -0.0000 | 51 |
| `old` tensorneat-legacy (JAX) — aggregate | `jax` | 42, 43 | -0.2505 | -0.2465 | -0.1332 | -0.0101 | -0.0000 | -0.0000 | 47 |
| `new` tensorneat (evoxc) | `numpy` | 42 | -0.2500 | -0.2161 | -0.0847 | -0.0001 | -0.0000 | -0.0000 | 36 |
| `new` tensorneat (evoxc) | `numpy` | 43 | -0.2500 | -0.2500 | -0.0796 | -0.0000 | -0.0000 | -0.0000 | 35 |
| `new` tensorneat (evoxc) — aggregate | `numpy` | 42, 43 | -0.2500 | -0.2330 | -0.0821 | -0.0000 | -0.0000 | -0.0000 | 36 |

### xor_p1000

| Implementation | Backend | Seed | Best @1 | Best @10 | Best @25% | Best @50% | Best @100% | Final | Gens to target |
|---|---|---|---|---|---|---|---|---|---|
| `old` tensorneat-legacy (JAX) | `jax` | 42 | -0.2506 | -0.2427 | -0.0762 | -0.0005 | -0.0000 | -0.0000 | 35 |
| `old` tensorneat-legacy (JAX) | `jax` | 43 | -0.2505 | -0.2376 | -0.0234 | -0.0000 | -0.0000 | -0.0000 | 29 |
| `old` tensorneat-legacy (JAX) — aggregate | `jax` | 42, 43 | -0.2505 | -0.2402 | -0.0498 | -0.0003 | -0.0000 | -0.0000 | 32 |
| `new` tensorneat (evoxc) | `numpy` | 42 | -0.2501 | -0.2450 | -0.0255 | -0.0000 | -0.0000 | -0.0000 | 28 |
| `new` tensorneat (evoxc) | `numpy` | 43 | -0.2500 | -0.2486 | -0.0674 | -0.0017 | -0.0000 | -0.0000 | 31 |
| `new` tensorneat (evoxc) — aggregate | `numpy` | 42, 43 | -0.2500 | -0.2468 | -0.0464 | -0.0009 | -0.0000 | -0.0000 | 30 |

### xor3d_p100

| Implementation | Backend | Seed | Best @1 | Best @10 | Best @25% | Best @50% | Best @100% | Final | Gens to target |
|---|---|---|---|---|---|---|---|---|---|
| `old` tensorneat-legacy (JAX) | `jax` | 42 | -0.2524 | -0.2500 | -0.2500 | -0.2447 | -0.0278 | -0.0278 | 149 |
| `old` tensorneat-legacy (JAX) | `jax` | 43 | -0.2547 | -0.2500 | -0.2500 | -0.2500 | -0.1224 | -0.1224 | >153 |
| `old` tensorneat-legacy (JAX) — aggregate | `jax` | 42, 43 | -0.2535 | -0.2500 | -0.2500 | -0.2473 | -0.0751 | -0.0751 | 149 |
| `new` tensorneat (evoxc) | `numpy` | 42 | -0.2516 | -0.2500 | -0.2500 | -0.2500 | -0.2500 | -0.2500 | >152 |
| `new` tensorneat (evoxc) | `numpy` | 43 | -0.2503 | -0.2500 | -0.2500 | -0.2500 | -0.2500 | -0.2500 | >152 |
| `new` tensorneat (evoxc) — aggregate | `numpy` | 42, 43 | -0.2509 | -0.2500 | -0.2500 | -0.2500 | -0.2500 | -0.2500 | >152 |

### xor3d_p500

| Implementation | Backend | Seed | Best @1 | Best @10 | Best @25% | Best @50% | Best @100% | Final | Gens to target |
|---|---|---|---|---|---|---|---|---|---|
| `old` tensorneat-legacy (JAX) | `jax` | 42 | -0.2524 | -0.2500 | -0.2500 | -0.2500 | -0.2500 | -0.2500 | >153 |
| `old` tensorneat-legacy (JAX) | `jax` | 43 | -0.2507 | -0.2500 | -0.2500 | -0.2500 | -0.1789 | -0.1789 | >153 |
| `old` tensorneat-legacy (JAX) — aggregate | `jax` | 42, 43 | -0.2516 | -0.2500 | -0.2500 | -0.2500 | -0.2144 | -0.2144 | >153 |
| `new` tensorneat (evoxc) | `numpy` | 42 | -0.2502 | -0.2500 | -0.2500 | -0.2500 | -0.2500 | -0.2500 | >152 |
| `new` tensorneat (evoxc) | `numpy` | 43 | -0.2503 | -0.2500 | -0.2500 | -0.2500 | -0.2500 | -0.2500 | >152 |
| `new` tensorneat (evoxc) — aggregate | `numpy` | 42, 43 | -0.2503 | -0.2500 | -0.2500 | -0.2500 | -0.2500 | -0.2500 | >152 |

### xor3d_p1000

| Implementation | Backend | Seed | Best @1 | Best @10 | Best @25% | Best @50% | Best @100% | Final | Gens to target |
|---|---|---|---|---|---|---|---|---|---|
| `old` tensorneat-legacy (JAX) | `jax` | 42 | -0.2524 | -0.2500 | -0.1850 | -0.0175 | -0.0000 | -0.0000 | 64 |
| `old` tensorneat-legacy (JAX) | `jax` | 43 | -0.2507 | -0.2500 | -0.2449 | -0.0794 | -0.0000 | -0.0000 | 86 |
| `old` tensorneat-legacy (JAX) — aggregate | `jax` | 42, 43 | -0.2515 | -0.2500 | -0.2149 | -0.0484 | -0.0000 | -0.0000 | 75 |
| `new` tensorneat (evoxc) | `numpy` | 42 | -0.2501 | -0.2500 | -0.2500 | -0.2500 | -0.0002 | -0.0002 | 135 |
| `new` tensorneat (evoxc) | `numpy` | 43 | -0.2502 | -0.2500 | -0.2386 | -0.1039 | -0.0001 | -0.0001 | 90 |
| `new` tensorneat (evoxc) — aggregate | `numpy` | 42, 43 | -0.2501 | -0.2500 | -0.2443 | -0.1770 | -0.0001 | -0.0001 | 113 |

### custom_pagie_p500

| Implementation | Backend | Seed | Best @1 | Best @10 | Best @25% | Best @50% | Best @100% | Final | Gens to target |
|---|---|---|---|---|---|---|---|---|---|
| `old` tensorneat-legacy (JAX) | `jax` | 42 | -0.0490 | -0.0473 | -0.0347 | -0.0233 | -0.0064 | -0.0064 | 125 |
| `old` tensorneat-legacy (JAX) | `jax` | 43 | -0.0493 | -0.0477 | -0.0243 | -0.0221 | -0.0040 | -0.0040 | 117 |
| `old` tensorneat-legacy (JAX) — aggregate | `jax` | 42, 43 | -0.0491 | -0.0475 | -0.0295 | -0.0227 | -0.0052 | -0.0052 | 121 |
| `new` tensorneat (evoxc) | `numpy` | 42 | -0.0461 | -0.0460 | -0.0240 | -0.0221 | -0.0219 | -0.0219 | >152 |
| `new` tensorneat (evoxc) | `numpy` | 43 | -0.0461 | -0.0460 | -0.0236 | -0.0220 | -0.0218 | -0.0218 | >152 |
| `new` tensorneat (evoxc) — aggregate | `numpy` | 42, 43 | -0.0461 | -0.0460 | -0.0238 | -0.0221 | -0.0219 | -0.0219 | >152 |

### cartpole_p100

| Implementation | Backend | Seed | Best @1 | Best @10 | Best @25% | Best @50% | Best @100% | Final | Gens to target |
|---|---|---|---|---|---|---|---|---|---|
| `new` tensorneat (evoxc) | `numpy` | 42 | 145.0000 | 200.0000 | 200.0000 | 200.0000 | 200.0000 | 200.0000 | 2 |
| `new` tensorneat (evoxc) | `numpy` | 43 | 200.0000 | 200.0000 | 200.0000 | 200.0000 | 200.0000 | 200.0000 | 1 |
| `new` tensorneat (evoxc) — aggregate | `numpy` | 42, 43 | 172.5000 | 200.0000 | 200.0000 | 200.0000 | 200.0000 | 200.0000 | 2 |

Percent columns are generation `round(fraction × N)` of the N recorded generations (warmup + timed — the budget `generations_to_target` is indexed against), and `>N` means the case's fitness target was not reached within those N recorded generations. Each repeat (seed) has its own row, and the `— aggregate` row is the element-wise median of those repeats: a bimodal case (one seed at the target, one not) shows up as two different `Final` values rather than one average.

## Reference numbers (declared baseline)

| Case | Pop | Declared legacy ms/gen | Measured old ms/gen | Ratio (measured/declared) |
|---|---|---|---|---|
| xor_p100 | 100 | 51.00 | 43.51 | 0.85x |
| xor_p500 | 500 | 109.00 | 268.12 | 2.46x |
| xor_p1000 | 1000 | 157.00 | 816.69 | 5.20x |

The performance contract in `../CONTEXT.md` declares legacy CPU numbers measured on a
1-thread-bound box (51 ms/gen at pop 100, 109 ms/gen at pop 500, 157 ms/gen at pop 1000).
The generator prints those next to the old row measured on this box; the verdict uses the
measured row.

## What is not compared

- **Same workload, not the same search.** The sides use different PRNGs (JAX `PRNGKey` vs evoxc splitmix64), so one seed identifies the same *workload*, not the same trajectory or a bitwise-equal population. Numerical fidelity is `test/parity/`'s job: it compares kernels against `benchmarks/legacy_jax`, and `pytest test/parity --collect-only -q` collects 53 tests on this box.
- **No accelerator row in this report.** The old-vs-new verdict is CPU-only and numpy-only by protocol, so the matrix here is the numpy reference interpreter; an `iree`/`xla`/`tvm` row is informational and never enters the verdict. The compiled leg is measured separately in `benchmarks/BACKENDS.md`: `iree` lowers, compiles and runs every shipped case, and that file carries the measured per-generation comparison with numpy — an accelerator row exists, it simply stays outside this verdict.
- **CartPole has no old-side row.** The frozen library's RL problems need gym/brax/mujoco, so `cartpole_p100` is new-only and marked `n/a` for old.
- **Import time, visualization and code generation are not timed.** Only `rss_after_import_mb` is recorded for the import phase; `Pipeline.show`, network drawing and sympy codegen sit outside every timed region.
- **No trajectory equality is expected.** The two sides evolve different populations from the same workload seed; parity of kernels (not of search) is what `test/parity/` verifies.

## How to reproduce

```bash
# regenerate the committed report: the literal command of this run
python benchmarks/run_benchmark.py --generations 150 --warmup 3 --repeats 2 --json benchmarks/results/authoritative.json

# full matrix, both implementations on the primary backend
python benchmarks/run_benchmark.py --impl both --backend numpy --cases all --generations 150 --warmup 3 --repeats 2 --threads 0

# baseline only (frozen JAX)
python benchmarks/run_benchmark.py --impl old --cases all --generations 150 --warmup 3 --repeats 2 --threads 0

# candidate only, one backend at a time
python benchmarks/run_benchmark.py --impl new --backend numpy --cases all --generations 150 --warmup 3 --repeats 2 --threads 0
python benchmarks/run_benchmark.py --impl new --backend iree --cases xor_p500 --generations 150 --warmup 3 --repeats 2 --threads 0   # informational; needs the evoxc[iree] extra, and its numbers never enter the verdict

# inspect the matrix and the planned worker commands (no run, no report)
python benchmarks/run_benchmark.py --list
python benchmarks/run_benchmark.py --dry-run
```

Interpreter used: .venv/bin/python

Cases: xor_p100, xor_p500, xor_p1000, xor3d_p100, xor3d_p500, xor3d_p1000, custom_pagie_p500, cartpole_p100. Timed generations: 150, warmup generations (untimed): 3, repeats: 2 (seeds 42, 43), threads: runtime defaults (`--threads 0`). Backend(s) of the new side: numpy; the old side always runs on JAX, once per case. `--out` defaults to `benchmarks/RESULTS.md` and `--json` to `benchmarks/results/latest.json` (scratch, git-ignored); the record this report was rendered from (`authoritative.json`) is committed at `benchmarks/results/authoritative.json` beside that default, so every table cell is traceable. A thread-bound sensitivity run of the XOR family (`--threads 1`, same CLI) is recorded in `benchmarks/THREADS.md` next to this report.

Generated from `benchmarks/results/authoritative.json` into `benchmarks/RESULTS.md` at 2026-11-04T17:43:10+00:00 — do not hand-edit.
