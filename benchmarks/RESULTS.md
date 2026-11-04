# TensorNEAT benchmark — evoxc rewrite vs frozen JAX baseline

> **GENERATED FILE.** Produced by `python benchmarks/run_benchmark.py`, rendered from
> `benchmarks/results/authoritative.json` at 2026-11-02T07:29:02+00:00. It must not be hand-edited:
> Re-rendered from that same JSON at 2026-11-04T07:27:15+00:00 with `--render-from-json` (no worker was re-run; the dirty state is re-probed on this tree).
> rerun the CLI instead, or the numbers below stop matching the recorded evidence.

## Headline: old vs new

| Case | Pop | Backend | Old ms/gen | New ms/gen | Speed-up | Speed-up range (best–worst) | Quality overlap | Old compile (s) | New compile (s) | Old peak RSS (MiB) | New peak RSS (MiB) | Old final best | New final best | Gens to target (old) | Gens to target (new) | Timing | Quality |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| xor_p100 | 100 | `numpy` | 31.35 | 58.02 | 0.54x | 0.28x–0.74x | inside | 2.744 | 0.558 | 587.1 | 145.4 | -0.0198 | -0.0000 | 78 | 55 | FAIL | PASS |
| xor_p500 | 500 | `numpy` | 157.01 | 357.25 | 0.44x | 0.28x–0.47x | inside | 2.822 | 1.520 | 634.6 | 1049.8 | -0.0000 | -0.0000 | 37 | 33 | FAIL | PASS |
| xor_p1000 | 1000 | `numpy` | 246.76 | 671.73 | 0.37x | 0.34x–0.39x | inside | 2.759 | 2.185 | 631.3 | 1992.1 | -0.0000 | -0.0000 | 35 | 27 | FAIL | PASS |
| xor3d_p100 | 100 | `numpy` | 19.59 | 69.48 | 0.28x | 0.27x–0.50x | inside | 2.716 | 0.575 | 591.3 | 151.9 | -0.0915 | -0.2358 | 149 | >152 | FAIL | FAIL |
| xor3d_p500 | 500 | `numpy` | 81.42 | 380.42 | 0.21x | 0.20x–0.38x | inside | 2.890 | 1.538 | 650.6 | 1081.9 | -0.0652 | -0.0072 | 93 | 101 | FAIL | PASS |
| xor3d_p1000 | 1000 | `numpy` | 141.18 | 698.15 | 0.20x | 0.16x–0.35x | inside | 2.855 | 2.198 | 655.1 | 2055.9 | -0.0000 | -0.0179 | 81 | 59 | FAIL | FAIL |
| custom_pagie_p500 | 500 | `numpy` | 371.83 | 3443.44 | 0.11x | 0.09x–0.14x | inside | 2.876 | 4.688 | 756.0 | 6041.7 | -0.0064 | -0.0221 | 121 | 77 | FAIL | FAIL |
| cartpole_p100 | 100 | `numpy` | n/a (no old-side equivalent) | 412.02 | n/a (needs both sides) | n/a (needs both sides) | n/a (needs both sides) | n/a (no old-side equivalent) | 0.916 | n/a (no old-side equivalent) | 137.0 | n/a (no old-side equivalent) | 200.0000 | n/a (no old-side equivalent) | 1 | n/a | n/a |

One row per case and new-side backend (the old side always runs on JAX). Speed-up is `old / new` per generation: above 1.00x the new side is faster. Speed-up range is `old_min/new_max` … `old_max/new_min` over the per-repeat medians, i.e. the best and worst case over the repeats. Quality overlap states where the new side's per-repeat final-best range sits against the old side's: `inside` when the ranges intersect (every new repeat inside the old range, or the ranges partially overlap), `new worse` when every new repeat is below the old range, `new better` when every new repeat is above it — an observation, never a replacement for the verdict. Timing PASS means new ms/generation is at or below the old one; quality PASS means the new final best stays within `max(10% × |old|, 0.001)` of the old one. Every cell comes from the aggregated JSON records of this run; `n/a (reason)` marks a cell that could not be reached (not run, skipped, error, or not recorded).

## Environment

| Item | Value |
|---|---|
| Date (UTC) | 2026-11-02T07:29:02+00:00 |
| Python | 3.12.3 |
| Platform | Linux-6.8.0-146-generic-x86_64-with-glibc2.39 |
| CPU model | INTEL(R) XEON(R) PLATINUM 8568Y+ |
| CPU cores (`os.cpu_count`) | 20 |
| Memory | 241611 MiB |
| Load average (1/5/15 min) | 3.17 / 3.88 / 3.41 (1/5/15 min, document build); workers 1.17–8.48 at start, 1.17–8.26 at end (1-min, 45 run(s)) |
| numpy / jax / evoxc / tensorneat (installed metadata) | numpy 2.5.4 / jax 0.12.0.1 / evoxc 0.1.0 / tensorneat n/a |
| Git commit | `89f6200d6bba` (clean — generated report artifacts are not counted) — `benchmarks/legacy_jax` and `src/` both come from this checkout |
| Thread pinning (`--threads`) | `--threads 0` — runtime defaults (as users run it), nothing exported |
| Thread env (set/unset) | OMP_NUM_THREADS unset · MKL_NUM_THREADS unset · OPENBLAS_NUM_THREADS unset · NUMEXPR_NUM_THREADS unset · XLA_FLAGS unset (CLI process, inherited unchanged by every worker — `--threads 0` exports nothing) |

**Shared box.** These numbers were measured on a machine that was not exclusively ours: other jobs ran on it while the workers timed generations, so the per-repeat spread (the headline's speed-up range and the per-repeat quality-curve rows) is the honest uncertainty measure — a median alone would hide it.

## Protocol

| Parameter | Value |
|---|---|
| Backend(s) of the new implementation | `numpy` |
| Warmup generations (untimed) | 3 |
| Timed generations | 150 |
| Repeats (seed + r) | 3 (`seed + r`, seeds 42, 43, 44) |
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
| `old` tensorneat-legacy (JAX) | `jax` | 2.744 | 31.35 | 31.90 | 587.1 | -0.0198 | 78 | 3 |
| `new` tensorneat (evoxc) | `numpy` | 0.558 | 58.02 | 17.24 | 145.4 | -0.0000 | 55 | 3 |

### xor_p500

| Implementation | Backend | Compile (s) | ms/generation | generations/s | peak RSS (MiB) | Final best | Gens to target | Repeats |
|---|---|---|---|---|---|---|---|---|
| `old` tensorneat-legacy (JAX) | `jax` | 2.822 | 157.01 | 6.37 | 634.6 | -0.0000 | 37 | 3 |
| `new` tensorneat (evoxc) | `numpy` | 1.520 | 357.25 | 2.80 | 1049.8 | -0.0000 | 33 | 3 |

### xor_p1000

| Implementation | Backend | Compile (s) | ms/generation | generations/s | peak RSS (MiB) | Final best | Gens to target | Repeats |
|---|---|---|---|---|---|---|---|---|
| `old` tensorneat-legacy (JAX) | `jax` | 2.759 | 246.76 | 4.05 | 631.3 | -0.0000 | 35 | 3 |
| `new` tensorneat (evoxc) | `numpy` | 2.185 | 671.73 | 1.49 | 1992.1 | -0.0000 | 27 | 3 |

### xor3d_p100

| Implementation | Backend | Compile (s) | ms/generation | generations/s | peak RSS (MiB) | Final best | Gens to target | Repeats |
|---|---|---|---|---|---|---|---|---|
| `old` tensorneat-legacy (JAX) | `jax` | 2.716 | 19.59 | 51.06 | 591.3 | -0.0915 | 149 | 3 |
| `new` tensorneat (evoxc) | `numpy` | 0.575 | 69.48 | 14.39 | 151.9 | -0.2358 | >152 | 3 |

### xor3d_p500

| Implementation | Backend | Compile (s) | ms/generation | generations/s | peak RSS (MiB) | Final best | Gens to target | Repeats |
|---|---|---|---|---|---|---|---|---|
| `old` tensorneat-legacy (JAX) | `jax` | 2.890 | 81.42 | 12.28 | 650.6 | -0.0652 | 93 | 3 |
| `new` tensorneat (evoxc) | `numpy` | 1.538 | 380.42 | 2.63 | 1081.9 | -0.0072 | 101 | 3 |

### xor3d_p1000

| Implementation | Backend | Compile (s) | ms/generation | generations/s | peak RSS (MiB) | Final best | Gens to target | Repeats |
|---|---|---|---|---|---|---|---|---|
| `old` tensorneat-legacy (JAX) | `jax` | 2.855 | 141.18 | 7.08 | 655.1 | -0.0000 | 81 | 3 |
| `new` tensorneat (evoxc) | `numpy` | 2.198 | 698.15 | 1.43 | 2055.9 | -0.0179 | 59 | 3 |

### custom_pagie_p500

| Implementation | Backend | Compile (s) | ms/generation | generations/s | peak RSS (MiB) | Final best | Gens to target | Repeats |
|---|---|---|---|---|---|---|---|---|
| `old` tensorneat-legacy (JAX) | `jax` | 2.876 | 371.83 | 2.69 | 756.0 | -0.0064 | 121 | 3 |
| `new` tensorneat (evoxc) | `numpy` | 4.688 | 3443.44 | 0.29 | 6041.7 | -0.0221 | 77 | 3 |

### cartpole_p100

| Implementation | Backend | Compile (s) | ms/generation | generations/s | peak RSS (MiB) | Final best | Gens to target | Repeats |
|---|---|---|---|---|---|---|---|---|
| `new` tensorneat (evoxc) | `numpy` | 0.916 | 412.02 | 2.43 | 137.0 | 200.0000 | 1 | 3 |

## Old vs new

| Case | Backend | Old ms/gen | New ms/gen | Speed-up | Delta ms/gen | Old final best | New final best | Quality delta | Timing | Quality |
|---|---|---|---|---|---|---|---|---|---|---|
| xor_p100 | `numpy` | 31.35 | 58.02 | 0.54x | +26.67 | -0.0198 | -0.0000 | +0.0198 | FAIL | PASS |
| xor_p500 | `numpy` | 157.01 | 357.25 | 0.44x | +200.24 | -0.0000 | -0.0000 | +0.0000 | FAIL | PASS |
| xor_p1000 | `numpy` | 246.76 | 671.73 | 0.37x | +424.96 | -0.0000 | -0.0000 | +0.0000 | FAIL | PASS |
| xor3d_p100 | `numpy` | 19.59 | 69.48 | 0.28x | +49.90 | -0.0915 | -0.2358 | -0.1443 | FAIL | FAIL |
| xor3d_p500 | `numpy` | 81.42 | 380.42 | 0.21x | +298.99 | -0.0652 | -0.0072 | +0.0580 | FAIL | PASS |
| xor3d_p1000 | `numpy` | 141.18 | 698.15 | 0.20x | +556.97 | -0.0000 | -0.0179 | -0.0178 | FAIL | FAIL |
| custom_pagie_p500 | `numpy` | 371.83 | 3443.44 | 0.11x | +3071.60 | -0.0064 | -0.0221 | -0.0157 | FAIL | FAIL |

Timing PASS means new ms/generation <= old; quality PASS means the new final best stays within the active tolerance `max(10% × |old|, 0.001)` of the old final best (the absolute floor keeps two runs that are both at the optimum from failing on float noise).

## Verdict

**FAIL**

- timing: xor_p100/numpy new 58.017 ms/gen > old 31.352 ms/gen (speed-up 0.540x)
- timing: xor_p500/numpy new 357.246 ms/gen > old 157.010 ms/gen (speed-up 0.440x)
- timing: xor_p1000/numpy new 671.727 ms/gen > old 246.763 ms/gen (speed-up 0.367x)
- timing: xor3d_p100/numpy new 69.485 ms/gen > old 19.585 ms/gen (speed-up 0.282x)
- timing: xor3d_p500/numpy new 380.417 ms/gen > old 81.423 ms/gen (speed-up 0.214x)
- timing: xor3d_p1000/numpy new 698.152 ms/gen > old 141.184 ms/gen (speed-up 0.202x)
- timing: custom_pagie_p500/numpy new 3443.435 ms/gen > old 371.833 ms/gen (speed-up 0.108x)
- quality: xor3d_p100/numpy new final best -0.2358 < threshold -0.1006 (old -0.0915 - tolerance 0.0091, i.e. max(10% × \|old\|, 0.001 absolute floor))
- quality: xor3d_p1000/numpy new final best -0.0179 < threshold -0.0010 (old -0.0000 - tolerance 0.0010, i.e. max(10% × \|old\|, 0.001 absolute floor))
- quality: custom_pagie_p500/numpy new final best -0.0221 < threshold -0.0074 (old -0.0064 - tolerance 0.0010, i.e. max(10% × \|old\|, 0.001 absolute floor))

Sub-verdicts: timing `FAIL`, quality `FAIL` — timing compares measured ms/generation, quality passes when `new >= old - max(10% × |old|, 0.001)`.

## Why the new side wins and loses

Per case, derived from the measured `old / new` speed-up:

- `xor_p100` (pop 100, family func_fit, backend `numpy`): **new loses** — 0.54x of the old speed (old 31.35 ms/gen → new 58.02 ms/gen; the old side is 1.85x faster).
- `xor_p500` (pop 500, family func_fit, backend `numpy`): **new loses** — 0.44x of the old speed (old 157.01 ms/gen → new 357.25 ms/gen; the old side is 2.28x faster).
- `xor_p1000` (pop 1000, family func_fit, backend `numpy`): **new loses** — 0.37x of the old speed (old 246.76 ms/gen → new 671.73 ms/gen; the old side is 2.72x faster).
- `xor3d_p100` (pop 100, family func_fit, backend `numpy`): **new loses** — 0.28x of the old speed (old 19.59 ms/gen → new 69.48 ms/gen; the old side is 3.55x faster).
- `xor3d_p500` (pop 500, family func_fit, backend `numpy`): **new loses** — 0.21x of the old speed (old 81.42 ms/gen → new 380.42 ms/gen; the old side is 4.67x faster).
- `xor3d_p1000` (pop 1000, family func_fit, backend `numpy`): **new loses** — 0.20x of the old speed (old 141.18 ms/gen → new 698.15 ms/gen; the old side is 4.94x faster).
- `custom_pagie_p500` (pop 500, family func_fit, backend `numpy`): **new loses** — 0.11x of the old speed (old 371.83 ms/gen → new 3443.44 ms/gen; the old side is 9.26x faster).

By population size:

- pop 100: **new loses** on all 2 comparable run(s) — 0.28x–0.54x of the old speed (the old side is 1.85x–3.55x faster).
- pop 500: **new loses** on all 3 comparable run(s) — 0.11x–0.44x of the old speed (the old side is 2.28x–9.26x faster).
- pop 1000: **new loses** on all 2 comparable run(s) — 0.20x–0.37x of the old speed (the old side is 2.72x–4.94x faster).

**Where the new side loses:** at the populations and edge counts above, the same per-op Python-loop dispatch is no longer amortised by edge-proportional work, while the old side's XLA program fuses the whole generation into compiled kernels and pays that dispatch once.

**Memory.** Peak RSS (`rss_final_mb`, the timed worker's high-water mark, taken as the max over the repeats) from the same runs, new vs old:

- `custom_pagie_p500` (pop 500, `numpy`): 6041.7 MiB vs 756.0 MiB — 7.99x (the new side uses more memory).
- `xor_p1000` (pop 1000, `numpy`): 1992.1 MiB vs 631.3 MiB — 3.16x (the new side uses more memory).
- `xor3d_p1000` (pop 1000, `numpy`): 2055.9 MiB vs 655.1 MiB — 3.14x (the new side uses more memory).
- `xor3d_p500` (pop 500, `numpy`): 1081.9 MiB vs 650.6 MiB — 1.66x (the new side uses more memory).
- `xor_p500` (pop 500, `numpy`): 1049.8 MiB vs 634.6 MiB — 1.65x (the new side uses more memory).
- `xor3d_p100` (pop 100, `numpy`): 151.9 MiB vs 591.3 MiB — 0.26x (the new side uses less memory).
- `xor_p100` (pop 100, `numpy`): 145.4 MiB vs 587.1 MiB — 0.25x (the new side uses less memory).

Worst case: `custom_pagie_p500` at 7.99x (6041.7 vs 756.0 MiB old, +5286 MiB).
The new side's peak grows with the population in the XOR family (145.4 → 1049.8 → 1992.1 MiB at pop 100, 500, 1000), while the old row moves only from 587.1 to 634.6 MiB.

Mechanism, as far as these runs support one: the old side's floor is the JAX/XLA runtime itself (193.8–193.9 MiB right after import); the new side starts from 72.9 MiB. Beyond those floors the new peak tracks the population while the old peak stays close to its runtime floor, so the working set the numpy interpreter materialises per generation is what grows with `pop_size × max_nodes × max_conns`; these runs do not attribute that growth to a single dominant term.

This section is written from the same `speedup` values as the headline and old-vs-new tables; it does not assume the new side is the winner. A rerun that changes those numbers changes this text with them.

Where the new side lands relative to the declared contract:

- Contract at pop 100 (`xor_p100`): 51.00 ms/gen declared; measured old 31.35 ms/gen (0.61x declared); measured new 58.02 ms/gen (1.14x declared, i.e. the new side is slower than the contract here).
- Contract at pop 500 (`xor_p500`): 109.00 ms/gen declared; measured old 157.01 ms/gen (1.44x declared); measured new 357.25 ms/gen (3.28x declared, i.e. the new side is slower than the contract here).
- Contract at pop 1000 (`xor_p1000`): 157.00 ms/gen declared; measured old 246.76 ms/gen (1.57x declared); measured new 671.73 ms/gen (4.28x declared, i.e. the new side is slower than the contract here).

The declared contract (51/109/157 ms/gen at pop 100/500/1000) was measured on a 1-thread-bound box; this run used `--threads 0`, so the comparison to the contract is informational and the measured old row on this box is what the verdict uses.

## Quality curves

### xor_p100

| Implementation | Backend | Seed | Best @1 | Best @10 | Best @25% | Best @50% | Best @100% | Final | Gens to target |
|---|---|---|---|---|---|---|---|---|---|
| `old` tensorneat-legacy (JAX) | `jax` | 42 | -0.2507 | -0.2500 | -0.1929 | -0.0542 | -0.0000 | -0.0000 | 62 |
| `old` tensorneat-legacy (JAX) | `jax` | 43 | -0.2519 | -0.2500 | -0.2364 | -0.1515 | -0.0198 | -0.0198 | 93 |
| `old` tensorneat-legacy (JAX) | `jax` | 44 | -0.2549 | -0.2500 | -0.2500 | -0.2500 | -0.2500 | -0.2500 | >153 |
| `old` tensorneat-legacy (JAX) — aggregate | `jax` | 42, 43, 44 | -0.2519 | -0.2500 | -0.2364 | -0.1515 | -0.0198 | -0.0198 | 78 |
| `new` tensorneat (evoxc) | `numpy` | 42 | -0.2502 | -0.2500 | -0.2205 | -0.0067 | -0.0000 | -0.0000 | 55 |
| `new` tensorneat (evoxc) | `numpy` | 43 | -0.2505 | -0.2500 | -0.2500 | -0.2500 | -0.0606 | -0.0606 | 140 |
| `new` tensorneat (evoxc) | `numpy` | 44 | -0.2501 | -0.2476 | -0.1375 | -0.0182 | -0.0000 | -0.0000 | 47 |
| `new` tensorneat (evoxc) — aggregate | `numpy` | 42, 43, 44 | -0.2502 | -0.2500 | -0.2205 | -0.0182 | -0.0000 | -0.0000 | 55 |

### xor_p500

| Implementation | Backend | Seed | Best @1 | Best @10 | Best @25% | Best @50% | Best @100% | Final | Gens to target |
|---|---|---|---|---|---|---|---|---|---|
| `old` tensorneat-legacy (JAX) | `jax` | 42 | -0.2506 | -0.2477 | -0.1152 | -0.0055 | -0.0000 | -0.0000 | 41 |
| `old` tensorneat-legacy (JAX) | `jax` | 43 | -0.2505 | -0.2453 | -0.0795 | -0.0002 | -0.0000 | -0.0000 | 37 |
| `old` tensorneat-legacy (JAX) | `jax` | 44 | -0.2510 | -0.2484 | -0.0717 | -0.0006 | -0.0000 | -0.0000 | 34 |
| `old` tensorneat-legacy (JAX) — aggregate | `jax` | 42, 43, 44 | -0.2506 | -0.2477 | -0.0795 | -0.0006 | -0.0000 | -0.0000 | 37 |
| `new` tensorneat (evoxc) | `numpy` | 42 | -0.2500 | -0.2421 | -0.0235 | -0.0000 | -0.0000 | -0.0000 | 28 |
| `new` tensorneat (evoxc) | `numpy` | 43 | -0.2500 | -0.2500 | -0.0496 | -0.0000 | -0.0000 | -0.0000 | 33 |
| `new` tensorneat (evoxc) | `numpy` | 44 | -0.2502 | -0.2500 | -0.1885 | -0.0008 | -0.0000 | -0.0000 | 48 |
| `new` tensorneat (evoxc) — aggregate | `numpy` | 42, 43, 44 | -0.2500 | -0.2500 | -0.0496 | -0.0000 | -0.0000 | -0.0000 | 33 |

### xor_p1000

| Implementation | Backend | Seed | Best @1 | Best @10 | Best @25% | Best @50% | Best @100% | Final | Gens to target |
|---|---|---|---|---|---|---|---|---|---|
| `old` tensorneat-legacy (JAX) | `jax` | 42 | -0.2506 | -0.2427 | -0.0780 | -0.0004 | -0.0000 | -0.0000 | 35 |
| `old` tensorneat-legacy (JAX) | `jax` | 43 | -0.2505 | -0.2376 | -0.0234 | -0.0000 | -0.0000 | -0.0000 | 29 |
| `old` tensorneat-legacy (JAX) | `jax` | 44 | -0.2507 | -0.2491 | -0.1176 | -0.0001 | -0.0000 | -0.0000 | 39 |
| `old` tensorneat-legacy (JAX) — aggregate | `jax` | 42, 43, 44 | -0.2506 | -0.2427 | -0.0780 | -0.0001 | -0.0000 | -0.0000 | 35 |
| `new` tensorneat (evoxc) | `numpy` | 42 | -0.2501 | -0.2416 | -0.0391 | -0.0000 | -0.0000 | -0.0000 | 31 |
| `new` tensorneat (evoxc) | `numpy` | 43 | -0.2500 | -0.2391 | -0.0047 | -0.0000 | -0.0000 | -0.0000 | 27 |
| `new` tensorneat (evoxc) | `numpy` | 44 | -0.2500 | -0.2446 | -0.0095 | -0.0000 | -0.0000 | -0.0000 | 24 |
| `new` tensorneat (evoxc) — aggregate | `numpy` | 42, 43, 44 | -0.2500 | -0.2416 | -0.0095 | -0.0000 | -0.0000 | -0.0000 | 27 |

### xor3d_p100

| Implementation | Backend | Seed | Best @1 | Best @10 | Best @25% | Best @50% | Best @100% | Final | Gens to target |
|---|---|---|---|---|---|---|---|---|---|
| `old` tensorneat-legacy (JAX) | `jax` | 42 | -0.2524 | -0.2500 | -0.2500 | -0.2447 | -0.0278 | -0.0278 | 149 |
| `old` tensorneat-legacy (JAX) | `jax` | 43 | -0.2547 | -0.2500 | -0.2500 | -0.2500 | -0.2500 | -0.2500 | >153 |
| `old` tensorneat-legacy (JAX) | `jax` | 44 | -0.2530 | -0.2500 | -0.2500 | -0.2496 | -0.0915 | -0.0915 | >153 |
| `old` tensorneat-legacy (JAX) — aggregate | `jax` | 42, 43, 44 | -0.2530 | -0.2500 | -0.2500 | -0.2496 | -0.0915 | -0.0915 | 149 |
| `new` tensorneat (evoxc) | `numpy` | 42 | -0.2515 | -0.2500 | -0.2500 | -0.2500 | -0.2358 | -0.2358 | >152 |
| `new` tensorneat (evoxc) | `numpy` | 43 | -0.2511 | -0.2500 | -0.2500 | -0.2500 | -0.2500 | -0.2500 | >152 |
| `new` tensorneat (evoxc) | `numpy` | 44 | -0.2539 | -0.2500 | -0.2500 | -0.2500 | -0.2031 | -0.2031 | >152 |
| `new` tensorneat (evoxc) — aggregate | `numpy` | 42, 43, 44 | -0.2515 | -0.2500 | -0.2500 | -0.2500 | -0.2358 | -0.2358 | >152 |

### xor3d_p500

| Implementation | Backend | Seed | Best @1 | Best @10 | Best @25% | Best @50% | Best @100% | Final | Gens to target |
|---|---|---|---|---|---|---|---|---|---|
| `old` tensorneat-legacy (JAX) | `jax` | 42 | -0.2524 | -0.2500 | -0.2500 | -0.2500 | -0.2500 | -0.2500 | >153 |
| `old` tensorneat-legacy (JAX) | `jax` | 43 | -0.2507 | -0.2500 | -0.2500 | -0.2463 | -0.0652 | -0.0652 | >153 |
| `old` tensorneat-legacy (JAX) | `jax` | 44 | -0.2530 | -0.2500 | -0.2500 | -0.1479 | -0.0003 | -0.0003 | 93 |
| `old` tensorneat-legacy (JAX) — aggregate | `jax` | 42, 43, 44 | -0.2524 | -0.2500 | -0.2500 | -0.2463 | -0.0652 | -0.0652 | 93 |
| `new` tensorneat (evoxc) | `numpy` | 42 | -0.2502 | -0.2500 | -0.2500 | -0.2500 | -0.0072 | -0.0072 | 140 |
| `new` tensorneat (evoxc) | `numpy` | 43 | -0.2506 | -0.2500 | -0.2500 | -0.2500 | -0.2500 | -0.2500 | >152 |
| `new` tensorneat (evoxc) | `numpy` | 44 | -0.2501 | -0.2500 | -0.1734 | -0.0034 | -0.0000 | -0.0000 | 62 |
| `new` tensorneat (evoxc) — aggregate | `numpy` | 42, 43, 44 | -0.2502 | -0.2500 | -0.2500 | -0.2500 | -0.0072 | -0.0072 | 101 |

### xor3d_p1000

| Implementation | Backend | Seed | Best @1 | Best @10 | Best @25% | Best @50% | Best @100% | Final | Gens to target |
|---|---|---|---|---|---|---|---|---|---|
| `old` tensorneat-legacy (JAX) | `jax` | 42 | -0.2524 | -0.2500 | -0.1850 | -0.0192 | -0.0000 | -0.0000 | 62 |
| `old` tensorneat-legacy (JAX) | `jax` | 43 | -0.2507 | -0.2500 | -0.2500 | -0.2500 | -0.2500 | -0.2500 | >153 |
| `old` tensorneat-legacy (JAX) | `jax` | 44 | -0.2515 | -0.2500 | -0.2400 | -0.1265 | -0.0000 | -0.0000 | 100 |
| `old` tensorneat-legacy (JAX) — aggregate | `jax` | 42, 43, 44 | -0.2515 | -0.2500 | -0.2400 | -0.1265 | -0.0000 | -0.0000 | 81 |
| `new` tensorneat (evoxc) | `numpy` | 42 | -0.2501 | -0.2500 | -0.1789 | -0.0195 | -0.0179 | -0.0179 | 59 |
| `new` tensorneat (evoxc) | `numpy` | 43 | -0.2501 | -0.2500 | -0.2500 | -0.2500 | -0.0000 | -0.0000 | 108 |
| `new` tensorneat (evoxc) | `numpy` | 44 | -0.2502 | -0.2500 | -0.1066 | -0.0183 | -0.0179 | -0.0179 | 49 |
| `new` tensorneat (evoxc) — aggregate | `numpy` | 42, 43, 44 | -0.2501 | -0.2500 | -0.1789 | -0.0195 | -0.0179 | -0.0179 | 59 |

### custom_pagie_p500

| Implementation | Backend | Seed | Best @1 | Best @10 | Best @25% | Best @50% | Best @100% | Final | Gens to target |
|---|---|---|---|---|---|---|---|---|---|
| `old` tensorneat-legacy (JAX) | `jax` | 42 | -0.0490 | -0.0473 | -0.0347 | -0.0233 | -0.0064 | -0.0064 | 125 |
| `old` tensorneat-legacy (JAX) | `jax` | 43 | -0.0493 | -0.0477 | -0.0243 | -0.0221 | -0.0040 | -0.0040 | 117 |
| `old` tensorneat-legacy (JAX) | `jax` | 44 | -0.0489 | -0.0474 | -0.0322 | -0.0232 | -0.0140 | -0.0140 | >153 |
| `old` tensorneat-legacy (JAX) — aggregate | `jax` | 42, 43, 44 | -0.0490 | -0.0474 | -0.0322 | -0.0232 | -0.0064 | -0.0064 | 121 |
| `new` tensorneat (evoxc) | `numpy` | 42 | -0.0461 | -0.0459 | -0.0332 | -0.0233 | -0.0221 | -0.0221 | >152 |
| `new` tensorneat (evoxc) | `numpy` | 43 | -0.0461 | -0.0424 | -0.0184 | -0.0102 | -0.0057 | -0.0057 | 77 |
| `new` tensorneat (evoxc) | `numpy` | 44 | -0.0460 | -0.0460 | -0.0460 | -0.0317 | -0.0245 | -0.0245 | >152 |
| `new` tensorneat (evoxc) — aggregate | `numpy` | 42, 43, 44 | -0.0461 | -0.0459 | -0.0332 | -0.0233 | -0.0221 | -0.0221 | 77 |

### cartpole_p100

| Implementation | Backend | Seed | Best @1 | Best @10 | Best @25% | Best @50% | Best @100% | Final | Gens to target |
|---|---|---|---|---|---|---|---|---|---|
| `new` tensorneat (evoxc) | `numpy` | 42 | 200.0000 | 200.0000 | 200.0000 | 200.0000 | 200.0000 | 200.0000 | 1 |
| `new` tensorneat (evoxc) | `numpy` | 43 | 200.0000 | 200.0000 | 200.0000 | 200.0000 | 200.0000 | 200.0000 | 1 |
| `new` tensorneat (evoxc) | `numpy` | 44 | 200.0000 | 200.0000 | 200.0000 | 200.0000 | 200.0000 | 200.0000 | 1 |
| `new` tensorneat (evoxc) — aggregate | `numpy` | 42, 43, 44 | 200.0000 | 200.0000 | 200.0000 | 200.0000 | 200.0000 | 200.0000 | 1 |

Percent columns are generation `round(fraction × N)` of the N recorded generations (warmup + timed — the budget `generations_to_target` is indexed against), and `>N` means the case's fitness target was not reached within those N recorded generations. Each repeat (seed) has its own row, and the `— aggregate` row is the element-wise median of those repeats: a bimodal case (one seed at the target, one not) shows up as two different `Final` values rather than one average.

## Reference numbers (declared baseline)

| Case | Pop | Declared legacy ms/gen | Measured old ms/gen | Ratio (measured/declared) |
|---|---|---|---|---|
| xor_p100 | 100 | 51.00 | 31.35 | 0.61x |
| xor_p500 | 500 | 109.00 | 157.01 | 1.44x |
| xor_p1000 | 1000 | 157.00 | 246.76 | 1.57x |

The performance contract in `../CONTEXT.md` declares legacy CPU numbers measured on a
1-thread-bound box (51 ms/gen at pop 100, 109 ms/gen at pop 500, 157 ms/gen at pop 1000).
The generator prints those next to the old row measured on this box; the verdict uses the
measured row.

## What is not compared

- **Same workload, not the same search.** The sides use different PRNGs (JAX `PRNGKey` vs evoxc splitmix64), so one seed identifies the same *workload*, not the same trajectory or a bitwise-equal population. Numerical fidelity is `test/parity/`'s job: it compares kernels against `benchmarks/legacy_jax`, and `pytest test/parity --collect-only -q` collects 18 tests on this box.
- **No accelerator row in this report.** The old-vs-new verdict is CPU-only and numpy-only by protocol, so the matrix here is the numpy reference interpreter; an `iree`/`xla`/`tvm` row is informational and never enters the verdict. The compiled leg is measured separately in `benchmarks/BACKENDS.md`: `iree` lowers, compiles and runs every shipped case, and that file carries the measured per-generation comparison with numpy — an accelerator row exists, it simply stays outside this verdict.
- **CartPole has no old-side row.** The frozen library's RL problems need gym/brax/mujoco, so `cartpole_p100` is new-only and marked `n/a` for old.
- **Import time, visualization and code generation are not timed.** Only `rss_after_import_mb` is recorded for the import phase; `Pipeline.show`, network drawing and sympy codegen sit outside every timed region.
- **No trajectory equality is expected.** The two sides evolve different populations from the same workload seed; parity of kernels (not of search) is what `test/parity/` verifies.

## How to reproduce

```bash
# regenerate the committed report: the literal command of this run
python benchmarks/run_benchmark.py --impl both --backend numpy --cases all --generations 150 --warmup 3 --repeats 3 --json benchmarks/results/authoritative.json

# full matrix, both implementations on the primary backend
python benchmarks/run_benchmark.py --impl both --backend numpy --cases all --generations 150 --warmup 3 --repeats 3 --threads 0

# baseline only (frozen JAX)
python benchmarks/run_benchmark.py --impl old --cases all --generations 150 --warmup 3 --repeats 3 --threads 0

# candidate only, one backend at a time
python benchmarks/run_benchmark.py --impl new --backend numpy --cases all --generations 150 --warmup 3 --repeats 3 --threads 0
python benchmarks/run_benchmark.py --impl new --backend iree --cases xor_p500 --generations 150 --warmup 3 --repeats 3 --threads 0   # informational; needs the evoxc[iree] extra, and its numbers never enter the verdict

# inspect the matrix and the planned worker commands (no run, no report)
python benchmarks/run_benchmark.py --list
python benchmarks/run_benchmark.py --dry-run
```

Interpreter used: /tmp/tn-venv/bin/python

Cases: xor_p100, xor_p500, xor_p1000, xor3d_p100, xor3d_p500, xor3d_p1000, custom_pagie_p500, cartpole_p100. Timed generations: 150, warmup generations (untimed): 3, repeats: 3 (seeds 42, 43, 44), threads: runtime defaults (`--threads 0`). Backend(s) of the new side: numpy; the old side always runs on JAX, once per case. `--out` defaults to `benchmarks/RESULTS.md` and `--json` to `benchmarks/results/latest.json` (scratch, git-ignored); the record this report was rendered from (`authoritative.json`) is committed at `benchmarks/results/authoritative.json` beside that default, so every table cell is traceable. A thread-bound sensitivity run of the XOR family (`--threads 1`, same CLI) is recorded in `benchmarks/THREADS.md` next to this report.

Generated from `benchmarks/results/authoritative.json` into `benchmarks/RESULTS.md` at 2026-11-02T07:29:02+00:00 — do not hand-edit.
