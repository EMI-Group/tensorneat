# TensorNEAT benchmark — evoxc rewrite vs frozen JAX baseline

> **GENERATED FILE.** Produced by `python benchmarks/run_benchmark.py`, rendered from `benchmarks/results/authoritative.json` at 2026-11-04T19:01:27+00:00.
> Re-rendered from that same JSON at 2026-11-05T04:48:30+00:00 with `--render-from-json` (no worker was re-run; the dirty state is re-probed on this tree).
> It must not be hand-edited: rerun the CLI instead, or the numbers below stop matching the recorded evidence.

*Pre-fix record, not a description of the merged tree: measured at `ebeaf1f8` on 2026-11-04, before the tiled-evaluate / RL-rollout-hoist performance pass (`90d97b4`) and before the search-quality fixes, so the new-side rows below are the pre-fix cost (pagie 3156.73 ms/gen and 4577.1 MiB) and the pre-fix convergence — `PERF_NUMPY.md` carries the post-fix case-scoped numbers and `results/xor3d_quality_2026-11-04.md` the post-fix quality record.*

## Headline: old vs new

| Case | Pop | Backend | Old ms/gen | New ms/gen | Speed-up | Speed-up range (best–worst) | Quality overlap | Old compile (s) | New compile (s) | Old peak RSS (MiB) | New peak RSS (MiB) | Old final best | New final best | Gens to target (old) | Gens to target (new) | Timing | Quality |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| xor_p100 | 100 | `numpy` | 38.99 | 36.44 | 1.07x | 0.59x–1.37x | inside | 2.941 | 0.297 | 676.7 | 114.9 | -0.0034 | -0.0000 | 78 | 56 | PASS | PASS |
| xor_p500 | 500 | `numpy` | 295.33 | 161.73 | 1.83x | 1.49x–2.33x | inside | 2.926 | 0.527 | 720.7 | 241.3 | -0.0000 | -0.0000 | 42 | 35 | PASS | PASS |
| xor_p1000 | 1000 | `numpy` | 807.30 | 279.92 | 2.88x | 2.72x–3.00x | inside | 2.956 | 0.801 | 705.1 | 392.2 | -0.0000 | -0.0000 | 35 | 28 | PASS | PASS |
| xor3d_p100 | 100 | `numpy` | 24.53 | 39.12 | 0.63x | 0.59x–1.12x | new worse | 2.982 | 0.308 | 681.5 | 120.0 | -0.0894 | -0.2500 | 149 | >152 | FAIL | FAIL |
| xor3d_p500 | 500 | `numpy` | 107.59 | 171.16 | 0.63x | 0.53x–0.79x | inside | 2.998 | 0.536 | 749.1 | 265.5 | -0.1789 | -0.2500 | 87 | 95 | FAIL | FAIL |
| xor3d_p1000 | 1000 | `numpy` | 255.41 | 300.17 | 0.85x | 0.56x–1.27x | inside | 3.050 | 0.800 | 732.3 | 441.0 | -0.0000 | -0.0001 | 73 | 90 | FAIL | PASS |
| custom_pagie_p500 | 500 | `numpy` | 496.45 | 3156.73 | 0.16x | 0.15x–0.19x | new worse | 2.927 | 2.256 | 853.3 | 4577.1 | -0.0064 | -0.0219 | 121 | >152 | FAIL | FAIL |
| cartpole_p100 | 100 | `numpy` | n/a (no old-side equivalent) | 381.73 | n/a (needs both sides) | n/a (needs both sides) | n/a (needs both sides) | n/a (no old-side equivalent) | 0.549 | n/a (no old-side equivalent) | 110.6 | n/a (no old-side equivalent) | 200.0000 | n/a (no old-side equivalent) | 1 | n/a | n/a |

One row per case and new-side backend (the old side always runs on JAX). Speed-up is `old / new` per generation: above 1.00x the new side is faster. Speed-up range is `old_min/new_max` … `old_max/new_min` over the per-repeat medians, i.e. the best and worst case over the repeats. Quality overlap states where the new side's per-repeat final-best range sits against the old side's: `inside` when the ranges intersect (every new repeat inside the old range, or the ranges partially overlap), `new worse` when every new repeat is below the old range, `new better` when every new repeat is above it — an observation, never a replacement for the verdict. Timing PASS means new ms/generation is at or below the old one; quality PASS means the new final best stays within `max(10% × |old|, 0.001)` of the old one. Every cell comes from the aggregated JSON records of this run; `n/a (reason)` marks a cell that could not be reached (not run, skipped, error, or not recorded).

## Environment

| Item | Value |
|---|---|
| Date (UTC) | 2026-11-04T19:01:27+00:00 |
| Python | 3.12.3 |
| Platform | Linux-6.8.0-146-generic-x86_64-with-glibc2.39 |
| CPU model | INTEL(R) XEON(R) PLATINUM 8568Y+ |
| CPU cores (`os.cpu_count`) | 20 |
| Memory | 241611 MiB |
| Load average (1/5/15 min) | 1.52 / 2.29 / 2.43 (1/5/15 min, document build); workers 0.70–5.51 at start, 0.75–5.51 at end (1-min, 45 run(s)) |
| numpy / jax / evoxc / tensorneat (installed metadata) | numpy 2.5.4 / jax 0.10.2 / evoxc 0.1.0 / tensorneat 0.2.0 |
| Git commit | `ebeaf1f8d07e` (clean — generated report artifacts are not counted) — `benchmarks/legacy_jax` and `src/` both come from this checkout |
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
| `old` tensorneat-legacy (JAX) | `jax` | 2.941 | 38.99 | 25.65 | 676.7 | -0.0034 | 78 | 3 |
| `new` tensorneat (evoxc) | `numpy` | 0.297 | 36.44 | 27.45 | 114.9 | -0.0000 | 56 | 3 |

### xor_p500

| Implementation | Backend | Compile (s) | ms/generation | generations/s | peak RSS (MiB) | Final best | Gens to target | Repeats |
|---|---|---|---|---|---|---|---|---|
| `old` tensorneat-legacy (JAX) | `jax` | 2.926 | 295.33 | 3.39 | 720.7 | -0.0000 | 42 | 3 |
| `new` tensorneat (evoxc) | `numpy` | 0.527 | 161.73 | 6.18 | 241.3 | -0.0000 | 35 | 3 |

### xor_p1000

| Implementation | Backend | Compile (s) | ms/generation | generations/s | peak RSS (MiB) | Final best | Gens to target | Repeats |
|---|---|---|---|---|---|---|---|---|
| `old` tensorneat-legacy (JAX) | `jax` | 2.956 | 807.30 | 1.24 | 705.1 | -0.0000 | 35 | 3 |
| `new` tensorneat (evoxc) | `numpy` | 0.801 | 279.92 | 3.57 | 392.2 | -0.0000 | 28 | 3 |

### xor3d_p100

| Implementation | Backend | Compile (s) | ms/generation | generations/s | peak RSS (MiB) | Final best | Gens to target | Repeats |
|---|---|---|---|---|---|---|---|---|
| `old` tensorneat-legacy (JAX) | `jax` | 2.982 | 24.53 | 40.76 | 681.5 | -0.0894 | 149 | 3 |
| `new` tensorneat (evoxc) | `numpy` | 0.308 | 39.12 | 25.56 | 120.0 | -0.2500 | >152 | 3 |

### xor3d_p500

| Implementation | Backend | Compile (s) | ms/generation | generations/s | peak RSS (MiB) | Final best | Gens to target | Repeats |
|---|---|---|---|---|---|---|---|---|
| `old` tensorneat-legacy (JAX) | `jax` | 2.998 | 107.59 | 9.29 | 749.1 | -0.1789 | 87 | 3 |
| `new` tensorneat (evoxc) | `numpy` | 0.536 | 171.16 | 5.84 | 265.5 | -0.2500 | 95 | 3 |

### xor3d_p1000

| Implementation | Backend | Compile (s) | ms/generation | generations/s | peak RSS (MiB) | Final best | Gens to target | Repeats |
|---|---|---|---|---|---|---|---|---|
| `old` tensorneat-legacy (JAX) | `jax` | 3.050 | 255.41 | 3.92 | 732.3 | -0.0000 | 73 | 3 |
| `new` tensorneat (evoxc) | `numpy` | 0.800 | 300.17 | 3.33 | 441.0 | -0.0001 | 90 | 3 |

### custom_pagie_p500

| Implementation | Backend | Compile (s) | ms/generation | generations/s | peak RSS (MiB) | Final best | Gens to target | Repeats |
|---|---|---|---|---|---|---|---|---|
| `old` tensorneat-legacy (JAX) | `jax` | 2.927 | 496.45 | 2.01 | 853.3 | -0.0064 | 121 | 3 |
| `new` tensorneat (evoxc) | `numpy` | 2.256 | 3156.73 | 0.32 | 4577.1 | -0.0219 | >152 | 3 |

### cartpole_p100

| Implementation | Backend | Compile (s) | ms/generation | generations/s | peak RSS (MiB) | Final best | Gens to target | Repeats |
|---|---|---|---|---|---|---|---|---|
| `new` tensorneat (evoxc) | `numpy` | 0.549 | 381.73 | 2.62 | 110.6 | 200.0000 | 1 | 3 |

## Old vs new

| Case | Backend | Old ms/gen | New ms/gen | Speed-up | Delta ms/gen | Old final best | New final best | Quality delta | Timing | Quality |
|---|---|---|---|---|---|---|---|---|---|---|
| xor_p100 | `numpy` | 38.99 | 36.44 | 1.07x | -2.56 | -0.0034 | -0.0000 | +0.0034 | PASS | PASS |
| xor_p500 | `numpy` | 295.33 | 161.73 | 1.83x | -133.60 | -0.0000 | -0.0000 | -0.0000 | PASS | PASS |
| xor_p1000 | `numpy` | 807.30 | 279.92 | 2.88x | -527.37 | -0.0000 | -0.0000 | -0.0000 | PASS | PASS |
| xor3d_p100 | `numpy` | 24.53 | 39.12 | 0.63x | +14.59 | -0.0894 | -0.2500 | -0.1606 | FAIL | FAIL |
| xor3d_p500 | `numpy` | 107.59 | 171.16 | 0.63x | +63.57 | -0.1789 | -0.2500 | -0.0711 | FAIL | FAIL |
| xor3d_p1000 | `numpy` | 255.41 | 300.17 | 0.85x | +44.76 | -0.0000 | -0.0001 | -0.0001 | FAIL | PASS |
| custom_pagie_p500 | `numpy` | 496.45 | 3156.73 | 0.16x | +2660.28 | -0.0064 | -0.0219 | -0.0155 | FAIL | FAIL |

Timing PASS means new ms/generation <= old; quality PASS means the new final best stays within the active tolerance `max(10% × |old|, 0.001)` of the old final best (the absolute floor keeps two runs that are both at the optimum from failing on float noise).

## Verdict

**FAIL**

- timing: xor3d_p100/numpy new 39.120 ms/gen > old 24.531 ms/gen (speed-up 0.627x)
- timing: xor3d_p500/numpy new 171.163 ms/gen > old 107.594 ms/gen (speed-up 0.629x)
- timing: xor3d_p1000/numpy new 300.174 ms/gen > old 255.411 ms/gen (speed-up 0.851x)
- timing: custom_pagie_p500/numpy new 3156.733 ms/gen > old 496.450 ms/gen (speed-up 0.157x)
- quality: xor3d_p100/numpy new final best -0.2500 < threshold -0.0983 (old -0.0894 - tolerance 0.0089, i.e. max(10% × \|old\|, 0.001 absolute floor))
- quality: xor3d_p500/numpy new final best -0.2500 < threshold -0.1968 (old -0.1789 - tolerance 0.0179, i.e. max(10% × \|old\|, 0.001 absolute floor))
- quality: custom_pagie_p500/numpy new final best -0.0219 < threshold -0.0074 (old -0.0064 - tolerance 0.0010, i.e. max(10% × \|old\|, 0.001 absolute floor))

Sub-verdicts: timing `FAIL`, quality `FAIL` — timing compares measured ms/generation, quality passes when `new >= old - max(10% × |old|, 0.001)`.

## Why the new side wins and loses

Per case, derived from the measured `old / new` speed-up:

- `xor_p100` (pop 100, family func_fit, backend `numpy`): **new wins** — 1.07x faster than the old side (old 38.99 ms/gen → new 36.44 ms/gen).
- `xor_p500` (pop 500, family func_fit, backend `numpy`): **new wins** — 1.83x faster than the old side (old 295.33 ms/gen → new 161.73 ms/gen).
- `xor_p1000` (pop 1000, family func_fit, backend `numpy`): **new wins** — 2.88x faster than the old side (old 807.30 ms/gen → new 279.92 ms/gen).
- `xor3d_p100` (pop 100, family func_fit, backend `numpy`): **new loses** — 0.63x of the old speed (old 24.53 ms/gen → new 39.12 ms/gen; the old side is 1.59x faster).
- `xor3d_p500` (pop 500, family func_fit, backend `numpy`): **new loses** — 0.63x of the old speed (old 107.59 ms/gen → new 171.16 ms/gen; the old side is 1.59x faster).
- `xor3d_p1000` (pop 1000, family func_fit, backend `numpy`): **new loses** — 0.85x of the old speed (old 255.41 ms/gen → new 300.17 ms/gen; the old side is 1.18x faster).
- `custom_pagie_p500` (pop 500, family func_fit, backend `numpy`): **new loses** — 0.16x of the old speed (old 496.45 ms/gen → new 3156.73 ms/gen; the old side is 6.36x faster).

By population size:

- pop 100: **mixed** — new wins 1 of 2 comparable run(s) — speed-up range 0.63x–1.07x.
- pop 500: **mixed** — new wins 1 of 3 comparable run(s) — speed-up range 0.16x–1.83x.
- pop 1000: **mixed** — new wins 1 of 2 comparable run(s) — speed-up range 0.85x–2.88x.

**Where the new side wins:** the evoxc graph's *compute* is edge-proportional and depth-parallel — the forward pass walks the real connections and the network depth instead of evaluating `max_nodes × max_nodes` padded pairs — and on the numpy reference interpreter that useful work amortises its per-op Python-loop dispatch, while XLA's compiled whole-generation program carries more fixed overhead than it saves at small populations and low edge counts.

**Where the new side loses:** at the populations and edge counts above, the same per-op Python-loop dispatch is no longer amortised by edge-proportional work, while the old side's XLA program fuses the whole generation into compiled kernels and pays that dispatch once.

**Memory.** Peak RSS (`rss_final_mb`, the timed worker's high-water mark, taken as the max over the repeats) from the same runs, new vs old:

- `custom_pagie_p500` (pop 500, `numpy`): 4577.1 MiB vs 853.3 MiB — 5.36x (the new side uses more memory).
- `xor3d_p1000` (pop 1000, `numpy`): 441.0 MiB vs 732.3 MiB — 0.60x (the new side uses less memory).
- `xor_p1000` (pop 1000, `numpy`): 392.2 MiB vs 705.1 MiB — 0.56x (the new side uses less memory).
- `xor3d_p500` (pop 500, `numpy`): 265.5 MiB vs 749.1 MiB — 0.35x (the new side uses less memory).
- `xor_p500` (pop 500, `numpy`): 241.3 MiB vs 720.7 MiB — 0.33x (the new side uses less memory).
- `xor3d_p100` (pop 100, `numpy`): 120.0 MiB vs 681.5 MiB — 0.18x (the new side uses less memory).
- `xor_p100` (pop 100, `numpy`): 114.9 MiB vs 676.7 MiB — 0.17x (the new side uses less memory).

Worst case: `custom_pagie_p500` at 5.36x (4577.1 vs 853.3 MiB old, +3724 MiB).
The new side's peak grows with the population in the XOR family (114.9 → 241.3 → 392.2 MiB at pop 100, 500, 1000), while the old row moves only from 676.7 to 720.7 MiB.

Mechanism, as far as these runs support one: the old side's floor is the JAX/XLA runtime itself (182.2–182.3 MiB right after import); the new side starts from 73.8–73.9 MiB. Beyond those floors the new peak tracks the population while the old peak stays close to its runtime floor, so the footprint is the other axis from the compute claim above — the numpy interpreter materialises padded `pop_size × max_nodes × max_conns` tensors every generation, and it is that tensor geometry, not the live edge count, that the peak follows; these runs do not attribute that growth to a single dominant term.

This section is written from the same `speedup` values as the headline and old-vs-new tables; it does not assume the new side is the winner. A rerun that changes those numbers changes this text with them.

Where the new side lands relative to the declared contract:

- Contract at pop 100 (`xor_p100`): 51.00 ms/gen declared; measured old 38.99 ms/gen (0.76x declared); measured new 36.44 ms/gen (0.71x declared).
- Contract at pop 500 (`xor_p500`): 109.00 ms/gen declared; measured old 295.33 ms/gen (2.71x declared); measured new 161.73 ms/gen (1.48x declared, i.e. the new side is slower than the contract here).
- Contract at pop 1000 (`xor_p1000`): 157.00 ms/gen declared; measured old 807.30 ms/gen (5.14x declared); measured new 279.92 ms/gen (1.78x declared, i.e. the new side is slower than the contract here).

The declared contract (51/109/157 ms/gen at pop 100/500/1000) is the recorded reference baseline (`reference_ms_per_generation` in `benchmarks/cases.py`, copied into every run record), not a number this box reproduces: `--threads 0` measured 38.99/295.33/807.30 ms/gen on the old side and `--threads 1` measured 35.79/127.71/252.06 ms/gen on the old side (`benchmarks/results/authoritative.json`, `benchmarks/results/threads1.json`). This run used `--threads 0`, so the comparison to the contract is informational and the measured old row on this box is what the verdict uses.

## Quality curves

### xor_p100

| Implementation | Backend | Seed | Best @1 | Best @10 | Best @25% | Best @50% | Best @100% | Final | Gens to target |
|---|---|---|---|---|---|---|---|---|---|
| `old` tensorneat-legacy (JAX) | `jax` | 42 | -0.2507 | -0.2500 | -0.1929 | -0.0542 | -0.0000 | -0.0000 | 62 |
| `old` tensorneat-legacy (JAX) | `jax` | 43 | -0.2519 | -0.2500 | -0.2364 | -0.1515 | -0.0034 | -0.0034 | 93 |
| `old` tensorneat-legacy (JAX) | `jax` | 44 | -0.2549 | -0.2500 | -0.2500 | -0.2500 | -0.2500 | -0.2500 | >153 |
| `old` tensorneat-legacy (JAX) — aggregate | `jax` | 42, 43, 44 | -0.2519 | -0.2500 | -0.2364 | -0.1515 | -0.0034 | -0.0034 | 78 |
| `new` tensorneat (evoxc) | `numpy` | 42 | -0.2508 | -0.2482 | -0.0428 | -0.0000 | -0.0000 | -0.0000 | 33 |
| `new` tensorneat (evoxc) | `numpy` | 43 | -0.2503 | -0.2500 | -0.2500 | -0.0596 | -0.0189 | -0.0189 | 68 |
| `new` tensorneat (evoxc) | `numpy` | 44 | -0.2501 | -0.2500 | -0.2111 | -0.0177 | -0.0000 | -0.0000 | 56 |
| `new` tensorneat (evoxc) — aggregate | `numpy` | 42, 43, 44 | -0.2503 | -0.2500 | -0.2111 | -0.0177 | -0.0000 | -0.0000 | 56 |

### xor_p500

| Implementation | Backend | Seed | Best @1 | Best @10 | Best @25% | Best @50% | Best @100% | Final | Gens to target |
|---|---|---|---|---|---|---|---|---|---|
| `old` tensorneat-legacy (JAX) | `jax` | 42 | -0.2506 | -0.2477 | -0.1203 | -0.0199 | -0.0000 | -0.0000 | 42 |
| `old` tensorneat-legacy (JAX) | `jax` | 43 | -0.2505 | -0.2453 | -0.1461 | -0.0002 | -0.0000 | -0.0000 | 51 |
| `old` tensorneat-legacy (JAX) | `jax` | 44 | -0.2510 | -0.2484 | -0.0717 | -0.0006 | -0.0000 | -0.0000 | 34 |
| `old` tensorneat-legacy (JAX) — aggregate | `jax` | 42, 43, 44 | -0.2506 | -0.2477 | -0.1203 | -0.0006 | -0.0000 | -0.0000 | 42 |
| `new` tensorneat (evoxc) | `numpy` | 42 | -0.2500 | -0.2161 | -0.0847 | -0.0001 | -0.0000 | -0.0000 | 36 |
| `new` tensorneat (evoxc) | `numpy` | 43 | -0.2500 | -0.2500 | -0.0796 | -0.0000 | -0.0000 | -0.0000 | 35 |
| `new` tensorneat (evoxc) | `numpy` | 44 | -0.2501 | -0.2372 | -0.0550 | -0.0000 | -0.0000 | -0.0000 | 31 |
| `new` tensorneat (evoxc) — aggregate | `numpy` | 42, 43, 44 | -0.2500 | -0.2372 | -0.0796 | -0.0000 | -0.0000 | -0.0000 | 35 |

### xor_p1000

| Implementation | Backend | Seed | Best @1 | Best @10 | Best @25% | Best @50% | Best @100% | Final | Gens to target |
|---|---|---|---|---|---|---|---|---|---|
| `old` tensorneat-legacy (JAX) | `jax` | 42 | -0.2506 | -0.2427 | -0.0762 | -0.0005 | -0.0000 | -0.0000 | 35 |
| `old` tensorneat-legacy (JAX) | `jax` | 43 | -0.2505 | -0.2376 | -0.0234 | -0.0000 | -0.0000 | -0.0000 | 29 |
| `old` tensorneat-legacy (JAX) | `jax` | 44 | -0.2507 | -0.2491 | -0.1176 | -0.0001 | -0.0000 | -0.0000 | 39 |
| `old` tensorneat-legacy (JAX) — aggregate | `jax` | 42, 43, 44 | -0.2506 | -0.2427 | -0.0762 | -0.0001 | -0.0000 | -0.0000 | 35 |
| `new` tensorneat (evoxc) | `numpy` | 42 | -0.2501 | -0.2450 | -0.0255 | -0.0000 | -0.0000 | -0.0000 | 28 |
| `new` tensorneat (evoxc) | `numpy` | 43 | -0.2500 | -0.2486 | -0.0674 | -0.0017 | -0.0000 | -0.0000 | 31 |
| `new` tensorneat (evoxc) | `numpy` | 44 | -0.2500 | -0.2277 | -0.0033 | -0.0000 | -0.0000 | -0.0000 | 23 |
| `new` tensorneat (evoxc) — aggregate | `numpy` | 42, 43, 44 | -0.2500 | -0.2450 | -0.0255 | -0.0000 | -0.0000 | -0.0000 | 28 |

### xor3d_p100

| Implementation | Backend | Seed | Best @1 | Best @10 | Best @25% | Best @50% | Best @100% | Final | Gens to target |
|---|---|---|---|---|---|---|---|---|---|
| `old` tensorneat-legacy (JAX) | `jax` | 42 | -0.2524 | -0.2500 | -0.2500 | -0.2447 | -0.0278 | -0.0278 | 149 |
| `old` tensorneat-legacy (JAX) | `jax` | 43 | -0.2547 | -0.2500 | -0.2500 | -0.2500 | -0.1224 | -0.1224 | >153 |
| `old` tensorneat-legacy (JAX) | `jax` | 44 | -0.2530 | -0.2500 | -0.2500 | -0.2496 | -0.0894 | -0.0894 | >153 |
| `old` tensorneat-legacy (JAX) — aggregate | `jax` | 42, 43, 44 | -0.2530 | -0.2500 | -0.2500 | -0.2496 | -0.0894 | -0.0894 | 149 |
| `new` tensorneat (evoxc) | `numpy` | 42 | -0.2516 | -0.2500 | -0.2500 | -0.2500 | -0.2500 | -0.2500 | >152 |
| `new` tensorneat (evoxc) | `numpy` | 43 | -0.2503 | -0.2500 | -0.2500 | -0.2500 | -0.2500 | -0.2500 | >152 |
| `new` tensorneat (evoxc) | `numpy` | 44 | -0.2530 | -0.2500 | -0.2500 | -0.2500 | -0.2500 | -0.2500 | >152 |
| `new` tensorneat (evoxc) — aggregate | `numpy` | 42, 43, 44 | -0.2516 | -0.2500 | -0.2500 | -0.2500 | -0.2500 | -0.2500 | >152 |

### xor3d_p500

| Implementation | Backend | Seed | Best @1 | Best @10 | Best @25% | Best @50% | Best @100% | Final | Gens to target |
|---|---|---|---|---|---|---|---|---|---|
| `old` tensorneat-legacy (JAX) | `jax` | 42 | -0.2524 | -0.2500 | -0.2500 | -0.2500 | -0.2500 | -0.2500 | >153 |
| `old` tensorneat-legacy (JAX) | `jax` | 43 | -0.2507 | -0.2500 | -0.2500 | -0.2500 | -0.1789 | -0.1789 | >153 |
| `old` tensorneat-legacy (JAX) | `jax` | 44 | -0.2530 | -0.2500 | -0.2449 | -0.1325 | -0.0000 | -0.0000 | 87 |
| `old` tensorneat-legacy (JAX) — aggregate | `jax` | 42, 43, 44 | -0.2524 | -0.2500 | -0.2500 | -0.2500 | -0.1789 | -0.1789 | 87 |
| `new` tensorneat (evoxc) | `numpy` | 42 | -0.2502 | -0.2500 | -0.2500 | -0.2500 | -0.2500 | -0.2500 | >152 |
| `new` tensorneat (evoxc) | `numpy` | 43 | -0.2503 | -0.2500 | -0.2500 | -0.2500 | -0.2500 | -0.2500 | >152 |
| `new` tensorneat (evoxc) | `numpy` | 44 | -0.2501 | -0.2500 | -0.2493 | -0.0677 | -0.0179 | -0.0179 | 95 |
| `new` tensorneat (evoxc) — aggregate | `numpy` | 42, 43, 44 | -0.2502 | -0.2500 | -0.2500 | -0.2500 | -0.2500 | -0.2500 | 95 |

### xor3d_p1000

| Implementation | Backend | Seed | Best @1 | Best @10 | Best @25% | Best @50% | Best @100% | Final | Gens to target |
|---|---|---|---|---|---|---|---|---|---|
| `old` tensorneat-legacy (JAX) | `jax` | 42 | -0.2524 | -0.2500 | -0.1850 | -0.0175 | -0.0000 | -0.0000 | 64 |
| `old` tensorneat-legacy (JAX) | `jax` | 43 | -0.2507 | -0.2500 | -0.2449 | -0.0794 | -0.0000 | -0.0000 | 86 |
| `old` tensorneat-legacy (JAX) | `jax` | 44 | -0.2515 | -0.2500 | -0.2378 | -0.0332 | -0.0000 | -0.0000 | 73 |
| `old` tensorneat-legacy (JAX) — aggregate | `jax` | 42, 43, 44 | -0.2515 | -0.2500 | -0.2378 | -0.0332 | -0.0000 | -0.0000 | 73 |
| `new` tensorneat (evoxc) | `numpy` | 42 | -0.2501 | -0.2500 | -0.2500 | -0.2500 | -0.0002 | -0.0002 | 135 |
| `new` tensorneat (evoxc) | `numpy` | 43 | -0.2502 | -0.2500 | -0.2386 | -0.1039 | -0.0001 | -0.0001 | 90 |
| `new` tensorneat (evoxc) | `numpy` | 44 | -0.2501 | -0.2500 | -0.2463 | -0.0014 | -0.0000 | -0.0000 | 60 |
| `new` tensorneat (evoxc) — aggregate | `numpy` | 42, 43, 44 | -0.2501 | -0.2500 | -0.2463 | -0.1039 | -0.0001 | -0.0001 | 90 |

### custom_pagie_p500

| Implementation | Backend | Seed | Best @1 | Best @10 | Best @25% | Best @50% | Best @100% | Final | Gens to target |
|---|---|---|---|---|---|---|---|---|---|
| `old` tensorneat-legacy (JAX) | `jax` | 42 | -0.0490 | -0.0473 | -0.0347 | -0.0233 | -0.0064 | -0.0064 | 125 |
| `old` tensorneat-legacy (JAX) | `jax` | 43 | -0.0493 | -0.0477 | -0.0243 | -0.0221 | -0.0040 | -0.0040 | 117 |
| `old` tensorneat-legacy (JAX) | `jax` | 44 | -0.0489 | -0.0474 | -0.0322 | -0.0232 | -0.0140 | -0.0140 | >153 |
| `old` tensorneat-legacy (JAX) — aggregate | `jax` | 42, 43, 44 | -0.0490 | -0.0474 | -0.0322 | -0.0232 | -0.0064 | -0.0064 | 121 |
| `new` tensorneat (evoxc) | `numpy` | 42 | -0.0461 | -0.0460 | -0.0240 | -0.0221 | -0.0219 | -0.0219 | >152 |
| `new` tensorneat (evoxc) | `numpy` | 43 | -0.0461 | -0.0460 | -0.0236 | -0.0220 | -0.0218 | -0.0218 | >152 |
| `new` tensorneat (evoxc) | `numpy` | 44 | -0.0461 | -0.0461 | -0.0460 | -0.0262 | -0.0248 | -0.0248 | >152 |
| `new` tensorneat (evoxc) — aggregate | `numpy` | 42, 43, 44 | -0.0461 | -0.0460 | -0.0240 | -0.0221 | -0.0219 | -0.0219 | >152 |

### cartpole_p100

| Implementation | Backend | Seed | Best @1 | Best @10 | Best @25% | Best @50% | Best @100% | Final | Gens to target |
|---|---|---|---|---|---|---|---|---|---|
| `new` tensorneat (evoxc) | `numpy` | 42 | 145.0000 | 200.0000 | 200.0000 | 200.0000 | 200.0000 | 200.0000 | 2 |
| `new` tensorneat (evoxc) | `numpy` | 43 | 200.0000 | 200.0000 | 200.0000 | 200.0000 | 200.0000 | 200.0000 | 1 |
| `new` tensorneat (evoxc) | `numpy` | 44 | 200.0000 | 200.0000 | 200.0000 | 200.0000 | 200.0000 | 200.0000 | 1 |
| `new` tensorneat (evoxc) — aggregate | `numpy` | 42, 43, 44 | 200.0000 | 200.0000 | 200.0000 | 200.0000 | 200.0000 | 200.0000 | 1 |

Percent columns are generation `round(fraction × N)` of the N recorded generations (warmup + timed — the budget `generations_to_target` is indexed against), and `>N` means the case's fitness target was not reached within those N recorded generations. Each repeat (seed) has its own row, and the `— aggregate` row is the element-wise median of those repeats: a bimodal case (one seed at the target, one not) shows up as two different `Final` values rather than one average.

## Reference numbers (declared baseline)

| Case | Pop | Declared legacy ms/gen | Measured old ms/gen | Ratio (measured/declared) |
|---|---|---|---|---|
| xor_p100 | 100 | 51.00 | 38.99 | 0.76x |
| xor_p500 | 500 | 109.00 | 295.33 | 2.71x |
| xor_p1000 | 1000 | 157.00 | 807.30 | 5.14x |

The performance contract in `../CONTEXT.md` declares the legacy CPU reference
baseline (51 ms/gen at pop 100, 109 ms/gen at pop 500, 157 ms/gen at pop 1000),
recorded as `reference_ms_per_generation` in `cases.py` and copied into every run
record. It is a reference figure, not a reading of this box: the declared numbers
reproduce at neither thread setting here, so the generator prints them next to the
old row measured in the same run and the verdict uses that measured row. The
win/lose section quotes the measured readings at both thread settings whenever it
has comparable pairs.

## What is not compared

- **Same workload, not the same search.** The sides use different PRNGs (JAX `PRNGKey` vs evoxc splitmix64), so one seed identifies the same *workload*, not the same trajectory or a bitwise-equal population. Numerical fidelity is `test/parity/`'s job: it compares kernels against `benchmarks/legacy_jax`, and `pytest test/parity --collect-only -q` collects 53 tests on this box.
- **No accelerator row in this report.** The old-vs-new verdict is CPU-only and numpy-only by protocol, so the matrix here is the numpy reference interpreter; an `iree`/`xla`/`tvm` row is informational and never enters the verdict. The compiled leg is measured separately in `benchmarks/BACKENDS.md`: `iree` lowers, compiles and runs every shipped case, and that file carries the measured per-generation comparison with numpy — an accelerator row exists, it simply stays outside this verdict.
- **CartPole has no old-side row.** The frozen library's RL problems need gym/brax/mujoco, so `cartpole_p100` is new-only and marked `n/a` for old.
- **Import time, visualization and code generation are not timed.** Only `rss_after_import_mb` is recorded for the import phase; `Pipeline.show`, network drawing and sympy codegen sit outside every timed region.
- **No trajectory equality is expected.** The two sides evolve different populations from the same workload seed; parity of kernels (not of search) is what `test/parity/` verifies.

## How to reproduce

```bash
# regenerate the committed report: the literal command of this run
python benchmarks/run_benchmark.py --generations 150 --warmup 3 --repeats 3 --json benchmarks/results/authoritative.json

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

Interpreter used: python

Cases: xor_p100, xor_p500, xor_p1000, xor3d_p100, xor3d_p500, xor3d_p1000, custom_pagie_p500, cartpole_p100. Timed generations: 150, warmup generations (untimed): 3, repeats: 3 (seeds 42, 43, 44), threads: runtime defaults (`--threads 0`). Backend(s) of the new side: numpy; the old side always runs on JAX, once per case. `--out` defaults to `benchmarks/RESULTS.md` and `--json` to `benchmarks/results/latest.json` (scratch, git-ignored); the record this report was rendered from (`authoritative.json`) is committed at `benchmarks/results/authoritative.json` beside that default, so every table cell is traceable. A thread-bound sensitivity run of the XOR family (`--threads 1`, same CLI) is recorded in `benchmarks/THREADS.md` next to this report.

Generated from `benchmarks/results/authoritative.json` into `benchmarks/RESULTS.md` at 2026-11-04T19:01:27+00:00 — do not hand-edit.
