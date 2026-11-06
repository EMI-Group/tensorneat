# TensorNEAT benchmark — evoxc rewrite vs frozen JAX baseline

> **GENERATED FILE.** Produced by `python benchmarks/run_benchmark.py`, rendered from `benchmarks/results/authoritative.json` at 2026-11-06T12:09:27+00:00.
> Re-rendered from that same JSON at 2026-11-06T12:22:13+00:00 with `--render-from-json` (no worker was re-run; the dirty state is re-probed on this tree).
> It must not be hand-edited: rerun the CLI instead, or the numbers below stop matching the recorded evidence.

*Full-matrix old-vs-new record of the merged tree: measured at `63f0fc67` on 2026-11-06 (producer run 11:31–12:09Z, document-build load average 1.00/1.09/1.39 on an otherwise idle box), so every cell here describes that tree and that session only — `PERF_NUMPY.md` holds the hand-measured per-pass before/after pairs and `results/xor3d_quality_2026-11-04.md` the narrower pre-tiling quality record.*

## Headline: old vs new

| Case | Pop | Backend | Old ms/gen | New ms/gen | Speed-up | Speed-up range (worst–best) | Quality overlap | Old compile (s) | New compile (s) | Old peak RSS (MiB) | New peak RSS (MiB) | Old final best | New final best | Gens to target (old) | Gens to target (new) | Timing | Quality |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| xor_p100 | 100 | `numpy` | 38.87 | 41.42 | 0.94x | 0.53x–1.17x | inside | 2.994 | 0.303 | 680.1 | 109.2 | -0.0034 | -0.0000 | 78 (2/3) | 40 | FAIL | PASS |
| xor_p500 | 500 | `numpy` | 308.49 | 119.02 | 2.59x | 2.08x–3.12x | new better | 2.965 | 0.448 | 693.6 | 191.5 | -0.0000 | -0.0000 | 42 | 28 | PASS | PASS |
| xor_p1000 | 1000 | `numpy` | 809.34 | 199.16 | 4.06x | 3.55x–4.12x | new better | 2.961 | 0.608 | 703.5 | 291.6 | -0.0000 | -0.0000 | 35 | 21 | PASS | PASS |
| xor3d_p100 | 100 | `numpy` | 24.63 | 36.02 | 0.68x | 0.56x–1.25x | inside | 2.940 | 0.314 | 690.0 | 110.9 | -0.0894 | -0.2500 | 149 (1/3) | 128 (1/3) | FAIL | FAIL |
| xor3d_p500 | 500 | `numpy` | 105.63 | 127.98 | 0.83x | 0.68x–0.99x | inside | 2.938 | 0.465 | 709.3 | 199.7 | -0.1789 | -0.0000 | 87 (1/3) | 56 | FAIL | PASS |
| xor3d_p1000 | 1000 | `numpy` | 226.66 | 212.41 | 1.07x | 0.83x–1.79x | inside | 2.924 | 0.599 | 715.8 | 310.9 | -0.0000 | -0.0000 | 73 | 57 | PASS | PASS |
| custom_pagie_p500 | 500 | `numpy` | 480.21 | 1258.04 | 0.38x | 0.34x–0.46x | inside | 2.942 | 1.547 | 849.9 | 1299.0 | -0.0064 | -0.0066 | 121 (2/3) | 95 | FAIL | PASS |
| cartpole_p100 | 100 | `numpy` | n/a (no old-side equivalent) | 307.16 | n/a (needs both sides) | n/a (needs both sides) | n/a (needs both sides) | n/a (no old-side equivalent) | 0.463 | n/a (no old-side equivalent) | 107.8 | n/a (no old-side equivalent) | 200.0000 | n/a (no old-side equivalent) | n/a (no informative target) | n/a | n/a |

One row per case and new-side backend (the old side always runs on JAX). Speed-up is `old / new` per generation: above 1.00x the new side is faster. Speed-up range is `old_min/new_max` … `old_max/new_min` over the per-repeat medians, printed worst first: the left value is the pessimistic bound for the new side and the right one the optimistic bound. Quality overlap states where the new side's per-repeat final-best range sits against the old side's: `inside` when the ranges intersect (every new repeat inside the old range, or the ranges partially overlap), `new worse` when every new repeat is below the old range, `new better` when every new repeat is above it — an observation, never a replacement for the verdict. Timing PASS means new ms/generation is at or below the old one; quality PASS means the new final best stays within `max(10% × |old|, 0.001)` of the old one. `Gens to target` is the median generation of the repeats that reached the case's fitness target, annotated `(reached/total)` when only part of them did — `78 (2/3)` — or when none did — `>152 (0/3)`, and rendered `n/a (no informative target)` for a case measured without a fitness target at all. Every cell comes from the aggregated JSON records of this run; `n/a (reason)` marks a cell that could not be reached (not run, skipped, error, or not recorded).

## Environment

| Item | Value |
|---|---|
| Date (UTC) | 2026-11-06T12:09:27+00:00 |
| Python | 3.12.3 |
| Platform | Linux-6.8.0-146-generic-x86_64-with-glibc2.39 |
| CPU model | INTEL(R) XEON(R) PLATINUM 8568Y+ |
| CPU cores (`os.cpu_count`) | 20 |
| Memory | 241611 MiB |
| Load average (1/5/15 min) | 1.00 / 1.09 / 1.39 (1/5/15 min, document build); workers 0.66–3.81 at start, 1.00–3.81 at end (1-min, 45 run(s)) |
| numpy / jax / evoxc / tensorneat (installed metadata) | numpy 2.5.4 / jax 0.10.2 / evoxc 0.1.0 / tensorneat 0.2.0 |
| Git commit | `63f0fc67a156` (clean — generated report artifacts are not counted) — `benchmarks/legacy_jax` and `src/` both come from this checkout |
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
| `old` tensorneat-legacy (JAX) | `jax` | 2.994 | 38.87 | 25.73 | 680.1 | -0.0034 | 78 (2/3) | 3 |
| `new` tensorneat (evoxc) | `numpy` | 0.303 | 41.42 | 24.14 | 109.2 | -0.0000 | 40 | 3 |

### xor_p500

| Implementation | Backend | Compile (s) | ms/generation | generations/s | peak RSS (MiB) | Final best | Gens to target | Repeats |
|---|---|---|---|---|---|---|---|---|
| `old` tensorneat-legacy (JAX) | `jax` | 2.965 | 308.49 | 3.24 | 693.6 | -0.0000 | 42 | 3 |
| `new` tensorneat (evoxc) | `numpy` | 0.448 | 119.02 | 8.40 | 191.5 | -0.0000 | 28 | 3 |

### xor_p1000

| Implementation | Backend | Compile (s) | ms/generation | generations/s | peak RSS (MiB) | Final best | Gens to target | Repeats |
|---|---|---|---|---|---|---|---|---|
| `old` tensorneat-legacy (JAX) | `jax` | 2.961 | 809.34 | 1.24 | 703.5 | -0.0000 | 35 | 3 |
| `new` tensorneat (evoxc) | `numpy` | 0.608 | 199.16 | 5.02 | 291.6 | -0.0000 | 21 | 3 |

### xor3d_p100

| Implementation | Backend | Compile (s) | ms/generation | generations/s | peak RSS (MiB) | Final best | Gens to target | Repeats |
|---|---|---|---|---|---|---|---|---|
| `old` tensorneat-legacy (JAX) | `jax` | 2.940 | 24.63 | 40.60 | 690.0 | -0.0894 | 149 (1/3) | 3 |
| `new` tensorneat (evoxc) | `numpy` | 0.314 | 36.02 | 27.76 | 110.9 | -0.2500 | 128 (1/3) | 3 |

### xor3d_p500

| Implementation | Backend | Compile (s) | ms/generation | generations/s | peak RSS (MiB) | Final best | Gens to target | Repeats |
|---|---|---|---|---|---|---|---|---|
| `old` tensorneat-legacy (JAX) | `jax` | 2.938 | 105.63 | 9.47 | 709.3 | -0.1789 | 87 (1/3) | 3 |
| `new` tensorneat (evoxc) | `numpy` | 0.465 | 127.98 | 7.81 | 199.7 | -0.0000 | 56 | 3 |

### xor3d_p1000

| Implementation | Backend | Compile (s) | ms/generation | generations/s | peak RSS (MiB) | Final best | Gens to target | Repeats |
|---|---|---|---|---|---|---|---|---|
| `old` tensorneat-legacy (JAX) | `jax` | 2.924 | 226.66 | 4.41 | 715.8 | -0.0000 | 73 | 3 |
| `new` tensorneat (evoxc) | `numpy` | 0.599 | 212.41 | 4.71 | 310.9 | -0.0000 | 57 | 3 |

### custom_pagie_p500

| Implementation | Backend | Compile (s) | ms/generation | generations/s | peak RSS (MiB) | Final best | Gens to target | Repeats |
|---|---|---|---|---|---|---|---|---|
| `old` tensorneat-legacy (JAX) | `jax` | 2.942 | 480.21 | 2.08 | 849.9 | -0.0064 | 121 (2/3) | 3 |
| `new` tensorneat (evoxc) | `numpy` | 1.547 | 1258.04 | 0.79 | 1299.0 | -0.0066 | 95 | 3 |

### cartpole_p100

| Implementation | Backend | Compile (s) | ms/generation | generations/s | peak RSS (MiB) | Final best | Gens to target | Repeats |
|---|---|---|---|---|---|---|---|---|
| `new` tensorneat (evoxc) | `numpy` | 0.463 | 307.16 | 3.26 | 107.8 | 200.0000 | n/a (no informative target) | 3 |

## Old vs new

| Case | Backend | Old ms/gen | New ms/gen | Speed-up | Delta ms/gen | Old final best | New final best | Quality delta | Timing | Quality |
|---|---|---|---|---|---|---|---|---|---|---|
| xor_p100 | `numpy` | 38.87 | 41.42 | 0.94x | +2.55 | -0.0034 | -0.0000 | +0.0034 | FAIL | PASS |
| xor_p500 | `numpy` | 308.49 | 119.02 | 2.59x | -189.46 | -0.0000 | -0.0000 | +0.0000 | PASS | PASS |
| xor_p1000 | `numpy` | 809.34 | 199.16 | 4.06x | -610.18 | -0.0000 | -0.0000 | +0.0000 | PASS | PASS |
| xor3d_p100 | `numpy` | 24.63 | 36.02 | 0.68x | +11.39 | -0.0894 | -0.2500 | -0.1606 | FAIL | FAIL |
| xor3d_p500 | `numpy` | 105.63 | 127.98 | 0.83x | +22.34 | -0.1789 | -0.0000 | +0.1789 | FAIL | PASS |
| xor3d_p1000 | `numpy` | 226.66 | 212.41 | 1.07x | -14.25 | -0.0000 | -0.0000 | +0.0000 | PASS | PASS |
| custom_pagie_p500 | `numpy` | 480.21 | 1258.04 | 0.38x | +777.84 | -0.0064 | -0.0066 | -0.0001 | FAIL | PASS |

Timing PASS means new ms/generation <= old; quality PASS means the new final best stays within the active tolerance `max(10% × |old|, 0.001)` of the old final best (the absolute floor keeps two runs that are both at the optimum from failing on float noise).

## Verdict

**FAIL**

- timing: xor_p100/numpy new 41.417 ms/gen > old 38.868 ms/gen (speed-up 0.938x)
- timing: xor3d_p100/numpy new 36.020 ms/gen > old 24.628 ms/gen (speed-up 0.684x)
- timing: xor3d_p500/numpy new 127.976 ms/gen > old 105.635 ms/gen (speed-up 0.825x)
- timing: custom_pagie_p500/numpy new 1258.042 ms/gen > old 480.205 ms/gen (speed-up 0.382x)
- quality: xor3d_p100/numpy new final best -0.2500 < threshold -0.0983 (old -0.0894 - tolerance 0.0089, i.e. max(10% × \|old\|, 0.001 absolute floor))

Sub-verdicts: timing `FAIL`, quality `FAIL` — timing compares measured ms/generation, quality passes when `new >= old - max(10% × |old|, 0.001)`.

## Why the new side wins and loses

Per case, derived from the measured `old / new` speed-up:

- `xor_p100` (pop 100, family func_fit, backend `numpy`): **new loses** — 0.94x of the old speed (old 38.87 ms/gen → new 41.42 ms/gen; the old side is 1.07x faster).
- `xor_p500` (pop 500, family func_fit, backend `numpy`): **new wins** — 2.59x faster than the old side (old 308.49 ms/gen → new 119.02 ms/gen).
- `xor_p1000` (pop 1000, family func_fit, backend `numpy`): **new wins** — 4.06x faster than the old side (old 809.34 ms/gen → new 199.16 ms/gen).
- `xor3d_p100` (pop 100, family func_fit, backend `numpy`): **new loses** — 0.68x of the old speed (old 24.63 ms/gen → new 36.02 ms/gen; the old side is 1.46x faster).
- `xor3d_p500` (pop 500, family func_fit, backend `numpy`): **new loses** — 0.83x of the old speed (old 105.63 ms/gen → new 127.98 ms/gen; the old side is 1.21x faster).
- `xor3d_p1000` (pop 1000, family func_fit, backend `numpy`): **new wins** — 1.07x faster than the old side (old 226.66 ms/gen → new 212.41 ms/gen).
- `custom_pagie_p500` (pop 500, family func_fit, backend `numpy`): **new loses** — 0.38x of the old speed (old 480.21 ms/gen → new 1258.04 ms/gen; the old side is 2.62x faster).

By population size:

- pop 100: **new loses** on all 2 comparable run(s) — 0.68x–0.94x of the old speed (the old side is 1.07x–1.46x faster).
- pop 500: **mixed** — new wins 1 of 3 comparable run(s) — speed-up range 0.38x–2.59x.
- pop 1000: **new wins** on all 2 comparable run(s) — 1.07x–4.06x faster.

**Where the new side wins:** the evoxc graph's *compute* is edge-proportional and depth-parallel — the forward pass walks the real connections and the network depth instead of evaluating `max_nodes × max_nodes` padded pairs — and on the numpy reference interpreter that useful work amortises its per-op Python-loop dispatch, while XLA's compiled whole-generation program carries more fixed overhead than it saves at small populations and low edge counts.

**Where the new side loses:** at the populations and edge counts above, the same per-op Python-loop dispatch is no longer amortised by edge-proportional work, while the old side's XLA program fuses the whole generation into compiled kernels and pays that dispatch once.

**Memory.** Peak RSS (`rss_final_mb`, the timed worker's high-water mark, taken as the max over the repeats) from the same runs, new vs old:

- `custom_pagie_p500` (pop 500, `numpy`): 1299.0 MiB vs 849.9 MiB — 1.53x (the new side uses more memory).
- `xor3d_p1000` (pop 1000, `numpy`): 310.9 MiB vs 715.8 MiB — 0.43x (the new side uses less memory).
- `xor_p1000` (pop 1000, `numpy`): 291.6 MiB vs 703.5 MiB — 0.41x (the new side uses less memory).
- `xor3d_p500` (pop 500, `numpy`): 199.7 MiB vs 709.3 MiB — 0.28x (the new side uses less memory).
- `xor_p500` (pop 500, `numpy`): 191.5 MiB vs 693.6 MiB — 0.28x (the new side uses less memory).
- `xor3d_p100` (pop 100, `numpy`): 110.9 MiB vs 690.0 MiB — 0.16x (the new side uses less memory).
- `xor_p100` (pop 100, `numpy`): 109.2 MiB vs 680.1 MiB — 0.16x (the new side uses less memory).

Worst case: `custom_pagie_p500` at 1.53x (1299.0 vs 849.9 MiB old, +449 MiB).
The new side's peak grows with the population in the XOR family (109.2 → 191.5 → 291.6 MiB at pop 100, 500, 1000), while the old row moves only from 680.1 to 703.5 MiB.

Mechanism, as far as these runs support one: the old side's floor is the JAX/XLA runtime itself (182.1–182.3 MiB right after import); the new side starts from 74.3–74.5 MiB. Beyond those floors the new peak tracks the population while the old peak stays close to its runtime floor, so the footprint is the other axis from the compute claim above — the numpy interpreter materialises padded `pop_size × max_nodes × max_conns` tensors every generation, and it is that tensor geometry, not the live edge count, that the peak follows; these runs do not attribute that growth to a single dominant term.

This section is written from the same `speedup` values as the headline and old-vs-new tables; it does not assume the new side is the winner. A rerun that changes those numbers changes this text with them.

Where the new side lands relative to the declared contract:

- Contract at pop 100 (`xor_p100`): 51.00 ms/gen declared; measured old 38.87 ms/gen (0.76x declared); measured new 41.42 ms/gen (0.81x declared).
- Contract at pop 500 (`xor_p500`): 109.00 ms/gen declared; measured old 308.49 ms/gen (2.83x declared); measured new 119.02 ms/gen (1.09x declared, i.e. the new side is slower than the contract here).
- Contract at pop 1000 (`xor_p1000`): 157.00 ms/gen declared; measured old 809.34 ms/gen (5.16x declared); measured new 199.16 ms/gen (1.27x declared, i.e. the new side is slower than the contract here).

The declared contract (51/109/157 ms/gen at pop 100/500/1000) is the recorded reference baseline (`reference_ms_per_generation` in `benchmarks/cases.py`, copied into every run record), not a number this box reproduces: `--threads 0` measured 38.87/308.49/809.34 ms/gen on the old side and `--threads 1` measured 35.79/127.71/252.06 ms/gen on the old side (`benchmarks/results/authoritative.json`, `benchmarks/results/threads1.json`). This run used `--threads 0`, so the comparison to the contract is informational and the measured old row on this box is what the verdict uses.

## Quality curves

### xor_p100

| Implementation | Backend | Seed | Best @1 | Best @10 | Best @25% | Best @50% | Best @100% | Final | Gens to target |
|---|---|---|---|---|---|---|---|---|---|
| `old` tensorneat-legacy (JAX) | `jax` | 42 | -0.2507 | -0.2500 | -0.1929 | -0.0542 | -0.0000 | -0.0000 | 62 |
| `old` tensorneat-legacy (JAX) | `jax` | 43 | -0.2519 | -0.2500 | -0.2364 | -0.1515 | -0.0034 | -0.0034 | 93 |
| `old` tensorneat-legacy (JAX) | `jax` | 44 | -0.2549 | -0.2500 | -0.2500 | -0.2500 | -0.2500 | -0.2500 | >153 |
| `old` tensorneat-legacy (JAX) — aggregate | `jax` | 42, 43, 44 | -0.2519 | -0.2500 | -0.2364 | -0.1515 | -0.0034 | -0.0034 | 78 (2/3) |
| `new` tensorneat (evoxc) | `numpy` | 42 | -0.2508 | -0.2437 | -0.1045 | -0.0034 | -0.0000 | -0.0000 | 40 |
| `new` tensorneat (evoxc) | `numpy` | 43 | -0.2506 | -0.2497 | -0.0962 | -0.0001 | -0.0000 | -0.0000 | 38 |
| `new` tensorneat (evoxc) | `numpy` | 44 | -0.2501 | -0.2500 | -0.1729 | -0.0402 | -0.0000 | -0.0000 | 65 |
| `new` tensorneat (evoxc) — aggregate | `numpy` | 42, 43, 44 | -0.2506 | -0.2497 | -0.1045 | -0.0034 | -0.0000 | -0.0000 | 40 |

### xor_p500

| Implementation | Backend | Seed | Best @1 | Best @10 | Best @25% | Best @50% | Best @100% | Final | Gens to target |
|---|---|---|---|---|---|---|---|---|---|
| `old` tensorneat-legacy (JAX) | `jax` | 42 | -0.2506 | -0.2477 | -0.1203 | -0.0199 | -0.0000 | -0.0000 | 42 |
| `old` tensorneat-legacy (JAX) | `jax` | 43 | -0.2505 | -0.2453 | -0.1461 | -0.0002 | -0.0000 | -0.0000 | 51 |
| `old` tensorneat-legacy (JAX) | `jax` | 44 | -0.2510 | -0.2484 | -0.0717 | -0.0006 | -0.0000 | -0.0000 | 34 |
| `old` tensorneat-legacy (JAX) — aggregate | `jax` | 42, 43, 44 | -0.2506 | -0.2477 | -0.1203 | -0.0006 | -0.0000 | -0.0000 | 42 |
| `new` tensorneat (evoxc) | `numpy` | 42 | -0.2500 | -0.2379 | -0.0350 | -0.0000 | -0.0000 | -0.0000 | 32 |
| `new` tensorneat (evoxc) | `numpy` | 43 | -0.2500 | -0.2414 | -0.0306 | -0.0000 | -0.0000 | -0.0000 | 28 |
| `new` tensorneat (evoxc) | `numpy` | 44 | -0.2501 | -0.2274 | -0.0270 | -0.0000 | -0.0000 | -0.0000 | 26 |
| `new` tensorneat (evoxc) — aggregate | `numpy` | 42, 43, 44 | -0.2500 | -0.2379 | -0.0306 | -0.0000 | -0.0000 | -0.0000 | 28 |

### xor_p1000

| Implementation | Backend | Seed | Best @1 | Best @10 | Best @25% | Best @50% | Best @100% | Final | Gens to target |
|---|---|---|---|---|---|---|---|---|---|
| `old` tensorneat-legacy (JAX) | `jax` | 42 | -0.2506 | -0.2427 | -0.0762 | -0.0005 | -0.0000 | -0.0000 | 35 |
| `old` tensorneat-legacy (JAX) | `jax` | 43 | -0.2505 | -0.2376 | -0.0234 | -0.0000 | -0.0000 | -0.0000 | 29 |
| `old` tensorneat-legacy (JAX) | `jax` | 44 | -0.2507 | -0.2491 | -0.1176 | -0.0001 | -0.0000 | -0.0000 | 39 |
| `old` tensorneat-legacy (JAX) — aggregate | `jax` | 42, 43, 44 | -0.2506 | -0.2427 | -0.0762 | -0.0001 | -0.0000 | -0.0000 | 35 |
| `new` tensorneat (evoxc) | `numpy` | 42 | -0.2501 | -0.2129 | -0.0002 | -0.0000 | -0.0000 | -0.0000 | 20 |
| `new` tensorneat (evoxc) | `numpy` | 43 | -0.2500 | -0.2143 | -0.0027 | -0.0000 | -0.0000 | -0.0000 | 21 |
| `new` tensorneat (evoxc) | `numpy` | 44 | -0.2500 | -0.2224 | -0.0024 | -0.0000 | -0.0000 | -0.0000 | 22 |
| `new` tensorneat (evoxc) — aggregate | `numpy` | 42, 43, 44 | -0.2500 | -0.2143 | -0.0024 | -0.0000 | -0.0000 | -0.0000 | 21 |

### xor3d_p100

| Implementation | Backend | Seed | Best @1 | Best @10 | Best @25% | Best @50% | Best @100% | Final | Gens to target |
|---|---|---|---|---|---|---|---|---|---|
| `old` tensorneat-legacy (JAX) | `jax` | 42 | -0.2524 | -0.2500 | -0.2500 | -0.2447 | -0.0278 | -0.0278 | 149 |
| `old` tensorneat-legacy (JAX) | `jax` | 43 | -0.2547 | -0.2500 | -0.2500 | -0.2500 | -0.1224 | -0.1224 | >153 |
| `old` tensorneat-legacy (JAX) | `jax` | 44 | -0.2530 | -0.2500 | -0.2500 | -0.2496 | -0.0894 | -0.0894 | >153 |
| `old` tensorneat-legacy (JAX) — aggregate | `jax` | 42, 43, 44 | -0.2530 | -0.2500 | -0.2500 | -0.2496 | -0.0894 | -0.0894 | 149 (1/3) |
| `new` tensorneat (evoxc) | `numpy` | 42 | -0.2516 | -0.2500 | -0.2500 | -0.1547 | -0.0143 | -0.0143 | 128 |
| `new` tensorneat (evoxc) | `numpy` | 43 | -0.2504 | -0.2500 | -0.2500 | -0.2500 | -0.2500 | -0.2500 | >152 |
| `new` tensorneat (evoxc) | `numpy` | 44 | -0.2530 | -0.2500 | -0.2500 | -0.2500 | -0.2500 | -0.2500 | >152 |
| `new` tensorneat (evoxc) — aggregate | `numpy` | 42, 43, 44 | -0.2516 | -0.2500 | -0.2500 | -0.2500 | -0.2500 | -0.2500 | 128 (1/3) |

### xor3d_p500

| Implementation | Backend | Seed | Best @1 | Best @10 | Best @25% | Best @50% | Best @100% | Final | Gens to target |
|---|---|---|---|---|---|---|---|---|---|
| `old` tensorneat-legacy (JAX) | `jax` | 42 | -0.2524 | -0.2500 | -0.2500 | -0.2500 | -0.2500 | -0.2500 | >153 |
| `old` tensorneat-legacy (JAX) | `jax` | 43 | -0.2507 | -0.2500 | -0.2500 | -0.2500 | -0.1789 | -0.1789 | >153 |
| `old` tensorneat-legacy (JAX) | `jax` | 44 | -0.2530 | -0.2500 | -0.2449 | -0.1325 | -0.0000 | -0.0000 | 87 |
| `old` tensorneat-legacy (JAX) — aggregate | `jax` | 42, 43, 44 | -0.2524 | -0.2500 | -0.2500 | -0.2500 | -0.1789 | -0.1789 | 87 (1/3) |
| `new` tensorneat (evoxc) | `numpy` | 42 | -0.2502 | -0.2498 | -0.1493 | -0.0004 | -0.0000 | -0.0000 | 56 |
| `new` tensorneat (evoxc) | `numpy` | 43 | -0.2504 | -0.2500 | -0.2500 | -0.1959 | -0.0000 | -0.0000 | 105 |
| `new` tensorneat (evoxc) | `numpy` | 44 | -0.2502 | -0.2500 | -0.1670 | -0.0041 | -0.0000 | -0.0000 | 53 |
| `new` tensorneat (evoxc) — aggregate | `numpy` | 42, 43, 44 | -0.2502 | -0.2500 | -0.1670 | -0.0041 | -0.0000 | -0.0000 | 56 |

### xor3d_p1000

| Implementation | Backend | Seed | Best @1 | Best @10 | Best @25% | Best @50% | Best @100% | Final | Gens to target |
|---|---|---|---|---|---|---|---|---|---|
| `old` tensorneat-legacy (JAX) | `jax` | 42 | -0.2524 | -0.2500 | -0.1850 | -0.0175 | -0.0000 | -0.0000 | 64 |
| `old` tensorneat-legacy (JAX) | `jax` | 43 | -0.2507 | -0.2500 | -0.2449 | -0.0794 | -0.0000 | -0.0000 | 86 |
| `old` tensorneat-legacy (JAX) | `jax` | 44 | -0.2515 | -0.2500 | -0.2378 | -0.0332 | -0.0000 | -0.0000 | 73 |
| `old` tensorneat-legacy (JAX) — aggregate | `jax` | 42, 43, 44 | -0.2515 | -0.2500 | -0.2378 | -0.0332 | -0.0000 | -0.0000 | 73 |
| `new` tensorneat (evoxc) | `numpy` | 42 | -0.2501 | -0.2500 | -0.1366 | -0.0000 | -0.0000 | -0.0000 | 46 |
| `new` tensorneat (evoxc) | `numpy` | 43 | -0.2501 | -0.2500 | -0.2483 | -0.1306 | -0.0000 | -0.0000 | 89 |
| `new` tensorneat (evoxc) | `numpy` | 44 | -0.2502 | -0.2500 | -0.2180 | -0.0028 | -0.0000 | -0.0000 | 57 |
| `new` tensorneat (evoxc) — aggregate | `numpy` | 42, 43, 44 | -0.2501 | -0.2500 | -0.2180 | -0.0028 | -0.0000 | -0.0000 | 57 |

### custom_pagie_p500

| Implementation | Backend | Seed | Best @1 | Best @10 | Best @25% | Best @50% | Best @100% | Final | Gens to target |
|---|---|---|---|---|---|---|---|---|---|
| `old` tensorneat-legacy (JAX) | `jax` | 42 | -0.0490 | -0.0473 | -0.0347 | -0.0233 | -0.0064 | -0.0064 | 125 |
| `old` tensorneat-legacy (JAX) | `jax` | 43 | -0.0493 | -0.0477 | -0.0243 | -0.0221 | -0.0040 | -0.0040 | 117 |
| `old` tensorneat-legacy (JAX) | `jax` | 44 | -0.0489 | -0.0474 | -0.0322 | -0.0232 | -0.0140 | -0.0140 | >153 |
| `old` tensorneat-legacy (JAX) — aggregate | `jax` | 42, 43, 44 | -0.0490 | -0.0474 | -0.0322 | -0.0232 | -0.0064 | -0.0064 | 121 (2/3) |
| `new` tensorneat (evoxc) | `numpy` | 42 | -0.0461 | -0.0443 | -0.0258 | -0.0143 | -0.0066 | -0.0066 | 95 |
| `new` tensorneat (evoxc) | `numpy` | 43 | -0.0461 | -0.0384 | -0.0227 | -0.0176 | -0.0068 | -0.0068 | 105 |
| `new` tensorneat (evoxc) | `numpy` | 44 | -0.0461 | -0.0456 | -0.0130 | -0.0052 | -0.0049 | -0.0049 | 43 |
| `new` tensorneat (evoxc) — aggregate | `numpy` | 42, 43, 44 | -0.0461 | -0.0443 | -0.0227 | -0.0143 | -0.0066 | -0.0066 | 95 |

### cartpole_p100

| Implementation | Backend | Seed | Best @1 | Best @10 | Best @25% | Best @50% | Best @100% | Final | Gens to target |
|---|---|---|---|---|---|---|---|---|---|
| `new` tensorneat (evoxc) | `numpy` | 42 | 145.0000 | 200.0000 | 200.0000 | 200.0000 | 200.0000 | 200.0000 | n/a (no informative target) |
| `new` tensorneat (evoxc) | `numpy` | 43 | 200.0000 | 200.0000 | 200.0000 | 200.0000 | 200.0000 | 200.0000 | n/a (no informative target) |
| `new` tensorneat (evoxc) | `numpy` | 44 | 200.0000 | 200.0000 | 200.0000 | 200.0000 | 200.0000 | 200.0000 | n/a (no informative target) |
| `new` tensorneat (evoxc) — aggregate | `numpy` | 42, 43, 44 | 200.0000 | 200.0000 | 200.0000 | 200.0000 | 200.0000 | 200.0000 | n/a (no informative target) |

Percent columns are generation `round(fraction × N)` of the N recorded generations (warmup + timed — the budget `generations_to_target` is indexed against). `>N` means the case's fitness target was not reached within those N recorded generations, and a `(reached/total)` suffix counts how many repeats of the group reached it: `78 (2/3)` is a median of 78 over two of three repeats, `>152 (0/3)` nothing reached it, and `n/a (no informative target)` marks a run measured without a fitness target at all. Each repeat (seed) has its own row, and the `— aggregate` row is the element-wise median of those repeats: a bimodal case (one seed at the target, one not) shows up as two different `Final` values rather than one average.

## Reference numbers (declared baseline)

| Case | Pop | Declared legacy ms/gen | Measured old ms/gen | Ratio (measured/declared) |
|---|---|---|---|---|
| xor_p100 | 100 | 51.00 | 38.87 | 0.76x |
| xor_p500 | 500 | 109.00 | 308.49 | 2.83x |
| xor_p1000 | 1000 | 157.00 | 809.34 | 5.16x |

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
python benchmarks/run_benchmark.py --impl both --backend numpy --cases all --generations 150 --warmup 3 --repeats 3 --json benchmarks/results/authoritative.json --out benchmarks/RESULTS.md

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

Interpreter used: .venv/bin/python

Cases: xor_p100, xor_p500, xor_p1000, xor3d_p100, xor3d_p500, xor3d_p1000, custom_pagie_p500, cartpole_p100. Timed generations: 150, warmup generations (untimed): 3, repeats: 3 (seeds 42, 43, 44), threads: runtime defaults (`--threads 0`). Backend(s) of the new side: numpy; the old side always runs on JAX, once per case. `--out` defaults to `benchmarks/RESULTS.md` and `--json` to `benchmarks/results/latest.json` (scratch, git-ignored); the record this report was rendered from (`authoritative.json`) is committed at `benchmarks/results/authoritative.json` beside that default, so every table cell is traceable. A thread-bound sensitivity run of the XOR family (`--threads 1`, same CLI) is recorded in `benchmarks/THREADS.md` next to this report.

Generated from `benchmarks/results/authoritative.json` into `benchmarks/RESULTS.md` at 2026-11-06T12:09:27+00:00 — do not hand-edit.
