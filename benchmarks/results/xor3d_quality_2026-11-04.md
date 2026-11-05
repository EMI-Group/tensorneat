# TensorNEAT post-fix quality measurement — XOR3d / Pagie (numpy backend, seeds 42/43/44)

> **GENERATED FILE.** Produced by `python benchmarks/run_benchmark.py`, rendered from `benchmarks/results/xor3d_quality_2026-11-04.json` at 2026-11-05T01:50:30+00:00.
> Re-rendered from that same JSON at 2026-11-05T04:52:08+00:00 with `--render-from-json` (no worker was re-run; the dirty state is re-probed on this tree).
> It must not be hand-edited: rerun the CLI instead, or the numbers below stop matching the recorded evidence.

*Post-search-quality-fix measurement of the xor3d and pagie cases; the committed `RESULTS.md`/`results/authoritative.json` pair is the pre-fix record and is left untouched.*

## Headline: old vs new

| Case | Pop | Backend | Old ms/gen | New ms/gen | Speed-up | Speed-up range (worst–best) | Quality overlap | Old compile (s) | New compile (s) | Old peak RSS (MiB) | New peak RSS (MiB) | Old final best | New final best | Gens to target (old) | Gens to target (new) | Timing | Quality |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| xor3d_p100 | 100 | `numpy` | 24.04 | 43.22 | 0.56x | 0.48x–1.05x | inside | 2.909 | 0.323 | 696.6 | 122.9 | -0.0894 | -0.2500 | 149 (1/3) | 128 (1/3) | FAIL | FAIL |
| xor3d_p500 | 500 | `numpy` | 104.87 | 189.93 | 0.55x | 0.48x–0.66x | inside | 2.933 | 0.566 | 707.4 | 272.9 | -0.1789 | -0.0000 | 87 (1/3) | 56 | FAIL | PASS |
| xor3d_p1000 | 1000 | `numpy` | 222.97 | 357.49 | 0.62x | 0.51x–1.04x | inside | 2.937 | 0.818 | 712.1 | 455.0 | -0.0000 | -0.0000 | 73 | 57 | FAIL | PASS |
| custom_pagie_p500 | 500 | `numpy` | 467.51 | 4530.73 | 0.10x | 0.09x–0.14x | inside | 2.949 | 2.256 | 850.7 | 4574.8 | -0.0064 | -0.0066 | 121 (2/3) | 95 | FAIL | PASS |

One row per case and new-side backend (the old side always runs on JAX). Speed-up is `old / new` per generation: above 1.00x the new side is faster. Speed-up range is `old_min/new_max` … `old_max/new_min` over the per-repeat medians, printed worst first: the left value is the pessimistic bound for the new side and the right one the optimistic bound. Quality overlap states where the new side's per-repeat final-best range sits against the old side's: `inside` when the ranges intersect (every new repeat inside the old range, or the ranges partially overlap), `new worse` when every new repeat is below the old range, `new better` when every new repeat is above it — an observation, never a replacement for the verdict. Timing PASS means new ms/generation is at or below the old one; quality PASS means the new final best stays within `max(10% × |old|, 0.001)` of the old one. `Gens to target` is the median generation of the repeats that reached the case's fitness target, annotated `(reached/total)` when only part of them did — `78 (2/3)` — or when none did — `>152 (0/3)`, and rendered `n/a (no informative target)` for a case measured without a fitness target at all. Every cell comes from the aggregated JSON records of this run; `n/a (reason)` marks a cell that could not be reached (not run, skipped, error, or not recorded).

## Environment

| Item | Value |
|---|---|
| Date (UTC) | 2026-11-05T01:50:30+00:00 |
| Python | 3.12.3 |
| Platform | Linux-6.8.0-146-generic-x86_64-with-glibc2.39 |
| CPU model | INTEL(R) XEON(R) PLATINUM 8568Y+ |
| CPU cores (`os.cpu_count`) | 20 |
| Memory | 241611 MiB |
| Load average (1/5/15 min) | 1.00 / 1.00 / 1.05 (1/5/15 min, document build); workers 1.00–3.45 at start, 1.00–3.45 at end (1-min, 24 run(s)) |
| numpy / jax / evoxc / tensorneat (installed metadata) | numpy 2.5.4 / jax 0.10.2 / evoxc 0.1.0 / tensorneat 0.2.0 |
| Git commit | `58657f68ae48` (clean — generated report artifacts are not counted) — `benchmarks/legacy_jax` and `src/` both come from this checkout |
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

## Case matrix

| Case | Family | Problem | Pop | Timed generations | Fitness target | Old | New | Notes |
|---|---|---|---|---|---|---|---|---|
| xor3d_p100 | func_fit | XOR3d | 100 | 150 | -0.05 | yes | yes | — |
| xor3d_p500 | func_fit | XOR3d | 500 | 150 | -0.05 | yes | yes | — |
| xor3d_p1000 | func_fit | XOR3d | 1000 | 150 | -0.05 | yes | yes | — |
| custom_pagie_p500 | func_fit | CustomFuncFit | 500 | 150 | -0.01 | yes | yes | per-impl divergence: the legacy side seeds the global numpy RNG and vmaps a jax-typed func; the new side samples from default_rng(seed) with a numpy func — same distribution, different draws |

## Results per case

### xor3d_p100

| Implementation | Backend | Compile (s) | ms/generation | generations/s | peak RSS (MiB) | Final best | Gens to target | Repeats |
|---|---|---|---|---|---|---|---|---|
| `old` tensorneat-legacy (JAX) | `jax` | 2.909 | 24.04 | 41.60 | 696.6 | -0.0894 | 149 (1/3) | 3 |
| `new` tensorneat (evoxc) | `numpy` | 0.323 | 43.22 | 23.14 | 122.9 | -0.2500 | 128 (1/3) | 3 |

### xor3d_p500

| Implementation | Backend | Compile (s) | ms/generation | generations/s | peak RSS (MiB) | Final best | Gens to target | Repeats |
|---|---|---|---|---|---|---|---|---|
| `old` tensorneat-legacy (JAX) | `jax` | 2.933 | 104.87 | 9.54 | 707.4 | -0.1789 | 87 (1/3) | 3 |
| `new` tensorneat (evoxc) | `numpy` | 0.566 | 189.93 | 5.27 | 272.9 | -0.0000 | 56 | 3 |

### xor3d_p1000

| Implementation | Backend | Compile (s) | ms/generation | generations/s | peak RSS (MiB) | Final best | Gens to target | Repeats |
|---|---|---|---|---|---|---|---|---|
| `old` tensorneat-legacy (JAX) | `jax` | 2.937 | 222.97 | 4.48 | 712.1 | -0.0000 | 73 | 3 |
| `new` tensorneat (evoxc) | `numpy` | 0.818 | 357.49 | 2.80 | 455.0 | -0.0000 | 57 | 3 |

### custom_pagie_p500

| Implementation | Backend | Compile (s) | ms/generation | generations/s | peak RSS (MiB) | Final best | Gens to target | Repeats |
|---|---|---|---|---|---|---|---|---|
| `old` tensorneat-legacy (JAX) | `jax` | 2.949 | 467.51 | 2.14 | 850.7 | -0.0064 | 121 (2/3) | 3 |
| `new` tensorneat (evoxc) | `numpy` | 2.256 | 4530.73 | 0.22 | 4574.8 | -0.0066 | 95 | 3 |

## Old vs new

| Case | Backend | Old ms/gen | New ms/gen | Speed-up | Delta ms/gen | Old final best | New final best | Quality delta | Timing | Quality |
|---|---|---|---|---|---|---|---|---|---|---|
| xor3d_p100 | `numpy` | 24.04 | 43.22 | 0.56x | +19.18 | -0.0894 | -0.2500 | -0.1606 | FAIL | FAIL |
| xor3d_p500 | `numpy` | 104.87 | 189.93 | 0.55x | +85.06 | -0.1789 | -0.0000 | +0.1789 | FAIL | PASS |
| xor3d_p1000 | `numpy` | 222.97 | 357.49 | 0.62x | +134.52 | -0.0000 | -0.0000 | +0.0000 | FAIL | PASS |
| custom_pagie_p500 | `numpy` | 467.51 | 4530.73 | 0.10x | +4063.22 | -0.0064 | -0.0066 | -0.0001 | FAIL | PASS |

Timing PASS means new ms/generation <= old; quality PASS means the new final best stays within the active tolerance `max(10% × |old|, 0.001)` of the old final best (the absolute floor keeps two runs that are both at the optimum from failing on float noise).

## Verdict

**FAIL**

- timing: xor3d_p100/numpy new 43.216 ms/gen > old 24.040 ms/gen (speed-up 0.556x)
- timing: xor3d_p500/numpy new 189.929 ms/gen > old 104.867 ms/gen (speed-up 0.552x)
- timing: xor3d_p1000/numpy new 357.489 ms/gen > old 222.967 ms/gen (speed-up 0.624x)
- timing: custom_pagie_p500/numpy new 4530.735 ms/gen > old 467.513 ms/gen (speed-up 0.103x)
- quality: xor3d_p100/numpy new final best -0.2500 < threshold -0.0983 (old -0.0894 - tolerance 0.0089, i.e. max(10% × \|old\|, 0.001 absolute floor))

Sub-verdicts: timing `FAIL`, quality `FAIL` — timing compares measured ms/generation, quality passes when `new >= old - max(10% × |old|, 0.001)`.

## Why the new side wins and loses

Per case, derived from the measured `old / new` speed-up:

- `xor3d_p100` (pop 100, family func_fit, backend `numpy`): **new loses** — 0.56x of the old speed (old 24.04 ms/gen → new 43.22 ms/gen; the old side is 1.80x faster).
- `xor3d_p500` (pop 500, family func_fit, backend `numpy`): **new loses** — 0.55x of the old speed (old 104.87 ms/gen → new 189.93 ms/gen; the old side is 1.81x faster).
- `xor3d_p1000` (pop 1000, family func_fit, backend `numpy`): **new loses** — 0.62x of the old speed (old 222.97 ms/gen → new 357.49 ms/gen; the old side is 1.60x faster).
- `custom_pagie_p500` (pop 500, family func_fit, backend `numpy`): **new loses** — 0.10x of the old speed (old 467.51 ms/gen → new 4530.73 ms/gen; the old side is 9.69x faster).

By population size:

- pop 100: **new loses** on all 1 comparable run(s) — 0.56x of the old speed (the old side is 1.80x faster).
- pop 500: **new loses** on all 2 comparable run(s) — 0.10x–0.55x of the old speed (the old side is 1.81x–9.69x faster).
- pop 1000: **new loses** on all 1 comparable run(s) — 0.62x of the old speed (the old side is 1.60x faster).

**Where the new side loses:** at the populations and edge counts above, the same per-op Python-loop dispatch is no longer amortised by edge-proportional work, while the old side's XLA program fuses the whole generation into compiled kernels and pays that dispatch once.

**Memory.** Peak RSS (`rss_final_mb`, the timed worker's high-water mark, taken as the max over the repeats) from the same runs, new vs old:

- `custom_pagie_p500` (pop 500, `numpy`): 4574.8 MiB vs 850.7 MiB — 5.38x (the new side uses more memory).
- `xor3d_p1000` (pop 1000, `numpy`): 455.0 MiB vs 712.1 MiB — 0.64x (the new side uses less memory).
- `xor3d_p500` (pop 500, `numpy`): 272.9 MiB vs 707.4 MiB — 0.39x (the new side uses less memory).
- `xor3d_p100` (pop 100, `numpy`): 122.9 MiB vs 696.6 MiB — 0.18x (the new side uses less memory).

Worst case: `custom_pagie_p500` at 5.38x (4574.8 vs 850.7 MiB old, +3724 MiB).

Mechanism, as far as these runs support one: the old side's floor is the JAX/XLA runtime itself (182.2–193.9 MiB right after import); the new side starts from 73.8–75.9 MiB. Beyond those floors the new peak stays within the range above while the old peak stays close to its runtime floor, so the footprint is the other axis from the compute claim above — the numpy interpreter materialises padded `pop_size × max_nodes × max_conns` tensors every generation, and it is that tensor geometry, not the live edge count, that the peak follows; these runs do not attribute that growth to a single dominant term.

This section is written from the same `speedup` values as the headline and old-vs-new tables; it does not assume the new side is the winner. A rerun that changes those numbers changes this text with them.

## Quality curves

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

Percent columns are generation `round(fraction × N)` of the N recorded generations (warmup + timed — the budget `generations_to_target` is indexed against). `>N` means the case's fitness target was not reached within those N recorded generations, and a `(reached/total)` suffix counts how many repeats of the group reached it: `78 (2/3)` is a median of 78 over two of three repeats, `>152 (0/3)` nothing reached it, and `n/a (no informative target)` marks a run measured without a fitness target at all. Each repeat (seed) has its own row, and the `— aggregate` row is the element-wise median of those repeats: a bimodal case (one seed at the target, one not) shows up as two different `Final` values rather than one average.

## Reference numbers (declared baseline)

no case declares a reference baseline

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
- **No accelerator row in this report.** The old-vs-new verdict is CPU-only and numpy-only by protocol, so the matrix here is the numpy reference interpreter; an `iree`/`xla`/`tvm` row is informational and never enters the verdict.
- **CartPole has no old-side row.** The frozen library's RL problems need gym/brax/mujoco, so `cartpole_p100` is new-only and marked `n/a` for old.
- **Import time, visualization and code generation are not timed.** Only `rss_after_import_mb` is recorded for the import phase; `Pipeline.show`, network drawing and sympy codegen sit outside every timed region.
- **No trajectory equality is expected.** The two sides evolve different populations from the same workload seed; parity of kernels (not of search) is what `test/parity/` verifies.

## How to reproduce

```bash
# regenerate the committed report: the literal command of this run
python benchmarks/run_benchmark.py --impl both --backend numpy --cases xor3d_p100,xor3d_p500,xor3d_p1000,custom_pagie_p500 --generations 150 --warmup 3 --repeats 3 --seed 42 --threads 0 --json benchmarks/results/xor3d_quality_2026-11-04.json --out benchmarks/results/xor3d_quality_2026-11-04.md --report-title 'TensorNEAT post-fix quality measurement — XOR3d / Pagie (numpy backend, seeds 42/43/44)' --report-note 'Post-search-quality-fix measurement of the xor3d and pagie cases; the committed `RESULTS.md`/`results/authoritative.json` pair is the pre-fix record and is left untouched.'

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

Cases: xor3d_p100, xor3d_p500, xor3d_p1000, custom_pagie_p500. Timed generations: 150, warmup generations (untimed): 3, repeats: 3 (seeds 42, 43, 44), threads: runtime defaults (`--threads 0`). Backend(s) of the new side: numpy; the old side always runs on JAX, once per case. `--out` defaults to `benchmarks/RESULTS.md` and `--json` to `benchmarks/results/latest.json` (scratch, git-ignored); the record this report was rendered from (`xor3d_quality_2026-11-04.json`) is committed at `benchmarks/results/xor3d_quality_2026-11-04.json` beside that default, so every table cell is traceable.

Generated from `benchmarks/results/xor3d_quality_2026-11-04.json` into `benchmarks/results/xor3d_quality_2026-11-04.md` at 2026-11-05T01:50:30+00:00 — do not hand-edit.
