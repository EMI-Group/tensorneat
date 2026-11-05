# TensorNEAT benchmark — thread-bound sensitivity (XOR family, --threads 1)

> **GENERATED FILE.** Produced by `python benchmarks/run_benchmark.py`, rendered from `benchmarks/results/threads1.json` at 2026-11-02T06:31:32+00:00.
> Re-rendered from that same JSON at 2026-11-05T04:05:58+00:00 with `--render-from-json` (no worker was re-run; the dirty state is re-probed on this tree).
> It must not be hand-edited: rerun the CLI instead, or the numbers below stop matching the recorded evidence.

*This is the thread-bound sensitivity run (`--threads 1`, the declared-contract condition) of the XOR family; the authoritative default-thread matrix is `RESULTS.md`.*

## Headline: old vs new

| Case | Pop | Backend | Old ms/gen | New ms/gen | Speed-up | Speed-up range (best–worst) | Quality overlap | Old compile (s) | New compile (s) | Old peak RSS (MiB) | New peak RSS (MiB) | Old final best | New final best | Gens to target (old) | Gens to target (new) | Timing | Quality |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| xor_p100 | 100 | `numpy` | 35.79 | 53.92 | 0.66x | 0.56x–0.77x | inside | 2.816 | 0.539 | 578.4 | 145.3 | -0.0099 | -0.0303 | 78 | 98 | FAIL | FAIL |
| xor_p500 | 500 | `numpy` | 127.71 | 360.12 | 0.35x | 0.28x–0.43x | new better | 2.713 | 1.469 | 615.9 | 1049.6 | -0.0000 | -0.0000 | 39 | 31 | FAIL | PASS |
| xor_p1000 | 1000 | `numpy` | 252.06 | 665.46 | 0.38x | 0.37x–0.39x | inside | 2.689 | 2.163 | 627.5 | 1991.6 | -0.0000 | -0.0000 | 32 | 29 | FAIL | PASS |

One row per case and new-side backend (the old side always runs on JAX). Speed-up is `old / new` per generation: above 1.00x the new side is faster. Speed-up range is `old_min/new_max` … `old_max/new_min` over the per-repeat medians, i.e. the best and worst case over the repeats. Quality overlap states where the new side's per-repeat final-best range sits against the old side's: `inside` when the ranges intersect (every new repeat inside the old range, or the ranges partially overlap), `new worse` when every new repeat is below the old range, `new better` when every new repeat is above it — an observation, never a replacement for the verdict. Timing PASS means new ms/generation is at or below the old one; quality PASS means the new final best stays within `max(10% × |old|, 0.001)` of the old one. Every cell comes from the aggregated JSON records of this run; `n/a (reason)` marks a cell that could not be reached (not run, skipped, error, or not recorded).

## Environment

| Item | Value |
|---|---|
| Date (UTC) | 2026-11-02T06:31:32+00:00 |
| Python | 3.12.3 |
| Platform | Linux-6.8.0-146-generic-x86_64-with-glibc2.39 |
| CPU model | INTEL(R) XEON(R) PLATINUM 8568Y+ |
| CPU cores (`os.cpu_count`) | 20 |
| Memory | 241611 MiB |
| Load average (1/5/15 min) | 1.44 / 2.25 / 2.35 (1/5/15 min, document build); workers 1.12–5.20 at start, 1.26–5.20 at end (1-min, 12 run(s)) |
| numpy / jax / evoxc / tensorneat (installed metadata) | numpy 2.5.4 / jax 0.12.0.1 / evoxc 0.1.0 / tensorneat n/a |
| Git commit | `89f6200d6bba` (clean — generated report artifacts are not counted) — `benchmarks/legacy_jax` and `src/` both come from this checkout |
| Thread pinning (`--threads`) | `--threads 1` — exported to OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS and, on the old side, to `XLA_FLAGS` intra-op parallelism |
| Thread env (set/unset) | OMP_NUM_THREADS unset · MKL_NUM_THREADS unset · OPENBLAS_NUM_THREADS unset · NUMEXPR_NUM_THREADS unset · XLA_FLAGS unset (CLI process, inherited by every worker before the export) — every worker exports the four `*_NUM_THREADS` variables as 1 and sets `XLA_FLAGS` on the old side before numpy/jax import |

**Shared box.** These numbers were measured on a machine that was not exclusively ours: other jobs ran on it while the workers timed generations, so the per-repeat spread (the headline's speed-up range and the per-repeat quality-curve rows) is the honest uncertainty measure — a median alone would hide it.

## Protocol

| Parameter | Value |
|---|---|
| Backend(s) of the new implementation | `numpy` |
| Warmup generations (untimed) | 3 |
| Timed generations | 150 |
| Repeats (seed + r) | 2 (`seed + r`, seeds 42, 43) |
| Thread pinning | `--threads 1` — exported to OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS and, on the old side, to `XLA_FLAGS` intra-op parallelism |
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
| xor_p100 | func_fit | XOR | 100 | 150 | -0.1 | yes | yes | — |
| xor_p500 | func_fit | XOR | 500 | 150 | -0.1 | yes | yes | — |
| xor_p1000 | func_fit | XOR | 1000 | 150 | -0.1 | yes | yes | — |

## Results per case

### xor_p100

| Implementation | Backend | Compile (s) | ms/generation | generations/s | peak RSS (MiB) | Final best | Gens to target | Repeats |
|---|---|---|---|---|---|---|---|---|
| `old` tensorneat-legacy (JAX) | `jax` | 2.816 | 35.79 | 27.94 | 578.4 | -0.0099 | 78 | 2 |
| `new` tensorneat (evoxc) | `numpy` | 0.539 | 53.92 | 18.54 | 145.3 | -0.0303 | 98 | 2 |

### xor_p500

| Implementation | Backend | Compile (s) | ms/generation | generations/s | peak RSS (MiB) | Final best | Gens to target | Repeats |
|---|---|---|---|---|---|---|---|---|
| `old` tensorneat-legacy (JAX) | `jax` | 2.713 | 127.71 | 7.83 | 615.9 | -0.0000 | 39 | 2 |
| `new` tensorneat (evoxc) | `numpy` | 1.469 | 360.12 | 2.78 | 1049.6 | -0.0000 | 31 | 2 |

### xor_p1000

| Implementation | Backend | Compile (s) | ms/generation | generations/s | peak RSS (MiB) | Final best | Gens to target | Repeats |
|---|---|---|---|---|---|---|---|---|
| `old` tensorneat-legacy (JAX) | `jax` | 2.689 | 252.06 | 3.97 | 627.5 | -0.0000 | 32 | 2 |
| `new` tensorneat (evoxc) | `numpy` | 2.163 | 665.46 | 1.50 | 1991.6 | -0.0000 | 29 | 2 |

## Old vs new

| Case | Backend | Old ms/gen | New ms/gen | Speed-up | Delta ms/gen | Old final best | New final best | Quality delta | Timing | Quality |
|---|---|---|---|---|---|---|---|---|---|---|
| xor_p100 | `numpy` | 35.79 | 53.92 | 0.66x | +18.14 | -0.0099 | -0.0303 | -0.0204 | FAIL | FAIL |
| xor_p500 | `numpy` | 127.71 | 360.12 | 0.35x | +232.41 | -0.0000 | -0.0000 | +0.0000 | FAIL | PASS |
| xor_p1000 | `numpy` | 252.06 | 665.46 | 0.38x | +413.40 | -0.0000 | -0.0000 | -0.0000 | FAIL | PASS |

Timing PASS means new ms/generation <= old; quality PASS means the new final best stays within the active tolerance `max(10% × |old|, 0.001)` of the old final best (the absolute floor keeps two runs that are both at the optimum from failing on float noise).

## Verdict

**FAIL**

- timing: xor_p100/numpy new 53.923 ms/gen > old 35.786 ms/gen (speed-up 0.664x)
- timing: xor_p500/numpy new 360.124 ms/gen > old 127.709 ms/gen (speed-up 0.355x)
- timing: xor_p1000/numpy new 665.458 ms/gen > old 252.061 ms/gen (speed-up 0.379x)
- quality: xor_p100/numpy new final best -0.0303 < threshold -0.0109 (old -0.0099 - tolerance 0.0010, i.e. max(10% × \|old\|, 0.001 absolute floor))

Sub-verdicts: timing `FAIL`, quality `FAIL` — timing compares measured ms/generation, quality passes when `new >= old - max(10% × |old|, 0.001)`.

## Why the new side wins and loses

Per case, derived from the measured `old / new` speed-up:

- `xor_p100` (pop 100, family func_fit, backend `numpy`): **new loses** — 0.66x of the old speed (old 35.79 ms/gen → new 53.92 ms/gen; the old side is 1.51x faster).
- `xor_p500` (pop 500, family func_fit, backend `numpy`): **new loses** — 0.35x of the old speed (old 127.71 ms/gen → new 360.12 ms/gen; the old side is 2.82x faster).
- `xor_p1000` (pop 1000, family func_fit, backend `numpy`): **new loses** — 0.38x of the old speed (old 252.06 ms/gen → new 665.46 ms/gen; the old side is 2.64x faster).

By population size:

- pop 100: **new loses** on all 1 comparable run(s) — 0.66x of the old speed (the old side is 1.51x faster).
- pop 500: **new loses** on all 1 comparable run(s) — 0.35x of the old speed (the old side is 2.82x faster).
- pop 1000: **new loses** on all 1 comparable run(s) — 0.38x of the old speed (the old side is 2.64x faster).

**Where the new side loses:** at the populations and edge counts above, the same per-op Python-loop dispatch is no longer amortised by edge-proportional work, while the old side's XLA program fuses the whole generation into compiled kernels and pays that dispatch once.

**Memory.** Peak RSS (`rss_final_mb`, the timed worker's high-water mark, taken as the max over the repeats) from the same runs, new vs old:

- `xor_p1000` (pop 1000, `numpy`): 1991.6 MiB vs 627.5 MiB — 3.17x (the new side uses more memory).
- `xor_p500` (pop 500, `numpy`): 1049.6 MiB vs 615.9 MiB — 1.70x (the new side uses more memory).
- `xor_p100` (pop 100, `numpy`): 145.3 MiB vs 578.4 MiB — 0.25x (the new side uses less memory).

Worst case: `xor_p1000` at 3.17x (1991.6 vs 627.5 MiB old, +1364 MiB).
The new side's peak grows with the population in the XOR family (145.3 → 1049.6 → 1991.6 MiB at pop 100, 500, 1000), while the old row moves only from 578.4 to 627.5 MiB.

Mechanism, as far as these runs support one: the old side's floor is the JAX/XLA runtime itself (192.2–192.4 MiB right after import); the new side starts from 72.8 MiB. Beyond those floors the new peak tracks the population while the old peak stays close to its runtime floor, so the footprint is the other axis from the compute claim above — the numpy interpreter materialises padded `pop_size × max_nodes × max_conns` tensors every generation, and it is that tensor geometry, not the live edge count, that the peak follows; these runs do not attribute that growth to a single dominant term.

This section is written from the same `speedup` values as the headline and old-vs-new tables; it does not assume the new side is the winner. A rerun that changes those numbers changes this text with them.

Where the new side lands relative to the declared contract:

- Contract at pop 100 (`xor_p100`): 51.00 ms/gen declared; measured old 35.79 ms/gen (0.70x declared); measured new 53.92 ms/gen (1.06x declared, i.e. the new side is slower than the contract here).
- Contract at pop 500 (`xor_p500`): 109.00 ms/gen declared; measured old 127.71 ms/gen (1.17x declared); measured new 360.12 ms/gen (3.30x declared, i.e. the new side is slower than the contract here).
- Contract at pop 1000 (`xor_p1000`): 157.00 ms/gen declared; measured old 252.06 ms/gen (1.61x declared); measured new 665.46 ms/gen (4.24x declared, i.e. the new side is slower than the contract here).

The declared contract (51/109/157 ms/gen at pop 100/500/1000) is the recorded reference baseline (`reference_ms_per_generation` in `benchmarks/cases.py`, copied into every run record), not a number this box reproduces: `--threads 0` measured 38.99/295.33/807.30 ms/gen on the old side and `--threads 1` measured 35.79/127.71/252.06 ms/gen on the old side (`benchmarks/results/authoritative.json`, `benchmarks/results/threads1.json`). This run used `--threads 1`, so the comparison to the contract is informational and the measured old row on this box is what the verdict uses.

## Quality curves

### xor_p100

| Implementation | Backend | Seed | Best @1 | Best @10 | Best @25% | Best @50% | Best @100% | Final | Gens to target |
|---|---|---|---|---|---|---|---|---|---|
| `old` tensorneat-legacy (JAX) | `jax` | 42 | -0.2507 | -0.2500 | -0.1929 | -0.0542 | -0.0000 | -0.0000 | 62 |
| `old` tensorneat-legacy (JAX) | `jax` | 43 | -0.2519 | -0.2500 | -0.2364 | -0.1515 | -0.0198 | -0.0198 | 93 |
| `old` tensorneat-legacy (JAX) — aggregate | `jax` | 42, 43 | -0.2513 | -0.2500 | -0.2147 | -0.1028 | -0.0099 | -0.0099 | 78 |
| `new` tensorneat (evoxc) | `numpy` | 42 | -0.2502 | -0.2500 | -0.2205 | -0.0067 | -0.0000 | -0.0000 | 55 |
| `new` tensorneat (evoxc) | `numpy` | 43 | -0.2505 | -0.2500 | -0.2500 | -0.2500 | -0.0606 | -0.0606 | 140 |
| `new` tensorneat (evoxc) — aggregate | `numpy` | 42, 43 | -0.2504 | -0.2500 | -0.2352 | -0.1283 | -0.0303 | -0.0303 | 98 |

### xor_p500

| Implementation | Backend | Seed | Best @1 | Best @10 | Best @25% | Best @50% | Best @100% | Final | Gens to target |
|---|---|---|---|---|---|---|---|---|---|
| `old` tensorneat-legacy (JAX) | `jax` | 42 | -0.2506 | -0.2477 | -0.1152 | -0.0055 | -0.0000 | -0.0000 | 41 |
| `old` tensorneat-legacy (JAX) | `jax` | 43 | -0.2505 | -0.2453 | -0.0795 | -0.0002 | -0.0000 | -0.0000 | 37 |
| `old` tensorneat-legacy (JAX) — aggregate | `jax` | 42, 43 | -0.2505 | -0.2465 | -0.0973 | -0.0028 | -0.0000 | -0.0000 | 39 |
| `new` tensorneat (evoxc) | `numpy` | 42 | -0.2500 | -0.2421 | -0.0235 | -0.0000 | -0.0000 | -0.0000 | 28 |
| `new` tensorneat (evoxc) | `numpy` | 43 | -0.2500 | -0.2500 | -0.0496 | -0.0000 | -0.0000 | -0.0000 | 33 |
| `new` tensorneat (evoxc) — aggregate | `numpy` | 42, 43 | -0.2500 | -0.2461 | -0.0365 | -0.0000 | -0.0000 | -0.0000 | 31 |

### xor_p1000

| Implementation | Backend | Seed | Best @1 | Best @10 | Best @25% | Best @50% | Best @100% | Final | Gens to target |
|---|---|---|---|---|---|---|---|---|---|
| `old` tensorneat-legacy (JAX) | `jax` | 42 | -0.2506 | -0.2427 | -0.0780 | -0.0004 | -0.0000 | -0.0000 | 35 |
| `old` tensorneat-legacy (JAX) | `jax` | 43 | -0.2505 | -0.2376 | -0.0234 | -0.0000 | -0.0000 | -0.0000 | 29 |
| `old` tensorneat-legacy (JAX) — aggregate | `jax` | 42, 43 | -0.2505 | -0.2402 | -0.0507 | -0.0002 | -0.0000 | -0.0000 | 32 |
| `new` tensorneat (evoxc) | `numpy` | 42 | -0.2501 | -0.2416 | -0.0391 | -0.0000 | -0.0000 | -0.0000 | 31 |
| `new` tensorneat (evoxc) | `numpy` | 43 | -0.2500 | -0.2391 | -0.0047 | -0.0000 | -0.0000 | -0.0000 | 27 |
| `new` tensorneat (evoxc) — aggregate | `numpy` | 42, 43 | -0.2500 | -0.2403 | -0.0219 | -0.0000 | -0.0000 | -0.0000 | 29 |

Percent columns are generation `round(fraction × N)` of the N recorded generations (warmup + timed — the budget `generations_to_target` is indexed against), and `>N` means the case's fitness target was not reached within those N recorded generations. Each repeat (seed) has its own row, and the `— aggregate` row is the element-wise median of those repeats: a bimodal case (one seed at the target, one not) shows up as two different `Final` values rather than one average.

## Reference numbers (declared baseline)

| Case | Pop | Declared legacy ms/gen | Measured old ms/gen | Ratio (measured/declared) |
|---|---|---|---|---|
| xor_p100 | 100 | 51.00 | 35.79 | 0.70x |
| xor_p500 | 500 | 109.00 | 127.71 | 1.17x |
| xor_p1000 | 1000 | 157.00 | 252.06 | 1.61x |

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
python benchmarks/run_benchmark.py --impl both --backend numpy --cases xor_p100,xor_p500,xor_p1000 --generations 150 --warmup 3 --repeats 2 --threads 1 --json benchmarks/results/threads1.json --out benchmarks/THREADS.md --report-title 'TensorNEAT benchmark — thread-bound sensitivity (XOR family, --threads 1)' --report-note 'This is the thread-bound sensitivity run (`--threads 1`, the declared-contract condition) of the XOR family; the authoritative default-thread matrix is `RESULTS.md`.'

# full matrix, both implementations on the primary backend
python benchmarks/run_benchmark.py --impl both --backend numpy --cases all --generations 150 --warmup 3 --repeats 2 --threads 1

# baseline only (frozen JAX)
python benchmarks/run_benchmark.py --impl old --cases all --generations 150 --warmup 3 --repeats 2 --threads 1

# candidate only, one backend at a time
python benchmarks/run_benchmark.py --impl new --backend numpy --cases all --generations 150 --warmup 3 --repeats 2 --threads 1
python benchmarks/run_benchmark.py --impl new --backend iree --cases xor_p500 --generations 150 --warmup 3 --repeats 2 --threads 1   # informational; needs the evoxc[iree] extra, and its numbers never enter the verdict

# inspect the matrix and the planned worker commands (no run, no report)
python benchmarks/run_benchmark.py --list
python benchmarks/run_benchmark.py --dry-run
```

Interpreter used: python

Cases: xor_p100, xor_p500, xor_p1000. Timed generations: 150, warmup generations (untimed): 3, repeats: 2 (seeds 42, 43), threads: `--threads 1`. Backend(s) of the new side: numpy; the old side always runs on JAX, once per case. `--out` defaults to `benchmarks/RESULTS.md` and `--json` to `benchmarks/results/latest.json` (scratch, git-ignored); the record this report was rendered from (`threads1.json`) is committed at `benchmarks/results/threads1.json` beside that default, so every table cell is traceable.

Generated from `benchmarks/results/threads1.json` into `benchmarks/THREADS.md` at 2026-11-02T06:31:32+00:00 — do not hand-edit.
