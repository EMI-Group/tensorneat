# TensorNEAT benchmark — IREE backend attempt (llvm-cpu)

> **GENERATED FILE.** Produced by `python benchmarks/run_benchmark.py`, rendered from
> `benchmarks/results/iree.json` at 2026-11-02T07:29:26+00:00. It must not be hand-edited:
> Re-rendered from that same JSON at 2026-11-04T07:48:56+00:00 with `--render-from-json` (no worker was re-run; the dirty state is re-probed on this tree).
> rerun the CLI instead, or the numbers below stop matching the recorded evidence.

*Historical pre-fix record, superseded by `BACKENDS.md` — the current measured `iree` evidence, where the backend lowers, compiles and runs every shipped case. Every case below errored in `Pipeline.setup` before generation 0, so no row here is current. The measured CPU contract is `RESULTS.md` (numpy reference backend).*

## Headline: old vs new

| Case | Pop | Backend | Old ms/gen | New ms/gen | Speed-up | Speed-up range (best–worst) | Quality overlap | Old compile (s) | New compile (s) | Old peak RSS (MiB) | New peak RSS (MiB) | Old final best | New final best | Gens to target (old) | Gens to target (new) | Timing | Quality |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| xor_p100 | 100 | `iree` | n/a (not run) | n/a (error) | n/a (needs both sides) | n/a (needs both sides) | n/a (needs both sides) | n/a (not run) | n/a (error) | n/a (not run) | n/a (error) | n/a (not run) | n/a (error) | n/a (not run) | n/a (error) | n/a | n/a |
| xor_p500 | 500 | `iree` | n/a (not run) | n/a (error) | n/a (needs both sides) | n/a (needs both sides) | n/a (needs both sides) | n/a (not run) | n/a (error) | n/a (not run) | n/a (error) | n/a (not run) | n/a (error) | n/a (not run) | n/a (error) | n/a | n/a |
| xor_p1000 | 1000 | `iree` | n/a (not run) | n/a (error) | n/a (needs both sides) | n/a (needs both sides) | n/a (needs both sides) | n/a (not run) | n/a (error) | n/a (not run) | n/a (error) | n/a (not run) | n/a (error) | n/a (not run) | n/a (error) | n/a | n/a |
| xor3d_p100 | 100 | `iree` | n/a (not run) | n/a (error) | n/a (needs both sides) | n/a (needs both sides) | n/a (needs both sides) | n/a (not run) | n/a (error) | n/a (not run) | n/a (error) | n/a (not run) | n/a (error) | n/a (not run) | n/a (error) | n/a | n/a |
| xor3d_p500 | 500 | `iree` | n/a (not run) | n/a (error) | n/a (needs both sides) | n/a (needs both sides) | n/a (needs both sides) | n/a (not run) | n/a (error) | n/a (not run) | n/a (error) | n/a (not run) | n/a (error) | n/a (not run) | n/a (error) | n/a | n/a |
| xor3d_p1000 | 1000 | `iree` | n/a (not run) | n/a (error) | n/a (needs both sides) | n/a (needs both sides) | n/a (needs both sides) | n/a (not run) | n/a (error) | n/a (not run) | n/a (error) | n/a (not run) | n/a (error) | n/a (not run) | n/a (error) | n/a | n/a |
| custom_pagie_p500 | 500 | `iree` | n/a (not run) | n/a (error) | n/a (needs both sides) | n/a (needs both sides) | n/a (needs both sides) | n/a (not run) | n/a (error) | n/a (not run) | n/a (error) | n/a (not run) | n/a (error) | n/a (not run) | n/a (error) | n/a | n/a |
| cartpole_p100 | 100 | `iree` | n/a (no old-side equivalent) | n/a (error) | n/a (needs both sides) | n/a (needs both sides) | n/a (needs both sides) | n/a (no old-side equivalent) | n/a (error) | n/a (no old-side equivalent) | n/a (error) | n/a (no old-side equivalent) | n/a (error) | n/a (no old-side equivalent) | n/a (error) | n/a | n/a |

One row per case and new-side backend (this document records no old-side row, so every case here is new-side only). Speed-up is `old / new` per generation: above 1.00x the new side is faster. Speed-up range is `old_min/new_max` … `old_max/new_min` over the per-repeat medians, i.e. the best and worst case over the repeats. Quality overlap states where the new side's per-repeat final-best range sits against the old side's: `inside` when the ranges intersect (every new repeat inside the old range, or the ranges partially overlap), `new worse` when every new repeat is below the old range, `new better` when every new repeat is above it — an observation, never a replacement for the verdict. Timing PASS means new ms/generation is at or below the old one; quality PASS means the new final best stays within `max(10% × |old|, 0.001)` of the old one. Every cell comes from the aggregated JSON records of this run; `n/a (reason)` marks a cell that could not be reached (not run, skipped, error, or not recorded).

- `xor_p100` / `new` / `iree`: **error** — evoxc.core.errors.BackendError: stablehlo export: op 'scatter' at /root/tensorneat/.genesis/workers/worker_T1_A293/src/tensorneat/algorithm/neat/species.py:295 on a rank-3 operand along axis 0 is not…
- `xor_p500` / `new` / `iree`: **error** — evoxc.core.errors.BackendError: stablehlo export: op 'scatter' at /root/tensorneat/.genesis/workers/worker_T1_A293/src/tensorneat/algorithm/neat/species.py:295 on a rank-3 operand along axis 0 is not…
- `xor_p1000` / `new` / `iree`: **error** — evoxc.core.errors.BackendError: stablehlo export: op 'scatter' at /root/tensorneat/.genesis/workers/worker_T1_A293/src/tensorneat/algorithm/neat/species.py:295 on a rank-3 operand along axis 0 is not…
- `xor3d_p100` / `new` / `iree`: **error** — evoxc.core.errors.BackendError: stablehlo export: op 'scatter' at /root/tensorneat/.genesis/workers/worker_T1_A293/src/tensorneat/algorithm/neat/species.py:295 on a rank-3 operand along axis 0 is not…
- `xor3d_p500` / `new` / `iree`: **error** — evoxc.core.errors.BackendError: stablehlo export: op 'scatter' at /root/tensorneat/.genesis/workers/worker_T1_A293/src/tensorneat/algorithm/neat/species.py:295 on a rank-3 operand along axis 0 is not…
- `xor3d_p1000` / `new` / `iree`: **error** — evoxc.core.errors.BackendError: stablehlo export: op 'scatter' at /root/tensorneat/.genesis/workers/worker_T1_A293/src/tensorneat/algorithm/neat/species.py:295 on a rank-3 operand along axis 0 is not…
- `custom_pagie_p500` / `new` / `iree`: **error** — evoxc.core.errors.BackendError: stablehlo export: op 'scatter' at /root/tensorneat/.genesis/workers/worker_T1_A293/src/tensorneat/algorithm/neat/species.py:295 on a rank-3 operand along axis 0 is not…
- `cartpole_p100` / `new` / `iree`: **error** — evoxc.core.errors.BackendError: stablehlo export: op 'scatter' at /root/tensorneat/.genesis/workers/worker_T1_A293/src/tensorneat/algorithm/neat/species.py:295 on a rank-3 operand along axis 0 is not…

## Environment

| Item | Value |
|---|---|
| Date (UTC) | 2026-11-02T07:29:26+00:00 |
| Python | 3.12.3 |
| Platform | Linux-6.8.0-146-generic-x86_64-with-glibc2.39 |
| CPU model | INTEL(R) XEON(R) PLATINUM 8568Y+ |
| CPU cores (`os.cpu_count`) | 20 |
| Memory | 241611 MiB |
| Load average (1/5/15 min) | 2.44 / 3.65 / 3.34 (1/5/15 min, document build) |
| numpy / jax / evoxc / tensorneat (installed metadata) | numpy 2.5.4 / jax n/a / evoxc 0.1.0 / tensorneat n/a |
| Git commit | `89f6200d6bba` (clean — generated report artifacts are not counted) — `benchmarks/legacy_jax` and `src/` both come from this checkout |
| Thread pinning (`--threads`) | `--threads 0` — runtime defaults (as users run it), nothing exported |
| Thread env (set/unset) | OMP_NUM_THREADS unset · MKL_NUM_THREADS unset · OPENBLAS_NUM_THREADS unset · NUMEXPR_NUM_THREADS unset · XLA_FLAGS unset (CLI process, inherited unchanged by every worker — `--threads 0` exports nothing) |

**Shared box.** These numbers were measured on a machine that was not exclusively ours: other jobs ran on it while the workers timed generations, so the per-repeat spread (the headline's speed-up range and the per-repeat quality-curve rows) is the honest uncertainty measure — a median alone would hide it.

## Protocol

| Parameter | Value |
|---|---|
| Backend(s) of the new implementation | `iree` |
| Warmup generations (untimed) | 1 |
| Timed generations | 2 |
| Repeats (seed + r) | 1 (`seed + r`, seeds 42) |
| Thread pinning | `--threads 0` — runtime defaults (as users run it), nothing exported |
| New-side build | inside the first `Pipeline.step` call (the evoxc build); the generation it consumes is recorded as `generations_consumed_before_timing`, excluded from the timed samples and subtracted from the untimed warmup, so both sides' timing samples cover the same generation indices |
| Compile time | reported separately as `compile_time_s` (old: `jax.jit(pipeline.step).lower(state).compile()`; new: the first `Pipeline.step` build) and never charged to a generation |
| Scheduling | one worker process per run, strictly sequential — the two implementations never share a process |

Metrics: `compile_time_s` (one-off build), `ms_per_generation` (median wall clock per generation
after warmup, including host sync), `gens_per_second`, `rss_final_mb`, and quality
(`final_best_fitness`, `generations_to_target`, best-fitness curve).
The build generation is also consumed from the new side's quality curve, so that curve starts one generation later than the old side's. Compile/build seconds are never charged to a generation.
Definitions, fairness rules and verdict rules live in `benchmarks/CONTEXT.md`.
A thread-bound sensitivity run (`--threads 1`, XOR family, same CLI) is recorded in `benchmarks/THREADS.md` next to this report.

## Case matrix

| Case | Family | Problem | Pop | Timed generations | Fitness target | Old | New | Notes |
|---|---|---|---|---|---|---|---|---|
| xor_p100 | func_fit | XOR | 100 | 2 | -0.1 | not run | yes | — |
| xor_p500 | func_fit | XOR | 500 | 2 | -0.1 | not run | yes | — |
| xor_p1000 | func_fit | XOR | 1000 | 2 | -0.1 | not run | yes | — |
| xor3d_p100 | func_fit | XOR3d | 100 | 2 | -0.05 | not run | yes | — |
| xor3d_p500 | func_fit | XOR3d | 500 | 2 | -0.05 | not run | yes | — |
| xor3d_p1000 | func_fit | XOR3d | 1000 | 2 | -0.05 | not run | yes | — |
| custom_pagie_p500 | func_fit | CustomFuncFit | 500 | 2 | -0.01 | not run | yes | per-impl divergence: the legacy side seeds the global numpy RNG and vmaps a jax-typed func; the new side samples from default_rng(seed) with a numpy func — same distribution, different draws |
| cartpole_p100 | rl | CartPole | 100 | 2 | 180 | n/a | yes | new-only: the legacy library has no equivalent problem |

## Results per case

### xor_p100

| Implementation | Backend | Compile (s) | ms/generation | generations/s | peak RSS (MiB) | Final best | Gens to target | Repeats |
|---|---|---|---|---|---|---|---|---|
| `new` tensorneat (evoxc) — error | `iree` | n/a | n/a | n/a | n/a | n/a | n/a | 1 |

- `new` / `iree`: **error** — evoxc.core.errors.BackendError: stablehlo export: op 'scatter' at /root/tensorneat/.genesis/workers/worker_T1_A293/src/tensorneat/algorithm/neat/species.py:295 on a rank-3 operand along axis 0 is not…

### xor_p500

| Implementation | Backend | Compile (s) | ms/generation | generations/s | peak RSS (MiB) | Final best | Gens to target | Repeats |
|---|---|---|---|---|---|---|---|---|
| `new` tensorneat (evoxc) — error | `iree` | n/a | n/a | n/a | n/a | n/a | n/a | 1 |

- `new` / `iree`: **error** — evoxc.core.errors.BackendError: stablehlo export: op 'scatter' at /root/tensorneat/.genesis/workers/worker_T1_A293/src/tensorneat/algorithm/neat/species.py:295 on a rank-3 operand along axis 0 is not…

### xor_p1000

| Implementation | Backend | Compile (s) | ms/generation | generations/s | peak RSS (MiB) | Final best | Gens to target | Repeats |
|---|---|---|---|---|---|---|---|---|
| `new` tensorneat (evoxc) — error | `iree` | n/a | n/a | n/a | n/a | n/a | n/a | 1 |

- `new` / `iree`: **error** — evoxc.core.errors.BackendError: stablehlo export: op 'scatter' at /root/tensorneat/.genesis/workers/worker_T1_A293/src/tensorneat/algorithm/neat/species.py:295 on a rank-3 operand along axis 0 is not…

### xor3d_p100

| Implementation | Backend | Compile (s) | ms/generation | generations/s | peak RSS (MiB) | Final best | Gens to target | Repeats |
|---|---|---|---|---|---|---|---|---|
| `new` tensorneat (evoxc) — error | `iree` | n/a | n/a | n/a | n/a | n/a | n/a | 1 |

- `new` / `iree`: **error** — evoxc.core.errors.BackendError: stablehlo export: op 'scatter' at /root/tensorneat/.genesis/workers/worker_T1_A293/src/tensorneat/algorithm/neat/species.py:295 on a rank-3 operand along axis 0 is not…

### xor3d_p500

| Implementation | Backend | Compile (s) | ms/generation | generations/s | peak RSS (MiB) | Final best | Gens to target | Repeats |
|---|---|---|---|---|---|---|---|---|
| `new` tensorneat (evoxc) — error | `iree` | n/a | n/a | n/a | n/a | n/a | n/a | 1 |

- `new` / `iree`: **error** — evoxc.core.errors.BackendError: stablehlo export: op 'scatter' at /root/tensorneat/.genesis/workers/worker_T1_A293/src/tensorneat/algorithm/neat/species.py:295 on a rank-3 operand along axis 0 is not…

### xor3d_p1000

| Implementation | Backend | Compile (s) | ms/generation | generations/s | peak RSS (MiB) | Final best | Gens to target | Repeats |
|---|---|---|---|---|---|---|---|---|
| `new` tensorneat (evoxc) — error | `iree` | n/a | n/a | n/a | n/a | n/a | n/a | 1 |

- `new` / `iree`: **error** — evoxc.core.errors.BackendError: stablehlo export: op 'scatter' at /root/tensorneat/.genesis/workers/worker_T1_A293/src/tensorneat/algorithm/neat/species.py:295 on a rank-3 operand along axis 0 is not…

### custom_pagie_p500

| Implementation | Backend | Compile (s) | ms/generation | generations/s | peak RSS (MiB) | Final best | Gens to target | Repeats |
|---|---|---|---|---|---|---|---|---|
| `new` tensorneat (evoxc) — error | `iree` | n/a | n/a | n/a | n/a | n/a | n/a | 1 |

- `new` / `iree`: **error** — evoxc.core.errors.BackendError: stablehlo export: op 'scatter' at /root/tensorneat/.genesis/workers/worker_T1_A293/src/tensorneat/algorithm/neat/species.py:295 on a rank-3 operand along axis 0 is not…

### cartpole_p100

| Implementation | Backend | Compile (s) | ms/generation | generations/s | peak RSS (MiB) | Final best | Gens to target | Repeats |
|---|---|---|---|---|---|---|---|---|
| `new` tensorneat (evoxc) — error | `iree` | n/a | n/a | n/a | n/a | n/a | n/a | 1 |

- `new` / `iree`: **error** — evoxc.core.errors.BackendError: stablehlo export: op 'scatter' at /root/tensorneat/.genesis/workers/worker_T1_A293/src/tensorneat/algorithm/neat/species.py:295 on a rank-3 operand along axis 0 is not…

## Old vs new

no old-vs-new comparison available.

## Verdict

**INCOMPLETE**

- xor_p100/new/iree: error — evoxc.core.errors.BackendError: stablehlo export: op 'scatter' at /root/tensorneat/.genesis/workers/worker_T1_A293/src/tensorneat/algorithm/neat/species.py:295 on a rank-3 operand along axis 0 is not…
- xor_p500/new/iree: error — evoxc.core.errors.BackendError: stablehlo export: op 'scatter' at /root/tensorneat/.genesis/workers/worker_T1_A293/src/tensorneat/algorithm/neat/species.py:295 on a rank-3 operand along axis 0 is not…
- xor_p1000/new/iree: error — evoxc.core.errors.BackendError: stablehlo export: op 'scatter' at /root/tensorneat/.genesis/workers/worker_T1_A293/src/tensorneat/algorithm/neat/species.py:295 on a rank-3 operand along axis 0 is not…
- xor3d_p100/new/iree: error — evoxc.core.errors.BackendError: stablehlo export: op 'scatter' at /root/tensorneat/.genesis/workers/worker_T1_A293/src/tensorneat/algorithm/neat/species.py:295 on a rank-3 operand along axis 0 is not…
- xor3d_p500/new/iree: error — evoxc.core.errors.BackendError: stablehlo export: op 'scatter' at /root/tensorneat/.genesis/workers/worker_T1_A293/src/tensorneat/algorithm/neat/species.py:295 on a rank-3 operand along axis 0 is not…
- xor3d_p1000/new/iree: error — evoxc.core.errors.BackendError: stablehlo export: op 'scatter' at /root/tensorneat/.genesis/workers/worker_T1_A293/src/tensorneat/algorithm/neat/species.py:295 on a rank-3 operand along axis 0 is not…
- custom_pagie_p500/new/iree: error — evoxc.core.errors.BackendError: stablehlo export: op 'scatter' at /root/tensorneat/.genesis/workers/worker_T1_A293/src/tensorneat/algorithm/neat/species.py:295 on a rank-3 operand along axis 0 is not…
- cartpole_p100/new/iree: error — evoxc.core.errors.BackendError: stablehlo export: op 'scatter' at /root/tensorneat/.genesis/workers/worker_T1_A293/src/tensorneat/algorithm/neat/species.py:295 on a rank-3 operand along axis 0 is not…
- no old-vs-new comparison available

Sub-verdicts: timing `n/a`, quality `n/a` — timing compares measured ms/generation, quality passes when `new >= old - max(10% × |old|, 0.001)`.

## Why the new side wins and loses

No old-vs-new comparison is recorded in this document yet, so there is no measured direction to explain. Rerun with `--impl both` (and a functioning new-side backend) to populate this section.

## Quality curves

no fitness curves recorded

## Reference numbers (declared baseline)

| Case | Pop | Declared legacy ms/gen | Measured old ms/gen | Ratio (measured/declared) |
|---|---|---|---|---|
| xor_p100 | 100 | 51.00 | n/a | n/a |
| xor_p500 | 500 | 109.00 | n/a | n/a |
| xor_p1000 | 1000 | 157.00 | n/a | n/a |

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
python benchmarks/run_benchmark.py --impl new --backend iree --cases all --generations 2 --warmup 1 --repeats 1 --json benchmarks/results/iree.json --out benchmarks/iree.md --report-title 'TensorNEAT benchmark — IREE backend attempt (llvm-cpu)' --report-note 'Every case errors in `Pipeline.setup` before generation 0; see `BACKENDS.md` for the failure signature and the minimal reproducer. The measured CPU contract is `RESULTS.md` (numpy reference backend).'

# full matrix, both implementations on the primary backend
python benchmarks/run_benchmark.py --impl both --backend iree --cases all --generations 2 --warmup 1 --repeats 1 --threads 0

# baseline only (frozen JAX)
python benchmarks/run_benchmark.py --impl old --cases all --generations 2 --warmup 1 --repeats 1 --threads 0

# candidate only, one backend at a time
python benchmarks/run_benchmark.py --impl new --backend iree --cases all --generations 2 --warmup 1 --repeats 1 --threads 0

# inspect the matrix and the planned worker commands (no run, no report)
python benchmarks/run_benchmark.py --list
python benchmarks/run_benchmark.py --dry-run
```

Interpreter used: /tmp/genesis/task_d7be275380ce6fc0/iree-venv/bin/python

Cases: xor_p100, xor_p500, xor_p1000, xor3d_p100, xor3d_p500, xor3d_p1000, custom_pagie_p500, cartpole_p100. Timed generations: 2, warmup generations (untimed): 1, repeats: 1 (seeds 42), threads: runtime defaults (`--threads 0`). Backend(s) of the new side: iree; no old-side row is recorded in this document, so every case here is new-side only. `--out` defaults to `benchmarks/RESULTS.md` and `--json` to `benchmarks/results/latest.json` (scratch, git-ignored); the record this report was rendered from (`iree.json`) is committed at `benchmarks/results/iree.json` beside that default, so every table cell is traceable. A thread-bound sensitivity run of the XOR family (`--threads 1`, same CLI) is recorded in `benchmarks/THREADS.md` next to this report.

Generated from `benchmarks/results/iree.json` into `benchmarks/iree.md` at 2026-11-02T07:29:26+00:00 — do not hand-edit.
