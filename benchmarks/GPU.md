# Accelerator measurement: `numpy`, `iree` llvm-cpu and `iree` rocm on one MI300X VF

**Verdict for the record this file renders: both CPU legs measure green across a 3 × 3 matrix — six `ok` rows and three blocked `iree-rocm` rows — `numpy` and `iree-llvm-cpu` run the canonical case matrix end to end and agree bit-for-bit on the integer genes at generation 1, while every `iree-rocm` row is an `error` row whose message is a device-placement refusal in `Pipeline.setup()`, not a compile failure and not a number.**
On the canonical matrix (`xor_p100`, `xor_p500`, `xor3d_p500`; seed 42; 3 warmup + 10 timed generations) the numpy medians are 41.646 / 184.222 / 180.004 ms per generation against 428.811 / 14912.799 / 15032.370 ms for `iree-llvm-cpu`, so the compiled leg is ≈10.3x (428.811 / 41.646) / ≈81.0x (14912.799 / 184.222) / ≈83.5x (15032.370 / 180.004) slower per generation on this box.
Each static shape pays one `iree-compile` of 8.506 / 9.045 / 8.792 s (the `build` column), which is 2.0x the whole 10-generation timed region at pop 100 (8.506 s against 4.288 s) and ≈6 % of it on the two pop-500 rows (9.045 s against 149.128 s, 8.792 s against 150.324 s).
Peak RSS runs the other way: the compiled leg's per-process high-water mark is 150.3 MiB against 161.7 MiB at pop 100 (7.0 % lower), 194.9 against 281.6 MiB on `xor_p500` (30.8 % lower) and 197.2 against 289.6 MiB on `xor3d_p500` (31.9 % lower) — driver RSS, never device memory, and the interpreter's retained intermediates are the larger footprint.
Numerically the two CPU legs agree where the gate is hard: generation 1 has bit-identical `node_keys`/`conn_keys` on all three compiled rows, attributes within 4.8e-7, fitness within 6.0e-8 and zero NaN-mask mismatches.
On the two pop-500 cases a 14-generation lockstep trace then drifts structurally (first float exceedance at generation 5 / 8, first key divergence at 6 / 9) while the population's best fitness stays identical for all 14 traced generations on `xor_p500` and within 6.0e-4 on `xor3d_p500`, which is a measured near-tie selection flip amplified by the ranking, not a kernel defect; at pop 100 all 14 traced generations stay key-identical.
Nothing below states or implies a GPU speed-up, a GPU compile time or a GPU fitness: the `iree-rocm` path never reached generation 1, so it has none of the three.
The refusal is `evoxc.core.errors.DeviceError: input at path [0] is on device Device(kind='cpu', index=0), but the executable runs on device Device(kind='rocm', index=0): no implicit device-to-host or host-to-device transfer happens at the run boundary — move the tensor to the executable's device explicitly via t.to(Device(kind='rocm', index=0))`.
Its cause is on the TensorNEAT side, not in IREE: the pipeline hands its initial tensors to `evoxc.run` on the host while the executable was compiled for the ROCm device, so the initial-tensor device placement that `src/tensorneat` is gaining in parallel is the missing piece.
The one-off generation-graph compile for `rocm` is a *later* step of this protocol and cannot be reached until that placement exists: `build_s` is filled by the first `Pipeline.step()`, and the first step needs a state that `Pipeline.setup()` never returns on this target.
Nothing below invents a number for the blocked rows: the measured block reports `n/a` for every metric an error row does not have, and the full traceback stays in the record.
No accelerator row is part of the numpy-only verdict in `RESULTS.md`; this file is informational evidence, and `## What this does not change` says exactly why.

## Provenance: which tree and which session

Every number in the measured block below belongs to `benchmarks/results/gpu_2026-11-05.json` and to the git commit its fingerprint names — `2d86a89ffd56` — and no sentence here is a statement about HEAD.
That record is the accelerator evidence this directory now commits: one `gpu_measure.py matrix` session, 2026-11-05T11:14:24Z – 2026-11-05T11:30:03Z (15 min 39 s wall clock), 9 rows over the cases `xor_p100,xor_p500,xor3d_p500` × the targets `numpy,iree-llvm-cpu,iree-rocm`, seed 42, 3 warmup and 10 timed generations, `--trace 14`, one repeat, one fresh process per row — 6 `ok` rows and 3 `error` rows, the three `iree-rocm` ones.
The block is rendered from that record, which carries its own command, its session timestamps, its environment fingerprint and the `protocol` block it was measured under (`trajectory_generations: 14` included).
If that file is absent or is still the git-ignored scratch default (`benchmarks/results/gpu.json`), the provenance paragraph inside the block says so and the block describes one session only, never "the GPU performance of TensorNEAT".
The record stores the recording worker's absolute interpreter path in `meta.command`; the portable form of the same argv is step 4 of `## Reproduce`, and that is the command to regenerate it with.
The environment table's `evoxc 0.1.0` row is the installed distribution's metadata — PyPI's newest release — while `pyproject.toml` pins the `v0.1.1` git tag through `[tool.uv.sources]`: the measured tree is the tag, the table row is what a version query reports.
The machinery was smoke-tested before this record was measured: one row per CPU target at small settings (`--warmup 1`, two to three timed generations) to prove the protocol, the rocm `error` row and the cross-check verdict; those smoke records are scratch, are not committed, and their timings are deliberately not quoted here because two generations are not a benchmark.
The CPU compiled-backend record is `BACKENDS.md` (llvm-cpu, its own session), and the numpy-only old-vs-new contract is `RESULTS.md` (its own record); both stay what they are, and neither is re-rendered from this protocol.

## Protocol

`gpu_measure.py` measures exactly one `(case, target)` row per process, and one row per process is not a style choice: `peak_rss_mib` is `resource.getrusage(RUSAGE_SELF).ru_maxrss`, a process-lifetime high-water mark, so two rows in one interpreter would report the higher of the two as both.
`matrix` plans `cases × targets` and spawns one `run` subprocess per row **strictly sequentially**, one fresh interpreter each and the same interpreter as the driver, so no two measurements ever overlap on the box.
Per row: `Pipeline.setup()` is timed into `setup_s`; the first `Pipeline.step()` is timed into `first_step_s` and its build cost, `Pipeline.build_seconds`, is reported as `build_s` — the build is never charged to a generation and that step is excluded from every timing sample; then `--warmup` untimed generations and `--generations` timed ones, with stdout/stderr redirected to `os.devnull` for the whole warmup + timed region.
Each timed generation is `time.perf_counter()` wall clock around one `step()` call, which includes the host materialisation of the result; the row keeps mean, median, min, max and every sample, plus `generations_per_second = 1000 / mean`.
`peak_rss_mib` is sampled immediately after the timed region, then `final_generation`, `final_max_fitness`, `device_info` (the target's `backend`/`device`/`compile_options` plus the IREE HIP device query) and the exact `protocol` block are recorded.
`--trace N` (default `0`, off) adds a lockstep trajectory *after* the timed region and after the `peak_rss_mib` sample: one fresh pair of legs — the accelerated target and a fresh `backend="numpy"` pipeline, each paying its own build — driven generation by generation, so it costs `N × (accelerated generation + numpy generation)` and can never enter a timing sample or the RSS figure.
A failing trace leg is recorded as `trajectory.status == "error"` *inside* an otherwise `ok` row (the row keeps its timings), `--trace 0` records `skipped` with a reason, and the committed matrix ran with `--trace 14`.
`--json` writes the row; its `status` is `ok` or `error`, and any exception during the measurement becomes an `error` row carrying the full traceback plus whatever was already measured, with the last traceback line printed to stderr and exit code 1 — a failed row is recorded, never dropped, and a hard crash that kills the process is turned into an `error` row by the matrix driver (returncode plus stderr tail).
The defaults are the canonical matrix of `benchmarks/cases.py` (`xor_p100`, `xor_p500`, `xor3d_p500`), all three targets, seed 42, three warmup and ten timed generations, one repeat, no trajectory (`--trace 0`): the case table is imported, never duplicated here.
Targets are data, and the raw flags override a named target for hardware this file does not name: `--device KIND` with `--device-index N` (together they build `evoxc.Device(KIND, N)`), `--target-backends a,b`, `--rocm-target ARCH` (appends `--iree-rocm-target=ARCH` to the `iree_compile_args` compile option, overriding the adapter's auto-detected arch) and `--iree-compile-arg FLAG` (repeatable). `device` and `target_backends` must stay aligned: a combination that pairs a host device with a device-only target becomes an `error` row instead of being repaired silently.
Nothing falls back to another backend: if a target cannot run, the exception is the row, and `matrix` exits 0 only when every planned row is `ok` — a blocked device path is loud by construction, which is exactly the state the ROCm rows are in today.

## Cross-check: what is compared and at what tolerance

The cross-check runs *after* the timed region, in the same process, and it never changes a timing: a fresh `backend="numpy"` pipeline with the same case, seed and step count is driven over the same number of generations, so the comparison is one implementation against itself on two backends — no JAX enters the process, and "one implementation per process" is untouched.
Two snapshots are compared, and they are recorded separately because they mean different things.
**Generation 1** is both legs' first evaluated population (the payload and the `(P,)` fitness vector of the first `step()`): integer keys (`node_keys`, `conn_keys`) must be `np.array_equal`, the attribute payloads must have an identical `np.isnan` mask, and the fitness vector an identical NaN *and* `-inf` mask, with every finite entry inside `FLOAT_RTOL = 1e-5` / `FLOAT_ATOL = 1e-6`.
**Final state** is the post-`tell` population after the same number of steps plus the last step's fitness, compared with the same rules and labelled as a search-trajectory observation: both legs draw the same evoxc splitmix64 stream from one seed and one graph, so generation 1 must match, while beyond it a float-reassociation difference between the backends can reorder a near-tied selection and legitimately diverge the integer keys — that is trajectory drift, not necessarily a kernel defect, and the record carries the sentence (`TRAJECTORY_NOTE`) next to the verdict.
Unlike the old-vs-new harness, which compares two different PRNGs (JAX `PRNGKey` vs evoxc splitmix64) and therefore no trajectories at all, both legs here share one random stream, which is why generation 1 is a hard comparison rather than a statistical one.
The tolerances are the ones `test/backends/test_iree_pipeline.py` documents: float32 addition is not associative, so a reduction that IREE reassociates differs from the numpy interpreter's order by about one ulp (~1.19e-7 at unit magnitude) per reordering, the error grows with the reduction depth but stays a small multiple of eps (~6e-8 measured on small populations), and 1e-5 keeps two orders of magnitude of headroom for a box- or build-dependent reordering while still failing on any real divergence; `1e-6` is the absolute floor for attributes that sit near zero, where a pure relative bound is meaningless.
Every row records the max absolute difference over entries finite on both sides, every mask-mismatch count and the first failing field, so a `fail` names the field instead of the row.
The `numpy` target records `{"status": "skipped", "reason": "target is the numpy reference"}`, `--no-cross-check` records `{"status": "skipped", "reason": "--no-cross-check"}`, and a reference leg that cannot run records `status="error"` with its traceback while the measured row keeps its timings — missing numerical evidence stays visible instead of silently green.
What that contract measured on the committed record: generation 1 passes on all three compiled rows — `node_keys` and `conn_keys` bit-identical, attributes within 4.8e-7 (2.38e-7 at pop 100, 4.77e-7 on both pop-500 rows), fitness within 6.0e-8 and no NaN-mask mismatch — so the gate that is hard is the gate that is green.
The final state passes at pop 100 and fails on both pop-500 rows (`fail (final state: node_keys)`), which is the observation the `--trace 14` trajectory answers: over 14 lockstep generations `xor_p100` has no key divergence and no float exceedance at all (attribute difference ≤ 3.6e-7, species count growing 1 → 6), `xor_p500` first exceeds the float tolerance at generation 5 (node-attribute max absolute difference 0.403, fitness 4.4e-3) and first diverges in the keys at generation 6 (31 node-key and 66 conn-key mismatches), and `xor3d_p500` does the same at generation 8 (0.272) and 9 (85 / 250), its species count reaching the 20-slot cap by generation 5.
The order is identical on both pop-500 cases — attributes first, structural genes one generation later — and that is what a selection near-tie looks like: a difference worth a few float32 ulps at generation 1 only becomes visible in the payload once it flips which row wins a rank comparison.
The mechanism, read from the code rather than inferred: generation 1's integer keys are identical and its attribute payload is *drawn*, not computed, so the 2.38e-7–4.77e-7 spread is the backend-specific float32 Box–Muller tail of `evoxc.random.normal` (the numpy kernel's float32 transcendentals against the StableHLO export's, the documented f32 fast-path deviation at the ~1e-6 relative scale) and the 6.0e-8 fitness spread is that payload passed through the forward graph.
It is then amplified by fitness ranking in `src/tensorneat/algorithm/neat/_kernels.py`: `sort_desc_legacy` (`:151`) is a stable ascending argsort gathered through the reversal permutation, i.e. a strict total order in which an exact tie is broken by row index, and `pair_parents` (`:411`) ranks with it at `:454`, draws parents at `:459` and pins elite slot *k* to that block's rank *k* at `:482-491`.
The benchmark cases carry `survival_threshold = 0.01` and `species_size = 20` (`benchmarks/cases.py`), so the survivor count `clamp(floor(0.01 × members), 1, …)` leaves every species under 200 members exactly one breeding parent while the pop-500 tells fill all 20 species slots: that is 20 independent rank-0 decisions per generation, and one ulp-level tie flip changes a species' pinned elites and its single parent one generation before it can change the structural genes — precisely the 5 → 6 and 8 → 9 order the trace shows.
At pop 100 only 1–6 species exist per tell inside the 14 traced generations and nothing flips, which is why the pop-100 row stays key-identical and the pop-500 rows do not.
So generation 1 is the equality gate and nothing else is: every later generation is a search-trajectory observation, and the entity to compare over a long horizon is the fitness view — identical on `xor_p500` for all 14 traced generations, within 6.0e-4 on `xor3d_p500` at generation 14 — never the key arrays.
The trace is part of the record (`trajectory` per row, `protocol.trajectory_generations: 14`), so the table re-renders from it without re-measuring and `--trace N` reproduces it.

<!-- BEGIN MEASURED (gpu_measure.py) -->
## Environment (measured, not assumed)

| Item | Value |
|---|---|
| Date / session (UTC) | 2026-11-05T11:14:24+00:00 – 2026-11-05T11:30:03+00:00 |
| GPU model (gfx version) | AMD Instinct MI300X VF (gfx942) |
| HIP devices (IREE runtime query) | device_id=1 AMD Instinct MI300X VF |
| ROCm version | 7.1.0 |
| Driver version | 6.16.6 |
| IREE compiler / runtime | 3.11.0 / 3.11.0 |
| evoxc | 0.1.0 |
| numpy | 2.5.4 |
| Python | 3.12.3 |
| Platform | Linux-6.8.0-146-generic-x86_64-with-glibc2.39 |
| CPU model / count | INTEL(R) XEON(R) PLATINUM 8568Y+ / 20 logical |
| Memory | 241611 MiB |
| Threads | runtime defaults (nothing exported; inherited environment recorded) |
| Git commit | `2d86a89ffd56` (clean — generated report artifacts are not counted) |
| Load average (1/5/15 min, document build) | 1.81 / 2.31 / 2.44 |

## Measured: numpy vs iree llvm-cpu vs iree rocm

One row per planned `(case, target)`; `n/a` marks a metric the row does not have (an error row has no timings at all). `build` is the one-off generation-graph compile and is excluded from every `ms/gen` sample; `peak RSS` is the process high-water mark of that single row, which is why each row runs in its own process.

| Case | Target | Backend | Status | setup (s) | build (s) | ms/gen median | ms/gen mean | ms/gen min–max | gens/s | peak RSS (MiB) | final max fitness | cross-check |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| xor_p100 | `numpy` | `numpy` | ok | 0.015 | 0.284 | 41.646 | 43.699 | 40.368 – 48.505 | 22.884 | 161.7 | -0.233823 | skipped (target is the numpy reference) |
| xor_p100 | `iree-llvm-cpu` | `iree` | ok | 0.974 | 8.506 | 428.811 | 608.456 | 363.051 – 1026.009 | 1.643 | 150.3 | -0.233823 | pass |
| xor_p100 | `iree-rocm` | `iree` | error | 1.257 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| xor_p500 | `numpy` | `numpy` | ok | 0.016 | 0.343 | 184.222 | 182.066 | 174.357 – 187.159 | 5.492 | 281.6 | -0.224597 | skipped (target is the numpy reference) |
| xor_p500 | `iree-llvm-cpu` | `iree` | ok | 1.021 | 9.045 | 14912.799 | 14925.771 | 14656.627 – 15199.429 | 0.067 | 194.9 | -0.224597 | fail (final state: node_keys) |
| xor_p500 | `iree-rocm` | `iree` | error | 1.199 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| xor3d_p500 | `numpy` | `numpy` | ok | 0.015 | 0.339 | 180.004 | 179.372 | 164.263 – 187.284 | 5.575 | 289.6 | -0.247870 | skipped (target is the numpy reference) |
| xor3d_p500 | `iree-llvm-cpu` | `iree` | ok | 0.970 | 8.792 | 15032.370 | 15022.081 | 14637.265 – 15291.337 | 0.067 | 197.2 | -0.247267 | fail (final state: node_keys) |
| xor3d_p500 | `iree-rocm` | `iree` | error | 1.409 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |

## Numerical cross-check

Each non-numpy row is compared against a fresh `backend="numpy"` run of the same case, seed and step count in the same process. Generation 1 must match (one seed, one graph, one PRNG): integer keys exactly, attributes and fitness to rtol 1e-05 / atol 1e-06 over finite entries with an identical NaN mask. The final state is a search-trajectory observation, not a pass/fail gate on its own: beyond generation 1 a float-reassociation difference can reorder a near-tied selection and legitimately diverge the keys.

| Case | Target | Generation 1 | Keys (gen 1) | Attr max abs diff (gen 1) | Fitness max abs diff | Mask mismatches | Final state | Keys (final) | Attr max abs diff (final) | Fitness max abs diff (final) | Mask mismatches (final) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| xor_p100 | `iree-llvm-cpu` | pass | node_keys exact; conn_keys exact | node_attrs 0.000000238; conn_attrs 0.000000238 | 0.000000060 | 0 | pass | node_keys exact; conn_keys exact | node_attrs 0.000000238; conn_attrs 0.000000238 | 0.000000030 | 0 |
| xor_p100 | `iree-rocm` | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| xor_p500 | `iree-llvm-cpu` | pass | node_keys exact; conn_keys exact | node_attrs 0.000000477; conn_attrs 0.000000294 | 0.000000060 | 0 | fail | node_keys 578 mismatch(es); conn_keys 1858 mismatch(es) | node_attrs 2.506511688; conn_attrs 4.956173897 | 0.090268850 | 3498 |
| xor_p500 | `iree-rocm` | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| xor3d_p500 | `iree-llvm-cpu` | pass | node_keys exact; conn_keys exact | node_attrs 0.000000477; conn_attrs 0.000000298 | 0.000000060 | 0 | fail | node_keys 850 mismatch(es); conn_keys 3155 mismatch(es) | node_attrs 3.936978817; conn_attrs 3.674550533 | 0.060001105 | 5844 |
| xor3d_p500 | `iree-rocm` | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |

## Per-generation trajectory

Every traced row was re-run after the timed region, with the peak-RSS sample already taken, as a lockstep pair: a fresh accelerated pipeline and a fresh `backend="numpy"` pipeline of the same case and seed, each paying its own build, compared per generation with the cross-check rules above. The trace touches neither the timings nor the recorded RSS.

**xor_p100 / `iree-llvm-cpu`** — 14 generation(s) compared in lockstep; first node-key divergence: none within 14 generations; first float exceedance: none within 14 generations; wall time 17.667 s.

*Both legs share one PRNG from one seed and one graph, so a first key divergence beyond generation 1 is a selection near-tie flip caused by a sub-ulp float difference between the backends, not a kernel defect.*

| Generation | Node-key mismatches | Conn-key mismatches | Node attr max abs diff | Conn attr max abs diff | Attr NaN-mask mismatches | Fitness max abs diff | Fitness mask mismatches | Max fitness (numpy / accelerated) | Species (numpy / accelerated) |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 0 | 0 | 0.000000238 | 0.000000238 | 0 | 0.000000060 | 0 | -0.253215 / -0.253215 | 1 / 1 |
| 2 | 0 | 0 | 0.000000119 | 0.000000119 | 0 | 0.000000060 | 0 | -0.250795 / -0.250795 | 1 / 1 |
| 3 | 0 | 0 | 0.000000067 | 0.000000119 | 0 | 0.000000060 | 0 | -0.250101 / -0.250101 | 2 / 2 |
| 4 | 0 | 0 | 0.000000209 | 0.000000134 | 0 | 0.000000030 | 0 | -0.250050 / -0.250050 | 2 / 2 |
| 5 | 0 | 0 | 0.000000119 | 0.000000119 | 0 | 0.000000060 | 0 | -0.249919 / -0.249919 | 2 / 2 |
| 6 | 0 | 0 | 0.000000238 | 0.000000119 | 0 | 0.000000060 | 0 | -0.249629 / -0.249629 | 2 / 2 |
| 7 | 0 | 0 | 0.000000119 | 0.000000179 | 0 | 0.000000060 | 0 | -0.248715 / -0.248715 | 2 / 2 |
| 8 | 0 | 0 | 0.000000119 | 0.000000238 | 0 | 0.000000060 | 0 | -0.247675 / -0.247675 | 2 / 2 |
| 9 | 0 | 0 | 0.000000119 | 0.000000119 | 0 | 0.000000060 | 0 | -0.246592 / -0.246592 | 2 / 2 |
| 10 | 0 | 0 | 0.000000119 | 0.000000179 | 0 | 0.000000060 | 0 | -0.245527 / -0.245527 | 3 / 3 |
| 11 | 0 | 0 | 0.000000119 | 0.000000238 | 0 | 0.000000060 | 0 | -0.243673 / -0.243673 | 6 / 6 |
| 12 | 0 | 0 | 0.000000179 | 0.000000358 | 0 | 0.000000060 | 0 | -0.241268 / -0.241268 | 6 / 6 |
| 13 | 0 | 0 | 0.000000119 | 0.000000179 | 0 | 0.000000060 | 0 | -0.237129 / -0.237129 | 6 / 6 |
| 14 | 0 | 0 | 0.000000119 | 0.000000238 | 0 | 0.000000030 | 0 | -0.233823 / -0.233823 | 6 / 6 |

**xor_p500 / `iree-llvm-cpu`** — 14 generation(s) compared in lockstep; first node-key divergence: 6; first float exceedance: 5; wall time 224.446 s.

*Both legs share one PRNG from one seed and one graph, so a first key divergence beyond generation 1 is a selection near-tie flip caused by a sub-ulp float difference between the backends, not a kernel defect.*

| Generation | Node-key mismatches | Conn-key mismatches | Node attr max abs diff | Conn attr max abs diff | Attr NaN-mask mismatches | Fitness max abs diff | Fitness mask mismatches | Max fitness (numpy / accelerated) | Species (numpy / accelerated) |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 0 | 0 | 0.000000477 | 0.000000294 | 0 | 0.000000060 | 0 | -0.250251 / -0.250251 | 1 / 1 |
| 2 | 0 | 0 | 0.000000119 | 0.000000238 | 0 | 0.000000060 | 0 | -0.250017 / -0.250017 | 6 / 6 |
| 3 | 0 | 0 | 0.000000238 | 0.000000358 | 0 | 0.000000089 | 0 | -0.250007 / -0.250007 | 16 / 16 |
| 4 | 0 | 0 | 0.000000209 | 0.000000160 | 0 | 0.000000060 | 0 | -0.250001 / -0.250001 | 20 / 20 |
| 5 | 0 | 0 | 0.402902961 | 0.003084093 | 0 | 0.004423261 | 0 | -0.249823 / -0.249823 | 20 / 20 |
| 6 | 31 | 66 | 0.508377314 | 1.976041079 | 7 | 0.005131155 | 0 | -0.248735 / -0.248735 | 20 / 20 |
| 7 | 159 | 439 | 1.245799303 | 1.014402628 | 444 | 0.014755279 | 1 | -0.247461 / -0.247461 | 20 / 20 |
| 8 | 237 | 845 | 1.494967699 | 4.651563644 | 843 | 0.082444668 | 4 | -0.243918 / -0.243918 | 20 / 20 |
| 9 | 425 | 1436 | 1.595991731 | 4.251160622 | 1282 | 0.100294918 | 10 | -0.241320 / -0.241320 | 20 / 20 |
| 10 | 411 | 1302 | 1.772526622 | 4.653420925 | 1056 | 0.033620209 | 8 | -0.240657 / -0.240657 | 20 / 20 |
| 11 | 547 | 1944 | 2.092477322 | 4.572396278 | 1585 | 0.096894860 | 13 | -0.237867 / -0.237867 | 20 / 20 |
| 12 | 558 | 1945 | 2.593549967 | 3.371931553 | 1761 | 0.107947081 | 15 | -0.233631 / -0.233631 | 20 / 20 |
| 13 | 533 | 1782 | 4.043320179 | 4.658003807 | 1340 | 0.130599111 | 14 | -0.228627 / -0.228627 | 20 / 20 |
| 14 | 486 | 1850 | 1.724698424 | 4.736943245 | 1340 | 0.090268850 | 9 | -0.224597 / -0.224597 | 20 / 20 |

**xor3d_p500 / `iree-llvm-cpu`** — 14 generation(s) compared in lockstep; first node-key divergence: 9; first float exceedance: 8; wall time 219.993 s.

*Both legs share one PRNG from one seed and one graph, so a first key divergence beyond generation 1 is a selection near-tie flip caused by a sub-ulp float difference between the backends, not a kernel defect.*

| Generation | Node-key mismatches | Conn-key mismatches | Node attr max abs diff | Conn attr max abs diff | Attr NaN-mask mismatches | Fitness max abs diff | Fitness mask mismatches | Max fitness (numpy / accelerated) | Species (numpy / accelerated) |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 0 | 0 | 0.000000477 | 0.000000298 | 0 | 0.000000060 | 0 | -0.251171 / -0.251171 | 1 / 1 |
| 2 | 0 | 0 | 0.000000238 | 0.000000358 | 0 | 0.000000060 | 0 | -0.250194 / -0.250194 | 1 / 1 |
| 3 | 0 | 0 | 0.000000238 | 0.000000358 | 0 | 0.000000060 | 0 | -0.250011 / -0.250011 | 3 / 3 |
| 4 | 0 | 0 | 0.000000209 | 0.000000343 | 0 | 0.000000089 | 0 | -0.250004 / -0.250004 | 3 / 3 |
| 5 | 0 | 0 | 0.000000238 | 0.000000313 | 0 | 0.000000060 | 0 | -0.250001 / -0.250001 | 20 / 20 |
| 6 | 0 | 0 | 0.000000238 | 0.000000343 | 0 | 0.000000060 | 0 | -0.250000 / -0.250000 | 20 / 20 |
| 7 | 0 | 0 | 0.000000253 | 0.000000328 | 0 | 0.000000030 | 0 | -0.250000 / -0.250000 | 20 / 20 |
| 8 | 0 | 0 | 0.272031397 | 0.057659566 | 0 | 0.000539750 | 0 | -0.249999 / -0.249999 | 20 / 20 |
| 9 | 85 | 250 | 2.416074753 | 1.378281713 | 103 | 0.002465039 | 0 | -0.249980 / -0.249980 | 20 / 20 |
| 10 | 450 | 1679 | 2.131389618 | 3.445238590 | 1607 | 0.128810614 | 13 | -0.249943 / -0.249943 | 20 / 20 |
| 11 | 702 | 2619 | 3.213420391 | 3.445238590 | 2385 | 0.083280176 | 22 | -0.249839 / -0.249839 | 20 / 20 |
| 12 | 788 | 2957 | 3.996948242 | 3.178690195 | 2214 | 0.112219185 | 21 | -0.249462 / -0.249462 | 20 / 20 |
| 13 | 742 | 2850 | 3.996948242 | 4.114693642 | 2168 | 0.098949313 | 29 | -0.249007 / -0.249007 | 20 / 20 |
| 14 | 879 | 3273 | 4.208442688 | 3.649823666 | 2422 | 0.060001105 | 18 | -0.247870 / -0.247267 | 20 / 20 |

## Blocked rows

The full traceback of every row below stays in the record; the cell prints its last non-empty line.

| Case | Target | Error (last non-empty line) |
|---|---|---|
| xor_p100 | `iree-rocm` | evoxc.core.errors.DeviceError: input at path [0] is on device Device(kind='cpu', index=0), but the executable runs on device Device(kind='rocm', index=0): no implicit device-to-host or host-to-device transfer happens at the run boundary — move the tensor to the executable's device explicitly via t.to(Device(kind='rocm', index=0)) |
| xor_p500 | `iree-rocm` | evoxc.core.errors.DeviceError: input at path [0] is on device Device(kind='cpu', index=0), but the executable runs on device Device(kind='rocm', index=0): no implicit device-to-host or host-to-device transfer happens at the run boundary — move the tensor to the executable's device explicitly via t.to(Device(kind='rocm', index=0)) |
| xor3d_p500 | `iree-rocm` | evoxc.core.errors.DeviceError: input at path [0] is on device Device(kind='cpu', index=0), but the executable runs on device Device(kind='rocm', index=0): no implicit device-to-host or host-to-device transfer happens at the run boundary — move the tensor to the executable's device explicitly via t.to(Device(kind='rocm', index=0)) |

## Provenance of this block

This block is rendered from `benchmarks/results/gpu_2026-11-05.json`: 9 record(s), 6 ok and 3 error, measured in the session 2026-11-05T11:14:24+00:00 – 2026-11-05T11:30:03+00:00 (UTC) at git `2d86a89ffd56` (clean — generated report artifacts are not counted).
`benchmarks/results/gpu_2026-11-05.json` is committed evidence (it is in the re-include list of `benchmarks/results/.gitignore`), so every cell above is traceable to the committed record.
Nothing here is a statement about HEAD: every number belongs to that record and to the tree its fingerprint names.
<!-- END MEASURED (gpu_measure.py) -->

## Limitations

No legacy-GPU comparison exists: the frozen JAX baseline in `benchmarks/legacy_jax` is CPU-only by protocol, so an accelerator number here is informational and is never part of the numpy-only old-vs-new verdict in `RESULTS.md`; there is no "old side" for a GPU.
One device: a single `AMD Instinct MI300X VF` at one device index (`iree.runtime.get_driver("hip").query_available_devices()` reports one entry), so nothing here says how the numbers scale to a second GPU, and no multi-device or multi-stream path is exercised.
`xla` and `tvm` are not attempted: `xla` needs a PJRT C API plugin `.so` that nothing provides on this box, and `tvm` is untested in this repository.
Transfer cost is inside `ms/generation` and not separated from it: the host materialisation of each generation's result happens inside the timed `step()` call, and the HIP host/device traffic is part of that wall clock; this protocol does not report a transfer-only number, and PCIe versus unified-memory behaviour is not instrumented.
`peak RSS` is per process and is the driver's resident memory, not the device's; no device-memory figure is recorded, so a row says nothing about VRAM occupancy.
No per-generation dispatch or op count is instrumented here — `PERF_NUMPY.md` is where executed-op counts live for the CPU interpreter — so an accelerator row gives wall clock and nothing about what the device was asked to do.
Timing is wall clock with a warmup, one repeat, one seed: repeat spread is not measured by this protocol, and a single run of a row is not a distribution.
The cross-check's later-generation divergence makes a long-horizon accelerator-vs-numpy *population* comparison a trajectory comparison, not an equality test: the committed trace's first key divergence lands at generation 6 on `xor_p500` and 9 on `xor3d_p500`, so the fitness (and species-count) view is what to compare across many generations, and the key arrays are not.
`--trace 14` is why the committed matrix took 15 min 39 s on this box: the three trajectory legs alone cost 17.667 s, 224.446 s and 219.993 s (≈7.7 min), each one re-paying a full build for both of its legs on top of the 14 generations.

## Reproduce

```bash
# 0. environment: the compiled-backend extra, plus a working ROCm runtime for the rocm rows
uv sync --extra iree
/opt/rocm/.info/version && rocm-smi --showproductname --showdriverversion   # 7.1.0 / MI300X VF / driver 6.16.6 here

# 1. numpy-free planning: targets, passthrough flags, the canonical case matrix
python3 benchmarks/gpu_measure.py --list

# 2. one row, one fresh process (this is what `matrix` spawns per row)
uv run --extra iree python benchmarks/gpu_measure.py run \
    --case xor_p100 --target iree-rocm --json $TMPDIR/row.json

# 3. one compiled row with the lockstep trajectory the committed record carries
uv run --extra iree python benchmarks/gpu_measure.py run \
    --case xor_p500 --target iree-llvm-cpu \
    --warmup 3 --generations 10 --trace 14 --seed 42 --json $TMPDIR/trace_row.json

# 4. the full matrix: cases x targets, strictly sequential, one process per row
uv run --extra iree python benchmarks/gpu_measure.py matrix \
    --cases xor_p100,xor_p500,xor3d_p500 --targets numpy,iree-llvm-cpu,iree-rocm \
    --warmup 3 --generations 10 --trace 14 --seed 42 \
    --json benchmarks/results/gpu_2026-11-05.json --out benchmarks/GPU.md

# 5. re-render the measured block from the existing record: no measurement, no JSON rewrite
uv run --extra iree python benchmarks/gpu_measure.py render \
    --json benchmarks/results/gpu_2026-11-05.json --out benchmarks/GPU.md
```

Step 4 is the exact argv the record stores, with the recording worker's absolute interpreter path replaced by the portable `uv run --extra iree python`.
It is a ~15.6-minute job on this box (15 min 39 s for the committed record, of which the three trajectory legs are ≈7.7 min), so run it in the background rather than under a short foreground timeout.
`matrix` exits 0 only when every planned row is `ok`, and 1 as soon as any row is `error` (it prints a `BLOCKED: …` line per blocked row) — with the three ROCm rows blocked today, step 4 therefore exits 1 by design while still writing the record and refreshing the block, which is the intended loud failure and not a broken run.
`render` and `matrix --out` replace only the interior of this file's two markers (the pair above, exactly one of each in the file) and exit 2 with a named error on a missing marker, an unreadable record, a wrong schema or a missing `records` list.
A single row is cheap on the CPU legs and pays the compile on the compiled ones: budget an `iree-*` row at the 8.5–9.0 s `iree-compile` of its shape plus its timed region, and a traced row at that plus `N × (accelerated + numpy generation)`.
The IREE runtime can print `nanobind: leaked …` lines to stderr when the interpreter exits; that is a binding teardown message emitted *after* the row is written, never a failure and never an `error` row.
The Python-free paths are `--list`, `--help` and nothing else: `run` and `matrix` need `numpy` and `tensorneat`, and the `iree-*` targets need the `iree` extra.

## What this does not change

`RESULTS.md` stays the generated numpy-only old-vs-new record, with its own verdict computed from `results/authoritative.json`; no accelerator row is added to it, and no number in it is touched by this protocol.
`BACKENDS.md` stays the CPU compiled-backend record of its own session, and `PERF_NUMPY.md` stays the numpy-interpreter performance record.
Accelerator rows are never wired into the harness verdict, and that is deliberate: `protocol.comparisons` pairs every old-side record with every new-side record, so an accelerator row in `results/` would enter the numpy-only verdict as if it were a numpy measurement and could flip a timing verdict that the CPU contract owns.
Keeping the accelerator protocol standalone (`benchmarks/gpu_measure.py`, its own `tensorneat-gpu/1` schema, its own record and report) keeps that contract intact: the harness verdict compares what it declares it compares, and an accelerated row is evidence about a different question.
