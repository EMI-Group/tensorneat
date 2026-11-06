# TensorNEAT accelerator measurement — numpy reference vs iree rocm on one MI300X VF

**Verdict for the record this file renders: the ROCm path is green end to end — twelve planned rows, twelve `ok`, zero blocked — so every one of the six cases now carries a real `iree` rocm compile time, a real per-generation timing and a real numerical cross-check against the `numpy` reference.**
`Pipeline(backend="iree", device="rocm", compile_options={"target_backends": ["rocm"]})` compiles the generation graph for `gfx942` and runs it on the device, so `build_s`, `ms_per_generation` and `cross_check` are filled on every rocm row instead of being `n/a`.
The device is faster per generation than the numpy reference on all six cases: 30.575 against 48.502 ms/gen on `xor_p100` (1.6x), 49.534 against 171.239 on `xor_p500` (3.5x), 66.134 against 313.623 on `xor_p1000` (4.7x), 49.503 against 198.987 on `xor3d_p500` (4.0x), 62.981 against 1039.650 on `custom_pagie_p500` (17x) and 179.644 against 283.515 on `cartpole_p100` (1.6x).
The headline caveat is that the per-generation win is bought with a one-off compile: each static shape pays an `iree-compile` of 9.037–15.091 s (15.091 s on `custom_pagie_p500`), which is 3.1x–14.8x the whole 20-generation timed region of that same row on the device, so amortising it at these measured rates takes 15 generations on `custom_pagie_p500`, 39 on `xor_p1000`, 68 on `xor3d_p500`, 80 on `xor_p500`, 109 on `cartpole_p100` and 504 on `xor_p100`.
Numerically, generation 1 is the equality gate and it is green on all six rocm rows: `node_keys` and `conn_keys` bit-identical, attributes within 4.8e-7, fitness within 2.4e-7 and zero NaN-mask mismatches.
Beyond generation 1 the two legs are a search trajectory, and the 14-generation lockstep trace shows the same order wherever a row drifts: attributes first, structural genes one to three generations later, while the population's best fitness stays within a few float32 ulps (and within 6.0e-4 on `xor3d_p500`).
Nothing below quotes a number that is not in `benchmarks/results/gpu_rocm_2026-11-05.json`; the speed-up column and the break-even generation counts are arithmetic on that record's two medians and its `build_s`, and nothing here is a statement about HEAD.

## Headline: what the device buys per generation

| Case | numpy ms/gen (median) | rocm ms/gen (median) | Speed-up (numpy ÷ rocm) | rocm `build_s` (s) | peak RSS numpy / rocm (MiB) | final max fitness numpy / rocm | Cross-check |
|---|---|---|---|---|---|---|---|
| `xor_p100` | 48.502 | 30.575 | 1.6x | 9.037 | 162.1 / 1903.5 | -0.176966 / -0.176966 | pass (gen 1 and final state) |
| `xor_p500` | 171.239 | 49.534 | 3.5x | 9.751 | 281.9 / 1931.9 | -0.147786 / -0.147786 | gen 1 pass; final-state drift (node_keys) |
| `xor_p1000` | 313.623 | 66.134 | 4.7x | 9.685 | 425.8 / 1954.5 | -0.053339 / -0.052824 | gen 1 pass; final-state drift (node_keys) |
| `xor3d_p500` | 198.987 | 49.503 | 4.0x | 10.166 | 289.9 / 1932.8 | -0.223466 / -0.207055 | gen 1 pass; final-state drift (node_keys) |
| `custom_pagie_p500` | 1039.650 | 62.981 | 17x | 15.091 | 1310.0 / 1978.7 | -0.032613 / -0.032613 | gen 1 pass; final-state drift (node_keys) |
| `cartpole_p100` | 283.515 | 179.644 | 1.6x | 11.316 | 160.4 / 2064.3 | 200.000000 / 200.000000 | pass (gen 1 and final state) |

Every cell except the speed-up column is a column of the generated block below; the speed-up column is the numpy median divided by the rocm median of the same row and is hand-written here because the generated table has no ratio column.

The biggest win is `custom_pagie_p500` at 17x (1039.650 against 62.981 ms/gen).
That is the expected shape of the result rather than a surprise: the func-fit forward is the sample-invariant schedule evaluated in tiles, which makes it the widest per-generation device workload in the matrix, and the numpy interpreter's per-op overhead scales with exactly that work — this row is also the slowest cell of the whole matrix on the CPU leg (1039.650 ms/gen against 313.623 on the next-slowest, `xor_p1000`).

No case loses or ties on this session: the two narrowest margins are `cartpole_p100` at 1.6x (283.515 against 179.644 ms/gen) and `xor_p100` at 1.6x (48.502 against 30.575 ms/gen), which are the two smallest per-generation tensors in the matrix.
A generation's cost at that size is dominated by the fixed per-generation overhead — dispatch, host materialisation and the run-boundary device traffic, all inside the timed `step()` — which the device cannot compress the way it compresses a wide kernel; `cartpole_p100` additionally drives a per-step rollout loop over its horizon, so its per-generation launch count is the largest of the shipped problems (the op-count pass records 31 604 executed ops per generation for this case on the CPU interpreter, `PERF_NUMPY.md`), and that fixed cost is why its margin is narrow even though each individual simulated step is cheap.

Memory runs the other way, and it is driver RSS, never device memory: the rocm leg sits at 1903.5–2064.3 MiB on every row and barely tracks the workload, while the numpy leg grows with it (162.1 MiB at `xor_p100` to 1310.0 MiB at `custom_pagie_p500`) and is lower than the rocm leg on all six rows.
The rocm figure is a single process high-water mark dominated by the HIP runtime and the loaded executable, which is why it is flat from pop 100 to pop 1000 instead of scaling like the numpy interpreter's retained intermediates.

## Provenance: which tree and which session

Every number in this file belongs to `benchmarks/results/gpu_rocm_2026-11-05.json` and to the git commit its fingerprint names, `a5973c89bebf39aa2363a2abfae75d83070b919c` — which is also this branch's HEAD at recording time, confirmed from the record rather than assumed — and no sentence here is a statement about HEAD.
That record is one `gpu_measure.py matrix` session: 2026-11-05T11:54:45Z – 2026-11-05T11:59:49Z (5 min 4 s wall clock), 12 rows over the cases `xor_p100,xor_p500,xor_p1000,xor3d_p500,custom_pagie_p500,cartpole_p100` × the targets `numpy,iree-rocm`, seed 42, 3 warmup and 20 timed generations, `--trace 14`, one repeat per row, one fresh process per row — 12 `ok` and 0 `error`.
The environment it ran on is the block below: IREE 3.13.0 compiler and runtime, ROCm 7.1.0, driver 6.16.6, `AMD Instinct MI300X VF` (`gfx942`) as the single HIP device, and a clean tree.
The block is rendered from that record, which carries its own command, session timestamps, environment fingerprint and `protocol` block (`trajectory_generations: 14` included); re-rendering it is `gpu_measure.py render` against the same file, and it is the only thing that rewrites the interior of the two markers.
The record stores the recording worker's absolute interpreter path in `meta.command`; the portable form of the same argv is step 4 of `## Reproduce`, and that is the command to regenerate it with.
The `evoxc 0.1.0` row in the environment table is the installed distribution's metadata — PyPI's newest release — while `pyproject.toml` pins the `v0.1.1` git tag through `[tool.uv.sources]`: the measured tree is the tag, the table row is what a version query reports.
A GPU optimizer pass landing in `src/tensorneat` in parallel may improve these numbers afterwards: nothing in this record observes that pass, and its own tree is the one the fingerprint names.

**Repeat agreement (a scratch check, not committed evidence).** The identical argv was run a second time into `$TMPDIR/gpu_rocm_repeat.json` (no `--out`, so it touched neither this report nor the committed record): 2026-11-05T12:00:22Z – 2026-11-05T12:05:27Z (5 min 5 s), same tree `a5973c89bebf`, 12/12 rows `ok`.
Across the two sessions no per-row `ms_per_generation.median` moved by more than 6.8 % — 10 of the 12 rows stayed within ±5 %, and the two exceptions are the numpy legs of `xor_p100` (+6.5 %) and `xor_p500` (+6.8 %); the rocm legs moved by +0.1 % to +4.2 % on five rows and -0.4 % to -4.4 % on the sixth — and every one of the 12 `final_max_fitness` values reproduced bit-for-bit on both legs, so what moved between the sessions is timing, not search.
The repeat is a scratch record: it is git-ignored, it is not committed evidence, and it describes the same session conditions rather than a second experiment.

**Measurement conditions, stated rather than implied.** The box was shared during both runs: the primary record's document-build load average is 7.76 / 4.99 / 2.81 (1/5/15 min) and the repeat's is 3.43 / 4.59 / 3.36, because another job on the same host was running while this matrix measured.
That is exactly why the repeat is reported above: the two sessions are the honest uncertainty measure for these rows, and a single median is not a distribution.

## Protocol

`gpu_measure.py` measures exactly one `(case, target)` row per process, and one row per process is not a style choice: `peak_rss_mib` is `resource.getrusage(RUSAGE_SELF).ru_maxrss`, a process-lifetime high-water mark, so two rows in one interpreter would report the higher of the two as both.
`matrix` plans `cases × targets` and spawns one `run` subprocess per row **strictly sequentially**, one fresh interpreter each and the same interpreter as the driver, so no two measurements ever overlap on the box.
Per row: `Pipeline.setup()` is timed into `setup_s` (0.015–0.017 s on the numpy leg, 1.234–1.638 s on the rocm leg); the first `Pipeline.step()` is timed into `first_step_s` and its build cost, `Pipeline.build_seconds`, is reported as `build_s` (0.282–0.863 s on the numpy leg, 9.037–15.091 s on the rocm leg, which is where `iree-compile` lands) — the build is never charged to a generation and that step is excluded from every timing sample; then `--warmup` untimed generations and `--generations` timed ones, with stdout/stderr redirected to `os.devnull` for the whole warmup + timed region.
Each timed generation is `time.perf_counter()` wall clock around one `step()` call, which includes the host materialisation of the result; the row keeps mean, median, min, max and every sample (20 per row here), plus `generations_per_second = 1000 / mean`.
`peak_rss_mib` is sampled immediately after the timed region, then `final_generation`, `final_max_fitness`, `device_info` (the target's `backend`/`device`/`compile_options` — `{"target_backends": ["rocm"]}` on the rocm rows — plus the IREE HIP device query) and the exact `protocol` block are recorded.
`--trace N` (default `0`, off) adds a lockstep trajectory *after* the timed region and after the `peak_rss_mib` sample: one fresh pair of legs — the accelerated target and a fresh `backend="numpy"` pipeline, each paying its own build — driven generation by generation, so it costs `N × (accelerated generation + numpy generation)` and can never enter a timing sample or the RSS figure; the six traced rows cost 108.2 s of wall clock in total (11.473 s on `xor_p100` up to 32.707 s on `custom_pagie_p500`).
A failing trace leg is recorded as `trajectory.status == "error"` *inside* an otherwise `ok` row (the row keeps its timings), `--trace 0` records `skipped` with a reason, and this matrix ran with `--trace 14`, so all six rocm rows carry 14 traced generations.
`--json` writes the row; its `status` is `ok` or `error`, and any exception during the measurement becomes an `error` row carrying the full traceback plus whatever was already measured, with the last traceback line printed to stderr and exit code 1 — a failed row is recorded, never dropped, and a hard crash that kills the process is turned into an `error` row by the matrix driver (returncode plus stderr tail).
The defaults are the canonical matrix of `benchmarks/cases.py` (`xor_p100`, `xor_p500`, `xor3d_p500`), all three targets, seed 42, three warmup and ten timed generations, one repeat, no trajectory (`--trace 0`): the case table is imported, never duplicated here.
Targets are data, and the raw flags override a named target for hardware this file does not name: `--device KIND` with `--device-index N` (together they build `evoxc.Device(KIND, N)`), `--target-backends a,b`, `--rocm-target ARCH` (appends `--iree-rocm-target=ARCH` to the `iree_compile_args` compile option) and `--iree-compile-arg FLAG` (repeatable). `device` and `target_backends` must stay aligned: a combination that pairs a host device with a device-only target becomes an `error` row instead of being repaired silently.
Nothing falls back to another backend: if a target cannot run, the exception is the row, and `matrix` exits 0 only when every planned row is `ok` — a blocked device path is loud by construction, and this matrix exited 0 because no row was blocked.

## Cross-check: what is compared and at what tolerance

The cross-check runs *after* the timed region, in the same process, and it never changes a timing: a fresh `backend="numpy"` pipeline with the same case, seed and step count is driven over the same number of generations, so the comparison is one implementation against itself on two backends — no JAX enters the process, and "one implementation per process" is untouched.
Two snapshots are compared, and they are recorded separately because they mean different things.
**Generation 1** is both legs' first evaluated population (the payload and the `(P,)` fitness vector of the first `step()`): integer keys (`node_keys`, `conn_keys`) must be `np.array_equal`, the attribute payloads must have an identical `np.isnan` mask, and the fitness vector an identical NaN *and* `-inf` mask, with every finite entry inside `FLOAT_RTOL = 1e-5` / `FLOAT_ATOL = 1e-6`.
**Final state** is the post-`tell` population after the same number of steps plus the last step's fitness, compared with the same rules and labelled as a search-trajectory observation: both legs draw the same evoxc splitmix64 stream from one seed and one graph, so generation 1 must match, while beyond it a float-reassociation difference between the backends can reorder a near-tied decision and legitimately diverge the integer keys — that is trajectory drift, not necessarily a kernel defect, and the record carries the sentence (`TRAJECTORY_NOTE`) next to the verdict.
Unlike the old-vs-new harness, which compares two different PRNGs (JAX `PRNGKey` vs evoxc splitmix64) and therefore no trajectories at all, both legs here share one random stream, which is why generation 1 is a hard comparison rather than a statistical one.
The tolerances are the ones `test/backends/test_iree_pipeline.py` documents: float32 addition is not associative, so a reduction that a compiled backend reassociates differs from the numpy interpreter's order by about one ulp (~1.19e-7 at unit magnitude) per reordering, the error grows with the reduction depth but stays a small multiple of eps, and 1e-5 keeps two orders of magnitude of headroom for a box- or build-dependent reordering while still failing on any real divergence; `1e-6` is the absolute floor for attributes that sit near zero, where a pure relative bound is meaningless.
Every row records the max absolute difference over entries finite on both sides, every mask-mismatch count and the first failing field, so a `fail` names the field instead of the row.
The `numpy` target records `{"status": "skipped", "reason": "target is the numpy reference"}`, `--no-cross-check` records `{"status": "skipped", "reason": "--no-cross-check"}`, and a reference leg that cannot run records `status="error"` with its traceback while the measured row keeps its timings — missing numerical evidence stays visible instead of silently green.

What that contract measured on the six rocm rows of this record:

| Case (`iree` rocm) | `node_keys` / `conn_keys` mismatches (gen 1) | node attrs max abs diff | conn attrs max abs diff | fitness max abs diff | NaN-mask mismatches | Trace: first float exceedance / first key divergence | Trace: best fitness at generation 14 (numpy / rocm) |
|---|---|---|---|---|---|---|---|
| `xor_p100` | 0 / 0 | 2.384e-7 | 2.384e-7 | 5.960e-8 | 0 | none within 14 / none within 14 | -0.233823 / -0.233823 |
| `xor_p500` | 0 / 0 | 4.768e-7 | 2.938e-7 | 5.960e-8 | 0 | 5 / 6 | -0.224597 / -0.224597 |
| `xor_p1000` | 0 / 0 | 4.768e-7 | 2.980e-7 | 8.941e-8 | 0 | 8 / 9 | -0.187691 / -0.187691 |
| `xor3d_p500` | 0 / 0 | 4.768e-7 | 2.980e-7 | 5.960e-8 | 0 | 8 / 9 | -0.247870 / -0.247267 |
| `custom_pagie_p500` | 0 / 0 | 4.768e-7 | 2.938e-7 | 2.384e-7 | 0 | 7 / 10 | -0.042274 / -0.042274 |
| `cartpole_p100` | 0 / 0 | 4.768e-7 | 2.938e-7 | 0.0 | 0 | none within 14 / none within 14 | 200.000000 / 200.000000 |

So the gate that is hard is the gate that is green: generation 1 is bit-identical on the integer genes on all six rows, its largest attribute difference is 4.8e-7 (two float32 ulps at unit magnitude, four orders of magnitude inside the 1e-5 tolerance) and its largest fitness difference is 2.4e-7.
The final state passes on `xor_p100` and `cartpole_p100` (`pass (gen 1 and final state)`) and drifts in the integer keys on the other four (`gen 1 pass; final-state drift (node_keys)`), which is the observation the `--trace 14` trajectory answers.
Over 14 lockstep generations `xor_p100` and `cartpole_p100` never diverge at all — no key mismatch, no float exceedance (attribute difference ≤ 4.8e-7; `cartpole_p100`'s fitness difference is exactly 0.0 on every generation because its horizon-capped return saturates at 200.0, and its species count stays 1 vs 1).
`xor_p500` first exceeds the float tolerance at generation 5 and first diverges in the keys at generation 6, ending generation 14 with 486 node-key and 1850 conn-key mismatches; `xor_p1000` does the same at generations 8 and 9 and ends with 1713 / 7687; `xor3d_p500` again at 8 and 9 and ends with 884 / 3200, its species count reaching the 20-slot cap.
`custom_pagie_p500` is the informative outlier: the float tolerance is exceeded at generation 7 (node-attribute difference 0.306) while the keys stay identical through generation 9, they diverge for exactly one generation at 10 (1 node key, 4 conn keys, 6 mask mismatches) and they are identical again from generation 11 to 14, with a fitness difference of 1.2e-5 and the same best fitness at generation 14; its species count grows 1 → 8 and stays below the cap.
The order is the same on every row that drifts — attributes first, structural genes one to three generations later — and the population's best fitness stays put while it happens: within 3.0e-8 on the `xor` and pagie rows at every traced generation and within 6.0e-4 on `xor3d_p500`, which is what a selection near-tie looks like rather than a diverging evaluation.
The mechanism, read from the code rather than inferred: a sub-ulp difference in the attribute payload is amplified by fitness ranking in `src/tensorneat/algorithm/neat/_kernels.py`: `sort_desc_legacy` (`:151`) is a stable ascending argsort gathered through the reversal permutation, i.e. a strict total order in which an exact tie is broken by row index, and `pair_parents` (`:411`) ranks with it at `:454`, draws parents at `:459` and pins elite slot *k* to that block's rank *k* at `:482-491`.
The benchmark cases carry `survival_threshold = 0.01` and `species_size = 20` (`benchmarks/cases.py`), so the survivor count `clamp(floor(0.01 × members), 1, …)` leaves every species under 200 members exactly one breeding parent while the pop-500 and pop-1000 tells fill all 20 species slots: that is up to 20 independent rank-0 decisions per generation, and one ulp-level tie flip changes a species' pinned elites and its single parent a generation or more before it can change the structural genes — precisely the 5 → 6, 8 → 9 and 7 → 10 orders the trace shows.
At pop 100 only 1–6 species exist per tell inside the 14 traced generations and nothing flips, which is why `xor_p100` stays key-identical; `cartpole_p100` never exceeds one species and its return is saturated, so it has nothing to flip either.
`custom_pagie_p500`'s single-generation divergence and re-convergence is the same class of event seen from the other side: a discrete decision resolved differently on a sub-ulp difference, whose pay-off disappeared once the two populations ranked the same way again; this record records the effect and does not localise which individual gate flipped, because no per-decision instrumentation is part of the protocol.
So generation 1 is the equality gate and nothing else is: every later generation is a search-trajectory observation, and the entity to compare over a long horizon is the fitness view — the population's best, agreeing to float32 ulps on every traced row — never the key arrays.
The trace is part of the record (`trajectory` per row, `protocol.trajectory_generations: 14`), so the table re-renders from it without re-measuring and `--trace N` reproduces it.

<!-- BEGIN MEASURED (gpu_measure.py) -->
## Environment (measured, not assumed)

| Item | Value |
|---|---|
| Date / session (UTC) | 2026-11-06T07:11:39+00:00 – 2026-11-06T07:17:50+00:00 |
| GPU model (gfx version) | AMD Instinct MI300X VF (gfx942) |
| HIP devices (IREE runtime query) | device_id=1 AMD Instinct MI300X VF |
| ROCm version | 7.1.0 |
| Driver version | 6.16.6 |
| IREE compiler / runtime | 3.13.0 / 3.13.0 |
| PJRT plugin (xla targets) | `xla_rocm_plugin.so` (480.1 MiB, sha256 `8e8319b48fcc…`) |
| evoxc | 0.1.0 |
| numpy | 2.5.4 |
| Python | 3.12.3 |
| Platform | Linux-6.8.0-146-generic-x86_64-with-glibc2.39 |
| CPU model / count | INTEL(R) XEON(R) PLATINUM 8568Y+ / 20 logical |
| Memory | 241611 MiB |
| Threads | runtime defaults (nothing exported; inherited environment recorded) |
| Git commit | `ddd7295afc5a` (clean — generated report artifacts are not counted) |
| Load average (1/5/15 min, document build) | 5.23 / 5.04 / 3.49 |

## Measured: numpy vs iree-rocm vs xla-rocm

One row per planned `(case, target)`; `n/a` marks a metric the row does not have (an error row has no timings at all). `build` is the one-off generation-graph compile and is excluded from every `ms/gen` sample; `peak RSS` is the process high-water mark of that single row, which is why each row runs in its own process.

| Case | Target | Backend | Status | setup (s) | build (s) | ms/gen median | ms/gen mean | ms/gen min–max | gens/s | peak RSS (MiB) | final max fitness | cross-check |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| xor_p100 | `numpy` | `numpy` | ok | 0.016 | 0.287 | 43.300 | 43.573 | 35.268 – 56.056 | 22.950 | 156.6 | -0.176966 | skipped (target is the numpy reference) |
| xor_p100 | `iree-rocm` | `iree` | ok | 1.305 | 8.017 | 10.295 | 10.403 | 9.926 – 12.005 | 96.124 | 1894.9 | -0.184767 | gen 1 pass; final-state drift (node_keys) |
| xor_p100 | `xla-rocm` | `xla` | ok | 4.460 | 2.149 | 3.610 | 3.611 | 3.500 – 3.746 | 276.918 | 9368.5 | -0.176966 | pass (gen 1) |
| xor_p500 | `numpy` | `numpy` | ok | 0.016 | 0.318 | 120.186 | 119.846 | 112.608 – 126.576 | 8.344 | 240.0 | -0.147786 | skipped (target is the numpy reference) |
| xor_p500 | `iree-rocm` | `iree` | ok | 1.218 | 8.212 | 12.820 | 12.947 | 12.698 – 14.543 | 77.238 | 1913.3 | -0.112459 | gen 1 pass; final-state drift (node_keys) |
| xor_p500 | `xla-rocm` | `xla` | ok | 4.334 | 2.197 | 4.535 | 4.523 | 4.386 – 4.649 | 221.094 | 9387.8 | -0.140949 | gen 1 pass; final-state drift (node_keys) |
| xor_p1000 | `numpy` | `numpy` | ok | 0.017 | 0.382 | 182.567 | 184.049 | 175.568 – 199.017 | 5.433 | 341.6 | -0.053339 | skipped (target is the numpy reference) |
| xor_p1000 | `iree-rocm` | `iree` | ok | 1.326 | 8.161 | 15.117 | 15.194 | 14.787 – 15.859 | 65.814 | 1936.6 | -0.136390 | gen 1 pass; final-state drift (node_keys) |
| xor_p1000 | `xla-rocm` | `xla` | ok | 4.354 | 2.367 | 5.497 | 5.503 | 5.317 – 5.625 | 181.720 | 9423.4 | -0.053339 | gen 1 pass; final-state drift (node_keys) |
| xor3d_p500 | `numpy` | `numpy` | ok | 0.018 | 0.323 | 136.063 | 137.903 | 126.384 – 168.140 | 7.252 | 248.0 | -0.223466 | skipped (target is the numpy reference) |
| xor3d_p500 | `iree-rocm` | `iree` | ok | 1.466 | 8.617 | 12.920 | 13.085 | 12.822 – 14.637 | 76.421 | 1914.9 | -0.250000 | gen 1 pass; final-state drift (node_keys) |
| xor3d_p500 | `xla-rocm` | `xla` | ok | 4.435 | 2.332 | 4.580 | 4.594 | 4.432 – 4.988 | 217.683 | 9413.5 | -0.223466 | gen 1 pass; final-state drift (node_keys) |
| custom_pagie_p500 | `numpy` | `numpy` | ok | 0.018 | 0.805 | 965.043 | 971.508 | 888.500 – 1092.814 | 1.029 | 1306.2 | -0.032613 | skipped (target is the numpy reference) |
| custom_pagie_p500 | `iree-rocm` | `iree` | ok | 1.305 | 9.027 | 15.768 | 15.282 | 14.298 – 16.103 | 65.438 | 1942.7 | -0.046049 | gen 1 pass; final-state drift (node_keys) |
| custom_pagie_p500 | `xla-rocm` | `xla` | ok | 4.455 | 2.672 | 6.319 | 6.363 | 6.146 – 6.816 | 157.167 | 9427.1 | -0.032613 | gen 1 pass; final-state drift (node_keys) |
| cartpole_p100 | `numpy` | `numpy` | ok | 0.016 | 0.283 | 236.023 | 256.083 | 229.720 – 370.927 | 3.905 | 155.2 | 200.000000 | skipped (target is the numpy reference) |
| cartpole_p100 | `iree-rocm` | `iree` | ok | 1.234 | 8.711 | 61.495 | 61.785 | 59.606 – 64.012 | 16.185 | 1934.3 | 200.000000 | pass (gen 1) |
| cartpole_p100 | `xla-rocm` | `xla` | ok | 4.357 | 2.227 | 13.362 | 13.433 | 13.182 – 13.734 | 74.444 | 9390.9 | 200.000000 | gen 1 pass; final-state drift (fitness) |

## Numerical cross-check

Each non-numpy row is compared against a fresh `backend="numpy"` run of the same case, seed and step count in the same process. Generation 1 must match (one seed, one graph, one PRNG): integer keys exactly, attributes and fitness to rtol 1e-05 / atol 1e-06 over finite entries with an identical NaN mask. The final state is a search-trajectory observation, not a pass/fail gate on its own: beyond generation 1 a float-reassociation difference can reorder a near-tied selection and legitimately diverge the keys. The table's `cross-check` column summarises both snapshots: generation 1 is the equality gate; a final-state-only difference renders `gen 1 pass; final-state drift (…)`.

| Case | Target | Generation 1 | Keys (gen 1) | Attr max abs diff (gen 1) | Fitness max abs diff | Mask mismatches | Final state | Keys (final) | Attr max abs diff (final) | Fitness max abs diff (final) | Mask mismatches (final) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| xor_p100 | `iree-rocm` | pass | node_keys exact; conn_keys exact | node_attrs 0.000000238; conn_attrs 0.000000238 | 0.000000060 | 0 | fail | node_keys 189 mismatch(es); conn_keys 686 mismatch(es) | node_attrs 1.492010832; conn_attrs 4.826976299 | 0.131897777 | 1080 |
| xor_p100 | `xla-rocm` | pass | node_keys exact; conn_keys exact | node_attrs 0.000000238; conn_attrs 0.000000238 | 0.000000060 | 0 | pass | node_keys exact; conn_keys exact | node_attrs 0.000000149; conn_attrs 0.000000238 | 0.000000045 | 0 |
| xor_p500 | `iree-rocm` | pass | node_keys exact; conn_keys exact | node_attrs 0.000000477; conn_attrs 0.000000294 | 0.000000060 | 0 | fail | node_keys 734 mismatch(es); conn_keys 3545 mismatch(es) | node_attrs 2.440064669; conn_attrs 7.131760120 | 0.266454309 | 4278 |
| xor_p500 | `xla-rocm` | pass | node_keys exact; conn_keys exact | node_attrs 0.000000238; conn_attrs 0.000000238 | 0.000000089 | 0 | fail | node_keys 857 mismatch(es); conn_keys 3765 mismatch(es) | node_attrs 2.720985651; conn_attrs 7.074343204 | 0.230899110 | 5990 |
| xor_p1000 | `iree-rocm` | pass | node_keys exact; conn_keys exact | node_attrs 0.000000477; conn_attrs 0.000000298 | 0.000000089 | 0 | fail | node_keys 1687 mismatch(es); conn_keys 8767 mismatch(es) | node_attrs 3.298459530; conn_attrs 6.690523148 | 0.420150518 | 7554 |
| xor_p1000 | `xla-rocm` | pass | node_keys exact; conn_keys exact | node_attrs 0.000000238; conn_attrs 0.000000238 | 0.000000119 | 0 | fail | node_keys 1527 mismatch(es); conn_keys 7624 mismatch(es) | node_attrs 4.058683395; conn_attrs 6.799370766 | 0.418266296 | 9380 |
| xor3d_p500 | `iree-rocm` | pass | node_keys exact; conn_keys exact | node_attrs 0.000000477; conn_attrs 0.000000298 | 0.000000060 | 0 | fail | node_keys 976 mismatch(es); conn_keys 4456 mismatch(es) | node_attrs 4.066220284; conn_attrs 4.489749908 | 0.153934240 | 7562 |
| xor3d_p500 | `xla-rocm` | pass | node_keys exact; conn_keys exact | node_attrs 0.000000238; conn_attrs 0.000000238 | 0.000000060 | 0 | fail | node_keys 959 mismatch(es); conn_keys 4229 mismatch(es) | node_attrs 4.066220284; conn_attrs 4.200792313 | 0.152276725 | 4512 |
| custom_pagie_p500 | `iree-rocm` | pass | node_keys exact; conn_keys exact | node_attrs 0.000000477; conn_attrs 0.000000294 | 0.000000238 | 0 | fail | node_keys 507 mismatch(es); conn_keys 2853 mismatch(es) | node_attrs 3.389557362; conn_attrs 3.321706295 | 0.095183104 | 5962 |
| custom_pagie_p500 | `xla-rocm` | pass | node_keys exact; conn_keys exact | node_attrs 0.000000238; conn_attrs 0.000000238 | 0.000000089 | 0 | fail | node_keys 237 mismatch(es); conn_keys 1458 mismatch(es) | node_attrs 3.406351328; conn_attrs 2.775112629 | 0.112810448 | 1896 |
| cartpole_p100 | `iree-rocm` | pass | node_keys exact; conn_keys exact | node_attrs 0.000000477; conn_attrs 0.000000294 | 0.000000000 | 0 | pass | node_keys exact; conn_keys exact | node_attrs 0.000000119; conn_attrs 0.000000238 | 0.000000000 | 0 |
| cartpole_p100 | `xla-rocm` | pass | node_keys exact; conn_keys exact | node_attrs 0.000000238; conn_attrs 0.000000238 | 0.000000000 | 0 | fail | node_keys exact; conn_keys exact | node_attrs 0.000000119; conn_attrs 0.000000238 | 1.000000000 | 0 |

## Per-generation trajectory

Every traced row was re-run after the timed region, with the peak-RSS sample already taken, as a lockstep pair: a fresh accelerated pipeline and a fresh `backend="numpy"` pipeline of the same case and seed, each paying its own build, compared per generation with the cross-check rules above. The trace touches neither the timings nor the recorded RSS.

**xor_p100 / `iree-rocm`** — 14 generation(s) compared in lockstep; first node-key divergence: 13; first float exceedance: 13; wall time 9.924 s.

*Both legs share one PRNG from one seed and one graph, so a first key divergence beyond generation 1 is a selection near-tie flip caused by a sub-ulp float difference between the backends, not a kernel defect.*

| Generation | Node-key mismatches | Conn-key mismatches | Node attr max abs diff | Conn attr max abs diff | Attr NaN-mask mismatches | Fitness max abs diff | Fitness mask mismatches | Max fitness (numpy / accelerated) | Species (numpy / accelerated) |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 0 | 0 | 0.000000238 | 0.000000238 | 0 | 0.000000060 | 0 | -0.253215 / -0.253215 | 1 / 1 |
| 2 | 0 | 0 | 0.000000119 | 0.000000238 | 0 | 0.000000060 | 0 | -0.250795 / -0.250795 | 1 / 1 |
| 3 | 0 | 0 | 0.000000119 | 0.000000119 | 0 | 0.000000060 | 0 | -0.250101 / -0.250101 | 2 / 2 |
| 4 | 0 | 0 | 0.000000119 | 0.000000119 | 0 | 0.000000060 | 0 | -0.250050 / -0.250050 | 2 / 2 |
| 5 | 0 | 0 | 0.000000119 | 0.000000119 | 0 | 0.000000060 | 0 | -0.249919 / -0.249919 | 2 / 2 |
| 6 | 0 | 0 | 0.000000119 | 0.000000238 | 0 | 0.000000060 | 0 | -0.249629 / -0.249629 | 2 / 2 |
| 7 | 0 | 0 | 0.000000060 | 0.000000119 | 0 | 0.000000060 | 0 | -0.248715 / -0.248715 | 2 / 2 |
| 8 | 0 | 0 | 0.000000119 | 0.000000238 | 0 | 0.000000030 | 0 | -0.247675 / -0.247675 | 2 / 2 |
| 9 | 0 | 0 | 0.000000119 | 0.000000238 | 0 | 0.000000060 | 0 | -0.246592 / -0.246592 | 2 / 2 |
| 10 | 0 | 0 | 0.000000119 | 0.000000119 | 0 | 0.000000060 | 0 | -0.245527 / -0.245527 | 3 / 3 |
| 11 | 0 | 0 | 0.000000060 | 0.000000119 | 0 | 0.000000060 | 0 | -0.243673 / -0.243673 | 6 / 6 |
| 12 | 0 | 0 | 0.000000119 | 0.000000238 | 0 | 0.000000060 | 0 | -0.241268 / -0.241268 | 6 / 6 |
| 13 | 27 | 118 | 2.164435387 | 2.167837143 | 101 | 0.113686353 | 4 | -0.237129 / -0.238927 | 6 / 6 |
| 14 | 53 | 187 | 0.813701153 | 1.681092739 | 202 | 0.086849481 | 4 | -0.233823 / -0.235706 | 6 / 6 |

**xor_p100 / `xla-rocm`** — 14 generation(s) compared in lockstep; first node-key divergence: none within 14 generations; first float exceedance: none within 14 generations; wall time 2.646 s.

*Both legs share one PRNG from one seed and one graph, so a first key divergence beyond generation 1 is a selection near-tie flip caused by a sub-ulp float difference between the backends, not a kernel defect.*

| Generation | Node-key mismatches | Conn-key mismatches | Node attr max abs diff | Conn attr max abs diff | Attr NaN-mask mismatches | Fitness max abs diff | Fitness mask mismatches | Max fitness (numpy / accelerated) | Species (numpy / accelerated) |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 0 | 0 | 0.000000238 | 0.000000238 | 0 | 0.000000060 | 0 | -0.253215 / -0.253215 | 1 / 1 |
| 2 | 0 | 0 | 0.000000119 | 0.000000238 | 0 | 0.000000060 | 0 | -0.250795 / -0.250795 | 1 / 1 |
| 3 | 0 | 0 | 0.000000119 | 0.000000119 | 0 | 0.000000060 | 0 | -0.250101 / -0.250101 | 2 / 2 |
| 4 | 0 | 0 | 0.000000119 | 0.000000119 | 0 | 0.000000060 | 0 | -0.250050 / -0.250050 | 2 / 2 |
| 5 | 0 | 0 | 0.000000119 | 0.000000119 | 0 | 0.000000060 | 0 | -0.249919 / -0.249919 | 2 / 2 |
| 6 | 0 | 0 | 0.000000179 | 0.000000238 | 0 | 0.000000060 | 0 | -0.249629 / -0.249629 | 2 / 2 |
| 7 | 0 | 0 | 0.000000119 | 0.000000119 | 0 | 0.000000060 | 0 | -0.248715 / -0.248715 | 2 / 2 |
| 8 | 0 | 0 | 0.000000119 | 0.000000238 | 0 | 0.000000045 | 0 | -0.247675 / -0.247675 | 2 / 2 |
| 9 | 0 | 0 | 0.000000238 | 0.000000238 | 0 | 0.000000060 | 0 | -0.246592 / -0.246592 | 2 / 2 |
| 10 | 0 | 0 | 0.000000119 | 0.000000119 | 0 | 0.000000060 | 0 | -0.245527 / -0.245527 | 3 / 3 |
| 11 | 0 | 0 | 0.000000238 | 0.000000119 | 0 | 0.000000060 | 0 | -0.243673 / -0.243673 | 6 / 6 |
| 12 | 0 | 0 | 0.000000179 | 0.000000238 | 0 | 0.000000045 | 0 | -0.241268 / -0.241268 | 6 / 6 |
| 13 | 0 | 0 | 0.000000119 | 0.000000119 | 0 | 0.000000030 | 0 | -0.237129 / -0.237129 | 6 / 6 |
| 14 | 0 | 0 | 0.000000119 | 0.000000119 | 0 | 0.000000060 | 0 | -0.233823 / -0.233823 | 6 / 6 |

**xor_p500 / `iree-rocm`** — 14 generation(s) compared in lockstep; first node-key divergence: 4; first float exceedance: 4; wall time 11.223 s.

*Both legs share one PRNG from one seed and one graph, so a first key divergence beyond generation 1 is a selection near-tie flip caused by a sub-ulp float difference between the backends, not a kernel defect.*

| Generation | Node-key mismatches | Conn-key mismatches | Node attr max abs diff | Conn attr max abs diff | Attr NaN-mask mismatches | Fitness max abs diff | Fitness mask mismatches | Max fitness (numpy / accelerated) | Species (numpy / accelerated) |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 0 | 0 | 0.000000477 | 0.000000294 | 0 | 0.000000060 | 0 | -0.250251 / -0.250251 | 1 / 1 |
| 2 | 0 | 0 | 0.000000119 | 0.000000238 | 0 | 0.000000060 | 0 | -0.250017 / -0.250017 | 6 / 3 |
| 3 | 0 | 0 | 0.000000149 | 0.000000238 | 0 | 0.000000060 | 0 | -0.250007 / -0.250007 | 16 / 6 |
| 4 | 308 | 1720 | 2.618898392 | 3.687131405 | 1789 | 0.097838163 | 14 | -0.250001 / -0.250001 | 20 / 6 |
| 5 | 433 | 2154 | 3.048482180 | 3.928130150 | 1764 | 0.078658432 | 25 | -0.249823 / -0.249994 | 20 / 6 |
| 6 | 563 | 2590 | 2.242585182 | 3.767443895 | 1436 | 0.162336916 | 27 | -0.248735 / -0.249380 | 20 / 6 |
| 7 | 667 | 2650 | 2.098993301 | 3.576029778 | 1588 | 0.102464199 | 22 | -0.247461 / -0.245810 | 20 / 6 |
| 8 | 645 | 2686 | 2.583975077 | 4.072798729 | 1524 | 0.112536043 | 18 | -0.243918 / -0.231725 | 20 / 6 |
| 9 | 658 | 2802 | 1.784778357 | 4.102497101 | 1700 | 0.141785219 | 23 | -0.241320 / -0.217541 | 20 / 6 |
| 10 | 635 | 2735 | 1.898266077 | 4.312487602 | 1567 | 0.121345967 | 31 | -0.240657 / -0.208400 | 20 / 6 |
| 11 | 698 | 2788 | 2.298183441 | 4.652419090 | 1816 | 0.129715905 | 23 | -0.237867 / -0.196763 | 20 / 6 |
| 12 | 722 | 2918 | 2.593549967 | 4.535817146 | 1813 | 0.125310659 | 25 | -0.233631 / -0.192127 | 20 / 6 |
| 13 | 688 | 2979 | 4.043320179 | 4.633800030 | 1700 | 0.103745818 | 35 | -0.228627 / -0.184734 | 20 / 6 |
| 14 | 754 | 3099 | 2.290542126 | 4.885830402 | 1975 | 0.181158334 | 21 | -0.224597 / -0.174005 | 20 / 6 |

**xor_p500 / `xla-rocm`** — 14 generation(s) compared in lockstep; first node-key divergence: 6; first float exceedance: 5; wall time 3.994 s.

*Both legs share one PRNG from one seed and one graph, so a first key divergence beyond generation 1 is a selection near-tie flip caused by a sub-ulp float difference between the backends, not a kernel defect.*

| Generation | Node-key mismatches | Conn-key mismatches | Node attr max abs diff | Conn attr max abs diff | Attr NaN-mask mismatches | Fitness max abs diff | Fitness mask mismatches | Max fitness (numpy / accelerated) | Species (numpy / accelerated) |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 0 | 0 | 0.000000238 | 0.000000238 | 0 | 0.000000089 | 0 | -0.250251 / -0.250251 | 1 / 1 |
| 2 | 0 | 0 | 0.000000477 | 0.000000238 | 0 | 0.000000060 | 0 | -0.250017 / -0.250017 | 6 / 6 |
| 3 | 0 | 0 | 0.000000238 | 0.000000238 | 0 | 0.000000060 | 0 | -0.250007 / -0.250007 | 16 / 16 |
| 4 | 0 | 0 | 0.000000238 | 0.000000238 | 0 | 0.000000060 | 0 | -0.250001 / -0.250001 | 20 / 20 |
| 5 | 0 | 0 | 0.217691779 | 0.003084108 | 0 | 0.000197470 | 0 | -0.249823 / -0.249823 | 20 / 20 |
| 6 | 119 | 375 | 2.586089373 | 1.691174030 | 72 | 0.039434135 | 6 | -0.248735 / -0.248735 | 20 / 20 |
| 7 | 332 | 1141 | 1.609423399 | 2.881172419 | 1056 | 0.093943715 | 5 | -0.247461 / -0.246979 | 20 / 20 |
| 8 | 405 | 1263 | 1.491860032 | 4.651563644 | 1086 | 0.082730204 | 3 | -0.243918 / -0.245092 | 20 / 20 |
| 9 | 589 | 2079 | 1.595991731 | 4.324651241 | 1713 | 0.100329936 | 13 | -0.241320 / -0.243271 | 20 / 20 |
| 10 | 574 | 1972 | 1.595991731 | 4.690880775 | 1618 | 0.124292791 | 17 | -0.240657 / -0.242666 | 20 / 20 |
| 11 | 627 | 2289 | 2.092477322 | 5.117555618 | 1926 | 0.097553045 | 15 | -0.237867 / -0.239880 | 20 / 20 |
| 12 | 625 | 2364 | 2.593549967 | 4.449964523 | 1521 | 0.113933265 | 21 | -0.233631 / -0.236103 | 20 / 20 |
| 13 | 634 | 2507 | 4.043320179 | 4.839536667 | 1667 | 0.137924224 | 17 | -0.228627 / -0.231327 | 20 / 20 |
| 14 | 639 | 2907 | 2.290542126 | 4.711293697 | 1713 | 0.103182733 | 16 | -0.224597 / -0.227711 | 20 / 20 |

**xor_p1000 / `iree-rocm`** — 14 generation(s) compared in lockstep; first node-key divergence: 4; first float exceedance: 4; wall time 13.194 s.

*Both legs share one PRNG from one seed and one graph, so a first key divergence beyond generation 1 is a selection near-tie flip caused by a sub-ulp float difference between the backends, not a kernel defect.*

| Generation | Node-key mismatches | Conn-key mismatches | Node attr max abs diff | Conn attr max abs diff | Attr NaN-mask mismatches | Fitness max abs diff | Fitness mask mismatches | Max fitness (numpy / accelerated) | Species (numpy / accelerated) |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 0 | 0 | 0.000000477 | 0.000000298 | 0 | 0.000000089 | 0 | -0.250251 / -0.250251 | 2 / 2 |
| 2 | 0 | 0 | 0.000000238 | 0.000000238 | 0 | 0.000000060 | 0 | -0.250060 / -0.250060 | 12 / 5 |
| 3 | 0 | 0 | 0.000000238 | 0.000000238 | 0 | 0.000000060 | 0 | -0.250001 / -0.250001 | 20 / 12 |
| 4 | 600 | 3449 | 3.142985821 | 2.540262222 | 2201 | 0.108110696 | 30 | -0.249913 / -0.250001 | 20 / 12 |
| 5 | 1088 | 5767 | 2.857068777 | 2.704190731 | 3712 | 0.109995067 | 62 | -0.249371 / -0.249985 | 20 / 12 |
| 6 | 1739 | 7095 | 3.530684710 | 2.946155310 | 5409 | 0.107642204 | 45 | -0.248355 / -0.249422 | 20 / 12 |
| 7 | 1762 | 7061 | 3.324771166 | 3.175319195 | 5089 | 0.110316157 | 50 | -0.246130 / -0.249157 | 20 / 12 |
| 8 | 1806 | 7173 | 3.536395550 | 4.040146351 | 5196 | 0.168323487 | 50 | -0.235605 / -0.248230 | 20 / 12 |
| 9 | 1713 | 7147 | 3.307558060 | 4.064034462 | 4586 | 0.123532623 | 50 | -0.232334 / -0.245386 | 20 / 12 |
| 10 | 1674 | 6902 | 3.386581659 | 3.137895107 | 4795 | 0.127675205 | 51 | -0.218614 / -0.247435 | 20 / 12 |
| 11 | 1485 | 7060 | 3.233250618 | 4.073899269 | 4282 | 0.127774924 | 39 | -0.212860 / -0.243339 | 20 / 12 |
| 12 | 1499 | 6868 | 3.050926685 | 4.206381798 | 3604 | 0.117527127 | 65 | -0.206379 / -0.236054 | 20 / 12 |
| 13 | 1510 | 7107 | 3.303441048 | 4.303811550 | 4321 | 0.193081409 | 64 | -0.199304 / -0.223004 | 20 / 12 |
| 14 | 1428 | 6903 | 3.186868191 | 4.832839966 | 4277 | 0.198184818 | 53 | -0.187691 / -0.215279 | 20 / 12 |

**xor_p1000 / `xla-rocm`** — 14 generation(s) compared in lockstep; first node-key divergence: 9; first float exceedance: 8; wall time 5.594 s.

*Both legs share one PRNG from one seed and one graph, so a first key divergence beyond generation 1 is a selection near-tie flip caused by a sub-ulp float difference between the backends, not a kernel defect.*

| Generation | Node-key mismatches | Conn-key mismatches | Node attr max abs diff | Conn attr max abs diff | Attr NaN-mask mismatches | Fitness max abs diff | Fitness mask mismatches | Max fitness (numpy / accelerated) | Species (numpy / accelerated) |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 0 | 0 | 0.000000238 | 0.000000238 | 0 | 0.000000119 | 0 | -0.250251 / -0.250251 | 2 / 2 |
| 2 | 0 | 0 | 0.000000477 | 0.000000238 | 0 | 0.000000060 | 0 | -0.250060 / -0.250060 | 12 / 12 |
| 3 | 0 | 0 | 0.000000238 | 0.000000238 | 0 | 0.000000060 | 0 | -0.250001 / -0.250001 | 20 / 20 |
| 4 | 0 | 0 | 0.000000119 | 0.000000238 | 0 | 0.000000060 | 0 | -0.249913 / -0.249913 | 20 / 20 |
| 5 | 0 | 0 | 0.000000119 | 0.000000238 | 0 | 0.000000060 | 0 | -0.249371 / -0.249371 | 20 / 20 |
| 6 | 0 | 0 | 0.000000119 | 0.000000238 | 0 | 0.000000060 | 0 | -0.248355 / -0.248355 | 20 / 20 |
| 7 | 0 | 0 | 0.000000238 | 0.000000238 | 0 | 0.000000060 | 0 | -0.246130 / -0.246130 | 20 / 20 |
| 8 | 0 | 0 | 0.130168200 | 0.005506754 | 0 | 0.000000060 | 0 | -0.235605 / -0.235605 | 20 / 20 |
| 9 | 18 | 92 | 3.740734100 | 2.835820675 | 66 | 0.010366738 | 1 | -0.232334 / -0.232334 | 20 / 20 |
| 10 | 243 | 1357 | 3.740734100 | 4.559720993 | 1427 | 0.087913930 | 1 | -0.218614 / -0.218614 | 20 / 20 |
| 11 | 614 | 2870 | 3.159790277 | 5.169526100 | 1591 | 0.102426678 | 8 | -0.212860 / -0.212860 | 20 / 20 |
| 12 | 1257 | 5537 | 3.268372774 | 5.513322830 | 3801 | 0.156097621 | 30 | -0.206379 / -0.206379 | 20 / 20 |
| 13 | 896 | 3740 | 3.268372774 | 5.399968147 | 2690 | 0.185953230 | 16 | -0.199304 / -0.199304 | 20 / 20 |
| 14 | 1257 | 4970 | 3.783902168 | 5.535475731 | 4239 | 0.218329474 | 25 | -0.187691 / -0.187691 | 20 / 20 |

**xor3d_p500 / `iree-rocm`** — 14 generation(s) compared in lockstep; first node-key divergence: 5; first float exceedance: 5; wall time 11.785 s.

*Both legs share one PRNG from one seed and one graph, so a first key divergence beyond generation 1 is a selection near-tie flip caused by a sub-ulp float difference between the backends, not a kernel defect.*

| Generation | Node-key mismatches | Conn-key mismatches | Node attr max abs diff | Conn attr max abs diff | Attr NaN-mask mismatches | Fitness max abs diff | Fitness mask mismatches | Max fitness (numpy / accelerated) | Species (numpy / accelerated) |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 0 | 0 | 0.000000477 | 0.000000298 | 0 | 0.000000060 | 0 | -0.251171 / -0.251171 | 1 / 1 |
| 2 | 0 | 0 | 0.000000179 | 0.000000358 | 0 | 0.000000060 | 0 | -0.250194 / -0.250194 | 1 / 1 |
| 3 | 0 | 0 | 0.000000119 | 0.000000328 | 0 | 0.000000060 | 0 | -0.250011 / -0.250011 | 3 / 3 |
| 4 | 0 | 0 | 0.000000238 | 0.000000328 | 0 | 0.000000089 | 0 | -0.250004 / -0.250004 | 3 / 3 |
| 5 | 86 | 792 | 1.953010201 | 2.348107815 | 671 | 0.091053963 | 12 | -0.250001 / -0.250002 | 20 / 3 |
| 6 | 127 | 1537 | 1.840878963 | 1.091112375 | 1183 | 0.103558660 | 9 | -0.250000 / -0.250001 | 20 / 3 |
| 7 | 556 | 2668 | 1.980885267 | 3.377946854 | 2372 | 0.114442766 | 41 | -0.250000 / -0.250000 | 20 / 3 |
| 8 | 529 | 2592 | 2.809998512 | 2.657910347 | 1947 | 0.085243106 | 23 | -0.249999 / -0.250000 | 20 / 3 |
| 9 | 576 | 2906 | 3.217855453 | 2.737298012 | 1999 | 0.095178545 | 9 | -0.249980 / -0.250000 | 20 / 3 |
| 10 | 654 | 2968 | 2.132001162 | 1.575432777 | 2310 | 0.139350057 | 26 | -0.249943 / -0.250000 | 20 / 3 |
| 11 | 732 | 3174 | 4.228101254 | 2.438176394 | 2161 | 0.080694973 | 23 | -0.249839 / -0.250000 | 20 / 3 |
| 12 | 802 | 3245 | 4.228100777 | 3.425137281 | 2424 | 0.117904425 | 25 | -0.249462 / -0.250000 | 20 / 3 |
| 13 | 852 | 3352 | 4.248557568 | 2.914254904 | 2660 | 0.080749840 | 27 | -0.249007 / -0.250000 | 20 / 3 |
| 14 | 988 | 3647 | 3.936978817 | 2.440686226 | 3330 | 0.047573268 | 16 | -0.247870 / -0.250000 | 20 / 3 |

**xor3d_p500 / `xla-rocm`** — 14 generation(s) compared in lockstep; first node-key divergence: 8; first float exceedance: 8; wall time 4.090 s.

*Both legs share one PRNG from one seed and one graph, so a first key divergence beyond generation 1 is a selection near-tie flip caused by a sub-ulp float difference between the backends, not a kernel defect.*

| Generation | Node-key mismatches | Conn-key mismatches | Node attr max abs diff | Conn attr max abs diff | Attr NaN-mask mismatches | Fitness max abs diff | Fitness mask mismatches | Max fitness (numpy / accelerated) | Species (numpy / accelerated) |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 0 | 0 | 0.000000238 | 0.000000238 | 0 | 0.000000060 | 0 | -0.251171 / -0.251171 | 1 / 1 |
| 2 | 0 | 0 | 0.000000119 | 0.000000238 | 0 | 0.000000060 | 0 | -0.250194 / -0.250194 | 1 / 1 |
| 3 | 0 | 0 | 0.000000119 | 0.000000238 | 0 | 0.000000060 | 0 | -0.250011 / -0.250011 | 3 / 3 |
| 4 | 0 | 0 | 0.000000238 | 0.000000238 | 0 | 0.000000060 | 0 | -0.250004 / -0.250004 | 3 / 3 |
| 5 | 0 | 0 | 0.000000238 | 0.000000238 | 0 | 0.000000060 | 0 | -0.250001 / -0.250001 | 20 / 20 |
| 6 | 0 | 0 | 0.000000238 | 0.000000119 | 0 | 0.000000060 | 0 | -0.250000 / -0.250000 | 20 / 20 |
| 7 | 0 | 0 | 0.000000119 | 0.000000238 | 0 | 0.000000030 | 0 | -0.250000 / -0.250000 | 20 / 20 |
| 8 | 43 | 220 | 1.093454957 | 1.200019002 | 221 | 0.066319168 | 4 | -0.249999 / -0.249999 | 20 / 20 |
| 9 | 127 | 550 | 1.899353027 | 2.193921089 | 295 | 0.090819627 | 4 | -0.249980 / -0.249980 | 20 / 20 |
| 10 | 562 | 2135 | 2.144493103 | 3.445238590 | 2216 | 0.133314312 | 22 | -0.249943 / -0.249943 | 20 / 20 |
| 11 | 791 | 2750 | 2.467591524 | 3.774496555 | 2480 | 0.107461154 | 19 | -0.249839 / -0.249839 | 20 / 20 |
| 12 | 711 | 2824 | 3.996948242 | 3.603818178 | 2223 | 0.150442004 | 23 | -0.249462 / -0.249462 | 20 / 20 |
| 13 | 853 | 3114 | 3.996948242 | 4.945690155 | 1948 | 0.098944753 | 18 | -0.249007 / -0.249007 | 20 / 20 |
| 14 | 906 | 3311 | 2.749063015 | 5.074888229 | 2843 | 0.089159191 | 20 | -0.247870 / -0.247870 | 20 / 20 |

**custom_pagie_p500 / `iree-rocm`** — 14 generation(s) compared in lockstep; first node-key divergence: 4; first float exceedance: 4; wall time 22.996 s.

*Both legs share one PRNG from one seed and one graph, so a first key divergence beyond generation 1 is a selection near-tie flip caused by a sub-ulp float difference between the backends, not a kernel defect.*

| Generation | Node-key mismatches | Conn-key mismatches | Node attr max abs diff | Conn attr max abs diff | Attr NaN-mask mismatches | Fitness max abs diff | Fitness mask mismatches | Max fitness (numpy / accelerated) | Species (numpy / accelerated) |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 0 | 0 | 0.000000477 | 0.000000294 | 0 | 0.000000238 | 0 | -0.046480 / -0.046480 | 1 / 1 |
| 2 | 0 | 0 | 0.000000134 | 0.000000238 | 0 | 0.000000063 | 0 | -0.046061 / -0.046061 | 3 / 3 |
| 3 | 0 | 0 | 0.000000238 | 0.000000238 | 0 | 0.000000089 | 0 | -0.046049 / -0.046049 | 3 / 3 |
| 4 | 90 | 510 | 2.348528624 | 1.000000000 | 579 | 0.034655090 | 1 | -0.046049 / -0.046049 | 4 / 3 |
| 5 | 123 | 700 | 2.387268543 | 2.084776640 | 781 | 0.045766756 | 1 | -0.046031 / -0.046049 | 5 / 3 |
| 6 | 241 | 1292 | 2.348528624 | 1.299737096 | 1484 | 0.117517307 | 2 | -0.046018 / -0.046049 | 5 / 3 |
| 7 | 319 | 1777 | 1.973827958 | 1.592835426 | 1930 | 0.083048046 | 7 | -0.045835 / -0.046049 | 6 / 3 |
| 8 | 360 | 1793 | 2.140910149 | 2.081655264 | 2015 | 0.083571717 | 7 | -0.045799 / -0.046049 | 6 / 3 |
| 9 | 369 | 1890 | 2.140910149 | 2.273530960 | 2114 | 0.089705303 | 7 | -0.045137 / -0.046049 | 6 / 3 |
| 10 | 358 | 1857 | 2.196461439 | 2.397579670 | 2039 | 0.072265916 | 2 | -0.044979 / -0.046049 | 6 / 3 |
| 11 | 374 | 1954 | 2.276385069 | 1.619795561 | 2140 | 0.072071239 | 3 | -0.044345 / -0.046049 | 7 / 3 |
| 12 | 427 | 2098 | 2.318641424 | 2.457735062 | 2331 | 0.086077839 | 3 | -0.043920 / -0.046049 | 8 / 3 |
| 13 | 432 | 2099 | 2.306272745 | 2.457735300 | 2336 | 0.058883082 | 5 | -0.042594 / -0.046049 | 8 / 3 |
| 14 | 459 | 2382 | 2.287573338 | 1.607507110 | 2414 | 0.125538796 | 18 | -0.042274 / -0.046049 | 8 / 3 |

**custom_pagie_p500 / `xla-rocm`** — 14 generation(s) compared in lockstep; first node-key divergence: 10; first float exceedance: 7; wall time 16.050 s.

*Both legs share one PRNG from one seed and one graph, so a first key divergence beyond generation 1 is a selection near-tie flip caused by a sub-ulp float difference between the backends, not a kernel defect.*

| Generation | Node-key mismatches | Conn-key mismatches | Node attr max abs diff | Conn attr max abs diff | Attr NaN-mask mismatches | Fitness max abs diff | Fitness mask mismatches | Max fitness (numpy / accelerated) | Species (numpy / accelerated) |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 0 | 0 | 0.000000238 | 0.000000238 | 0 | 0.000000089 | 0 | -0.046480 / -0.046480 | 1 / 1 |
| 2 | 0 | 0 | 0.000000238 | 0.000000238 | 0 | 0.000000022 | 0 | -0.046061 / -0.046061 | 3 / 3 |
| 3 | 0 | 0 | 0.000000238 | 0.000000238 | 0 | 0.000000030 | 0 | -0.046049 / -0.046049 | 3 / 3 |
| 4 | 0 | 0 | 0.000000119 | 0.000000238 | 0 | 0.000000030 | 0 | -0.046049 / -0.046049 | 4 / 4 |
| 5 | 0 | 0 | 0.000000238 | 0.000000238 | 0 | 0.000000060 | 0 | -0.046031 / -0.046031 | 5 / 5 |
| 6 | 0 | 0 | 0.000000238 | 0.000000238 | 0 | 0.000000030 | 0 | -0.046018 / -0.046018 | 5 / 5 |
| 7 | 0 | 0 | 0.306329429 | 0.008525796 | 0 | 0.000130326 | 0 | -0.045835 / -0.045835 | 6 / 6 |
| 8 | 0 | 0 | 0.259954870 | 0.010306090 | 0 | 0.000155181 | 0 | -0.045799 / -0.045799 | 6 / 6 |
| 9 | 0 | 0 | 0.259954810 | 0.010306120 | 0 | 0.000231959 | 0 | -0.045137 / -0.045137 | 6 / 6 |
| 10 | 1 | 4 | 1.414301872 | 0.234590173 | 6 | 0.001782279 | 0 | -0.044979 / -0.044979 | 6 / 6 |
| 11 | 0 | 0 | 0.259954810 | 0.009333849 | 0 | 0.000296667 | 0 | -0.044345 / -0.044345 | 7 / 7 |
| 12 | 0 | 0 | 0.259954840 | 0.009333819 | 0 | 0.000089884 | 0 | -0.043920 / -0.043920 | 8 / 8 |
| 13 | 0 | 0 | 0.259954870 | 0.009333819 | 0 | 0.000088129 | 0 | -0.042594 / -0.042594 | 8 / 8 |
| 14 | 0 | 0 | 0.380616546 | 0.009333789 | 0 | 0.000011727 | 0 | -0.042274 / -0.042274 | 8 / 8 |

**cartpole_p100 / `iree-rocm`** — 14 generation(s) compared in lockstep; first node-key divergence: none within 14 generations; first float exceedance: none within 14 generations; wall time 14.431 s.

*Both legs share one PRNG from one seed and one graph, so a first key divergence beyond generation 1 is a selection near-tie flip caused by a sub-ulp float difference between the backends, not a kernel defect.*

| Generation | Node-key mismatches | Conn-key mismatches | Node attr max abs diff | Conn attr max abs diff | Attr NaN-mask mismatches | Fitness max abs diff | Fitness mask mismatches | Max fitness (numpy / accelerated) | Species (numpy / accelerated) |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 0 | 0 | 0.000000477 | 0.000000294 | 0 | 0.000000000 | 0 | 56.000000 / 56.000000 | 1 / 1 |
| 2 | 0 | 0 | 0.000000119 | 0.000000238 | 0 | 0.000000000 | 0 | 145.000000 / 145.000000 | 1 / 1 |
| 3 | 0 | 0 | 0.000000119 | 0.000000238 | 0 | 0.000000000 | 0 | 200.000000 / 200.000000 | 1 / 1 |
| 4 | 0 | 0 | 0.000000119 | 0.000000119 | 0 | 0.000000000 | 0 | 200.000000 / 200.000000 | 1 / 1 |
| 5 | 0 | 0 | 0.000000119 | 0.000000238 | 0 | 0.000000000 | 0 | 200.000000 / 200.000000 | 1 / 1 |
| 6 | 0 | 0 | 0.000000119 | 0.000000238 | 0 | 0.000000000 | 0 | 200.000000 / 200.000000 | 1 / 1 |
| 7 | 0 | 0 | 0.000000179 | 0.000000238 | 0 | 0.000000000 | 0 | 200.000000 / 200.000000 | 1 / 1 |
| 8 | 0 | 0 | 0.000000179 | 0.000000477 | 0 | 0.000000000 | 0 | 200.000000 / 200.000000 | 1 / 1 |
| 9 | 0 | 0 | 0.000000119 | 0.000000477 | 0 | 0.000000000 | 0 | 200.000000 / 200.000000 | 1 / 1 |
| 10 | 0 | 0 | 0.000000179 | 0.000000477 | 0 | 0.000000000 | 0 | 200.000000 / 200.000000 | 1 / 1 |
| 11 | 0 | 0 | 0.000000149 | 0.000000477 | 0 | 0.000000000 | 0 | 200.000000 / 200.000000 | 1 / 1 |
| 12 | 0 | 0 | 0.000000119 | 0.000000238 | 0 | 0.000000000 | 0 | 200.000000 / 200.000000 | 1 / 1 |
| 13 | 0 | 0 | 0.000000119 | 0.000000238 | 0 | 0.000000000 | 0 | 200.000000 / 200.000000 | 1 / 1 |
| 14 | 0 | 0 | 0.000000119 | 0.000000358 | 0 | 0.000000000 | 0 | 200.000000 / 200.000000 | 1 / 1 |

**cartpole_p100 / `xla-rocm`** — 14 generation(s) compared in lockstep; first node-key divergence: none within 14 generations; first float exceedance: 4; wall time 5.969 s.

*Both legs share one PRNG from one seed and one graph, so a first key divergence beyond generation 1 is a selection near-tie flip caused by a sub-ulp float difference between the backends, not a kernel defect.*

| Generation | Node-key mismatches | Conn-key mismatches | Node attr max abs diff | Conn attr max abs diff | Attr NaN-mask mismatches | Fitness max abs diff | Fitness mask mismatches | Max fitness (numpy / accelerated) | Species (numpy / accelerated) |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 0 | 0 | 0.000000238 | 0.000000238 | 0 | 0.000000000 | 0 | 56.000000 / 56.000000 | 1 / 1 |
| 2 | 0 | 0 | 0.000000119 | 0.000000238 | 0 | 0.000000000 | 0 | 145.000000 / 145.000000 | 1 / 1 |
| 3 | 0 | 0 | 0.000000119 | 0.000000238 | 0 | 0.000000000 | 0 | 200.000000 / 200.000000 | 1 / 1 |
| 4 | 0 | 0 | 0.000000119 | 0.000000119 | 0 | 3.000000000 | 0 | 200.000000 / 200.000000 | 1 / 1 |
| 5 | 0 | 0 | 0.000000119 | 0.000000238 | 0 | 12.000000000 | 0 | 200.000000 / 200.000000 | 1 / 1 |
| 6 | 0 | 0 | 0.000000119 | 0.000000238 | 0 | 1.000000000 | 0 | 200.000000 / 200.000000 | 1 / 1 |
| 7 | 0 | 0 | 0.000000119 | 0.000000238 | 0 | 1.000000000 | 0 | 200.000000 / 200.000000 | 1 / 1 |
| 8 | 0 | 0 | 0.000000119 | 0.000000477 | 0 | 1.000000000 | 0 | 200.000000 / 200.000000 | 1 / 1 |
| 9 | 0 | 0 | 0.000000119 | 0.000000477 | 0 | 23.000000000 | 0 | 200.000000 / 200.000000 | 1 / 1 |
| 10 | 0 | 0 | 0.000000119 | 0.000000477 | 0 | 0.000000000 | 0 | 200.000000 / 200.000000 | 1 / 1 |
| 11 | 0 | 0 | 0.000000119 | 0.000000477 | 0 | 2.000000000 | 0 | 200.000000 / 200.000000 | 1 / 1 |
| 12 | 0 | 0 | 0.000000060 | 0.000000238 | 0 | 1.000000000 | 0 | 200.000000 / 200.000000 | 1 / 1 |
| 13 | 0 | 0 | 0.000000119 | 0.000000238 | 0 | 0.000000000 | 0 | 200.000000 / 200.000000 | 1 / 1 |
| 14 | 0 | 0 | 0.000000119 | 0.000000358 | 0 | 0.000000000 | 0 | 200.000000 / 200.000000 | 1 / 1 |

## Blocked rows

No row was blocked: every planned row is `ok`.

## Provenance of this block

This block is rendered from `benchmarks/results/gpu_rocm_xla_2026-11-06.json`: 18 record(s), 18 ok and 0 error, measured in the session 2026-11-06T07:11:39+00:00 – 2026-11-06T07:17:50+00:00 (UTC) at git `ddd7295afc5a` (clean — generated report artifacts are not counted).
`benchmarks/results/gpu_rocm_xla_2026-11-06.json` is a scratch record: it is git-ignored and **not** committed evidence, so the cells above describe this session only.
Nothing here is a statement about HEAD: every number belongs to that record and to the tree its fingerprint names.
<!-- END MEASURED (gpu_measure.py) -->

## Limitations

No legacy-GPU comparison exists: the frozen JAX baseline in `benchmarks/legacy_jax` is CPU-only by protocol, so an accelerator number here is informational and is never part of the numpy-only old-vs-new verdict in `RESULTS.md`; there is no "old side" for a GPU.
One device: a single `AMD Instinct MI300X VF` at one device index (`iree.runtime.get_driver("hip").query_available_devices()` reports one entry, `device_id=1`), so nothing here says how the numbers scale to a second GPU, and no multi-device or multi-stream path is exercised.
`xla` and `tvm` are not attempted: `xla` needs a PJRT C API plugin `.so` that nothing provides on this box, and `tvm` is untested in this repository.
Transfer cost is inside `ms/generation` and not separated from it: the host materialisation of each generation's result happens inside the timed `step()` call, and the HIP host/device traffic is part of that wall clock; this protocol does not report a transfer-only number, and unified-memory behaviour is not instrumented.
`peak RSS` is per process and is the driver's resident memory, not the device's, so the 1903.5–2064.3 MiB on the rocm rows says nothing about VRAM occupancy and no device-memory figure is recorded anywhere in this protocol.
No per-generation dispatch or op count is instrumented here — `PERF_NUMPY.md` is where executed-op counts live for the CPU interpreter — so an accelerator row gives wall clock and nothing about what the device was asked to do.
Timing is wall clock with a warmup, one seed and one committed repeat: per-row spread is the scratch repeat described in `## Provenance` (no median moved by more than 6.8 % between the two sessions), and neither run is a distribution over seeds.
The one-off compile is per static shape, so a shape change repays it: the 9.037–15.091 s `build_s` is excluded from every `ms/gen` sample but is real wall clock for a run that evaluates few generations, which is what the 15–504 generation break-evens in the verdict paragraph quantify.
The cross-check's later-generation divergence makes a long-horizon accelerator-vs-numpy *population* comparison a trajectory comparison, not an equality test: this record's first key divergence lands at generation 6 on `xor_p500`, 9 on `xor_p1000` and `xor3d_p500`, and transiently at 10 on `custom_pagie_p500`, so the fitness (and species-count) view is what to compare across many generations, and the key arrays are not.
`--trace 14` is a real cost: the six trajectory legs add 108.2 s of wall clock (11.473 s to 32.707 s per row), each re-paying a full build for both of its legs on top of the 14 generations.

## Reproduce

```bash
# 0. environment: the compiled-backend extra, plus a working ROCm runtime for the rocm rows
uv sync --extra iree
/opt/rocm/.info/version && rocm-smi --showproductname --showdriverversion   # 7.1.0 / MI300X VF / driver 6.16.6 here

# 1. numpy-free planning: targets, passthrough flags, the case matrix
python3 benchmarks/gpu_measure.py --list

# 2. one row, one fresh process (this is what `matrix` spawns per row)
uv run --extra iree python benchmarks/gpu_measure.py run \
    --case xor_p500 --target iree-rocm --warmup 1 --generations 3 --json $TMPDIR/row.json

# 3. the full matrix this report renders: 6 cases x 2 targets, strictly sequential
uv run --extra iree python benchmarks/gpu_measure.py matrix \
    --cases xor_p100,xor_p500,xor_p1000,xor3d_p500,custom_pagie_p500,cartpole_p100 \
    --targets numpy,iree-rocm \
    --warmup 3 --generations 20 --trace 14 --seed 42 \
    --json benchmarks/results/gpu_rocm_2026-11-05.json --out benchmarks/GPU.md \
    --report-title "TensorNEAT accelerator measurement — numpy reference vs iree rocm on one MI300X VF" \
    --report-note "One accelerator session at tree a5973c8 (IREE 3.13.0, evoxc v0.1.1 tag): six cases measured on the numpy reference and on iree rocm — the ROCm path layers, compiles and runs end to end for the first time, so no row here is blocked. These rows measure that tree only, before the GPU optimizer pass landing in src/tensorneat in parallel; they may improve afterwards."

# 4. re-render the measured block from the existing record: no measurement, no JSON rewrite
uv run --extra iree python benchmarks/gpu_measure.py render \
    --json benchmarks/results/gpu_rocm_2026-11-05.json --out benchmarks/GPU.md
```

Step 3 is the exact argv the record stores, with the recording worker's absolute interpreter path replaced by the portable `uv run --extra iree python`.
It is a 5 min 4 s job on this box (12 rows, of which the six trajectory legs are 108.2 s and the six `iree-compile`s of the rocm rows are 65.0 s), so run it in the background rather than under a short foreground timeout.
`matrix` exits 0 only when every planned row is `ok` and 1 as soon as any row is `error` (it prints a `BLOCKED: …` line per blocked row); this matrix exited 0 with 12/12 `ok`, which is the state the block below records.
`render` and `matrix --out` replace only the interior of this file's two markers (exactly one of each in the file) and exit 2 with a named error on a missing marker, an unreadable record, a wrong schema or a missing `records` list.
A single row is cheap on the numpy leg (0.282–0.863 s build) and pays the compile on the rocm leg (9.037–15.091 s), so budget a rocm row at its `iree-compile` plus its timed region, and a traced row at that plus `N × (accelerated + numpy generation)`.
The IREE runtime can print `nanobind: leaked …` lines to stderr when the interpreter exits; that is a binding teardown message emitted *after* the row is written, never a failure and never an `error` row.
The Python-free paths are `--list`, `--help` and nothing else: `run` and `matrix` need `numpy` and `tensorneat`, and the `iree-*` targets need the `iree` extra.

## What this does not change

`RESULTS.md` stays the generated numpy-only old-vs-new record, with its own verdict computed from `results/authoritative.json`; no accelerator row is added to it, and no number in it is touched by this protocol.
`BACKENDS.md` stays the compiled-backend CPU-leg record of its own session — the `iree` llvm-cpu numbers, which this matrix does not measure — and `PERF_NUMPY.md` stays the numpy-interpreter performance record; neither is re-rendered from here.
Accelerator rows are never wired into the harness verdict, and that is deliberate: `protocol.comparisons` pairs every old-side record with every new-side record, so an accelerator row in `results/` would enter the numpy-only verdict as if it were a numpy measurement and could flip a timing verdict that the CPU contract owns.
Keeping the accelerator protocol standalone (`benchmarks/gpu_measure.py`, its own `tensorneat-gpu/1` schema, its own record and report) keeps that contract intact: the harness verdict compares what it declares it compares, and an accelerated row is evidence about a different question.
