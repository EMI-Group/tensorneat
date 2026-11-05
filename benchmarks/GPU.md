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
| Date / session (UTC) | 2026-11-05T11:54:45+00:00 – 2026-11-05T11:59:49+00:00 |
| GPU model (gfx version) | AMD Instinct MI300X VF (gfx942) |
| HIP devices (IREE runtime query) | device_id=1 AMD Instinct MI300X VF |
| ROCm version | 7.1.0 |
| Driver version | 6.16.6 |
| IREE compiler / runtime | 3.13.0 / 3.13.0 |
| evoxc | 0.1.0 |
| numpy | 2.5.4 |
| Python | 3.12.3 |
| Platform | Linux-6.8.0-146-generic-x86_64-with-glibc2.39 |
| CPU model / count | INTEL(R) XEON(R) PLATINUM 8568Y+ / 20 logical |
| Memory | 241611 MiB |
| Threads | runtime defaults (nothing exported; inherited environment recorded) |
| Git commit | `a5973c89bebf` (clean — generated report artifacts are not counted) |
| Load average (1/5/15 min, document build) | 7.76 / 4.99 / 2.81 |

## Measured: numpy vs iree llvm-cpu vs iree rocm

One row per planned `(case, target)`; `n/a` marks a metric the row does not have (an error row has no timings at all). `build` is the one-off generation-graph compile and is excluded from every `ms/gen` sample; `peak RSS` is the process high-water mark of that single row, which is why each row runs in its own process.

| Case | Target | Backend | Status | setup (s) | build (s) | ms/gen median | ms/gen mean | ms/gen min–max | gens/s | peak RSS (MiB) | final max fitness | cross-check |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| xor_p100 | `numpy` | `numpy` | ok | 0.015 | 0.282 | 48.502 | 47.894 | 38.758 – 65.355 | 20.879 | 162.1 | -0.176966 | skipped (target is the numpy reference) |
| xor_p100 | `iree-rocm` | `iree` | ok | 1.234 | 9.037 | 30.575 | 30.431 | 28.434 – 32.771 | 32.861 | 1903.5 | -0.176966 | pass (gen 1) |
| xor_p500 | `numpy` | `numpy` | ok | 0.016 | 0.353 | 171.239 | 173.032 | 167.856 – 192.457 | 5.779 | 281.9 | -0.147786 | skipped (target is the numpy reference) |
| xor_p500 | `iree-rocm` | `iree` | ok | 1.330 | 9.751 | 49.534 | 49.426 | 47.568 – 56.867 | 20.232 | 1931.9 | -0.147786 | gen 1 pass; final-state drift (node_keys) |
| xor_p1000 | `numpy` | `numpy` | ok | 0.016 | 0.430 | 313.623 | 315.972 | 302.022 – 343.304 | 3.165 | 425.8 | -0.053339 | skipped (target is the numpy reference) |
| xor_p1000 | `iree-rocm` | `iree` | ok | 1.383 | 9.685 | 66.134 | 65.787 | 63.840 – 69.514 | 15.201 | 1954.5 | -0.052824 | gen 1 pass; final-state drift (node_keys) |
| xor3d_p500 | `numpy` | `numpy` | ok | 0.016 | 0.343 | 198.987 | 197.629 | 179.884 – 220.262 | 5.060 | 289.9 | -0.223466 | skipped (target is the numpy reference) |
| xor3d_p500 | `iree-rocm` | `iree` | ok | 1.384 | 10.166 | 49.503 | 49.679 | 48.062 – 56.097 | 20.129 | 1932.8 | -0.207055 | gen 1 pass; final-state drift (node_keys) |
| custom_pagie_p500 | `numpy` | `numpy` | ok | 0.017 | 0.863 | 1039.650 | 1040.777 | 906.630 – 1137.642 | 0.961 | 1310.0 | -0.032613 | skipped (target is the numpy reference) |
| custom_pagie_p500 | `iree-rocm` | `iree` | ok | 1.291 | 15.091 | 62.981 | 63.210 | 61.381 – 65.092 | 15.820 | 1978.7 | -0.032613 | gen 1 pass; final-state drift (node_keys) |
| cartpole_p100 | `numpy` | `numpy` | ok | 0.017 | 0.304 | 283.515 | 313.064 | 276.780 – 510.828 | 3.194 | 160.4 | 200.000000 | skipped (target is the numpy reference) |
| cartpole_p100 | `iree-rocm` | `iree` | ok | 1.638 | 11.316 | 179.644 | 193.456 | 165.156 – 257.777 | 5.169 | 2064.3 | 200.000000 | pass (gen 1) |

## Numerical cross-check

Each non-numpy row is compared against a fresh `backend="numpy"` run of the same case, seed and step count in the same process. Generation 1 must match (one seed, one graph, one PRNG): integer keys exactly, attributes and fitness to rtol 1e-05 / atol 1e-06 over finite entries with an identical NaN mask. The final state is a search-trajectory observation, not a pass/fail gate on its own: beyond generation 1 a float-reassociation difference can reorder a near-tied selection and legitimately diverge the keys. The table's `cross-check` column summarises both snapshots: generation 1 is the equality gate; a final-state-only difference renders `gen 1 pass; final-state drift (…)`.

| Case | Target | Generation 1 | Keys (gen 1) | Attr max abs diff (gen 1) | Fitness max abs diff | Mask mismatches | Final state | Keys (final) | Attr max abs diff (final) | Fitness max abs diff (final) | Mask mismatches (final) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| xor_p100 | `iree-rocm` | pass | node_keys exact; conn_keys exact | node_attrs 0.000000238; conn_attrs 0.000000238 | 0.000000060 | 0 | pass | node_keys exact; conn_keys exact | node_attrs 0.000000149; conn_attrs 0.000000238 | 0.000000045 | 0 |
| xor_p500 | `iree-rocm` | pass | node_keys exact; conn_keys exact | node_attrs 0.000000477; conn_attrs 0.000000294 | 0.000000060 | 0 | fail | node_keys 786 mismatch(es); conn_keys 3248 mismatch(es) | node_attrs 2.583666325; conn_attrs 6.162846565 | 0.252536833 | 4526 |
| xor_p1000 | `iree-rocm` | pass | node_keys exact; conn_keys exact | node_attrs 0.000000477; conn_attrs 0.000000298 | 0.000000089 | 0 | fail | node_keys 2268 mismatch(es); conn_keys 10185 mismatch(es) | node_attrs 4.307779312; conn_attrs 6.632864952 | 0.402970970 | 17050 |
| xor3d_p500 | `iree-rocm` | pass | node_keys exact; conn_keys exact | node_attrs 0.000000477; conn_attrs 0.000000298 | 0.000000060 | 0 | fail | node_keys 979 mismatch(es); conn_keys 4424 mismatch(es) | node_attrs 3.908831835; conn_attrs 4.325877190 | 0.172152549 | 6028 |
| custom_pagie_p500 | `iree-rocm` | pass | node_keys exact; conn_keys exact | node_attrs 0.000000477; conn_attrs 0.000000294 | 0.000000238 | 0 | fail | node_keys 237 mismatch(es); conn_keys 1458 mismatch(es) | node_attrs 3.406351328; conn_attrs 2.775112629 | 0.112810463 | 1896 |
| cartpole_p100 | `iree-rocm` | pass | node_keys exact; conn_keys exact | node_attrs 0.000000477; conn_attrs 0.000000294 | 0.000000000 | 0 | pass | node_keys exact; conn_keys exact | node_attrs 0.000000119; conn_attrs 0.000000238 | 0.000000000 | 0 |

## Per-generation trajectory

Every traced row was re-run after the timed region, with the peak-RSS sample already taken, as a lockstep pair: a fresh accelerated pipeline and a fresh `backend="numpy"` pipeline of the same case and seed, each paying its own build, compared per generation with the cross-check rules above. The trace touches neither the timings nor the recorded RSS.

**xor_p100 / `iree-rocm`** — 14 generation(s) compared in lockstep; first node-key divergence: none within 14 generations; first float exceedance: none within 14 generations; wall time 11.473 s.

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
| 13 | 0 | 0 | 0.000000119 | 0.000000119 | 0 | 0.000000060 | 0 | -0.237129 / -0.237129 | 6 / 6 |
| 14 | 0 | 0 | 0.000000119 | 0.000000119 | 0 | 0.000000060 | 0 | -0.233823 / -0.233823 | 6 / 6 |

**xor_p500 / `iree-rocm`** — 14 generation(s) compared in lockstep; first node-key divergence: 6; first float exceedance: 5; wall time 14.174 s.

*Both legs share one PRNG from one seed and one graph, so a first key divergence beyond generation 1 is a selection near-tie flip caused by a sub-ulp float difference between the backends, not a kernel defect.*

| Generation | Node-key mismatches | Conn-key mismatches | Node attr max abs diff | Conn attr max abs diff | Attr NaN-mask mismatches | Fitness max abs diff | Fitness mask mismatches | Max fitness (numpy / accelerated) | Species (numpy / accelerated) |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 0 | 0 | 0.000000477 | 0.000000294 | 0 | 0.000000060 | 0 | -0.250251 / -0.250251 | 1 / 1 |
| 2 | 0 | 0 | 0.000000119 | 0.000000238 | 0 | 0.000000060 | 0 | -0.250017 / -0.250017 | 6 / 6 |
| 3 | 0 | 0 | 0.000000149 | 0.000000238 | 0 | 0.000000060 | 0 | -0.250007 / -0.250007 | 16 / 16 |
| 4 | 0 | 0 | 0.000000127 | 0.000000238 | 0 | 0.000000060 | 0 | -0.250001 / -0.250001 | 20 / 20 |
| 5 | 0 | 0 | 0.402902961 | 0.003084075 | 0 | 0.004423261 | 0 | -0.249823 / -0.249823 | 20 / 20 |
| 6 | 31 | 66 | 0.508377314 | 1.976041079 | 7 | 0.005131155 | 0 | -0.248735 / -0.248735 | 20 / 20 |
| 7 | 159 | 439 | 1.245799303 | 1.014402628 | 444 | 0.014945000 | 1 | -0.247461 / -0.247461 | 20 / 20 |
| 8 | 237 | 845 | 1.494967699 | 4.651563644 | 843 | 0.082444668 | 4 | -0.243918 / -0.243918 | 20 / 20 |
| 9 | 425 | 1436 | 1.595991731 | 4.251160622 | 1282 | 0.100294918 | 10 | -0.241320 / -0.241320 | 20 / 20 |
| 10 | 411 | 1302 | 1.772526622 | 4.653420925 | 1056 | 0.033620209 | 8 | -0.240657 / -0.240657 | 20 / 20 |
| 11 | 547 | 1944 | 2.092477322 | 4.572396278 | 1585 | 0.096894860 | 13 | -0.237867 / -0.237867 | 20 / 20 |
| 12 | 558 | 1945 | 2.593549967 | 3.371931553 | 1761 | 0.107947081 | 15 | -0.233631 / -0.233631 | 20 / 20 |
| 13 | 533 | 1782 | 4.043320179 | 4.658003807 | 1340 | 0.130599111 | 14 | -0.228627 / -0.228627 | 20 / 20 |
| 14 | 486 | 1850 | 1.724698305 | 4.736943245 | 1340 | 0.090268880 | 9 | -0.224597 / -0.224597 | 20 / 20 |

**xor_p1000 / `iree-rocm`** — 14 generation(s) compared in lockstep; first node-key divergence: 9; first float exceedance: 8; wall time 16.285 s.

*Both legs share one PRNG from one seed and one graph, so a first key divergence beyond generation 1 is a selection near-tie flip caused by a sub-ulp float difference between the backends, not a kernel defect.*

| Generation | Node-key mismatches | Conn-key mismatches | Node attr max abs diff | Conn attr max abs diff | Attr NaN-mask mismatches | Fitness max abs diff | Fitness mask mismatches | Max fitness (numpy / accelerated) | Species (numpy / accelerated) |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 0 | 0 | 0.000000477 | 0.000000298 | 0 | 0.000000089 | 0 | -0.250251 / -0.250251 | 2 / 2 |
| 2 | 0 | 0 | 0.000000238 | 0.000000238 | 0 | 0.000000060 | 0 | -0.250060 / -0.250060 | 12 / 12 |
| 3 | 0 | 0 | 0.000000238 | 0.000000238 | 0 | 0.000000060 | 0 | -0.250001 / -0.250001 | 20 / 20 |
| 4 | 0 | 0 | 0.000000164 | 0.000000238 | 0 | 0.000000060 | 0 | -0.249913 / -0.249913 | 20 / 20 |
| 5 | 0 | 0 | 0.000000119 | 0.000000238 | 0 | 0.000000060 | 0 | -0.249371 / -0.249371 | 20 / 20 |
| 6 | 0 | 0 | 0.000000134 | 0.000000238 | 0 | 0.000000060 | 0 | -0.248355 / -0.248355 | 20 / 20 |
| 7 | 0 | 0 | 0.000000238 | 0.000000238 | 0 | 0.000000089 | 0 | -0.246130 / -0.246130 | 20 / 20 |
| 8 | 0 | 0 | 0.290338397 | 1.006935835 | 0 | 0.003362447 | 0 | -0.235605 / -0.235605 | 20 / 20 |
| 9 | 18 | 92 | 3.740734100 | 2.835820675 | 66 | 0.010366708 | 1 | -0.232334 / -0.232334 | 20 / 20 |
| 10 | 243 | 1359 | 3.740734100 | 4.559720993 | 1428 | 0.087913960 | 1 | -0.218614 / -0.218614 | 20 / 20 |
| 11 | 613 | 2869 | 3.159790277 | 3.706911087 | 1564 | 0.102426678 | 9 | -0.212860 / -0.212860 | 20 / 20 |
| 12 | 1047 | 4615 | 3.743532181 | 3.842892885 | 3566 | 0.184486270 | 20 | -0.206379 / -0.206379 | 20 / 20 |
| 13 | 1346 | 6073 | 3.743532181 | 4.476229668 | 4761 | 0.180017292 | 41 | -0.199304 / -0.199304 | 20 / 20 |
| 14 | 1713 | 7687 | 3.439329386 | 4.645004272 | 5624 | 0.211708605 | 39 | -0.187691 / -0.187691 | 20 / 20 |

**xor3d_p500 / `iree-rocm`** — 14 generation(s) compared in lockstep; first node-key divergence: 9; first float exceedance: 8; wall time 15.608 s.

*Both legs share one PRNG from one seed and one graph, so a first key divergence beyond generation 1 is a selection near-tie flip caused by a sub-ulp float difference between the backends, not a kernel defect.*

| Generation | Node-key mismatches | Conn-key mismatches | Node attr max abs diff | Conn attr max abs diff | Attr NaN-mask mismatches | Fitness max abs diff | Fitness mask mismatches | Max fitness (numpy / accelerated) | Species (numpy / accelerated) |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 0 | 0 | 0.000000477 | 0.000000298 | 0 | 0.000000060 | 0 | -0.251171 / -0.251171 | 1 / 1 |
| 2 | 0 | 0 | 0.000000179 | 0.000000358 | 0 | 0.000000060 | 0 | -0.250194 / -0.250194 | 1 / 1 |
| 3 | 0 | 0 | 0.000000119 | 0.000000328 | 0 | 0.000000060 | 0 | -0.250011 / -0.250011 | 3 / 3 |
| 4 | 0 | 0 | 0.000000238 | 0.000000328 | 0 | 0.000000089 | 0 | -0.250004 / -0.250004 | 3 / 3 |
| 5 | 0 | 0 | 0.000000238 | 0.000000328 | 0 | 0.000000060 | 0 | -0.250001 / -0.250001 | 20 / 20 |
| 6 | 0 | 0 | 0.000000238 | 0.000000328 | 0 | 0.000000060 | 0 | -0.250000 / -0.250000 | 20 / 20 |
| 7 | 0 | 0 | 0.000000119 | 0.000000328 | 0 | 0.000000030 | 0 | -0.250000 / -0.250000 | 20 / 20 |
| 8 | 0 | 0 | 0.272031397 | 0.057659566 | 0 | 0.000539750 | 0 | -0.249999 / -0.249999 | 20 / 20 |
| 9 | 85 | 250 | 2.416074753 | 1.378281713 | 103 | 0.002465039 | 0 | -0.249980 / -0.249980 | 20 / 20 |
| 10 | 450 | 1679 | 2.217846394 | 3.445238590 | 1607 | 0.128810614 | 13 | -0.249943 / -0.249943 | 20 / 20 |
| 11 | 702 | 2619 | 3.213420391 | 3.445238590 | 2385 | 0.082215965 | 22 | -0.249839 / -0.249839 | 20 / 20 |
| 12 | 788 | 2957 | 3.996948242 | 3.178690195 | 2214 | 0.112219185 | 21 | -0.249462 / -0.249462 | 20 / 20 |
| 13 | 734 | 2824 | 3.996948242 | 4.032338619 | 2145 | 0.098949313 | 29 | -0.249007 / -0.249007 | 20 / 20 |
| 14 | 884 | 3200 | 4.208443165 | 3.956594229 | 2590 | 0.060001105 | 17 | -0.247870 / -0.247267 | 20 / 20 |

**custom_pagie_p500 / `iree-rocm`** — 14 generation(s) compared in lockstep; first node-key divergence: 10; first float exceedance: 7; wall time 32.707 s.

*Both legs share one PRNG from one seed and one graph, so a first key divergence beyond generation 1 is a selection near-tie flip caused by a sub-ulp float difference between the backends, not a kernel defect.*

| Generation | Node-key mismatches | Conn-key mismatches | Node attr max abs diff | Conn attr max abs diff | Attr NaN-mask mismatches | Fitness max abs diff | Fitness mask mismatches | Max fitness (numpy / accelerated) | Species (numpy / accelerated) |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 0 | 0 | 0.000000477 | 0.000000294 | 0 | 0.000000238 | 0 | -0.046480 / -0.046480 | 1 / 1 |
| 2 | 0 | 0 | 0.000000134 | 0.000000238 | 0 | 0.000000063 | 0 | -0.046061 / -0.046061 | 3 / 3 |
| 3 | 0 | 0 | 0.000000238 | 0.000000238 | 0 | 0.000000089 | 0 | -0.046049 / -0.046049 | 3 / 3 |
| 4 | 0 | 0 | 0.000000179 | 0.000000238 | 0 | 0.000000119 | 0 | -0.046049 / -0.046049 | 4 / 4 |
| 5 | 0 | 0 | 0.000000238 | 0.000000238 | 0 | 0.000000298 | 0 | -0.046031 / -0.046031 | 5 / 5 |
| 6 | 0 | 0 | 0.000000238 | 0.000000238 | 0 | 0.000000119 | 0 | -0.046018 / -0.046018 | 5 / 5 |
| 7 | 0 | 0 | 0.306329310 | 0.008525796 | 0 | 0.000130322 | 0 | -0.045835 / -0.045835 | 6 / 6 |
| 8 | 0 | 0 | 0.259954870 | 0.010306150 | 0 | 0.000155196 | 0 | -0.045799 / -0.045799 | 6 / 6 |
| 9 | 0 | 0 | 0.259954810 | 0.010306165 | 0 | 0.000231951 | 0 | -0.045137 / -0.045137 | 6 / 6 |
| 10 | 1 | 4 | 1.414301872 | 0.234590173 | 6 | 0.001782283 | 0 | -0.044979 / -0.044979 | 6 / 6 |
| 11 | 0 | 0 | 0.259954810 | 0.009333879 | 0 | 0.000296637 | 0 | -0.044345 / -0.044345 | 7 / 7 |
| 12 | 0 | 0 | 0.259954840 | 0.009333879 | 0 | 0.000089876 | 0 | -0.043920 / -0.043920 | 8 / 8 |
| 13 | 0 | 0 | 0.259954870 | 0.009333879 | 0 | 0.000088107 | 0 | -0.042594 / -0.042594 | 8 / 8 |
| 14 | 0 | 0 | 0.380616546 | 0.009333849 | 0 | 0.000011727 | 0 | -0.042274 / -0.042274 | 8 / 8 |

**cartpole_p100 / `iree-rocm`** — 14 generation(s) compared in lockstep; first node-key divergence: none within 14 generations; first float exceedance: none within 14 generations; wall time 17.908 s.

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

## Blocked rows

No row was blocked: every planned row is `ok`.

## Provenance of this block

This block is rendered from `benchmarks/results/gpu_rocm_2026-11-05.json`: 12 record(s), 12 ok and 0 error, measured in the session 2026-11-05T11:54:45+00:00 – 2026-11-05T11:59:49+00:00 (UTC) at git `a5973c89bebf` (clean — generated report artifacts are not counted).
`benchmarks/results/gpu_rocm_2026-11-05.json` is committed evidence (it is in the re-include list of `benchmarks/results/.gitignore`), so every cell above is traceable to the committed record.
Nothing here is a statement about HEAD: every number belongs to that record and to the tree its fingerprint names.
One accelerator session at tree a5973c8 (IREE 3.13.0, evoxc v0.1.1 tag): six cases measured on the numpy reference and on iree rocm — the ROCm path layers, compiles and runs end to end for the first time, so no row here is blocked. These rows measure that tree only, before the GPU optimizer pass landing in src/tensorneat in parallel; they may improve afterwards.
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
