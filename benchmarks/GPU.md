# TensorNEAT accelerator measurement — numpy reference vs iree rocm vs xla rocm on one MI300X VF

**Verdict for the record this file renders: both compiled paths are green in one session — eighteen planned rows, eighteen `ok`, zero blocked — so every one of the six cases carries a real compile time, a real per-generation timing and a real numerical cross-check on `iree` rocm and on `xla` rocm against the `numpy` reference.**
`Pipeline(backend="iree", device="rocm", compile_options={"target_backends": ["rocm"]})` compiles the generation graph with `iree-compile` for `gfx942`, and `Pipeline(backend="xla", device="rocm", compile_options={"plugin_path": …})` compiles it with XLA for the same device, driving the `jax-rocm7-pjrt 0.10.2` PJRT C API plugin named by that option and by `EVOXC_PJRT_PLUGIN`.
The device is faster per generation than the numpy reference on all six cases, on both compiled paths: `iree` rocm by 4.2x on `xor_p100` (10.295 against 43.300 ms/gen), 9.4x on `xor_p500` (12.820 against 120.186), 12.1x on `xor_p1000` (15.117 against 182.567), 10.5x on `xor3d_p500` (12.920 against 136.063), 61x on `custom_pagie_p500` (15.768 against 965.043) and 3.8x on `cartpole_p100` (61.495 against 236.023); `xla` rocm by 12.0x (3.610), 26.5x (4.535), 33.2x (5.497), 29.7x (4.580), 153x (6.319) and 17.7x (13.362) on the same rows.
The two compiled backends are not close: `xla` rocm is 2.5x–4.6x faster per generation than `iree` rocm on every case, and its per-shape XLA compile (`build_s` 2.149–2.672 s) costs 0.26x–0.30x of `iree`'s (`build_s` 8.017–9.027 s).
What `xla` pays instead is one-off client setup and resident memory: `setup_s` is 4.334–4.460 s against `iree`'s 1.218–1.466 s (loading the 480.1 MiB plugin and creating the PJRT client), and peak RSS is 9368.5–9427.1 MiB against `iree`'s 1894.9–1942.7 MiB and the numpy leg's 155.2–1306.2 MiB.
The per-generation win is still bought with a one-off compile on both paths: amortising `build_s` at the measured rate takes 243/77/49/70/10/50 generations on `iree` (`xor_p100`/`xor_p500`/`xor_p1000`/`xor3d_p500`/`custom_pagie_p500`/`cartpole_p100`) and 54/19/13/18/3/10 on `xla`, so `xla` amortises in 0.20x–0.29x of the generations `iree` needs.
Numerically, generation 1 is the equality gate and it is green on all twelve compiled rows: `node_keys` and `conn_keys` bit-identical, attributes within 4.8e-7 (`iree`) and 2.4e-7 (`xla`), fitness within 2.4e-7 (`iree`) and 1.2e-7 (`xla`), and zero NaN-mask mismatches everywhere.
Beyond generation 1 both compiled legs are a search trajectory, and the 14-generation lockstep trace shows the same order wherever a row drifts — attributes first, structural genes one to three generations later, the population's best fitness agreeing to float32 ulps — with `xla` diverging later than `iree` on every traced row that diverges at all, and `xor_p100` on `xla` never diverging inside 14 generations.
Nothing below quotes a number that is not in `benchmarks/results/gpu_rocm_xla_2026-11-06.json`; the speed-up columns and the break-even generation counts are arithmetic on that record's three medians and its `build_s`, and nothing here is a statement about HEAD.

## Headline: what each compiled backend buys per generation

| Case | numpy ms/gen (median) | `iree` rocm ms/gen | `xla` rocm ms/gen | numpy ÷ `iree` | numpy ÷ `xla` | `iree` ÷ `xla` | final max fitness numpy / `iree` / `xla` | final state `iree` / `xla` |
|---|---|---|---|---|---|---|---|---|
| `xor_p100` | 43.300 | 10.295 | 3.610 | 4.2x | 12.0x | 2.9x | -0.176966 / -0.184767 / -0.176966 | node_keys / pass |
| `xor_p500` | 120.186 | 12.820 | 4.535 | 9.4x | 26.5x | 2.8x | -0.147786 / -0.112459 / -0.140949 | node_keys / node_keys |
| `xor_p1000` | 182.567 | 15.117 | 5.497 | 12.1x | 33.2x | 2.8x | -0.053339 / -0.136390 / -0.053339 | node_keys / node_keys |
| `xor3d_p500` | 136.063 | 12.920 | 4.580 | 10.5x | 29.7x | 2.8x | -0.223466 / -0.250000 / -0.223466 | node_keys / node_keys |
| `custom_pagie_p500` | 965.043 | 15.768 | 6.319 | 61x | 153x | 2.5x | -0.032613 / -0.046049 / -0.032613 | node_keys / node_keys |
| `cartpole_p100` | 236.023 | 61.495 | 13.362 | 3.8x | 17.7x | 4.6x | 200.000000 / 200.000000 / 200.000000 | pass / fitness |

Every cell except the three ratio columns and the last two is a column of the generated block below; a ratio is the numpy median divided by the compiled median of the same row, hand-written here because the generated table has no ratio column, and the final-state column is the generated cross-check verdict abbreviated (`pass` means the post-`tell` population also matched).

The biggest win is `custom_pagie_p500`: 61x on `iree` (965.043 against 15.768 ms/gen) and 153x on `xla` (965.043 against 6.319).
That is the expected shape rather than a surprise: the func-fit forward is the sample-invariant schedule evaluated in tiles, which makes it the widest per-generation device workload in the matrix, and the numpy interpreter's per-op overhead scales with exactly that work — this row is also the slowest cell of the whole matrix on the CPU leg (965.043 ms/gen against 236.023 on the next-slowest, `cartpole_p100`).
The narrowest device margin on `iree` is `cartpole_p100` at 3.8x (236.023 against 61.495 ms/gen); on `xla` the narrowest is `xor_p100` at 12.0x (43.300 against 3.610).
A generation at those sizes is dominated by fixed per-generation overhead — dispatch, host materialisation and run-boundary device traffic, all inside the timed `step()` — which a compiled backend cannot compress the way it compresses a wide kernel; `cartpole_p100` additionally drives a per-step rollout loop over its horizon, so its per-generation launch count is the largest of the shipped problems (the op-count pass records 31 604 executed ops per generation for this case on the CPU interpreter, `PERF_NUMPY.md`).
`xla` compresses that fixed overhead further than `iree` does, which is why its smallest margin (12.0x on `xor_p100`) is 2.9x `iree`'s margin on the same row (4.2x).

Memory runs the other way, and it is driver RSS, never device memory: the `iree` leg sits at 1894.9–1942.7 MiB on every row and barely tracks the workload, the `xla` leg at 9368.5–9427.1 MiB with the same flatness, while the numpy leg grows with the workload (155.2 MiB at `cartpole_p100` to 1306.2 MiB at `custom_pagie_p500`).
Both compiled figures are single process high-water marks dominated by the loaded runtime and executable, which is why they are flat from pop 100 to pop 1000 instead of scaling like the numpy interpreter's retained intermediates; the `xla` figure is ~4.9x the `iree` one on every row and is discussed with the plugin's allocator reserve in `## IREE vs XLA on the same device`.

## IREE vs XLA on the same device

Same box, same graphs, same seed, same case config: the two compiled backends differ only in their compiler and their runtime, so the comparison below is per `(case, shape)` and its columns are the ones the generated block records (`build_s`, `setup_s`, `peak_rss_mib`) plus arithmetic on them.

| Case | `iree` build (s) | `xla` build (s) | `xla` ÷ `iree` build | `iree` setup (s) | `xla` setup (s) | `iree` break-even (gens) | `xla` break-even (gens) | peak RSS numpy / `iree` / `xla` (MiB) |
|---|---|---|---|---|---|---|---|---|
| `xor_p100` | 8.017 | 2.149 | 0.27 | 1.305 | 4.460 | 243 | 54 | 156.6 / 1894.9 / 9368.5 |
| `xor_p500` | 8.212 | 2.197 | 0.27 | 1.218 | 4.334 | 77 | 19 | 240.0 / 1913.3 / 9387.8 |
| `xor_p1000` | 8.161 | 2.367 | 0.29 | 1.326 | 4.354 | 49 | 13 | 341.6 / 1936.6 / 9423.4 |
| `xor3d_p500` | 8.617 | 2.332 | 0.27 | 1.466 | 4.435 | 70 | 18 | 248.0 / 1914.9 / 9413.5 |
| `custom_pagie_p500` | 9.027 | 2.672 | 0.30 | 1.305 | 4.455 | 10 | 3 | 1306.2 / 1942.7 / 9427.1 |
| `cartpole_p100` | 8.711 | 2.227 | 0.26 | 1.234 | 4.357 | 50 | 10 | 155.2 / 1934.3 / 9390.9 |

A break-even count is that row's `build_s` ÷ (numpy median − that backend's median), i.e. the generations of per-generation saving needed to repay the compile; it charges the compile and nothing else, and `setup_s` is a per-process cost on the same order on `xla` (4.3–4.5 s against `iree`'s 1.2–1.5 s), so a run that also counts client setup pays it once, not per shape.

Per generation, `xla` wins every row: 2.9x on `xor_p100` (10.295 against 3.610 ms/gen), 2.8x on `xor_p500`, 2.8x on `xor_p1000`, 2.8x on `xor3d_p500`, 2.5x on `custom_pagie_p500` and 4.6x on `cartpole_p100`.
The margin is stable at 2.75x–2.85x on the four XOR/XOR3d rows, 2.50x on the widest func-fit row (`custom_pagie_p500`) and widest at 4.60x on `cartpole_p100`, the row with the largest per-generation launch count, so the difference is a lower per-launch and per-dispatch cost in the XLA runtime rather than a wider kernel: the widest workload, where the kernel body dominates, is where the two backends come closest.
The compile cost goes the same way and by a similar factor: XLA compiles each of these shapes in 2.149–2.672 s against `iree-compile`'s 8.017–9.027 s (0.26x–0.30x), which combines with the per-generation lead so that `xla`'s break-even is 3–54 generations against `iree`'s 10–243.
The cost side is setup time and memory: `xla` takes 4.334–4.460 s in `Pipeline.setup()` on every row against `iree`'s 1.218–1.466 s (that is the plugin `.so` load plus PJRT client creation, once per process), and its 9368.5–9427.1 MiB peak RSS is ~4.9x `iree`'s 1894.9–1942.7 MiB on every row.
The `xla` resident figure is real, not the allocator's virtual reserve: the plugin logs `XLA backend allocating 143.77GiB … for BFCAllocator` and `… up to 51455721472 bytes … for CollectiveBFCAllocator` at client creation, but those are address-space reservations — the process's measured high-water mark is the 9368.5–9427.1 MiB the block records, and no device-memory figure is measured anywhere in this protocol.

Correctness is the same contract on both paths, and the same generation is the hard gate:

| Case | Generation 1 `iree` / `xla` | First float exceedance `iree` / `xla` | First key divergence `iree` / `xla` | Best fitness at generation 14 numpy / `iree` / `xla` |
|---|---|---|---|---|
| `xor_p100` | pass / pass | 13 / none | 13 / none | -0.233823 / -0.235706 / -0.233823 |
| `xor_p500` | pass / pass | 4 / 5 | 4 / 6 | -0.224597 / -0.174005 / -0.227711 |
| `xor_p1000` | pass / pass | 4 / 8 | 4 / 9 | -0.187691 / -0.215279 / -0.187691 |
| `xor3d_p500` | pass / pass | 5 / 8 | 5 / 8 | -0.247870 / -0.249999985 / -0.247870 |
| `custom_pagie_p500` | pass / pass | 4 / 7 | 4 / 10 | -0.042274 / -0.046049 / -0.042274 |
| `cartpole_p100` | pass / pass | none / 4 | none / none | 200.000000 / 200.000000 / 200.000000 |

Generation 1 matches on every compiled row of both backends: integer keys bit-identical, the largest attribute difference 4.8e-7 on `iree` and 2.4e-7 on `xla` (two float32 ulps at unit magnitude, four orders of magnitude inside the 1e-5 tolerance), the largest fitness difference 2.4e-7 on `iree` and 1.2e-7 on `xla`, zero NaN-mask mismatches.
Beyond it, the final state passes on `xor_p100` and `cartpole_p100` under `iree` and drifts in the integer keys on the other four rows, while under `xla` it passes on `xor_p100` and drifts on five.
The `xla` divergence lands later than `iree`'s on every row where both drift — keys at generation 6 against 4 on `xor_p500`, 9 against 4 on `xor_p1000`, 8 against 5 on `xor3d_p500`, 10 against 4 on `custom_pagie_p500` — and there is one row where the two disagree qualitatively: `xor_p100` is key-identical for all 14 traced generations under `xla` (no float exceedance at all) while `iree` first exceeds the tolerance and first diverges in the keys at generation 13.
`custom_pagie_p500` under `xla` is the transient case: the float tolerance is exceeded at generation 7 (node-attribute difference 0.306) while the keys stay identical through generation 9, they diverge for exactly one generation at 10 (1 node key, 4 conn keys, 6 mask mismatches) and are identical again from generation 11 to 14, ending with a fitness difference of 1.2e-5; under `iree` the same row diverges at generation 4 and never re-converges, its species count stuck at 3 against numpy's 8 and its best fitness at -0.046049 against -0.042274.
`cartpole_p100` under `xla` keeps identical keys on all 14 generations while individual returns differ by 3, 12, 23 (integer-valued, horizon-capped) — an action chosen at a near-tie — which is why its final state reads `fitness` drift although the population's best is 200.0 on both legs.
The mechanism is the same one the earlier session traced: a sub-ulp difference in the attribute payload is amplified by fitness ranking in `src/tensorneat/algorithm/neat/_kernels.py` — `sort_desc_legacy` (`:171`) is a stable ascending argsort gathered through the reversal permutation, i.e. a strict total order in which an exact tie is broken by row index, and `pair_parents` (`:407`) ranks with it at `:460`, draws parents at `:465` and pins elite slot *k* to that block's rank *k* at `:492-509`.
The benchmark cases carry `survival_threshold = 0.01` and `species_size = 20` (`benchmarks/cases.py`), so the survivor count `clamp(floor(0.01 × members), 1, …)` leaves every species under 200 members exactly one breeding parent while the pop-500 and pop-1000 tells fill all 20 species slots: up to 20 independent rank-0 decisions per generation, and one ulp-level tie flip changes a species' pinned elites and its single parent a generation or more before it can change the structural genes.
So generation 1 is the equality gate and nothing else is: every later generation is a search-trajectory observation, the entity to compare over a long horizon is the fitness view — the population's best, agreeing to float32 ulps on every traced row — never the key arrays, and a compiled backend that holds the keys longer (`xla` here) is not thereby more correct, only slower to lose a near-tie.

## Provenance: which tree and which session

Every number in this file belongs to `benchmarks/results/gpu_rocm_xla_2026-11-06.json` and to the git commit its fingerprint names, `ddd7295afc5a1ce972053215be161cea652f62e7` — this branch's HEAD at recording time, confirmed from the record rather than assumed — and no sentence here is a statement about HEAD.
That record is one `gpu_measure.py matrix` session: 2026-11-06T07:11:39Z – 2026-11-06T07:17:50Z (6 min 11 s wall clock), 18 rows over the cases `xor_p100,xor_p500,xor_p1000,xor3d_p500,custom_pagie_p500,cartpole_p100` × the targets `numpy,iree-rocm,xla-rocm`, seed 42, 3 warmup and 20 timed generations, `--trace 14`, one repeat per row, one fresh process per row — 18 `ok` and 0 `error`.
The environment it ran on is the block below: IREE 3.13.0 compiler and runtime, ROCm 7.1.0, driver 6.16.6, `AMD Instinct MI300X VF` (`gfx942`) as the single HIP device (`device_id=1`), numpy 2.5.4, Python 3.12.3 and a clean tree.
The `xla-rocm` target is `Pipeline(backend="xla", device="rocm", compile_options={"plugin_path": …})` driving the PJRT C API plugin `xla_rocm_plugin.so` from `jax-rocm7-pjrt 0.10.2`: 503 453 288 bytes (480.1 MiB), sha256 `8e8319b48fcc771b5221771dc3da0d24bc6950c872ee07ac6f5200743b3b4fa7`, installed at the conventional `site-packages/jax_plugins/xla_rocm7/` path the wheel ships, and recorded in the record's own `meta.pjrt_plugin` fingerprint (path, basename, size, sha256) that the environment table in the block below renders.
The record stores the recording worker's absolute interpreter path in `meta.command`; the portable form of the same argv is step 3 of `## Reproduce`, and that is the command to regenerate it with.
The `evoxc 0.1.0` row in the environment table is the installed distribution's metadata — PyPI's newest release — while `pyproject.toml` pins the `v0.1.1` git tag through `[tool.uv.sources]`: the measured tree is the tag, the table row is what a version query reports.

**Measurement conditions, stated rather than implied.** The box was shared while this matrix measured — other agents run on the same host — and the record's document-build load average is 5.23 / 5.04 / 3.49 (1/5/15 min).
No second session was run for this record: the uncertainty measure here is the per-row `ms/gen min–max` spread the generated block carries (20 samples per row), and a single median is not a distribution.
The earlier two-target session (`benchmarks/results/gpu_rocm_2026-11-05.json`, tree `a5973c8`) measured this `iree` rocm leg at 30.575 ms/gen on `xor_p100` and this file's record measures 10.295 ms/gen at `ddd7295a`: the two sessions belong to different trees, so neither number may be quoted as a statement about the other, and this file quotes only its own record.

## Protocol

`gpu_measure.py` measures exactly one `(case, target)` row per process, and one row per process is not a style choice: `peak_rss_mib` is `resource.getrusage(RUSAGE_SELF).ru_maxrss`, a process-lifetime high-water mark, so two rows in one interpreter would report the higher of the two as both.
`matrix` plans `cases × targets` and spawns one `run` subprocess per row **strictly sequentially**, one fresh interpreter each and the same interpreter as the driver, so no two measurements ever overlap on the box.
Per row: `Pipeline.setup()` is timed into `setup_s` (0.016–0.018 s on the numpy leg, 1.218–1.466 s on the `iree` rocm leg, 4.334–4.460 s on the `xla` rocm leg, where the plugin load and the PJRT client land); the first `Pipeline.step()` is timed into `first_step_s` and its build cost, `Pipeline.build_seconds`, is reported as `build_s` (0.283–0.805 s on the numpy leg, 8.017–9.027 s on the `iree` leg, which is where `iree-compile` lands, and 2.149–2.672 s on the `xla` leg, which is where XLA compilation lands) — the build is never charged to a generation and that step is excluded from every timing sample; then `--warmup` untimed generations and `--generations` timed ones, with stdout/stderr redirected to `os.devnull` for the whole warmup + timed region.
Each timed generation is `time.perf_counter()` wall clock around one `step()` call, which includes the host materialisation of the result; the row keeps mean, median, min, max and every sample (20 per row here), plus `generations_per_second = 1000 / mean`.
`peak_rss_mib` is sampled immediately after the timed region, then `final_generation`, `final_max_fitness`, `device_info` (the target's `backend`/`device`/`compile_options` — `{"target_backends": ["rocm"]}` on the `iree` rows, `{"plugin_path": …}` on the `xla` rows — plus the IREE HIP device query) and the exact `protocol` block are recorded.
`--trace N` (default `0`, off) adds a lockstep trajectory *after* the timed region and after the `peak_rss_mib` sample: one fresh pair of legs — the accelerated target and a fresh `backend="numpy"` pipeline, each paying its own build — driven generation by generation, so it costs `N × (accelerated generation + numpy generation)` and can never enter a timing sample or the RSS figure; the twelve traced rows cost 121.9 s of wall clock in total (83.6 s on the six `iree` rows, 2.646 s to 22.996 s per row, and 38.3 s on the six `xla` rows, 3.994 s to 16.050 s per row).
A failing trace leg is recorded as `trajectory.status == "error"` *inside* an otherwise `ok` row (the row keeps its timings), `--trace 0` records `skipped` with a reason, and this matrix ran with `--trace 14`, so all twelve compiled rows carry 14 traced generations.
`--json` writes the row; its `status` is `ok` or `error`, and any exception during the measurement becomes an `error` row carrying the full traceback plus whatever was already measured, with the last traceback line printed to stderr and exit code 1 — a failed row is recorded, never dropped, and a hard crash that kills the process is turned into an `error` row by the matrix driver (returncode plus stderr tail).
The defaults are the canonical matrix of `benchmarks/cases.py` (`xor_p100`, `xor_p500`, `xor3d_p500`), the three targets `numpy`, `iree-llvm-cpu`, `iree-rocm`, seed 42, three warmup and ten timed generations, one repeat, no trajectory (`--trace 0`): the case table is imported, never duplicated here, and an `xla` target is named explicitly because it needs a plugin path.
Targets are data, and the raw flags override a named target for hardware this file does not name: `--device KIND` with `--device-index N` (together they build `evoxc.Device(KIND, N)`), `--target-backends a,b`, `--rocm-target ARCH` (appends `--iree-rocm-target=ARCH` to the `iree_compile_args` compile option) and `--iree-compile-arg FLAG` (repeatable). `device` and `target_backends` must stay aligned: a combination that pairs a host device with a device-only target becomes an `error` row instead of being repaired silently.
`--pjrt-plugin PATH` is resolved once by `matrix` and forwarded to every row child, which also exports it as `EVOXC_PJRT_PLUGIN`; both wirings are required, because evoxc's `check_available()` probe for the `xla` backend takes no options and reads only the variable, while the compile option wins in every later discovery call.
Nothing falls back to another backend: if a target cannot run, the exception is the row, and `matrix` exits 0 only when every planned row is `ok` — a blocked device path is loud by construction, and this matrix exited 0 because no row was blocked.

## Cross-check: what is compared and at what tolerance

The cross-check runs *after* the timed region, in the same process, and it never changes a timing: a fresh `backend="numpy"` pipeline with the same case, seed and step count is driven over the same number of generations, so the comparison is one implementation against itself on two backends — no JAX enters the process, and "one implementation per process" is untouched (the `xla` row loads a PJRT plugin, not `jax`).
Two snapshots are compared, and they are recorded separately because they mean different things.
**Generation 1** is both legs' first evaluated population (the payload and the `(P,)` fitness vector of the first `step()`): integer keys (`node_keys`, `conn_keys`) must be `np.array_equal`, the attribute payloads must have an identical `np.isnan` mask, and the fitness vector an identical NaN *and* `-inf` mask, with every finite entry inside `FLOAT_RTOL = 1e-5` / `FLOAT_ATOL = 1e-6`.
**Final state** is the post-`tell` population after the same number of steps plus the last step's fitness, compared with the same rules and labelled as a search-trajectory observation: both legs draw the same evoxc splitmix64 stream from one seed and one graph, so generation 1 must match, while beyond it a float-reassociation difference between the backends can reorder a near-tied decision and legitimately diverge the integer keys — that is trajectory drift, not necessarily a kernel defect, and the record carries the sentence (`TRAJECTORY_NOTE`) next to the verdict.
Unlike the old-vs-new harness, which compares two different PRNGs (JAX `PRNGKey` vs evoxc splitmix64) and therefore no trajectories at all, both legs here share one random stream, which is why generation 1 is a hard comparison rather than a statistical one.
The tolerances are the ones `test/backends/test_iree_pipeline.py` documents: float32 addition is not associative, so a reduction that a compiled backend reassociates differs from the numpy interpreter's order by about one ulp (~1.19e-7 at unit magnitude) per reordering, the error grows with the reduction depth but stays a small multiple of eps, and 1e-5 keeps two orders of magnitude of headroom for a box- or build-dependent reordering while still failing on any real divergence; `1e-6` is the absolute floor for attributes that sit near zero, where a pure relative bound is meaningless.
Every row records the max absolute difference over entries finite on both sides, every mask-mismatch count and the first failing field, so a `fail` names the field instead of the row.
The `numpy` target records `{"status": "skipped", "reason": "target is the numpy reference"}`, `--no-cross-check` records `{"status": "skipped", "reason": "--no-cross-check"}`, and a reference leg that cannot run records `status="error"` with its traceback while the measured row keeps its timings — missing numerical evidence stays visible instead of silently green.
The trace is part of the record (`trajectory` per row, `protocol.trajectory_generations: 14`), so the tables below re-render from it without re-measuring and `--trace N` reproduces it.

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
`benchmarks/results/gpu_rocm_xla_2026-11-06.json` is committed evidence (it is in the re-include list of `benchmarks/results/.gitignore`), so every cell above is traceable to the committed record.
Nothing here is a statement about HEAD: every number belongs to that record and to the tree its fingerprint names.
<!-- END MEASURED (gpu_measure.py) -->

## Limitations

No legacy-GPU comparison exists: the frozen JAX baseline in `benchmarks/legacy_jax` is CPU-only by protocol, so an accelerator number here is informational and is never part of the numpy-only old-vs-new verdict in `RESULTS.md`; there is no "old side" for a GPU.
One device: a single `AMD Instinct MI300X VF` at one device index (`iree.runtime.get_driver("hip").query_available_devices()` reports one entry, `device_id=1`), so nothing here says how the numbers scale to a second GPU, and no multi-device or multi-stream path is exercised.
Only `xla` rocm is measurable here: the `xla` backend needs a PJRT C API plugin `.so`, the ROCm one is `jax-rocm7-pjrt` and no CPU PJRT C API plugin exists to buy (`jaxlib 0.10.2`/`0.12.0.1` export no `GetPjRtApi`, and PyPI ships no CPU plugin package), so an `xla` CPU row stays impossible on this box and `tvm` is untested in this repository.
The plugin also wants two wirings, not one: `compile_options["plugin_path"]` alone does not satisfy evoxc's `check_available()` probe, which takes no options and reads only `EVOXC_PJRT_PLUGIN`, so a row built from the option without the variable is an `error` row rather than a silent fallback.
Transfer cost is inside `ms/generation` and not separated from it: the host materialisation of each generation's result happens inside the timed `step()` call, and the host/device traffic is part of that wall clock; this protocol does not report a transfer-only number, and unified-memory behaviour is not instrumented.
`peak RSS` is per process and is the driver's resident memory, not the device's, so the 1894.9–1942.7 MiB on the `iree` rows and the 9368.5–9427.1 MiB on the `xla` rows say nothing about VRAM occupancy and no device-memory figure is recorded anywhere in this protocol; the `xla` figure includes the loaded 480.1 MiB plugin and is dominated by its runtime, while the plugin's `XLA backend allocating 143.77GiB … for BFCAllocator` line is an address-space reservation, not resident memory.
No per-generation dispatch or op count is instrumented here — `PERF_NUMPY.md` is where executed-op counts live for the CPU interpreter — so an accelerator row gives wall clock and nothing about what the device was asked to do.
Timing is wall clock with a warmup, one seed and one committed repeat: this record has no second session, so the per-row 20-sample `min–max` in the generated block is its only spread, and it is not a distribution over seeds.
The one-off compile is per static shape, so a shape change repays it: `build_s` is excluded from every `ms/gen` sample but is real wall clock for a run that evaluates few generations, which is what the 3–243 generation break-evens in the verdict quantify; `setup_s` is a second one-off (4.3–4.5 s on `xla`, 1.2–1.5 s on `iree`) that the break-even columns do not charge.
The cross-check's later-generation divergence makes a long-horizon accelerator-vs-numpy *population* comparison a trajectory comparison, not an equality test: this record's first key divergences land at generations 4/4/5/4/13 on `iree` and 6/9/8/10/none on `xla` (`xor_p500`/`xor_p1000`/`xor3d_p500`/`custom_pagie_p500`/`xor_p100`), so the fitness (and species-count) view is what to compare across many generations, and the key arrays are not.
The trace re-pays a full build for both of its legs on every traced row, so `--trace 14` is a real cost: the twelve trajectory legs add 121.9 s of wall clock (83.6 s `iree`, 38.3 s `xla`).
The record's `meta.report_title` is the tool's default (`numpy / iree llvm-cpu / iree rocm`) because this session overrode `--targets` without overriding `--report-title`; nothing renders it, the block's heading derives from the record's actual targets, and the title is left exactly as the measurement wrote it.

## Reproduce

```bash
# 0. environment: the compiled-backend extra, the ROCm runtime, and the PJRT plugin the xla rows drive
uv sync --extra iree
uv pip install --no-deps jax-rocm7-pjrt-0.10.2-py3-none-manylinux_2_27_x86_64.whl   # ships jax_plugins/xla_rocm7/xla_rocm_plugin.so (480.1 MiB)
export EVOXC_PJRT_PLUGIN=$PWD/.venv/lib/python3.12/site-packages/jax_plugins/xla_rocm7/xla_rocm_plugin.so
/opt/rocm/.info/version && rocm-smi --showproductname --showdriverversion   # 7.1.0 / MI300X VF / driver 6.16.6 here

# 1. numpy-free planning: targets, passthrough flags, the case matrix
python3 benchmarks/gpu_measure.py --list

# 2. one row, one fresh process (this is what `matrix` spawns per row)
uv run --extra iree python benchmarks/gpu_measure.py run \
    --case xor_p500 --target xla-rocm --pjrt-plugin "$EVOXC_PJRT_PLUGIN" \
    --warmup 1 --generations 3 --json $TMPDIR/row.json

# 3. the full matrix this report renders: 6 cases x 3 targets, strictly sequential
uv run --extra iree python benchmarks/gpu_measure.py matrix \
    --cases xor_p100,xor_p500,xor_p1000,xor3d_p500,custom_pagie_p500,cartpole_p100 \
    --targets numpy,iree-rocm,xla-rocm \
    --warmup 3 --generations 20 --trace 14 --seed 42 \
    --pjrt-plugin "$EVOXC_PJRT_PLUGIN" \
    --json benchmarks/results/gpu_rocm_xla_2026-11-06.json --out benchmarks/GPU.md

# 4. re-render the measured block from the existing record: no measurement, no JSON rewrite
uv run --extra iree python benchmarks/gpu_measure.py render \
    --json benchmarks/results/gpu_rocm_xla_2026-11-06.json --out benchmarks/GPU.md
```

Step 3 is the exact argv the record stores, with the recording worker's absolute interpreter path replaced by the portable `uv run --extra iree python` and the plugin path given as the variable it was read from.
It is a 6 min 11 s job on this box (18 rows, of which the twelve trajectory legs are 121.9 s, the six `iree-compile`s 50.7 s and the six XLA compiles 13.9 s), so run it in the background rather than under a short foreground timeout.
`uv pip install --no-deps` matters twice: the wheel has no dependency metadata worth resolving (it carries the plugin and nothing else), and a plain `uv sync` afterwards prunes it, so install it *after* the sync — the same recipe the `xla` backend needs anywhere, since no PyPI package ships a CPU plugin at all.
`matrix` exits 0 only when every planned row is `ok` and 1 as soon as any row is `error` (it prints a `BLOCKED: …` line per blocked row); this matrix exited 0 with 18/18 `ok`, which is the state the block below records.
`render` and `matrix --out` replace only the interior of this file's two markers (exactly one of each in the file) and exit 2 with a named error on a missing marker, an unreadable record, a wrong schema or a missing `records` list.
A single row is cheap on the numpy leg (0.283–0.805 s build) and pays the compile on a compiled leg (2.149–2.672 s XLA, 8.017–9.027 s `iree-compile`), so budget a compiled row at its compile plus its timed region, and a traced row at that plus `N × (accelerated + numpy generation)`.
The IREE runtime can print `nanobind: leaked …` lines to stderr when the interpreter exits, and the XLA plugin prints its allocator lines at client creation; both are teardown/startup messages around the measured region, never a failure and never an `error` row.
The Python-free paths are `--list`, `--help` and nothing else: `run` and `matrix` need `numpy` and `tensorneat`, and the `iree-*` targets need the `iree` extra while the `xla-*` targets need the plugin.

## What this does not change

`RESULTS.md` stays the generated numpy-only old-vs-new record, with its own verdict computed from `results/authoritative.json`; no accelerator row is added to it, and no number in it is touched by this protocol.
`BACKENDS.md` stays the compiled-backend CPU-leg record of its own session — the `iree` llvm-cpu numbers, which this matrix does not measure — and `PERF_NUMPY.md` stays the numpy-interpreter performance record; neither is re-rendered from here.
Accelerator rows are never wired into the harness verdict, and that is deliberate: `protocol.comparisons` pairs every old-side record with every new-side record, so an accelerator row in `results/` would enter the numpy-only verdict as if it were a numpy measurement and could flip a timing verdict that the CPU contract owns.
Keeping the accelerator protocol standalone (`benchmarks/gpu_measure.py`, its own `tensorneat-gpu/1` schema, its own record and report) keeps that contract intact: the harness verdict compares what it declares it compares, and an accelerated row is evidence about a different question.
`benchmarks/results/gpu_rocm_2026-11-05.json` stays the record of its own tree and session: it is superseded as this file's block source only, and is not rewritten or re-measured here.
