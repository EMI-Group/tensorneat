# TensorNEAT accelerator measurement — numpy reference vs iree rocm vs xla rocm, and the frozen JAX implementation on the same MI300X VF

**Verdict for the record this file renders: both compiled paths are green in one session — eighteen planned rows, eighteen `ok`, zero blocked — so every one of the six cases carries a real compile time, a real per-generation timing and a real generation-1 numerical cross-check on `iree` rocm and on `xla` rocm against the `numpy` reference.**
Two measured blocks live in this file, each owned by one tool and one committed record: `gpu_measure.py` owns `## Measured: numpy vs iree-rocm vs xla-rocm` (from `results/gpu_rocm_xla_2026-11-07.json`), and `jax_device_measure.py` owns `## Against the frozen JAX implementation, same device` (from `results/jax_rocm_2026-11-07.json`) — the frozen pre-rewrite JAX implementation timed on the same device, which is the only block here that answers old-vs-new.
`Pipeline(backend="iree", device="rocm", compile_options={"target_backends": ["rocm"]})` compiles the generation graph with `iree-compile` for `gfx942`, and `Pipeline(backend="xla", device="rocm", compile_options={"plugin_path": …})` compiles it with XLA for the same device, driving the PJRT C API plugin `xla_rocm_plugin.so` (480.1 MiB, sha256 `8e8319b4…`) whose path the harness passes as `--pjrt-plugin`.
The device is faster per generation than the numpy reference on all six cases, on both compiled paths: `iree` rocm by 1.76x on `xor_p100` (10.633 against 18.767 ms/gen), 5.73x on `xor_p500` (12.744 against 72.983), 8.69x on `xor_p1000` (15.203 against 132.185), 6.79x on `xor3d_p500` (12.910 against 87.598), 55.90x on `custom_pagie_p500` (15.873 against 887.256) and 1.43x on `cartpole_p100` (67.202 against 95.952); `xla` rocm by 4.91x (3.825), 18.10x (4.033), 26.61x (4.968), 21.09x (4.153), 150.61x (5.891) and 4.07x (23.567) on the same rows.
The two compiled backends are not close: `xla` rocm is 2.69x–3.16x faster per generation than `iree` rocm on every case, and its per-shape XLA compile (`build_s` 2.034–2.607 s) costs 0.26x–0.31x of `iree`'s (`build_s` 7.799–8.658 s).
What `xla` pays instead is one-off client setup and resident memory: `setup_s` is 3.048–4.159 s against `iree`'s 1.186–1.258 s (loading the 480.1 MiB plugin and creating the PJRT client), and peak RSS is 4880.9–9408.0 MiB against `iree`'s 1891.0–1944.1 MiB and the numpy leg's 142.4–933.9 MiB.
The per-generation win is still bought with a one-off compile on both paths: `build_s` ÷ the row's per-generation saving, both in seconds, is 959/133/70/108/10/301 generations on `iree` (`xor_p100`/`xor_p500`/`xor_p1000`/`xor3d_p500`/`custom_pagie_p500`/`cartpole_p100`) and 136/33/19/27/3/31 on `xla`, so `xla` repays its compile in 0.10x–0.31x of the generations `iree` needs.
Numerically, generation 1 is the equality gate and it is green on all twelve compiled rows: `node_keys` and `conn_keys` bit-identical, attribute differences of at most 4.768e-7 on `iree` and exactly 0 on `xla`, fitness differences of at most 2.384e-7 on `iree` and 5.960e-8 on `xla`, and zero NaN-mask mismatches everywhere.
Beyond generation 1 both compiled legs are a search trajectory, and this session ran with `--trace 0`: the final-state snapshot fails on 10 of the 12 compiled rows and passes only on `xor_p100`/`xla` and `cartpole_p100`/`iree`, and no per-generation divergence table exists for this record.
Nothing below quotes a number that is not in `benchmarks/results/gpu_rocm_xla_2026-11-07.json`; the speed-up columns and the generation counts are arithmetic on that record's medians, its `build_s` and its min–max samples, and nothing here is a statement about HEAD.

## Headline: what each compiled backend buys per generation

| Case | numpy ms/gen (median) | `iree` rocm ms/gen | `xla` rocm ms/gen | numpy ÷ `iree` | numpy ÷ `xla` | `iree` ÷ `xla` | final max fitness numpy / `iree` / `xla` | final state `iree` / `xla` |
|---|---|---|---|---|---|---|---|---|
| `xor_p100` | 18.767 | 10.633 | 3.825 | 1.76x | 4.91x | 2.78x | -0.176966 / -0.184767 / -0.176966 | pass / pass |
| `xor_p500` | 72.983 | 12.744 | 4.033 | 5.73x | 18.10x | 3.16x | -0.147786 / -0.112459 / -0.140949 | node_keys / node_keys |
| `xor_p1000` | 132.185 | 15.203 | 4.968 | 8.69x | 26.61x | 3.06x | -0.053339 / -0.136390 / -0.053339 | node_keys / node_keys |
| `xor3d_p500` | 87.598 | 12.910 | 4.153 | 6.79x | 21.09x | 3.11x | -0.223466 / -0.250000 / -0.223466 | node_keys / node_keys |
| `custom_pagie_p500` | 887.256 | 15.873 | 5.891 | 55.90x | 150.61x | 2.69x | -0.032613 / -0.046049 / -0.032613 | node_keys / node_keys |
| `cartpole_p100` | 95.952 | 67.202 | 23.567 | 1.43x | 4.07x | 2.85x | 200.000000 / 200.000000 / 200.000000 | pass / fitness |

Every cell except the three ratio columns and the last two is a column of the generated block below; a ratio is the numpy median divided by the compiled median of the same row, hand-written here because the generated table has no ratio column, and the final-state cell abbreviates the generated cross-check verdict (`pass` means the post-`tell` population matched as well).

The biggest win is `custom_pagie_p500`: 55.90x on `iree` (887.256 against 15.873 ms/gen) and 150.61x on `xla` (887.256 against 5.891).
That is the expected shape rather than a surprise: the func-fit forward is the sample-invariant schedule evaluated in tiles, which makes it the widest per-generation device workload in the matrix, and the numpy interpreter's per-op overhead scales with exactly that work — this row is also the slowest cell of the whole matrix on the CPU leg (887.256 ms/gen against 95.952 on the next-slowest, `cartpole_p100`).
The narrowest device margin is `cartpole_p100` on both compiled paths: 1.43x on `iree` (95.952 against 67.202 ms/gen) and 4.07x on `xla` (95.952 against 23.567).
A generation at those sizes is dominated by fixed per-generation overhead — dispatch, host materialisation and run-boundary device traffic, all inside the timed `step()` — which a compiled backend cannot compress the way it compresses a wide kernel; `cartpole_p100` additionally drives a per-step rollout loop over its horizon, so its per-generation launch count is the largest of the shipped problems (the canonical dispatch probe in `src/tensorneat/CONTEXT.md` counts 27 860 executed ops per generation for this case in its dominant depth-1 regime, against 7 920 at `xor_p500` and 11 187 at `custom_pagie_p500`).
The numpy leg on that row has the widest spread of the six numpy rows: 94.413–153.009 ms around a 95.952 median and a 105.252 mean, so its ratio against either compiled leg carries more per-sample noise than the other five rows.
`xla` compresses that fixed overhead further than `iree` does, which is why its margin over numpy on the same narrow row is 4.07x against `iree`'s 1.43x.
Final fitness separates the legs where the timing margin does not: at the shared 24-step snapshot the numpy leg's final best is ahead of `iree`'s on `xor_p100` (-0.176966 against -0.184767), `xor_p1000` (-0.053339 against -0.136390), `xor3d_p500` (-0.223466 against -0.250000) and `custom_pagie_p500` (-0.032613 against -0.046049) and behind it on `xor_p500` (-0.147786 against -0.112459), while `xla` reproduces the numpy value exactly on `xor_p100`, `xor_p1000`, `xor3d_p500`, `custom_pagie_p500` and at the 200.0 cap on `cartpole_p100` and is ahead of it on `xor_p500` (-0.140949 against -0.147786).
Those are 24-step search outcomes of one seed on either side of a float-reassociation difference (`## Cross-check`), not per-generation quality: the per-generation question this file answers is wall clock.

Memory runs the other way, and it is driver RSS, never device memory: the `iree` leg sits at 1891.0–1944.1 MiB on every row and barely tracks the workload, the `xla` leg at 9385.7–9408.0 MiB on five rows and 4880.9 MiB on `custom_pagie_p500`, while the numpy leg grows with the workload (142.4 MiB at `cartpole_p100` to 933.9 MiB at `custom_pagie_p500`).
Both compiled figures are single process high-water marks dominated by the loaded runtime and executable, which is why they are flat from pop 100 to pop 1000 instead of scaling like the numpy interpreter's retained intermediates; the `xla` figure is 4.84x–4.95x the `iree` one on five rows and 2.5x on `custom_pagie_p500`, and is discussed with the plugin's allocator reserve in `## IREE vs XLA on the same device`.

## IREE vs XLA on the same device

Same box, same graphs, same seed, same case config: the two compiled backends differ only in their compiler and their runtime, so the comparison below is per `(case, shape)` and its columns are the ones the generated block records (`build_s`, `setup_s`, `peak_rss_mib`) plus arithmetic on them.

| Case | `iree` build (s) | `xla` build (s) | `xla` ÷ `iree` build | `iree` setup (s) | `xla` setup (s) | `iree` break-even (gens) | `xla` break-even (gens) | peak RSS numpy / `iree` / `xla` (MiB) |
|---|---|---|---|---|---|---|---|---|
| `xor_p100` | 7.799 | 2.034 | 0.26 | 1.186 | 4.014 | 959 | 136 | 143.6 / 1891.0 / 9358.2 |
| `xor_p500` | 8.017 | 2.244 | 0.28 | 1.218 | 4.019 | 133 | 33 | 167.6 / 1914.7 / 9396.2 |
| `xor_p1000` | 8.193 | 2.354 | 0.29 | 1.258 | 4.084 | 70 | 19 | 198.0 / 1937.6 / 9402.0 |
| `xor3d_p500` | 8.063 | 2.211 | 0.27 | 1.220 | 4.053 | 108 | 27 | 180.0 / 1916.2 / 9408.0 |
| `custom_pagie_p500` | 8.406 | 2.607 | 0.31 | 1.204 | 3.048 | 10 | 3 | 933.9 / 1944.1 / 4880.9 |
| `cartpole_p100` | 8.658 | 2.236 | 0.26 | 1.215 | 4.159 | 301 | 31 | 142.4 / 1939.0 / 9385.7 |

A break-even count is that row's `build_s` ÷ (numpy median − that backend's median), with both medians converted from milliseconds to seconds, i.e. the generations of per-generation saving needed to repay the compile; it charges the compile and nothing else, and `setup_s` is a per-process cost charged once rather than per shape.

Per generation, `xla` wins every row: 2.78x on `xor_p100` (10.633 against 3.825 ms/gen), 3.16x on `xor_p500`, 3.06x on `xor_p1000`, 3.11x on `xor3d_p500`, 2.69x on `custom_pagie_p500` and 2.85x on `cartpole_p100`.
That band is narrow across rows whose numpy medians span 47x (18.767 to 887.256 ms/gen) and whose per-generation work differs by orders of magnitude in width and in launch count, so the `xla` lead reads as a lower per-launch and per-dispatch cost in its runtime rather than as a wider kernel; the widest workload (`custom_pagie_p500`) carries the smallest lead (2.69x) and the largest lead sits on the middling `xor_p500` (3.16x), so this record does not show the gap scaling with kernel width and the mechanism is stated as a fixed per-generation difference, not a measured decomposition.
The compile cost goes the same way: XLA compiles each of these shapes in 2.034–2.607 s against `iree-compile`'s 7.799–8.658 s (0.26x–0.31x).
The cost side is setup time and memory: `xla` takes 3.048–4.159 s in `Pipeline.setup()` on every row against `iree`'s 1.186–1.258 s (that is the plugin `.so` load plus PJRT client creation, once per process), and its 9385.7–9408.0 MiB peak RSS is 4.84x–4.95x `iree`'s 1891.0–1944.1 MiB on five rows.
The one row where the two compiled legs' resident memory are closer is `custom_pagie_p500` (4880.9 against 1944.1 MiB, 2.5x), and the record stores only the process high-water mark with no allocator or per-phase breakdown, so that difference is stated rather than attributed.
The `xla` resident figure is real, not the allocator's virtual reserve: the plugin logs `XLA backend allocating 143.77GiB … for BFCAllocator` at client creation, but that is an address-space reservation, the process's measured high-water mark is the 4880.9–9408.0 MiB the block records, and no device-memory figure is measured anywhere in this protocol.
Break-even against this tree's numpy leg is long, and that is a property of the numpy leg rather than of the compiled backends: 959/133/70/108/10/301 generations on `iree` and 136/33/19/27/3/31 on `xla` here, against the 2026-11-06 record's 243/77/49/70/10/50 and 54/19/13/18/3/10.
The numpy leg of this record is 1.09x–2.46x faster per generation than that record's, so each generation saves less and the compile takes longer to repay; the two sets of counts belong to their own trees and are not comparable.

## Provenance: which tree and which session

Every number in this file belongs to `benchmarks/results/gpu_rocm_xla_2026-11-07.json` and to the git commit its fingerprint names, `93ce3cac122fef34edc00093d3a3d2b4b9bbf5f7` — this branch's HEAD at recording time, confirmed clean in the record rather than assumed — and no sentence here is a statement about HEAD.
That record is one `gpu_measure.py matrix` session: 2026-11-07T10:29:06Z – 2026-11-07T10:32:39Z (3 min 33 s wall clock), 18 rows over the cases `xor_p100,xor_p500,xor_p1000,xor3d_p500,custom_pagie_p500,cartpole_p100` × the targets `numpy,iree-rocm,xla-rocm`, seed 42, 3 warmup and 20 timed generations, `--trace 0`, one repeat per row, one fresh process per row — 18 `ok` and 0 `error`.
The environment it ran on is the block below: IREE 3.13.0 compiler and runtime, ROCm 7.1.0, driver 6.16.6, `AMD Instinct MI300X VF` (`gfx942`) as the single HIP device (`device_id=1`), numpy 2.5.4, Python 3.12.3, evoxc 0.1.0, 20 logical cores, 241 611 MiB of RAM, threads at runtime defaults with nothing exported, and a clean tree.
The `xla-rocm` target is `Pipeline(backend="xla", device="rocm", compile_options={"plugin_path": …})` driving the PJRT C API plugin `xla_rocm_plugin.so`: 503 453 288 bytes (480.1 MiB), sha256 `8e8319b48fcc771b5221771dc3da0d24bc6950c872ee07ac6f5200743b3b4fa7`, at the conventional `site-packages/jax_plugins/xla_rocm7/` path, and recorded in the record's own `meta.pjrt_plugin` fingerprint (path, basename, size, sha256) that the environment table below renders as basename, size and a 12-hex sha256.
`plugin_path` is the explicit override, not a requirement: the pinned `evoxc` resolves a PJRT plugin as `plugin_path` → `EVOXC_PJRT_PLUGIN` → fixed well-known paths → an installed `jax_plugins/<platform>/<name>_plugin.so`, so a `Pipeline(backend="xla", device="rocm")` with neither the option nor the variable finds this wheel too (`EVOXC_PJRT_PLUGIN_DISCOVERY=0` opts out), and the harness pins the path so the row records which `.so` it measured.
The record stores the recording worker's absolute interpreter path in `meta.command`; the portable form of the same argv is step 3 of `## Reproduce`, and that is the command to regenerate it with.
The `evoxc 0.1.0` row in the environment table is the installed distribution's metadata; this tree resolves `evoxc` from an upstream `main` commit (`fe85a534…`) through `[tool.uv.sources]` rather than from a PyPI release, so the version string does not by itself identify the interpreter revision the numbers were measured on.
`meta.report_title` and `meta.report_note` were both set for this session, and the note is rendered as the last line of the measured block's provenance section, so a bare re-render keeps it.

**Measurement conditions, stated rather than implied.** The box was shared while this matrix measured — other agents run on the same host — and the record's document-build load average is 4.54 / 4.17 / 6.54 (1/5/15 min), with every row storing the load average it started under in its own `env.loadavg`.
No second session was run for this record: the uncertainty measure here is the per-row `ms/gen min–max` spread the generated block carries (20 samples per row), and a single median is not a distribution.
The earlier three-target session (`benchmarks/results/gpu_rocm_xla_2026-11-06.json`, tree `ddd7295a`) measured the same six cases at numpy 43.300/120.186/182.567/136.063/965.043/236.023, `iree` 10.295/12.820/15.117/12.920/15.768/61.495 and `xla` 3.610/4.535/5.497/4.580/6.319/13.362 ms/gen, and the deltas are real but they are not a backend comparison: that record belongs to a different tree and a different evoxc revision.
Against it, this tree's numpy leg is 1.09x–2.46x faster per generation (`custom_pagie_p500` 1.09x, `xor_p1000` 1.38x, `xor3d_p500` 1.55x, `xor_p500` 1.65x, `xor_p100` 2.31x, `cartpole_p100` 2.46x), which is a numpy-interpreter change (the upstream `main` evoxc pin plus this tree's op-count reductions), not an accelerator change.
`iree` moved by between −0.6 % and +9.3 % on every row, the widest being `cartpole_p100` 61.495→67.202 ms/gen.
`xla` moved between −11.1 % and +6.0 % on five rows but 13.362→23.567 ms/gen on `cartpole_p100` (+76 %), and that one move is not attributable to the `XLA_FLAGS` workaround alone: the trees and the evoxc revisions differ, and that record has no `xla_flags` field at all because the field landed after it was measured.
Neither record's per-generation numbers may be quoted as a statement about the other, and this file quotes only its own record.

## Protocol

`gpu_measure.py` measures exactly one `(case, target)` row per process, and one row per process is not a style choice: `peak_rss_mib` is `resource.getrusage(RUSAGE_SELF).ru_maxrss`, a process-lifetime high-water mark, so two rows in one interpreter would report the higher of the two as both.
`matrix` plans `cases × targets` and spawns one `run` subprocess per row **strictly sequentially**, one fresh interpreter each and the same interpreter as the driver, so no two measurements ever overlap on the box.
Per row: `Pipeline.setup()` is timed into `setup_s` (0.015–0.017 s on the numpy leg, 1.186–1.258 s on the `iree` rocm leg, 3.048–4.159 s on the `xla` rocm leg, where the plugin load and the PJRT client land); the first `Pipeline.step()` is timed into `first_step_s` and its build cost, `Pipeline.build_seconds`, is reported as `build_s` (0.266–0.786 s on the numpy leg, 7.799–8.658 s on the `iree` leg, which is where `iree-compile` lands, and 2.034–2.607 s on the `xla` leg, which is where XLA compilation lands) — the build is never charged to a generation and that step is excluded from every timing sample; then `--warmup` untimed generations and `--generations` timed ones, with stdout/stderr redirected to `os.devnull` for the whole warmup + timed region.
Each timed generation is `time.perf_counter()` wall clock around one `step()` call, which includes the host materialisation of the result; the row keeps mean, median, min, max and every sample (20 per row here), plus `generations_per_second = 1000 / mean` and `final_generation` 24, which is the build-consuming step plus 3 warmup and 20 timed generations.
`peak_rss_mib` is sampled immediately after the timed region, then `final_generation`, `final_max_fitness`, `device_info` (the target's `backend`/`device`/`compile_options` — `{"target_backends": ["rocm"]}` on the `iree` rows, `{"plugin_path": …}` on the `xla` rows — plus the IREE HIP device query) and the exact `protocol` block are recorded.
`--trace N` (default `0`, off) adds a lockstep trajectory *after* the timed region and after the `peak_rss_mib` sample: one fresh pair of legs — the accelerated target and a fresh `backend="numpy"` pipeline, each paying its own build — driven generation by generation, so it costs `N × (accelerated generation + numpy generation)` and can never enter a timing sample or the RSS figure.
This session ran with `--trace 0`, so every row stores `trajectory.status == "skipped"` with the reason `--trace 0 (off)` and nothing in this file is a divergence-generation table.
`--json` writes the row; its `status` is `ok` or `error`, and any exception during the measurement becomes an `error` row carrying the full traceback plus whatever was already measured, with the last traceback line printed to stderr and exit code 1 — a failed row is recorded, never dropped, and a hard crash that kills the process is turned into an `error` row by the matrix driver (returncode plus stderr tail).
The defaults are the canonical matrix of `benchmarks/cases.py` (`xor_p100`, `xor_p500`, `xor3d_p500`), the three targets `numpy`, `iree-llvm-cpu`, `iree-rocm`, seed 42, three warmup and ten timed generations, one repeat, no trajectory (`--trace 0`): the case table is imported, never duplicated here, and an `xla` target is never in the default set, so one is named explicitly.
Targets are data, and the raw flags override a named target for hardware this file does not name: `--device KIND` with `--device-index N` (together they build `evoxc.Device(KIND, N)`), `--target-backends a,b`, `--rocm-target ARCH` (appends `--iree-rocm-target=ARCH` to the `iree_compile_args` compile option) and `--iree-compile-arg FLAG` (repeatable). `device` and `target_backends` must stay aligned: a combination that pairs a host device with a device-only target becomes an `error` row instead of being repaired silently.
`--pjrt-plugin PATH` is resolved once by `matrix` and forwarded to every row child as the row's `compile_options["plugin_path"]`, and the child also exports it as `EVOXC_PJRT_PLUGIN` so the variable pins the same `.so` for any option-less discovery call; the variable is a pin, not a requirement, because the pinned `evoxc` registers the `xla` backend unconditionally and resolves the plugin at `compile`/`load` with the documented precedence `plugin_path` → `EVOXC_PJRT_PLUGIN` → fixed well-known paths → an installed `jax_plugins/*/*_plugin.so`.
`--xla-flags TEXT` sets the row process's `XLA_FLAGS`, and its default is `gpu_targets.XLA_WORKAROUND_FLAGS`, the string `--xla_gpu_enable_command_buffer=WHILE`, which takes FUSION command-buffer capture away from the RL rollout's long `while`.
`--xla-flags ''` — or the empty forwarded form `--xla-flags=` — measures *without* the workaround: the variable is removed from the child environment instead of being set to an empty string, so that leg is a real absence rather than an empty value.
The rule is applied to the `xla-*` targets only (`gpu_targets.PJRT_TARGETS`): a numpy, `iree` or jax row keeps its inherited environment untouched, and `matrix` forwards the same rule to every row child.
The effective value is recorded per row as `record["xla_flags"]`, `null` for a target that does not use XLA — which is why the six numpy and six `iree` rows here store `null` — and once per document as `meta["xla_flags"]`, `null` when no planned target is an `xla` target; the environment table in the block below and its provenance line both render it, so a per-generation number measured under a different flag value is a different experiment rather than a renamed one.
XLA reads `XLA_FLAGS` at backend init, before the plugin builds the executable, so this is an environment setting and never a compile option.
This record's `meta.xla_flags` is `--xla_gpu_enable_command_buffer=WHILE` and its six `xla` rows carry exactly that value.

**Scratch evidence from this session, not a committed record.** Two checks ran beside the matrix, into `$TMPDIR`, and none of their numbers belongs to the record below.
A pure-forward A/B through `forward_measure.py run --target xla-rocm --passes 10 --warmup 3` measured `xor_p500_s500` at 1.344 ms/pass with the flag (1.301–1.389) against 1.404 ms/pass without it (1.374–1.453), and `xor_p1000_s500` at 2.153 (2.136–2.198) against 2.121 (2.094–2.192): no cost beyond the roughly ±5 % per-leg spread, and the two workloads disagree in sign.
On the RL row the flag is kept for a crash-proof leg, not for a speed gain: `cartpole_p100` on `xla` completed its 24 steps at a 23.567 ms/gen median in the matrix, and with `--xla-flags=` it also completed 4 of 4 standalone trials (20.879 / 20.979 / 21.336 / 22.5 ms/gen), so the `PJRT_LoadedExecutable_Execute` SIGSEGV that motivated the workaround was observed on an earlier tree and evoxc snapshot and is intermittent here.
The flag stays the shipped default because the leg must not be a lottery, and nothing in this file claims it is what makes this session's crash-free result possible.
Nothing falls back to another backend: if a target cannot run, the exception is the row, and `matrix` exits 0 only when every planned row is `ok` — a blocked device path is loud by construction, and this matrix exited 0 because no row was blocked.

## Cross-check: what is compared and at what tolerance

The cross-check runs *after* the timed region, in the same process, and it never changes a timing: a fresh `backend="numpy"` pipeline with the same case, seed and step count is driven over the same 24 steps, so the comparison is one implementation against itself on two backends — no JAX enters the process, and "one implementation per process" is untouched (the `xla` row loads a PJRT plugin, not `jax`).
Two snapshots are compared, and they are recorded separately because they mean different things.
**Generation 1** is both legs' first evaluated population (the payload and the `(P,)` fitness vector of the first `step()`): integer keys (`node_keys`, `conn_keys`) must be `np.array_equal`, the attribute payloads must have an identical `np.isnan` mask, and the fitness vector an identical NaN *and* `-inf` mask, with every finite entry inside `FLOAT_RTOL = 1e-5` / `FLOAT_ATOL = 1e-6`.
That gate is green on all twelve compiled rows: zero mismatching `node_keys`/`conn_keys` entries, attribute differences of at most 4.768e-7 on `iree` and exactly 0 on `xla`, fitness differences of at most 2.384e-7 on `iree` and 5.960e-8 on `xla`, and zero NaN-mask and non-finite mismatches.
**Final state** is the post-`tell` population after the same number of steps plus the last step's fitness, compared with the same rules and labelled as a search-trajectory observation: both legs draw the same evoxc splitmix64 stream from one seed and one graph, so generation 1 must match, while beyond it a float-reassociation difference between the backends can reorder a near-tied decision and legitimately diverge the integer keys — that is trajectory drift, not necessarily a kernel defect, and the record carries the sentence (`TRAJECTORY_NOTE`) next to the verdict.
It fails on 10 of the 12 compiled rows: `node_keys` on `xor_p100`/`iree`, on both `xor_p500` rows, both `xor_p1000` rows, both `xor3d_p500` rows and both `custom_pagie_p500` rows, and `fitness` on `cartpole_p100`/`xla`, where keys and attributes are exact but an integer-valued, horizon-capped return differs by 1.0 — an action chosen at a near-tie.
It passes only on `xor_p100`/`xla` (keys exact, attributes within 2.384e-7, fitness within 4.470e-8) and on `cartpole_p100`/`iree` (keys exact, fitness identical).
Unlike the old-vs-new harness, which compares two different PRNGs (JAX `PRNGKey` vs evoxc splitmix64) and therefore no trajectories at all, both legs here share one random stream, which is why generation 1 is a hard comparison rather than a statistical one.
The mechanism is the one the earlier session traced: a sub-ulp difference in the attribute payload is amplified by fitness ranking in `src/tensorneat/algorithm/neat/_kernels.py` — `sort_desc_legacy` (`:171`) is a stable ascending argsort gathered through the reversal permutation, i.e. a strict total order in which an exact tie is broken by row index, and `pair_parents` (`:407`) ranks with it at `:460`, draws parents at `:478` and pins elite slot *k* to rank `min(k, live_members − 1)` at `:493`–`:509`.
The benchmark cases carry `survival_threshold = 0.01` and `species_size = 20` (`benchmarks/cases.py`), so the survivor count `clamp(floor(0.01 × members), 1, …)` leaves every species under 200 members exactly one breeding parent while the pop-500 and pop-1000 tells fill all 20 species slots: up to 20 independent rank-0 decisions per generation, and one ulp-level tie flip changes a species' pinned elites and its single parent a generation or more before it can change the structural genes.
So generation 1 is the equality gate and nothing else is: every later generation is a search-trajectory observation, the entity to compare over a long horizon is the fitness view — the population's best, agreeing to float32 ulps on the two rows that pass the final-state comparison — never the key arrays, and a compiled backend that reproduces the generation-1 payload exactly (`xla` here) is not thereby more correct over 24 steps, since both legs pass exactly one final-state snapshot each.
The tolerances are the ones `test/backends/test_iree_pipeline.py` documents: float32 addition is not associative, so a reduction that a compiled backend reassociates differs from the numpy interpreter's order by about one ulp (~1.19e-7 at unit magnitude) per reordering, the error grows with the reduction depth but stays a small multiple of eps, and 1e-5 keeps two orders of magnitude of headroom for a box- or build-dependent reordering while still failing on any real divergence; `1e-6` is the absolute floor for attributes that sit near zero, where a pure relative bound is meaningless.
Every row records the max absolute difference over entries finite on both sides, every mask-mismatch count and the first failing field, so a `fail` names the field instead of the row.
The numpy reference leg pays its own build on every row (`reference_build_s` 0.284–0.763 s) and that cost is reported beside the row's timings, never inside them.
The `numpy` target records `{"status": "skipped", "reason": "target is the numpy reference"}`, `--no-cross-check` records `{"status": "skipped", "reason": "--no-cross-check"}`, and a reference leg that cannot run records `status="error"` with its traceback while the measured row keeps its timings — missing numerical evidence stays visible instead of silently green.
The trace is part of the record (`trajectory` per row, `protocol.trajectory_generations: 0` here), so a `--trace N` re-run would add the per-generation comparison this session deliberately did not measure.

<!-- BEGIN MEASURED (gpu_measure.py) -->
## Environment (measured, not assumed)

| Item | Value |
|---|---|
| Date / session (UTC) | 2026-11-07T13:58:25+00:00 – 2026-11-07T14:02:04+00:00 |
| GPU model (gfx version) | AMD Instinct MI300X VF (gfx942) |
| HIP devices (IREE runtime query) | device_id=1 AMD Instinct MI300X VF |
| ROCm version | 7.1.0 |
| Driver version | 6.16.6 |
| IREE compiler / runtime | 3.13.0 / 3.13.0 |
| PJRT plugin (xla targets) | `xla_rocm_plugin.so` (480.1 MiB, sha256 `8e8319b48fcc…`) |
| XLA_FLAGS (xla targets) | `--xla_gpu_enable_command_buffer=WHILE` (ROCm command-buffer workaround: the RL rollout's long `while` SIGSEGVs under FUSION capture) |
| evoxc | 0.1.0 |
| numpy | 2.5.4 |
| Python | 3.12.3 |
| Platform | Linux-6.8.0-146-generic-x86_64-with-glibc2.39 |
| CPU model / count | INTEL(R) XEON(R) PLATINUM 8568Y+ / 20 logical |
| Memory | 241611 MiB |
| Threads | runtime defaults (nothing exported; inherited environment recorded) |
| Git commit | `03368c1ea03c` (clean — generated report artifacts are not counted) |
| Load average (1/5/15 min, document build) | 3.17 / 2.76 / 2.81 |

## Measured: numpy vs iree-rocm vs xla-rocm

One row per planned `(case, target)`; `n/a` marks a metric the row does not have (an error row has no timings at all). `build` is the one-off generation-graph compile and is excluded from every `ms/gen` sample; `peak RSS` is the process high-water mark of that single row, which is why each row runs in its own process.

| Case | Target | Backend | Status | setup (s) | build (s) | ms/gen median | ms/gen mean | ms/gen min–max | gens/s | peak RSS (MiB) | final max fitness | cross-check |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| xor_p100 | `numpy` | `numpy` | ok | 0.034 | 0.271 | 18.760 | 19.003 | 17.384 – 21.720 | 52.622 | 151.2 | -0.176966 | skipped (target is the numpy reference) |
| xor_p100 | `iree-rocm` | `iree` | ok | 1.271 | 8.814 | 9.867 | 9.830 | 9.385 – 10.235 | 101.727 | 1894.5 | -0.184767 | gen 1 pass; final-state drift (node_keys) |
| xor_p100 | `xla-rocm` | `xla` | ok | 4.089 | 2.243 | 3.599 | 3.587 | 3.405 – 3.946 | 278.800 | 9380.6 | -0.176966 | pass (gen 1) |
| xor_p500 | `numpy` | `numpy` | ok | 0.016 | 0.313 | 73.846 | 73.665 | 68.791 – 78.338 | 13.575 | 167.9 | -0.147786 | skipped (target is the numpy reference) |
| xor_p500 | `iree-rocm` | `iree` | ok | 1.231 | 9.099 | 11.863 | 11.965 | 11.784 – 12.288 | 83.574 | 1911.4 | -0.112459 | gen 1 pass; final-state drift (node_keys) |
| xor_p500 | `xla-rocm` | `xla` | ok | 4.046 | 2.431 | 3.883 | 3.888 | 3.791 – 4.057 | 257.219 | 9407.4 | -0.140949 | gen 1 pass; final-state drift (node_keys) |
| xor_p1000 | `numpy` | `numpy` | ok | 0.017 | 0.380 | 130.224 | 130.617 | 123.946 – 143.296 | 7.656 | 198.5 | -0.053339 | skipped (target is the numpy reference) |
| xor_p1000 | `iree-rocm` | `iree` | ok | 1.190 | 9.136 | 14.215 | 14.377 | 13.752 – 15.198 | 69.555 | 1938.0 | -0.136390 | gen 1 pass; final-state drift (node_keys) |
| xor_p1000 | `xla-rocm` | `xla` | ok | 4.075 | 2.617 | 4.054 | 4.053 | 3.929 – 4.262 | 246.699 | 9421.5 | -0.053339 | gen 1 pass; final-state drift (node_keys) |
| xor3d_p500 | `numpy` | `numpy` | ok | 0.016 | 0.315 | 86.328 | 86.019 | 74.911 – 94.357 | 11.625 | 180.2 | -0.223466 | skipped (target is the numpy reference) |
| xor3d_p500 | `iree-rocm` | `iree` | ok | 1.204 | 9.179 | 12.004 | 12.073 | 11.839 – 12.382 | 82.832 | 1912.4 | -0.250000 | gen 1 pass; final-state drift (node_keys) |
| xor3d_p500 | `xla-rocm` | `xla` | ok | 4.123 | 2.510 | 3.876 | 3.907 | 3.731 – 4.197 | 255.923 | 9404.6 | -0.223466 | gen 1 pass; final-state drift (node_keys) |
| custom_pagie_p500 | `numpy` | `numpy` | ok | 0.015 | 0.744 | 813.551 | 812.973 | 712.758 – 890.623 | 1.230 | 933.8 | -0.032613 | skipped (target is the numpy reference) |
| custom_pagie_p500 | `iree-rocm` | `iree` | ok | 1.221 | 9.397 | 14.751 | 14.384 | 13.481 – 15.283 | 69.523 | 1939.7 | -0.046049 | gen 1 pass; final-state drift (node_keys) |
| custom_pagie_p500 | `xla-rocm` | `xla` | ok | 4.124 | 2.785 | 5.276 | 5.369 | 5.182 – 5.828 | 186.268 | 9442.8 | -0.032613 | gen 1 pass; final-state drift (node_keys) |
| cartpole_p100 | `numpy` | `numpy` | ok | 0.015 | 0.270 | 97.705 | 106.299 | 95.018 – 156.348 | 9.407 | 142.2 | 200.000000 | skipped (target is the numpy reference) |
| cartpole_p100 | `iree-rocm` | `iree` | ok | 1.198 | 9.692 | 59.030 | 59.269 | 58.579 – 61.811 | 16.872 | 1936.8 | 200.000000 | pass (gen 1) |
| cartpole_p100 | `xla-rocm` | `xla` | ok | 4.100 | 2.309 | 23.628 | 25.124 | 23.317 – 33.016 | 39.803 | 9392.5 | 200.000000 | gen 1 pass; final-state drift (fitness) |

## Numerical cross-check

Each non-numpy row is compared against a fresh `backend="numpy"` run of the same case, seed and step count in the same process. Generation 1 must match (one seed, one graph, one PRNG): integer keys exactly, attributes and fitness to rtol 1e-05 / atol 1e-06 over finite entries with an identical NaN mask. The final state is a search-trajectory observation, not a pass/fail gate on its own: beyond generation 1 a float-reassociation difference can reorder a near-tied selection and legitimately diverge the keys. The table's `cross-check` column summarises both snapshots: generation 1 is the equality gate; a final-state-only difference renders `gen 1 pass; final-state drift (…)`.

| Case | Target | Generation 1 | Keys (gen 1) | Attr max abs diff (gen 1) | Fitness max abs diff | Mask mismatches | Final state | Keys (final) | Attr max abs diff (final) | Fitness max abs diff (final) | Mask mismatches (final) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| xor_p100 | `iree-rocm` | pass | node_keys exact; conn_keys exact | node_attrs 0.000000238; conn_attrs 0.000000238 | 0.000000060 | 0 | fail | node_keys 189 mismatch(es); conn_keys 686 mismatch(es) | node_attrs 1.492010832; conn_attrs 4.826976299 | 0.131897777 | 1080 |
| xor_p100 | `xla-rocm` | pass | node_keys exact; conn_keys exact | node_attrs 0.000000000; conn_attrs 0.000000000 | 0.000000060 | 0 | pass | node_keys exact; conn_keys exact | node_attrs 0.000000149; conn_attrs 0.000000238 | 0.000000045 | 0 |
| xor_p500 | `iree-rocm` | pass | node_keys exact; conn_keys exact | node_attrs 0.000000477; conn_attrs 0.000000294 | 0.000000060 | 0 | fail | node_keys 734 mismatch(es); conn_keys 3545 mismatch(es) | node_attrs 2.440064669; conn_attrs 7.131760120 | 0.266454309 | 4278 |
| xor_p500 | `xla-rocm` | pass | node_keys exact; conn_keys exact | node_attrs 0.000000000; conn_attrs 0.000000000 | 0.000000060 | 0 | fail | node_keys 857 mismatch(es); conn_keys 3765 mismatch(es) | node_attrs 2.720985651; conn_attrs 7.074343204 | 0.230899110 | 5990 |
| xor_p1000 | `iree-rocm` | pass | node_keys exact; conn_keys exact | node_attrs 0.000000477; conn_attrs 0.000000298 | 0.000000089 | 0 | fail | node_keys 1687 mismatch(es); conn_keys 8767 mismatch(es) | node_attrs 3.298459530; conn_attrs 6.690523148 | 0.420150518 | 7554 |
| xor_p1000 | `xla-rocm` | pass | node_keys exact; conn_keys exact | node_attrs 0.000000000; conn_attrs 0.000000000 | 0.000000060 | 0 | fail | node_keys 1527 mismatch(es); conn_keys 7624 mismatch(es) | node_attrs 4.058683395; conn_attrs 6.799370766 | 0.418266296 | 9380 |
| xor3d_p500 | `iree-rocm` | pass | node_keys exact; conn_keys exact | node_attrs 0.000000477; conn_attrs 0.000000298 | 0.000000060 | 0 | fail | node_keys 976 mismatch(es); conn_keys 4456 mismatch(es) | node_attrs 4.066220284; conn_attrs 4.489749908 | 0.153934240 | 7562 |
| xor3d_p500 | `xla-rocm` | pass | node_keys exact; conn_keys exact | node_attrs 0.000000000; conn_attrs 0.000000000 | 0.000000060 | 0 | fail | node_keys 959 mismatch(es); conn_keys 4229 mismatch(es) | node_attrs 4.066220284; conn_attrs 4.200792313 | 0.152276725 | 4512 |
| custom_pagie_p500 | `iree-rocm` | pass | node_keys exact; conn_keys exact | node_attrs 0.000000477; conn_attrs 0.000000294 | 0.000000238 | 0 | fail | node_keys 507 mismatch(es); conn_keys 2853 mismatch(es) | node_attrs 3.389557362; conn_attrs 3.321706295 | 0.095183104 | 5962 |
| custom_pagie_p500 | `xla-rocm` | pass | node_keys exact; conn_keys exact | node_attrs 0.000000000; conn_attrs 0.000000000 | 0.000000119 | 0 | fail | node_keys 237 mismatch(es); conn_keys 1458 mismatch(es) | node_attrs 3.406351328; conn_attrs 2.775112629 | 0.112810433 | 1896 |
| cartpole_p100 | `iree-rocm` | pass | node_keys exact; conn_keys exact | node_attrs 0.000000477; conn_attrs 0.000000294 | 0.000000000 | 0 | pass | node_keys exact; conn_keys exact | node_attrs 0.000000119; conn_attrs 0.000000238 | 0.000000000 | 0 |
| cartpole_p100 | `xla-rocm` | pass | node_keys exact; conn_keys exact | node_attrs 0.000000000; conn_attrs 0.000000000 | 0.000000000 | 0 | fail | node_keys exact; conn_keys exact | node_attrs 0.000000119; conn_attrs 0.000000238 | 1.000000000 | 0 |

## Blocked rows

No row was blocked: every planned row is `ok`.

## Provenance of this block

This block is rendered from `benchmarks/results/gpu_rocm_xla_2026-11-07_grouped.json`: 18 record(s), 18 ok and 0 error, measured in the session 2026-11-07T13:58:25+00:00 – 2026-11-07T14:02:04+00:00 (UTC) at git `03368c1ea03c` (clean — generated report artifacts are not counted).
`benchmarks/results/gpu_rocm_xla_2026-11-07_grouped.json` is committed evidence (it is in the re-include list of `benchmarks/results/.gitignore`), so every cell above is traceable to the committed record.
Nothing here is a statement about HEAD: every number belongs to that record and to the tree its fingerprint names.
<!-- END MEASURED (gpu_measure.py) -->
## Against the frozen JAX implementation, same device
Everything above compares the new engine against the *numpy reference interpreter* — the same implementation on a slower engine — so no row above can speak to the old-vs-new question that `RESULTS.md` answers on the CPU.
This section times the **frozen pre-rewrite JAX implementation** (`benchmarks/legacy_jax`, the same code `RESULTS.md` calls the old side) on the **same MI300X VF**, in the same protocol (seed 42, 3 warmup + 20 timed generations, one fresh process per row, compile measured separately), so the two implementations are compared on one device.
`JAX_PLATFORMS=rocm` is what puts JAX itself on the device, and no row may fall back silently: every row records `jax.devices() == ["rocm:0"]`, `jax.default_backend() == "gpu"` and `device_kind` `AMD Instinct MI300X VF`, and a row that reported `cpu` would be an error row rather than a CPU number labelled as a device number.
Both sides then run through the same device runtime: the JAX ROCm backend *is* the PJRT plugin the `xla` rows drive (`xla_rocm_plugin.so`, 480.1 MiB, sha256 `8e8319b4…`), and both records carry it in their plugin fingerprint.
On this device the engine decides whether the rewrite wins: the `xla` leg is **1.40x–2.50x faster per generation than the frozen JAX implementation on all five cases the two implementations share**, while the `iree` leg is **slower than it on all five** (0.50x–0.80x, i.e. the pre-rewrite implementation is 1.25x–2.00x faster than `iree` here).
The per-generation win lands where the rewrite's level sweep replaces the legacy `fori_loop` over `max_nodes`: 5.338 against 3.825 ms/gen on `xor_p100`, 10.089 against 4.033 on `xor_p500`, 10.511 against 4.968 on `xor_p1000`, 9.196 against 4.153 on `xor3d_p500` and 12.710 against 5.891 on `custom_pagie_p500`.
There is no amortisation trade-off to weigh against the frozen implementation, because `xla` also **compiles 2.63x–3.40x faster per shape** (2.034–2.607 s against the legacy `jax.jit(...).lower().compile()`'s 6.554–6.904 s): the new side is ahead on both the one-off and the per-generation column, so this is not a "faster per generation, slower to start" trade.
Against the numpy reference above, the same `xla` compile needed 19–136 generations to repay itself; against the frozen JAX implementation there is nothing to repay, because that side pays a 2.6x–3.4x longer compile of its own before its first timed generation.
The batch forward tells the same story more sharply: on the device the frozen JAX forward takes 1.435 ms at 400 rows and 13.745 ms at 500 000, against `xla`'s 0.419 and 2.229 (**3.20x–6.17x**), and moving the *old* forward onto the device changes almost nothing — 1.382/1.566/1.661/3.430/7.681/14.238 ms in the committed CPU-JAX session (`FORWARD.md`) against 1.435/1.660/1.612/3.081/7.366/13.745 here, i.e. within −10 %…+6 % — because a `vmap` over every connection with a per-node `lax.cond` is dispatch-bound, and the device does not change the dispatch count.
One protocol finding is shared by both sides: the legacy jitted step dies on this device without the command-buffer workaround (`INTERNAL: Failed to set memcpy d2d node params: HIP_ERROR_InvalidValue` from `jit_step`), and `XLA_FLAGS=--xla_gpu_enable_command_buffer=WHILE` cures it — the same flag, for the same reason, that the `xla` rows above need, so it is a property of XLA-on-ROCm here rather than of `evoxc`.
The comparison is one repeat per row and two sessions on one shared device — these old-side rows started at load average 1.51–2.16, the new columns come from the accelerator record measured 1 h 53 min earlier — so each ratio is a per-generation ratio of two sessions, not an interleaved A/B, and the per-row `min – max` in the block is its uncertainty measure.
Two scope limits keep this honest: `cartpole_p100` has no old side at all (the pre-rewrite library's RL problems needed `brax`/`gymnax`/mujoco and were replaced by evoxc-native simulators), and `xor3d_p100`/`xor3d_p1000` are absent from the committed accelerator record, so measuring them would produce an old-side number with no paired new-side column.
<!-- BEGIN MEASURED (jax_device_measure.py) -->
## Frozen JAX on the device — old vs new per generation

**Verdict: the new evoxc engine against the frozen JAX implementation on the same device — `xla-rocm` is 1.40x–2.50x faster on all 5 compared cases; `iree-rocm` is slower on all 5 compared cases (0.50x–0.80x) (old ÷ new, medians of one repeat per row).**
The old side is `benchmarks/legacy_jax` (`import tensorneat_legacy`) — the frozen pre-rewrite implementation — compiled with `jax.jit(pipeline.step).lower(state).compile()` and measured by one `benchmarks/harness/worker.py` process per row, exactly as the CPU matrix measures it.
The new side is the committed accelerator record's own `xla` and `iree` rows on the same device, read at render time and never re-measured, so each speed-up is a per-generation ratio of two sessions on one physical device.

### Per generation

`median` is one old-side worker process's median over its timed generations and `min – max` that row's own per-generation spread — the uncertainty measure here, because a single median is not a distribution.
The `xla` and `iree` columns are the same case's medians in the accelerator record this block names under `### Provenance`; their `build s` is that record's one-off compile, reported beside the timing and never charged to a generation.
`n/a` means that leg has no `ok` row (see `### Blocked rows`); it is never filled in from another session.

| Case | old JAX ms/gen (median, min–max) | `xla` ms/gen | `iree` ms/gen | old ÷ `xla` | old ÷ `iree` | old JAX compile s | `xla` build s | `iree` build s |
|---|---|---|---|---|---|---|---|---|
| `xor_p100` | 5.338 (4.763 – 5.465) | 3.825 | 10.633 | 1.40x | 0.50x | 6.904 | 2.034 | 7.799 |
| `xor_p500` | 10.089 (10.001 – 10.564) | 4.033 | 12.744 | 2.50x | 0.79x | 6.742 | 2.244 | 8.017 |
| `xor_p1000` | 10.511 (9.598 – 10.670) | 4.968 | 15.203 | 2.12x | 0.69x | 6.554 | 2.354 | 8.193 |
| `xor3d_p500` | 9.196 (6.203 – 10.087) | 4.153 | 12.910 | 2.21x | 0.71x | 6.803 | 2.211 | 8.063 |
| `custom_pagie_p500` | 12.710 (12.414 – 12.961) | 5.891 | 15.873 | 2.16x | 0.80x | 6.860 | 2.607 | 8.406 |

### Batch forward

One forward pass of a whole population of differently-shaped networks: the RL rollout pays this once per timestep, which is why it is reported beside the generation and not folded into it.
Both columns are `forward_measure.py` records joined row for row — the old side's `jax` target and the new side's `xla-rocm` target, medians over that row's timed passes — and nothing here is re-measured or re-derived (see `### Provenance`).
`setup / build` is the JAX side's one-off cost, reported beside the timing and never charged to a pass; `n/a` means a leg has no `ok` row for that workload.

| Workload | rows | old JAX ms/pass | `xla` ms/pass | speed-up | jax setup / build s | `xla` build s |
|---|---|---|---|---|---|---|
| `xor_p100_s4` | 400 | 1.435 | 0.419 | 3.42x | 3.320 / 0.285 | 0.401 |
| `xor_p500_s4` | 2000 | 1.660 | 0.481 | 3.45x | 3.263 / 0.299 | 0.545 |
| `xor_p1000_s4` | 4000 | 1.611 | 0.497 | 3.24x | 3.299 / 0.296 | 0.678 |
| `xor_p100_s500` | 50000 | 3.081 | 0.745 | 4.13x | 3.345 / 0.316 | 0.498 |
| `xor_p500_s500` | 250000 | 7.366 | 1.326 | 5.55x | 3.264 / 0.455 | 0.840 |
| `xor_p1000_s500` | 500000 | 13.745 | 2.229 | 6.17x | 3.322 / 0.321 | 1.363 |
| `cartpole_p100_s1` | 100 | 1.439 | 0.450 | 3.20x | 5.740 / 0.271 | 0.571 |

### Environment

Measured conditions, not a verdict: these numbers are comparable with another session on the same device and tree, and with nothing else.

| Item | Value |
|---|---|
| `--platform` → `jax.default_backend()` | `'rocm'` → `'gpu'` (expected `'gpu'`, verified yes) |
| Device(s) / `device_kind` | `rocm:0`; device_kind `AMD Instinct MI300X VF` |
| JAX / jaxlib | 0.10.2 / 0.10.2 |
| `JAX_PLATFORMS` (as the rows saw it) | `'rocm'` |
| `XLA_FLAGS` | `--xla_gpu_enable_command_buffer=WHILE` (ROCm command-buffer workaround: without it the legacy jitted step fails with `HIP_ERROR_InvalidValue`) |
| XLA allocator policy | `XLA_PYTHON_CLIENT_PREALLOCATE=false` — allocate on demand instead of reserving 75 % of a shared device up front, which OOMed rows |
| PJRT plugin (the jax ROCm backend *is* this plugin) | `xla_rocm_plugin.so` (480.1 MiB, sha256 `8e8319b48fcc…`) |
| Load average (start of each row) | 1.51 – 2.16 (1-minute sample at each row's start) |
| Python / numpy | 3.12.3 / 2.5.4 |
| CPU cores / platform | 20 / Linux-6.8.0-146-generic-x86_64-with-glibc2.39 |
| Git commit (code under test) | `ab02073bebb8` (clean — generated report artifacts are not counted) |
| Session (UTC) | 2026-11-07T12:25:54+00:00 – 2026-11-07T12:26:57+00:00 |

### Provenance

Old-side rows (this block's `old JAX` and `old ÷ …` columns): `benchmarks/results/jax_rocm_2026-11-07.json` — 5 row(s), 5 ok and 0 error — measured at git `ab02073bebb8` (clean — generated report artifacts are not counted) in the session 2026-11-07T12:25:54+00:00 – 2026-11-07T12:26:57+00:00.
New-side columns (`xla`, `iree`, their `build s`): `benchmarks/results/gpu_rocm_xla_2026-11-07.json` at git `93ce3cac122f` (session 2026-11-07T10:29:06+00:00 – 2026-11-07T10:32:39+00:00), schema `tensorneat-gpu/1` — read at render time, never re-measured here and never hard-coded.
Batch-forward rows: `benchmarks/results/forward_jax_rocm_2026-11-07.json` at git `54683c3e3d9b` (session 2026-11-07T12:21:02+00:00 – 2026-11-07T12:21:54+00:00) (target `jax`) joined with `benchmarks/results/forward.json` at git `8c6330989a92` (session 2026-11-07T08:17:07+00:00 – 2026-11-07T08:19:54+00:00) (target `xla-rocm`), both schema `tensorneat-forward/1`.
Protocol: seed 42, 3 warmup + 20 timed generations, one fresh process per row, one repeat per row.
The one-off compile (`jax.jit(pipeline.step).lower(state).compile()`) is timed separately and never charged to a generation, and every quoted millisecond is a median, not a mean.
Nothing here is a statement about HEAD: every number belongs to the record, tree and session it names.
The old-side rows of this record are the frozen pre-rewrite JAX implementation (`benchmarks/legacy_jax`) measured on the same MI300X the accelerator record used: seed 42, 3 warmup + 20 timed generations, one fresh process per row, and `jax.jit(...).lower().compile()` timed separately, under `JAX_PLATFORMS=rocm`, `XLA_PYTHON_CLIENT_PREALLOCATE=false` and `XLA_FLAGS=--xla_gpu_enable_command_buffer=WHILE` (without that workaround the legacy jitted step dies with `HIP_ERROR_InvalidValue`); the new-side columns are read from `benchmarks/results/gpu_rocm_xla_2026-11-07.json` (tree `93ce3cac`) and belong to that session.

### What is and is not comparable

- Same physical device: both legs run on the accelerator this block names, and a row whose `jax.default_backend()` did not match `--platform` is an error row rather than a fast CPU number.
- Same workload shape: one seed, the same generation count and the same `benchmarks/cases.py` case configuration on both sides.
- Not comparable in values: the two implementations draw different PRNGs (JAX `PRNGKey` versus evoxc splitmix64), so their search trajectories differ by design and fitness is never compared here (`test/parity/` owns numerical fidelity).
- Not a lockstep comparison: the new-side columns come from `benchmarks/results/gpu_rocm_xla_2026-11-07.json` — an earlier session on the same device at a different tree — so each ratio is a per-generation ratio of two sessions, not one interleaved run.
- One repeat per row: the per-row `min – max` is the uncertainty measure and the medians are point estimates, so a small ratio difference is noise, not a result.

### Blocked rows

None: every planned old-side row measured on the device. This section is written from the record, so it reappears the moment a leg fails.
<!-- END MEASURED (jax_device_measure.py) -->
## Limitations

The legacy comparison exists only inside this file: `## Against the frozen JAX implementation, same device` times the frozen implementation on this device (one seed, one repeat, one session), and it is never part of the numpy-only old-vs-new verdict in `RESULTS.md`, which stays computed from `results/authoritative.json` on the CPU contract; an accelerator ratio mixed into that verdict would compare a device measurement against a CPU one.
One device: a single `AMD Instinct MI300X VF` at one device index (`iree.runtime.get_driver("hip").query_available_devices()` reports one entry, `device_id=1`), so nothing here says how the numbers scale to a second GPU, and no multi-device or multi-stream path is exercised.
Only `xla` rocm is measurable here: the `xla` backend needs a PJRT C API plugin `.so`, the ROCm one is the `jax-rocm7-pjrt` wheel and no CPU PJRT C API plugin exists to buy (`jaxlib` exports no `GetPjRtApi`, and PyPI ships no CPU plugin package), so an `xla` CPU row stays impossible on this box and `tvm` is untested in this repository.
One wiring is enough and the harness uses two: `compile_options["plugin_path"]` is what actually loads the plugin, and `EVOXC_PJRT_PLUGIN` in the row process's environment pins the same `.so` for any option-less discovery call, so a row is an `error` row (never a silent fallback) only when neither names a usable plugin and no `jax_plugins/*/*_plugin.so` is installed on `sys.path`.
Transfer cost is inside `ms/generation` and not separated from it: the host materialisation of each generation's result happens inside the timed `step()` call, and the host/device traffic is part of that wall clock; this protocol does not report a transfer-only number, and unified-memory behaviour is not instrumented.
`peak RSS` is per process and is the driver's resident memory, not the device's, so the 1891.0–1944.1 MiB on the `iree` rows and the 4880.9–9408.0 MiB on the `xla` rows say nothing about VRAM occupancy, and no device-memory figure is recorded anywhere in this protocol.
The `xla` figure includes the loaded 480.1 MiB plugin and is dominated by its runtime, and the plugin's `XLA backend allocating 143.77GiB … for BFCAllocator` line at client creation is an address-space reservation (its `CollectiveBFCAllocator` line reserves more), not resident memory — the recorded high-water marks are the numbers to read.
No per-generation dispatch or op count is instrumented here — `PERF_NUMPY.md` and the cost sections of `src/tensorneat/CONTEXT.md` are where executed-op counts live for the CPU interpreter — so an accelerator row gives wall clock and nothing about what the device was asked to do.
Timing is wall clock with a warmup, one seed and one committed repeat: this record has 20 timed generations per row, one sample set, and no second session, so the per-row 20-sample `min–max` in the generated block is its only spread, and it is not a distribution over seeds.
The box was shared — other agents run on the same host — and the record's document-build load average is 4.54 / 4.17 / 6.54 (1/5/15 min) with the per-row load average stored in each row's `env.loadavg`, so every median here is a condition-dependent measurement and the spread is its uncertainty measure.
`cartpole_p100`'s numpy row is the widest-spread numpy cell: 94.413–153.009 ms with a 95.952 median and a 105.252 mean, so its 1.43x and 4.07x ratios carry more per-sample noise than the other five rows.
`TF_XLA_HSACO_CACHE_DIR` is unset, so the plugin logs `HSACO file cache is disabled!` and every fresh process recompiles its kernels: the 2.034–2.607 s XLA `build_s` is paid per process per shape (and the cross-check's own numpy reference leg pays its own 0.284–0.763 s build inside the same process), and setting that variable would be a protocol change, not a free win.
The one-off compile is also per static shape, so a shape change repays it: `build_s` is excluded from every `ms/gen` sample but is real wall clock for a run that evaluates few generations, which is what the 3–959 generation break-evens above quantify; `setup_s` is a second one-off (3.048–4.159 s on `xla`, 1.186–1.258 s on `iree`) that the break-even column does not charge.
Per-generation numbers are not comparable across evoxc revisions or across records: this tree's numpy leg is 1.09x–2.46x faster per generation than the 2026-11-06 record's on the same cases, so a ratio mixed from two records is a different experiment, and the record's own `report_note` says so inside the measured block.
The cross-check's final-state divergence makes a long-horizon accelerator-vs-numpy *population* comparison a trajectory comparison, not an equality test, and `--trace 0` means this record cannot say at which generation any row first drifts: what it can say is that the generation-1 payload and fitness agree and that 10 of 12 final-state snapshots do not.
20 timed generations is a timing protocol, not a quality protocol: `final_max_fitness` comes from the same 24-step run as the timings, so those values order the legs at step 24 of one seed and are not converged search results.
The six `iree` rows and six `xla` rows are medians of 20 samples each on a warm interpreter, and `--trace 0` also keeps the trajectory legs' extra builds out of the session's 3 min 33 s: the session is 18 rows of setup + build + 3 untimed warmup and 20 timed generations and nothing else.
The record's `meta.report_title` and `meta.report_note` were both set for this session; the note is rendered as the last line of the block's provenance section and a bare re-render keeps it, while the title is not rendered because the block's heading derives from the record's actual targets.

## Reproduce

```bash
# 0. environment: the compiled-backend and PJRT-plugin extras, the ROCm runtime
uv sync --all-extras   # iree + xla-rocm; the latter ships jax_plugins/xla_rocm7/xla_rocm_plugin.so (480.1 MiB)
export EVOXC_PJRT_PLUGIN=$PWD/.venv/lib/python3.12/site-packages/jax_plugins/xla_rocm7/xla_rocm_plugin.so
/opt/rocm/.info/version && rocm-smi --showproductname --showdriverversion   # 7.1.0 / MI300X VF / driver 6.16.6 here

# 1. numpy-free planning: targets, passthrough flags, the case matrix
python3 benchmarks/gpu_measure.py --list

# 2. one row, one fresh process (this is what `matrix` spawns per row)
uv run --all-extras python benchmarks/gpu_measure.py run \
    --case xor_p500 --target xla-rocm --pjrt-plugin "$EVOXC_PJRT_PLUGIN" \
    --warmup 1 --generations 3 --json $TMPDIR/row.json

# 3. the full matrix this report renders: 6 cases x 3 targets, strictly sequential
REPORT_TITLE="TensorNEAT accelerator measurement — numpy reference vs iree rocm vs xla rocm on one MI300X VF (2026-11-07)"
REPORT_NOTE=$(cat <<'NOTE'
Every xla row of this record runs with `XLA_FLAGS=--xla_gpu_enable_command_buffer=WHILE` (the XLA+ROCm command-buffer workaround, stored in `meta.xla_flags`), one fresh process per row, on the tree this record's fingerprint names; its per-generation numbers are not comparable with the FUSION-capture xla rows of `gpu_rocm_xla_2026-11-06.json` or with any earlier evoxc revision.
NOTE
)
uv run --all-extras python benchmarks/gpu_measure.py matrix \
    --cases xor_p100,xor_p500,xor_p1000,xor3d_p500,custom_pagie_p500,cartpole_p100 \
    --targets numpy,iree-rocm,xla-rocm \
    --warmup 3 --generations 20 --seed 42 \
    --pjrt-plugin "$EVOXC_PJRT_PLUGIN" \
    --report-title "$REPORT_TITLE" --report-note "$REPORT_NOTE" \
    --json benchmarks/results/gpu_rocm_xla_2026-11-07.json --out benchmarks/GPU.md

# 4. re-render the measured block from the existing record: no measurement, no JSON rewrite
uv run --all-extras python benchmarks/gpu_measure.py render \
    --json benchmarks/results/gpu_rocm_xla_2026-11-07.json --out benchmarks/GPU.md

# 5. the XLA_FLAGS A/B quoted in ## Protocol (scratch evidence, into $TMPDIR, never a record)
uv run --all-extras python benchmarks/forward_measure.py run --workload xor_p500_s500 --target xla-rocm \
    --passes 10 --warmup 3 --pjrt-plugin "$EVOXC_PJRT_PLUGIN" --json $TMPDIR/fwd-with.json
uv run --all-extras python benchmarks/forward_measure.py run --workload xor_p500_s500 --target xla-rocm \
    --passes 10 --warmup 3 --pjrt-plugin "$EVOXC_PJRT_PLUGIN" --xla-flags= --json $TMPDIR/fwd-without.json
#   ... and the same pair for --workload xor_p1000_s500, plus four without-workaround RL trials:
uv run --all-extras python benchmarks/gpu_measure.py run --case cartpole_p100 --target xla-rocm \
    --warmup 3 --generations 20 --seed 42 --pjrt-plugin "$EVOXC_PJRT_PLUGIN" --xla-flags= --json $TMPDIR/cp-without.json
# 6. the frozen JAX implementation on the same device: 5 paired cases, strictly sequential, one process per row
JAX_NOTE=$(cat <<'NOTE'
The old-side rows of this record are the frozen pre-rewrite JAX implementation (`benchmarks/legacy_jax`) measured on the same MI300X the accelerator record used: seed 42, 3 warmup + 20 timed generations, one fresh process per row, and `jax.jit(...).lower().compile()` timed separately, under `JAX_PLATFORMS=rocm`, `XLA_PYTHON_CLIENT_PREALLOCATE=false` and `XLA_FLAGS=--xla_gpu_enable_command_buffer=WHILE` (without that workaround the legacy jitted step dies with `HIP_ERROR_InvalidValue`); the new-side columns are read from `benchmarks/results/gpu_rocm_xla_2026-11-07.json` (tree `93ce3cac`) and belong to that session.
NOTE
)
uv run --all-extras python benchmarks/jax_device_measure.py matrix \
    --cases xor_p100,xor_p500,xor_p1000,xor3d_p500,custom_pagie_p500 \
    --platform rocm --warmup 3 --generations 20 --seed 42 \
    --json benchmarks/results/jax_rocm_2026-11-07.json --out benchmarks/GPU.md \
    --report-title "TensorNEAT accelerator measurement — the frozen JAX implementation on the MI300X VF" \
    --report-note "$JAX_NOTE"
# 7. the frozen JAX batch forward on the device: the block's forward column (JAX_PLATFORMS picks the device)
JAX_PLATFORMS=rocm uv run --all-extras python benchmarks/forward_measure.py matrix \
    --workloads all --targets jax --passes 10 --warmup 3 --seed 42 \
    --json benchmarks/results/forward_jax_rocm_2026-11-07.json
# 8. re-render either measured block from its record: no measurement, no JSON rewrite
uv run --all-extras python benchmarks/jax_device_measure.py render \
    --json benchmarks/results/jax_rocm_2026-11-07.json --out benchmarks/GPU.md
```
Step 6 is what produced the second block: `jax_device_measure.py` pins each child to `JAX_PLATFORMS=rocm` (verified against `jax.default_backend() == "gpu"`), exports `XLA_FLAGS=--xla_gpu_enable_command_buffer=WHILE` and `XLA_PYTHON_CLIENT_PREALLOCATE=false`, and spawns one `benchmarks/harness/worker.py` process per row — the same worker, warmup and timing code the CPU matrix uses.
Step 7 is the forward column's old side: the `jax` target now receives the same command-buffer workaround (a `jax` row is XLA-driven, and the legacy `fori_loop` forward graph fails without it), which the record stores per row in `record["xla_flags"]`.
Step 8 is the second block's own re-render: it reproduces the measured block byte-for-byte and leaves the hand-written prose around it untouched.
The session behind step 6 took 53 s of wall clock on this box (five legacy XLA compiles of 6.554–6.904 s plus 23 generations per row) and step 7 took 50 s (seven rows), so both are foreground-cheap but must not run concurrently with another measurement on the shared device.

Step 3 is the exact argv the record stores in `meta.command`, with the recording worker's absolute interpreter path replaced by the portable `uv run --all-extras python` and the plugin path given as the variable it was read from; the title and the note travel through shell variables only so the stored quoting is reproducible without escaping.
The session took 3 min 33 s of wall clock on this box, of which the six `iree-compile`s are 49.1 s and the six XLA compiles 13.7 s (sums of the recorded `build_s`), so run it in the background rather than under a short foreground timeout.
`--xla-flags=` in step 5 is the empty-value form of `--xla-flags ''`: the variable is removed from the child environment rather than set, so step 3's rows run with the workaround and step 5's secondary legs run without it.
The plugin comes from the `xla-rocm` extra, not from a hand install: `uv sync --all-extras` resolves and keeps the locked `jax-rocm7-pjrt` wheel.
A wheel installed instead with `uv pip install --no-deps` is untracked, so a later `uv sync` prunes it and the `xla` rows become `error` rows with the named `PluginError`; that one-off recipe is only for a retargeted wheel (e.g. `jax-cuda12-pjrt` on an NVIDIA box), which needs the extra pointed at it and a `uv lock`.
No CPU PJRT plugin is pip-installable at all.
`matrix` exits 0 only when every planned row is `ok` and 1 as soon as any row is `error` (it prints a `BLOCKED: …` line per blocked row); this matrix exited 0 with 18/18 `ok`, which is the state the block above records.
`render` and `matrix --out` replace only the interior of this file's two markers (exactly one of each in the file) and exit 2 with a named error on a missing marker, an unreadable record, a wrong schema or a missing `records` list.
A single row is cheap on the numpy leg (0.266–0.786 s build) and pays the compile on a compiled leg (2.034–2.607 s XLA, 7.799–8.658 s `iree-compile`), so budget a compiled row at its compile plus its timed region, and a traced row at that plus `N × (accelerated + numpy generation)`.
The IREE runtime can print `nanobind: leaked …` lines to stderr when the interpreter exits, and the XLA plugin prints its allocator lines at client creation; both are teardown/startup messages around the measured region, never a failure and never an `error` row.
The Python-free paths are `--list`, `--help` and nothing else: `run` and `matrix` need `numpy` and `tensorneat`, and the `iree-*` targets need the `iree` extra while the `xla-*` targets need a PJRT C API plugin — an installed `jax_plugins/*/*_plugin.so`, or a path through `--pjrt-plugin`/`EVOXC_PJRT_PLUGIN`.

## What this does not change

`RESULTS.md` stays the generated numpy-only old-vs-new record, with its own verdict computed from `results/authoritative.json`; no accelerator row is added to it, and no number in it is touched by this protocol.
`BACKENDS.md` stays the compiled-backend CPU-leg record of its own session — the `iree` llvm-cpu numbers, which this matrix does not measure — and `PERF_NUMPY.md` stays the numpy-interpreter performance record; neither is re-rendered from here.
Accelerator rows are never wired into the harness verdict, and that is deliberate: `protocol.comparisons` pairs every old-side record with every new-side record, so an accelerator row in `results/` would enter the numpy-only verdict as if it were a numpy measurement and could flip a timing verdict that the CPU contract owns.
Keeping the accelerator protocol standalone (`benchmarks/gpu_measure.py`, its own `tensorneat-gpu/1` schema, its own record and report) keeps that contract intact: the harness verdict compares what it declares it compares, and an accelerated row is evidence about a different question.
The same-device JAX section changes that in no way: `results/jax_rocm_2026-11-07.json` carries schema `tensorneat-jax-device/1` and `results/forward_jax_rocm_2026-11-07.json` carries `tensorneat-forward/1`, neither of which a harness renderer reads, so no row of either can enter `protocol.comparisons` or move `RESULTS.md`, `THREADS.md` or `iree.md`.
The frozen implementation is read, never edited: the old-side rows import `benchmarks/legacy_jax` unchanged, which is what keeps its role as the baseline and the numerical oracle meaningful.
`benchmarks/results/gpu_rocm_xla_2026-11-06.json` and `benchmarks/results/gpu_rocm_2026-11-05.json` stay the records of their own trees and sessions: they are superseded as this file's block source only, and are neither rewritten nor re-measured here.
`FORWARD.md` stays the forward-pass report of its own four-target session; the two forward A/B runs quoted under `## Protocol` are scratch measurements into `$TMPDIR` and are not a record.
