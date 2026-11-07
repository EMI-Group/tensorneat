# spikes/rocm — ROCm (MI300X) cost model of one TensorNEAT generation
## Intent
Hold the ROCm evidence for this node: the pre-3.13 abort-era report (`ROCM_FINDINGS.md`, IREE 3.11: nothing ran end to end) and the per-generation cost model below, measured on the tree that does run (IREE 3.13).
`ROCM_FINDINGS.md` + `repro_minimal.py` + `rocm_a_pipeline.py` / `rocm_abort_ladder.py` / `rocm_b_stages.py` / `rocm_c_kernels.py` / `rocm_run_all.sh` / `rocm_support.py` are the 2026-11-05 pre-fix era record: their per-stage numbers are directional only, because IREE 3.13 lifts the `while`/`scan` aborts they worked around.
The cost model below supersedes `ROCM_FINDINGS.md` for anything about a whole generation on ROCm; nothing here is a pass/fail benchmark row (`../CONTEXT.md` and `../../GPU.md` own that).
## Measured cost model (tree `3c0eb95`, 2026-11-05 15:21–15:36 UTC, IREE 3.13.0, ROCm 7.1, MI300X VF, rocprofv3 1.0.0)
Protocol: one `(case, target)` per process, `benchmarks/cases.py` geometry, seed 42, 3 warmup + 5–10 timed generations, `time.perf_counter` around one `evoxc.run` + host materialisation; device time from `rocprofv3 --kernel-trace` on a *single* warm generation (1 warmup + 1 timed, only the second segmented out).
| case | dispatches / generation | device ms / gen | wall ms / gen (`run` + copies) | non-device ms | non-device µs / dispatch |
|---|---|---|---|---|---|
| `xor_p100` | 1 623 | 14.28 | 28.27 (+0.48) | 14.0 | 8.6 |
| `xor_p500` | 2 828 | 42.62 | 46.34 (+0.80) | 3.7 | 1.3 |
| `xor_p1000` | 2 826 | 57.51 | 62.79 (+0.97) | 5.3 | 1.9 |
| `custom_pagie_p500` | 3 258 | 44.30 | 60.93 (+0.77) | 16.6 | 5.1 |
| `cartpole_p100` | 8 410 | 31.80 | 164.05 (+0.50) | 132.3 | 15.7 |
`device` is the sum of kernel durations in the warm generation; `non-device` is wall − device − the per-generation D2H cost, i.e. the part of `evoxc.run` not covered by GPU work.
Read it as `wall ≈ max(host_dispatch_total, device_total) + ≈0.8 ms` with a host dispatch cost of **≈17–19 µs per dispatch** wherever the host is the binding side: the per-dispatch column is low exactly where device time already exceeds the host submission time (`xor_p500`), and it is the *whole* wall where the GPU is idle (`cartpole_p100`: device is 19 % of the generation).
CPU time inside `evoxc.run` (`time.process_time`, all threads) is 1.45–1.8× the wall for every case — `xor_p500` 82.2 ms CPU against 46.0 ms wall, `cartpole_p100` 254.5 against 163.8 — so ≈25–30 µs of host CPU per dispatch is spent across ~1.6 cores; the limit is IREE VM/HAL submission work, not kernel duration.
Executed dispatches are 7.3× the static `hal.executable.export` count (386 for `xor_p500`) because the genome-forward level loop, both speciation scans and the RL rollout repeat their dispatch chains inside loop regions.
Phase split (measured by stubbing one phase out of the traced graph; each variant froze the population at generation 0, so the deltas are an initial-population decomposition, not an additive decomposition of the evolved run):
| case | tell | evaluate (incl. genome forward) | transform |
|---|---|---|---|
| `xor_p500` | 42.0 ms removed from 47.1 → 5.1 | 4.3 ms (47.1 → 42.8) | 2.5 ms whole-graph |
| `cartpole_p100` | 58.3 ms (164.6 → 106.3) | 138.9 ms (164.6 → 25.6), of which genome forward 63.5 ms and the rollout arithmetic the rest | 1.0 ms whole-graph |
Dispatch split of the same variant graphs (initial population): `xor_p500` transform 29, evaluate/genome-forward ~62, tell ~2 737; `cartpole_p100` transform 29, tell ~1 531, evaluate (the 200-step rollout) ~5 457.
The rollout's marginal cost is measured directly by shrinking `CartPole.max_step`: 50 → 62.0 ms, 100 → 94.1 ms, 200 → 162.3 ms, i.e. **0.67 ms per timestep** and ≈28.6 ms of fixed per-generation cost (≈27 dispatches per timestep).
Worst kernels by device time on `xor_p500` (share of the warm generation's 42.6 ms): `sort_500x252xi64` 5.5 ms (13 %, one sort per generation), the speciation distance family `reduction_500x201x201_{i64,i32...}` ×240 instances ≈ 20 ms, `sort_500x200xi32` 2.0 ms, `scatter_500x252` / `scatter_201000` / `scatter_500x100` ≈ 6.5 ms, and 333 `__amd_rocclr_copyBuffer` 1.6 ms; 2 257 of 2 828 kernels are shorter than 10 µs and the median kernel is 4.9 µs.
Phase attribution of those kernels (matching dispatch-shape suffixes across variant traces) puts 80.4 % of `xor_p500` device time in tell, 13.9 % in kernels shared by all three phases, and 4.7 % in evaluate.
`cartpole_p100` device time is spread thin instead: the two sorts are 10.7 + 3.7 ms, and 4 016 `copyBuffer` + 2 167 `slow_memcpy` + 2 066 `elementwise` dispatches at 1–2.5 µs each carry the rollout.
Per-generation host round trips cost `xor_p500` 0.69 ms for the four payload arrays (1.1 MB, 1.6 GB/s in step, 4.6 GB/s isolated), 0.074 ms for the `(P,)` fitness vector and 0.054 ms for the overflow scalar; every single `Tensor.to(cpu)` carries a fixed ≈47–60 µs regardless of size (a 8-byte scalar costs the same as 100 KB).
## Host round-trip cost model: the pinned 1-byte staging buffer (tree `81011094`, 2026-11-06 04:22–04:45 UTC, IREE 3.13.0, evoxc `be145ee` — then the pinned `v0.1.1` tag — ROCm 7.1, MI300X VF gfx942, rocprofv3 1.0.0)
Protocol: artifact prepare/run split (an unprofiled process traces+lowers+compiles the generation graph and `Executable.save`s it, the profiled process only `Executable.load`s and runs it, exiting with `sys.exit`); `rocprofv3 -r` (HIP runtime + memory + kernel); generation windows cut on `CLOCK_MONOTONIC` markers the profiled script prints around each `evoxc.run` + its one D2H.
Profiling inflates wall 1.6–2.0× at ~1.9 µs per extra HIP call, so the trace supplies **counts only** and every unit cost below was measured separately (ctypes on `libamdhip64`, unprofiled, medians of 100–300 calls).
**Mechanism**: every control-flow decision whose predicate lives on the device — a `while` trip condition or a `cond` branch predicate — is read back to the host through a **fresh 1-byte pinned, mapped host buffer**: `hipHostMalloc(1)` → `hipHostGetDevicePointer` → copy → `hipHostFree`.
Unit costs: that exact sequence costs **106.7 µs** (p10 104.7 / p90 111.8, n=300), size-independent to 64 KiB (`hipHostMalloc` 22.5 µs + `hipHostFree` 86.9 µs; 1 MiB → 184 + 159 µs); with the copy and its events a readback costs **~157 µs** end to end (`hipMemcpyAsync` D2H 4 KiB submit 11.9 µs, submit + `hipEventRecord` + `hipEventSynchronize` 19.2 µs).
A **static/counter** condition is host-counted instead: ~21 µs per iteration with an empty body, 57 µs with a fused body.
Per-generation counts (one warm generation, generation 8, seed 42; `wall` is the unprofiled 3-warm/5-timed median except cartpole 152.2 = the 200-step horizon):
| case | wall ms | kernels | device ms | `hipMemcpyAsync` | pinned 1-byte pairs | launches | eventRecord | eventSync | streamWait | ctxSet |
|---|---|---|---|---|---|---|---|---|---|---|
| `cartpole_p100` | 152.2 | 6 917 | 12.9–17.3 | 2 223 | **647** | 4 408 | 8 720 | 4 360 | 3 149 | 15 713 |
| `xor_p100` (gated) | 16.8 | 1 205 | 3.5 | 239 | **53** | 884 | 1 192 | 596 | 533 | 2 509 |
| `xor_p100` (ungated) | 10.5 | 1 602 | 4.5 | 201 | **13** | 1 345 | 618 | 309 | 249 | 3 163 |
| `xor_p500` (ungated, the geometry's own path) | 13.6 | 1 604 | 7.0 | 201 | **13** | 1 347 | 618 | 309 | 258 | 3 165 |
`cartpole_p100` closes to 133 ms of host HIP API time against a 152 ms wall: 647 × 106.7 µs = **69 ms of pinned alloc/free**, 2 223 copies ≈ 27 ms, 4 408 launches ≈ 18 ms, the event family ≈ 20 ms, device 12.9–17.3 ms mostly overlapped — **the missing ~110 ms is the pinned-alloc churn plus the copy/event churn of the same round trips, not device work and not the loop machinery**.
Per rollout iteration the arithmetic is 2.85 pinned-alloc pairs + 9.5 copies + 17.2 launches + 18.4 event syncs + 36.7 event records (0.65–0.69 ms/iteration measured: slope 0.694 between horizons 100→200 and 0.646 across 25→200 over a 23–28 ms fixed intercept).
The 2.85 pairs per rollout iteration are the genome level sweep: `genome/default.py`'s `while_loop` predicate is `level <= prepared.limit` with `limit` a **device** tensor, so it is data-dependent even though its value is trace-static.
`xor_p100`'s 53 pairs decompose as **2 per species slot (40 at `species_size` 20) + 13 fixed**: `species_size` 5 gives 23 pairs and 9.2 ms less wall, and `xor_p500` (`20 × 500 = 10 000 > GATED_ROWS`, i.e. ungated) gives exactly the 13 fixed pairs.
The 2-per-slot term is the `cond` that `_kernels.gated()` wraps each scan iteration in — **on ROCm a `cond` iteration costs ~157 µs and an extra unconditional dispatch ~2 µs, so the gate is a loss on an accelerator** even though it wins on the numpy interpreter.
Measured fix, no source change (the trace-time predicate reads the module global): `GATED_ROWS = 0` takes `xor_p100` from 16.8 to **10.5 ms/gen (1.60×)** with 1 602 kernels and 4.5 ms device, and takes `xor_p500` from 18.2 (forced gated) to 13.6 ms (1.34×).
Scratch-graph structural experiment (100-row f32, one graph per process, iree/rocm; `$TMPDIR`, not committed):
| form | wall ms | per iteration | copies | pinned pairs |
|---|---|---|---|---|
| flat 200 × 0 body ops | 4.20 | 21 µs | 4 | 1 |
| flat 200 × 20 / 40 / 60 body ops | 11.54 / 11.43 / 11.37 | **57 µs, saturated** | 204 | 1 |
| flat 600 × 20 body ops | 33.17 | 55 µs | — | — |
| nested 200 × (3 × 20 ops), static condition | 35.6 | 60 µs per inner | 604 | 1 |
| nested 200 × (3 × 20 ops), data-dependent condition | **157.7** | 263 µs per inner | 2 204 | 800 |
So **nesting costs +7 % at equal op count** (36.0 vs 33.2 ms) while a **data-dependent condition costs 4.4×** (157.7 vs 35.6 ms for identical ops), and per-iteration cost saturates at ~57 µs from 20 to 60 fused ops — extra statically unrolled device work is free inside a loop body.
The 263 µs is this ladder's per-inner-iteration cost on a 100-row `f32` scratch graph; the tree's canonical figure for the same mechanism is **~230 µs per policy query** on the `cartpole_p100` payload (`max_step` 200, pop 100, level sweep nested in the rollout loop), so the nested-`while` bound re-staging reads **230–263 µs** and neither number means anything without its payload.
Both measure one mechanism: IREE host-distributes a dispatch-bearing `stablehlo.while` (one `stream.cmd.execute` plus a blocking `hal.fence.await` and a `hal.buffer.load` per trip), and a nested loop re-stages its bound on every enclosing iteration; the ladder pays three inner trips per outer on its own 100-row graph, while a CartPole query pays one level sweep per timestep at that case's node and connection geometry.
Prediction for the flat formulation (level sweep unrolled over `max_depth`, mask-gated, one flat rollout loop): 200 × 57–90 µs + the 23–28 ms fixed part = **36–43 ms/gen against today's 152 ms (3.5–4.2×)**.
Break-even for statically unrolling the speciation scan instead of dropping its gate: the gate costs 6.3 ms/gen and the unroll costs +5.6–8.4 s of `iree-compile`, i.e. ~890–1 330 generations — still not worth it on a 13.6 ms generation, and the gate removal buys the same 6.3 ms for zero compile time.
## Numerical fidelity of the device-gated kernels (same tree and session)
`gpu_measure.py run --target iree-rocm --warmup 2 --generations 5` (8 steps, seed 42): gen-1 int keys **exact** on all three cases, float attrs ≤ 4.77e-7 and fitness ≤ 2.38e-7 (rtol 1e-5 / atol 1e-6), no NaN/`-inf` mask mismatch.
`cartpole_p100` passes even the final state (fitness pinned at 200.0, keys exact); `xor_p500` and `custom_pagie_p500` drift in the final state (fitness max |Δ| 0.1125 / 0.0836) as documented.
Lockstep from one seed at pop 500: gen-1 payload keys 0 mismatches and species leaves 8/10 equal (the two differ by ≤ 8.9e-8, float32 ulps), gen 2 has 5/10 species leaves equal and `species_count` 19 vs 18, gen 3 diverges the genomes (308 of 94 999 node keys); the RNG key is identical every generation, so this is not a PRNG-stream difference.
`--trace 8` reports `first_key_divergence 4`.
Running **speciate on one fixed bit-identical payload** gives **all 10 species leaves exactly equal** across numpy and rocm, so the device-gated distance/alignment kernels are numerically identical and the drift is a threshold amplification of float32 ulps: species membership is a discrete decision taken on a continuous distance.
The cross-check's generation-1 gate compares payload + fitness only, so a species-level ulp difference at generation 1 is invisible to it; comparing the species leaves at generation 1 is the gate that would cover speciate.
## Ceiling (current architecture, one graph per generation)
`xor_p500`: device 42.6 ms is the floor once host submission stops binding, so the reachable wall is ≈43.5 ms against today's 47.1 ms — halving tell's 2 737-dispatch host cost alone buys ≈3 ms and then device time binds; below that the speciation distance matrix (201² per individual per scan iteration) and the per-genome 252-key sort must shrink.
`cartpole_p100`: device is 31.8 ms, so the reachable floor is ≈33 ms against today's 164 ms; halving tell's host cost leaves ≈134 ms because the 200-step rollout's ~5 460 dispatches (0.67 ms per step) are the binding term.
`custom_pagie_p500`: host-bound at ≈60 ms with 44.3 ms of device time, so the floor is ≈45 ms.
`xor_p100`: host-bound (14.0 ms host vs 14.3 ms device) with a 1 623-dispatch gated-speciation graph, so the floor is ≈15 ms.
## Workgroup shape of every scatter dispatch (static census, tree `141974d`, IREE 3.13.0rc20261016, gfx942)
A static census — `evoxc.trace` → `evoxc.lower(backend="iree", target_backends=["rocm"])` → `iree-compile --compile-to=executable-configurations` plus a `--compile-to=hal` grid dump, nothing compiled for or run on the device — shows **every** `iree_linalg_ext.scatter` dispatch is one workgroup of one thread: `workgroup_size = [1, 1, 1]` with **no** `subgroup_size`, root-op `lowering_config = #iree_codegen.lowering_config<tile_sizes = [[]]>` (untiled), codegen pipeline `#iree_gpu.pipeline<Distribute>`, and a materialised grid of `workgroups([%c1, %c1, %c1])` at every dispatch site.
That is the rocprofv3 "(1,1,1)x1 thread" signature seen in the cost-model traces, and it holds for the three `common/tensorization.py` kernels in isolation and for a whole `xor_p500` generation graph.
`xor_p500` generation graph (`benchmarks/cases.py` geometry, seed 42): 10 001 618 bytes of StableHLO, 42 `stablehlo.scatter`, 5 `stablehlo.sort`, 6 `stablehlo.while`, 6 109 ops; 330 dispatch entry points over 458 dispatch sites, of which 20 are scatter executables reached from 41 sites (`iree-flow-deduplicate-executables` folds the 42 scatter ops onto those 20 bodies), and all 20 are single-workgroup/single-thread.
The other 72 single-workgroup entries are reduction/elementwise/copy dispatches (`reduction_20`, `reduction_500`, `slow_memcpy`, `BaseLowering` copies), not scatters.
Isolated kernels: `key_slot_table` 3 entry points (1 scatter), `edge_order` 12 (1), `incoming_table` 14 (1); each scatter is `[1,1,1]`/untiled/grid 1×1×1 while their elementwise/reduction/sort neighbours get real thread blocks (`[64,1,1]`, `[100,1,1]`, `[128,1,1]`, `[50,2,1]` with `subgroup_size = 64`).
Consequence: a 25 000- to 100 000-element scatter is executed serially by one work-item on one SIMD lane of one CU on gfx942, so the scatter family is not a tiling-tuning problem — it needs a different formulation (or a codegen path that tiles `iree_linalg_ext.scatter`).
IREE 3.13 names its GPU pipelines `#iree_gpu.pipeline<Distribute|TileAndFuse|BaseLowering>`; the older `LLVMGPUVectorDistribute`/`GPUDefault` names no longer appear in these dumps, so a parser for 3.11 artifacts does not read 3.13 ones.
An export's count region is `iree_tensor_ext.dispatch.workgroup_count_from_slice()` in 310 of the 330 entries and a constant `hal.return %c1, %c1, %c1` in 20 (`BaseLowering` copies plus `__builtin_splat_i64`); every scatter export is the from-slice form, so the 1×1×1 grid must be read from the HAL dump, not inferred from the export.
The static capture is exact: `Pipeline.setup()` + `Pipeline._generation_kernel(p._static_config(state), evoxc.tree_structure(state.species))` + `evoxc.trace` + `evoxc.lower(backend="iree", target_backends=["rocm"])` produced bytes identical (sha256 `829020ed…`) to the last executable a real `Pipeline.step` builds, so a census needs no device and no execution.
The census scripts are session scratch (not committed): `$TMPDIR/t1a117sub/{taskA_trace.py,taskB_capture.py,census.py}`, the latter joining `<k>.cfg.mlir` metadata with the `hal.command_buffer.dispatch … workgroups([…])` lines of `<k>.hal.mlir`.
## Compile-flag sweep (report only, `xor_p500` + `custom_pagie_p500`, 3 warmup + 5 timed)
| configuration | `xor_p500` ms/gen | `custom_pagie_p500` ms/gen | build s | dispatches (2 gens) |
|---|---|---|---|---|
| baseline | 46.05 | 59.76 | 8.5 / 12.9 | 5 664 / 6 409 |
| `opt_level="O3"` | 46.14 | 60.22 | 8.5 / 13.0 | 5 664 / 6 409 |
| `iree_compile_args=["--iree-dispatch-creation-enable-aggressive-fusion"]` | 46.09 | 59.90 | 8.6 / 13.0 | 5 664 / 6 409 |
No flag moves wall time or build time, and none changes the executed dispatch count — the only ROCm lever that matters is the dispatch count, which these flags do not touch on these graphs.
`opt_level` is a compile-stage evoxc option; `dispatch_formation` was not in the adapter this session measured (its compile-stage set was `target_backends`/`iree_compile_args`/`opt_level`/`vulkan_target`, the then-pinned `v0.1.1` tag), so aggressive fusion had to be passed as the raw `--iree-dispatch-creation-enable-aggressive-fusion` flag; the lower stage rejects a `device` keyword.
The current pin (`fe85a534`) does accept `dispatch_formation` as a compile option, so a flag sweep written today can use it instead of the raw flag — a different option on a different evoxc revision, never a re-reading of the table above.
Runtime flags are equally inert: `--hip_use_streams=true`, `--hip_allow_inline_execution=true`, `--hip_async_allocations=false` and their combination all leave `xor_p500` at 45.8–46.1 ms and `cartpole_p100` at 159–165 ms (`EVOXC_IREE_RUNTIME_ARGS`, load stage).
## Method and its traps
The IREE 3.13 Python runtime has no execution-tracing switch: `iree.runtime.flags` exposes only `FUNCTION_INPUT_VALIDATION`/`parse_flags`, the wheel's `_runtime.abi3.so` contains no `trace_execution`/`execution_trace` string, and `--hip_tracing=true` prints nothing, so per-dispatch data has to come from `rocprofv3`.
Under `rocprofv3` the IREE `local-task` driver cannot create its worker threads (`thread_pthreads.c:160: INTERNAL; thread creation failed with 22`), so a profiled process cannot run `Pipeline.setup` (it builds the setup graphs on CPU) or even `rt.get_driver("local-task")`; the `hip` driver is unaffected.
rocprofv3 also injects its tool library into the `iree-compile` child evoxc spawns, which then dies in consteval with the same thread error, so a profiled process must drop `LD_PRELOAD`/`ROCP_TOOL_LIBRARIES` from its environment before it compiles.
What works is a prepare/run split: an unprofiled process traces+lowers+compiles the generation graph (`evoxc.trace`/`lower`/`compile`, `artifact.save`), and a profiled process only `evoxc.Executable.load`s that artifact and runs it (mode `run` of the session's scratch script), which needs neither the compiler nor `local-task`.
The profiled process must exit cleanly (`sys.exit`, not `os._exit`): rocprofv3 writes its CSV at finalization, so an `os._exit(0)` produces no trace at all.
Only a *single* warm generation inside a trace is usable: the first generation in a process carries a ~23 ms one-time setup gap, and the gap between one `run` and the next is also ~23 ms of profiler artifact, so wall-minus-device must be taken from the unprofiled run and device time from the warm generation's kernels.
## Constraints
- Never quote a number here without the tree and session above; a re-measurement is a new record with a new date, not an edit of this table.
- These are cost-model measurements, not benchmark rows: nothing here may enter `../../RESULTS.md`/`GPU.md`, which only `run_benchmark.py`/`gpu_measure.py` write.
- The stub variants (tell/evaluate/forward replaced by a constant) are decomposition probes, never correctness evidence: they freeze the population and their graphs are not valid NEAT generations.
- Keep the scripts throwaway: `ROCM_FINDINGS.md` and its probes are a historical record and stay as they are; a new ROCm probe belongs in this directory as its own file with its protocol printed.
## Known Issues
- No committed artifact backs either cost-model table: the probe scripts are session scratch scripts outside the repo (this node is read-only apart from CONTEXT.md), so reproducing a row means re-running the prepare/run split described in `## Method and its traps`.
- The prepare/run split's profiled half runs the artifact outside `Pipeline`, so it measures `evoxc.run` + one D2H only: the host bookkeeping `Pipeline.step` adds (`to_numpy`, the overflow check) is not in the trace and is 0.1 % of a generation.
- `rocprofv3 -r` has no HIP-argument capture in 1.0.0, so copy **sizes** are inferred from the `__amd_rocclr_copyBuffer` grid sizes (2 KiB and 6 KiB dominate at `cartpole_p100`) and pinned-buffer sizes from `memory_allocation_trace.csv` (every per-readback allocation is 1 byte).
- `ROCm_FINDINGS.md` still says "no full ROCm generation runs today" and extrapolates ≈45–65 ms per generation; that forecast is now superseded by the measured 28–164 ms range above and should not be read as a current claim.
- The phase table's three deltas do not add up to the full generation: each stub freezes the population at generation 0, while the full run measures evolved (deeper, denser) genomes whose loops iterate more.
## Routing Table
| Path | Purpose |
|---|---|
| `./ROCM_FINDINGS.md` | Pre-3.13 (IREE 3.11) abort-era report: which construct died where, stage costs that did run, and the extrapolation that the model above replaces. |
| `./repro_minimal.py` | Minimal `evoxc`-level reproducer of the 3.11 HIP `while`/`scan` aborts (one variant per process). |
| `./rocm_support.py` | Shared helpers for the abort-era probes (pipeline factory at `cases.py` geometry, device hop, `PROBE` reporter). |
| `./rocm_a_pipeline.py`, `./rocm_abort_ladder.py`, `./rocm_b_stages.py`, `./rocm_c_kernels.py`, `./rocm_run_all.sh` | Abort-era probes: placement, construct ladder, per-stage cost, per-kernel micro-benchmarks, and their driver. |
| `../../GPU.md` | The measured accelerator benchmark matrix (numpy / iree llvm-cpu / iree rocm rows); the pass/fail record this node does not own. |
