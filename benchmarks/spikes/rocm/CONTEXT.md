# spikes/rocm — ROCm (MI300X) cost model of one TensorNEAT generation
## Intent
Hold the ROCm evidence for this node: the pre-3.13 abort-era report (`ROCM_FINDINGS.md`, IREE 3.11: nothing ran end to end) and the per-generation cost model below, measured on the tree that does run (IREE 3.13).
`ROCM_FINDINGS.md` + `repro_minimal.py` + `rocm_a_pipeline.py` / `rocm_abort_ladder.py` / `rocm_b_stages.py` / `rocm_c_kernels.py` / `rocm_run_all.sh` / `rocm_support.py` are the 2026-11-05 pre-fix era record: their per-stage numbers are directional only, because IREE 3.13 lifts the `while`/`scan` aborts they worked around.
The cost model below supersedes `ROCM_FINDINGS.md` for anything about a whole generation on ROCm; nothing here is a pass/fail benchmark row (`../CONTEXT.md` and `../../GPU.md` own that).
## Measured cost model (tree `3c0eb95`, 2026-11-05 15:21–15:45 UTC, IREE 3.13.0, ROCm 7.1, MI300X VF, rocprofv3 1.0.0)
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
## Ceiling (current architecture, one graph per generation)
`xor_p500`: device 42.6 ms is the floor once host submission stops binding, so the reachable wall is ≈43.5 ms against today's 47.1 ms — halving tell's 2 737-dispatch host cost alone buys ≈3 ms and then device time binds; below that the speciation distance matrix (201² per individual per scan iteration) and the per-genome 252-key sort must shrink.
`cartpole_p100`: device is 31.8 ms, so the reachable floor is ≈33 ms against today's 164 ms; halving tell's host cost leaves ≈134 ms because the 200-step rollout's ~5 460 dispatches (0.67 ms per step) are the binding term.
`custom_pagie_p500`: host-bound at ≈60 ms with 44.3 ms of device time, so the floor is ≈45 ms.
`xor_p100`: host-bound (14.0 ms host vs 14.3 ms device) with a 1 623-dispatch gated-speciation graph, so the floor is ≈15 ms.
## Compile-flag sweep (report only, `xor_p500` + `custom_pagie_p500`, 3 warmup + 5 timed)
| configuration | `xor_p500` ms/gen | `custom_pagie_p500` ms/gen | build s | dispatches (2 gens) |
|---|---|---|---|---|
| baseline | 46.05 | 59.76 | 8.5 / 12.9 | 5 664 / 6 409 |
| `opt_level="O3"` | 46.14 | 60.22 | 8.5 / 13.0 | 5 664 / 6 409 |
| `iree_compile_args=["--iree-dispatch-creation-enable-aggressive-fusion"]` | 46.09 | 59.90 | 8.6 / 13.0 | 5 664 / 6 409 |
No flag moves wall time or build time, and none changes the executed dispatch count — the only ROCm lever that matters is the dispatch count, which these flags do not touch on these graphs.
`opt_level` is a compile-stage evoxc option; `dispatch_formation` exists in the newer `/root/evoxc` checkout but **not** in the pinned v0.1.1 adapter the tree installs (its compile-stage set is `target_backends`/`iree_compile_args`/`opt_level`/`vulkan_target`), so aggressive fusion has to be passed as the raw `--iree-dispatch-creation-enable-aggressive-fusion` flag; the lower stage rejects a `device` keyword.
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
- No committed artifact backs the cost-model table above: the probe scripts were session scratch scripts outside the repo (this node is read-only apart from CONTEXT.md), so reproducing a row means re-running the prepare/run split described in `## Method and its traps`.
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
