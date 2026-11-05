# ROCm (MI300X VF) measurement / decomposition report

Environment measured 2026-11-05, one box: AMD Instinct MI300X VF, ROCm 7.1, IREE 3.11
(`iree.runtime.get_driver("hip").query_available_devices()` → one device, `device_id 1`;
`evoxc.Device("rocm", 0)` maps to it). Python 3.12 via `uv`, local `evoxc` checkout
(`uv pip install -e /root/evoxc --no-deps`, the ROCm staging fixes are there; the
committed tree stays on the pinned v0.1.1 API).

Reproduce everything:

```bash
export PATH=$HOME/.local/bin:$PATH
uv sync --all-extras
uv pip install -e /root/evoxc --no-deps
bash src/tensorneat/_rocm_spikes/rocm_run_all.sh        # logs/ next to the scripts
```

Every probe is ONE graph in its own subprocess under `timeout 900` (a failed HIP
device poisons the process and teardown hangs), prints
`PROBE <name> <OK|FAIL> <detail>`, and ends with `os._exit`. `nanobind: leaked …`
at exit is harmless and filtered.

Scripts: `rocm_support.py` (shared: Pipeline factory at `benchmarks/cases.py`
geometry, device-hop placement, `PROBE` reporter), `rocm_a_pipeline.py` (task A),
`rocm_b_stages.py` (task B + stage A/B cost, `--backend rocm|numpy --repeats N`),
`rocm_c_kernels.py` (task C micro-kernels, same flags), `rocm_abort_ladder.py`
(the pre-existing construct ladder, kept for reference), `logs/` (raw output).

Bottom line: **no full ROCm generation runs today** — the two structural
`while_loop`s (`node_levels` inside `transform`, the level loop inside `forward`)
abort at run time and `speciate`'s scans segfault, so every per-generation number
below is a stage decomposition plus an explicitly flagged extrapolation.

---

## A. Device placement on the real GPU path

`Pipeline(..., backend="iree", device="rocm")` at xor_p500
(`NEAT(pop_size=500, species_size=20, survival_threshold=0.01)`,
`DefaultGenome(2→1, max_nodes=50, max_conns=100, output_transform=ACT.sigmoid)`, `XOR()`, seed 42).

| case | spelling | result |
|---|---|---|
| `a1` | `device="rocm"` **+** `compile_options={"target_backends": ["rocm"]}` | **setup itself fails after 1.027 s**: `evoxc.BackendError: the iree artifact was compiled for target(s) rocm and cannot run on the requested device Device(kind='cpu', index=0) (device kind 'cpu' requires compile target 'llvm-cpu')`. `Pipeline.setup` registers `device=None` for the setup graphs but forwards `compile_options` verbatim, so the setup graph is compiled for rocm and run on cpu:0. |
| `a1_infer` | `device="rocm"`, **no** `compile_options` (evoxc infers the target per build) | setup builds and runs on cpu:0 (1.012 s), then **`_place_state` fails**: `evoxc.DeviceError: Tensor.to cannot copy Device(kind='cpu', index=0) data directly to Device(kind='rocm', index=0): cross-device copies are not implemented in v1`. The setup graphs return IREE **payload**-backed `cpu:0` tensors, and `_place_state` copies them in one hop. |
| `a1_hop` | same as `a1_infer` + the missing host hop applied by hand (`leaf.numpy()` → `Tensor(host).to(rocm)`, rank-0 shape restored) | **placement works**: `pop[0]=rocm pop[3]=rocm randkey=rocm generation=rocm species[10 leaves]=rocm`; generation-graph **build 9.005 s** (signature `(500, 50, 100, 4, 16, 150)`); `step` then **ABORTs**: `c/runtime/src/iree/hal/drivers/hip/event_semaphore.c:786: ABORTED; the semaphore was aborted; while invoking native function hal.fence.await`. |
| `a1_numpy` | `backend="numpy"`, `device="rocm"` | setup **succeeds** and the state is device-labelled rocm (ndarray-backed tensors upload fine); the first build then raises `BackendError: the numpy backend v1 supports CPU devices only, got device kind 'rocm'`. Placement ≠ runnability. |
| `a2numpy` | 2 generations, `device="cpu"` vs `device=None` | **bit-identical** (fitness and all four payload arrays, `equal_nan`), e.g. gen0 `[-0.26183051, -0.28553843, -0.38008713, -0.34660804]` in both |
| `a2iree` | 2 generations, `target_backends=["llvm-cpu"]`, `device="cpu"` vs `None` | **bit-identical** within the iree side (`[-0.26183051, -0.28553843, -0.38008713, -0.34660804]`) |
| `a3` | rocm-resident state → `_host_view(state)` | view registers `device=None` and every runtime leaf is `cpu` (`pop[0] pop[3] randkey generation species[0]`); `.numpy()` raises nothing; `to_numpy(state.pop.node_keys)` D2H equals the view's values; a **real reporting graph** (`_schedule_of`, the transform stage) builds and runs through the view on the CPU side-channel in **1.734 s** from a rocm-resident population |

Two committed-code defects this exposes (reported, not fixed here — outside this
task's scope):

1. `Pipeline.setup` forwards `compile_options` to the CPU setup graphs, so
   `device=<gpu>` + `compile_options={"target_backends": ["rocm"]}` cannot get past
   `setup`. Leaving `compile_options` empty works because evoxc infers the target
   from each build's device (rocm for the generation graph, llvm-cpu for the setup
   graphs).
2. `Pipeline._place_state` does a single-hop `Tensor.to(device)`; for an `iree`
   backend the setup outputs are payload-backed `cpu:0` tensors that need the
   documented host hop (`t.numpy()` → new host tensor → `.to(device)`), exactly as
   `common.tools.to_numpy` already does for D2H. `device="cpu"` is unaffected
   (identity) and the numpy backend is unaffected (ndarray-backed).

With the hop applied by hand, the placement story is validated end to end: setup on
CPU → one placement → device-resident state → graph compiled for rocm → the run
reaches the HAL and dies in it.

---

## B. Stage decomposition (xor_p500): which stage dies first

One stage graph per subprocess, calling the real kernels (`NEAT.setup`,
`algorithm.transform`, `algorithm.forward`, `SpeciesController.update_species`,
`NEAT._create_next_generation`, `genome.prepare_distance` + `species.speciate`,
`NEAT.tell`). Run cost is the median of 3 warmup + 10 timed runs; the numpy column
is the same graph on `backend="numpy"` (same session).

| stage (xor_p500) | construct used | ROCm | numpy (same graph) |
|---|---|---|---|
| genome init + seed species | the real `NEAT.setup` graph | **OK** — run 34 ms, build+run 1.114 s | OK, run 34 ms |
| `transform`, table half | `key_slot_table` → `_slot_of` → `edge_order` → `incoming_table` (argsort/sort/scatter/window arithmetic) | **OK** — build 1.370 s, run **7.142 ms** | build 0.101 s, run **4.133 ms** |
| `transform`, `node_levels` (`acyclic=True`, no `cond`) | the max-plus relaxation `while_loop` alone | **ABORT** (runtime `event_semaphore.c:786 hal.fence.await`) | run ~9 ms |
| `transform`, `node_levels` (shipped: `cond` + fallback loop) | `while_loop` nested in `evoxc.cond` | **`iree-compile` dies with SIGSEGV** (`Error invoking IREE compiler tool iree-compile`, exit code −11) — a *compile-time* crash, not the runtime abort | — |
| `algorithm.transform` (whole stage) | `derive_schedule` | **ABORT** (same runtime signature) | — |
| `algorithm.forward`, schedule fed in as graph inputs | `_hit_conn_attrs` + the level `while_loop` | **ABORT** | run 4.165 ms (outputs `(2000,1)`, all finite) |
| `transform` + `forward` in one graph | the pipeline's own order | **ABORT** | — |
| `species.update_species` | ranking, elites, parent pairs | **OK** — build 0.846 s, run **0.276 ms** | build 0.046 s, run **1.843 ms** |
| `_create_next_generation` (over `update_species`) | crossover/mutation chain, incl. `cycle.closes_cycle`'s `while_loop` | **OK** — build 4.657 s, run **22.547 ms** | build 0.142 s, run **22.024 ms** |
| `genome.prepare_distance` | one population digest | **OK** — build 0.589 s, run **0.194 ms** | build 0.048 s, run **3.316 ms** |
| `species.speciate` (digest + the fixed-length scans) | `evoxc.scan` × 2-3 | build **OK 3.193 s**, then the **run SIGSEGVs the process** (`exit 139`, no Python exception) | — |
| `algorithm.tell` (whole stage) | everything after `evaluate` | build **OK 7.591 s**, then **SIGSEGV** (`exit 139`) | — |

Reading: the first fatal stage of a generation is `transform`'s `node_levels`, and
`forward`'s level loop is independently fatal; the tell-side scan path dies later
with a different (harsher) signature. What *does* work is instructive: the
data-movement half of the schedule, the whole `update_species` + mutation/
crossover chain (including a `while_loop` with gather/select/scatter in its body),
the distance digest, and the level step itself as an un-looped graph.

---

## C. Per-kernel cost at each case's real geometry

Geometry from `benchmarks/cases.py` (`get_cases`); `level_step` mirrors
`genome/step.py::_node_step`'s op mix at `(rows, max_nodes=50, max_in_degree=4)`
(flat gather, `xv == xv` liveness, weight multiply, K-fold reduce, bias add, `any`
mask, select); `rows = pop * samples` and a width above `EVAL_CHUNK_ROWS` = 12 500
is measured one chunk at a time, like `host_payload._prepared_query`. 3 warmup +
10 timed runs, median.

### C1. `level_step` (one level of a forward pass)

| case | rows × chunks | ROCm ms | numpy ms | ratio |
|---|---|---|---|---|
| xor_p100 (S=4) | 400 × 1 | 0.100 | 1.133 | 11× |
| xor_p500 (S=4) | 2 000 × 1 | 0.095 | 5.070 | 53× |
| xor3d_p500 (S=8) | 4 000 × 1 | 0.101 | 9.564 | 95× |
| custom_pagie_p500 (S=500) | 12 500 × 20 | 0.121/chunk → **2.425/batch** | 32.240/chunk → **644.798/batch** | **266×** |
| cartpole_p100 (per query) | 100 × 1 | 0.100 | 0.334 | 3.3× |

Effective throughput of the dominant element volume (`rows·N·K·4 B`):
ROCm 3.2 / 16.9 / 31.8 / 82.5 / 0.8 GB/s in the table order (the 2.5 M-element
pagie chunk is the only one that leaves the ~0.1 ms per-graph floor: every ROCm
number below ≈0.1 ms is a **launch/fence floor, not a compute time**), numpy
0.24-0.33 GB/s throughout (numpy `gather`/`select` run at 0.3 GB/s here, `take` and
`where` on 50 M-element index arrays — consistent with `PERF_NUMPY.md`).

### C2. `level_step_x3` (a depth-3 forward's level work, no loop)

| case | ROCm ms | numpy ms | ratio |
|---|---|---|---|
| xor_p500 | 0.120 (0.040/level) | 14.417 (4.806/level) | 120× |
| custom_pagie_p500 | 0.196/chunk → 3.925/batch | 106.387/chunk → 2 127.746/batch | 542× |
| cartpole_p100 | 0.119 | 0.965 | 8× |

### C3. Speciation distance

| case | digest only ROCm | digest only numpy | digest + 1 row ROCm | digest + 1 row numpy | row alone (difference) ROCm / numpy |
|---|---|---|---|---|---|
| xor_p100 | — | — | 0.427 | 2.382 | 0.24 / 1.06 |
| xor_p500 | 0.194 | 3.316 | 0.562 | 9.807 | 0.37 / 6.49 |
| xor3d_p500 | — | — | 0.572 | 9.869 | 0.38 / 6.55 |
| custom_pagie_p500 | — | — | 0.571 | 9.903 | 0.38 / 6.59 |
| cartpole_p100 | 0.185 | 1.318 | 0.424 | 2.344 | 0.24 / 1.03 |

`speciate` needs `species_size * 2 = 40` rows per generation plus the one digest.

### C4. RL per-query forward (unrolled, numpy only — the level loop aborts on ROCm)

`forward_prepared` + 20 unrolled queries on a CPU-derived schedule:
xor_p500 **0.9424 ms/query** (18.849 ms for 20, so 188.5 ms scaled to 200);
cartpole_p100 **0.5252 ms/query → 105.03 ms for 200 queries** (the committed
`PERF_NUMPY.md` generation for cartpole is 275.24 ms, so the 200-query forward is
~38 % of it and the rollout arithmetic the rest).

### C5. Build cost (one-off per shape)

ROCm `iree-compile` per static shape, same graphs: transform tables 1.37 s,
`update_species` 0.85 s, next generation 4.66 s, digest 0.59 s, speciate 3.19 s,
full generation graph **9.005 s** (`a1_hop`) — against 0.10-0.14 s per stage and
0.55 s for the whole generation on numpy. Compilation is 10-30× more expensive on
the GPU leg.

---

## C6. Extrapolated per-generation cost (measured stages + flagged assumptions)

Model: `t_gen ≈ Σ stage times`, with the stages that cannot run on ROCm replaced by
the measured ROCm cost of their un-looped building blocks. Committed numpy
per-generation references (`PERF_NUMPY.md` pass 2, 150 generations, 3 repeats):
xor_p500 178.24 ms, custom_pagie_p500 833.81 ms, cartpole_p100 275.24 ms.

| case | measured ROCm stages | assumed (not measured) | extrapolated ROCm ms/gen | vs numpy |
|---|---|---|---|---|
| xor_p500 | transform tables 7.14, `update_species` 0.28, next generation 22.55, speciate 0.19 + 40×0.37 = 14.9 | forward = 1-2 × `level_step` (0.095-0.19 ms; numpy evidence: the real forward is 4.165 ms ≈ 0.8 level steps) and transform levels ≈ 8.5 ms / 53 = 0.16 ms | **≈ 45** | ~4× |
| custom_pagie_p500 | transform tables 7.14, `update_species` 0.28, next generation 22.55, speciate 14.9, forward 2.425 × levels | the pagie forward is 1-2 levels (numpy evidence: one measured level step is 644.8 ms against an 833.81 ms whole generation, so ≤1.3 levels), and the 20 unrolled chunks stay element-bound rather than launch-bound | **≈ 48** | ~17× |
| cartpole_p100 | transform tables ~7.1, `update_species` 0.28, next generation ~22.5, speciate 0.19 + 40×0.24 = 9.8 | forward 0.1-0.3 ms (the measured 0.100 ms `level_step` is a launch floor), rollout 200 × ~0.1 ms ≈ 20 ms (**the rollout `while_loop` cannot run, so this is pure assumption**) | **≈ 45-65** | ~4-6× |

How much of each number is measured: xor_p500 and custom_pagie_p500 are **>99 %
measured stage costs** (the assumed parts are <0.3 ms); cartpole_p100 is
**~55 % measured** with the whole 200-step rollout assumed. The estimates assume
(a) the two fatal `while_loop`s would cost their per-iteration micro-benchmark cost,
(b) ops inside one graph pay no per-op launch cost on ROCm (the measured per-graph
floor is ~0.1 ms, and a whole generation is ONE graph run, so the floor is paid
once), (c) stage costs add. They are *upper bounds* on the GPU side only in the
sense that a working graph would fuse nothing extra. **No end-to-end ROCm
generation number exists until evoxc fixes the while/scan abort.**

---

## D. Verdict for the benchmark work

1. **Large-batch func-fit (`custom_pagie_p500`) is the most promising case**: the
   level step is 266× (single level) to 542× (three levels) faster on the MI300X
   because 250 000 tiled rows move 2.5 M-element windows per level and the GPU
   leaves the launch floor (82.5 GB/s vs numpy's 0.31 GB/s). Extrapolated
   generation ≈48 ms vs 834 ms (≈17×).
2. **Mid-batch XOR (`xor_p500`, `xor3d_p500`) wins on the network kernel (53×/95×)
   but not whole generations**: at 2 000-4 000 rows the generation is dominated by
   the many small tell-side ops, where ROCm is *at best equal* to numpy —
   `_create_next_generation` is 22.55 ms on ROCm against 22.02 ms on numpy, and the
   transform table half is **slower** on ROCm (7.14 vs 4.13 ms). Extrapolated ≈45 ms
   vs 178 ms (≈4×).
3. **The RL rollout (`cartpole_p100`) is the least promising**: 100-row queries sit
   entirely in the ~0.1 ms per-graph launch floor (level step 3.3× faster, 0.100 vs
   0.334 ms), and its per-generation cost on numpy is 38 % network / 62 % rollout
   arithmetic — the part no kernel micro-benchmark covers. Extrapolated ≈45-65 ms vs
   275 ms (≈4-6×).
4. **Compilation is a real cost on this leg**: 9.005 s to build the xor_p500
   generation graph (0.55 s on numpy) and 0.6-4.7 s per stage graph. Any benchmark
   protocol on ROCm must exclude build time or amortise it over hundreds of
   generations.
5. **Nothing on ROCm is end-to-end measurable today.** `transform`'s `node_levels`
   loop aborts at run time (and crashes `iree-compile` when nested in `cond`),
   `forward`'s level loop aborts, and `speciate`'s scans segfault the process — so
   `benchmarks/RESULTS.md` must not gain a ROCm row until evoxc ships a fix.

---

## Smallest failing patterns to hand the evoxc owner

In decreasing order of usefulness (all reproduced from this tree, one subprocess
each, `backend="iree", device=evoxc.Device("rocm", 0), target_backends=["rocm"]`):

1. **`scan` whose carry is a wide species table (run-time SIGSEGV, no Python
   exception)** — `species.speciate`: two/three `evoxc.scan`s over `species_size`
   with a carry of the `(S, …)` species table plus population-shaped `(P,)` rows.
   Build succeeds (3.19 s); the run kills the process (`exit 139`). The existing
   ladder's `p1`/`p2`/`p3` show the same family at minimal size (runtime
   `event_semaphore.c:786 ABORTED`) and `p12` shows a trivial scan failing as
   `stream_command_buffer.c:452 INVALID_ARGUMENT … hal.device.queue.alloca`.
2. **`while_loop` nested in `cond` (compile-time SIGSEGV)** —
   `common/tensorization.py::node_levels(acyclic=False)`: the level-only loop and the
   cycle-detecting loop inside `evoxc.cond`. `iree-compile` exits with code −11, so
   this fails earlier and with a different signature than the runtime abort.
3. **`while_loop` whose body reduces a block-gathered window (run-time abort)** —
   `node_levels(acyclic=True)` and the genome level step: body = `concatenate` a dump
   column, flat-`gather` with a **rank-3** `(rows, N, K)` index block, K pairwise
   `maximum`/`sum`, `select`, `not_equal`, `reduce_max`, carries
   `(int32 step, (rows, N) values, bool changed)` → `event_semaphore.c:786 ABORTED …
   hal.fence.await`.

Hypothesis worth one ladder probe (not yet isolated): the rank of the gather index
inside the loop body is what separates the two `while_loop` outcomes — the loop that
**works** on this box (`genome/operations/mutation/cycle.py::closes_cycle`, exercised
inside the passing `next_generation` stage) gathers with a **rank-2** `(P, E)` index,
while both failing loops gather with a **rank-3** `(rows, N, K)` index. Probe: take
ladder `p14` and (a) gather a `(P, E)` index, (b) gather a `(rows, N, K)` index, same
carries, same cap.
