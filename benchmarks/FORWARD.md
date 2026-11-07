# TensorNEAT batch forward pass — new evoxc (numpy / iree rocm / xla rocm) vs frozen JAX on one MI300X VF
**Verdict for the record this file renders: every planned row measured — twenty-eight of twenty-eight `ok`, zero blocked — and the fastest new-side leg, `xla` rocm, is ahead of the frozen JAX baseline on all seven workloads, from 3.07x (the RL shape) to 6.39x (500 000 evaluated rows) per forward pass, on the same device, in the same session.**
The new side is a leg-by-leg story, not one range: `xla` rocm wins every row (3.30x/3.25x/3.34x/4.60x/5.79x/6.39x/3.07x in declaration order), `iree` rocm sits at parity on the narrow XOR tiles (0.92x at 400 and 2 000 rows, 1.00x at 4 000) and pulls ahead only from 50 000 rows up (1.82x, 2.66x, 3.68x), and the CPU `numpy` reference is behind the JAX device on all seven (0.47x/0.14x/0.08x/0.04x/0.01x/0.01x/0.77x).
Against `iree` rocm on the same device, `xla` rocm is 3.58x/3.54x/3.35x/2.53x/2.18x/1.73x/2.62x faster per forward pass: the widest advantage at the narrowest tiles, where the fixed per-pass cost dominates, shrinking as the tile widens and the kernel work takes over.
The RL shape — one observation per individual, `(100, 4) -> (100, 2)` — is the row that matters most for a rollout, since an RL generation calls exactly this forward once per environment timestep: `xla` rocm 0.450 ms against the JAX side's 1.383 ms (3.07x), `iree` rocm 1.177 ms (1.18x) and the CPU leg 1.801 ms.
The shape of the curves, not any single number, is what this record establishes: `xla` rocm is nearly flat in the row count (0.419 ms at 400 rows, 2.229 ms at 500 000 — a 5.3x growth for 1 250x the work, 224 286 308 rows/s at the widest tile), `iree` rocm grows 2.6x over the same span (1.498 ms -> 3.867 ms, 129 313 677 rows/s), the JAX baseline grows 10.3x (1.382 ms -> 14.238 ms, 35 117 638 rows/s) and the CPU leg 371x (2.950 ms -> 1 094.888 ms, 456 668 rows/s). Every leg is dominated by a fixed per-pass cost — launch, synchronisation and the host materialisation of the output block — and a mechanism is not attributed here, because one forward-pass record cannot separate kernel time from readback time.
Against the `numpy` leg — the CPU reference interpreter, which is what a reader without a GPU runs — the JAX device leg is 2.1x/7.2x/12.0x/27.6x/73.1x/76.9x/1.30x faster on the seven workloads in declaration order (`xor_p100_s4`, `xor_p500_s4`, `xor_p1000_s4`, `xor_p100_s500`, `xor_p500_s500`, `xor_p1000_s500`, `cartpole_p100_s1`), and `xla` rocm is 7.0x/23.5x/40.2x/127.1x/423.3x/491.2x/4.00x faster; the CPU leg's cost is the one that scales hardest with the tiled rows (2.950 ms at 400 rows to 1 094.888 ms at 500 000).
Setup and build are one-off and are never charged to a pass: the new side's `Pipeline.setup()` costs 0.015–0.018 s on the numpy leg, 1.217–1.280 s on `iree` rocm (loading the IREE runtime) and 3.989–4.122 s on `xla` rocm (the PJRT plugin client; the first xla row pays 8.443 s because it also loads the 480.1 MiB plugin library), the forward graph builds in 0.054–1.086 s (numpy), 1.558–2.485 s (`iree`) and 0.401–1.363 s (`xla`); the JAX side's setup — XLA client plus `vmap(transform)` — costs 3.302–5.944 s and its `jit(...).lower().compile()` 0.294–0.487 s.
Numerically every new-side target is the same computation: `xla` rocm matches the `numpy` leg's output mean to `0.000e+00` on all seven workloads and `iree` rocm on six of the seven plus `3.725e-08` on `cartpole_p100_s1`, the one row that takes the `samples == 1` accelerated kernel formulation. The JAX side draws its own population from its own PRNG, so its output is a different individual set and is deliberately not compared here (`test/parity/` owns numerical fidelity).
Nothing below quotes a number that is not in `benchmarks/results/forward.json`; the ratios in the headline table and in the prose are arithmetic on that record's medians, and nothing here is a statement about HEAD.
## Headline: one forward pass of the whole population
| Workload | Evaluated rows | JAX ms/forward | new `xla` rocm ms/forward | new `iree` rocm ms/forward | new `numpy` ms/forward | JAX ÷ `xla` | JAX ÷ `iree` | JAX ÷ `numpy` | `xla` rows/s | `iree` rows/s | `jax` rows/s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `xor_p100_s4` | 400 | 1.382 | 0.419 | 1.498 | 2.950 | 3.30x | 0.92x | 0.47x | 953 968 | 267 029 | 289 350 |
| `xor_p500_s4` | 2 000 | 1.566 | 0.481 | 1.701 | 11.316 | 3.25x | 0.92x | 0.14x | 4 155 326 | 1 175 958 | 1 277 240 |
| `xor_p1000_s4` | 4 000 | 1.661 | 0.497 | 1.663 | 19.975 | 3.34x | 1.00x | 0.08x | 8 053 458 | 2 404 940 | 2 408 064 |
| `xor_p100_s500` | 50 000 | 3.430 | 0.745 | 1.885 | 94.724 | 4.60x | 1.82x | 0.04x | 67 075 840 | 26 526 830 | 14 579 008 |
| `xor_p500_s500` | 250 000 | 7.681 | 1.326 | 2.892 | 561.243 | 5.79x | 2.66x | 0.01x | 188 488 703 | 86 459 743 | 32 546 479 |
| `xor_p1000_s500` | 500 000 | 14.238 | 2.229 | 3.867 | 1 094.888 | 6.39x | 3.68x | 0.01x | 224 286 308 | 129 313 677 | 35 117 638 |
| `cartpole_p100_s1` | 100 | 1.383 | 0.450 | 1.177 | 1.801 | 3.07x | 1.18x | 0.77x | 222 184 | 84 996 | 72 334 |
Every cell is a column of the generated block below (`Median ms` recomputed as rows/s here, `Rows` as the second column); `JAX ÷ xla`, `JAX ÷ iree` and `JAX ÷ numpy` are the `speed-up vs jax` columns of that block's speed-up table, where a value below 1.00x means the JAX side is faster.
## What is measured, and what is deliberately outside the timed region
One row is one forward pass of a fixed population of NEAT networks, each with its own evolved topology, evaluated through the tensorized forward: `population x samples` rows in, `population x samples x num_outputs` out.
The population comes from `Pipeline.setup()` on the new side and from `algorithm.setup()` + `vmap(algorithm.transform)` on the old side, so the two sides forward *their own initial population* of the same size, drawn from their own PRNG (evoxc splitmix64 versus JAX `PRNGKey`) — one seed identifies the same workload, never the same search.
The forward is the only thing timed: no mutation, no selection, no speciation, no environment step, and no `tell`. The func-fit-shaped rows pass a fixed `(rows, 2)` float32 block drawn once from `default_rng(42)`; the RL row passes `(100, 4)`, the observation shape `cartpole_p100` queries once per environment timestep.
`setup_s` and `build_s` are measured in the same process and reported beside the timing, never inside it; each timed pass includes the host materialisation of the output block, so an asynchronous backend cannot report before its pass completed, and the input block is placed on the device before the timed region so no transfer is counted as forward cost.
The 500-sample rows use the XOR genome only for its geometry (2 inputs, 1 output, `max_nodes` 50 / `max_conns` 100, sigmoid read-out), which is byte-identical to `custom_pagie_p500`'s genome config; they measure the Pagie *tile shape*, not the Pagie dataset, and the values are a random block because the per-forward cost of the tensorized forward is bounded by the schedule and the tile width, not by the data.
## Protocol (binding for this record)
One `(workload, target)` pair per fresh process: the matrix spawns one `forward_measure.py run` per row and never measures two targets in one interpreter, so no JAX thread pool or XLA client shares an address space with the evoxc interpreter and the reported `peak_rss_mib` is that one row's high-water mark.
Every row runs `--warmup 3` untimed passes and `--passes 10` timed ones with stdout/stderr redirected to `os.devnull`; a pass is timed with `time.perf_counter()` around the call, and the row reports mean, median, min and max of those ten samples plus the `min – max` spread — the spread, not the median, is this record's uncertainty measure.
The JAX baseline pins `XLA_PYTHON_CLIENT_PREALLOCATE=false` before importing JAX and records the resolved policy as `jax_allocator`: the default 75 % preallocation of a shared 205.8 GB `MI300X VF` failed rows with `HIP_ERROR_OutOfMemory` and inflated their RSS, and the policy changes neither the compiled computation nor the device op count.
The `numpy` leg is the CPU reference interpreter on the same box (no device placement), reported for scale: it is what the library does with no optional dependency.
`xla-rocm` is a shipped target and part of the tool's default matrix (`DEFAULT_TARGETS` is `numpy,iree-rocm,xla-rocm,jax`): it drives the `jax-rocm7-pjrt==0.10.2` PJRT C API plugin from the `xla-rocm` extra (`.so` loaded over ctypes, no `jax` import — every new-side row records `jax_imported: false`), resolved once per session from `--pjrt-plugin PATH` or `$EVOXC_PJRT_PLUGIN` and never from a fallback. The plugin is fingerprinted in the environment table above (`xla_rocm_plugin.so`, 480.1 MiB, sha256 `8e8319b48fcc…`) because a number measured against a different `.so` is a different experiment. The leg builds and runs on this tree because `Pipeline.setup()` now runs the host setup graphs on the numpy reference backend and registers the real backend and device afterwards (`src/tensorneat/pipeline.py`, `fcbf677f`); before that fix the host-phase device load refused the ROCm plugin (`BackendError: … it cannot serve Device(kind='cpu', index=0)`) and the leg was an error row.
<!-- BEGIN MEASURED (forward_measure.py) -->
## Environment

Measured conditions, not a verdict: these numbers are comparable only with another session on the same box and tree.

| Item | Value |
|---|---|
| Date / session (UTC) | 2026-11-07T08:17:07+00:00 – 2026-11-07T08:19:54+00:00 |
| GPU model (gfx version) | AMD Instinct MI300X VF (gfx942) |
| HIP devices (IREE runtime query) | device_id=1 AMD Instinct MI300X VF |
| Device memory during the rows | 286 – 193939 MiB used of 196288 MiB total |
| ROCm version | 7.1.0 |
| IREE compiler / runtime | 3.13.0 / 3.13.0 |
| JAX (old side) | 0.10.2; backend gpu; devices rocm:0; allocator XLA_PYTHON_CLIENT_PREALLOCATE=false |
| PJRT plugin (xla targets) | `xla_rocm_plugin.so` (480.1 MiB, sha256 `8e8319b48fcc…`) |
| XLA_FLAGS (xla targets) | not recorded (record predates the workaround) |
| tensorneat / evoxc | 0.2.0 / 0.1.0 |
| numpy | 2.5.4 |
| Python | 3.12.3 |
| Platform | Linux-6.8.0-146-generic-x86_64-with-glibc2.39 |
| CPU cores / RAM | 20 / 241611 MB |
| Threads (inherited env) | runtime defaults |
| Git commit (code under test) | `8c6330989a92` (clean — generated report artifacts are not counted) |
| Load average (1/5/15 min) | 6.42 / 5.17 / 3.67 |

## Measured (one forward pass of the whole population)

`median` is the median over the timed passes of one fresh process and `min – max` that row's own per-pass spread -- the uncertainty measure here (a single median is not a distribution). Build and setup are one-off: they are reported in the record, never charged to a pass.

| Workload | Target | Rows | Median ms | Min – max ms | Rows/s | Peak RSS MiB | Status |
|---|---|---|---|---|---|---|---|
| `xor_p100_s4` | numpy | 400 | 2.950 | 2.885 – 3.094 | 135570 | 134.3 | ok |
| `xor_p100_s4` | iree-rocm | 400 | 1.498 | 1.479 – 1.554 | 267027 | 1879.8 | ok |
| `xor_p100_s4` | xla-rocm | 400 | 0.419 | 0.413 – 0.455 | 953968 | 9167.3 | ok |
| `xor_p100_s4` | jax | 400 | 1.382 | 1.343 – 1.423 | 289350 | 4702.8 | ok |
| `xor_p500_s4` | numpy | 2000 | 11.316 | 10.879 – 11.655 | 176747 | 153.0 | ok |
| `xor_p500_s4` | iree-rocm | 2000 | 1.701 | 1.663 – 1.755 | 1175958 | 1891.3 | ok |
| `xor_p500_s4` | xla-rocm | 2000 | 0.481 | 0.454 – 0.529 | 4155326 | 9185.1 | ok |
| `xor_p500_s4` | jax | 2000 | 1.566 | 1.532 – 1.610 | 1277240 | 4712.4 | ok |
| `xor_p1000_s4` | numpy | 4000 | 19.975 | 19.808 – 20.079 | 200252 | 176.4 | ok |
| `xor_p1000_s4` | iree-rocm | 4000 | 1.663 | 1.639 – 1.724 | 2404940 | 1907.5 | ok |
| `xor_p1000_s4` | xla-rocm | 4000 | 0.497 | 0.473 – 0.513 | 8053458 | 9211.7 | ok |
| `xor_p1000_s4` | jax | 4000 | 1.661 | 1.619 – 1.773 | 2408064 | 4723.7 | ok |
| `xor_p100_s500` | numpy | 50000 | 94.724 | 91.651 – 98.956 | 527850 | 311.0 | ok |
| `xor_p100_s500` | iree-rocm | 50000 | 1.885 | 1.682 – 2.566 | 26526830 | 1885.0 | ok |
| `xor_p100_s500` | xla-rocm | 50000 | 0.745 | 0.727 – 0.820 | 67075840 | 9200.4 | ok |
| `xor_p100_s500` | jax | 50000 | 3.430 | 3.350 – 3.460 | 14579008 | 4893.8 | ok |
| `xor_p500_s500` | numpy | 250000 | 561.243 | 530.058 – 626.080 | 445440 | 920.3 | ok |
| `xor_p500_s500` | iree-rocm | 250000 | 2.892 | 2.870 – 2.949 | 86459743 | 1919.5 | ok |
| `xor_p500_s500` | xla-rocm | 250000 | 1.326 | 1.311 – 1.368 | 188488703 | 9247.6 | ok |
| `xor_p500_s500` | jax | 250000 | 7.681 | 7.608 – 7.747 | 32546479 | 4928.2 | ok |
| `xor_p1000_s500` | numpy | 500000 | 1094.888 | 1088.296 – 1140.784 | 456668 | 1654.1 | ok |
| `xor_p1000_s500` | iree-rocm | 500000 | 3.867 | 3.793 – 3.931 | 129313677 | 1967.8 | ok |
| `xor_p1000_s500` | xla-rocm | 500000 | 2.229 | 2.172 – 2.317 | 224286308 | 9304.5 | ok |
| `xor_p1000_s500` | jax | 500000 | 14.238 | 14.061 – 17.609 | 35117638 | 4936.7 | ok |
| `cartpole_p100_s1` | numpy | 100 | 1.801 | 1.777 – 1.982 | 55511 | 132.6 | ok |
| `cartpole_p100_s1` | iree-rocm | 100 | 1.177 | 1.133 – 1.207 | 84996 | 1879.9 | ok |
| `cartpole_p100_s1` | xla-rocm | 100 | 0.450 | 0.427 – 0.550 | 222184 | 9202.3 | ok |
| `cartpole_p100_s1` | jax | 100 | 1.383 | 1.353 – 1.492 | 72334 | 4682.2 | ok |

## Speed-up against the frozen JAX baseline

Speed-up is `jax median / new median` per forward pass (higher means the new side is faster), the same `old/new` convention as `RESULTS.md`; `n/a` means one of the two legs has no timing in this record — see `## Blocked rows`.

| Workload | jax median ms | numpy median ms | speed-up vs jax | iree-rocm median ms | speed-up vs jax | xla-rocm median ms | speed-up vs jax |
|---|---|---|---|---|---|---|---|
| `xor_p100_s4` | 1.382 | 2.950 | 0.47x | 1.498 | 0.92x | 0.419 | 3.30x |
| `xor_p500_s4` | 1.566 | 11.316 | 0.14x | 1.701 | 0.92x | 0.481 | 3.25x |
| `xor_p1000_s4` | 1.661 | 19.975 | 0.08x | 1.663 | 1.00x | 0.497 | 3.34x |
| `xor_p100_s500` | 3.430 | 94.724 | 0.04x | 1.885 | 1.82x | 0.745 | 4.60x |
| `xor_p500_s500` | 7.681 | 561.243 | 0.01x | 2.892 | 2.66x | 1.326 | 5.79x |
| `xor_p1000_s500` | 14.238 | 1094.888 | 0.01x | 3.867 | 3.68x | 2.229 | 6.39x |
| `cartpole_p100_s1` | 1.383 | 1.801 | 0.77x | 1.177 | 1.18x | 0.450 | 3.07x |

## Numerical agreement (new-side targets vs the numpy leg)

Every new-side target runs the same graph on the same payload, so its output must match the `numpy` leg's: the absolute difference of the output means is checked against `1e-06 + 1e-05 x |reference|`. This is a self-consistency check that the timed pass computed the forward, not a cross-implementation comparison: the JAX side draws a different population from a different PRNG, so its values are not comparable by design (`test/parity/` owns fidelity).

| Workload | Target | numpy mean | target mean | abs delta | Status |
|---|---|---|---|---|---|
| `xor_p100_s4` | iree-rocm | 0.52595609 | 0.52595609 | 0.000e+00 | pass |
| `xor_p100_s4` | xla-rocm | 0.52595609 | 0.52595609 | 0.000e+00 | pass |
| `xor_p500_s4` | iree-rocm | 0.50019968 | 0.50019968 | 0.000e+00 | pass |
| `xor_p500_s4` | xla-rocm | 0.50019968 | 0.50019968 | 0.000e+00 | pass |
| `xor_p1000_s4` | iree-rocm | 0.49793524 | 0.49793524 | 0.000e+00 | pass |
| `xor_p1000_s4` | xla-rocm | 0.49793524 | 0.49793524 | 0.000e+00 | pass |
| `xor_p100_s500` | iree-rocm | 0.52333850 | 0.52333850 | 0.000e+00 | pass |
| `xor_p100_s500` | xla-rocm | 0.52333850 | 0.52333850 | 0.000e+00 | pass |
| `xor_p500_s500` | iree-rocm | 0.49729583 | 0.49729583 | 0.000e+00 | pass |
| `xor_p500_s500` | xla-rocm | 0.49729583 | 0.49729583 | 0.000e+00 | pass |
| `xor_p1000_s500` | iree-rocm | 0.49589536 | 0.49589536 | 0.000e+00 | pass |
| `xor_p1000_s500` | xla-rocm | 0.49589536 | 0.49589536 | 0.000e+00 | pass |
| `cartpole_p100_s1` | iree-rocm | -0.02731544 | -0.02731547 | 3.725e-08 | pass |
| `cartpole_p100_s1` | xla-rocm | -0.02731544 | -0.02731544 | 0.000e+00 | pass |

## Blocked rows

None: every planned row measured. This section is written from the record, so it reappears the moment a leg fails.

## Provenance of this block

This block is rendered from `benchmarks/results/forward.json`: 28 row(s), 28 ok and 0 error, measured in the session 2026-11-07T08:17:07+00:00 – 2026-11-07T08:19:54+00:00 (UTC) at git `8c6330989a92` (clean — generated report artifacts are not counted).
Every row is one `forward_measure.py run` in its own process, so the reported peak RSS is that process's high-water mark and no JAX thread pool shared a process with the evoxc interpreter.
Nothing here is a statement about HEAD: every number belongs to that record and to the tree its fingerprint names.
This is the batch forward-pass record of this session: one forward pass of a fixed population, measured per target (numpy CPU reference, iree rocm, xla rocm and the frozen JAX baseline) in its own fresh process. It says nothing about HEAD beyond the tree its fingerprint names.
<!-- END MEASURED (forward_measure.py) -->
## Limitations
- The box was shared and busy during this session (`/proc/loadavg` 5.60/3.95/3.03 at start, 4.77–8.27 across the rows, 6.42/5.17/3.67 at document build), so the per-row `min – max` spread is the honest uncertainty measure and a single median is not a distribution.
- One seed (42) and ten timed passes per row: this is a per-forward cost measurement, not a search-quality measurement, and it says nothing about the generations in `RESULTS.md`.
- `xor_p500_s500` and `xor_p1000_s500` are the widest tiles shipped here (250 000 and 500 000 evaluated rows); a compiled one-shot forward at that width is where a kernel amortises best, so the ratios above are a function of tile width, not a single headline number.
- The JAX side's numbers are its ROCm backend on the same device, with the allocator policy stated above; a default-allocator JAX run on a shared device cannot even reach these rows (`HIP_ERROR_OutOfMemory` on `jit_topological_sort`), which is a property of the box, not of the baseline.
- Peak RSS is driver RSS and it separates the legs cleanly: the `numpy` leg stays 132.6–1 654.1 MiB (growing with the tile width), `iree` rocm 1 879.8–1 967.8 MiB (runtime-dominated and flat), the JAX side 4 682.2–4 936.7 MiB (XLA client plus executable) and `xla` rocm 9 167.3–9 304.5 MiB — the largest footprint here, the PJRT plugin plus its executable, which is real resident memory and not the allocator's address-space reserve.
- The `xla` leg's BFC allocator reserves device memory per row — each `xla` row logs `XLA backend allocating 143.77GiB … for BFCAllocator` at client creation — so this leg needs a device where ~144 GiB is free; a fully shared MI300X would fail those rows with a device OOM rather than a wrong number.
- The `Device memory during the rows` range in the table above (286–193 939 MiB) is a co-tenancy reading, not this record's own footprint: `device_memory` is sampled at the *start* of a row, before that row's own device client exists, and during this session another process on the box intermittently held ~144 GiB of the 196 288 MiB device (75 % of total, the default preallocation of an XLA client), while the same device read 286 MiB used again once the session ended.
- The `xla` leg's setup (3.989–4.122 s) is its one-off client creation plus plugin load, `TF_XLA_HSACO_CACHE_DIR` is unset so every fresh process recompiles its kernel (`HSACO file cache is disabled!`), and the first xla row's 8.443 s is that load paid cold; none of it is charged to a pass.
## Reproduce
```
# the committed record and the block below (one (workload, target) row per fresh process, ~3 min)
export EVOXC_PJRT_PLUGIN="$PWD/.venv/lib/python3.12/site-packages/jax_plugins/xla_rocm7/xla_rocm_plugin.so"
uv run --all-extras python benchmarks/forward_measure.py matrix --workloads all \
    --targets numpy,iree-rocm,xla-rocm,jax --passes 10 --warmup 3 \
    --pjrt-plugin "$EVOXC_PJRT_PLUGIN" \
    --json benchmarks/results/forward.json --out benchmarks/FORWARD.md \
    --report-note "This is the batch forward-pass record of this session: one forward pass of a fixed population, measured per target (numpy CPU reference, iree rocm, xla rocm and the frozen JAX baseline) in its own fresh process. It says nothing about HEAD beyond the tree its fingerprint names."
# one row, one process
uv run --all-extras python benchmarks/forward_measure.py run --workload xor_p500_s4 --target iree-rocm
# the xla leg (needs the plugin; --pjrt-plugin also accepts the path directly)
uv run --all-extras python benchmarks/forward_measure.py run --workload xor_p100_s4 \
    --target xla-rocm --pjrt-plugin .venv/lib/python3.12/site-packages/jax_plugins/xla_rocm7/xla_rocm_plugin.so
# re-render this block from the committed record without measuring anything
uv run --all-extras python benchmarks/forward_measure.py render \
    --json benchmarks/results/forward.json --out benchmarks/FORWARD.md
# no optional dependency needed to list the matrix
python3 benchmarks/forward_measure.py --list
```
## What this does not change
Whole-generation cost stays `RESULTS.md`'s business (CPU, old vs new, 150 generations) and `GPU.md`'s (per-generation device cost on the same box); this file isolates one forward pass and claims nothing about a generation's mutation, selection or speciation cost.
The forward measured here is the same graph the pipeline runs inside a generation — `algorithm.transform` then the prepared `act_fn` — so the per-forward numbers move together with `PERF_NUMPY.md`'s whole-generation numpy numbers; they are not a second implementation.
