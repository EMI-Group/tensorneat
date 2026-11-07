# TensorNEAT batch forward pass — new evoxc (numpy / iree rocm) vs frozen JAX on one MI300X VF
**Verdict for the record this file renders: every planned row measured — twenty-one of twenty-one `ok`, zero blocked — and the new evoxc forward pass sits between 0.91x (the JAX baseline marginally ahead) and 3.68x (the new side ahead) of the frozen JAX one per forward, on the same device, in the same session.**
That single range is the whole story and it is not a win everywhere: below ~4 000 evaluated rows per pass the two sides are at parity and JAX is nominally ahead (0.96x at 400 rows, 0.91x at 2 000, 0.99x at 4 000), because at that size both sides are paying a fixed launch-and-readback cost and JAX's is marginally smaller; from 50 000 rows up the new side wins, by 1.94x at 50 000, 2.65x at 250 000 and 3.68x at 500 000 evaluated rows.
The RL shape — one observation per individual, `(100, 4) -> (100, 2)` — is 1.21x faster on the new side (1.119 against 1.358 ms), the row that matters most for a rollout, since an RL generation calls exactly this forward once per environment timestep.
The shape of the two curves, not any single number, is what this record establishes: the new side is nearly flat in the row count (1.481 ms at 400 rows, 3.833 ms at 500 000 — a 2.6x growth for 1 250x the work, 130 431 153 rows/s at the widest tile), while the JAX baseline grows with it (1.415 ms -> 14.110 ms, a 10.0x growth, 35 436 381 rows/s). Both are dominated by a fixed per-pass cost — launch, synchronisation and the host materialisation of the output block — and the new side's fixed cost is the larger one at small tiles and the smaller coefficient on the row count at wide ones; a mechanism is not attributed here, because one forward-pass record cannot separate kernel time from readback time.
Against the `numpy` leg — the CPU reference interpreter, which is what a reader without a GPU runs — the JAX device leg is 2.0x/7.5x/13.0x/25.0x/70.1x/77.4x/1.35x faster on the seven workloads in declaration order (`xor_p100_s4`, `xor_p500_s4`, `xor_p1000_s4`, `xor_p100_s500`, `xor_p500_s500`, `xor_p1000_s500`, `cartpole_p100_s1`), and the CPU leg's cost is the one that scales hardest with the tiled rows (2.807 ms at 400 rows to 1 092.027 ms at 500 000).
Setup and build are one-off and are never charged to a pass: the new side's `Pipeline.setup()` costs 0.015–0.018 s on the numpy leg and 1.193–1.407 s on `iree` rocm (loading the IREE runtime), the forward graph builds in 0.056–1.103 s (numpy) and 1.584–2.358 s (`iree`); the JAX side's setup — XLA client plus `vmap(transform)` — costs 3.305–5.913 s and its `jit(...).lower().compile()` 0.274–0.465 s.
Numerically the two new-side targets are the same computation: `iree` rocm matches the `numpy` leg's output mean to `0.000e+00` on six of the seven workloads and to `3.725e-08` on `cartpole_p100_s1`, the one row that takes the `samples == 1` accelerated kernel formulation. The JAX side draws its own population from its own PRNG, so its output is a different individual set and is deliberately not compared here (`test/parity/` owns numerical fidelity).
Nothing below quotes a number that is not in `benchmarks/results/forward.json`; the ratios in the headline table and in the prose are arithmetic on that record's medians, and nothing here is a statement about HEAD.
## Headline: one forward pass of the whole population
| Workload | Evaluated rows | JAX ms/forward | new `iree` rocm ms/forward | new `numpy` ms/forward | JAX ÷ `iree` | JAX ÷ `numpy` | `iree` rows/s | `jax` rows/s |
|---|---|---|---|---|---|---|---|---|
| `xor_p100_s4` | 400 | 1.415 | 1.481 | 2.807 | 0.96x | 0.50x | 270 029 | 282 719 |
| `xor_p500_s4` | 2 000 | 1.554 | 1.716 | 11.655 | 0.91x | 0.13x | 1 165 868 | 1 286 883 |
| `xor_p1000_s4` | 4 000 | 1.636 | 1.645 | 21.258 | 0.99x | 0.08x | 2 431 325 | 2 444 592 |
| `xor_p100_s500` | 50 000 | 3.426 | 1.765 | 85.785 | 1.94x | 0.04x | 28 334 414 | 14 591 945 |
| `xor_p500_s500` | 250 000 | 7.726 | 2.918 | 541.941 | 2.65x | 0.01x | 85 683 499 | 32 358 332 |
| `xor_p1000_s500` | 500 000 | 14.110 | 3.833 | 1 092.027 | 3.68x | 0.01x | 130 431 153 | 35 436 381 |
| `cartpole_p100_s1` | 100 | 1.358 | 1.119 | 1.831 | 1.21x | 0.74x | 89 383 | 73 620 |
Every cell is a column of the generated block below (`Median ms` recomputed as rows/s here, `Rows` as the second column); `JAX ÷ iree` and `JAX ÷ numpy` are the `speed-up vs jax` columns of that block's speed-up table, where a value below 1.00x means the JAX side is faster.
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
`xla-rocm` is an opt-in target (`--targets numpy,iree-rocm,xla-rocm,jax`) and is **not** part of this record: on this tree its host phase fails inside `Pipeline.setup()`'s device load, `evoxc.core.errors.BackendError: the xla PJRT plugin reports platform 'rocm', which serves device kind(s) ('rocm', 'hip'); it cannot serve Device(kind='cpu', index=0)`, so the leg is an explicit `error` row and no `xla` number is quoted here. A later session adds those rows.
<!-- BEGIN MEASURED (forward_measure.py) -->
## Environment

Measured conditions, not a verdict: these numbers are comparable only with another session on the same box and tree.

| Item | Value |
|---|---|
| Date / session (UTC) | 2026-11-07T08:08:22+00:00 – 2026-11-07T08:10:15+00:00 |
| GPU model (gfx version) | AMD Instinct MI300X VF (gfx942) |
| HIP devices (IREE runtime query) | device_id=1 AMD Instinct MI300X VF |
| Device memory during the rows | 286 – 286 MiB used of 196288 MiB total |
| ROCm version | 7.1.0 |
| IREE compiler / runtime | 3.13.0 / 3.13.0 |
| JAX (old side) | 0.10.2; backend gpu; devices rocm:0; allocator XLA_PYTHON_CLIENT_PREALLOCATE=false |
| tensorneat / evoxc | 0.2.0 / 0.1.0 |
| numpy | 2.5.4 |
| Python | 3.12.3 |
| Platform | Linux-6.8.0-146-generic-x86_64-with-glibc2.39 |
| CPU cores / RAM | 20 / 241611 MB |
| Threads (inherited env) | runtime defaults |
| Git commit (code under test) | `713f4a618223` (clean — generated report artifacts are not counted) |
| Load average (1/5/15 min) | 2.58 / 2.89 / 2.56 |

## Measured (one forward pass of the whole population)

`median` is the median over the timed passes of one fresh process and `min – max` that row's own per-pass spread -- the uncertainty measure here (a single median is not a distribution). Build and setup are one-off: they are reported in the record, never charged to a pass.

| Workload | Target | Rows | Median ms | Min – max ms | Rows/s | Peak RSS MiB | Status |
|---|---|---|---|---|---|---|---|
| `xor_p100_s4` | numpy | 400 | 2.807 | 2.767 – 2.846 | 142526 | 134.3 | ok |
| `xor_p100_s4` | iree-rocm | 400 | 1.481 | 1.449 – 1.535 | 270029 | 1879.4 | ok |
| `xor_p100_s4` | jax | 400 | 1.415 | 1.380 – 1.497 | 282719 | 4702.4 | ok |
| `xor_p500_s4` | numpy | 2000 | 11.655 | 11.276 – 11.986 | 171602 | 153.2 | ok |
| `xor_p500_s4` | iree-rocm | 2000 | 1.716 | 1.671 – 1.770 | 1165868 | 1891.5 | ok |
| `xor_p500_s4` | jax | 2000 | 1.554 | 1.520 – 1.617 | 1286883 | 4720.7 | ok |
| `xor_p1000_s4` | numpy | 4000 | 21.258 | 20.394 – 21.758 | 188167 | 176.5 | ok |
| `xor_p1000_s4` | iree-rocm | 4000 | 1.645 | 1.608 – 1.657 | 2431325 | 1904.2 | ok |
| `xor_p1000_s4` | jax | 4000 | 1.636 | 1.594 – 1.758 | 2444592 | 4716.6 | ok |
| `xor_p100_s500` | numpy | 50000 | 85.785 | 84.271 – 87.391 | 582852 | 311.0 | ok |
| `xor_p100_s500` | iree-rocm | 50000 | 1.765 | 1.728 – 1.792 | 28334414 | 1885.6 | ok |
| `xor_p100_s500` | jax | 50000 | 3.426 | 3.372 – 3.536 | 14591945 | 4889.8 | ok |
| `xor_p500_s500` | numpy | 250000 | 541.941 | 527.711 – 568.462 | 461305 | 920.2 | ok |
| `xor_p500_s500` | iree-rocm | 250000 | 2.918 | 2.860 – 2.961 | 85683499 | 1918.3 | ok |
| `xor_p500_s500` | jax | 250000 | 7.726 | 7.622 – 7.900 | 32358332 | 4926.5 | ok |
| `xor_p1000_s500` | numpy | 500000 | 1092.027 | 1083.183 – 1108.297 | 457864 | 1654.0 | ok |
| `xor_p1000_s500` | iree-rocm | 500000 | 3.833 | 3.794 – 3.882 | 130431153 | 1968.1 | ok |
| `xor_p1000_s500` | jax | 500000 | 14.110 | 13.839 – 14.230 | 35436381 | 4936.5 | ok |
| `cartpole_p100_s1` | numpy | 100 | 1.831 | 1.773 – 1.940 | 54611 | 132.5 | ok |
| `cartpole_p100_s1` | iree-rocm | 100 | 1.119 | 1.111 – 1.151 | 89383 | 1879.9 | ok |
| `cartpole_p100_s1` | jax | 100 | 1.358 | 1.338 – 1.461 | 73620 | 4683.2 | ok |

## Speed-up against the frozen JAX baseline

Speed-up is `jax median / new median` per forward pass (higher means the new side is faster), the same `old/new` convention as `RESULTS.md`; `n/a` means one of the two legs has no timing in this record — see `## Blocked rows`.

| Workload | jax median ms | numpy median ms | speed-up vs jax | iree-rocm median ms | speed-up vs jax |
|---|---|---|---|---|---|
| `xor_p100_s4` | 1.415 | 2.807 | 0.50x | 1.481 | 0.96x |
| `xor_p500_s4` | 1.554 | 11.655 | 0.13x | 1.716 | 0.91x |
| `xor_p1000_s4` | 1.636 | 21.258 | 0.08x | 1.645 | 0.99x |
| `xor_p100_s500` | 3.426 | 85.785 | 0.04x | 1.765 | 1.94x |
| `xor_p500_s500` | 7.726 | 541.941 | 0.01x | 2.918 | 2.65x |
| `xor_p1000_s500` | 14.110 | 1092.027 | 0.01x | 3.833 | 3.68x |
| `cartpole_p100_s1` | 1.358 | 1.831 | 0.74x | 1.119 | 1.21x |

## Numerical agreement (new-side targets vs the numpy leg)

Every new-side target runs the same graph on the same payload, so its output must match the `numpy` leg's: the absolute difference of the output means is checked against `1e-06 + 1e-05 x |reference|`. This is a self-consistency check that the timed pass computed the forward, not a cross-implementation comparison: the JAX side draws a different population from a different PRNG, so its values are not comparable by design (`test/parity/` owns fidelity).

| Workload | Target | numpy mean | target mean | abs delta | Status |
|---|---|---|---|---|---|
| `xor_p100_s4` | iree-rocm | 0.52595609 | 0.52595609 | 0.000e+00 | pass |
| `xor_p500_s4` | iree-rocm | 0.50019968 | 0.50019968 | 0.000e+00 | pass |
| `xor_p1000_s4` | iree-rocm | 0.49793524 | 0.49793524 | 0.000e+00 | pass |
| `xor_p100_s500` | iree-rocm | 0.52333850 | 0.52333850 | 0.000e+00 | pass |
| `xor_p500_s500` | iree-rocm | 0.49729583 | 0.49729583 | 0.000e+00 | pass |
| `xor_p1000_s500` | iree-rocm | 0.49589536 | 0.49589536 | 0.000e+00 | pass |
| `cartpole_p100_s1` | iree-rocm | -0.02731544 | -0.02731547 | 3.725e-08 | pass |

## Blocked rows

None: every planned row measured. This section is written from the record, so it reappears the moment a leg fails.

## Provenance of this block

This block is rendered from `benchmarks/results/forward.json`: 21 row(s), 21 ok and 0 error, measured in the session 2026-11-07T08:08:22+00:00 – 2026-11-07T08:10:15+00:00 (UTC) at git `713f4a618223` (clean — generated report artifacts are not counted).
Every row is one `forward_measure.py run` in its own process, so the reported peak RSS is that process's high-water mark and no JAX thread pool shared a process with the evoxc interpreter.
Nothing here is a statement about HEAD: every number belongs to that record and to the tree its fingerprint names.
This is the batch forward-pass record of this session: one forward pass of a fixed population, measured per target in its own process. It says nothing about HEAD beyond the tree its fingerprint names.
<!-- END MEASURED (forward_measure.py) -->
## Limitations
- The box was quiet but shared during this session (`/proc/loadavg` 2.58/2.89/2.56 at start, 285.9 MiB of 205 822 MiB device VRAM in use by a co-tenant), so the per-row `min – max` spread is the honest uncertainty measure and a single median is not a distribution.
- One seed (42) and ten timed passes per row: this is a per-forward cost measurement, not a search-quality measurement, and it says nothing about the generations in `RESULTS.md`.
- `xor_p500_s500` and `xor_p1000_s500` are the widest tiles shipped here (250 000 and 500 000 evaluated rows); a compiled one-shot forward at that width is where the new side's kernel amortises best, so the range above is a range over tile widths, not a single headline number.
- The JAX side's numbers are its ROCm backend on the same device, with the allocator policy stated above; a default-allocator JAX run on a shared device cannot even reach these rows (`HIP_ERROR_OutOfMemory` on `jit_topological_sort`), which is a property of the box, not of the baseline.
- Peak RSS is driver RSS and it separates the sides cleanly: the `numpy` leg stays 132.5–1 654.0 MiB (growing with the tile width), `iree` rocm 1 879.4–1 968.1 MiB (runtime-dominated and flat), the JAX side 4 683.2–4 936.5 MiB (XLA client plus executable).
## Reproduce
```
# the committed record and the block below (one (workload, target) row per fresh process, ~2 min)
uv run --all-extras python benchmarks/forward_measure.py matrix --workloads all \
    --targets numpy,iree-rocm,jax --passes 10 --warmup 3 \
    --json benchmarks/results/forward.json --out benchmarks/FORWARD.md \
    --report-note "This is the batch forward-pass record of this session: one forward pass of a fixed population, measured per target in its own process. It says nothing about HEAD beyond the tree its fingerprint names."
# one row, one process
uv run --all-extras python benchmarks/forward_measure.py run --workload xor_p500_s4 --target iree-rocm
# the xla leg (opt-in; blocked on this tree, see "Protocol" above)
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
