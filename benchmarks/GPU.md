# Accelerator measurement: `numpy`, `iree` llvm-cpu and `iree` rocm on one MI300X VF

**Verdict for the tree this file ships with: the two CPU legs measure green through this protocol, and the ROCm device path is blocked before the first generation — `numpy` and `iree-llvm-cpu` run the canonical case matrix end to end and agree numerically at generation 1, while every `iree-rocm` row is an `error` row whose message is a device-placement refusal in `Pipeline.setup()`, not a compile failure and not a number.**
The refusal is `evoxc.core.errors.DeviceError: input at path [0] is on device Device(kind='cpu', index=0), but the executable runs on device Device(kind='rocm', index=0): no implicit device-to-host or host-to-device transfer happens at the run boundary — move the tensor to the executable's device explicitly via t.to(Device(kind='rocm', index=0))`.
Its cause is on the TensorNEAT side, not in IREE: the pipeline hands its initial tensors to `evoxc.run` on the host while the executable was compiled for the ROCm device, so the initial-tensor device placement that `src/tensorneat` is gaining in parallel is the missing piece.
The one-off generation-graph compile for `rocm` is a *later* step of this protocol and cannot be reached until that placement exists: `build_s` is filled by the first `Pipeline.step()`, and the first step needs a state that `Pipeline.setup()` never returns on this target.
Nothing below invents a number for the blocked rows: the measured block reports `n/a` for every metric an error row does not have, and the full traceback stays in the record.
No accelerator row is part of the numpy-only verdict in `RESULTS.md`; this file is informational evidence, and `## What this does not change` says exactly why.

## Provenance: which tree and which session

Every number in the measured block below belongs to the record that block names and to the git commit its fingerprint names; no sentence here is a statement about HEAD, and no committed artifact in this directory measures an accelerator row yet.
The block is rendered from `benchmarks/results/gpu_2026-11-05.json`, the record of one `gpu_measure.py matrix` session, and that record carries its own command, its session timestamps, its environment fingerprint and the `protocol` block it was measured under.
If that file is absent or is still the git-ignored scratch default (`benchmarks/results/gpu.json`), the provenance paragraph inside the block says so and the block describes one session only, never "the GPU performance of TensorNEAT".
The machinery was smoke-tested before this file shipped: one row per CPU target at small settings (`--warmup 1`, two to three timed generations) to prove the protocol, the rocm `error` row and the cross-check verdict; those smoke records are scratch, are not committed, and their timings are deliberately not quoted here because two generations are not a benchmark.
The CPU compiled-backend record is `BACKENDS.md` (llvm-cpu, its own session), and the numpy-only old-vs-new contract is `RESULTS.md` (its own record); both stay what they are, and neither is re-rendered from this protocol.

## Protocol

`gpu_measure.py` measures exactly one `(case, target)` row per process, and one row per process is not a style choice: `peak_rss_mib` is `resource.getrusage(RUSAGE_SELF).ru_maxrss`, a process-lifetime high-water mark, so two rows in one interpreter would report the higher of the two as both.
`matrix` plans `cases × targets` and spawns one `run` subprocess per row **strictly sequentially**, one fresh interpreter each and the same interpreter as the driver, so no two measurements ever overlap on the box.
Per row: `Pipeline.setup()` is timed into `setup_s`; the first `Pipeline.step()` is timed into `first_step_s` and its build cost, `Pipeline.build_seconds`, is reported as `build_s` — the build is never charged to a generation and that step is excluded from every timing sample; then `--warmup` untimed generations and `--generations` timed ones, with stdout/stderr redirected to `os.devnull` for the whole warmup + timed region.
Each timed generation is `time.perf_counter()` wall clock around one `step()` call, which includes the host materialisation of the result; the row keeps mean, median, min, max and every sample, plus `generations_per_second = 1000 / mean`.
`peak_rss_mib` is sampled immediately after the timed region, then `final_generation`, `final_max_fitness`, `device_info` (the target's `backend`/`device`/`compile_options` plus the IREE HIP device query) and the exact `protocol` block are recorded.
`--json` writes the row; its `status` is `ok` or `error`, and any exception during the measurement becomes an `error` row carrying the full traceback plus whatever was already measured, with the last traceback line printed to stderr and exit code 1 — a failed row is recorded, never dropped, and a hard crash that kills the process is turned into an `error` row by the matrix driver (returncode plus stderr tail).
The defaults are the canonical matrix of `benchmarks/cases.py` (`xor_p100`, `xor_p500`, `xor3d_p500`), all three targets, seed 42, three warmup and ten timed generations, one repeat: the case table is imported, never duplicated here.
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

<!-- BEGIN MEASURED (gpu_measure.py) -->
_This block is generated by `benchmarks/gpu_measure.py` (`matrix --out` or `render`) from a measured record; do not hand-edit it. A record has not been measured and committed yet, so the block is still a placeholder._
<!-- END MEASURED (gpu_measure.py) -->

## Limitations

No legacy-GPU comparison exists: the frozen JAX baseline in `benchmarks/legacy_jax` is CPU-only by protocol, so an accelerator number here is informational and is never part of the numpy-only old-vs-new verdict in `RESULTS.md`; there is no "old side" for a GPU.
One device: a single `AMD Instinct MI300X VF` at one device index (`iree.runtime.get_driver("hip").query_available_devices()` reports one entry), so nothing here says how the numbers scale to a second GPU, and no multi-device or multi-stream path is exercised.
`xla` and `tvm` are not attempted: `xla` needs a PJRT C API plugin `.so` that nothing provides on this box, and `tvm` is untested in this repository.
Transfer cost is inside `ms/generation` and not separated from it: the host materialisation of each generation's result happens inside the timed `step()` call, and the HIP host/device traffic is part of that wall clock; this protocol does not report a transfer-only number, and PCIe versus unified-memory behaviour is not instrumented.
`peak RSS` is per process and is the driver's resident memory, not the device's; no device-memory figure is recorded, so a row says nothing about VRAM occupancy.
No per-generation dispatch or op count is instrumented here — `PERF_NUMPY.md` is where executed-op counts live for the CPU interpreter — so an accelerator row gives wall clock and nothing about what the device was asked to do.
Timing is wall clock with a warmup, one repeat, one seed: repeat spread is not measured by this protocol, and a single run of a row is not a distribution.

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

# 3. the full matrix: cases x targets, strictly sequential, one process per row
uv run --extra iree python benchmarks/gpu_measure.py matrix \
    --cases xor_p100,xor_p500,xor3d_p500 --targets numpy,iree-llvm-cpu,iree-rocm \
    --warmup 3 --generations 10 --seed 42 \
    --json benchmarks/results/gpu_2026-11-05.json --out benchmarks/GPU.md

# 4. re-render the measured block from the existing record: no measurement, no JSON rewrite
uv run --extra iree python benchmarks/gpu_measure.py render \
    --json benchmarks/results/gpu_2026-11-05.json --out benchmarks/GPU.md
```

`matrix` exits 0 only when every planned row is `ok`, and 1 as soon as any row is `error` (it prints a `BLOCKED: …` line per blocked row) — with the ROCm rows blocked today, step 3 therefore exits 1 while still writing the record and refreshing the block, which is the intended loud failure and not a broken run.
`render` and `matrix --out` replace only the text between `<!-- BEGIN MEASURED (gpu_measure.py) -->` and `<!-- END MEASURED (gpu_measure.py) -->` and exit 2 with a named error on a missing marker, an unreadable record, a wrong schema or a missing `records` list.
A single row is cheap on the CPU legs and pays the compile on the compiled ones: budget the iree legs at roughly the `iree-compile` cost of the shape plus the timed region, and run the matrix as a background job rather than under a short foreground timeout.
The IREE runtime can print `nanobind: leaked …` lines to stderr when the interpreter exits; that is a binding teardown message emitted *after* the row is written, never a failure and never an `error` row.
The Python-free paths are `--list`, `--help` and nothing else: `run` and `matrix` need `numpy` and `tensorneat`, and the `iree-*` targets need the `iree` extra.

## What this does not change

`RESULTS.md` stays the generated numpy-only old-vs-new record, with its own verdict computed from `results/authoritative.json`; no accelerator row is added to it, and no number in it is touched by this protocol.
`BACKENDS.md` stays the CPU compiled-backend record of its own session, and `PERF_NUMPY.md` stays the numpy-interpreter performance record.
Accelerator rows are never wired into the harness verdict, and that is deliberate: `protocol.comparisons` pairs every old-side record with every new-side record, so an accelerator row in `results/` would enter the numpy-only verdict as if it were a numpy measurement and could flip a timing verdict that the CPU contract owns.
Keeping the accelerator protocol standalone (`benchmarks/gpu_measure.py`, its own `tensorneat-gpu/1` schema, its own record and report) keeps that contract intact: the harness verdict compares what it declares it compares, and an accelerated row is evidence about a different question.
