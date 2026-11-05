# FORMULATION — measured evidence for the depth-parallel forward

## Purpose

This record holds the measurements that justify the rewrite's *design* decision — an edge-proportional, depth-parallel forward — as numbers separate from the end-to-end benchmark.
It answers one question: how much work does each formulation execute per forward, and what does that work cost on this box.
Every number below carries its population, `max_nodes`, connection count, depth, interpreter/backend and box, because a number without conditions is a rumour.

## Recorded reference baseline (not measured by this spike)

`../cases.py` carries the recorded legacy reference as `reference_ms_per_generation = {100: 51.0, 500: 109.0, 1000: 157.0}` ms/generation.
Those are whole-generation numbers (transform + forward + `tell`); reproducing and comparing them is the end-to-end harness's job (`../run_benchmark.py`, verdict in `../RESULTS.md`), not this spike's.
They are a recorded reference, not this box's measured 1-thread condition: the measured `--threads 1` old-side rows here are 35.79 / 127.71 / 252.06 ms/generation (`../THREADS.md`), and the default-thread old-side rows are 38.99 / 295.33 / 807.30 ms/generation (`../RESULTS.md`).

## Where this sits in the measured end-to-end run

`../RESULTS.md` (rendered 2026-11-04T19:01:27Z from `results/authoritative.json`, git `ebeaf1f`, `--threads 0`, new side on the `numpy` backend, 150 timed generations, 3 repeats) measures XOR at **38.99 / 295.33 / 807.30 ms per generation on the old JAX side** and **36.44 / 161.73 / 279.92 ms per generation on the new evoxc side**, at pop 100 / 500 / 1000.
The same CLI with `--threads 1` (record in `../THREADS.md`, 2 repeats) gives **35.79 / 127.71 / 252.06 ms on the old side** and **53.92 / 360.12 / 665.46 ms on the new side** at the same populations; in that record the new side is slower at every population.
That `--threads 1` record resolves jax 0.12.0.1 while the 0-thread `RESULTS.md` was regenerated with jax 0.10.2 (`../THREADS.md` and `../RESULTS.md` environment tables), so the two pairs are not a controlled thread-only comparison.
That is the whole generation (transform + forward + `tell` + the species controller + the problem's `evaluate`), and the 1-thread old-side pop-1000 row (252.06 ms) is 1.6× the recorded 157 ms reference while the 0-thread row (807.30 ms) is 5.1× it; the measured row, not the reference, is what the harness verdict uses.
Of the rows below only the legacy transform+forward at pop 1000 (27.84 ms) maps onto that generation: it is ≈ 3.4 % of the measured 0-thread old-side 807.30 ms (≈ 11.0 % of the 1-thread 252.06 ms), and the legacy forward alone (8.95 ms) is ≈ 1.1 % (≈ 3.6 %).
The remaining ≈ 779 ms of the 0-thread old-side generation (≈ 224 ms of the 1-thread row) is work this spike never times (`tell`, the species controller and its `analysis`, the problem's `evaluate` loop, the per-generation host sync), so these numbers explain the transform+forward part of the old side and no more.
They say even less about the new side's 279.92 ms/generation (or 665.46 ms under `--threads 1`): the 0.69–15.90 ms prototype rows time a level loop over prebuilt tables, with no evoxc build/run, no transform of an evolving population and no host orchestration.
The spike also fixes P = 1000 in every row it records, so the pop 100 and pop 500 end-to-end rows have no counterpart here at all.
Nothing below is a prediction of, or an argument about, that end-to-end timing verdict: the micro-kernel rows cannot adjudicate the end-to-end gap, because they measure formulations rather than the compiled pipeline; the verdict is the harness's, and what this record supports is work per forward and how it scales with depth and real edges.

## Conditions shared by every measured number

When: measured 2026-11-02, 06:17–06:19 UTC, with the `uptime` 1-minute load average between 1.4 and 2.9 around the runs.
Box: Linux 6.8.0-146 x86_64 on a shared 20-core QEMU/KVM guest (`INTEL(R) XEON(R) PLATINUM 8568Y+`, 241611 MiB RAM), `sys.executable` = `…/tn-venv/bin/python` (a virtualenv outside the checkout, so the path is redacted to its environment name), Python 3.12.3, numpy 2.5.4, jax 0.12.0.1 (`cpu` backend for the legacy rows); `evoxc` 0.1.0 is installed but the micro-benchmark never imports it.
Load: co-tenant jobs share those 20 cores, so the load average is part of the conditions and the spread line below — never a single median — is the uncertainty measure.
Threads: `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `NUMEXPR_NUM_THREADS` and `XLA_FLAGS` unset, i.e. runtime defaults — this is the `--threads 0` condition, not the `--threads 1` condition of `../THREADS.md` and not the recorded reference baseline.
Population: P = 1000 synthetic individuals, `max_nodes` = 50 padded node slots, 150 real connections per individual (144 at depth 4, where the level skeleton has no room for the last 6).
Population layout: node keys equal the slot index in the legacy genome's order (slots 0–1 inputs, slot 2 output, slots 3–49 hidden), 2 inputs / 1 output.
Population shape: per individual a random, level-ordered DAG (so no cycles, and the legacy `topological_sort` sees a legal graph), fan-in per node capped by the window K, weights uniform in [-1, 1].
Both formulations are fed the exact same graph and the legacy forward returns finite output for 1000/1000 individuals.
Timing: median of 20 iterations of the steady-state forward after one warmup call; build/compile medians over 5 fresh builds (jax's compilation cache is cleared before each compile sample).
Each value below is the median across repeats of that per-run median (7 `--mode unrolled` runs, 4 `--mode legacy` runs); rows that quote an `(a–b ms over the N runs)` range state the spread of those per-run medians.
Run-to-run spread on this box: legacy forward 8.92–9.12 ms, transform+forward 27.61–28.41 ms, compile 315–336 ms, window-gather at depth 8 7.91–8.08 ms, edge-segment 0.64–0.80 ms, per-edge-loop build 91.7–102.7 ms, segment-sort build 5.2–8.6 ms, vectorised window build 6.8–11.3 ms — i.e. ±5 % on the forward rows and up to ±20 % on the build rows, so only differences of that order or larger are meaningful.

## Legacy per-node forward (`--mode legacy`)

Formulation: one `fori_loop` step per padded node slot, each vmap'ing over all `max_conns` connection slots — work per forward ≈ P × `max_nodes` × `max_conns` = 7.50 M padded element-ops.
Depth is not an input of the legacy formulation (it walks the 50 node slots in topological order); in the depth-8 population the graph itself is 8 levels deep.
Measured: **forward only = 8.95 ms per forward for the whole population (8.92–9.12 ms over the 4 runs), 8.95 µs per individual**, P=1000, `max_nodes`=50, 150 connections, jax 0.12.0.1 CPU on the 20-core box, jit-compiled, transform hoisted out — command in "How to re-run" below.
`transform` + forward together cost **27.84 ms per generation (27.61–28.41 ms over the 4 runs)** at the same conditions, so the graph transform (`topological_sort`'s data-dependent `while_loop`) costs ≈ 18.9 ms of that and is the larger half.
One-off costs at the same conditions: jax compile of the transform+forward program 327 ms median (315–336 ms), plus an informational 1790 ms for `NEAT.setup` (it builds the population and the species controller; it is not inside any timed region).
The recorded reference 157 ms/generation at pop 1000 (`reference_ms_per_generation` in `../cases.py`) is a *whole* generation (transform + forward + `tell` + the problem's `evaluate`), so it is ≈ 5.6× the 27.84 ms transform+forward measured here and is not a competing measurement of the same thing.
On this box the whole-generation old-side rows are 807.30 ms (`--threads 0`, `../RESULTS.md`) and 252.06 ms (`--threads 1`, `../THREADS.md`), so the transform+forward above is 3.4 % / 11.0 % of them, not 5.6×.
The expectation for this row was ≈ 6 ms at P=1000 / `max_nodes`=50 / ~150 connections; the measured value on this box is 8.95 ms (≈ 1.5× the expectation), and 8.95 ms is what the record uses.

## Depth-unrolled prototype, window-gather form (`--mode unrolled`)

Formulation: one Python-unrolled level step per depth level; each step gathers a `(P, max_nodes, K)` window of source values from a prebuilt index table, multiplies by the level's weights, reduces over the window, and applies `tanh(bias + sum)` — work per forward ≈ depth × P × `max_nodes` × K.
Conditions: P=1000, `max_nodes`=50, 150 connections, K = 4 (auto = `ceil(max_conns / real_nodes)`), numpy 2.5.4 interpreter on the same 20-core box, one gather + one window reduce + one activation per level.
Measured at depth 8: **8.02 ms per forward for the whole population (7.91–8.08 ms over the 7 runs), 8.02 µs per individual**, 1.60 M padded element-ops, from `python benchmarks/spikes/forward_micro.py --mode unrolled` with the defaults below.
The expectation for this row was ≈ 14 ms at depth 8 / P=1000 / `max_nodes`=50 / K≈2; the measured 8.02 ms is ≈ 1.8× faster while doing twice the window work (K=4), and 8.02 ms is what the record uses.

## Depth-unrolled prototype, edge-segment form (`--mode unrolled`)

Formulation: the same unrolled level loop, but each step gathers exactly the level's *real* edges (no window padding) and reduces them by destination with one `np.add.reduceat` segment sum — work per forward ≈ P × real_edges = 0.15 M element-ops, i.e. proportional to the connections that exist.
Conditions: P=1000, `max_nodes`=50, 150 connections, numpy 2.5.4 interpreter on the same box; the per-level segment tables are the stable-sorted edge lists plus their segment starts.
Measured at depth 8: **0.72 ms per forward for the whole population (0.71–0.75 ms over the 7 runs), 0.72 µs per individual**, from the same `--mode unrolled` run (both forms are measured side by side on the same tables).
Both prototypes compute the same function on the same graph: their outputs agree to 1.5e-7 absolute (float32 summation order is the only difference), and the script aborts if they ever diverge by more than 1e-5.

## Depth sweep (P=1000, max_nodes=50, 150 connections, 20-core box, numpy 2.5.4)

| depth | K | window-gather forward | edge-segment forward | build (window, per-edge loop) | build (window, vectorised) | build (segment, sort) | window elem-ops | segment elem-ops |
|---|---|---|---|---|---|---|---|---|
| 4 | 4 | 4.08 ms | 0.69 ms | 92.6 ms | 9.5 ms | 8.3 ms | 0.80 M | 0.14 M |
| 8 | 4 | 8.02 ms | 0.72 ms | 94.8 ms | 10.8 ms | 5.8 ms | 1.60 M | 0.15 M |
| 16 | 4 | 15.90 ms | 0.77 ms | 100.5 ms | 10.8 ms | 7.4 ms | 3.20 M | 0.15 M |

The window-gather forward grows linearly with depth (4.08 → 8.02 → 15.90 ms) because its per-level cost is fixed at P × `max_nodes` × K, whatever the graph holds.
The edge-segment forward is flat in depth (0.69 → 0.72 → 0.77 ms) because its per-level cost is the real edges of that level, and the population's total edge count is fixed at 150.
The one-off build cost at depth 4/8/16 is **92.6 / 94.8 / 100.5 ms** when the window tables are filled by the obvious per-edge Python loop, **9.5 / 10.8 / 10.8 ms** when the same tables are filled by one sorted scatter per level (both builders produce identical tables, which the script asserts), and **8.3 / 5.8 / 7.4 ms** for the segment tables' per-level stable sort.
The forwards repeat to ±5 % across the 7 runs (window 4.01–4.11 / 7.91–8.08 / 15.80–16.06 ms, segment 0.64–0.80 ms over all three depths); the build rows spread more (segment sort at depth 4 5.2–8.6 ms, vectorised window build at depth 4 6.8–10.6 ms), which is why they are quoted as medians.
The expectation for the one-off build was 117–396 ms across depths 4/8/16; the measured per-edge-loop build is 93–101 ms (≈ 1.3× lower at the low end) and the measured vectorised build is 9.5–10.8 ms, so 93–101 ms is what the record keeps as the "host-side, one pass per edge" build.

## Design conclusion

Work per node per forward is (depth × K) in the padded window form and (real fan-in × depth) in the segment form, instead of the legacy `max_conns` per node: 4.69× and 50× fewer padded element-ops at depth 8 with the measured 7.50 M / 1.60 M / 0.15 M counts.
Measured against the legacy per-node forward of 8.95 ms, the edge-segment form is 13.0×/12.4×/11.6× faster at depth 4/8/16 (0.69/0.72/0.77 ms), while the window-gather form is 2.19×/1.12×/0.56× — it loses at depth 16 even though it still executes 2.3× fewer padded element-ops than the legacy formulation (3.20 M vs 7.50 M), because every unrolled level pays a full gather + window reduce through the numpy interpreter and that per-level overhead, not the element count, dominates once there are 16 of them.
The forward is the hot spot and the build is amortised over generations: the build is paid once per population/topology at 6–101 ms depending on how the tables are filled, while the forward is paid every generation at 0.69–0.77 ms (edge-segment) or 4.08–15.90 ms (window-gather) across depths 4/8/16, and the legacy side pays both a ≈ 18.9 ms transform and an 8.95 ms forward per generation.
The structural win is that nothing scales with `max_nodes × max_nodes` padding: the padded dimensions are depth, the window K (a real fan-in bound), and the real edge count.

## What these numbers do NOT prove

This is a micro-benchmark of formulations, not the end-to-end benchmark: it does not execute the evoxc kernels, and the end-to-end evidence (timing and quality verdict) lives in `../RESULTS.md`.
The edge-segment and window-gather rows are numpy prototypes, so their absolute milliseconds say nothing about the compiled evoxc backend's cost — they bound the *op count* and show how it scales, which is what the design argument rests on.
The population is synthetic and every individual shares the same node/level skeleton (only the edges differ), so the measurements do not include the cost of building a per-individual level schedule from a real evolving population.
The legacy row includes `transform` and forward of the frozen implementation but not the problem's `evaluate` loop, and it runs with runtime-default thread settings, so it is neither the recorded reference generation baseline nor either end-to-end old-side row.
The build cost cannot be assumed amortised across generations unless the topology is unchanged: NEAT mutates structure every generation, and how much of the table build can be reused or made incremental is not measured here.
Accelerator behaviour (`iree`/`xla`/`tvm`) is not measured; the contract these numbers support is CPU.

## How to re-run

Every number above comes from one of these three commands, run from the repository root with the bench environment (`numpy`, plus `jax` for the legacy rows):

```
python benchmarks/spikes/forward_micro.py --mode all
python benchmarks/spikes/forward_micro.py --mode unrolled
python benchmarks/spikes/forward_micro.py --mode legacy
```

The script prints the conditions table, the environment (python, numpy, installed jax, CPU count, thread variables), the exact command line, then the numbers; `--mode legacy`/`all` exits non-zero with a message when jax is not importable.
Defaults reproduce this record's conditions (one run lands inside the spread documented above; quote a median of repeats, never a single run): `--pop 1000 --max-nodes 50 --max-conns 150 --depth 8 --levels 4,8,16 --iters 20 --build-iters 5 --seed 0`.
Useful variants: `--levels 4,8,16 --depth 8` for the sweep shape, `--window 2` to force a narrower pad (only ~98 of the 150 edges then fit), `--pop 100` or `--pop 500` to match the recorded reference populations, and `--iters 50 --build-iters 7` for tighter medians.
`--mode unrolled` never imports jax: it is the numpy-only prototype and the way to reproduce the segment/window rows on a machine without the `bench` extra.
