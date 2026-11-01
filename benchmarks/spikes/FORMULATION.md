# FORMULATION — measured evidence for the depth-parallel forward

## Purpose

This record holds the measurements that justify the rewrite's *design* decision — an edge-proportional, depth-parallel forward — as numbers separate from the end-to-end benchmark.
It answers one question: how much work does each formulation execute per forward, and what does that work cost on this box.
Every number below carries its population, `max_nodes`, connection count, depth, interpreter/backend and box, because a number without conditions is a rumour.

## Declared baseline (not measured by this spike)

`../CONTEXT.md` declares the frozen legacy cost on a 1-thread-bound CPU box as 51 ms/generation at pop 100, 109 ms/generation at pop 500 and 157 ms/generation at pop 1000.
Those are whole-generation numbers (transform + forward + `tell`); reproducing and comparing them is the end-to-end harness's job (`../run_benchmark.py`, verdict in `../RESULTS.md`), not this spike's.

## Conditions shared by every measured number

Box: Linux 6.8.0 x86_64, 20 logical CPUs, `sys.executable` = the bench venv, Python 3.12.3, numpy 2.5.4, jax 0.12.0.1 (`cpu` backend for the legacy rows).
Threads: `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `NUMEXPR_NUM_THREADS` and `XLA_FLAGS` unset, i.e. runtime defaults — this is *not* the 1-thread-bound box of the declared baseline.
Population: P = 1000 synthetic individuals, `max_nodes` = 50 padded node slots, 150 real connections per individual (144 at depth 4, where the level skeleton has no room for the last 6).
Population layout: node keys equal the slot index in the legacy genome's order (slots 0–1 inputs, slot 2 output, slots 3–49 hidden), 2 inputs / 1 output.
Population shape: per individual a random, level-ordered DAG (so no cycles, and the legacy `topological_sort` sees a legal graph), fan-in per node capped by the window K, weights uniform in [-1, 1].
Both formulations are fed the exact same graph and the legacy forward returns finite output for 1000/1000 individuals.
Timing: median of 20 iterations of the steady-state forward after one warmup call; build/compile medians over 5 fresh builds (jax's compilation cache is cleared before each compile sample).
Run-to-run spread on this box over the runs behind this record: legacy forward 8.8–9.6 ms, window-gather at depth 8 7.3–8.1 ms, edge-segment 0.63–0.77 ms — i.e. ±5 %, so only differences of that order or larger are meaningful.

## Legacy per-node forward (`--mode legacy`)

Formulation: one `fori_loop` step per padded node slot, each vmap'ing over all `max_conns` connection slots — work per forward ≈ P × `max_nodes` × `max_conns` = 7.50 M padded element-ops.
Depth is not an input of the legacy formulation (it walks the 50 node slots in topological order); in the depth-8 population the graph itself is 8 levels deep.
Measured: **forward only = 8.82 ms per forward for the whole population (min 8.39 / max 9.42 ms), 8.82 µs per individual**, P=1000, `max_nodes`=50, 150 connections, jax 0.12.0.1 CPU on the 20-core box, jit-compiled, transform hoisted out — command in "How to re-run" below.
`transform` + forward together cost **27.24 ms per generation (min 26.00 / max 31.69 ms)** at the same conditions, so the graph transform (`topological_sort`'s data-dependent `while_loop`) costs ≈ 18.4 ms of that and is the larger half.
One-off costs at the same conditions: jax compile of the transform+forward program 318 ms median, plus an informational 1722 ms for `NEAT.setup` (it builds the population and the species controller; it is not inside any timed region).
The declared 157 ms/generation at pop 1000 in `../CONTEXT.md` is the *whole* generation (transform + forward + `tell` + the problem's `evaluate`) on a 1-thread-bound box, so it is ≈ 5.8× the 27.24 ms transform+forward measured here and is not a competing measurement of the same thing.
The expectation for this row was ≈ 6 ms at P=1000 / `max_nodes`=50 / ~150 connections; the measured value on this box is 8.82 ms (≈ 1.5× the expectation), and 8.82 ms is what the record uses.

## Depth-unrolled prototype, window-gather form (`--mode unrolled`)

Formulation: one Python-unrolled level step per depth level; each step gathers a `(P, max_nodes, K)` window of source values from a prebuilt index table, multiplies by the level's weights, reduces over the window, and applies `tanh(bias + sum)` — work per forward ≈ depth × P × `max_nodes` × K.
Conditions: P=1000, `max_nodes`=50, 150 connections, K = 4 (auto = `ceil(max_conns / real_nodes)`), numpy 2.5.4 interpreter on the same 20-core box, one gather + one window reduce + one activation per level.
Measured at depth 8: **7.64 ms per forward for the whole population (min 7.29 / max 8.25 ms), 7.64 µs per individual**, 1.60 M padded element-ops.
The expectation for this row was ≈ 14 ms at depth 8 / P=1000 / `max_nodes`=50 / K≈2; the measured 7.64 ms is ≈ 1.8× faster while doing twice the window work (K=4), and 7.64 ms is what the record uses.

## Depth-unrolled prototype, edge-segment form (`--mode unrolled`)

Formulation: the same unrolled level loop, but each step gathers exactly the level's *real* edges (no window padding) and reduces them by destination with one `np.add.reduceat` segment sum — work per forward ≈ P × real_edges = 0.15 M element-ops, i.e. proportional to the connections that exist.
Conditions: P=1000, `max_nodes`=50, 150 connections, numpy 2.5.4 interpreter on the same box; the per-level segment tables are the stable-sorted edge lists plus their segment starts.
Measured at depth 8: **0.68 ms per forward for the whole population (min 0.66 / max 0.75 ms), 0.68 µs per individual.**
Both prototypes compute the same function on the same graph: their outputs agree to 1.5e-7 absolute (float32 summation order is the only difference), and the script aborts if they ever diverge by more than 1e-5.

## Depth sweep (P=1000, max_nodes=50, 150 connections, 20-core box, numpy 2.5.4)

| depth | K | window-gather forward | edge-segment forward | build (window, per-edge loop) | build (window, vectorised) | build (segment, sort) | window elem-ops | segment elem-ops |
|---|---|---|---|---|---|---|---|---|
| 4 | 4 | 3.74 ms | 0.65 ms | 86.2 ms | 6.2 ms | 4.6 ms | 0.80 M | 0.14 M |
| 8 | 4 | 7.64 ms | 0.68 ms | 91.1 ms | 7.3 ms | 5.5 ms | 1.60 M | 0.15 M |
| 16 | 4 | 15.87 ms | 0.80 ms | 90.8 ms | 10.8 ms | 7.7 ms | 3.20 M | 0.15 M |

The window-gather forward grows linearly with depth (3.74 → 7.64 → 15.87 ms) because its per-level cost is fixed at P × `max_nodes` × K, whatever the graph holds.
The edge-segment forward is flat in depth (0.65 → 0.68 → 0.80 ms) because its per-level cost is the real edges of that level, and the population's total edge count is fixed at 150.
The one-off build cost at depth 4/8/16 is **86.2 / 91.1 / 90.8 ms** when the window tables are filled by the obvious per-edge Python loop, **6.2 / 7.3 / 10.8 ms** when the same tables are filled by one sorted scatter per level (both builders produce identical tables, which the script asserts), and **4.6 / 5.5 / 7.7 ms** for the segment tables' per-level stable sort.
The expectation for the one-off build was 117–396 ms across depths 4/8/16; the measured per-edge-loop build is 86–91 ms (≈ 1.35× lower at the low end) and the measured vectorised build is 6–11 ms, so 86–91 ms is what the record keeps as the "host-side, one pass per edge" build.

## Design conclusion

Work per node per forward is (depth × K) in the padded window form and (real fan-in × depth) in the segment form, instead of the legacy `max_conns` per node: 4.69× and 50× fewer padded element-ops at depth 8 with the measured 7.50 M / 1.60 M / 0.15 M counts.
Measured against the legacy per-node forward of 8.82 ms, the edge-segment form is 13.6×/12.9×/11.1× faster at depth 4/8/16 (0.65/0.68/0.80 ms), while the window-gather form is 2.36×/1.16×/0.56× — the padded form stops winning once depth × K (64 at depth 16) exceeds the 150 connection slots it replaces.
The forward is the hot spot and the build is amortised over generations: the build is paid once per population/topology at 5–91 ms depending on how the tables are filled, while the forward is paid every generation at 0.65–0.80 ms (edge-segment) or 3.74–15.87 ms (window-gather) across depths 4/8/16, and the legacy side pays both a ≈ 18.4 ms transform and an 8.82 ms forward per generation.
The structural win is that nothing scales with `max_nodes × max_nodes` padding: the padded dimensions are depth, the window K (a real fan-in bound), and the real edge count.

## What these numbers do NOT prove

This is a micro-benchmark of formulations, not the end-to-end benchmark: it does not execute the evoxc kernels, and the end-to-end evidence (timing and quality verdict) lives in `../RESULTS.md`.
The edge-segment and window-gather rows are numpy prototypes, so their absolute milliseconds say nothing about the compiled evoxc backend's cost — they bound the *op count* and show how it scales, which is what the design argument rests on.
The population is synthetic and every individual shares the same node/level skeleton (only the edges differ), so the measurements do not include the cost of building a per-individual level schedule from a real evolving population.
The legacy row includes `transform` and forward of the frozen implementation but not the problem's `evaluate` loop, and it runs with runtime-default thread settings, so it is not the declared 1-thread-bound generation baseline.
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
Defaults reproduce this record exactly: `--pop 1000 --max-nodes 50 --max-conns 150 --depth 8 --levels 4,8,16 --iters 20 --build-iters 5 --seed 0`.
Useful variants: `--levels 4,8,16 --depth 8` for the sweep shape, `--window 2` to force a narrower pad (only ~98 of the 150 edges then fit), `--pop 100` or `--pop 500` to match the declared baseline's population sizes, and `--iters 50 --build-iters 7` for tighter medians.
`--mode unrolled` never imports jax: it is the numpy-only prototype and the way to reproduce the segment/window rows on a machine without the `bench` extra.
