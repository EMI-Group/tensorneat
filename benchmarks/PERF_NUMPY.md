# numpy reference backend — per-generation cost, before vs after the op-count pass

> **Hand-maintained record** of one focused performance pass on the `numpy` reference interpreter.
> `RESULTS.md` remains the *generated* old-vs-new harness report (rendered from
> `results/authoritative.json`); its new-side rows predate this pass and are stale until the harness is
> re-run. This file holds the measured before/after pair of *this* pass, taken in one session with the
> box staggered between the two trees so drift cancels.

## Headline

One `Pipeline.step` (XOR, `NEAT(pop_size=P, species_size=20, survival_threshold=0.01)`,
`DefaultGenome(num_inputs=2, num_outputs=1, max_nodes=50, max_conns=100,
output_transform=ACT.sigmoid)`, seed 42, `backend="numpy"`, build excluded):

| cell | before mean / median (ms) | after mean / median (ms) | median change | peak RSS before → after | build before → after |
| --- | --- | --- | --- | --- | --- |
| xor p100 | 44.18 / 43.47 | **40.05 / 38.58** | −11 % | 118.2 → 114.4 MiB | 0.288 → 0.255 s |
| xor p500 | 256.41 / 268.39 | **161.37 / 159.90** | **−40 %** | 260.8 → **240.5** MiB | 0.351 → 0.314 s |
| xor p1000 | 408.28 / 342.21 | **280.26 / 270.08** | **−21 %** | 431.6 → **391.5** MiB | 0.435 → 0.399 s |
| xor3d p500 | 289.17 / 290.04 | **177.70 / 176.98** | **−39 %** | 291.3 → **264.7** MiB | 0.335 → 0.320 s |

Executed IR ops per generation (dispatch calls counted on the interpreter): **10 767 → 8 150** at
both populations (−24 %). Peak RSS *fell* while wall time fell: the winning restructuring moved the
speciation distance table back inside a single `evoxc.scan` (one centre per iteration) after first
measuring it as 20 hoisted top-level calls.

`before` = commit `a697531`; `after` = the tree carrying this file. Both cells per row were run in the
same session, alternating trees, one fresh process per cell.

## Where the time was

Profiling 20 generations at pop 500 (`cProfile` + an op-name timer wrapped around
`evoxc.backends.numpy.kernels.dispatch`, plus a host/graph split around `evoxc.run`):

| share | what | evidence |
| --- | --- | --- |
| **0.1 %** | host-side Python in `Pipeline.step` outside `evoxc.run` (state update, `to_numpy`, payload views) | timer around `evoxc.run` vs `Pipeline.step` |
| ~63 % | evoxc interpreter *per-op* overhead — operand resolution, dtype/shape `_validate_result`, `Tensor` construction (~13 µs/op) | `evoxc/backends/numpy/interpreter.py::_run_block` |
| ~37 % | numpy kernel bodies | `evoxc/backends/numpy/kernels/*` |

Executed op mix before the pass: `reshape` 1294, `add` 867, `gather` 755, `select` 704, `slice` 555,
`cast` 529, `constant` 467 per generation. **Two `speciate` scans were 9 160 of 10 767 executed ops
(85 %)**: each scan body is traced once (226 / 232 static ops) but executes `species_size = 20`
times. The forward level loop is only 8 ops per iteration; the mutation chain is ~340 static ops, not
the 1 360 older notes claimed.

## What changed (each measured, each value-identical)

1. **Speciation distance table measured inside one `evoxc.scan`** (`algorithm/neat/_distance.py`,
   `_kernels.py`, `species.py`). Scan 1's centre for slot `i` is the pre-scan carry row `i`, so all
   `S` rows are known before the loop; they are now produced by a fixed-length scan (one centre per
   iteration) instead of 20 top-level `measure_distance` calls. Wall −28 %, peak RSS +74 % when the
   rows were hoisted to the top level of the block — hoisting was then reverted in favour of the scan,
   which keeps one call's transients live instead of twenty: −40 % wall, −8 % RSS versus base.
   Rule: **the interpreter keeps every SSA result of a block alive until that block returns, so a
   kernel's live set is charged once per enclosing block — never hoist a kernel out of a
   `while`/`scan` without measuring peak RSS.**
2. **Digest-side constant and index hoisting** (`genome/operations/distance/*`, `gene/base.py`,
   `gene/blocks.py`, `alignment.py`, `step.py`, `window.py`, `crossover/*`): per-call rank-0 constants
   moved into the per-generation digest (measure now emits zero `constant` ops), key-axis fold via one
   `dot`, trailing pad column instead of a successor clamp, width-1 attribute gather, one flat
   `scatter` for the two unshared mutation write slots, encoded-rank crossover inversion. Kernel
   effects: `measure_distance` 150 → 94 ops (3.27 → 2.39 ms/call), `execute_mutation` 452 → 438 ops,
   `execute_crossover` 152 → 142, `forward` 110 → 98 ops.
3. **AGG innermost-axis reduction folded, not reduced** (`common/functions/agg.py`): the static `K`
   axis is emitted as `K` slices and folded in numpy's `pairwise_sum` association order
   (`_FOLD_MAX_WIDTH = 128`, backend reduce above it), so `sum`/`mean`/`maxabs` stay bit-identical.
   `forward` on the tiled `4P` batch 13.9 → 10.3 ms (−26 %); whole step −2…−4 %.

## Value identity

`test/parity/` compares every kernel against `benchmarks/legacy_jax` bit-for-bit (53 passed).
Independently, 3 generations of the full pipeline at pop 500 and pop 1000 hash identically to
`a697531` over all four payload arrays, the `(P,)` fitness and the returned `State`
(`fitness_history`, `SpeciesState` leaves, `randkey`, `generation`). Full suite: **548 passed,
1 skipped, 0 failed** (the skip is the `requires_iree` case; baseline at `a697531` is 540 passed,
1 skipped, and the 8 new tests are `test/unit/genome/test_distance_digest.py`).

## How to reproduce

```bash
# one cell per process; the venv installs tensorneat editable from the working tree, so pin
# PYTHONPATH when the tree under test is a second checkout of another commit
git archive a697531 | tar -x -C $TMPDIR/base
PYTHONPATH=$TMPDIR/base/src python benchmarks/backend_measure.py --backend numpy --pop 500 --generations 20 --warmup 3
PYTHONPATH=$PWD/src        python benchmarks/backend_measure.py --backend numpy --pop 500 --generations 20 --warmup 3
# same-session old-vs-new (JAX) comparison of the XOR family, without touching the committed reports
python benchmarks/run_benchmark.py --impl both --backend numpy --cases xor_p100,xor_p500,xor_p1000 \
    --generations 50 --warmup 3 --repeats 2 --out $TMPDIR/perf.md --json $TMPDIR/perf.json
```

Protocol: `seed=42`, 3 warmup + 20 timed generations through `Pipeline.step`, build reported
separately and never charged to a generation, peak RSS = `resource.getrusage(RUSAGE_SELF).ru_maxrss`
of the process (a high-water mark, hence one process per cell).

## Same-session old-vs-new harness (XOR family, 50 generations, 2 repeats)

`run_benchmark.py --impl both --backend numpy --cases xor_p100,xor_p500,xor_p1000 --generations 50
--warmup 3 --repeats 2 --out $TMPDIR/perf.md --json $TMPDIR/perf.json` (rendered to `$TMPDIR`, so the
committed `RESULTS.md` is untouched):

| Case | Old (JAX) ms/gen | New (numpy) ms/gen | Speed-up | Timing | Old / new peak RSS |
| --- | --- | --- | --- | --- | --- |
| xor_p100 | 48.72 | 37.86 | 1.29× | PASS | 676.5 / 115.1 MiB |
| xor_p500 | 321.37 | 157.16 | 2.04× | PASS | 722.5 / 240.9 MiB |
| xor_p1000 | 830.10 | 271.70 | 3.06× | PASS | 706.3 / 391.9 MiB |

Read this as *internal* evidence only: this environment resolves `jax 0.10.2`, while the committed
`RESULTS.md` was measured with `jax 0.12.0.1`, whose old-side rows are much faster (157 / 246.76 ms at
p500 / p1000). The contract verdict must therefore be re-rendered by the harness on the same JAX
version as the record it replaces; this file does not claim a contract PASS.
The `xor_p1000` quality row fails against the JAX side, which is a search-trajectory difference, not a
kernel one: the two sides use different PRNGs, and the new tree is bit-identical to `a697531` (same
seed ⇒ same evolution), so that verdict is unchanged by this pass.

## Caveats

- The box drifts ±30 % between sessions; every comparison above is *within* one session, alternating
  the two trees, and the median is the load-bearing statistic (the pop-1000 mean is skewed by one slow
  generation and is reported for completeness).
- The contract verdict against the frozen JAX baseline still has to be re-rendered by
  `run_benchmark.py` (it is generated evidence); this file states the before/after pair of the pass,
  not a re-verdict.
- Build time is one-off per shape signature and improved on every cell, but a `T=50`-step build now
  traces the distance kernel inside a scan body instead of 20 times, which is why it fell.
