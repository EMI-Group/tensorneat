# numpy reference backend — per-generation cost across the three measured passes

> **Hand-maintained record** of three measured passes on the `numpy` reference interpreter.
> Pass 1 is the op-count pass (`a697531` → the tree the pass was recorded on, `f619f46`), pass 2 is the tiled func-fit evaluate plus the RL rollout hoist (`ce5bb88` → `90d97b4`), and pass 3 is the dependency-side A/B of the pinned `evoxc` tag against the numpy-interpreter performance pass on the evoxc task branch (`fe85a534`), measured at `3437e556` whose `src/` is byte-identical to `63f0fc67`; each pass keeps its own measured pair, its own commits and its own protocol below.
> `RESULTS.md` is the *generated* old-vs-new harness report rendered from `results/authoritative.json`, re-measured on the merged tree at `63f0fc67` on 2026-11-06, so its new-side rows carry pass 1, pass 2 and the search-quality fixes and are no longer stale.
> This file is not a re-verdict: it holds the measured before/after pair of each pass — pass 1 and pass 3 with both cells taken in one session, alternating trees so drift cancels, pass 2 with an interleaved same-session no-regression control beside a cross-session `before` column that is labelled indicative.
> Pass 3's headline is the reverse of a null result: the same graph, at the same executed op count, runs **1.07–2.25× faster per generation** (1.56–2.25× on the three dispatch-bound cases) on the post-pass interpreter at **12–37 % lower peak RSS**, and the win lives on an evoxc task branch, so it needs an evoxc release beyond the pinned `v0.1.1` before a `uv sync` user sees it.

## Which tree each record belongs to

Each record in this directory speaks only for the tree and session it names — the provenance table in `CONTEXT.md` is the canonical map — and the merged tree now has its own committed record instead of only these case-scoped pairs.
`results/authoritative.json`, and therefore the committed `RESULTS.md`, is that merged-tree record: re-measured at `63f0fc67` on 2026-11-06 (session 11:31Z–12:09Z, 45/45 runs `ok`), after pass 1, pass 2 and the search-quality fixes, with verdict FAIL on both timing and quality.
Its new-side `numpy` legs are (`ms/gen`) `xor_p100` 41.417, `xor_p500` 119.023, `xor_p1000` 199.160, `xor3d_p100` 36.020, `xor3d_p500` 127.976, `xor3d_p1000` 212.409, `custom_pagie_p500` 1258.042 and `cartpole_p100` 307.164, against old-side (JAX) rows of 38.868, 308.485, 809.342, 24.628, 105.635, 226.660, 480.205 and `n/a`.
`results/xor3d_quality_2026-11-04.json` was measured at `58657f6` on 2026-11-05T00:57Z on a branch that carries the search-quality fixes but not pass 2, so it is post-quality and pre-tiling and is not comparable with the merged-tree record.
Pass 2's `after` record `results/fix_after.json` was measured at `90d97b4` on 2026-11-04T21:51Z, after both of its passes and before the search-quality fixes; it stays git-ignored scratch, and the rows in pass 2 below remain `90d97b4`'s numbers rather than the merged tree's.
Pass 3's sixteen case cells and four control cells were measured at `3437e556` on 2026-11-06 (12:34Z–12:52Z) and live in `$TMPDIR` only; that tree's `src/`, `benchmarks/cases.py` and `benchmarks/harness/` are byte-identical to `63f0fc67`, so its per-generation numbers speak for the merged tree.

## Pass 1 — op-count pass (`a697531` → `f619f46`)

### Headline

One `Pipeline.step` (XOR, `NEAT(pop_size=P, species_size=20, survival_threshold=0.01)`,
`DefaultGenome(num_inputs=2, num_outputs=1, max_nodes=50, max_conns=100,
output_transform=ACT.sigmoid)`, seed 42, `backend="numpy"`, build excluded):

| cell | before mean / median (ms) | after mean / median (ms) | median change | peak RSS before → after | build before → after |
| --- | --- | --- | --- | --- | --- |
| xor p100 | 44.18 / 43.47 | **40.05 / 38.58** | −11 % | 118.2 → 114.4 MiB | 0.288 → 0.255 s |
| xor p500 (3 interleaved rounds) | 258.7 / 262.5 · 192.2 / 191.8 · 259.4 / 262.8 | **162.8 / 160.3 · 161.2 / 159.5 · 158.6 / 156.8** | **−39 %** (median of rounds) | 260.8 → **240.5** MiB | 0.342 → 0.319 s |
| xor p1000 (3 interleaved rounds) | 509.9 / 526.8 · 473.8 / 484.1 · 492.3 / 504.2 | **321.2 / 282.9 · 294.6 / 277.2 · 289.1 / 272.7** | **−45 %** (median of rounds) | 431.3 → **391.7** MiB | 0.424 → 0.395 s |
| xor3d p500 | 289.17 / 290.04 | **177.70 / 176.98** | **−39 %** | 291.3 → **264.7** MiB | 0.335 → 0.320 s |

The three interleaved rounds are the honest form: the base median itself drifts between 191.8 and
262.8 ms at pop 500 depending on what else the shared box is doing, and the new side stays inside
156.8–162.8 ms across all of them, so even the most conservative pairing is −17 % at pop 500 and
−43 % at pop 1000, while the median-of-rounds pairing is −39 % / −45 %.

Executed IR ops per generation (dispatch calls counted on the interpreter): **10 767 → 8 336** at
pop 500 and **8 340** at pop 1000 (−23 %). Note the op count is not monotone with wall time in the
last step: the `common/functions/agg.py` fold *added* 16 executed ops and made `forward` 26 % faster,
so op count is a leading indicator, never a proxy — validate every change end to end.
Peak RSS *fell* while wall time fell: the winning restructuring moved the speciation distance table
back inside a single `evoxc.scan` (one centre per iteration) after first measuring it as 20 hoisted
top-level calls.

`before` = commit `a697531`; `after` = the tree the pass was recorded on (`f619f46`, the commit that
first wrote this pass up), not the merged HEAD. Both cells per row were run in the
same session, alternating trees, one fresh process per cell.

### Where the time was

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

### What changed (each measured, each value-identical)

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

### Value identity

`test/parity/` compares every kernel against `benchmarks/legacy_jax` within the suite's documented
`rtol`/`atol` tolerances (53 passed).
Independently, 3 generations of the full pipeline at pop 500 and pop 1000 hash identically to
`a697531` over all four payload arrays, the `(P,)` fitness and the returned `State`
(`fitness_history`, `SpeciesState` leaves, `randkey`, `generation`). The suite was green on the tree
this pass was recorded on (`f619f46`); the only other count quoted below is 674 passed on pass 2's
tree, and an older count describes that older tree, never HEAD (`../CONTEXT.md` holds the current
merged-tree count).
The counts need the optional extras: `test/parity` needs the `bench` extra (`jax`) and `test/backends`
needs `evoxc[iree]` (without it those three tests skip rather than fail).

### IREE stayed compilable (and one regression was caught and fixed)

The compiled-backend portability gate is `pytest test/backends` (one XOR generation end to end on
`iree` 3.11 llvm-cpu against numpy), not a StableHLO export audit and not a per-kernel `lower`:
`iree-compile` SIGSEGVs (`Error code: -11`) on the mutation rewrite's frontier seed, and neither a
clean export nor a successful standalone compile of `closes_cycle` or `execute_mutation` reproduced
it — only the assembled generation graph did. `genome/operations/mutation/cycle.py` seeded a
`while_loop`-carried value with a **boolean `scatter`**; the same scatter inside the loop body is
fine. The seed is now a one-hot `equal`/`logical_and` (2 traced ops cheaper, bit-identical), and
`pytest test/backends` passes 3/3 plus a real `backend_measure.py --backend iree --pop 100` run.
Rule: **never seed a `while`-carried value with a `scatter`; the only IREE gate is the backend test.**

### How to reproduce

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

### Same-session old-vs-new harness (XOR family, 50 generations, 2 repeats)

`run_benchmark.py --impl both --backend numpy --cases xor_p100,xor_p500,xor_p1000 --generations 50
--warmup 3 --repeats 2 --out $TMPDIR/perf.md --json $TMPDIR/perf.json` (rendered to `$TMPDIR`, so the
committed `RESULTS.md` is untouched):

| Case | Old (JAX) ms/gen | New (numpy) ms/gen | Speed-up | Timing | Old / new peak RSS |
| --- | --- | --- | --- | --- | --- |
| xor_p100 | 48.72 | 37.86 | 1.29× | PASS | 676.5 / 115.1 MiB |
| xor_p500 | 321.37 | 157.16 | 2.04× | PASS | 722.5 / 240.9 MiB |
| xor_p1000 | 830.10 | 271.70 | 3.06× | PASS | 706.3 / 391.9 MiB |

Read this as *internal* evidence only. This box resolves `jax 0.10.2` for both sides of the comparison, and the committed `RESULTS.md` records old-side rows of 308.49 / 809.34 ms at p500 / p1000 on the merged tree at `63f0fc67` (the `ebeaf1f8` record it replaced held 295.33 / 807.30 ms) against 321.37 / 830.10 ms for this 50-generation pair, so the session, not the interpreter version, is what separates them. The contract verdict must therefore
be re-rendered by the harness on the same interpreter and box as the record it replaces; this file
does not claim a contract PASS.
The `xor_p1000` quality row fails against the JAX side, which is a search-trajectory difference, not a
kernel one: the two sides use different PRNGs, and the new tree is bit-identical to `a697531` (same
seed ⇒ same evolution), so that verdict is unchanged by this pass.
### Caveats

- The box drifts ±30 % between sessions; every comparison above is *within* one session, alternating
  the two trees, and the median is the load-bearing statistic (the pop-1000 mean is skewed by one slow
  generation and is reported for completeness).
- The contract verdict is still owned by the generated report; this file states the before/after pair of the pass, not a re-verdict.
  When this pass was recorded the committed `RESULTS.md`/`authoritative.json` pair was the `ebeaf1f8` record (2026-11-04T19:01Z), which postdated this pass and predated pass 2; it has since been re-measured on the merged tree (`63f0fc67`, 2026-11-06) and that record is post-pass-1, post-pass-2 and post-search-quality, so neither form is this pass's `after` column.
- Build time is one-off per shape signature and improved on every cell, but a `T=50`-step build now
  traces the distance kernel inside a scan body instead of 20 times, which is why it fell.

## Pass 2 — tiled func-fit evaluate and the RL rollout hoist (`ce5bb88` → `90d97b4`)

Protocol: same box, `backend="numpy"`, harness seeds 42/43/44 (repeat `r` uses `seed + r`),
`--generations 150 --warmup 3 --repeats 3`. `after` = the case-scoped run on `90d97b4` (the perf
stream's tip, before the search-quality merge, so not the merged HEAD), record
`benchmarks/results/fix_after.json` (a git-ignored scratch record, rendered to `$TMPDIR/fix_after.md`,
so the committed `RESULTS.md` was never rewritten). `before` = the `authoritative.json` record this
pass was measured against, since replaced (that one measured at `ebeaf1f8`, the commit its own
environment fingerprint stores; the report rendered from it, `RESULTS.md`, was committed at `ce5bb88`;
same protocol, same box family). Both sides of the old-vs-new column come from the same pass-2 run.

### Headline

| case | before — new side, `authoritative.json` measured at `ebeaf1f8`<br>ms/gen · peak RSS · build | after — new side, `fix_after.json` @ `90d97b4`<br>per repeat → median ms/gen · peak RSS · build | old side — JAX, same run<br>per repeat → median ms/gen | speed-up old/new |
| --- | --- | --- | --- | --- |
| `custom_pagie_p500` | 3156.73 · 4577.1 MiB · 2.256 s (0.16× of the old 496.45 ms/gen) | 833.81 / 960.63 / 799.88 → **833.81** · 1361.4 MiB · 1.63 s | 629.22 / 520.84 / 539.19 → 539.19 | **0.65×** |
| `cartpole_p100` (new-only: the pre-rewrite library has no evoxc CartPole) | 381.73 · 110.6 MiB · 0.549 s | 268.51 / 310.58 / 275.24 → **275.24** · 111.0 MiB · 0.46 s | n/a | n/a |
| `xor_p500` (control) | 161.73 · 241.3 MiB (1.83× of old) | 178.24 / 178.64 / 177.34 → **178.24** · 228.6 MiB · 0.55 s | 349.98 / 248.17 / 403.98 → 349.98 | 1.96× |
| `xor3d_p500` (control) | 171.16 · 265.5 MiB (0.63× of old) | 188.50 / 176.91 / 183.22 → **183.22** · 237.2 MiB · 0.54 s | 111.84 / 94.68 / 132.84 → 111.84 | 0.61× |

Every `before` cell is the new-side row of the `results/authoritative.json` record of its day (measured
at `ebeaf1f8`, rendered into the `RESULTS.md` committed at `ce5bb88`, since replaced by the merged-tree
record); the `(1.83× of old)` /
`(0.63× of old)` notes are that record's own same-run old-vs-new ratios. Every `after` cell is the
new-side row of `results/fix_after.json` (measured at `90d97b4`, same case config, 3 repeats, median
taken over the three repeats, peak RSS the per-run high-water mark). The old-side column and the
speed-up column are both taken from that pass-2 run, so the ratio never mixes sessions;
`cartpole_p100` is `n/a` on the old side because the frozen pre-rewrite library ships no evoxc
CartPole.

The `custom_pagie_p500` cell is the point of the pass: wall time and peak RSS both fall sharply,
about **−74 % wall and −70 % RSS** against the `before` record (3156.73 → 833.81 ms/gen,
4577.1 → 1361.4 MiB), while the same case's old-side row moved from 496.45 ms/gen in the `before`
record to 539.19 ms/gen in this run — the box, not the code.
`cartpole_p100` falls about **−28 %** (381.73 → 275.24 ms/gen) and remains new-only.
The two XOR control rows move by roughly **±10 % against the `before` column's record** (161.73 → 178.24,
171.16 → 183.22), which is **box drift, not a regression**: an interleaved same-session A/B at pop 500
(20 timed generations, 3 rounds, pre-fix tree `ce5bb88` vs `90d97b4`) gives xor medians base
175.9 / 178.1 / 176.6 ms vs `90d97b4` 179.7 / 172.0 / 170.0 ms, and xor3d base 181.3 / 184.8 / 189.1 ms vs
`90d97b4` 174.3 / 180.4 / 174.9 ms — neutral to slightly better at `S = 4` and `S = 8`.
The old-side rows of `RESULTS.md` and of this record were measured with the same `jax 0.10.2`, and
today's box runs about 10 % slower than the record that supplied the `before` column (old `xor_p500`
349.98 ms here vs 295.33 ms there; the merged-tree record at `63f0fc67` now holds 308.49 ms). So the
same-run old-vs-new ratio is the load-bearing number, never the cross-session delta.

### Where the time was

Root cause measured on one `custom_pagie_p500` generation: **100 % of the excess over `xor_p500` sat
in `problem.evaluate`.** Substituting a zeros kernel for the evaluate call isolates it:

| variant (one `custom_pagie_p500` generation) | measured |
| --- | --- |
| zeros kernel in place of `problem.evaluate` | 182 ms, 218 MiB |
| real evaluate | 2490 ms, 4555 MiB |

The phase split of that fanned evaluate:

| phase | measured |
| --- | --- |
| fanned copies | 240 ms, 2.56 GiB |
| statics | 785 ms, 3.33 GiB |
| level loop | 1306 ms, 3.43 GiB |

Largest single temporary: `(250000, 50, 4)` float32 = 190.7 MiB.
Mechanically, the evaluate fanned the schedule and payload to `P*S` rows and derived the per-pass
statics at `P*S` rows even though that data is sample-invariant, so `gather`/`select` ran at
0.6–0.8 GB/s at that size while only ~1.3 % of gene slots are live — and the numpy interpreter retains
every SSA result of a block until it returns (evoxc upstream).
On the RL side, 58 of the rollout body's 114 direct ops were loop-invariant: 27 % of a `cartpole_p100`
generation, 11 600 of 42 549 executed dispatches.

### What changed

1. **Tiled func-fit evaluate** (`src/tensorneat/host_payload.py`, `genome/step.py`, `genome/window.py`,
   `genome/default.py`, `genome/recurrent.py`): the evaluate no longer fans the schedule/payload to
   `P*S` rows, and no longer derives the per-pass statics at `P*S` rows — that data is
   sample-invariant, so it derives at `P` rows and expands via `by_samples`. A tiled batch wider than
   `EVAL_CHUNK_ROWS = 12 500` rows is evaluated in row groups of `12 500 // S` individuals, with the
   per-group outputs concatenated in individual-major order. Same-session A/B of this fix alone
   (2 rounds, 3 warmup + 30 timed generations, alternating trees, pre-fix tree = the commit before
   the pass): 3282 / 3315 ms → 838 / 841 ms per generation, peak RSS 4578 → 1315 MiB, build
   0.37–0.42 → 0.76–0.79 s. A read-set-only interim step measured 994 ms / 2798 MiB.
2. **RL rollout hoist** (`genome/base.py`, `genome/default.py`, `genome/recurrent.py`,
   `algorithm/{base,neat/neat,hyperneat/hyperneat}.py`, `host_payload.py::_NetworkActFn`,
   `problem/base.py`, `problem/rl/{_common,cartpole,pendulum,mountaincar}.py`, `pipeline.py`): a
   prepare-then-query pair (`Genome.prepare_pass` / `forward_prepared`, `Algorithm.prepare_forward` /
   `forward_prepared`, `act_fn.prepare(batch_rows)`) derives the genome-forward prelude once per
   generation instead of once per environment timestep; `forward` is exactly `prepare_pass` +
   `forward_prepared`, so this is pure hoisting and no semantics move. Loop-invariant direct ops in
   the rollout body: 58 of 114 → 3. Same session, pre-change tree measured in that session: cartpole
   380.8 → 305.7 ms/gen median (1.25×), dispatches 42 549 → 31 604 (the single-step pair 371.7 →
   278.7 ms), peak RSS 110.4 → 111.1 MiB, build 0.55 → 0.46 s. Same class, single-step pairs,
   identical fitness: `repeat_times=4` 50 349 → 31 643 dispatches (587.3 → 417.6 ms, 1.41×, so
   `repeat_times > 1` is now free); Pendulum p100 19 615 → 16 370 dispatches (137 → 103 ms);
   MountainCarContinuous p100 13 590 → 10 345 (100 → 81 ms). Op law after the hoist:
   `ops = 3 804 + 139 × max_step`, and the remaining per-step budget is about 80 forward ops (the
   network output is the next observation, so it cannot be hoisted) plus 54 dynamics ops.

### Value identity

5-generation captures of the per-generation fitness and all four `state.pop` arrays compare
byte-exactly against the pre-fix tree for `custom_pagie_p500`, `xor_p100`, `xor_p500` and `xor_p1000`;
CartPole over 8 generations; Pendulum and MountainCar at `R = 1` and `R = 2`. Executed dispatch counts
are unchanged on the untouched paths (pagie 11 490 before and after the hoist, `xor_p500` 8 263).
Executed ops per generation at the pass-2 tree (`90d97b4`): pagie 11 490 (chunked, unchanged by the
hoist), `xor_p500` 8 263, cartpole 31 604. Tests at the pass-2 tree (`90d97b4`): `uv sync --all-extras` then
`uv run pytest -o addopts="" -q` → 674 passed, 0 failed, 0 skipped; a suite count belongs to the tree
it was run on, so any earlier count quoted for a pass describes that pass's tree, never HEAD.

### IREE

The StableHLO v1 export of the pass-2 generation graph still succeeds, and an `iree` llvm-cpu build plus
one generation of `custom_pagie_p500` agrees with numpy to **2.38e-07** — float32 fusion, not
bit-equal across compilers, so the durability gate stays `pytest test/backends` (which passes 3/3).
This is a durability check at `90d97b4`, not a timing one, and it predates the search-quality merge:
`BACKENDS.md` holds the only compiled-backend per-generation numbers in this node and they belong to
its own earlier session, so no `iree` speed number here describes the merged tree.

### How to reproduce

```bash
# case-scoped after-record: scratch json + report rendered into $TMPDIR, so RESULTS.md stays untouched
python benchmarks/run_benchmark.py --impl both --backend numpy \
    --cases custom_pagie_p500,cartpole_p100,xor_p500,xor3d_p500 \
    --generations 150 --warmup 3 --repeats 3 \
    --json benchmarks/results/fix_after.json --out $TMPDIR/fix_after.md
# interleaved no-regression control at pop 500, alternating trees
git archive ce5bb88 | tar -x -C $TMPDIR/pre_fix
PYTHONPATH=$TMPDIR/pre_fix/src python benchmarks/backend_measure.py --case xor --pop 500 --generations 20 --warmup 3
PYTHONPATH=$PWD/src            python benchmarks/backend_measure.py --case xor --pop 500 --generations 20 --warmup 3
```

`backend_measure.py` reports `setup_s`/`build_s` and runs 3 warmup + 20 timed generations per process,
so one process per cell; the alternating lines above are one *round*, and the reported A/B is three
such rounds.

### Caveats

- The box drifts ±30 % between sessions, so only the interleaved same-session pairs and the same-run
  old-vs-new ratio are load-bearing.
- The `before` column comes from a different session and is indicative, not a same-session control.
- `RESULTS.md` / `results/authoritative.json` held the `ebeaf1f8` record, which predates this pass, while this pass was measured; they have since been re-measured on the merged tree (`63f0fc67`, 2026-11-06), a tree that includes this pass, so the current committed report supersedes the `before` column above instead of repeating it.
- `results/fix_after.json` is git-ignored scratch, not committed evidence.

## Pass 3 — dependency side (pinned `v0.1.1` tag vs the numpy-interpreter pass at `fe85a534`)

Protocol: `3437e556` in the worktree (`src/`, `benchmarks/cases.py` and `benchmarks/harness/` are byte-identical to `63f0fc67`, so this is the merged tree), `backend="numpy"`, new side only, one fresh process per cell, `--generations 150 --warmup 3 --repeats 1` at seed 42, `--threads 0`, cells strictly sequential, and per case the cell order **A B B A** so each leg is measured on both sides of the other (round 1 = A then B, round 2 = B then A).
The box was idle for three of the four cases (per-run `loadavg_start` 0.28–1.97); the `xor_p500` window ran under outside load of 2–3 and both legs sat inside it, the pinned cells reading `loadavg_start`/`loadavg_end` 0.56/3.05 and 2.35/1.97 against the pass cells' 3.05/2.60 and 2.60/2.35.
Each leg owns a venv over one shared base (identical CPython 3.12.3 and numpy 2.5.4, the lock's 3.12 numpy), installed from a `git archive` copy of its tree with `uv pip install --no-deps --reinstall`:
**leg A** is the pinned dependency `[tool.uv.sources]` resolves — `evoxc` git tag `v0.1.1` (`be145ee`, what `uv.lock` records); **leg B** is the numpy-interpreter performance pass — evoxc task-branch tip `fe85a534` (`v0.1.1-127-gfe85a53`, 127 commits past `v0.1.1` and 93 past `main`, carrying `3c3c8e0` per-block dispatch plans + flat value store + static-result fast path + dead-value release, `5deca1b` the concrete-int dim fast path, `dda3ec0` the kernel hot path and `51d5829` the shared `Device('cpu', 0)`); **leg C** (control only) is `main` at `474bf72` (`v0.1.1-34-g474bf72`), the newest pure-`main` tree, measured here as the control that makes the pinned tag's number legible: a `main`-based checkout is exactly what a tag-versus-`HEAD` A/B of this dependency compares.
All three dists report `evoxc 0.1.0` in their metadata, so each leg is identified by a sha256 over every `*.py` under the installed package: **`d0d1f4b4b7777f1b`** (150 files) pinned, **`5498cbc31b361688`** (153 files) branch tip, **`90b801c7bbc89a8d`** (151 files) `main`, and the recipe is below.

### Headline

| case | leg A — pinned `v0.1.1`<br>A1 · A2 (ms/gen) | leg B — branch tip `fe85a534`<br>B1 · B2 (ms/gen) | matched pairs<br>A1/B1 · A2/B2 | leg A mean / leg B mean | peak RSS A / B (MiB) | build A / B (s) | trajectory |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `xor_p100` | 42.52 · 42.12 | 19.36 · 19.46 | 2.20× · 2.16× | **2.18×** (42.32 → 19.41) | 108.8 / 95.8–96.0 | 0.305–0.308 / 0.290–0.295 | bit-identical |
| `xor_p500` | 115.06 · 114.52 | 73.03 · 74.08 | 1.58× · 1.55× | **1.56×** (114.79 → 73.56) | 191.3–191.5 / 119.4–120.5 | 0.467–0.487 / 0.380–0.387 | bit-identical |
| `cartpole_p100` | 302.49 · 298.31 | 130.28 · 136.28 | 2.32× · 2.19× | **2.25×** (300.40 → 133.28) | 107.6–107.7 / 94.8–95.0 | 0.441–0.448 / 0.336–0.339 | bit-identical |
| `custom_pagie_p500` | 1250.56 · 1201.60 | 1119.99 · 1165.43 | 1.12× · 1.03× | **1.07×** (1226.08 → 1142.71) | 1293.1–1294.9 / 888.6–894.3 | 1.515–1.571 / 1.442–1.470 | bit-identical |

**The dependency side carries 1.07–2.25× of this backend's per-generation cost — 1.56–2.25× wherever the graph is dispatch-bound.** Nothing in this repository changed between the cells — same `src/`, same seed, same case config, same 150 generations — so every millisecond in that column is the interpreter loop.
The pass moves wall clock, peak RSS and build time in the same direction on every case, and the two rounds reverse the order without moving the verdict: the pinned tag's own three `xor_p100` cells (42.52, 42.12 and 41.31 in the control round below) span 3 % and `main` adds a fourth at 42.65, while the pass leg sits at 19.36–19.46.
The size of the win tracks how much of an op is *dispatch* rather than arithmetic: `xor_p100` and `cartpole_p100` are thousands of small tensor ops per generation and gain 54–56 %, `xor_p500` gains 36 %, and `custom_pagie_p500` — whose ops run on 250 000-row arrays — gains only 7 % even though it executes the same kind of graph.
Peak RSS follows the pass's dead-value release rather than its dispatch plans: −12 % on the two pop-100 cases, −37 % on `xor_p500` (191.3 → 120.5 MiB) and −31 % on `custom_pagie_p500` (1293.1 → 888.6 MiB), i.e. the cases that keep the most live SSA values gain the most memory.
Build time falls too (0.487 → 0.380 s on `xor_p500`, 1.571 → 1.442 s on pagie, 0.441 → 0.336 s on cartpole): the branch also removed a per-op `Device('cpu', 0)` allocation on the `Tensor` path.

### The control: why a tag-vs-`main` A/B reads null

Leg C is the tree the newest pure-`main` commit provides (`474bf72`, 34 commits past the tag) and it carries **none** of the pass: `dispatch_plan`/`_BlockPlan`, the flat `_values` store and the dead-value release first appear at `fe85a534`, 93 commits further up the task branch.
Measured in this same session, minutes apart, with `xor_p100` and `xor_p500` in both orders:

| control pair (cell order) | pinned `v0.1.1` | `main` `474bf72` | `main` vs pinned |
| --- | --- | --- | --- |
| `xor_p100` (C1 then A3) | 41.31 (A3) · 42.52 (A1) | 42.65 | +0.3 % vs A1, +3.2 % vs A3 |
| `xor_p500` (A3 then C1) | 112.70 (A3) | 112.93 | +0.2 % |

So `main` and the pinned tag are the same speed within run-to-run drift, and both are 1.5–2.2× slower than the branch tip on the same box.
That is the trap in a tag-versus-checkout reading: `git describe` of `main` promises 34 commits of post-tag work and not one of them touches the numpy path, so a diff taken against it reports the dependency as inert while the pass exists one branch away — with the same `evoxc 0.1.0` metadata on both sides.
`main` is not stale for this purpose in general (it is the newest tree by date); it simply does not contain the numpy-interpreter work, which lives on the task branch until it is released.

### Where the gain sits: identical op counts, lower per-op cost

Executed interpreter kernels per generation, counted on both legs for all four cases by wrapping every entry of evoxc's numpy `KERNEL_TABLE` after the build and before the first run (the counts are equal to the op on both legs):

| case | executed ops / generation | leg A µs / op | leg B µs / op | per-op cost |
| --- | --- | --- | --- | --- |
| `xor_p100` | 3 505 | 12.07 | 5.54 | **−54 %** |
| `xor_p500` | 7 922 | 14.49 | 9.29 | **−36 %** |
| `cartpole_p100` | 21 638 | 13.88 | 6.16 | **−56 %** |
| `custom_pagie_p500` | 11 227 | 109.21 | 101.78 | **−7 %** |

The op counts are a property of this repository's graphs, so the two legs dispatch exactly the same work; what changed is what one dispatch costs.
That is the branch's own stated mechanism (`3c3c8e0`): the plan resolves each op's kernel and its operand/result value ids once per block instead of walking an env stack per operand per execution, the flat value store drops the env-stack push/pop, the static-result fast path replaces a symbolic shape evaluation per result with one `(dtype, shape)` compare, and the dead-value release bounds the live set.
The µs/op column also explains pagie: at ~110 µs per op the Python-side dispatch cost is noise next to the numpy kernel body, so the same pass buys 7 % there and 54 % on XOR.
`xor_p100`'s 3 505 dispatches at pop 100 versus `xor_p500`'s 7 922 confirm the fixed cost is per-shape, not per-individual: the population scales the arrays, the op count scales with the schedule.

### Trajectories are bit-identical

Both legs reproduce the same search exactly: the per-generation `best_fitness` / `mean_fitness` / `valid_count` curves (152 entries each), `final_best_fitness` and `generations_to_target` are equal element-for-element across all four cells of all four cases — `xor_p100` ends on `-1.636888373468537e-05` (target at generation 40), `xor_p500` on `-0.0` (32), `custom_pagie_p500` on `-0.006567014846950769` (95) and `cartpole_p100` on `200.0` (no informative target, `gens_to_target=False`).
That is what the design predicts: the dependency supplies the interpreter loop and the splitmix64 stream over this repository's own numpy kernels, and the pass changes how an op executes — never which ops run, in which order, or with which RNG words.
It is also what the branch's own record claims ("bit-identical fitness trajectories, unchanged executed op counts"), measured here from this side of the dependency.

### What this means for the pin

`pyproject.toml` and `uv.lock` keep `evoxc` on tag `v0.1.1`, and this pass does not change that: the pin is a released tag, and the improvement is on an **unreleased task branch**.
Concretely, a `uv sync` user today gets leg A's numbers; moving the pin is what would hand them 2.18× on `xor_p100`, 1.56× on `xor_p500`, 2.25× on `cartpole_p100` and 1.07× on `custom_pagie_p500`, at 12–37 % lower peak RSS — and it needs an evoxc release that contains `3c3c8e0`, `5deca1b`, `dda3ec0` and `51d5829`, not a checkout pointer.
Any such move is a protocol change with a measurement attached: the new-side rows of `RESULTS.md` (and of every committed record whose new side is `numpy`) would land 1.07–2.25× faster, so the report is regenerated in the same commit that moves the pin.
Until then this section is evidence *about a candidate dependency*, not about the shipped one, and the pinned leg is the number a user actually gets.

### How to reproduce

Three venvs over one base interpreter, so all legs exist at once and a cell is a plain interpreter swap — no reinstall between cells, and no `uv sync` to undo it:

```bash
export PATH="$HOME/.local/bin:$PATH"
mkdir -p $TMPDIR/evoxc-pinned $TMPDIR/evoxc-pass $TMPDIR/evoxc-main
git -C /root/evoxc archive v0.1.1  | tar -x -C $TMPDIR/evoxc-pinned   # leg A — the pinned tag
git -C /root/evoxc archive fe85a534 | tar -x -C $TMPDIR/evoxc-pass    # leg B — task-branch tip
git -C /root/evoxc archive 474bf72  | tar -x -C $TMPDIR/evoxc-main    # leg C — main (control)
uv venv $TMPDIR/venvA --python 3.12
uv pip install --python $TMPDIR/venvA/bin/python numpy networkx matplotlib sympy
uv pip install --python $TMPDIR/venvA/bin/python --no-deps --reinstall $TMPDIR/evoxc-pinned
cp -a $TMPDIR/venvA $TMPDIR/venvB && cp -a $TMPDIR/venvA $TMPDIR/venvC
uv pip install --python $TMPDIR/venvB/bin/python --no-deps --reinstall $TMPDIR/evoxc-pass
uv pip install --python $TMPDIR/venvC/bin/python --no-deps --reinstall $TMPDIR/evoxc-main
# one cell = one fresh process, strictly sequential; per case the order is A1 B1 B2 A2
$TMPDIR/venvA/bin/python benchmarks/run_benchmark.py --impl new --backend numpy \
    --cases xor_p500 --generations 150 --warmup 3 --repeats 1 --seed 42 --threads 0 \
    --json $TMPDIR/p3_xor_p500_A1.json --out $TMPDIR/p3_xor_p500_A1.md
$TMPDIR/venvB/bin/python benchmarks/run_benchmark.py --impl new --backend numpy \
    --cases xor_p500 --generations 150 --warmup 3 --repeats 1 --seed 42 --threads 0 \
    --json $TMPDIR/p3_xor_p500_B1.json --out $TMPDIR/p3_xor_p500_B1.md   # then B2, then A2
```

All three legs self-report `evoxc 0.1.0`, so verify which bytes are installed before believing a cell:

```bash
$TMPDIR/venvA/bin/python - <<'PY'
import hashlib, importlib.metadata as md, pathlib, evoxc
pkg = pathlib.Path(evoxc.__file__).parent
h = hashlib.sha256()
for f in sorted(pkg.rglob("*.py")):
    h.update(f.relative_to(pkg).as_posix().encode())
    h.update(f.read_bytes())
print(md.version("evoxc"), len(list(pkg.rglob("*.py"))), h.hexdigest()[:16])
PY
# pinned tag (v0.1.1)                    0.1.0 150 d0d1f4b4b7777f1b
# task-branch tip (v0.1.1-127-gfe85a53)  0.1.0 153 5498cbc31b361688
# main (v0.1.1-34-g474bf72)              0.1.0 151 90b801c7bbc89a8d
```

Executed ops per generation come from wrapping `KERNEL_TABLE` **after** the build and **before** the first `run` (a v0.1.1 dispatcher reads the table per op; the post-pass interpreter caches the kernels into its block plan at the plan's first build, so a wrapper installed later is never called):

```python
from evoxc.backends.numpy import kernels
exe = pipeline._executable_for(state)      # build + load; no block plan exists yet
counts = {}
for name, kernel in list(kernels.KERNEL_TABLE.items()):
    def counting(ctx, op, operands, _k=kernel, _n=name):
        counts[_n] = counts.get(_n, 0) + 1
        return _k(ctx, op, operands)
    kernels.KERNEL_TABLE[_n] = counting
pipeline.step(state)
total = sum(counts.values())
```

Quality identity is checked from the records themselves: load two cells' JSON and compare `runs[0].quality.{best_fitness,mean_fitness,valid_count,final_best_fitness,generations_to_target}` element-for-element (all 152 entries agree, on every pair of cells of every case).
Never `uv run` a leg: `uv run` performs an implicit sync that reverts to the pinned tag in `.venv`, which silently measures leg A twice — drive the venv interpreter directly.
Every leg record is a git-ignored scratch file written to `$TMPDIR`, with `--out` beside it, so no committed report is touched.

### Caveats

- One repeat per seed per leg: each cell is one 150-generation curve, so the uncertainty is the spread between the two reversed rounds, not a repeat distribution. That spread is small (pinned `xor_p100` 42.52 / 42.12 / 42.65 / 41.31 across four cells, i.e. 3 %) and the leg gap is 54 %, but no cell carries a confidence interval.
- A single run's own 150 sample spread is far wider than the leg medians: leg A's generation samples span 32.5–73.2 ms on `xor_p100` (15.4–25.9 ms on leg B), so only medians are quoted and a per-sample comparison would be meaningless.
- `custom_pagie_p500` is the least resolved cell: 7 % on the two-cell means, its matched pairs reading 1.12× and 1.03×, while the pinned leg's own two cells differ by 4 % (1250.56 vs 1201.60). The direction is consistent in both orders; the magnitude is not resolved by two cells.
- The `xor_p500` window ran under outside load (load average 2–3, one pinned cell ending on 3.05) with both legs interleaved inside it; each leg's two cells stayed within 1 % of each other (pinned 115.06 / 114.52, pass 73.03 / 74.08), so the load did not produce the gap.
- Leg C is a diagnostic, not a third A/B arm: one cell per case, enough to show `main` ≈ pinned and not enough to resolve a difference between them.
- The legs are `git archive` copies installed with `--no-deps --reinstall`, not editable installs: the fingerprint proves which bytes ran and the venvs share one base interpreter and one numpy build, but both trees are pure Python, so nothing here depends on compile flags.
- The op-count column is a dispatch count, never a proxy for wall time — and that is exactly the pass's shape: the count does not move at all (3 505 / 7 922 / 21 638 / 11 227 on *both* legs) while wall time halves, so the entire win is per-dispatch cost.
- This pass changes no code and no dependency here: `pyproject.toml`/`uv.lock` keep `v0.1.1`, and the measurement's job is to price the *next* pin, not to move it.
- Numerical identity between the legs is expected and observed, and says nothing about the frozen oracle — `test/parity/` owns fidelity against `benchmarks/legacy_jax`.

## Rules the performance passes establish

**Rule A — the interpreter's blocks are the unit of memory.** The pinned numpy interpreter keeps every
SSA result of an interpreter block alive until that block returns, so bound the row group a problem asks
the network to evaluate (pass 2's `EVAL_CHUNK_ROWS` tiling) and never hoist a kernel out of a
`while`/`scan` without measuring peak RSS (pass 1's speciation table paid +74 % RSS for exactly that
hoist, and only fell back to −8 % once the rows returned to a scan).
The post-pass interpreter of pass 3 releases a value at its last use inside the block instead of at
block exit, which bounds the live set by liveness rather than by block extent — that is where its
−31 % / −37 % RSS comes from at unchanged op counts — but it cannot shrink the peak of one wide
fan-out, so width stays a tiling decision either way.
**Rule B — loop-invariant data is prepared once per generation, never once per query.** Anything a
forward derives from loop-invariant input (the genome-forward prelude, the per-pass statics, the
schedule) belongs to a prepare step whose result the query path reuses, which is why the RL body's
loop-invariant direct ops went 58 → 3 and why the evaluate stopped deriving statics at `P*S` rows.
Both rules are memory rules before they are speed rules: in both passes peak RSS fell alongside wall
time (260.8 → 240.5 MiB and 431.3 → 391.7 MiB at pop 500/1000 in pass 1, 4577.1 → 1361.4 MiB for
pagie in pass 2), so a change that moves wall time without moving RSS deserves a second look.
Neither rule replaces the identity checks: each pass was accepted only after byte-exact captures of
the fitness and payload arrays against the tree it replaced.
**Standing caveat — executed op count is a leading indicator, never a proxy for wall time:** pass 1's
`agg` fold added 16 executed ops and made `forward` 26 % faster, while pass 2 cut dispatches 42 549 →
31 604 and still banked only 1.25×, so every change is validated end to end on wall time, peak RSS and
the identity checks, in one session with the trees interleaved.
**Rule C — a dependency A/B measures a commit, never a branch name.** A tag-versus-`HEAD` diff of a
dependency is only as good as the tree that `HEAD` happens to point at: here `main` was 34 commits past
the tag, carried no interpreter work at all, and measured within ±3 % of the pinned tag, while the pass
that halves dispatch-bound generations sat 93 commits further up a task branch — same repository, same
version metadata (`0.1.0`), a 2.18× difference in wall time. Name the commit and fingerprint the bytes
(`d0d1f4b4b7777f1b` / `5498cbc31b361688` / `90b801c7bbc89a8d` here), and read a null result as "these
two trees are the same tree for this purpose", not as "this dependency has nothing to give".
Pass 3 also sharpens the standing caveat from the other side: executed op count says nothing about wall
time on its own — it stayed exactly constant (3 505 / 7 922 / 21 638 / 11 227) while wall time fell
54 % / 36 % / 56 % / 7 %, so what a pass buys is decided by which part of an op dominates (dispatch
overhead on XOR and CartPole, numpy kernel bodies on Pagie), never by how many ops run.
