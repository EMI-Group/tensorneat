# numpy reference backend — per-generation cost, before vs after the two measured performance passes

> **Hand-maintained record** of two focused performance passes on the `numpy` reference interpreter.
> Pass 1 is the op-count pass (`a697531` → its pass-1 tree) and pass 2 is the tiled func-fit evaluate
> plus the RL rollout hoist (`ce5bb88` → `90d97b4`); each pass keeps its own measured pair, its own
> commits and its own protocol below.
> `RESULTS.md` remains the *generated* old-vs-new harness report (rendered from
> `results/authoritative.json`); its new-side rows **predate both passes** and are stale until the
> harness is re-run. This file is not a re-verdict: it holds the measured before/after pair of each
> pass — pass 1 with both cells taken in one session, alternating trees so drift cancels, pass 2 with
> an interleaved same-session no-regression control beside a cross-session `before` column that is
> labelled indicative.

## Pass 1 — op-count pass (`a697531` → pass-1 tree)

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

`before` = commit `a697531`; `after` = the tree carrying this file. Both cells per row were run in the
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

`test/parity/` compares every kernel against `benchmarks/legacy_jax` bit-for-bit (53 passed).
Independently, 3 generations of the full pipeline at pop 500 and pop 1000 hash identically to
`a697531` over all four payload arrays, the `(P,)` fitness and the returned `State`
(`fitness_history`, `SpeciesState` leaves, `randkey`, `generation`). Full suite: **551 passed,
0 failed, 0 skipped** (baseline at `a697531` with the same extras: 540 passed, 1 skipped; the 8 new
tests are `test/unit/genome/test_distance_digest.py`, and the 3 `test/backends` IREE tests now pass
because the `evoxc[iree]` extra is installed and the blocker below is fixed).

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

Read this as *internal* evidence only: this environment resolves `jax 0.10.2`, while the committed
`RESULTS.md` was measured with `jax 0.12.0.1`, whose old-side rows are much faster (157 / 246.76 ms at
p500 / p1000). The contract verdict must therefore be re-rendered by the harness on the same JAX
version as the record it replaces; this file does not claim a contract PASS.
The `xor_p1000` quality row fails against the JAX side, which is a search-trajectory difference, not a
kernel one: the two sides use different PRNGs, and the new tree is bit-identical to `a697531` (same
seed ⇒ same evolution), so that verdict is unchanged by this pass.

### Caveats

- The box drifts ±30 % between sessions; every comparison above is *within* one session, alternating
  the two trees, and the median is the load-bearing statistic (the pop-1000 mean is skewed by one slow
  generation and is reported for completeness).
- The contract verdict against the frozen JAX baseline still has to be re-rendered by
  `run_benchmark.py` (it is generated evidence); this file states the before/after pair of the pass,
  not a re-verdict.
- Build time is one-off per shape signature and improved on every cell, but a `T=50`-step build now
  traces the distance kernel inside a scan body instead of 20 times, which is why it fell.

## Pass 2 — tiled func-fit evaluate and the RL rollout hoist (`ce5bb88` → `90d97b4`)

Protocol: same box, `backend="numpy"`, harness seeds 42/43/44 (repeat `r` uses `seed + r`),
`--generations 150 --warmup 3 --repeats 3`. `after` = the case-scoped run on HEAD `90d97b4`, record
`benchmarks/results/fix_after.json` (a git-ignored scratch record, rendered to `$TMPDIR/fix_after.md`,
so the committed `RESULTS.md` was never rewritten). `before` = the committed
`benchmarks/results/authoritative.json` (HEAD `ce5bb88`, same protocol, same box family). Both sides
of the old-vs-new column come from the same pass-2 run.

### Headline

| case | before — new side, `authoritative.json` @ `ce5bb88`<br>ms/gen · peak RSS · build | after — new side, `fix_after.json` @ `90d97b4`<br>per repeat → median ms/gen · peak RSS · build | old side — JAX, same run<br>per repeat → median ms/gen | speed-up old/new |
| --- | --- | --- | --- | --- |
| `custom_pagie_p500` | 3156.73 · 4577.1 MiB · 2.256 s (0.16× of the old 496.45 ms/gen) | 833.81 / 960.63 / 799.88 → **833.81** · 1361.4 MiB · 1.63 s | 629.22 / 520.84 / 539.19 → 539.19 | **0.65×** |
| `cartpole_p100` (new-only: the pre-rewrite library has no evoxc CartPole) | 381.73 · 110.6 MiB · 0.549 s | 268.51 / 310.58 / 275.24 → **275.24** · 111.0 MiB · 0.46 s | n/a | n/a |
| `xor_p500` (control) | 161.73 · 241.3 MiB (1.83× of old) | 178.24 / 178.64 / 177.34 → **178.24** · 228.6 MiB · 0.55 s | 349.98 / 248.17 / 403.98 → 349.98 | 1.96× |
| `xor3d_p500` (control) | 171.16 · 265.5 MiB (0.63× of old) | 188.50 / 176.91 / 183.22 → **183.22** · 237.2 MiB · 0.54 s | 111.84 / 94.68 / 132.84 → 111.84 | 0.61× |

Every `before` cell is the new-side row of the committed `results/authoritative.json` (measured at
`ce5bb88`, rendered into the committed `RESULTS.md`); the `(1.83× of old)` / `(0.63× of old)` notes are
that record's own same-run old-vs-new ratios. Every `after` cell is the new-side row of
`results/fix_after.json` (measured at `90d97b4`, same case config, 3 repeats, median taken over the
three repeats, peak RSS the per-run high-water mark). The old-side column and the speed-up column are
both taken from that pass-2 run, so the ratio never mixes sessions; `cartpole_p100` is `n/a` on the
old side because the frozen pre-rewrite library ships no evoxc CartPole.

The `custom_pagie_p500` cell is the point of the pass: wall time and peak RSS both fall sharply,
about **−74 % wall and −70 % RSS** against the `before` record (3156.73 → 833.81 ms/gen,
4577.1 → 1361.4 MiB), while the same case's old-side row moved from 496.45 ms/gen in the committed
record to 539.19 ms/gen in this run — the box, not the code.
`cartpole_p100` falls about **−28 %** (381.73 → 275.24 ms/gen) and remains new-only.
The two XOR control rows move by roughly **±10 % against the committed record** (161.73 → 178.24,
171.16 → 183.22), which is **box drift, not a regression**: an interleaved same-session A/B at pop 500
(20 timed generations, 3 rounds, pre-fix tree `ce5bb88` vs `90d97b4`) gives xor medians base
175.9 / 178.1 / 176.6 ms vs HEAD 179.7 / 172.0 / 170.0 ms, and xor3d base 181.3 / 184.8 / 189.1 ms vs
HEAD 174.3 / 180.4 / 174.9 ms — neutral to slightly better at `S = 4` and `S = 8`.
The old-side rows of `RESULTS.md` and of this record were measured with the same `jax 0.10.2`, and
today's box runs about 10 % slower than the committed record (old `xor_p500` 349.98 ms here vs
295.33 ms there). So the same-run old-vs-new ratio is the load-bearing number, never the
cross-session delta.

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
Executed ops per generation at HEAD: pagie 11 490 (chunked, unchanged by the hoist), `xor_p500` 8 263,
cartpole 31 604. Tests at HEAD: `uv run pytest -o addopts="" -q` → 674 passed (659 before the pass);
`-m parity` → 53 passed; `test/backends` → 3 passed; `test/release_gate_no_jax.py` → 4 passed.

### IREE

The StableHLO v1 export of the new generation graph still succeeds, and an `iree` llvm-cpu build plus
one generation of `custom_pagie_p500` agrees with numpy to **2.38e-07** — float32 fusion, not
bit-equal across compilers, so the durability gate stays `pytest test/backends` (which passes 3/3).

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
- `RESULTS.md` / `results/authoritative.json` remain a generated record from before this pass; only a
  full `run_benchmark.py` re-run replaces them, which this file does not claim to do.
- `results/fix_after.json` is git-ignored scratch, not committed evidence.

## Rules the two passes establish

**Rule A — the interpreter's blocks are the unit of memory.** The numpy interpreter keeps every SSA
result of an interpreter block alive until that block returns, so bound the row group a problem asks
the network to evaluate (pass 2's `EVAL_CHUNK_ROWS` tiling) and never hoist a kernel out of a
`while`/`scan` without measuring peak RSS (pass 1's speciation table paid +74 % RSS for exactly that
hoist, and only fell back to −8 % once the rows returned to a scan).
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
