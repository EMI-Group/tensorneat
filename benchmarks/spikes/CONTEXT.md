# spikes — measured formulation evidence
## Intent
Keep the measurements that justify the rewrite's *design* (edge-proportional, depth-parallel forward) separate from the end-to-end benchmark, with their conditions spelled out so nobody has to trust a number without a method.
`FORMULATION.md` is the record: it cites the frozen legacy per-node forward cost, the depth-unrolled prototype cost in both forms, the one-off build cost per unrolled depth, and a cross-reference to the measured end-to-end generation times in `../RESULTS.md` that says which rows do and do not explain them.
`forward_micro.py` is the runnable micro-benchmark behind those numbers — it measures formulations, not libraries: the legacy side needs `jax` plus the frozen `benchmarks/legacy_jax` tree (the script inserts that directory on `sys.path` itself), the numpy prototype needs only `numpy` and never imports `jax` (the version is read from the installed metadata).
`iree_probe.py` is the runnable evidence behind `../BACKENDS.md`: it builds the benchmark's own graphs on `backend="iree"` and exits 0/1 per probe, so a non-deterministic failure shows up as a pass/fail pattern over fresh processes.
## API Surface
`python benchmarks/spikes/forward_micro.py --mode {legacy,unrolled,all} [--pop 1000] [--max-nodes 50] [--max-conns 150] [--depth 8] [--levels 4,8,16] [--window 0] [--iters 20] [--build-iters 5] [--seed 0]`.
- `legacy` — times `tensorneat_legacy` `DefaultGenome` `vmap`'d transform+forward for a synthetic population (per-node `u_conns` formulation, build cost reported separately).
- `unrolled` — times two forms of a self-contained numpy prototype of the depth-parallel formulation, both Python-unrolled by depth: a `(P, max_nodes, window)` gather + window reduce per level, and a real-edge gather + `np.add.reduceat` segment sum per level; each form reports its own one-off build cost and the two are asserted to compute the same outputs.
- `all` — both, plus the ratio table that the design argument rests on.
`python benchmarks/spikes/iree_probe.py --mode {sanity,scatter,node-levels,while,pipeline} [--repeat N] [--case NAME] [--bypass scatter[,uint32]] [--generations N] [--max-nodes N] [--carries N] [--body select|plain] [--pop N] [--window N]`.
- `sanity` / `scatter` / `node-levels` / `while` — isolate the iree/stablehlo constructs (a 64x64 matmul, a rank-3 axis-0 row scatter, `common.tensorization.node_levels`, a synthetic `while_loop`) before blaming TensorNEAT for them.
- `pipeline` — the real case through `Pipeline.setup`/`step` on `backend="iree"`, with probe-only trace-time bypasses (`scatter`, `uint32`) installed on the imported modules by that process only.
- `--repeat N` re-runs the same probe in N fresh interpreters and prints the pass/fail pattern.
`forward_micro.py` prints the conditions table, the environment (python, numpy, installed jax, CPU count, thread variables), the exact command line and then the numbers; `iree_probe.py` prints one `OK …`/`FAIL …` line per probe.
## Constraints
- Prototype code here is throwaway by design: it exists to measure a formulation, never to be imported by `src/` or by the end-to-end benchmark.
- Only `numpy` (+ `jax` for the legacy mode) may be imported by `forward_micro.py`; nothing from `src/tensorneat` — the whole point is to measure the target formulation before it exists.
- `iree_probe.py` imports only stdlib at module scope, so the numpy-free harness path (`run_benchmark.py --list`) never depends on it; `evoxc`/`numpy`/`tensorneat` are imported inside the modes.
- Numbers in `FORMULATION.md` must state population, `max_nodes`, connection count, depth and backend; a number without conditions is a rumour.
- Timing rows are medians across repeats of the per-run median, with the observed spread next to them (±5 % on the forward rows, up to ±20 % on the build rows) and the date, box and load average in the shared-conditions block.
## Routing Table
| Path | Purpose |
|---|---|
| `./FORMULATION.md` | Recorded numbers with conditions, the end-to-end cross-reference, how to reproduce, and what the design must achieve. |
| `./forward_micro.py` | Runnable micro-benchmark (`--mode legacy|unrolled|all`). |
| `./iree_probe.py` | Runnable iree-backend probes behind `../BACKENDS.md`. |
| `./rocm` | ROCm (MI300X) evidence: the pre-3.13 abort-era report plus the current per-generation cost model (dispatch count, device-vs-host time, phase split, ceiling) — see `./rocm/CONTEXT.md`. |
| `../CONTEXT.md` | End-to-end protocol and verdict rules (parent level). |
