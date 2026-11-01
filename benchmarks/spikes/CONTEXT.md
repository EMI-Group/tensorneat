# spikes — measured formulation evidence
## Intent
Keep the measurements that justify the rewrite's *design* (edge-proportional, depth-parallel forward) separate from the end-to-end benchmark, with their conditions spelled out so nobody has to trust a number without a method.
`FORMULATION.md` is the record: it cites the frozen legacy per-node forward cost, the depth-unrolled prototype cost, and the one-off build cost per unrolled depth.
`forward_micro.py` is the runnable micro-benchmark behind those numbers — it measures formulations, not libraries: the legacy side needs `jax` + `benchmarks/legacy_jax` on the path, the numpy prototype needs only `numpy`.
## API Surface
`python benchmarks/spikes/forward_micro.py --mode {legacy,unrolled,all} [--pop 1000] [--max-nodes 50] [--max-conns 150] [--depth 8] [--levels 4,8,16]`.
- `legacy` — times `tensorneat_legacy` `DefaultGenome` `vmap`'d transform+forward for a synthetic population (per-node `u_conns` formulation, build cost reported separately).
- `unrolled` — times a self-contained numpy prototype of the depth-parallel formulation: per-level gather over real edges plus window reduce, Python-unrolled by depth, with its own build cost.
- `all` — both, plus the ratio table that the design argument rests on.
Output is a plain table plus the exact command lines and environment, so a number quoted in `FORMULATION.md` can be re-derived.
## Constraints
- Prototype code here is throwaway by design: it exists to measure a formulation, never to be imported by `src/` or by the end-to-end benchmark.
- Only `numpy` (+ `jax` for the legacy mode) may be imported; nothing from `src/tensorneat` — the whole point is to measure the target formulation before it exists.
- Numbers in `FORMULATION.md` must state population, `max_nodes`, connection count, depth and backend; a number without conditions is a rumour.
## Routing Table
| Path | Purpose |
|---|---|
| `./FORMULATION.md` | Recorded numbers, measurement conditions, how to reproduce, and what the design must achieve. |
| `./forward_micro.py` | Runnable micro-benchmark (`--mode legacy|unrolled|all`). |
| `../CONTEXT.md` | End-to-end protocol and verdict rules (parent level). |
