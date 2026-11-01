# tensorneat — package architecture

## Intent
This package is the NEAT/HyperNEAT toolchain built on the EvoX Compiler (`evoxc`), replacing the pre-rewrite JAX implementation module for module.
It owns three things only: (1) **tensorization** of variable topology into fixed-shape arrays, (2) the **kernel contract** shared by genome / algorithm / problem, and (3) the **compile site** (`pipeline.py`) that turns one generation into a single evoxc executable.

## Kernel Contract (binding for all child nodes)
**Batch-first.** The population is axis 0 and every shape is static at build time.
`vmap` is never applied over a function containing `scan`/`cond`/`while_loop` (evoxc defers that), so kernels are written already batched, with control flow only at their outermost level — and genome kernels unroll the depth loop at trace time instead of using `scan`, because a region op cannot be batched or differentiated.
**Split gene payload.** `evoxc` has no bitcast/reinterpret op, so a mixed-dtype gene row is impossible: `node_keys (P,max_nodes) int32` + `node_attrs (P,max_nodes,A) float32`, `conn_keys (P,max_conns,2..3) int32` + `conn_attrs (P,max_conns,W) float32`. Absent rows are `I_INF` in key arrays and NaN in attr arrays. Mutation therefore twiddles int keys and float attrs separately.
**Execution schedule.** `Genome.transform(state, node_keys, node_attrs, conn_keys, conn_attrs)` returns the `Transformed` pytree defined in `./common/CONTEXT.md` (authority: that table, not this one) — topological level, per-node incoming-edge window of static width `K` (`in_src`, `in_conn`, `in_valid`), `node_exists`, `input_slot`/`output_slot`, `overflow`.
`common/tensorization.py::derive_schedule` derives it **inside the graph** with cost `O(P·(C log C + N·K·D))` — never the legacy `max_nodes²` matrix.
**Edge proportionality.** A forward pass touches work proportional to real connections and depth: the pre-rewrite `u_conns` formulation gathered a full `max_nodes`-wide column per node per genome, which is the main reason the legacy baseline is slow.
**Genome.** `initialize(state, key, pop_size)` (pop_size is static; evoxc keys are rank-0 so randomness is one key plus batched output shapes); `transform`; `forward(state, transformed, node_keys, node_attrs, conn_keys, conn_attrs, inputs (P,*in)) -> (P,*out)`; `grad(...) -> (loss (P,), (grad_attrs, ...))` built by `evoxc.grad` over the float attrs only (the schedule, being `sort`-derived, has no AD rule); `execute_mutation/crossover/distance`.
Host-only helpers (`network_dict`, `sympy_func`, `visualize`, `repr`, `hash`) stay pure Python and never enter a graph.
**Algorithm.** `ask(state) -> pop`, `transform(state, pop)`, `forward(...)`, `tell(state, fitness) -> state`, `num_inputs`, `num_outputs`, `pop_size`.
**Problem.** `evaluate(state, key, act_fn, transformed) -> fitness (P,)` executed inside the active trace; **`key` is a single rank-0 evoxc key** (there is no batched key type), `act_fn` maps `(P*S, *input_shape)` to `(P*S, *output_shape)`, and `input_shape`/`output_shape` are **per-sample**. Datasets and rollouts are tiled into the population axis inside the graph; NaN fitness becomes `-inf` as the last op of every problem. `evaluate` is a plain helper, never `@evoxc.defn` (a defn cannot call another defn).
**Randomness.** `evoxc.random` stateless keys only; no `numpy.random`/`jax.random` anywhere on the compute path. Draw order intentionally differs from the legacy library, so parity is kernel-level on identical inputs, never trajectory-level.
**State.** Immutable host container (`State.register/update/remove` + pytree registration) holding configuration, evoxc keys and tensors.

## Contract Reconciliations (authoritative; resolves the divergences found in Phase 1)
1. **Gene payload is split and integer-keyed**: `node_keys (P,max_nodes) int32` (`I_INF` padding), `node_attrs (P,max_nodes,A) float32`, `conn_keys (P,max_conns,2) int32` (`I_INF` padding; a third column only when the conn gene uses `historical_marker`), `conn_attrs (P,max_conns,W) float32` (NaN padding). Keys are never floats; `host_graph`/`tensorization` detect absence with `key != I_INF`.
2. **Payload arity is four arrays**, bundled as `genome.utils.GenomeTensors` (a `typing.NamedTuple` in order `node_keys, node_attrs, conn_keys, conn_attrs`): `algorithm.transform(state, pop=None)` and `execute_*(state, key, payload, ...)` accept the container or the four positional arrays; `algorithm.forward(state, transformed, inputs, pop=None)` and the genome kernel `forward(state, transformed, node_keys, node_attrs, conn_keys, conn_attrs, inputs)` spell the arrays out. `BaseAlgorithm.forward(state, transformed, inputs)` has no `pop`; `SpeciesState` centers store the split payload (never packed ints).
3. **`Transformed` authority is `common/tensorization.py`**, with the Phase-1 review amendments: fields `src, dst, edge_valid, in_conn, in_src, in_valid, level, node_level, topo_key, input_slot, output_slot, node_exists, edge_overflow` where `in_valid (P,N,K)` marks a live window slot, `node_level (P,N) int32` is the topological level (authority for `level`), and `edge_overflow (P,)` counts connections that did not fit the static `K`.
4. **Public names**: `common.tensorization.derive_schedule` (no `build_schedule` alias); **all static arguments are positional** (evoxc cannot pass keyword-only statics to `build`, so no `functools.partial` dance at call sites); `common.host_graph.{decode_graph, topological_sort, find_useful_nodes, node_depth, check_cycles}` and `common.__init__` re-exports the first three; `ACT.name_of`/`AGG.name_of` replace `get_func_name`; `common.tools.to_numpy(tensor)` is the host-side array accessor.
5. **No default `State()` in signatures**: `setup(self, state=None)` builds `State()` in the body (a default argument would run at class-definition time and break `import tensorneat`).
6. **Genome helper surface used by `algorithm/`**: `node_exists`, `conn_exists`, `max_node_key`, `max_conn_marker`, `conn_gene.uses_historical_marker`, `node_gene.from_attrs`, `conn_gene.from_attrs`, plus statics `num_inputs/num_outputs/max_nodes/max_conns/max_in_degree/node_gene.length/conn_gene.length`.
7. **`CartPole`** validates `fitness_target <= max_step` at construction and demotes NaN logits to `-inf` fitness (an explicit `evoxc.isnan` guard, not an implicit NaN comparison).

## Public API Surface
`tensorneat.Pipeline`, `tensorneat.algorithm.{NEAT,HyperNEAT,HyperNEATFeedForward}`, `tensorneat.genome.{DefaultGenome,RecurrentGenome,DefaultNode,BiasNode,OriginNode,DefaultConn,OriginConn,...}`, `tensorneat.problem.{XOR,XOR3d,CustomFuncFit,FuncFit,CartPole}`, `tensorneat.common.{ACT,AGG,State,TensorNEATError}`.
`pipeline.py` exposes `Pipeline(algorithm, problem, seed=42, fitness_target=1, generation_limit=1000, backend="numpy", device=None, compile_options=None, is_save=False, save_dir=None, is_detailed=True, show_problem_details=False, pop_batch_size=None)` with `setup`, `step`, `auto_run`, `show`.
`Pipeline.step` is the single compile site: one evoxc graph per shape signature `(P, max_nodes, max_conns, max_in_degree, key_span)` covering ask → transform → evaluate → tell; `auto_run` caches the executable per shape, rebuilds when `overflow` occurs, and loops on the host for logging / checkpointing / early stop.
Fitness is maximized; NaN or infinite fitness is mapped to `-inf` before selection so broken topologies cannot win.
There is no `problem.jitable` flag any more: whether a graph compiles is the backend's business and `evoxc.BackendError` propagates.

## Constraints
- `import tensorneat` must not import `jax`, `torch`, `brax`, `gymnax`, `mujoco` (release-gate test); optional backends are pulled in only by `evoxc.build(backend=...)`.
- evoxc idioms are mandatory: `@evoxc.defn` graph entry points, plain helpers otherwise, `evoxc.numpy as enp` for tensor creation (float literals become f64 constants otherwise), `evoxc.random` for keys, `evoxc.constant` only for genuinely static data, non-negative axes in `concatenate`/`split`.
- No eager fallback: if a backend cannot compile a kernel the error propagates; we never quietly compute on the host.
- NaN padding semantics of the pre-rewrite library are preserved (`test/parity/` depends on them); `select` chains stand in for `lax.switch`.
- Errors: user/config validation raises `common.errors.TensorNEATError` (and its subclasses); tensor/backend errors are never caught and re-wrapped.
- Files ≤ ~1000 lines; shared math lives in `common/`, never duplicated per genome/algorithm.

## Routing Table
| Path | Responsibility |
|---|---|
| `./common` | Backend primitives: `State`, errors, RNG/tensor helpers, **tensorization (execution schedule)**, `ACT`/`AGG` registries, host graph decoding, sympy codegen. |
| `./genome` | Genome classes, gene types, genetic operators, forward/grad kernels over the schedule. |
| `./algorithm` | `NEAT` (ask/tell/speciation), `HyperNEAT` and substrates — batched, fixed-length scans, no vmap-over-control-flow. |
| `./problem` | Task layer: `FuncFit`/`XOR`/`XOR3d`/`CustomFuncFit` and the evoxc-native `CartPole` rollout. |
| `./pipeline.py` | Host-side compile site, generation loop, logging/checkpointing, `show`. |
| `./__init__.py` | Re-exports `Pipeline` and the four subpackages. |
