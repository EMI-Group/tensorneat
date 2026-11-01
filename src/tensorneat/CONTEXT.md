# tensorneat — package architecture

## Intent
This package is the NEAT/HyperNEAT toolchain built on the EvoX Compiler.
It owns three things only: (1) the **tensorization** of variable topology into fixed-shape arrays, (2) the **kernel contract** shared by genome / algorithm / problem, and (3) the **compile site** (`pipeline.py`) that turns one generation into a single evoxc executable.
Everything below `genome/`, `algorithm/`, `problem/`, `common/` implements that contract; nothing outside `src/` may be imported by it.

## Kernel Contract (binding for all child nodes)
**Batch-first.** Population is always axis 0 and shapes are static at build time. `vmap` is NOT used over functions containing `scan`/`cond`/`while_loop` (evoxc defers that) — every kernel is written once, already batched, with control flow appearing at its outermost level.
**Gene payload.** `nodes (P, max_nodes, node_len)` and `conns (P, max_conns, conn_len)` are integer arrays carrying gene keys in the leading columns (`nodes[:,:,:1]` = node key, `conns[:,:,:2]` = (in_key, out_key)); float gene attributes are bitcast into int slots exactly as the pre-rewrite library did, so genetic operators remain integer-bit-twiddling kernels. Unused rows are NaN-encoded.
**Execution schedule.** `Genome.transform(state, nodes, conns)` returns a batch-first `Transformed` pytree describing *how to execute*: topological order, a per-node incoming-edge table (padded to a static width, `I_INF`-free), and per-node depth/level plus aggregation masks.
The schedule derivation is an in-graph kernel (no host round-trip) and its cost must be edge-proportional, never `max_nodes²` per node.
**Edge proportionality.** A forward pass touches work proportional to real connections and depth: the pre-rewrite `u_conns` formulation gathered a full `max_nodes`-wide column per node per genome, and that quadratic waste is the main reason the legacy baseline is slow.
**Genome.** `initialize(state, key) -> (nodes, conns)`; `transform`; `forward(state, transformed, inputs (P,*in)) -> (P,*out)`; `grad(state, nodes, conns, inputs, loss_fn) -> (loss (P,), grads)`; `execute_mutation/crossover/distance(state, keys, ...)`. Host-only helpers (`network_dict`, `sympy_func`, `visualize`, `repr`, `hash`) stay pure Python and never enter a graph.
**Algorithm.** `ask(state) -> pop`, `transform(state, pop)`, `forward(state, transformed, inputs)`, `tell(state, fitness) -> state`, `num_inputs`, `num_outputs`, `pop_size`.
**Problem.** `evaluate(state, keys, act_fn, transformed) -> fitness (P,)` where `act_fn` maps a `(P, *input_shape)` batch to `(P, *output_shape)`; datasets are tiled into the population axis inside the graph instead of being vmapped per sample. `input_shape`/`output_shape` are declared.
**State.** Immutable host container (`State.register/update/remove`) holding configuration, evoxc RNG keys and tensors; the only mutation-free store used by `Pipeline`.

## Public API Surface
`tensorneat.Pipeline`, `tensorneat.algorithm.{NEAT,HyperNEAT}`, `tensorneat.genome.{DefaultGenome,RecurrentGenome,...}`, `tensorneat.problem.{XOR,XOR3d,CustomFuncFit,CartPole}`, `tensorneat.common.{ACT,AGG,State}`.
`pipeline.py` exposes `Pipeline(algorithm, problem, seed=42, fitness_target=1, generation_limit=1000, backend="numpy", device=None, compile_options=None, is_save=False, save_dir=None, is_detailed=True, show_problem_details=False)` with `setup`, `step`, `auto_run`, `show`.
`Pipeline.step` is the single compile site: one evoxc graph per (shape signature) covering ask → transform → evaluate → tell; `auto_run` caches the executable and loops on the host (logging, checkpointing, early stop).
Fitness is maximized; NaN/infinite fitness is mapped to `-inf` before selection so broken topologies cannot win.

## Constraints
- `import tensorneat` must not import `jax`, `torch`, `brax`, `gymnax`, `mujoco` (release-gate test); optional backends are pulled in only by `evoxc.build(backend=...)`.
- evoxc idioms are mandatory: `@evoxc.defn` graph entry points, plain helpers otherwise, `evoxc.numpy as enp` for tensor creation, `evoxc.random` for keys (no `numpy.random` inside kernels), `evoxc.constant` only for genuinely static data.
- No eager fallback: if a backend cannot compile a kernel, the error propagates (`evoxc.BackendError`); we never quietly compute on host.
- NaN padding semantics of the pre-rewrite library are preserved (`test/parity/` depends on them).
- Errors: user/config validation raises `common.errors.TensorNEATError`; tensor/backend errors are never caught and re-wrapped.
- Files ≤ ~1000 lines; shared math lives in `common/`, never duplicated per genome/algorithm.

## Routing Table
| Path | Responsibility |
|---|---|
| `./common` | Backend primitives: `State`, errors, RNG/tensor helpers, **tensorization (execution schedule)**, `ACT`/`AGG` registries, sympy codegen. |
| `./genome` | Genome classes, gene types, genetic operators, forward/grad kernels in schedule form. |
| `./algorithm` | `NEAT` (ask/tell/speciation), `HyperNEAT` and substrates — batched, control flow at kernel top level only. |
| `./problem` | Task layer: function fitting (`XOR`, `XOR3d`, `CustomFuncFit`) and an evoxc-native `CartPole` rollout. |
| `./pipeline.py` | Host-side compile site, generation loop, logging/checkpointing, `show`. |
| `./__init__.py` | Re-exports `Pipeline` and the four subpackages. |
