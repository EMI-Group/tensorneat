# tensorneat — package architecture

## Intent
This package is the NEAT/HyperNEAT toolchain built on the EvoX Compiler (`evoxc`), replacing the pre-rewrite JAX implementation module for module.
It owns three things only: (1) **tensorization** of variable topology into fixed-shape arrays, (2) the **kernel contract** shared by genome / algorithm / problem, and (3) the **compile site** (`pipeline.py`) that turns one generation into a single evoxc executable.

## Kernel Contract (binding for all child nodes)
**Batch-first.** The population is axis 0 and every shape is static at build time.
Kernels are written already batched, with control flow only at their outermost level; `evoxc.vmap` is never applied to a kernel.
**Split gene payload.** `evoxc` has no bitcast/reinterpret op, so a mixed-dtype gene row is impossible: `node_keys (P,max_nodes) int32` + `node_attrs (P,max_nodes,A) float32`, `conn_keys (P,max_conns,2..3) int32` + `conn_attrs (P,max_conns,W) float32`. Absent rows are `I_INF` in key arrays and NaN in attr arrays. Mutation therefore twiddles int keys and float attrs separately.
**Execution schedule.** `Genome.transform(state, *arrays)` accepts the four-array payload (or the two key arrays) and returns the `Transformed` pytree defined in `./common/CONTEXT.md` (authority: that table, not this one) — `in_src`/`in_conn`/`in_valid (P,N,K)`, `node_level`/`node_exists`, `input_slot`/`output_slot`, `edge_overflow (P,)`.
`common/tensorization.py::derive_schedule` derives it **inside the graph** at cost `O(P·(C log C + N·K·D))` — never the legacy `max_nodes²` matrix.
**Edge proportionality.** A forward pass touches work proportional to real connections and depth (`(depth+1) × max_nodes × K` elementwise/reduce/gather ops), never `max_nodes²` and never `max_depth`.
**Genome.** `initialize(state, key, pop_size)` (pop_size is static; evoxc keys are rank-0 so randomness is one key plus batched output shapes); `transform(state, *arrays) -> Transformed`; `forward(state, transformed, node_keys, node_attrs, conn_keys, conn_attrs, inputs (P,*in)) -> (P,*out)`; `grad(...) -> (loss (P,), (grad_attrs, ...))` built by `evoxc.grad` over the float attrs only (the `sort`-derived schedule has no AD rule); `execute_mutation/crossover/distance`; and the digest pair `prepare_distance`/`measure_distance` that `speciate` consumes.
The feedforward forward is one `evoxc.while_loop` over topological levels; the recurrent one unrolls its `activate_time` ticks in Python.
Host-only helpers (`network_dict`, `sympy_func`, `visualize`, `repr`, `hash`) stay pure Python and never enter a graph.
**Algorithm.** `ask(state) -> payload`, `transform(state, pop=None)`, `forward(state, transformed, inputs, pop=None)`, `tell(state, fitness) -> State`, `num_inputs`, `num_outputs`, `pop_size`, plus the host-only `show_details`.
**Problem.** `evaluate(state, key, act_fn, transformed) -> fitness (P,)` executed inside the active trace; **`key` is a single rank-0 evoxc key** (there is no batched key type), `act_fn` maps `(P*S, *input_shape)` to `(P*S, *output_shape)`, and `input_shape`/`output_shape` are **per-sample**. Datasets and rollouts are tiled into the population axis inside the graph; NaN fitness becomes `-inf` as the last op of every problem. `evaluate` is a plain helper, never `@evoxc.defn` (a defn cannot call another defn).
**Randomness.** `evoxc.random` stateless keys only; no `numpy.random`/`jax.random` anywhere on the compute path. Draw order intentionally differs from the legacy library, so parity is kernel-level on identical inputs, never trajectory-level.
**State.** Immutable host container (`State.register/update/remove/registered_keys` + pytree registration) holding configuration, evoxc keys and tensors.

## Contract Reconciliations (authoritative; resolves the divergences found in Phase 1)
1. **Gene payload is split and integer-keyed**: `node_keys (P,max_nodes) int32` (`I_INF` padding), `node_attrs (P,max_nodes,A) float32`, `conn_keys (P,max_conns,2) int32` (`I_INF` padding; a third column only when the conn gene uses `historical_marker`), `conn_attrs (P,max_conns,W) float32` (NaN padding). Keys are never floats; `host_graph`/`tensorization` detect absence with `key != I_INF`.
2. **Payload arity is four arrays**, bundled as `genome.utils.GenomeTensors`: `initialize(state, key, pop_size)`, `transform(state, payload)`, `forward(state, transformed, payload, inputs)`, `execute_*(state, key, payload, ...)`. `BaseAlgorithm.forward(state, transformed, payload, inputs)` therefore takes the payload too, and `SpeciesState` centers store the split payload (never packed ints).
3. **`Transformed` authority is `common/tensorization.py`**, with the Phase-1 review amendments: fields `src, dst, edge_valid, in_conn, in_src, in_valid, level, node_level, topo_key, input_slot, output_slot, node_exists, edge_overflow` where `in_valid (P,N,K)` marks a live window slot, `node_level (P,N) int32` is the topological level (authority for `level`), and `edge_overflow (P,)` counts connections that did not fit the static `K`.
4. **Public names**: `common.tensorization.derive_schedule` (no `build_schedule` alias); **all static arguments are positional** (evoxc cannot pass keyword-only statics to `build`, so no `functools.partial` dance at call sites); `common.host_graph.{decode_graph, topological_sort, find_useful_nodes, node_depth, check_cycles}` and `common.__init__` re-exports the first three; `ACT.name_of`/`AGG.name_of` replace `get_func_name`; `common.tools.to_numpy(tensor)` is the host-side array accessor.
5. **No default `State()` in signatures**: `setup(self, state=None)` builds `State()` in the body (a default argument would run at class-definition time and break `import tensorneat`).
6. **Genome helper surface used by `algorithm/`**: `node_exists`, `conn_exists`, `max_node_key`, `max_conn_marker`, `conn_gene.uses_historical_marker`, `node_gene.from_attrs`, `conn_gene.from_attrs`, plus statics `num_inputs/num_outputs/max_nodes/max_conns/max_in_degree/node_gene.length/conn_gene.length`.
7. **`CartPole`** validates `fitness_target <= max_step` at construction and demotes NaN logits to `-inf` fitness (an explicit `evoxc.isnan` guard, not an implicit NaN comparison).

## API Surface (as implemented)
`tensorneat.Pipeline`, `tensorneat.algorithm.{NEAT,HyperNEAT,HyperNEATFeedForward}`, `tensorneat.genome.{DefaultGenome,RecurrentGenome,DefaultNode,BiasNode,OriginNode,DefaultConn,OriginConn,...}`, `tensorneat.problem.{XOR,XOR3d,CustomFuncFit,FuncFit,CartPole}`, `tensorneat.common.{ACT,AGG,State,TensorNEATError}`.
`Pipeline(algorithm, problem, seed=42, fitness_target=1, generation_limit=1000, backend="numpy", device=None, compile_options=None, is_save=False, save_dir=None, is_detailed=True, show_problem_details=False, pop_batch_size=None)` — host-only construction, no graph built and nothing printed until the first `step`.
`setup(state=None) -> State` registers `randkey`, `backend`/`device`/`compile_options`, the algorithm's payload/species and the problem's data, resets the run's history and, with `is_save`, writes `config.txt` and the `log.txt` header.
`step(state) -> (state, previous_pop, fitnesses)` is the single compile site and one compiled-graph run; fitness is **last**, a host `numpy` `(P,)` float32 array, `previous_pop` is a `GenomeTensors` of host arrays (the population that produced those fitnesses) and `step` prints nothing.
`auto_run(state) -> (state, best_genome)` loops on the host until `fitness_target` or `generation_limit`, owning all output, logging, checkpointing and early stop.
`show(state, best)` prints one genome's behaviour through the problem's host-side `show` (schedule derived once, network re-baked as `evoxc.constant`), and `show_config() -> dict` is the JSON-serialisable description of the run.
`best_genome` is spelled **batch-first** everywhere: a one-individual batch — `node_keys (1, max_nodes)`, `node_attrs (1, max_nodes, A)`, `conn_keys (1, max_conns, key_width)`, `conn_attrs (1, max_conns, W)` — i.e. exactly `state.pop` at `P = 1`; `None` when no generation produced a finite fitness.
`auto_run`/`_analyse` store the selected individual with its leading axis added back, and `show` accepts that batch as well as the bare rows a checkpoint stores (`host_payload._one_individual` promotes bare rows and rejects a payload whose arrays disagree about the population axis or use an impossible rank).
`tensorneat.__version__` is `importlib.metadata.version("tensorneat")` with a `PackageNotFoundError` fallback to `"0.0.0"`; it imports nothing that reaches a backend.
Host-visible attributes: `fitness_history` (list of `(generation, max_fitness)`), `best_fitness`, `best_genome`, `shape_signature`, `build_seconds`, `builds`, `_executables`, `save_dir`.
Fitness is maximized; NaN or infinite fitness is mapped to `-inf` before selection so broken topologies cannot win.
No `problem.jitable` flag exists: whether a graph compiles is the backend's business and `evoxc.BackendError` propagates.

## Compile Site
One `evoxc` graph per shape signature covers `split randkey → ask → transform → problem.evaluate(eval_key, act_fn) → tell`, so a generation is one `evoxc.run` plus host bookkeeping.
The graph boundary carries **tensors only** (the four payload arrays, `generation`, `randkey`, the `SpeciesState` leaves, then `fitness` and the eval-phase `edge_overflow` maximum); the tensor-free configuration `State` is baked through the closure (`Pipeline._static_config` drops every tensor and every `_RUNTIME_KEYS` entry, the `NEAT._static_state` pattern), so no concrete tensor is ever captured by a trace.
`act_fn` is a closure over the already-transformed schedule: it repeats the schedule and payload `S = rows / P` times with `host_payload._repeat_batch` in **individual-major** order, matching how a problem tiles a dataset or a rollout batch, and runs `algorithm.forward` through the arity adapter (`NEAT` takes the payload, `HyperNEAT` hides it in its substrate schedule).
`ask` costs zero ops, so the returned `previous_pop` *is* the population that produced `fitnesses`.

## Caching Policy
The signature is `(pop_size, max_nodes, max_conns, max_in_degree, max_depth, key_span)`; `_executable_for` keeps one `evoxc.Executable` per signature in `_executables` and rebuilds only when the signature changes.
A second pipeline with a different `max_nodes` is a second entry and a second build; a repeated `step` on the same pipeline reuses the cached executable (`builds` stays at 1).
The first build is timed into `build_seconds` and counted in `builds`; `auto_run` pays it before its first generation and then only runs.

## Overflow Policy
A non-zero `Transformed.edge_overflow` means the schedule dropped connections, so `step` raises `TensorNEATError` naming the count, the population and `max_in_degree` instead of evaluating a lossy phenotype.
`max_in_degree` is a **genome** static, so the pipeline cannot widen it silently — the caller rebuilds the genome, and a lossy schedule is never used.
The check is host-side on the graph's returned maximum: a traced value never drives Python control flow.

## Key Discipline
Exactly one `evoxc.random.split` happens at the pipeline boundary per generation: `eval_key` goes to the problem and the advanced key goes to `tell`, which advances it again before the generation returns.
`step` stores that returned key on the new state, so one `step` is exactly one generation of key advance and `(seed, config, shape)` reproduces a run.

## Checkpoint Layout (`is_save=True`, written by `setup`/`auto_run`)
`config.txt` — `json.dumps(show_config(), indent=4)`.
`log.txt` — header `Generation,Max,Min,Mean,Std,Cost Time`, then one row per generation.
`genomes/<generation>.npz` and `genomes/best_genome.npz` — `node_keys`, `node_attrs`, `conn_keys`, `conn_attrs` and `fitness`, one individual per file (no population axis), so `np.load` round-trips to a payload `show` accepts and re-scores to its recorded fitness.
`best_genome.npz` is omitted when no generation produced a finite fitness; a bare `step` loop writes no files (`auto_run`/`_analyse` is the writer).

## Measured Cost
numpy reference backend, one config per fresh process, build excluded, one warmup + 6 timed `step` calls, `DefaultGenome(num_inputs=2, num_outputs=1, max_nodes=50, max_conns=100, output_transform=ACT.sigmoid)` + `NEAT(pop_size=P, species_size=20, survival_threshold=0.01)` + `XOR()`, seed 42:

| pop | build (one-off) | ms/generation (median / min) | legacy JAX | ratio |
| --- | --- | --- | --- | --- |
| 100 | 2.2 s | 54.7 / 51.3 | 51 | 1.07× |
| 500 | 3.9 s | 352.5 / 343.7 | 109 | 3.2× |
| 1000 | 4.5 s | 642.4 / 603.5 | 157 | 4.1× |

The box drifts ±30 % between sessions, so the ratio against the frozen legacy baseline is the load-bearing number, never the absolute milliseconds.
Remaining hot spots at `pop ≥ 500`: `speciate` dominates `tell` (2 phases × 20 species slots × P distance measurements = 4 000 / 20 000 / 40 000 pairs per generation, ~7.5 µs/pair at pop 100 to ~13 µs/pair at pop 1000 after the digest API), `breed` is second (~30 / ~76 / ~148 ms in the phase-sum protocol) and `evaluate` (`transform` + `forward` on the tiled `4P` batch) is last.
The pair count is irreducible under NEAT's assignment semantics, and the reference interpreter pays per traced op and per element while the legacy baseline was the same work fused by XLA — so parity at large `P` is not reachable on `backend="numpy"`.
The compiled route is blocked inside `evoxc`: `scan` desugars to `while` + gather/scatter and a matrix-valued scan carry needs a rank-3 scatter the v1 StableHLO exporter rejects, `iree`'s llvm-cpu compiler SIGSEGVs non-deterministically on a `while` with ≥3 carried values plus a `select`, compile costs ~3–7 s per few-thousand-op graph, and there is no committed llvm-cpu measurement above ~8 k ops.
The shipped backend is therefore the reference interpreter, and a compiled backend must be re-evaluated when `evoxc` lifts those limits — no compiled win is claimed.
Current per-generation op counts: `forward` 112 traced ops (749 before the level loop was reformulated), `execute_mutation` 1360 (3114), `measure_distance` 7.5 µs/pair (25–48 µs before the digest API).

## Known Issues / Notes for Agents
`test/integration/` drives `Pipeline` end-to-end on the numpy backend (`setup`/`step`/`auto_run`, the batch-first best-genome payload, determinism per seed, one build per shape signature, checkpoint round-trip); `test/unit/` covers the kernels and `test/parity/` the legacy comparison, the latter needing the `bench` extra so its jax-dependent cases skip without it.
`benchmarks/RESULTS.md` still carries the old-side baseline only (`jax 0.12`, `evoxc n/a`); its owner must re-render it from a harness run that includes the new-side rows above.
`benchmarks/run_benchmark.py --impl new --backend numpy --cases xor_p100` drives this `step` contract unchanged (it reads fitness from the last element of the step result) and reports `INCOMPLETE` until an old-side row is measured on the same box.

## Constraints
- `import tensorneat` must not import `jax`, `torch`, `brax`, `gymnax`, `mujoco` (release-gate test); optional backends are pulled in only by `evoxc.build(backend=...)`.
- evoxc idioms are mandatory: `@evoxc.defn` graph entry points, plain helpers otherwise, `evoxc.numpy as enp` for tensor creation (float literals become f64 constants otherwise), `evoxc.random` for keys, `evoxc.constant` only for genuinely static data, non-negative axes in `concatenate`/`split`.
- No eager fallback: if a backend cannot compile a kernel the error propagates; we never quietly compute on the host.
- NaN padding semantics of the pre-rewrite library are preserved (`test/parity/` depends on them); `select` chains stand in for `lax.switch`.
- Errors: user/config validation raises `common.errors.TensorNEATError` (and its subclasses); tensor/backend errors are never caught and re-wrapped.
- Files ≤ ~1000 lines; shared math lives in `common/`, never duplicated per genome/algorithm.
- `pipeline.py` (1132 lines) is one job — the single compile site plus the design, caching, overflow and measured-cost documentation that belongs next to it — and is a deliberate exception to the line budget; the code is ~900 lines and the rest is the module docstring.

## Routing Table
| Path | Responsibility |
|---|---|
| `./common` | Backend primitives: `State`, errors, RNG/tensor helpers, **tensorization (execution schedule)**, `ACT`/`AGG` registries, host graph decoding, sympy codegen. |
| `./genome` | Genome classes, gene types, genetic operators, forward/grad kernels over the schedule. |
| `./algorithm` | `NEAT` (ask/tell/speciation), `HyperNEAT` and substrates — batched, fixed-length scans, no vmap-over-control-flow. |
| `./problem` | Task layer: `FuncFit`/`XOR`/`XOR3d`/`CustomFuncFit` and the evoxc-native `CartPole` rollout. |
| `./pipeline.py` | Host-side compile site, generation loop, logging/checkpointing, `show`. |
| `./__init__.py` | Re-exports `Pipeline` and the four subpackages. |
