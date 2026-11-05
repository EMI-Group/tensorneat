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
7. **Three RL environments ship, each validating `fitness_target` against its own return ceiling at construction** and each reaching `-inf` for a NaN-poisoned genome: **`CartPole`** (ceiling `max_step`, return = survival time) demotes NaN logits with an explicit `evoxc.isnan` census, while **`Pendulum`** (ceiling `0`, negated cumulative cost) and **`MountainCarContinuous`** (ceiling `100`, the one-time goal bonus) are poisoned *implicitly* through their action penalty — `tanh(NaN)` is NaN, so `0.001*u**2` and `0.1*a**2` respectively carry the NaN into the return. Every shipped problem ends with `problem.base.finalize_fitness`, the single NaN → `-inf` gate.

## API Surface (as implemented)
`tensorneat.Pipeline`, `tensorneat.algorithm.{BaseAlgorithm,NEAT,HyperNEAT,HyperNEATFeedForward}`, `tensorneat.genome.{DefaultGenome,RecurrentGenome,DefaultNode,BiasNode,OriginNode,DefaultConn,OriginConn,...}`, `tensorneat.problem.{XOR,XOR3d,CustomFuncFit,FuncFit,CartPole,MountainCarContinuous,Pendulum}`, `tensorneat.common.{ACT,AGG,State,TensorNEATError}`.
`tensorneat.algorithm` re-exports `BaseAlgorithm`, `NEAT`, `HyperNEAT` and `HyperNEATFeedForward` (the last from `tensorneat.algorithm.hyperneat`) with exactly those four names in `__all__`, pinned by `test/unit/algorithm/test_exports.py`.
`Pipeline(algorithm, problem, seed=42, fitness_target=1, generation_limit=1000, backend="numpy", device=None, compile_options=None, is_save=False, save_dir=None, is_detailed=True, show_problem_details=False, pop_batch_size=None)` — host-only construction, no graph built and nothing printed until the first `step`.
`setup(state=None) -> State` registers `randkey`, `backend`/`device`/`compile_options`, the algorithm's payload/species and the problem's data, resets the run's history and, with `is_save`, writes `config.txt` and the `log.txt` header.
`step(state) -> (state, previous_pop, fitnesses)` is the single compile site and one compiled-graph run; fitness is **last**, a host `numpy` `(P,)` float32 array, `previous_pop` is a `GenomeTensors` of host arrays (the population that produced those fitnesses) and `step` prints nothing.
`auto_run(state) -> (state, best_genome)` loops on the host until `fitness_target` or `generation_limit`, owning all output, logging, checkpointing and early stop.
`show(state, best)` prints one genome's behaviour through the problem's host-side `show` (schedule derived once, network re-baked as `evoxc.constant`), and `show_config() -> dict` is the JSON-serialisable description of the run.
`best_genome` is spelled **batch-first** everywhere: a one-individual batch — `node_keys (1, max_nodes)`, `node_attrs (1, max_nodes, A)`, `conn_keys (1, max_conns, key_width)`, `conn_attrs (1, max_conns, W)` — i.e. exactly `state.pop` at `P = 1`; `None` when no generation produced a finite fitness.
`auto_run`/`_analyse` store the selected individual with its leading axis added back, and `show` accepts that batch as well as the bare rows a checkpoint stores: `host_payload._one_individual` accepts **only** the one-individual batch (`node_keys (1, max_nodes)`, i.e. `state.pop`/`best_genome` at `P = 1`) or those bare per-individual rows, and rejects a `P > 1` payload **and** a missing/`None` payload with a `TensorNEATError` naming both accepted spellings (never a bare `TypeError`, never the population-axis message); a genuine disagreement between the four arrays' leading axes still raises the population-axis message.
`Pipeline._constant_act_fn` therefore bakes its payload **as given** — `self._analyse` hands it the whole population the diagnostics hook receives, `show` hands it the one individual `_one_individual` normalised.
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
`Pipeline.setup` creates the one root key (`evoxc.random.key(seed)`); `NEAT.setup` then splits it once inside the init graph so `init_key` draws the whole population's attributes and the other half is carried into generation 0.
Exactly one `evoxc.random.split` happens at the pipeline boundary per generation: `eval_key` goes to the problem and the advanced key goes to `tell`, which advances it again before the generation returns.
`step` stores that returned key on the new state, so one `step` is exactly one generation of key advance and `(seed, config, shape)` reproduces a run.

## Checkpoint Layout (`is_save=True`, written by `setup`/`auto_run`)
`config.txt` — `json.dumps(show_config(), indent=4)`.
`log.txt` — header `Generation,Max,Min,Mean,Std,Cost Time`, then one row per generation.
`genomes/<generation>.npz` and `genomes/best_genome.npz` — `node_keys`, `node_attrs`, `conn_keys`, `conn_attrs` and `fitness`, one individual per file: the batch-first payload flattened (no population axis), so `np.load` round-trips to a payload `show` accepts and re-scores to its recorded fitness.
`best_genome.npz` is omitted when no generation produced a finite fitness; a bare `step` loop writes no files (`auto_run`/`_analyse` is the writer).

## Measured Cost
numpy reference backend, XOR, seed 42, `DefaultGenome(num_inputs=2, num_outputs=1, max_nodes=50, max_conns=100, output_transform=ACT.sigmoid)`, `NEAT(pop_size=P, species_size=20, survival_threshold=0.01)`, build excluded, 3 warmup + 20 timed `Pipeline.step`, one process per cell, before/after measured in one session that alternated the two trees:

| cell | build (one-off) | median / mean ms per generation now | median / mean before the pass | peak RSS now / before |
| --- | --- | --- | --- | --- |
| xor p100 | 0.255 s | 38.58 / 40.05 | 43.47 / 44.18 | 114.4 / 118.2 MiB |
| xor p500 | 0.314 s | 159.90 / 161.37 | 268.39 / 256.41 | 240.5 / 260.8 MiB |
| xor p1000 | 0.399 s | 270.08 / 280.26 | 342.21 / 408.28 | 391.5 / 431.6 MiB |
| xor3d p500 | 0.320 s | 176.98 / 177.70 | 290.04 / 289.17 | 264.7 / 291.3 MiB |

The box drifts ±30 % between sessions, so the ratio against the frozen legacy baseline (51 / 109 / 157 ms per generation at pop 100 / 500 / 1000) is the load-bearing number, never the absolute milliseconds.
`benchmarks/PERF_NUMPY.md` holds this hand-maintained before/after record, its profile split and the reproduce commands.
`benchmarks/RESULTS.md` predates this pass, so its new-side rows are stale and only `run_benchmark.py` may re-render it — never hand-edit it.
A generation executes **8 150 IR ops** (10 767 before the pass), and ~63 % of the remaining wall time is the numpy interpreter's fixed per-op cost (~13 µs of operand resolution, dtype/shape validation and `Tensor` construction in `evoxc/backends/numpy/interpreter.py::_run_block`), so **executed op count is the lever for any further win**; host-side bookkeeping in `Pipeline.step` is 0.1 %.
`speciate` is still the largest phase — two fixed-length scans over `species_size` slots, one population-shaped distance row per slot, the rows produced inside one dedicated `evoxc.scan` — followed by the mutation/crossover chain and `transform`; the forward level loop is a `while` whose body is 5 real IR ops (`gather`, `select`, `scatter`, `not_equal`, `reduce_max`), run `depth + 1` times.
The digest is paid once per generation (`prepare_distance`), while the distance rows are measured `species_size * 2` times, so the population is tiled into the digest rather than re-prepared per centre.
The pair count `species_size * 2 * pop_size` is irreducible under NEAT's assignment semantics, and the reference interpreter pays per traced op and per element while the legacy baseline was the same work fused by XLA — so parity at large `P` is not reachable on `backend="numpy"` by kernel-level tuning alone.
`backend="iree"` runs every shipped case end to end since the graph formulations were made v1-exportable, but on llvm-cpu it is 5–24× slower per generation than the interpreter and costs 8–11 s of `iree-compile` per static shape; `benchmarks/BACKENDS.md` holds the measured table, the protocol and the per-gate fix record.
`backend="xla"` needs a user-supplied PJRT plugin `.so` (`EVOXC_PJRT_PLUGIN`); `backend="tvm"` is untested.
No compiled backend beats the numpy interpreter on this box, so the reference interpreter stays the default.
The generation graph's static entry block is **1 743 ops** at pop 500 and is executed once per generation; its `while` regions hold bodies of 22 and 39 ops (the two `derive_schedule` loops in `transform`), 6 ops (the forward level loop: `gather`, `select`, `scatter`, `not_equal`, `reduce_max` plus the block terminator, i.e. 5 real ops per level, run `depth + 1` times), and 119 / 53 / 163 ops (the distance-table scan, the speciate reserve scan and the speciate assign-member scan, `species_size` = 20 iterations each).
The three 20-iteration scan bodies therefore own 6 700 of the 8 150 executed ops, while `execute_mutation` is ~438 traced ops with the whole mutation chain ~340 executed ops per generation, and `measure_distance` is 94 traced ops / ~2.39 ms per call at pop 500 against 163 traced ops for the once-per-generation `prepare_distance`.

## Search Quality (current, measured on the numpy backend)
The quality contract for the search layer is `benchmarks/cases.py`'s XOR family: `DefaultGenome(num_inputs=2|3, num_outputs=1, max_nodes=50, max_conns=100, output_transform=ACT.sigmoid)` + `NEAT(pop_size=P, species_size=20, survival_threshold=0.01)`, 150 generations, no early stop, fitness = −mse.
XOR converges: pop 100 seed 42 ends at −1.6e-05 (target at gen 51) and pop 500 seed 42 at −4.0e-24 (gen 32).
XOR3d pop 500 converges on 3 of seeds 42-47 within 150 generations, 3-repeat median **−4.1e-07** over seeds 42/43/44; pop 1000 converges on 2 of 3 (median **−6.1e-09**) and seed 43 does not.
XOR3d pop 100 stays at the constant-0.5 attractor (mse exactly 0.25) for all of seeds 42-47 within 150 generations and escapes on 2 of 6 within 400 (gens 254 / 350), so it is a small-population search-throughput limit rather than an evaluation defect: hand-built payloads score bit-identically through `Pipeline.step` and a host numpy oracle, and all three input channels measurably move the output.
The frozen oracle shows the same class of failure at pop 100 (seeds 42/43/44 = −0.0397 / −0.1350 / −0.0922: it leaves the attractor but meets its own −0.05 target on 1 of 3 seeds).
`custom_pagie_p500` seed 42 reaches −0.0103 at 150 generations against the −0.01 target and the legacy −0.0064, a slow climb that runs out of budget rather than a plateau.
Weight-mutation frequency is not the pop-100 lever: the frozen tree's law (0.2/0.015, its ±5 clip inert at max |w| 3.8) and 0.5/0.05 both give 0 of 6 escapes at pop 100 while flattening cells that converge at the shipped 0.8/0.1 defaults; `genome/operations/mutation/CONTEXT.md` holds the full sweep and the rejected experiments.
Two structural contracts keep the search from silently losing genes: every allocated structural key lies inside `[0, key_span)` (`algorithm/neat/_keys.py`; `key_span <= max_nodes` raises at `NEAT` construction) and every structural operation is gated independently (`genome/operations/mutation`).
`algorithm/neat/_kernels.py::pair_parents` pins elite slot k of a species block to that block's rank-k member, so a species never clones its champion into both elite slots.

## Known Issues / Notes for Agents
Two cases in `test/unit/genome/test_mutation.py` (`test_at_most_one_structural_change_per_individual`, `test_shapes_and_dtypes_are_preserved`) still assert the superseded single-draw structural contract and fail against the four-independent-gate law; they need a test-scope rewrite to "≤1 change per operation, sequential composition, refusals bit-exact no-ops" (`genome/operations/mutation/CONTEXT.md` states the replacement contract).
`test/integration/` drives `Pipeline` end-to-end on the numpy backend (`setup`/`step`/`auto_run`, the batch-first best-genome payload, determinism per seed, one build per shape signature, checkpoint round-trip); `test/unit/` covers the kernels and `test/parity/` the legacy comparison, the latter needing the `bench` extra so its jax-dependent cases skip without it.
`benchmarks/RESULTS.md` records the CPU speed contract as `FAIL` but predates this pass, so its new-side rows are stale and only `run_benchmark.py` may re-render the verdict; against the numbers above a generation is 0.76× / 1.47× / 1.72× legacy at pop 100 / 500 / 1000.
`docs/source/` carries one autodoc stub per public module here, so a new module needs a matching `docs/source/tensorneat.<module>.rst` plus its toctree entry (the rule is in `docs/CONTEXT.md`).

## Constraints
- `import tensorneat` must not import `jax`, `torch`, `brax`, `gymnax`, `mujoco` (release-gate test); optional backends are pulled in only by `evoxc.build(backend=...)`.
- evoxc idioms are mandatory: `@evoxc.defn` graph entry points, plain helpers otherwise, `evoxc.numpy as enp` for tensor creation (float literals become f64 constants otherwise), `evoxc.random` for keys, `evoxc.constant` only for genuinely static data, non-negative axes in `concatenate`/`split`.
- No eager fallback: if a backend cannot compile a kernel the error propagates; we never quietly compute on the host.
- NaN padding semantics of the pre-rewrite library are preserved (`test/parity/` depends on them); `select` chains stand in for `lax.switch`.
- Errors: user/config validation raises `common.errors.TensorNEATError` (and its subclasses); tensor/backend errors are never caught and re-wrapped.
- Files ≤ ~1000 lines; shared math lives in `common/`, never duplicated per genome/algorithm.
- The numpy interpreter keeps every SSA result of a block alive until that block returns, so a kernel's live set is charged once per enclosing block: never hoist a kernel out of a `while`/`scan` without measuring peak RSS (hoisting the 20 speciation rows was 28 % faster and 1.7× the memory; measuring them inside one `evoxc.scan` is 40 % faster and below the starting memory).
- Never seed a `while`-carried value with a `scatter`: it SIGSEGVs `iree-compile` in the assembled graph while exporting cleanly; seed with a one-hot `equal`/`logical_and`, and treat only `test/backends` as proof of portability.
- `pipeline.py` (1003 lines) is one job — the single compile site plus the design, caching, overflow and measured-cost documentation that belongs next to it; the host-side payload helpers live in `./host_payload.py` so the file stays under the line budget.

## Routing Table
| Path | Responsibility |
|---|---|
| `./common` | Backend primitives: `State`, errors, RNG/tensor helpers, **tensorization (execution schedule)**, `ACT`/`AGG` registries, host graph decoding, sympy codegen. |
| `./genome` | Genome classes, gene types, genetic operators, forward/grad kernels over the schedule. |
| `./algorithm` | `NEAT` (ask/tell/speciation), `HyperNEAT` and substrates — batched, fixed-length scans, no vmap-over-control-flow. |
| `./problem` | Task layer: `FuncFit`/`XOR`/`XOR3d`/`CustomFuncFit` plus the three evoxc-native RL rollouts — `CartPole` (discrete), `Pendulum` (continuous torque), `MountainCarContinuous` (continuous force with a one-time goal bonus) — each a `while_loop` over timesteps, with the shared host helpers in `problem/rl/_common.py`. |
| `./pipeline.py` | Host-side compile site, generation loop, logging/checkpointing, `show`. |
| `./host_payload.py` | Host-only payload helpers: config rendering (`_jsonable`/`_config_value`/`_declared_shape`), batch tiling (`_repeat_batch`/`_run_network`), schedule overflow lookup and `_one_individual`, which normalises exactly one individual (a `(1, max_nodes)` batch or the bare checkpoint rows) and rejects `P > 1` / `None` with `TensorNEATError`. |
| `./__init__.py` | Re-exports `Pipeline`, the four subpackages and `__version__`. |
