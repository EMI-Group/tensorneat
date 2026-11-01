# Algorithm (`tensorneat.algorithm`)
## Intent
Evolutionary algorithm layer that sits between `tensorneat.pipeline.Pipeline` and `tensorneat.genome`: it owns the population, the ask → transform → forward → tell loop and all evolutionary bookkeeping.
Everything here is written against the immutable `State` pytree and pure functions so that one whole generation compiles as a single `jax.jit` (see JAX Dependency Map).
Two concrete algorithms: `NEAT` (plain NEAT over an evolvable `BaseGenome`) and `HyperNEAT` (evolves a CPPN with an inner `NEAT` and decodes it into a fixed-topology substrate network).
## API Surface
`__init__.py` re-exports exactly three names: `BaseAlgorithm`, `NEAT`, `HyperNEAT` (import order matters — `base` must be imported first because `neat.py` does `from .. import BaseAlgorithm`).
`base.py::BaseAlgorithm(StatefulBaseClass)` is the contract every algorithm must satisfy; its members all raise `NotImplementedError`:
`setup(state=State())` (inherited from `StatefulBaseClass`, must return the threaded `State`),
`ask(state) -> population` (hand out the genomes to be evaluated),
`tell(state, fitness) -> State` (consume a 1-D fitness array of length `pop_size`),
`transform(state, individual) -> transformed` (decode one genome into an executable network),
`forward(state, transformed, inputs) -> outputs` (run the decoded network),
`show_details(state, fitness)` (host-side printing, called outside jit),
properties `num_inputs`, `num_outputs` (checked against `problem.input_shape[-1]`/`output_shape[-1]` by `Pipeline`).
`NEAT.__init__` takes only algorithm hyperparameters plus a pre-configured genome — `NEAT(genome, pop_size, species_size=10, max_stagnation=15, species_elitism=2, spawn_number_change_rate=0.5, genome_elitism=2, survival_threshold=0.1, min_species_size=1, compatibility_threshold=2.0, species_fitness_func=jnp.max, species_number_calculate_by="rank", pop_batch_size=None)`.
There is no `max_nodes`, `max_conns`, mutation-rate or crossover-rate argument at algorithm level: those belong to the genome, e.g. `DefaultGenome(num_inputs=…, num_outputs=…, max_nodes=…, max_conns=…, mutation=DefaultMutation(conn_add=…, node_add=…, …))`.
`pop_batch_size` (default `None`) must evenly divide `pop_size` and switches population-level work from one full `vmap` to `jax.lax.scan` batches (GPU memory O(pop) → O(pop_batch_size), issue #41).
`NEAT` exposes `num_inputs`/`num_outputs` purely by forwarding to `self.genome`, plus attributes `genome`, `pop_size`, `pop_batch_size`, `species_controller`.
`transform` in both algorithms is pure delegation: `NEAT.transform` unpacks `(nodes, conns)` and calls `genome.transform`; `NEAT.forward` calls `genome.forward` — all topology/traversal logic lives in `tensorneat/genome`.
The number of values returned by `transform` is genome-dependent (`DefaultGenome` → `(seqs, nodes, conns, u_conns)`, `RecurrentGenome` → `(nodes, conns, u_conns)`); `forward` must accept whatever `transform` produced, and `ask`'s population items are the `(nodes, conns)` tuples returned by `ask`.
## JAX Dependency Map
Compilation entry point is outside this package: `pipeline.py:193` `jax.jit(self.step).lower(state).compile()`, where `step` calls `algorithm.transform` / `algorithm.forward` / `algorithm.ask` / `algorithm.tell` — so ALL of `setup`-after-compile, `ask`, `tell`, `transform`, `forward` and the entire speciation bookkeeping are traced into one jitted graph.
`pipeline.py:197` additionally compiles `jax.jit(jax.vmap(self.algorithm.transform, in_axes=(None, 0)))` for `show_problem_details`.
No function in this package carries `@jax.jit` itself, and there is no `static_argnames`/`donate_argnums` anywhere in this subtree.
`neat.py`: `setup` (`jax.random.split`, `jax.vmap`), `ask` (pure attribute access, zero JAX ops), `tell` (Python call graph + traced `state.generation + 1`), `_create_next_generation` (`jnp.max(where=…, initial=…)`, `jnp.isnan`, `jnp.arange`, `jnp.full`, `jnp.where`, `jax.random.split`, `jax.vmap`), `_vmap_or_scan` (`jax.tree_util.tree_map` + `jax.lax.scan` with `carry=None`), `show_details` (the only host-side function: `jax.device_get` + numpy + `print`).
`species.py`: `setup` (`jnp.full`/`jnp.zeros`/`.at[...].set`), `update_species` (`jnp.argsort(...)[::-1]` reorder, `jax.random.split`, static Python branch on `species_number_calculate_by`), `_stagnation` (`jax.vmap` + `jax.lax.cond` + jitted `rank_elements`), `_cal_spawn_numbers_by_rank`/`_by_fitness` (`jnp.floor`/`jnp.where`/`.at[0].add(error)`), `_create_crossover_pair` (nested `jax.vmap`, `jax.random.choice(..., p=…)`, `jnp.cumsum`/`jnp.argmax`), `speciate` (two `jax.lax.while_loop`s with data-dependent trip counts bounded by `species_size`, per-slot `jax.lax.cond`, `jax.vmap`ed genome distance, `fetch_first`).
Jitted helpers imported from `tensorneat/common/tools.py`: `fetch_first` (`@jit`), `rank_elements` (`@partial(jit, static_argnames=["reverse"])`); `argmin_with_mask` is plain jnp and not jitted.
Pure-Python bookkeeping that still runs inside the trace (must stay trace-safe): spawn-number allocation, elite masks, `new_conn_markers` allocation, all Python `if`/`elif` on static config strings (`species_number_calculate_by`, `"historical_marker" in conn_gene.fixed_attrs`, `pop_batch_size is None`).
`hyperneat` follows the same model: no local `@jax.jit`; `transform`/`forward` are traced by `Pipeline`; substrate construction is eager NumPy and must happen outside any trace.
## Gradients
The algorithm layer has NO gradient channel: `NEAT.ask` returns only `(pop_nodes, pop_conns)` and `NEAT.tell` accepts only `fitness` — no `grad`, `value_and_grad`, `jax.grad` or `optax` appears anywhere under `src/tensorneat/algorithm`.
Gradient support lives one level down at `genome/base.py::BaseGenome.grad(state, nodes, conns, inputs, loss_fn)`, which uses `jax.value_and_grad(_inner, argnums=(0, 1))`, zeroes gradients at NaN padding (`jnp.where(jnp.isnan(nodes), 0.0, grad_nodes)`) and returns `(loss, (grad_nodes, grad_conns))`.
Callers apply the SGD step themselves (see `examples/func_fit/xor_grad.py`, which vmaps `grad` over the population); `optax` is a declared dependency in `pyproject.toml` but is imported nowhere in `src/`.
## Constraints
`State` (`tensorneat/common/state.py`) is an immutable `register_pytree_node_class` pytree: `__setattr__` raises `AttributeError`, keys are string literals in `aux_data`, `register` rejects duplicate keys and `update` rejects unknown keys — a new algorithm field must be introduced with `state.register(...)` before any `state.update(...)`.
Every algorithm method that changes anything returns a new `State`; `transform`/`forward` are pure `(state, …)` → arrays with no state writes.
Never mutate arrays in place: only `.at[idx].set/add(...)`.
Never branch in Python on a traced value: use `jnp.where` / `jax.lax.cond`; never loop `for i in range(pop_size)` — use `jax.vmap` (or `jax.lax.scan` for memory batching).
NaN is the universal "empty/invalid" sentinel for float arrays (padded node/conn rows, dead species slots, unassigned `idx2species`); int32 paths use `I_INF = np.iinfo(jnp.int32).max` instead; `generation` is `jnp.float32` rather than int so it flows through float species bookkeeping.
Population shapes are fixed and NaN-padded: `pop_nodes (pop_size, max_nodes, node_gene.length)`, `pop_conns (pop_size, max_conns, conn_gene.length)`, column 0 = node key / connection historical marker.
A new algorithm must keep its state fields JAX-pytree-compatible and must not read host values (`jax.device_get`, `.item()`, `int()`) inside `ask`/`tell`/`transform`/`forward`; only `show_details` may do that.
Algorithm ↔ genome coupling contract: `genome.initialize / execute_crossover / execute_mutation / execute_distance / transform / forward` must all be vmappable with `state` broadcast as the first arg, and the algorithm reads `genome.conn_gene.fixed_attrs`, `genome.max_nodes`, `genome.max_conns` directly.
Any new population-level matrix allocation keyed off "one mutation may add up to N genes" needs the same hard-coded width convention as `new_conn_markers` (3 conns per mutation in `neat.py`).
## Test Coverage
`test/test_grad.py` covers `BaseGenome.grad` for `DefaultGenome`/`RecurrentGenome` forward; algorithm-level behaviour is exercised only through the examples (`examples/func_fit/xor*.py`, `examples/gymnax/*.py`).
## Routing Table
| Path | Contents |
| --- | --- |
| `./base.py` | `BaseAlgorithm` abstract contract (7 members) shared by every algorithm. |
| `./neat` | `NEAT` (population, crossover/mutation, elitism, `_vmap_or_scan`) and `SpeciesController` (speciation, stagnation, spawn allocation, parent selection). |
| `./hyperneat` | `HyperNEAT` / `HyperNEATFeedForward` CPPN→substrate decoding, `HyperNEATNode`/`HyperNEATConn`, and the `./hyperneat/substrate` package (`FullSubstrate`, `MLPSubstrate`, coordinates → nodes/conns analysis). |
