# Intent
Pure-JAX NEAT (NeuroEvolution of Augmenting Topologies) implementation: population-level genetic operators plus fitness-based speciation, written as pure functions over the immutable `State` pytree so the whole ask → transform → evaluate → tell cycle compiles as ONE `jax.jit`.
`NEAT` (`neat.py`) is the algorithm entry point used by `tensorneat.pipeline.Pipeline`; `SpeciesController` (`species.py`) owns all speciation, stagnation and reproduction bookkeeping.
Everything is functional: no in-place mutation, no Python loops over the population (vmap/scan instead), no host-side branching on traced values (`jnp.where` / `jax.lax.cond` instead).

# API Surface

## `NEAT(BaseAlgorithm)` — neat.py
`BaseAlgorithm` (`../base.py`) is a `StatefulBaseClass` declaring only `NotImplementedError` stubs for `ask/tell/transform/forward/show_details/num_inputs/num_outputs`.

```python
NEAT(
    genome: BaseGenome,
    pop_size: int,
    species_size: int = 10,
    max_stagnation: int = 15,
    species_elitism: int = 2,
    spawn_number_change_rate: float = 0.5,
    genome_elitism: int = 2,
    survival_threshold: float = 0.1,
    min_species_size: int = 1,
    compatibility_threshold: float = 2.0,
    species_fitness_func: Callable = jnp.max,
    species_number_calculate_by: str = "rank",   # "rank" | "fitness"
    pop_batch_size: int = None,
)
```
- `__init__` stores `self.genome`, `self.pop_size`, `self.pop_batch_size`, builds `self.species_controller = SpeciesController(...)` (all 11 species kwargs passed positionally).
- Asserts `species_number_calculate_by in ["rank", "fitness"]` and `pop_size % pop_batch_size == 0` when `pop_batch_size` is set.
- `pop_batch_size` docstring (issue #41): splits population-level vmap work into `lax.scan` batches to cut GPU peak memory from O(pop) to O(pop_batch_size).
- `setup(self, state=State())` → registers `randkey` untouched from genome setup, splits a key, `vmap(self.genome.initialize, in_axes=(None, 0))` over `pop_size` keys, registers `pop_nodes`, `pop_conns`, `generation=jnp.float32(0)`, then `species_controller.setup(state, pop_nodes[0], pop_conns[0])` (species 0 seeded with individual 0 as center).
- `ask(self, state)` → `return state.pop_nodes, state.pop_conns` (a 2-tuple, no copy, no randomness).
- `tell(self, state, fitness)` → increments `generation`, calls `species_controller.update_species` (returns `winner`, `loser`, `elite_mask`), then `_create_next_generation`, then `species_controller.speciate(state, self.genome.execute_distance)`; returns new state.
- `transform(self, state, individual)` → unpacks `(nodes, conns)` and delegates to `self.genome.transform`.
- `forward(self, state, transformed, inputs)` → delegates to `self.genome.forward`.
- `show_details(self, state, fitness)` → host-side only (`jax.device_get` + numpy); prints max/min/mean node & conn counts (counted as non-NaN first columns) and the list of non-zero species sizes.
- Properties: `num_inputs`, `num_outputs` (both forwarded from `self.genome`); public attributes `genome`, `pop_size`, `pop_batch_size`, `species_controller`.
- `_create_next_generation(self, state, winner, loser, elite_mask)` — allocates `new_node_keys = arange(pop_size) + (max node key + 1)` using `jnp.max(..., where=~isnan, initial=0)`; allocates `new_conn_markers` of shape `(pop_size, 3)` likewise only if `"historical_marker" in self.genome.conn_gene.fixed_attrs`, else `jnp.full((pop_size, 3), 0)`; splits randkeys into crossover keys and mutation keys; `vmap(self.genome.execute_crossover, in_axes=(None,0,0,0,0,0))` over gathered winner/loser parents; `vmap(self.genome.execute_mutation, in_axes=(None,0,0,0,0,0))` through `_vmap_or_scan`; then elitism: `jnp.where(elite_mask[:, None, None], n_nodes, m_n_nodes)` keeps elite children unmutated.
- `_vmap_or_scan(self, vmap_fn, args)` — if `pop_batch_size` is None/≥ pop_size runs `vmap_fn(args)`; otherwise reshapes args to `(n_batches, bs, ...)` with `jax.tree_util.tree_map`, runs `jax.lax.scan` with `carry=None`, and reshapes results back.

## `SpeciesController(StatefulBaseClass)` — species.py
- `__init__(pop_size, species_size, max_stagnation, species_elitism, spawn_number_change_rate, genome_elitism, survival_threshold, min_species_size, compatibility_threshold, species_fitness_func, species_number_calculate_by)`; stores `self.species_arange = np.arange(species_size)` (host numpy constant used as the vmap/loop index source).
- `setup(self, state, first_nodes, first_conns)` → registers a nested `State` under key `species` with keys `species_keys`, `best_fitness`, `last_improved`, `member_count` (each `(species_size,)` float, NaN = empty slot), `idx2species` (`(pop_size,)` float), `center_nodes`/`center_conns` (`(species_size, *gene_shape)`, NaN-padded), `next_species_key=jnp.float32(1)`; slot 0 is seeded as a live species holding all `pop_size` members with `best_fitness=-inf`, `last_improved=0`.
- `update_species(self, state, fitness)` → per-species fitness (`_update_species_fitness`), stagnation (`_stagnation`), descending sort of all species arrays by `jnp.argsort(species_fitness)[::-1]`, spawn-number allocation (`_cal_spawn_numbers_by_rank` or `_cal_spawn_numbers_by_fitness`, chosen by a Python `if` on the static string), then `_create_crossover_pair`; returns `(state_with_new_randkey_and_species, winner, loser, elite_mask)`.
- `_update_species_fitness(self, species_state, fitness)` → `vmap` over `species_arange`; per slot masks `fitness` with `idx2species == species_keys[idx]` (`-inf` elsewhere) and applies `self.species_fitness_func` (default `jnp.max`).
- `_stagnation(self, species_state, species_fitness, generation)` → `vmap` over species with `jax.lax.cond` to update `best_fitness`/`last_improved`; flags stagnation when fitness did not improve for `> max_stagnation` generations; `rank_elements(species_fitness) < species_elitism` species are exempt; stagnated slots get NaN keys/counts/centers and `species_fitness = -inf` (deliberately -inf, not NaN, so the later descending argsort keeps them last).
- `_cal_spawn_numbers_by_rank(species_state)` → linear ranking over valid (non-NaN) species: `rank_score = valid_species_num - arange`, `denominator = (n+1)*n/2`, `target = floor(rate * pop_size)`, then damped `previous_size + (target - previous_size) * spawn_number_change_rate`, floored at `min_species_size`, cast to `jnp.int32` (NaN → 0), and the residual `pop_size - sum` is added to species 0.
- `_cal_spawn_numbers_by_fitness(species_state)` → same damping/clamping/fix-up path but rates from `best_fitness - min(best_fitness) + 1` normalized over non-NaN entries.
- `_create_crossover_pair(self, species_state, randkey, spawn_number, fitness)` → per species `vmap`: member mask from `idx2species`, `survive_size = floor(survival_threshold * members_num)`, `select_pro = (p_idx < survive_size) / survive_size`, `jax.random.choice(..., shape=(2, pop_size), replace=True, p=select_pro)` for father/mother; the first `genome_elitism` slots are forced to the best member and marked in `elite_mask`; second `vmap` maps each population slot to its species via `jnp.argmax(idx < jnp.cumsum(spawn_number))`; winner/loser chosen by `fitness[father] >= fitness[mother]`.
- `speciate(self, state, genome_distance_func)` → builds `o2p_distance_func = vmap(genome_distance_func, in_axes=(None, None, None, 0, 0))` (one-to-population compatibility distance) and then runs two `jax.lax.while_loop`s: (1) re-elect center genomes — phase 1 assigns each existing species a new center as the nearest still-unassigned genome (`argmin_with_mask`); (2) assign genomes to species — `jax.lax.while_loop` with `jax.lax.cond` per slot dispatching to `create_new_species` (via `fetch_first`, consumes `next_species_key`, then `speciate_by_threshold`) or `update_exist_specie`; leftovers when the species cap is hit are dumped into the last species key; new species get `best_fitness=-inf`, `last_improved=state.generation`; finally `member_count` is recomputed by `vmap`-ed `jnp.sum(idx2species == key)`.

# JAX Dependency Map (most important)
- No function in this subtree carries `@jax.jit`, `jax.jit(...)` or `partial(jax.jit, ...)`, and there are no `static_argnames` / `donate_argnums` here.
- Compilation comes from OUTSIDE: `Pipeline.auto_run` does `jax.jit(self.step).lower(state).compile()` where `step` calls `algorithm.ask/transform/tell`; `Pipeline.analysis` separately jits `vmap(algorithm.transform)` when `show_problem_details` is on. So every traced function below must be jit-compatible.
- Jitted helpers consumed here live in `tensorneat/common/tools.py`: `fetch_first` (`@jit`), `rank_elements` (`@partial(jit, static_argnames=["reverse"])`), while `argmin_with_mask` is plain Python composed of `jnp.where`/`jnp.argmin` (no decorator).
- `neat.py::setup` — `jax.random.split`, `jax.vmap`, `jnp.float32`; Python-level `state.register`. Runs once, outside the training loop.
- `neat.py::ask` — no JAX ops at all (pure attribute access), always traced-inline.
- `neat.py::tell` — Python call graph only; arithmetic on `state.generation` (+1) is traced jnp math.
- `neat.py::transform` / `::forward` — thin delegations to the genome (all real XLA work is in `tensorneat/genome/`).
- `neat.py::_create_next_generation` — `jnp.max` with `where=`/`initial=`, `jnp.isnan`, `jnp.arange`, `jnp.full`, `jnp.where`, `jax.random.split`, Python `if "historical_marker" in self.genome.conn_gene.fixed_attrs` (static, resolved at trace time), gather indexing `state.pop_nodes[winner]`, and `jax.vmap` over `execute_crossover`/`execute_mutation`.
- `neat.py::_vmap_or_scan` — `jax.tree_util.tree_map`, `jax.lax.scan` with `carry=None`; the Python `if bs is None or bs >= pop_size` branch is static (shape changes require recompilation).
- `neat.py::show_details` — `jax.device_get` + `numpy`, `print`: host-side, must stay outside jit (called from `Pipeline.analysis` after `compiled_step`).
- `species.py::setup` — `jnp.full`, `jnp.zeros`, functional `.at[...].set`; no randomness.
- `species.py::update_species` — `jnp.argsort(...)[::-1]` (fancy-index reorder of every species array), `jax.random.split`, Python `if/elif` on `species_number_calculate_by` (static).
- `species.py::_update_species_fitness` — `jnp.where`, `jnp.isnan`-free, `jax.vmap` over a host `numpy` index array.
- `species.py::_stagnation` — `jax.vmap` + `jax.lax.cond` (traced `if`), `jnp.where`, `rank_elements` (jitted helper), `jnp.full_like`.
- `species.py::_cal_spawn_numbers_by_rank` / `_by_fitness` — `jnp.isnan`, `jnp.sum`, `jnp.floor`, `jnp.where`, `.astype(jnp.int32)`, `.at[0].add(error)`, `jnp.min`/`jnp.sum(..., where=)`.
- `species.py::_create_crossover_pair` — `jax.vmap` (nested: per species and per population slot), `jax.random.split`, `jax.random.choice` with `p=`, `jnp.argsort`, `jnp.floor`, `jnp.cumsum`, `jnp.argmax`, `jnp.where`.
- `species.py::speciate` — two `jax.lax.while_loop`s with data-dependent trip counts (bounded by `species_size`), `jax.lax.cond`, `jax.vmap` of the genome distance function, `jnp.isnan`/`jnp.any`/`jnp.where`, `argmin_with_mask`, `fetch_first`.
- Bottom line: the entire generation bookkeeping (speciation, species sorting, stagnation, survival/spawn allocation, parent selection, mutation/crossover dispatch) is compiled; only `show_details` and the `species_arange` construction are host-side Python.

# Gradients
- Nothing in this subtree references `grad`, `value_and_grad` or `optax`: `NEAT.ask` returns only `(pop_nodes, pop_conns)` and `NEAT.tell` accepts only `fitness` — fitness-based evolution with no gradient channel.
- Gradient capability lives one level down in `tensorneat/genome/base.py::BaseGenome.grad(state, nodes, conns, inputs, loss_fn)` → `jax.value_and_grad(_inner, argnums=(0, 1))` with NaN entries zeroed (`jnp.where(jnp.isnan(nodes), 0.0, grad_nodes)`), returning `(loss, (grad_nodes, grad_conns))`; callers apply it themselves (see `examples/func_fit/xor_grad.py`, which vmaps it over the population and does plain SGD).
- `optax >= 0.2.2` is a declared dependency in `pyproject.toml` but is not imported anywhere under `src/tensorneat`.

# Constraints & Conventions
- `State` (`tensorneat/common/state.py`) is an immutable, `register_pytree_node_class`-registered pytree: `__setattr__` raises `AttributeError("State is immutable")`, children = values, aux_data = literal string keys.
- `state.register(**kwargs)` REJECTS existing keys (`ValueError`), `state.update(**kwargs)` REJECTS unknown keys — every key must be introduced by `register` before `update` can touch it; `remove(*keys)` exists too.
- Keys live as string literals: algorithm state `randkey`, `pop_nodes`, `pop_conns`, `generation`; nested `state.species.<name>` for the 8 species fields. Reads go through `__getattr__` (`state.pop_nodes`).
- `SpeciesController` extends `StatefulBaseClass` (`tensorneat/common/stateful_class.py`), whose `setup(state=State())`/`show_config` are the only shared behavior; controller attributes are plain Python (config), not state.
- NaN is the universal "empty/invalid" sentinel: padded nodes/conns rows, dead species slots, and unassigned `idx2species`; for int32 arrays NaN is impossible so `I_INF = np.iinfo(jnp.int32).max` (`tensorneat/common/tools.py`) is used, e.g. as `fetch_first` default.
- Population shapes: `pop_nodes (pop_size, max_nodes, node_gene.length)`, `pop_conns (pop_size, max_conns, conn_gene.length)`; column 0 holds node keys / connection historical markers; the arrays are fixed-size and NaN-padded.
- `generation` is `jnp.float32` (not int) so it can flow through float-typed species bookkeeping (`last_improved`, `next_species_key` likewise float).
- No in-place mutation anywhere: only `x.at[idx].set/add(...)` functional updates and rebuilt `State`s.
- Control flow on traced values must be `jnp.where` or `jax.lax.cond`; loops over species are `vmap(species_arange)`/`lax.while_loop` with dynamic conditions — never `for i in range(pop_size)`.
- `spawn_number` is always reconciled to sum exactly to `pop_size` by adding the residual to species 0, and cast to `jnp.int32` (NaN → 0).
- Species are kept sorted best-first after `update_species`, which is what lets `_cal_spawn_numbers_by_rank` use `species_arange` itself as the rank.
- Elite individuals bypass mutation: `elite_mask` (first `genome_elitism` slots of each species) selects the unmutated crossover children via `jnp.where(elite_mask[:, None, None], ...)`.
- Coupling to the genome package: `self.genome.initialize / execute_crossover / execute_mutation / execute_distance / transform / forward` (all vmappable, `in_axes=(None, 0, ...)` with `state` as a broadcast first arg), plus `genome.conn_gene.fixed_attrs` (presence of `"historical_marker"` decides whether new conn markers are allocated), `genome.node_gene.length`, `genome.max_nodes`/`max_conns`.
- `new_conn_markers` has a hard-coded width of 3: one mutation call may add at most 3 connections.
- Known edge case: in `_create_crossover_pair`, `select_pro = (p_idx < survive_size) / survive_size` divides by zero for a species whose `floor(survival_threshold * members_num) == 0` (e.g. a 1-member species), yielding NaN selection probabilities for `jax.random.choice`.
- No `donate_argnums`/`jit` here means state buffers are not donated — every generation allocates new `pop_nodes`/`pop_conns`.
