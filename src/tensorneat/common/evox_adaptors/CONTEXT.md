# tensorneat/common/evox_adaptors — EvoX <-> TensorNEAT glue (optional dependency)

## Intent
Lets a TensorNEAT algorithm run as an EvoX `Algorithm` and report per-generation progress from inside a jitted workflow (host `io_callback`).
This is the only package that imports `evox`, and it is never imported by `tensorneat.common.__init__`, so `evox` stays an optional extra even though it is absent from `pyproject.toml` dependencies.
Entry point for users: `from tensorneat.common.evox_adaptors import EvoXAlgorithmAdaptor, TensorNEATMonitor` (see `README.md:343`, `examples/with_evox/walker2d_evox.py`).

## API Surface
### `algorithm_adaptor.py` — `@jit_class class EvoXAlgorithmAdaptor(EvoXAlgorithm)`
`__init__(self, algorithm: tensorneat.algorithm.BaseAlgorithm)` — stores the wrapped algorithm and `self.fixed_state = None`.
`setup(self, key)` — host-side: builds `TensorNEATState(randkey=key)`, runs `algorithm.setup(state)`, caches the result in `self.fixed_state` (a python attribute, *not* part of the EvoX state), and returns `EvoXState(alg_state=neat_algorithm_state)`.
`ask(self, state)` → `(algorithm.ask(state.alg_state), state)` — population plus the unchanged EvoX state.
`tell(self, state, fitness)` — first maps NaN fitness to `-jnp.inf` (`jnp.where(jnp.isnan(fitness), -jnp.inf, fitness)`), then returns `state.replace(alg_state=algorithm.tell(state.alg_state, fitness))`.
`transform(self, individual)` / `forward(self, transformed, inputs)` — delegate using the cached `self.fixed_state`, so both are pure functions of the individual (jit/vmap-safe) and share the same constants captured at `setup`.
Translation convention: EvoX `State` field `alg_state` holds the TensorNEAT `State`; fitness maximisation with `-inf` for invalid individuals.

### `tensorneat_monitor.py` — `class TensorNEATMonitor(Monitor)`
`__init__(self, tensorneat_algorithm: BaseAlgorithm, save_dir: str = None, is_save: bool = False)` — records `generation_timestamp`, `alg_state=None`, `fitness=None`, `best_fitness=-inf`, `best_genome=None`; when `is_save` it creates `save_dir` (and `save_dir/genomes`) and sets `genome_dir`.
`clear_history()` — resets the cached state/fitness/best genome.
`hooks()` → `["pre_tell"]`.
`pre_tell(self, monitor_state, workflow_state, transformed_fitness)` — schedules `jax.experimental.io_callback(self.store_info, None, workflow_state, transformed_fitness)` (no results) and returns `monitor_state`; the callback is the only way to read device-side fitness out of the jitted workflow.
`store_info(self, state: EvoXState, fitness)` — host callback: `self.alg_state = state.query_state("algorithm").alg_state`, `self.fitness = jax.device_get(fitness)`.
`show()` — `io_callback(self._show, None)`, i.e. asks the user to invoke it outside a trace to print/persist the last generation.
`_show()` — host-side: regenerates the population with `algorithm.ask(self.alg_state)`, computes max/min/mean/std over finite fitnesses, tracks `best_genome`/`best_fitness`, optionally writes `genome_dir/{generation}.npz` (`nodes`, `conns`, `fitness`) and appends `log.txt` (`generation,max,min,mean,std,cost_time`), then prints and calls `algorithm.show_details(self.alg_state, self.fitness)`.

## Constraints & Conventions
No host reads inside jit: all host work happens inside `io_callback` targets (`store_info`, `_show`), which must not return device values (result shape `None`).
The adaptor keeps `fixed_state` in a python attribute, so multiple adaptors wrapping different algorithms are independent; `setup` must be called before `transform`/`forward`.
EvoX `State` is treated as immutable (`state.replace(...)`); TensorNEAT's `State` is stored inside it as a pytree child.
`evox` must be installed separately (not in `pyproject.toml` dependencies); importing this package without it raises `ModuleNotFoundError` at import time.

## Known Issues
`tensorneat_monitor.py` calls `datetime.datetime.now()` for the default `save_dir` but only imports `warnings`, `os`, `time`, `numpy`, `jax`, `io_callback`, `evox` — so `is_save=True` with `save_dir=None` raises `NameError`; `warnings` is imported but unused.
`show()` → `_show()` dereferences `self.alg_state` and `self.fitness`, which are `None` until the first `pre_tell` callback has fired.
