# tensorneat/common — shared primitives: immutable `State`, jit-able index/graph helpers, ACT/AGG registries

## Intent
Framework-wide foundation layer: immutable pytree `State`, `StatefulBaseClass` mixin, jit-able index/hash/graph helpers, the ACT/AGG function registries, host-side sympy code generation, and optional evox adaptors.
Depends only on `jax`, `numpy`, `sympy` (and `evox` only inside `evox_adaptors/`); the only modules importing `genome`/`algorithm` types are the `evox_adaptors` modules, which are not reached from `common/__init__.py`.

## API Surface
`__init__.py`: `from .tools import *`, `from .graph import *`, `from .state import State`, `from .stateful_class import StatefulBaseClass`, `from .functions import ACT, AGG, apply_activation, apply_aggregation, get_func_name`.
The two star-imports have no `__all__`, so `tensorneat.common` also re-exports `np`, `jax`, `jnp`, `jit`, `vmap`, `partial`, `Array`, `Tuple/Set/List/Union` and every tool/graph name (including `fetch_first`, `I_INF`).
`sympy_tools` and `evox_adaptors` are NOT imported by `__init__.py`; import them as `tensorneat.common.sympy_tools` / `tensorneat.common.evox_adaptors`.

### state.py — `State` (immutable pytree)
`State(**kwargs)` stores a dict in `self.__dict__["state_dict"]`; `@register_pytree_node_class` makes it a JAX pytree whose children are the dict values and aux_data the key list (so insertion order defines flatten order).
`registered_keys()` → `dict_keys`; `__getattr__(name)` → `state_dict[name]`, so a missing key raises `KeyError` (not `AttributeError`, which breaks `hasattr`/`getattr(state, k, default)` probes); `__contains__`; `__repr__`.
`register(**kwargs)` → new `State` (raises `ValueError` if any key already exists), `update(**kwargs)` → new `State` (raises if any key is missing), `remove(*keys)` → new `State` (raises if any key is missing); all three are pure and preserve insertion order.
`__setattr__` always raises `AttributeError("State is immutable")`, and `__init__` bypasses it by writing `self.__dict__` directly.
`save(file_name)` / `classmethod load(file_name)` pickle the whole `State` via `__getstate__`/`__setstate__`.
`tree_flatten()` / `tree_unflatten(aux_data, children)` are the explicit pytree hooks.
Consequence: `state.register/update(...)` must be *reassigned*; `state.x = v` and in-place mutation are impossible.

### stateful_class.py — `StatefulBaseClass`
`setup(self, state=State())` returns the state unchanged (default arg is a shared immutable `State`, harmless because `State` is immutable); subclasses override to register own keys and recurse into children.
`show_config(self, registered_objects=None)` → nested dict of `str(attr)` with cycle protection via the `registered_objects` list; used for config logging.
`save`/`load` pickle helpers are commented out (TODO "Bug need be fixed"); there is no working persistence for stateful objects beyond `State.save`.

### tools.py — index/sentinel/hash helpers
`I_INF = np.iinfo(jnp.int32).max` (2147483647), the "no index / unavailable" sentinel for int32 arrays (int arrays cannot hold NaN).
`attach_with_inf(arr, idx)` — gather-with-sentinel: expands `idx` to `arr.ndim + idx.ndim - 1` dims (`jnp.expand_dims(idx, range(idx.ndim, target_dim))`) so the trailing custom-attr axes of `arr` broadcast, then `jnp.where(expand_idx == I_INF, jnp.nan, arr[idx])`; absent slots become NaN while real hits keep the attr axis.
`fetch_first(mask, default=I_INF)` `@jit` — `jnp.argmax` trick: index of first True, else `default`.
`fetch_random(randkey, mask, default=I_INF)` `@jit` — draws `jax.random.randint(k, (), 1, sum(mask)+1)` against `cumsum(mask)` and delegates to `fetch_first`; an all-False mask is forced to False so `default` is returned.
`rank_elements(array, reverse=False)` `@partial(jit, static_argnames=["reverse"])` — dense ranks from `argsort(argsort)`; by default the largest value gets rank 0.
`mutate_float(randkey, val, init_mean, init_std, mutate_power, mutate_rate, replace_rate)` `@jit` and `mutate_int(randkey, val, options, replace_rate)` `@jit` — NEAT float/int mutation kernels (`jax.random.split`, `jnp.where`, `jax.random.choice`).
`argmin_with_mask(arr, mask)` — plain jnp, NOT jitted: `argmin(jnp.where(mask, arr, jnp.inf))`.
`hash_array(arr)` — `jax.lax.bitcast_convert_type(arr, jnp.uint32)` then `jax.lax.fori_loop` xor-mix (`0x9E3779B9`, shifts 6/2) from `jnp.uint32(0)`; jit/vmap-compatible despite no decorator; input must be 32-bit (used on float32 gene rows).

### graph.py — jit-able vs host-side graph algorithms
`topological_sort(nodes, conns)` `@jit` — the jit-able Kahn loop; `conns` is a boolean `(N, N)` adjacency matrix, `nodes` is `(N, attrs)` with NaN rows for absent nodes.
- in-degree: `jnp.where(jnp.isnan(nodes[:, 0]), jnp.nan, jnp.sum(conns, axis=0))`, so NaN rows never compare equal to 0 and are skipped; `res = jnp.full(shape, I_INF)`.
- `lax.while_loop(cond_fun, body_func, (res, 0, in_degree))`; cond is `fetch_first(in_degree == 0.0) != I_INF`, body sets `res[idx] = i`, `in_degree[i] = -1`, then `in_degree = jnp.where(conns[i, :], in_degree - 1, in_degree)`.
- Returns `(max_nodes,)` int32: topological order of node **row** indices, `I_INF` in leftover slots; a cycle silently truncates the sequence (no error), which is why cyclic graphs must use the recurrent forward.
`topological_sort_python(nodes, conns)` — host-side (python sets/lists), returns `(topo_order: List[int], topo_layer: List[List[int]])` (layers sorted ascending), raises `ValueError("Graph has at least one cycle, topological sort not possible")` on cycles; note it mutates its copies and the docstring credits gpt4.
`find_useful_nodes(nodes, conns, output_idx)` → `Set[int]` — host-side backward reachability from outputs (iterative set closure).
`check_cycles(nodes, conns, from_idx, to_idx)` `@jit` — sets `conns[from_idx, to_idx] = True`, then a boolean transitive-closure `lax.while_loop` (`jnp.dot(visited, conns)` OR-ed into `visited`), stopping when nothing new is visited or `from_idx` is reached; returns `visited[from_idx]` (True = the new edge closes a cycle).

### sympy_tools.py — host-side code generation (sympy + regex, never traced)
`analysis_nodes_exprs(nodes_exprs)` → `(input_cnt, hidden_cnt, output_cnt, norm_symbols)` by string prefix of the symbols (`i*`, `h*`, `o*`, `norm*`).
`round_expr(expr, precision=2)` — `expr.xreplace` rounding every `sp.Number` atom.
`replace_variable_names(expression, mode)` — `mode in {"python","latex"}`; regex `\b[ohi](\d+)\b` → `o[0]` / `o_{0}`; asserts on other modes.
`to_latex_code(...)` / `to_python_code(...)` — both take the 8-tuple produced by `DefaultGenome.sympy_func` (`symbols, args_symbols, input_symbols, nodes_exprs, output_exprs, forward_func, topo_order, useful_nodes`); iterate `topo_order[input_cnt:]`, skip nodes not in `useful_nodes`, `subs` args/norm symbols, round to 2 (latex) / 6 (python), and emit `\begin{align}` lines or `h = np.zeros(...)`/`o = np.zeros(...)` assignments.
The `input_symbols`, `output_exprs` and `forward_func` parameters are accepted but unused by both code generators.

## JAX Usage Map
`@jit`: `tools.fetch_first`, `tools.fetch_random`, `tools.rank_elements` (`@partial(jit, static_argnames=["reverse"])`), `tools.mutate_float`, `tools.mutate_int`, `graph.topological_sort`, `graph.check_cycles`.
Jit-compatible but undecorated (pure jnp/lax, traced by callers): `tools.attach_with_inf`, `tools.argmin_with_mask`, `tools.hash_array`, `functions.apply_activation`, `functions.apply_aggregation`, all `act_jnp` / `agg_jnp` functions.
`lax.while_loop`: `graph.topological_sort` (Kahn loop over `in_degree`), `graph.check_cycles` (boolean transitive closure). `lax.fori_loop`: `tools.hash_array` (over `arr.size`).
`lax.switch`: `apply_activation` (per-node activation) and `apply_aggregation` (per-node aggregation). `lax.cond`: both of those plus the empty-mask shortcut.
`vmap`: not used in this node; callers vmap `attach_with_inf`/`hash_array`/gene `forward` (e.g. `genome/operations/crossover/default.py`, `genome/recurrent.py`).
Host-side python/numpy (never traced): `graph.topological_sort_python`, `graph.find_useful_nodes`, all of `sympy_tools.py`, all `FunctionManager` methods, `State.save/load`, `StatefulBaseClass.show_config`, everything in `evox_adaptors` outside `io_callback`.

## Tensorization Trick (how this node's helpers form it)
Step 1, host/`transform`: `../genome/utils.py::unflatten_conns(nodes, conns)` maps the ragged `(max_conns, ...)` rows to a dense `(max_nodes, max_nodes)` int32 matrix `u_conns` holding the connection row index, with `I_INF` where no connection exists (`jnp.full(..., I_INF).at[i_idxs, o_idxs].set(jnp.arange(C))`).
Step 2, `transform` (feedforward): `topological_sort(nodes, u_conns != I_INF)` returns `cal_seqs`, a fixed-length `(max_nodes,)` int32 order, so the variable-length graph becomes a `lax.fori_loop(0, max_nodes, ...)` with per-iteration validity masks.
Step 2', `transform` (recurrent): no ordering at all — `attach_with_inf(conns_attrs, u_conns)` yields the dense `(max_nodes, max_nodes, attrs)` weight tensor consumed by `lax.fori_loop(0, activate_time, ...)`.
Step 3, `forward`: per destination node `i`, `u_conns[:, i]` gives every incoming edge, `attach_with_inf(conns_attrs, conn_indices)` gathers their weights (NaN for absent edges) and NaN-padded rows/values are filtered by `conn_exists & ~jnp.isnan(values)` masks, so all arrays stay rectangular and `vmap`/`lax.switch` can run branch-free.
This is why `attach_with_inf` must emit NaN rather than 0 (it doubles as the "edge absent" mask) and why `I_INF` and NaN are both required.

## Constraints & Conventions
No host-side reads inside jit: everything traced must be jnp/lax; data-dependent trip counts use `lax.while_loop` (Kahn's algorithm, transitive closure), never python `for`.
Two disjoint sentinels: NaN for "absent/invalid" float data (padded node/conn rows, missing source values), `I_INF` (`int32` max) for "no index / unavailable index"; never mix them within one array.
`attach_with_inf` returning NaN is the reason every consumer combines its mask with `~jnp.isnan(values)` (`src_valid`) before aggregating.
`State` is immutable and a pytree: all mutation goes through `register`/`update`/`remove` returning a new `State`, which keeps algorithms functional under jit/vmap.
`lax.switch` operand lists (activation/aggregation options) must be static python lists fixed at trace time — options are a construction-time choice, indices stored per-node in the gene row.
Activation callables take `(z)`; aggregation callables take `(z, mask)` and must reduce over axis 0 with `where=mask` and an `initial=` (see `functions/agg_jnp.py`).
There are no `NUM_ACT`/`NUM_AGG` constants anywhere in the repo — derive counts from `len(ACT.name2jnp)` / `len(ACT.name2sympy)`.

## Known Issues
`evox_adaptors/tensorneat_monitor.py` calls `datetime.datetime.now()` but never imports `datetime` (`warnings` is imported unused) → `NameError` when `is_save=True` with `save_dir=None`.
`functions/__init__.py`: `act_name2sympy["inv"] = SympyIdentity`, so sympy/interpretability output for an `inv` activation is the identity instead of `1/z`, and the defined `SympyInv` class is unused; `act_name2sympy` also has a `"clip"` entry with no jnp counterpart (13 sympy vs 12 jnp activations).
`tools.argmin_with_mask` returns 0 for an all-False mask (argmin over an all-`inf` array).
`attach_with_inf` gathers with `I_INF` indices, which are out of bounds; the garbage is discarded by the surrounding `jnp.where`, so correctness relies on JAX's index clamping rather than on in-bounds indices.
`tools.fetch_random` still draws `jax.random.randint(..., minval=1, maxval=true_cnt+1)` before guarding an empty mask, so `true_cnt == 0` relies on JAX tolerating `minval == maxval`.

## Routing Table
- `functions/` — ACT/AGG registries (`apply_activation`/`apply_aggregation` with `lax.switch`), jnp/sympy function pairs, `FunctionManager`. See `functions/CONTEXT.md`.
- `evox_adaptors/` — `EvoXAlgorithmAdaptor`, `TensorNEATMonitor`; the only place `evox` is imported (not a pyproject dependency). See `evox_adaptors/CONTEXT.md`.
- Consumers outside this node: `../../genome/default.py` (`topological_sort`, `attach_with_inf`, `find_useful_nodes`, `topological_sort_python`), `../../genome/recurrent.py` (`attach_with_inf`, `I_INF`), `../../genome/utils.py` (`fetch_first`, `I_INF` in `unflatten_conns`), `../../genome/gene/**`, `../../algorithm/**` (`fetch_first`, `rank_elements`, `argmin_with_mask`), `../../pipeline.py`.
