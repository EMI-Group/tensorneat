# tensorneat/common/functions — ACT/AGG registries used by node genes

## Intent
Single source of truth for the activation and aggregation functions that a NEAT node gene may select, holding a jnp implementation (traced forward) and a sympy counterpart (interpretability) for each name, plus the `lax.switch` dispatchers.
Consumed by `../../genome/gene/node/{default,bias}.py` (`apply_activation`, `apply_aggregation`, `ACT.obtain_sympy`, `AGG.obtain_sympy`), `../../algorithm/hyperneat/*`, `../../genome/default.py` (`AGG/ACT.sympy_module(backend)` for `sp.lambdify`), and user configs (`ACT.tanh`, `AGG.sum`).

## API Surface
### `__init__.py` (registry tables + dispatchers)
Four plain dicts, in this key order: `act_name2jnp` = scaled_sigmoid, sigmoid, scaled_tanh, tanh, sin, relu, lelu, identity, inv, log, exp, abs (12 entries, each a `*_` function from `act_jnp`); `act_name2sympy` = the same 12 plus `clip` (13 entries); `agg_name2jnp` / `agg_name2sympy` = sum, product, max, min, maxabs, mean (6 each).
`ACT = FunctionManager(act_name2jnp, act_name2sympy)`, `AGG = FunctionManager(agg_name2jnp, agg_name2sympy)`.
`apply_activation(idx, z, act_funcs)` — casts `idx` to `jnp.int32`, then `jax.lax.cond(idx == -1, lambda: z, lambda: jax.lax.switch(idx, act_funcs, z))`; `idx == -1` is the identity sentinel produced by `BiasNode.new_identity_attrs`, otherwise `idx` indexes the static python list `act_funcs`.
`apply_aggregation(idx, z, agg_funcs, mask)` — casts `idx` to int32, then `jax.lax.cond(jnp.all(~mask), lambda: 0.0, lambda: jax.lax.switch(idx, agg_funcs, z, mask))`, i.e. an empty input mask short-circuits to `0.0` and aggregation functions always receive `(z, mask)`.
`get_func_name(func)` — `func.__name__` with one trailing `_` stripped (used by node `repr`/`to_dict` to recover the registry name).

### `act_jnp.py` — scalar→scalar, pure jnp
`SCALE = 3`; `scaled_sigmoid_`, `sigmoid_`, `scaled_tanh_`, `tanh_`, `sin_`, `relu_` (`jnp.maximum(z, 0)`), `lelu_` (leaky 0.005), `identity_`, `inv_` (clamps `|z| >= 1e-7` then `1/z`), `log_` (`jnp.maximum(z, 1e-7)` then `log`), `exp_`, `abs_`.
Signature convention: `f(z)` only, no mask, no side data.

### `agg_jnp.py` — `f(z, mask)` reducing over axis 0
`sum_`, `product_`, `max_`, `min_` use `jnp.{sum,prod,max,min}(z, axis=0, where=mask, initial=<0/1/-inf/+inf>)`; `maxabs_` zeroes masked entries then returns `z[jnp.argmax(jnp.abs(z))]`; `mean_` = masked sum / `jnp.sum(mask, axis=0)`.

### `act_sympy.py` — `sp.Function` subclasses with `eval(cls, z)`
`SympySigmoid`, `SympyScaledSigmoid` (= `SympySigmoid(z)*3`), `SympyTanh`, `SympyScaledTanh`, `SympySin`, `SympyRelu` (`sp.Max(z, 0)`), `SympyLelu` (`sp.Piecewise`), `SympyIdentity`, `SympyInv`, `SympyLog`, `SympyExp` (wraps `SympyClip(z, -10, 10)`), `SympyAbs`.
`SympyClip` has three args and defines `numerical_eval(val, min_val, max_val, backend=np)` plus `_sympystr`/`_latex` so `lambdify` can evaluate it.

### `agg_sympy.py` — `sp.Function` over an iterable of exprs
`SympySum` (`sp.Add(*z)`), `SympyProduct` (`sp.Mul(*z)`), `SympyMax`, `SympyMin`, `SympyMaxabs` (`sp.Max(*z, key=sp.Abs)`), `SympyMean` (`sp.Add(*z)/len(z)`).

### `manager.py` — `FunctionManager(name2jnp, name2sympy)`
`__init__` stores both dicts and `setattr(self, name, func)` for every jnp entry, so registries are used as `ACT.tanh`, `AGG.sum`, `ACT.identity`.
`get_all_funcs()` → list of the jnp callables in `name2jnp` order.
`add_func(name, func)` → validates `callable`, rejects an existing name, appends to `name2jnp` and `setattr`s it (jnp half of a registration).
`update_sympy(name, sympy_cls: sp.Function)` → fills the sympy half.
`obtain_sympy(func)` → accepts a registry name (`str`) or a callable (matched with `f == func` over `name2jnp`) and returns the sympy class; raises `ValueError` for unknown names/functions.
`_obtain_sympy_by_name(name)`.
`sympy_module(backend)` → `assert backend in ["jax","numpy"]`, then a dict `{sympy_cls.__name__: partial(sympy_cls.numerical_eval, backend)}` for the classes that define `numerical_eval` (only `SympyClip` today); it is passed to `sp.lambdify(..., modules=[backend, AGG.sympy_module(backend), ACT.sympy_module(backend)])`.

## Registering a new function
1. Add `foo_(z)` (activation, single arg) or `foo_(z, mask)` (aggregation, axis-0 reduction with `where`/`initial`) in `act_jnp.py` / `agg_jnp.py`.
2. Optionally add `class SympyFoo(sp.Function)` with a classmethod `eval` in `act_sympy.py` / `agg_sympy.py` (and `numerical_eval` if `lambdify` must evaluate it).
3. At runtime: `ACT.add_func("foo", foo_)` (or `AGG.add_func`) and `ACT.update_sympy("foo", SympyFoo)` — see `examples/func_fit/custom_func_fit.py`; adding to the module-level dicts is not enough, since `__init__.py` snapshots them at import.
4. Reference it in the node gene: `activation_options=[ACT.identity, ACT.foo]` / `aggregation_options=[...]`; the stored gene value is the **position in that user list**, not the registry index.

## Constraints & Conventions
Everything in `act_jnp.py`/`agg_jnp.py` is traced: jnp only, no numpy, no python branching on traced values.
`lax.switch` requires a static python list of branches with identical signatures; the option list is fixed at gene-construction time (`DefaultNode.__init__` normalises a bare callable to `[callable]` and stores `*_default` as `options.index(default)`), so per-individual dispatch happens only via the integer stored in the gene row.
`apply_activation` / `apply_aggregation` do not bound-check `idx`; `lax.switch` clamps out-of-range indices, so a stale index silently selects the last option.
Neither `act_jnp` nor `act_sympy` is imported lazily: both are `*`-imported eagerly by `__init__.py`, so `import tensorneat` pulls in `sympy` (a declared pyproject dependency); only `sympy_module()`'s `partial(..., numerical_eval)` bindings are created on demand.
Registering a name that collides with a `FunctionManager` method or attribute (`add_func`, `sympy_module`, `name2jnp`, ...) silently shadows that method through `setattr` — avoid such names.

## Known Issues
`act_name2sympy["inv"] = SympyIdentity`, so sympy/interpretability output for an `inv` activation renders as the identity instead of `1/z`, and the existing `SympyInv` class is dead code.
`act_name2sympy` has a `clip` entry with no `act_name2jnp` counterpart, so `ACT.clip` does not exist, `ACT.get_all_funcs()` yields 12 activations while `ACT.name2sympy` has 13, and the two indices are not interchangeable.
`agg_jnp.mean_` divides by the mask count with no guard, so it yields NaN for an all-False mask; correctness depends on `apply_aggregation`'s `jnp.all(~mask)` shortcut rather than on the function itself.
`FunctionManager.obtain_sympy` matches callables by `==`, so a `functools.partial`/wrapped or lambdified variant of a registered function raises `ValueError("Func ... doesn't not registered.")`.
