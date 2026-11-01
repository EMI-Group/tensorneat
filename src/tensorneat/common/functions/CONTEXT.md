# tensorneat/common/functions — ACT/AGG option registries and their evoxc kernels

## Intent
Single source of truth for the activation and aggregation options a NEAT node gene may select: the evoxc-native kernels, the per-node dispatchers that replace `lax.switch`, the sympy counterparts used for interpretability, and the `FunctionManager` that pairs them in a fixed, parity-critical order.
Everything in `act.py` / `agg.py` is graph code (runs only inside an `@evoxc.defn` trace); everything in `act_sympy.py` / `agg_sympy.py` / `manager.py` is host-only.
Nothing here knows about genomes, populations or the Pipeline: a node gene hands in the option list it was configured with and a per-node id array, and gets back values.

## Module map (no child directories)
| File | Responsibility |
|---|---|
| `act.py` | 12 elementwise activation kernels, the constants they share with the sympy counterparts, `ACT_FUNCS` (canonical order), `apply_activation`. |
| `agg.py` | 6 masked aggregations over the incoming-edge axis, `AGG_FUNCS` (canonical order), `apply_aggregation`. |
| `act_sympy.py` | `sp.Function` counterparts of `ACT_FUNCS` plus `SympyClip`, `ACT_SYMPY_FUNCS`, `ACT_SYMPY_HELPERS`. |
| `agg_sympy.py` | `sp.Function` counterparts of the aggregations, `AGG_SYMPY_FUNCS`. |
| `manager.py` | `FunctionManager`: names/ids/lookups, pairing validation, `sympy_module`, `random_index`. |
| `__init__.py` | Wiring only: builds `ACT` / `AGG` and re-exports exactly five names. |

## Registry contract
A node gene stores an integer id = **position in the user's option list** (`activation_options` / `aggregation_options`), not a global registry index; `ACT_FUNCS` / `AGG_FUNCS` are the canonical order used when a genome passes the registry itself as its option list.
An option's registry name is its kernel's `__name__`, so kernels must be named after the option they implement; `FunctionManager` derives `names`, `name2index`, `index2name` from the tuples.
The order is part of the gene encoding: **append only** — reordering or deleting changes sampled ids and breaks checkpoint/parity.
Ids are sampled on the host side with `evoxc.random` (`FunctionManager.random_index(key)` → rank-0 int32 in `[0, len)`); genome code mutates them within `[0, len)` only.

### ACT — canonical order (12)
| id | name | definition (`z` elementwise, float32) |
|---|---|---|
| 0 | `scaled_sigmoid` | `SCALE / (1 + exp(-z))`, `SCALE = 3.0` |
| 1 | `sigmoid` | `1 / (1 + exp(-z))` |
| 2 | `scaled_tanh` | `SCALE * tanh(z)` |
| 3 | `tanh` | `tanh(z)` |
| 4 | `sin` | `sin(z)` |
| 5 | `relu` | `maximum(z, 0)` |
| 6 | `lelu` | `select(z > 0, z, LELU_SLACK * z)`, `LELU_SLACK = 0.005` |
| 7 | `identity` | `z` (also the `-1` sentinel target) |
| 8 | `inv` | `1 / select(z > 0, maximum(z, INV_EPS), minimum(z, -INV_EPS))`, `INV_EPS = 1e-7` |
| 9 | `log` | `log(maximum(z, LOG_EPS))`, `LOG_EPS = 1e-7` |
| 10 | `exp` | `exp(z)`, unclipped (overflows to `inf` above ~88) |
| 11 | `abs` | `abs(z)` |
Activations are shape-polymorphic and elementwise: canonical call shape `(P, N)`, also valid for `(P, out_dim)` output transforms and HyperNEAT substrate nodes; `NaN` in → `NaN` out, no masking.

### AGG — canonical order (6)
| id | name | reduce over the last axis of `z (P, N, K)` | neutral padding |
|---|---|---|---|
| 0 | `sum` | `reduce_sum` | `0.0` |
| 1 | `product` | `reduce_prod` | `1.0` |
| 2 | `max` | `reduce_max` | `-inf` |
| 3 | `min` | `reduce_min` | `+inf` |
| 4 | `maxabs` | signed value of largest `abs` (argmax + flat gather) | `0.0` |
| 5 | `mean` | `reduce_sum / maximum(count, 1.0)` | `0.0` |
Aggregations are rank-3 only (`(P, N, K)` → `(P, N)`); a consumer with a different incoming layout reshapes to `(..., K)` and passes an all-`True` mask, or reduces on its own.

## Aggregation contract over the `K` axis
`z` and `mask` are both `(P, N, K)`: axis 0 population, axis 1 node slot, axis 2 the static in-degree bound `K`; the reduce axis is always the last one and the result is `(P, N) float32`.
`mask` is supplied by the genome (`level == l` ∧ edge matched ∧ source value not `NaN`); it — not `isnan(z)` — decides which slots count, so a genuinely `NaN` contribution still poisons its node exactly as in the pre-rewrite library.
evoxc reductions have no `where=` / `initial=`, so each kernel first rewrites masked-out slots to its neutral element with `evoxc.select` and then reduces; there is no `inf` constant in evoxc, so `±inf` neutrals are built in-graph (`evoxc.numpy.full` with a float fill, or a scalar literal).
`maxabs` is **signed** and legacy-exact: masked-out slots become `0.0`, `evoxc.argmax(abs)` picks the first maximum, and one `flat_gather` (`..tools.flat_gather`) returns `z[argmax]`; reducing `abs` alone would return a magnitude and flip the sign of every negative-dominant node.
`mean` divides by `maximum(count, 1.0)` with `count = reduce_sum(cast(mask, float32), -1)`, so it is `0.0` — never `NaN` — for a node with no valid incoming edge; the pre-rewrite kernel divided by the raw count and relied on the dispatcher's guard.
`has_valid`: a node whose mask column is entirely `False` gets exactly `0.0` from the dispatcher (the legacy `all(~mask) -> 0.0` short-circuit), never `-inf` / `+inf` / `NaN`; the genome still decides via its own `has_valid` select whether the node's value is written at all (an uncomputed node keeps `NaN`).

## Dispatch contract (no `lax.switch` in evoxc)
`apply_activation(z, act_ids, act_fns=ACT_FUNCS)` — `z (...,) float32`, `act_ids (...,) float32` integral or `int32` broadcastable against `z`, `act_fns` = the node gene's static option tuple; returns `(...,) float32`.
Activation dispatch is a `select` cascade starting from `z` (the identity / `-1` branch) and folding `select(ids == i, act_fns[i](z), acc)` for every `i`: **12 evaluations + 12 selects** for the canonical list, over the `(P, N)` level slice only.
That cost is independent of `K` and of the population's actual id histogram, and it is subdominant to the `(P, N, K) × attrs` gathers the forward pass already performs; there is no static way to evaluate only the used options because the id is gene data, not a shape.
`apply_aggregation(z, mask, agg_ids, agg_fns=AGG_FUNCS)` — `z (P, N, K) float32`, `mask (P, N, K) bool`, `agg_ids (P, N)` integral, `agg_fns` = the node gene's static option tuple; returns `(P, N) float32`.
Aggregation dispatch is the same cascade (accumulator starts at `0.0`, then one `select` per option) over **6 masked reductions + 6 selects + 1 count reduction**, with `mean` reusing the zero-padded tensor of `sum`; it is the only place where the whole `(P, N, K)` tensor is traversed several times per level.
There is no `-1` sentinel for aggregation (the pre-rewrite library had none); `-1` means identity for activation only.
An id outside `[-1, len(options))` (activation) / `[0, len(options))` (aggregation) falls back to the cascade's initial value — `z` / `0.0`; the pre-rewrite `lax.switch` clamped such ids to the last option, an unreachable divergence for well-formed genomes (see Known Divergences).
`act_fns` / `agg_fns` must be **trace-time static** Python sequences (they are host configuration), and every entry must be a kernel with the documented signature; ids are cast with `evoxc.cast(ids, evoxc.int32)`, so a `NaN` id (only possible on absent nodes, whose values are discarded) casts to an unspecified int32 and never faults.

## Shapes, dtypes, sentinels
All kernels take and return `float32`; ids arrive as integral-valued `float32` gene columns and are cast to `int32` inside the dispatchers.
`NaN` is the only float sentinel (absent node row, uncomputed node value, unmatched attribute slot); `I_INF = int32 max` is the only integer sentinel and never appears in this package — the mask plus `(P, N, K)` padding already encode absence on the aggregation axis.
No bitcast exists in evoxc, which changes nothing here: activations and aggregations are numerics, not payload bit-twiddling.

## Sympy / interpretability contract (host-only)
`ACT` / `AGG` pair every option with an `sp.Function` subclass in the same order; `FunctionManager.__init__` validates the pairing by class name (`SympyScaledSigmoid` → `scaled_sigmoid`) and raises `ValueError` on a mismatch, which is what makes the legacy `"inv" → SympyIdentity` bug impossible to reintroduce.
Each sympy class is a constructor-time rewriter (`SympySigmoid(z)` returns `1/(1+exp(-z))`), so generated formulas are plain sympy expressions; `SympyClip` is the exception and defines `numerical_eval`, exposed through `sympy_module()` for `sp.lambdify`.
`sp.lambdify(..., modules=["numpy", AGG.sympy_module(), ACT.sympy_module()])` is the sanctioned codegen path; `sympy_module(backend)` accepts `"numpy"` only and raises `ValueError` for anything else (the JAX backend disappeared with the rewrite).
`sympy_module()` emits a `partial(cls.numerical_eval, backend=numpy)` for every counterpart **and** every `extra_sympy_classes` entry that defines `numerical_eval` — `SympyClip` is such a helper (`ACT_SYMPY_HELPERS`), because `SympyExp` builds it while not being an option itself; the binding is by keyword, since `numerical_eval(val, min_val, max_val, backend=np)` would otherwise receive numpy as `val`.
The codegen path is exercised, not aspirational: every activation and aggregation round-trips through `cls(expr)` → `sp.lambdify` → numeric evaluation (a module-level smoke test in `test/unit/` should keep that true).
Shared numeric constants (`SCALE`, `LELU_SLACK`, `INV_EPS`, `LOG_EPS`, `EXP_CLIP`) live in `act.py` and are imported by `act_sympy.py`, so the two representations cannot drift.

## Design Decisions
1. **`inv` pairs with `SympyInv`** (`1 / clamped(z)`), fixing the pre-rewrite `"inv" → SympyIdentity` bug: the numeric kernel was always `1/z`, so numeric parity is untouched and only interpretability output changes (formulas for `inv` nodes now match the executed kernel).
2. **`clip` is not a registry option**: the pre-rewrite sympy table carried a 13th name with no numeric counterpart, which made `ACT`'s two tables disagree in length; `clip` stays internal to `SympyExp` (`SympyClip`) and the ACT option list is exactly the 12 names above, so sampled activation ids are unchanged.
3. **Dispatch is a `select` cascade, not a branch**: evoxc has no `switch` and `cond` demands a 0-d predicate, so every option is evaluated and selected per node; the documented op counts above are the accepted price.
4. **Aggregations are rank-3 and last-axis reducing**: one shape class, one reduce axis, testable against a numpy oracle; HyperNEAT's substrate shape adaptation belongs to that node.
5. **`maxabs` stays signed** (argmax + `flat_gather`) rather than the cheaper `reduce_max(abs(z))`, because the pre-rewrite kernel returned the signed value and parity tests compare against it.
6. **The manager is immutable** (no `add_func` / `update_sympy`): a gene id indexes the *user's* option list, so custom options need no global registration; a genome wanting sympy support for a custom option constructs its own `FunctionManager`.
7. **`get_func_name` is gone**; names are registry data (`ACT.name_of(fn)`, `ACT.index2name[id]`), which is also what `to_dict` / `repr` need.
8. **Registry validation raises `ValueError`, not `ConfigError`**: a mismatched table is a programmer error in this package, whereas `common.errors.ConfigError` is reserved for user configuration raised by the layer that owns the config.

## Known Divergences from the pre-rewrite library
`SympyExp` clips its argument to `[-EXP_CLIP, EXP_CLIP] = [-10, 10]` while `exp` does not — generated formulas over/under-estimate strongly positive/negative pre-activations (kept for formula stability, as before).
Out-of-range activation/aggregation ids fall back to identity / `0.0` instead of clamping to the last option (`lax.switch` behaviour); mutation only samples valid ids, so this is reachable only through hand-written genes or a stale checkpoint replayed against a shortened option list.
`mean` is `0.0` instead of `NaN` for an all-`False` mask; indistinguishable through the dispatcher, different only when the kernel is called standalone.
`sympy_module` no longer accepts `"jax"`; `obtain_sympy` accepts an option id as well as a name or callable.
`sympy_module()` binds numpy as the *keyword* `backend` (the pre-rewrite `partial(fn, jnp_or_np)` bound it to `val`, so `lambdify`d `clip` evaluations raised `AttributeError`); and `SympyMaxabs` renders as an explicit `Piecewise` over `Eq(Abs(t_i), max_j Abs(t_j))`, because `sp.Max(*z, key=sp.Abs)` is rejected by current sympy — the signed term and its first-maximum tie-break are unchanged.
The aggregation reduce axis moved from the legacy per-node axis 0 to the batch-first last axis of `(P, N, K)`; numerically identical, but HyperNEAT substrate code that used to call `AGG.sum(inputs)` directly must reshape to `(..., K)`.

## Constraints
- No `jax`, `torch`, `numpy.random` anywhere in this package; graph code uses `evoxc` ops only, host code may use `numpy` / `sympy`.
- Nothing here imports `genome`, `algorithm`, `problem`, `pipeline`, `benchmarks` or `test`; the only intra-package dependency is `..tools.flat_gather` (used by `maxabs`).
- Files ≤ ~1000 lines; each file here is a single responsibility (kernels+dispatcher per side, sympy per side, registry bookkeeping).
- Do not reorder `ACT_FUNCS` / `AGG_FUNCS` or their sympy tuples; append only.
- `sum`, `max`, `min`, `abs` shadow builtins inside `agg.py` / `act.py` — that is intentional (the registry names must equal the kernel names); never use the builtin of the same name in those modules.
- Errors: broken tables raise `ValueError` at import time (fail fast); unknown option names/ids raise `ValueError` / `IndexError`; nothing is swallowed.

## Consumers (the API sibling nodes may rely on)
`from tensorneat.common import ACT, AGG, apply_activation, apply_aggregation` is the only public surface of this package (`__all__` is exactly those five names, `FunctionManager` included).
Node genes: `apply_aggregation(z, mask, agg_ids, self.aggregation_options)` then `apply_activation(z, act_ids, self.activation_options)`; the option lists are plain tuples of kernels, normalised once in the gene constructor.
Configs reference an option by name through attribute access (`AGG.sum`, `ACT.sigmoid`) and resolve ids through `ACT.name2index` / `ACT.get_evoxc_func`; `ACT.random_index(key)` samples an id in-graph from a stateless evoxc key.
Interpretability: `ACT.obtain_sympy(name_or_id_or_func)` / `AGG.obtain_sympy(...)` return the sympy class for a `to_dict`-style name.
The parent `common/__init__.py` must import exactly `ACT, AGG, apply_activation, apply_aggregation` (plus optionally `FunctionManager`) — the legacy `get_func_name` import no longer exists.

## Test Hooks
`test/unit/` must be able to test every kernel on the numpy backend with no optional dependency: each activation against a numpy reference on a `(P, N)` tensor, each aggregation against a numpy oracle (including an all-`False` mask column and a `maxabs` tie), and the dispatchers against a per-element Python loop over the option list.
Host-side tests need no backend at all: registry order/names, pairing validation (feed a deliberately wrong sympy tuple and expect `ValueError`), `name_of` / `get_sympy_func` / `obtain_sympy`, and `sympy_module()` shape.
`test/parity/` compares each kernel and the dispatchers against `benchmarks/legacy_jax/tensorneat_legacy/common/functions/` with the same inputs and option lists — the legacy `maxabs` sign, the `mean` guard and the `-1` identity sentinel are the three cases most likely to regress.
