# tensorneat/common/functions — ACT/AGG option registries and their evoxc kernels

## Intent
Single source of truth for the activation and aggregation options a NEAT node gene may select: the evoxc-native kernels, the per-node dispatchers that replace `lax.switch`, the sympy counterparts used for interpretability, and the `FunctionManager` that pairs them in a fixed, parity-critical order.
Everything in `act.py` / `agg.py` is graph code (runs only inside an `@evoxc.defn` trace); everything in `act_sympy.py` / `agg_sympy.py` / `manager.py` is host-only.
Nothing here knows about genomes, populations or the Pipeline: a node gene hands in the option list it was configured with and a per-node id array, and gets back values.

## Module map (no child directories)
| File | Responsibility |
|---|---|
| `act.py` | 12 elementwise activation kernels, the constants they share with the sympy counterparts, `ACT_FUNCS` (canonical order), `apply_activation`. |
| `agg.py` | 6 masked aggregations over the incoming-edge axis, the static `K`-plane folds (`_planes`/`_sum_last`/`_fold_last`) that replace `reduce_*` over that axis, the `_Window` memo that shares their padded tensors and live-edge count, `AGG_FUNCS` (canonical order), `apply_aggregation`. |
| `act_sympy.py` | `sp.Function` counterparts of `ACT_FUNCS` plus `SympyClip`, `ACT_SYMPY_FUNCS`, `ACT_SYMPY_HELPERS`. |
| `agg_sympy.py` | `sp.Function` counterparts of the aggregations, `AGG_SYMPY_FUNCS`. |
| `manager.py` | `FunctionManager`: names/ids/lookups, pairing validation, `sympy_module`, `random_index`. |
| `__init__.py` | Wiring only: builds `ACT` / `AGG`; `__all__` is exactly five names (the canonical tuples stay importable but are not public surface). |

## Registry contract
A node gene stores an integer id = **position in the user's option list** (`activation_options` / `aggregation_options`), not a global registry index; `ACT_FUNCS` / `AGG_FUNCS` are the canonical order used when a genome passes the registry itself as its option list.
An option's registry name is its kernel's `__name__`, so kernels must be named after the option they implement; `FunctionManager` derives `names`, `name2index`, `index2name` from the tuples.
The order is part of the gene encoding: **append only** — reordering or deleting changes sampled ids and breaks checkpoint/parity.
Ids are sampled with `evoxc.random` stateless keys (`FunctionManager.random_index(key)` → rank-0 int32 in `[0, len)`, in-graph) and mutated within `[0, len)` only.
`random_index` draws `evoxc.random.randint(key, (1,), 0, len(self))` and reshapes it to rank 0: the numpy reference backend returns a bare NumPy scalar for a 0-d `randint` result and then rejects it while building the output tensor, so the `(1,)` draw plus `evoxc.reshape` is the working spelling of the documented rank-0 contract.
Host lookups (`get_evoxc_func`, `get_sympy_func`, `obtain_sympy`) reject ids outside `[0, len)` with `IndexError` instead of wrapping, so a corrupted gene id fails loudly.
Every kernel is a plain module-level function, so an option tuple stays picklable (checkpoint path) and importable by name from `act.py` / `agg.py`.

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
| 7 | `identity` | `z` (also the `-1` sentinel target; carries `is_identity = True`) |
| 8 | `inv` | `1 / select(z > 0, maximum(z, INV_EPS), minimum(z, -INV_EPS))`, `INV_EPS = 1e-7` |
| 9 | `log` | `log(maximum(z, LOG_EPS))`, `LOG_EPS = 1e-7` |
| 10 | `exp` | `exp(z)`, unclipped (overflows to `inf` above ~88) |
| 11 | `abs` | `abs(z)` |
Activations are shape-polymorphic and elementwise: canonical call shape `(P, N)`, also valid for `(P, out_dim)` output transforms and HyperNEAT substrate nodes; `NaN` in → `NaN` out, no masking.
The kernel formulas are bit-identical to the pre-rewrite library (same expression order); `evoxc.sigmoid`/`evoxc.relu` and friends are **not** used, because their rounding differs.

### AGG — canonical order (6)
| id | name | reduce over the last axis of `z (P, N, K)` | neutral padding | `zero_on_empty` |
|---|---|---|---|---|
| 0 | `sum` | `reduce_sum` | `0.0` | `True` |
| 1 | `product` | `reduce_prod` | `1.0` | `False` |
| 2 | `max` | `reduce_max` | `-inf` | `False` |
| 3 | `min` | `reduce_min` | `+inf` | `False` |
| 4 | `maxabs` | signed value of largest `abs` (argmax + flat gather) | `0.0` | `True` |
| 5 | `mean` | `reduce_sum / maximum(count, 1.0)` | `0.0` | `True` |
Aggregations are rank-3 only (`(P, N, K)` → `(P, N)`); a consumer with a different incoming layout reshapes to `(..., K)` and passes an all-`True` mask, or reduces on its own.

## Static kernel flags (the contract the dispatchers read)
Three attributes are attached to a kernel and read at trace time; a custom kernel supplied by a genome simply lacks them, and the defaults (`False` / absent) always mean "emit the general path", never "skip work":
- `identity.is_identity = True` — selecting this option returns the incoming value unchanged, so `apply_activation` emits no `equal`/`select` pair for it.
- `agg.<kernel>.window_body = _<kernel>_body` — the `(window) -> (P, N)` body that takes the shared `_Window` of one `(z, mask)` pair; absent means the dispatcher calls `fn(z, mask)` directly.
- `agg.<kernel>.zero_on_empty` — `True` iff the kernel already returns exactly `0.0` for an all-`False` mask column (`sum`, `mean`, `maxabs`); `False` (`product`, `max`, `min`, and every unannotated custom kernel) means the option needs the `has_valid` guard.
No dispatcher branches on a kernel's *identity* (`fn is sum`); the flags are the extension point for a genome's own option lists.

## Aggregation contract over the `K` axis
`z` and `mask` are both `(P, N, K)`: axis 0 population, axis 1 node slot, axis 2 the static in-degree bound `K`; the reduce axis is always the last one and the result is `(P, N) float32`.
`mask` is supplied by the genome (`level == l` ∧ edge matched ∧ source value not `NaN`); it — not `isnan(z)` — decides which slots count, so a genuinely `NaN` contribution still poisons its node exactly as in the pre-rewrite library.
evoxc reductions have no `where=` / `initial=`, so each kernel first rewrites masked-out slots to its neutral element with `evoxc.select` and then **folds** the `K` planes (`_sum_last` / `_fold_last`), never `reduce_*` over the last axis; the `±inf` neutrals are scalar literals that `evoxc.select` weak-promotes to the tensor's float32 dtype.

### The `K` axis is folded, not reduced
`K` is static, so every kernel replaces its `reduce_*`-over-`-1` with a fixed-length chain of `(P, N, 1)` `evoxc.slice` views and elementwise ops; `_planes(x)` emits the `K` slices and the result is reshaped to `(P, N)` once.
The reason is measured, not stylistic: one `np.sum(..., axis=-1)` is the most expensive op of a forward-pass level (≈1.4 ms of a 3.1 ms level at `(P, N, K) = (2000, 50, 4)`), because numpy runs its buffered outer-loop path over a 4-element inner axis while a chain of `K` adds streams `(P, N)` planes that stay cache resident.
**The association order is the reference backend's, on purpose**, because `float32` addition is not associative and `test/unit/` + `test/parity/` compare results bit for bit: `_sum_last` reproduces numpy's `pairwise_sum` — a left-to-right accumulation starting from `+0.0` for `K < 8`, and eight partial accumulators (`r[j]` over `j, j+8, …`) combined as `((r0+r1)+(r2+r3)) + ((r4+r5)+(r6+r7))` plus a sequential `K % 8` tail for `K >= 8`.
The `+0.0` start is load-bearing for exactly one case (an all-`-0.0` column sums to `+0.0`) and costs one op, so it is never elided.
`_fold_last` covers `product` / `max` / `min`, which the reference backend reduces left to right at every width, so a left fold from the neutral element is the whole recipe there (`1.0 * -0.0` is `-0.0`, `maximum(-inf, a)` and `minimum(+inf, a)` are `a`, `NaN` included — which is why the padding neutral and the fold identity share one constant).
The fold cost is `K` slices + `K - 1` adds + one `reshape` (plus the `+0.0` start) — at most `K + 1` more ops than the single reduction it replaces, all on `(P, N, 1)` instead of `(P, N, K)`.
Nothing here is deferred-op or unsigned dtype territory: `slice` / `add` / `multiply` / `maximum` / `minimum` / `reshape` are all v1-exportable, and no `scatter`, `cumsum` or `flip` is involved.
`_Window.padded(neutral)` memoises that `select(mask, z, neutral)` per distinct neutral, and `_Window.count()` memoises `reduce_sum(cast(mask, float32), -1)`, so a dispatch that mixes `sum`, `mean` and `maxabs` materialises the zero-padded window **once** and the live-edge count **once**; sharing is exact (identical expressions, no reassociation), not approximate.
`maxabs` is **signed** and legacy-exact: masked-out slots become `0.0`, `evoxc.argmax(abs)` picks the first maximum (int64, cast to int32 for the gather), and one `flat_gather` (`..tools.flat_gather`) reads the **zeroed** tensor at `argmax`, reshaping `(P, N, K)` to `(P*N, K)` and the index block to `(P*N,)` first; reducing `abs` alone would return a magnitude and flip the sign of every negative-dominant node, and reading `z` instead of the zeroed tensor would let an all-`False` column return a padding slot instead of `0.0`.
`mean` divides by `maximum(count, 1.0)` with `count = reduce_sum(cast(mask, float32), -1)`, so it is `0.0` — never `NaN` — for a node with no valid incoming edge; the pre-rewrite kernel divided by the raw count and relied on the dispatcher's guard.
`has_valid`: a node whose mask column is entirely `False` gets exactly `0.0` from the dispatcher (the legacy `all(~mask) -> 0.0` short-circuit), never `-inf` / `+inf` / `NaN`; the genome still decides via its own `has_valid` select whether the node's value is written at all (an uncomputed node keeps `NaN`).

## Dispatch contract (no `lax.switch` in evoxc)
`apply_activation(z, act_ids, act_fns=ACT_FUNCS)` — `z (...,) float32`, `act_ids (...,) float32` integral or `int32` broadcastable against `z`, `act_fns` = the node gene's static option tuple; returns `(...,) float32`.
Activation dispatch is a `select` cascade starting from `z` (the identity branch, which is also the `-1` / out-of-range branch) and folding `select(ids == i, act_fns[i](z), acc)` for every option **except** an `is_identity` option; the id column is used as is when it is already `int32` (the gene's `prepare` casts once per forward pass) and cast only for a `float32` gene column.
`apply_aggregation(z, mask, agg_ids, agg_fns=AGG_FUNCS)` — `z (P, N, K) float32`, `mask (P, N, K) bool`, `agg_ids (P, N)` integral, `agg_fns` = the node gene's static option tuple; returns `(P, N) float32`.
Aggregation dispatch builds one `_Window(z, mask)` and folds `select(ids == i, body_i(window), acc)` for every option; the **first** `select` carries the `else` value `0.0`, which is exactly the pre-rewrite zeroed accumulator for an out-of-range id, so no separate `enp.zeros` accumulator op is emitted.
The final `has_valid` select (`select(count > 0, acc, 0.0)`) is emitted only when at least one option is not `zero_on_empty` — the case where a padded `product` / `max` / `min` would leak `1.0` / `-inf` / `+inf` into a node with no computable incoming edge; a one-option list of `sum` / `mean` / `maxabs` needs no guard at all, and its `else` value already returns exactly `0.0` for an out-of-range id.
`act_fns` / `agg_fns` must be **trace-time static** Python sequences (they are host configuration), and every entry must be a callable with the documented signature.
An id outside `[-1, len(options))` (activation) / `[0, len(options))` (aggregation) falls back to the cascade's initial value — `z` / `0.0`; the pre-rewrite `lax.switch` clamped such ids to the last option, an unreachable divergence for well-formed genomes.

## Measured op counts (numpy reference backend, graph ops incl. `constant` and `return`)
One dispatch call, `(P, N, K) = (100, 50, 8)`:

| graph | 1-option list | canonical list |
|---|---|---|
| `apply_activation` | 5 (`tanh`; 10 when the kernel itself needs more ops) | 74 (12 options, identity skipped) |
| `apply_aggregation` | 7 (`sum`; 12 for `max`) | 57 (6 options) |
| one level: `apply_activation(apply_aggregation(...))` | 11 | 130 |
| whole `DefaultGenome.forward` (`max_nodes=50`, `max_conns=100`, `max_depth = 16` unrolled steps) | 621 | 2509 |

What keeps the count there: the id `cast` is skipped (the gene's `prepare` already casts), an `is_identity` option emits no `equal`/`select` pair, the neutral-padded window is built once per distinct neutral through `_Window` (`sum`/`mean`/`maxabs` share `select(mask, z, 0.0)`), `sum` and `mean` share one `reduce_sum` of that window (`_Window.total()`), `mean` and the `has_valid` guard share one live-edge count, one 0-d `float32` `constant` is emitted per distinct literal (`_Window.scalar()`), the guard itself is skipped when every option is `zero_on_empty`, and no separate zeroed accumulator is emitted (the first `select` carries the `else` value).
Per level this node owns ~29 % of the `DefaultGenome` forward-pass op count (11 of the ~38 ops a one-option level costs); the rest is the genome's gather/combine work.
The dispatch is at its op floor: three ops per non-identity option (`constant` index, `equal`, `select`) is the minimum for an id that is gene data, and a one-option list cannot drop the `select` because an out-of-range id must still fall back to identity / `0.0`.

## Measured cost (median of 15 `evoxc.run` calls, build cached, single-threaded)
One level kernel (`apply_activation ∘ apply_aggregation`) over `(P, 50, 8)` messages, `agg_ids`/`act_ids` int32, one option each: 0.32 ms / 1.29 ms / 2.42 ms for P = 100 / 500 / 1000; with both canonical option lists 3.10 ms / 12.9 ms / 23.7 ms (25.1 ms at P = 1000 before the shared-reduce pass, i.e. one `(P, N, K)` reduce saved).
`sum` + `mean` in one option list: 0.45 / 1.89 / 3.62 ms (from 0.54 / 2.37 / 4.43), the whole difference being the shared `total()`.
The shared literals earn op count, not wall time: 12 duplicate `constant` ops inside a 60-op select graph cost 0.059 ms in total.
Build (cached per shape by the Pipeline, not per generation): 5–6 ms for the one-option level graph, ~45 ms for the canonical one.
End-to-end `Pipeline.step` (XOR, `DefaultGenome`, `NEAT`, `max_nodes=50`, `max_conns=100`, `species_size=20`, pop 1000, build excluded): ≈713 ms per generation, with individual generations scattering ±30 % on a shared machine; the whole level loop is ≈3–5 % of that at P=500–1000 (a level is 11 of the forward pass's ops but the forward pass is not the generation).
The rest of a generation is schedule derivation, mutation/crossover, speciation and selection — not this node.

Remaining hot spots inside this node, in order:
1. `select(mask, z, 0.0)` (≈1.37 ms) plus `reduce_sum` (≈0.87 ms) at P = 1000 — the mandatory neutral-padded window of a single aggregation level, ≈90 % of its 2.42 ms run time. Dropping the padding would require the *genome* to guarantee that dead slots are exactly `0.0` (it currently only NaN-sanitizes them), i.e. a genome-side contract change plus a `zero_on_empty`-style opt-in, not a change here.
2. The `equal`/`select` cascade of a multi-option list (3 ops per option over `(P, N)`): unavoidable while the id is gene data rather than a shape.
3. `maxabs` costs 0.85 ms more per level than `sum` at P = 1000 (argmax + 7 `reshape`s + `flat_gather`); `flat_gather` additionally emits a `(P, 1)` int32 offsets constant plus ~4 index-range guards per call even though an `argmax` result is provably in range (see `common/tools.py`, not this node).
4. `constant` ops are distinct per literal and per dispatch: evoxc has no graph-level constant cache, so each call re-materialises its own literals (the shared-scalar pass only removes the duplicates *within* one call); 3 of the 11 one-option ops are such constants.

## Shapes, dtypes, sentinels
All kernels take and return `float32`; ids arrive as integral-valued `float32` gene columns (cast to `int32`) or as an already-`int32` id column (used as is); an already-`int32` column reaches the dispatcher from `gene/node/base.prepare`, which casts once per forward pass.
`NaN` is the only float sentinel (absent node row, uncomputed node value, unmatched attribute slot); `I_INF = int32 max` is the only integer sentinel and never appears in this package — the mask plus `(P, N, K)` padding already encode absence on the aggregation axis.
No bitcast exists in evoxc, which changes nothing here: activations and aggregations are numerics, not payload bit-twiddling.

## Sympy / interpretability contract (host-only)
`ACT` / `AGG` pair every option with an `sp.Function` subclass in the same order; `FunctionManager.__init__` validates the pairing by class name (`SympyScaledSigmoid` → `scaled_sigmoid`) and raises `ValueError` on a mismatch, so an `inv` / `SympyIdentity` mix-up cannot reach the codegen path.
Each sympy class is a constructor-time rewriter (`SympySigmoid(z)` returns `1/(1+exp(-z))`), so generated formulas are plain sympy expressions; `SympyClip` is the exception and defines `numerical_eval`, exposed through `sympy_module()` for `sp.lambdify`.
`sp.lambdify(..., modules=["numpy", AGG.sympy_module(), ACT.sympy_module()])` is the sanctioned codegen path; `sympy_module(backend)` accepts `"numpy"` only and raises `ValueError` for anything else, because there is no JAX backend any more and sympy codegen is host-only.
`sympy_module()` emits a `partial(cls.numerical_eval, backend=numpy)` for every counterpart **and** every `extra_sympy_classes` entry that defines `numerical_eval` — `SympyClip` is such a helper (`ACT_SYMPY_HELPERS`), because `SympyExp` builds it while not being an option itself; the binding is by keyword, since `numerical_eval(val, min_val, max_val, backend=np)` would otherwise receive numpy as `val`.
The codegen path is exercised, not aspirational: every activation and aggregation round-trips through `cls(expr)` → `sp.lambdify` → numeric evaluation (a module-level smoke test in `test/unit/` should keep that true).
Shared numeric constants (`SCALE`, `LELU_SLACK`, `INV_EPS`, `LOG_EPS`, `EXP_CLIP`) live in `act.py` and are imported by `act_sympy.py`, so the two representations cannot drift.

## Design Decisions
1. **`inv` pairs with `SympyInv`** (`1 / clamped(z)`): the numeric kernel is `1/z`, so the interpretability output states the executed function; pairing `inv` with an identity would make generated formulas disagree with the compiled graph and is rejected by the pairing check.
2. **`clip` is not a registry option**: a 13th sympy-only name would make `ACT`'s two tables disagree in length and shift no numeric behaviour; `clip` stays internal to `SympyExp` (`SympyClip`, exported as a codegen helper) and the ACT option list is exactly the 12 names above, so sampled activation ids stay stable.
3. **Dispatch is a `select` cascade, not a branch**: evoxc has no `switch` and `cond` demands a 0-d predicate, so every option is evaluated and selected per node.
4. **Dispatch overhead is minimised by static flags, not by inspecting kernel identity**: `is_identity` / `window_body` / `zero_on_empty` let a genome's custom option list opt in, and their defaults keep the fully general path.
5. **`_Window` is the single owner of the masked reduction inputs**: `padded(neutral)` and `count()` are the only places this package builds `select(mask, z, neutral)` and `reduce_sum(cast(mask, float32))`, so a kernel body and the dispatcher cannot drift apart.
6. **A gather/`stack`-based dispatcher was rejected**: stacking every option result and selecting with one `flat_gather` trades 2 ops per option for a `(len, P, N)` stack, which is *more* wall time on the numpy backend (~2.4 MB of traffic for 12 options at P = 1000) and needs clamp/mask plumbing for the `-1` and out-of-range rules.
7. **Aggregations are rank-3 and last-axis reducing**: one shape class, one reduce axis, testable against a numpy oracle; HyperNEAT's substrate shape adaptation belongs to that node.
8. **`maxabs` stays signed** (argmax + `flat_gather`) rather than the cheaper `reduce_max(abs(z))`, because the pre-rewrite kernel returned the signed value and parity tests compare against it.
9. **The manager is immutable** (no `add_func` / `update_sympy`): a gene id indexes the *user's* option list, so custom options need no global registration; a genome wanting sympy support for a custom option constructs its own `FunctionManager`.
10. **Names are registry data** (`ACT.name_of(fn)`, `ACT.index2name[id]`), with no module-level `get_func_name` helper: `to_dict` / `repr` ask the manager, which knows the canonical spelling of an option.
11. **Registry validation raises `ValueError`, not `ConfigError`**: a mismatched table is a programmer error in this package, whereas `common.errors.ConfigError` is reserved for user configuration raised by the layer that owns the config.

## Known Divergences from the pre-rewrite library
`SympyExp` clips its argument to `[-EXP_CLIP, EXP_CLIP] = [-10, 10]` while `exp` does not — generated formulas over/under-estimate strongly positive/negative pre-activations (kept for formula stability, as before).
Out-of-range activation/aggregation ids fall back to identity / `0.0` instead of clamping to the last option (`lax.switch` behaviour); mutation only samples valid ids, so this is reachable only through hand-written genes or a stale checkpoint replayed against a shortened option list.
`mean` is `0.0` instead of `NaN` for an all-`False` mask; indistinguishable through the dispatcher, different only when the kernel is called standalone.
`sympy_module` accepts only `"numpy"` (a `"jax"` argument raises `ValueError`), and `obtain_sympy` accepts an option id as well as a name or callable.
`sympy_module()` binds numpy as the *keyword* `backend` (the pre-rewrite `partial(fn, jnp_or_np)` bound it to `val`, so `lambdify`d `clip` evaluations raised `AttributeError`); and `SympyMaxabs` renders as an explicit `Piecewise` over `Eq(Abs(t_i), max_j Abs(t_j))`, because `sp.Max(*z, key=sp.Abs)` is rejected by current sympy — the signed term and its first-maximum tie-break are unchanged.
The aggregation reduce axis is the last axis of `(P, N, K)` (the pre-rewrite library reduced a per-node axis 0); numerically identical for the same edge order, but HyperNEAT substrate code must reshape its contributions to `(..., K)` instead of calling `AGG.sum(inputs)` on a raw input vector.

## Constraints
- No `jax`, `torch`, `numpy.random` anywhere in this package; graph code uses `evoxc` ops only, host code may use `numpy` / `sympy`.
- Nothing here imports `genome`, `algorithm`, `problem`, `pipeline`, `benchmarks` or `test`; the only intra-package dependency is `..tools.flat_gather` (used by `maxabs`).
- Files ≤ ~1000 lines; each file here is a single responsibility (kernels+dispatcher per side, sympy per side, registry bookkeeping).
- Do not reorder `ACT_FUNCS` / `AGG_FUNCS` or their sympy tuples; append only.
- Do not change a kernel's arithmetic: the formulas are compared bit for bit against the pre-rewrite library, so `evoxc.sigmoid`-style fused ops are off limits.
- `sum`, `max`, `min`, `abs` shadow builtins inside `agg.py` / `act.py` — that is intentional (the registry names must equal the kernel names); never use the builtin of the same name in those modules.
- Errors: broken tables raise `ValueError` at import time (fail fast); unknown option names/ids raise `ValueError` / `IndexError`; nothing is swallowed.

## Consumers (the API sibling nodes may rely on)
`from tensorneat.common import ACT, AGG, apply_activation, apply_aggregation` is the only public surface of this package (`__all__` is exactly those five names, `FunctionManager` included).
Node genes: `apply_aggregation(z, mask, agg_ids, self.aggregation_options)` then `apply_activation(z, act_ids, self.activation_options)`; the option lists are plain tuples of kernels, normalised once in the gene constructor, and an id column that is already `int32` (as produced by `gene/node/base.prepare`) reaches the dispatchers without a traced cast.
Configs reference an option by name through attribute access (`AGG.sum`, `ACT.sigmoid`) and resolve ids through `ACT.name2index` / `ACT.get_evoxc_func`; `ACT.random_index(key)` samples an id in-graph from a stateless evoxc key.
Interpretability: `ACT.obtain_sympy(name_or_id_or_func)` / `AGG.obtain_sympy(...)` return the sympy class for a `to_dict`-style name.
The parent `common/__init__.py` imports exactly `ACT, AGG, apply_activation, apply_aggregation` (plus optionally `FunctionManager`); there is no `get_func_name` symbol in this package, so `repr` / `to_dict` code takes names from `ACT.name_of` / `AGG.name_of`.

## Test Hooks
`test/unit/` must be able to test every kernel on the numpy backend with no optional dependency: each activation against a numpy reference on a `(P, N)` tensor, each aggregation against a numpy oracle (including an all-`False` mask column and a `maxabs` tie), and the dispatchers against a per-element Python loop over the option list.
Host-side tests need no backend at all: registry order/names, pairing validation (feed a deliberately wrong sympy tuple and expect `ValueError`), `name_of` / `get_sympy_func` / `obtain_sympy`, and `sympy_module()` shape.
Falsifiable invariants the flags must keep: a list holding only `zero_on_empty` kernels and a list holding `product`/`max`/`min` give the same `0.0` on an all-`False` column through `apply_aggregation`; `identity.is_identity` changes no output but removes two ops; `agg.sum` standalone still returns `0.0`, `product` `1.0`, `max` `-inf`, `min` `+inf`.
`test/parity/` compares each kernel and the dispatchers against `benchmarks/legacy_jax/tensorneat_legacy/common/functions/` with the same inputs and option lists — the legacy `maxabs` sign, the `mean` guard and the `-1` identity sentinel are the three cases most likely to regress.

## Verification Status
Every kernel and dispatcher is checked on the numpy reference backend against `test/support/functions_ref.py` on random `(P, N, K)` data that includes `NaN` in masked-out slots, a live `NaN`, an all-`False` and an all-`True` column, `maxabs` sign/tie cases, `mean` edge counts, `-1` / out-of-range ids, integral `float32` id columns, `int64` id columns, custom option lists, and the ``-1``-only row block (bitwise identity, `NaN` included).
The optimised dispatchers were additionally diffed bit for bit against the previous implementation on the same inputs: identical outputs for the one-option and the canonical option lists, identical `NaN` patterns, identical whole-`DefaultGenome.forward` outputs for pop 100 (default and full option lists), and identical 5-generation `pipeline.fitness_history` at seed 42.
Environment facts the tests depend on: `maxabs` needs a working `..tools.flat_gather` (`(P*N, K)` rows indexed by a `(P*N,)` block), and `FunctionManager.random_index` needs the numpy backend's `random.randint` (compiler backends reject every random op).
The dispatchers take `act_fns` / `agg_fns` as trace-time static arguments, so a test that supplies a custom option tuple must bind it (``functools.partial(apply_activation, act_fns=...)``) instead of passing a tuple of callables as a traced argument.
