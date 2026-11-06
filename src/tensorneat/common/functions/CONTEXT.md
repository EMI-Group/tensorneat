# tensorneat/common/functions — ACT/AGG option registries, their evoxc kernels and sympy twins

## Intent
Single source of truth for the activation and aggregation options a NEAT node gene may select: the evoxc kernels, the per-node `select` cascades that stand in for `lax.switch`, the sympy counterparts used for interpretability, and the immutable `FunctionManager` pairing them in one fixed, parity-critical order.
`act.py` / `agg.py` are graph code (called only during an `@evoxc.defn` trace); `act_sympy.py` / `agg_sympy.py` / `manager.py` are host-only.
Nothing here knows about genomes, populations or the Pipeline, and this node owns no `kernel_mode` / `distance_*` formulation static: it emits the same graph on every backend.

## Module map (no child directories; every file is one responsibility and none nears the ~1000-line rule)
| File | Lines | Responsibility |
|---|---|---|
| `act.py` | 378 | 12 elementwise activations, the constants shared with the sympy twins (`SCALE`, `LELU_SLACK`, `INV_EPS`, `LOG_EPS`, `EXP_CLIP`), `ACT_IDENTITY = -1`, `ACT_FUNCS`, `apply_activation`. |
| `agg.py` | 720 | 6 masked aggregations, the `K`-plane folds (`_planes` / `_sum_last` / `_fold_last`, bounded by `_FOLD_MAX_WIDTH = 128`), the `_Window` memo, `AGG_FUNCS`, `apply_aggregation`. |
| `act_sympy.py` | 221 | `sp.Function` counterparts of `ACT_FUNCS`, `SympyClip`, `ACT_SYMPY_FUNCS`, `ACT_SYMPY_HELPERS`. |
| `agg_sympy.py` | 118 | `sp.Function` counterparts of `AGG_FUNCS`, `AGG_SYMPY_FUNCS`. |
| `manager.py` | 381 | `FunctionManager`: ordered pairing, name/id lookups, pairing validation, `sympy_module`, `random_index`. |
| `__init__.py` | 42 | Wiring: builds `ACT` / `AGG`; `__all__` is exactly `ACT`, `AGG`, `FunctionManager`, `apply_activation`, `apply_aggregation`, while the canonical tuples stay importable but unexported. |

## Registry law
An option id is the **position in the gene's own option tuple**, never a global index: `activation_options` / `aggregation_options` are plain callables normalised once in the node-gene constructor, and `ACT_FUNCS` / `AGG_FUNCS` are only the canonical order a genome gets when it passes the registry itself (`ACT.evoxc_fns`) as its option list.
A custom callable is therefore registered nowhere: pass it in the option tuple (activation `z -> z`, aggregation `(z, mask) -> (P, N)`) and it dispatches on the general path because it carries none of the static flags; for sympy support, build a private `FunctionManager(kernels, sympy_classes, extra_helpers)`, which validates the pairing by class name (`SympyScaledSigmoid` → `scaled_sigmoid`) and raises `ValueError` for an empty table, a non-callable, a duplicate name, a length mismatch or a mismatched counterpart (a `None` counterpart is allowed and `get_sympy_func` then raises `ValueError`).
The manager is immutable (no `add_func` / `update_sympy`).
An option's registry name is its kernel `__name__` — which is why `sum` / `max` / `min` / `abs` shadow builtins inside `act.py` / `agg.py`; `names` *is* `index2name`, `name2index` is its inverse, and `manager.<name>` resolves through `__getattr__` (a `_`-prefixed or unknown name raises `AttributeError`, so copy/pickle stays safe), with `get_evoxc_func(index)` as the escape hatch when a real attribute shadows an option.
Order is part of the gene encoding: **append only** — reordering or deleting changes sampled ids and breaks checkpoints and parity.
Host lookups reject ids outside `[0, len)` with `IndexError` (never a negative wrap), and `obtain_sympy` accepts an id, a name or a registered kernel.
Ids are drawn in-graph by the gene layer (`genome/gene/blocks.py`: `evoxc.random.randint` over `len(spec.options)`, `options[0]` when only one is configured), never through `FunctionManager.random_index` — that helper answers a rank-0 `int32` in `[0, len)` (`randint(key, (1,), 0, len(self))` reshaped, because the numpy backend rejects a bare 0-d draw while building the output tensor) but has no `src/` caller.

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
| 7 | `identity` | `z` (target of the `ACT_IDENTITY = -1` gene sentinel; carries `is_identity = True`) |
| 8 | `inv` | `1 / select(z > 0, maximum(z, INV_EPS), minimum(z, -INV_EPS))`, `INV_EPS = 1e-7` |
| 9 | `log` | `log(maximum(z, LOG_EPS))`, `LOG_EPS = 1e-7` |
| 10 | `exp` | `exp(z)`, unclipped (overflows to `inf` above ~88) |
| 11 | `abs` | `abs(z)` |

Activations are shape-polymorphic and elementwise — canonical call shape `(P, N)`, and the same kernels serve `(P, out_dim)` output transforms and HyperNEAT substrate nodes; `NaN` in → `NaN` out, no masking.
The formulas are bit-identical to the frozen legacy library (same expression order), so `evoxc.sigmoid` / `evoxc.relu` and friends are **not** used — their rounding differs.
`ACT_IDENTITY` is written into output-node rows by `gene/node/base.py` and read by `BiasNode` / `DefaultNode`, so it is a cross-node constant, not a private literal.

### AGG — canonical order (6)
| id | name | combine over the last axis of `z (P, N, K)` | neutral padding | `zero_on_empty` |
|---|---|---|---|---|
| 0 | `sum` | `_sum_last` (numpy `pairwise_sum` order) | `0.0` | `True` |
| 1 | `product` | `_fold_last(…, multiply, 1.0)` | `1.0` | `False` |
| 2 | `max` | `_fold_last(…, maximum, -inf)` | `-inf` | `False` |
| 3 | `min` | `_fold_last(…, minimum, +inf)` | `+inf` | `False` |
| 4 | `maxabs` | signed value of largest `abs` (argmax + one flat gather) | `0.0` | `True` |
| 5 | `mean` | `_sum_last` of the padded window `/ maximum(count, 1.0)` | `0.0` | `True` |

Aggregations are rank-3 only (`(P, N, K)` → `(P, N)`); a consumer with another layout reshapes to `(..., K)` with an all-`True` mask, or combines on its own.

## Static kernel flags (the dispatchers' extension point)
Three attributes ride on a kernel and are read at trace time; a custom kernel lacks them, and the defaults (`False` / absent) always mean "emit the general path", never "skip work":
- `identity.is_identity = True` — selecting it returns the incoming value, so `apply_activation` emits no `equal`/`select` pair. Read only by `apply_activation`; the node gene's activation cascade evaluates every option (`identity` is free) and selects.
- `agg.<kernel>.window_body = _<kernel>_body` — the `(window) -> (P, N)` body over the shared `_Window`; absent means the dispatcher calls `fn(z, mask)`. Read only by `apply_aggregation`.
- `agg.<kernel>.zero_on_empty` — `True` iff the kernel already returns exactly `0.0` for an all-`False` column (`sum`, `mean`, `maxabs`); `False` (`product`, `max`, `min`, any unannotated custom kernel) means the option needs the `has_valid` guard. Read only by `apply_aggregation`; the node gene keeps its own stricter `BaseNode.self_neutral_aggregations = (AGG.sum, AGG.mean)` (identity comparison, so a wrapper keeps the guard — the conservative direction).
No dispatcher branches on a kernel's *identity* (`fn is sum`).

## Aggregation over the `K` axis
`z` and `mask` are `(P, N, K)` — population, node slot, static in-degree bound — the combine axis is always the last one and the result is `(P, N) float32`.
`mask` comes from the genome (level matches ∧ edge matched ∧ source not `NaN`) and, not `isnan(z)`, decides which slots count, so a genuinely `NaN` contribution still poisons its node as in the frozen legacy library.
`_Window` is the single owner of the shared inputs: `padded(neutral)` memoises one `select(mask, z, neutral)` per distinct neutral (`0.0` and `-0.0` share a key), `total()` folds the zero-padded window once, `count()` folds `cast(mask, float32)` once, and `scalar(value)` builds one 0-d float32 constant per distinct literal (evoxc has no constant CSE); reuse is exact — same expressions, no reassociation — so a dispatch mixing `sum`, `mean` and `maxabs` materialises the zero-padded window once and the live-edge count once.
`maxabs` is **signed** and legacy-exact: masked-out slots become `0.0`, `evoxc.argmax(abs)` takes the first maximum (int64 cast to int32) and one `common.tools.flat_gather` reads the **zeroed** tensor at it — reducing `abs` alone would return a magnitude and flip a negative-dominant node's sign, and reading `z` would let an all-`False` column return a padding slot.
`mean` divides by `maximum(count, 1.0)`, so a node with no live edge is `0.0`, never `NaN`; `product` / `max` / `min` return their neutral on an all-`False` column, which the dispatcher's `has_valid` guard overwrites.

### The `K` axis is folded, not reduced — up to `_FOLD_MAX_WIDTH` (128)
`K` is static, so for `K <= 128` every kernel replaces its `reduce_*`-over-`-1` with a chain of `(P, N, 1)` `evoxc.slice` views plus elementwise ops (`_planes` emits the `K` slices, one closing `reshape` gives `(P, N)`).
The bound is numpy's `PW_BLOCKSIZE`: `pairwise_sum` uses the 8-accumulator branch the fold reproduces only while `n <= 128` and halves recursively above it, so a wider window returns the backend reduce (`_sum_last` → `reduce_sum`, `_fold_last(..., reduce_op)` → `reduce_prod` / `reduce_max` / `reduce_min`) — numpy's own reduction, identical by construction and one op instead of thousands unrolled.
The choice is a trace-time `if` on the static width, never `evoxc.cond` and never a data-dependent branch; the shipped widths (`K = 4` / `8`) always fold, and a folded width costs `K` slices + `K - 1` combines + one `reshape` (plus the sum's `+0.0` start), at most `K + 1` more ops than the reduction it replaces, all on `(P, N, 1)` instead of `(P, N, K)`.
**The association order is the reference backend's, on purpose**, because `float32` addition is not associative and `test/unit/` + `test/parity/` compare bit for bit: `_sum_last` starts at `+0.0` for `K < 8`, and for `8 <= K <= 128` keeps eight partial accumulators (`r[j]` over `j, j+8, …`) combined as `((r0+r1)+(r2+r3)) + ((r4+r5)+(r6+r7))` plus a sequential `K % 8` tail; the `+0.0` start is load-bearing for exactly one case (an all-`-0.0` column sums to `+0.0`) and costs one op, so it is never elided.
`_fold_last` covers `product` / `max` / `min`, which the reference backend reduces left to right at every width, so a left fold from the neutral is the whole recipe (`1.0 * -0.0` is `-0.0`; `maximum(-inf, a)` / `minimum(+inf, a)` are `a`, `NaN` included — which is why padding neutral and fold identity share one constant).
Bit-identity with `np.sum` / `np.prod` / `np.max` / `np.min` (`view(np.uint32)`) is verified across `K = 1`–`300` including the 127/128/129 boundary on `±0.0`, `NaN` and `±inf` data, with the fold-vs-reduce dispatch asserted at every width tested.

## Dispatch contract (no `lax.switch` in evoxc)
`apply_activation(z, act_ids, act_fns=ACT_FUNCS)` — `z (...,) float32`, `act_ids (...,)` integral float32 or `int32` broadcastable against `z`, `act_fns` the gene's static tuple; returns `(...,) float32`.
It is a `select` cascade starting from `z` (the identity branch, also the `-1` / out-of-range branch) that folds `select(ids == i, act_fns[i](z), acc)` for every option **except** an `is_identity` one; an already-`int32` id column is used as is and cast only otherwise.
`apply_aggregation(z, mask, agg_ids, agg_fns=AGG_FUNCS)` — `z (P, N, K) float32`, `mask (P, N, K) bool`, `agg_ids (P, N)` integral, `agg_fns` the gene's static tuple; returns `(P, N) float32`.
It builds one `_Window(z, mask)` and folds `select(ids == i, body_i(window), acc)` per option; the **first** `select` carries the `else` value `0.0` (the legacy zeroed accumulator for an out-of-range id), so no separate zeros-accumulator op is emitted, and an empty option list returns `zeros(z.shape[:-1])`.
The trailing `has_valid` select (`select(count > 0, acc, 0.0)`) is emitted only when some option is not `zero_on_empty` — when a padded `product` / `max` / `min` would leak `1.0` / `-inf` / `+inf` into a node with no computable incoming edge; a list of `sum` / `mean` / `maxabs` needs none.
`act_fns` / `agg_fns` must be **trace-time static** sequences of callables with the documented signature, which is why a test with a custom tuple binds it (`functools.partial`) instead of passing it as a traced argument.
An id outside `[-1, len(options))` (activation) / `[0, len(options))` (aggregation) falls back to the cascade's initial value, `z` / `0.0`; the legacy `lax.switch` clamped to the last option, an unreachable divergence for well-formed genomes.

## Dtypes, NaN and sentinels
Kernels take and return `float32`; ids arrive as integral-valued `float32` gene columns (cast to `int32`) or already-`int32` (used as is).
`NaN` is the only float sentinel here (absent node row, uncomputed value, unmatched attribute slot) and propagates through every activation; `I_INF = int32 max` never appears in this package — the mask plus `(P, N, K)` padding already encode absence on the aggregation axis.
No bitcast exists in evoxc, which changes nothing here: these are numerics, not payload bit-twiddling.

## Cost
The lever is the interpreter's per-op overhead, not arithmetic: a dispatched op costs ≈19–20 µs on the numpy reference interpreter at population-sized tensors (the package-wide law in `../CONTEXT.md`), so a reduction on `(P, N, K)` is worth `K` ops on `(P, N, 1)` only when it does not buy a wider pass.
That is the whole case for the fold: one reduce-over-`-1` at `(P, N, K) = (2000, 50, 4)` measures ≈1.4 ms per level where the `K`-plane fold of the same window measures ≈0.4 ms — also what the per-op law predicts for `K + 2` ops.
The mandatory neutral-padded `select(mask, z, neutral)` is the single most expensive op of a level (≈1.0 ms of a 2.0 ms level at that shape) and cannot become a masked multiply: a *dead* slot may hold `NaN` (`NaN * 0` is `NaN`) or `±inf` (`inf * 0` is `NaN`), and `nan_to_num` would also rewrite a live `±inf`.
Everything else in a dispatch is 3 ops per non-identity option over `(P, N)` (index `constant` + `equal` + `select`), the floor for an id that is gene data rather than a shape.
`maxabs` is the one expensive option (argmax + 7 `reshape`s + a `flat_gather` that emits its `(P, 1)` int32 offset block plus ~4 range guards even though an `argmax` result is provably in range — see `common/tools.py`); constants are ~1 µs each, so `_Window.scalar` is what stops one dispatch emitting the same `0.0` four times.
Rejected levers (measured; do not re-derive): a halving tree instead of the per-plane fold (numpy's order only at `K = 8`, so a different sum at every other width, for ~10 ops); per-plane `select` folded with the adds (1.606 vs 1.609 ms, +10 ops, and it takes `maxabs` off the shared window); a masked multiply (~0.2 vs ~1.3 ms, rejected for the `NaN` / `±inf` reason); a gather/`stack` dispatcher (2 ops per option traded for a `(len, P, N)` stack, ≈2.4 MB at `P = 1000`, plus clamp plumbing for the `-1` rule).

## Portability
Every op this package emits is `v1` in `evoxc.backends.stablehlo.ops.status()`: `slice`, `reshape`, `add`, `multiply`, `divide`, `negate`, `maximum`, `minimum`, `abs`, `exp`, `log`, `sin`, `tanh`, `select`, `equal`, `greater`, `cast`, `argmax`, `gather`, and `reduce_*` on the wide-window fallback.
`erf` / `gelu` are deferred in that table and no ACT option is either; nothing here emits an unsigned dtype, a `scatter`, `cumsum`, `flip`, `tril`/`triu` or a `while`.
That is the recorded rule from evoxc's op table, not a fresh export from this node; `test/backends` is the only proof of portability.

## Sympy / interpretability contract (host-only)
`ACT` / `AGG` pair every option with an `sp.Function` subclass in the same order, validated by class name at import, so a mismatched table fails immediately; each class is a constructor-time rewriter (`SympySigmoid(z)` returns `1/(1+exp(-z))`), and `SympyClip` is the exception — no closed form, it defines `numerical_eval` for `sp.lambdify`.
`sp.lambdify(..., modules=["numpy", AGG.sympy_module(), ACT.sympy_module()])` is the sanctioned path (`genome/default.py` builds exactly that list), and `sympy_module(backend)` accepts `"numpy"` only because codegen is host-only.
`sympy_module()` emits `partial(cls.numerical_eval, backend=np)` for every counterpart **and** every `extra_sympy_classes` entry that defines `numerical_eval`, binding numpy as the *keyword* `backend` (`numerical_eval(val, min_val, max_val, backend=np)` would otherwise receive numpy as `val`), which is why `ACT.sympy_module()` holds `SympyClip` alone and `AGG.sympy_module()` is empty.
Shared numeric constants live in `act.py` and are imported by `act_sympy.py`, so the representations cannot drift; aggregation counterparts take the *valid* incoming terms as an iterable (no mask), so `sympy_func` must pass `[0]` (or the node's bias) for a node with no incoming edge.
Two deliberate pairings: `inv` ↔ `SympyInv` (`1 / clamped(z)`), so the formula states the executed function, and `clip` is **not** an ACT option — a 13th sympy-only name would make `ACT`'s two tables disagree in length without changing numerics, so `SympyClip` stays internal to `SympyExp` and sampled ids stay stable.

## Divergences from the frozen legacy implementation (parity-relevant)
`SympyExp` clips its argument to `[-10, 10]` (`EXP_CLIP`) while `exp` does not, so generated formulas over/under-estimate strongly positive/negative pre-activations (kept for formula stability).
Out-of-range ids fall back to identity / `0.0` instead of clamping to the last option (reachable only through hand-written genes or a stale checkpoint replayed against a shortened list), `mean` is `0.0` instead of `NaN` for an all-`False` mask (different only standalone), `sympy_module` accepts only `"numpy"` and binds it by keyword, and `SympyMaxabs` renders as an explicit `Piecewise` over `Eq(Abs(t_i), max_j Abs(t_j))` because `sp.Max(*z, key=sp.Abs)` is rejected by current sympy (signed term and first-maximum tie-break unchanged).
The combine axis is the last axis of `(P, N, K)` (the legacy library reduced a per-node axis 0): identical for the same edge order, but a consumer holding a raw input vector must reshape to `(..., K)` rather than calling `AGG.sum(inputs)`.

## Known Issues
`SympyMaxabs` returns a `Piecewise` whose conditions are `Eq(Abs(t_i), Max(Abs(...)))` per node, so `Genome.sympy_func`'s closed-form substitution multiplies those branch sets layer by layer: ≈27 s at 2 hidden layers, >110 s at 3 and 0.02 s at 1 on a `maxabs` genome (interpretability only; no generation, fitness or trace path is affected). The fix is a cheaper signed-argmax formulation that keeps the first-maximum tie-break without a per-node `Piecewise` over `Max`.
`apply_aggregation`'s docstring op counts (7 for a one-option `sum` list, 12 for `max`, 57 for the canonical list at `(100, 50, 8)`) predate the `K`-plane fold and understate a folded dispatch — a folded width alone is `K` slices + `K - 1` adds + one `reshape`; `act.py`'s counts (5 for one-option `tanh`, 10 for `sigmoid`, 74 for the canonical list) are unaffected.

## Constraints
- No `jax`, `torch`, `numpy.random` anywhere in this package; graph code uses `evoxc` ops only, host code may use `numpy` / `sympy`.
- Nothing here imports `genome`, `algorithm`, `problem`, `pipeline`, `benchmarks` or `test`; the only intra-package dependency is `..tools.flat_gather` (used by `maxabs`).
- Do not reorder `ACT_FUNCS` / `AGG_FUNCS` or their sympy tuples; append only, and do not change a kernel's arithmetic — the formulas are compared bit for bit against the frozen legacy library, so `evoxc.sigmoid`-style fused ops are off limits.
- Never use the builtin that `sum` / `max` / `min` / `abs` shadow inside `agg.py` / `act.py`; the kernel names must equal the registry names.
- Errors: broken tables raise `ValueError` at import time (fail fast); unknown option names/ids raise `ValueError` / `IndexError`; nothing is swallowed.

## Consumers
`from tensorneat.common import ACT, AGG, FunctionManager, apply_activation, apply_aggregation` is the public surface (`common/__init__.py` re-exports exactly those in `__all__`); `ACT_IDENTITY`, `ACT_FUNCS`, `AGG_FUNCS`, `ACT_SYMPY_FUNCS` and `ACT_SYMPY_HELPERS` are importable module symbols (gene nodes import `ACT_IDENTITY` from `act.py`) without being public surface.
The node-gene forward path calls the **registry kernels directly**, not the dispatchers: `BaseNode.aggregate` runs `[fn(messages, valid_mask) for fn in self.aggregation_options]` and `select_cascade` picks the row's option, `BaseNode.forward` does the same for activations, and the id cast plus the `ids == index` masks are hoisted into `prepare` (`gene/base.py::option_masks` / `select_cascade`) — one op fewer per option per level, with each kernel building its own `_Window`, so the memo is a dispatcher-only win.
`apply_activation` therefore has exactly one `src/` caller, `genome/grad.py` (the differentiable path), and `apply_aggregation` has none; both stay public API and are what `test/unit` pins the semantics against.
Configs name an option by attribute access (`AGG.sum`, `ACT.sigmoid`) or by passing the callable into an option tuple; `ACT.name_of` / `AGG.name_of` name one for `to_dict` / `repr`, and `obtain_sympy` resolves the sympy class for a name or id; `random_index`, `get_evoxc_func`, `get_sympy_func` and `index2name` have test callers only, while `sympy_module`, `obtain_sympy` and `name_of` have production callers.

## Test contract
`test/unit/common/test_functions.py` (24 test functions) pins the literal tables (12 / 6) and their sympy twins, the name↔id↔kernel↔sympy pairing plus its `ValueError` / `IndexError` / `AttributeError` paths, `random_index` staying in `[0, len)` over seeded keys, and both dispatchers against a per-option Python loop and `test/support/functions_ref.py`, on data with `NaN` in masked-out slots, a live `NaN`, all-`False` / all-`True` columns, `maxabs` sign/tie cases, `mean` edge counts, `-1` / out-of-range ids, integral-float32 and int64 id columns, custom option lists and a `-1`-only row block (bitwise identity).
Falsifiable invariants: a list of `zero_on_empty` kernels and a list holding `product` / `max` / `min` both give `0.0` on an all-`False` column through `apply_aggregation`; `identity.is_identity` changes no output but removes two ops; each aggregation standalone still returns `sum` `0.0`, `product` `1.0`, `max` `-inf`, `min` `+inf`, `maxabs` `0.0`, `mean` `0.0`; `maxabs` needs a working `..tools.flat_gather`, and `random_index` needs the numpy backend's `random.randint` (compiler backends reject every random op).
`test/parity/` compares every kernel and both dispatchers against `benchmarks/legacy_jax/tensorneat_legacy/common/functions/` on the same inputs and option lists — the legacy `maxabs` sign, the `mean` guard and the `-1` identity sentinel are the three cases most likely to regress — and the kernels were diffed bit for bit against their previous implementation (identical one-option and canonical outputs, `NaN` patterns, `DefaultGenome.forward` outputs at pop 100 for the default and the full option list, and a 5-generation `fitness_history` at seed 42).
