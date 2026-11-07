# tensorneat/common/functions — ACT/AGG option registries, their evoxc kernels and sympy twins

## Intent
Single source of truth for the activation and aggregation options a NEAT node gene may select: the evoxc kernels, the `select` cascades that stand in for `lax.switch`, the sympy counterparts, and the immutable `FunctionManager` pairing them in one fixed, parity-critical order.
`act.py` / `agg.py` are graph code (called only during an `@evoxc.defn` trace); `act_sympy.py` / `agg_sympy.py` / `manager.py` are host-only, and nothing here knows about genomes, populations or the Pipeline.
This node owns no `kernel_mode` / `distance_*` static: it emits the same graph on every backend.

## Module map (leaf; one responsibility per file, all far under ~1000 lines)
| File | Lines | Responsibility |
|---|---|---|
| `act.py` | 391 | 12 activations, their constants, `ACT_IDENTITY`, `ACT_FUNCS`, `apply_activation`. |
| `agg.py` | 760 | 6 aggregations, the `K`-plane folds (`_FOLD_MAX_WIDTH = 128`), the `_Window` memo, `AGG_FUNCS`, `apply_aggregation`. |
| `act_sympy.py` | 220 | sympy twins of `ACT_FUNCS`, `SympyClip`, `ACT_SYMPY_HELPERS`. |
| `agg_sympy.py` | 117 | sympy twins of `AGG_FUNCS`. |
| `manager.py` | 380 | `FunctionManager`: pairing, lookups, validation, `sympy_module`, `random_index`. |

`__init__.py` (41 lines) only wires `ACT` / `AGG` together and declares the five public names.

## Registry law
An option id is the **position in the gene's own option tuple**, never a global index: the tuple's callables are normalised once in the node-gene constructor, and `ACT_FUNCS` / `AGG_FUNCS` are only the canonical order a genome gets when it passes the registry itself (`ACT.evoxc_fns`) as its option list.
A custom callable is registered nowhere — pass it in the tuple (activation `z -> z`, aggregation `(z, mask) -> (P, N)`) and it dispatches on the general path because it carries none of the static flags; a private `FunctionManager(kernels, sympy_classes, extra_helpers)` adds sympy support, validating the pairing by class name (`SympyScaledSigmoid` → `scaled_sigmoid`) and raising `ValueError` on a length mismatch, a duplicate or non-callable name or a mismatched counterpart (`None` is allowed; `get_sympy_func` then raises), and it is immutable (no `add_func` / `update_sympy`).
An option's registry name is its kernel `__name__` (hence the builtin shadowing); `names` *is* `index2name`, `name2index` its inverse, `name_of` maps back, and `manager.<name>` resolves through `__getattr__`, where an unknown name raises `AttributeError`.
**Append only**: an id encodes a position, so reordering or deleting shifts sampled ids and breaks checkpoints and parity; a host lookup outside `[0, len)` raises `IndexError`, and `obtain_sympy` accepts an id, a name or a kernel.
Ids are drawn by the gene layer (`genome/gene/blocks.py`: `evoxc.random.randint` over `len(spec.options)`, `options[0]` when one is configured), never by `FunctionManager.random_index` (a rank-0 `int32` helper with test callers only).

### ACT — canonical order (12)
Ids 0–11 are `scaled_sigmoid, sigmoid, scaled_tanh, tanh, sin, relu, lelu, identity, inv, log, exp, abs`, built from the module constants `SCALE = 3.0` (the two `scaled_*`), `LELU_SLACK = 0.005` (`lelu` = `select(z>0, z, 0.005*z)`), `INV_EPS = LOG_EPS = 1e-7` (`inv` = `1/clamped(z)`, `log` = `log(maximum(z, 1e-7))`) and an unclipped `exp` (overflows to `inf` above ~88); `identity` is `z`, the target of `ACT_IDENTITY = -1`, and carries `is_identity = True`.
The expressions and their evaluation order are bit-identical to the frozen legacy library, so `evoxc.sigmoid` / `evoxc.relu` and friends are **not** used; `act_sympy.py` imports the same constants, so the two representations cannot drift.
Activations are shape-polymorphic and elementwise — canonical call shape `(P, N)`, and the same kernels serve `(P, out_dim)` output transforms and HyperNEAT substrate nodes; `NaN` in → `NaN` out, no masking, and `ACT_IDENTITY` is a cross-node constant (written into output rows by `gene/node/base.py`, read by `BiasNode` / `DefaultNode`).

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
Three attributes ride on a kernel and are read at trace time; a custom kernel lacks them, and the defaults (`False` / absent) always mean "emit the general path", never "skip work": `identity.is_identity` (selecting it returns the incoming value, so `apply_activation` emits no `equal`/`select` pair), `agg.<kernel>.window_body` (the `(window) -> (P, N)` body over the shared `_Window`; absent means the dispatcher calls `fn(z, mask)`), and `agg.<kernel>.zero_on_empty` (`True` iff the kernel already returns exactly `0.0` for an all-`False` column: `sum`, `mean`, `maxabs`; `False` for `product` / `max` / `min` / any unannotated kernel, which therefore needs the `has_valid` guard).
The first flag is read only by `apply_activation`, the other two only by `apply_aggregation`; the node gene keeps its own stricter, identity-compared `BaseNode.self_neutral_aggregations = (AGG.sum, AGG.mean)` (so a wrapper keeps the guard), and no dispatcher branches on a kernel's identity.

## Aggregation over the `K` axis
`z` and `mask` are `(P, N, K)` — population, node slot, static in-degree bound — the combine axis is always last and the result `(P, N) float32`; `mask` comes from the genome (level matches ∧ edge matched ∧ source not `NaN`) and, not `isnan(z)`, decides which slots count, so a genuinely `NaN` contribution still poisons its node as in the frozen legacy library.
`_Window` shares the dispatcher's inputs: `padded(neutral)` memoises `select(mask, z, neutral)` per distinct neutral, `total()` folds that zero-padded window once, `count()` folds `cast(mask, float32)` once, and `scalar(value)` builds one 0-d float32 constant per literal (evoxc has no constant CSE).
`maxabs` is **signed** and legacy-exact: masked-out slots become `0.0`, `evoxc.argmax(abs)` takes the first maximum and one `common.tools.flat_gather` reads the **zeroed** tensor at it (reducing `abs` alone would return a magnitude; reading `z` would let an all-`False` column return a padding slot); `mean` divides by `maximum(count, 1.0)`, so a node with no live edge is `0.0`, never `NaN`, while `product` / `max` / `min` return their neutral there, which the dispatcher's `has_valid` guard overwrites.

### The `K` axis is folded, not reduced — up to `_FOLD_MAX_WIDTH` (128)
`K` is static, so for `K <= 128` every kernel replaces its `reduce_*`-over-`-1` with `K` `(P, N, 1)` `evoxc.slice` views plus elementwise ops (`_planes` emits the slices, one closing `reshape` gives `(P, N)`), at most `K + 1` more ops than the reduction; a wider window returns the backend reduce (`reduce_sum` / `reduce_prod` / `reduce_max` / `reduce_min`), because numpy's `pairwise_sum` uses the 8-accumulator branch the fold reproduces only while `n <= PW_BLOCKSIZE (128)`. The choice is a trace-time `if` on the static width, never `evoxc.cond`, and the shipped widths (`K = 4` / `8`) always fold.
**The association order is the reference backend's, on purpose** (`float32` addition is not associative and `test/unit/` + `test/parity/` compare bit for bit): `_sum_last` accumulates from `+0.0` for `K < 8` (load-bearing for an all-`-0.0` column) and reproduces `pairwise_sum`'s eight partial accumulators plus its `K % 8` tail for `8 <= K <= 128`; `_fold_last` is a plain left fold from the neutral, exactly how the reference backend reduces `product` / `max` / `min` (`maximum(-inf, a)` / `minimum(+inf, a)` are `a`, `NaN` included, so the padding neutral and the fold identity share one constant).
Bit-identity with `np.sum` / `np.prod` / `np.max` / `np.min` (`view(np.uint32)`) is verified across `K = 1`–`300`, including the 127/128/129 boundary, on `±0.0`, `NaN` and `±inf` data, with the fold-vs-reduce dispatch asserted at every width tested.

## Dispatch contract (no `lax.switch` in evoxc)
`apply_activation(z, act_ids, act_fns=ACT_FUNCS)` — `z (...,) float32`, `act_ids (...,)` integral float32 or `int32` broadcastable against `z`, `act_fns` the gene's static tuple; returns `(...,) float32` as a `select` cascade starting from `z` (the identity branch, also the `-1` / out-of-range branch) that folds `select(ids == i, act_fns[i](z), acc)` for every option **except** an `is_identity` one, using an already-`int32` id column as is.
`apply_aggregation(z, mask, agg_ids, agg_fns=AGG_FUNCS)` — `z (P, N, K) float32`, `mask (P, N, K) bool`, `agg_ids (P, N)` integral; returns `(P, N) float32`, built as one `_Window(z, mask)` folded with `select(ids == i, body_i(window), acc)` per option, where the **first** `select` carries the `else` value `0.0` (the legacy zeroed accumulator for an out-of-range id, so no zeros op is emitted) and an empty option list returns `zeros(z.shape[:-1])`.
The trailing `has_valid` select (`select(count > 0, acc, 0.0)`) is emitted only when some option is not `zero_on_empty`, i.e. when a padded `product` / `max` / `min` would leak `1.0` / `-inf` / `+inf` into a node with no computable incoming edge.
`act_fns` / `agg_fns` must be **trace-time static** sequences of callables with the documented signature, which is why a test with a custom tuple binds it (`functools.partial`).

## Dtypes, NaN and sentinels
Kernels take and return `float32`, ids are integral-valued `float32` gene columns or already `int32`, and `NaN` is the only sentinel (absent node row, uncomputed value, unmatched attribute slot), propagating through every activation; `I_INF` never appears, because the mask plus `(P, N, K)` padding encode absence on the aggregation axis.

## Cost
The lever is the interpreter's per-op overhead: ≈19–20 µs per dispatched op on the numpy reference interpreter at population-sized tensors (the package-wide law in `../CONTEXT.md`), so a reduction on `(P, N, K)` is worth `K` ops on `(P, N, 1)` only when it does not buy a wider pass — the case for the fold: one reduce-over-`-1` at `(P, N, K) = (2000, 50, 4)` measures ≈1.4 ms per level against ≈0.4 ms for the `K`-plane fold of the same window, as the per-op law predicts for `K + 2` ops.
The mandatory neutral-padded `select(mask, z, neutral)` is the single most expensive op of a level (≈1.0 ms of a 2.0 ms level at that shape) and cannot become a masked multiply: a dead slot may hold `NaN` or `±inf`, whose product with `0` is `NaN` (`../CONTEXT.md` records the rule).
Everything else in a dispatch is 3 ops per non-identity option over `(P, N)` (`constant` + `equal` + `select`), the floor for an id that is gene data rather than a shape; `maxabs` is the one expensive option (argmax + 7 `reshape`s + a `flat_gather` whose ~4 range guards are redundant for an `argmax` result), and constants cost ~1 µs each, so `_Window.scalar` stops one dispatch emitting the same `0.0` four times.
Rejected levers (measured; do not re-derive): a halving tree, which matches numpy's order only at `K = 8`; per-plane `select` folded with the adds (1.606 vs 1.609 ms, +10 ops, and it takes `maxabs` off the shared window); the masked multiply above; a gather/`stack` dispatcher, which trades 2 ops per option for a `(len, P, N)` stack (≈2.4 MB at `P = 1000`) plus clamp plumbing for the `-1` rule.

## Portability
Every op this package emits is `v1` in `evoxc.backends.stablehlo.ops.status()` (elementwise ops, `slice`, `reshape`, `select`, `equal`, `greater`, `cast`, `argmax`, `gather`, and `reduce_*` on the wide-window fallback); `erf` / `gelu` are deferred there and no ACT option is either, so nothing here emits an unsigned dtype, a `scatter`, `cumsum`, `flip`, `tril`/`triu` or a `while`.
Recorded rule from evoxc's op table rather than a fresh export from this node; `test/backends` is the only proof of portability.

## Sympy / interpretability contract (host-only)
`ACT` / `AGG` pair every option with an `sp.Function` subclass in the same order, validated by class name at import; each class is a constructor-time rewriter (`SympySigmoid(z)` returns `1/(1+exp(-z))`), and `SympyClip` is the exception — no closed form, it defines `numerical_eval` for `sp.lambdify`.
`sp.lambdify(..., modules=["numpy", AGG.sympy_module(), ACT.sympy_module()])` is the sanctioned path (`genome/default.py` builds exactly that list); `sympy_module(backend)` accepts `"numpy"` only and binds numpy as the *keyword* `backend` (`numerical_eval(val, min_val, max_val, backend=np)` would otherwise receive numpy as `val`), so `ACT.sympy_module()` holds `SympyClip` alone and `AGG.sympy_module()` is empty.
Aggregation counterparts take the *valid* incoming terms as an iterable (no mask), so `sympy_func` must pass `[0]` (or the node's bias) for a node with no incoming edge; `inv` pairs with `SympyInv` (`1/clamped(z)`) so the formula states the executed function, and `clip` is **not** an ACT option — a 13th sympy-only name would make `ACT`'s two tables disagree in length without changing numerics, so `SympyClip` stays internal to `SympyExp` and sampled ids stay stable.

## Divergences from the frozen legacy implementation (parity-relevant)
`SympyExp` clips its argument to `[-10, 10]` (`EXP_CLIP`) while `exp` does not, so generated formulas over/under-estimate strongly positive/negative pre-activations (kept for formula stability).
Out-of-range ids fall back to identity / `0.0` instead of clamping to the last option (reachable only through hand-written genes or a stale checkpoint), `mean` is `0.0` instead of `NaN` for an all-`False` mask (different only standalone), `sympy_module` accepts only `"numpy"` and binds it by keyword, and the combine axis is the last axis of `(P, N, K)` where the legacy library reduced a per-node axis 0 — identical for the same edge order, but a consumer holding a raw input vector must reshape to `(..., K)` rather than calling `AGG.sum(inputs)`.

## Known Issues
`SympyMaxabs` returns a `Piecewise` whose conditions are `Eq(Abs(t_i), Max(Abs(...)))` per node, so `Genome.sympy_func`'s closed-form substitution multiplies those branch sets layer by layer: ≈27 s at 2 hidden layers, >110 s at 3 and 0.02 s at 1 on a `maxabs` genome (interpretability only; no generation, fitness or trace path is affected). The fix is a cheaper signed-argmax formulation that keeps the first-maximum tie-break without a per-node `Piecewise` over `Max`.

## Constraints
- No `jax`, `torch`, `numpy.random` anywhere in this package; graph code uses `evoxc` ops only, host code may use `numpy` / `sympy`.
- Nothing here imports `genome`, `algorithm`, `problem`, `pipeline`, `benchmarks` or `test`; the only intra-package dependency is `..tools.flat_gather` (used by `maxabs`).
- Never change a kernel's arithmetic or reorder the tuples — both are compared bit for bit against the frozen legacy library, so `evoxc.sigmoid`-style fused ops are off limits.
- Never use the builtin that `sum` / `max` / `min` / `abs` shadow inside `agg.py` / `act.py`; kernel names must equal the registry names.
- Errors: broken tables raise `ValueError` at import time (fail fast); unknown option names/ids raise `ValueError` / `IndexError`; nothing is swallowed.

## Consumers
`from tensorneat.common import ACT, AGG, FunctionManager, apply_activation, apply_aggregation` is the public surface (`common/__init__.py` re-exports exactly those in `__all__`); `ACT_IDENTITY`, `ACT_FUNCS`, `AGG_FUNCS`, `ACT_SYMPY_FUNCS` and `ACT_SYMPY_HELPERS` are importable module symbols (gene nodes import `ACT_IDENTITY` from `act.py`) without being public surface.
The node-gene forward path calls the **registry kernels directly**, not the dispatchers: `BaseNode.aggregate` runs `[fn(messages, valid_mask) for fn in self.aggregation_options]` and `BaseNode.forward` likewise for activations, with `select_cascade` picking the row's option and the id cast plus `ids == index` masks hoisted into `prepare` — one op fewer per option per level, at the price that each kernel builds its own `_Window`, so the memo is a dispatcher-only win.
`apply_activation` therefore has exactly one `src/` caller, `genome/grad.py` (the differentiable path), and `apply_aggregation` has none; both stay public API and are what `test/unit` pins the semantics against, while `random_index`, `get_evoxc_func`, `get_sympy_func` and `index2name` have test callers only.

## Test contract
`test/unit/common/test_functions.py` (24 test functions) pins the literal tables (12 / 6) and their sympy twins, the name↔id↔kernel↔sympy pairing plus its `ValueError` / `IndexError` / `AttributeError` paths, `random_index` staying in `[0, len)` over seeded keys, and both dispatchers against a per-option Python loop and `test/support/functions_ref.py` on the documented edge cases (`NaN` in masked-out slots, a live `NaN`, all-`False` / all-`True` columns, `maxabs` sign/tie, `mean` edge counts, `-1` / out-of-range ids, int64 id columns, custom lists).
Falsifiable invariants: `identity.is_identity` changes no output but removes two ops; a `zero_on_empty`-only list and a list holding `product` / `max` / `min` both give `0.0` on an all-`False` column; each aggregation standalone returns its neutral (`sum` `0.0`, `product` `1.0`, `max` `-inf`, `min` `+inf`); `maxabs` needs a working `..tools.flat_gather`.
`test/parity/` compares every kernel and both dispatchers against `benchmarks/legacy_jax/tensorneat_legacy/common/functions/` on the same inputs and option lists — the legacy `maxabs` sign, the `mean` guard and the `-1` identity sentinel are the three cases most likely to regress.

## Routing Table
None — this directory is a leaf; `functions/` has no child nodes.
