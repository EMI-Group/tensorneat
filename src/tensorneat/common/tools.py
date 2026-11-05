"""Batched array helpers shared by every evoxc kernel in TensorNEAT.

Dependency DAG position: bottom of ``tensorneat.common`` (with
:mod:`~tensorneat.common.errors`); it imports nothing from ``genome``,
``algorithm``, ``problem`` or ``pipeline``, and no ``jax``/``torch`` symbol and no
NumPy global-RNG call — the only host-side numpy uses are :func:`hash_rows` and
:func:`to_numpy`, which are explicitly marked host-only.

Everything in this module is a **graph helper**: it is called *inside* an
active ``@evoxc.defn`` trace, receives batch-first tensors (population on axis
0) and returns batch-first tensors of static shape. Nothing here branches on a
traced value, allocates host memory, or touches NumPy's global random module.

Why this module exists (validated evoxc facts that shape every signature):

* ``evoxc.gather(x, idx, axis)`` / ``evoxc.scatter(x, idx, updates, axis)``
  **replace** the indexed axis — they are not batched ``take_along_axis``.
  Per-row (batched) indexing is therefore only possible through fully
  flattened 1-D indices ``flat = row * row_len + col``; that is exactly what
  :func:`batch_offsets`, :func:`flat_gather`, :func:`masked_flat_gather`,
  :func:`flat_scatter` and :func:`masked_flat_scatter` provide.
* evoxc's numpy backend **raises** on out-of-range gather/scatter indices
  (JAX used to clamp), so callers must clamp *and* mask. A masked gather must
  never rely on clamping to ``0`` (slot 0 is a real node), and a masked
  scatter must route dropped writes to a trailing **dump slot**, never to
  index ``0``.
* There is no ``broadcast_to`` on ``evoxc`` (use ``evoxc.numpy.broadcast_to``),
  no ``searchsorted``, no ``choice``, no ``bitcast`` and no ``where`` (it is
  ``evoxc.select(pred, on_true, on_false)``).
* ``scatter`` is set-only (no add/max mode) and the reductions have no
  ``where=``/``initial=``; masking is done by padding with the neutral element.
* **One index block, one scatter.** A prepared index block (or any index
  expression, after IREE's CSE) must feed exactly one ``scatter``:
  ``iree-compile`` collapses the emitted rank-2 ``(n, 1)`` index vector to
  rank 1 for ``iree_linalg_ext.scatter`` through a materialization it cannot
  legalize while a second scatter still consumes the rank-2 value, and fails
  with ``failed to legalize unresolved materialization``. Two payloads that
  write the *same* cells (e.g. ``in_conn`` and ``in_src``) therefore go into
  one wider buffer and one scatter, never into two calls.
* ``argsort`` returns ``int64`` and requires ``stable=True`` wherever
  tie-breaking must be deterministic (group ranks).

Sentinel contract (shared with :mod:`~tensorneat.common.tensorization`):

* ``I_INF = 2147483647`` (``int32`` max) means "no index / unmatched /
  unavailable". It lives in **integer** arrays only, never mixed with ``NaN``
  in the same array.
* ``NaN`` in a **float** array means "absent row" (``isnan(x[:, 0])``) or
  "not computed yet".

Sentinel **constants are hoistable**: every sentinel is emitted as a rank-0
``constant`` op (never a bare Python scalar — it would be weakly promoted and
change the result dtype), so a helper called from inside a ``scan``/``cond``
body would pay one such op per iteration. :func:`fetch_first` and
:func:`argmin_with_mask` therefore also accept a pre-built constant, which is
used verbatim; build it once at trace time and close over it. Helpers that do
not offer that spelling yet are listed in ``common/CONTEXT.md``.

Public surface without a production caller: :func:`hash_rows`,
:func:`fetch_random`, :func:`mutate_float` and :func:`mutate_int` are exported
by ``common.__init__`` for tests and legacy parity only — nothing in ``src/``
calls them. The genome path has its own primitives: attribute mutation is
:func:`tensorneat.genome.gene.blocks.mutate_attrs` (its own per-spec draws,
not :func:`mutate_float`/:func:`mutate_int`), the ``(row, hit)`` pair in
``genome/operations/mutation/arrays.py`` replaces :func:`fetch_random`, and
gene hashing is the in-graph ``hash_int32`` (``genome/gene/base.py``), whose
host-only twin is :func:`hash_rows`. Each docstring states this individually.
"""

import numpy as np

import evoxc
import evoxc.numpy as enp

from .errors import ConfigError

#: ``int32`` max: "no index / unmatched / unavailable" in integer arrays.
#: Never store it in a float array (use ``NaN`` there) and never use it as a
#: real index — every consumer must test for it before indexing.
I_INF: int = int(np.iinfo(np.int32).max)

#: Exclusive upper bound of gene keys. Canonical payload keys are ``int32`` in
#: ``[0, KEY_MAX)``; the pre-rewrite/parity spelling keeps the same integer
#: values in ``float32``, where ``KEY_MAX = 2**24`` is the first integer the
#: format can no longer distinguish from its neighbour. Staying below it makes
#: key equality exact under either spelling.
KEY_MAX: int = 1 << 24


# --------------------------------------------------------------------------- #
# host-side validation helpers (raise before a graph is traced)
# --------------------------------------------------------------------------- #


def _positive_static(value, name):
    """Return ``value`` as a positive Python ``int``.

    The single validator for every trace-time static count in this package
    (population, node/connection bounds, in-degree window, key span, row
    lengths): a static is baked into the trace, so it is checked on the host,
    once, before any graph is built.

    Args:
        value: Static value to validate; an ``int`` or a NumPy integer scalar
            (``numpy.shape`` entries and ``array.shape`` members are the usual
            source). ``bool`` is refused — it is not a count.
        name (str): Argument name used in the error message.

    Returns:
        int: ``value`` as a Python ``int`` (a NumPy integer is normalised, so
        the trace never sees a NumPy scalar).

    Raises:
        tensorneat.common.errors.ConfigError: If ``value`` is not an
            ``int``/``numpy.integer`` (``bool`` excluded), or is not positive.
    """
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ConfigError(f"{name} must be a positive int, got {value!r}")
    value = int(value)
    if value <= 0:
        raise ConfigError(f"{name} must be a positive int, got {value!r}")
    return value


def _rate(name, value, *, allow_one=False):
    """Validate a mutation probability.

    Args:
        name (str): Argument name used in the error message.
        value: Rate to validate; any object ``float()`` accepts.
        allow_one (bool): Whether ``1.0`` is a legal value (``mutate_float``'s
            perturb band and ``mutate_int``'s replace band are "always") or not
            (a replacement band needs a non-empty keep band).

    Returns:
        float: The validated rate.

    Raises:
        tensorneat.common.errors.ConfigError: If the rate is not a real number,
            is outside ``[0, 1)`` (or ``[0, 1]`` when ``allow_one``), or is
            ``NaN``.
    """
    try:
        rate = float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{name} must be a real number, got {value!r}") from exc
    upper_ok = rate <= 1.0 if allow_one else rate < 1.0
    if not (0.0 <= rate and upper_ok):
        bound = "[0, 1]" if allow_one else "[0, 1)"
        raise ConfigError(f"{name} must lie in {bound}, got {rate}")
    return rate


def _scalar(value):
    """Normalize a scalar parameter that is handed to an ``evoxc.random`` op.

    ``evoxc.random.normal``/``uniform`` accept Python scalars and symbolic
    tensors but reject NumPy scalars, so a ``np.float32`` argument is unwrapped
    to its Python counterpart here.

    Args:
        value: Python scalar, NumPy scalar, or ``evoxc`` symbolic tensor.

    Returns:
        The value unchanged, except that a NumPy scalar becomes a Python scalar.
    """
    return value.item() if isinstance(value, np.generic) else value


def _scalar_constant(value, dtype):
    """Zero-dimensional graph constant of ``dtype`` holding ``value``.

    Used instead of a bare Python scalar wherever the *result dtype* of an
    ``evoxc.select``/comparison must stay pinned: a bare Python int is weakly
    promoted against an integer tensor via plain ``np.result_type`` (``int64``),
    which would silently widen an ``int32`` block.

    Args:
        value: Static Python scalar (int/float) baked into the trace.
        dtype: ``evoxc``/NumPy dtype of the constant.

    Returns:
        A rank-0 ``SymbolicTensor`` inside the active trace.
    """
    return enp.full((), value, dtype=dtype)


def _sentinel_constant(value, fallback, dtype):
    """Resolve a sentinel argument into a rank-0 graph constant, or pass it on.

    The single resolver behind the hoistable sentinels of :func:`fetch_first` and
    :func:`argmin_with_mask`. Every call emits one ``constant`` op, so a caller
    whose helper call sits inside a ``scan``/``cond`` body pays that op once per
    iteration; building the constant **once** at trace time (``enp.full((),
    value, dtype=...)``, the same expression used here) and passing it in makes
    the body close over an enclosing SSA value instead, which costs zero ops per
    iteration.

    Args:
        value: The caller's argument. ``None`` builds ``fallback``; a Python or
            NumPy number is built as a rank-0 constant of ``dtype``; anything
            else is treated as an already-built graph constant.
        fallback: The static sentinel ``None`` selects (``+inf`` for the masked
            ``argmin``, ``I_INF`` for a "no candidate" index).
        dtype: ``evoxc``/NumPy dtype a *built* constant is pinned to.

    Returns:
        A rank-0 ``SymbolicTensor`` of ``dtype``, or a supplied tensor
        **verbatim** — no cast, no reshape, so the caller's own pinning holds.

    Raises:
        tensorneat.common.errors.ConfigError: If a supplied tensor's dtype is
            not ``dtype``. ``evoxc.select`` promotes instead of rejecting, so
            without this check a ``float64`` filler would silently widen the
            ``argmin`` input and a ``float64``/``int64`` default would change
            the returned index's dtype. Checked at trace time on a static
            dtype, never on a value.
    """
    if value is None:
        value = fallback
    if isinstance(value, (int, float, np.integer, np.floating)):
        return _scalar_constant(value, dtype)
    got = getattr(value, "dtype", None)
    if got is not None and np.dtype(got) != np.dtype(dtype):
        raise ConfigError(
            "a supplied sentinel constant must have dtype "
            f"{np.dtype(dtype)}, got {np.dtype(got)}; build it with "
            "enp.full((), value, dtype=...)"
        )
    return value


# --------------------------------------------------------------------------- #
# flat-index plumbing
# --------------------------------------------------------------------------- #


def _flat_shape(x):
    """``(pop_size, row_len)`` of a flat-index target given as a tensor or a shape.

    Args:
        x: Batch-first tensor of shape ``(P, R)``, or a ``(P, R)`` pair of
            Python ints when only the shape is known (an index block that is
            built once and reused against a different tensor of the same
            shape, e.g. a loop-invariant index inside ``while_loop``).

    Returns:
        tuple: ``(pop_size, row_len)`` as Python ints.
    """
    if isinstance(x, tuple):
        pop_size, row_len = x
    else:
        pop_size, row_len = x.shape
    return int(pop_size), int(row_len)


def prepare_flat_index(x, idx, present=None, fallback=0, trusted=False):
    """Turn per-row column indices into flat 1-D indices into the flattened ``x``.

    ``idx`` holds one *column* per position; the flat index of position
    ``(p, ...)`` is ``p * R + idx[p, ...]`` with ``R = x.shape[1]``.

    Two safety rules, both mandated by the contract that evoxc's numpy backend
    must raise rather than clamp:

    * Positions with ``present == False`` are replaced by ``fallback`` *before*
      the index is formed, so a masked ``I_INF`` never reaches the backend.
    * Any remaining index outside ``[0, R)`` is redirected to ``P * R`` — one
      past the end of the flattened payload — so the backend raises
      ``evoxc.ShapeError`` instead of silently reading a neighbouring row
      (plain ``row * R + col`` arithmetic would land inside the array for
      ``R <= col < 2R``, and negative indices would wrap).

    The result depends only on ``idx``, ``present`` and the static shape, so a
    caller whose index block is **loop-invariant** (the relaxation in
    :func:`~tensorneat.common.tensorization.node_levels`) builds it once
    outside its ``while_loop`` and then only calls
    :func:`flat_gather_prepared` / :func:`flat_scatter_prepared` inside.
    :func:`flat_gather` and :func:`masked_flat_gather` are the one-shot
    spellings of the same pair, so this function is the single owner of the
    index arithmetic.

    Args:
        x: Batch-first destination/source of static shape ``(P, R)``, or the
            ``(P, R)`` shape pair when only the shape is needed.
        idx: ``(P, *shape)`` integer column indices.
        present: ``(P, *shape)`` bool mask broadcastable to ``idx``, or ``None``
            for "every position is real".
        fallback (int): In-range replacement for masked-out positions (``0`` for
            a gather, the dump column for a scatter).
        trusted (bool): ``False`` (default) keeps the out-of-range redirect.
            ``True`` asserts that **every position selected by ``present``**
            already holds an index in ``[0, R)``, so the test and its redirect
            are dropped. Only for a caller that can prove the bound from the
            construction of ``idx`` (a sanitized key index, a ``dst * K + col``
            cell with ``col < K``, a permutation); it trades the "backend
            raises" net for op count, so never pass it speculatively.

    Returns:
        ``(P, n)`` integer flat indices, ``n`` the product of ``idx.shape[1:]``.
    """
    pop_size, row_len = _flat_shape(x)
    if present is None:
        safe = idx
    else:
        safe = evoxc.select(present, idx, _scalar_constant(fallback, idx.dtype))
    if not trusted:
        escaped = evoxc.logical_or(
            evoxc.less(safe, 0), evoxc.greater_equal(safe, row_len)
        )
        safe = evoxc.select(
            escaped, _scalar_constant(pop_size * row_len, idx.dtype), safe
        )
    return evoxc.reshape(safe, (pop_size, -1)) + batch_offsets(pop_size, row_len)


def flat_gather_prepared(x, flat_idx, shape):
    """Gather through an index block already built by :func:`prepare_flat_index`.

    Args:
        x: Batch-first source of shape ``(P, R)`` (any dtype), flattened
            row-wise.
        flat_idx: ``(P, n)`` ``int32`` flat indices from
            :func:`prepare_flat_index` **for the same shape** ``(P, R)``.
        shape (tuple): ``(P, *shape)`` result shape of the gather, i.e. the
            index block reshaped back to its logical layout.

    Returns:
        ``(P, *shape)`` tensor with ``x``'s dtype.

    Raises:
        evoxc.ShapeError / evoxc.BackendError: Propagated unchanged if a flat
            index escapes the flattened payload.
    """
    pop_size, row_len = _flat_shape(x)
    flat = evoxc.reshape(x, (pop_size * row_len,))
    return evoxc.reshape(
        evoxc.gather(flat, flat_idx, axis=0), tuple(int(d) for d in shape)
    )


def batch_offsets(pop_size, row_len):
    """Row offsets that turn ``(P, ...)`` column indices into flat 1-D indices.

    The sanctioned workaround for evoxc's axis-replacing ``gather``/
    ``scatter``: flatten a ``(P, R)`` payload to ``1-D`` and index it with
    ``flat = row * row_len + col``.

    Args:
        pop_size (int): Population size ``P``; static at build time.
        row_len (int): Row length ``R`` of the flattened payload; static at
            build time.

    Returns:
        A ``(P, 1)`` ``int32`` tensor ``[[0], [R], [2R], ...]`` that broadcasts
        against any ``(P, K)`` column-index block, so
        ``columns + batch_offsets(P, R)`` is the flat index tensor of shape
        ``(P, K)``.

    Raises:
        tensorneat.common.errors.ConfigError: If ``pop_size`` or ``row_len``
            is not a positive ``int``, or if ``pop_size * row_len`` would
            overflow the ``int32`` flat-index range (``>= I_INF``).

    Notes:
        The offsets are built with ``evoxc.numpy.arange`` (``enp``) plus
        ``enp.reshape``; there is no ``evoxc.broadcast_to``, so broadcasting the
        ``(P, 1)`` table against a ``(P, K)`` block is left to the caller's
        index arithmetic (``evoxc`` broadcasting rules do the rest).
    """
    _positive_static(pop_size, "pop_size")
    _positive_static(row_len, "row_len")
    if pop_size * row_len >= I_INF:
        raise ConfigError(
            "pop_size * row_len must stay below I_INF for a flat int32 index, "
            f"got pop_size={pop_size} * row_len={row_len}"
        )
    offsets = enp.arange(0, pop_size * row_len, row_len, dtype=evoxc.int32)
    return enp.reshape(offsets, (pop_size, 1))


def flat_gather(x, idx, trusted=False):
    """Batched gather implemented through flattened 1-D indices.

    Args:
        x: Batch-first source of shape ``(P, R)`` (any dtype); it is flattened
            row-wise, so column ``c`` of row ``p`` lives at ``p * R + c``.
        idx: ``(P, *shape)`` ``int32`` indices, each in ``[0, R)``. Values
            equal to ``I_INF`` (or any other value outside ``[0, R)``) are a
            caller bug here — use :func:`masked_flat_gather` for batches that
            legitimately carry absent positions.
        trusted (bool): ``False`` (default) redirects an escaping index so the
            backend raises; ``True`` skips that test for a caller that can
            prove every index is inside ``[0, R)``.

    Returns:
        ``(P, *shape)`` tensor with ``x``'s dtype, where
        ``out[p, ...] == x_flat[p * R + idx[p, ...]]``.

    Raises:
        evoxc.ShapeError / evoxc.BackendError: Propagated unchanged if ``idx``
            escapes ``[0, R)`` (the numpy backend raises where JAX clamped).
            Never caught or re-wrapped by TensorNEAT.

    Notes:
        ``idx`` shapes must be fully static; ``P * R`` must stay inside the
        ``int32`` flat-index range. An escaping index is redirected to
        ``P * R`` (one past the flattened payload) before the gather, because
        ``row * R + col`` alone would land **inside** the array for
        ``R <= col < 2R`` (silently reading the next row) and numpy wraps
        negative indices; the redirected value makes the backend raise.
        :func:`prepare_flat_index` builds the index block and
        :func:`flat_gather_prepared` performs the gather, so a caller with a
        loop-invariant index block can hoist the former.
    """
    flat_idx = prepare_flat_index(x, idx, trusted=trusted)
    return flat_gather_prepared(x, flat_idx, idx.shape)


def masked_flat_gather(x, idx, mask=None, fill=float("nan"), trusted=False):
    """Batched gather where absent positions yield ``fill`` instead of a value.

    This is the workhorse of the forward pass: every per-connection gene
    attribute is gathered from a flattened ``(P, max_conns * attrs)`` payload
    with the schedule's ``in_conn`` block, and every source value is gathered
    with ``in_src``. Both index blocks are ``I_INF``-padded, so an unmasked
    gather would index out of range.

    Args:
        x: Batch-first source of shape ``(P, R)`` (any dtype), flattened
            row-wise.
        idx: ``(P, *shape)`` ``int32`` indices; positions equal to ``I_INF``
            (or outside ``[0, R)``) must be masked out by ``mask``.
        mask: ``(P, *shape)`` ``bool`` broadcastable to ``idx``, or ``None``
            (default) for "every position is real". ``False`` marks an absent
            position.
        fill: Value written where the position is masked out **or**
            ``idx == I_INF``. Defaults to ``NaN`` ("absent / not computed");
            pass the reduction's neutral element (``0``, ``-inf``, ``+inf``,
            ``1``) when the result feeds an unmasked aggregation.
        trusted (bool): ``False`` (default) keeps the out-of-range redirect;
            ``True`` is the ``prepare_flat_index`` opt-in for a caller that can
            prove every *selected* index is inside ``[0, R)``. Aborting an
            ``I_INF`` index is still done here.

    Returns:
        ``(P, *shape)`` tensor with ``x``'s dtype (``fill`` is cast to it).

    Raises:
        evoxc.BackendError: Propagated unchanged when a *selected* index is
            outside ``[0, R)`` — masking happens before the gather, so a
            masked-out ``I_INF`` never reaches the backend.

    Notes:
        Implemented as: sanitize ``idx`` by replacing masked-out / ``I_INF``
        slots with ``0``, :func:`flat_gather`, then ``evoxc.select`` to
        ``fill``. Replacing with ``0`` is safe only because the result is
        discarded by the same predicate that selected the replacement, and the
        out-of-range redirect of :func:`flat_gather` runs after the
        sanitization, so only *selected* indices can raise.
    """
    present = evoxc.not_equal(idx, _scalar_constant(I_INF, idx.dtype))
    if mask is not None:
        present = evoxc.logical_and(present, mask)
    flat_idx = prepare_flat_index(x, idx, present, 0, trusted=trusted)
    values = flat_gather_prepared(x, flat_idx, idx.shape)
    return evoxc.select(present, values, _scalar_constant(fill, x.dtype))


def flat_scatter(x, idx, updates, trusted=False):
    """Batched scatter implemented through flattened 1-D indices.

    Args:
        x: Batch-first destination of shape ``(P, R)`` (any dtype).
        idx: ``(P, *shape)`` ``int32`` indices, each in ``[0, R)``.
        updates: ``(P, *shape)`` values to write; dtype must equal ``x``'s.
        trusted (bool): ``False`` (default) redirects an escaping index so the
            backend raises; ``True`` skips that test for a caller that can
            prove every index is inside ``[0, R)`` (e.g. a permutation).

    Returns:
        ``(P, R)`` tensor: ``x`` with ``out_flat[p * R + idx[p, ...]] =
        updates[p, ...]``.

    Raises:
        evoxc.ShapeError / evoxc.BackendError: Propagated unchanged for
            out-of-range indices or dtype mismatches.

    Notes:
        ``evoxc.scatter`` is set-only (no add/max mode), so duplicate indices
        resolve in flat traversal order — the last write wins. Callers that
        need accumulation must reduce first (see the ``K``-axis aggregation in
        ``functions``). Masked writes belong in :func:`masked_flat_scatter`.
        :func:`prepare_flat_index` builds the index block and
        :func:`flat_scatter_prepared` performs the scatter, so a caller that
        writes two payloads to the same cells hoists the former — but it must
        **not** then call :func:`flat_scatter_prepared` twice with that block:
        one index block feeds exactly one scatter (see the module docstring),
        so the two payloads belong in one concatenated buffer and one call.
    """
    flat_idx = prepare_flat_index(x, idx, trusted=trusted)
    return flat_scatter_prepared(x, flat_idx, updates)


def flat_scatter_prepared(x, flat_idx, updates):
    """Scatter through an index block already built by :func:`prepare_flat_index`.

    Args:
        x: Batch-first destination of shape ``(P, R)`` (any dtype).
        flat_idx: ``(P, n)`` ``int32`` flat indices from
            :func:`prepare_flat_index` **for the same shape** ``(P, R)``.
        updates: ``(P, *shape)`` values to write, ``n`` elements per row; dtype
            must equal ``x``'s.

    Returns:
        ``(P, R)`` tensor with ``x``'s dtype.

    Raises:
        evoxc.ShapeError / evoxc.BackendError: Propagated unchanged if a flat
            index escapes the flattened payload.

    Notes:
        Call this **once per index block**: two calls with the same
        ``flat_idx`` emit two scatters sharing one index value, which
        ``iree-compile`` rejects (see the module docstring). Two payloads that
        write the same cells go through one call on a buffer wide enough to
        hold both — and where the cells are cheap to *read* instead of write,
        a gather is the better formulation: ``evoxc`` lowers ``scatter`` to one
        workgroup of one thread on ROCm (~16-21 ns per updated cell), while
        ``gather`` is parallel (``tensorization.incoming_table`` is the worked
        example of replacing a write with a read).
    """
    pop_size, row_len = _flat_shape(x)
    out = evoxc.scatter(
        evoxc.reshape(x, (pop_size * row_len,)),
        evoxc.reshape(flat_idx, (-1,)),
        evoxc.reshape(updates, (-1,)),
        axis=0,
    )
    return evoxc.reshape(out, (pop_size, row_len))


def masked_flat_scatter(x, idx, updates, mask, trusted=False):
    """Batched scatter whose dropped writes land in a trailing dump slot.

    Routing masked writes to index ``0`` would corrupt a real element, and
    routing an ``I_INF`` index to the backend would raise; both are avoided by
    giving the destination one extra trailing column that no reader ever looks
    at (the *dump slot*).

    Args:
        x: Batch-first destination of shape ``(P, R + 1)``; column ``R`` is the
            dump slot and its incoming content is irrelevant.
        idx: ``(P, *shape)`` ``int32`` indices into ``[0, R)``. Positions that
            are masked out may hold anything at all, ``I_INF`` included —
            they are replaced by ``R`` before the scatter.
        updates: ``(P, *shape)`` values to write; dtype must equal ``x``'s.
            Values written to the dump slot are discarded.
        mask: ``(P, *shape)`` ``bool`` broadcastable to ``idx``; ``False``
            redirects that write to the dump slot.
        trusted (bool): ``False`` (default) keeps the out-of-range redirect;
            ``True`` is the ``prepare_flat_index`` opt-in for a caller that can
            prove every position with ``mask == True`` is inside ``[0, R)``
            (the dump redirect for masked writes still happens here).

    Returns:
        ``(P, R + 1)`` tensor; callers slice the dump column off with
        ``evoxc.slice(x, (0, 0), (P, R))`` before reading the payload.

    Raises:
        evoxc.ShapeError / evoxc.BackendError: Propagated unchanged, e.g. when
            ``x`` does not have the expected ``(P, R + 1)`` width.

    Notes:
        Also used to build the dense key→slot table
        (:func:`tensorization.key_slot_table`), where the masked positions are
        the node rows whose key is absent (``I_INF``) or outside
        ``[0, key_span)`` — the caller's ``in_range`` mask, never an ``isnan``
        test, because canonical keys are ``int32``.
    """
    pop_size, width = _flat_shape(x)
    dump_column = width - 1
    flat_idx = prepare_flat_index(x, idx, mask, dump_column, trusted=trusted)
    out = evoxc.scatter(
        evoxc.reshape(x, (pop_size * width,)),
        evoxc.reshape(flat_idx, (-1,)),
        evoxc.reshape(updates, (-1,)),
        axis=0,
    )
    return evoxc.reshape(out, (pop_size, width))


def fetch_first(mask, default=I_INF):
    """Index of the first ``True`` entry along the last axis, per batch row.

    Args:
        mask: ``(P, L)`` ``bool`` tensor.
        default: ``int32`` value returned where ``mask`` has no ``True``
            entry; ``I_INF`` by default (``None`` also selects that
            sentinel, so "unset" and "no candidate" read the same). A
            pre-built ``int32`` graph constant (``enp.full((), v,
            dtype=evoxc.int32)``) may be passed instead of an ``int``: it is
            used **verbatim** (no cast, no reshape), which is how a caller inside a
            ``scan``/``cond`` body hoists the sentinel out of the loop. Any
            Python/NumPy number keeps the original behaviour: one rank-0
            ``int32`` constant pinned to ``int32``.

    Returns:
        ``(P,)`` ``int32`` tensor of indices in ``[0, L)``, or ``default``.

    Raises:
        evoxc.ShapeError: Propagated unchanged if ``mask`` is not 2-D.
        tensorneat.common.errors.ConfigError: If a supplied constant's dtype is
            not ``int32`` (``evoxc.select`` would promote silently).

    Notes:
        Computed as ``argmax`` of the boolean mask followed by
        ``evoxc.select(mask_selected, idx, default)`` — no ``where`` (evoxc has
        none), no ``cumsum`` (a StableHLO export deferral).

        The sentinel is a **tensor, not a bare scalar**, to pin the result dtype
        (a bare Python int would be weakly promoted to ``int64`` against the
        ``int32`` index); passing a pre-built one leaves the dtype pinning to
        the caller's constant. ``algorithm/neat/_kernels.py::speciate`` calls
        this from inside its two ``scan`` bodies and is the intended hoister —
        the constant it supplies must be built at trace time, outside the loop.
    """
    if default is None:  # "unset" -> the documented I_INF sentinel
        default = I_INF
    hit = evoxc.reduce_max(mask, axes=-1)
    first = evoxc.cast(evoxc.argmax(mask, axis=-1), evoxc.int32)
    return evoxc.select(hit, first, _sentinel_constant(default, I_INF, evoxc.int32))


def fetch_random(key, mask, default=I_INF):
    """Index of a uniformly random ``True`` entry along the last axis.

    Args:
        key: An ``evoxc.random`` key (rank-0 ``int64`` tensor, see
            :func:`evoxc.random.key`); draw the whole batch from it so the
            result is a function of the key alone.
        mask: ``(P, L)`` ``bool`` tensor.
        default: ``int32`` value returned where ``mask`` has no ``True``
            entry; ``I_INF`` by default (``None`` also selects that sentinel).

    Returns:
        ``(P,)`` ``int32`` tensor of indices in ``[0, L)``, or ``default``.

    Raises:
        evoxc.BackendError: Propagated unchanged — ``evoxc.random`` is
            numpy-backend-only in v1 and every compiler backend rejects random
            ops explicitly. There is no host-side fallback.

    Notes:
        The legacy JAX version built a ``cumsum`` mask and drew a rank; the
        evoxc port must not, because ``cumsum`` is a StableHLO export deferral
        (works on the numpy interpreter only). Equivalent sampling that stays
        exportable: draw a ``(P, L)`` block of i.i.d. uniforms
        (``evoxc.random.uniform``), set masked-out positions to ``-inf`` and
        take the ``argmax`` — the maximiser of i.i.d. draws restricted to a set
        is uniform over that set.

        **No production caller**: the genome's ``(row, hit)`` primitive in
        ``genome/operations/mutation/arrays.py`` is the cheaper replacement
        production uses; this helper is exported for tests and legacy parity.
    """
    if default is None:  # "unset" -> the documented I_INF sentinel
        default = I_INF
    draws = evoxc.random.uniform(key, tuple(mask.shape), 0.0, 1.0, dtype=evoxc.float32)
    candidate = evoxc.select(
        mask, draws, _scalar_constant(float("-inf"), evoxc.float32)
    )
    hit = evoxc.reduce_max(mask, axes=-1)
    chosen = evoxc.cast(evoxc.argmax(candidate, axis=-1), evoxc.int32)
    return evoxc.select(hit, chosen, _scalar_constant(default, evoxc.int32))


def rank_elements(x, reverse=False):
    """Ordinal rank of each element along the last axis (legacy orientation).

    Args:
        x: ``(P, L)`` numeric tensor to rank.
        reverse (bool): Orientation flag. ``False`` (default, legacy
            behaviour) ranks **large to small** — rank ``0`` is the largest
            element. ``True`` ranks small to large. Trace-time static.

    Returns:
        ``(P, L)`` ``int32`` tensor of ranks in ``[0, L)``.

    Raises:
        evoxc.ShapeError: Propagated unchanged if ``x`` is not 2-D.

    Notes:
        The rank of an element is its position in the sorted order, i.e. the
        inverse permutation of ``argsort``. ``evoxc.argsort`` returns
        ``int64`` (cast down here) and needs ``stable=True`` so equal values
        keep a deterministic order — the schedule's group ranks depend on it.
    """
    oriented = x if reverse else evoxc.negate(x)
    order = evoxc.argsort(oriented, axis=-1, stable=True)
    inverse = evoxc.argsort(order, axis=-1, stable=True)
    return evoxc.cast(inverse, evoxc.int32)


def argmin_with_mask(arr, mask, *, filler=None, default=None):
    """Index of the smallest selectable element, per batch row.

    Args:
        arr: ``(P, L)`` ``float32`` tensor of scores.
        mask: ``(P, L)`` ``bool`` tensor; ``False`` entries are not selectable.
        filler: ``+inf`` of ``arr.dtype`` by default (``None``) — the value the
            masked-out entries are replaced with before the ``argmin``. A
            pre-built graph constant of ``arr.dtype`` may be passed instead of
            a number; it is used **verbatim** (no cast, no reshape). Must stay
            *above* every real score, or a masked entry wins the ``argmin``.
        default: ``I_INF`` by default (``None``), i.e. the returned index for a
            row whose mask has no ``True`` entry. An ``int`` is built as a
            rank-0 ``int32`` constant; a pre-built ``int32`` graph constant is
            used verbatim.

    Returns:
        ``(P,)`` ``int32`` tensor: the index of the minimal element among the
        ``True`` positions, or ``default`` when a row has no ``True`` position.

    Raises:
        evoxc.ShapeError: Propagated unchanged if the shapes disagree.
        tensorneat.common.errors.ConfigError: If a supplied constant's dtype is
            not ``arr.dtype`` (``filler``) / ``int32`` (``default``) — a wider
            or narrower constant would change the ``select``'s result dtype.

    Notes:
        Masked-out entries are replaced by ``+inf`` (``evoxc.select``) before
        ``evoxc.argmin``, since evoxc reductions have no ``where=``/
        ``initial=``. Returning ``I_INF`` for an empty mask is a deliberate
        sentinel: the legacy JAX helper silently returned slot ``0``, which is
        a real individual, so callers of this port must test for ``I_INF``.

        Both sentinels are **tensors, not bare scalars**, to pin the result
        dtypes (``filler`` must keep ``arr.dtype``, ``default`` must stay
        ``int32``). Each one costs a ``constant`` op, so a caller inside a
        ``scan``/``cond`` body passes constants built once at trace time
        (``enp.full((), float("inf"), dtype=evoxc.float32)`` / ``... I_INF ...
        evoxc.int32``) and the loop body closes over them instead of re-emitting
        them per iteration — ``algorithm/neat/_kernels.py::speciate`` is that
        caller. ``None`` keeps the one-shot behaviour unchanged, op for op.
    """
    scored = evoxc.select(
        mask, arr, _sentinel_constant(filler, float("inf"), arr.dtype)
    )
    hit = evoxc.reduce_max(mask, axes=-1)
    smallest = evoxc.cast(evoxc.argmin(scored, axis=-1), evoxc.int32)
    return evoxc.select(hit, smallest, _sentinel_constant(default, I_INF, evoxc.int32))


def mutate_float(key, val, init_mean, init_std, mutate_power, mutate_rate, replace_rate):
    """Mutate a batch of float gene values (legacy three-way rule).

    Draw ``r ~ U[0, 1)`` per element:

    * ``r < mutate_rate`` → add ``normal(0, 1) * mutate_power`` perturbation,
    * ``mutate_rate <= r < mutate_rate + replace_rate`` → replace with a fresh
      ``normal(init_mean, init_std)`` sample,
    * otherwise → keep the original value.

    Args:
        key: ``evoxc.random`` key (rank-0 ``int64`` tensor); split it into the
            three independent streams the rule consumes (perturbation,
            replacement, decision).
        val: ``(P,)`` ``float32`` (or any batch-first shape) current values;
            ``NaN`` slots (absent genes) must be filtered by the caller.
        init_mean: Scalar ``float32``/Python float: mean of a replacement draw.
        init_std: Scalar: standard deviation of a replacement draw.
        mutate_power: Scalar: standard deviation of the additive perturbation.
        mutate_rate: Scalar in ``[0, 1]``: probability of perturbation; ``1``
            means "always perturb" and is legal (the perturb band has no upper
            keep bound to protect).
        replace_rate: Scalar in ``[0, 1)``: probability of replacement;
            ``mutate_rate + replace_rate`` must not exceed ``1``.

    Returns:
        Tensor with ``val``'s shape and dtype.

    Raises:
        evoxc.BackendError: Propagated unchanged (random ops are numpy-backend
            only in v1).
        tensorneat.common.errors.ConfigError: If the host-side rates are
            outside ``[0, 1]`` (``mutate_rate``) / ``[0, 1)`` (``replace_rate``)
            or sum above ``1`` — validated before tracing.

    Notes:
        Branching is ``evoxc.select``, never ``where``/Python ``if``; the
        legacy ``jnp.where`` nests map one-to-one onto two nested selects.
        Standalone public kernel with **no production caller**: the genome
        attribute-mutation path is
        :func:`tensorneat.genome.gene.blocks.mutate_attrs`, which does not call
        this helper; it is exported for tests and legacy parity.
    """
    mutate_rate = _rate("mutate_rate", mutate_rate, allow_one=True)
    replace_rate = _rate("replace_rate", replace_rate)
    if mutate_rate + replace_rate > 1.0:
        raise ConfigError(
            "mutate_rate + replace_rate must not exceed 1, got "
            f"{mutate_rate} + {replace_rate}"
        )
    shape = tuple(val.shape)
    key_noise, key_replace, key_draw = evoxc.random.split_n(key, 3)
    noise = evoxc.random.normal(
        key_noise, shape, 0.0, _scalar(mutate_power), dtype=val.dtype
    )
    replacement = evoxc.random.normal(
        key_replace, shape, _scalar(init_mean), _scalar(init_std), dtype=val.dtype
    )
    draw = evoxc.random.uniform(key_draw, shape, 0.0, 1.0, dtype=val.dtype)
    replaced = evoxc.select(
        evoxc.less(draw, mutate_rate + replace_rate), replacement, val
    )
    return evoxc.select(evoxc.less(draw, mutate_rate), evoxc.add(val, noise), replaced)


def mutate_int(key, val, options, replace_rate):
    """Mutate a batch of integer-valued gene slots (legacy replace-only rule).

    Draw ``r ~ U[0, 1)`` per element: ``r < replace_rate`` → pick a uniformly
    random entry of ``options``, otherwise keep the value.

    Args:
        key: ``evoxc.random`` key (rank-0 ``int64`` tensor).
        val: ``(P,)`` ``int32``-valued gene slot (the *position* in the user's
            option list, not the registry index).
        options: ``(M,)`` ``int32`` tensor of the legal replacement values
            (typically ``evoxc.numpy.arange(M)``). Static at build time. There
            is no ``random.choice`` in evoxc, so the draw is
            ``gather(options, randint(key, shape, 0, M))``.
        replace_rate: Scalar in ``[0, 1]``: probability of replacement.

    Returns:
        Tensor with ``val``'s shape and dtype.

    Raises:
        tensorneat.common.errors.ConfigError: If ``options`` is empty or the
            rate is outside ``[0, 1]`` (host-side validation, before tracing).
        evoxc.BackendError: Propagated unchanged.

    Notes:
        ``options`` must be non-empty and static; an empty option list is a
        configuration bug, not a runtime condition.
        Standalone public kernel with **no production caller**: the genome
        attribute-mutation path is
        :func:`tensorneat.genome.gene.blocks.mutate_attrs`, which does not call
        this helper; it is exported for tests and legacy parity.
    """
    replace_rate = _rate("replace_rate", replace_rate, allow_one=True)
    count = int(options.shape[0])
    if count <= 0:
        raise ConfigError(
            "options must be a non-empty 1-D tensor, got an empty option list"
        )
    shape = tuple(val.shape)
    key_draw, key_pick = evoxc.random.split_n(key, 2)
    draw = evoxc.random.uniform(key_draw, shape, 0.0, 1.0, dtype=evoxc.float32)
    picked = evoxc.random.randint(key_pick, shape, 0, count, dtype=evoxc.int32)
    replacement = evoxc.cast(evoxc.gather(options, picked, axis=0), val.dtype)
    return evoxc.select(evoxc.less(draw, replace_rate), replacement, val)


def hash_rows(rows):
    """Hash each row of a gene array to a single ``uint32`` — **host only**.

    Args:
        rows: ``(N, L)`` numpy array of gene rows — ``int32`` key columns,
            ``float32`` attribute columns, or a whole mixed row array kept
            integer-valued — in host memory, not a traced tensor.

    Returns:
        ``(N,)`` ``numpy.uint32`` array; equal rows give equal hashes, unequal
        rows almost certainly different ones. This is the host-only twin of the
        in-graph ``hash_int32`` (:mod:`tensorneat.genome.gene.base`), with
        **no production caller** — it is exported for tests and legacy parity.

    Raises:
        ValueError: If ``rows`` is not 2-D.

    Notes:
        Must **never** be called inside an ``@evoxc.defn`` graph: it needs
        numpy bit-viewing (``view(np.uint32)``) and evoxc has no ``bitcast``
        op. ``Genome.hash()`` is the in-graph ``hash_int32`` and does not call
        this function; in-graph gene identity therefore compares the key
        columns with ``evoxc.equal`` (crossover matching), never a row hash.
        Fold order is left-to-right with the legacy ``0x9E3779B9`` constant so
        the numbers stay comparable with the pre-rewrite library.
    """
    array = np.ascontiguousarray(rows)
    if array.ndim != 2:
        raise ValueError(
            f"hash_rows expects a 2-D (rows, columns) array, got {array.ndim}-D"
        )
    row_count = array.shape[0]
    if array.shape[1] == 0:  # the empty fold: no word is mixed in
        return np.zeros(row_count, dtype=np.uint32)
    raw = array.view(np.uint8).reshape(row_count, -1)
    padding = (-raw.shape[1]) % 4  # uint32 needs whole 4-byte words per row
    if padding:
        padded = np.zeros((row_count, raw.shape[1] + padding), dtype=np.uint8)
        padded[:, : raw.shape[1]] = raw
        raw = padded
    words = raw.view(np.uint32)
    hashes = np.zeros(row_count, dtype=np.uint32)
    golden = np.uint32(0x9E3779B9)
    shift_left, shift_right = np.uint32(6), np.uint32(2)
    for column in range(words.shape[1]):
        word = words[:, column]
        hashes = hashes ^ (
            word + golden + (hashes << shift_left) + (hashes >> shift_right)
        )
    return hashes


def to_numpy(tensor):
    """Host-side accessor: an ``evoxc.Tensor`` (or array) as a NumPy array.

    **Host only** — never call this inside an ``@evoxc.defn`` graph. Every
    logging, checkpoint and ``show`` path goes through it because
    ``evoxc.Tensor`` has no ``__array__``, so ``np.asarray`` would leave it
    untouched instead of unwrapping it.

    Args:
        tensor: An ``evoxc.Tensor`` (has ``.numpy()``), a ``numpy.ndarray``
            (returned unchanged) or anything else ``np.asarray`` accepts.

    Returns:
        ``numpy.ndarray`` sharing the tensor's buffer where the backend allows.

    Notes:
        Kept deliberately shallow: a pytree of tensors is unwrapped by the
        caller (``evoxc.tree_map``) or by ``test/support/runner.to_numpy``; this
        helper converts one leaf.
        This is the **single explicit device-to-host step** of the library: a
        tensor whose ``.device.kind`` is not ``"cpu"`` is transferred with
        ``Tensor.to(Device("cpu", 0))`` first, because ``Tensor.numpy()``
        raises ``evoxc.DeviceError`` on device memory (evoxc never transfers
        implicitly). The CPU path is unchanged -- a CPU tensor (or an ndarray)
        is never copied here.
    """
    if isinstance(tensor, np.ndarray):
        return tensor
    numpy_method = getattr(tensor, "numpy", None)
    if callable(numpy_method):
        device = getattr(tensor, "device", None)
        if device is not None and getattr(device, "kind", "cpu") != "cpu":
            tensor = tensor.to(evoxc.Device("cpu", 0))
        return tensor.numpy()
    return np.asarray(tensor)
