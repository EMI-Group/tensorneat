"""Batched array helpers shared by every evoxc kernel in TensorNEAT.

Dependency DAG position: bottom of ``tensorneat.common`` (with
:mod:`~tensorneat.common.errors`); it imports nothing from ``genome``,
``algorithm``, ``problem`` or ``pipeline``, and no ``jax``/``torch`` symbol and no
NumPy global-RNG call — the only host-side numpy use is :func:`hash_rows`, which is
explicitly marked host-only.

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
* ``argsort`` returns ``int64`` and requires ``stable=True`` wherever
  tie-breaking must be deterministic (group ranks).

Sentinel contract (shared with :mod:`~tensorneat.common.tensorization`):

* ``I_INF = 2147483647`` (``int32`` max) means "no index / unmatched /
  unavailable". It lives in **integer** arrays only, never mixed with ``NaN``
  in the same array.
* ``NaN`` in a **float** array means "absent row" (``isnan(x[:, 0])``) or
  "not computed yet".
"""

import numpy as np

#: ``int32`` max: "no index / unmatched / unavailable" in integer arrays.
#: Never store it in a float array (use ``NaN`` there) and never use it as a
#: real index — every consumer must test for it before indexing.
I_INF: int = int(np.iinfo(np.int32).max)

#: Exclusive upper bound of gene keys: keys are integer-valued ``float32`` in
#: ``[0, KEY_MAX)`` so that *exact* equality (and therefore key matching)
#: survives float32 representation. ``KEY_MAX = 2**24`` is the first integer
#: that float32 can no longer distinguish from its neighbour.
KEY_MAX: int = 1 << 24


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
        The offsets are built with ``evoxc.numpy.arange`` (``enp``) and
        ``enp.broadcast_to`` — there is no ``evoxc.broadcast_to``.
    """
    raise NotImplementedError(
        "build the (P, 1) int32 row-offset table used to flatten batch indices"
    )


def flat_gather(x, idx):
    """Batched gather implemented through flattened 1-D indices.

    Args:
        x: Batch-first source of shape ``(P, R)`` (any dtype); it is flattened
            row-wise, so column ``c`` of row ``p`` lives at ``p * R + c``.
        idx: ``(P, *shape)`` ``int32`` indices, each in ``[0, R)``. Values
            equal to ``I_INF`` (or any other value outside ``[0, R)``) are a
            caller bug here — use :func:`masked_flat_gather` for batches that
            legitimately carry absent positions.

    Returns:
        ``(P, *shape)`` tensor with ``x``'s dtype, where
        ``out[p, ...] == x_flat[p * R + idx[p, ...]]``.

    Raises:
        evoxc.ShapeError / evoxc.BackendError: Propagated unchanged if ``idx``
            escapes ``[0, R)`` (the numpy backend raises where JAX clamped).
            Never caught or re-wrapped by TensorNEAT.

    Notes:
        ``idx`` shapes must be fully static; ``P * R`` must stay inside the
        ``int32`` flat-index range.
    """
    raise NotImplementedError(
        "gather x through flat 1-D indices and reshape back to idx's batch shape"
    )


def masked_flat_gather(x, idx, mask=None, fill=float("nan")):
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
        discarded by the same predicate that selected the replacement.
    """
    raise NotImplementedError(
        "gather through flat indices with absent positions replaced by fill"
    )


def flat_scatter(x, idx, updates):
    """Batched scatter implemented through flattened 1-D indices.

    Args:
        x: Batch-first destination of shape ``(P, R)`` (any dtype).
        idx: ``(P, *shape)`` ``int32`` indices, each in ``[0, R)``.
        updates: ``(P, *shape)`` values to write; dtype must equal ``x``'s.

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
    """
    raise NotImplementedError(
        "scatter updates into x through flat 1-D indices and reshape back"
    )


def masked_flat_scatter(x, idx, updates, mask):
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

    Returns:
        ``(P, R + 1)`` tensor; callers slice the dump column off with
        ``evoxc.slice(x, (0, 0), (P, R))`` before reading the payload.

    Raises:
        evoxc.ShapeError / evoxc.BackendError: Propagated unchanged, e.g. when
            ``x`` does not have the expected ``(P, R + 1)`` width.

    Notes:
        Also used to build the dense key→slot table
        (:func:`tensorization.key_slot_table`), where absent (``NaN``) node
        rows are the masked ones.
    """
    raise NotImplementedError(
        "scatter with masked writes redirected to a trailing dump column"
    )


def fetch_first(mask, default=I_INF):
    """Index of the first ``True`` entry along the last axis, per batch row.

    Args:
        mask: ``(P, L)`` ``bool`` tensor.
        default: ``int32`` value returned where ``mask`` has no ``True``
            entry; ``I_INF`` by default.

    Returns:
        ``(P,)`` ``int32`` tensor of indices in ``[0, L)``, or ``default``.

    Raises:
        evoxc.ShapeError: Propagated unchanged if ``mask`` is not 2-D.

    Notes:
        Computed as ``argmax`` of the boolean mask followed by
        ``evoxc.select(mask_selected, idx, default)`` — no ``where`` (evoxc has
        none), no ``cumsum`` (a StableHLO export deferral).
    """
    raise NotImplementedError(
        "return the first True index of each mask row, or default when none"
    )


def fetch_random(key, mask, default=I_INF):
    """Index of a uniformly random ``True`` entry along the last axis.

    Args:
        key: An ``evoxc.random`` key (rank-0 ``int64`` tensor, see
            :func:`evoxc.random.key`); draw the whole batch from it so the
            result is a function of the key alone.
        mask: ``(P, L)`` ``bool`` tensor.
        default: ``int32`` value returned where ``mask`` has no ``True``
            entry; ``I_INF`` by default.

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
    """
    raise NotImplementedError(
        "pick a uniform random True index per mask row without cumsum"
    )


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
    raise NotImplementedError(
        "compute ordinal ranks along the last axis via a stable argsort inverse"
    )


def argmin_with_mask(arr, mask):
    """Index of the smallest selectable element, per batch row.

    Args:
        arr: ``(P, L)`` ``float32`` tensor of scores.
        mask: ``(P, L)`` ``bool`` tensor; ``False`` entries are not selectable.

    Returns:
        ``(P,)`` ``int32`` tensor: the index of the minimal element among the
        ``True`` positions, or ``I_INF`` when a row has no ``True`` position.

    Raises:
        evoxc.ShapeError: Propagated unchanged if the shapes disagree.

    Notes:
        Masked-out entries are replaced by ``+inf`` (``evoxc.select``) before
        ``evoxc.argmin``, since evoxc reductions have no ``where=``/
        ``initial=``. Returning ``I_INF`` for an empty mask is a deliberate
        sentinel: the legacy JAX helper silently returned slot ``0``, which is
        a real individual, so callers of this port must test for ``I_INF``.
    """
    raise NotImplementedError(
        "return the argmin over mask-selected entries, I_INF when none"
    )


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
        mutate_rate: Scalar in ``[0, 1)``: probability of perturbation.
        replace_rate: Scalar in ``[0, 1)``: probability of replacement;
            ``mutate_rate + replace_rate`` must not exceed ``1``.

    Returns:
        Tensor with ``val``'s shape and dtype.

    Raises:
        evoxc.BackendError: Propagated unchanged (random ops are numpy-backend
            only in v1).
        tensorneat.common.errors.ConfigError: If the host-side rates are
            outside ``[0, 1)`` or sum above ``1`` — validated before tracing.

    Notes:
        Branching is ``evoxc.select``, never ``where``/Python ``if``; the
        legacy ``jnp.where`` nests map one-to-one onto two nested selects.
    """
    raise NotImplementedError(
        "apply the perturb-or-replace-or-keep float mutation rule with evoxc.random"
    )


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
    """
    raise NotImplementedError(
        "replace-or-keep int mutation using gather(options, randint(...))"
    )


def hash_rows(rows):
    """Hash each row of a gene array to a single ``uint32`` — **host only**.

    Args:
        rows: ``(N, L)`` numpy array of gene rows (float32 gene payload or
            int-valued columns). Host memory, not a traced tensor.

    Returns:
        ``(N,)`` ``numpy.uint32`` array; equal rows give equal hashes, unequal
        rows almost certainly different ones. Used by ``Genome.hash()`` for
        legacy-compatible debugging/inspection only.

    Raises:
        ValueError: If ``rows`` is not 2-D.

    Notes:
        Must **never** be called inside an ``@evoxc.defn`` graph: it needs
        numpy bit-viewing (``view(np.uint32)``) and evoxc has no ``bitcast``
        op. In-graph gene identity therefore compares the key columns with
        ``evoxc.equal`` (crossover matching), never a row hash. Fold order is
        left-to-right with the legacy ``0x9E3779B9`` constant so the numbers
        stay comparable with the pre-rewrite library.
    """
    raise NotImplementedError(
        "hash each row to uint32 on the host via numpy bit-viewing"
    )
