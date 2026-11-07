"""Masked aggregation kernels over the incoming-edge axis and the per-node dispatcher.

Every function here is *graph code*: it runs while an ``@evoxc.defn`` graph is
traced and may only use ``evoxc`` ops.

Contract shared by all kernels in this module:

* **Rank-3, batch-first.** ``z`` and ``mask`` are ``(P, N, K)``: population on
  axis 0, node slots on axis 1, the static in-degree bound ``K`` on the last
  axis. The reduction is always over the **last** axis and the result is
  ``(P, N)``. A consumer with a different incoming layout (e.g. a HyperNEAT
  substrate node) must reshape its contributions to ``(..., K)`` and pass an
  all-``True`` mask, or reduce on its own.
* **float32 contributions, bool mask.** ``mask`` marks the real incoming edges
  of each node; a slot with ``mask == False`` is padding (unmatched key,
  ``I_INF`` edge row, or a ``NaN`` source value) and must not influence the
  result.
* **Neutral padding, not ``where=``.** evoxc reductions accept neither
  ``where=`` nor ``initial=``, so each kernel first rewrites the masked-out
  slots to its neutral element (``0`` for sum/maxabs, ``1`` for product,
  ``-inf`` for max, ``+inf`` for min) with ``evoxc.select`` and then reduces.
* **The ``K`` axis is folded, not reduced -- up to ``_FOLD_MAX_WIDTH``.** For
  ``K <= _FOLD_MAX_WIDTH`` (128) ``K`` is a static, so every
  ``reduce_*``-over-``-1`` here is emitted as a fixed-length chain of
  ``(P, N)``/``(P, N, 1)`` slices and elementwise ops (see :func:`_sum_last`
  and :func:`_fold_last`); ``np.sum(..., axis=-1)`` on the reference backend is
  the single most expensive op of a forward pass at the canonical window
  width, and the folds reproduce the reference backend's summation order bit
  for bit. A **wider** window falls back to the backend ``reduce_*`` itself,
  because numpy's ``pairwise_sum`` only uses the 8-accumulator branch the fold
  reproduces while ``n <= PW_BLOCKSIZE (128)`` and halves recursively above
  it; the reduce is then identical by construction. The choice is a trace-time
  ``if`` on a static shape, never a data-dependent branch.
* **Empty input is the caller's problem.** With ``K >= 1`` an all-``False``
  mask column yields the neutral element (``0``, ``1``, ``-inf``, ``+inf``),
  except :func:`mean`, whose ``maximum(count, 1.0)`` floor already gives
  ``0.0``; :func:`apply_aggregation` overrides every such column to exactly
  ``0.0``.

Every kernel is a plain module-level function ``(z, mask, scalars=None) -> (P, N)``,
so it stays importable by name and picklable as a gene option. ``scalars`` is
the optional ``float -> tensor`` mapping of rank-0 constants a forward pass
prepared once (``genome.step._PreparedScalars.mapping``); it seeds
:meth:`_Window.scalar`, and ``None`` -- every direct call -- builds the same
literals locally, so the graph and the values are unchanged. Each kernel is a
thin wrapper over a private ``_body(window)`` that takes the shared
:class:`_Window` of one ``(z, mask)`` pair; the dispatcher calls the bodies and
reuses one window, which is what keeps a multi-option dispatch from rebuilding
the same neutral-padded tensor, the same ``(P, N, K)`` reduction and the same
scalar constant once per option (see the ``Notes`` of
:func:`apply_aggregation`).

Two static flags are attached to a kernel and read by the dispatcher:

* ``window_body`` -- the ``(window) -> (P, N)`` body shared with the
  dispatcher. Absent on a custom kernel, which the dispatcher then calls as
  ``fn(z, mask)``.
* ``zero_on_empty`` -- ``True`` for a kernel that already returns exactly
  ``0.0`` for an all-``False`` mask column (``bool``, default ``False`` for a
  custom kernel, i.e. "needs the ``has_valid`` guard")

Note that ``sum``, ``max``, ``min`` and ``abs`` shadow the builtins inside this
module; none of them is used here (``evoxc.maximum`` / ``evoxc.reduce_sum`` are).
"""

from typing import Any, Callable, Sequence

import evoxc
import evoxc.numpy as enp

from ..tools import flat_gather


#: numpy's ``pairwise_sum`` uses its 8-accumulator branch only up to
#: ``PW_BLOCKSIZE``; a wider window falls back to the backend reduce.
_FOLD_MAX_WIDTH = 128


def _planes(x):
    """The ``K`` planes of a ``(P, N, K)`` tensor as ``(P, N, 1)`` strided views.

    Parameters
    ----------
    x : ``(P, N, K) float32`` or ``(P, N, K) bool``
        A batch-first window tensor.

    Returns
    -------
    list
        ``K`` tensors of shape ``(P, N, 1)``, one per slot of the last axis.

    Notes
    -----
    One ``evoxc.slice`` op per plane and no copy: ``evoxc`` has no squeeze, so
    a plane keeps a trailing unit axis and the fold's result is reshaped to
    ``(P, N)`` once at the end instead of once per operand. Slicing the
    innermost axis is a strided read of the source; at the canonical widths
    (``K`` <= 8) the whole window is a few MiB and stays cache resident, which
    is what makes the fold beat one ``reduce_sum`` over the same axis.
    """
    pop_size, num_nodes = int(x.shape[0]), int(x.shape[1])
    return [
        evoxc.slice(x, (0, 0, k), (pop_size, num_nodes, 1))
        for k in range(int(x.shape[2]))
    ]


def _sum_last(x, zero):
    """``reduce_sum(x, axes=-1)`` as a static fold in the reference backend's order.

    Parameters
    ----------
    x : ``(P, N, K) float32``
        Summand window (already neutral-padded by the caller).
    zero : ``() float32``
        The ``+0.0`` the accumulation starts from (one shared constant).

    Returns
    -------
    ``(P, N) float32``
        Byte-identical to ``evoxc.reduce_sum(x, axes=-1)``.

    Notes
    -----
    ``reduce_sum`` lowers to one ``np.sum(x, axis=-1)``, which costs ~1.4 ms
    per level at ``(P, N, K) = (2000, 50, 4)`` on the numpy interpreter --
    numpy's buffered outer-loop path, not a bandwidth limit. ``K`` is static,
    so the same sum is emitted as ``K`` strided slice ops plus ``K - 1``
    ``(P, N, 1)`` adds (~0.4 ms for the same tensor).

    **Wide windows are not folded.** The recipe below reproduces numpy's
    ``pairwise_sum`` only while ``n <= PW_BLOCKSIZE (128)``; above that numpy
    halves the range recursively and the fold's association order diverges, so
    a window wider than :data:`_FOLD_MAX_WIDTH` returns
    ``evoxc.reduce_sum(x, axes=-1)`` (which *is* that reduction) instead. The
    guard is a trace-time ``if`` on the static window width, never ``cond``,
    and it never runs at the canonical widths (``K`` = 4 / 8).

    **The association order is numpy's, on purpose.** ``float32`` addition is
    not associative, so a fold only reproduces ``np.sum`` if it pairs the terms
    the way numpy's ``pairwise_sum`` does:

    * ``K < 8`` -- a plain left-to-right accumulation starting from ``+0.0``
      (numpy's small-``n`` branch is ``res = 0.; res += a[i]``). The ``+0.0``
      start is load-bearing for one case only: a column of ``-0.0`` sums to
      ``+0.0``, and starting the fold at the first plane would keep ``-0.0``.
    * ``8 <= K <= 128`` -- eight partial accumulators (``r[j]`` over ``j``,
      ``j + 8``, ...), combined as
      ``((r0+r1)+(r2+r3)) + ((r4+r5)+(r6+r7))``, then the
      ``K % 8`` tail accumulated sequentially.

    Verified bit for bit against ``np.sum`` (``view(uint32)`` equality, so
    signed zeros and ``NaN`` payloads count) for ``K`` = 1, 2, 3, 7, 8, 9, 15,
    16, 17, 31, 32, 64, 96, 127, 128, 129, 130, 200, 256, 300 on random,
    ``+-0.0``, ``NaN`` and ``+-inf`` data. A folded width costs ``K`` slices
    and ``K - 1`` adds plus the ``+0.0`` start and one closing ``reshape``.
    """
    if int(x.shape[2]) > _FOLD_MAX_WIDTH:
        return evoxc.reduce_sum(x, axes=-1)
    planes = _planes(x)
    width = len(planes)
    # ``+0.0 + a[0]`` is numpy's first accumulation and is the only place the
    # signed-zero rule can be decided, so it is never elided.
    acc = evoxc.add(zero, planes[0])
    if width < 8:
        for plane in planes[1:]:
            acc = evoxc.add(acc, plane)
    else:
        chains = [acc, *planes[1:8]]
        whole = width - width % 8
        for start in range(8, whole, 8):
            for slot in range(8):
                chains[slot] = evoxc.add(chains[slot], planes[start + slot])
        acc = evoxc.add(
            evoxc.add(
                evoxc.add(chains[0], chains[1]), evoxc.add(chains[2], chains[3])
            ),
            evoxc.add(
                evoxc.add(chains[4], chains[5]), evoxc.add(chains[6], chains[7])
            ),
        )
        for slot in range(whole, width):
            acc = evoxc.add(acc, planes[slot])
    return evoxc.reshape(acc, (int(x.shape[0]), int(x.shape[1])))


def _fold_last(x, binary, identity, reduce_op):
    """``reduce_*`` over the last axis of ``(P, N, K)`` as a left fold.

    Parameters
    ----------
    x : ``(P, N, K) float32``
        Window tensor (already neutral-padded by the caller).
    binary : Callable
        ``evoxc`` elementwise binary op: :func:`evoxc.multiply` for the
        product, :func:`evoxc.maximum` / :func:`evoxc.minimum` for max / min.
    identity : ``() float32``
        The op's neutral element, the value the accumulation starts from.
    reduce_op : Callable
        The matching backend reduction, used for a window wider than
        :data:`_FOLD_MAX_WIDTH`: :func:`evoxc.reduce_prod`,
        :func:`evoxc.reduce_max`, :func:`evoxc.reduce_min`.

    Returns
    -------
    ``(P, N) float32``
        Byte-identical to ``reduce_prod`` / ``reduce_max`` / ``reduce_min``
        over the same axis.

    Notes
    -----
    Unlike the sum, these three reductions are *not* pairwise on the reference
    backend: ``np.prod`` / ``np.max`` / ``np.min`` accumulate left to right at
    every width, so a left fold starting at the neutral element is the whole
    recipe (verified bit for bit against ``np.prod`` / ``np.max`` / ``np.min``
    for ``K`` = 1, 2, 3, 7, 8, 9, 15, 16, 17, 31, 32, 64, 96, 127, 128, 129,
    130, 200, 256, 300 on ``+-0.0``, ``NaN`` and ``+-inf`` data). Starting at
    the neutral element is exact for all three: ``1.0 * -0.0`` is ``-0.0``,
    and ``maximum(-inf, a)`` / ``minimum(+inf, a)`` are ``a`` -- ``NaN``
    included, which is why the padding and the identity can share one
    constant.

    A window wider than :data:`_FOLD_MAX_WIDTH` returns
    ``reduce_op(x, axes=-1)`` instead: the fold is one op per plane, so a wide
    window would emit thousands of unrolled ops for no accuracy gain, and the
    backend reduce is the same arithmetic at any width. The guard is a
    trace-time ``if`` on the static window width, never ``cond``.
    """
    if int(x.shape[2]) > _FOLD_MAX_WIDTH:
        return reduce_op(x, axes=-1)
    planes = _planes(x)
    acc = binary(identity, planes[0])
    for plane in planes[1:]:
        acc = binary(acc, plane)
    return evoxc.reshape(acc, (int(x.shape[0]), int(x.shape[1])))


class _Window:
    """Memo of the shared masked reductions of one ``(z, mask)`` incoming window.

    Trace-time scaffolding only: it owns no input data (``z`` and ``mask`` stay
    borrowed), and each accessor memoises the one graph value it derives, so the
    options of one dispatch share the neutral-padded tensor, the live-edge
    count, the zero-padded total and the scalar constants instead of rebuilding
    them per option. Nothing here is data dependent -- reuse is exact, not
    approximate:

    * :meth:`padded` returns ``select(mask, z, neutral)``, byte for byte the
      expression every kernel documents for its own neutral element, so sharing
      it between :func:`sum`, :func:`mean` and :func:`maxabs` (all neutral
      ``0.0``) cannot change a result.
    * :meth:`total` returns the ``K``-plane fold of ``padded(0.0)``
      (:func:`_sum_last`, byte-identical to ``reduce_sum(padded(0.0), -1)``),
      the body shared by :func:`sum` and :func:`mean`.
    * :meth:`count` returns the same fold of ``cast(mask, float32)``, the count
      :func:`mean` divides by and the ``has_valid`` guard compares against.
    * :meth:`scalar` returns a 0-d float32 constant, so every literal the
      dispatch needs (the ``0.0`` of the padding *and* of the ``else`` / guard
      values, the ``1.0`` of :func:`product`'s neutral and of :func:`mean`'s
      divisor floor) is one graph constant instead of one per use site --
      ``evoxc`` has no common-subexpression elimination for constants.

    Parameters
    ----------
    z : ``(P, N, K) float32``
        Contribution of every incoming edge of every node.
    mask : ``(P, N, K) bool``
        ``True`` for real incoming edges.
    scalars : mapping of ``float`` to ``() float32``, optional
        Loop-invariant rank-0 constants prepared once per forward pass
        (``genome.step._PreparedScalars.mapping``); ``None`` builds them
        locally. They seed the :meth:`scalar` memo, so a neutral element the
        pass already materialised (``0.0``/``1.0``) costs no ``constant`` op
        here, and every other literal is still built and memoised per window.
    """

    __slots__ = ("_z", "_mask", "_padded", "_count", "_total", "_scalars")

    def __init__(self, z: Any, mask: Any, scalars: Any = None) -> None:
        self._z = z
        self._mask = mask
        self._padded: dict[float, Any] = {}
        self._count: Any = None
        self._total: Any = None
        self._scalars: dict[float, Any] = dict(scalars) if scalars else {}

    def scalar(self, value: float):
        """0-d ``float32`` constant, built once per distinct ``value``.

        Parameters
        ----------
        value : float
            The literal to materialise (``0.0``, ``1.0``, ``-inf``, ``+inf``
            appear as neutral elements, ``else`` values and guard bounds).

        Returns
        -------
        ``() float32``
            A scalar tensor, broadcast by every consumer that needs a full
            ``(P, N)`` or ``(P, N, K)`` operand.

        Notes
        -----
        A Python literal passed straight to ``evoxc.select`` / ``evoxc.greater``
        becomes its own ``constant`` op at every call site, and ``evoxc`` does
        not merge them, so routing every literal of one dispatch through this
        memo is what keeps a multi-option dispatch from emitting the same
        ``0.0`` four times.

        A window built by a prepared forward pass is seeded from that pass's
        rank-0 constants (:class:`_Window`'s ``scalars``), so a ``0.0`` or
        ``1.0`` the pass already materialised resolves to its tensor and costs
        no op at all; every other value is built here once and memoised.
        """
        key = float(value)
        cached = self._scalars.get(key)
        if cached is None:
            cached = enp.full((), value, dtype=evoxc.float32)
            self._scalars[key] = cached
        return cached

    def padded(self, neutral: float):
        """``select(mask, z, neutral)``, built once per distinct ``neutral``.

        Parameters
        ----------
        neutral : float
            The reduction's neutral element (``0.0``, ``1.0``, ``-inf``,
            ``+inf``). ``0.0`` and ``-0.0`` share one tensor, which is safe
            because they are the same float.

        Returns
        -------
        ``(P, N, K) float32``
            ``z`` where ``mask`` is ``True`` and ``neutral`` elsewhere.
        """
        key = float(neutral)
        cached = self._padded.get(key)
        if cached is None:
            cached = evoxc.select(self._mask, self._z, self.scalar(neutral))
            self._padded[key] = cached
        return cached

    def total(self):
        """``reduce_sum(padded(0.0), -1)``, built once.

        Returns
        -------
        ``(P, N) float32``
            Sum of the masked-in contributions of every node slot, ``0.0`` for
            an all-``False`` column.

        Notes
        -----
        :func:`sum` returns it as is and :func:`mean` divides it by
        :meth:`count`, so one reduction of the window serves both options of a
        dispatch instead of one each. It is emitted as the static ``K``-plane
        fold of :func:`_sum_last` for ``K <= _FOLD_MAX_WIDTH`` -- numpy's
        ``axis=-1`` sum is the most expensive op of a forward-pass level, and
        the fold reproduces its result bit for bit -- and as the backend
        ``reduce_sum`` above that width, where numpy's own association order
        changes.
        """
        if self._total is None:
            self._total = _sum_last(self.padded(0.0), self.scalar(0.0))
        return self._total

    def count(self):
        """``reduce_sum(cast(mask, float32), -1)``, built once.

        Returns
        -------
        ``(P, N) float32``
            Number of live incoming edges of every node slot.

        Notes
        -----
        Folded like :meth:`total` (and falling back to ``reduce_sum`` above
        :data:`_FOLD_MAX_WIDTH`). The summands are exact integers (``0.0`` /
        ``1.0``, ``K`` far below ``2**24``), so every association order gives
        the same bit pattern and either path is byte-identical by construction.
        """
        if self._count is None:
            self._count = _sum_last(
                evoxc.cast(self._mask, evoxc.float32), self.scalar(0.0)
            )
        return self._count


def _sum_body(window: _Window):
    """Body of :func:`sum` (see its docstring for the contract)."""
    return window.total()


def sum(z, mask, scalars=None):
    """Masked sum over the incoming-edge axis.

    Parameters
    ----------
    z : ``(P, N, K) float32``
        Contribution of every incoming edge of every node.
    mask : ``(P, N, K) bool``
        ``True`` for real incoming edges.
    scalars : mapping of ``float`` to ``() float32``, optional
        Loop-invariant rank-0 constants prepared once per forward pass; ``None``
        builds them locally.

    Returns
    -------
    ``(P, N) float32``
        ``0.0`` where nothing is masked in (neutral element of the sum).

    Notes
    -----
    The ``K``-plane fold of ``select(mask, z, 0.0)`` (:func:`_sum_last`),
    byte-identical to ``reduce_sum(select(mask, z, 0.0), axes=-1)`` -- which is
    what matches the pre-rewrite ``jnp.sum(z, axis=0, where=mask, initial=0)``;
    a ``NaN`` contribution is *not* skipped (it poisons the node) because the
    mask, not ``isnan``, decides.
    An all-``False`` column is therefore already ``0.0`` (``zero_on_empty``), so
    the dispatcher needs no ``has_valid`` guard for this option.
    The padding and the fold's seed are the *same* ``0.0``, so a prepared pass
    supplies one shared constant for both.
    """
    return _sum_body(_Window(z, mask, scalars))


sum.window_body = _sum_body
sum.zero_on_empty = True


def _product_body(window: _Window):
    """Body of :func:`product` (see its docstring for the contract)."""
    return _fold_last(
        window.padded(1.0), evoxc.multiply, window.scalar(1.0), evoxc.reduce_prod
    )


def product(z, mask, scalars=None):
    """Masked product over the incoming-edge axis.

    Parameters
    ----------
    z : ``(P, N, K) float32``
        Contribution of every incoming edge of every node.
    mask : ``(P, N, K) bool``
        ``True`` for real incoming edges.
    scalars : mapping of ``float`` to ``() float32``, optional
        Loop-invariant rank-0 constants prepared once per forward pass; ``None``
        builds them locally.

    Returns
    -------
    ``(P, N) float32``
        ``1.0`` where nothing is masked in (neutral element of the product);
        :func:`apply_aggregation` rewrites that case to ``0.0``.

    Notes
    -----
    The ``K``-plane fold of ``select(mask, z, 1.0)`` (:func:`_fold_last`),
    byte-identical to ``reduce_prod(select(mask, z, 1.0), axes=-1)``. The empty
    column is ``1.0``, not ``0.0``, so this kernel is *not* ``zero_on_empty``
    and a dispatch that includes it keeps the ``has_valid`` guard.
    The neutral padding and the fold's identity are the same ``1.0``, so a
    prepared pass supplies one shared constant for both.
    """
    return _product_body(_Window(z, mask, scalars))


product.window_body = _product_body
product.zero_on_empty = False


def _max_body(window: _Window):
    """Body of :func:`max` (see its docstring for the contract)."""
    neutral = window.scalar(float("-inf"))
    return _fold_last(
        window.padded(float("-inf")), evoxc.maximum, neutral, evoxc.reduce_max
    )


def max(z, mask, scalars=None):
    """Masked maximum over the incoming-edge axis.

    Parameters
    ----------
    z : ``(P, N, K) float32``
        Contribution of every incoming edge of every node.
    mask : ``(P, N, K) bool``
        ``True`` for real incoming edges.
    scalars : mapping of ``float`` to ``() float32``, optional
        Loop-invariant rank-0 constants prepared once per forward pass; ``None``
        builds them locally.

    Returns
    -------
    ``(P, N) float32``
        ``-inf`` where nothing is masked in (neutral element of the maximum);
        :func:`apply_aggregation` rewrites that case to ``0.0``.

    Notes
    -----
    The ``K``-plane fold of ``select(mask, z, -inf)`` (:func:`_fold_last`),
    byte-identical to ``reduce_max(select(mask, z, -inf), axes=-1)``; the
    ``-inf`` is a scalar constant that ``evoxc.select`` weak-promotes to the
    tensor's float32 dtype, shared with the fold's starting value. Not
    ``zero_on_empty`` (an empty column is ``-inf``). A prepared pass supplies
    that one constant instead of building it per level.
    """
    return _max_body(_Window(z, mask, scalars))


max.window_body = _max_body
max.zero_on_empty = False


def _min_body(window: _Window):
    """Body of :func:`min` (see its docstring for the contract)."""
    neutral = window.scalar(float("inf"))
    return _fold_last(
        window.padded(float("inf")), evoxc.minimum, neutral, evoxc.reduce_min
    )


def min(z, mask, scalars=None):
    """Masked minimum over the incoming-edge axis.

    Parameters
    ----------
    z : ``(P, N, K) float32``
        Contribution of every incoming edge of every node.
    mask : ``(P, N, K) bool``
        ``True`` for real incoming edges.
    scalars : mapping of ``float`` to ``() float32``, optional
        Loop-invariant rank-0 constants prepared once per forward pass; ``None``
        builds them locally.

    Returns
    -------
    ``(P, N) float32``
        ``+inf`` where nothing is masked in (neutral element of the minimum);
        :func:`apply_aggregation` rewrites that case to ``0.0``.

    Notes
    -----
    The ``K``-plane fold of ``select(mask, z, +inf)`` (:func:`_fold_last`),
    byte-identical to ``reduce_min(select(mask, z, +inf), axes=-1)``. Not
    ``zero_on_empty`` (an empty column is ``+inf``). A prepared pass supplies
    the one ``+inf`` constant instead of building it per level.
    """
    return _min_body(_Window(z, mask, scalars))


min.window_body = _min_body
min.zero_on_empty = False


def _maxabs_body(window: _Window):
    """Body of :func:`maxabs` (see its docstring for the contract)."""
    zeroed = window.padded(0.0)
    magnitudes = evoxc.abs(zeroed)
    column = evoxc.cast(evoxc.argmax(magnitudes, axis=-1), evoxc.int32)
    pop_size, num_nodes, width = zeroed.shape
    flat = evoxc.reshape(zeroed, (pop_size * num_nodes, width))
    picked = flat_gather(flat, evoxc.reshape(column, (pop_size * num_nodes,)))
    return evoxc.reshape(picked, (pop_size, num_nodes))


def maxabs(z, mask, scalars=None):
    """Signed contribution with the largest magnitude, over the incoming-edge axis.

    Parameters
    ----------
    z : ``(P, N, K) float32``
        Contribution of every incoming edge of every node.
    mask : ``(P, N, K) bool``
        ``True`` for real incoming edges.
    scalars : mapping of ``float`` to ``() float32``, optional
        Loop-invariant rank-0 constants prepared once per forward pass; ``None``
        builds them locally.

    Returns
    -------
    ``(P, N) float32``
        ``z[p, n, j]`` for the *first* ``j`` maximizing ``|z[p, n, j]|`` among
        masked-in slots (``0.0`` when nothing is masked in).

    Notes
    -----
    Legacy-exact: the pre-rewrite kernel zeroed masked-out slots, took
    ``argmax(|z|)`` and returned ``z[argmax]`` -- the **signed** value, with
    numpy's first-maximum tie-break. Reducing ``|z|`` instead would return a
    magnitude and flip the sign of every negative-dominant node, so the kernel
    is ``argmax`` + one flat gather::

        zeroed = select(mask, z, 0.0)                                # (P, N, K)
        col    = cast(argmax(abs(zeroed), axis=-1), int32)           # (P, N), first max
        flat   = reshape(zeroed, (P * N, K))                         # (P*N, K)
        val    = flat_gather(flat, reshape(col, (P * N,)))           # (P*N,)
        return reshape(val, (P, N))

    The gather reads the *zeroed* tensor, not ``z``: an all-``False`` column then
    yields its neutral element ``0.0`` instead of an arbitrary (possibly ``NaN``)
    padding slot, exactly like the pre-rewrite ``z[argmax]`` after zeroing.
    Two extra ops beyond the magnitude variant, both ``O(P * N * K)`` gathers.
    ``zero_on_empty``: an empty column is already ``0.0``. A prepared pass
    supplies the ``0.0`` padding instead of building it per level.
    """
    return _maxabs_body(_Window(z, mask, scalars))


maxabs.window_body = _maxabs_body
maxabs.zero_on_empty = True


def _mean_body(window: _Window):
    """Body of :func:`mean` (see its docstring for the contract)."""
    floor = evoxc.maximum(window.count(), window.scalar(1.0))
    return evoxc.divide(window.total(), floor)


def mean(z, mask, scalars=None):
    """Masked arithmetic mean over the incoming-edge axis.

    Parameters
    ----------
    z : ``(P, N, K) float32``
        Contribution of every incoming edge of every node.
    mask : ``(P, N, K) bool``
        ``True`` for real incoming edges.
    scalars : mapping of ``float`` to ``() float32``, optional
        Loop-invariant rank-0 constants prepared once per forward pass; ``None``
        builds them locally.

    Returns
    -------
    ``(P, N) float32``
        ``reduce_sum(select(mask, z, 0.0), -1) / maximum(count, 1.0)`` with
        ``count = reduce_sum(cast(mask, float32), -1)``, i.e. ``0.0`` where
        nothing is masked in (never ``NaN``).

    Notes
    -----
    The pre-rewrite kernel divided by the raw mask count with no guard, so an
    all-``False`` mask produced ``NaN``; correctness there relied on the
    ``all(~mask) -> 0.0`` short-circuit in the dispatcher. The ``maximum(count, 1)``
    floor reproduces that value without relying on the guard, which makes the
    kernel safe to call standalone -- and is why it is ``zero_on_empty``.
    The zero-padded sum it divides and the count it floors are both shared with
    the other options of the same dispatch (see :class:`_Window`), so a
    dispatch holding :func:`sum` and :func:`mean` reduces the ``(P, N, K)``
    window once, not twice. A prepared pass supplies the ``0.0`` sum seed and
    the ``1.0`` divisor floor as its shared rank-0 constants.
    """
    return _mean_body(_Window(z, mask, scalars))


mean.window_body = _mean_body
mean.zero_on_empty = True


AGG_FUNCS = (sum, product, max, min, maxabs, mean)
"""Canonical AGG option order -- part of the gene encoding, do not reorder.

Appending is safe; reordering or deleting changes sampled aggregation ids and
checkpoint compatibility.
"""


def apply_aggregation(
    z, mask, agg_ids, agg_fns: Sequence[Callable] = AGG_FUNCS
):
    """Apply a per-node aggregation selected by gene id, without ``lax.switch``.

    Parameters
    ----------
    z : ``(P, N, K) float32``
        Contributions of the ``K`` incoming-edge slots of every node slot.
    mask : ``(P, N, K) bool``
        ``True`` for real incoming edges (see the module docstring).
    agg_ids : ``(P, N) float32`` or ``(P, N) int32``
        Per-node aggregation ids; integral-valued float32 gene columns are cast
        to int32, an already-int32 id column is used as is (no traced cast).
        There is no ``-1`` sentinel for aggregation (the pre-rewrite library had
        none).
    agg_fns : sequence of callables, optional
        The node gene's ``aggregation_options`` -- a static, trace-time Python
        tuple of ``(z, mask) -> (P, N)`` kernels. Defaults to :data:`AGG_FUNCS`
        (the global registry order).

    Returns
    -------
    ``(P, N) float32``
        ``agg_fns[id](z, mask)`` per node, and exactly ``0.0`` for a node whose
        mask column is entirely ``False``.

    Notes
    -----
    ``evoxc.cond`` needs a 0-d predicate, so the dispatch is an elementwise
    ``select`` cascade: every option is evaluated on the whole ``(P, N, K)``
    tensor and selected per node with ``select(agg_ids == i, ...)``. The cost is
    one ``constant``/``equal``/``select`` triple over ``(P, N)`` per option plus
    its reductions over ``(P, N, K)`` -- at most one per option, fewer when
    options share one. The padding, the live-edge count and the scalar literals
    are *not* per option:

    * every kernel that shares a neutral element shares one
      ``select(mask, z, neutral)`` through :class:`_Window` (:func:`sum`,
      :func:`mean` and :func:`maxabs` all pad with ``0.0``),
    * :func:`sum` and :func:`mean` share one ``(P, N, K)`` ``reduce_sum``, and
    * :func:`mean` and the ``has_valid`` guard share one live-edge count.

    Every scalar the cascade needs -- the ``0.0`` of the first ``select``'s
    ``else`` value and of the guard's bound and rewrite, the ``1.0`` of
    :func:`product`'s neutral and of :func:`mean`'s divisor floor -- is one
    ``constant`` op per distinct value through :meth:`_Window.scalar`
    (``evoxc`` does not eliminate duplicate constants).

    Measured graph ops at ``(P, N, K) = (100, 50, 8)`` int32 ids: 7 for a
    one-option :func:`sum` list, 12 for one-option :func:`max` (its extra four
    are the ``has_valid`` guard and the live-edge count), and 57 for the
    canonical list.

    The ``has_valid`` guard (a final ``select(count > 0, result, 0.0)``) is
    emitted only when some option is not ``zero_on_empty`` -- i.e. when a
    padded :func:`product` / :func:`max` / :func:`min` would leak ``1.0`` /
    ``-inf`` / ``+inf`` into a node with no computable incoming edge. A list of
    ``zero_on_empty`` kernels (:func:`sum`, :func:`mean`, :func:`maxabs`) and a
    single-option list of one of them need no guard at all: every option already
    yields exactly ``0.0`` there. A custom kernel without the flag counts as
    "needs the guard", so an unannotated option never changes a result.

    An id outside ``[0, len(agg_fns))`` yields ``0.0`` (the first ``select``'s
    ``else`` value); the pre-rewrite ``lax.switch`` clamped such ids to the last
    option, an unreachable divergence for well-formed genomes. That ``else``
    value is why a single-option list still pays its ``select``: the id column
    is gene data, so the option cannot be selected statically even when the list
    has one entry.
    """
    ids = agg_ids if agg_ids.dtype == evoxc.int32 else evoxc.cast(agg_ids, evoxc.int32)
    window = _Window(z, mask)
    out = None
    for index, fn in enumerate(agg_fns):
        body = getattr(fn, "window_body", None)
        value = body(window) if body is not None else fn(z, mask)
        # The first ``select`` carries the ``else`` value: an id outside the
        # option list keeps exactly ``0.0``, the value the pre-rewrite cascade's
        # zeroed accumulator produced (no separate ``enp.zeros`` accumulator op).
        out = evoxc.select(
            evoxc.equal(ids, index), value, window.scalar(0.0) if out is None else out
        )
    if out is None:  # an empty option list: every node aggregates to 0.0
        return enp.zeros(z.shape[:-1], dtype=evoxc.float32)
    if all(getattr(fn, "zero_on_empty", False) for fn in agg_fns):
        return out
    # ``has_valid``: a node with no real incoming edge gets exactly ``0.0``, so
    # the ``-inf`` / ``+inf`` neutral padding of ``max`` / ``min`` cannot leak
    # into a node value (and a padded product cannot become ``1``).
    zero = window.scalar(0.0)
    return evoxc.select(evoxc.greater(window.count(), zero), out, zero)
