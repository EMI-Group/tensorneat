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
* **Empty input is the caller's problem.** With ``K >= 1`` an all-``False``
  mask column yields the neutral element (``0``, ``1``, ``-inf``, ``+inf``),
  except :func:`mean`, whose ``maximum(count, 1.0)`` floor already gives
  ``0.0``; :func:`apply_aggregation` overrides every such column to exactly
  ``0.0``.

Note that ``sum``, ``max``, ``min`` and ``abs`` shadow the builtins inside this
module; none of them is used here (``evoxc.maximum`` / ``evoxc.reduce_sum`` are).
"""

from typing import Callable, Sequence

import evoxc

from ..tools import flat_gather


def sum(z, mask):
    """Masked sum over the incoming-edge axis.

    Parameters
    ----------
    z : ``(P, N, K) float32``
        Contribution of every incoming edge of every node.
    mask : ``(P, N, K) bool``
        ``True`` for real incoming edges.

    Returns
    -------
    ``(P, N) float32``
        ``0.0`` where nothing is masked in (neutral element of the sum).

    Notes
    -----
    ``reduce_sum(select(mask, z, 0.0), axes=-1)`` -- matches the pre-rewrite
    ``jnp.sum(z, axis=0, where=mask, initial=0)``; a ``NaN`` contribution is
    *not* skipped (it poisons the node) because the mask, not ``isnan``, decides.
    """
    raise NotImplementedError("agg.sum: evoxc kernel, implemented in the implementation phase")


def product(z, mask):
    """Masked product over the incoming-edge axis.

    Parameters
    ----------
    z : ``(P, N, K) float32``
        Contribution of every incoming edge of every node.
    mask : ``(P, N, K) bool``
        ``True`` for real incoming edges.

    Returns
    -------
    ``(P, N) float32``
        ``1.0`` where nothing is masked in (neutral element of the product).

    Notes
    -----
    ``reduce_prod(select(mask, z, 1.0), axes=-1)``.
    """
    raise NotImplementedError("agg.product: evoxc kernel, implemented in the implementation phase")


def max(z, mask):
    """Masked maximum over the incoming-edge axis.

    Parameters
    ----------
    z : ``(P, N, K) float32``
        Contribution of every incoming edge of every node.
    mask : ``(P, N, K) bool``
        ``True`` for real incoming edges.

    Returns
    -------
    ``(P, N) float32``
        ``-inf`` where nothing is masked in (neutral element of the maximum);
        :func:`apply_aggregation` rewrites that case to ``0.0``.

    Notes
    -----
    ``reduce_max(select(mask, z, -inf), axes=-1)``. ``-inf`` is built in-graph
    (``evoxc.numpy.full`` with a float ``-inf`` fill, or an ``evoxc`` scalar
    literal) -- evoxc exports no ``inf`` constant.
    """
    raise NotImplementedError("agg.max: evoxc kernel, implemented in the implementation phase")


def min(z, mask):
    """Masked minimum over the incoming-edge axis.

    Parameters
    ----------
    z : ``(P, N, K) float32``
        Contribution of every incoming edge of every node.
    mask : ``(P, N, K) bool``
        ``True`` for real incoming edges.

    Returns
    -------
    ``(P, N) float32``
        ``+inf`` where nothing is masked in (neutral element of the minimum);
        :func:`apply_aggregation` rewrites that case to ``0.0``.

    Notes
    -----
    ``reduce_min(select(mask, z, +inf), axes=-1)``.
    """
    raise NotImplementedError("agg.min: evoxc kernel, implemented in the implementation phase")


def maxabs(z, mask):
    """Signed contribution with the largest magnitude, over the incoming-edge axis.

    Parameters
    ----------
    z : ``(P, N, K) float32``
        Contribution of every incoming edge of every node.
    mask : ``(P, N, K) bool``
        ``True`` for real incoming edges.

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

        a     = select(mask, abs(z), 0.0)                            # (P, N, K)
        col   = evoxc.argmax(a, axis=-1)                             # (P, N) int64, first max
        flat  = reshape(z, (P * N, K))                               # (P*N, K)
        rows  = arange(P * N) * K                                    # (P*N,)
        val   = flat_gather(flat, rows + reshape(col, (P * N,)))     # (P, N)

    Two extra ops beyond the magnitude variant, both O(P * N * K) gathers.
    """
    raise NotImplementedError("agg.maxabs: evoxc kernel, implemented in the implementation phase")


def mean(z, mask):
    """Masked arithmetic mean over the incoming-edge axis.

    Parameters
    ----------
    z : ``(P, N, K) float32``
        Contribution of every incoming edge of every node.
    mask : ``(P, N, K) bool``
        ``True`` for real incoming edges.

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
    kernel safe to call standalone.
    """
    raise NotImplementedError("agg.mean: evoxc kernel, implemented in the implementation phase")


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
        to int32. There is no ``-1`` sentinel for aggregation (the pre-rewrite
        library had none).
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
    tensor and selected per node with ``select(agg_ids == i, ...)``. Cost is
    ``len(agg_fns)`` masked reductions plus ``len(agg_fns)`` selects -- 6 + 6
    more passes over ``(P, N, K)`` for the canonical list, on top of the
    gathers the forward pass already performs. :func:`mean` reuses the
    zero-padded tensor of :func:`sum`, so the constant factors stay small.

    The ``has_valid`` guard is a final
    ``select(reduce_sum(cast(mask, float32), -1) > 0, result, 0.0)``; it is what
    keeps a node with no computable incoming edge from absorbing ``-inf`` /
    ``+inf`` / ``NaN``. An id outside ``[0, len(agg_fns))`` yields ``0.0`` (the
    cascade's initial value); the pre-rewrite ``lax.switch`` clamped such ids to
    the last option, an unreachable divergence for well-formed genomes.
    """
    raise NotImplementedError("agg.apply_aggregation: evoxc kernel, implemented in the implementation phase")
