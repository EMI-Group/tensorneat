"""Numpy reference kernels of the ``ACT`` / ``AGG`` option registries, plus dispatchers.

Written from the documented definitions in ``src/tensorneat/common/functions/CONTEXT.md``
(the canonical option order, the ``SCALE``/``LELU_SLACK``/``INV_EPS``/``LOG_EPS``
constants) — never by calling the kernels they are compared against. Plain numpy in,
plain numpy out: elementwise activations of any shape, aggregations that reduce the
last axis of a ``(P, N, K)`` contribution tensor.

Conventions shared with the gene contract:

* an option **id** is the position of the kernel in the user's option list
  (``activation_options`` / ``aggregation_options``), not a global registry index;
* ``-1`` in the activation id column means identity and is never an option index;
* dead in-edge slots are neutral-padded per option, and a node whose mask column is
  entirely ``False`` aggregates to exactly ``0.0`` (the dispatcher's ``has_valid``
  guard);
* ``sum``/``product``/``max``/``min``/``abs`` shadow builtins inside the evoxc module
  these mirror, so the kernels here carry an ``act_``/``agg_`` prefix instead.

``support.oracle`` re-exports every name below, so a test can import from either
module.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

import numpy as np

__all__ = [
    "ACT_FUNCS",
    "AGG_FUNCS",
    "act_abs",
    "act_exp",
    "act_identity",
    "act_inv",
    "act_lelu",
    "act_log",
    "act_relu",
    "act_scaled_sigmoid",
    "act_scaled_tanh",
    "act_sigmoid",
    "act_sin",
    "act_tanh",
    "agg_max",
    "agg_maxabs",
    "agg_mean",
    "agg_min",
    "agg_product",
    "agg_sum",
    "apply_activation",
    "apply_aggregation",
]


# ============================================================ activation kernels
# Canonical ``ACT`` order and constants, per ``common/functions/CONTEXT.md``. The
# kernels are elementwise, shape-polymorphic (``(...)`` in, same shape out) and let
# ``NaN`` through: ``NaN`` in, ``NaN`` out, with no masking of their own.
ACT_SCALE = 3.0
LELU_SLACK = 0.005
INV_EPS = 1e-7
LOG_EPS = 1e-7


def act_scaled_sigmoid(z: Any) -> np.ndarray:
    """``SCALE / (1 + exp(-z))`` with ``SCALE = 3.0`` (canonical ``ACT`` id 0)."""
    z = np.asarray(z, dtype=np.float64)
    return ACT_SCALE / (1.0 + np.exp(-z))


def act_sigmoid(z: Any) -> np.ndarray:
    """``1 / (1 + exp(-z))`` (canonical ``ACT`` id 1, the ``DefaultNode`` default)."""
    z = np.asarray(z, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-z))


def act_scaled_tanh(z: Any) -> np.ndarray:
    """``SCALE * tanh(z)`` with ``SCALE = 3.0`` (canonical ``ACT`` id 2)."""
    return ACT_SCALE * np.tanh(np.asarray(z, dtype=np.float64))


def act_tanh(z: Any) -> np.ndarray:
    """``tanh(z)`` (canonical ``ACT`` id 3)."""
    return np.tanh(np.asarray(z, dtype=np.float64))


def act_sin(z: Any) -> np.ndarray:
    """``sin(z)`` (canonical ``ACT`` id 4)."""
    return np.sin(np.asarray(z, dtype=np.float64))


def act_relu(z: Any) -> np.ndarray:
    """``maximum(z, 0)`` (canonical ``ACT`` id 5)."""
    return np.maximum(np.asarray(z, dtype=np.float64), 0.0)


def act_lelu(z: Any) -> np.ndarray:
    """``z`` for ``z > 0``, ``LELU_SLACK * z`` otherwise (canonical ``ACT`` id 6)."""
    z = np.asarray(z, dtype=np.float64)
    return np.where(z > 0, z, LELU_SLACK * z)


def act_identity(z: Any) -> np.ndarray:
    """``z`` unchanged (canonical ``ACT`` id 7; also the ``-1`` identity sentinel)."""
    return np.asarray(z, dtype=np.float64)


def act_inv(z: Any) -> np.ndarray:
    """``1 / sign-preserving-clamped z`` with ``INV_EPS = 1e-7`` (canonical ``ACT`` id 8)."""
    z = np.asarray(z, dtype=np.float64)
    return 1.0 / np.where(z > 0, np.maximum(z, INV_EPS), np.minimum(z, -INV_EPS))


def act_log(z: Any) -> np.ndarray:
    """``log(maximum(z, LOG_EPS))`` with ``LOG_EPS = 1e-7`` (canonical ``ACT`` id 9)."""
    return np.log(np.maximum(np.asarray(z, dtype=np.float64), LOG_EPS))


def act_exp(z: Any) -> np.ndarray:
    """``exp(z)``, unclipped (canonical ``ACT`` id 10; overflows above ~88)."""
    return np.exp(np.asarray(z, dtype=np.float64))


def act_abs(z: Any) -> np.ndarray:
    """``abs(z)`` (canonical ``ACT`` id 11)."""
    return np.abs(np.asarray(z, dtype=np.float64))


#: Canonical ``ACT`` option order — the order a genome gets by passing ``ACT_FUNCS``.
ACT_FUNCS: tuple[Callable[[Any], np.ndarray], ...] = (
    act_scaled_sigmoid,
    act_sigmoid,
    act_scaled_tanh,
    act_tanh,
    act_sin,
    act_relu,
    act_lelu,
    act_identity,
    act_inv,
    act_log,
    act_exp,
    act_abs,
)


def apply_activation(
    z: Any,
    act_ids: Any,
    act_fns: Sequence[Callable[[Any], np.ndarray]] = (act_sigmoid,),
) -> np.ndarray:
    """Apply the per-node activation selected by a gene id, without branching.

    Args:
        z: ``(P, N)`` (or any shape) float values to activate.
        act_ids: Same-shape integral ids *broadcastable* against ``z``, i.e. the
            position of the chosen kernel in ``act_fns``; ``-1`` means identity and
            is never an option index.
        act_fns: The node gene's static option tuple; a rank-0 id outside
            ``[0, len(act_fns))`` yields ``0.0`` (the cascade's initial value).

    Returns:
        ``z``'s shape, ``float64``: ``act_fns[id](z)`` per element, ``z`` itself
        where the id is ``-1``.

    Raises:
        ValueError: If ``act_fns`` is empty.
        TypeError: If an entry of ``act_fns`` is not callable.
    """
    fns = tuple(act_fns)
    if not fns:
        raise ValueError("act_fns must hold at least one activation")
    if not all(callable(fn) for fn in fns):
        raise TypeError("every entry of act_fns must be callable")
    z = np.asarray(z, dtype=np.float64)
    ids = np.asarray(act_ids)
    result = np.zeros_like(z)
    for index, fn in enumerate(fns):
        result = np.where(ids == index, fn(z), result)
    return np.where(ids == -1, z, result)


# =========================================================== aggregation kernels
# Each kernel reduces the last axis of a ``(P, N, K)`` contribution tensor over the
# live slots. evoxc reductions have no ``where=``/``initial=``, so the documented
# implementation pads dead slots with its neutral element; the numpy equivalent is
# ``np.where(mask, z, neutral)`` before the reduce.


def agg_sum(z: Any, mask: Any) -> np.ndarray:
    """Masked sum over the last axis; ``0.0`` where nothing is live (``AGG`` id 0)."""
    z = np.asarray(z, dtype=np.float64)
    return np.sum(np.where(np.asarray(mask), z, 0.0), axis=-1)


def agg_product(z: Any, mask: Any) -> np.ndarray:
    """Masked product over the last axis; ``1.0`` where nothing is live (``AGG`` id 1)."""
    z = np.asarray(z, dtype=np.float64)
    return np.prod(np.where(np.asarray(mask), z, 1.0), axis=-1)


def agg_max(z: Any, mask: Any) -> np.ndarray:
    """Masked maximum over the last axis; ``-inf`` where nothing is live (``AGG`` id 2)."""
    z = np.asarray(z, dtype=np.float64)
    return np.max(np.where(np.asarray(mask), z, -np.inf), axis=-1)


def agg_min(z: Any, mask: Any) -> np.ndarray:
    """Masked minimum over the last axis; ``+inf`` where nothing is live (``AGG`` id 3)."""
    z = np.asarray(z, dtype=np.float64)
    return np.min(np.where(np.asarray(mask), z, np.inf), axis=-1)


def agg_maxabs(z: Any, mask: Any) -> np.ndarray:
    """The *signed* contribution of largest magnitude, first max on a tie (``AGG`` id 4).

    The legacy kernel zeroed masked-out slots, took ``argmax(|z|)`` and returned
    ``z[argmax]`` — the signed value with numpy's first-maximum tie-break — so this is
    ``argmax`` plus a gather, not a plain ``reduce_max(|z|)``. Columns with nothing
    live return ``0.0``.
    """
    z = np.asarray(z, dtype=np.float64)
    mask = np.asarray(mask)
    magnitudes = np.where(mask, np.abs(z), 0.0)
    column = np.argmax(magnitudes, axis=-1)[..., None]
    picked = np.take_along_axis(z, column, axis=-1)[..., 0]
    return np.where(np.sum(mask, axis=-1) > 0, picked, 0.0)


def agg_mean(z: Any, mask: Any) -> np.ndarray:
    """Masked mean over the last axis, ``maximum(count, 1)`` floored (``AGG`` id 5).

    Dividing by the raw count (the pre-rewrite kernel) would give ``NaN`` for an
    all-dead column; the floor reproduces the dispatcher's ``0.0`` guard without
    needing it.
    """
    z = np.asarray(z, dtype=np.float64)
    mask = np.asarray(mask)
    count = np.sum(mask, axis=-1)
    return np.sum(np.where(mask, z, 0.0), axis=-1) / np.maximum(count, 1.0)


#: Canonical ``AGG`` option order — the order a genome gets by passing ``AGG_FUNCS``.
AGG_FUNCS: tuple[Callable[[Any, Any], np.ndarray], ...] = (
    agg_sum,
    agg_product,
    agg_max,
    agg_min,
    agg_maxabs,
    agg_mean,
)


def apply_aggregation(
    z: Any,
    mask: Any,
    agg_ids: Any,
    agg_fns: Sequence[Callable[[Any, Any], np.ndarray]] = (agg_sum,),
) -> np.ndarray:
    """Aggregate the incoming window per node with the option the gene id selects.

    Args:
        z: ``(P, N, K)`` float32 contributions of the ``K`` in-edge slots of each node.
        mask: ``(P, N, K)`` bool, ``True`` for live slots.
        agg_ids: ``(P, N)`` integral aggregation ids, i.e. positions in ``agg_fns``
            (there is no ``-1`` sentinel for aggregations).
        agg_fns: The node gene's static option tuple.

    Returns:
        ``(P, N)`` float64 aggregated values, and exactly ``0.0`` for a node whose
        mask column is entirely ``False`` (the documented ``has_valid`` guard). An id
        outside ``[0, len(agg_fns))`` yields ``0.0`` as well.

    Raises:
        ValueError: If ``agg_fns`` is empty or ``z``/``mask`` disagree in shape.
        TypeError: If an entry of ``agg_fns`` is not callable.
    """
    fns = tuple(agg_fns)
    if not fns:
        raise ValueError("agg_fns must hold at least one aggregation")
    if not all(callable(fn) for fn in fns):
        raise TypeError("every entry of agg_fns must be callable")
    z = np.asarray(z, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    if z.shape != mask.shape:
        raise ValueError(f"z and mask must have the same shape, got {z.shape} and {mask.shape}")
    ids = np.asarray(agg_ids)
    result = np.zeros(z.shape[:-1], dtype=np.float64)
    for index, fn in enumerate(fns):
        result = np.where(ids == index, fn(z, mask), result)
    return np.where(np.sum(mask, axis=-1) > 0, result, 0.0)
