"""Host-side helpers shared by the evoxc reinforcement-learning environments.

This module is the small, genuinely generic part of the environment recipe: the
``evoxc.Tensor`` accessor ``show`` needs, the constructor validators every
environment repeats verbatim, and the ``(B,) -> (P, R)`` return layout.

Two rules keep it honest:

* **No host-only logic that is actually environment algebra.** Dynamics, reward
  and reset expressions belong to the environment that owns the kernel, never
  here, so adding an environment never edits this file.
* **No module-scope constant.** ``enp.*`` creators are graph constructors, so a
  call at import time would materialize a concrete array outside a trace (and
  would break ``import tensorneat``); they are only legal inside a helper that
  the caller invokes while a trace is active.

``returns_of`` is therefore the one traced helper here -- it is inlined into the
caller's graph exactly like ``problem.base.finalize_fitness`` -- while every
other name runs on the host only.
"""

from typing import Any

import numpy as np

import evoxc
import evoxc.numpy as enp

from ...common.errors import TensorNEATError

__all__ = [
    "require_non_negative",
    "require_positive",
    "require_positive_int",
    "returns_of",
    "to_numpy",
]


def to_numpy(value: Any) -> np.ndarray:
    """Return a host numpy view of a concrete ``evoxc`` tensor or an array.

    Host-side helper for ``show`` only: a concrete ``evoxc.Tensor`` exposes its
    data through ``numpy()`` (zero-copy for the numpy-backed reference
    backend), while everything else is coerced with ``np.asarray``.

    Args:
        value: An ``evoxc.Tensor``, a numpy array, or a numpy scalar.

    Returns:
        A numpy array (or 0-d array) holding the same values.
    """
    if isinstance(value, evoxc.Tensor):
        return value.numpy()
    return np.asarray(value)


def require_positive_int(owner: str, name: str, value: Any) -> int:
    """Validate a strictly positive integer configuration value.

    Args:
        owner: Environment name, used as the message prefix.
        name: Configuration name, used in the error message.
        value: The value to validate; a Python or numpy integer.

    Returns:
        The value as a Python ``int``.

    Raises:
        TensorNEATError: If ``value`` is not an ``int`` or is not positive.
            ``bool`` is rejected explicitly: ``True`` is an ``int`` in Python
            but never a meaningful horizon or repeat count.
    """
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TensorNEATError(f"{owner}: {name} must be an int, got {value!r}")
    number = int(value)
    if number <= 0:
        raise TensorNEATError(f"{owner}: {name} must be positive, got {number}")
    return number


def require_positive(owner: str, name: str, value: Any) -> float:
    """Validate a strictly positive scalar configuration value.

    The comparison is ``number > 0.0`` rather than ``number <= 0.0`` so a NaN
    is rejected as well.

    Args:
        owner: Environment name, used as the message prefix.
        name: Configuration name, used in the error message.
        value: The value to validate.

    Returns:
        The value as a Python ``float``.

    Raises:
        TensorNEATError: If ``value`` is not positive (or is NaN).
    """
    number = float(value)
    if not number > 0.0:
        raise TensorNEATError(f"{owner}: {name} must be positive, got {number}")
    return number


def require_non_negative(owner: str, name: str, value: Any) -> float:
    """Validate a non-negative scalar configuration value.

    Args:
        owner: Environment name, used as the message prefix.
        name: Configuration name, used in the error message.
        value: The value to validate.

    Returns:
        The value as a Python ``float``.

    Raises:
        TensorNEATError: If ``value`` is negative (a NaN passes, as in the
            reference ``CartPole`` validators).
    """
    number = float(value)
    if number < 0.0:
        raise TensorNEATError(f"{owner}: {name} must be non-negative, got {number}")
    return number


def returns_of(total: Any, pop_size: int, repeat_times: int) -> Any:
    """Group per-episode returns into one row of ``repeat_times`` per individual.

    Must be called inside the active trace (``enp.reshape`` is a graph op). The
    episodes are individual-major -- row ``p * R + r`` is repeat ``r`` of
    individual ``p`` -- so reshaping ``(B,)`` to ``(P, R)`` puts repeat ``r`` of
    individual ``p`` at ``[p, r]``; ``evaluate`` means over axis 1.

    Args:
        total: Per-episode returns of shape ``(B,)`` with ``B = P * R``.
        pop_size: Population size ``P`` (static).
        repeat_times: Repeats per individual ``R`` (static).

    Returns:
        ``(P, R)`` returns.
    """
    return enp.reshape(total, (pop_size, repeat_times))
