"""Host-side helpers shared by the evoxc reinforcement-learning environments.

This module is the small, genuinely generic part of the environment recipe: the
``evoxc.Tensor`` accessor ``show`` needs, the constructor validators every
environment repeats verbatim, the fixed-horizon trip condition every rollout loop
shares, and the ``(B,) -> (P, R)`` return layout.

Every validator is a **configuration** check, so it raises
``TensorNEATError`` and nothing else: a non-numeric knob (a string, a list, a
``None``) and a non-finite one ``NaN``/``+/-inf`` are rejected by
:func:`require_finite` before any comparison runs, because ``NaN`` compares
false against everything and would otherwise silently pass a ``<``/``>`` gate.
That matters numerically here: a ``NaN`` physics knob poisons every episode
through the dynamics, and a ``NaN`` ``fitness_target`` never terminates a run,
so both are refused at construction instead of surfacing as ``-inf`` fitness on
every individual.

Two rules keep it honest:

* **No host-only logic that is actually environment algebra.** Dynamics, reward
  and reset expressions belong to the environment that owns the kernel, never
  here, so adding an environment never edits this file.
* **No module-scope constant.** ``enp.*`` creators are graph constructors, so a
  call at import time would materialize a concrete array outside a trace (and
  would break ``import tensorneat``); they are only legal inside a helper that
  the caller invokes while a trace is active.

``horizon_cond``, ``returns_of`` and ``prepared_query`` are therefore the three
traced helpers here -- they are inlined into the caller's graph exactly like
``problem.base.finalize_fitness`` -- while every other name runs on the host only.
"""

import math
from typing import Any, Callable

import numpy as np

import evoxc
import evoxc.numpy as enp

from ...common.errors import TensorNEATError

__all__ = [
    "horizon_cond",
    "prepared_query",
    "require_finite",
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


def require_finite(owner: str, name: str, value: Any) -> float:
    """Validate that a scalar configuration value is a finite number.

    This is the base of every float validator here: it owns the conversion, so
    a knob that is not a number at all (``None``, a string, a sequence) raises
    the same ``TensorNEATError`` as one that is a number but not finite
    (``NaN``, ``+inf``, ``-inf``), instead of escaping as a bare ``TypeError``
    or ``ValueError``. ``NaN`` in particular must be rejected: it compares
    false against every ``<``/``>`` gate, so it would silently pass a range
    check and then poison the dynamics (or make a ``fitness_target``
    unreachable) with no diagnostic at all.

    Args:
        owner: Environment name, used as the message prefix.
        name: Configuration name, used in the error message.
        value: The value to validate; anything ``float`` can convert, except
            ``bool`` (``True`` is an ``int`` in Python but never a physical
            constant).

    Returns:
        The value as a Python ``float``.

    Raises:
        TensorNEATError: If ``value`` is a ``bool``, cannot be converted to a
            ``float``, or converts to ``NaN``/``+inf``/``-inf``.
    """
    message = f"{owner}: {name} must be a finite number, got {value!r}"
    if isinstance(value, bool):
        raise TensorNEATError(message)
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise TensorNEATError(message) from exc
    if not math.isfinite(number):
        raise TensorNEATError(message)
    return number


def require_positive(owner: str, name: str, value: Any) -> float:
    """Validate a strictly positive, finite scalar configuration value.

    Args:
        owner: Environment name, used as the message prefix.
        name: Configuration name, used in the error message.
        value: The value to validate.

    Returns:
        The value as a Python ``float``.

    Raises:
        TensorNEATError: If ``value`` is not a finite number (see
            :func:`require_finite`), or is not strictly positive.
    """
    number = require_finite(owner, name, value)
    if not number > 0.0:
        raise TensorNEATError(f"{owner}: {name} must be positive, got {number}")
    return number


def require_non_negative(owner: str, name: str, value: Any) -> float:
    """Validate a non-negative, finite scalar configuration value.

    Args:
        owner: Environment name, used as the message prefix.
        name: Configuration name, used in the error message.
        value: The value to validate.

    Returns:
        The value as a Python ``float``.

    Raises:
        TensorNEATError: If ``value`` is not a finite number (see
            :func:`require_finite`), or is negative. A ``NaN`` is rejected by
            the finiteness check, so ``0.0`` is the only admitted zero.
    """
    number = require_finite(owner, name, value)
    if number < 0.0:
        raise TensorNEATError(f"{owner}: {name} must be non-negative, got {number}")
    return number


def horizon_cond(consts: Any, step_index: int = 3) -> Callable:
    """Build the trip condition of an environment's rollout loop.

    Must be called inside the active trace (``enp.less`` is a graph op), exactly
    like :func:`returns_of`. The horizon is static config, so every rollout runs
    exactly ``max_step`` iterations: the condition counts up from ``0`` against
    ``consts.horizon`` instead of testing a terminal flag ("is any episode still
    alive"), which would make the cost depend on the population's behaviour and
    would let a ``NaN`` logit arriving after every episode finished escape the
    accumulator.

    Args:
        consts: The environment's constant bundle; its ``horizon`` field is the
            int32 ``max_step`` the counter is compared against.
        step_index (int, optional): Position of the loop counter in the carry --
            ``3`` for the four-leaf ``(obs, alive, total, step)`` carry the three
            shipped environments share, ``1`` for ``Pendulum``'s packed
            ``(row, step)`` accelerator carry. Defaults to ``3``.

    Returns:
        ``cond(carry) -> () bool`` -- true while ``step < max_step``.
    """

    def cond(carry):
        """Test the loop counter against the static horizon.

        Args:
            carry: The rollout carry; its entry ``step_index`` is the counter.

        Returns:
            The rank-0 boolean ``step < max_step``.
        """
        return enp.less(carry[step_index], consts.horizon)

    return cond


def prepared_query(act_fn: Any, batch: int) -> Any:
    """Prepare ``act_fn``'s policy pass once for the whole rollout.

    A rollout queries the same network once per timestep, so deriving the
    loop-invariant operands per query would repeat that work ``max_step`` times;
    this returns the prepared-query callable instead.
    A genome-free closure (a unit suite's policy, a user kernel) exposes no
    ``prepare`` and is returned unchanged, so ``evaluate`` keeps working with a
    plain callable.

    Args:
        act_fn: The ``act_fn`` the problem received.
        batch: Rows one query will carry, ``B = P * R`` (static).

    Returns:
        ``act_fn.prepare(batch)`` when ``act_fn`` supports it, else ``act_fn``.
    """
    if getattr(act_fn, "prepare", None) is None:
        return act_fn
    return act_fn.prepare(batch)


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
