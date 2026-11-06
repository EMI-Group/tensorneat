"""Host-side configuration validation shared by ``NEAT`` and ``SpeciesController``.

Every helper runs before anything is traced, so a bad configuration fails with a
named :class:`~tensorneat.common.errors.TensorNEATError` instead of an
``evoxc`` shape/dtype error deep inside a trace.
They live here rather than in ``species.py``/``neat.py`` because both modules
validate the same kinds of knob (counts, rates, thresholds) and a duplicated
validator is how the two surfaces drift apart.
Nothing in this module touches ``evoxc`` tensors: the arguments are Python
configuration values, and the return values are plain ``int``/``float``.
"""
from __future__ import annotations

import operator
from typing import Any

from tensorneat.common.errors import TensorNEATError

__all__ = [
    "DISTANCE_CENTRE_KEY",
    "DISTANCE_CENTRE_MODES",
    "distance_centre_mode",
    "non_negative_int",
    "positive_float",
    "positive_int",
    "unit_interval",
]

#: State entry selecting the width of every speciation distance centre payload.
#:
#: ``"tiled"`` hands the genome a population-shaped centre (the row repeated
#: ``pop_size`` times), ``"single"`` a one-individual one (leading axis ``1``).
#: The entry is a plain Python string on the trace-time configuration ``State``
#: and outranks nothing else: a bare ``State`` gets ``"tiled"``.
DISTANCE_CENTRE_KEY = "distance_centre"

#: Accepted values of :data:`DISTANCE_CENTRE_KEY`.
DISTANCE_CENTRE_MODES = ("tiled", "single")


def distance_centre_mode(state: Any) -> str:
    """Resolve :data:`DISTANCE_CENTRE_KEY` from the trace-time ``State``.

    Args:
        state (Any): The configuration ``State`` the speciation call received;
            the entry is a host value, never a traced one.

    Returns:
        str: ``"tiled"`` when the entry is absent, else the stored mode.

    Raises:
        TensorNEATError: If the entry holds anything but the two documented
            modes, so a typo fails loudly instead of silently picking a width.
    """
    mode = getattr(state, DISTANCE_CENTRE_KEY, DISTANCE_CENTRE_MODES[0])
    if mode not in DISTANCE_CENTRE_MODES:
        raise TensorNEATError(
            f"state.{DISTANCE_CENTRE_KEY} must be one of "
            f"{DISTANCE_CENTRE_MODES}, got {mode!r}"
        )
    return mode


def positive_int(value: Any, name: str) -> int:
    """Validate a strictly positive integer configuration value.

    Args:
        value: The candidate value; anything :func:`operator.index` accepts (a
            Python ``int`` or a NumPy integer) is allowed.
        name (str): Parameter name, quoted in the error message.

    Returns:
        int: ``value`` as a plain Python ``int``.

    Raises:
        TensorNEATError: If ``value`` is a ``bool``, not integral, or not
            positive.  ``bool`` is rejected on purpose even though it is an
            ``int`` subclass: ``pop_size=True`` is a typo, not a size of one.
    """
    if isinstance(value, bool):
        raise TensorNEATError(f"{name} must be a positive int, got {value!r}")
    try:
        integer = operator.index(value)
    except TypeError:
        raise TensorNEATError(
            f"{name} must be a positive int, got {value!r} "
            f"({type(value).__name__})"
        ) from None
    if integer <= 0:
        raise TensorNEATError(f"{name} must be a positive int, got {value!r}")
    return int(integer)


def non_negative_int(value: Any, name: str) -> int:
    """Validate an integer configuration value that may be zero.

    Args:
        value: The candidate value.
        name (str): Parameter name, quoted in the error message.

    Returns:
        int: ``value`` as a plain Python ``int``.

    Raises:
        TensorNEATError: If ``value`` is a ``bool``, not integral, or negative.
    """
    if isinstance(value, bool):
        raise TensorNEATError(f"{name} must be a non-negative int, got {value!r}")
    try:
        integer = operator.index(value)
    except TypeError:
        raise TensorNEATError(
            f"{name} must be a non-negative int, got {value!r} "
            f"({type(value).__name__})"
        ) from None
    if integer < 0:
        raise TensorNEATError(f"{name} must be a non-negative int, got {value!r}")
    return int(integer)


def unit_interval(value: Any, name: str, allow_zero: bool = True) -> float:
    """Validate a rate that must live in ``[0, 1]`` or ``(0, 1]``.

    Args:
        value: The candidate value.
        name (str): Parameter name, quoted in the error message.
        allow_zero (bool, optional): Whether ``0.0`` is legal; defaults to
            ``True``.  ``survival_threshold`` disallows it because it multiplies
            a member count that must yield at least one survivor.

    Returns:
        float: ``value`` as a Python ``float``.

    Raises:
        TensorNEATError: If ``value`` is not a real number inside the interval.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TensorNEATError(f"{name} must be a float in [0, 1], got {value!r}")
    number = float(value)
    lower_ok = number >= 0.0 if allow_zero else number > 0.0
    if not lower_ok or number > 1.0:
        interval = "[0, 1]" if allow_zero else "(0, 1]"
        raise TensorNEATError(f"{name} must be a float in {interval}, got {value!r}")
    return number


def positive_float(value: Any, name: str) -> float:
    """Validate a strictly positive, finite ``float`` configuration value.

    Args:
        value: The candidate value.
        name (str): Parameter name, quoted in the error message.

    Returns:
        float: ``value`` as a Python ``float``.

    Raises:
        TensorNEATError: If ``value`` is not a finite positive number.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TensorNEATError(f"{name} must be a positive float, got {value!r}")
    number = float(value)
    if not number > 0.0 or number != number or number == float("inf"):
        raise TensorNEATError(f"{name} must be a finite positive float, got {value!r}")
    return number
