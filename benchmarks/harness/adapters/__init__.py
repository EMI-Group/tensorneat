"""Lazy factory for the per-implementation adapters.

This module is import-safe on both sides: nothing here imports ``jax``,
``tensorneat`` or ``evoxc`` at module scope. The implementation-specific modules
(``./old``, ``./new``) are imported *inside* the factory, after the benchmark
worker has put the tree under test on ``sys.path``.

The contract every adapter implements is documented in ``../CONTEXT.md``::

    class Adapter:
        impl: str          # "old" | "new"
        display: str       # human row label
        backend_label: str # backend actually used
        build(case, backend: str, seed: int) -> pipeline
        setup(pipeline) -> state
        prepare(pipeline, state) -> (state, compile_s, gens_consumed, step_fn)
        # step_fn(state) -> (state, float64 ndarray of shape (P,))
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol

__all__ = ["ADAPTERS", "Adapter", "get_adapter"]


class Adapter(Protocol):
    """Structural type of an implementation adapter (documentation only)."""

    impl: str
    """``"old"`` or ``"new"``; the identity written into a ``RunResult``."""

    display: str
    """Human-readable row label, e.g. ``"tensorneat-legacy (JAX)"``."""

    backend_label: str
    """Backend actually used: ``"jax"`` for the old side, else the evoxc backend."""

    def build(self, case: Any, backend: str, seed: int) -> Any:
        """Construct the pipeline for ``case`` (``backend`` is new-side only)."""

    def setup(self, pipeline: Any) -> Any:
        """Return the initial :class:`State` of ``pipeline``."""

    def prepare(
        self, pipeline: Any, state: Any
    ) -> tuple[Any, float, int, Callable[[Any], tuple[Any, Any]]]:
        """Do the one-off compile work and return the timed step function."""


def _make_old() -> Adapter:
    """Import and instantiate the frozen-JAX adapter (deferred import)."""
    from .old import OldAdapter

    return OldAdapter()


def _make_new() -> Adapter:
    """Import and instantiate the evoxc adapter (deferred import)."""
    from .new import NewAdapter

    return NewAdapter()


ADAPTERS: dict[str, Callable[[], Adapter]] = {"old": _make_old, "new": _make_new}
"""Implementation name -> zero-argument factory; the import happens in the call."""


def get_adapter(impl: str) -> Adapter:
    """Return a fresh adapter for ``impl``.

    The module holding the implementation under test is imported here, not at
    this package's import time, so that the worker can fix ``sys.path`` first.

    Args:
        impl: ``"old"`` (frozen JAX baseline) or ``"new"`` (evoxc rewrite).

    Returns:
        An object satisfying the :class:`Adapter` contract.

    Raises:
        ValueError: If ``impl`` is neither ``"old"`` nor ``"new"``.
    """
    try:
        factory = ADAPTERS[impl]
    except KeyError:
        raise ValueError(
            f"unknown impl {impl!r}; expected 'old' or 'new'"
        ) from None
    return factory()
