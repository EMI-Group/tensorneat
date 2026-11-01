"""``tensorneat.common`` — host primitives and the tensorization engine.

Dependency DAG position: bottom of the ``tensorneat`` package. Nothing here
imports ``genome``, ``algorithm``, ``problem`` or ``pipeline``, so every other
node may import this one without a cycle. Only ``numpy``, ``sympy`` and
``evoxc`` are used — never ``jax`` or ``torch``, never NumPy's global RNG.

Public surface:

* :class:`State` / :class:`StatefulBaseClass` — the immutable host container
  and the configuration mixin that extends it.
* :class:`TensorNEATError` and its subclasses — host-side validation errors;
  evoxc backend/shape errors are never caught or re-wrapped.
* :data:`I_INF`, :data:`KEY_MAX` and the batched helper kernels of
  :mod:`~tensorneat.common.tools` (flat gather/scatter, selection, mutation).
* :class:`Transformed`, :func:`derive_schedule`, :func:`validate_schedule` —
  the execution schedule that makes a ragged evolution problem static-shape.
* ``ACT`` / ``AGG`` registries, their dispatchers and :class:`FunctionManager`
  from :mod:`~tensorneat.common.functions`.
"""

from .state import State
from .stateful_class import StatefulBaseClass

from .errors import (
    BackendMismatchError,
    ConfigError,
    GraphError,
    ShapeBoundError,
    TensorNEATError,
)

from .tools import (
    I_INF,
    KEY_MAX,
    argmin_with_mask,
    batch_offsets,
    fetch_first,
    fetch_random,
    flat_gather,
    flat_scatter,
    hash_rows,
    masked_flat_gather,
    masked_flat_scatter,
    mutate_float,
    mutate_int,
    rank_elements,
)

from .tensorization import Transformed, derive_schedule, validate_schedule

from .functions import (
    ACT,
    AGG,
    FunctionManager,
    apply_activation,
    apply_aggregation,
)

__all__ = [
    # host containers
    "State",
    "StatefulBaseClass",
    # error hierarchy
    "TensorNEATError",
    "ConfigError",
    "ShapeBoundError",
    "GraphError",
    "BackendMismatchError",
    # sentinels and batched helpers
    "I_INF",
    "KEY_MAX",
    "batch_offsets",
    "flat_gather",
    "masked_flat_gather",
    "flat_scatter",
    "masked_flat_scatter",
    "fetch_first",
    "fetch_random",
    "rank_elements",
    "argmin_with_mask",
    "mutate_float",
    "mutate_int",
    "hash_rows",
    # tensorization (execution schedule)
    "Transformed",
    "derive_schedule",
    "validate_schedule",
    # activation / aggregation registries
    "ACT",
    "AGG",
    "FunctionManager",
    "apply_activation",
    "apply_aggregation",
]
