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
  :mod:`~tensorneat.common.tools` (flat gather/scatter, selection, mutation),
  plus :func:`~tensorneat.common.tools.to_numpy`, the host-side array accessor
  every logging/checkpoint path uses.
* :class:`Transformed`, :func:`derive_schedule`, :func:`validate_schedule` —
  the execution schedule that makes a ragged evolution problem static-shape,
  plus :func:`kernel_mode` — the resolver of the generic formulation static
  ``state.kernel_mode`` that selects the schedule's host or accelerator
  spellings (:mod:`~tensorneat.common.schedule_accelerator`).
* :func:`~tensorneat.common.host_graph.decode_graph`,
  :func:`~tensorneat.common.host_graph.topological_sort` and
  :func:`~tensorneat.common.host_graph.find_useful_nodes` — the host-only graph
  walks, re-exported for convenience; ``host_graph`` also owns ``node_depth``
  and ``check_cycles``, which consumers import from the module directly.
* ``ACT`` / ``AGG`` registries, their dispatchers and :class:`FunctionManager`
  from :mod:`~tensorneat.common.functions`.

:mod:`~tensorneat.common.sympy_tools` (host-only formula/Python code
generation) is deliberately **not** re-exported: it is only reached from
``Genome.sympy_func``, so ``import tensorneat.common.sympy_tools`` at the
call site keeps the package surface as small as the contract specifies.
"""

from .state import State
from .stateful_class import StatefulBaseClass

from .errors import (
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
    to_numpy,
)

from .tensorization import Transformed, derive_schedule, validate_schedule
from .schedule_accelerator import kernel_mode

from .host_graph import decode_graph, find_useful_nodes, topological_sort

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
    "to_numpy",
    # tensorization (execution schedule)
    "Transformed",
    "derive_schedule",
    "validate_schedule",
    "kernel_mode",
    # host-only graph walks
    "decode_graph",
    "topological_sort",
    "find_useful_nodes",
    # activation / aggregation registries
    "ACT",
    "AGG",
    "FunctionManager",
    "apply_activation",
    "apply_aggregation",
]
