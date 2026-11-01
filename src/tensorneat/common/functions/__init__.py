"""ACT / AGG option registries and the per-node dispatchers.

This package is the single source of truth for the activation and aggregation
options a NEAT node gene may select. It exposes exactly five names:

* ``ACT`` / ``AGG`` -- module-level :class:`FunctionManager` instances pairing
  the evoxc kernels with their sympy counterparts, in canonical option order.
* ``FunctionManager`` -- the registry class, for genomes that need a private
  registry (custom options with their own sympy form).
* ``apply_activation`` / ``apply_aggregation`` -- the in-graph dispatchers.

Import order matters only in that ``.act``/``.agg`` own the canonical option
order and ``.act_sympy``/``.agg_sympy`` own the paired sympy tuple; the manager
validates the pairing at import time, so a mismatched table fails immediately
rather than producing a wrong formula later.

``import tensorneat`` pulls in ``sympy`` through this package (a declared
dependency); no backend module, no JAX, no torch.
"""

from .act import ACT_FUNCS, apply_activation
from .act_sympy import ACT_SYMPY_FUNCS, ACT_SYMPY_HELPERS
from .agg import AGG_FUNCS, apply_aggregation
from .agg_sympy import AGG_SYMPY_FUNCS
from .manager import FunctionManager

ACT = FunctionManager(ACT_FUNCS, ACT_SYMPY_FUNCS, ACT_SYMPY_HELPERS)
AGG = FunctionManager(AGG_FUNCS, AGG_SYMPY_FUNCS)

__all__ = [
    "ACT",
    "AGG",
    "FunctionManager",
    "apply_activation",
    "apply_aggregation",
]
