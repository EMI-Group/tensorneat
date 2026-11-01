"""Unit-suite guard: the numpy-backend tests must not pull in an optional framework.

The release gate (`../release_gate_no_jax.py`) checks the *package* in a clean
subprocess; this fixture checks the *tests*: after every unit test, no optional
framework may have entered ``sys.modules`` that was not already there when the
test started (the "already there" allowance keeps a mixed run, e.g.
``pytest`` over parity and unit together, from failing spuriously).
"""

from __future__ import annotations

import sys

import pytest

#: Modules whose presence means the suite silently acquired an optional dependency.
FORBIDDEN = ("jax", "torch", "brax", "gymnax", "mujoco")


@pytest.fixture(autouse=True)
def _stays_free_of_optional_frameworks(request: pytest.FixtureRequest):
    """Fail the test that imports an optional framework at run time or at module scope."""
    preexisting = {name for name in FORBIDDEN if name in sys.modules}
    yield
    leaked = sorted(
        name for name in FORBIDDEN if name in sys.modules and name not in preexisting
    )
    assert not leaked, (
        f"{request.node.nodeid} imported {leaked}; the unit suite runs on the numpy "
        "backend only (parity tests are the only place allowed to import jax)"
    )
