"""Unit tests for the public name surface of ``tensorneat.algorithm``.

What this module proves
-----------------------
The package advertises ``tensorneat.algorithm.{BaseAlgorithm,NEAT,HyperNEAT,
HyperNEATFeedForward}`` (see ``src/tensorneat/CONTEXT.md``), so every one of those
spellings has to resolve after a plain ``from tensorneat import algorithm`` -- and
resolve to the *same* object the defining submodule exposes, not to a re-created
alias:

* ``HyperNEATFeedForward`` is importable from the package even though it is defined
  one package deeper, in ``tensorneat.algorithm.hyperneat.hyperneat_feedforward``;
* it is listed in ``__all__``, which is what ``from tensorneat.algorithm import *``
  and the docs' public-surface checks read;
* ``__all__`` itself is a duplicate-free list of strings and every entry resolves
  through ``getattr`` -- an entry naming a removed or misspelled symbol is a
  documentation bug that only this direction of the check catches.

Fast and host-only: no graph is built, no evolution runs, no backend is involved.
"""

from __future__ import annotations

import tensorneat
import tensorneat.algorithm as algorithm
import tensorneat.algorithm.hyperneat as hyperneat
from tensorneat.algorithm.base import BaseAlgorithm
from tensorneat.algorithm.neat import NEAT

#: Names the package documentation promises on ``tensorneat.algorithm`` itself.
DOCUMENTED_NAMES = ("BaseAlgorithm", "NEAT", "HyperNEAT", "HyperNEATFeedForward")


def test_hyperneat_feedforward_is_exported_from_the_package() -> None:
    """The feedforward variant is the very class ``.hyperneat`` defines."""
    exported = algorithm.HyperNEATFeedForward

    assert exported is hyperneat.HyperNEATFeedForward, (
        "tensorneat.algorithm.HyperNEATFeedForward must be the class re-exported by "
        f"tensorneat.algorithm.hyperneat, got {exported!r}"
    )
    assert exported.__module__ == (
        "tensorneat.algorithm.hyperneat.hyperneat_feedforward"
    ), exported.__module__
    assert exported.__name__ == "HyperNEATFeedForward"


def test_every_documented_name_is_in_the_package_all() -> None:
    """``__all__`` must list the documented spellings, without duplicates."""
    assert isinstance(algorithm.__all__, list), type(algorithm.__all__)
    assert all(isinstance(name, str) for name in algorithm.__all__), algorithm.__all__
    assert len(set(algorithm.__all__)) == len(algorithm.__all__), (
        f"duplicate names in tensorneat.algorithm.__all__: {algorithm.__all__}"
    )
    missing = sorted(set(DOCUMENTED_NAMES) - set(algorithm.__all__))
    assert not missing, f"documented but absent from __all__: {missing}"


def test_every_name_in_the_package_all_resolves() -> None:
    """Each ``__all__`` entry is a real attribute, and the classes are the classes."""
    for name in algorithm.__all__:
        resolved = getattr(algorithm, name)
        assert resolved is not None, f"tensorneat.algorithm.{name} resolved to None"

    assert algorithm.BaseAlgorithm is BaseAlgorithm
    assert algorithm.NEAT is NEAT
    assert algorithm.HyperNEAT is hyperneat.HyperNEAT
    assert algorithm.HyperNEATFeedForward is hyperneat.HyperNEATFeedForward


def test_the_documented_attribute_path_works_from_the_top_level_package() -> None:
    """``tensorneat.algorithm`` is the same module object the top level exposes."""
    assert tensorneat.algorithm is algorithm
    assert tensorneat.algorithm.HyperNEATFeedForward is hyperneat.HyperNEATFeedForward
