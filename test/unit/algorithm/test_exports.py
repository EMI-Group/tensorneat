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
* ``__all__`` itself is a duplicate-free list of strings, is *exactly* the
  documented set (an entry added or renamed in one place and not the other
  fails), and every entry resolves through ``getattr`` to a class whose
  ``__module__`` lives inside ``tensorneat.algorithm`` -- so a removed or
  misspelled symbol, a module, or an object re-exported from another package is
  a documentation bug that only this direction of the check catches;
* every documented spelling resolves to the very class its defining submodule
  exposes, not to a look-alike alias.

Fast and host-only: no graph is built, no evolution runs, no backend is involved.
"""

from __future__ import annotations

import tensorneat
import tensorneat.algorithm as algorithm
import tensorneat.algorithm.hyperneat as hyperneat
from tensorneat.algorithm.base import BaseAlgorithm
from tensorneat.algorithm.neat import NEAT

#: Class each documented spelling must resolve to, imported from its *defining*
#: submodule -- never from ``tensorneat.algorithm`` -- so the identity check has
#: an oracle the package under test cannot influence.
DOCUMENTED_CLASSES = {
    "BaseAlgorithm": BaseAlgorithm,
    "NEAT": NEAT,
    "HyperNEAT": hyperneat.HyperNEAT,
    "HyperNEATFeedForward": hyperneat.HyperNEATFeedForward,
}

#: Names the package documentation promises on ``tensorneat.algorithm`` itself.
DOCUMENTED_NAMES = tuple(DOCUMENTED_CLASSES)


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
    """``__all__`` is exactly the documented spellings, without duplicates."""
    assert isinstance(algorithm.__all__, list), type(algorithm.__all__)
    assert all(isinstance(name, str) for name in algorithm.__all__), algorithm.__all__
    assert len(set(algorithm.__all__)) == len(algorithm.__all__), (
        f"duplicate names in tensorneat.algorithm.__all__: {algorithm.__all__}"
    )
    unexpected = sorted(set(algorithm.__all__) - set(DOCUMENTED_NAMES))
    missing = sorted(set(DOCUMENTED_NAMES) - set(algorithm.__all__))
    assert not unexpected, (
        f"tensorneat.algorithm.__all__ advertises undocumented names: {unexpected}"
    )
    assert not missing, f"documented but absent from __all__: {missing}"
    assert set(algorithm.__all__) == set(DOCUMENTED_NAMES), (
        "tensorneat.algorithm.__all__ must be exactly the documented set "
        f"{sorted(DOCUMENTED_NAMES)}, got {sorted(algorithm.__all__)}"
    )


def test_every_name_in_the_package_all_resolves() -> None:
    """Each ``__all__`` entry is a class defined in the package and the right one.

    Two independent directions of the check: an entry has to be a callable whose
    ``__module__`` lives inside ``tensorneat.algorithm`` (so a module, a removed
    or misspelled symbol, or a foreign re-export fails), and every documented
    spelling has to be the very class imported from its defining submodule (so an
    alias that merely *looks* right fails too).
    """
    for name in algorithm.__all__:
        resolved = getattr(algorithm, name)
        assert callable(resolved), (
            f"tensorneat.algorithm.{name} must be a class or callable public "
            f"entry, got {type(resolved).__name__} ({resolved!r})"
        )
        module = getattr(resolved, "__module__", "")
        assert module == "tensorneat.algorithm" or module.startswith(
            "tensorneat.algorithm."
        ), (
            f"tensorneat.algorithm.{name} is defined in {module!r}, outside the "
            "tensorneat.algorithm package; a public entry must not be an object "
            "re-exported from another package"
        )

    for name, defining_class in DOCUMENTED_CLASSES.items():
        resolved = getattr(algorithm, name)
        assert resolved is defining_class, (
            f"tensorneat.algorithm.{name} must be the object imported from its "
            f"defining submodule (expected {defining_class!r}), got {resolved!r}"
        )


def test_the_documented_attribute_path_works_from_the_top_level_package() -> None:
    """``tensorneat.algorithm`` is the same module object the top level exposes."""
    assert tensorneat.algorithm is algorithm
    assert tensorneat.algorithm.HyperNEATFeedForward is hyperneat.HyperNEATFeedForward
