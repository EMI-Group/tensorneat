"""Package marker: the parity modules share basenames with ``../unit/genome``.

``test/`` is not a package, so pytest's default *prepend* import mode imports
every test module under its bare basename (``test_grad``).  This directory mirrors
the source tree exactly like ``../unit`` does and therefore reuses three module
names (``test_forward``, ``test_distance``, ``test_grad``), which collides with
``../unit/genome/'s`` modules and aborts the whole session with an
``import file mismatch``.

An ``__init__.py`` here makes pytest resolve this directory's modules through
their package path (``parity.test_grad``) instead of the bare basename, so the two
suites can coexist without renaming either one.  It holds no code: nothing in this
directory is imported as part of a package by another module, and ``conftest.py``
still puts this directory on ``sys.path`` so ``import convert`` keeps working.
"""
