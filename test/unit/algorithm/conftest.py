"""Fixtures shared by the ``./unit/algorithm`` suites: the lazily resolved inputs.

Every module in this directory depends on something that is **not importable at
collection time**, so each is resolved through a fixture instead of a module-scope
import:

* the algorithm modules under test — ``tensorneat.algorithm.neat.species``
  (:func:`species`), ``tensorneat.algorithm.neat.neat`` (:func:`neat_module`) —
  and the two containers they share, ``tensorneat.common`` (:func:`common`) and
  ``tensorneat.common.errors`` (:func:`errors`).  ``import tensorneat``
  currently raises ``NotImplementedError``: ``tensorneat/algorithm/base.py``
  evaluates ``State()`` as a default argument while the class body is being
  created, and ``State`` is a design skeleton.  Deferring the import to fixture
  time keeps every module collectable and turns the missing implementation into
  a named skip instead of a collection error.  Once ``src/`` is implemented the
  imports can move back to module scope and these fixtures become one-line
  accessors.
* ``support.payloads`` (:func:`payloads`) — the hand-built gene payloads and the
  canonical genome factories of ``../../support``, a sibling deliverable, so it
  is a skip while absent rather than a failure.  ``support.runner`` imports only
  ``evoxc`` and is therefore safe to import at module scope; the suites do that
  and never re-implement the build/run dance.

Two layers of fixtures
----------------------
The ``scope="module"`` fixtures return **modules** and build nothing, so a suite
that only needs ``TensorNEATError`` does not pay for a genome.  The
function-scoped factories build the small host objects every suite shares — a
``SpeciesController`` over ``P = 8`` / ``S = 4``, a ``SpeciesState`` from
hand-written tables, a ``NEAT`` over the canonical genome — and may be written
as plain constructor calls because their *first* argument is the module fixture,
which skips before the body runs.

How a kernel is executed
------------------------
Only through ``support.runner`` (``run_defn`` for a ``@evoxc.defn`` or any plain
closure, ``run_graph`` for an already traced ``Graph``).  ``evoxc.build`` turns
only a ``numpy.ndarray`` argument into a graph input; every other argument is
baked into the trace as a constant, so a test passes the tensor *leaves*
positionally and rebuilds the immutable container (``State`` /
``SpeciesState``) **inside** the traced closure.  :func:`species_kernel` is that
closure for every ``SpeciesController`` helper that takes a ``SpeciesState``::

    spawn = species_kernel("_cal_spawn_numbers_by_rank", controller(...), bookkeeping)

``run_defn`` returns host numpy leaves (``to_numpy`` unwraps a nested result),
which is what lets a test compare against a plain-numpy oracle and print it on
failure.  A ``Defn`` object is not callable and statics are passed positionally,
so a kernel with keyword-only configuration is bound with ``functools.partial``.

Nothing here imports ``jax``/``torch`` or builds an evoxc graph.  The per-test
guard in ``../conftest.py`` (no optional framework may enter ``sys.modules``)
therefore stays honest for every module in this directory, and the numpy
reference backend is the only one exercised.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from support.lazy import require
from support.runner import run_defn, to_numpy

#: Population size of the hand-built tables.  Small on purpose: every assertion
#: below is readable by hand, and the species algebra is O(species_size ×
#: pop_size) so the whole suite stays under a second once implemented.
POP_SIZE = 8
#: Species-slot capacity of the hand-built tables (two live, two dead by default).
SPECIES_SIZE = 4
#: Static node/connection capacities used by the fake payloads.
MAX_NODES = 8
MAX_CONNS = 16
#: ``DefaultNode.length`` = one key column + ``bias, response, aggregation,
#: activation``; ``DefaultConn.length`` = ``(input_index, output_index)`` +
#: ``weight``.  Only the species centre tensors use them, and those are never
#: read by the algebra under test except in ``speciate``.
NODE_LEN = 5
CONN_LEN = 3


# --------------------------------------------------------------------------- #
# modules (scope="module": resolve once, build nothing)
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def species():
    """The ``tensorneat.algorithm.neat.species`` module: ``SpeciesController`` + ``SpeciesState``.

    Returns:
        The imported module, so a test reads ``species.SpeciesController`` and
        constructs ``species.SpeciesState`` tables directly.

    Notes:
        The import is deferred (see the module docstring) and only the pending
        ``NotImplementedError`` becomes a skip: an ``ImportError`` or any other
        failure still surfaces, because those mean the package is broken, not
        unfinished.
    """
    return require("tensorneat.algorithm.neat.species")


@pytest.fixture(scope="module")
def neat_module():
    """The ``tensorneat.algorithm.neat.neat`` module: ``NEAT`` and ``_sanitize_fitness``.

    Returns:
        The imported module.  It is named ``neat_module`` rather than ``neat`` so
        an *algorithm instance* fixture can keep the shorter, more readable name
        in the suites that need one.

    Notes:
        ``_sanitize_fitness`` lives at module scope (it is a pure function of the
        fitness vector and ``pop_size``), so it is reached through this module
        rather than through an instance.
    """
    return require("tensorneat.algorithm.neat.neat")


@pytest.fixture(scope="module")
def common():
    """The ``tensorneat.common`` package: ``State`` plus the tools the kernels call.

    Returns:
        The imported package.  ``State`` is the immutable container the controller
        methods receive; the suites build one with
        :func:`algorithm_state` rather than constructing it themselves.
    """
    return require("tensorneat.common")


@pytest.fixture(scope="module")
def errors():
    """The ``tensorneat.common.errors`` module: the ``TensorNEATError`` family.

    Returns:
        The imported module, so a constructor-rejection test asserts the
        documented *type* (``errors.TensorNEATError``) rather than a message.
    """
    return require("tensorneat.common.errors")


@pytest.fixture(scope="module")
def payloads():
    """The hand-built payload builders of ``../../support/payloads.py``.

    Returns:
        The ``support.payloads`` module: ``new_payload``, ``set_node``,
        ``set_conn``, the canonical topologies, ``default_genome`` and
        ``I_INF``.

    Notes:
        ``support.payloads`` imports ``tensorneat.common.tools`` at module scope,
        so it is a skip while the package is a skeleton — exactly like the
        modules under test.
    """
    return require("support.payloads")


# --------------------------------------------------------------------------- #
# host objects (function scope: built per test, small)
# --------------------------------------------------------------------------- #


@pytest.fixture
def controller(species):
    """Factory: ``make(**overrides) -> SpeciesController`` over ``P = 8``, ``S = 4``.

    Every constructor argument defaults to the value the suites assert against,
    so a test overrides exactly the knob it is about::

        controller(spawn_number_change_rate=0.0, species_number_calculate_by="fitness")

    Returns:
        A callable taking ``SpeciesController`` keyword arguments and returning
        the controller.

    Notes:
        The controller is a pure host-side configuration object (no tensor is
        built in ``__init__``), which is why this factory is not traced.
    """

    def make(**overrides):
        config = dict(
            pop_size=POP_SIZE,
            species_size=SPECIES_SIZE,
            max_stagnation=15,
            species_elitism=2,
            spawn_number_change_rate=0.5,
            genome_elitism=2,
            survival_threshold=0.1,
            min_species_size=1,
            compatibility_threshold=2.0,
        )
        config.update(overrides)
        return species.SpeciesController(**config)

    return make


@pytest.fixture
def species_state(species):
    """Factory: ``make(**overrides) -> SpeciesState`` with two live and two dead slots.

    The default table is the canonical "half-full species list" the algebra tests
    start from (``S = 4``, ``P = 8``)::

        slot        0     1     2     3
        state     live  live  dead  dead
        keys       0.0   1.0   NaN   NaN
        best_fit   0.5   0.2   NaN   NaN
        improved   0.0   3.0   NaN   NaN
        members    5.0   3.0   NaN   NaN

    ``idx2species`` assigns population slots ``0..4`` to species key ``0`` and
    ``5..7`` to key ``1``; ``next_species_key`` is ``2.0`` (``0`` is the initial
    species, so the counter starts at ``1`` and two keys are already in use).

    Returns:
        A callable taking ``SpeciesState`` keyword overrides and returning the
        dataclass with every field present.

    Notes:
        A dead slot is dead *only* because its key is ``NaN`` — the centre rows are
        zero-filled integers here and are equally never read, which is what the
        dead-slot test asserts.
    """

    def make(**overrides):
        fields = dict(
            species_keys=np.array([0.0, 1.0, np.nan, np.nan], np.float32),
            best_fitness=np.array([0.5, 0.2, np.nan, np.nan], np.float32),
            last_improved=np.array([0.0, 3.0, np.nan, np.nan], np.float32),
            member_count=np.array([5.0, 3.0, np.nan, np.nan], np.float32),
            idx2species=np.array([0, 0, 0, 0, 0, 1, 1, 1], np.float32),
            center_nodes=np.zeros((SPECIES_SIZE, MAX_NODES, NODE_LEN), np.int32),
            center_conns=np.zeros((SPECIES_SIZE, MAX_CONNS, CONN_LEN), np.int32),
            next_species_key=np.float32(2.0),
        )
        fields.update(overrides)
        return species.SpeciesState(**fields)

    return make


@pytest.fixture
def algorithm_state(common, species_state):
    """Factory: wrap a ``SpeciesState`` into the ``State`` the controller methods take.

    ``SpeciesController.update_species``/``speciate`` read ``state.species``,
    ``state.pop_nodes``, ``state.pop_conns``, ``state.generation`` and (for the
    parent draw) ``state.randkey``; this factory registers exactly those entries::

        state = algorithm_state(bookkeeping, pop_nodes=nodes, pop_conns=conns,
                                generation=1.0, randkey=evoxc_key)

    Returns:
        A callable ``make(bookkeeping, *, pop_nodes, pop_conns, generation=0.0,
        randkey=None)`` returning a ``tensorneat.common.State``.

    Notes:
        ``generation`` is stored as a 0-d ``float32`` because that is what the
        algorithm registers (stagnation arithmetic writes it into the float32
        ``last_improved`` field).  ``randkey`` is optional so a test about a
        deterministic step (speciation, spawn allocation) cannot accidentally
        depend on one.
    """

    def make(bookkeeping, *, pop_nodes, pop_conns, generation=0.0, randkey=None):
        entries = dict(
            pop_nodes=pop_nodes,
            pop_conns=pop_conns,
            generation=np.float32(generation),
            species=bookkeeping,
        )
        if randkey is not None:
            entries["randkey"] = randkey
        return common.State(**entries)

    return make


@pytest.fixture
def neat(neat_module, payloads):
    """Factory: ``make(**overrides) -> NEAT`` over the canonical ``DefaultGenome``.

    The genome comes from ``payloads.default_genome()`` (2 inputs, 1 output,
    ``max_nodes = 8``, ``max_conns = 16``, default gene/operator stack) and the
    population/species sizes default to this suite's ``P = 8`` / ``S = 4`` with no
    ``pop_batch_size`` chunking.

    Returns:
        A callable taking ``NEAT`` keyword arguments and returning the algorithm.

    Notes:
        The genome is constructed, never initialized: ``NEAT.setup`` is the
        workload that draws the first population, and a test that only checks
        constructor validation must not pay for it.
    """

    def make(**overrides):
        config = dict(genome=payloads.default_genome(), pop_size=POP_SIZE, species_size=SPECIES_SIZE)
        config.update(overrides)
        return neat_module.NEAT(**config)

    return make


@pytest.fixture
def species_kernel(species):
    """Factory: run one ``SpeciesController`` method in a graph and return host numpy.

    Usage::

        spawn = species_kernel("_cal_spawn_numbers_by_rank", controller(), bookkeeping)
        winner, loser, elite = species_kernel(
            "_pair_parents", controller(), bookkeeping, selection_key, spawn, fitness
        )

    The dataclass fields travel as graph inputs (in ``dataclasses.fields`` order, so
    the fixture never hard-codes the field list) and any ``extra`` numpy array — a
    selection key, a spawn vector, a fitness vector — travels after them.  The
    ``SpeciesState`` is rebuilt inside the traced closure because
    ``evoxc.build`` only turns an ``ndarray`` argument into an input.

    Returns:
        A callable ``run(method_name, controller, bookkeeping, *extra)`` returning
        the method's output with ``evoxc.Tensor`` leaves replaced by numpy.

    Notes:
        The method is looked up on the *class* and called with the controller
        explicitly, so the controller stays a host-side static closure value rather
        than something a test has to re-register inside the trace.
    """
    names = [field.name for field in dataclasses.fields(species.SpeciesState)]

    def run(method_name, controller, bookkeeping, *extra):
        arrays = [np.asarray(getattr(bookkeeping, name)) for name in names]
        method = getattr(species.SpeciesController, method_name)

        def kernel(*values):
            table = species.SpeciesState(*values[: len(names)])
            return method(controller, table, *values[len(names) :])

        return to_numpy(run_defn(kernel, *(arrays + list(extra))))

    return run

