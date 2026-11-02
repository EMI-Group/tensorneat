"""Fixtures shared by the ``./unit/algorithm`` suites: modules, tables and graph runners.

Two layers of fixtures
----------------------
The ``scope="module"`` fixtures return **modules** and build nothing, so a suite that
only needs ``TensorNEATError`` does not pay for a genome.  The function-scoped
factories build the small host objects every suite shares -- a ``SpeciesController``
over ``P = 8`` / ``S = 4``, a ``SpeciesState`` from hand-written tables, a ``NEAT``
over the canonical ``DefaultGenome`` -- and may be written as plain constructor calls
because their *first* argument is a module fixture.

How a kernel is executed
------------------------
Only through ``support.runner`` (``run_defn`` for a ``@evoxc.defn`` or any plain
closure).  ``evoxc.build`` turns only a ``numpy.ndarray`` argument into a graph input;
every other argument is baked into the trace as a constant, so a test passes the
tensor *leaves* positionally and rebuilds the immutable container (``State`` /
``SpeciesState``) **inside** the traced closure.  Two runners cover the whole species
surface:

* :func:`species_kernel` -- ``run(method, bookkeeping, *extra)``.  The ten
  ``SpeciesState`` fields travel first, in ``dataclasses.fields`` order, so the
  runner never hard-codes the field list; the table is rebuilt inside the trace.
* :func:`raw_kernel` -- ``run(method, *arrays)``.  No ``SpeciesState`` at all, for the
  pure helpers (``_prefix_sum_exclusive``, ``_sort_desc_legacy``, ``_block_layout``).

Both look the method up on the *class*, pass the controller explicitly for an
instance method and omit it for a ``staticmethod`` (``inspect.getattr_static``
decides), so the controller stays a host-side configuration value baked into the
trace instead of something a test re-registers.  ``run_defn`` returns host numpy
leaves, which is what lets a test compare against a plain-numpy oracle and print it
on failure.

Import policy
-------------
``tensorneat`` is imported at module scope here: ``src/`` is implemented, so the
former ``support.lazy.require`` skip dance is gone.  The import happens at collection
time, before the per-test guard in ``../conftest.py`` snapshots ``sys.modules``.
``import tensorneat`` still loads the legacy JAX ``src/tensorneat/pipeline.py``
through ``tensorneat/__init__``, so this suite needs ``jax`` installed until that file
becomes evoxc-native; no test *uses* jax, and the algorithm subtree itself is
jax-free.  ``support.payloads`` is imported at module scope for the same reason.

Nothing here builds a graph, and backend is always ``"numpy"``.
"""

from __future__ import annotations

import dataclasses
import inspect

import numpy as np
import pytest

from support import payloads as payloads_module
from support.runner import run_defn, to_numpy

import tensorneat.algorithm.neat.neat as neat_module_under_test
import tensorneat.algorithm.neat.species as species_module
import tensorneat.common as common_module
import tensorneat.common.errors as errors_module

#: Population size of the hand-built tables.  Small on purpose: every assertion in
#: the suites is readable by hand, and the species algebra is O(species_size ×
#: pop_size) so the whole directory stays under a few seconds.
POP_SIZE = 8
#: Species-slot capacity of the hand-built tables (two live, two dead by default).
SPECIES_SIZE = 4
#: Static node/connection capacities used by the fake payloads.
MAX_NODES = 8
MAX_CONNS = 16
#: Gene widths of the default gene stack (``payloads`` documents the column order):
#: ``DefaultNode`` has four attribute columns, ``DefaultConn`` two key columns
#: (``input_index``/``output_index``, no ``historical_marker``) and one ``weight``.
NODE_ATTR_WIDTH = 4
CONN_KEY_WIDTH = 2
CONN_ATTR_WIDTH = 1


# --------------------------------------------------------------------------- #
# modules (scope="module": resolve once, build nothing)
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def species():
    """The ``tensorneat.algorithm.neat.species`` module: ``SpeciesController`` + ``SpeciesState``.

    Returns:
        The imported module, so a test reads ``species.SpeciesController`` and
        constructs ``species.SpeciesState`` tables directly.
    """
    return species_module


@pytest.fixture(scope="module")
def neat_module():
    """The ``tensorneat.algorithm.neat.neat`` module: ``NEAT`` and ``_sanitize_fitness``.

    Returns:
        The imported module.  It is named ``neat_module`` rather than ``neat`` so an
        *algorithm instance* fixture keeps the shorter, more readable name in the
        suites that need one.

    Notes:
        ``_sanitize_fitness`` lives at module scope (a pure function of the fitness
        vector and ``pop_size``), so it is reached through this module rather than
        through an instance.
    """
    return neat_module_under_test


@pytest.fixture(scope="module")
def common():
    """The ``tensorneat.common`` package: ``State`` plus the tools the kernels call.

    Returns:
        The imported package.  ``State`` is the immutable container the controller
        methods receive; the suites build one with :func:`algorithm_state` rather than
        constructing it themselves.
    """
    return common_module


@pytest.fixture(scope="module")
def errors():
    """The ``tensorneat.common.errors`` module: the ``TensorNEATError`` family.

    Returns:
        The imported module, so a constructor-rejection test asserts the documented
        *type* (``errors.TensorNEATError``) rather than a message.
    """
    return errors_module


@pytest.fixture(scope="module")
def payloads():
    """The hand-built payload builders of ``../../support/payloads.py``.

    Returns:
        ``support.payloads``: ``new_payload``, ``set_node``, ``set_conn``, the
        canonical topologies, ``default_genome`` and ``I_INF``.
    """
    return payloads_module


# --------------------------------------------------------------------------- #
# host objects (function scope: built per test, small)
# --------------------------------------------------------------------------- #


@pytest.fixture
def controller(species):
    """Factory: ``make(**overrides) -> SpeciesController`` over ``P = 8``, ``S = 4``.

    Every constructor argument defaults to the value the suites assert against, so a
    test overrides exactly the knob it is about::

        controller(spawn_number_change_rate=0.0, species_number_calculate_by="fitness")

    Returns:
        A callable taking ``SpeciesController`` keyword arguments and returning the
        controller.

    Notes:
        The controller is a pure host-side configuration object (no tensor is built in
        ``__init__``), which is why this factory is not traced.
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

    The four centre fields are the **split payload arrays** of the documented
    ``SpeciesState`` (``center_node_keys``/``center_node_attrs``/
    ``center_conn_keys``/``center_conn_attrs``) -- never packed integer gene rows.
    They are zero-filled here because the algebra under test does not read them; only
    ``speciate`` does, and its suite overrides them with real payloads.

    Returns:
        A callable taking ``SpeciesState`` keyword overrides and returning the
        dataclass with every field present.

    Notes:
        A dead slot is dead *only* because its key is ``NaN`` -- its centre rows hold
        stale zeros (``int32`` cannot carry ``NaN``) and are equally never read, which
        is what the dead-slot test asserts.
    """

    def make(**overrides):
        fields = dict(
            species_keys=np.array([0.0, 1.0, np.nan, np.nan], np.float32),
            best_fitness=np.array([0.5, 0.2, np.nan, np.nan], np.float32),
            last_improved=np.array([0.0, 3.0, np.nan, np.nan], np.float32),
            member_count=np.array([5.0, 3.0, np.nan, np.nan], np.float32),
            idx2species=np.array([0, 0, 0, 0, 0, 1, 1, 1], np.float32),
            center_node_keys=np.zeros((SPECIES_SIZE, MAX_NODES), np.int32),
            center_node_attrs=np.zeros((SPECIES_SIZE, MAX_NODES, NODE_ATTR_WIDTH), np.float32),
            center_conn_keys=np.zeros((SPECIES_SIZE, MAX_CONNS, CONN_KEY_WIDTH), np.int32),
            center_conn_attrs=np.zeros((SPECIES_SIZE, MAX_CONNS, CONN_ATTR_WIDTH), np.float32),
            next_species_key=np.float32(2.0),
        )
        fields.update(overrides)
        return species.SpeciesState(**fields)

    return make


@pytest.fixture
def algorithm_state(common, species_state):
    """Factory: wrap a ``SpeciesState`` into the ``State`` the controller methods take.

    ``SpeciesController.speciate`` reads ``state.species``, the population payload and
    ``state.generation``; ``update_species`` additionally reads ``state.randkey``.  The
    payload is registered under the documented split keys
    (``pop_node_keys``/``pop_node_attrs``/``pop_conn_keys``/``pop_conn_attrs``) or as
    one ``pop`` entry holding a ``GenomeTensors``::

        state = algorithm_state(bookkeeping, pop=payload,
                                generation=1.0, randkey=evoxc_key)

    Returns:
        A callable ``make(bookkeeping, *, pop=None, generation=0.0, randkey=None)``
        returning a ``tensorneat.common.State``.

    Notes:
        ``generation`` is stored as a 0-d ``float32`` because that is what the
        algorithm registers (stagnation arithmetic writes it into the float32
        ``last_improved`` field).  ``randkey`` is optional so a test about a
        deterministic step (speciation, spawn allocation) cannot accidentally depend
        on one.  The payload arrays stay plain numpy: a runner hands them to
        ``evoxc.build`` as graph inputs and rebuilds the container inside the trace.
    """

    def make(bookkeeping, *, pop=None, generation=0.0, randkey=None):
        entries = dict(
            generation=np.float32(generation),
            species=bookkeeping,
        )
        if randkey is not None:
            entries["randkey"] = randkey
        if pop is not None:
            split = getattr(pop, "_fields", None)
            if split and len(split) == 4:
                entries["pop_node_keys"] = np.asarray(pop.node_keys)
                entries["pop_node_attrs"] = np.asarray(pop.node_attrs)
                entries["pop_conn_keys"] = np.asarray(pop.conn_keys)
                entries["pop_conn_attrs"] = np.asarray(pop.conn_attrs)
            else:
                entries["pop"] = pop
        return common.State(**entries)

    return make


@pytest.fixture
def neat(neat_module, payloads):
    """Factory: ``make(**overrides) -> NEAT`` over the canonical ``DefaultGenome``.

    The genome comes from ``payloads.default_genome()`` (2 inputs, 1 output, its own
    default ``max_nodes``/``max_conns`` and the default gene/operator stack) and the
    population/species sizes default to this suite's ``P = 8`` / ``S = 4`` with no
    ``pop_batch_size`` chunking.

    Returns:
        A callable taking ``NEAT`` keyword arguments and returning the algorithm.

    Notes:
        The genome is constructed, never initialized: ``NEAT.setup`` is the workload
        that draws the first population, so a test that only checks constructor
        validation must not pay for it.
    """

    def make(**overrides):
        config = dict(
            genome=payloads.default_genome(), pop_size=POP_SIZE, species_size=SPECIES_SIZE
        )
        config.update(overrides)
        return neat_module.NEAT(**config)

    return make


# --------------------------------------------------------------------------- #
# graph runners
# --------------------------------------------------------------------------- #


def _needs_controller(species, method_name):
    """True when ``method_name`` is a plain instance method (vs. a ``staticmethod``)."""
    raw = inspect.getattr_static(species.SpeciesController, method_name)
    return not isinstance(raw, staticmethod)


def _host(value, species):
    """Host numpy view of a kernel result, including a returned ``SpeciesState``."""
    value = to_numpy(value)
    if isinstance(value, species.SpeciesState):
        return {
            field.name: to_numpy(getattr(value, field.name))
            for field in dataclasses.fields(species.SpeciesState)
        }
    return value


@pytest.fixture
def species_kernel(species):
    """Factory: run one ``SpeciesController`` method over a ``SpeciesState`` in a graph.

    Usage::

        spawn = species_kernel("_cal_spawn_numbers_by_rank", bookkeeping)
        winner, loser, elite = species_kernel(
            "_pair_parents", bookkeeping, selection_key, spawn, fitness
        )
        sorted_fields = species_kernel("_gather_species", bookkeeping, order)

    The dataclass fields travel as graph inputs in ``dataclasses.fields`` order (the
    fixture never hard-codes the field list) and any ``extra`` numpy array -- a
    selection key, a spawn vector, a fitness vector -- travels after them.  The
    ``SpeciesState`` is rebuilt inside the traced closure because ``evoxc.build`` only
    turns an ``ndarray`` argument into an input.

    Returns:
        A callable ``run(method_name, bookkeeping, *extra, controller=None)``.  When
        ``controller`` is omitted a default controller (this suite's standard
        configuration) is built, so a test that only reads a documented default needs
        no factory.  A ``SpeciesState`` result is returned as a field-name → numpy
        ``dict``; anything else is returned as ``to_numpy`` produced it.
    """
    names = [field.name for field in dataclasses.fields(species.SpeciesState)]

    def run(method_name, bookkeeping, *extra, controller=None):
        if controller is None:
            controller = species.SpeciesController(
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
        arrays = [np.asarray(getattr(bookkeeping, name)) for name in names]
        method = getattr(species.SpeciesController, method_name)
        bound = _needs_controller(species, method_name)

        def kernel(*values):
            table = species.SpeciesState(*values[: len(names)])
            tail = values[len(names) :]
            if bound:
                return method(controller, table, *tail)
            return method(table, *tail)

        return _host(run_defn(kernel, *(arrays + list(extra))), species)

    return run


@pytest.fixture
def raw_kernel(species):
    """Factory: run a ``SpeciesController`` helper that takes no ``SpeciesState``.

    Usage::

        loc, idx_in_block = raw_kernel("_block_layout", spawn)
        prefix = raw_kernel("_prefix_sum_exclusive", values)
        order = raw_kernel("_sort_desc_legacy", values)

    Returns:
        A callable ``run(method_name, *arrays, controller=None)`` returning host numpy.

    Notes:
        ``_block_layout`` is an instance method (it needs ``pop_size``/``species_size``)
        while the two utilities are ``staticmethod``s; the runner decides from the
        class attribute so the call sites stay uniform.
    """
    def run(method_name, *arrays, controller=None):
        if controller is None:
            controller = species.SpeciesController(
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
        method = getattr(species.SpeciesController, method_name)
        bound = _needs_controller(species, method_name)
        if bound:
            return _host(run_defn(lambda *values: method(controller, *values), *arrays), species)
        return _host(run_defn(lambda *values: method(*values), *arrays), species)

    return run
