"""Unit test for the SPECIATION gate of the accelerator formulation.

What this module proves
-----------------------
``_kernels.gated(species_size, pop_size, mode)`` is the pure host predicate that
decides whether the two speciation scans compute each distance row lazily behind
one ``evoxc.cond`` (``_kernels.gated_row``):

* ``"host"`` gates iff ``species_size * pop_size <= GATED_ROWS`` (6000);
* ``"accelerator"`` **never** gates, whatever the table size, because on a
  compiled backend each ``cond`` reads its predicate back to the host while the
  extra unconditional row is nearly free in the already-launched sequence.

Three facts are pinned: the predicate's truth table; the traced SPECIATE graph
carrying exactly the ``"if"`` ops the predicate predicts (four on the gated host
leg -- one per traced scan-body instance, ``evoxc.scan`` peels step 0 and traces
its body again, of each of the two scans -- zero on the host-dense leg and zero
on the accelerator leg at the *same* small table that gates on the host leg);
and non-vacuity (every graph has ops; the gated and accelerator histograms
differ).  The accelerator zero is the regression: that route must never emit a
host read-back ``cond``.

What was traced
---------------
Exactly ``SpeciesController.speciate(state, distance_fn, pop)``, with the
``DistanceRows`` callable ``NEAT.tell`` builds from the generation's digest,
preceded inside the same closure by the one ``genome.prepare_distance`` call
that builds it -- the whole ``tell`` tail.  ``tell`` itself is never traced, so
mutation/crossover/transform ``cond`` ops cannot pollute the count.  The closure
rebuilds ``State``/``GenomeTensors``/``SpeciesState`` from positional tensor
leaves inside the trace (``evoxc`` turns only a tensor argument into a graph
input).

Oracles
-------
The predicate's documented arithmetic, the traced IR itself, and the host-gated
run of the same starting state (the other branch).

Gotchas
-------
* ``GATED_ROWS`` is read through the ``_kernels`` module at trace time; patch
  that attribute, never a local ``from ... import``.
* Op histograms are value-independent; only shapes matter.
* ``evoxc.Tensor`` has no ``__array__``; ``support.runner.run_graph`` returns
  host numpy.  pytest hangs when an exception escapes a trace under
  ``--tb=auto``.

Backend is ``"numpy"``: no GPU, no ``iree``, no optional extra, no jax/torch.
"""

from __future__ import annotations

import collections
import dataclasses
from typing import Any

import numpy as np
import pytest

import evoxc

from support.runner import run_graph, spec_like

import tensorneat.algorithm.neat._kernels as kernels_module
import tensorneat.common.errors as errors_module

#: Population size / species-slot capacity of the speciation table.
POP_SIZE = 8
SPECIES_SIZE = 4
#: The four split payload fields of ``genome.utils.GenomeTensors``, in order.
POP_FIELDS = ("node_keys", "node_attrs", "conn_keys", "conn_attrs")
#: ``State`` entries that are runtime tensors; everything else is static config.
STATE_TENSOR_KEYS = ("pop", "generation", "species", "randkey")
#: ``"if"`` ops of the gated host leg, measured: one ``cond`` per traced
#: scan-body instance (``evoxc.scan`` peels step 0 and traces the body again)
#: of each of the two speciation scans, so ``2 * 2``.
HOST_GATED_IF_OPS = 4
#: The two accelerator state spellings the pipeline can register: the bare
#: formulation switch, and the full GPU-class triple ``Pipeline.setup`` derives.
ACCELERATOR_STATICS = (
    {"kernel_mode": "accelerator"},
    {
        "kernel_mode": "accelerator",
        "distance_centre": "tiled",
        "distance_formulation": "index_free",
    },
)


# --------------------------------------------------------------------------- #
# traced-IR helpers
# --------------------------------------------------------------------------- #


def _count_block(block: Any, counter: collections.Counter) -> None:
    """Count op names of one IR block, nested region blocks included.

    The ``return`` terminator is not an executed op; a region's bodies live in
    ``.blocks``, and a plain block is accepted too.
    """
    for op in block.ops:
        if op.name == "return":
            continue
        counter[op.name] += 1
        for region in getattr(op, "regions", ()) or ():
            for inner in getattr(region, "blocks", (region,)):
                _count_block(inner, counter)


def _op_histogram(graph: Any) -> collections.Counter:
    """Op-name histogram of every function of a traced ``evoxc`` graph."""
    counter: collections.Counter = collections.Counter()
    for function in graph.module.functions:
        _count_block(function.entry_block, counter)
    return counter


# --------------------------------------------------------------------------- #
# the traced speciate closure
# --------------------------------------------------------------------------- #


def _trace_speciate(algorithm: Any, state: Any) -> tuple:
    """Trace ``speciate`` for ``state``: ``(graph, leaves, field_names)``.

    The closure is the ``NEAT.tell`` tail only -- digest preparation, the
    ``DistanceRows`` wrapper ``tell`` builds, then
    ``SpeciesController.speciate`` -- so the ``"if"`` count is speciate's own.
    """
    field_names = tuple(field.name for field in dataclasses.fields(type(state.species)))
    static = {
        name: getattr(state, name)
        for name in state.registered_keys()
        if name not in STATE_TENSOR_KEYS
    }
    state_type = type(state)
    payload_type = type(state.pop)
    species_type = type(state.species)

    def kernel(*values: Any) -> tuple:
        payload = payload_type(*values[: len(POP_FIELDS)])
        cursor = len(POP_FIELDS)
        generation = values[cursor]
        cursor += 1
        species_state = species_type(*values[cursor : cursor + len(field_names)])
        current = state_type(**static).register(
            pop=payload, generation=generation, species=species_state
        )
        digest = algorithm.genome.prepare_distance(current, current.pop)
        distance_fn = algorithm._distance_rows(digest, current.pop)
        next_state = algorithm.species.speciate(current, distance_fn, current.pop)
        return tuple(getattr(next_state.species, name) for name in field_names)

    leaves = [
        *(getattr(state.pop, name) for name in POP_FIELDS),
        state.generation,
        *(getattr(state.species, name) for name in field_names),
    ]
    graph = evoxc.trace(kernel, *[spec_like(leaf) for leaf in leaves])
    return graph, leaves, field_names


def _assert_identical(label: str, expected: Any, actual: Any) -> None:
    """Fail naming the field when two leaf arrays differ (``NaN == NaN``)."""
    expected = np.asarray(expected)
    actual = np.asarray(actual)
    assert expected.shape == actual.shape and expected.dtype == actual.dtype, (
        f"{label}: shape/dtype {expected.shape}/{expected.dtype} != "
        f"{actual.shape}/{actual.dtype}"
    )
    if np.issubdtype(expected.dtype, np.floating):
        same = np.array_equal(expected, actual, equal_nan=True)
    else:
        same = np.array_equal(expected, actual)
    assert same, f"{label} differs between the two formulations: {actual!r}"


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture
def algorithm(neat):
    """The ``NEAT`` under test: ``P = 8``, ``S = 4``, no batch chunking."""
    return neat(pop_size=POP_SIZE, species_size=SPECIES_SIZE)


@pytest.fixture
def base_state(algorithm, common, evoxc_key):
    """Generation-0 state carrying no formulation statics (the host defaults)."""
    return algorithm.setup(common.State(randkey=evoxc_key))


@pytest.fixture
def forced_gate(monkeypatch):
    """Set ``_kernels.GATED_ROWS`` so ``gated(S, P)`` answers exactly ``value``.

    ``0`` forces the dense host path; the factory asserts the predicate really
    answered ``False`` so the trace cannot land on the other branch.
    """

    def force(value: int) -> None:
        monkeypatch.setattr(kernels_module, "GATED_ROWS", value)
        expected = SPECIES_SIZE * POP_SIZE <= value
        assert kernels_module.gated(SPECIES_SIZE, POP_SIZE) is expected, (
            f"GATED_ROWS={value} did not make gated({SPECIES_SIZE}, "
            f"{POP_SIZE}) answer {expected}"
        )

    return force


# --------------------------------------------------------------------------- #
# the predicate's truth table
# --------------------------------------------------------------------------- #


def test_gated_host_leg_follows_the_table_bound():
    """Host mode gates a sparse table and not a dense one.

    Oracle: ``species_size * pop_size <= GATED_ROWS`` against the module constant.
    """
    assert kernels_module.gated(SPECIES_SIZE, POP_SIZE, "host") is True, (
        f"a {SPECIES_SIZE} x {POP_SIZE} host table is far below "
        f"GATED_ROWS={kernels_module.GATED_ROWS} and must gate"
    )
    assert kernels_module.gated(kernels_module.GATED_ROWS + 1, 1, "host") is False
    assert kernels_module.gated(100, 100, "host") is False


def test_gated_boundary_is_inclusive():
    """``S * P == GATED_ROWS`` gates; one row more does not (``<=``, not ``<``)."""
    gate = kernels_module.GATED_ROWS
    assert kernels_module.gated(gate, 1, "host") is True, (
        f"S * P == GATED_ROWS == {gate} must gate (the comparison is `<=`)"
    )
    assert kernels_module.gated(gate + 1, 1, "host") is False


def test_gated_accelerator_leg_never_gates():
    """The accelerator leg answers ``False`` for a tiny *and* a large table.

    Oracle: ``logical_and(not accelerator, S * P <= GATED_ROWS)`` -- the tiny
    table is one that *does* gate on the host leg, which is the regression.
    """
    gate = kernels_module.GATED_ROWS
    for species_size, pop_size in (
        (SPECIES_SIZE, POP_SIZE),  # 32: gated on the host leg
        (100, 100),  # 10 000: dense on the host leg too
        (gate, 1),  # exactly the boundary
        (gate + 1, 1),  # one past it
    ):
        assert kernels_module.gated(species_size, pop_size, "accelerator") is False, (
            f"accelerator mode must never gate, but gated({species_size}, "
            f"{pop_size}, 'accelerator') answered True"
        )


def test_gated_default_mode_is_host():
    """The omitted third argument behaves like ``"host"``."""
    for species_size, pop_size in ((SPECIES_SIZE, POP_SIZE), (100, 100)):
        assert kernels_module.gated(species_size, pop_size) is kernels_module.gated(
            species_size, pop_size, "host"
        )


def test_gated_rejects_an_unknown_mode():
    """An unknown formulation is a ``ConfigError``, i.e. a ``TensorNEATError``.

    Oracle: the documented error policy -- a configuration typo raises instead
    of silently selecting one route.
    """
    assert issubclass(errors_module.ConfigError, errors_module.TensorNEATError)
    with pytest.raises(errors_module.ConfigError):
        kernels_module.gated(SPECIES_SIZE, POP_SIZE, "not-a-formulation")


# --------------------------------------------------------------------------- #
# the traced speciate graph follows the gate
# --------------------------------------------------------------------------- #


def test_host_gated_speciate_graph_carries_cond_ops(algorithm, base_state):
    """Sparse host mode traces the lazy rows: ``"if"`` ops are present.

    Oracle: the traced IR -- ``evoxc.cond`` traces as the op name ``"if"``.  The
    count is pinned exactly so a scan that silently loses its ``cond`` fails.
    """
    state = base_state.register(kernel_mode="host")
    assert kernels_module.gated(SPECIES_SIZE, POP_SIZE, "host") is True

    graph, _leaves, _names = _trace_speciate(algorithm, state)
    histogram = _op_histogram(graph)

    assert histogram["if"] == HOST_GATED_IF_OPS, (
        "the gated host leg must build one cond per traced scan-body instance "
        f"(2 scans x (step 0 + re-traced body) = {HOST_GATED_IF_OPS}); the "
        f"traced speciate graph carries {histogram['if']} 'if' op(s) among "
        f"{sum(histogram.values())} ops"
    )


def test_host_dense_speciate_graph_carries_no_cond(algorithm, base_state, forced_gate):
    """A host table forced above the gate traces the dense rows: no ``"if"``.

    Oracle: ``forced_gate(0)`` proves the predicate answered ``False`` first.
    """
    forced_gate(0)
    state = base_state.register(kernel_mode="host")

    graph, _leaves, _names = _trace_speciate(algorithm, state)
    histogram = _op_histogram(graph)

    assert histogram["if"] == 0, (
        "the dense host leg must not build a lazy row; the traced speciate "
        f"graph carries {histogram['if']} 'if' op(s)"
    )


@pytest.mark.parametrize(
    "statics", ACCELERATOR_STATICS, ids=("kernel-mode", "gpu-triple")
)
def test_accelerator_speciate_graph_carries_no_cond(algorithm, base_state, statics):
    """The accelerator leg never traces a ``cond``, even at a table that gates.

    Oracle: the same ``(S, P)`` table answers ``gated(..., "host") is True``, so
    a nonzero count here would be a fallback to the host read-back.
    """
    state = base_state.register(**statics)
    assert kernels_module.gated(SPECIES_SIZE, POP_SIZE, "accelerator") is False

    graph, _leaves, _names = _trace_speciate(algorithm, state)
    histogram = _op_histogram(graph)

    assert histogram["if"] == 0, (
        "accelerator mode must construct zero evoxc.cond operations; the "
        f"traced speciate graph carries {histogram['if']} 'if' op(s) under "
        f"{statics}"
    )


def test_the_three_legs_trace_non_vacuously_different_graphs(
    algorithm, base_state, forced_gate
):
    """All three graphs carry ops, and the gated/accelerator ones differ.

    Oracle: the traced IR.  A trace that produced the same (or an empty) graph
    for every leg would satisfy the zero-``"if"`` assertions above.
    """
    host_gated_graph, _leaves, _names = _trace_speciate(
        algorithm, base_state.register(kernel_mode="host")
    )
    host_gated = _op_histogram(host_gated_graph)

    accelerator_graph, _leaves, _names = _trace_speciate(
        algorithm, base_state.register(kernel_mode="accelerator")
    )
    accelerator = _op_histogram(accelerator_graph)

    forced_gate(0)
    host_dense_graph, _leaves, _names = _trace_speciate(
        algorithm, base_state.register(kernel_mode="host")
    )
    host_dense = _op_histogram(host_dense_graph)

    assert sum(host_gated.values()) > 0, "the gated host graph traced no ops"
    assert sum(host_dense.values()) > 0, "the host dense graph traced no ops"
    assert sum(accelerator.values()) > 0, "the accelerator graph traced no ops"
    assert host_gated["if"] == HOST_GATED_IF_OPS
    assert host_dense["if"] == 0
    assert accelerator["if"] == 0

    assert "if" in host_gated and "if" not in accelerator, (
        "the gated host graph must carry the 'if' op the accelerator graph "
        f"lacks; host gated = {host_gated.most_common(5)}, accelerator = "
        f"{accelerator.most_common(5)}"
    )
    assert host_gated != accelerator, (
        "the gated host and accelerator graphs must not be the same trace: "
        f"host gated = {host_gated.most_common(5)}, accelerator = "
        f"{accelerator.most_common(5)}"
    )


def test_accelerator_and_host_gated_speciate_agree(algorithm, base_state):
    """Both formulations return one bit-identical ``SpeciesState`` from one state.

    Oracle: the *other branch* -- two graphs traced from the same generation-0
    population, ``generation`` and species table, so all ten fields must agree.
    """
    host_graph, host_leaves, field_names = _trace_speciate(
        algorithm, base_state.register(kernel_mode="host")
    )
    accelerator_graph, accelerator_leaves, _names = _trace_speciate(
        algorithm, base_state.register(kernel_mode="accelerator")
    )

    expected = run_graph(host_graph, *host_leaves)
    actual = run_graph(accelerator_graph, *accelerator_leaves)

    for name, left, right in zip(field_names, expected, actual):
        _assert_identical(f"species.{name} (host vs accelerator)", left, right)

    idx2species = np.asarray(actual[field_names.index("idx2species")])
    assert not np.isnan(idx2species).any(), (
        f"the accelerator run left unassigned rows: {idx2species}"
    )
