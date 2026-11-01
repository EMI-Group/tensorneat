"""Unit tests for the genome *payload* and configuration surface.

What this module proves
-----------------------
Before any arithmetic, a genome family owns two contracts: the shape/layout of the
four payload arrays, and the host-side configuration it hands to the algorithm.
Both are pinned down here, on ``BaseGenome``/``DefaultGenome``:

* :meth:`BaseGenome.initialize` writes the dense initial network in key order, so
  the input keys occupy rows ``0..num_inputs-1`` and the output keys rows
  ``max_nodes-num_outputs..max_nodes-1``; unused rows keep the padding sentinels
  (``I_INF`` in the integer key arrays, ``NaN`` in the float attribute arrays) and
  never a leftover value from a previous generation.
* The row/key invariant is what makes :meth:`get_input_idx` /
  :meth:`get_output_idx` usable as *row* indices, which every later kernel and
  every host helper relies on; it is asserted separately from the key values.
* :meth:`setup` *chains* the gene/operator configuration onto an immutable
  ``State`` (``node_gene`` → ``conn_gene`` → ``mutation`` → ``crossover`` →
  ``distance``, the order ``BaseGenome.setup`` documents) and returns a new
  container, leaving the base state untouched.
* The constructor refuses configurations the static shapes cannot honour — a
  ``max_nodes``/``max_conns`` too small for the initial network, or an
  ``init_hidden_layers`` layout needing an in-degree above ``max_in_degree`` — with
  :class:`TensorNEATError`, rather than discovering it inside a trace.
* :meth:`transform` output is a *valid* schedule: ``common.validate_schedule``
  accepts it, and its fields agree with an independent derivation.

Oracles used here (never the kernel itself)
-------------------------------------------
* the row/key convention written into ``support.payloads`` (node key == row
  index; the two input rows first, the output rows last) — the expected keys are
  enumerated literally;
* ``common.tensorization.validate_schedule`` — the package's own invariant
  checker, which must return ``None`` for a schedule the derivation produced;
* ``support.oracle.schedule`` — a second, numpy-only derivation (dense key→slot
  table, in-group rank, counting trick, longest-path levels), compared field by
  field.

Gotchas this module encodes
---------------------------
* Two sentinels, two dtypes: ``I_INF`` in ``*_keys`` (``int32``), ``NaN`` in
  ``*_attrs`` (``float32``). "Row absent" is ``node_keys[p, row] == I_INF`` and
  must coincide with ``isnan(node_attrs[p, row, 0])``; a test that checks only one
  of them cannot see a half-cleared row.
* ``get_output_idx()`` names rows ``max_nodes-num_outputs..max_nodes-1``, *not*
  the last ``num_outputs`` rows of the real network: the invariant is about the
  padded payload, so ``max_nodes`` — never the initial node count — is the offset.
* ``support.oracle.schedule`` documents ``float32`` keys with ``NaN`` padding
  while the payload stores ``int32`` keys with ``I_INF`` padding, so a comparison
  has to reconcile the two sentinels on both sides (the same reconciliation
  ``test/unit/common/test_tensorization.py`` performs; see the report to the
  parent — it is a doc conflict in ``derive_schedule``, not a payload bug).
* ``edge_overflow`` is *data*, never an exception: a too-small ``max_in_degree``
  is reported per individual, and the retained window columns are the lowest
  in-group ranks. Nothing here expects a raise for that case.

Backend is ``"numpy"`` throughout, through the shared runner only; no
``jax``/``torch``, no network, no GPU.
"""

from __future__ import annotations

import pytest


# --------------------------------------------------------------------------- #
# shape and layout of the payload
# --------------------------------------------------------------------------- #


def test_initialize_uses_the_documented_shapes_and_dtypes(payloads, run_kernel):
    """Invariant: the four payload arrays are batch-first with the documented shape and dtype.

    Oracle: the shapes spelled out from the configuration alone —
    ``node_keys (P, N) int32``, ``node_attrs (P, N, node_gene.attr_width) float32``,
    ``conn_keys (P, E, conn_gene.key_width) int32``, ``conn_attrs (P, E,
    conn_gene.attr_width) float32`` — with the fields in ``GenomeTensors`` order,
    so a kernel can take them positionally. ``DefaultNode`` declares four
    attribute columns and ``DefaultConn`` one, and ``BaseConn.fixed_attrs`` names
    two key columns.
    """
    pytest.skip(
        "not implemented: initialize shapes/dtypes/field order of GenomeTensors "
        "(owner: genome/base.BaseGenome.initialize)"
    )


def test_initialize_places_input_keys_on_the_first_rows(payloads, run_kernel):
    """Invariant: ``node_keys[p, 0:num_inputs] == [0, 1, ..., num_inputs-1]``.

    Oracle: hand enumeration of the documented key order — ``initialize`` writes
    nodes in key order, so the input keys are the first rows *and* the first keys.
    Checked on every individual, because a per-individual offset (a key column
    derived from the population index) would still look right in row ``0``.
    """
    pytest.skip(
        "not implemented: input keys == rows 0..num_inputs-1 for every individual "
        "(owner: genome/base.BaseGenome.initialize)"
    )


def test_initialize_places_output_keys_on_the_last_rows(payloads, run_kernel):
    """Invariant: ``node_keys[p, max_nodes-num_outputs:] == [N-num_outputs, ..., N-1]``.

    Oracle: the row/key invariant, read with ``max_nodes`` as the offset — the
    output rows sit at the *end of the padded payload*, not right after the real
    hidden nodes, and they are never moved afterwards.
    """
    pytest.skip(
        "not implemented: output keys == rows max_nodes-num_outputs..max_nodes-1 "
        "(owner: genome/base.BaseGenome.initialize)"
    )


def test_initialize_pads_unused_rows_with_the_documented_sentinels(payloads, run_kernel):
    """Invariant: every row beyond the initial network is ``I_INF`` *and* ``NaN``.

    Oracle: the documented padding pair — ``node_keys == I_INF`` in the integer
    array and ``node_attrs == NaN`` in the float array must coincide row by row,
    and the same for ``conn_keys``/``conn_attrs``. A configuration with
    ``init_hidden_layers`` leaves pads on both sides of the real rows, so the test
    looks at "rows not named by the initial network", not at a single tail slice.
    """
    pytest.skip(
        "not implemented: I_INF/NaN padding of unused node and connection rows "
        "(owner: genome/base.BaseGenome.initialize)"
    )


def test_initialize_writes_the_dense_initial_network(payloads, run_kernel):
    """Invariant: the initial network is ``num_inputs -> init_hidden_layers -> num_outputs``, dense.

    Oracle: hand enumeration. With no hidden layer the payload holds exactly
    ``num_inputs * num_outputs`` connections, each with key columns
    ``(input_key, output_key)``, finite weights and no duplicate
    ``(in_key, out_key)`` pair; with one hidden layer the edge count is
    ``num_inputs * h + h * num_outputs``. The ``historical_marker`` column, when
    the conn gene declares one, is the connection's row index.
    """
    pytest.skip(
        "not implemented: dense layered initial network edge set and key columns "
        "(owner: genome/base.BaseGenome.initialize)"
    )


def test_initialize_is_reproducible_for_one_key_and_shape(payloads, run_kernel, evoxc_key):
    """Invariant: the same ``(key, config, pop_size)`` triple reproduces the payload exactly.

    Oracle: two independent ``initialize`` calls on the same rank-0 ``evoxc.random``
    key must agree bit for bit (``np.array_equal`` on all four arrays), and the
    *structure* must also agree across two different keys (the dense layout is
    deterministic even though the attribute draws are not). This is the
    ``(seed, config, shape)`` reproduction contract, checked at its source.
    """
    pytest.skip(
        "not implemented: initialize determinism for one key, structure-stability across keys "
        "(owner: genome/base.BaseGenome.initialize)"
    )


# --------------------------------------------------------------------------- #
# the row/key invariant, as an index contract
# --------------------------------------------------------------------------- #


def test_get_input_idx_and_get_output_idx_are_row_indices(payloads):
    """Invariant: the two helpers return exactly the rows their keys occupy.

    Oracle: ``get_input_idx() == list(range(num_inputs))`` and
    ``get_output_idx() == list(range(max_nodes - num_outputs, max_nodes))``, then
    re-derived from a payload: ``node_keys[0, index] == index`` for every returned
    index. The second half is what makes the helpers usable as *rows* by
    ``network_dict``/``visualize`` and by the output read-out, and it fails for a
    genome that stores keys and rows differently (e.g. an ``OriginConn`` payload
    whose hidden keys were renumbered).
    """
    pytest.skip(
        "not implemented: get_input_idx/get_output_idx == the rows their keys occupy "
        "(owner: genome/base.BaseGenome.get_input_idx, get_output_idx)"
    )


# --------------------------------------------------------------------------- #
# setup(): the host-side configuration the algorithm reads
# --------------------------------------------------------------------------- #


def test_setup_registers_the_gene_and_operator_configuration(payloads):
    """Invariant: ``setup(State())`` carries one entry per gene and per operator.

    Oracle: the documented chain order — ``node_gene``, ``conn_gene``,
    ``mutation``, ``crossover``, ``distance`` — registered on the returned
    ``State`` (its ``registered_keys()`` order is insertion order), each entry
    being the very object the genome was constructed with. A genome whose genes
    were built per instance must register *its own* objects; a shared mutable
    default would show up here as two genomes registering the same instance.
    """
    pytest.skip(
        "not implemented: setup registers node_gene/conn_gene/mutation/crossover/distance "
        "(owner: genome/base.BaseGenome.setup)"
    )


def test_setup_returns_a_new_state_and_leaves_the_base_untouched(payloads):
    """Invariant: ``setup`` is functional — the base ``State`` is never mutated.

    Oracle: a fresh ``State()`` is empty before the call and still empty after it,
    while the returned container is a *different* object carrying the five new
    entries. ``State`` is deliberately immutable (see ``test/unit/common``), so a
    genome that registered on ``state`` in place would silently corrupt a caller's
    copy.
    """
    pytest.skip(
        "not implemented: setup returns a new State and leaves its argument unchanged "
        "(owner: genome/base.BaseGenome.setup, common/state.State.register)"
    )


# --------------------------------------------------------------------------- #
# constructor validation
# --------------------------------------------------------------------------- #


def test_constructor_rejects_max_nodes_smaller_than_the_initial_network(payloads):
    """Invariant: a ``max_nodes`` below the initial node count raises ``TensorNEATError``.

    Oracle: the node count of the configured layout, ``num_inputs +
    sum(init_hidden_layers) + num_outputs``; one below it must fail at
    construction (the docstring puts the check in ``__init__``/``initialize``),
    and the error must be ``TensorNEATError`` — a bare ``ValueError`` would escape
    the package's configuration-error hierarchy.
    """
    pytest.skip(
        "not implemented: max_nodes below the initial node count raises TensorNEATError "
        "(owner: genome/default.DefaultGenome.__init__, genome/base.BaseGenome.__init__)"
    )


def test_constructor_rejects_max_conns_smaller_than_the_initial_network(payloads):
    """Invariant: a ``max_conns`` below the dense edge count raises ``TensorNEATError``.

    Oracle: the edge count of the configured layout — ``num_inputs * h1 + ... +
    hk * num_outputs`` — computed on the host from ``init_hidden_layers``; one
    below it must fail at construction instead of dropping the surplus edges,
    because a silently truncated initial network is a different experiment.
    """
    pytest.skip(
        "not implemented: max_conns below the dense initial edge count raises TensorNEATError "
        "(owner: genome/default.DefaultGenome.__init__, genome/base.BaseGenome.__init__)"
    )


def test_constructor_rejects_a_hidden_layer_wider_than_max_in_degree(payloads):
    """Invariant: ``init_hidden_layers`` needing an in-degree above ``max_in_degree`` raises.

    Oracle: the documented static window width ``K`` is ``max(initial in-degree,
    4)`` when left at ``None``; passing a hidden layer wider than an explicit
    ``K`` must raise ``TensorNEATError`` rather than produce a schedule with a
    positive ``edge_overflow`` on generation zero. ``max_in_degree`` is a
    performance knob, but it is never silently lossy at construction time.
    """
    pytest.skip(
        "not implemented: init_hidden_layers wider than max_in_degree raises TensorNEATError "
        "(owner: genome/base.BaseGenome.__init__)"
    )


# --------------------------------------------------------------------------- #
# transform(): the schedule of an initialized payload
# --------------------------------------------------------------------------- #


def test_transform_is_accepted_by_validate_schedule(payloads, common, run_transform):
    """Invariant: ``validate_schedule(transform(...))`` returns ``None``.

    Oracle: the package's own invariant checker — ``in_conn``/``in_src``
    ``I_INF`` in the same positions, ``level == I_INF`` iff ``topo_key ==
    I_INF``, a non-negative ``edge_overflow`` and in-range ``input_slot``/
    ``output_slot``. A derivation that returns a self-consistent *looking* but
    internally inconsistent schedule (a padded ``in_src`` slot, an absent node
    with a finite level) passes a value comparison against a similarly wrong
    oracle but fails this one.
    """
    pytest.skip(
        "not implemented: validate_schedule accepts the derivation on a fresh payload "
        "(owner: common/tensorization.validate_schedule, genome/base.BaseGenome.transform)"
    )


def test_transform_matches_the_independent_oracle(payloads, oracle, run_transform):
    """Invariant: every ``Transformed`` field equals ``support.oracle.schedule``.

    Oracle: the numpy-only derivation, field by field (``level``, ``in_conn``/
    ``in_src`` windows in connection-row order, ``src``/``dst``/``edge_valid``,
    ``input_slot``/``output_slot``, ``edge_overflow == 0``). Both sides are
    brought to one dtype/sentinel convention first: the payload stores ``int32``
    keys with ``I_INF`` while ``oracle.schedule`` documents ``float32`` keys with
    ``NaN``.
    """
    pytest.skip(
        "not implemented: transform equals support.oracle.schedule on a fresh payload "
        "(owner: genome/base.BaseGenome.transform, common/tensorization.derive_schedule)"
    )


def test_transform_reports_edge_overflow_when_max_in_degree_is_too_small(
    payloads, run_transform
):
    """Invariant: a window narrower than a node's in-degree shows up as ``edge_overflow > 0``.

    Oracle: the documented lossy-schedule diagnostic. On the dense initial
    network every output node has in-degree ``num_inputs``, so constructing with
    ``max_in_degree=1`` must report ``edge_overflow > 0`` for every individual and
    *drop* the surplus window columns — it must not raise, and it must not
    silently pretend the node has a single incoming edge.
    """
    pytest.skip(
        "not implemented: edge_overflow reports a too-small max_in_degree as data "
        "(owner: common/tensorization.derive_schedule, genome/base.BaseGenome.transform)"
    )
