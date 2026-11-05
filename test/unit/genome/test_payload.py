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
* :meth:`setup` *chains* the gene/operator configuration — ``node_gene`` →
  ``conn_gene`` → ``mutation`` → ``crossover`` → ``distance``, the order
  ``BaseGenome.setup`` documents — on an immutable ``State``. The gene and
  operator configuration itself is constructor-level (``genome/CONTEXT.md``:
  "nothing on the compute path reads ``State``"), so the chain validates each
  piece on the host, registers no entries of its own, and hands the incoming
  container forward with the caller's entries intact.
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
  field, dtype included.

Gotchas this module encodes
---------------------------
* Two sentinels, two dtypes: ``I_INF`` in ``*_keys`` (``int32``), ``NaN`` in
  ``*_attrs`` (``float32``). "Row absent" is ``node_keys[p, row] == I_INF`` and
  must coincide with ``isnan(node_attrs[p, row, 0])``; a test that checks only one
  of them cannot see a half-cleared row.
* ``get_output_idx()`` names rows ``max_nodes-num_outputs..max_nodes-1``, *not*
  the last ``num_outputs`` rows of the real network: the invariant is about the
  padded payload, so ``max_nodes`` — never the initial node count — is the offset.
  The tests therefore run a genome that *has* hidden layers, where the two
  readings differ.
* ``support.oracle.schedule`` documents ``float32`` keys with ``NaN`` padding
  while the payload stores ``int32`` keys with ``I_INF`` padding. Both spellings
  are accepted by the derivation (``common.tensorization._key_present``) and by
  the oracle (``support.oracle._present_keys``), so the payload arrays are fed to
  it unchanged — the comparison is on values and dtypes, never on a re-encoded
  copy.
* ``edge_overflow`` is *data*, never an exception: a too-small ``max_in_degree``
  is reported per individual, and the retained window columns are the lowest
  in-group ranks. Since the *constructor* already refuses a ``max_in_degree``
  below the initial in-degree, the lossy case is exercised on a hand-built
  payload whose in-degree exceeds a legal ``K`` — nothing here expects a raise.

Backend is ``"numpy"`` throughout, through the shared runner only; no
``jax``/``torch``, no network, no GPU.
"""

from __future__ import annotations

import evoxc
import numpy as np
import pytest


# --------------------------------------------------------------------------- #
# host-side helpers
# --------------------------------------------------------------------------- #


def _numpy_payload(tensors):
    """The four payload arrays of a ``GenomeTensors`` result, as plain numpy.

    Args:
        tensors: A ``GenomeTensors`` (or any four-array sequence) returned by
            ``run_kernel`` — its leaves are already host numpy.

    Returns:
        ``(node_keys, node_attrs, conn_keys, conn_attrs)``, in payload order.
    """
    return tuple(np.asarray(array) for array in tensors)


def _present_edges(conn_keys_one, i_inf):
    """The ``(in_key, out_key)`` pair of every present connection row, in row order.

    Args:
        conn_keys_one: ``(E, CK)`` connection keys of one individual.
        i_inf: The ``int32`` padding sentinel.

    Returns:
        A list of ``(in_key, out_key)`` int pairs; absent rows are skipped.
    """
    present = conn_keys_one[:, 0] != i_inf
    return [(int(source), int(target)) for source, target in conn_keys_one[present, :2]]


def _recording(cls, calls, name):
    """A ``cls`` subclass whose ``setup`` logs itself and then does the real work.

    Used to observe the *order* of ``BaseGenome.setup``'s chain without
    re-implementing it: the subclass still calls ``super().setup(state)``, so a
    failing gene/operator validation surfaces exactly as it would in production.

    Args:
        cls: The real gene/operator class to wrap (``DefaultNode``, ...).
        calls: List the recorded names are appended to, in call order.
        name: The attribute name this instance is installed under.

    Returns:
        An instance of the recording subclass.
    """

    class _Recording(cls):
        def setup(self, state=None):
            calls.append(name)
            return super().setup(state)

    return _Recording()


# --------------------------------------------------------------------------- #
# shape and layout of the payload
# --------------------------------------------------------------------------- #


def test_initialize_uses_the_documented_shapes_and_dtypes(
    payloads, genome, common, run_kernel, evoxc_key
):
    """Invariant: the four payload arrays are batch-first with the documented shape and dtype.

    Oracle: the shapes spelled out from the configuration alone —
    ``node_keys (P, N) int32``, ``node_attrs (P, N, node_gene.attr_width) float32``,
    ``conn_keys (P, E, conn_gene.key_width) int32``, ``conn_attrs (P, E,
    conn_gene.attr_width) float32`` — with the fields in ``GenomeTensors`` order,
    so a kernel can take them positionally. ``DefaultNode`` declares four
    attribute columns and ``DefaultConn`` one, and ``BaseConn.fixed_attrs`` names
    two key columns.

    The same test pins the two documented sentinels, which the dtype assertions
    alone do not: an *absent* row is ``I_INF`` in every key column **and** ``NaN``
    in every attribute column, a *present* row carries keys inside
    ``[0, key_span)`` with finite attributes, and the initial network is the dense
    ``num_inputs -> num_outputs`` wiring (``num_inputs * num_outputs`` edges).
    """
    pop_size, max_nodes, max_conns = 3, 8, 16
    g = payloads.default_genome(
        num_inputs=2, num_outputs=1, max_nodes=max_nodes, max_conns=max_conns
    )
    state = payloads.genome_state(g)

    result = run_kernel(g.initialize, state, evoxc_key, pop_size)

    # The field order is the positional contract every kernel relies on.
    assert isinstance(result, genome.GenomeTensors)
    assert result.field_names == ("node_keys", "node_attrs", "conn_keys", "conn_attrs")
    node_keys, node_attrs, conn_keys, conn_attrs = _numpy_payload(result)

    # The declared gene widths, spelled out: DefaultNode has 4 attr columns,
    # BaseConn 2 key columns and DefaultConn 1 attr column.
    assert g.node_gene.attr_width == 4
    assert g.conn_gene.key_width == 2
    assert g.conn_gene.attr_width == 1

    assert node_keys.shape == (pop_size, max_nodes)
    assert node_attrs.shape == (pop_size, max_nodes, g.node_gene.attr_width)
    assert conn_keys.shape == (pop_size, max_conns, g.conn_gene.key_width)
    assert conn_attrs.shape == (pop_size, max_conns, g.conn_gene.attr_width)

    assert node_keys.dtype == np.int32
    assert conn_keys.dtype == np.int32
    assert node_attrs.dtype == np.float32
    assert conn_attrs.dtype == np.float32

    # The two sentinels never mix.  "Row absent" is exactly ``I_INF`` in every key
    # column *and* ``NaN`` in every attribute column of that row; a present row
    # carries keys inside ``[0, key_span)`` and finite attributes.  (An ``int32``
    # array cannot hold ``NaN``, so a ``np.isnan`` probe on a key array proves
    # nothing; the assertions below are the actual contract.)
    num_inputs, num_outputs = g.num_inputs, g.num_outputs
    node_absent = node_keys == common.I_INF
    assert node_absent.any() and not node_absent.all()
    # The initial network is ``num_inputs -> num_outputs`` dense: three node rows
    # are present (2 inputs + 1 output) and the rest are padding.
    assert node_absent.sum(axis=1).tolist() == [
        max_nodes - (num_inputs + num_outputs)
    ] * pop_size
    present_node_keys = node_keys[~node_absent]
    assert np.all(present_node_keys >= 0)
    assert np.all(present_node_keys < g.key_span)
    for p in range(pop_size):
        keys = node_keys[p][~node_absent[p]]
        # No duplicate key inside an individual, and the present keys ascend
        # because ``initialize`` writes the input/hidden keys first and the
        # output keys (the highest ones) last.
        assert np.array_equal(keys, np.sort(keys)), f"individual {p} keys {keys}"
        assert len(set(keys.tolist())) == len(keys), f"individual {p} repeats a key"
    # The float attribute rows are NaN in *every* column exactly where the key row
    # is absent (no half-cleared row), and finite in every column where it is not.
    assert np.array_equal(np.isnan(node_attrs).all(axis=-1), node_absent)
    assert np.all(np.isfinite(node_attrs[~node_absent]))

    # The same two-sentinel contract for connections, plus the initial wiring:
    # ``num_inputs * num_outputs`` edges from every input row to the output row.
    conn_absent = conn_keys[:, :, 0] == common.I_INF
    assert conn_absent.any() and not conn_absent.all()
    assert conn_absent.sum(axis=1).tolist() == [
        max_conns - num_inputs * num_outputs
    ] * pop_size
    assert np.array_equal(
        conn_keys == common.I_INF,
        np.broadcast_to(conn_absent[:, :, None], conn_keys.shape),
    ), "every key column of a connection row shares the sentinel"
    present_conn_keys = conn_keys[~conn_absent]
    assert np.all(present_conn_keys >= 0)
    assert np.all(present_conn_keys < g.key_span)
    assert np.array_equal(np.isnan(conn_attrs).all(axis=-1), conn_absent)
    assert np.all(np.isfinite(conn_attrs[~conn_absent]))
    expected_edges = {
        (source, max_nodes - 1) for source in range(num_inputs)
    }
    for p in range(pop_size):
        edges = {
            (int(row[0]), int(row[1]))
            for row in conn_keys[p, ~conn_absent[p], :2]
        }
        assert edges == expected_edges, f"individual {p} writes {sorted(edges)}"


def test_initialize_places_input_keys_on_the_first_rows(
    payloads, common, run_kernel, evoxc_key
):
    """Invariant: ``node_keys[p, 0:num_inputs] == [0, 1, ..., num_inputs-1]``.

    Oracle: hand enumeration of the documented key order — ``initialize`` writes
    nodes in key order, so the input keys are the first rows *and* the first keys.
    Checked on every individual, because a per-individual offset (a key column
    derived from the population index) would still look right in row ``0``. The
    hidden keys of the initial network continue right after the inputs (row ``i``
    holds key ``i``), which is the other half of the same convention.
    """
    num_inputs, hidden = 2, 3
    g = payloads.default_genome(
        num_inputs=num_inputs,
        num_outputs=1,
        max_nodes=8,
        max_conns=16,
        init_hidden_layers=(hidden,),
    )
    state = payloads.genome_state(g)
    node_keys, *_ = _numpy_payload(run_kernel(g.initialize, state, evoxc_key, 4))

    expected_inputs = np.arange(num_inputs, dtype=np.int32)
    expected_hidden = np.arange(num_inputs, num_inputs + hidden, dtype=np.int32)
    for p in range(node_keys.shape[0]):
        assert np.array_equal(node_keys[p, :num_inputs], expected_inputs), f"individual {p}"
        assert np.array_equal(
            node_keys[p, num_inputs : num_inputs + hidden], expected_hidden
        ), f"individual {p}"

    # Every input key names exactly one row: no duplicate and no stray copy.
    for key in expected_inputs:
        counts = (node_keys == key).sum(axis=1)
        assert np.all(counts == 1), f"key {int(key)} appears {counts.tolist()} times"

    # The rows between the real network and the output block are padding, not
    # further inputs: the input keys are the first rows and only the first rows.
    assert np.all(
        node_keys[:, num_inputs + hidden : g.max_nodes - g.num_outputs] == common.I_INF
    )


def test_initialize_places_output_keys_on_the_last_rows(
    payloads, common, run_kernel, evoxc_key
):
    """Invariant: ``node_keys[p, max_nodes-num_outputs:] == [N-num_outputs, ..., N-1]``.

    Oracle: the row/key invariant, read with ``max_nodes`` as the offset — the
    output rows sit at the *end of the padded payload*, not right after the real
    hidden nodes, and they are never moved afterwards. The genome therefore
    carries a hidden layer, so the two readings ("tail of the padded payload" vs
    "after the hidden nodes") give different rows: the rows between the hidden
    nodes and the output block must be padding.
    """
    num_inputs, hidden, num_outputs, max_nodes = 2, 3, 1, 10
    g = payloads.default_genome(
        num_inputs=num_inputs,
        num_outputs=num_outputs,
        max_nodes=max_nodes,
        max_conns=16,
        init_hidden_layers=(hidden,),
    )
    state = payloads.genome_state(g)
    node_keys, *_ = _numpy_payload(run_kernel(g.initialize, state, evoxc_key, 4))

    expected_outputs = np.arange(max_nodes - num_outputs, max_nodes, dtype=np.int32)
    for p in range(node_keys.shape[0]):
        assert np.array_equal(
            node_keys[p, max_nodes - num_outputs :], expected_outputs
        ), f"individual {p}"
    # The output block is not "the rows right after the hidden nodes": those are pads.
    gap_start = num_inputs + hidden
    assert gap_start < max_nodes - num_outputs
    assert np.all(node_keys[:, gap_start : max_nodes - num_outputs] == common.I_INF)
    # An output key names exactly one row.
    for key in expected_outputs:
        assert np.all((node_keys == key).sum(axis=1) == 1)


def test_initialize_pads_unused_rows_with_the_documented_sentinels(
    payloads, common, run_kernel, evoxc_key
):
    """Invariant: every row beyond the initial network is ``I_INF`` *and* ``NaN``.

    Oracle: the documented padding pair — ``node_keys == I_INF`` in the integer
    array and ``node_attrs == NaN`` in the float array must coincide row by row,
    and the same for ``conn_keys``/``conn_attrs``. A configuration with
    ``init_hidden_layers`` leaves pads on both sides of the real rows, so the test
    looks at "rows not named by the initial network", not at a single tail slice.
    """
    num_inputs, hidden, num_outputs, max_nodes = 2, 3, 1, 8
    max_conns = 16
    g = payloads.default_genome(
        num_inputs=num_inputs,
        num_outputs=num_outputs,
        max_nodes=max_nodes,
        max_conns=max_conns,
        init_hidden_layers=(hidden,),
    )
    state = payloads.genome_state(g)
    node_keys, node_attrs, conn_keys, conn_attrs = _numpy_payload(
        run_kernel(g.initialize, state, evoxc_key, 3)
    )

    # ---- node rows: key sentinel and attribute sentinel coincide, everywhere.
    absent = node_keys == common.I_INF
    node_count = num_inputs + hidden + num_outputs
    assert int(absent[0].sum()) == max_nodes - node_count
    expected_absent = np.zeros(max_nodes, dtype=bool)
    expected_absent[num_inputs + hidden : max_nodes - num_outputs] = True
    assert np.array_equal(absent, np.broadcast_to(expected_absent, absent.shape))
    assert np.array_equal(absent, np.isnan(node_attrs).all(axis=2))
    assert np.array_equal(absent, np.isnan(node_attrs[:, :, 0]))
    # Present rows carry real values in the first column — never a half-cleared row.
    assert not np.isnan(node_attrs[:, :, 0][~absent]).any()

    # ---- connection rows: exactly the dense edges, the rest I_INF + NaN.
    conn_absent = conn_keys == common.I_INF
    conn_count = num_inputs * hidden + hidden * num_outputs
    # "Row absent" is all-or-nothing: the sentinel never sits next to a real key.
    row_absent = conn_absent.all(axis=2)
    assert np.array_equal(row_absent, conn_absent.any(axis=2))
    assert np.array_equal(
        conn_absent, np.broadcast_to(row_absent[:, :, None], conn_absent.shape)
    )
    assert int(row_absent[0].sum()) == max_conns - conn_count
    # The float side agrees row by row, and only absent rows are NaN.
    assert np.array_equal(row_absent, np.isnan(conn_attrs).all(axis=2))
    assert not np.isnan(conn_attrs[:, :, 0][~row_absent]).any()

    # The padding *pattern* is a property of the shape, not of the individual
    # (the attribute values themselves are per-individual draws).
    for p in range(1, node_keys.shape[0]):
        assert np.array_equal(node_keys[p], node_keys[0])
        assert np.array_equal(conn_keys[p], conn_keys[0])
        assert np.array_equal(np.isnan(node_attrs[p]), np.isnan(node_attrs[0]))
        assert np.array_equal(np.isnan(conn_attrs[p]), np.isnan(conn_attrs[0]))


def test_initialize_writes_the_dense_initial_network(
    payloads, common, genome, run_kernel, evoxc_key
):
    """Invariant: the initial network is ``num_inputs -> init_hidden_layers -> num_outputs``, dense.

    Oracle: hand enumeration. With no hidden layer the payload holds exactly
    ``num_inputs * num_outputs`` connections, each with key columns
    ``(input_key, output_key)``, finite weights and no duplicate
    ``(in_key, out_key)`` pair; with one hidden layer the edge count is
    ``num_inputs * h + h * num_outputs``. The ``historical_marker`` column, when
    the conn gene declares one, is the connection's row index.
    """
    # ---- no hidden layer: 2 inputs -> 1 output.
    g = payloads.default_genome(
        num_inputs=2, num_outputs=1, max_nodes=8, max_conns=16
    )
    state = payloads.genome_state(g)
    node_keys, _, conn_keys, conn_attrs = _numpy_payload(
        run_kernel(g.initialize, state, evoxc_key, 2)
    )
    assert g.conn_gene.uses_historical_marker is False
    assert g.conn_gene.key_width == 2
    edges = _present_edges(conn_keys[0], common.I_INF)
    assert edges == [(0, 7), (1, 7)]
    assert len(set(edges)) == len(edges)
    assert np.all(np.isfinite(conn_attrs[0, : len(edges), 0]))
    # Nodes: inputs 0, 1 and the output key at the tail of the padded payload.
    present_keys = node_keys[0][node_keys[0] != common.I_INF]
    assert sorted(present_keys.tolist()) == [0, 1, 7]

    # ---- one hidden layer: 2 inputs -> 3 hidden -> 1 output, in layer order.
    gh = payloads.default_genome(
        num_inputs=2, num_outputs=1, max_nodes=8, max_conns=16, init_hidden_layers=(3,)
    )
    hstate = payloads.genome_state(gh)
    node_keys, _, conn_keys, conn_attrs = _numpy_payload(
        run_kernel(gh.initialize, hstate, evoxc_key, 2)
    )
    edges = _present_edges(conn_keys[0], common.I_INF)
    assert edges == [
        (0, 2),
        (0, 3),
        (0, 4),
        (1, 2),
        (1, 3),
        (1, 4),
        (2, 7),
        (3, 7),
        (4, 7),
    ]
    assert len(edges) == 2 * 3 + 3 * 1
    assert len(set(edges)) == len(edges)
    assert np.all(np.isfinite(conn_attrs[0, : len(edges), 0]))
    assert np.all(np.isnan(conn_attrs[0, len(edges) :, 0]))

    # ---- an OriginConn payload carries the historical marker == row index.
    marker_genome = genome.DefaultGenome(
        num_inputs=2,
        num_outputs=1,
        max_nodes=8,
        max_conns=16,
        node_gene=genome.DefaultNode(),
        conn_gene=genome.OriginConn(),
        mutation=genome.DefaultMutation(),
        crossover=genome.DefaultCrossover(),
        distance=genome.DefaultDistance(),
    )
    assert marker_genome.conn_gene.uses_historical_marker is True
    assert marker_genome.conn_gene.key_width == 3
    marker_state = payloads.genome_state(marker_genome)
    _, _, marker_keys, _ = _numpy_payload(
        run_kernel(marker_genome.initialize, marker_state, evoxc_key, 2)
    )
    marker_count = 2
    assert np.array_equal(
        marker_keys[0, :marker_count, 2], np.arange(marker_count, dtype=np.int32)
    )
    assert np.all(marker_keys[0, marker_count:, :] == common.I_INF)


def test_initialize_is_reproducible_for_one_key_and_shape(
    payloads, common, run_kernel, evoxc_key, seed
):
    """Invariant: the same ``(key, config, pop_size)`` triple reproduces the payload exactly.

    Oracle: two independent ``initialize`` calls on the same rank-0 ``evoxc.random``
    key must agree bit for bit (``np.array_equal`` on all four arrays), and the
    *structure* must also agree across two different keys (the dense layout is
    deterministic even though the attribute draws are not). This is the
    ``(seed, config, shape)`` reproduction contract, checked at its source.
    """
    pop_size = 4
    g = payloads.default_genome(
        num_inputs=2, num_outputs=1, max_nodes=8, max_conns=16
    )
    state = payloads.genome_state(g)
    names = ("node_keys", "node_attrs", "conn_keys", "conn_attrs")

    first = _numpy_payload(run_kernel(g.initialize, state, evoxc_key, pop_size))
    second = _numpy_payload(run_kernel(g.initialize, state, evoxc_key, pop_size))
    for name, left, right in zip(names, first, second):
        assert np.array_equal(left, right, equal_nan=True), f"{name} is not reproducible"

    # A different key keeps the structure and re-draws the attributes.
    other_key = evoxc.random.key(seed + 1)
    other = _numpy_payload(run_kernel(g.initialize, state, other_key, pop_size))
    assert np.array_equal(first[0], other[0])  # node_keys: the dense layout
    assert np.array_equal(first[2], other[2])  # conn_keys: the dense layer order
    assert not np.array_equal(first[1], other[1], equal_nan=True)  # node_attrs
    assert not np.array_equal(first[3], other[3], equal_nan=True)  # conn_attrs

    # The single draw feeds every individual: no two individuals share a stream.
    present = first[0][0] != common.I_INF
    node_attrs = first[1][:, present, :]
    conn_attrs = first[3][:, :2, :]
    for i in range(pop_size):
        for j in range(i + 1, pop_size):
            assert not np.array_equal(node_attrs[i], node_attrs[j], equal_nan=True)
            assert not np.array_equal(conn_attrs[i], conn_attrs[j], equal_nan=True)


# --------------------------------------------------------------------------- #
# the row/key invariant, as an index contract
# --------------------------------------------------------------------------- #


def test_get_input_idx_and_get_output_idx_are_row_indices(
    payloads, common, run_kernel, evoxc_key
):
    """Invariant: the two helpers return exactly the rows their keys occupy.

    Oracle: ``get_input_idx() == list(range(num_inputs))`` and
    ``get_output_idx() == list(range(max_nodes - num_outputs, max_nodes))``, then
    re-derived from a payload: ``node_keys[p, index] == index`` for every returned
    index and every individual. That second half is what makes the helpers usable
    as *rows* by ``network_dict``/``visualize`` and by the output read-out, and it
    fails for a genome that stores keys and rows differently (e.g. an
    ``OriginConn`` payload whose hidden keys were renumbered).
    """
    num_inputs, hidden, num_outputs, max_nodes = 2, 3, 1, 10
    g = payloads.default_genome(
        num_inputs=num_inputs,
        num_outputs=num_outputs,
        max_nodes=max_nodes,
        max_conns=16,
        init_hidden_layers=(hidden,),
    )
    input_idx = g.get_input_idx()
    output_idx = g.get_output_idx()
    assert input_idx == list(range(num_inputs)) == [0, 1]
    assert output_idx == list(range(max_nodes - num_outputs, max_nodes)) == [9]

    state = payloads.genome_state(g)
    node_keys, *_ = _numpy_payload(run_kernel(g.initialize, state, evoxc_key, 3))
    for p in range(node_keys.shape[0]):
        for row in input_idx:
            assert node_keys[p, row] == row, f"input row {row} of individual {p}"
        for row in output_idx:
            assert node_keys[p, row] == row, f"output row {row} of individual {p}"

    # The helpers are drawn from the padded payload, so the two sets are disjoint
    # and every index is an in-range row.
    assert not set(input_idx) & set(output_idx)
    for index in input_idx + output_idx:
        assert 0 <= index < max_nodes
    # Hidden rows are *not* named by either helper.
    assert set(input_idx + output_idx).isdisjoint(range(num_inputs, num_inputs + hidden))


# --------------------------------------------------------------------------- #
# setup(): the host-side configuration the algorithm reads
# --------------------------------------------------------------------------- #


def test_setup_registers_the_gene_and_operator_configuration(genome, common):
    """Invariant: ``setup`` chains every gene and operator setup, in the documented order.

    Oracle: the chain order ``BaseGenome.setup`` documents — ``node_gene``,
    ``conn_gene``, ``mutation``, ``crossover``, ``distance``. Each of the five
    parts is a recording subclass of the *real* class, so the assertion also proves
    the real ``setup`` ran (and, on a broken gene/operator table, would surface the
    same :class:`ConfigError` production would raise). The gene/operator
    configuration is constructor-level (``genome/CONTEXT.md``: nothing on the
    compute path reads ``State``), so the chain validates the parts and registers
    no entries of its own — the assertion on the returned container pins that.
    """
    calls: list[str] = []
    parts = {
        "node_gene": _recording(genome.DefaultNode, calls, "node_gene"),
        "conn_gene": _recording(genome.DefaultConn, calls, "conn_gene"),
        "mutation": _recording(genome.DefaultMutation, calls, "mutation"),
        "crossover": _recording(genome.DefaultCrossover, calls, "crossover"),
        "distance": _recording(genome.DefaultDistance, calls, "distance"),
    }
    g = genome.DefaultGenome(
        num_inputs=2,
        num_outputs=1,
        max_nodes=8,
        max_conns=16,
        **parts,
    )
    assert calls == []  # construction is not a setup

    state = g.setup(common.State())

    assert calls == ["node_gene", "conn_gene", "mutation", "crossover", "distance"]
    assert isinstance(state, common.State)
    # Constructor-level configuration: the chain adds nothing to the container.
    assert list(state.registered_keys()) == []

    # The genome holds the very objects it was constructed with — no shared
    # mutable default, and no per-instance copy that the chain ignored.
    for name, part in parts.items():
        assert getattr(g, name) is part, name


def test_setup_returns_a_new_state_and_leaves_the_base_untouched(
    payloads, common, run_kernel, evoxc_key
):
    """Invariant: ``setup`` is functional — the base ``State`` is never mutated.

    Oracle: ``State`` is deliberately immutable (``test/unit/common``), so a
    genome that registered on ``state`` in place would silently corrupt a
    caller's copy. ``setup(base)`` must hand the *caller's* entries on
    (``pop_size`` here, which ``initialize`` then reads through
    ``_resolve_pop_size``) and must not drop or rewrite anything the caller
    registered; an empty base stays empty, and ``setup()`` with no argument starts
    from a fresh ``State`` (reconciliation #5: no ``State()`` default argument).
    """
    g = payloads.default_genome(
        num_inputs=2, num_outputs=1, max_nodes=8, max_conns=16
    )

    base = common.State(pop_size=3)
    state = g.setup(base)
    assert list(base.registered_keys()) == ["pop_size"]
    assert base.pop_size == 3
    assert "pop_size" in state
    assert state.pop_size == 3

    # The returned container is what a kernel takes: initialize reads pop_size off it.
    node_keys, *_ = _numpy_payload(run_kernel(g.initialize, state, evoxc_key))
    assert node_keys.shape == (3, 8)

    # ... and the base state is still the untouched, three-individual one.
    assert _numpy_payload(run_kernel(g.initialize, base, evoxc_key))[0].shape == (3, 8)
    assert list(base.registered_keys()) == ["pop_size"]

    # An empty base stays empty; a missing base is a fresh State, not a shared one.
    empty = common.State()
    assert list(g.setup(empty).registered_keys()) == []
    assert list(empty.registered_keys()) == []
    assert isinstance(g.setup(), common.State)
    assert list(g.setup().registered_keys()) == []


# --------------------------------------------------------------------------- #
# constructor validation
# --------------------------------------------------------------------------- #


def test_constructor_rejects_max_nodes_smaller_than_the_initial_network(
    payloads, common
):
    """Invariant: a ``max_nodes`` below the initial node count raises ``TensorNEATError``.

    Oracle: the node count of the configured layout, ``num_inputs +
    sum(init_hidden_layers) + num_outputs``; one below it must fail at
    construction (the docstring puts the check in ``__init__``/``initialize``),
    and the error must be ``TensorNEATError`` — a bare ``ValueError`` would escape
    the package's configuration-error hierarchy. Exactly at the bound the genome
    must construct, because that is a legal (if tight) configuration.
    """
    # 2 inputs + 1 output = 3 nodes; the hidden-layer case adds three more.
    with pytest.raises(common.TensorNEATError):
        payloads.default_genome(num_inputs=2, num_outputs=1, max_nodes=2, max_conns=16)

    with pytest.raises(common.TensorNEATError):
        payloads.default_genome(
            num_inputs=2,
            num_outputs=1,
            max_nodes=5,
            max_conns=16,
            init_hidden_layers=(3,),
        )

    # The bound itself is legal, and the output block moves with max_nodes.
    tight = payloads.default_genome(
        num_inputs=2, num_outputs=1, max_nodes=3, max_conns=16
    )
    assert tight.max_nodes == 3
    assert tight.get_output_idx() == [2]

    hidden_tight = payloads.default_genome(
        num_inputs=2, num_outputs=1, max_nodes=6, max_conns=16, init_hidden_layers=(3,)
    )
    assert hidden_tight.get_output_idx() == [5]


def test_constructor_rejects_max_conns_smaller_than_the_initial_network(
    payloads, common
):
    """Invariant: a ``max_conns`` below the dense edge count raises ``TensorNEATError``.

    Oracle: the edge count of the configured layout — ``num_inputs * h1 + ... +
    hk * num_outputs`` — computed on the host from ``init_hidden_layers``; one
    below it must fail at construction instead of dropping the surplus edges,
    because a silently truncated initial network is a different experiment. The
    bound itself must construct.
    """
    # 2 inputs -> 1 output: two dense edges.
    with pytest.raises(common.TensorNEATError):
        payloads.default_genome(num_inputs=2, num_outputs=1, max_nodes=8, max_conns=1)

    # 2 -> 3 -> 1: nine dense edges.
    with pytest.raises(common.TensorNEATError):
        payloads.default_genome(
            num_inputs=2,
            num_outputs=1,
            max_nodes=8,
            max_conns=8,
            init_hidden_layers=(3,),
        )

    tight = payloads.default_genome(
        num_inputs=2, num_outputs=1, max_nodes=8, max_conns=2
    )
    assert tight.max_conns == 2
    tight_hidden = payloads.default_genome(
        num_inputs=2, num_outputs=1, max_nodes=8, max_conns=9, init_hidden_layers=(3,)
    )
    assert tight_hidden.max_conns == 9


def test_constructor_rejects_a_hidden_layer_wider_than_max_in_degree(
    payloads, common
):
    """Invariant: ``init_hidden_layers`` needing an in-degree above ``max_in_degree`` raises.

    Oracle: the documented static window width ``K`` is ``max(initial in-degree,
    4)`` when left at ``None``; passing a hidden layer wider than an explicit
    ``K`` must raise ``TensorNEATError`` rather than produce a schedule with a
    positive ``edge_overflow`` on generation zero. ``max_in_degree`` is a
    performance knob, but it is never silently lossy at construction time.
    """
    with pytest.raises(common.TensorNEATError):
        payloads.default_genome(
            num_inputs=2,
            num_outputs=1,
            max_nodes=16,
            max_conns=16,
            init_hidden_layers=(5,),
            max_in_degree=4,
        )

    # Left at None, the window widens to the initial in-degree (5, not the floor 4).
    widened = payloads.default_genome(
        num_inputs=2,
        num_outputs=1,
        max_nodes=16,
        max_conns=16,
        init_hidden_layers=(5,),
    )
    assert widened.max_in_degree == 5
    # An explicit width at the bound is accepted; the floor still applies without
    # a hidden layer (max(initial in-degree 2, 4) == 4).
    exact = payloads.default_genome(
        num_inputs=2,
        num_outputs=1,
        max_nodes=16,
        max_conns=16,
        init_hidden_layers=(5,),
        max_in_degree=5,
    )
    assert exact.max_in_degree == 5
    assert payloads.default_genome(
        num_inputs=2, num_outputs=1, max_nodes=8, max_conns=16
    ).max_in_degree == 4


# --------------------------------------------------------------------------- #
# transform(): the schedule of an initialized payload
# --------------------------------------------------------------------------- #


def test_transform_is_accepted_by_validate_schedule(
    payloads, common, run_transform, run_kernel, evoxc_key
):
    """Invariant: ``validate_schedule(transform(...))`` returns ``None``.

    Oracle: the package's own invariant checker — ``in_conn``/``in_src``
    ``I_INF`` in the same positions, ``level == I_INF`` iff ``topo_key ==
    I_INF``, a non-negative ``edge_overflow`` and in-range ``input_slot``/
    ``output_slot``. A derivation that returns a self-consistent *looking* but
    internally inconsistent schedule (a padded ``in_src`` slot, an absent node
    with a finite level) passes a value comparison against a similarly wrong
    oracle but fails this one.
    """
    g = payloads.default_genome(
        num_inputs=2, num_outputs=1, max_nodes=8, max_conns=16
    )
    state = payloads.genome_state(g)
    payload = payloads.Payload(*run_kernel(g.initialize, state, evoxc_key, 3))

    transformed = run_transform(g, state, payload)

    assert common.validate_schedule(transformed) is None
    # A fresh, dense payload never overflows its window.
    assert np.array_equal(np.asarray(transformed.edge_overflow), np.zeros(3, np.int32))


def test_transform_matches_the_independent_oracle(
    payloads, common, oracle, run_transform, run_kernel, evoxc_key
):
    """Invariant: every ``Transformed`` field equals ``support.oracle.schedule``.

    Oracle: the numpy-only derivation, field by field (``node_level``, ``in_conn``/
    ``in_src`` windows in connection-row order, ``src``/``dst``/``edge_valid``,
    ``input_slot``/``output_slot``, ``edge_overflow == 0``), compared with the
    dtype of each field. Both sides are brought to one sentinel convention for
    free: the payload stores ``int32`` keys with ``I_INF`` while
    ``oracle.schedule`` documents ``float32`` keys with ``NaN``, and *both*
    spellings are accepted on either side, so no re-encoding happens here.
    """
    # A hidden layer makes the schedule non-trivial: three levels, a two-slot
    # window for the hidden nodes and a three-slot window for the output.
    g = payloads.default_genome(
        num_inputs=2, num_outputs=1, max_nodes=8, max_conns=16, init_hidden_layers=(3,)
    )
    state = payloads.genome_state(g)
    payload = payloads.Payload(*run_kernel(g.initialize, state, evoxc_key, 3))

    transformed = run_transform(g, state, payload)
    reference = oracle.schedule(
        payload.node_keys,
        payload.conn_keys[:, :, 0],
        payload.conn_keys[:, :, 1],
        np.asarray(g.input_idx, dtype=np.int32),
        np.asarray(g.output_idx, dtype=np.int32),
        max_in_degree=g.max_in_degree,
        key_span=g.key_span,
    )

    assert transformed._fields == reference._fields
    for name in transformed._fields:
        actual = np.asarray(getattr(transformed, name))
        expected = np.asarray(getattr(reference, name))
        assert actual.shape == expected.shape, f"{name} shape"
        assert actual.dtype == expected.dtype, f"{name} dtype"
        assert np.array_equal(actual, expected), f"{name} values"

    assert np.array_equal(np.asarray(transformed.edge_overflow), np.zeros(3, np.int32))
    # The levels are the documented layer depths, not all zero.
    expected_levels = np.array([0, 0, 1, 1, 1, common.I_INF, common.I_INF, 2])
    assert np.array_equal(np.asarray(transformed.node_level)[0], expected_levels)


def test_transform_reports_edge_overflow_when_max_in_degree_is_too_small(
    payloads, common, run_transform
):
    """Invariant: a window narrower than a node's in-degree shows up as ``edge_overflow > 0``.

    Oracle: the documented lossy-schedule diagnostic. The *constructor* already
    refuses a ``max_in_degree`` below the initial in-degree, so the lossy case is
    built by hand: a legal genome with ``max_in_degree=2`` and a payload whose
    output node has three incoming edges. ``transform`` must report
    ``edge_overflow == 1`` per individual and *drop* the surplus window column —
    it must not raise, and it must not silently pretend the node has two incoming
    edges: the retained columns are the lowest in-group ranks, i.e. connection
    rows 0 and 1 (sources ``0`` and ``2``), and row 2 is the one counted.
    """
    g = payloads.default_genome(
        num_inputs=2, num_outputs=1, max_nodes=8, max_conns=8, max_in_degree=2
    )
    assert g.max_in_degree == 2
    state = payloads.genome_state(g)

    payload = payloads.new_payload(pop_size=2, max_nodes=8, max_conns=8)
    for p in range(payload.pop_size):
        payloads.set_node(payload, p, 0, 0)  # input key 0
        payloads.set_node(payload, p, 1, 1)  # input key 1
        payloads.set_node(payload, p, 2, 2)  # hidden key 2
        payloads.set_node(payload, p, 7, 7)  # output key 7
        payloads.set_conn(payload, p, 0, 0, 7)
        payloads.set_conn(payload, p, 1, 2, 7)
        payloads.set_conn(payload, p, 2, 1, 7)

    transformed = run_transform(g, state, payload)

    overflow = np.asarray(transformed.edge_overflow)
    assert overflow.shape == (2,)
    assert np.array_equal(overflow, np.ones(2, dtype=np.int32))
    assert np.all(overflow > 0)

    # The window keeps the first two edges into node 7 and drops the third.
    in_conn = np.asarray(transformed.in_conn)
    in_src = np.asarray(transformed.in_src)
    in_valid = np.asarray(transformed.in_valid)
    assert np.array_equal(in_conn[0, 7], np.array([0, 1], dtype=np.int32))
    assert np.array_equal(in_src[0, 7], np.array([0, 2], dtype=np.int32))
    assert np.array_equal(in_valid[0, 7], np.array([True, True]))
    # Every individual is lossy in the same way.
    assert np.array_equal(in_conn[1, 7], in_conn[0, 7])
    assert np.array_equal(in_src[1, 7], in_src[0, 7])
    # The dropped edge is counted, not aliased into a live slot.
    assert not np.any(in_conn[0, 7] == 2)
    assert not np.any(in_src[0, 7] == 1)
    # A lossy schedule is still internally consistent.
    assert common.validate_schedule(transformed) is None
