"""Unit tests for key-aligned crossover — parent 1's structure, recombined attributes.

What this module proves
-----------------------
``DefaultCrossover`` is the only operator that combines two parents, and its
contract is deliberately asymmetric: **parent 1 is the fitter parent**, so the
offspring inherits its *structure* verbatim (``node_keys``/``conn_keys`` are
returned unchanged), and only the float attribute rows may come from parent 2 —
and only for genes that have a homologue there, aligned on the whole key-column
tuple. The tests pin down:

* structure: the returned key arrays are parent 1's, so the offspring's gene set
  is a subset of parent 1's (equal, in fact) and can never gain a gene;
* alignment: a gene present in both parents recombines its attributes, a gene
  present only in parent 1 keeps parent 1's attributes unchanged, and a gene
  present only in parent 2 contributes nothing;
* the recombination rule itself: ``BaseGene.crossover`` is a **per-attribute
  50/50 pick**, so each child attribute equals one of the two parents' values —
  never their mean and never a value from another row;
* padding: absent rows stay ``NaN``/``I_INF``, so an offspring has exactly the
  same row layout as parent 1;
* the degenerate case: crossing a parent with itself is the identity, which any
  blend-based implementation would also satisfy but a mis-aligned gather would not.

Oracles used here (never the kernel itself)
-------------------------------------------
* set/permutation algebra on the payloads — which keys exist, in which row, and
  whether a child attribute came from parent 1 or parent 2 (a membership test, not
  a value comparison, because the pick is random);
* hand-built parents from ``support.payloads`` that differ in *one* gene, so "did
  this attribute move?" has a single possible cause;
* ``find_homologous`` is checked against a plain Python ``{key_tuple: row}`` map
  built from parent 2 on the host, which is the definition of "homologous" rather
  than the operator's own alignment machinery.

Gotchas this module encodes
---------------------------
* Row order is *not* the alignment key: two parents hold their genes in different
  rows and still align on ``(in_key, out_key)`` (plus ``historical_marker`` for a
  conn gene that declares it). A test that compared row ``i`` with row ``i`` would
  pass a broken implementation whenever both parents happen to be sorted, so the
  ``find_homologous`` fixtures shuffle parent 2's rows and a three-column key
  table shows that a differing ``historical_marker`` makes a gene non-homologous.
* ``find_homologous`` returns ``I_INF`` for a parent-1 gene with no homologue and
  for parent 1's own padded rows; ``crossover_gene_attrs`` must then keep parent
  1's attributes instead of gathering row ``I_INF`` from parent 2 (which would be
  an out-of-range read, not merely a wrong value).
* The 50/50 pick is *per attribute*, so a node row may take its ``bias`` from one
  parent and its ``activation`` from the other. Asserting "the whole row came from
  one parent" is the ``OriginNode``/``OriginConn`` behaviour and would fail here.
* Crossing is defined for equal shapes only: ``max_nodes``/``max_conns`` are
  build-time constants, so a shape mismatch raises ``TensorNEATError`` rather than
  splicing two differently sized payloads.

Backend is ``"numpy"`` throughout, through the shared runner only; no
``jax``/``torch``, no network, no GPU.
"""

from __future__ import annotations

import numpy as np
import pytest


#: Node capacity ``N`` of every fixture here (the payload's node-row axis).
MAX_NODES = 8
#: Connection capacity ``E`` of every fixture here (the payload's connection axis).
MAX_CONNS = 6


# --------------------------------------------------------------------------- #
# hand-built fixtures, so every expectation below is a value we can print
# --------------------------------------------------------------------------- #


def _genome(payloads):
    """The genome whose operator is under test, with the default operator stack.

    ``support.payloads.default_genome`` wires ``crossover=DefaultCrossover()``
    explicitly, which is also what ``tensorneat/genome/default.py`` builds when the
    keyword is omitted — so what is traced below is the production operator, not a
    stand-in. Shapes match the fixtures (``MAX_NODES``/``MAX_CONNS``, two inputs,
    one output row ``7``).
    """
    return payloads.default_genome(
        max_nodes=MAX_NODES,
        max_conns=MAX_CONNS,
        num_inputs=2,
        num_outputs=1,
        max_in_degree=4,
    )


def _payload(payloads, *, pop_size=2, nodes=(), conns=(), max_conns=MAX_CONNS):
    """Build one parent payload from ``(row, key, attrs)`` node and ``(row, in, out, attrs)`` edges.

    The same genes are written for every individual, so a test can state its oracle
    per row instead of per individual; ``attrs=None`` writes the neutral row of
    ``support.payloads``. Rows that are not written keep the padding sentinels
    (``I_INF`` keys, ``NaN`` attributes), which is what makes "an absent row stays
    absent" assertable.
    """
    payload = payloads.new_payload(pop_size, MAX_NODES, max_conns)
    for p in range(pop_size):
        for row, key, attrs in nodes:
            payloads.set_node(payload, p, row, key, attrs)
        for row, in_key, out_key, attrs in conns:
            payloads.set_conn(payload, p, row, in_key, out_key, attrs)
    return payload


def _cross(run_kernel, genome, state, key, parent1, parent2):
    """Cross two parents through ``execute_crossover`` and return the offspring payload."""
    return run_kernel(
        genome.execute_crossover, state, key, *parent1.args(), *parent2.args()
    )


def _key_map(payloads, keys):
    """``{key tuple: row}`` of one payload's present genes — the host definition of homology.

    ``keys`` is one individual's node-key column ``(N,)`` or connection-key table
    ``(E, CK)``; rows whose first column is ``I_INF`` are absent and left out.
    """
    table = np.asarray(keys)
    rows = table.reshape(table.shape[0], -1)
    return {
        tuple(int(column) for column in row): row_index
        for row_index, row in enumerate(rows)
        if int(row[0]) != payloads.I_INF
    }


def _key_table(rows):
    """``(P, M, K)`` ``int32`` key table from a list of key rows, repeated for the batch."""
    return np.array([rows, rows], dtype=np.int32)


# --------------------------------------------------------------------------- #
# structure: parent 1 wins
# --------------------------------------------------------------------------- #


def test_parent_one_structure_is_kept(payloads, run_kernel, evoxc_key):
    """Invariant: the two key arrays returned are parent 1's, bit for bit.

    Oracle: ``np.array_equal`` against parent 1's ``node_keys``/``conn_keys``.
    The offspring cannot gain, lose or reorder a gene, whatever parent 2 holds —
    deliberately constructed so that parent 2 has *more* genes than parent 1. A
    union of the two parents (the usual "genetic algorithm" instinct) fails here.
    """
    genome = _genome(payloads)
    state = payloads.genome_state(genome)
    parent1 = _payload(
        payloads,
        nodes=[(0, 0, None), (1, 1, None), (7, 7, None)],
        conns=[(0, 0, 7, None), (1, 1, 7, None)],
    )
    parent2 = _payload(
        payloads,
        nodes=[(0, 0, None), (1, 1, None), (3, 3, None), (5, 5, None), (7, 7, None)],
        conns=[
            (0, 0, 7, None),
            (1, 1, 7, None),
            (2, 3, 5, None),
            (3, 5, 7, None),
        ],
    )

    # Fixture sanity: parent 2 really is the richer parent, so a union would differ.
    nodes1 = set(_key_map(payloads, parent1.node_keys[0]))
    nodes2 = set(_key_map(payloads, parent2.node_keys[0]))
    conns1 = set(_key_map(payloads, parent1.conn_keys[0]))
    conns2 = set(_key_map(payloads, parent2.conn_keys[0]))
    assert nodes2 > nodes1
    assert conns2 > conns1

    offspring = _cross(run_kernel, genome, state, evoxc_key, parent1, parent2)

    np.testing.assert_array_equal(offspring.node_keys, parent1.node_keys)
    np.testing.assert_array_equal(offspring.conn_keys, parent1.conn_keys)


def test_offspring_keys_are_a_subset_of_parent_one_keys(
    payloads, run_kernel, evoxc_key
):
    """Invariant: every gene of the offspring exists in parent 1.

    Oracle: set membership of the ``(in_key, out_key)`` pairs and of the node keys,
    per individual. Stated separately from equality because it is the property the
    algorithm relies on (an offspring never carries an innovation number its fitter
    parent did not have), and it is the assertion that survives a future
    multi-parent operator whose key array is a *selection* of parent 1's rows.
    """
    genome = _genome(payloads)
    state = payloads.genome_state(genome)
    parent1 = _payload(
        payloads,
        nodes=[(0, 0, None), (1, 1, None), (2, 2, None), (7, 7, None)],
        conns=[(0, 0, 7, None), (1, 1, 7, None), (2, 2, 7, None)],
    )
    parent2 = _payload(
        payloads,
        nodes=[
            (0, 0, None),
            (1, 1, None),
            (2, 2, None),
            (4, 4, None),
            (5, 5, None),
            (7, 7, None),
        ],
        conns=[
            (0, 0, 7, None),
            (1, 1, 7, None),
            (2, 2, 7, None),
            (3, 4, 5, None),
            (4, 5, 7, None),
        ],
    )
    offspring = _cross(run_kernel, genome, state, evoxc_key, parent1, parent2)

    parent2_only_nodes = set(_key_map(payloads, parent2.node_keys[0])) - set(
        _key_map(payloads, parent1.node_keys[0])
    )
    parent2_only_conns = set(_key_map(payloads, parent2.conn_keys[0])) - set(
        _key_map(payloads, parent1.conn_keys[0])
    )
    assert parent2_only_nodes and parent2_only_conns  # fixture sanity

    for p in range(parent1.pop_size):
        nodes1 = set(_key_map(payloads, parent1.node_keys[p]))
        conns1 = set(_key_map(payloads, parent1.conn_keys[p]))
        child_nodes = set(_key_map(payloads, offspring.node_keys[p]))
        child_conns = set(_key_map(payloads, offspring.conn_keys[p]))

        assert child_nodes <= nodes1
        assert child_conns <= conns1
        # The genes only parent 2 has contribute nothing at all.
        assert not (child_nodes & parent2_only_nodes)
        assert not (child_conns & parent2_only_conns)


# --------------------------------------------------------------------------- #
# attributes: parent 2 wins only where keys align, and only per attribute
# --------------------------------------------------------------------------- #


def test_attributes_come_from_the_fitter_parent_where_keys_align(
    payloads, run_kernel, evoxc_key
):
    """Invariant: a homologous gene's child attribute equals one parent's value, per attribute.

    Oracle: membership in the two-element set ``{attrs1, attrs2}`` of that gene —
    ``BaseGene.crossover`` is documented as a per-attribute 50/50 pick, so a child
    value that is neither parent's (a mean, a sum, a value from a neighbouring row)
    is a failure. Both parents carry the *same* topology with different attribute
    rows, so every gene is homologous and the alignment is not what is being
    tested here.

    The two attribute rows are chosen so that their arithmetic mean is a third
    value, and the pick is also required to actually happen: both parents' values
    must appear somewhere in the offspring, which an "always parent 1" operator
    would not do.
    """
    genome = _genome(payloads)
    state = payloads.genome_state(genome)
    nodes = [(0, 0, None), (1, 1, None), (2, 2, None), (7, 7, None)]
    conns = [(0, 0, 7, None), (1, 1, 7, None), (2, 2, 7, None)]
    rows1, rows2 = (1.0, 2.0, 0.0, 0.0), (4.0, 8.0, 1.0, 1.0)
    parent1 = _payload(
        payloads,
        pop_size=4,
        nodes=[(row, key, rows1) for row, key, _ in nodes],
        conns=[(row, a, b, (1.0,)) for row, a, b, _ in conns],
    )
    parent2 = _payload(
        payloads,
        pop_size=4,
        nodes=[(row, key, rows2) for row, key, _ in nodes],
        conns=[(row, a, b, (9.0,)) for row, a, b, _ in conns],
    )
    offspring = _cross(run_kernel, genome, state, evoxc_key, parent1, parent2)

    # No mean is a member of its parent pair, so "picked one of the two" is the
    # only way this elementwise test can pass.
    for first, second in zip(rows1, rows2):
        assert (first + second) / 2.0 not in (first, second)

    live_nodes = parent1.node_keys != payloads.I_INF
    child_nodes = offspring.node_attrs[live_nodes]
    nodes1 = parent1.node_attrs[live_nodes]
    nodes2 = parent2.node_attrs[live_nodes]
    assert child_nodes.shape == nodes1.shape
    # Every attribute value in play is one of the two parents', so no third value
    # (a mean, a neighbouring row) can hide in the comparison below.
    assert np.array_equal(
        np.unique(np.concatenate([nodes1, nodes2])),
        np.array([0.0, 1.0, 2.0, 4.0, 8.0], dtype=np.float32),
    )
    assert ((child_nodes == nodes1) | (child_nodes == nodes2)).all()

    live_conns = (parent1.conn_keys != payloads.I_INF).all(axis=-1)
    child_conns = offspring.conn_attrs[live_conns]
    conns1 = parent1.conn_attrs[live_conns]
    conns2 = parent2.conn_attrs[live_conns]
    assert ((child_conns == conns1) | (child_conns == conns2)).all()

    # The draw is real: parent 2 wins somewhere, so an "always parent 1" operator
    # (and one whose key never reaches the gene hook) fails here.
    assert (child_nodes == nodes2).any()
    assert (child_conns == conns2).any()


def test_genes_without_a_homologue_keep_parent_one_attributes(
    payloads, run_kernel, evoxc_key
):
    """Invariant: a gene absent from parent 2 keeps parent 1's attributes unchanged.

    Oracle: value equality against parent 1 on exactly the rows whose key has no
    homologue (identified on the host from the two payloads). Parent 2 is built to
    lack one node and one connection that parent 1 has, and its attributes
    elsewhere are set to a value that could only come from a wrong gather — so a
    "gather row ``I_INF``" bug shows up as a value that is neither parent's, not as
    a silent pass.
    """
    genome = _genome(payloads)
    state = payloads.genome_state(genome)
    parent1 = _payload(
        payloads,
        nodes=[
            (0, 0, (1.0, 2.0, 0.0, 0.0)),
            (1, 1, (1.5, 2.5, 0.0, 0.0)),
            (2, 2, (3.0, 4.0, 0.0, 0.0)),
            (7, 7, (5.0, 6.0, 0.0, 0.0)),
        ],
        conns=[
            (0, 0, 7, (1.0,)),
            (1, 1, 7, (2.0,)),
            (2, 2, 7, (3.0,)),
        ],
    )
    parent2 = _payload(
        payloads,
        nodes=[
            (0, 0, (99.0, 99.0, 1.0, 1.0)),
            (1, 1, (7.0, 8.0, 1.0, 1.0)),
            (7, 7, (9.0, 10.0, 1.0, 1.0)),
        ],
        conns=[(0, 0, 7, (7.0,)), (2, 2, 7, (9.0,))],
    )
    offspring = _cross(run_kernel, genome, state, evoxc_key, parent1, parent2)

    for p in range(parent1.pop_size):
        parent2_nodes = _key_map(payloads, parent2.node_keys[p])
        parent2_conns = _key_map(payloads, parent2.conn_keys[p])
        lonely_nodes = [
            row
            for row, key in enumerate(parent1.node_keys[p])
            if int(key) != payloads.I_INF
            and (int(key),) not in parent2_nodes
        ]
        lonely_conns = [
            row
            for row, key in enumerate(parent1.conn_keys[p])
            if int(key[0]) != payloads.I_INF
            and tuple(int(column) for column in key) not in parent2_conns
        ]
        assert len(lonely_nodes) == 1, lonely_nodes
        assert len(lonely_conns) == 1, lonely_conns
        node_row, conn_row = lonely_nodes[0], lonely_conns[0]

        # The wrong-gather value is parent 2's row 0 (the row an ``I_INF`` index is
        # redirected to), which no correct row of the offspring contains.
        assert not np.array_equal(
            parent1.node_attrs[p, node_row], parent2.node_attrs[p, 0]
        )
        assert not np.array_equal(
            parent1.conn_attrs[p, conn_row], parent2.conn_attrs[p, 0]
        )

        np.testing.assert_array_equal(
            offspring.node_attrs[p, node_row], parent1.node_attrs[p, node_row]
        )
        np.testing.assert_array_equal(
            offspring.conn_attrs[p, conn_row], parent1.conn_attrs[p, conn_row]
        )

        # ... while the genes that do have a homologue still recombine normally.
        for row, key in enumerate(parent1.node_keys[p]):
            if int(key) == payloads.I_INF or (int(key),) not in parent2_nodes:
                continue
            aligned = parent2.node_attrs[p, parent2_nodes[(int(key),)]]
            child = offspring.node_attrs[p, row]
            assert ((child == parent1.node_attrs[p, row]) | (child == aligned)).all()


# --------------------------------------------------------------------------- #
# padding and shapes
# --------------------------------------------------------------------------- #


def test_nan_padding_is_preserved(payloads, run_kernel, evoxc_key):
    """Invariant: an offspring's absent rows stay ``I_INF``/``NaN``.

    Oracle: the padding mask of parent 1, row for row — a child row is padded iff
    parent 1's row is padded, and a padded row's attribute columns are all ``NaN``.
    Crossover must not fill a padded row from parent 2's gene that happens to sit
    at the same row index, which is exactly what a row-wise (rather than
    key-aligned) implementation would do.
    """
    genome = _genome(payloads)
    state = payloads.genome_state(genome)
    parent1 = _payload(
        payloads,
        nodes=[(0, 0, None), (1, 1, None), (7, 7, None)],
        conns=[(0, 0, 7, None), (1, 1, 7, None)],
    )
    parent2 = _payload(
        payloads,
        nodes=[
            (0, 0, None),
            (1, 1, None),
            (2, 2, None),
            (3, 3, None),
            (7, 7, None),
        ],
        conns=[
            (0, 0, 7, None),
            (1, 1, 7, None),
            (2, 2, 7, None),
            (3, 3, 7, None),
            (4, 0, 2, None),
            (5, 3, 7, None),
        ],
    )
    offspring = _cross(run_kernel, genome, state, evoxc_key, parent1, parent2)

    absent_nodes = parent1.node_keys == payloads.I_INF
    absent_conns = (parent1.conn_keys == payloads.I_INF).all(axis=-1)
    # Fixture sanity: parent 2 carries genes exactly where parent 1 is padded, so a
    # row-wise implementation would fill those rows.
    assert absent_nodes.any() and absent_conns.any()
    assert (parent2.node_keys[absent_nodes] != payloads.I_INF).any()
    assert (parent2.conn_keys[absent_conns][..., 0] != payloads.I_INF).any()

    np.testing.assert_array_equal(offspring.node_keys == payloads.I_INF, absent_nodes)
    np.testing.assert_array_equal(
        (offspring.conn_keys == payloads.I_INF).all(axis=-1), absent_conns
    )
    assert np.isnan(offspring.node_attrs[absent_nodes]).all()
    assert np.isnan(offspring.conn_attrs[absent_conns]).all()
    # Rows that are present stay finite: nothing padding-related leaks the other way.
    assert not np.isnan(offspring.node_attrs[~absent_nodes]).any()
    assert not np.isnan(offspring.conn_attrs[~absent_conns]).any()


def test_shapes_and_dtypes_are_preserved(payloads, run_kernel, evoxc_key, common):
    """Invariant: crossover neither resizes the payload nor changes a dtype.

    Oracle: shape/dtype equality with parent 1 for all four arrays, and a
    differently shaped parent 2 is rejected with ``TensorNEATError`` rather than
    spliced. Shapes are static at build time (the pipeline caches an executable per
    shape signature), so an operator that returned parent 2's shape — or a float
    key column — would break the next generation rather than this one, which is why
    it is asserted explicitly.
    """
    genome = _genome(payloads)
    state = payloads.genome_state(genome)
    parent1 = _payload(
        payloads,
        nodes=[(0, 0, None), (1, 1, None), (7, 7, None)],
        conns=[(0, 0, 7, None), (1, 1, 7, None)],
    )
    parent2 = _payload(
        payloads,
        nodes=[(0, 0, None), (1, 1, None), (2, 2, None), (7, 7, None)],
        conns=[(0, 0, 7, None), (1, 1, 7, None), (2, 2, 7, None)],
    )
    offspring = _cross(run_kernel, genome, state, evoxc_key, parent1, parent2)

    for name in ("node_keys", "node_attrs", "conn_keys", "conn_attrs"):
        first = getattr(parent1, name)
        child = getattr(offspring, name)
        assert child.shape == first.shape, (name, child.shape, first.shape)
        assert child.dtype == first.dtype, (name, child.dtype, first.dtype)
    assert offspring.node_keys.dtype == np.int32
    assert offspring.conn_keys.dtype == np.int32
    assert offspring.node_attrs.dtype == np.float32
    assert offspring.conn_attrs.dtype == np.float32

    taller = payloads.new_payload(parent1.pop_size, MAX_NODES + 1, MAX_CONNS)
    with pytest.raises(common.TensorNEATError):
        _cross(run_kernel, genome, state, evoxc_key, parent1, taller)


def test_self_crossover_is_the_identity(payloads, run_kernel, evoxc_key):
    """Invariant: crossing a parent with itself returns that parent unchanged.

    Oracle: full payload equality (all four arrays) when both arguments are the
    same payload. Since the two candidate values are equal, any correct 50/50 pick
    is the identity — so a failure here means the operator is not picking between
    the aligned rows at all (a scrambled gather, a wrong homologue table, a padded
    row overwritten by an unaligned one).
    """
    genome = _genome(payloads)
    state = payloads.genome_state(genome)
    parent = _payload(
        payloads,
        nodes=[
            (0, 0, (1.0, 2.0, 0.0, 0.0)),
            (1, 1, (1.5, 2.5, 0.0, 0.0)),
            (2, 2, (3.0, 4.0, 0.0, 0.0)),
            (7, 7, (5.0, 6.0, 0.0, 0.0)),
        ],
        conns=[(0, 0, 7, (1.0,)), (1, 1, 7, (2.0,)), (2, 2, 7, (3.0,))],
    )
    offspring = _cross(run_kernel, genome, state, evoxc_key, parent, parent)

    # The fixture carries padded rows in both families, so this also pins that the
    # identity holds for the ``NaN``/``I_INF`` rows, not only for live genes.
    assert (parent.node_keys == payloads.I_INF).any()
    assert (parent.conn_keys == payloads.I_INF).any()

    for name in ("node_keys", "node_attrs", "conn_keys", "conn_attrs"):
        np.testing.assert_array_equal(
            getattr(offspring, name), getattr(parent, name), err_msg=name
        )


# --------------------------------------------------------------------------- #
# the alignment primitive
# --------------------------------------------------------------------------- #


def test_find_homologous_is_exact_on_the_key_tuple(payloads, runner):
    """Invariant: ``find_homologous`` maps every parent-1 gene to its parent-2 row by key.

    Oracle: a plain Python ``{key_tuple: row}`` map built from parent 2. Parent 2's
    rows are **shuffled**, so a row-index-based alignment gives a different (and
    detectably wrong) answer; a key present in both parents must map to the row
    that actually holds it, and the hash-collision window must not change that.
    The operator method takes no ``state``, so it is traced through
    ``support.runner.run_defn`` directly rather than through ``run_kernel``.

    The connection half uses a three-column key table, the ``OriginConn`` layout:
    the same ``(in_key, out_key)`` pair under a different ``historical_marker`` is
    a *different* gene, which is what makes the match exact on the whole tuple.
    """
    genome = _genome(payloads)

    def find(keys1, keys2):
        return genome.crossover.find_homologous(keys1, keys2)

    node_rows1 = [0, 1, 2, 3, 7, payloads.I_INF, payloads.I_INF, payloads.I_INF]
    node_rows2 = [7, 3, payloads.I_INF, 0, payloads.I_INF, 2, 1, payloads.I_INF]
    node_table = runner.run_defn(
        find,
        _key_table([[key] for key in node_rows1]),
        _key_table([[key] for key in node_rows2]),
    )
    expected = [
        _key_map(payloads, node_rows2).get((key,), payloads.I_INF)
        for key in node_rows1
    ]
    np.testing.assert_array_equal(node_table, np.array([expected, expected]))
    # A row-index alignment would have answered "row i", so the fixture can tell them apart.
    assert expected != list(range(len(node_rows1)))

    conn_rows1 = [[0, 7], [1, 7], [2, 7], [3, 7]]
    conn_rows2 = [[2, 7], [3, 7], [0, 7], [1, 7]]
    conn_table = runner.run_defn(
        find,
        _key_table(conn_rows1),
        _key_table(conn_rows2),
    )
    expected = [
        _key_map(payloads, conn_rows2)[tuple(row)] for row in conn_rows1
    ]
    np.testing.assert_array_equal(conn_table, np.array([expected, expected]))
    assert expected != list(range(len(conn_rows1)))

    # Three key columns: the historical marker is part of the gene's identity.
    marked1 = [[0, 7, 5], [1, 7, 6], [payloads.I_INF] * 3]
    marked2 = [[1, 7, 7], [0, 7, 5], [payloads.I_INF] * 3]
    marked_table = runner.run_defn(
        find,
        _key_table(marked1),
        _key_table(marked2),
    )
    np.testing.assert_array_equal(
        marked_table, np.array([[1, payloads.I_INF, payloads.I_INF]] * 2)
    )


def test_find_homologous_returns_i_inf_for_absent_genes(payloads, runner):
    """Invariant: ``I_INF`` marks a parent-1 gene with no homologue and parent 1's padding.

    Oracle: the key map of parent 2 — every key it does not contain maps to
    ``I_INF``, and every padded row of parent 1 maps there too (a padded row must
    never align with parent 2's padded row, or ``crossover_gene_attrs`` would
    "recombine" two ``NaN`` rows into a gene that is supposed to be absent).
    """
    genome = _genome(payloads)

    def find(keys1, keys2):
        return genome.crossover.find_homologous(keys1, keys2)

    # Rows 3..6 are padding in BOTH parents, row 2 is a parent-1 gene parent 2 lacks.
    node_rows1 = [0, 1, 2, payloads.I_INF, payloads.I_INF, payloads.I_INF,
                  payloads.I_INF, 7]
    node_rows2 = [0, 7, payloads.I_INF, payloads.I_INF, payloads.I_INF,
                  payloads.I_INF, payloads.I_INF, payloads.I_INF]
    node_table = runner.run_defn(
        find,
        _key_table([[key] for key in node_rows1]),
        _key_table([[key] for key in node_rows2]),
    )
    absent = payloads.I_INF
    expected = [0, absent, absent, absent, absent, absent, absent, 1]
    np.testing.assert_array_equal(node_table, np.array([expected, expected]))

    conn_rows1 = [[0, 7], [1, 7], [2, 7], [payloads.I_INF] * 2]
    conn_rows2 = [[0, 7], [payloads.I_INF] * 2, [payloads.I_INF] * 2,
                  [payloads.I_INF] * 2]
    conn_table = runner.run_defn(
        find,
        _key_table(conn_rows1),
        _key_table(conn_rows2),
    )
    expected = [0, absent, absent, absent]
    np.testing.assert_array_equal(conn_table, np.array([expected, expected]))

    # And the padded rows never align with each other, whatever their index.
    padded1 = [[payloads.I_INF] * 2] * 3
    padded2 = [[payloads.I_INF] * 2] * 3
    padded_table = runner.run_defn(find, _key_table(padded1), _key_table(padded2))
    assert (padded_table == absent).all()
