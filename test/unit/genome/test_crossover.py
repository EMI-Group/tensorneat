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
  pass a broken implementation whenever both parents happen to be sorted.
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

import pytest


# --------------------------------------------------------------------------- #
# structure: parent 1 wins
# --------------------------------------------------------------------------- #


def test_parent_one_structure_is_kept(payloads, runner, run_kernel, evoxc_key):
    """Invariant: the two key arrays returned are parent 1's, bit for bit.

    Oracle: ``np.array_equal`` against parent 1's ``node_keys``/``conn_keys``.
    The offspring cannot gain, lose or reorder a gene, whatever parent 2 holds —
    deliberately constructed so that parent 2 has *more* genes than parent 1. A
    union of the two parents (the usual "genetic algorithm" instinct) fails here.
    """
    pytest.skip(
        "not implemented: offspring key arrays equal parent 1's key arrays "
        "(owner: genome/operations/crossover/default.DefaultCrossover.__call__)"
    )


def test_offspring_keys_are_a_subset_of_parent_one_keys(payloads, runner, run_kernel, evoxc_key):
    """Invariant: every gene of the offspring exists in parent 1.

    Oracle: set membership of the ``(in_key, out_key)`` pairs and of the node keys,
    per individual. Stated separately from equality because it is the property the
    algorithm relies on (an offspring never carries an innovation number its fitter
    parent did not have), and it is the assertion that survives a future
    multi-parent operator whose key array is a *selection* of parent 1's rows.
    """
    pytest.skip(
        "not implemented: offspring gene set is a subset of parent 1's gene set "
        "(owner: genome/operations/crossover/default.DefaultCrossover.__call__)"
    )


# --------------------------------------------------------------------------- #
# attributes: parent 2 wins only where keys align, and only per attribute
# --------------------------------------------------------------------------- #


def test_attributes_come_from_the_fitter_parent_where_keys_align(
    payloads, runner, run_kernel, evoxc_key
):
    """Invariant: a homologous gene's child attribute equals one parent's value, per attribute.

    Oracle: membership in the two-element set ``{attrs1, attrs2}`` of that gene —
    ``BaseGene.crossover`` is documented as a per-attribute 50/50 pick, so a child
    value that is neither parent's (a mean, a sum, a value from a neighbouring row)
    is a failure. Both parents carry the *same* topology with different attribute
    rows, so every gene is homologous and the alignment is not what is being
    tested here.
    """
    pytest.skip(
        "not implemented: homologous attributes are picked 50/50 per attribute "
        "(owner: genome/operations/crossover/default.DefaultCrossover.crossover_gene_attrs, "
        "gene/base.BaseGene.crossover)"
    )


def test_genes_without_a_homologue_keep_parent_one_attributes(
    payloads, runner, run_kernel, evoxc_key
):
    """Invariant: a gene absent from parent 2 keeps parent 1's attributes unchanged.

    Oracle: value equality against parent 1 on exactly the rows whose key has no
    homologue (identified on the host from the two payloads). Parent 2 is built to
    lack one node and one connection that parent 1 has, and its attributes
    elsewhere are set to a value that could only come from a wrong gather — so a
    "gather row ``I_INF``" bug shows up as a value that is neither parent's, not as
    a silent pass.
    """
    pytest.skip(
        "not implemented: genes without a homologue keep parent 1's attributes "
        "(owner: genome/operations/crossover/default.DefaultCrossover.crossover_gene_attrs)"
    )


# --------------------------------------------------------------------------- #
# padding and shapes
# --------------------------------------------------------------------------- #


def test_nan_padding_is_preserved(payloads, runner, run_kernel, evoxc_key):
    """Invariant: an offspring's absent rows stay ``I_INF``/``NaN``.

    Oracle: the padding mask of parent 1, row for row — a child row is padded iff
    parent 1's row is padded, and a padded row's attribute columns are all ``NaN``.
    Crossover must not fill a padded row from parent 2's gene that happens to sit
    at the same row index, which is exactly what a row-wise (rather than
    key-aligned) implementation would do.
    """
    pytest.skip(
        "not implemented: padding rows stay padded in the offspring "
        "(owner: genome/operations/crossover/default.DefaultCrossover.crossover_gene_attrs)"
    )


def test_shapes_and_dtypes_are_preserved(payloads, runner, run_kernel, evoxc_key):
    """Invariant: crossover neither resizes the payload nor changes a dtype.

    Oracle: shape/dtype equality with parent 1 for all four arrays. Shapes are
    static at build time (the pipeline caches an executable per shape signature),
    so an operator that returned parent 2's shape — or a float key column — would
    break the next generation rather than this one, which is why it is asserted
    explicitly.
    """
    pytest.skip(
        "not implemented: crossover preserves payload shapes and dtypes "
        "(owner: genome/base.BaseGenome.execute_crossover, genome/utils.GenomeTensors)"
    )


def test_self_crossover_is_the_identity(payloads, runner, run_kernel, evoxc_key):
    """Invariant: crossing a parent with itself returns that parent unchanged.

    Oracle: full payload equality (all four arrays) when both arguments are the
    same payload. Since the two candidate values are equal, any correct 50/50 pick
    is the identity — so a failure here means the operator is not picking between
    the aligned rows at all (a scrambled gather, a wrong homologue table, a padded
    row overwritten by an unaligned one).
    """
    pytest.skip(
        "not implemented: self-crossover returns the parent unchanged "
        "(owner: genome/operations/crossover/default.DefaultCrossover.__call__)"
    )


# --------------------------------------------------------------------------- #
# the alignment primitive
# --------------------------------------------------------------------------- #


def test_find_homologous_is_exact_on_the_key_tuple(payloads, runner, evoxc_key):
    """Invariant: ``find_homologous`` maps every parent-1 gene to its parent-2 row by key.

    Oracle: a plain Python ``{key_tuple: row}`` map built from parent 2. Parent 2's
    rows are **shuffled**, so a row-index-based alignment gives a different (and
    detectably wrong) answer; a key present in both parents must map to the row
    that actually holds it, and the hash-collision window must not change that.
    The operator method takes no ``state``, so it is traced through
    ``support.runner.run_defn`` directly rather than through ``run_kernel``.
    """
    pytest.skip(
        "not implemented: find_homologous aligns on the exact key-column tuple "
        "(owner: genome/operations/crossover/default.DefaultCrossover.find_homologous)"
    )


def test_find_homologous_returns_i_inf_for_absent_genes(payloads, runner, evoxc_key):
    """Invariant: ``I_INF`` marks a parent-1 gene with no homologue and parent 1's padding.

    Oracle: the key map of parent 2 — every key it does not contain maps to
    ``I_INF``, and every padded row of parent 1 maps there too (a padded row must
    never align with parent 2's padded row, or ``crossover_gene_attrs`` would
    "recombine" two ``NaN`` rows into a gene that is supposed to be absent).
    """
    pytest.skip(
        "not implemented: unmatched and padded genes map to I_INF "
        "(owner: genome/operations/crossover/default.DefaultCrossover.find_homologous)"
    )
