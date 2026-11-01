"""Parity of ``execute_distance``: the compatibility distance of two genomes.

The quantity, identical on both sides (``genome/operations/distance/default.py``)
------------------------------------------------------------------------------
``distance = (non_homologous_count * compatibility_disjoint
             + homologous_attribute_distance * compatibility_weight)
             / max(cnt1, cnt2)``

reduced over the two gene families (nodes, connections), with the default weights
``compatibility_disjoint=1.0`` and ``compatibility_weight=0.4`` on both sides.
Homologous genes are those with equal key columns; the attribute term is the
gene's own ``distance`` summed over homologous pairs; an empty pair yields ``0``
rather than ``NaN``.

Hand-computed oracles (verified against the real legacy code, not just the new one)
----------------------------------------------------------------------------------
With ``chain_payload`` (7 edges, 8 nodes) and its one-extra-connection variant
(8 edges, 8 nodes), all weights ``1``:

* identical genomes: ``0`` exactly, both sides;
* one extra edge: node term ``0`` (8 homologous, 0 detached) and connection term
  ``(1 * 1.0 + 0 * 0.4) / 8 = 0.125`` — so ``0.125`` is the exact answer, not a
  tolerance question;
* all weights doubled (``weight=2``): the connection term is
  ``(0 * 1.0 + 7 * 1.0 * 0.4) / 7 = 0.4``;
* all weights doubled *and* one node's ``bias`` raised by ``1``: the same ``0.4``
  connection term plus a node term ``(0 * 1.0 + 1 * 1.0 * 0.4) / 8 = 0.05``, so
  ``0.45``;
* ``skip_payload`` (5 nodes, 4 edges) against the chain (8 nodes, 7 edges): the
  node term is ``3 / 8`` (three chain-only nodes) and the connection term is
  ``7 / 7`` (two homologous edges, seven detached), so ``1.375``.

These numbers pin the normalisation (``max`` of the two counts) and the weight of
each term independently, which is why they are asserted exactly rather than
through a loose tolerance.

Scope rules (``../CONTEXT.md``)
------------------------------
Only *kernels on identical payloads* are compared: neither library's RNG, neither
one's ``initialize``/mutation/crossover, and never ``genome.hash()``.  The genome
pair is the same one the forward tests build — one ACT/AGG option each and
``max_in_degree == max_nodes`` — because this kernel reads the raw attribute
columns: an option list of a different length would make attribute id ``0`` mean
different activations on the two sides, and a different ``max_in_degree`` would
make the pair "the same genome" only by accident.  ``execute_distance`` itself
derives no schedule, so ``edge_overflow`` is not a precondition *here*; the
payloads are the ones ``test_forward`` proves lossless.
"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = [pytest.mark.parity]

pytest.importorskip("jax", reason="parity needs the `bench` extra (jax)")


def _stack(payloads, individuals):
    """Concatenate ``P == 1`` payloads along the population axis.

    Args:
        payloads: The ``support.payloads`` module, for the ``Payload`` constructor.
        individuals: Payloads of identical static shape, one per parent pair.

    Returns:
        A ``Payload`` with ``P == len(individuals)`` whose row ``p`` is
        ``individuals[p]`` — the layout both distance kernels pair individual-wise.
    """
    return payloads.Payload(
        *(
            np.concatenate([getattr(item, field) for item in individuals], axis=0)
            for field in payloads.Payload.field_names
        )
    )


def test_identical_genomes_yield_exactly_zero(payloads, genome_pair, new_distance, legacy_distance):
    """Invariant: a genome's distance to itself is ``0`` on both sides, and it is deterministic.

    Oracle: the definition (``non_homologous_count == 0`` and every homologous
    attribute distance ``0``) plus the legacy operator on the same payload.
    ``payloads.chain_payload()`` compared with a byte-identical copy.

    Assertion: ``new == 0.0`` and ``legacy == 0.0`` **exactly** (``==``, not
    ``allclose``) — ``0`` is the value the speciation code depends on — and a
    second call returns the identical value, i.e. ``execute_distance`` draws no
    random numbers.
    """
    pair = genome_pair()
    left = payloads.chain_payload()
    right = payloads.chain_payload()

    new = new_distance(pair, left, right)
    legacy = legacy_distance(pair, left.of(0), right.of(0))

    assert new.shape == (1,)
    assert new[0] == 0.0
    assert legacy == 0.0

    again = new_distance(pair, left, right)
    assert again[0] == new[0]


def test_one_extra_connection_matches_legacy(payloads, genome_pair, new_distance, legacy_distance):
    """Invariant: the disjoint-count term is ``non_homologous / max(cnt1, cnt2)``.

    Oracle: hand computation *and* the legacy operator.  ``payloads.chain_payload()``
    (7 edges) versus the same payload plus the edge ``0->3`` (8 edges, one extra
    connection row).  Node genes are identical (8 homologous, no attribute
    difference), connections are 7 homologous + 1 non-homologous, so the answer is
    ``(1 * 1.0 + 0 * 0.4) / 8 = 0.125``.

    Assertion: ``np.isclose(new, legacy, rtol=0, atol=1e-6)`` and
    ``np.isclose(new, 0.125, rtol=0, atol=1e-6)`` — matching the legacy kernel and
    the closed form at the same time, so neither a shared bug nor a drifted
    normalisation can pass.
    """
    pair = genome_pair()
    left = payloads.chain_payload()
    # Rows 0..6 hold the chain's seven edges, so row 7 is the free row for the extra one.
    right = payloads.set_conn(payloads.chain_payload(), 0, 7, 0, 3)
    assert int(np.sum(~np.isnan(right.conn_attrs[0, :, 0]))) == 8

    new = new_distance(pair, left, right)
    legacy = legacy_distance(pair, left.of(0), right.of(0))

    assert np.isclose(new[0], legacy, rtol=0, atol=1e-6)
    assert np.isclose(new[0], 0.125, rtol=0, atol=1e-6)


def test_attribute_only_difference_matches_legacy(payloads, genome_pair, new_distance, legacy_distance):
    """Invariant: the homologous attribute term is ``sum(|Δ|) * weight / max(cnt1, cnt2)``.

    Oracle: hand computation *and* the legacy operator.  ``payloads.chain_payload()``
    with ``weight=1`` versus ``weight=2``: same 8 node keys (identical attributes),
    same 7 connection keys with ``|Δw| = 1`` each, so
    ``(0 * 1.0 + 7 * 1.0 * 0.4) / 7 = 0.4``.  This is the term the forward-parity
    tests never touch, and it is where a wrong ``compatibility_weight`` or a wrong
    ``max``/``sum`` normalisation shows up.

    Assertion: ``np.isclose(new, legacy, rtol=0, atol=1e-6)`` and
    ``np.isclose(new, 0.4, rtol=0, atol=1e-6)``.  A node-attribute variant is
    included the same way (``bias`` of one hidden row shifted by ``1`` adds
    ``1.0 * 0.4 / 8``), so the node gene's own ``distance`` is covered too.
    """
    pair = genome_pair()
    left = payloads.chain_payload(weight=1.0)
    right = payloads.chain_payload(weight=2.0)

    new = new_distance(pair, left, right)
    legacy = legacy_distance(pair, left.of(0), right.of(0))

    assert np.isclose(new[0], legacy, rtol=0, atol=1e-6)
    assert np.isclose(new[0], 0.4, rtol=0, atol=1e-6)

    # The node gene's own distance: the same doubled-weight network, with the bias of
    # the hidden row 3 raised from 0 to 1.  Its |Δ| is 1, so the node term adds
    # (0 * 1.0 + 1 * 1.0 * 0.4) / 8 = 0.05 on top of the 0.4 connection term.
    shifted = payloads.set_node(
        payloads.chain_payload(weight=2.0), 0, 3, 3, attrs=(1.0, 1.0, 0.0, 0.0)
    )
    new_node = new_distance(pair, left, shifted)
    legacy_node = legacy_distance(pair, left.of(0), shifted.of(0))

    assert np.isclose(new_node[0], legacy_node, rtol=0, atol=1e-6)
    assert np.isclose(new_node[0], 0.4 + 1.0 * 0.4 / 8, rtol=0, atol=1e-6)


def test_batch_of_pairs_matches_pairwise_legacy(payloads, genome_pair, new_distance, legacy_distance):
    """Invariant: the batched operator equals the pairwise one — the population axis is a pure batch axis.

    Oracle: ``P`` separate legacy calls.  The batch stacks four pairs whose
    distances differ by construction (identical, one extra connection, doubled
    weights, a skipped-node topology against the chain), and the legacy side
    evaluates each pair through ``payload.of(p)`` on both parents, because a
    legacy genome is one individual and ``convert.to_legacy`` refuses a batch.

    Assertion: ``new[p]`` equals the legacy distance of pair ``p`` within
    ``atol=1e-6`` for every ``p``, and the four values are not all equal — an
    operator that reduced across the population axis would return one repeated
    number and fail both halves.
    """
    pair = genome_pair()
    chain = payloads.chain_payload()
    left = _stack(
        payloads,
        [
            chain,
            chain,
            payloads.chain_payload(weight=1.0),
            payloads.skip_payload(),
        ],
    )
    right = _stack(
        payloads,
        [
            payloads.chain_payload(),
            payloads.set_conn(payloads.chain_payload(), 0, 7, 0, 3),
            payloads.chain_payload(weight=2.0),
            payloads.chain_payload(),
        ],
    )
    assert left.pop_size == right.pop_size == 4

    new = new_distance(pair, left, right)
    assert new.shape == (4,)

    for p in range(4):
        legacy = legacy_distance(pair, left.of(p), right.of(p))
        assert np.isclose(new[p], legacy, rtol=0, atol=1e-6), f"pair {p}"

    assert len(set(np.asarray(new, dtype=float).tolist())) == 4


def test_empty_pair_yields_zero_not_nan(payloads, genome_pair, new_distance, legacy_distance):
    """Invariant: two empty genomes have distance ``0``, not ``NaN``.

    Oracle: the documented ``max_cnt == 0`` guard (``val = where(max_cnt == 0, 0,
    val / max_cnt)`` in the legacy operator) plus the legacy kernel.  Both parents
    are ``payloads.new_payload()`` — every node and connection row absent — which
    is the "two freshly allocated genomes" edge case the speciation code can hit
    after heavy deletion.

    Assertion: ``new == 0.0`` and ``legacy == 0.0`` exactly, and
    ``not np.isnan(new)``: a divide-by-zero would give ``NaN`` here, and ``NaN``
    silently poisons the species assignment that consumes this number.
    """
    pair = genome_pair()
    empty = payloads.new_payload(pop_size=1)
    assert empty.pop_size == 1
    assert np.all(empty.node_keys == np.iinfo(np.int32).max)
    assert np.all(empty.conn_keys == np.iinfo(np.int32).max)

    new = new_distance(pair, empty, empty)
    legacy = legacy_distance(pair, empty, empty)

    assert new.shape == (1,)
    assert not np.isnan(new[0])
    assert new[0] == 0.0
    assert legacy == 0.0
