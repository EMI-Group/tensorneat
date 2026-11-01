"""Parity of ``forward``: the rewrite's unrolled pass vs the legacy ``fori_loop``.

What "identical inputs" means here
----------------------------------
One payload built on the host by ``support.payloads`` (node keys == row indices,
``I_INF``/``NaN`` padding, one activation and one aggregation option), converted
by ``./convert.py`` into the legacy ``(nodes, conns)`` float matrices, and one
input batch.  Nothing is drawn: no ``initialize``, no mutation, no crossover — the
two libraries only *evaluate* what the test wrote down.

Every test asserts three things
------------------------------
1. ``Transformed.edge_overflow == 0`` — asserted inside the ``new_schedule``
   fixture, before any value is produced.  The legacy forward aggregates over a
   full ``max_nodes``-wide column and truncates nothing, so a rewrite schedule
   that dropped an edge would compare two different networks.  The fixture is
   built with ``max_in_degree == max_nodes`` (``convert.max_in_degree_for``)
   precisely so this holds.
2. values: ``np.allclose(new, legacy, rtol=1e-5, atol=1e-6)`` on ``float32``;
3. the NaN pattern: ``np.isnan(new) == np.isnan(legacy)`` elementwise.  ``NaN`` is
   *data* in this library (an unconnected node keeps ``NaN`` and poisons its
   descendants), so a matching ``NaN`` is as binding as a matching number, and a
   test that compared ``nan_to_num`` values only would accept a different network.

Excluded by construction (``../CONTEXT.md``, "Known divergences …")
-------------------------------------------------------------------
* activation id ``-1`` — identity in the rewrite, clamped to the last option by
  the legacy ``lax.switch``; every payload here uses option ``0`` of a
  one-element activation list, and no attribute row is written with ``-1``;
* multi-option ACT/AGG lists — the rewrite evaluates every configured option and
  selects, so a non-total option could poison a branch it was not selected into;
  both sides are configured with one option each (``./conftest.genome_pair``);
* RNG draws, initial populations, mutations, crossovers and ``genome.hash()`` —
  never compared, they differ by construction (splitmix64 vs ``jax.random``).
"""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.parity]


# --------------------------------------------------------------------------- topologies
def test_dense_chain_matches_legacy(
    payloads, genome_pair, new_forward, legacy_forward
):
    """Invariant: on a dense chain both forwards produce the same values and the same NaN pattern.

    Oracle: the *legacy kernel itself* on the identical payload.  The genome is
    ``payloads.chain_payload()`` — rows ``0..7`` all present, edges ``0->2``,
    ``1->2``, ``2->3``, ``3->4``, ``4->5``, ``5->6``, ``6->7``, one hidden row per
    level — evaluated on ``inputs = [[0, 1]]`` (one individual, one sample), so
    the deepest path (six levels) exercises the whole unroll.

    Assertion: ``np.allclose(new, legacy, rtol=1e-5, atol=1e-6)`` and
    ``np.isnan(new) == np.isnan(legacy)``; the output must be finite (the chain is
    connected end to end), which the test states explicitly so a both-sides-``NaN``
    coincidence cannot pass as equality.
    """
    pytest.skip(
        "not implemented: dense-chain forward equals the legacy forward "
        "(owner: genome/default.DefaultGenome.forward, genome/base.BaseGenome._node_step)"
    )


def test_skipped_node_matches_legacy(
    payloads, genome_pair, new_forward, legacy_forward
):
    """Invariant: a bypassed node's level is ``1``, not ``2``, on both sides.

    Oracle: the legacy kernel on ``payloads.skip_payload()`` (``0->2``, ``1->2``,
    ``0->3``, ``3->7``): row ``2`` receives the full input signal but nothing
    consumes it, while row ``3`` reads input ``0`` directly.  A level-synchronous
    schedule that incremented the level through *every* out-edge would place row
    ``3`` one level too deep and read a stale (``NaN``) value in a longer network;
    here it shows up as ``NaN`` in the read-out.

    Assertion: values within ``rtol=1e-5, atol=1e-6``, identical NaN pattern, and
    the read-out is finite on both sides — plus the perturbation half: rewriting
    row ``2``'s attribute row to ``NaN`` (or a large bias) must leave *both*
    forwards' read-outs bit-identical, which is the property the schedule has to
    reproduce, not just the values.
    """
    pytest.skip(
        "not implemented: skipped-node forward equals the legacy forward, under perturbation "
        "(owner: genome/default.DefaultGenome.forward, common/tensorization.derive_schedule)"
    )


def test_isolated_node_stays_nan_and_poisons_the_output(
    payloads, genome_pair, new_forward, legacy_forward
):
    """Invariant: a present node with no edges keeps ``NaN``, and so does its subtree.

    Oracle: the legacy kernel on ``payloads.isolated_payload()`` (``0->7``,
    ``1->7``; row ``2`` present but edge-less).  Two halves, both compared against
    the legacy result on the same payload:

    * row ``2`` alone: the read-out is finite on both sides (the isolated node
      does not reach the output, so nothing changes — the "its value is ``NaN``
      but nobody reads it" case);
    * the same payload plus the single edge ``2->7``: the read-out is ``NaN`` on
      **both** sides, because the only live slot of the output row carries a
      ``NaN`` source.  Asserting ``np.isnan(...)`` on both results is what makes
      this a parity assertion rather than a "both happened to be finite" one.
    """
    pytest.skip(
        "not implemented: isolated node stays NaN and poisons its descendants on both sides "
        "(owner: genome/base.BaseGenome._node_step, genome/default.DefaultGenome.forward)"
    )


@pytest.mark.parametrize("case", [0, 1, 2, 3, 4])
def test_random_dags_match_legacy(
    case, payloads, genome_pair, rng, new_forward, legacy_forward
):
    """Invariant: on random feedforward DAGs the two kernels agree, values and NaN pattern.

    Oracle: the legacy kernel on the same randomly built payload.  Topologies are
    drawn on the host from the ``rng`` fixture (seeded from the test id, so each
    ``case`` is its own reproducible DAG): a random acyclic edge set among the
    present rows, edge count well inside ``max_conns``, in-degree inside
    ``max_in_degree == max_nodes``, node key == row index, weights and biases
    drawn uniformly from ``[-1, 1]``, ``response = 1``, one ACT option and one AGG
    option.  Five cases make an accidentally special-cased schedule (a chain, a
    single fan-in) visible.

    Assertion: ``np.allclose(new, legacy, rtol=1e-5, atol=1e-6)`` where both are
    finite, and ``np.isnan(new) == np.isnan(legacy)`` everywhere — an unreachable
    node in the drawing is expected to yield ``NaN`` on both sides.
    """
    pytest.skip(
        "not implemented: random feedforward DAGs equal the legacy forward "
        "(owner: genome/default.DefaultGenome.forward)"
    )


def test_batch_of_individuals_matches_legacy_per_individual(
    payloads, genome_pair, new_forward, legacy_forward
):
    """Invariant: the population axis is a pure batch axis — ``P`` in one call == ``P`` calls.

    Oracle: the legacy kernel, which has no batch axis (one ``(nodes, conns)`` pair
    per call).  The payload stacks three *different* individuals —
    ``chain_payload(weight=1.0)``, ``chain_payload(weight=0.5)`` and
    ``skip_payload()``, built with ``pop_size=1`` and concatenated along axis 0 —
    and the legacy side evaluates them one at a time via ``payload.of(p)`` (a
    legacy genome is one individual; ``convert.to_legacy`` refuses a batch).
    Differing weights *and* differing topologies make an accidental cross-row
    gather visible.

    Assertion: row ``p`` of the ``(3, num_outputs)`` rewrite result equals the
    legacy result of individual ``p`` within ``rtol=1e-5, atol=1e-6``, for every
    ``p``, with identical NaN patterns.
    """
    pytest.skip(
        "not implemented: the population axis is a pure batch axis "
        "(owner: genome/default.DefaultGenome.forward, genome/base.BaseGenome._node_step)"
    )


# --------------------------------------------------------------------------- transforms
def test_input_transform_matches_legacy(payloads, genome_pair, new_forward, legacy_forward):
    """Invariant: ``input_transform`` is applied to ``inputs`` *before* the input rows are seeded.

    Oracle: the legacy kernel with the same transform.  Both genomes are built
    from the one host-side description ``input_transform="affine"``
    (``lambda x: x * 2.0 + 1.0`` — operator-only, so the very same callable is
    valid inside an evoxc trace and on a ``jax`` array), on
    ``payloads.chain_payload()`` with inputs ``[[0, 1]]``.

    Assertion: values and NaN pattern equal the legacy result, *and* differ from
    the untransformed run on the same payload — the second half is what proves the
    transform is actually applied, which a broken genome that ignored it would
    otherwise pass by matching the plain run.
    """
    pytest.skip(
        "not implemented: input_transform applies before the input rows are seeded "
        "(owner: genome/default.DefaultGenome.forward, genome/base.BaseGenome.__init__)"
    )


def test_output_transform_matches_legacy(payloads, genome_pair, new_forward, legacy_forward):
    """Invariant: ``output_transform`` maps the read-out only, once, on both sides.

    Oracle: the legacy kernel with the same transform.  The genome is built from
    ``output_transform="affine"`` (``lambda x: x * 2.0 + 1.0``) on
    ``payloads.chain_payload()``; because the map is not monotone in the sense of
    "cannot hide a mistake", the test also compares against the untransformed run
    and asserts ``new == 2 * plain + 1`` elementwise — a genome that applied the
    transform to the node *values* would fail that relation while still matching
    the legacy side.

    Assertion: values within ``rtol=1e-5, atol=1e-6`` against the legacy result,
    identical NaN pattern, and the affine relation to the untransformed rewrite run.
    """
    pytest.skip(
        "not implemented: output_transform applies to the read-out only "
        "(owner: genome/default.DefaultGenome.forward, genome/base.BaseGenome._output_values)"
    )


# --------------------------------------------------------------------------- the guard
def test_schedule_is_lossless_for_every_parity_payload(
    payloads, genome_pair, new_schedule
):
    """Invariant: ``Transformed.edge_overflow == 0`` for every payload used in this directory.

    Oracle: none needed — this is the *precondition* of every other comparison in
    ``./``.  It is asserted (not skipped over) so a payload whose in-degree exceeds
    the window can never be compared: the legacy forward aggregates over the full
    ``max_nodes`` column and the rewrite over ``max_in_degree`` slots, so an
    overflow means the two kernels saw different edge sets.

    Assertion: for ``chain_payload``, ``skip_payload``, ``isolated_payload`` and
    ``ring_payload`` (the last one is expected to be *unusable* here — its cycle
    has no finite level — so it is built with ``max_nodes == max_in_degree`` and
    only the overflow counter is asserted, not the values),
    ``np.all(np.asarray(transformed.edge_overflow) == 0)``, plus the constructive
    half: building the same payload pair with ``max_in_degree`` one smaller than
    the widest in-degree makes ``edge_overflow`` non-zero, which is exactly what
    ``convert.max_in_degree_for`` prevents.
    """
    pytest.skip(
        "not implemented: the schedule never drops an edge for the parity window "
        "(owner: common/tensorization.derive_schedule, common/errors.ShapeBoundError)"
    )
