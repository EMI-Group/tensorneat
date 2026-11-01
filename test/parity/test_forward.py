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
   precisely so this holds.  The one deliberate exception is the constructive half
   of :func:`test_schedule_is_lossless_for_every_parity_payload`, which builds an
   undersized window to *show* the overflow and never compares any value.
2. values: ``np.allclose(new, legacy, rtol=1e-5, atol=1e-6)`` on ``float32``;
3. the NaN pattern: ``np.isnan(new) == np.isnan(legacy)`` elementwise.  ``NaN`` is
   *data* in this library, so a matching ``NaN`` is as binding as a matching
   number, and a test that compared ``nan_to_num`` values only would accept a
   different network.  Both forwards implement the same NaN rule: a node is
   updated only when at least one of its incoming sources holds a number, so a
   ``NaN`` source is *ignored* next to a live one and a node whose every source is
   ``NaN`` keeps its own ``NaN`` seed (the legacy ``valid_mask = conn_exists &
   src_valid``, mirrored by the rewrite's per-slot ``live`` mask).

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

import numpy as np
import pytest

pytestmark = [pytest.mark.parity]

pytest.importorskip("jax", reason="parity needs the `bench` extra (jax)")

#: The single input row every topology is evaluated on: two inputs, one sample.
INPUTS = np.array([[0.0, 1.0]], dtype=np.float32)
#: The tolerance pair for a forward value (``../CONTEXT.md``, "Tolerances").
RTOL = 1e-5
ATOL = 1e-6


def _copy(payloads, payload):
    """A writable deep copy of ``payload`` (the builders mutate in place)."""
    node_keys, node_attrs, conn_keys, conn_attrs = (
        array.copy() for array in payload.args()
    )
    return payloads.Payload(node_keys, node_attrs, conn_keys, conn_attrs)


def _assert_matches(new_out, legacy_out):
    """Compare a rewrite ``(1, num_outputs)`` read-out with one legacy row.

    Asserts the NaN pattern elementwise (``NaN`` is data, so a matching ``NaN`` is
    as binding as a matching number) and then the values on the positions both
    sides filled with a number.
    """
    new_row = np.asarray(new_out)[0]
    legacy_row = np.asarray(legacy_out)
    assert new_row.shape == legacy_row.shape
    assert np.array_equal(np.isnan(new_row), np.isnan(legacy_row)), (
        f"NaN pattern differs: rewrite {np.isnan(new_row).tolist()} vs "
        f"legacy {np.isnan(legacy_row).tolist()}"
    )
    finite = np.isfinite(new_row) & np.isfinite(legacy_row)
    np.testing.assert_allclose(new_row[finite], legacy_row[finite], rtol=RTOL, atol=ATOL)


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
    pair = genome_pair()
    payload = payloads.chain_payload()

    new_out = new_forward(pair, payload, INPUTS)
    legacy_out = legacy_forward(pair, payload, INPUTS[0])

    assert new_out.shape == (1, pair.num_outputs)
    assert np.all(np.isfinite(new_out))
    assert np.all(np.isfinite(legacy_out))
    _assert_matches(new_out, legacy_out)


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
    row ``2``'s attribute row to a large bias and a different response must leave
    *both* forwards' read-outs bit-identical, which is the property the schedule
    has to reproduce, not just the values.
    """
    pair = genome_pair()
    payload = payloads.skip_payload()

    new_out = new_forward(pair, payload, INPUTS)
    legacy_out = legacy_forward(pair, payload, INPUTS[0])

    assert new_out.shape == (1, pair.num_outputs)
    assert np.all(np.isfinite(new_out))
    assert np.all(np.isfinite(legacy_out))
    _assert_matches(new_out, legacy_out)

    perturbed = _copy(payloads, payload)
    payloads.set_node(perturbed, 0, 2, 2, attrs=[10.0, 1.0, 0.0, 0.0])

    new_perturbed = new_forward(pair, perturbed, INPUTS)
    legacy_perturbed = legacy_forward(pair, perturbed, INPUTS[0])

    np.testing.assert_array_equal(new_perturbed, new_out)
    np.testing.assert_array_equal(legacy_perturbed, legacy_out)
    _assert_matches(new_perturbed, legacy_perturbed)


def test_isolated_node_stays_nan_and_poisons_the_output(
    payloads, genome_pair, new_forward, legacy_forward
):
    """Invariant: a present node with no edge holds ``NaN``, and a consumer of *only* that node does too.

    Oracle: the legacy kernel on ``payloads.isolated_payload()`` (``0->7``,
    ``1->7``; row ``2`` present but edge-less).  Three halves, all compared against
    the legacy result on the same payload:

    * plain payload: the read-out is finite on both sides — the isolated node's
      ``NaN`` never reaches the output, so nothing changes;
    * the same payload plus the single edge ``2->7`` while ``0->7``/``1->7`` stay:
      the read-out is still finite on both sides, because both forwards drop a
      ``NaN`` source when the destination also has a live one (the output row's
      other two slots), which is the "a ``NaN`` edge is ignored" rule;
    * the edge ``2->7`` as the output row's *only* incoming edge: the read-out is
      ``NaN`` on **both** sides, because the output then has no live source and
      keeps its ``NaN`` seed.  Asserting ``np.isnan`` on both results is what makes
      this a parity assertion rather than a "both happened to be finite" one.
    """
    pair = genome_pair()
    payload = payloads.isolated_payload()

    new_out = new_forward(pair, payload, INPUTS)
    legacy_out = legacy_forward(pair, payload, INPUTS[0])
    assert np.all(np.isfinite(new_out))
    assert np.all(np.isfinite(legacy_out))
    _assert_matches(new_out, legacy_out)

    mixed = _copy(payloads, payload)
    payloads.set_conn(mixed, 0, 2, 2, 7)
    mixed_new = new_forward(pair, mixed, INPUTS)
    mixed_legacy = legacy_forward(pair, mixed, INPUTS[0])
    assert np.all(np.isfinite(mixed_new))
    assert np.all(np.isfinite(mixed_legacy))
    _assert_matches(mixed_new, mixed_legacy)

    poisoned = _copy(payloads, payload)
    payloads.set_conn(poisoned, 0, 0, payloads.I_INF, payloads.I_INF)
    payloads.set_conn(poisoned, 0, 1, payloads.I_INF, payloads.I_INF)
    payloads.set_conn(poisoned, 0, 2, 2, 7)
    poisoned_new = new_forward(pair, poisoned, INPUTS)
    poisoned_legacy = legacy_forward(pair, poisoned, INPUTS[0])
    assert np.isnan(poisoned_new).all()
    assert np.isnan(poisoned_legacy).all()
    _assert_matches(poisoned_new, poisoned_legacy)


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
    finite, and ``np.isnan(new) == np.isnan(legacy)`` everywhere — a node the
    drawing leaves unreachable yields ``NaN`` on both sides.
    """
    pair = genome_pair()
    payload = payloads.new_payload(pop_size=1, max_nodes=pair.max_nodes, max_conns=pair.max_conns)
    biases = rng.uniform(-1.0, 1.0, size=pair.max_nodes)
    weights = rng.uniform(-1.0, 1.0, size=pair.max_conns)
    for row in range(pair.max_nodes):
        payloads.set_node(
            payload, 0, row, row, attrs=[float(biases[row]), 1.0, 0.0, 0.0]
        )

    # Every edge goes from a lower row to a higher one, so the drawing is acyclic
    # by construction; input rows are destination-free (they are seeded, never
    # computed), and the in-degree cap is the rewrite's incoming window.
    candidates = [
        (source, target)
        for source in range(pair.max_nodes)
        for target in range(max(source + 1, pair.num_inputs), pair.max_nodes)
    ]
    wanted = int(rng.integers(1, min(len(candidates), pair.max_conns) + 1))
    in_degree = np.zeros(pair.max_nodes, dtype=np.int64)
    chosen: list[tuple[int, int]] = []
    for index in rng.permutation(len(candidates)):
        if len(chosen) >= wanted:
            break
        source, target = candidates[int(index)]
        if in_degree[target] >= pair.max_in_degree:
            continue
        chosen.append((source, target))
        in_degree[target] += 1
    assert chosen, f"case {case} drew an empty DAG"

    for row, (source, target) in enumerate(chosen):
        payloads.set_conn(payload, 0, row, source, target, attrs=[float(weights[row])])

    inputs = np.array([[0.3, -0.7]], dtype=np.float32)
    new_out = new_forward(pair, payload, inputs)
    legacy_out = legacy_forward(pair, payload, inputs[0])

    assert new_out.shape == (1, pair.num_outputs)
    _assert_matches(new_out, legacy_out)


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
    pair = genome_pair()
    individuals = [
        payloads.chain_payload(weight=1.0),
        payloads.chain_payload(weight=0.5),
        payloads.skip_payload(),
    ]
    payload = payloads.Payload(
        *[
            np.concatenate([getattr(ind, field) for ind in individuals], axis=0)
            for field in payloads.Payload.field_names
        ]
    )
    assert payload.pop_size == len(individuals)
    inputs = np.tile(INPUTS, (len(individuals), 1))

    new_out = new_forward(pair, payload, inputs)
    assert new_out.shape == (len(individuals), pair.num_outputs)

    for p in range(len(individuals)):
        legacy_out = legacy_forward(pair, payload.of(p), inputs[p])
        _assert_matches(new_out[p : p + 1], legacy_out)

    assert not np.allclose(new_out[0], new_out[1], rtol=RTOL, atol=ATOL), (
        "the two chains differ only in weight, so their read-outs must differ"
    )
    assert not np.allclose(new_out[0], new_out[2], rtol=RTOL, atol=ATOL), (
        "the chain and the skipped-node topology must not read out the same value"
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
    payload = payloads.chain_payload()
    transformed_pair = genome_pair(input_transform="affine")
    plain_pair = genome_pair()

    new_out = new_forward(transformed_pair, payload, INPUTS)
    legacy_out = legacy_forward(transformed_pair, payload, INPUTS[0])
    plain_out = new_forward(plain_pair, payload, INPUTS)

    assert np.all(np.isfinite(new_out))
    _assert_matches(new_out, legacy_out)
    assert not np.allclose(new_out, plain_out, rtol=RTOL, atol=ATOL), (
        "the affine input transform must change the read-out of the chain"
    )


def test_output_transform_matches_legacy(payloads, genome_pair, new_forward, legacy_forward):
    """Invariant: ``output_transform`` maps the read-out only, once, on both sides.

    Oracle: the legacy kernel with the same transform.  The genome is built from
    ``output_transform="affine"`` (``lambda x: x * 2.0 + 1.0``) on
    ``payloads.chain_payload()``; because that map is easily confused with a
    per-node transform, the test also compares against the untransformed run and
    asserts ``new == 2 * plain + 1`` elementwise — a genome that applied the
    transform to the node *values* would fail that relation while still matching
    the legacy side.

    Assertion: values within ``rtol=1e-5, atol=1e-6`` against the legacy result,
    identical NaN pattern, and the affine relation to the untransformed rewrite run.
    """
    payload = payloads.chain_payload()
    transformed_pair = genome_pair(output_transform="affine")
    plain_pair = genome_pair()

    new_out = new_forward(transformed_pair, payload, INPUTS)
    legacy_out = legacy_forward(transformed_pair, payload, INPUTS[0])
    plain_out = new_forward(plain_pair, payload, INPUTS)

    assert np.all(np.isfinite(new_out))
    _assert_matches(new_out, legacy_out)
    np.testing.assert_allclose(
        2.0 * plain_out + 1.0, new_out, rtol=RTOL, atol=ATOL
    )


# --------------------------------------------------------------------------- the guard
def test_schedule_is_lossless_for_every_parity_payload(
    payloads, genome_pair, new_schedule, runner
):
    """Invariant: ``Transformed.edge_overflow == 0`` for every payload used in this directory.

    Oracle: none needed — this is the *precondition* of every other comparison in
    ``./``.  It is asserted (not skipped over) so a payload whose in-degree exceeds
    the window can never be compared: the legacy forward aggregates over the full
    ``max_nodes`` column and the rewrite over ``max_in_degree`` slots, so an
    overflow means the two kernels saw different edge sets.

    Assertion: for ``chain_payload``, ``skip_payload``, ``isolated_payload`` and
    ``ring_payload`` (the last one is *unusable* here — its cycle has no finite
    level — so only the overflow counter is asserted, never the values),
    ``np.all(np.asarray(transformed.edge_overflow) == 0)``, plus the constructive
    half: the same layout built with ``max_in_degree == 5`` (the constructor's
    floor for this genome) and a payload whose output row has in-degree ``7``
    reports ``edge_overflow == [2]``, which is exactly what
    ``convert.max_in_degree_for`` prevents.  That lossy pair is never compared
    value by value: it sees a different edge set, by construction.
    """
    pair = genome_pair()
    for builder in (
        payloads.chain_payload,
        payloads.skip_payload,
        payloads.isolated_payload,
        payloads.ring_payload,
    ):
        payload = builder()
        transformed = new_schedule(pair, payload)
        overflow = np.asarray(transformed.edge_overflow)
        assert np.all(overflow == 0), (
            f"{builder.__name__} overflows a {pair.max_in_degree}-wide window: "
            f"{overflow.tolist()}"
        )

    lossy_pair = genome_pair(max_in_degree=5)
    payload = payloads.new_payload(pop_size=1)
    for row in range(lossy_pair.max_nodes):
        payloads.set_node(payload, 0, row, row)
    output_row = lossy_pair.max_nodes - 1
    for row in range(output_row):
        payloads.set_conn(payload, 0, row, row, output_row)

    transformed = runner.run_defn(
        lambda node_keys, conn_keys: lossy_pair.new_genome.transform(
            lossy_pair.new_state, node_keys, conn_keys
        ),
        payload.node_keys,
        payload.conn_keys,
    )
    overflow = np.asarray(transformed.edge_overflow)
    np.testing.assert_array_equal(overflow, np.array([2], dtype=overflow.dtype))
