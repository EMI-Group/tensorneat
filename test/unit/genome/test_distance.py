"""Unit tests for compatibility distance — the speciation metric.

What this module proves
-----------------------
``DefaultDistance`` answers one question per parent pair: *how different are these
two genomes?* The algorithm's speciation threshold is only meaningful if the answer
is a stable, symmetric, zero-at-identity number, so the tests assert exactly those
properties of ``node_distance + conn_distance``:

* **zero at identity** — the same genome against itself is ``0``, and so is a pair
  of empty gene families (``0``, never ``NaN``: a ``NaN`` distance would send every
  genome to its own species);
* **symmetry** — ``d(a, b) == d(b, a)``, which the ``/ max(cnt1, cnt2)``
  normalisation is what makes true; an unnormalised count is symmetric too, so the
  test also covers pairs of *different* sizes and pins the absolute value;
* **monotonicity** — adding a gene that has no homologue in the other parent
  strictly increases the distance (``compatibility_disjoint`` counts non-homologous
  genes, disjoint *and* excess alike), and adding an attribute difference increases
  it in proportion to ``compatibility_weight``;
* **composition** — the batched call on ``P`` pairs equals ``P`` single-pair calls
  (the population axis is a pure batch axis here too), and repeated calls agree;
* **the digested route's key window** — ``measure_distance`` against a
  ``prepare_distance`` digest refuses a present key outside
  ``[0, genome.key_span)`` (centre and population, both gene families) on the
  dense single-centre route with a message naming the offender, accepts the same
  poisoned payload on ``DefaultDistance(dense_single_centre=False)``, and both
  routes answer bit-identically to ``execute_distance`` on the legal domain.

Oracles used here (never the kernel itself)
-------------------------------------------
* hand arithmetic on the documented formula
  ``(non_homologous_count * compatibility_disjoint + homologous_attribute_distance *
  compatibility_weight) / max(cnt1, cnt2)`` with the attribute term itself
  ``sum(|delta|)`` over the float columns and ``sum(delta != 0)`` over the int
  columns (``BaseGene.distance``);
* ``support.payloads`` topologies whose gene counts and attribute values are known
  exactly, built in pairs that differ by one controlled edit;
* ``P`` single-pair runs for the batching check — the same kernel, but with the
  batch axis of size one, which is the only way to see a cross-pair reduction.

Gotchas this module encodes
---------------------------
* The two terms are **added** over gene families (node genes and connection genes),
  so a distance of ``0`` requires both families to be identical; a test that only
  varied connection weights could not tell an implemented node term from an absent
  one. Both are varied here.
* ``max(cnt1, cnt2)`` — not the sum, not the minimum. A pair where one parent has
  twice the genes is therefore normalised by the larger count; the empty-pair case
  would divide by zero, which is why the docstring pins the guarded ``0``.
* Int attribute columns (``aggregation``/``activation`` ids) contribute ``!=``,
  i.e. ``1`` per differing option, not ``|delta|``: changing activation from
  option ``0`` to option ``2`` is *one* difference, not two. ``DefaultNode`` is a
  single-option configuration by default, so the test configures three options to
  make the two readings distinguishable at all.
* Nothing here is differentiable and nothing is random, so distance is the one
  operator whose output parity against the frozen implementation is exact rather
  than ``allclose``; this module only asserts its algebra.

Backend is ``"numpy"`` throughout, through the shared runner only; no
``jax``/``torch``, no network, no GPU.
"""

from __future__ import annotations

import numpy as np
import pytest

#: ``payloads.chain_payload()`` at the default shapes (``max_nodes=8``,
#: ``num_inputs=2``, ``num_outputs=1``): every node row is present, so the chain
#: carries 8 node genes and 7 connection genes. ``test/parity/test_distance.py``
#: hand-computes with the same two counts, which is what makes these numbers
#: comparable across suites.
CHAIN_NODES = 8
CHAIN_CONNS = 7

#: Static key bound ``S`` of the small genome the key-span tests build:
#: ``max(max_nodes, num_inputs + num_outputs) + max_conns`` = ``max(8, 3) + 16``.
#: ``[0, KEY_SPAN)`` is the window the dense single-centre key table can address,
#: so ``KEY_SPAN`` (``24``) and ``-1`` are the two ways of stepping outside it.
KEY_SPAN = 24

#: Gene capacities of every key-span payload below (``(N, E)`` of the payload's
#: ``(P, N)``/``(P, E, CK)`` arrays), chosen so ``key_span`` is :data:`KEY_SPAN`.
SPAN_NODES = 8
SPAN_CONNS = 16


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _distance(run, genome_object, state, left, right):
    """One ``(P,)`` distance vector for ``P`` parent pairs.

    Args:
        run: The ``run_kernel`` fixture — ``run(method, state, *arrays)``.
        genome_object: The genome owning the distance operator.
        state: Host configuration container (``payloads.genome_state``).
        left: ``(P, ...)`` payload of parent 1, one individual per pair.
        right: Payload of parent 2 in the same static shape.

    Returns:
        ``(P,) float32`` numpy distances: entry ``p`` compares ``left[p]`` with
        ``right[p]``.

    Notes:
        The kernel never forms pairs itself — the algorithm does — so the two
        payloads must already be aligned on the population axis. Both are passed
        as their four arrays (the genome's positional payload spelling) because
        the shared runner hands only top-level arrays to ``evoxc`` as data.
    """
    return run(
        genome_object.execute_distance,
        state,
        *left.args(),
        *right.args(),
    )


def _stack(*individuals):
    """One batched payload built from single-individual payloads.

    Args:
        *individuals: ``Payload`` objects of one shape but possibly different
            gene counts and attribute values.

    Returns:
        A ``Payload`` whose population axis is the concatenation of theirs, so
        individual ``p`` is ``individuals[p]``.

    Raises:
        ValueError: Propagated from ``numpy.concatenate``/``Payload`` if the
            payloads do not share their static shape.
    """
    payload_type = type(individuals[0])
    return payload_type(
        *(
            np.concatenate([getattr(payload, field) for payload in individuals], axis=0)
            for field in payload_type.field_names
        )
    )


def _cleared(payloads, payload):
    """The same payload with every connection row absent (its node genes untouched).

    Args:
        payloads: The ``support.payloads`` module (for ``set_conn``/``I_INF``).
        payload: A ``pop_size == 1`` payload, edited in place.

    Returns:
        The same payload, its whole connection family at the ``I_INF``/``NaN``
        padding sentinels.
    """
    for row in range(payload.max_conns):
        payloads.set_conn(payload, 0, row, payloads.I_INF, payloads.I_INF)
    return payload


def _span_genome(genome, *, dense):
    """A ``DefaultGenome`` whose ``key_span`` is :data:`KEY_SPAN`, on a chosen route.

    Args:
        genome: The ``tensorneat.genome`` package under test.
        dense (bool): ``DefaultDistance(dense_single_centre=dense)`` — ``True`` is
            the shipped dense key-table route for a batch-1 centre, ``False`` the
            row-wise merge that matches any key exactly.

    Returns:
        The genome. For ``dense=False`` the attribute survives on the operator
        (``genome.distance.dense_single_centre``), so the route a test exercises
        is the configuration it named, not a default.
    """
    genome_object = genome.DefaultGenome(
        num_inputs=2,
        num_outputs=1,
        max_nodes=SPAN_NODES,
        max_conns=SPAN_CONNS,
        node_gene=genome.DefaultNode(),
        conn_gene=genome.DefaultConn(),
        mutation=genome.DefaultMutation(),
        crossover=genome.DefaultCrossover(),
        distance=genome.DefaultDistance(dense_single_centre=dense),
    )
    assert genome_object.key_span == KEY_SPAN
    assert genome_object.distance.dense_single_centre is dense
    return genome_object


def _span_payload(payloads, count=3):
    """``count`` copies of a clean in-span genome: node keys ``0/1/2``, edge ``0 -> 2``.

    Args:
        payloads: The ``support.payloads`` module.
        count (int): Population axis of the payload.

    Returns:
        A payload whose every individual is that three-node, one-edge genome, with
        neutral ``DefaultNode``/``DefaultConn`` attribute rows.

    Notes:
        Every key is inside ``[0, KEY_SPAN)``, so the payload's digest reports
        ``node_oor == conn_oor == 0``. A test that poisons one key therefore knows
        which family and which individual the refusal is about.
    """
    payload = payloads.new_payload(count, SPAN_NODES, SPAN_CONNS)
    for p in range(count):
        for row, key in enumerate((0, 1, 2)):
            payloads.set_node(payload, p, row, key)
        payloads.set_conn(payload, p, 0, 0, 2)
    return payload


def _tiled(payloads, one, count):
    """``count`` broadcast copies of a one-individual payload.

    Args:
        payloads: The ``support.payloads`` module (for the ``Payload`` type).
        one: A ``pop_size == 1`` payload — the centre of a ``measure_distance`` call.
        count (int): Target population axis.

    Returns:
        The payload ``execute_distance`` needs to compare that centre with every
        individual of a ``count``-row population.

    Notes:
        A broadcast view is enough: the runner makes each array contiguous before
        it reaches ``evoxc``, and nothing downstream writes to a payload.
    """
    return payloads.Payload(
        *(np.broadcast_to(array, (count,) + array.shape[1:]) for array in one.args())
    )


# --------------------------------------------------------------------------- #
# configuration
# --------------------------------------------------------------------------- #


def test_negative_weights_are_rejected(genome, common):
    """Invariant: a negative ``compatibility_disjoint``/``compatibility_weight`` raises.

    Oracle: the documented constructor validation — the error must be
    ``TensorNEATError`` (the package hierarchy), raised at construction rather than
    at the first ``evoxc.build``. A negative weight would make "more different" mean
    "closer", which the algorithm cannot detect on its own.

    Assertion: each invalid weight (negative, and the ``NaN``/``inf`` values that
    would poison every distance) raises ``TensorNEATError`` whose message names the
    offending argument; ``0`` is a legal boundary (the documented "count this term
    not at all" configuration) and must construct.
    """
    invalid = (-1.0, -1e-9, float("nan"), float("inf"))
    for name in ("compatibility_disjoint", "compatibility_weight"):
        for value in invalid:
            with pytest.raises(common.TensorNEATError) as caught:
                genome.DefaultDistance(**{name: value})
            assert name in str(caught.value)

    boundary = genome.DefaultDistance(
        compatibility_disjoint=0.0, compatibility_weight=0.0
    )
    assert boundary.compatibility_disjoint == 0.0
    assert boundary.compatibility_weight == 0.0


# --------------------------------------------------------------------------- #
# the identity and symmetry axioms
# --------------------------------------------------------------------------- #


def test_identical_genomes_have_zero_distance(payloads, run_kernel, evoxc_key):
    """Invariant: ``d(g, g) == 0`` for every individual of a non-trivial payload.

    Oracle: the identity axiom, on a payload that has both gene families populated
    (nodes *and* connections) and padding rows, so an over-counted padding row or a
    node-gene term applied to absent rows would show as a non-zero distance. Also
    asserted for a freshly *initialized* population, whose attribute rows are
    randomly drawn per individual and therefore not the neutral rows above.

    Assertion: the returned vector is ``(P,) float32`` and every entry is exactly
    ``0.0`` — ``array_equal``, not ``allclose``, because a non-zero identity
    distance is a behaviour change, not a numerical wobble.
    """
    genome_object = payloads.default_genome(
        max_nodes=8,
        max_conns=16,
        num_inputs=2,
        num_outputs=1,
        max_in_degree=4,
    )
    state = payloads.genome_state(genome_object)

    mixed = _stack(
        payloads.chain_payload(),  # every node row present, dense connections
        payloads.skip_payload(),  # padding rows in both gene families
        payloads.isolated_payload(),  # one unconnected node, few connections
        payloads.new_payload(),  # the empty genome
    )
    distances = _distance(run_kernel, genome_object, state, mixed, mixed)
    assert distances.shape == (mixed.pop_size,)
    assert distances.dtype == np.float32
    np.testing.assert_array_equal(distances, np.zeros(mixed.pop_size, np.float32))

    # ``initialize`` draws a *different* dense network per individual, so this is the
    # identity axiom on genomes whose attributes are not the neutral rows above.
    population = tuple(run_kernel(genome_object.initialize, state, evoxc_key, 3))
    assert population[0].shape[0] == 3
    assert not np.array_equal(population[1][0], population[1][1])  # not degenerate
    drawn = run_kernel(genome_object.execute_distance, state, *population, *population)
    np.testing.assert_array_equal(drawn, np.zeros(3, np.float32))


def test_distance_is_symmetric(payloads, run_kernel):
    """Invariant: ``d(a, b) == d(b, a)`` within one ulp, including for unequal gene counts.

    Oracle: the two calls. Asymmetry is the classic symptom of an unnormalised or
    half-counted term (``cnt1`` instead of ``max(cnt1, cnt2)``), and it only shows
    on parents with different numbers of genes — so the pair is built with parent 2
    strictly larger than parent 1, and the reverse call is run on the same arrays.

    Assertion: the two vectors agree within one ``float32`` ulp, and both equal the
    hand value ``3/8 + 7/7 = 1.375`` (3 non-homologous node genes out of 8, 7
    non-homologous connection genes out of 7). The absolute value matters because an
    unnormalised count would be symmetric as well.
    """
    genome_object = payloads.default_genome(
        max_nodes=8,
        max_conns=16,
        num_inputs=2,
        num_outputs=1,
        max_in_degree=4,
    )
    state = payloads.genome_state(genome_object)

    smaller = payloads.skip_payload()  # 5 node genes, 4 connection genes
    larger = payloads.chain_payload()  # 8 node genes, 7 connection genes

    forward = _distance(run_kernel, genome_object, state, smaller, larger)
    backward = _distance(run_kernel, genome_object, state, larger, smaller)

    assert np.isfinite(forward).all()
    np.testing.assert_allclose(forward, backward, rtol=1e-7, atol=0)
    expected = 3 / CHAIN_NODES + 7 / CHAIN_CONNS
    np.testing.assert_allclose(forward, [expected], rtol=0, atol=1e-6)


# --------------------------------------------------------------------------- #
# monotonicity
# --------------------------------------------------------------------------- #


def test_distance_grows_with_non_homologous_genes(payloads, run_kernel):
    """Invariant: one added non-homologous gene strictly increases the distance.

    Oracle: hand arithmetic of the disjoint term. Starting from identical parents,
    adding one node (or one connection) to parent 2 that parent 1 does not have
    must add ``compatibility_disjoint / max(cnt1, cnt2)``-worth of distance, and
    adding a second one must add more again — *disjoint* and *excess* genes are
    counted by the same term, so the test walks the count ``0 -> 1 -> 2`` and
    checks strict monotonicity rather than a particular value.

    Assertion: ``d0 < d1 < d2`` for a node-gene walk and for a connection-gene walk,
    with ``d0`` exactly ``0`` and every entry finite. Both families are walked
    because the two terms are added: a missing node term would leave the connection
    walk green.
    """
    genome_object = payloads.default_genome(
        max_nodes=8,
        max_conns=16,
        num_inputs=2,
        num_outputs=1,
        max_in_degree=4,
    )
    state = payloads.genome_state(genome_object)

    # Node family: ``skip_payload`` has 5 present rows, so rows 4 and 5 are free.
    node_free_rows = (4, 5)
    node_steps = []
    for added in range(len(node_free_rows) + 1):
        step = payloads.skip_payload()
        for row in node_free_rows[:added]:
            payloads.set_node(step, 0, row, row)
        node_steps.append(step)

    # Connection family: the chain fills rows 0..6, so rows 7 and 8 are free.
    conn_edges = ((0, 3), (1, 4))
    conn_steps = []
    for added in range(len(conn_edges) + 1):
        step = payloads.chain_payload()
        for row, (source, target) in enumerate(conn_edges[:added], start=CHAIN_CONNS):
            payloads.set_conn(step, 0, row, source, target)
        conn_steps.append(step)

    node_baseline = payloads.skip_payload()
    node_distances = [
        _distance(run_kernel, genome_object, state, node_baseline, step)[0]
        for step in node_steps
    ]
    conn_baseline = payloads.chain_payload()
    conn_distances = [
        _distance(run_kernel, genome_object, state, conn_baseline, step)[0]
        for step in conn_steps
    ]

    for distances in (node_distances, conn_distances):
        assert np.isfinite(distances).all()
        assert distances[0] == 0.0
        assert distances[0] < distances[1] < distances[2], distances


def test_attribute_differences_are_weighted_by_compatibility_weight(
    payloads, run_kernel
):
    """Invariant: doubling a float attribute delta doubles that gene's distance contribution.

    Oracle: hand arithmetic of ``homologous_attribute_distance * compatibility_weight``
    with ``|delta|`` summed over the float columns. Two payloads that differ only in
    the weight of one known connection, by ``delta`` and ``2 * delta`` respectively,
    must give distances whose *differences from the identical-pair baseline* are in
    a ``1 : 2`` ratio — which pins both the absolute-value and the weight factor.

    Assertion: the identical pair is exactly ``0``; each edited pair equals
    ``compatibility_weight * |delta| / max(cnt1, cnt2)`` (``0.4 * delta / 7`` by
    hand — one changed connection gene out of the chain's 7); the doubled delta is
    twice the single one; and a *negative* delta gives the same distance as its
    magnitude, which is what ``|delta|`` means.
    """
    genome_object = payloads.default_genome(
        max_nodes=8,
        max_conns=16,
        num_inputs=2,
        num_outputs=1,
        max_in_degree=4,
    )
    state = payloads.genome_state(genome_object)
    weight = genome_object.distance.compatibility_weight

    baseline_payload = payloads.chain_payload()

    def with_weight(delta):
        """The chain with connection row 0's weight moved by ``delta``."""
        payload = payloads.chain_payload()
        payloads.set_conn(payload, 0, 0, 0, 2, attrs=[1.0 + delta])
        return payload

    baseline = _distance(
        run_kernel, genome_object, state, baseline_payload, baseline_payload
    )
    np.testing.assert_array_equal(baseline, np.zeros(1, np.float32))

    measured = {}
    for delta in (0.25, 0.5, -0.25):
        measured[delta] = _distance(
            run_kernel, genome_object, state, baseline_payload, with_weight(delta)
        )[0]
        expected = weight * abs(delta) / CHAIN_CONNS
        np.testing.assert_allclose(measured[delta], expected, rtol=1e-6, atol=0)

    # ``|delta|``, not ``delta``: the sign of a weight difference is irrelevant.
    np.testing.assert_allclose(measured[-0.25], measured[0.25], rtol=1e-6, atol=0)
    # ... and the term is linear in the delta, so doubling it doubles the distance.
    np.testing.assert_allclose(measured[0.5], 2 * measured[0.25], rtol=1e-6, atol=0)


def test_int_attribute_columns_contribute_one_per_changed_option(
    genome, common, payloads, run_kernel
):
    """Invariant: a changed option id contributes ``1``, not ``|id1 - id2|``.

    Oracle: ``BaseGene.distance``'s documented per-column rule — ``!=`` for an
    ``"int"`` spec, ``|delta|`` for a ``"float"`` spec. A ``DefaultNode`` configured
    with three activation options, whose rows hold option ``0`` and option ``2``, must
    contribute ``1``; reading the column as a float would contribute ``2``. This is
    the difference between counting changed *genes* and measuring a meaningless
    numeric gap between option indices.

    Assertion: the pair differs in exactly one node's activation id, so the distance
    is ``compatibility_weight * 1 / max(cnt1, cnt2)`` = ``0.4 / 8`` = ``0.05``
    exactly — and explicitly *not* the ``0.1`` the ``|delta|`` reading would give.
    """
    # ``payloads.default_genome`` hard-codes a single-option ``DefaultNode``, so the
    # genome is built here: three options are needed for ids ``0`` and ``2`` to be
    # distinguishable from a float delta of ``2``.
    node_gene = genome.DefaultNode(
        activation_options=[common.ACT.sigmoid, common.ACT.relu, common.ACT.tanh]
    )
    assert [spec.kind for spec in node_gene.attr_specs][3] == "int"
    assert len(node_gene.attr_specs[3].options) == 3

    genome_object = genome.DefaultGenome(
        num_inputs=2,
        num_outputs=1,
        max_nodes=8,
        max_conns=16,
        node_gene=node_gene,
        conn_gene=genome.DefaultConn(),
        mutation=genome.DefaultMutation(),
        crossover=genome.DefaultCrossover(),
        distance=genome.DefaultDistance(),
        max_in_degree=4,
    )
    state = payloads.genome_state(genome_object)

    baseline_payload = payloads.chain_payload()
    edited = payloads.chain_payload()
    edited.node_attrs[0, 3, 3] = 2.0  # activation id of node row 3: option 0 -> 2

    distance = _distance(run_kernel, genome_object, state, baseline_payload, edited)
    expected = genome_object.distance.compatibility_weight / CHAIN_NODES
    np.testing.assert_allclose(distance, [expected], rtol=0, atol=1e-6)
    assert not np.isclose(distance[0], 2 * expected, atol=1e-6)


# --------------------------------------------------------------------------- #
# degenerate and batched calls
# --------------------------------------------------------------------------- #


def test_empty_pair_distance_is_zero_not_nan(payloads, run_kernel):
    """Invariant: two empty gene families are ``0`` apart, and no pair ever yields ``NaN``.

    Oracle: the guarded ``max(cnt1, cnt2) == 0`` branch documented on
    ``gene_distance``. A payload with no connections (an isolated-node payload with
    its edge removed, or a whole payload of padding) is legal — a mutation can
    produce it — and it must not poison the speciation step with ``NaN``. The whole
    returned vector is asserted finite.

    Assertion: two fully padded payloads are exactly ``0.0``; the guard is per pair,
    so an empty parent next to a populated one still reports its non-zero distance
    in the same vector; and two *edge-free* payloads of different sizes differ by
    their node genes only (``4 / 8 = 0.5``), proving the empty connection family
    contributes ``0`` rather than ``NaN``.
    """
    genome_object = payloads.default_genome(
        max_nodes=8,
        max_conns=16,
        num_inputs=2,
        num_outputs=1,
        max_in_degree=4,
    )
    state = payloads.genome_state(genome_object)

    empty = payloads.new_payload(pop_size=2)
    zero = _distance(run_kernel, genome_object, state, empty, empty)
    assert not np.isnan(zero).any()
    np.testing.assert_array_equal(zero, np.zeros(2, np.float32))

    mixed = _distance(
        run_kernel,
        genome_object,
        state,
        _stack(payloads.new_payload(), payloads.chain_payload()),
        _stack(payloads.new_payload(), payloads.new_payload()),
    )
    assert np.isfinite(mixed).all()
    assert mixed[0] == 0.0
    assert mixed[1] > 0.0

    edge_free_small = _cleared(payloads, payloads.isolated_payload())  # 4 node genes
    edge_free_large = _cleared(payloads, payloads.chain_payload())  # 8 node genes
    only_nodes = _distance(
        run_kernel, genome_object, state, edge_free_small, edge_free_large
    )
    assert np.isfinite(only_nodes).all()
    np.testing.assert_allclose(only_nodes, [4 / CHAIN_NODES], rtol=0, atol=1e-6)


def test_batched_pairs_equal_pairwise_calls(payloads, run_kernel):
    """Invariant: one ``P``-pair call equals ``P`` single-pair calls.

    Oracle: the per-pair calls, run separately with their own payloads of
    ``pop_size=1``. The population axis is a pure batch axis for every kernel of
    this package, so a reduce that accidentally spans it (a ``max`` over all pairs
    instead of per pair, say) is visible here and nowhere else. Pairs are chosen to
    have different sizes and different edit distances, so a single shared answer
    cannot pass.

    Assertion: the batched vector equals the stacked pairwise results element by
    element (exactly — the same arrays through the same arithmetic), and the ``P``
    answers are pairwise distinct, so no reduction collapsed them.
    """
    genome_object = payloads.default_genome(
        max_nodes=8,
        max_conns=16,
        num_inputs=2,
        num_outputs=1,
        max_in_degree=4,
    )
    state = payloads.genome_state(genome_object)

    chain = payloads.chain_payload()
    extra_conn = payloads.chain_payload()
    payloads.set_conn(extra_conn, 0, CHAIN_CONNS, 0, 3)  # one non-homologous edge
    doubled_weights = payloads.chain_payload(weight=2.0)  # attribute-only edit
    skip = payloads.skip_payload()  # a different topology, 5 of 8 node genes
    empty = payloads.new_payload()

    left = _stack(chain, chain, chain, skip, empty)
    right = _stack(chain, extra_conn, doubled_weights, chain, chain)
    batched = _distance(run_kernel, genome_object, state, left, right)

    pairwise = np.concatenate(
        [
            _distance(run_kernel, genome_object, state, first, second)
            for first, second in (
                (chain, chain),
                (chain, extra_conn),
                (chain, doubled_weights),
                (skip, chain),
                (empty, chain),
            )
        ]
    )

    assert batched.shape == (5,)
    np.testing.assert_array_equal(batched, pairwise)
    assert len(set(batched.tolist())) == 5


def test_distance_is_deterministic(payloads, run_kernel):
    """Invariant: the same pair gives bit-identical distances on repeated calls.

    Oracle: two runs of the same kernel on the same arrays. Distance consumes no
    RNG, so it must be a pure function of its inputs — which is also why it is the
    one operator whose parity against the frozen implementation is exact rather
    than tolerance-based. A ``hash``/``argsort``-based alignment that leaked
    iteration order would surface as a run-to-run difference here.

    Assertion: two independent builds+runs of the same ``P``-pair batch return
    ``array_equal`` vectors, and the values are finite and non-trivial (the batch
    mixes a ``0`` pair with disjoint- and attribute-only differences).
    """
    genome_object = payloads.default_genome(
        max_nodes=8,
        max_conns=16,
        num_inputs=2,
        num_outputs=1,
        max_in_degree=4,
    )
    state = payloads.genome_state(genome_object)

    chain = payloads.chain_payload()
    extra_conn = payloads.chain_payload()
    payloads.set_conn(extra_conn, 0, CHAIN_CONNS, 0, 3)
    doubled_weights = payloads.chain_payload(weight=2.0)
    skip = payloads.skip_payload()

    left = _stack(chain, chain, skip)
    right = _stack(chain, extra_conn, doubled_weights)

    first = _distance(run_kernel, genome_object, state, left, right)
    second = _distance(run_kernel, genome_object, state, left, right)

    assert np.isfinite(first).all()
    assert first[0] == 0.0 and first[1] > 0.0 and first[2] > 0.0
    np.testing.assert_array_equal(first, second)


# --------------------------------------------------------------------------- #
# the digested route's ``[0, key_span)`` window
# --------------------------------------------------------------------------- #
def test_measure_distance_rejects_a_centre_node_key_outside_the_key_span(
    genome, common, payloads, run_kernel
):
    """Invariant: the dense digest route refuses a centre node key ``>= key_span``.

    Oracle: the documented precondition of ``BaseDistance.measure`` — the dense
    key table addresses a node gene by its key, so a present key outside
    ``[0, genome.key_span)`` is unaddressable there while the row-wise merge matches
    it exactly. ``genome.key_span`` is :data:`KEY_SPAN`, so key ``24`` is the first
    value outside the window.

    Assertion: the population's own digest is clean (both ``oor`` counts ``0``), and
    a host ``measure_distance`` with a centre whose third node row carries key
    ``24`` raises ``TensorNEATError`` whose message names the centre, that row and
    the window, and offers ``DefaultDistance(dense_single_centre=False)`` as the
    remedy. The call is made on host arrays because the guard is documented as a
    host-call guarantee only: inside a trace every leaf is a ``SymbolicTensor`` and
    the check emits no op.
    """
    dense = _span_genome(genome, dense=True)
    state = payloads.genome_state(dense)

    population = _span_payload(payloads)
    digest = run_kernel(dense.prepare_distance, state, *population.args())
    np.testing.assert_array_equal(digest.node_oor, np.zeros(3, np.int64))
    np.testing.assert_array_equal(digest.conn_oor, np.zeros(3, np.int64))

    centre = _span_payload(payloads, 1)
    payloads.set_node(centre, 0, 2, KEY_SPAN)  # 24: the first unaddressable key
    with pytest.raises(common.TensorNEATError) as caught:
        dense.measure_distance(state, *centre.args(), digest)

    message = str(caught.value)
    assert "the centre's present node gene row(s) 2" in message
    assert "outside [0, genome.key_span) = [0, 24)" in message
    assert "DefaultDistance(dense_single_centre=False)" in message


def test_measure_distance_rejects_a_centre_conn_key_outside_the_key_span(
    genome, common, payloads, run_kernel
):
    """Invariant: the dense digest route refuses a centre conn key ``< 0``.

    Oracle: the same documented precondition for the connection family. A centre
    whose node family is entirely in-span isolates the connection check, and ``-1``
    covers the lower end of the window (the node case above covers the upper end).

    Assertion: ``measure_distance`` raises ``TensorNEATError`` naming the centre's
    present *conn* gene row ``0`` with the offending value and the window
    ``[0, 24)`` — never the node-family message, because the node family of this
    centre is clean.
    """
    dense = _span_genome(genome, dense=True)
    state = payloads.genome_state(dense)
    population = _span_payload(payloads)
    digest = run_kernel(dense.prepare_distance, state, *population.args())

    centre = _span_payload(payloads, 1)
    payloads.set_conn(centre, 0, 0, -1, 2)  # negative source key of the one edge
    with pytest.raises(common.TensorNEATError) as caught:
        dense.measure_distance(state, *centre.args(), digest)

    message = str(caught.value)
    assert "the centre's present conn gene row(s) 0" in message
    assert "span -1..-1" in message
    assert "outside [0, genome.key_span) = [0, 24)" in message
    assert "present node gene row(s)" not in message


def test_measure_distance_rejects_a_population_node_key_outside_the_key_span(
    genome, common, payloads, run_kernel
):
    """Invariant: the dense route refuses a *digested* population node key out of span.

    Oracle: ``prepare_distance`` records the count of present genes outside
    ``[0, key_span)`` in ``node_oor``/``conn_oor``; a non-zero count is the dense
    route's violated precondition. Individual ``0`` carries node key ``24`` while
    the centre is clean, so the refusal cannot be the centre's.

    Assertion: the digest counts exactly ``node_oor == [1, 0, 0]`` and a clean
    ``conn_oor``, and a host ``measure_distance`` raises ``TensorNEATError`` naming
    one present node gene of the population digest, individual ``0``, the leaf
    (``node_oor``), the window and the ``dense_single_centre=False`` remedy. The
    merge route's acceptance of the same digest is pinned separately.
    """
    dense = _span_genome(genome, dense=True)
    state = payloads.genome_state(dense)

    population = _span_payload(payloads)
    payloads.set_node(population, 0, 2, KEY_SPAN)
    digest = run_kernel(dense.prepare_distance, state, *population.args())
    np.testing.assert_array_equal(digest.node_oor, np.array([1, 0, 0], np.int64))
    np.testing.assert_array_equal(digest.conn_oor, np.zeros(3, np.int64))

    centre = _span_payload(payloads, 1)
    with pytest.raises(common.TensorNEATError) as caught:
        dense.measure_distance(state, *centre.args(), digest)

    message = str(caught.value)
    assert "1 present node gene(s) of the population digest" in message
    assert "individuals 0 carry 1 such gene(s) (node_oor)" in message
    assert "outside [0, genome.key_span) = [0, 24)" in message
    assert "DefaultDistance(dense_single_centre=False)" in message


def test_measure_distance_rejects_a_population_conn_key_outside_the_key_span(
    genome, common, payloads, run_kernel
):
    """Invariant: the same refusal for a digested population *conn* key, named as such.

    Oracle: ``conn_oor`` counts present connection genes whose key leaves the
    window. Individual ``1`` carries the negative endpoint ``-1``; the node family
    of the whole population stays in-span, so the node check passes and the message
    is the connection one — a single test that could not tell the two families'
    errors apart would pass on a node-family bug.

    Assertion: ``node_oor`` is all zero, ``conn_oor == [0, 1, 0]``, and the raised
    ``TensorNEATError`` names one present conn gene, individual ``1`` and
    ``conn_oor``, with the window and the row-wise remedy stated.
    """
    dense = _span_genome(genome, dense=True)
    state = payloads.genome_state(dense)

    population = _span_payload(payloads)
    payloads.set_conn(population, 1, 0, -1, 2)
    digest = run_kernel(dense.prepare_distance, state, *population.args())
    np.testing.assert_array_equal(digest.node_oor, np.zeros(3, np.int64))
    np.testing.assert_array_equal(digest.conn_oor, np.array([0, 1, 0], np.int64))

    centre = _span_payload(payloads, 1)
    with pytest.raises(common.TensorNEATError) as caught:
        dense.measure_distance(state, *centre.args(), digest)

    message = str(caught.value)
    assert "1 present conn gene(s) of the population digest" in message
    assert "individuals 1 carry 1 such gene(s) (conn_oor)" in message
    assert "present node gene(s)" not in message
    assert "DefaultDistance(dense_single_centre=False)" in message


def test_measure_distance_merge_route_accepts_out_of_range_keys(
    genome, common, payloads, run_kernel
):
    """Invariant: ``dense_single_centre=False`` matches an out-of-range key exactly.

    Oracle: ``execute_distance`` on the same centre tiled to the population — the
    pair-wise kernel, which aligns genes by their key and therefore counts an
    out-of-range gene as the disjoint gene it is. The row-wise merge is documented
    to be bit-identical to it, while the dense table would treat the unaddressable
    key as absent (which is why the dense route refuses instead of answering).

    Assertion: for a payload whose individual ``0`` carries both a node key ``24``
    and an edge ending at ``24``, the merge route's ``measure_distance`` is
    ``tobytes()``-equal to ``execute_distance`` on the tiled centre and answers
    ``[2.6666667, 0.0, 0.5]`` — individual ``1`` is the centre itself (``0``),
    individual ``0`` is one disjoint node gene *and* one disjoint edge away
    (``1/3 + 1``), and individual ``2`` differs by one in-span edge (``0.5``), so
    the equality is not an equality of two zeros. A second payload poisons only the
    connection family (individual ``2``'s extra edge ends at ``24``) and repeats
    the bit comparison, so neither family passes by accident.
    """
    merge = _span_genome(genome, dense=False)
    state = payloads.genome_state(merge)
    centre = _span_payload(payloads, 1)

    # Individual 0: node key 24 and an edge ending at it; individual 1: the centre
    # itself; individual 2: one extra in-span edge.
    population = _span_payload(payloads)
    payloads.set_node(population, 0, 2, KEY_SPAN)
    payloads.set_conn(population, 0, 0, 0, KEY_SPAN)
    payloads.set_conn(population, 2, 1, 1, 2)
    digest = run_kernel(merge.prepare_distance, state, *population.args())
    np.testing.assert_array_equal(digest.node_oor, np.array([1, 0, 0], np.int64))
    np.testing.assert_array_equal(digest.conn_oor, np.array([1, 0, 0], np.int64))

    measured = run_kernel(
        merge.measure_distance, state, *centre.args(), digest
    )
    reference = run_kernel(
        merge.execute_distance,
        state,
        *_tiled(payloads, centre, 3).args(),
        *population.args(),
    )
    assert measured.shape == (3,)
    assert measured.dtype == np.float32
    assert np.isfinite(measured).all()
    assert measured.tobytes() == reference.tobytes()
    # Hand arithmetic of the documented formula on the poisoned pair: individual 0
    # is two non-homologous node genes out of ``max(3, 3)`` plus two non-homologous
    # edges out of ``max(1, 1)``; individual 2 is one non-homologous edge out of
    # ``max(2, 1)``; individual 1 is the centre itself.
    np.testing.assert_allclose(measured, [2 / 3 + 2.0, 0.0, 0.5], rtol=0, atol=1e-6)
    assert measured[1] == 0.0
    assert measured[0] > 0.0 and measured[2] > 0.0

    # The very same poisoned population is the dense route's refused case (the centre
    # is clean, so the refusal is the digest's): the two routes differ in contract,
    # not in the comparison they are asked to perform.
    dense = _span_genome(genome, dense=True)
    dense_state = payloads.genome_state(dense)
    dense_digest = run_kernel(dense.prepare_distance, dense_state, *population.args())
    with pytest.raises(common.TensorNEATError):
        dense.measure_distance(dense_state, *centre.args(), dense_digest)

    # Connection-family-only poison: every node key is in-span, individual 2's extra
    # edge steps outside the window.
    conn_poisoned = _span_payload(payloads)
    payloads.set_conn(conn_poisoned, 2, 1, 1, KEY_SPAN)
    conn_digest = run_kernel(merge.prepare_distance, state, *conn_poisoned.args())
    np.testing.assert_array_equal(conn_digest.node_oor, np.zeros(3, np.int64))
    np.testing.assert_array_equal(conn_digest.conn_oor, np.array([0, 0, 1], np.int64))
    conn_measured = run_kernel(
        merge.measure_distance, state, *centre.args(), conn_digest
    )
    conn_reference = run_kernel(
        merge.execute_distance,
        state,
        *_tiled(payloads, centre, 3).args(),
        *conn_poisoned.args(),
    )
    assert conn_measured.tobytes() == conn_reference.tobytes()
    # One non-homologous edge out of ``max(2, 1)`` for individual 2, nothing else.
    np.testing.assert_allclose(conn_measured, [0.0, 0.0, 0.5], rtol=0, atol=1e-6)
    assert conn_measured[0] == 0.0 and conn_measured[1] == 0.0
    assert conn_measured[2] > 0.0


def test_measure_distance_routes_agree_on_a_clean_payload(
    genome, payloads, run_kernel
):
    """Invariant: the two ``DefaultDistance`` configurations agree on the legal domain.

    Oracle: ``execute_distance`` on the same centre tiled to the population. The
    dense refusal is about *addressability* — a key the table cannot index — not
    about the metric: while every key is inside ``[0, key_span)`` the dense
    single-centre route and the row-wise merge must answer the same bits.

    Assertion: on a three-individual payload that is entirely in-span (one equal to
    the centre, one with an extra edge, one with an extra node gene) the dense and
    merge ``measure_distance`` results are ``tobytes()``-equal to each other and to
    ``execute_distance``, and the vector mixes an exact ``0`` with two non-zero
    values, so the agreement is not an agreement of zeros.
    """
    dense = _span_genome(genome, dense=True)
    merge = _span_genome(genome, dense=False)
    dense_state = payloads.genome_state(dense)
    merge_state = payloads.genome_state(merge)

    centre = _span_payload(payloads, 1)
    population = _span_payload(payloads)
    payloads.set_conn(population, 1, 1, 1, 2)  # one in-span extra edge
    payloads.set_node(population, 2, 3, 7)  # one in-span extra node gene

    dense_digest = run_kernel(dense.prepare_distance, dense_state, *population.args())
    merge_digest = run_kernel(merge.prepare_distance, merge_state, *population.args())
    for digest in (dense_digest, merge_digest):
        np.testing.assert_array_equal(digest.node_oor, np.zeros(3, np.int64))
        np.testing.assert_array_equal(digest.conn_oor, np.zeros(3, np.int64))

    via_dense = run_kernel(
        dense.measure_distance, dense_state, *centre.args(), dense_digest
    )
    via_merge = run_kernel(
        merge.measure_distance, merge_state, *centre.args(), merge_digest
    )
    reference = run_kernel(
        dense.execute_distance,
        dense_state,
        *_tiled(payloads, centre, 3).args(),
        *population.args(),
    )

    assert via_dense.shape == (3,)
    assert via_dense.tobytes() == via_merge.tobytes()
    assert via_dense.tobytes() == reference.tobytes()
    # Individual 1 adds one edge out of ``max(2, 1)``; individual 2 adds one node
    # gene out of ``max(4, 3)`` (one non-homologous gene, normalised by the larger
    # count); individual 0 is the centre itself.
    np.testing.assert_allclose(via_dense, [0.0, 0.5, 0.25], rtol=0, atol=1e-6)
    assert via_dense[0] == 0.0
    assert via_dense[1] > 0.0 and via_dense[2] > 0.0
