"""Regression tests for the two-phase compatibility distance (``prepare`` + ``measure``).

What this module proves
-----------------------
Speciation measures **one centre genome against the whole population** twice per
generation, so ``2 * species_size`` calls of ``measure_distance`` run on one
``prepare_distance`` digest.  That split is only legitimate if it answers exactly
the number ``execute_distance`` answers, per pair and bit for bit: a route that
were merely *close* would move individuals across species thresholds as the
population grows, and NEAT's search would no longer be the one the parity suites
compare against the frozen implementation.

The suite therefore pins five properties on a **real evolved payload**
(a seed-42 ``Pipeline`` XOR run, two ``step`` calls, pop 100 — so the gene rows
carry structure and attribute values the mutation operator actually produced):

* **bit identity of the three routes** — the shipped population-shaped centre,
  the batch-1 centre (which takes the dense key-table fast path in
  ``distance/dense.py``) and ``DefaultDistance(dense_single_centre=False)`` (the
  row-wise merge) all reproduce ``execute_distance`` on the same centre tiled to
  the whole population, compared with ``tobytes()``;
* **zero at identity** — self-distance is exactly ``0.0`` for every individual,
  and so is an all-padded (empty) pair, never ``NaN``;
* **determinism** — two independently traced digests of the same population and
  the distances measured from them are ``tobytes()``-equal;
* **the digest's static contract** — the per-individual counts and their stacked
  ``tail_counts`` twin line up with the population axis, and a centre whose static
  shapes disagree with the digest is refused with ``ConfigError`` instead of
  reaching a reshape or a gather;
* **an op-budget ceiling** — the number of traced IR ops a ``measure`` call adds
  on top of ``prepare`` stays under a generous cap.

Oracles used here (never the kernel itself)
-------------------------------------------
* ``genome.execute_distance`` — the pair-wise kernel of
  ``operations/distance``, the contract every other route must reproduce
  (``test/parity`` holds it against the frozen legacy implementation);
* ``DefaultDistance(dense_single_centre=False)`` — the row-wise merge route, which
  re-aligns the population per call and therefore shares no alignment code with
  the digest's pre-sorted one;
* hand arithmetic on the documented formula (``0`` at identity, ``0`` for an
  empty pair) for the axiom checks, and the digest's own field names for the
  static-shape checks.

Gotchas this module encodes
---------------------------
* **``tobytes()``, never ``np.array_equal``, for the route comparison.**  The
  distance vector itself is ``NaN``-free (asserted), but the digest's attribute
  blocks are ``NaN``-padded, and ``NaN != NaN`` would make ``array_equal`` report
  a false difference on padding rows.  ``tobytes()`` compares the bit pattern,
  which is exactly the promise ("bit-identical", not "close").
* **The dense route's documented precondition.**  ``distance/dense.py`` addresses
  a gene by its key, so a key outside ``[0, genome.key_span)`` is unmatched there
  while the merge matches it exactly.  ``prepare_distance`` reports those genes as
  ``node_oor``/``conn_oor``; the fixture's payload has both at zero (asserted), so
  the three routes are comparable *and* the assertion doubles as the canary for
  whatever allocates structural keys.
* **A digest belongs to one generation.**  ``prepare_distance`` sorts the
  population once, so reusing a digest built from another population (or another
  payload shape) would silently compare the wrong genomes; the shape contract is
  what the ``ConfigError`` cases exercise.
* **The op count is a cost budget, not a style rule.**  The numpy reference
  interpreter charges per *executed* IR op (measured ~13 µs per op, almost
  independent of element count at these shapes), so an edit that doubles
  ``measure``'s op count doubles a generation's distance cost.  The ceiling is
  deliberately loose (the measured value is well below it) so that an honest
  small change does not trip it.

Everything runs on the numpy reference backend through the shared runner; no
``jax``/``torch``, no network, no GPU.
"""

from __future__ import annotations

import numpy as np
import pytest

from ..ir_ops import count_graph_ops, count_ops

#: Population size of the evolved fixture: big enough for a real speciate pass to
#: have shuffled the payload, small enough for a module-scoped build.
POP_SIZE = 100

#: Static gene capacities of the fixture's genome, i.e. the payload widths the
#: digest is built for (``(P, 50)`` node rows, ``(P, 100, 2)`` connection rows).
MAX_NODES = 50
MAX_CONNS = 100

#: ``step`` calls that produce the fixture.  One ``step`` returns the population
#: that *produced* its fitnesses (the pre-mutation parents), so two calls are the
#: cheapest payload that has been through a mutation/crossover round.
STEPS = 2

#: Traced-op ceiling of one ``measure`` call (the marginal cost on top of a
#: ``prepare``).  Measured: 98 (119 before the connection family folded its key
#: columns with one ``dot`` and the merge dropped its successor clamp).  The cap
#: is a doubling guard, not a target.
MEASURE_OP_CEILING = 120

#: Traced-op ceiling of one ``prepare`` call, which runs once per generation.
#: Measured: 167 (161 before the per-generation ``key_fold`` constant and the
#: merge's trailing padding column were added there).  ``prepare`` runs once per
#: generation against ``2 * species_size`` ``measure`` calls, so the two ceilings
#: are not the same budget.
PREPARE_OP_CEILING = 210
#: The digest's field order, spelled out literally: the documented payload contract
#: (node family first) that ``field_names`` reports.  ``field_names`` itself is
#: ``self._fields``, so this tuple is the assertion that can fail.
DIGEST_FIELDS = (
    "node_keys",
    "node_attrs",
    "node_count",
    "node_key_cols",
    "conn_keys",
    "conn_attrs",
    "conn_count",
    "conn_key_cols",
    "node_dense",
    "node_real",
    "node_oor",
    "conn_dense",
    "conn_real",
    "conn_oor",
    "node_layout",
    "conn_layout",
    "tail_counts",
    "node_statics",
    "conn_statics",
)


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #
def _distance_genome(genome, *, dense):
    """A ``DefaultGenome`` over the fixture's static shape and gene layout.

    Args:
        genome: The ``tensorneat.genome`` package under test.
        dense (bool): ``DefaultDistance(dense_single_centre=...)`` — ``True``
            keeps the shipped single-centre fast path, ``False`` forces the
            row-wise merge everywhere.

    Returns:
        A genome whose payload shapes, gene classes and ``key_span`` match the
        evolved fixture, so its digest and the fixture's payload agree.
    """
    return genome.DefaultGenome(
        num_inputs=2,
        num_outputs=1,
        max_nodes=MAX_NODES,
        max_conns=MAX_CONNS,
        node_gene=genome.DefaultNode(),
        conn_gene=genome.DefaultConn(),
        mutation=genome.DefaultMutation(),
        crossover=genome.DefaultCrossover(),
        distance=genome.DefaultDistance(dense_single_centre=dense),
    )


def _oor_genome(genome, *, dense):
    """A small-span genome whose key window a hand-written payload can overstep.

    ``key_span`` defaults to ``max(max_nodes, num_inputs + num_outputs) + max_conns``,
    so ``max_nodes=8``/``max_conns=16`` gives ``S = 24``.  That is small enough to
    write an out-of-range key by hand (``24``, or a negative one) while every table
    still fits, so the digest's ``node_oor``/``conn_oor`` counts can be checked
    against a value the test named rather than against another kernel.
    """
    return genome.DefaultGenome(
        num_inputs=2,
        num_outputs=1,
        max_nodes=8,
        max_conns=16,
        node_gene=genome.DefaultNode(),
        conn_gene=genome.DefaultConn(),
        mutation=genome.DefaultMutation(),
        crossover=genome.DefaultCrossover(),
        distance=genome.DefaultDistance(dense_single_centre=dense),
    )


@pytest.fixture(scope="module")
def evolved(payloads):
    """A real evolved population: seed-42 ``Pipeline`` XOR, pop 100, two steps.

    Args:
        payloads: The ``support.payloads`` module (for ``Payload`` and the
            ``default_genome`` factory that fixes the gene layout).

    Returns:
        ``support.payloads.Payload`` of the population that produced the second
        generation's fitnesses — 100 individuals with evolved structure, evolved
        attribute values and ``I_INF``/``NaN`` padding rows.

    Notes:
        The pipeline is built here rather than in a helper module because the
        digest tests need a payload the mutation operator actually produced, and a
        hand-written one would not exercise the structural key allocation the
        dense route's precondition is about.  ``seed=42`` and a fixed
        ``pop_size`` make the payload reproducible, so the assertions below are
        exact rather than tolerance-based.
    """
    from tensorneat import Pipeline, algorithm, problem

    pipeline = Pipeline(
        algorithm=algorithm.NEAT(
            pop_size=POP_SIZE,
            species_size=5,
            genome=payloads.default_genome(max_nodes=MAX_NODES, max_conns=MAX_CONNS),
        ),
        problem=problem.XOR(),
        seed=42,
    )
    state = pipeline.setup()
    previous = None
    for _ in range(STEPS):
        state, previous, _fitness = pipeline.step(state)
    return payloads.Payload(*tuple(previous))


@pytest.fixture(scope="module")
def routes(genome, payloads, run_kernel, evolved):
    """The two digest routes over the evolved population, prepared once.

    Args:
        genome: The ``tensorneat.genome`` package under test.
        payloads: The ``support.payloads`` module (``genome_state``).
        run_kernel: The shared runner fixture — ``run(method, state, *args)``.
        evolved: The evolved population fixture.

    Returns:
        ``{name: (genome_object, state, digest)}`` for ``"dense"`` (the shipped
        route) and ``"merge"`` (``dense_single_centre=False``).  Each digest is a
        host pytree of numpy leaves, rebuilt from the ``DistanceDigest``
        namedtuple the kernel returns.

    Notes:
        Both digests come from the same payload, so the two routes differ only in
        how a centre is aligned against it — which is exactly the claim under
        test.  Building them at module scope keeps the suite to two traces.
    """
    built = {}
    for name, dense in (("dense", True), ("merge", False)):
        genome_object = _distance_genome(genome, dense=dense)
        state = payloads.genome_state(genome_object)
        digest = run_kernel(genome_object.prepare_distance, state, *evolved.args())
        built[name] = (genome_object, state, digest)
    return built


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _measure(run, routes, name, centre):
    """``measure_distance`` of one centre payload against the route's digest.

    Args:
        run: The ``run_kernel`` fixture.
        routes: The ``routes`` fixture.
        name (str): ``"dense"`` or ``"merge"``.
        centre: A one-individual or population-shaped ``Payload``.

    Returns:
        ``(P,) float32`` numpy distances.
    """
    genome_object, state, digest = routes[name]
    return run(genome_object.measure_distance, state, *centre.args(), digest)


def _tiled(payloads, one, count):
    """Broadcast one individual to a ``count``-row batch (``count`` copies of it).

    Args:
        payloads: The ``support.payloads`` module (for the ``Payload`` type).
        one: A ``pop_size == 1`` ``Payload``.
        count (int): Target population axis.

    Returns:
        A ``Payload`` whose every row is ``one`` — the shape ``execute_distance``
        needs to compare one centre against a whole population.

    Notes:
        A broadcast view is enough: the runner makes each array contiguous before
        it reaches ``evoxc``, and nothing downstream writes to a payload.
    """
    return payloads.Payload(
        *(np.broadcast_to(array, (count,) + array.shape[1:]) for array in one.args())
    )


# --------------------------------------------------------------------------- #
# route equivalence
# --------------------------------------------------------------------------- #
def test_routes_are_bit_identical_to_the_pair_kernel(
    payloads, routes, run_kernel, evolved
):
    """Invariant: all three routes return the same bits as ``execute_distance``.

    Oracle: ``execute_distance`` on the same centre tiled to the population — the
    pair-wise kernel ``test/parity`` holds against the frozen legacy library.  The
    centre is a single individual, which is the shape speciation actually sends,
    and two different individuals are measured so neither a padding-heavy genome
    nor an evolved one can pass on its own.

    Assertion: the dense route (batch-1 centre, key table), the merge route
    (``dense_single_centre=False``) and the pair-wise kernel agree with
    ``tobytes()`` on every row, the vector is ``(P,) float32`` and ``NaN``-free.
    ``tobytes()`` rather than ``array_equal`` because the promise is
    *bit-identical*: the two digest routes are required to reproduce the float
    summation order, not to land inside a tolerance.
    """
    genome_object, state, _digest = routes["dense"]
    for index in (0, 1):
        centre = evolved.of(index)
        tiled = _tiled(payloads, centre, POP_SIZE)
        reference = run_kernel(
            genome_object.execute_distance, state, *tiled.args(), *evolved.args()
        )
        assert reference.shape == (POP_SIZE,)
        assert reference.dtype == np.float32
        assert np.isfinite(reference).all()
        assert (
            _measure(run_kernel, routes, "dense", centre).tobytes()
            == reference.tobytes()
        )
        assert (
            _measure(run_kernel, routes, "merge", centre).tobytes()
            == reference.tobytes()
        )


def test_population_shaped_centre_agrees_on_every_route(routes, run_kernel, evolved):
    """Invariant: the shipped population-shaped centre matches both other routes.

    Oracle: ``execute_distance(pop, pop)`` — the same payload compared with
    itself, one pair per digest row.

    Assertion: ``measure_distance`` with a batch-``P`` centre is ``tobytes()``-equal
    to the pair-wise kernel on both the dense and the merge route.  This is the
    shape the algorithm sends (``DistanceRows.matrix``), and the shape in which a
    per-row pairing mistake would be invisible: a route that paired centre row
    ``0`` with every digest row would still return ``P`` numbers.
    """
    genome_object, state, _digest = routes["dense"]
    reference = run_kernel(
        genome_object.execute_distance, state, *evolved.args(), *evolved.args()
    )
    assert np.isfinite(reference).all()
    assert _measure(run_kernel, routes, "dense", evolved).tobytes() == (
        reference.tobytes()
    )
    assert _measure(run_kernel, routes, "merge", evolved).tobytes() == (
        reference.tobytes()
    )


# --------------------------------------------------------------------------- #
# the identity axiom
# --------------------------------------------------------------------------- #
def test_self_distance_is_exactly_zero(routes, payloads, run_kernel, evolved):
    """Invariant: ``d(g, g) == 0`` for every individual, and ``0`` for an empty pair.

    Oracle: the identity axiom — a non-zero self-distance would make every genome
    its own species, and a ``NaN`` would send it to one too.  The empty pair is
    the documented guard ``0 / max(1, …)``: two gene families with no present row
    at all must answer ``0.0``, not ``NaN``.

    Assertion: every entry of the population-shaped ``measure`` is exactly ``0.0``
    (on both routes), and an all-padded centre measured against an all-padded
    digest — and paired with itself in ``execute_distance`` — is exactly ``0.0``
    and finite.
    """
    for name in ("dense", "merge"):
        distances = _measure(run_kernel, routes, name, evolved)
        np.testing.assert_array_equal(distances, np.zeros(POP_SIZE, np.float32))
    empty = payloads.new_payload(1, MAX_NODES, MAX_CONNS)
    for name in ("dense", "merge"):
        genome_object, state, _digest = routes[name]
        own_digest = run_kernel(genome_object.prepare_distance, state, *empty.args())
        distances = run_kernel(
            genome_object.measure_distance, state, *empty.args(), own_digest
        )
        assert np.isfinite(distances).all()
        np.testing.assert_array_equal(distances, np.zeros(1, np.float32))
        paired = run_kernel(
            genome_object.execute_distance, state, *empty.args(), *empty.args()
        )
        np.testing.assert_array_equal(paired, np.zeros(1, np.float32))


# --------------------------------------------------------------------------- #
# determinism
# --------------------------------------------------------------------------- #
def test_two_independent_digests_measure_identically(
    genome, payloads, run_kernel, routes, evolved
):
    """Invariant: digesting and measuring twice gives the same bits.

    Oracle: a second, independently constructed genome and state — a second trace
    and a second build, so the comparison covers the kernel and not only a cached
    executable.  Nothing in this path draws randomness (no ``evoxc.random`` key is
    threaded through ``prepare``/``measure``), so any difference would be a hidden
    dependence on build order or host state.

    Assertion: every digest leaf is ``tobytes()``-equal, and the distances of two
    centres are ``tobytes()``-equal to the first route's answer.  ``tobytes()``
    again: the attribute blocks carry ``NaN`` padding, which ``array_equal`` would
    report as unequal even for identical bytes.
    """
    _genome_object, _state, digest = routes["dense"]
    second = _distance_genome(genome, dense=True)
    second_state = payloads.genome_state(second)
    again = run_kernel(second.prepare_distance, second_state, *evolved.args())
    for field in (
        "node_keys",
        "node_attrs",
        "conn_keys",
        "conn_attrs",
        "node_count",
        "tail_counts",
    ):
        assert getattr(again, field).tobytes() == getattr(digest, field).tobytes()
    for index in (0, 1):
        centre = evolved.of(index)
        first = _measure(run_kernel, routes, "dense", centre)
        repeated = run_kernel(
            second.measure_distance, second_state, *centre.args(), again
        )
        assert repeated.tobytes() == first.tobytes()


# --------------------------------------------------------------------------- #
# digest contract
# --------------------------------------------------------------------------- #
def test_digest_static_contract(routes, evolved):
    """Invariant: the digest's shapes and counts describe *this* population.

    Oracle: the payload's own static shape and the documented field contract —
    ``node_count``/``conn_count`` are ``(P,) int32`` real-gene counts,
    ``tail_counts`` is the two families' counts stacked on a leading axis (the
    fast path's ``cnt2`` pair), the packed key blocks are ``(P, M) int64``, the
    attribute blocks ``(P, M, A) float32`` in the same gene order, and
    ``node_oor``/``conn_oor`` are the per-individual out-of-range gene counts.

    Assertion: the counts and their stacked twin agree with each other and with
    the population axis, the gene and attribute axes match the payload, and both
    ``oor`` counts are ``0`` — the dense route's documented precondition, which is
    what makes the bit-identity assertions above comparable at all.
    """
    _genome_object, _state, digest = routes["dense"]
    pop_size = evolved.pop_size
    for name in ("node_count", "conn_count"):
        count = getattr(digest, name)
        assert count.shape == (pop_size,)
        assert count.dtype == np.int32
    assert tuple(digest.node_keys.shape) == (pop_size, evolved.max_nodes)
    assert digest.node_keys.dtype == np.int64
    assert tuple(digest.node_attrs.shape) == (
        pop_size,
        evolved.max_nodes,
        evolved.node_attr_width,
    )
    assert digest.node_attrs.dtype == np.float32
    assert tuple(digest.conn_keys.shape) == (pop_size, evolved.max_conns)
    assert tuple(digest.conn_attrs.shape) == (
        pop_size,
        evolved.max_conns,
        evolved.conn_attr_width,
    )
    assert digest.tail_counts.shape == (2, pop_size)
    assert digest.tail_counts.dtype == np.int32
    np.testing.assert_array_equal(
        digest.tail_counts, np.stack([digest.node_count, digest.conn_count])
    )
    # The real contract is the *literal* field order: ``field_names`` is implemented
    # as ``self._fields``, so comparing the two is tautological, while pinning the
    # tuple catches a renamed, reordered or added field.
    assert digest.field_names == DIGEST_FIELDS
    assert digest._fields == DIGEST_FIELDS
    # On the evolved fixture both counts are 0 — the dense route's documented
    # precondition, which is what makes the bit-identity assertions above comparable.
    # The non-zero contract (the exact count of keys outside ``[0, key_span)``) is
    # pinned differentially by ``test_out_of_range_keys_are_counted_per_family``.
    for name in ("node_oor", "conn_oor"):
        oor = getattr(digest, name)
        assert oor.shape == (pop_size,)
        assert oor.dtype == np.int64
        np.testing.assert_array_equal(oor, np.zeros(pop_size, np.int64))


def test_out_of_range_keys_are_counted_per_family(genome, common, payloads, run_kernel):
    """Invariant: ``node_oor``/``conn_oor`` count exactly the *present* genes outside ``key_span``.

    Oracle: hand-written keys.  ``_oor_genome`` has ``key_span == 24``, so
    individual 0 carries one out-of-range node key (``24``, the first value the dense
    table cannot address) and one out-of-range connection endpoint, individual 1
    carries a *negative* out-of-range node key and a negative connection endpoint, and
    individual 2 is clean.  Padding rows are ``I_INF`` — out of range but *absent* —
    and must never be counted.
    Assertion: the exact per-individual counts ``[1, 1, 0]`` for both families, as
    ``(P,) int64``, next to the payload's present-row counts (``[4, 4, 4]`` nodes,
    ``[2, 2, 2]`` connections) — so counting present genes instead of out-of-range
    ones, or counting every slot, fails.  Then the documented consequence: a host
    ``measure_distance`` on the dense route rejects the digest with ``TensorNEATError``
    naming the family (``node_oor`` first, ``conn_oor`` on a clean-node payload),
    while the row-wise merge (``dense_single_centre=False``) still answers, with
    self-distance exactly ``0.0`` for every individual — proof that the merge matches
    the out-of-range keys instead of dropping them like the dense table would.
    """
    dense = _oor_genome(genome, dense=True)
    merge = _oor_genome(genome, dense=False)
    assert dense.key_span == merge.key_span == 24  # the window the payload oversteps
    payload = payloads.new_payload(3, 8, 16)
    # Individual 0: in-range nodes plus key 24 (the first out-of-range value); the one
    # connection ending at that key is out of range too.
    for row, key in enumerate((0, 1, 7, 24)):
        payloads.set_node(payload, 0, row, key)
    payloads.set_conn(payload, 0, 0, 0, 7)
    payloads.set_conn(payload, 0, 1, 1, 24)
    # Individual 1: a negative node key and a negative connection endpoint.
    for row, key in enumerate((0, 1, 7, -1)):
        payloads.set_node(payload, 1, row, key)
    payloads.set_conn(payload, 1, 0, 0, 7)
    payloads.set_conn(payload, 1, 1, -1, 7)
    # Individual 2: clean, so a hard-coded count cannot pass for every row.
    for row, key in enumerate((0, 1, 2, 7)):
        payloads.set_node(payload, 2, row, key)
    payloads.set_conn(payload, 2, 0, 0, 7)
    payloads.set_conn(payload, 2, 1, 2, 7)

    state = payloads.genome_state(dense)
    digest = run_kernel(dense.prepare_distance, state, *payload.args())
    np.testing.assert_array_equal(digest.node_count, np.array([4, 4, 4], np.int32))
    np.testing.assert_array_equal(digest.conn_count, np.array([2, 2, 2], np.int32))
    for name in ("node_oor", "conn_oor"):
        oor = getattr(digest, name)
        assert oor.shape == (3,)
        assert oor.dtype == np.int64
    np.testing.assert_array_equal(digest.node_oor, np.array([1, 1, 0], np.int64))
    np.testing.assert_array_equal(digest.conn_oor, np.array([1, 1, 0], np.int64))

    # The dense route's precondition, rejected on the host with the family named.
    with pytest.raises(common.TensorNEATError) as caught:
        dense.measure_distance(state, *payload.of(0).args(), digest)
    message = str(caught.value)
    assert "node_oor" in message
    assert "[0, 24)" in message

    # A clean node family isolates the connection count's own rejection message.  The
    # payload keeps the same shapes, so the prepared executable is reused.
    conn_only = payloads.new_payload(3, 8, 16)
    for p in range(3):
        for row, key in enumerate((0, 1, 2, 7)):
            payloads.set_node(conn_only, p, row, key)
        payloads.set_conn(conn_only, p, 0, 0, 7)
    payloads.set_conn(conn_only, 0, 1, 0, 24)
    other = run_kernel(dense.prepare_distance, state, *conn_only.args())
    np.testing.assert_array_equal(other.node_oor, np.zeros(3, np.int64))
    np.testing.assert_array_equal(other.conn_oor, np.array([1, 0, 0], np.int64))
    with pytest.raises(common.TensorNEATError) as caught:
        dense.measure_distance(state, *conn_only.of(0).args(), other)
    assert "conn_oor" in str(caught.value)

    # The merge route is the documented alternative for such a payload: it matches the
    # keys exactly, so every individual is at distance 0 from itself.
    merge_state = payloads.genome_state(merge)
    merge_digest = run_kernel(merge.prepare_distance, merge_state, *payload.args())
    np.testing.assert_array_equal(merge_digest.node_oor, np.array([1, 1, 0], np.int64))
    np.testing.assert_array_equal(merge_digest.conn_oor, np.array([1, 1, 0], np.int64))
    distances = run_kernel(
        merge.measure_distance, merge_state, *payload.args(), merge_digest
    )
    assert distances.shape == (3,)
    assert distances.dtype == np.float32
    np.testing.assert_array_equal(distances, np.zeros(3, np.float32))


def test_measure_rejects_a_centre_that_disagrees_with_the_digest(
    common, payloads, run_kernel, routes
):
    """Invariant: a centre whose static shapes disagree with the digest is refused.

    Oracle: the documented validation — the digest is prepared for one payload
    shape and reused for every centre of a generation, so a mismatched centre is a
    configuration error at the call site, never a shape error inside a gather that
    silently compares the wrong rows.

    Assertion: three mismatches raise ``ConfigError`` (a ``TensorNEATError``
    subclass) with the offending field named in the message — a centre with fewer
    node rows, a centre with fewer connection rows, and a centre batch that is
    neither ``1`` nor the digest's ``P``.  The checks read ``.shape`` only, so they
    fire during the trace and propagate out of the build; the error type is
    asserted, never a bare ``Exception``.
    """
    genome_object, state, digest = routes["dense"]
    cases = (
        (payloads.new_payload(1, MAX_NODES - 1, MAX_CONNS), "node digest holds"),
        (payloads.new_payload(1, MAX_NODES, MAX_CONNS - 1), "conn digest holds"),
        (payloads.new_payload(2, MAX_NODES, MAX_CONNS), "the centre batch is 2"),
    )
    for payload, needle in cases:
        with pytest.raises(common.ConfigError) as caught:
            run_kernel(genome_object.measure_distance, state, *payload.args(), digest)
        assert needle in str(caught.value)
        assert isinstance(caught.value, common.TensorNEATError)


# --------------------------------------------------------------------------- #
# op budget
# --------------------------------------------------------------------------- #
def test_padding_rows_carry_the_sentinel_in_every_key_column(payloads, evolved):
    """Invariant: an absent gene row is ``I_INF`` in *every* key column.
    Oracle: the payload contract (and ``initialize``, which writes each absent
    gene row as ``I_INF`` across its whole key row).  ``digest._pack_centre``
    relies on it: the centre's connection family folds its key columns into one
    ``int64`` composite with a ``dot`` and derives row validity from that
    composite (``valid = composite != _PAD_CENTRE``, the composite of an
    all-``I_INF`` row) instead of reading the leading key column, which is the
    same predicate only while no absent row carries a real digit beside a padded
    one.
    Assertion: every connection row of the evolved payload whose first key column
    is ``I_INF`` is ``I_INF`` in all of its key columns, and at least one such row
    exists so the assertion is not vacuous (the node rows are rank 2, i.e. one key
    column, so the invariant is trivial there but still checked).
    """
    sentinel = np.iinfo(np.int32).max
    keys = np.asarray(evolved.conn_keys)
    absent = keys[:, :, 0] == sentinel
    assert absent.any(), "the fixture carries no padding connection row"
    assert (keys[absent] == sentinel).all(), (
        "a padding connection row carries a non-sentinel key digit, so folding the "
        "key columns would not reproduce that row's validity"
    )
    node_keys = np.asarray(evolved.node_keys)
    assert (node_keys == sentinel).any(), "the fixture carries no padding node row"


def _count_ops(block):
    """Traced IR ops of one block, nested region blocks included.

    Args:
        block: An ``evoxc`` IR block (``entry_block`` or a region body).

    Returns:
        int: Non-terminator ops of the block plus every ``op.regions`` body — the
        shared walk of ``..ir_ops``, which descends ``region.blocks`` (a region is
        an ordered list of blocks, not a block).

    Notes:
        The ``return`` terminator is not an executed op, so counting it would make
        the budget depend on how many blocks a kernel happens to have.
    """
    return count_ops(block)


def _graph_ops(graph):
    """Traced IR ops over every function of a traced ``evoxc`` graph."""
    return count_graph_ops(graph)


def test_measure_stays_under_its_op_budget(runner, routes, evolved):
    """Cost guard: one ``measure`` call adds at most ``MEASURE_OP_CEILING`` ops.

    Oracle: the traced IR itself.  The numpy reference interpreter charges per
    *executed* op (~13 µs per op at these shapes, almost independent of element
    count), so the op count is the cost model: ``measure`` runs ``2 *
    species_size`` times per generation, and an edit that doubles its op count
    doubles the generation's distance cost.  This is a budget, not a style rule —
    the ceiling sits well above the measured value so a justified small change
    does not trip it.

    Assertion: tracing ``prepare`` and ``prepare + measure`` on the evolved
    payload gives a marginal cost at or below ``MEASURE_OP_CEILING`` (measured:
    98) and a ``prepare`` cost at or below ``PREPARE_OP_CEILING`` (measured:
    167), with the marginal cost strictly positive so an accidental
    ``measure``-free graph cannot pass.
    """
    import evoxc

    genome_object, state, _digest = routes["dense"]
    specs = [runner.spec_like(array) for array in evolved.args()]
    prepare = _graph_ops(
        evoxc.trace(
            lambda *arrays: genome_object.prepare_distance(state, *arrays), *specs
        )
    )
    both = _graph_ops(
        evoxc.trace(
            lambda *arrays: genome_object.measure_distance(
                state, *arrays, genome_object.prepare_distance(state, *arrays)
            ),
            *specs,
        )
    )
    marginal = both - prepare
    assert marginal > 0, f"measure added no op on top of prepare ({both} == {prepare})"
    assert marginal <= MEASURE_OP_CEILING, (
        f"one measure call now traces {marginal} ops (prepare {prepare}, prepare + "
        f"measure {both}); the budget is {MEASURE_OP_CEILING} at ~13 µs per executed "
        "op and 2 * species_size calls per generation"
    )
    assert prepare <= PREPARE_OP_CEILING, (
        f"one prepare call now traces {prepare} ops; the budget is "
        f"{PREPARE_OP_CEILING} (it runs once per generation)"
    )
