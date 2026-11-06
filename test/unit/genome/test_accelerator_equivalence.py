"""Regression tests: the two device-gated accelerator formulations answer the host routes exactly.

``state.kernel_mode`` is a host string on the trace-time ``common.State`` that selects a second
spelling of two kernels without changing their answer.  The default ``"host"`` route is the
shipped one; ``"accelerator"`` trades ``scatter`` / ``sort`` dispatches for reductions, which is
what a compiled device needs.  Both are traced helpers chosen by one static ``if``, so a graph
contains exactly one spelling -- sound only if the spellings agree bit for bit:

* ``genome/operations/mutation/cycle.py::closes_cycle`` (the add-conn acyclicity veto).  Host: a
  dense ``(P, key_span + 1)`` key table written with ``scatter`` plus one per frontier hop;
  accelerator: an equality-template key resolution plus broadcast-reduction hops.  Covered: edges
  that close nothing, self-loops on live and absent keys, short and six-hop back edges (direction
  matters -- ``6 -> 0`` closes a cycle, ``0 -> 6`` does not), a duplicate of an existing edge,
  every documented degenerate candidate (``I_INF`` endpoint, absent key, key outside
  ``[0, key_span)``, negative key), a per-individual mixed batch at ``pop_size=8``, a ``P=2``
  payload whose individuals carry *different* graphs, and the recurrent ring.
* ``genome/operations/alignment.py::homologous_pair`` (the homologue table crossover reads).
  Host: one injective ``int64`` identity per key row, one ``argsort`` merge and one encoded
  ``sort`` rank inversion; accelerator: a ``(P, M, M)`` equality table with a leftmost-row
  ``reduce_min``.  Covered: identical, disjoint and partially overlapping parents, padded
  parent-2 rows that must never match, a fully padded (empty) pair, a batch of different
  populations, a three-column key (``OriginConn``, the ``_confirm_rows`` path) and a repeated key
  row inside one parent (the one documented divergence, asserted per leg).  ``rows`` and
  ``present`` are asserted separately and exactly.
* The routes really are different graphs: each kernel is traced both ways and the op-name
  histograms compared -- host ``closes_cycle`` carries ``scatter``, accelerator none; host
  ``homologous_pair`` carries ``sort``/``argsort``, accelerator neither.  A ``selected`` that
  silently answered the cheap answer would otherwise pass the equivalence tests.
* The switch's edges: a state with no ``kernel_mode`` entry (and ``state=None``) traces the host
  formulation (proved by the histogram, since the legs answer alike), and any other value raises
  ``TensorNEATError``.

Oracles: the other leg (asserted first, so a disagreement names the two legs); the documented
answer written as a literal (the reachability rule, the homologue record's contract), so two
agreeing but wrong legs cannot pass; the traced IR for the route proof; and, on a payload that
violates the merge's uniqueness precondition, each leg's own documented law.

Gotchas: ``kernel_mode`` is a trace-time static, so each formulation needs its own ``State``
object (the executable cache keys on the state's identity) and it is never a data argument.  Both
key functions are free functions, not bound methods, and the house ``run_kernel`` fixture only
caches a bound method -- ``_Kernel`` binds them (with ``key_span`` baked in) so each formulation
traces one executable per shape.  ``evoxc.Tensor`` has no ``__array__`` (``run_kernel`` applies
``support.runner.to_numpy``), and an ``evoxc`` region is walked through ``region.blocks`` --
``op.regions`` yields ``Region`` objects, not blocks.  An exception escaping a trace can make
pytest's traceback formatting appear to hang under ``--tb=auto``; debug with ``--tb=short``.
Everything runs on the numpy backend: no GPU, no ``iree`` extra, no ``requires_iree`` marker, no
``jax``/``torch`` import.
"""

from __future__ import annotations

import numpy as np
import pytest

from ..ir_ops import op_names

#: ``I_INF``, the payload's absent-key sentinel.
MISSING = np.iinfo(np.int32).max
#: Population of the default chain geometry (``max_nodes`` 8, inputs 2, output 1):
#: rows 0/1 inputs, 2..6 hidden, 7 output, edges 0->2, 1->2, 2->3, ..., 6->7.
CHAIN_POP = 8
#: Parent-1 key rows shared by the pair tests (unique ``(in, out)`` pairs -- the
#: merge's documented precondition).
PARENT1 = [[0, 1], [1, 2], [2, 3], [3, 4]]
#: Attribute-less padding key row used by the pair tests.
PAD2 = [MISSING, MISSING]


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _key_span(payload):
    """``DefaultGenome.key_span`` for this payload geometry (a host static).

    The documented law is ``max(max_nodes, num_inputs + num_outputs) + max_conns``
    (``genome/base.py``); every hand-built key is below ``max_nodes``, hence inside it.
    """
    return max(payload.max_nodes, 3) + payload.max_conns


def _split_graphs(payloads):
    """A ``P=2`` payload whose two individuals carry *different* graphs.

    Individual 0 is ``0 -> 2 -> 3``; individual 1 adds the back edge ``3 -> 2``, so
    candidate ``2 -> 3`` closes no cycle for row 0 and does close one for row 1 -- a
    route that read one row's edges for the whole batch answers both rows alike.
    """
    payload = payloads.new_payload(2, 4, 4)
    for individual in range(2):
        for row in range(4):
            payloads.set_node(payload, individual, row, row)
    for individual, edges in (
        (0, ((0, 2), (2, 3))),
        (1, ((0, 2), (2, 3), (3, 2))),
    ):
        for row, (src, dst) in enumerate(edges):
            payloads.set_conn(payload, individual, row, src, dst)
    return payload


def _cycle_answers(run, legs, payload, src, dst, span):
    """``{leg: (P,) bool}`` -- each formulation's answer on one candidate batch."""
    return {leg: run(state, payload, src, dst, span) for leg, state in legs.items()}


def _pair_answers(run, legs, keys1, keys2):
    """``{leg: (rows, present)}`` -- each formulation's record on one key-pair batch."""
    return {leg: run(state, keys1, keys2) for leg, state in legs.items()}


def _assert_parts(label, leg, parts):
    """Assert one leg's ``(name, array, expected)`` parts exactly.

    Oracle: the documented answer, written by hand.  Shape, dtype and every value
    are compared, so a widened dtype or a reordered row cannot pass.
    """
    for name, got, want in parts:
        got, want = np.asarray(got), np.asarray(want)
        assert (got.shape, got.dtype, got.tolist()) == (
            want.shape,
            want.dtype,
            want.tolist(),
        ), (
            f"{label}: the {leg} leg returned {name} {got.shape}/{got.dtype} "
            f"{got.tolist()}, the documented {name} is {want.shape}/{want.dtype} "
            f"{want.tolist()}"
        )


def _assert_legs_agree(label, name, host, accelerator):
    """Assert the two formulations agree exactly on one part, naming both sides.

    Oracle: the other leg.  This runs before the documented-answer check, so a
    disagreement names the two legs instead of blaming the literal.
    """
    assert np.array_equal(host, accelerator), (
        f"{label}: the accelerator leg's {name} disagrees with the host's "
        f"(host {np.asarray(host).tolist()}, accelerator "
        f"{np.asarray(accelerator).tolist()})"
    )


def _assert_scalar_answers(label, answers, expected):
    """Assert every formulation answered ``expected`` exactly, and the legs agree.

    Oracle: the documented reachability answer, written by hand.
    """
    expected = np.asarray(expected, dtype=bool)
    _assert_legs_agree(label, "answer", answers["host"], answers["accelerator"])
    for leg, value in answers.items():
        _assert_parts(label, leg, (("answer", value, expected),))


def _assert_pair_answers(label, answers, expected_rows, expected_present):
    """Assert both halves of the homologue record exactly, per leg and between legs.

    Oracle: the documented ``(rows, present)`` contract -- parent 2's row per gene
    of parent 1, redirected to ``0`` where there is no homologue, plus the mask.
    """
    rows_expected = np.asarray(expected_rows, dtype=np.int32)
    present_expected = np.asarray(expected_present, dtype=bool)
    for part, name in ((0, "rows"), (1, "mask")):
        _assert_legs_agree(
            label, name, answers["host"][part], answers["accelerator"][part]
        )
    for leg, record in answers.items():
        _assert_parts(
            label,
            leg,
            (
                ("rows", record[0], rows_expected),
                ("mask", record[1], present_expected),
            ),
        )


def _op_histogram(fn, *specs):
    """``Counter`` of the traced op names over every function of a freshly traced graph.

    The walk is the shared one in ``..ir_ops``: non-terminator ops only, nested
    region bodies reached through ``op.regions -> region.blocks``.
    """
    import evoxc

    return op_names(evoxc.trace(fn, *specs))


# --------------------------------------------------------------------------- #
# kernels, states and cached executables
# --------------------------------------------------------------------------- #
class _Kernel:
    """Bind a free kernel and its statics so ``run_kernel`` caches the executable.

    ``run_kernel`` only caches a *bound method*: its key is (function, owner
    identity, state identity, argument shapes), so one stable owner per
    (formulation, ``key_span``, shape) keeps the trace count at one graph per leg
    instead of one per assertion, and the replay then serves every later data value.
    """

    __slots__ = ("_call",)

    def __init__(self, call):
        self._call = call

    def call(self, state, *arrays):
        return self._call(state, *arrays)


@pytest.fixture(scope="module")
def cycle():
    """The cycle-predicate module (``closes_cycle`` is the mutation's entry point)."""
    from support.lazy import require

    return require("tensorneat.genome.operations.mutation.cycle")


@pytest.fixture(scope="module")
def alignment():
    """The homologue module (``homologous_pair`` is ``DefaultCrossover``'s entry point)."""
    from support.lazy import require

    return require("tensorneat.genome.operations.alignment")


@pytest.fixture(scope="module")
def legs(common):
    """The two formulation states; the switch is a plain entry on the trace-time ``State``."""
    return {
        "host": common.State(kernel_mode="host"),
        "accelerator": common.State(kernel_mode="accelerator"),
    }


@pytest.fixture(scope="module")
def run_cycle(run_kernel, cycle):
    """``run(state, payload, src, dst, key_span) -> (P,) bool``, one executable per leg+shape."""
    owners = {}

    def run(state, payload, src_keys, dst_keys, key_span):
        owner = owners.get(key_span)
        if owner is None:
            owner = owners[key_span] = _Kernel(
                lambda state, n, c, s, d: cycle.closes_cycle(state, n, c, s, d, key_span)
            )
        return run_kernel(
            owner.call,
            state,
            payload.node_keys,
            payload.conn_keys,
            np.asarray(src_keys, dtype=np.int32),
            np.asarray(dst_keys, dtype=np.int32),
        )

    return run


@pytest.fixture(scope="module")
def run_pair(run_kernel, alignment):
    """``run(state, keys1, keys2) -> (rows, present)``, one executable per leg+shape."""
    owner = _Kernel(
        lambda state, keys1, keys2: alignment.homologous_pair(keys1, keys2, state=state)
    )

    def run(state, keys1, keys2):
        return run_kernel(
            owner.call,
            state,
            np.asarray(keys1, dtype=np.int32),
            np.asarray(keys2, dtype=np.int32),
        )

    return run


# --------------------------------------------------------------------------- #
# closes_cycle -- the add-conn acyclicity veto
# --------------------------------------------------------------------------- #
def test_cycle_legs_agree_on_directed_reachability(run_cycle, legs, payloads):
    """Every documented ``closes_cycle`` answer on the chain, both legs exact.

    Oracle: ``src`` is reachable from ``dst`` on the *existing* graph.  The table pins
    direction (``6 -> 0`` is a six-hop back edge and closes a cycle; ``0 -> 6`` does
    not), self-loops on live nodes (``True``) versus an absent one (``False``), a
    duplicate of the existing ``2 -> 3`` (observed ``False`` on both legs), and every
    degenerate endpoint (``I_INF``, an absent key inside the span, a key outside
    ``[0, key_span)``, a negative key), which answer ``False`` anyway.
    """
    payload = payloads.chain_payload(pop_size=CHAIN_POP)
    span = _key_span(payload)
    cases = (
        ("forward edge closing nothing", 0, 7, False),
        ("edge across two levels", 0, 3, False),
        ("edge from a hidden row forward", 2, 6, False),
        ("six-hop back edge", 6, 0, True),
        ("short back edge", 5, 2, True),
        ("back edge from the output", 7, 2, True),
        ("three-hop back edge", 3, 0, True),
        ("self-loop on an input", 0, 0, True),
        ("self-loop on a hidden node", 3, 3, True),
        ("self-loop on the output", 7, 7, True),
        ("self-loop on an absent key", 99, 99, False),
        ("duplicate of the existing 2->3", 2, 3, False),
        ("I_INF source", MISSING, 7, False),
        ("I_INF destination", 3, MISSING, False),
        ("both endpoints I_INF", MISSING, MISSING, False),
        ("absent key inside the span", 20, 7, False),
        ("key above key_span", span + 5, 7, False),
        ("negative key", -1, 7, False),
    )
    for label, src, dst, expected in cases:
        answers = _cycle_answers(
            run_cycle, legs, payload, [src] * CHAIN_POP, [dst] * CHAIN_POP, span
        )
        _assert_scalar_answers(
            f"chain {label} ({src}->{dst})", answers, [expected] * CHAIN_POP
        )


def test_cycle_legs_agree_on_the_recurrent_ring(run_cycle, legs, payloads):
    """The already-cyclic ring topology answers identically on both legs.

    The ring holds ``2 -> 3`` and ``3 -> 2`` plus the input/output spokes, so the
    predicate reads reachability off a cyclic graph instead of refusing it: ``3 -> 2``
    (a duplicate on the cycle) and ``7 -> 2`` (a back edge from the output) close a
    cycle, ``0 -> 7`` does not (the output is a sink), ``7 -> 7`` does (7 is live).
    """
    payload = payloads.ring_payload(pop_size=4)
    src = np.array([3, 0, 7, 7], dtype=np.int32)
    dst = np.array([2, 7, 7, 2], dtype=np.int32)
    answers = _cycle_answers(run_cycle, legs, payload, src, dst, _key_span(payload))
    _assert_scalar_answers("ring candidates", answers, [True, False, True, True])


def test_cycle_legs_agree_when_each_individual_gets_a_different_candidate(
    run_cycle, legs, payloads
):
    """A ``pop_size=8`` batch of mixed candidates gets one answer per population row.

    The eight individuals share one chain graph but ask eight different questions
    (no-cycle, self-loop, back edge, duplicate, both ``I_INF`` spellings, an absent
    key and a six-hop back edge), so a route that broadcast one row's candidate over
    the batch, or read the wrong row's answer, cannot produce the vector.
    """
    payload = payloads.chain_payload(pop_size=CHAIN_POP)
    src = np.array([0, 3, 5, 2, MISSING, 0, 20, 6], dtype=np.int32)
    dst = np.array([7, 3, 2, 3, 7, MISSING, 3, 2], dtype=np.int32)
    answers = _cycle_answers(run_cycle, legs, payload, src, dst, _key_span(payload))
    _assert_scalar_answers(
        "mixed per-individual candidates",
        answers,
        [False, True, True, False, False, False, False, True],
    )


def test_cycle_legs_agree_when_the_two_individuals_carry_different_graphs(
    run_cycle, legs, payloads
):
    """The same candidate answers differently per row when the graphs differ per row.

    Individual 0 is ``0 -> 2 -> 3``, individual 1 adds ``3 -> 2``, so candidate
    ``2 -> 3`` closes no cycle for row 0 and does close one for row 1.
    """
    payload = _split_graphs(payloads)
    span = _key_span(payload)
    answers = _cycle_answers(run_cycle, legs, payload, [2, 2], [3, 3], span)
    _assert_scalar_answers("mixed graphs, candidate 2->3", answers, [False, True])


# --------------------------------------------------------------------------- #
# homologous_pair -- the homologue table crossover reads
# --------------------------------------------------------------------------- #
def test_homologue_legs_agree_on_the_pair_matrix(run_pair, legs):
    """The whole contract-conforming pair matrix, exact per leg and between legs.

    Oracle: the documented ``(rows, present)`` record -- parent 2's row per gene of
    parent 1, redirected to ``0`` where there is none, plus the mask.  Padding never
    matches anything, and each population row is paired independently.
    """
    cases = (
        # label, parent 1, parent 2, expected rows, expected present
        ("identical", [PARENT1], [PARENT1], [[0, 1, 2, 3]], [[True] * 4]),
        ("disjoint", [PARENT1], [[[10, 11], [11, 12], [12, 13], [13, 14]]],
         [[0, 0, 0, 0]], [[False] * 4]),
        ("partial overlap", [PARENT1], [[[2, 3], [3, 4], [7, 8], [8, 9]]],
         [[0, 0, 0, 1]], [[False, False, True, True]]),
        ("padded parent-2 rows", [PARENT1], [[[0, 1], PAD2, PAD2, [3, 4]]],
         [[0, 0, 0, 3]], [[True, False, False, True]]),
        ("all-padded pair", [[PAD2] * 4], [[PAD2] * 4],
         [[0, 0, 0, 0]], [[False] * 4]),
        ("padded parent 1 vs real parent 2", [[PAD2] * 4], [PARENT1],
         [[0, 0, 0, 0]], [[False] * 4]),
        ("three populations in one batch",
         [PARENT1, [[5, 6], [6, 7], [7, 8], [8, 9]], [[0, 1], [9, 9], [2, 3], PAD2]],
         [PARENT1, PARENT1, PARENT1],
         [[0, 1, 2, 3], [0, 0, 0, 0], [0, 0, 2, 0]],
         [[True, True, True, True], [False] * 4, [True, False, True, False]]),
    )
    for label, keys1, keys2, rows, present in cases:
        answers = _pair_answers(
            run_pair, legs, np.array(keys1, np.int32), np.array(keys2, np.int32)
        )
        _assert_pair_answers(f"pair matrix: {label}", answers, rows, present)


def test_homologue_legs_agree_on_a_three_column_key(run_pair, legs):
    """A ``(P, M, 3)`` key (``OriginConn``) exercises the ``_confirm_rows`` width path.

    The packed ``int64`` identity covers only the first two columns, so a prefix match
    whose third column differs must be rejected *by both legs*; a full-tuple match must
    still pair.
    """
    pad = [MISSING, MISSING, MISSING]
    keys1 = np.array([[[1, 2, 5], [3, 4, 6], pad, pad]], dtype=np.int32)
    prefix_only = np.array([[[1, 2, 9], [3, 4, 6], pad, pad]], dtype=np.int32)
    _assert_pair_answers(
        "three-column keys, prefix-only match",
        _pair_answers(run_pair, legs, keys1, prefix_only),
        [[0, 1, 0, 0]], [[False, True, False, False]],
    )
    exact = np.array([[[1, 2, 5], [3, 4, 6], [7, 8, 9], pad]], dtype=np.int32)
    _assert_pair_answers(
        "three-column keys, exact match",
        _pair_answers(run_pair, legs, keys1, exact),
        [[0, 1, 0, 0]], [[True, True, False, False]],
    )


def test_homologue_duplicate_rows_follow_each_legs_documented_law(run_pair, legs):
    """A repeated key row inside one parent is the one documented divergence.

    The merge's precondition is *the first two key columns are unique within one
    parent*; a payload that repeats them leaves the group's order to an unstable
    ``argsort``, and ``alignment_accelerator`` documents that the merge then reads a
    tie it cannot reproduce, while the accelerator deterministically marks every
    duplicate and picks the leftmost matching row.  So this asserts each leg's own
    documented law instead of equality: the accelerator marks parent-1 rows 0 and 1
    (both ``(2, 3)``) present, both pointing at parent 2's leftmost ``(2, 3)`` row;
    the host marks exactly one of the two indistinguishable duplicates (the group's
    last row) and points it at one of parent 2's two ``(2, 3)`` rows.
    """
    keys1 = np.array([[[2, 3], [2, 3], [7, 8], PAD2]], dtype=np.int32)
    keys2 = np.array([[[0, 1], [1, 2], [2, 3], [2, 3]]], dtype=np.int32)
    answers = _pair_answers(run_pair, legs, keys1, keys2)
    acc_rows, acc_present = (np.asarray(part) for part in answers["accelerator"])
    host_rows, host_present = (np.asarray(part) for part in answers["host"])

    assert np.array_equal(acc_present, [[True, True, False, False]]), (
        "the accelerator leg must mark every parent-1 duplicate of a present key: "
        f"got {acc_present.tolist()}"
    )
    assert np.array_equal(acc_rows, [[2, 2, 0, 0]]), (
        "the accelerator leg must point every duplicate at parent 2's leftmost "
        f"matching row (2), got {acc_rows.tolist()}"
    )
    assert int(host_present.sum()) == 1, (
        "the host merge's successor test marks exactly one of two indistinguishable "
        f"parent-1 duplicates (the group's last row), got {host_present.tolist()}"
    )
    winner = int(np.argmax(host_present[0]))
    assert winner in (0, 1), (
        f"the host merge marked row {winner}, not one of the duplicated rows"
    )
    assert int(host_rows[0, winner]) in (2, 3), (
        "the host merge's homologue row must be one of parent 2's two (2, 3) rows, "
        f"got {int(host_rows[0, winner])}"
    )


# --------------------------------------------------------------------------- #
# the routes are different graphs, and the default really is the host one
# --------------------------------------------------------------------------- #
def _assert_route(label, host, accelerator, default, host_only=(), accel_only=()):
    """Assert the two traces are two routes, and that a bare state traces the host one.

    Oracle: the traced IR's op-name histogram plus the documented default
    ``kernel_mode="host"``.  An equality of *answers* cannot prove the default, since
    the two legs agree by construction.
    """
    host, accelerator, default = dict(host), dict(accelerator), dict(default)
    for op in host_only:
        assert host[op] >= 1, f"{label}: the host route no longer traces {op!r} ({host})"
        assert accelerator.get(op, 0) == 0, (
            f"{label}: the accelerator route traces {op!r}, the dispatch it exists "
            f"to remove ({accelerator})"
        )
    for op in accel_only:
        assert accelerator.get(op, 0) >= 1, (
            f"{label}: the accelerator route lost its {op!r} reduction ({accelerator})"
        )
    assert host != accelerator, (
        f"{label}: both formulations traced the same op histogram ({host}), so this "
        "module is not comparing two routes"
    )
    assert default == host, (
        f"{label}: a state without a kernel_mode entry must trace the host "
        f"formulation; got {default} against {host}"
    )


def test_cycle_formulations_trace_different_graphs(cycle, runner, payloads, legs):
    """The host ``closes_cycle`` graph scatters, the accelerator graph does not.

    Measured at ``pop_size=8`` on the chain geometry: host 78 ops with 2 ``scatter``
    ops, accelerator 78 ops with 0.
    """
    payload = payloads.chain_payload(pop_size=CHAIN_POP)
    span = _key_span(payload)
    spec = (
        runner.spec_like(payload.node_keys),
        runner.spec_like(payload.conn_keys),
        runner.spec_like(np.zeros(CHAIN_POP, dtype=np.int32)),
        runner.spec_like(np.zeros(CHAIN_POP, dtype=np.int32)),
    )

    def traced(state):
        return lambda node_keys, conn_keys, src, dst: cycle.closes_cycle(
            state, node_keys, conn_keys, src, dst, span
        )

    _assert_route(
        "closes_cycle",
        _op_histogram(traced(legs["host"]), *spec),
        _op_histogram(traced(legs["accelerator"]), *spec),
        _op_histogram(traced(None), *spec),
        host_only=("scatter",),
    )


def test_homologue_formulations_trace_different_graphs(alignment, runner, legs):
    """The host homologue graph sorts, the accelerator graph does not.

    Measured at ``(P, M, K) = (1, 4, 2)``: host 59 ops with one ``argsort`` and one
    ``sort`` (the rank inversion), accelerator 42 ops with neither and one
    ``reduce_min`` (the leftmost-match reduction).
    """
    keys = np.array([PARENT1], dtype=np.int32)
    spec = (runner.spec_like(keys), runner.spec_like(keys))

    def traced(state):
        return lambda keys1, keys2: alignment.homologous_pair(keys1, keys2, state=state)

    _assert_route(
        "homologous_pair",
        _op_histogram(traced(legs["host"]), *spec),
        _op_histogram(traced(legs["accelerator"]), *spec),
        _op_histogram(traced(None), *spec),
        host_only=("argsort", "sort"),
        accel_only=("reduce_min",),
    )


def test_an_unknown_kernel_mode_is_refused_by_both_kernels(
    cycle, alignment, common, runner, payloads
):
    """A misspelled formulation fails loudly instead of silently selecting a route.

    Oracle: the documented ``Raises: TensorNEATError`` of both entry points; the
    kernel-mode resolver reads the entry at trace time, before any backend sees it.
    """
    unknown = common.State(kernel_mode="gpu")
    payload = payloads.chain_payload(pop_size=2)
    zeros = np.zeros(2, dtype=np.int32)

    def closes(node_keys, conn_keys, src, dst):
        return cycle.closes_cycle(
            unknown, node_keys, conn_keys, src, dst, _key_span(payload)
        )

    with pytest.raises(common.TensorNEATError) as cycle_error:
        runner.run_defn(closes, payload.node_keys, payload.conn_keys, zeros, zeros)
    assert "kernel_mode" in str(cycle_error.value), (
        "the cycle predicate did not name the offending state entry: "
        f"{cycle_error.value}"
    )

    keys = np.zeros((2, 4, 2), dtype=np.int32)

    def pairs(keys1, keys2):
        return alignment.homologous_pair(keys1, keys2, state=unknown)

    with pytest.raises(common.TensorNEATError) as pair_error:
        runner.run_defn(pairs, keys, keys)
    assert "kernel_mode" in str(pair_error.value), (
        "the homologue table did not name the offending state entry: "
        f"{pair_error.value}"
    )
