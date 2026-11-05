"""Unit tests for structural and value mutation — invariants only.

What this module proves
-----------------------
Mutation is stochastic and batch-wide, so the tests assert the *invariants* an operator
must keep rather than a particular outcome (the exact draws differ from the legacy
library by construction — ``evoxc.random`` vs ``jax.random`` — and are covered by parity
only where they are deterministic):

* capacity is hard: ``max_nodes``, ``max_conns`` and the genome's ``max_in_degree`` are
  never exceeded, and a refused add is a *no-op for that individual* — never a silent
  repair, never a raised error;
* the four structural probabilities are **independent** per-individual Bernoulli gates
  (add-node ``0.1``, delete-node ``0.1``, add-conn ``0.2``, delete-conn ``0.2``), so one
  individual may both gain and lose a node in one generation — but no operation fires
  twice, and the stages compose sequentially (add-node → delete-node → add-conn →
  delete-conn), each planning against the payload the previous stage left, so a refused
  stage is a bit-exact no-op; value mutation always runs afterwards on every attribute
  row;
* keys stay unique and ``I_INF``-padded; structural keys are **caller-allocated** and
  honoured verbatim (``new_node_keys (P,)``, ``new_conn_keys (P, 3)``), so
  ``(seed, config, shape)`` — not an internal counter — decides the innovation numbers;
* rows the operator did not touch are bit-identical (no accidental rewrite of an
  attribute or a key column), and a probability of ``0`` removes the operation entirely.

The *magnitude* of a value mutation and its replace-vs-perturb branch are pinned
separately in ``test_mutation_weight.py``: this module only relies on value mutation as
the stage that must leave the *key* columns alone, so the scale and branch laws live
next to the payload builder that makes them exact.

Oracles and gotchas
-------------------
Every assertion is counting or set algebra on the returned payload — node/connection
counts, ``(in_key, out_key)`` membership, ``len(set(keys))``, ``isnan`` on attribute rows,
``== I_INF`` on key rows — over hand-built ``support.payloads`` topologies whose edge sets
are known exactly, plus a Python reachability walk for "is the graph still acyclic?" that
never consults the ``closes_cycle`` predicate under test.
``validate_schedule``-style reasoning does not apply: a mutation may leave a genome
temporarily inconsistent *as a network* (a node with no edges), which is legal and which
the forward pass handles through ``NaN``, so the invariants are the payload's, not the
topology's. Value mutation is unconditional — the four probabilities gate only
``mutate_structure`` — so a test that wants the *whole* kernel to be the identity also
builds the node/conn genes with every ``*_mutate_rate``/``*_replace_rate`` at ``0``
(:func:`_genome` does that by default).

Backend is ``"numpy"`` throughout, through the shared runner only; no ``jax``/``torch``,
no network, no GPU.
"""

from __future__ import annotations

import numpy as np
import pytest

#: Input width of every genome built here (rows ``0.._NUM_INPUTS-1``).
_NUM_INPUTS = 2
#: Output width of every genome built here (the last ``_NUM_OUTPUTS`` rows).
_NUM_OUTPUTS = 1
#: Node capacity ``N`` of every payload built here.
_MAX_NODES = 8
#: Connection capacity used unless a test needs a tighter one.
_MAX_CONNS = 16
#: Static key bound ``S``: the caller-allocated keys below (20..) must stay inside it.
_KEY_SPAN = 64
#: Output row of the hand-built payloads (``_MAX_NODES - _NUM_OUTPUTS``).
_OUTPUT_ROW = _MAX_NODES - _NUM_OUTPUTS


def _quiet_node(genome):
    """``DefaultNode`` whose attribute mutation is the identity (every rate is ``0``)."""
    return genome.DefaultNode(
        bias_mutate_rate=0.0,
        bias_replace_rate=0.0,
        response_mutate_rate=0.0,
        response_replace_rate=0.0,
        aggregation_replace_rate=0.0,
        activation_replace_rate=0.0,
    )


def _quiet_conn(genome, gene=None):
    """``DefaultConn`` (or ``gene``) whose weight mutation is the identity."""
    gene = genome.DefaultConn if gene is None else gene
    return gene(weight_mutate_rate=0.0, weight_replace_rate=0.0)


def _genome(
    genome, mutation, *, cls=None, node_gene=None, conn_gene=None, max_conns=_MAX_CONNS, **kwargs
):
    """Build a genome of the tested family around one mutation operator.

    The node/conn genes mutate values only if a test passes a gene explicitly, so "the
    operator touched only this row" is a bitwise statement.
    """
    builder = genome.DefaultGenome if cls is None else cls
    return builder(
        num_inputs=_NUM_INPUTS,
        num_outputs=_NUM_OUTPUTS,
        max_nodes=_MAX_NODES,
        max_conns=max_conns,
        node_gene=_quiet_node(genome) if node_gene is None else node_gene,
        conn_gene=_quiet_conn(genome) if conn_gene is None else conn_gene,
        mutation=mutation,
        crossover=genome.DefaultCrossover(),
        distance=genome.DefaultDistance(),
        key_span=_KEY_SPAN,
        **kwargs,
    )


def _keys(seed: int, count: int):
    """``count`` distinct rank-0 ``evoxc`` keys from the per-test ``seed``; a small fixed
    loop over them keeps a failure reproducible and the run cheap."""
    import evoxc

    return [evoxc.random.key(int(seed) + offset) for offset in range(count)]


def _node_slots(pop_size: int, first: int = 20) -> np.ndarray:
    """``(P,) int32`` caller-allocated node keys, above every hand-built key."""
    return np.arange(first, first + pop_size, dtype=np.int32)


def _conn_slots(pop_size: int, first: int = 40) -> np.ndarray:
    """``(P, 3) int32`` caller-allocated connection keys (identity/copy/add columns)."""
    return np.tile(np.array([first, first + 1, first + 2], dtype=np.int32), (pop_size, 1))


def _small_payload(payloads, pop_size=2, max_conns=_MAX_CONNS, conn_key_width=2, extra_edges=()):
    """``0->2`` and ``2->7`` between two inputs, one hidden row and one output row.

    Rows ``0``, ``1``, ``2`` and ``7`` are present; ``conn_key_width`` is ``2`` for
    ``BaseConn`` and ``3`` for ``OriginConn``, ``extra_edges`` adds more edges.
    """
    payload = payloads.new_payload(pop_size, _MAX_NODES, max_conns, conn_key_width=conn_key_width)
    for p in range(pop_size):
        for row in (0, 1, 2, _OUTPUT_ROW):
            payloads.set_node(payload, p, row, row)
        for row, (source, target) in enumerate([(0, 2), (2, _OUTPUT_ROW), *extra_edges]):
            payloads.set_conn(payload, p, row, source, target)
    return payload


def _one_edge_payload(payloads, pop_size=2, max_conns=4, weight=2.0, conn_key_width=2):
    """Rows ``0, 1, 2, 7`` present with exactly one edge ``2->7`` of ``weight``.

    ``conn_key_width`` must agree with the genome's conn gene (``2`` for
    ``DefaultConn``, ``3`` for ``OriginConn``) — the payload layout is a property
    of the gene, and a mismatch is a test bug, not a kernel behaviour.
    """
    payload = payloads.new_payload(pop_size, _MAX_NODES, max_conns, conn_key_width=conn_key_width)
    for p in range(pop_size):
        for row in (0, 1, 2, _OUTPUT_ROW):
            payloads.set_node(payload, p, row, row)
        payloads.set_conn(payload, p, 0, 2, _OUTPUT_ROW, [weight])
    return payload


def _dense_payload(payloads, pop_size=3, conn_key_width=2):
    """Every legal pair of rows ``{0, 1} -> {2, 7}`` plus ``2->7``, so adds collide."""
    return _small_payload(
        payloads,
        pop_size=pop_size,
        conn_key_width=conn_key_width,
        extra_edges=((0, _OUTPUT_ROW), (1, 2), (1, _OUTPUT_ROW)),
    )


def _mutate(
    run_kernel,
    genome_object,
    state,
    payload,
    key,
    pop=None,
    *,
    new_node_keys=None,
    new_conn_keys=None,
):
    """Run one ``execute_mutation`` call and return the four host arrays in order.

    The caller-allocated keys default to the fixed ``(P, 3)`` blocks above; a test
    that cares about them passes its own.
    """
    pop = payload.pop_size if pop is None else pop
    node_keys = _node_slots(pop) if new_node_keys is None else new_node_keys
    conn_keys = _conn_slots(pop) if new_conn_keys is None else new_conn_keys
    return run_kernel(
        genome_object.execute_mutation, state, key, *payload.args(), node_keys, conn_keys
    )


def _nodes(node_keys_row, sentinel) -> set:
    """Node keys present in one individual's ``(N,)`` key row."""
    return {int(key) for key in node_keys_row if int(key) != sentinel}


def _pairs(conn_keys_row, sentinel) -> list:
    """Every present ``(in_key, out_key, *extra)`` key row of one individual, in row order."""
    return [tuple(int(c) for c in row) for row in conn_keys_row if int(row[0]) != sentinel]


def _edges(conn_keys_row, sentinel) -> set:
    """The set of present ``(in_key, out_key)`` edges of one individual."""
    return {(source, target) for source, target, *_ in _pairs(conn_keys_row, sentinel)}


def _edges_with_attrs(conn_keys_row, conn_attrs_row, sentinel) -> dict:
    """Map every present ``(in_key, out_key)`` edge of one individual to its attribute row."""
    return {
        (int(key_row[0]), int(key_row[1])): conn_attrs_row[row]
        for row, key_row in enumerate(conn_keys_row)
        if int(key_row[0]) != sentinel
    }


def _indegree(edges) -> dict:
    """``{destination key: incoming edge count}`` for a set of ``(source, target)`` pairs."""
    counts: dict = {}
    for _, target in edges:
        counts[target] = counts.get(target, 0) + 1
    return counts


def _has_cycle(edges) -> bool:
    """True when the directed graph of ``(source, target)`` pairs contains a cycle.

    A plain Python colouring DFS — deliberately not ``closes_cycle``, which is
    part of the code under test.
    """
    adjacency: dict = {}
    for source, target in edges:
        adjacency.setdefault(source, set()).add(target)
        adjacency.setdefault(target, set())
    colour: dict = {}

    def visit(node) -> bool:
        colour[node] = 1
        for neighbour in adjacency[node]:
            if colour.get(neighbour) == 1:
                return True
            if colour.get(neighbour) is None and visit(neighbour):
                return True
        colour[node] = 2
        return False

    return any(colour.get(node) is None and visit(node) for node in adjacency)


def _changed_rows(before, after) -> list:
    """Indices of the rows of a ``(M, ...)`` payload slice that differ byte-wise."""
    return [
        row
        for row in range(before.shape[0])
        if before[row].tobytes() != after[row].tobytes()
    ]


def _assert_padding_is_paired(
    node_keys_row, node_attrs_row, conn_keys_row, conn_attrs_row, sentinel
):
    """Assert the "absent row" sentinels agree across the key and attribute arrays."""
    node_present = node_keys_row != sentinel
    assert (np.isnan(node_attrs_row).all(axis=-1) == ~node_present).all()
    conn_present = conn_keys_row[:, 0] != sentinel
    assert (np.isnan(conn_attrs_row).all(axis=-1) == ~conn_present).all()
    # An absent connection row carries the sentinel in *every* key column, so no
    # half-cleared row can alias a real edge (I_INF in column 0 but not column 1).
    assert (conn_keys_row[~conn_present] == sentinel).all()


def test_probabilities_outside_the_unit_interval_are_rejected(genome, common):
    """Invariant: a probability outside ``[0, 1]`` raises ``TensorNEATError`` at construction.

    Oracle: the documented configuration contract — validated in
    ``DefaultMutation.__init__`` so a bad configuration fails before the first
    ``evoxc.build``, not on the first generation. Each of the four knobs
    (``conn_add``, ``conn_delete``, ``node_add``, ``node_delete``) is exercised
    with ``-0.1`` and ``1.1``, and the error type is asserted (a bare
    ``ValueError`` escapes the package's hierarchy). The closed interval itself is
    accepted, and the documented gate order
    ``(node_add, node_delete, conn_add, conn_delete)`` is pinned.
    """
    knobs = ("conn_add", "conn_delete", "node_add", "node_delete")
    for knob in knobs:
        for value in (-0.1, 1.1):
            with pytest.raises(common.TensorNEATError):
                genome.DefaultMutation(**{knob: value})

    for value in (0.0, 1.0):
        mutation = genome.DefaultMutation(**{knob: value for knob in knobs})
        assert mutation.structural_probabilities == (value,) * 4

    # The gate draw's row order, from the constructor defaults.
    assert genome.DefaultMutation().structural_probabilities == (0.1, 0.1, 0.2, 0.2)


def test_zero_probabilities_are_the_identity(genome, payloads, run_kernel, evoxc_key):
    """Invariant: with all four probabilities ``0`` the payload is returned unchanged.

    Oracle: bit equality against the input payload (all four arrays, ``-0.0`` and
    ``NaN`` included). A probability of ``0`` is documented to remove its operation
    from the graph, so even a *legal* structural change is a failure here; the check
    also catches an operator that resets padding or re-sorts rows as a side effect
    of running. The value-mutation rates are zeroed as well (:func:`_genome`),
    because the four probabilities gate only the structural stage and the whole
    kernel is what is being pinned here.
    """
    mutation = genome.DefaultMutation(conn_add=0.0, conn_delete=0.0, node_add=0.0, node_delete=0.0)
    genome_object = _genome(genome, mutation)
    state = payloads.genome_state(genome_object)
    payload = payloads.chain_payload(pop_size=2)

    mutated = _mutate(run_kernel, genome_object, state, payload, evoxc_key)

    for returned, original in zip(mutated, payload.args()):
        assert returned.shape == original.shape
        assert returned.dtype == original.dtype
        assert returned.tobytes() == original.tobytes()


def test_shapes_and_dtypes_are_preserved(genome, payloads, run_kernel, evoxc_key):
    """Invariant: mutation never resizes the payload.

    Oracle: shape and dtype equality with the input for all four arrays —
    ``node_keys (P, N) int32``, ``node_attrs (P, N, NA) float32``,
    ``conn_keys (P, E, CK) int32``, ``conn_attrs (P, E, CA) float32``. Shapes are static
    at build time, so an operator that grew an array would force a rebuild (and break the
    pipeline's per-shape executable cache), and dtypes are part of the payload contract.
    Only add-node is open, so the check runs on a payload that really changed: in the
    full four-stage configuration a later stage may undo an earlier one, which would make
    the non-vacuity check fail for a correct kernel.
    """
    mutation = genome.DefaultMutation(conn_add=0.0, conn_delete=0.0, node_add=1.0, node_delete=0.0)
    genome_object = _genome(genome, mutation)
    state = payloads.genome_state(genome_object)
    payload = _small_payload(payloads, pop_size=3)

    mutated = _mutate(run_kernel, genome_object, state, payload, evoxc_key)

    dtypes = (np.int32, np.float32, np.int32, np.float32)
    for returned, original, dtype in zip(mutated, payload.args(), dtypes):
        assert returned.shape == original.shape
        assert returned.dtype == dtype

    # Non-vacuous: the open add-node gate really did restructure the payload.
    assert not np.array_equal(mutated[0], payload.node_keys)


def test_node_capacity_is_never_exceeded(genome, payloads, run_kernel, seed):
    """Invariant: ``node_add=1.0`` repeated over many generations never exceeds ``max_nodes``.

    Oracle: a row count per individual — the number of ``node_keys != I_INF``
    entries — must stay ``<= max_nodes``, and once the family is full (a payload
    whose only free rows are gone) the operator must leave every individual
    unchanged instead of writing into a padded row. Several generations from the
    same payload are run so the "full" case is reached deterministically.
    """
    mutation = genome.DefaultMutation(conn_add=0.0, conn_delete=0.0, node_add=1.0, node_delete=0.0)
    genome_object = _genome(genome, mutation)
    state = payloads.genome_state(genome_object)
    sentinel = payloads.I_INF
    current = _small_payload(payloads, pop_size=2)
    refused = 0

    for generation, key in enumerate(_keys(seed, 8)):
        # Fresh keys, still unique per individual: a repeat would alias a gene.
        mutated = _mutate(
            run_kernel, genome_object, state, current, key,
            new_node_keys=_node_slots(2) + generation,
        )
        before = (current.node_keys != sentinel).sum(axis=1)
        after = (mutated[0] != sentinel).sum(axis=1)
        assert (after <= _MAX_NODES).all()
        assert (after >= before).all()  # an add never removes a row

        full = before == _MAX_NODES
        for p in range(2):
            if full[p]:
                # A full family is a no-op for that individual, bit for bit.
                for returned, original in zip(mutated, current.args()):
                    assert returned[p].tobytes() == original[p].tobytes()
                refused += 1
        current = payloads.Payload(*mutated)

    assert (current.node_keys != sentinel).sum() == 2 * _MAX_NODES
    assert refused > 0  # the "family is full" path was actually exercised


def test_connection_capacity_is_never_exceeded(genome, payloads, run_kernel, seed):
    """Invariant: ``conn_add=1.0`` repeated over many generations never exceeds ``max_conns``.

    Oracle: the connection row count per individual (``conn_keys[:, :, 0] !=
    I_INF``) stays ``<= max_conns``; a full connection family refuses the add
    (bit-identical payload). ``node_add`` needs two free connection rows at once,
    so the same assertion is run with ``node_add=1.0`` on a payload with exactly
    one free row: the split must be refused rather than half-applied (node without
    its two edges).
    """
    sentinel = payloads.I_INF
    capacity = 4
    mutation = genome.DefaultMutation(conn_add=1.0, conn_delete=0.0, node_add=0.0, node_delete=0.0)
    genome_object = _genome(genome, mutation, max_conns=capacity)
    state = payloads.genome_state(genome_object)
    current = _small_payload(payloads, pop_size=2, max_conns=capacity)
    refused = 0
    highest = 0

    for generation, key in enumerate(_keys(seed, 8)):
        mutated = _mutate(
            run_kernel, genome_object, state, current, key,
            new_node_keys=_node_slots(2) + generation,
        )
        before = (current.conn_keys[:, :, 0] != sentinel).sum(axis=1)
        after = (mutated[2][:, :, 0] != sentinel).sum(axis=1)
        assert (after <= capacity).all()
        assert (after >= before).all()
        highest = max(highest, int(after.max()))

        full = before == capacity
        for p in range(2):
            if full[p]:
                assert mutated[2][p].tobytes() == current.conn_keys[p].tobytes()
                assert mutated[3][p].tobytes() == current.conn_attrs[p].tobytes()
                refused += 1
        current = payloads.Payload(*mutated)

    assert highest == capacity  # the bound was reached, not merely approached
    assert refused > 0

    # One free row: the node split needs two, so the whole kernel is a no-op.
    split = genome.DefaultMutation(conn_add=0.0, conn_delete=0.0, node_add=1.0, node_delete=0.0)
    split_genome = _genome(genome, split, max_conns=capacity - 1)
    split_state = payloads.genome_state(split_genome)
    payload = _small_payload(payloads, pop_size=2, max_conns=capacity - 1)
    assert (payload.conn_keys[:, :, 0] != sentinel).sum(axis=1).tolist() == [2, 2]

    mutated = _mutate(run_kernel, split_genome, split_state, payload, _keys(seed, 1)[0])
    for returned, original in zip(mutated, payload.args()):
        assert returned.tobytes() == original.tobytes()


def test_in_degree_never_exceeds_the_genome_window(genome, payloads, run_kernel, seed):
    """Invariant: no destination node gains more than ``max_in_degree`` incoming edges.

    Oracle: the in-degree histogram of every individual's ``conn_keys``, compared
    against ``genome.max_in_degree``, plus the per-add check "the destination's
    in-degree *before* the add was below the window". The window width is the
    schedule's static ``K``; a mutation that exceeded it would make the next
    ``transform`` report a positive ``edge_overflow`` and silently drop edges, so
    the operator must refuse that candidate (a no-op for the individual) rather
    than rely on the schedule to absorb it. The payload starts with node ``2``
    already at in-degree ``K``, so an implementation that ignored the check would
    show a violation on the first accepted add into it.
    """
    sentinel = payloads.I_INF
    limit = 2
    mutation = genome.DefaultMutation(conn_add=1.0, conn_delete=0.0, node_add=0.0, node_delete=0.0)
    genome_object = _genome(genome, mutation, max_in_degree=limit)
    state = payloads.genome_state(genome_object)
    # ``0->2`` and ``1->2`` put node ``2`` at the window; ``2->7`` gives it an outgoing edge.
    payload = _small_payload(payloads, pop_size=3, extra_edges=((1, 2),))
    assert _indegree(_edges(payload.conn_keys[0], sentinel))[2] == limit

    additions = 0
    for key in _keys(seed, 8):
        mutated = _mutate(run_kernel, genome_object, state, payload, key)
        for p in range(3):
            before = _edges(payload.conn_keys[p], sentinel)
            after = _edges(mutated[2][p], sentinel)
            added = after - before
            additions += len(added)
            for _, destination in added:
                assert _indegree(before).get(destination, 0) < limit
            assert all(count <= limit for count in _indegree(after).values())

    assert additions > 0  # the check was exercised on accepted candidates too


def test_new_node_and_connection_keys_are_caller_allocated_and_honoured(
    genome, payloads, run_kernel, seed
):
    """Invariant: an inserted gene carries exactly the key the caller allocated.

    Oracle: set membership. With ``node_add=1.0`` the inserted node's key is
    ``new_node_keys[p]``; with ``conn_add=1.0`` the new edge's ``(in_key, out_key)``
    is the chosen pair and its ``historical_marker`` column (when the conn gene
    declares one) is ``new_conn_keys[p, 2]``; an add-node split uses
    ``new_conn_keys[p, 0]`` for the identity edge and ``[p, 1]`` for the copy edge.
    Nothing is drawn from the RNG: the operator must not invent keys, because
    innovation numbers have to be globally unique across the population and are
    allocated by the algorithm.
    """
    sentinel = payloads.I_INF
    keys = _keys(seed, 8)
    new_node_keys = np.array([20, 21, 22], dtype=np.int32)
    markers = np.array([[40, 41, 42], [50, 51, 52], [60, 61, 62]], dtype=np.int32)

    # (a) node key, default conn gene (no historical-marker column).
    mutation = genome.DefaultMutation(conn_add=0.0, conn_delete=0.0, node_add=1.0, node_delete=0.0)
    genome_object = _genome(genome, mutation)
    state = payloads.genome_state(genome_object)
    payload = _small_payload(payloads, pop_size=3)

    mutated = _mutate(
        run_kernel, genome_object, state, payload, keys[0], new_node_keys=new_node_keys
    )
    for p in range(3):
        before = _nodes(payload.node_keys[p], sentinel)
        assert _nodes(mutated[0][p], sentinel) - before == {int(new_node_keys[p])}

    # (b) add-node markers: identity edge gets column 0, copy edge column 1.
    split = genome.DefaultMutation(conn_add=0.0, conn_delete=0.0, node_add=1.0, node_delete=0.0)
    origin_genome = _genome(
        genome, split, conn_gene=_quiet_conn(genome, genome.OriginConn), max_conns=4
    )
    origin_state = payloads.genome_state(origin_genome)
    payload = _one_edge_payload(payloads, pop_size=3, max_conns=4, conn_key_width=3)
    mutated = _mutate(
        run_kernel, origin_genome, origin_state, payload, keys[1],
        new_node_keys=new_node_keys, new_conn_keys=markers,
    )
    for p in range(3):
        new_key = int(new_node_keys[p])
        rows = {(row[0], row[1]): row[2] for row in _pairs(mutated[2][p], sentinel)}
        assert set(rows) == {(2, new_key), (new_key, _OUTPUT_ROW)}
        assert rows[(2, new_key)] == int(markers[p, 0])
        assert rows[(new_key, _OUTPUT_ROW)] == int(markers[p, 1])

    # (c) add-connection marker: the new edge carries column 2.
    add = genome.DefaultMutation(conn_add=1.0, conn_delete=0.0, node_add=0.0, node_delete=0.0)
    origin_add_genome = _genome(
        genome, add, conn_gene=_quiet_conn(genome, genome.OriginConn), max_conns=8
    )
    origin_add_state = payloads.genome_state(origin_add_genome)
    payload = _small_payload(payloads, pop_size=3, max_conns=8, conn_key_width=3)
    additions = 0
    for key in keys:
        mutated = _mutate(
            run_kernel, origin_add_genome, origin_add_state, payload, key,
            new_node_keys=new_node_keys, new_conn_keys=markers,
        )
        for p in range(3):
            before = _edges(payload.conn_keys[p], sentinel)
            rows = {(row[0], row[1]): row[2] for row in _pairs(mutated[2][p], sentinel)}
            for edge in set(rows) - before:
                additions += 1
                assert rows[edge] == int(markers[p, 2])

    assert additions > 0  # the marker column was actually written at least once


def test_keys_stay_unique_and_padding_sentinels_are_intact(genome, payloads, run_kernel, seed):
    """Invariant: no duplicate key within an individual, and absent rows keep their sentinels.

    Oracle: ``len(set(node_keys[p]))`` over the present rows, and the pairing
    "``node_keys == I_INF`` iff ``node_attrs`` is all-``NaN``" (likewise for
    connections). Both sentinels are checked together because a half-cleared row —
    an ``I_INF`` key with finite attributes, or a finite key with ``NaN``
    attributes — breaks every downstream kernel while looking correct in one array.
    Keys allocated above the current maximum (the algorithm's convention) must also
    come out *strictly greater* than every pre-existing key of that individual, so
    "new gene" stays decidable from the key alone and no inserted key aliases an
    older gene. The check runs after an insert and after a delete, the two ways a
    row can change.
    """
    sentinel = payloads.I_INF
    keys = _keys(seed, 3)
    new_node_keys = _node_slots(3)

    insert = genome.DefaultMutation(conn_add=0.0, conn_delete=0.0, node_add=1.0, node_delete=0.0)
    insert_genome = _genome(genome, insert)
    insert_state = payloads.genome_state(insert_genome)
    payload = _small_payload(payloads, pop_size=3)

    for key in keys:
        mutated = _mutate(
            run_kernel, insert_genome, insert_state, payload, key, new_node_keys=new_node_keys
        )
        for p in range(3):
            present = mutated[0][p] != sentinel
            assert len(set(mutated[0][p][present].tolist())) == int(present.sum())
            _assert_padding_is_paired(
                mutated[0][p], mutated[1][p], mutated[2][p], mutated[3][p], sentinel
            )
            connection_pairs = _pairs(mutated[2][p], sentinel)
            assert len(set(connection_pairs)) == len(connection_pairs)

            before = _nodes(payload.node_keys[p], sentinel)
            inserted = _nodes(mutated[0][p], sentinel) - before
            assert inserted <= {int(new_node_keys[p])}
            if inserted:
                assert min(inserted) > max(before)

    delete = genome.DefaultMutation(conn_add=0.0, conn_delete=0.0, node_add=0.0, node_delete=1.0)
    delete_genome = _genome(genome, delete)
    delete_state = payloads.genome_state(delete_genome)
    payload = payloads.chain_payload(pop_size=3)

    for key in keys:
        mutated = _mutate(run_kernel, delete_genome, delete_state, payload, key)
        for p in range(3):
            present = mutated[0][p] != sentinel
            assert len(set(mutated[0][p][present].tolist())) == int(present.sum())
            _assert_padding_is_paired(
                mutated[0][p], mutated[1][p], mutated[2][p], mutated[3][p], sentinel
            )
            # A delete leaves exactly one hole behind, never a duplicated survivor.
            assert int(present.sum()) == int((payload.node_keys[p] != sentinel).sum()) - 1


def test_rows_outside_the_mutated_gene_are_bit_identical(genome, payloads, run_kernel, seed):
    """Invariant: an operator touches only the rows it names.

    Oracle: a row-by-row diff against the input payload. With ``conn_delete=1.0``
    exactly the chosen connection's ``CA`` attribute columns and ``CK`` key columns
    change (to ``NaN``/``I_INF``), every other row is bit-identical; with
    ``node_delete=1.0`` exactly the deleted node's row and the rows of the
    connections touching it change. A kernel that rebuilt the whole table (a
    ``gather``-then-``scatter`` over all rows) would show up here as a diff in
    untouched rows even when the values happen to look plausible.
    """
    sentinel = payloads.I_INF
    key = _keys(seed, 1)[0]

    # (a) delete-connection: one cleared connection row, node arrays untouched.
    mutation = genome.DefaultMutation(conn_add=0.0, conn_delete=1.0, node_add=0.0, node_delete=0.0)
    genome_object = _genome(genome, mutation)
    state = payloads.genome_state(genome_object)
    payload = _small_payload(payloads, pop_size=4)

    mutated = _mutate(run_kernel, genome_object, state, payload, key)
    assert mutated[0].tobytes() == payload.node_keys.tobytes()
    assert mutated[1].tobytes() == payload.node_attrs.tobytes()
    for p in range(4):
        changed = _changed_rows(payload.conn_keys[p], mutated[2][p])
        assert len(changed) == 1
        row = changed[0]
        assert (mutated[2][p, row] == sentinel).all()
        assert np.isnan(mutated[3][p, row]).all()
        assert _changed_rows(payload.conn_attrs[p], mutated[3][p]) == [row]
        untouched = [other for other in range(payload.max_conns) if other != row]
        assert mutated[2][p, untouched].tobytes() == payload.conn_keys[p, untouched].tobytes()
        assert mutated[3][p, untouched].tobytes() == payload.conn_attrs[p, untouched].tobytes()

    # (b) delete-node: its node row, plus exactly the connection rows touching it.
    mutation = genome.DefaultMutation(conn_add=0.0, conn_delete=0.0, node_add=0.0, node_delete=1.0)
    genome_object = _genome(genome, mutation)
    state = payloads.genome_state(genome_object)
    payload = payloads.chain_payload(pop_size=4)

    mutated = _mutate(run_kernel, genome_object, state, payload, key)
    for p in range(4):
        lost = _nodes(payload.node_keys[p], sentinel) - _nodes(mutated[0][p], sentinel)
        assert len(lost) == 1
        (deleted_key,) = lost
        touching = [
            row
            for row, key_row in enumerate(payload.conn_keys[p])
            if int(key_row[0]) != sentinel and deleted_key in (int(key_row[0]), int(key_row[1]))
        ]
        assert touching  # the chain really has edges on the deleted node
        assert _changed_rows(payload.conn_keys[p], mutated[2][p]) == touching
        assert _changed_rows(payload.conn_attrs[p], mutated[3][p]) == touching
        for row in touching:
            assert (mutated[2][p, row] == sentinel).all()
            assert np.isnan(mutated[3][p, row]).all()
        changed_nodes = _changed_rows(payload.node_keys[p], mutated[0][p])
        assert len(changed_nodes) == 1
        assert int(payload.node_keys[p, changed_nodes[0]]) == deleted_key
        assert _changed_rows(payload.node_attrs[p], mutated[1][p]) == changed_nodes


def _classify_change(payload, mutated, new_node_keys, sentinel, interface):
    """Assert the per-operation mutation law; return ``(gained, lost, both, extra_added,
    unexplained_removed)`` totals over the population.

    The four gates are independent Bernoullis and the stages compose sequentially, so one
    individual may gain **and** lose a node in the same generation; what is bounded is the
    change *per operation*: at most one node added (carrying the caller-allocated
    ``new_node_keys[p]``), at most one node deleted (never an interface key), the split's
    two edges plus at most one added connection, and the lost node's edges plus the
    split's consumed edge plus at most one removed connection. ``extra_added`` counts the
    added edges touching no gained node (the add-conn ones) and ``unexplained_removed``
    the removals no lost node explains.
    """
    totals = [0, 0, 0, 0, 0]
    for p in range(payload.pop_size):
        nodes_before = _nodes(payload.node_keys[p], sentinel)
        nodes_after = _nodes(mutated[0][p], sentinel)
        edges_before = _edges(payload.conn_keys[p], sentinel)
        edges_after = _edges(mutated[2][p], sentinel)
        gained = nodes_after - nodes_before
        lost = nodes_before - nodes_after
        added = edges_after - edges_before
        removed = edges_before - edges_after

        # add-node: one node, carrying the one key the caller allocated for it.
        assert len(gained) <= 1, f"individual {p} gained {gained}"
        assert gained <= {int(new_node_keys[p])}, f"individual {p} gained {gained}"
        # delete-node: one node, and the interface rows are never candidates.
        assert len(lost) <= 1, f"individual {p} lost {lost}"
        assert not (lost & interface), f"individual {p} lost {lost}"

        entering = {edge for edge in added if edge[1] in gained}
        leaving = {edge for edge in added if edge[0] in gained}
        assert len(added) <= 3, f"individual {p} added {added}"
        assert len(entering) <= 2 and len(leaving) <= 2, f"individual {p} added {added}"
        assert len(added - entering - leaving) <= 1, f"individual {p} added {added}"
        if not gained:  # a deleted split node takes both of its edges with it
            assert len(added) <= 1, f"individual {p} added {added}"

        dangling = {edge for edge in edges_before if lost.intersection(edge)}
        assert dangling <= removed, f"individual {p} kept an edge of {lost}"
        assert len(removed - dangling) <= 2, f"individual {p} removed {removed}"

        totals[0] += bool(gained)
        totals[1] += bool(lost)
        totals[2] += bool(gained and lost)
        totals[3] += len(added - entering - leaving)
        totals[4] += len(removed - dangling)
    return tuple(totals)


def test_one_change_per_operation_type_but_several_per_individual(
    genome, payloads, run_kernel, seed
):
    """Invariant: each operation fires at most once, yet one individual may run several.

    Oracle: set algebra on the returned payload (:func:`_classify_change`). With all four
    probabilities at ``1.0`` every gate is open, so the payload alone decides the outcome:
    an individual gains the allocated node *and* loses a hidden one in the same
    generation, which is exactly what four independent Bernoullis running in sequence
    allow. The per-operation bounds still hold, and the totals show add-conn landing
    beside the split instead of replacing it.
    """
    sentinel = payloads.I_INF
    keys = _keys(seed, 4)
    new_node_keys = _node_slots(4)
    payload = _small_payload(payloads, pop_size=4)
    interface_rows = list(range(_NUM_INPUTS)) + list(range(_OUTPUT_ROW, _MAX_NODES))
    interface = _nodes(payload.node_keys[:, interface_rows].ravel(), sentinel)

    def totals(mutation):
        genome_object = _genome(genome, mutation)
        state = payloads.genome_state(genome_object)
        result = [0, 0, 0, 0, 0]
        for key in keys:
            mutated = _mutate(
                run_kernel, genome_object, state, payload, key, new_node_keys=new_node_keys
            )
            for index, value in enumerate(
                _classify_change(payload, mutated, new_node_keys, sentinel, interface)
            ):
                result[index] += value
        return result

    # (a) every gate open: the same individual both gains and loses a node.
    gained, lost, both, extra_added, _ = totals(
        genome.DefaultMutation(conn_add=1.0, conn_delete=1.0, node_add=1.0, node_delete=1.0)
    )
    assert gained > 0 and lost > 0
    assert both > 0, "no individual both gained and lost a node"
    assert extra_added > 0  # an add-conn edge landed beside the split

    # (b) a mixture: every operation is reachable, and the same bounds still hold.
    gained, lost, _, extra_added, unexplained = totals(
        genome.DefaultMutation(conn_add=0.5, conn_delete=0.5, node_add=0.5, node_delete=0.5)
    )
    assert gained > 0 and lost > 0  # both node operations were reached
    assert extra_added > 0 and unexplained > 0  # the connection operations too


def test_add_connection_never_duplicates_an_existing_edge(genome, payloads, run_kernel, seed):
    """Invariant: the added ``(in_key, out_key)`` pair is not already present.

    Oracle: set membership on the payload's own connection keys, computed per
    individual before and after. A duplicate edge would double that source's
    contribution in the aggregation without changing the topology, which is the
    classic silent bug of a random-pair add; ``support.payloads`` gives a dense
    network where every legal pair exists, so a naive draw collides almost always,
    and a sparse one where an add is accepted, so the check is not vacuous.
    """
    sentinel = payloads.I_INF
    mutation = genome.DefaultMutation(conn_add=1.0, conn_delete=0.0, node_add=0.0, node_delete=0.0)
    genome_object = _genome(genome, mutation)
    state = payloads.genome_state(genome_object)

    # Dense: a duplicate would be reachable for most draws, so all must be refused.
    dense = _dense_payload(payloads, pop_size=3)
    for key in _keys(seed, 8):
        mutated = _mutate(run_kernel, genome_object, state, dense, key)
        for p in range(3):
            rows = _pairs(mutated[2][p], sentinel)
            pairs = [(row[0], row[1]) for row in rows]
            assert len(set(pairs)) == len(pairs)
            assert _edges(dense.conn_keys[p], sentinel) <= set(pairs)

    # Sparse: adds are accepted, and none of them repeats an existing edge.
    sparse = _small_payload(payloads, pop_size=3)
    additions = 0
    for key in _keys(seed, 8):
        mutated = _mutate(run_kernel, genome_object, state, sparse, key)
        for p in range(3):
            before = _edges(sparse.conn_keys[p], sentinel)
            rows = _pairs(mutated[2][p], sentinel)
            pairs = [(row[0], row[1]) for row in rows]
            assert len(set(pairs)) == len(pairs)
            assert before <= set(pairs)
            additions += len(set(pairs) - before)

    assert additions > 0


def test_add_connection_never_closes_a_cycle_in_a_feedforward_genome(
    genome, payloads, run_kernel, seed
):
    """Invariant: a feedforward genome never gains an edge that closes a cycle.

    Oracle: the genome's own ``closes_cycle`` predicate is *not* used as the oracle
    (it is part of the code under test); the test recomputes reachability in Python
    on the returned topology — after the mutation, the directed graph must still be
    acyclic. On the dense chain every backward edge would close a cycle, so any
    accepted candidate is a visible violation; on a recurrent genome the very same
    mutation is legal, which the test asserts on the ring payload.
    """
    sentinel = payloads.I_INF
    new_node_keys = _node_slots(3)
    keys = _keys(seed, 8)

    # (a) feedforward: the graph stays acyclic, and edges really are added.
    mutation = genome.DefaultMutation(conn_add=1.0, conn_delete=0.0, node_add=0.0, node_delete=0.0)
    genome_object = _genome(genome, mutation)
    state = payloads.genome_state(genome_object)
    payload = _small_payload(payloads, pop_size=3)

    additions = 0
    for key in keys:
        mutated = _mutate(
            run_kernel, genome_object, state, payload, key, new_node_keys=new_node_keys
        )
        for p in range(3):
            before = _edges(payload.conn_keys[p], sentinel)
            after = _edges(mutated[2][p], sentinel)
            additions += len(after - before)
            assert not _has_cycle(after)

    assert additions > 0  # the cycle test was applied to accepted edges

    # (b) recurrent: the identical operator accepts a cycle-closing edge.
    recurrent_genome = _genome(genome, mutation, cls=genome.RecurrentGenome)
    recurrent_state = payloads.genome_state(recurrent_genome)
    ring = payloads.ring_payload(pop_size=3)
    assert _has_cycle(_edges(ring.conn_keys[0], sentinel))  # the ring is already cyclic

    cyclic_additions = 0
    for key in keys:
        mutated = _mutate(
            run_kernel, recurrent_genome, recurrent_state, ring, key, new_node_keys=new_node_keys
        )
        for p in range(3):
            before = _edges(ring.conn_keys[p], sentinel)
            for edge in _edges(mutated[2][p], sentinel) - before:
                if _has_cycle(before | {edge}):
                    cyclic_additions += 1

    assert cyclic_additions > 0


def test_add_node_splits_one_connection_into_two(genome, payloads, run_kernel, evoxc_key):
    """Invariant: the split node sits between the old endpoints with an identity and a copy edge.

    Oracle: hand enumeration on a payload with a single known edge ``a->b``:
    ``node_add=1.0`` must leave ``a->b`` deleted, add ``a->n`` with
    ``conn_gene.new_identity_attrs`` (``weight = 1.0``: it reproduces its source)
    and ``n->b`` with the *original* attributes of ``a->b`` (so the network's
    output is preserved). The new node's attribute row is the node gene's identity
    row. Counts: nodes ``+1``, connections ``+1``; the source and destination key
    sets otherwise unchanged.
    """
    sentinel = payloads.I_INF
    mutation = genome.DefaultMutation(conn_add=0.0, conn_delete=0.0, node_add=1.0, node_delete=0.0)
    genome_object = _genome(genome, mutation, max_conns=4)
    state = payloads.genome_state(genome_object)
    payload = _one_edge_payload(payloads, pop_size=2, max_conns=4, weight=2.0)
    new_node_keys = np.array([20, 21], dtype=np.int32)

    mutated = _mutate(
        run_kernel, genome_object, state, payload, evoxc_key, new_node_keys=new_node_keys
    )

    # ``DefaultNode.new_identity_attrs``: bias 0, response 1, option columns at
    # their configured default (option 0).
    identity_node_row = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)

    for p in range(2):
        new_key = int(new_node_keys[p])
        rows_before = _nodes(payload.node_keys[p], sentinel)
        assert _nodes(mutated[0][p], sentinel) == rows_before | {new_key}
        assert (mutated[0][p] != sentinel).sum() == len(rows_before) + 1

        edges_before = _edges_with_attrs(payload.conn_keys[p], payload.conn_attrs[p], sentinel)
        edges_after = _edges_with_attrs(mutated[2][p], mutated[3][p], sentinel)
        assert set(edges_before) == {(2, _OUTPUT_ROW)}
        assert (2, _OUTPUT_ROW) not in edges_after
        assert set(edges_after) == {(2, new_key), (new_key, _OUTPUT_ROW)}
        assert len(edges_after) == len(edges_before) + 1

        # Identity edge reproduces its source; the copy edge keeps the original weight.
        assert np.array_equal(edges_after[(2, new_key)], np.array([1.0], dtype=np.float32))
        assert np.array_equal(
            edges_after[(new_key, _OUTPUT_ROW)], edges_before[(2, _OUTPUT_ROW)]
        )

        new_row = int(np.flatnonzero(mutated[0][p] == new_key)[0])
        assert np.array_equal(mutated[1][p, new_row], identity_node_row)

        # The interface rows (inputs and output) are untouched.
        assert mutated[0][p, 0] == 0 and mutated[0][p, 1] == 1
        assert mutated[0][p, _OUTPUT_ROW] == _OUTPUT_ROW
        assert mutated[1][p, 0].tobytes() == payload.node_attrs[p, 0].tobytes()
        assert mutated[1][p, _OUTPUT_ROW].tobytes() == payload.node_attrs[p, _OUTPUT_ROW].tobytes()


def test_delete_node_never_removes_an_input_or_output_row(genome, payloads, run_kernel, seed):
    """Invariant: ``node_delete`` clears a hidden row and every edge touching it, nothing else.

    Oracle: the surviving key set of the input and output rows — ``0..num_inputs-1``
    and ``max_nodes-num_outputs..max_nodes-1`` must still be present and unchanged —
    plus "no connection row references the deleted key any more" (the
    ``DeleteNodePlan.drop`` detach mask, which is data-dependent because a node
    may have any number of edges). On ``payloads.isolated_payload`` the only hidden row
    has no edges, so the delete must clear it and change no connection row at all.
    """
    sentinel = payloads.I_INF
    interface_rows = list(range(_NUM_INPUTS)) + list(range(_OUTPUT_ROW, _MAX_NODES))
    mutation = genome.DefaultMutation(conn_add=0.0, conn_delete=0.0, node_add=0.0, node_delete=1.0)
    genome_object = _genome(genome, mutation)
    state = payloads.genome_state(genome_object)
    key = _keys(seed, 1)[0]

    # (a) isolated payload: the lone hidden row has no edges at all.
    payload = payloads.isolated_payload(pop_size=3)
    mutated = _mutate(run_kernel, genome_object, state, payload, key)
    assert mutated[2].tobytes() == payload.conn_keys.tobytes()
    assert mutated[3].tobytes() == payload.conn_attrs.tobytes()
    for p in range(3):
        for row in interface_rows:
            assert mutated[0][p, row] == payload.node_keys[p, row]
            assert mutated[1][p, row].tobytes() == payload.node_attrs[p, row].tobytes()
        # The only hidden row (row ``_NUM_INPUTS``) is gone, and it was hidden.
        assert _nodes(payload.node_keys[p], sentinel) - _nodes(mutated[0][p], sentinel) == {
            _NUM_INPUTS
        }

    # (b) chain payload: a hidden node with edges takes all its edges with it.
    payload = payloads.chain_payload(pop_size=3)
    mutated = _mutate(run_kernel, genome_object, state, payload, key)
    hidden_rows = set(range(_NUM_INPUTS, _OUTPUT_ROW))
    for p in range(3):
        for row in interface_rows:
            assert mutated[0][p, row] == payload.node_keys[p, row]
            assert mutated[1][p, row].tobytes() == payload.node_attrs[p, row].tobytes()

        lost = _nodes(payload.node_keys[p], sentinel) - _nodes(mutated[0][p], sentinel)
        assert len(lost) == 1
        (deleted_key,) = lost
        assert deleted_key in hidden_rows

        surviving = _pairs(mutated[2][p], sentinel)
        assert all(deleted_key not in (row[0], row[1]) for row in surviving)
        edges_before = _edges(payload.conn_keys[p], sentinel)
        assert edges_before - _edges(mutated[2][p], sentinel) == {
            edge for edge in edges_before if deleted_key in edge
        }


def test_execute_mutation_rejects_non_int32_key_tensors(
    genome, payloads, run_kernel, evoxc_key, common
):
    """Invariant: caller-allocated structural keys are ``int32`` or refused, never cast.

    Oracle: the documented ``TensorNEATError`` naming the field and the dtype
    (``must be an int32 key tensor``), for a float32 ``new_node_keys`` and, separately,
    for a float32 ``new_conn_keys`` at the otherwise correct shapes. Keys are ``int32``
    throughout the payload, so a cast would be a traced op hiding a host-side violation.
    """
    mutation = genome.DefaultMutation(
        conn_add=0.0, conn_delete=0.0, node_add=0.0, node_delete=0.0
    )
    genome_object = _genome(genome, mutation)
    state = payloads.genome_state(genome_object)
    payload = _small_payload(payloads)
    node_keys = _node_slots(payload.pop_size)
    conn_keys = _conn_slots(payload.pop_size)

    for label, new_node_keys, new_conn_keys in (
        ("new_node_keys", node_keys.astype(np.float32), conn_keys),
        ("new_conn_keys", node_keys, conn_keys.astype(np.float32)),
    ):
        with pytest.raises(common.TensorNEATError) as info:
            _mutate(
                run_kernel,
                genome_object,
                state,
                payload,
                evoxc_key,
                new_node_keys=new_node_keys,
                new_conn_keys=new_conn_keys,
            )
        message = str(info.value)
        assert "must be an int32 key tensor" in message, message
        assert label in message, message
        assert "float32" in message, message


def test_execute_mutation_rejects_wrong_key_shapes(
    genome, payloads, run_kernel, evoxc_key, common
):
    """Invariant: ``new_node_keys`` is ``(P,)`` and ``new_conn_keys`` is ``(P, 3)``.

    Oracle: the documented ``TensorNEATError`` naming the field and the expected shape
    (``must have shape``), for a ``(P, 3)`` node-key block and, separately, a ``(P,)``
    connection-key block. Both shapes are static at build time, so a wrong one is refused
    on the host instead of reaching ``evoxc.build`` as a mis-shaped graph input.
    """
    mutation = genome.DefaultMutation(
        conn_add=0.0, conn_delete=0.0, node_add=0.0, node_delete=0.0
    )
    genome_object = _genome(genome, mutation)
    state = payloads.genome_state(genome_object)
    payload = _small_payload(payloads)
    pop = payload.pop_size

    for label, new_node_keys, new_conn_keys, expected in (
        ("new_node_keys", np.zeros((pop, 3), np.int32), _conn_slots(pop), f"({pop},)"),
        ("new_conn_keys", _node_slots(pop), np.zeros((pop,), np.int32), f"({pop}, 3)"),
    ):
        with pytest.raises(common.TensorNEATError) as info:
            _mutate(
                run_kernel,
                genome_object,
                state,
                payload,
                evoxc_key,
                new_node_keys=new_node_keys,
                new_conn_keys=new_conn_keys,
            )
        message = str(info.value)
        assert f"{label} must have shape {expected}" in message, message
