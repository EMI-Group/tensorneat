"""Host-side graph algorithms for interpretability, visualisation and export.

Dependency DAG position: leaf of ``tensorneat.common``. Imports only the
standard library and the error hierarchy (:class:`GraphError`,
:class:`ConfigError`) from :mod:`tensorneat.common.errors`; no ``evoxc``, no
``numpy``, no optional backend.

Role: this module is the **host-side counterpart** of the in-graph execution
schedule (:func:`tensorneat.common.tensorization.derive_schedule`). It never
runs inside a trace, it is never called from a kernel, and it produces no
tensors — it exists so that the three host-only consumers
(``genome.network_dict``, ``genome.sympy_func``, ``genome.visualize``) share
one implementation of "walk this topology" instead of each re-deriving
adjacency from the gene payload.

Input representation: every function here deliberately operates on plain
Python ``set``/``dict``/``list`` structures decoded from a gene payload by the
caller (see :func:`decode_graph`) — node keys are ``int``, connections are
``(in_key, out_key)`` pairs of ``int``. No array, no sentinel (``I_INF``,
``NaN``) and no batch axis leaks into this module; a caller that holds
``(P, max_nodes, node_len)`` payloads selects one population member first.

Why the split from the schedule: the in-graph schedule answers *how to execute
a whole population with static shapes* and costs ``O(P * conns)`` inside a
compiled graph; the functions below answer *what does one genome look like*
and are allowed to be quadratic-ish, allocate dicts, and raise Python
exceptions. Keeping them apart is what lets the schedule stay tensor-only and
lets this module stay dependency-free.
"""

from .errors import ConfigError, GraphError

__all__ = [
    "decode_graph",
    "topological_sort",
    "find_useful_nodes",
    "check_cycles",
    "node_depth",
]

#: Mirror of :data:`tensorneat.common.tools.I_INF` (``int32`` max) — every
#: integer gene key strictly below it is a real key. It is duplicated here
#: instead of imported because ``tools`` pulls in ``evoxc``, and this module is
#: deliberately backend-free.
_I_INF = 2147483647


def _is_absent(value):
    """True when one row's key column marks an unused gene row.

    Both payload spellings are accepted, because the two coexist in this
    package: the canonical split payload stores keys as ``int32`` padded with
    ``I_INF``, while the pre-rewrite (and parity) spelling stores them as
    ``float32`` padded with ``NaN``. The rule is therefore "``NaN`` or
    ``I_INF``", applied to the raw scalar so that Python floats, NumPy float
    scalars, Python ints and NumPy integer scalars all work without importing
    ``numpy``.

    Parameters
    ----------
    value : float or int
        The key column (``row[0]``) of one gene row.

    Returns
    -------
    bool
        ``True`` iff the row is padding and must be skipped.
    """
    if value != value:  # NaN, true for Python and NumPy floating scalars
        return True
    return int(value) == _I_INF


def _unique_edges(conns):
    """Normalise a ``set``/``list`` of connections to a fresh ``set`` of pairs.

    The graph algorithms below treat a connection list as a *topology*, so a
    duplicated ``(in_key, out_key)`` entry must count once (otherwise the
    in-degree bookkeeping and the child lists disagree). The input collection is
    never mutated: a caller that reuses its genome between mutation rounds keeps
    its edges.

    Parameters
    ----------
    conns : set[tuple[int, int]] or list[tuple[int, int]]
        ``(in_key, out_key)`` pairs.

    Returns
    -------
    set[tuple[int, int]]
        A new set holding each pair exactly once.
    """
    edges = set()
    for pair in conns:
        in_key, out_key = pair
        edges.add((in_key, out_key))
    return edges


def decode_graph(nodes, conns, node_attrs=1, conn_attrs=2):
    """Decode one genome's gene payload into plain Python structures.

    This is the only place where the padded array payload crosses into
    host graph land: everything downstream of it works on ``int`` keys.

    Parameters
    ----------
    nodes : array-like of float or int, shape ``(max_nodes, node_len)``
        Node gene rows of a **single** genome (no population axis). Column 0
        is the node key; the remaining columns are gene attributes. A row whose
        column 0 is absent is unused padding and is skipped; absence is ``NaN``
        for a ``float32`` payload (the pre-rewrite spelling) and ``I_INF``
        (``int32`` max) for the canonical split payload (see Notes).
    conns : array-like of float or int, shape ``(max_conns, conn_len)``
        Connection gene rows of the same genome. Columns 0 and 1 are the
        ``(in_key, out_key)`` endpoints; the remaining columns are gene
        attributes (weight, marker, ...). A row whose column 0 is ``NaN``
        (or ``I_INF``) is unused padding and is skipped.
    node_attrs : int, optional
        Width of the *fixed* (structural) prefix of a node row, i.e. the number
        of leading columns that are not free gene attributes. ``1`` for the
        canonical payload, whose only fixed column is the key
        (``DefaultNodeGene.fixed_attrs == ["idx"]``).
    conn_attrs : int, optional
        Width of the *fixed* prefix of a connection row. ``2`` for the
        canonical payload, whose fixed columns are ``(in_key, out_key)``
        (``BaseConn.fixed_attrs == ["input_index", "output_index"]``).

    Returns
    -------
    nodes_out : dict[int, tuple[float, ...]]
        Key ``int(row[0])`` (the integer-valued float gene key, cast with
        ``int(...)`` exactly as the pre-rewrite ``to_dict`` helpers did) mapped
        to that node's gene attributes, i.e. ``row[node_attrs:]`` as a tuple of
        Python floats. Insertion order follows the payload row order.
    conns_out : set[tuple[int, int]]
        ``(int(row[0]), int(row[1]))`` for every used connection row. The
        connection *attribute* columns are not returned: every graph algorithm
        in this module consumes topology only, and a caller that needs weights
        for labels or sympy symbols re-reads the payload row it already holds.

    Raises
    ------
    tensorneat.common.errors.ConfigError
        If ``node_attrs < 1`` or ``conn_attrs < 2``, or if a used row is too
        short for its declared fixed prefix (validated on the host, at
        export time, before any graph walk).
    tensorneat.common.errors.GraphError
        If two used node rows carry the same key, or a connection endpoint key
        does not name a used node row.

    Notes
    -----
    **Padding detection accepts both key spellings.** The canonical payload
    stores keys as ``int32`` and pads with ``I_INF`` (``int32`` max); the
    pre-rewrite payload stores them as ``float32`` and pads with ``NaN``. A row
    is unused iff its column 0 is ``NaN`` *or* ``I_INF``, so one code path
    decodes either spelling — the check is spelled out here rather than
    special-cased silently, because a caller mixing the two in one payload is
    a bug the equality of the two sentinels would otherwise hide. A *used*
    connection row that carries a padding destination key cannot be keyed at
    all, and is reported as a :class:`GraphError` (the topology is broken).

    Keys are exact: they are integer-valued keys below ``KEY_MAX = 2**24``, so
    ``int(...)`` round-trips them and ``int`` equality is gene-key equality (no
    hashing of the whole row is involved).
    Duplicate connection pairs (same ``(in_key, out_key)`` twice) collapse into
    one element of the returned set; the schedule is the authority on whether
    that is legal, this module just makes the topology a set.
    """
    if int(node_attrs) < 1:
        raise ConfigError(
            f"node_attrs must be at least 1 (the key column), got {node_attrs!r}"
        )
    if int(conn_attrs) < 2:
        raise ConfigError(
            "conn_attrs must be at least 2 (the (in_key, out_key) columns), "
            f"got {conn_attrs!r}"
        )

    nodes_out = {}
    for row_index, row in enumerate(nodes):
        width = len(row)
        if width < 1:
            raise ConfigError(
                f"node row {row_index} has no columns; column 0 must hold the key"
            )
        if _is_absent(row[0]):
            continue
        if width < node_attrs:
            raise ConfigError(
                f"node row {row_index} has {width} column(s), fewer than the "
                f"declared fixed prefix node_attrs={node_attrs}"
            )
        key = int(row[0])
        if key in nodes_out:
            raise GraphError(f"duplicate node key {key} in the gene payload")
        nodes_out[key] = tuple(float(value) for value in row[node_attrs:])

    conns_out = set()
    for row_index, row in enumerate(conns):
        width = len(row)
        if width < 1:
            raise ConfigError(
                f"connection row {row_index} has no columns; columns 0 and 1 "
                "must hold the (in_key, out_key) endpoints"
            )
        if _is_absent(row[0]):
            continue
        if width < conn_attrs:
            raise ConfigError(
                f"connection row {row_index} has {width} column(s), fewer than "
                f"the declared fixed prefix conn_attrs={conn_attrs}"
            )
        if _is_absent(row[1]):
            raise GraphError(
                f"connection row {row_index} has a padding destination key; a "
                "used row must name both endpoints"
            )
        in_key, out_key = int(row[0]), int(row[1])
        for endpoint, role in ((in_key, "input"), (out_key, "output")):
            if endpoint not in nodes_out:
                raise GraphError(
                    f"connection row {row_index} names {role} node {endpoint}, "
                    "which has no used node row in this genome"
                )
        conns_out.add((in_key, out_key))

    return nodes_out, conns_out


def topological_sort(nodes, conns):
    """Kahn topological sort of one host-side genome graph.

    Parameters
    ----------
    nodes : set[int] or list[int]
        Node keys of the genome. The collection is *not* mutated (the
        pre-rewrite version copied its arguments).
    conns : set[tuple[int, int]] or list[tuple[int, int]]
        ``(in_key, out_key)`` connection pairs over ``nodes``.

    Returns
    -------
    topo_order : list[int]
        Every node key exactly once, in an order where each node follows all of
        its predecessors. Nodes that become available in the same round are
        emitted in ascending key order, so the result is deterministic and
        independent of set iteration order.
    topo_layers : list[list[int]]
        The same partition by readiness round: ``topo_layers[0]`` are the nodes
        with in-degree 0 (the input nodes of a feedforward genome),
        ``topo_layers[i]`` those freed by round ``i - 1``; each layer is sorted
        ascending. ``topo_order`` is the concatenation of the layers.

    Raises
    ------
    tensorneat.common.errors.GraphError
        If the graph has at least one cycle (some connection or node is left
        unprocessed when no zero-in-degree node remains), or if an endpoint of
        a connection is not in ``nodes``.
        The pre-rewrite library raised ``ValueError`` here; the package now has
        an error hierarchy, and a cycle in a genome whose execution requires
        feedforward is exactly :class:`GraphError` ("cycle where feedforward is
        required"). ``GraphError`` is a ``TensorNEATError``, so callers that
        only care about observability can catch the common root.

    Notes
    -----
    Host-only companion of the in-graph ``level``/``topo_key`` order: same
    topology order for a feedforward genome, but this version sorts eagerly,
    allocates Python containers, and reports a cycle by raising, whereas the
    schedule reports it as ``level == I_INF`` data.
    """
    node_set = set(nodes)
    edges = _unique_edges(conns)

    in_degree = {node: 0 for node in node_set}
    children = {node: [] for node in node_set}
    for in_key, out_key in edges:
        if in_key not in node_set:
            raise GraphError(
                f"connection source {in_key} is not one of the genome's nodes"
            )
        if out_key not in node_set:
            raise GraphError(
                f"connection destination {out_key} is not one of the genome's nodes"
            )
        in_degree[out_key] += 1
        children[in_key].append(out_key)

    topo_layers = []
    remaining = set(node_set)
    current = sorted(node for node in node_set if in_degree[node] == 0)
    while current:
        topo_layers.append(current)
        freed_next_round = []
        for node in current:
            remaining.discard(node)
            for child in children[node]:
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    freed_next_round.append(child)
        current = sorted(freed_next_round)

    if remaining:
        raise GraphError(
            "Graph has at least one cycle, topological sort not possible: "
            f"unprocessed nodes {sorted(remaining)}"
        )

    topo_order = [node for layer in topo_layers for node in layer]
    return topo_order, topo_layers


def find_useful_nodes(nodes, conns, output_idx):
    """Backward-reachable nodes: the ones that can influence an output.

    Parameters
    ----------
    nodes : set[int] or list[int]
        Node keys of the genome; used to bound the closure (keys outside it are
        ignored) and not mutated.
    conns : set[tuple[int, int]] or list[tuple[int, int]]
        ``(in_key, out_key)`` connection pairs of the genome.
    output_idx : set[int] or list[int]
        Output node keys — the seeds of the closure
        (``genome.output_idx`` / ``algorithm`` output keys).

    Returns
    -------
    set[int]
        ``output_idx`` plus every node that reaches one of them along a
        connection, computed as an iterative fixpoint: start from the outputs,
        repeatedly add the sources of connections whose destination is already
        in the set, stop when a round adds nothing.
        Input nodes are members only if they feed something useful, so a genome
        may legitimately report ``useful_nodes`` smaller than ``nodes``.

    Raises
    ------
    tensorneat.common.errors.ConfigError
        If ``output_idx`` contains a key that is not in ``nodes`` (the caller's
        genome and its declared outputs disagree).

    Notes
    -----
    Termination is guaranteed by monotonicity (the set only grows, bounded by
    ``nodes``), so this is ``O(conns)`` per round and at most ``nodes`` rounds.
    Cycles do not need special handling: a recurrent genome simply has nodes
    that reach the outputs through the cycle.
    """
    node_set = set(nodes)
    outputs = list(output_idx)
    for key in outputs:
        if key not in node_set:
            raise ConfigError(
                f"output node {key} is not one of the genome's nodes "
                f"{sorted(node_set)}"
            )

    useful_nodes = set(outputs)
    edges = _unique_edges(conns)
    while True:
        added = {
            in_key
            for in_key, out_key in edges
            if out_key in useful_nodes and in_key not in useful_nodes
        }
        # Bound the closure by the genome: a key outside ``nodes`` is ignored
        # rather than grown into the result.
        added &= node_set
        if not added:
            return useful_nodes
        useful_nodes |= added


def check_cycles(nodes, conns, from_idx, to_idx):
    """Would adding the edge ``from_idx -> to_idx`` close a cycle?

    Host-side, plain-Python counterpart of the pre-rewrite ``check_cycles``:
    build the adjacency of ``conns`` plus the candidate edge, then answer by
    boolean transitive closure — ``from_idx`` is reachable from ``to_idx``.

    Parameters
    ----------
    nodes : set[int] or list[int]
        Node keys of the genome; bounds the closure and is not mutated.
    conns : set[tuple[int, int]] or list[tuple[int, int]]
        Existing ``(in_key, out_key)`` pairs. The candidate edge is *not*
        inserted into this collection (the pre-rewrite version set it in a
        copied ``(N, N)`` matrix); pass the current topology only.
    from_idx : int
        Source node key of the proposed connection.
    to_idx : int
        Destination node key of the proposed connection.

    Returns
    -------
    bool
        ``True`` iff the proposed edge would make ``from_idx`` reachable from
        itself, i.e. iff the graph is still acyclic without it but acyclic no
        longer with it:
        ``from_idx == to_idx`` (a self-loop) always returns ``True``, and an
        edge that already exists returns ``False`` for an acyclic graph (it is
        a duplicate, not a new cycle — matching the pre-rewrite behaviour,
        which only OR-ed a bit into the adjacency matrix).
        Evolving an edge into a *recurrent* genome is the caller's decision:
        ``genome.network_type == "recurrent"`` skips this check entirely.

    Raises
    ------
    tensorneat.common.errors.ConfigError
        If ``from_idx`` or ``to_idx`` is not in ``nodes``.

    Notes
    -----
    A genome that is already cyclic returns ``True`` for almost any pair; this
    function answers "does this edge close a cycle", not "is the graph
    acyclic" — use :func:`topological_sort` for the latter. Cost is
    ``O(nodes * conns)`` worst case (iterative closure), which is fine for the
    host-only mutation paths that call it.
    """
    node_set = set(nodes)
    for key, role in ((from_idx, "from_idx"), (to_idx, "to_idx")):
        if key not in node_set:
            raise ConfigError(
                f"{role}={key} is not one of the genome's nodes "
                f"{sorted(node_set)}"
            )

    if from_idx == to_idx:
        # A self-loop closes a cycle by definition (the legacy adjacency
        # matrix answer was ``True`` here as well).
        return True

    # The candidate edge points *into* the closure's seed ``to_idx``, so it can
    # only ever lead back to an already-visited node and never extends the
    # reachable set. The answer is therefore "is ``from_idx`` reachable from
    # ``to_idx`` in the current topology" — which is exactly what the legacy
    # code computed after OR-ing the candidate bit into its matrix. ``conns``
    # is not mutated.
    adjacency = {}
    for in_key, out_key in _unique_edges(conns):
        if in_key in node_set and out_key in node_set:
            adjacency.setdefault(in_key, set()).add(out_key)

    reachable = {to_idx}
    frontier = [to_idx]
    while frontier:
        node = frontier.pop()
        for child in adjacency.get(node, ()):
            if child not in reachable:
                reachable.add(child)
                frontier.append(child)
    return from_idx in reachable


def node_depth(nodes, conns, input_idx):
    """Longest-path depth of every node key, with the inputs pinned at 0.

    Parameters
    ----------
    nodes : set[int] or list[int]
        Node keys of the genome; used to bound the relaxation and not mutated.
    conns : set[tuple[int, int]] or list[tuple[int, int]]
        ``(in_key, out_key)`` connection pairs of a feedforward genome.
    input_idx : iterable of int
        Node keys that start at depth 0 (``genome.input_idx`` /
        ``algorithm`` input keys).

    Returns
    -------
    dict[int, int]
        ``depth[key]`` = length of the longest path from any input node to
        ``key``; an input node itself is ``0``, a node fed only by inputs is
        ``1``, and so on. Every key of ``nodes`` appears exactly once.

    Raises
    ------
    tensorneat.common.errors.GraphError
        If the genome is cyclic, or if some node is not reachable from the
        inputs (the relaxation cannot assign it a depth).
        The in-graph field :attr:`tensorization.Transformed.level` encodes
        exactly those two cases as ``I_INF``; here, on the host, they raise
        instead of propagating a sentinel.
    tensorneat.common.errors.ConfigError
        If ``input_idx`` is empty or names a key absent from ``nodes``.

    Notes
    -----
    **Agreement with the schedule (test hook for ``test/unit/common/``):** for
    a well-formed feedforward genome — acyclic, every node reachable from the
    inputs — this mapping must equal
    ``{int(key): level}`` read off ``Transformed.level`` for the same genome,
    because both are the longest-path depth from the inputs. A mismatch means
    either the schedule or this walk is wrong, never "both are right".
    Recurrent genomes have no such agreement: their cyclic nodes carry
    ``level == I_INF`` in the schedule and raise here.
    """
    node_set = set(nodes)
    inputs = list(input_idx)
    if not inputs:
        raise ConfigError(
            "node_depth needs at least one input node to pin at depth 0"
        )
    for key in inputs:
        if key not in node_set:
            raise ConfigError(
                f"input node {key} is not one of the genome's nodes "
                f"{sorted(node_set)}"
            )
    input_set = set(inputs)

    # Reuse the topological sort for the cycle check (it raises ``GraphError``
    # exactly where this function must: a cycle, or a connection endpoint
    # outside the genome). The depth itself is then one forward pass.
    edges = _unique_edges(conns)
    topo_order, _ = topological_sort(node_set, edges)

    predecessors = {}
    for in_key, out_key in edges:
        predecessors.setdefault(out_key, []).append(in_key)

    depth = {}
    for node in topo_order:
        if node in input_set:
            # Inputs are pinned at 0 even if a recurrent edge points into them.
            depth[node] = 0
            continue
        preds = predecessors.get(node)
        if not preds:
            raise GraphError(
                f"node {node} is not reachable from the input nodes "
                f"{sorted(input_set)}"
            )
        depth[node] = max(depth[pred] for pred in preds) + 1
    return depth
