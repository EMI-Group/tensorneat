"""Host-side graph algorithms for interpretability, visualisation and export.

Dependency DAG position: leaf of ``tensorneat.common``. Imports only the
standard library and :class:`tensorneat.common.errors.GraphError`; no
``evoxc``, no ``numpy``, no optional backend.

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

from .errors import GraphError

__all__ = [
    "decode_graph",
    "topological_sort",
    "find_useful_nodes",
    "check_cycles",
    "node_depth",
]


def decode_graph(nodes, conns, node_attrs=1, conn_attrs=2):
    """Decode one genome's float32 gene payload into plain Python structures.

    This is the only place where the NaN-padded array payload crosses into
    host graph land: everything downstream of it works on ``int`` keys.

    Parameters
    ----------
    nodes : array-like of float, shape ``(max_nodes, node_len)``
        Node gene rows of a **single** genome (no population axis). Column 0
        is the node key; the remaining columns are gene attributes. A row whose
        column 0 is ``NaN`` is unused padding and is skipped.
    conns : array-like of float, shape ``(max_conns, conn_len)``
        Connection gene rows of the same genome. Columns 0 and 1 are the
        ``(in_key, out_key)`` endpoints; the remaining columns are gene
        attributes (weight, marker, ...). A row whose column 0 is ``NaN`` is
        unused padding and is skipped.
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
    Keys are exact: they are integer-valued ``float32`` below
    ``KEY_MAX = 2**24``, so ``int(...)`` round-trips them and ``int`` equality
    is gene-key equality (no hashing of the whole row is involved).
    Duplicate connection pairs (same ``(in_key, out_key)`` twice) collapse into
    one element of the returned set; the schedule is the authority on whether
    that is legal, this module just makes the topology a set.
    """
    raise NotImplementedError(
        "decode one genome's NaN-padded node/conn payload into "
        "{key: attrs} and {(in_key, out_key)}; stdlib only, no tracing"
    )


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
    raise NotImplementedError(
        "Kahn topological sort over Python sets; return (topo_order, "
        "topo_layers) and raise GraphError on a cycle"
    )


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
    raise NotImplementedError(
        "iterative backward set closure from output_idx over Python sets; "
        "returns the set of nodes that reach an output"
    )


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
    raise NotImplementedError(
        "boolean transitive closure over Python sets with the candidate edge "
        "applied; True means the edge would close a cycle"
    )


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
    raise NotImplementedError(
        "longest-path depth from input_idx over Python sets; must agree with "
        "Transformed.level for a feedforward genome"
    )
