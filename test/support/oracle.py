"""Independent numpy references for the algorithms ``src/tensorneat`` implements in evoxc.

Every function here is written from the *documented* algorithm — the docstrings of
``src/tensorneat/common/tensorization.py``, ``src/tensorneat/genome/base.py`` and
``src/tensorneat/problem/`` — never by calling the implementation it checks, so a
test comparing the two is a real comparison instead of a tautology. Plain numpy in,
plain numpy out: no ``evoxc``, no ``jax``, no ``torch``, no network, no GPU, and no
assertion (a caller asserts; this module only computes and raises ``ValueError`` for
input it cannot interpret).

Contents, in dependency order:

* the canonical activation and aggregation kernels (``act_*`` / ``agg_*``) and their
  per-node dispatchers, which live in :mod:`support.functions_ref` and are re-exported
  here so there is one import site;
* :func:`schedule` — the ``Transformed`` execution schedule, derived with a dense
  key→slot table, in-group ranks, the counting trick for cumulative in-degree and a
  longest-path level assignment that reports a node on (or downstream of) a cycle as
  ``I_INF``;
* :func:`forward` — the NaN-aware level-synchronous (feedforward) and
  tick-synchronous (recurrent) forward pass over that schedule;
* :func:`func_fit_loss` — the mse/rmse/mae/mape reduction over the sample **and**
  output axes;
* :func:`cartpole_rollout` (+ :func:`cartpole_params`) — the documented Gym
  ``CartPole-v1`` Euler rollout with the frozen-state-after-death rule.

Dtypes: inputs are the payload's ``float32``/``int32`` arrays; arithmetic is carried
out in ``float64`` so the oracle is the more accurate side of a comparison. Compare
against an ``evoxc`` result with a tolerance (``rtol=1e-5``), never with ``==``.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

import numpy as np

from tensorneat.common.tensorization import Transformed
from tensorneat.common.tools import I_INF

# The canonical ``ACT``/``AGG`` numpy kernels and their dispatchers live in their own
# module (option registries are a different concern from the network algorithms) and
# are re-exported here so a test has one import site.
from .functions_ref import (  # noqa: F401  (re-exported API)
    ACT_FUNCS,
    AGG_FUNCS,
    act_abs,
    act_exp,
    act_identity,
    act_inv,
    act_lelu,
    act_log,
    act_relu,
    act_scaled_sigmoid,
    act_scaled_tanh,
    act_sigmoid,
    act_sin,
    act_tanh,
    agg_max,
    agg_maxabs,
    agg_mean,
    agg_min,
    agg_product,
    agg_sum,
    apply_activation,
    apply_aggregation,
)


__all__ = [
    "Transformed",
    "I_INF",
    "ACT_FUNCS",
    "AGG_FUNCS",
    "act_abs",
    "act_exp",
    "act_identity",
    "act_inv",
    "act_lelu",
    "act_log",
    "act_relu",
    "act_scaled_sigmoid",
    "act_scaled_tanh",
    "act_sigmoid",
    "act_sin",
    "act_tanh",
    "agg_max",
    "agg_maxabs",
    "agg_mean",
    "agg_min",
    "agg_product",
    "agg_sum",
    "apply_activation",
    "apply_aggregation",
    "schedule",
    "forward",
    "default_node_attrs_of",
    "default_conn_attr_of",
    "func_fit_loss",
    "cartpole_params",
    "cartpole_rollout",
    "FUNC_FIT_ERROR_METHODS",
]

#: Loss reductions of ``problem/func_fit`` (``FuncFit.ERROR_METHODS``).
FUNC_FIT_ERROR_METHODS: tuple[str, ...] = ("mse", "rmse", "mae", "mape")


# ================================================================ the schedule
#: ``I_INF`` is ``int32`` max, which a ``float32`` cannot hold exactly (it rounds up
#: to ``2**31``), so an "is this key the padding sentinel" test compares against
#: ``I_INF - 1`` instead of relying on float equality. Real gene keys are below
#: ``tools.KEY_MAX = 2**24``, i.e. three orders of magnitude away from the sentinel.
_I_INF_FLOAT = float(I_INF - 1)


def _present_keys(keys: Any) -> np.ndarray:
    """Mask of the entries that hold a real key rather than the padding sentinel.

    The two sentinel families meet at this boundary: ``GenomeTensors`` pads integer
    key arrays with ``I_INF``, while ``derive_schedule`` documents ``NaN`` for an
    absent row. Both spellings mean "absent" here, and an integer array is compared
    exactly (no float rounding involved).

    Args:
        keys: ``int32``/``float32`` key array of any shape.

    Returns:
        Bool array of ``keys``' shape, ``True`` where the entry is a real key.
    """
    array = np.asarray(keys)
    if array.dtype.kind == "f":
        return np.isfinite(array) & (array < _I_INF_FLOAT)
    return array != I_INF


def _float_array(values: Any, name: str, ndim: int) -> np.ndarray:
    """``values`` as a ``float32`` array of rank ``ndim`` (keys are integer-valued)."""
    array = np.asarray(values)
    if array.dtype.kind not in "fiu":
        raise TypeError(f"{name} must hold real numbers, got dtype {array.dtype}")
    array = array.astype(np.float32 if array.dtype.kind == "f" else np.float64, copy=False)
    if array.ndim != ndim:
        raise ValueError(f"{name} must have rank {ndim}, got shape {array.shape}")
    return array


def _positive_int(value: Any, name: str) -> int:
    """``value`` as a positive Python ``int``."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an int, got {type(value).__name__}")
    if int(value) <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return int(value)


def _key_ints(keys: np.ndarray, present: np.ndarray, key_span: int, name: str) -> np.ndarray:
    """Integer keys of the ``present`` positions, validated against ``[0, key_span)``.

    Args:
        keys: Float key array; the padding sentinel marks an absent entry.
        present: Bool mask of the entries to convert (``_present_keys`` in every call).
        key_span: ``S``, the exclusive key bound of the dense table.
        name: Field name for error messages.

    Returns:
        ``int64`` array of ``keys.shape``; the ``~present`` positions hold ``0``.

    Raises:
        ValueError: If a present key is not integer-valued or not in ``[0, key_span)``
            — a key that cannot fit the table is a configuration error, exactly as
            ``derive_schedule`` refuses it before tracing.
    """
    ints = np.zeros(keys.shape, dtype=np.int64)
    values = np.asarray(keys, dtype=np.float64)[present]
    if values.size:
        rounded = np.rint(values)
        if not np.array_equal(values, rounded):
            raise ValueError(f"{name} holds a non-integer key: keys must be integer-valued")
        if rounded.min() < 0 or rounded.max() >= key_span:
            raise ValueError(
                f"{name} holds a key outside [0, {key_span}): every key must fit key_span"
            )
        ints[present] = rounded.astype(np.int64)
    return ints


def _lookup(table: np.ndarray, keys: np.ndarray, key_span: int, name: str) -> np.ndarray:
    """Resolve keys to node slots through the dense key→slot table.

    Misses (``NaN``) become ``I_INF``. The indices are made safe before the gather
    (``NaN -> 0`` and a clamp to ``[0, key_span-1]``) and the result is then
    overwritten with ``I_INF`` where the key was absent — the same sanitize-gather-mask
    order the evoxc kernel has to use because its numpy backend raises on an
    out-of-range index.
    """
    present = _present_keys(keys)
    ints = _key_ints(keys, present, key_span, name)
    slots = np.take_along_axis(table, np.clip(ints, 0, key_span - 1), axis=1)
    return np.where(present, slots, I_INF).astype(np.int32)


def schedule(
    node_keys: Any,
    conn_in: Any,
    conn_out: Any,
    input_keys: Any,
    output_keys: Any,
    *,
    max_in_degree: int,
    key_span: int,
) -> Transformed:
    """Derive the ``Transformed`` execution schedule of one population (numpy oracle).

    Re-implements ``common.tensorization.derive_schedule`` from its documented
    composition: dense key→slot table, sanitized slot lookup, in-group edge rank,
    the ``cumsum``-free cumulative in-degree, the packed ``(P, N, K)`` incoming table
    and the longest-path levels.

    Args:
        node_keys: ``(P, N)`` node keys; the padding sentinel marks an absent row.
        conn_in: ``(P, C)`` connection source keys; the padding sentinel marks an
            absent row.
        conn_out: ``(P, C)`` connection destination keys; likewise.
        input_keys: ``(I,)`` keys of the input nodes, shared by the batch.
        output_keys: ``(O,)`` keys of the output nodes, shared by the batch.
        max_in_degree: ``K``, the static in-degree window width; a node with more
            live in-edges than ``K`` loses the tail into ``edge_overflow``.
        key_span: ``S``, the exclusive key bound of the dense table; every present key
            must lie in ``[0, S)``.

    The absent-row sentinel may be spelled either way: ``NaN`` (the schedule's
    documented sentinel) or ``I_INF`` (what ``GenomeTensors`` pads the integer key
    arrays with), so a hand-built payload can be fed in unchanged — see
    :func:`_present_keys`.

    Returns:
        ``Transformed`` with ``src``/``dst``/``edge_valid``, the ``(P, N, K)``
        ``in_conn``/``in_src``, ``level``/``topo_key``, ``input_slot``/``output_slot``
        and ``edge_overflow``. Sentinels: ``I_INF`` in every integer field, ``False``
        in ``edge_valid``; ``NaN`` never appears (these are integer/bool arrays).

    Raises:
        ValueError: If a shape disagrees, a key is not integer-valued or does not fit
            ``key_span``, or ``max_in_degree``/``key_span`` is not a positive int.
    """
    node_keys = _float_array(node_keys, "node_keys", 2)
    conn_in = _float_array(conn_in, "conn_in", 2)
    conn_out = _float_array(conn_out, "conn_out", 2)
    key_span = _positive_int(key_span, "key_span")
    max_in_degree = _positive_int(max_in_degree, "max_in_degree")
    pop_size, max_nodes = node_keys.shape
    if conn_in.shape != conn_out.shape:
        raise ValueError(
            f"conn_in and conn_out must have the same shape, got {conn_in.shape} and "
            f"{conn_out.shape}"
        )
    if conn_in.shape[0] != pop_size:
        raise ValueError(
            f"conn_in/conn_out must share the population axis {pop_size}, got "
            f"{conn_in.shape[0]}"
        )
    max_conns = conn_in.shape[1]
    input_keys = _float_array(input_keys, "input_keys", 1)
    output_keys = _float_array(output_keys, "output_keys", 1)
    num_inputs, num_outputs = input_keys.shape[0], output_keys.shape[0]
    slots = np.arange(max_nodes, dtype=np.int32)[None, :].repeat(pop_size, axis=0)
    rows = np.arange(pop_size)[:, None]

    # 1. dense key -> slot table, with a dump column that no lookup ever reads.
    table = np.full((pop_size, key_span + 1), I_INF, dtype=np.int32)
    for p in range(pop_size):
        present = _present_keys(node_keys[p])
        keys = _key_ints(node_keys[p], present, key_span, "node_keys")[present]
        table[p, keys] = np.flatnonzero(present).astype(np.int32)

    # 2. connection endpoints, with an unmatched key invalidating the row.
    src = _lookup(table, conn_in, key_span, "conn_in")
    dst = _lookup(table, conn_out, key_span, "conn_out")
    edge_valid = _present_keys(conn_in) & _present_keys(conn_out) & (src != I_INF) & (dst != I_INF)

    # 3. in-group rank: position of the edge in its destination's in-edge list.
    rank = np.full((pop_size, max_conns), I_INF, dtype=np.int32)
    for p in range(pop_size):
        valid = np.flatnonzero(edge_valid[p])
        if valid.size:
            order = np.argsort(dst[p, valid], kind="stable")
            rank[p, valid[order]] = np.arange(valid.size, dtype=np.int32)

    # 4. cc[i] = number of valid edges whose destination slot is <= i.
    cumulative = np.zeros((pop_size, max_nodes), dtype=np.int32)
    for p in range(pop_size):
        counts = np.bincount(dst[p, edge_valid[p]], minlength=max_nodes)
        cumulative[p] = np.cumsum(counts)

    # 5. pack each node's in-edges into the (P, N, K) window.
    in_conn = np.full((pop_size, max_nodes, max_in_degree), I_INF, dtype=np.int32)
    in_src = np.full((pop_size, max_nodes, max_in_degree), I_INF, dtype=np.int32)
    edge_overflow = np.zeros(pop_size, dtype=np.int32)
    for p in range(pop_size):
        start = np.concatenate(([0], cumulative[p, :-1])).astype(np.int64)
        for conn_row in np.flatnonzero(edge_valid[p]):
            node = int(dst[p, conn_row])
            column = int(rank[p, conn_row]) - int(start[node])
            if 0 <= column < max_in_degree:
                in_conn[p, node, column] = conn_row
                in_src[p, node, column] = src[p, conn_row]
            else:
                edge_overflow[p] += 1

    # 6. longest-path levels, inputs pinned to 0.
    input_slot = _lookup(
        table, np.broadcast_to(input_keys, (pop_size, num_inputs)), key_span, "input_keys"
    )
    output_slot = _lookup(
        table, np.broadcast_to(output_keys, (pop_size, num_outputs)), key_span, "output_keys"
    )
    level = _levels(in_conn, in_src, input_slot, max_nodes)

    # 7. the derived fields.
    finite = level != I_INF
    topo_key = np.where(finite, np.where(finite, level, 0) * max_nodes + slots, I_INF)
    return Transformed(
        src=src,
        dst=dst,
        edge_valid=edge_valid,
        in_conn=in_conn,
        in_src=in_src,
        in_valid=in_conn != I_INF,
        node_level=level.astype(np.int32),
        node_exists=_present_keys(node_keys),
        topo_key=topo_key.astype(np.int32),
        input_slot=input_slot,
        output_slot=output_slot,
        edge_overflow=edge_overflow,
    )


def _levels(
    in_conn: np.ndarray,
    in_src: np.ndarray,
    input_slot: np.ndarray,
    max_nodes: int,
) -> np.ndarray:
    """Longest-path depth of every node: ``max(pred) + 1``, inputs pinned to ``0``.

    The documented contract of ``Transformed.level`` is an *iff*: ``I_INF`` exactly
    for a node that is absent, unreachable from the inputs, or on a cycle. The evoxc
    kernel obtains it by relaxing ``level[i] = max(level[in_src[i, k]] + 1)`` in a
    ``while_loop`` capped at ``N`` iterations, which never settles on a cycle; this
    oracle states the same property directly instead of imitating the loop, because a
    cycle (and everything downstream of it) simply has no finite longest path:

    1. build the node adjacency of the live edges,
    2. take the transitive closure (Warshall over ``N`` nodes): a node is on a cycle
       iff it reaches itself through a non-empty path, and every node reachable from
       a cyclic node is poisoned as well,
    3. relax the levels of the remaining DAG ``N`` times (a simple path has at most
       ``N-1`` edges, so this is exact), then blank the poisoned rows.

    Pinned input rows keep level ``0`` even when they sit on a cycle, matching the
    "inputs are pinned" half of the contract.

    Args:
        in_conn: ``(P, N, K)`` int32 incoming-edge table; ``I_INF`` = padded slot.
        in_src: ``(P, N, K)`` int32 source slots of those edges.
        input_slot: ``(P, I)`` int32 input node slots, pinned to level ``0``.
        max_nodes: ``N``, the slot count and the relaxation cap.

    Returns:
        ``(P, N)`` int32 levels in ``[0, N)``, or ``I_INF``.
    """
    pop_size, nodes, width = in_conn.shape
    levels = np.full((pop_size, nodes), I_INF, dtype=np.int32)
    for p in range(pop_size):
        live = in_conn[p] != I_INF
        adjacency = np.zeros((nodes, nodes), dtype=bool)
        slots = np.broadcast_to(np.arange(nodes)[:, None], live.shape)
        adjacency[in_src[p][live], slots[live]] = True

        reach = adjacency.copy()
        for through in range(nodes):
            reach |= reach[:, through][:, None] & reach[through][None, :]
        on_cycle = np.diagonal(reach).copy()
        poisoned = np.zeros(nodes, dtype=bool)
        if on_cycle.any():
            poisoned = (reach[on_cycle] | np.eye(nodes, dtype=bool)[on_cycle]).any(axis=0)

        pinned = np.zeros(nodes, dtype=bool)
        have_input = input_slot[p] != I_INF
        pinned[input_slot[p][have_input]] = True

        level = np.full(nodes, I_INF, dtype=np.int32)
        level[pinned] = 0
        for _ in range(nodes):
            for node in range(nodes):
                if pinned[node]:
                    continue
                best = -1
                for slot in range(width):
                    if live[node, slot] and level[in_src[p, node, slot]] != I_INF:
                        best = max(best, int(level[in_src[p, node, slot]]) + 1)
                if best >= 0:
                    level[node] = best
        level[poisoned & ~pinned] = I_INF
        levels[p] = level
    return levels


# =================================================================== forward pass
def default_node_attrs_of(node_attrs: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split ``DefaultNode`` attribute columns into the four fields the step needs.

    Column order is the ``attr_specs`` order of
    ``genome/gene/node/default.py::DefaultNode``: ``bias``, ``response``,
    ``aggregation`` (an option **position**), ``activation`` (likewise, ``-1`` =
    identity). A payload with fewer columns (``NA == 3``, the pre-rewrite layout) has
    no activation column: the ids are then ``0``, i.e. option ``0`` of the gene's
    configured lists — which is the documented default of both option tables.

    Args:
        node_attrs: ``(P, N, NA)`` float32 node attribute rows, ``NaN`` padded.

    Returns:
        ``(bias (P, N), response (P, N), aggregation_ids (P, N), activation_ids
        (P, N))``, all ``float64`` copies.
    """
    attrs = np.asarray(node_attrs, dtype=np.float64)
    if attrs.ndim != 3 or attrs.shape[-1] < 2:
        raise ValueError(
            "node_attrs must be (P, N, NA) with at least bias and response columns, "
            f"got {attrs.shape}"
        )
    bias = attrs[..., 0]
    response = attrs[..., 1]
    aggregation_ids = attrs[..., 2] if attrs.shape[-1] >= 3 else np.zeros(attrs.shape[:2])
    activation_ids = attrs[..., 3] if attrs.shape[-1] >= 4 else np.zeros(attrs.shape[:2])
    return bias, response, aggregation_ids, activation_ids


def default_conn_attr_of(conn_attrs: Any) -> np.ndarray:
    """Take the ``weight`` column of ``DefaultConn`` (column ``0``).

    Args:
        conn_attrs: ``(P, E, CA)`` float32 connection attribute rows, ``NaN`` padded.

    Returns:
        ``(P, E)`` float64 weights; ``NaN`` on absent rows, which the edge message
        sanitizes to ``0``.
    """
    attrs = np.asarray(conn_attrs, dtype=np.float64)
    if attrs.ndim != 3 or attrs.shape[-1] < 1:
        raise ValueError(f"conn_attrs must be (P, E, CA) with CA >= 1, got {attrs.shape}")
    return attrs[..., 0]


def _options(option: Any, name: str) -> tuple[Callable[..., Any], ...]:
    """Normalise an activation/aggregation parameter to a tuple of callables."""
    if callable(option):
        return (option,)
    try:
        return tuple(option)
    except TypeError:
        raise TypeError(f"{name} must be a callable or a sequence of callables") from None


def _output_mask(
    transformed: Transformed,
    is_output: Any,
    pop_size: int,
    max_nodes: int,
) -> np.ndarray:
    """The ``(P, N)`` mask of the rows the node gene leaves unactivated."""
    if is_output is None:
        mask = np.zeros((pop_size, max_nodes), dtype=bool)
        slot = np.asarray(transformed.output_slot)
        have = slot != I_INF
        mask[np.broadcast_to(np.arange(pop_size)[:, None], slot.shape)[have], slot[have]] = True
        return mask
    if isinstance(is_output, (int, np.integer)) and not isinstance(is_output, bool):
        count = int(is_output)
        if not 0 < count <= max_nodes:
            raise ValueError(f"is_output count {count} outside [1, {max_nodes}]")
        mask = np.zeros((pop_size, max_nodes), dtype=bool)
        mask[:, max_nodes - count :] = True
        return mask
    mask = np.asarray(is_output, dtype=bool)
    if mask.shape != (pop_size, max_nodes):
        raise ValueError(
            f"is_output must be (P, N) = {(pop_size, max_nodes)}, got {mask.shape}"
        )
    return mask


def forward(
    payload: Any,
    transformed: Transformed,
    inputs: Any,
    *,
    node_attrs_of: Callable[[Any], tuple] | None = None,
    conn_attr_of: Callable[[Any], np.ndarray] | None = None,
    activation: Any = None,
    aggregation: Any = None,
    is_output: Any = None,
    network_type: str = "feedforward",
    activate_time: int = 10,
) -> np.ndarray:
    """The documented NaN-aware forward pass of the whole population, in numpy.

    Both genomes seed ``values`` with ``NaN`` and then advance it one window reduce
    per step (``BaseGenome._node_step``), which is the algorithm re-implemented here:

    1. seed: ``values = NaN``, then write ``inputs`` into the input node slots;
    2. per slot: ``xv = values[in_src]``, ``live = in_valid & ~isnan(xv)``,
       ``xs = where(live, xv, 0)`` — a source that has no value yet must not poison
       the arithmetic through a ``NaN`` cotangent;
    3. ``msg = sanitized(weight) * xs`` (``DefaultConn``'s per-edge message);
    4. ``agg`` = the aggregation selected per node over the live slots (dead slots
       padded with the option's neutral element, ``0.0`` when nothing is live);
       ``z = bias + response * agg``; the activation is applied **except** on output
       rows, which are read out unactivated;
    5. ``values = where(update_mask & (live_count > 0), z, values)`` — a node with no
       live incoming edge keeps its previous value, so ``NaN`` stays ``NaN`` and its
       descendants lose their only live source (the legacy ``NaN`` poisoning).

    ``network_type`` selects the schedule of those steps:

    * ``"feedforward"`` — level-synchronous: one step per level ``1 .. max(level)``,
      with ``update_mask = (level == l) & exists & (valid_count > 0)``. Equivalent to
      ``DefaultGenome.forward`` for any ``max_depth >= max(level)``.
    * ``"recurrent"`` — ``activate_time`` synchronous ticks, each re-injecting the raw
      ``inputs`` into the input slots and then stepping every existing row with
      ``update_mask = exists & (valid_count > 0)``; every node reads the *previous*
      tick's value (Jacobi, no in-place ordering). ``input_transform`` is never
      applied — legacy parity, and what HyperNEAT relies on when it uses this genome
      as a CPPN.

    Args:
        payload: A ``GenomeTensors``-shaped batch (see :class:`support.payloads.Payload`)
            providing ``node_keys``, ``node_attrs`` and ``conn_attrs``.
        transformed: The schedule of the same population (``in_conn``/``in_src``/
            ``input_slot``/``output_slot``/``level``).
        inputs: ``(P, num_inputs)`` network inputs, already carrying any
            ``input_transform`` (the feedforward genome applies it before seeding;
            the recurrent genome never applies it).
        node_attrs_of: ``(P, N, NA) -> (bias, response, aggregation_ids,
            activation_ids)``; defaults to :func:`default_node_attrs_of`.
        conn_attr_of: ``(P, E, CA) -> (P, E)`` weights; defaults to
            :func:`default_conn_attr_of`.
        activation: A plain callable ``(z) -> z`` applied to every non-output row, or a
            sequence of callables dispatched by the stored activation id (``-1`` =
            identity); defaults to ``act_sigmoid``, the ``DefaultNode`` default option
            at position ``0``.
        aggregation: A plain callable ``(z, mask) -> agg`` or a sequence dispatched by
            the stored aggregation id; defaults to ``agg_sum``, the ``DefaultNode``
            default option at position ``0``.
        is_output: ``(P, N)`` bool mask, an ``int`` count of trailing output rows, or
            ``None`` to derive it from ``transformed.output_slot``.
        network_type: ``"feedforward"`` or ``"recurrent"``.
        activate_time: Number of synchronous ticks (``"recurrent"`` only).

    Returns:
        ``(P, num_outputs)`` float64 outputs, read at ``transformed.output_slot``;
        ``NaN`` for an individual whose output node never got a value.

    Raises:
        ValueError: If shapes disagree with the schedule, an option id is out of
            range, or ``network_type`` is unknown.
    """
    node_keys = np.asarray(payload.node_keys)
    node_attrs = np.asarray(payload.node_attrs)
    conn_attrs = np.asarray(payload.conn_attrs)
    node_attrs_of = default_node_attrs_of if node_attrs_of is None else node_attrs_of
    conn_attr_of = default_conn_attr_of if conn_attr_of is None else conn_attr_of
    act_fns = _options(act_sigmoid if activation is None else activation, "activation")
    agg_fns = _options(agg_sum if aggregation is None else aggregation, "aggregation")

    in_conn = np.asarray(transformed.in_conn)
    in_src = np.asarray(transformed.in_src)
    level = np.asarray(transformed.level)
    pop_size, max_nodes, width = in_conn.shape
    if node_keys.shape != (pop_size, max_nodes):
        raise ValueError(
            f"payload node_keys {node_keys.shape} does not match the schedule "
            f"{(pop_size, max_nodes)}"
        )
    live_slot = in_conn != I_INF
    if not np.array_equal(live_slot, in_src != I_INF):
        raise ValueError("in_conn and in_src must be I_INF in exactly the same positions")
    if live_slot.any():
        num_conn_rows = np.asarray(conn_attrs).shape[1]
        if int(in_conn[live_slot].max()) >= num_conn_rows:
            raise ValueError("in_conn indexes a connection row outside conn_attrs")

    exists = _present_keys(node_keys)
    batch = np.arange(pop_size)[:, None]
    slot_rows = np.arange(pop_size)[:, None, None]
    safe_src = np.where(live_slot, in_src, 0)
    weights = np.asarray(conn_attr_of(conn_attrs), dtype=np.float64)
    if weights.shape != (pop_size, np.asarray(conn_attrs).shape[1]):
        raise ValueError(
            "conn_attr_of must return (P, E) weights, got "
            f"{weights.shape} for conn_attrs {np.asarray(conn_attrs).shape}"
        )
    slot_weights = weights[slot_rows, np.where(live_slot, in_conn, 0)]
    slot_weights = np.where(live_slot, slot_weights, np.nan)

    bias, response, aggregation_ids, activation_ids = node_attrs_of(node_attrs)
    out_mask = _output_mask(transformed, is_output, pop_size, max_nodes)
    input_slot = np.asarray(transformed.input_slot)
    output_slot = np.asarray(transformed.output_slot)
    valid_count = live_slot.sum(axis=-1)

    def node_step(values: np.ndarray, update_mask: np.ndarray) -> np.ndarray:
        """One window reduce of ``_node_step``: ``values`` -> ``values``."""
        source_values = values[slot_rows, safe_src]
        live = live_slot & ~np.isnan(source_values)
        sanitized = np.where(live, source_values, 0.0)
        messages = sanitized * np.where(np.isnan(slot_weights), 0.0, slot_weights)
        aggregated = apply_aggregation(messages, live, aggregation_ids, agg_fns)
        updated = bias + response * aggregated
        activated = apply_activation(updated, activation_ids, act_fns)
        updated = np.where(out_mask, updated, activated)
        live_count = live.sum(axis=-1)
        return np.where(update_mask & (live_count > 0), updated, values)

    def seed(values: np.ndarray) -> np.ndarray:
        """Write ``inputs`` into the input node slots (rows and keys agree)."""
        have_input = input_slot != I_INF
        seed_rows = np.broadcast_to(batch, input_slot.shape)
        values[seed_rows[have_input], input_slot[have_input]] = np.asarray(
            inputs, dtype=np.float64
        )[have_input]
        return values

    values = np.full((pop_size, max_nodes), np.nan, dtype=np.float64)
    if network_type == "feedforward":
        values = seed(values)
        finite = level != I_INF
        max_level = int(level[finite].max()) if finite.any() else 0
        for current in range(1, max_level + 1):
            update_mask = (level == current) & exists & (valid_count > 0)
            values = node_step(values, update_mask)
    elif network_type == "recurrent":
        ticks = _positive_int(activate_time, "activate_time")
        for _ in range(ticks):
            values = seed(values)
            values = node_step(values, exists & (valid_count > 0))
    else:
        raise ValueError(
            f"network_type must be 'feedforward' or 'recurrent', got {network_type!r}"
        )

    have_output = output_slot != I_INF
    outputs = values[batch, np.where(have_output, output_slot, 0)]
    return np.where(have_output, outputs, np.nan)


# ================================================================ function fitting
def func_fit_loss(pred: Any, targets: Any, error_method: str = "mse") -> np.ndarray:
    """Per-individual function-fitting loss, reduced over the sample and output axes.

    Re-implements ``problem/func_fit``'s ``_loss``: for one individual the legacy
    prediction is ``(S, O)`` and ``jnp.mean`` reduces over *both* axes, so the batched
    form is ``np.mean(..., axis=(-2, -1))`` — a mean over samples **and** outputs, not
    two chained means of different denominators.

    Args:
        pred: ``(P, S, O)`` float predictions, one row block per individual.
        targets: ``(S, O)`` float targets, shared by the population.
        error_method: One of :data:`FUNC_FIT_ERROR_METHODS`.

    Returns:
        ``(P,)`` float64 loss, **not** negated: ``mse = mean(err**2)``,
        ``rmse = sqrt(mse)``, ``mae = mean(|err|)``, ``mape = mean(|err / targets|)``.

    Raises:
        ValueError: If the shapes disagree or ``error_method`` is unknown.

    Notes:
        ``mape`` divides by the target, so a zero target yields ``inf`` and a ``NaN``
        prediction yields ``NaN`` — both are the legacy behaviour (the problem then
        negates the loss and maps ``NaN`` fitness to ``-inf``). The numpy warnings for
        those two documented cases are silenced locally, nothing else is.
    """
    pred = np.asarray(pred, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    if pred.ndim != 3:
        raise ValueError(f"pred must be (P, S, O), got shape {pred.shape}")
    if targets.shape != pred.shape[1:]:
        raise ValueError(
            f"targets must be (S, O) = {pred.shape[1:]}, got shape {targets.shape}"
        )
    if error_method not in FUNC_FIT_ERROR_METHODS:
        raise ValueError(
            f"error_method must be one of {FUNC_FIT_ERROR_METHODS}, got {error_method!r}"
        )
    err = pred - targets[None, ...]
    with np.errstate(divide="ignore", invalid="ignore"):
        if error_method == "mse":
            return np.mean(err**2, axis=(-2, -1))
        if error_method == "rmse":
            return np.sqrt(np.mean(err**2, axis=(-2, -1)))
        if error_method == "mae":
            return np.mean(np.abs(err), axis=(-2, -1))
        return np.mean(np.abs(err / targets[None, ...]), axis=(-2, -1))


# ======================================================================== cartpole
def cartpole_params(
    *,
    gravity: float = 9.8,
    masscart: float = 1.0,
    masspole: float = 0.1,
    length: float = 0.5,
    force_mag: float = 10.0,
    dt: float = 0.02,
    x_threshold: float = 2.4,
    theta_threshold: float = 12.0,
    reset_scale: float = 0.05,
) -> dict[str, float]:
    """The documented Gym ``CartPole-v1`` physics bundle used by :func:`cartpole_rollout`.

    Args:
        gravity: Gravity constant.
        masscart: Cart mass.
        masspole: Pole mass.
        length: **Half**-pole length.
        force_mag: Magnitude of the applied force.
        dt: Euler integration step.
        x_threshold: ``|x|`` limit of the episode.
        theta_threshold: ``|theta|`` limit in **degrees** (Gym's convention); the
            returned bundle holds it in radians.
        reset_scale: Half-width of the uniform reset distribution.

    Returns:
        A dict with ``gravity``, ``masscart``, ``masspole``, ``polemass_length``
        (``masspole * length``), ``total_mass`` (``masscart + masspole``),
        ``length``, ``force_mag``, ``dt``, ``x_threshold``, ``theta_threshold``
        (radians) and ``reset_scale``, all Python floats.

    Raises:
        ValueError: If a mass/length/dt/force is not positive, or a threshold or
            ``reset_scale`` is negative.
    """
    for name, value in (
        ("gravity", gravity),
        ("masscart", masscart),
        ("masspole", masspole),
        ("length", length),
        ("force_mag", force_mag),
        ("dt", dt),
        ("theta_threshold", theta_threshold),
    ):
        if not value > 0:
            raise ValueError(f"{name} must be positive, got {value}")
    for name, value in (("x_threshold", x_threshold), ("reset_scale", reset_scale)):
        if value < 0:
            raise ValueError(f"{name} must be non-negative, got {value}")
    return {
        "gravity": float(gravity),
        "masscart": float(masscart),
        "masspole": float(masspole),
        "polemass_length": float(masspole * length),
        "total_mass": float(masscart + masspole),
        "length": float(length),
        "force_mag": float(force_mag),
        "dt": float(dt),
        "x_threshold": float(x_threshold),
        "theta_threshold": float(np.deg2rad(theta_threshold)),
        "reset_scale": float(reset_scale),
    }


def _euler_step(obs: np.ndarray, force: np.ndarray, params: Mapping[str, float]) -> np.ndarray:
    """One documented Euler step of the ``(B, 4)`` CartPole observations.

    With ``c = cos(theta)``, ``s = sin(theta)`` and
    ``temp = (force + polemass_length * theta_dot**2 * s) / total_mass``::

        thetaacc = (gravity * s - c * temp) / (length * (4/3 - masspole * c**2 / total_mass))
        xacc     = temp - polemass_length * thetaacc * c / total_mass
        x'       = x + dt * x_dot          x_dot'     = x_dot + dt * xacc
        theta'   = theta + dt * theta_dot   theta_dot' = theta_dot + dt * thetaacc

    Args:
        obs: ``(B, 4)`` current observations ``(x, x_dot, theta, theta_dot)``.
        force: ``(B, 1)`` applied force.
        params: The bundle from :func:`cartpole_params` (``theta`` in radians).

    Returns:
        ``(B, 4)`` next observations.
    """
    x, x_dot, theta, theta_dot = (obs[:, index] for index in range(4))
    cos = np.cos(theta)
    sin = np.sin(theta)
    temp = (
        force[:, 0] + params["polemass_length"] * theta_dot**2 * sin
    ) / params["total_mass"]
    thetaacc = (params["gravity"] * sin - cos * temp) / (
        params["length"] * (4.0 / 3.0 - params["masspole"] * cos**2 / params["total_mass"])
    )
    xacc = temp - params["polemass_length"] * thetaacc * cos / params["total_mass"]
    return np.stack(
        (
            x + params["dt"] * x_dot,
            x_dot + params["dt"] * xacc,
            theta + params["dt"] * theta_dot,
            theta_dot + params["dt"] * thetaacc,
        ),
        axis=1,
    )


def cartpole_rollout(
    actions: Any,
    obs0: Any,
    *,
    max_step: int = 500,
    params: Mapping[str, float] | None = None,
) -> np.ndarray:
    """Per-episode return of the documented Cartesian-pole rollout.

    Re-implements ``problem/rl/cartpole.py``'s scan body, step by step: reward the
    still-alive episodes **before** the transition (the failing step still scores, as
    in Gym), apply the Euler step, mark death on ``|x| > x_threshold`` or
    ``|theta| > theta_threshold`` (radians), and then freeze the state with the
    previous observation so the dead episode keeps producing the same (irrelevant)
    dynamics instead of needing a ``cond``.

    Because the state is frozen and the reward stops on death, an episode's return is
    exactly its survival time: an integer in ``[0, max_step]``, which is why
    ``max_step`` is a hard fitness ceiling.

    Args:
        actions: ``(B, max_step)`` discrete decisions, ``1`` = push right (``+force_mag``)
            and ``0`` = push left (a tie pushes left, as ``argmax`` index ``0`` would).
        obs0: ``(B, 4)`` initial observations ``(x, x_dot, theta, theta_dot)``.
        max_step: Fixed horizon ``T``, also the fitness ceiling; ``actions`` must have
            exactly ``T`` columns.
        params: The physics bundle from :func:`cartpole_params`; ``None`` uses its
            defaults.

    Returns:
        ``(B,)`` float64 returns — the summed reward of each episode.

    Raises:
        ValueError: If a shape is wrong, ``max_step`` is not positive, or the bundle is
            missing a documented key.
    """
    obs0 = np.asarray(obs0, dtype=np.float64)
    if obs0.ndim != 2 or obs0.shape[1] != 4:
        raise ValueError(f"obs0 must be (B, 4), got shape {obs0.shape}")
    episodes = obs0.shape[0]
    horizon = _positive_int(max_step, "max_step")
    actions = np.asarray(actions)
    if actions.ndim != 2 or actions.shape != (episodes, horizon):
        raise ValueError(
            f"actions must be (B, max_step) = {(episodes, horizon)}, got {actions.shape}"
        )
    constants = cartpole_params() if params is None else dict(params)
    missing = {
        "gravity",
        "polemass_length",
        "total_mass",
        "masspole",
        "length",
        "force_mag",
        "dt",
        "x_threshold",
        "theta_threshold",
    } - set(constants)
    if missing:
        raise ValueError(f"params is missing {sorted(missing)}; use cartpole_params()")

    obs = obs0.copy()
    alive = np.ones(episodes, dtype=bool)
    returns = np.zeros(episodes, dtype=np.float64)
    for step in range(horizon):
        returns = returns + alive.astype(np.float64)
        force = np.where(
            actions[:, step : step + 1] > 0, constants["force_mag"], -constants["force_mag"]
        )
        next_obs = _euler_step(obs, force, constants)
        dead = (np.abs(next_obs[:, 0]) > constants["x_threshold"]) | (
            np.abs(next_obs[:, 2]) > constants["theta_threshold"]
        )
        alive = alive & ~dead
        obs = np.where(alive[:, None], next_obs, obs)
    return returns
