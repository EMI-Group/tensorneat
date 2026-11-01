"""The exact gradient path: a dense-weight twin of the window forward pass.

``BaseGenome.grad`` returns an :class:`evoxc.Graph`; this module builds it.  The differentiated
pass is :func:`dense_forward`, never the window step of ``base.BaseGenome._node_step``: evoxc
0.1.0 derives a ``gather`` VJP with *put* semantics (``g_x = scatter(zeros_like(x), idx, ct)``,
``evoxc/transforms/rules.py``), so a node value read by several incoming-edge slots keeps only
one of their cotangents and the gradient is wrong wherever a node fans out.  A per-individual
dense weight matrix plus a batched ``dot`` has an exact transpose rule, so the gradient is
exact even though the ops differ from the plain forward.

Nothing here is per-generation work beyond one Python pass over the schedule:
:func:`dense_grad_plan` derives the host statics once per compiled graph, and the graph itself
is cached by the Pipeline per shape signature like any other genome graph.

A genome object is passed in rather than mixed in: these functions read its public statics
(``num_inputs``, ``max_in_degree``, ``max_depth``, ``network_type``), its genes and transforms,
plus the two output helpers ``_output_row_mask``/``_output_values`` that stay in ``base.py``
because the plain forward uses them too.  ``base.py`` imports this module; this module never
imports ``base.py`` at runtime.

Every kernel is batch-first (population on axis 0) and static in shape; the level loop is a
plain Python ``for`` over the plan's level masks, never ``scan``/``while_loop``/``cond``,
because a region op cannot be differentiated.
"""

from typing import TYPE_CHECKING, Any, NamedTuple

import evoxc
import evoxc.numpy as enp
import numpy as np

from ..common import (
    AGG,
    I_INF,
    ShapeBoundError,
    State,
    Transformed,
    apply_activation,
    to_numpy,
    validate_schedule,
)
from ..common.errors import ConfigError

if TYPE_CHECKING:  # pragma: no cover - typing only; keeps base -> grad a one-way import
    from .base import BaseGenome


#: numpy dtype -> evoxc dtype, for the graph-factory input specs of
#: :func:`build_grad`. A tensor that is already an evoxc tensor carries its dtype verbatim
#: and misses this table.
_NUMPY_DTYPES = {
    np.dtype("float32"): evoxc.float32,
    np.dtype("float64"): evoxc.float64,
    np.dtype("int32"): evoxc.int32,
    np.dtype("int64"): evoxc.int64,
    np.dtype("bool"): evoxc.bool_,
}


class _DenseGradPlan(NamedTuple):
    """Host statics of the exact (dense-weight) gradient path.

    Built once per :func:`build_grad` call from an already-derived schedule and baked into
    the traced graph as constants; nothing here is per-generation tensor data.

    Attributes:
        mean: ``True`` when the node gene aggregates with ``AGG.mean`` — the dense sum is then
            divided by the per-node live slot count.
        cell_index: ``(P * E,)`` flat index of every connection row into a padded
            ``(P, N * N + 1)`` weight matrix. A used row targets ``p * (N * N + 1) + dst * N +
            src``, an unused row the per-individual dump cell ``N * N`` (sliced off again), so
            the two never collide and the scatter writes each ``(destination, source)`` cell at
            most once.
        edge_used: ``(P, E) bool``, True where the connection row is a real edge *and* the
            schedule placed it in some node's incoming window (an edge that did not fit the
            static ``K`` is refused by :func:`dense_grad_plan` instead).
        live_matrix: ``(P, N, N) float32``, ``1.0`` on every ``(destination, source)`` pair the
            window holds: a weight-independent structural mask, so a zero weight still counts
            as a live slot.
        input_rows: ``(P, N) bool``, True on the input rows ``0..num_inputs-1``, the only rows
            with a value before level 1.
        level_masks: One ``(P, N) bool`` per unrolled level ``1..min(max finite node_level,
            max_depth)``: the static part of the update mask (``node_level == level`` &
            ``node_exists`` & "the row has an incoming connection"); the dynamic "has a live
            slot" term is ANDed in per step. Deeper levels leave the values unchanged, so the
            shorter tuple reproduces the ``1..max_depth`` sweep exactly.
    """

    mean: bool
    cell_index: np.ndarray
    edge_used: np.ndarray
    live_matrix: np.ndarray
    input_rows: np.ndarray
    level_masks: tuple


def _graph_constant(value: Any) -> evoxc.SymbolicTensor:
    """Embed one host array as an in-graph constant.

    Args:
        value: ``evoxc.Tensor``, ``numpy`` array or anything ``to_numpy`` unwraps.

    Returns:
        The value as a traced constant of the same shape/dtype.
    """
    if isinstance(value, evoxc.Tensor):
        return evoxc.constant(value)
    array = np.ascontiguousarray(to_numpy(value))
    return evoxc.constant(evoxc.from_numpy(array))


def _tensor_spec(value: Any) -> evoxc.TensorSpec:
    """Build the ``TensorSpec`` of one concrete tensor.

    Args:
        value: ``numpy`` array, ``evoxc.Tensor`` or traced tensor.

    Returns:
        ``TensorSpec`` with the value's static shape and ``evoxc`` dtype.
    """
    dtype = value.dtype
    return evoxc.TensorSpec(
        tuple(int(size) for size in value.shape), _NUMPY_DTYPES.get(dtype, dtype)
    )


def _sanitize_attrs(
    attrs: evoxc.SymbolicTensor, exists: evoxc.SymbolicTensor
) -> evoxc.SymbolicTensor:
    """Mask padding rows and clean NaN out of one attribute tensor.

    Applied once at the boundary of :func:`build_grad`, so nothing NaN-valued ever enters a
    differentiable op: the reverse sweep then produces a finite gradient that is exactly
    ``0`` on absent rows.

    Args:
        attrs: ``(P, R, W) float32`` attribute rows, NaN padded.
        exists: ``(P, R) bool`` row-validity mask (``node_exists``/``conn_exists``).

    Returns:
        ``(P, R, W) float32``: the attribute rows of existing, non-NaN entries, else ``0``.
    """
    shape = tuple(attrs.shape)
    if not shape[-1]:  # a family without evolvable attributes
        return attrs
    rows = enp.broadcast_to(
        evoxc.reshape(exists, (*shape[:-1], 1)), shape
    )
    keep = evoxc.logical_and(rows, evoxc.logical_not(evoxc.isnan(attrs)))
    return evoxc.select(keep, attrs, enp.zeros((), dtype=evoxc.float32))


def dense_grad_plan(genome: "BaseGenome", transformed: Transformed) -> _DenseGradPlan:
    """Derive and validate the host statics of the exact gradient path.

    Everything the dense pass needs beyond the payload — the scatter cell of every window
    slot, the structural live-edge matrix, the input-row mask and the per-level update
    masks — follows from the schedule alone, so it is computed here once, on the host, and
    later embedded as graph constants (:func:`_graph_constant`).

    The three configuration refusals below are not conservative guesses: each one names a
    case where the dense pass would silently disagree with
    :meth:`base.BaseGenome.forward`.

    Args:
        genome: The genome whose configuration and genes the pass reads.
        transformed: The schedule of the population, as host data.

    Returns:
        :class:`_DenseGradPlan` for :func:`dense_forward`.

    Raises:
        ConfigError: If the genome is not ``network_type == "feedforward"`` (the dense pass
            is the level-synchronous sweep), if ``node_gene.aggregation_options`` is not
            exactly one ``AGG.sum`` or ``AGG.mean`` (only those two are a ``dot``), if the
            connection gene declares no ``weight`` attribute column (the dense matrix is
            built from it), if the schedule violates its documented invariants, or if one
            individual has two connections sharing a ``(destination, source)`` pair (the
            scatter would keep one weight).
        ShapeBoundError: If an individual has an edge the schedule could not place in any
            incoming window, i.e. ``max_in_degree`` is too small for its in-degree: the
            dense matrix has no cell for it, so the host must rebuild before
            differentiating.
    """
    if genome.network_type != "feedforward":
        raise ConfigError(
            f"{type(genome).__name__}.grad differentiates the level-synchronous "
            f"feedforward sweep with a dense per-individual weight matrix; this genome's "
            f"network_type is {genome.network_type!r}"
        )
    options = tuple(genome.node_gene.aggregation_options)
    supported = (AGG.sum, AGG.mean)
    offender = next(
        (option for option in options if not any(option is fn for fn in supported)),
        None,
    )
    if offender is not None or len(options) != 1:
        names = ", ".join(
            getattr(option, "__name__", repr(option)) for option in options
        )
        raise ConfigError(
            f"{type(genome).__name__}.grad aggregates through a batched matmul, which "
            "carries only AGG.sum and AGG.mean; the offending option is "
            f"{offender!r} and node_gene.aggregation_options is ({names})"
        )
    try:
        genome.conn_gene.attr_index("weight")
    except ConfigError as error:
        raise ConfigError(
            f"{type(genome).__name__}.grad builds the dense weight matrix from the "
            f"connection gene's 'weight' column: {error}"
        ) from None

    validate_schedule(transformed)
    in_conn = np.asarray(to_numpy(transformed.in_conn))
    in_src = np.asarray(to_numpy(transformed.in_src))
    in_valid = np.asarray(to_numpy(transformed.in_valid))
    node_level = np.asarray(to_numpy(transformed.node_level))
    node_exists = np.asarray(to_numpy(transformed.node_exists))
    src = np.asarray(to_numpy(transformed.src))
    dst = np.asarray(to_numpy(transformed.dst))
    edge_valid = np.asarray(to_numpy(transformed.edge_valid))
    pop_size, max_nodes = (int(size) for size in in_valid.shape[:2])

    # One weight per (destination, source) cell: two connections sharing a pair would
    # make the scatter keep only one of the two weights, so refuse instead of drifting.
    cells = np.where(edge_valid, dst.astype(np.int64) * max_nodes + src, -1)
    ordered = np.sort(cells, axis=1)
    duplicated = ordered[:, 1:] == ordered[:, :-1]
    if np.any(duplicated & (ordered[:, 1:] >= 0)):
        member = int(np.argwhere(duplicated & (ordered[:, 1:] >= 0))[0, 0])
        raise ConfigError(
            f"{type(genome).__name__}.grad needs one weight per (destination, source) "
            f"pair; individual {member} carries two connections with the same endpoints"
        )

    padded_cells = max_nodes * max_nodes + 1
    in_window = np.zeros((pop_size, dst.shape[1]), dtype=np.bool_)
    members, sources, slots = np.nonzero(in_valid)
    in_window[members, in_conn[members, sources, slots]] = True
    if np.any(edge_valid & ~in_window):
        member = int(np.argwhere(edge_valid & ~in_window)[0, 0])
        raise ShapeBoundError(
            f"{type(genome).__name__}.grad: individual {member} has an edge outside every "
            f"incoming window (max_in_degree={genome.max_in_degree} is too small for its "
            "in-degree); rebuild the schedule with a larger max_in_degree before "
            "differentiating"
        )
    used = edge_valid & in_window
    edge_cell = np.where(
        used, dst.astype(np.int64) * max_nodes + src, max_nodes * max_nodes
    )
    edge_cell = (
        np.arange(pop_size, dtype=np.int64)[:, None] * padded_cells + edge_cell
    ).astype(np.int32).reshape(-1)

    live_matrix = np.zeros((pop_size, max_nodes, max_nodes), dtype=np.float32)
    live_matrix[members, sources, in_src[members, sources, slots]] = 1.0

    input_rows = np.zeros((pop_size, max_nodes), dtype=np.bool_)
    input_rows[:, : genome.num_inputs] = True

    finite = node_level != I_INF
    deepest = int(node_level[finite].max()) if finite.any() else 0
    has_slot = in_valid.any(axis=-1)
    return _DenseGradPlan(
        mean=options[0] is AGG.mean,
        cell_index=edge_cell,
        edge_used=used,
        live_matrix=live_matrix,
        input_rows=input_rows,
        level_masks=tuple(
            (node_level == level) & node_exists & has_slot
            for level in range(1, min(deepest, genome.max_depth) + 1)
        ),
    )


def dense_forward(
    genome: "BaseGenome",
    state: State,
    plan: _DenseGradPlan,
    node_attrs: evoxc.SymbolicTensor,
    conn_attrs: evoxc.SymbolicTensor,
    inputs: evoxc.SymbolicTensor,
) -> evoxc.SymbolicTensor:
    """Evaluate the population with a dense per-individual weight matrix (grad only).

    Same algebra as the level-synchronous :meth:`base.BaseGenome.forward` of a feedforward
    genome, with the incoming-window gather replaced by one matmul, so that reverse mode is
    exact despite evoxc 0.1.0's put-semantics ``gather`` transpose:

    1. ``W (P, N, N) float32`` is scattered once from the connection gene's ``weight``
       column: ``W[p, dst, src] = weight`` for every edge the schedule placed in a window.
       The ``(destination, source)`` pairs are unique, so both the scatter and its
       transpose (a ``gather`` at unique indices) are exact; unused rows write into a
       per-individual dump cell that is sliced off again.
    2. ``values = NaN``, then ``input_transform(inputs)`` on the input rows — the seeding of
       the plain forward.
    3. per unrolled level: ``xs = select(isnan(values), 0, values)``;
       ``agg = dot(W, xs)`` (the ``AGG.sum`` window reduction); for ``AGG.mean`` divide by
       ``dot(M, ~isnan(values))``, the per-node live slot count, floored at 1; then
       ``node_gene.combine`` + ``apply_activation`` except on output rows; and
       ``values = select(level_mask & (live_count > 0), z, values)`` — a node without a live
       source keeps ``NaN``, so the legacy NaN poisoning is preserved node for node.

    Args:
        genome: The genome whose genes and transforms the pass reads.
        state: Host configuration container.
        plan: Host statics from :func:`dense_grad_plan`, embedded as graph constants.
        node_attrs: ``(P, N, NA) float32`` node attributes, padding already cleaned to 0.
        conn_attrs: ``(P, E, CA) float32`` connection attributes, padding already cleaned
            to 0.
        inputs: ``(P, num_inputs) float32`` network inputs.

    Returns:
        ``(P, num_outputs) float32`` outputs after ``output_transform``, equal to
        :meth:`base.BaseGenome.forward`'s for the same payload up to float32 summation order.
    """
    pop_size, max_nodes = plan.input_rows.shape
    max_conns = int(conn_attrs.shape[1])
    padded_cells = max_nodes * max_nodes + 1
    zero = enp.zeros((), dtype=evoxc.float32)
    weight = genome.conn_gene.attr_column(conn_attrs, "weight")
    # An unused row writes the neutral weight into the dump cell, so no second mask op is
    # needed on the (P, E) update vector.
    updates = evoxc.select(_graph_constant(plan.edge_used), weight, zero)
    scattered = evoxc.scatter(
        enp.zeros((pop_size * padded_cells,), dtype=evoxc.float32),
        _graph_constant(plan.cell_index),
        evoxc.reshape(updates, (pop_size * max_conns,)),
    )
    weights = evoxc.reshape(
        evoxc.reshape(scattered, (pop_size, padded_cells))[
            (slice(None), slice(0, max_nodes * max_nodes))
        ],
        (pop_size, max_nodes, max_nodes),
    )

    seeded = (
        inputs if genome.input_transform is None else genome.input_transform(inputs)
    )
    if max_nodes > genome.num_inputs:
        seeded = evoxc.concatenate(
            [
                seeded,
                enp.zeros(
                    (pop_size, max_nodes - genome.num_inputs), dtype=evoxc.float32
                ),
            ],
            axis=1,
        )
    values = evoxc.select(
        _graph_constant(plan.input_rows),
        seeded,
        enp.full((pop_size, max_nodes), float("nan"), dtype=evoxc.float32),
    )

    live_matrix = _graph_constant(plan.live_matrix)
    for level_mask in plan.level_masks:
        values = dense_node_step(
            genome,
            state,
            weights,
            live_matrix,
            node_attrs,
            values,
            _graph_constant(level_mask),
            plan.mean,
        )
    return genome._output_values(values, genome.output_transform)


def dense_node_step(
    genome: "BaseGenome",
    state: State,
    weights: evoxc.SymbolicTensor,
    live_matrix: evoxc.SymbolicTensor,
    node_attrs: evoxc.SymbolicTensor,
    values: evoxc.SymbolicTensor,
    update_mask: evoxc.SymbolicTensor,
    mean: bool,
) -> evoxc.SymbolicTensor:
    """Advance the node values by one dense level (the matmul twin of
    ``BaseGenome._node_step``).

    Args:
        genome: The genome whose node gene reads the attributes.
        state: Host configuration container.
        weights: ``(P, N, N) float32`` dense edge-weight matrix of this population.
        live_matrix: ``(P, N, N) float32`` structural mask of the same shape, ``1.0`` on
            every edge the window holds.
        node_attrs: ``(P, N, NA) float32`` node attributes, padding already cleaned to 0.
        values: ``(P, N) float32`` current node values, NaN where not yet computed.
        update_mask: ``(P, N) bool`` static level mask; the dynamic "has a live source"
            term is ANDed in here.
        mean: ``True`` to divide the aggregated sum by the live slot count.

    Returns:
        ``(P, N) float32`` updated values; rows masked off are returned unchanged.
    """
    pop_size, max_nodes = (int(size) for size in values.shape)
    zero = enp.zeros((), dtype=evoxc.float32)
    one = enp.full((), 1.0, dtype=evoxc.float32)
    missing = evoxc.isnan(values)
    # NaN sources must read as 0 before the product: a zeroed cotangent times a NaN local
    # derivative would still be NaN in the reverse sweep.
    xs = evoxc.select(missing, zero, values)
    aggregated = evoxc.reshape(
        evoxc.dot(weights, evoxc.reshape(xs, (pop_size, max_nodes, 1))),
        (pop_size, max_nodes),
    )
    live = evoxc.select(missing, zero, one)
    live_count = evoxc.reshape(
        evoxc.dot(live_matrix, evoxc.reshape(live, (pop_size, max_nodes, 1))),
        (pop_size, max_nodes),
    )
    if mean:
        aggregated = evoxc.divide(aggregated, evoxc.maximum(live_count, one))
    z = genome.node_gene.combine(state, node_attrs, aggregated)
    activated = apply_activation(
        z,
        genome.node_gene.attr_column(node_attrs, "activation"),
        genome.node_gene.activation_options,
    )
    z = evoxc.select(genome._output_row_mask(pop_size, max_nodes), z, activated)
    update = evoxc.logical_and(update_mask, evoxc.greater(live_count, zero))
    return evoxc.select(update, z, values)


def build_grad(
    genome: "BaseGenome",
    state: State,
    transformed: Transformed,
    node_keys: evoxc.SymbolicTensor,
    node_attrs: evoxc.SymbolicTensor,
    conn_keys: evoxc.SymbolicTensor,
    conn_attrs: evoxc.SymbolicTensor,
    inputs: evoxc.SymbolicTensor,
    loss_fn: Any,
) -> evoxc.Graph:
    """Trace the dense pass and wrap it in ``evoxc.grad`` (the implementation of ``grad``).

    The public contract — one scalar output, float-only differentiation, the schedule outside
    the differentiated graph, and NaN cleaned at the input boundary — is documented on
    :meth:`base.BaseGenome.grad`; this function is that recipe applied to
    :func:`dense_forward`.

    Args:
        genome: The genome whose genes and statics the pass reads.
        state: Host configuration container.
        transformed: Execution schedule of the population, as host data (its fields are baked
            into the graph as constants).
        node_keys: ``(P, N) int32`` node keys, non-differentiated input.
        node_attrs: ``(P, N, NA) float32`` node attributes, differentiated input.
        conn_keys: ``(P, E, CK) int32`` connection keys, non-differentiated input.
        conn_attrs: ``(P, E, CA) float32`` connection attributes, differentiated input.
        inputs: ``(P, num_inputs) float32`` network inputs.
        loss_fn: ``(P, num_outputs) -> (*)`` float32 per-individual loss, summed before the
            gradient transform.

    Returns:
        ``evoxc.Graph`` mapping ``(node_keys, node_attrs, conn_keys, conn_attrs, inputs)`` to
        ``(node_attr_grads, conn_attr_grads)``.

    Raises:
        ConfigError: If the genome's topology or genes cannot be expressed by the dense pass
            (see :func:`dense_grad_plan`).
        ShapeBoundError: If an edge has no incoming-window slot, i.e. ``max_in_degree`` is too
            small for the individual's in-degree.
    """
    plan = dense_grad_plan(genome, transformed)

    def objective(node_keys, node_attrs, conn_keys, conn_attrs, inputs):
        safe_node_attrs = _sanitize_attrs(
            node_attrs, genome.node_exists(node_keys)
        )
        safe_conn_attrs = _sanitize_attrs(
            conn_attrs, genome.conn_exists(conn_keys)
        )
        outputs = dense_forward(
            genome,
            state,
            plan,
            safe_node_attrs,
            safe_conn_attrs,
            inputs,
        )
        return enp.sum(loss_fn(outputs), axis=None)

    graph = evoxc.trace(
        objective,
        *(
            _tensor_spec(value)
            for value in (node_keys, node_attrs, conn_keys, conn_attrs, inputs)
        ),
    )
    return evoxc.grad(graph, argnums=(1, 3))
