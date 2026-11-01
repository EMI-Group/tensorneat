"""Genome base class: payload contract, schedule glue and forward/grad API.

This module owns host-side configuration (static sizes, gene/operator objects, transforms),
the payload/row-key invariant, and the *shared* window-reduce machinery both concrete genomes
compose — ``default.py`` and ``recurrent.py`` must not duplicate it.
Every compute method is batch-first (population on axis 0) with static shapes: no
``vmap``/``vectorize`` and no ``scan``/``while_loop``/``cond`` — depth and tick loops are plain
Python ``for`` loops unrolled at trace time, which is what keeps ``evoxc.grad`` usable.

## Cross-module contracts
Every name below is provided by the ``common`` node and is never re-derived here.
``../common.tensorization.derive_schedule`` is the only schedule derivation.
``../common.functions.apply_activation``/``apply_aggregation`` are the only activation and
window-reduce kernels, reached through the node gene.
``../common.tools.I_INF`` is the ``int32`` "no index" sentinel padding every key array, and
``../common.tools.masked_flat_gather`` is the flat-index gather used by the window step.
``../common.State``/``StatefulBaseClass`` are the immutable host containers and
``../common.errors.TensorNEATError`` (``ConfigError``) covers every configuration violation.

## Cost budget (measured, numpy backend)
A forward pass costs about one unrolled level per network depth: measured 1.8 ms per level at `P=1000`, `max_nodes=50`, `K=2` with ~20 ops per level, i.e. ~14 ms for a depth-8 forward against the 157 ms legacy generation at the same population.
Op count per unrolled level — not element count — is the budget: keep the window step thin, gather connection attributes once per forward pass, and never re-derive the schedule inside a level.
Depth, tick and option loops are unrolled at trace time, so compile time grows with `max_depth`/`activate_time`/option count; single-option configurations emit no selection op at all.
"""

from collections.abc import Callable, Sequence
from typing import Any, NamedTuple

import evoxc
import evoxc.numpy as enp
import numpy as np

from ..common import (
    AGG,
    I_INF,
    ShapeBoundError,
    State,
    StatefulBaseClass,
    TensorNEATError,
    Transformed,
    apply_activation,
    derive_schedule,
    to_numpy,
    validate_schedule,
)
from ..common.errors import ConfigError
from .gene import BaseConn, BaseNode
from .operations import BaseCrossover, BaseDistance, BaseMutation
from .utils import GenomeTensors, check_cycles as _check_cycles, re_cound_idx

#: Scale applied to a float attribute before it is rounded to an ``int32`` hash word.
#: ``evoxc`` has no float/int bitcast, so :meth:`BaseGenome.hash` quantises instead; the
#: scale is coarse enough that two attributes that differ visibly hash differently and fine
#: enough that gene values stay far inside the ``int32`` range (they are clipped to +-5).
HASH_QUANT_SCALE: int = 1_000_000

#: numpy dtype -> evoxc dtype, for the graph-factory input specs of :meth:`BaseGenome.grad`.
#: A tensor that is already an evoxc tensor carries its dtype verbatim and misses this table.
_NUMPY_DTYPES = {
    np.dtype("float32"): evoxc.float32,
    np.dtype("float64"): evoxc.float64,
    np.dtype("int32"): evoxc.int32,
    np.dtype("int64"): evoxc.int64,
    np.dtype("bool"): evoxc.bool_,
}


# --------------------------------------------------------------------------- #
# module-level helpers: static scalars, sentinels, host conversion
# --------------------------------------------------------------------------- #


def _positive_int(value: Any, name: str) -> int:
    """Validate one positive constructor integer.

    Args:
        value: Candidate value; ``bool`` is refused because it is not a count.
        name: Argument name used in the error message.

    Returns:
        ``value`` as a Python ``int`` (NumPy integers are normalised, so a trace never sees a
        NumPy scalar).

    Raises:
        ValueError: If ``value`` is not an ``int``/``numpy.integer`` or is not positive — the
            documented error for a bad size argument.
    """
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be a positive int, got {value!r}")
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be a positive int, got {value!r}")
    return value


def _i32(value: int) -> evoxc.SymbolicTensor:
    """Rank-0 ``int32`` constant.

    Written out instead of a bare Python literal because an ``int32`` tensor plus a Python
    ``int`` promotes to ``int64`` in ``evoxc``, which would silently widen index arithmetic.

    Args:
        value: Static integer baked into the trace.

    Returns:
        Rank-0 ``int32`` tensor inside the active trace.
    """
    return enp.full((), int(value), dtype=evoxc.int32)


class _DenseGradPlan(NamedTuple):
    """Host statics of the exact (dense-weight) gradient path.

    Built once per :meth:`BaseGenome.grad` call from an already-derived schedule and baked into
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
            static ``K`` is refused by :meth:`_dense_plan` instead).
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


def _iinf(shape: tuple[int, ...], dtype: Any = evoxc.int32) -> evoxc.SymbolicTensor:
    """``I_INF`` broadcast to ``shape`` — the padding sentinel of every key array.

    Args:
        shape: Static shape of the result.
        dtype: Integer dtype of the result.

    Returns:
        ``shape``-shaped tensor full of ``I_INF``.
    """
    return enp.full(tuple(shape), I_INF, dtype=dtype)


def _constant_int32(values: Sequence[int]) -> evoxc.SymbolicTensor:
    """Embed a host integer sequence as an in-graph ``int32`` constant.

    Args:
        values: Static integers (node keys, column indices, ...).

    Returns:
        ``(len(values),)`` ``int32`` tensor. Genuinely static data (the initial key layout,
        the input/output key vector), never per-generation payload.
    """
    array = np.ascontiguousarray(np.asarray(values, dtype=np.int32))
    return evoxc.constant(evoxc.from_numpy(array))


def _row_slice(table: evoxc.SymbolicTensor, index: int) -> evoxc.SymbolicTensor:
    """Read one static column of a ``(P, M)`` traced table as a ``(P,)`` vector.

    A slice-then-reshape instead of ``table[:, index]``: ``evoxc``'s reverse-mode rule for an
    axis-dropping integer index has no usable cotangent shape, and the same idiom is used by
    the gene attribute readers.

    Args:
        table: ``(P, M)`` traced tensor.
        index: Static column index in ``[0, M)``.

    Returns:
        ``(P,)``-shaped view of column ``index``.
    """
    pop_size = int(table.shape[0])
    sliced = table[(slice(None), slice(index, index + 1))]
    return evoxc.reshape(sliced, (pop_size,))


class BaseGenome(StatefulBaseClass):
    """Configuration and compute contract of one NEAT genome family.

    Payload (``GenomeTensors``, batch-first, static; ``P`` population, ``N`` ``max_nodes``,
    ``E`` ``max_conns``): ``node_keys (P, N) int32``, ``node_attrs (P, N, NA) float32``,
    ``conn_keys (P, E, CK) int32``, ``conn_attrs (P, E, CA) float32``.  Absent key rows are
    ``I_INF`` (``int32`` max), absent attribute rows NaN; keys and attributes are separate
    tensors because evoxc has no float/int bitcast, so operators twiddle integer keys and only
    attributes do float math.

    Row/key invariant: ``initialize`` writes nodes in key order, so input keys are rows
    ``0..num_inputs-1`` and output keys rows ``max_nodes-num_outputs..max_nodes-1``; those rows
    are never deleted or moved, so ``get_input_idx()``/``get_output_idx()`` double as row
    indices, while hidden keys are arbitrary and may move rows (compare keys, never rows).
    Hidden keys of the initial network are contiguous with the input keys (rows
    ``num_inputs..num_inputs+sum(init_hidden_layers)-1``), so row ``i`` holds key ``i`` for
    every hidden node as well.

    ``max_in_degree`` is the static incoming-window width ``K`` (default
    ``max(initial in-degree, 4)``), ``max_depth`` the static unroll count ``D``; both are
    Python ints fixed at construction, so changing either forces a rebuild.

    ``key_span`` is the static key-magnitude bound ``S`` that
    ``common.tensorization.derive_schedule`` needs for its dense key->slot table: **every node
    key must lie in ``[0, S)``** and ``S`` is part of the Pipeline's shape signature, so a
    caller that allocates larger keys must raise it (or construct the genome with a larger
    ``key_span``).  The default is the compact bound
    ``max(max_nodes, num_inputs + num_outputs) + max_conns``, which covers the initial key
    layout plus one allocated key per connection row; a key outside ``S`` is *not* aliased —
    the schedule reports it as an unmatched endpoint, exactly like an unknown key.
    """

    network_type: str | None = None

    def __init__(
        self,
        num_inputs: int,
        num_outputs: int,
        max_nodes: int,
        max_conns: int,
        node_gene: BaseNode,
        conn_gene: BaseConn,
        mutation: BaseMutation,
        crossover: BaseCrossover,
        distance: BaseDistance,
        output_transform: Callable | None = None,
        input_transform: Callable | None = None,
        init_hidden_layers: Sequence[int] = (),
        max_in_degree: int | None = None,
        max_depth: int = 16,
        key_span: int | None = None,
    ):
        """Configure one genome family and derive its dense initial network.

        The initial network is fully connected and layered,
        ``num_inputs -> init_hidden_layers -> num_outputs``, keys assigned in layer order
        (input keys ``0..num_inputs-1``, hidden keys continuing from there, output keys the
        last ``num_outputs`` of the padded payload) and written to the payload by
        :meth:`initialize` together with the dense edges.
        ``input_transform``/``output_transform`` are elementwise evoxc graph callables over
        the last axis of a ``(P, n)`` batch; each is probed by tracing a ``(1, n)`` graph here,
        so a broken transform fails at construction instead of inside a later trace.

        Args:
            num_inputs: Input node count, also the network input width.
            num_outputs: Output node count, also the network output width.
            max_nodes: Static node capacity ``N``.
            max_conns: Static connection capacity ``E``.
            node_gene: Node gene supplying ``attr_width`` and the node step.
            conn_gene: Conn gene supplying ``key_width``/``attr_width`` and the edge message.
            mutation: Batched mutation operator.
            crossover: Batched key-alignment crossover operator.
            distance: Batched compatibility-distance operator.
            output_transform: Optional elementwise map of the ``(P, num_outputs)`` read-out.
            input_transform: Optional elementwise map of ``inputs``; never applied by
                ``RecurrentGenome`` (legacy parity).
            init_hidden_layers: Width of each dense hidden layer; empty means one dense
                input-to-output layer.
            max_in_degree: Static window width ``K``; ``None`` resolves to
                ``max(initial in-degree, 4)``.
            max_depth: Static depth unroll count ``D``.
            key_span: Static key bound ``S``; ``None`` resolves to
                ``max(max_nodes, num_inputs + num_outputs) + max_conns``.

        Raises:
            TensorNEATError: If ``max_nodes``/``max_conns`` cannot hold the initial network, if
                ``init_hidden_layers`` needs an in-degree above ``max_in_degree``, if
                ``key_span`` is below ``max_nodes``, if a gene/operator is not the expected
                type, or if a transform probe raises or returns a wrong shape.
            ValueError: If a size argument is not a positive int.
        """
        self.num_inputs = _positive_int(num_inputs, "num_inputs")
        self.num_outputs = _positive_int(num_outputs, "num_outputs")
        self.max_nodes = _positive_int(max_nodes, "max_nodes")
        self.max_conns = _positive_int(max_conns, "max_conns")
        self.max_depth = _positive_int(max_depth, "max_depth")
        hidden = tuple(
            _positive_int(width, "init_hidden_layers entry")
            for width in init_hidden_layers
        )
        self.init_hidden_layers = hidden

        for value, expected, name in (
            (node_gene, BaseNode, "node_gene"),
            (conn_gene, BaseConn, "conn_gene"),
            (mutation, BaseMutation, "mutation"),
            (crossover, BaseCrossover, "crossover"),
            (distance, BaseDistance, "distance"),
        ):
            if not isinstance(value, expected):
                raise TensorNEATError(
                    f"{type(self).__name__}: {name} must be a {expected.__name__}, "
                    f"got {type(value).__name__}"
                )

        layers = (self.num_inputs, *hidden, self.num_outputs)
        node_count = sum(layers)
        conn_count = sum(left * right for left, right in zip(layers, layers[1:]))
        if node_count > self.max_nodes:
            raise TensorNEATError(
                f"max_nodes={self.max_nodes} cannot hold the initial network of "
                f"{node_count} nodes ({' -> '.join(str(width) for width in layers)})"
            )
        if conn_count > self.max_conns:
            raise TensorNEATError(
                f"max_conns={self.max_conns} cannot hold the {conn_count} dense initial "
                f"connections of {' -> '.join(str(width) for width in layers)}"
            )

        # Row/key invariant: input keys start at row 0, hidden keys continue in row order and
        # the output keys sit at the tail of the *padded* payload (rows N-num_outputs..N-1).
        self.input_idx = tuple(range(self.num_inputs))
        self.output_idx = tuple(range(self.max_nodes - self.num_outputs, self.max_nodes))
        layer_keys: list[tuple[int, ...]] = []
        next_key = 0
        for width in layers[:-1]:
            layer_keys.append(tuple(range(next_key, next_key + width)))
            next_key += width
        layer_keys.append(self.output_idx)
        self._layer_keys = tuple(layer_keys)
        #: Node keys in payload row order: inputs and hidden nodes first, then the outputs.
        self._init_node_keys = tuple(range(sum(layers[:-1]))) + self.output_idx
        edges = [
            (src, dst)
            for left, right in zip(layer_keys, layer_keys[1:])
            for src in left
            for dst in right
        ]
        self._init_edges = tuple(edges)

        initial_in_degree = max(len(keys) for keys in layer_keys[:-1])
        if max_in_degree is None:
            self.max_in_degree = max(initial_in_degree, 4)
        else:
            self.max_in_degree = _positive_int(max_in_degree, "max_in_degree")
            if self.max_in_degree < initial_in_degree:
                raise TensorNEATError(
                    f"max_in_degree={self.max_in_degree} is below the initial in-degree "
                    f"{initial_in_degree} of the configured network; the dense initial edges "
                    "would not fit the static window"
                )
        if key_span is None:
            self.key_span = (
                max(self.max_nodes, self.num_inputs + self.num_outputs) + self.max_conns
            )
        else:
            self.key_span = _positive_int(key_span, "key_span")
            if self.key_span < self.max_nodes:
                raise TensorNEATError(
                    f"key_span={self.key_span} is below max_nodes={self.max_nodes}: the "
                    "initial output keys occupy rows up to max_nodes - 1, which must be "
                    "inside the schedule's key table"
                )

        self.node_gene = node_gene
        self.conn_gene = conn_gene
        self.mutation = mutation
        self.crossover = crossover
        self.distance = distance
        self.output_transform = output_transform
        self.input_transform = input_transform

        self._probe_transform(input_transform, self.num_inputs, "input_transform")
        self._probe_transform(output_transform, self.num_outputs, "output_transform")

    def _probe_transform(
        self, transform: Callable | None, width: int, name: str
    ) -> None:
        """Trace-probe an elementwise transform on a ``(1, width)`` batch.

        Args:
            transform: Optional graph callable over the last axis of a ``(1, width)`` batch.
            width: Expected (and required) last-axis length of the probe and its result.
            name: Argument name used in error messages.

        Returns:
            ``None``; the probe only traces, it never compiles or runs.

        Raises:
            TensorNEATError: If the transform raises while being traced or returns a shape
                other than ``(1, width)`` — a broken transform must fail at construction, not
                inside the generation graph.
        """
        if transform is None:
            return
        expected = (1, width)

        def probe(values: evoxc.SymbolicTensor) -> evoxc.SymbolicTensor:
            out = transform(values)
            shape = tuple(int(size) for size in out.shape)
            if shape != expected:
                raise TensorNEATError(
                    f"{name} returned shape {shape} on a {expected} probe; an elementwise "
                    "transform must preserve the batch shape"
                )
            return out

        try:
            evoxc.trace(probe, evoxc.TensorSpec(expected, evoxc.float32))
        except TensorNEATError:
            raise
        except Exception as error:  # noqa: BLE001 - re-raised as a configuration error
            raise TensorNEATError(
                f"{name} failed on a {expected} probe: {error}"
            ) from error

    # ------------------------------------------------------------- host plumbing
    def _resolve_pop_size(
        self, state: State, pop_size: int | None, caller: str
    ) -> int:
        """Resolve the static population size of a kernel call.

        ``pop_size`` is a static, so it is either passed explicitly or read from the
        ``pop_size`` entry the algorithm registers on ``state``.

        Args:
            state: Host configuration container.
            pop_size: Explicit population size, or ``None`` to read ``state.pop_size``.
            caller: Method name used in the error message.

        Returns:
            The validated population size ``P``.

        Raises:
            TensorNEATError: If no population size is available or it is not a positive int.
        """
        if pop_size is None:
            try:
                pop_size = state.pop_size
            except KeyError:
                raise TensorNEATError(
                    f"{caller} needs a static pop_size: pass it explicitly or register "
                    "'pop_size' on the state"
                ) from None
        try:
            return _positive_int(pop_size, "pop_size")
        except ValueError as error:
            raise TensorNEATError(f"{caller}: {error}") from error

    def _split_payload(self, *arrays: Any) -> tuple[GenomeTensors, tuple]:
        """Split a call's positional payload arguments into genes and trailing arguments.

        Both spellings are accepted, because the algorithm's doc spells the operators with the
        four arrays while ``genome/CONTEXT.md`` spells them with a ``GenomeTensors`` payload:

        * ``(payload, *trailing)`` — one ``GenomeTensors`` (or a 4-tuple of the four arrays);
        * ``(node_keys, node_attrs, conn_keys, conn_attrs, *trailing)``.

        Args:
            *arrays: The positional arguments after the ``state``/``key`` ones.

        Returns:
            ``(payload, trailing)``.

        Raises:
            TensorNEATError: If the arguments match neither spelling, or a payload leaf is not
                the documented ``(P, ...)`` tensor.
        """
        if arrays and isinstance(arrays[0], GenomeTensors):
            payload, trailing = arrays[0], tuple(arrays[1:])
        elif len(arrays) >= 4:
            payload, trailing = GenomeTensors(*arrays[:4]), tuple(arrays[4:])
        else:
            raise TensorNEATError(
                f"{type(self).__name__}: expected a GenomeTensors payload or its four arrays "
                f"(node_keys, node_attrs, conn_keys, conn_attrs), got {len(arrays)} argument(s)"
            )
        for name, value in zip(payload.field_names, payload):
            if len(value.shape) < 2:
                raise TensorNEATError(
                    f"{type(self).__name__}: payload field {name} must be batch-first "
                    f"(population on axis 0), got shape {tuple(value.shape)}"
                )
        return payload, trailing

    @staticmethod
    def _individual(payload: GenomeTensors) -> GenomeTensors:
        """Host numpy views of one individual's payload (never fed back into a graph).

        Args:
            payload: ``(P, ...)`` payload batch, or already unbatched arrays.

        Returns:
            The four arrays as ``numpy`` arrays without the population axis.

        Raises:
            ConfigError: If the payload holds more than one individual — these helpers convert
                one genome to host dicts, and a batch would be walked through the wrong axis.
        """
        arrays = tuple(np.asarray(to_numpy(value)) for value in payload)
        node_keys = arrays[0]
        if node_keys.ndim > 1:
            if node_keys.shape[0] != 1:
                raise ConfigError(
                    f"{len(payload)} batch entries: this helper takes one individual, "
                    f"got a batch of {node_keys.shape[0]} (pass payload.of(p))"
                )
            arrays = tuple(array[0] for array in arrays)
        return GenomeTensors(*arrays)

    # ---------------------------------------------------------------- lifecycle
    def setup(self, state: State | None = None) -> State:
        """Register the gene/operator configuration on an immutable state.

        Args:
            state: Base state to extend; never mutated in place. ``None`` starts from a fresh
                ``State()`` (a ``State()`` default argument would run at class-definition time).

        Returns:
            A new ``State`` after chaining ``node_gene``, ``conn_gene``, ``mutation``,
            ``crossover`` and ``distance`` setup, in that order.
        """
        state = State() if state is None else state
        state = self.node_gene.setup(state)
        state = self.conn_gene.setup(state)
        state = self.mutation.setup(state)
        state = self.crossover.setup(state)
        state = self.distance.setup(state)
        return state

    # ----------------------------------------------------------------- payload
    def initialize(
        self, state: State, key: evoxc.SymbolicTensor, pop_size: int | None = None
    ) -> GenomeTensors:
        """Draw the initial population of dense layered networks.

        One rank-0 ``evoxc.random`` key plus a static ``pop_size``; every tensor is created
        with its full batched shape, so individual ``i`` gets an independent stream from that
        single draw (keys are never split per individual).  The ``historical_marker`` key
        column, when the conn gene declares it, is the row index of the connection.

        Args:
            state: Host configuration container from :meth:`setup`.
            key: Rank-0 ``evoxc.random`` key, split internally per gene family.
            pop_size: Static population size ``P``; ``None`` reads ``state.pop_size``.

        Returns:
            ``GenomeTensors``: ``node_keys (P, N) int32`` (``I_INF`` padded, input/output rows
            at their invariant positions), ``node_attrs (P, N, NA) float32`` (NaN padded),
            ``conn_keys (P, E, CK) int32`` (``I_INF`` padded), ``conn_attrs (P, E, CA)
            float32`` (NaN padded).
        """
        pop_size = self._resolve_pop_size(state, pop_size, "initialize")
        node_count = len(self._init_node_keys)
        conn_count = len(self._init_edges)
        node_attr_width = self.node_gene.attr_width
        conn_attr_width = self.conn_gene.attr_width
        head_count = len(self._init_node_keys) - self.num_outputs

        node_rows = np.full(self.max_nodes, I_INF, dtype=np.int32)
        node_rows[:head_count] = np.asarray(self._init_node_keys[:head_count], dtype=np.int32)
        node_rows[self.max_nodes - self.num_outputs :] = np.asarray(
            self.output_idx, dtype=np.int32
        )
        node_keys = enp.broadcast_to(
            evoxc.reshape(_constant_int32(node_rows), (1, self.max_nodes)),
            (pop_size, self.max_nodes),
        )

        key_node, key_conn = evoxc.random.split_n(key, 2)
        drawn_nodes = self.node_gene.new_random_attrs(
            state, key_node, (pop_size, node_count)
        )
        head_attrs = drawn_nodes[(slice(None), slice(0, head_count))]
        tail_attrs = drawn_nodes[(slice(None), slice(head_count, node_count))]
        pad_rows = self.max_nodes - node_count
        if pad_rows:
            node_attrs = evoxc.concatenate(
                [
                    head_attrs,
                    enp.full(
                        (pop_size, pad_rows, node_attr_width),
                        float("nan"),
                        dtype=evoxc.float32,
                    ),
                    tail_attrs,
                ],
                axis=1,
            )
        else:
            node_attrs = evoxc.concatenate([head_attrs, tail_attrs], axis=1)

        conn_rows = np.full((self.max_conns, self.conn_gene.key_width), I_INF, dtype=np.int32)
        conn_rows[:conn_count, 0] = np.asarray(
            [edge[0] for edge in self._init_edges], dtype=np.int32
        )
        conn_rows[:conn_count, 1] = np.asarray(
            [edge[1] for edge in self._init_edges], dtype=np.int32
        )
        if self.conn_gene.uses_historical_marker:
            # The initial marker of a connection is its row index (legacy convention).
            conn_rows[:conn_count, 2] = np.arange(conn_count, dtype=np.int32)
        conn_keys = enp.broadcast_to(
            evoxc.reshape(_constant_int32(conn_rows.reshape(-1)), (1, *conn_rows.shape)),
            (pop_size, *conn_rows.shape),
        )

        drawn_conns = self.conn_gene.new_random_attrs(
            state, key_conn, (pop_size, conn_count)
        )
        if self.max_conns > conn_count:
            conn_attrs = evoxc.concatenate(
                [
                    drawn_conns,
                    enp.full(
                        (pop_size, self.max_conns - conn_count, conn_attr_width),
                        float("nan"),
                        dtype=evoxc.float32,
                    ),
                ],
                axis=1,
            )
        else:
            conn_attrs = drawn_conns

        return GenomeTensors(node_keys, node_attrs, conn_keys, conn_attrs)

    def transform(self, state: State, *arrays: Any) -> Transformed:
        """Derive the execution schedule of every individual (in-graph, no host round-trip).

        The only place the schedule is derived: it delegates to
        ``..common.tensorization.derive_schedule`` with the input/output keys as ``int32``
        in-graph constants and the five statics of this genome.  The composition goes through
        ``derive_schedule.fn`` — the ``@evoxc.defn`` object itself refuses to be called
        (``Defn`` is a graph definition, not a callable), so a graph that needs the schedule
        inline calls the plain function, which evoxc documents as the supported way to
        specialize it.  The derivation must stay outside any differentiated graph
        (``argsort`` has no AD rule), which is why :meth:`grad` takes an already-derived
        ``Transformed``.

        Args:
            state: Host configuration container.
            *arrays: The key arrays, in either accepted spelling: a ``GenomeTensors``
                payload; ``(node_keys, conn_keys)``; or the full four-array payload
                ``(node_keys, node_attrs, conn_keys, conn_attrs)`` — the attribute arrays are
                ignored, the schedule only needs the keys.

        Returns:
            ``Transformed`` with ``in_src``/``in_conn``/``in_valid (P, N, K)``,
            ``node_level``/``node_exists (P, N)``, ``overflow (P,)``; cost is
            edge-proportional, never ``max_nodes**2`` per node.

        Raises:
            TensorNEATError: If the arguments match none of the accepted spellings.
        """
        node_keys, conn_keys = self._schedule_inputs(arrays)
        pop_size, max_nodes = (int(size) for size in node_keys.shape)
        conn_in = self.conn_gene.key_column(conn_keys, "input_index")
        conn_out = self.conn_gene.key_column(conn_keys, "output_index")
        return derive_schedule.fn(
            node_keys,
            conn_in,
            conn_out,
            _constant_int32(self.input_idx),
            _constant_int32(self.output_idx),
            pop_size,
            max_nodes,
            self.max_conns,
            self.max_in_degree,
            self.key_span,
        )

    @staticmethod
    def _schedule_inputs(arrays: tuple) -> tuple:
        """Pick the node and connection key arrays out of a schedule call.

        Args:
            arrays: ``(payload,)``, ``(node_keys, conn_keys)`` or the four payload arrays.

        Returns:
            ``(node_keys, conn_keys)``.

        Raises:
            TensorNEATError: If the arguments match none of the accepted spellings.
        """
        if len(arrays) == 1 and isinstance(arrays[0], GenomeTensors):
            return arrays[0].node_keys, arrays[0].conn_keys
        if len(arrays) == 2:
            return arrays[0], arrays[1]
        if len(arrays) == 4:
            return arrays[0], arrays[2]
        raise TensorNEATError(
            "transform takes a GenomeTensors payload, (node_keys, conn_keys), or the four "
            f"payload arrays; got {len(arrays)} argument(s)"
        )

    def forward(
        self,
        state: State,
        transformed: Transformed,
        node_keys: evoxc.SymbolicTensor,
        node_attrs: evoxc.SymbolicTensor,
        conn_keys: evoxc.SymbolicTensor,
        conn_attrs: evoxc.SymbolicTensor,
        inputs: evoxc.SymbolicTensor,
    ) -> evoxc.SymbolicTensor:
        """Evaluate a whole population of networks on one batch of inputs (abstract).

        Args:
            state: Host configuration container.
            transformed: Execution schedule of the same population.
            node_keys, node_attrs, conn_keys, conn_attrs: Payload batch (see the class
                docstring); ``node_attrs``/``conn_attrs`` keep their NaN padding.
            inputs: ``(P, num_inputs) float32``; axis 0 is the population, so datasets are
                tiled over it instead of being vmapped per sample.

        Returns:
            ``(P, num_outputs) float32`` outputs, after ``output_transform``.

        Raises:
            NotImplementedError: Always — ``default.py``/``recurrent.py`` implement it.
        """
        raise NotImplementedError(
            "concrete genomes implement the depth- or tick-unrolled forward pass"
        )

    def grad(
        self,
        state: State,
        transformed: Transformed,
        node_keys: evoxc.SymbolicTensor,
        node_attrs: evoxc.SymbolicTensor,
        conn_keys: evoxc.SymbolicTensor,
        conn_attrs: evoxc.SymbolicTensor,
        inputs: evoxc.SymbolicTensor,
        loss_fn: Callable[[evoxc.SymbolicTensor], evoxc.SymbolicTensor],
    ) -> evoxc.Graph:
        """Build the gradient graph of one forward pass (graph factory, not a kernel).

        Like ``evoxc.grad`` itself, this traces its own graph and returns it for the Pipeline
        to cache per shape signature.  Four evoxc rules shape it: the transform needs exactly
        one scalar output, so the loss is scalarized as ``sum(loss_fn(outputs))``
        (per-individual gradients are unaffected — the population axis is independent of the
        sum); only float arguments are differentiable, so ``argnums`` selects
        ``node_attrs``/``conn_attrs`` while keys and inputs stay non-differentiated graph
        inputs (a differentiated ``int32`` argument is a hard ``TransformError``); the schedule
        must stay outside the differentiated graph because ``argsort`` has no AD rule, so every
        ``Transformed`` field is embedded as an in-graph constant of this factory call; and the
        NaN contract is enforced at the *input* boundary — each attribute tensor is first
        masked to its existing rows and NaN-cleaned (``select(isnan(a) | ~exists, 0, a)``),
        which makes the reverse sweep emit exactly ``0`` on padding (a zeroed cotangent times
        a NaN local derivative would otherwise be NaN) and leaves live rows untouched.

        The differentiated pass is :meth:`_dense_forward`, not the window step of
        :meth:`_node_step`.  evoxc 0.1.0 transposes ``gather`` with *put* semantics, so a node
        value read by several incoming-edge slots keeps only one of their cotangents; the
        exact path therefore aggregates with a per-individual dense weight matrix and a batched
        ``dot``, whose transpose rule is exact.  The plain forward keeps the cheap window op
        budget and is untouched.

        Args:
            state: Host configuration container.
            transformed: Execution schedule of the population, already derived. Its fields
                must be host data (numpy arrays or ``evoxc.Tensor``): they are baked into the
                graph, and a traced schedule belongs to another module.
            node_keys: ``(P, N) int32`` node keys, non-differentiated input.
            node_attrs: ``(P, N, NA) float32`` node attributes, differentiated input.
            conn_keys: ``(P, E, CK) int32`` connection keys, non-differentiated input.
            conn_attrs: ``(P, E, CA) float32`` connection attributes, differentiated input.
            inputs: ``(P, num_inputs) float32`` network inputs.
            loss_fn: ``(P, num_outputs) -> (*)`` float32 per-individual loss, summed before
                the gradient transform.

        Returns:
            ``evoxc.Graph`` mapping ``(node_keys, node_attrs, conn_keys, conn_attrs, inputs)``
            to ``(node_attr_grads, conn_attr_grads)``, gradients taking the shapes and dtypes
            of the differentiated inputs. ``evoxc.grad`` returns gradients only, so the loss
            value is not part of this graph; callers that need it read it from
            :meth:`forward` plus the same ``loss_fn``.

        Raises:
            ConfigError: If the genome is not the level-synchronous ``"feedforward"`` kind, if
                its node gene does not aggregate with a single ``AGG.sum``/``AGG.mean`` option,
                if its connection gene declares no ``weight`` attribute column, if the schedule
                violates its own invariants, or if one individual carries two connections with
                the same ``(destination, source)`` pair — each of those breaks the identity
                between this pass and :meth:`forward`.
            ShapeBoundError: If an individual's in-degree exceeds the schedule's static
                ``max_in_degree``, so an edge has no incoming-window slot to be evaluated in.
        """
        plan = self._dense_plan(transformed)

        def objective(node_keys, node_attrs, conn_keys, conn_attrs, inputs):
            safe_node_attrs = self._sanitize_attrs(
                node_attrs, self.node_exists(node_keys)
            )
            safe_conn_attrs = self._sanitize_attrs(
                conn_attrs, self.conn_exists(conn_keys)
            )
            outputs = self._dense_forward(
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
                self._tensor_spec(value)
                for value in (node_keys, node_attrs, conn_keys, conn_attrs, inputs)
            ),
        )
        return evoxc.grad(graph, argnums=(1, 3))

    @staticmethod
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

    @staticmethod
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
        self, attrs: evoxc.SymbolicTensor, exists: evoxc.SymbolicTensor
    ) -> evoxc.SymbolicTensor:
        """Mask padding rows and clean NaN out of one attribute tensor.

        Applied once at the boundary of :meth:`grad`, so nothing NaN-valued ever enters a
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

    # ---------------------------------------------------------------- shared kernel
    @staticmethod
    def _gather_slots(
        x: evoxc.SymbolicTensor,
        idx: evoxc.SymbolicTensor,
        mask: evoxc.SymbolicTensor,
        fill: float,
    ) -> evoxc.SymbolicTensor:
        """Gather ``(P, *shape)`` window slots out of one batch-first ``(P, R)`` table.

        The index is flattened to one dimension per batch item (``p * R + row``) because the
        reverse of a ``gather`` is a ``scatter``, and ``evoxc``'s scatter requires the indices
        and the scattered array to share a rank: a fully flat index (``p * R + ...``) keeps
        both operands one-dimensional, so the same helper works in the forward pass and inside
        :meth:`grad`.  Masked-out and ``I_INF`` slots are redirected to ``0`` before the gather
        and overwritten with ``fill`` afterwards, so the backend never sees an out-of-range
        index.

        Args:
            x: ``(P, R)`` source table (node values, or a reshaped attribute payload).
            idx: ``(P, *shape) int32`` row indices into ``x``'s second axis; ``I_INF`` (or a
                masked-out position) means "no slot".
            mask: ``(P, *shape) bool`` liveness of each slot.
            fill: Value written where the slot is absent.

        Returns:
            ``(P, *shape)`` tensor with ``x``'s dtype.
        """
        pop_size, row_len = (int(size) for size in x.shape)
        block = tuple(int(size) for size in idx.shape[1:])
        count = 1
        for size in block:
            count *= size
        present = evoxc.logical_and(
            enp.not_equal(idx, _iinf(tuple(idx.shape), idx.dtype)), mask
        )
        safe = evoxc.select(present, idx, enp.zeros(tuple(idx.shape), dtype=idx.dtype))
        offsets = evoxc.reshape(
            evoxc.multiply(enp.arange(0, pop_size, dtype=evoxc.int32), _i32(row_len)),
            (pop_size, *([1] * len(block))),
        )
        flat_idx = evoxc.reshape(evoxc.add(safe, offsets), (pop_size * count,))
        values = evoxc.gather(evoxc.reshape(x, (pop_size * row_len,)), flat_idx, axis=0)
        values = evoxc.select(
            evoxc.reshape(present, (pop_size * count,)),
            values,
            enp.full((), float(fill), dtype=x.dtype),
        )
        return evoxc.reshape(values, (pop_size, *block))

    def _hit_conn_attrs(
        self, transformed: Transformed, conn_attrs: evoxc.SymbolicTensor
    ) -> evoxc.SymbolicTensor:
        """Gather the attributes of every incoming edge slot, once per forward pass.

        Attributes are constant during a forward pass, so the gather is hoisted out of the
        level/tick loop instead of being repeated per step.

        Args:
            transformed: Schedule whose ``in_conn`` is the ``(P, N, K) int32`` connection
                **row** index of each slot.
            conn_attrs: ``(P, E, CA) float32`` connection attributes, NaN padded.

        Returns:
            ``(P, N, K, CA) float32`` per-slot attributes; absent slots hold NaN, which the
            conn gene sanitises.

        Notes:
            The flat index of slot ``(p, n, k, w)`` is ``p * (E * CA) + row * CA + w``, i.e.
            the slot row index times the attribute width plus the column index, because the
            payload is gathered from ``reshape(conn_attrs, (P, E * CA))``.
        """
        pop_size, max_nodes, window = (int(size) for size in transformed.in_conn.shape)
        conn_rows, attr_width = (int(size) for size in conn_attrs.shape[1:])
        if not attr_width:
            return enp.zeros((pop_size, max_nodes, window, 0), dtype=evoxc.float32)
        flat = evoxc.reshape(conn_attrs, (pop_size, conn_rows * attr_width))
        rows = evoxc.multiply(
            evoxc.reshape(transformed.in_conn, (pop_size, max_nodes, window, 1)),
            _i32(attr_width),
        )
        index = evoxc.add(
            rows,
            enp.reshape(
                enp.arange(0, attr_width, dtype=evoxc.int32), (1, 1, 1, attr_width)
            ),
        )
        mask = enp.broadcast_to(
            evoxc.reshape(transformed.in_valid, (pop_size, max_nodes, window, 1)),
            (pop_size, max_nodes, window, attr_width),
        )
        return self._gather_slots(flat, index, mask=mask, fill=float("nan"))

    def _node_step(
        self,
        state: State,
        transformed: Transformed,
        node_attrs: evoxc.SymbolicTensor,
        hit_conn_attrs: evoxc.SymbolicTensor,
        values: evoxc.SymbolicTensor,
        update_mask: evoxc.SymbolicTensor,
    ) -> evoxc.SymbolicTensor:
        """Advance the node values by one window reduce (the composition point).

        Written here once; ``default.py`` (per depth level) and ``recurrent.py`` (per tick)
        call it with their own ``update_mask`` and must not duplicate it.  With ``P`` the
        population, ``N`` ``max_nodes``, ``K`` the window width and ``CA`` the conn
        ``attr_width``:

        1. ``xv = gather(values, in_src)`` -> ``(P, N, K)`` source values per slot; ``values``
           stays ``(P, N)`` and is never widened into a ``max_nodes``-wide column per node.
        2. ``live = in_valid & ~isnan(xv)``; ``xs = select(live, xv, 0)`` — the NaN-to-0
           sanitisation is mandatory before any differentiable op, because a zeroed cotangent
           times a NaN local derivative is still NaN.
        3. ``msg = conn_gene.forward(state, hit_conn_attrs, xs)`` -> ``(P, N, K)``, one scalar
           message per edge slot, elementwise in the slot.
        4. ``z = node_gene.forward(state, node_attrs, msg, is_output_row, live)`` -> ``(P, N)``:
           the gene runs ``apply_aggregation`` (dead slots neutral-filled per option, then a
           ``select`` chain), then ``combine`` (``bias + response * agg``), then the activation
           ``select`` except on output rows.
        5. ``values = select(update_mask & (live_count > 0), z, values)``: a node with no live
           incoming edge keeps its previous value, so NaN stays NaN and its descendants lose
           their only live source — the legacy NaN poisoning, reproduced exactly.

        Args:
            state: Host configuration container (read-only during a trace).
            transformed: Execution schedule of the population.
            node_attrs: ``(P, N, NA) float32`` node attributes, NaN padded.
            hit_conn_attrs: ``(P, N, K, CA) float32`` output of :meth:`_hit_conn_attrs`.
            values: ``(P, N) float32`` current node values, NaN where not yet computed.
            update_mask: ``(P, N) bool`` caller gate — the level mask for the feedforward
                genome, ``node_exists & (valid_count > 0)`` for the recurrent one; the dynamic
                "has a live slot" term is ANDed in here (step 5).

        Returns:
            ``(P, N) float32`` updated values; rows masked off are returned unchanged.
        """
        pop_size, max_nodes = (int(size) for size in values.shape)
        xv = self._gather_slots(
            values, transformed.in_src, mask=transformed.in_valid, fill=float("nan")
        )
        live = evoxc.logical_and(
            transformed.in_valid, evoxc.logical_not(evoxc.isnan(xv))
        )
        xs = evoxc.select(live, xv, enp.zeros((), dtype=evoxc.float32))
        messages = self.conn_gene.forward(state, hit_conn_attrs, xs)
        z = self.node_gene.forward(
            state,
            node_attrs,
            messages,
            self._output_row_mask(pop_size, max_nodes),
            live,
        )
        live_count = evoxc.cast(
            enp.sum(evoxc.cast(live, evoxc.int32), axis=-1), evoxc.int32
        )
        update = evoxc.logical_and(update_mask, evoxc.greater(live_count, _i32(0)))
        return evoxc.select(update, z, values)

    # ---------------------------------------------------- exact gradient path
    def _dense_plan(self, transformed: Transformed) -> _DenseGradPlan:
        """Derive and validate the host statics of the exact gradient path.

        Everything the dense pass needs beyond the payload — the scatter cell of every window
        slot, the structural live-edge matrix, the input-row mask and the per-level update
        masks — follows from the schedule alone, so it is computed here once, on the host, and
        later embedded as graph constants (:meth:`_graph_constant`).

        The three configuration refusals below are not conservative guesses: each one names a
        case where the dense pass would silently disagree with :meth:`forward`.

        Args:
            transformed: The schedule of the population, as host data.

        Returns:
            :class:`_DenseGradPlan` for :meth:`_dense_forward`.

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
        if self.network_type != "feedforward":
            raise ConfigError(
                f"{type(self).__name__}.grad differentiates the level-synchronous "
                f"feedforward sweep with a dense per-individual weight matrix; this genome's "
                f"network_type is {self.network_type!r}"
            )
        options = tuple(self.node_gene.aggregation_options)
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
                f"{type(self).__name__}.grad aggregates through a batched matmul, which "
                "carries only AGG.sum and AGG.mean; the offending option is "
                f"{offender!r} and node_gene.aggregation_options is ({names})"
            )
        try:
            self.conn_gene.attr_index("weight")
        except ConfigError as error:
            raise ConfigError(
                f"{type(self).__name__}.grad builds the dense weight matrix from the "
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
                f"{type(self).__name__}.grad needs one weight per (destination, source) "
                f"pair; individual {member} carries two connections with the same endpoints"
            )

        padded_cells = max_nodes * max_nodes + 1
        in_window = np.zeros((pop_size, dst.shape[1]), dtype=np.bool_)
        members, sources, slots = np.nonzero(in_valid)
        in_window[members, in_conn[members, sources, slots]] = True
        if np.any(edge_valid & ~in_window):
            member = int(np.argwhere(edge_valid & ~in_window)[0, 0])
            raise ShapeBoundError(
                f"{type(self).__name__}.grad: individual {member} has an edge outside every "
                f"incoming window (max_in_degree={self.max_in_degree} is too small for its "
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
        input_rows[:, : self.num_inputs] = True

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
                for level in range(1, min(deepest, self.max_depth) + 1)
            ),
        )

    def _dense_forward(
        self,
        state: State,
        plan: _DenseGradPlan,
        node_attrs: evoxc.SymbolicTensor,
        conn_attrs: evoxc.SymbolicTensor,
        inputs: evoxc.SymbolicTensor,
    ) -> evoxc.SymbolicTensor:
        """Evaluate the population with a dense per-individual weight matrix (grad only).

        Same algebra as the level-synchronous :meth:`forward` of a feedforward genome, with the
        incoming-window gather replaced by one matmul, so that reverse mode is exact despite
        evoxc 0.1.0's put-semantics ``gather`` transpose:

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
            state: Host configuration container.
            plan: Host statics from :meth:`_dense_plan`, embedded as graph constants.
            node_attrs: ``(P, N, NA) float32`` node attributes, padding already cleaned to 0.
            conn_attrs: ``(P, E, CA) float32`` connection attributes, padding already cleaned
                to 0.
            inputs: ``(P, num_inputs) float32`` network inputs.

        Returns:
            ``(P, num_outputs) float32`` outputs after ``output_transform``, equal to
            :meth:`forward`'s for the same payload up to float32 summation order.
        """
        pop_size, max_nodes = plan.input_rows.shape
        max_conns = int(conn_attrs.shape[1])
        padded_cells = max_nodes * max_nodes + 1
        zero = enp.zeros((), dtype=evoxc.float32)
        weight = self.conn_gene.attr_column(conn_attrs, "weight")
        # An unused row writes the neutral weight into the dump cell, so no second mask op is
        # needed on the (P, E) update vector.
        updates = evoxc.select(self._graph_constant(plan.edge_used), weight, zero)
        scattered = evoxc.scatter(
            enp.zeros((pop_size * padded_cells,), dtype=evoxc.float32),
            self._graph_constant(plan.cell_index),
            evoxc.reshape(updates, (pop_size * max_conns,)),
        )
        weights = evoxc.reshape(
            evoxc.reshape(scattered, (pop_size, padded_cells))[
                (slice(None), slice(0, max_nodes * max_nodes))
            ],
            (pop_size, max_nodes, max_nodes),
        )

        seeded = inputs if self.input_transform is None else self.input_transform(inputs)
        if max_nodes > self.num_inputs:
            seeded = evoxc.concatenate(
                [
                    seeded,
                    enp.zeros(
                        (pop_size, max_nodes - self.num_inputs), dtype=evoxc.float32
                    ),
                ],
                axis=1,
            )
        values = evoxc.select(
            self._graph_constant(plan.input_rows),
            seeded,
            enp.full((pop_size, max_nodes), float("nan"), dtype=evoxc.float32),
        )

        live_matrix = self._graph_constant(plan.live_matrix)
        for level_mask in plan.level_masks:
            values = self._dense_node_step(
                state,
                weights,
                live_matrix,
                node_attrs,
                values,
                self._graph_constant(level_mask),
                plan.mean,
            )
        return self._output_values(values, self.output_transform)

    def _dense_node_step(
        self,
        state: State,
        weights: evoxc.SymbolicTensor,
        live_matrix: evoxc.SymbolicTensor,
        node_attrs: evoxc.SymbolicTensor,
        values: evoxc.SymbolicTensor,
        update_mask: evoxc.SymbolicTensor,
        mean: bool,
    ) -> evoxc.SymbolicTensor:
        """Advance the node values by one dense level (the matmul twin of ``_node_step``).

        Args:
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
        z = self.node_gene.combine(state, node_attrs, aggregated)
        activated = apply_activation(
            z,
            self.node_gene.attr_column(node_attrs, "activation"),
            self.node_gene.activation_options,
        )
        z = evoxc.select(self._output_row_mask(pop_size, max_nodes), z, activated)
        update = evoxc.logical_and(update_mask, evoxc.greater(live_count, zero))
        return evoxc.select(update, z, values)

    def _is_output_row(self, node_keys: evoxc.SymbolicTensor) -> evoxc.SymbolicTensor:
        """Flag the node rows read out as network outputs.

        Args:
            node_keys: ``(P, N) int32`` node keys, ``I_INF`` padded.

        Returns:
            ``(P, N) bool``, True on rows ``max_nodes-num_outputs..max_nodes-1``, which the
            node gene leaves unactivated.
        """
        pop_size, max_nodes = (int(size) for size in node_keys.shape)
        return self._output_row_mask(pop_size, max_nodes)

    def _output_row_mask(
        self, pop_size: int, max_nodes: int
    ) -> evoxc.SymbolicTensor:
        """Build the ``(P, N)`` output-row mask from static facts alone.

        Args:
            pop_size: Static population size ``P``.
            max_nodes: Static node capacity ``N``.

        Returns:
            ``(P, N) bool``, True on the last ``num_outputs`` rows.
        """
        rows = enp.arange(0, max_nodes, dtype=evoxc.int32)
        flag = evoxc.greater_equal(rows, _i32(max_nodes - self.num_outputs))
        return enp.broadcast_to(evoxc.reshape(flag, (1, max_nodes)), (pop_size, max_nodes))

    def _output_values(
        self, values: evoxc.SymbolicTensor, output_transform: Callable | None
    ) -> evoxc.SymbolicTensor:
        """Read the output rows out of the node values and apply the output transform.

        Args:
            values: ``(P, N) float32`` node values after the unrolled pass.
            output_transform: Optional elementwise map over the last axis; ``None`` is identity.

        Returns:
            ``(P, num_outputs) float32`` — rows ``max_nodes-num_outputs..max_nodes-1``, then
            ``output_transform`` when it is not ``None``.
        """
        max_nodes = int(values.shape[1])
        outputs = values[(slice(None), slice(max_nodes - self.num_outputs, max_nodes))]
        return outputs if output_transform is None else output_transform(outputs)

    # ------------------------------------------------------------------- operators
    def execute_mutation(
        self, state: State, key: evoxc.SymbolicTensor, *args: Any
    ) -> GenomeTensors:
        """Run the batched mutation operator on a whole population.

        Structural keys are allocated by the caller (not RNG draws) and must be unique across
        the population.  Accepts ``(payload, new_node_keys, new_conn_keys)`` and the
        four-array spelling ``(node_keys, node_attrs, conn_keys, conn_attrs, new_node_keys,
        new_conn_keys)``.

        Args:
            state: Host configuration container.
            key: Rank-0 ``evoxc.random`` key for this call.
            *args: Payload (``GenomeTensors`` or its four arrays) followed by
                ``new_node_keys`` and ``new_conn_keys``.

        Returns:
            ``GenomeTensors`` with identical static shapes — the mutated population.

        Raises:
            TensorNEATError: If the payload or the allocated key tensors have the wrong shapes.
        """
        payload, trailing = self._split_payload(*args)
        if len(trailing) != 2:
            raise TensorNEATError(
                "execute_mutation expects new_node_keys and new_conn_keys after the payload, "
                f"got {len(trailing)} trailing argument(s)"
            )
        new_node_keys, new_conn_keys = trailing
        pop_size = int(payload.node_keys.shape[0])
        self._check_key_shapes(
            (new_node_keys, "new_node_keys", (pop_size,)),
            (new_conn_keys, "new_conn_keys", (pop_size, 3)),
        )
        return self.mutation(
            state, self, key, *payload, new_node_keys, new_conn_keys
        )

    def execute_crossover(
        self, state: State, key: evoxc.SymbolicTensor, *args: Any
    ) -> GenomeTensors:
        """Recombine two batches of parents into one batch of offspring.

        Accepts ``(genome1, genome2)`` and the four-array-per-parent spelling
        ``(node_keys1, node_attrs1, conn_keys1, conn_attrs1, node_keys2, ...)``.

        Args:
            state: Host configuration container.
            key: Rank-0 ``evoxc.random`` key for this call.
            *args: Two payloads (``GenomeTensors`` or four arrays each); the first is the
                fitter parent, whose structure is kept.

        Returns:
            ``GenomeTensors`` — parent 1's keys with key-aligned recombined attributes.

        Raises:
            TensorNEATError: If the two parents' shapes differ.
        """
        first, remaining = self._split_payload(*args)
        second, trailing = self._split_payload(*remaining)
        if trailing:
            raise TensorNEATError(
                f"execute_crossover takes exactly two payloads, got {len(trailing)} extra "
                "argument(s)"
            )
        self._check_same_shapes(first, second, "execute_crossover")
        return self.crossover(state, self, key, *first, *second)

    def execute_distance(
        self, state: State, *args: Any
    ) -> evoxc.SymbolicTensor:
        """Compute one compatibility distance per aligned pair of genomes.

        Accepts ``(genome1, genome2)`` and the four-array-per-genome spelling.

        Args:
            state: Host configuration container.
            *args: Two payloads (``GenomeTensors`` or four arrays each).

        Returns:
            ``(P,) float32`` distances, ``0`` for an empty pair rather than NaN.

        Raises:
            TensorNEATError: If the two genomes' shapes differ.
        """
        first, remaining = self._split_payload(*args)
        second, trailing = self._split_payload(*remaining)
        if trailing:
            raise TensorNEATError(
                f"execute_distance takes exactly two payloads, got {len(trailing)} extra "
                "argument(s)"
            )
        self._check_same_shapes(first, second, "execute_distance")
        return self.distance(state, self, *first, *second)

    def _check_same_shapes(
        self, first: GenomeTensors, second: GenomeTensors, caller: str
    ) -> None:
        """Refuse two payloads whose static shapes differ.

        Args:
            first: Left payload.
            second: Right payload.
            caller: Method name used in the error message.

        Raises:
            TensorNEATError: If a field's shape disagrees between the two payloads.
        """
        for name, left, right in zip(first.field_names, first, second):
            if tuple(left.shape) != tuple(right.shape):
                raise TensorNEATError(
                    f"{caller}: payload field {name} has shape {tuple(left.shape)} in the "
                    f"first genome and {tuple(right.shape)} in the second"
                )

    @staticmethod
    def _check_key_shapes(*checks: tuple) -> None:
        """Validate the static shapes of caller-allocated structural key tensors.

        Args:
            *checks: ``(tensor, name, expected_shape)`` triples.

        Raises:
            TensorNEATError: If a tensor's shape differs from its expected shape, or its dtype
                is not an integer dtype (keys are never floats).
        """
        for tensor, name, expected in checks:
            shape = tuple(int(size) for size in tensor.shape)
            if shape != tuple(expected):
                raise TensorNEATError(
                    f"{name} must have shape {tuple(expected)}, got {shape}"
                )
            if tensor.dtype not in (evoxc.int8, evoxc.int16, evoxc.int32, evoxc.int64):
                raise TensorNEATError(
                    f"{name} must be an integer key tensor, got dtype {tensor.dtype}"
                )

    def check_cycles(
        self,
        node_keys: evoxc.SymbolicTensor,
        conn_keys: evoxc.SymbolicTensor,
        src_keys: evoxc.SymbolicTensor,
        dst_keys: evoxc.SymbolicTensor,
    ) -> evoxc.SymbolicTensor:
        """Ask whether adding one edge per individual would close a cycle.

        Documented delegation to ``.utils.check_cycles``; the mutation operator calls the
        predicate through the genome, so a genome can opt out by ``network_type`` — a
        recurrent genome accepts cycles, and its answer is a constant ``False``.

        Args:
            node_keys: ``(P, N) int32`` node keys, ``I_INF`` padded.
            conn_keys: ``(P, E, CK) int32`` connection keys, ``I_INF`` padded.
            src_keys: ``(P,) int32`` new edge source node keys.
            dst_keys: ``(P,) int32`` new edge destination node keys.

        Returns:
            ``(P,) bool`` True where the new edge would close a cycle; False for absent or
            degenerate keys, and always False outside a feedforward genome.
        """
        pop_size = int(node_keys.shape[0])
        if self.network_type != "feedforward":
            return enp.zeros((pop_size,), dtype=evoxc.bool_)
        return _check_cycles(node_keys, conn_keys, src_keys, dst_keys)

    # ------------------------------------------------------------- helper surface
    def node_exists(self, node_keys: evoxc.SymbolicTensor) -> evoxc.SymbolicTensor:
        """Mark the node rows that hold a real node key.

        Args:
            node_keys: ``(P, N) int32`` node keys, ``I_INF`` padded.

        Returns:
            ``(P, N) bool``, True where the row is not the padding sentinel.
        """
        return enp.not_equal(node_keys, _iinf(tuple(node_keys.shape), node_keys.dtype))

    def conn_exists(self, conn_keys: evoxc.SymbolicTensor) -> evoxc.SymbolicTensor:
        """Mark the connection rows that hold a real connection.

        Args:
            conn_keys: ``(P, E, CK) int32`` connection keys, ``I_INF`` padded.

        Returns:
            ``(P, E) bool``, True where the source-key column is not the padding sentinel.
        """
        columns = conn_keys[(slice(None), slice(None), slice(0, 1))]
        sources = evoxc.reshape(columns, (int(conn_keys.shape[0]), int(conn_keys.shape[1])))
        return enp.not_equal(
            sources, _iinf(tuple(sources.shape), sources.dtype)
        )

    def max_node_key(
        self, state: State, node_keys: evoxc.SymbolicTensor
    ) -> evoxc.SymbolicTensor:
        """Largest present node key per individual.

        Args:
            state: Host configuration container (unused; kept for the algorithm's call shape).
            node_keys: ``(P, N) int32`` node keys, ``I_INF`` padded.

        Returns:
            ``(P,) int32``: the largest non-padding node key, ``0`` for an individual with no
            node row at all (it has no key to continue from).
        """
        exists = self.node_exists(node_keys)
        safe = evoxc.select(
            exists, node_keys, enp.zeros(tuple(node_keys.shape), dtype=node_keys.dtype)
        )
        return evoxc.cast(enp.max(safe, axis=-1), evoxc.int32)

    def max_conn_marker(
        self, state: State, conn_keys: evoxc.SymbolicTensor
    ) -> evoxc.SymbolicTensor:
        """Largest present ``historical_marker`` per individual.

        Args:
            state: Host configuration container (unused; kept for the algorithm's call shape).
            conn_keys: ``(P, E, CK) int32`` connection keys, ``I_INF`` padded.

        Returns:
            ``(P,) int32``: the largest marker of a real connection row, or ``0`` for every
            individual when the conn gene declares no ``historical_marker`` column (there is
            then no marker to continue from and the third allocated key column is ignored).
        """
        pop_size = int(conn_keys.shape[0])
        if not self.conn_gene.uses_historical_marker:
            return enp.zeros((pop_size,), dtype=evoxc.int32)
        markers = self.conn_gene.key_column(conn_keys, "historical_marker")
        exists = self.conn_exists(conn_keys)
        safe = evoxc.select(
            exists, markers, enp.zeros(tuple(markers.shape), dtype=markers.dtype)
        )
        return evoxc.cast(enp.max(safe, axis=-1), evoxc.int32)

    # ------------------------------------------------------------------ host-only
    def get_input_idx(self) -> list[int]:
        """Keys and row positions of the input nodes.

        Returns:
            ``[0, 1, ..., num_inputs-1]``; the row/key invariant makes both coincide.
        """
        return list(self.input_idx)

    def get_output_idx(self) -> list[int]:
        """Keys and row positions of the output nodes.

        Returns:
            ``[max_nodes-num_outputs, ..., max_nodes-1]``; the row/key invariant makes both
            coincide.
        """
        return list(self.output_idx)

    def hash(self, genome: GenomeTensors | Any) -> evoxc.SymbolicTensor:
        """Hash every individual's payload for fast de-duplication (in-graph).

        An xor-mix over the ``int32`` key columns.  evoxc has no float/int bitcast, so float
        attributes are quantised (scaled by :data:`HASH_QUANT_SCALE`, rounded, cast to
        ``int32``) before mixing instead of being hashed by bit pattern; NaN padding quantises
        to one sentinel (``0``), so identical genomes hash identically and padding never
        distinguishes two payloads.

        Args:
            genome: ``(P, ...)`` payload batch, a ``GenomeTensors`` or its four arrays.

        Returns:
            ``(P,) uint32`` hashes, one per individual.
        """
        payload, _ = self._split_payload(genome)
        pop_size = int(payload.node_keys.shape[0])
        node_rows = self._row_hashes(
            self.node_gene, payload.node_keys, payload.node_attrs
        )
        conn_rows = self._row_hashes(
            self.conn_gene, payload.conn_keys, payload.conn_attrs
        )
        total = enp.zeros((pop_size,), dtype=evoxc.uint32)
        for index in range(int(node_rows.shape[1])):
            total = evoxc.bitwise_xor(total, _row_slice(node_rows, index))
        for index in range(int(conn_rows.shape[1])):
            total = evoxc.bitwise_xor(total, _row_slice(conn_rows, index))
        return total

    def _row_hashes(
        self,
        gene: Any,
        keys: evoxc.SymbolicTensor,
        attrs: evoxc.SymbolicTensor,
    ) -> evoxc.SymbolicTensor:
        """Hash every gene row of one family: key columns xor quantised attribute columns.

        Args:
            gene: The gene family owning ``hash`` (``node_gene`` or ``conn_gene``).
            keys: ``(P, R)`` or ``(P, R, CK)`` int32 key rows, ``I_INF`` padded.
            attrs: ``(P, R, CA)`` float32 attribute rows, NaN padded.

        Returns:
            ``(P, R) uint32`` row hashes.
        """
        key_rows = (
            keys if len(keys.shape) == 3 else evoxc.reshape(keys, (*keys.shape, 1))
        )
        quantised = enp.where(
            evoxc.isnan(attrs),
            enp.zeros((), dtype=evoxc.float32),
            evoxc.round(
                evoxc.multiply(
                    attrs, enp.full((), float(HASH_QUANT_SCALE), dtype=evoxc.float32)
                )
            ),
        )
        return evoxc.bitwise_xor(
            gene.hash(key_rows), gene.hash(evoxc.cast(quantised, evoxc.int32))
        )

    def network_dict(
        self, state: State, genome: GenomeTensors, whether_re_cound_idx: bool = True
    ) -> dict[str, dict]:
        """Convert one individual to host-side dicts (never traced).

        Host-only: converts with ``Tensor.numpy()`` and walks rows in Python, so the result must
        never be fed back into a graph.  ``genome`` is one individual (unbatched arrays, or a
        batch with ``P == 1``).

        Args:
            state: Host configuration container.
            genome: Payload of one individual.
            whether_re_cound_idx: Renumber hidden keys to be contiguous above
                ``max(input/output keys)`` via ``.utils.re_cound_idx``; input and output keys
                are untouched.

        Returns:
            ``{"nodes": {key: node_dict}, "conns": {(in_key, out_key): conn_dict}}`` from
            ``node_gene.to_dict``/``conn_gene.to_dict``; ``default.py`` adds
            ``topo_order``/``topo_layers``/``useful_nodes``.

        Raises:
            ConfigError: If ``genome`` holds more than one individual.
        """
        payload, _ = self._split_payload(genome)
        individual = self._individual(payload)
        node_keys, node_attrs, conn_keys, conn_attrs = individual
        if whether_re_cound_idx:
            node_keys, node_attrs, conn_keys = re_cound_idx(
                node_keys,
                node_attrs,
                conn_keys,
                self.get_input_idx(),
                self.get_output_idx(),
            )
        return {
            "nodes": self._get_node_dict(state, node_keys, node_attrs),
            "conns": self._get_conn_dict(state, conn_keys, conn_attrs),
        }

    def _get_node_dict(
        self,
        state: State,
        node_keys: Any,
        node_attrs: Any,
    ) -> dict:
        """Map node key -> node dict for one individual (host-only).

        Args:
            state: Host configuration container.
            node_keys: ``(N,)`` node key row of one individual, ``I_INF`` padded.
            node_attrs: ``(N, NA)`` attribute rows of the same individual, NaN padded.

        Returns:
            ``{node_key: node_gene.to_dict(...)}``, padded rows skipped.
        """
        node_dict = {}
        for row in range(node_keys.shape[0]):
            key = int(node_keys[row])
            if key == I_INF:
                continue
            node_dict[key] = self.node_gene.to_dict(
                state, node_keys[row : row + 1], node_attrs[row]
            )
        return node_dict

    def _get_conn_dict(
        self,
        state: State,
        conn_keys: Any,
        conn_attrs: Any,
    ) -> dict:
        """Map connection ``(in_key, out_key)`` -> conn dict for one individual (host-only).

        Args:
            state: Host configuration container.
            conn_keys: ``(E, CK)`` connection key rows of one individual, ``I_INF`` padded.
            conn_attrs: ``(E, CA)`` attribute rows of the same individual, NaN padded.

        Returns:
            ``{(in_key, out_key): conn_gene.to_dict(...)}``, padded rows skipped.
        """
        conn_dict = {}
        for row in range(conn_keys.shape[0]):
            if int(conn_keys[row, 0]) == I_INF:
                continue
            converted = self.conn_gene.to_dict(
                state, conn_keys[row], conn_attrs[row]
            )
            conn_dict[(converted["in"], converted["out"])] = converted
        return conn_dict

    def repr(self, state: State, genome: GenomeTensors, precision: int = 2) -> str:
        """Render one individual as a multi-line string (host-only).

        Args:
            state: Host configuration container.
            genome: Payload of one individual.
            precision: Decimal places forwarded to the gene ``repr`` hooks.

        Returns:
            ``"<ClassName>(nodes=<count>, conns=<count>):"`` then one line per node (annotated
            ``(input)``/``(output)``) and one line per connection; padded rows contribute no
            line at all.

        Raises:
            ConfigError: If ``genome`` holds more than one individual.
        """
        payload, _ = self._split_payload(genome)
        individual = self._individual(payload)
        node_keys, node_attrs, conn_keys, conn_attrs = individual
        present_nodes = [
            row for row in range(node_keys.shape[0]) if int(node_keys[row]) != I_INF
        ]
        present_conns = [
            row for row in range(conn_keys.shape[0]) if int(conn_keys[row, 0]) != I_INF
        ]
        inputs, outputs = set(self.get_input_idx()), set(self.get_output_idx())
        lines = [
            f"{type(self).__name__}(nodes={len(present_nodes)}, "
            f"conns={len(present_conns)}):"
        ]
        for row in present_nodes:
            text = self.node_gene.repr(
                state, node_keys[row : row + 1], node_attrs[row], precision
            )
            key = int(node_keys[row])
            if key in inputs:
                text += " (input)"
            elif key in outputs:
                text += " (output)"
            lines.append(text)
        for row in present_conns:
            lines.append(
                self.conn_gene.repr(state, conn_keys[row], conn_attrs[row], precision)
            )
        return "\n".join(lines) + "\n"

    def sympy_func(
        self,
        state: State,
        network: dict,
        sympy_input_transform: Any = None,
        sympy_output_transform: Any = None,
        backend: str = "numpy",
    ) -> tuple:
        """Build the sympy expression of a network and its NumPy evaluator (host-only).

        Declared here, implemented by the concrete genomes; the tuple is the contract with
        ``..common.sympy_tools``.

        Args:
            state: Host configuration container.
            network: Output of :meth:`network_dict`.
            sympy_input_transform: Sympy counterpart of ``input_transform``.
            sympy_output_transform: Sympy counterpart of ``output_transform``.
            backend: Codegen backend; ``"numpy"`` only in this rewrite (no JAX).

        Returns:
            ``(symbols, args_symbols, input_symbols, nodes_exprs, output_exprs, forward_func,
            topo_order, useful_nodes)`` — the legacy 8-tuple.

        Raises:
            NotImplementedError: Always — concrete genomes implement it.
        """
        raise NotImplementedError(
            "concrete genomes return the 8-tuple consumed by common.sympy_tools"
        )

    def visualize(self, network: dict, **kwargs: Any) -> None:
        """Draw a network (host-only; networkx/matplotlib imported inside the body).

        Declared here, implemented by the concrete genomes, so ``import tensorneat`` never
        imports ``networkx``/``matplotlib``.

        Args:
            network: Output of :meth:`network_dict`.
            **kwargs: Per-genome drawing options.

        Returns:
            ``None``; the figure goes to the path chosen by the concrete genome.

        Raises:
            NotImplementedError: Always — concrete genomes implement it.
        """
        raise NotImplementedError("concrete genomes draw the network and save the figure")
