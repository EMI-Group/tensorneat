"""Genome base class: payload contract, schedule glue and forward/grad API.

This module owns host-side configuration (static sizes, gene/operator objects, transforms),
the payload/row-key invariant, the schedule entry point and the operator entry points.
The *shared* window-reduce machinery both concrete genomes compose lives in ``step.py``
(:class:`NodeStepMixin`) — ``default.py`` and ``recurrent.py`` must not duplicate it.
Every compute method is batch-first (population on axis 0) with static shapes: batch
dimensions are written explicitly and ``vmap``/``vectorize`` is never applied over these
kernels. Control flow (``scan``/``cond``/``while_loop``) appears only at the outermost layer
of a kernel — the feedforward level sweep's ``while_loop`` (``default.py``), the mutation
cycle predicate, and the ``while_loop``/``cond`` of the schedule derivation
(``../common/tensorization.py``) — never inside a level step and never under ``vmap``. The
exact gradient path in ``grad.py`` uses no region op at all: its level loop is a plain Python
``for`` unrolled at trace time, which is what keeps ``evoxc.grad`` usable, and the recurrent
path unrolls its ``activate_time`` ticks the same way.
The exact gradient path — a dense-weight twin of the window forward, needed because evoxc
0.1.0's ``gather`` VJP does not accumulate duplicate indices — lives in ``grad.py``;
:meth:`BaseGenome.grad` is its public entry point and delegates there.

## Cross-module contracts
Every name below is provided by the ``common`` node and is never re-derived here.
``../common.tensorization.derive_schedule`` is the only schedule derivation.
``../common.functions.apply_activation``/``apply_aggregation`` are the only activation and
window-reduce kernels, reached through the node gene.
``../common.tools.I_INF`` is the ``int32`` "no index" sentinel padding every key array, and
``./window._gather_prepared`` is the flat-index gather the window step uses (prepared once per
forward pass by ``window._slot_index``; ``../common.tools.masked_flat_gather`` is not on this
path).
``../common.State``/``StatefulBaseClass`` are the immutable host containers and
``../common.errors.TensorNEATError`` (``ConfigError``) covers every configuration violation.

## Cost budget (measured, numpy backend)
A forward pass costs about one unrolled level per network depth: measured 1.8 ms per level at `P=1000`, `max_nodes=50`, `K=2` with ~20 ops per level.
Op count per unrolled level — not element count — is the budget: keep the window step thin, gather connection attributes once per forward pass, and never re-derive the schedule inside a level; depth, tick and option loops are unrolled at trace time, so compile time grows with `max_depth`/`activate_time`/option count, while single-option configurations emit no selection op at all.
"""

from collections.abc import Callable, Sequence
from typing import Any

import evoxc
import evoxc.numpy as enp
import numpy as np

from ..common import (
    I_INF,
    State,
    StatefulBaseClass,
    TensorNEATError,
    Transformed,
    derive_schedule,
)
from .gene import BaseConn, BaseNode
from .grad import build_grad
from .host import BaseGenomeHost
from .operations import BaseCrossover, BaseDistance, BaseMutation
from .operations.distance import DistanceDigest
from .step import NodeStepMixin
from .window import fill
from .utils import GenomeTensors


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


class BaseGenome(NodeStepMixin, BaseGenomeHost, StatefulBaseClass):
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
            # A single NaN element broadcast over the padding rows: a materialised
            # ``(P, pad_rows, NA)`` block is a graph constant proportional to ``P`` (see
            # ``.window.fill``) for a value every one of those rows shares.
            node_attrs = evoxc.concatenate(
                [
                    head_attrs,
                    fill(
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
                    fill(
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
        ``..common.tensorization.derive_schedule.fn`` (the ``@evoxc.defn`` object itself is not
        callable) with the input/output keys as ``int32`` in-graph constants and the six statics
        of this genome. The last static is ``acyclic``, derived from :attr:`network_type` (never
        a constructor argument): a feedforward genome cannot hold a cycle because
        ``execute_mutation`` refuses every cycle-closing edge, so the live-cycle detection is
        dropped (a *cyclic* payload, which its own operators cannot build, would then saturate
        its levels instead of reading ``I_INF``); a recurrent genome keeps ``acyclic=False`` and
        so keeps the ``I_INF`` level of every node on or downstream of a cycle. The derivation
        must stay outside any differentiated graph (``argsort`` has no AD rule), which is why
        :meth:`grad` takes an already-derived ``Transformed``.

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
            TensorNEATError: If the arguments match none of the accepted spellings, or if a
                selected array is not batch-first — ``node_keys`` must be ``(P, max_nodes)``,
                ``conn_keys`` ``(P, max_conns, key_width)``, all leading dimensions equal, and
                (four-array spelling) the attribute arrays led by the same population size.
                A bare per-individual row is the common defect: it has no population axis, so
                the error says so instead of letting the shape unpack raise ``ValueError``.
        """
        node_keys, conn_keys = self._schedule_inputs(arrays)
        self._check_schedule_shapes(arrays, node_keys, conn_keys)
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
            self.network_type == "feedforward",
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

    @staticmethod
    def _check_schedule_shapes(arrays: tuple, node_keys: Any, conn_keys: Any) -> None:
        """Validate the rank and population alignment of a schedule call's arrays.

        ``transform`` is the batch-first entry point of the genome, and a caller that slices
        one individual out of a payload hands it a bare row (``node_keys (N,)``,
        ``conn_keys (E, key_width)``).  Without this check the population unpack inside
        ``transform`` raises a raw ``ValueError`` from the shape tuple instead of the
        documented configuration error.  Only ``.shape`` is read — a tuple of ints both on a
        traced ``evoxc.SymbolicTensor`` and on a host array — so the check costs no traced op
        and no host conversion, and it runs identically inside a trace (where the exception
        propagates out of ``build``) and on host data.

        The static sizes are deliberately **not** compared with ``self.max_nodes``/
        ``self.max_conns``; only the *internal* agreement of one call is a contract.

        Args:
            arrays: The positional arguments of ``transform``, used to locate the attribute
                arrays and to name their offsets in the four-array spelling.
            node_keys: The selected ``(P, max_nodes) int32`` node keys.
            conn_keys: The selected ``(P, max_conns, key_width) int32`` connection keys.

        Raises:
            TensorNEATError: If either key array has the wrong rank (a rank-1 ``node_keys`` or
                rank-2 ``conn_keys`` is reported as a per-individual row that needs a leading
                population axis), if the two disagree on the population size, or if an
                attribute array's leading dimension is not that population size.
        """
        node_shape = tuple(int(size) for size in node_keys.shape)
        conn_shape = tuple(int(size) for size in conn_keys.shape)
        row_hint = (
            " This is a per-individual row, not a batch: it needs a leading population axis "
            "(e.g. payload[None] / node_keys[None, :]). Pipeline.auto_run returns a "
            "(1, ...) batch, so keep that axis when slicing one individual out."
        )
        if len(node_shape) != 2:
            raise TensorNEATError(
                "transform expects node_keys of shape (P, max_nodes) with the population on "
                f"axis 0, got shape {node_shape}."
                + (row_hint if len(node_shape) == 1 else "")
            )
        if len(conn_shape) != 3:
            raise TensorNEATError(
                "transform expects conn_keys of shape (P, max_conns, key_width) with the "
                f"population on axis 0, got shape {conn_shape}."
                + (row_hint if len(conn_shape) == 2 else "")
            )
        if conn_shape[0] != node_shape[0]:
            raise TensorNEATError(
                "transform got node_keys and conn_keys of different population sizes: "
                f"node_keys has shape {node_shape} and conn_keys has shape {conn_shape}"
            )
        if len(arrays) == 1 and isinstance(arrays[0], GenomeTensors):
            attributes = [
                (f"payload field {name}", tensor)
                for name, tensor in zip(arrays[0].field_names, arrays[0])
                if name in ("node_attrs", "conn_attrs")
            ]
        elif len(arrays) == 4:
            attributes = [
                (f"argument {offset} ({name})", arrays[offset])
                for offset, name in ((1, "node_attrs"), (3, "conn_attrs"))
            ]
        else:
            attributes = []
        for label, tensor in attributes:
            shape = tuple(int(size) for size in tensor.shape)
            if not shape or shape[0] != node_shape[0]:
                raise TensorNEATError(
                    "transform expects the attribute arrays to lead with the population size "
                    f"of node_keys ({node_shape[0]}), but {label} has shape {shape} "
                    f"(node_keys has shape {node_shape})"
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

    def prepare_pass(
        self,
        state: State,
        transformed: Transformed,
        node_keys: evoxc.SymbolicTensor,
        node_attrs: evoxc.SymbolicTensor,
        conn_keys: evoxc.SymbolicTensor,
        conn_attrs: evoxc.SymbolicTensor,
        samples: int = 1,
    ) -> Any:
        """Derive the loop-invariant operands of one forward pass (abstract).

        ``conn_keys`` is accepted for symmetry with :meth:`forward`; ``samples`` is the static
        tiling factor of the ``P * samples`` evaluated rows.
        """
        raise NotImplementedError(
            "concrete genomes derive one forward pass's loop-invariant operands"
        )

    def forward_prepared(
        self,
        state: State,
        transformed: Transformed,
        prepared: Any,
        inputs: evoxc.SymbolicTensor,
    ) -> evoxc.SymbolicTensor:
        """Run one forward pass from ``prepared`` (:meth:`prepare_pass`) (abstract)."""
        raise NotImplementedError(
            "concrete genomes run one forward pass from a prepared pass"
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

        The differentiated pass is ``.grad.dense_forward``, not the window step of
        :meth:`_node_step`: evoxc 0.1.0 transposes ``gather`` with *put* semantics, so the
        exact path aggregates with a per-individual dense weight ``dot`` instead
        (:func:`.grad.build_grad`).

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
        return build_grad(
            self,
            state,
            transformed,
            node_keys,
            node_attrs,
            conn_keys,
            conn_attrs,
            inputs,
            loss_fn,
        )

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

    def prepare_distance(self, state: State, *args: Any) -> "DistanceDigest":
        """Digest a batch of genomes for repeated compatibility distance.

        Run once per generation: the result is the static-length-per-genome data
        (packed sorted key rows, attribute rows in the same order, gene counts)
        that :meth:`measure_distance` needs, so measuring one centre against the
        whole population costs no per-pair sort of the population.

        Args:
            state: Host configuration container.
            *args: One payload (``GenomeTensors`` or the four-array spelling).

        Returns:
            ``DistanceDigest`` — a 19-field pytree (15 tensors plus four static
            scaffolds), batch-first over the input population, cacheable for one
            generation.

        Raises:
            TensorNEATError: If more than one payload is passed.
        """
        payload, trailing = self._split_payload(*args)
        if trailing:
            raise TensorNEATError(
                f"prepare_distance takes exactly one payload, got {len(trailing)} extra "
                "argument(s)"
            )
        return self.distance.prepare(state, self, *payload)

    def measure_distance(
        self, state: State, *args: Any
    ) -> evoxc.SymbolicTensor:
        """Compatibility distance of one centre genome against a digested population.

        Args:
            state: Host configuration container.
            *args: One centre payload (``GenomeTensors`` or four arrays; a batch of
                ``1`` is broadcast to the digest's population axis) followed by the
                ``DistanceDigest`` returned by :meth:`prepare_distance`.

        Returns:
            ``(P,) float32`` distances, bit-identical to :meth:`execute_distance`
            on the same centre and population, and ``0`` for an empty pair.

        Raises:
            TensorNEATError: If the arguments are not one payload plus one digest.
        """
        centre, remaining = self._split_payload(*args)
        if len(remaining) != 1:
            raise TensorNEATError(
                "measure_distance takes a centre payload followed by exactly one "
                f"DistanceDigest, got {len(remaining)} trailing argument(s)"
            )
        return self.distance.measure(state, self, *centre, remaining[0])

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
