"""``RecurrentGenome``: tick-unrolled Jacobi (synchronous) evaluation of a cyclic network.

No depth levels are used: the network is unrolled ``activate_time`` ticks in a plain Python
``for`` loop, each tick recomputing every existing node from the *previous* tick's values, so
the traced graph holds only ``gather``/``select``/``reduce_*`` and elementwise ops.
Everything is batch-first (population on axis 0) and the shared window reduce lives in
:class:`.base.BaseGenome`; HyperNEAT also uses this genome as its CPPN.
"""

from collections.abc import Callable, Sequence
from typing import Any

import evoxc
import numpy as np  # host-only: the ``activate_time`` validation of the constructor

from ..common import State, TensorNEATError
from ..common.tensorization import Transformed
from .base import BaseGenome
from .gene import BaseConn, BaseNode, DefaultConn, DefaultNode
from .operations import (
    BaseCrossover,
    BaseDistance,
    BaseMutation,
    DefaultCrossover,
    DefaultDistance,
    DefaultMutation,
)
from .window import any_slot, by_samples, fill


class RecurrentGenome(BaseGenome):
    """Recurrent NEAT genome, equivalent to the pre-rewrite ``RecurrentGenome``.

    ``network_type = "recurrent"``: the schedule is the same incoming-edge table as the
    feedforward one, but ``node_level`` is all zeros (there is no depth) and the whole network
    is evaluated by ``activate_time`` synchronous ticks.
    """

    network_type = "recurrent"

    def __init__(
        self,
        num_inputs: int,
        num_outputs: int,
        max_nodes: int = 50,
        max_conns: int = 100,
        node_gene: BaseNode | None = None,
        conn_gene: BaseConn | None = None,
        mutation: BaseMutation | None = None,
        crossover: BaseCrossover | None = None,
        distance: BaseDistance | None = None,
        output_transform: Callable | None = None,
        input_transform: Callable | None = None,
        init_hidden_layers: Sequence[int] = (),
        max_in_degree: int | None = None,
        max_depth: int = 16,
        activate_time: int = 10,
        key_span: int | None = None,
    ):
        """Configure a recurrent genome with NEAT-python default genes and operators.

        Gene/operator arguments default to ``None`` and are then built **per instance**
        (``DefaultNode()``, ``DefaultConn()``, ``DefaultMutation()``, ``DefaultCrossover()``,
        ``DefaultDistance()``), never shared through a mutable default argument, so two genomes
        never alias one gene object.  The initial network, the row/key invariant and the
        ``max_in_degree`` window width come from :class:`.base.BaseGenome`.

        Args:
            num_inputs: Input node count, also the network input width.
            num_outputs: Output node count, also the network output width.
            max_nodes: Static node capacity ``N``.
            max_conns: Static connection capacity ``E``.
            node_gene: Node gene; ``None`` builds a fresh ``DefaultNode()``.
            conn_gene: Connection gene; ``None`` builds a fresh ``DefaultConn()``.
            mutation: Mutation operator; ``None`` builds a fresh ``DefaultMutation()``.
            crossover: Crossover operator; ``None`` builds a fresh ``DefaultCrossover()``.
            distance: Distance operator; ``None`` builds a fresh ``DefaultDistance()``.
            output_transform: Optional elementwise map of the ``(P, num_outputs)`` read-out.
            input_transform: Kept for signature parity; **never applied** by this genome
                (legacy parity, see :meth:`forward`).
            init_hidden_layers: Width of each dense hidden layer; empty means one dense
                input-to-output layer.
            max_in_degree: Static window width ``K``; ``None`` resolves to
                ``max(initial in-degree, 4)``.
            max_depth: Static depth unroll count ``D``; unused by the tick loop, kept so the
                signature matches :class:`.default.DefaultGenome`.
            activate_time: Number of synchronous ticks unrolled in the forward pass; a static
                Python int, so changing it forces a rebuild.
            key_span: Static key bound ``S`` of the schedule's dense key table; ``None``
                resolves to ``max(max_nodes, num_inputs + num_outputs) + max_conns``. A caller
                that allocates structural keys beyond that default (the mutation operators take
                caller-allocated keys) must raise it here.

        Raises:
            TensorNEATError: If ``activate_time`` is not a positive int (zero ticks would read
                out the ``NaN`` seed, a silently dead population), or if the capacity
                invariants of the base class are violated.
            ValueError: If any other size argument is not a positive int.
        """
        if (
            isinstance(activate_time, bool)
            or not isinstance(activate_time, (int, np.integer))
            or int(activate_time) < 1
        ):
            raise TensorNEATError(
                f"activate_time must be a positive int, got {activate_time!r}"
            )
        self.activate_time = int(activate_time)
        super().__init__(
            num_inputs=num_inputs,
            num_outputs=num_outputs,
            max_nodes=max_nodes,
            max_conns=max_conns,
            node_gene=DefaultNode() if node_gene is None else node_gene,
            conn_gene=DefaultConn() if conn_gene is None else conn_gene,
            mutation=DefaultMutation() if mutation is None else mutation,
            crossover=DefaultCrossover() if crossover is None else crossover,
            distance=DefaultDistance() if distance is None else distance,
            output_transform=output_transform,
            input_transform=input_transform,
            init_hidden_layers=init_hidden_layers,
            max_in_degree=max_in_degree,
            max_depth=max_depth,
            key_span=key_span,
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
        """Evaluate the population by ``activate_time`` synchronous ticks, unrolled.

        The exact algebra, with ``P`` the population, ``rows = P * samples`` the evaluated row
        count (a problem tiles its dataset or rollout over the population axis, so row
        ``p * samples + j`` is individual ``p``), ``N`` ``max_nodes`` and ``T``
        ``activate_time``:

        1. ``values = full((rows, N), NaN)``; ``hit_conn_attrs =
           self._hit_conn_attrs(transformed, conn_attrs, samples)`` once, since edge attributes
           are constant over the ticks, and the attribute rows of absent nodes are rewritten to
           ``0`` in the same one-off step (``self._padding_safe_attrs``) so the node gene's
           ``int32`` option-index cast never sees a ``NaN``.
        2. ``for _ in range(self.activate_time)`` (a Python loop, unrolled at trace time):
           re-inject the raw ``inputs`` into the input rows, then
           ``values = self._node_step(state, transformed, node_attrs, hit_conn_attrs, values,
           update_mask=transformed.node_exists & (valid_count > 0))`` (``valid_count =
           reduce_sum(transformed.in_valid, axis=-1)``, the per-row live-slot count), which
           reads the *previous* tick's values for every node (Jacobi/synchronous, no in-place
           ordering dependency).  Re-injection is one ``concatenate`` of the input block with
           the non-input rows of the previous tick, so the values computed last tick survive in
           every non-input row.  ``input_transform`` is deliberately **not** applied — legacy
           parity: the pre-rewrite class fed raw inputs on every tick, which is also what
           HyperNEAT relies on when it uses this genome as a CPPN.
        3. ``return self._output_values(values, self.output_transform)`` — rows
           ``max_nodes-num_outputs..max_nodes-1`` of the last tick, then ``output_transform``.

        ``node_level`` is ignored on purpose: the schedule reports ``I_INF`` for every node on
        or downstream of a cycle, so a level-driven mask would produce an all-``NaN`` read-out.

        Args:
            state: Host configuration container.
            transformed: Execution schedule of the same population (``node_exists``,
                ``in_valid``/``in_src``/``in_conn (P, N, K)``; ``node_level`` is unused).
            node_keys: ``(P, N) int32`` node keys, ``I_INF`` padded.
            node_attrs: ``(P, N, NA) float32`` node attributes, NaN padded.
            conn_keys: ``(P, E, CK) int32`` connection keys, ``I_INF`` padded.
            conn_attrs: ``(P, E, CA) float32`` connection attributes, NaN padded.
            inputs: ``(P * samples, num_inputs) float32`` network inputs, tiled
                individual-major over the population axis and re-injected raw on every tick.

        Returns:
            ``(P * samples, num_outputs) float32`` outputs after ``output_transform``.

        Raises:
            TensorNEATError: If the batch is not a positive multiple of ``P`` (see
                :meth:`_sample_count`) — there is then no sample axis to expand the schedule
                with, and the two would silently disagree about their rows.
        """
        pop_size, max_nodes = (int(size) for size in transformed.node_exists.shape)
        samples = self._sample_count(inputs, pop_size)
        # One NaN element broadcast to the node block (see ``..window.fill``): the materialised
        # block would be the kernel's only population-scaled graph constant.
        values = fill(
            (pop_size * samples, max_nodes), float("nan"), dtype=evoxc.float32
        )
        # Absent rows read as ``0`` instead of ``NaN`` before the tick loop, so the option
        # index columns the node gene casts to ``int32`` are never NaN (see
        # ``BaseGenome._padding_safe_attrs``). The schedule's ``P`` rows are enough: the
        # statics and the mask are expanded to the evaluated ``P * samples`` rows below.
        node_attrs = self._padding_safe_attrs(node_attrs, node_keys)
        hit_conn_attrs = self._hit_conn_attrs(transformed, conn_attrs, samples)
        # "Has at least one live incoming slot" is a boolean *any* over the window, so it is
        # the same static slice fold as the once-per-tick liveness test of the window step
        # (:func:`..window.any_slot`) instead of a count: ``count > 0`` would be an
        # innermost-axis ``reduce_sum`` over the whole ``(rows, N, K)`` block, the most
        # expensive op shape on the reference backend, for a predicate that costs ``2K`` cheap
        # ops. Both terms are folded at ``P`` rows and expanded afterwards, which is the same
        # boolean per tiled row (the tiling repeats an individual, and an *any* over the
        # window is row-local).
        update_mask = evoxc.logical_and(
            by_samples(transformed.node_exists, samples, pop_size),
            by_samples(any_slot(transformed.in_valid), samples, pop_size),
        )
        for _ in range(self.activate_time):
            values = evoxc.concatenate(
                [
                    inputs,
                    values[(slice(None), slice(self.num_inputs, max_nodes))],
                ],
                axis=1,
            )
            values = self._node_step(
                state,
                transformed,
                node_attrs,
                hit_conn_attrs,
                values,
                update_mask,
            )
        return self._output_values(values, self.output_transform)

    def sympy_func(
        self,
        state: State,
        network: dict,
        sympy_input_transform: Any = None,
        sympy_output_transform: Any = None,
        backend: str = "numpy",
    ) -> tuple:
        """Symbolic form of a recurrent network — unsupported (host-only hook).

        A cyclic network has no finite closed form, so the legacy library refused; this
        rewrite keeps that behaviour instead of silently emitting a truncated formula.

        Args:
            state: Host configuration container.
            network: Output of :meth:`.base.BaseGenome.network_dict`.
            sympy_input_transform: Ignored.
            sympy_output_transform: Ignored.
            backend: Ignored.

        Returns:
            Never returns.

        Raises:
            TensorNEATError: Always — symbolic form is not defined for a recurrent genome, so
                this rewrite keeps the pre-rewrite refusal instead of emitting a truncated
                formula. Use ``..common.host_graph``/``network_dict`` to inspect the topology.
        """
        raise TensorNEATError(
            f"{type(self).__name__}.sympy_func is not supported: a cyclic network has no "
            "finite closed form, so no symbolic expression can be generated. Inspect the "
            "topology with network_dict(), or use DefaultGenome for a feedforward genome."
        )

    def visualize(self, network: dict, **kwargs: Any) -> None:
        """Drawing a recurrent network is unsupported (host-only hook).

        The layered layout assumes a depth-ordered feedforward graph, which a cyclic network
        does not have; the legacy library refused rather than drawing a misleading picture.

        Args:
            network: Output of :meth:`.base.BaseGenome.network_dict`.
            **kwargs: Ignored.

        Returns:
            Never returns.

        Raises:
            TensorNEATError: Always — the layered drawing assumes a depth-ordered feedforward
                graph, which a cyclic network does not have; the rewrite keeps the pre-rewrite
                refusal rather than drawing a misleading picture. Draw the topology yourself
                from ``network_dict()``, or use ``DefaultGenome``.
        """
        raise TensorNEATError(
            f"{type(self).__name__}.visualize is not supported: the layered layout assumes a "
            "depth-ordered feedforward graph, which a cyclic network does not have. Build the "
            "figure from network_dict(), or use DefaultGenome for a feedforward genome."
        )
