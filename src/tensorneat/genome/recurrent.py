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
import evoxc.numpy as enp  # noqa: F401 - tensor creation for the implementation wave
import numpy as np  # noqa: F401 - host-only helpers
import sympy as sp  # noqa: F401 - declared hooks raise; kept for the shared signature

from ..common import I_INF, State, TensorNEATError
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
from .utils import GenomeTensors


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

        Raises:
            NotImplementedError: Skeleton stub; the per-instance default construction and the
                ``super().__init__`` call land in the implementation wave.
            TensorNEATError: (implementation wave) if ``activate_time`` is not a positive int
                or the capacity invariants of the base class are violated.
        """
        raise NotImplementedError(
            "build DefaultNode/DefaultConn/DefaultMutation/DefaultCrossover/DefaultDistance "
            "per instance where the argument is None, store activate_time, then call "
            "super().__init__"
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

        The exact algebra, with ``P`` the population, ``N`` ``max_nodes`` and ``T``
        ``activate_time``:

        1. ``values = full((P, N), NaN)``; ``hit_conn_attrs = self._hit_conn_attrs(transformed,
           conn_attrs)`` once, since edge attributes are constant over the ticks.
        2. ``for _ in range(self.activate_time)`` (a Python loop, unrolled at trace time):
           re-inject the raw ``inputs`` into the input rows, then
           ``values = self._node_step(state, transformed, node_attrs, hit_conn_attrs, values,
           update_mask=transformed.node_exists & (valid_count > 0))`` (``valid_count =
           reduce_sum(transformed.in_valid, axis=-1)``, the per-row live-slot count), which
           reads the *previous* tick's values for every node (Jacobi/synchronous, no in-place
           ordering dependency).  ``input_transform`` is deliberately **not** applied — legacy parity:
           the pre-rewrite class fed raw inputs on every tick, which is also what HyperNEAT
           relies on when it uses this genome as a CPPN.
        3. ``return self._output_values(values, self.output_transform)`` — rows
           ``max_nodes-num_outputs..max_nodes-1`` of the last tick, then ``output_transform``.

        Args:
            state: Host configuration container.
            transformed: Execution schedule of the same population (``node_exists``,
                ``in_valid``/``in_src``/``in_conn (P, N, K)``; ``node_level`` is unused).
            node_keys: ``(P, N) int32`` node keys, ``I_INF`` padded.
            node_attrs: ``(P, N, NA) float32`` node attributes, NaN padded.
            conn_keys: ``(P, E, CK) int32`` connection keys, ``I_INF`` padded.
            conn_attrs: ``(P, E, CA) float32`` connection attributes, NaN padded.
            inputs: ``(P, num_inputs) float32`` network inputs, re-injected raw on every tick.

        Returns:
            ``(P, num_outputs) float32`` outputs after ``output_transform``.

        Raises:
            NotImplementedError: Skeleton stub; the tick loop is pending.
        """
        raise NotImplementedError(
            "NaN-seed values, gather hit_conn_attrs once, unroll activate_time ticks that "
            "re-inject the raw inputs and run _node_step with mask node_exists & "
            "(valid_count > 0), then read out the output rows and apply output_transform"
        )

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
            NotImplementedError: Always — symbolic form is not defined for a recurrent genome.
        """
        raise NotImplementedError(
            "symbolic form / visualization is not defined for a recurrent genome"
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
            NotImplementedError: Always — visualization is not defined for a recurrent genome.
        """
        raise NotImplementedError(
            "symbolic form / visualization is not defined for a recurrent genome"
        )
