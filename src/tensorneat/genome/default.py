"""``DefaultGenome``: depth-unrolled, level-synchronous feedforward evaluation.

The forward pass seeds the input rows once, then walks the *static* depth levels in a plain
Python ``for`` loop, so the traced graph contains only ``gather``/``select``/``reduce_*`` and
elementwise ops — no ``scan``/``while_loop``/``cond`` — which is what lets ``evoxc.grad``
differentiate it.
Everything is batch-first (population on axis 0); the shared window reduce lives in
:class:`.base.BaseGenome` and must not be re-implemented here.
"""

from collections.abc import Callable, Sequence
from typing import Any

import evoxc
import evoxc.numpy as enp  # noqa: F401 - tensor creation for the implementation wave
import numpy as np  # noqa: F401 - host-only graph utilities
import sympy as sp  # noqa: F401 - host-only code generation

from ..common import (
    I_INF,
    State,
    TensorNEATError,
    find_useful_nodes,
    topological_sort,
)
from ..common.functions import ACT, AGG
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


class DefaultGenome(BaseGenome):
    """Feedforward NEAT genome, behaviourally equivalent to NEAT-python.

    ``network_type = "feedforward"``: the schedule carries ``node_level`` (longest-path depth,
    inputs at level 0) and the forward pass evaluates one level per Python-unrolled step, so a
    node is computed only after all of its real sources already have a value.
    """

    network_type = "feedforward"

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
    ):
        """Configure a feedforward genome with NEAT-python default genes and operators.

        Gene/operator arguments default to ``None`` and are then built **per instance**
        (``DefaultNode()``, ``DefaultConn()``, ``DefaultMutation()``, ``DefaultCrossover()``,
        ``DefaultDistance()``), never shared through a mutable default argument: a genome owns
        its gene objects, and two genomes that shared one would silently mutate each other's
        attribute tables.

        See :class:`.base.BaseGenome` for the dense initial network
        (``num_inputs -> init_hidden_layers -> num_outputs``), the row/key invariant and the
        transform probes.

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
            input_transform: Optional elementwise map of ``inputs`` before it is written into
                the input rows.
            init_hidden_layers: Width of each dense hidden layer; empty means one dense
                input-to-output layer.
            max_in_degree: Static window width ``K``; ``None`` resolves to
                ``max(initial in-degree, 4)``.
            max_depth: Static depth unroll count ``D``; deeper networks leave their unreachable
                tail NaN (and raise ``Transformed.overflow``).

        Raises:
            NotImplementedError: Skeleton stub; the per-instance default construction and the
                ``super().__init__`` call land in the implementation wave.
            TensorNEATError: (implementation wave) if ``max_nodes``/``max_conns`` cannot hold
                the initial network or ``init_hidden_layers`` exceeds ``max_in_degree``.
        """
        raise NotImplementedError(
            "build DefaultNode/DefaultConn/DefaultMutation/DefaultCrossover/DefaultDistance "
            "per instance where the argument is None, then call super().__init__"
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
        """Evaluate the population level by level, in one unrolled graph.

        The exact algebra, with ``P`` the population, ``N`` ``max_nodes`` and ``D``
        ``max_depth``:

        1. ``values = full((P, N), NaN)``, then ``values = select(input_row, input_transform(
           inputs), values)`` — only the input rows (``0..num_inputs-1``) hold a number.
        2. ``hit_conn_attrs = self._hit_conn_attrs(transformed, conn_attrs)`` **once**: edge
           attributes are constant during a forward pass, so the per-slot gather is hoisted out
           of the loop.
        3. ``for level in range(1, self.max_depth + 1)`` (a Python loop, hence unrolled at trace
           time): ``values = self._node_step(state, transformed, node_attrs, hit_conn_attrs,
           values, update_mask=(transformed.node_level == level) & transformed.node_exists &
           (valid_count > 0))``, with ``valid_count = reduce_sum(transformed.in_valid,
           axis=-1)`` the static "this row has at least one incoming connection" gate (a
           per-row live-slot count, not ``.utils.valid_count``, which counts existing rows).
           The dynamic "at least one *live* slot" term is ANDed in inside ``_node_step``, so a
           node whose sources are all NaN keeps NaN and its descendants lose their only live
           source: the legacy NaN poisoning, reproduced exactly.
        4. ``return self._output_values(values, self.output_transform)`` — rows
           ``max_nodes-num_outputs..max_nodes-1``, then ``output_transform``.

        Cost is ``O(D x N x K)`` elementwise work: proportional to depth and real in-degree,
        never ``max_nodes**2`` per node.

        Args:
            state: Host configuration container.
            transformed: Execution schedule of the same population (``node_level``,
                ``node_exists (P, N)``, ``in_valid``/``in_src``/``in_conn (P, N, K)``).
            node_keys: ``(P, N) int32`` node keys, ``I_INF`` padded.
            node_attrs: ``(P, N, NA) float32`` node attributes, NaN padded.
            conn_keys: ``(P, E, CK) int32`` connection keys, ``I_INF`` padded.
            conn_attrs: ``(P, E, CA) float32`` connection attributes, NaN padded.
            inputs: ``(P, num_inputs) float32`` network inputs.

        Returns:
            ``(P, num_outputs) float32`` outputs after ``output_transform``.

        Raises:
            NotImplementedError: Skeleton stub; the unrolled loop is pending.
        """
        raise NotImplementedError(
            "NaN-seed values, write input_transform(inputs) into the input rows, gather "
            "hit_conn_attrs once, unroll _node_step over levels 1..max_depth with the level "
            "mask, then read out the output rows and apply output_transform"
        )

    def network_dict(
        self, state: State, genome: GenomeTensors, whether_re_cound_idx: bool = True
    ) -> dict[str, Any]:
        """Convert one individual to host dicts plus feedforward graph utilities.

        Extends :meth:`.base.BaseGenome.network_dict` with three host-only entries computed on
        the renumbered key sets — ``topological_sort_python`` and ``find_useful_nodes`` run in
        Python and never enter a graph.

        Args:
            state: Host configuration container.
            genome: Payload of one individual.
            whether_re_cound_idx: Renumber hidden keys to be contiguous above
                ``max(input/output keys)`` before converting.

        Returns:
            ``{"nodes", "conns", "topo_order", "topo_layers", "useful_nodes"}``: the key dicts
            of the base method plus the topological order, the layer grouping and the set of
            nodes that actually reach an output (nodes outside it may be NaN and are skipped by
            code generation).

        Raises:
            NotImplementedError: Skeleton stub.
        """
        raise NotImplementedError(
            "super().network_dict(...) plus topological_sort_python(nodes, conns) -> "
            "topo_order/topo_layers and find_useful_nodes(..., output_idx) -> useful_nodes"
        )

    def sympy_func(
        self,
        state: State,
        network: dict,
        sympy_input_transform: Any = None,
        sympy_output_transform: Any = None,
        backend: str = "numpy",
    ) -> tuple:
        """Build sympy expressions of a network and a NumPy evaluator (host-only).

        The legacy 8-tuple contract with ``..common.sympy_tools``: symbols are named by role
        (``i<k>`` raw input, ``norm<k>`` transformed input, ``h<k>`` hidden, ``o<k>`` output),
        nodes are emitted in ``topo_order`` and every node expression is substituted into its
        successors, so ``output_exprs`` are closed forms.  ``backend`` is ``"numpy"`` only in
        this rewrite (there is no JAX); ``AGG``/``ACT`` contribute their sympy modules to the
        ``lambdify`` call.  A missing ``sympy_*_transform`` while the genome has the
        corresponding transform emits a warning, as before.

        Args:
            state: Host configuration container.
            network: Output of :meth:`network_dict` (needs ``topo_order``/``useful_nodes``).
            sympy_input_transform: Sympy counterpart of ``input_transform``; a list with one
                entry per input, or a single callable applied to each.
            sympy_output_transform: Sympy counterpart of ``output_transform``.
            backend: Codegen backend, ``"numpy"`` only.

        Returns:
            ``(symbols, args_symbols, input_symbols, nodes_exprs, output_exprs, forward_func,
            topo_order, useful_nodes)`` — ``forward_func`` maps a sequence of inputs to a NumPy
            array of outputs with the gene values already substituted.

        Raises:
            NotImplementedError: Skeleton stub; the expression walk is pending.
            AssertionError: (implementation wave) if ``backend`` is not ``"numpy"``.
        """
        raise NotImplementedError(
            "walk topo_order, build per-node sympy expressions from gene sympy_func, substitute "
            "into successors, lambdify with the numpy backend -> the 8-tuple"
        )

    def visualize(
        self,
        network: dict,
        rotate: float = 0,
        reverse_node_order: bool = False,
        size: tuple = (300, 300, 300),
        color: tuple = ("yellow", "white", "blue"),
        with_labels: bool = False,
        edgecolors: str = "k",
        arrowstyle: str = "->",
        arrowsize: int = 3,
        edge_color: tuple = (0.3, 0.3, 0.3),
        save_path: str = "network.svg",
        save_dpi: int = 800,
        **kwargs: Any,
    ) -> None:
        """Draw a network as a layered directed graph (host-only).

        ``networkx`` and ``matplotlib`` are imported **inside** the body, so
        ``import tensorneat`` never pulls them in.  Nodes are placed with the multipartite
        layout of ``topo_layers``, reordered per layer to reduce edge crossings, then rotated by
        ``rotate`` degrees; input nodes use ``size[0]``/``color[0]``, hidden nodes the middle
        entry and output nodes the last.

        Args:
            network: Output of :meth:`network_dict` (needs ``topo_order``/``topo_layers``).
            rotate: Rotation of the layout in degrees.
            reverse_node_order: Reverse the node order within every layer.
            size: Node sizes ``(input, hidden, output)`` in drawing units.
            color: Node colours ``(input, hidden, output)``.
            with_labels: Draw the node keys.
            edgecolors: Marker edge colour.
            arrowstyle: Matplotlib arrow style of the edges.
            arrowsize: Arrow size in drawing units.
            edge_color: Edge colour.
            save_path: Output path of the figure.
            save_dpi: Resolution of the saved figure.
            **kwargs: Extra keyword arguments forwarded to ``networkx.draw``.

        Returns:
            ``None``; the figure is written to ``save_path`` and the matplotlib figure closed.

        Raises:
            NotImplementedError: Skeleton stub; the drawing code is pending.
        """
        raise NotImplementedError(
            "import networkx/matplotlib inside the body, build the DiGraph from "
            "network['conns'], multipartite_layout over topo_layers, rotate, draw and savefig"
        )
