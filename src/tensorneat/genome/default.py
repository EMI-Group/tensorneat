"""``DefaultGenome``: depth-unrolled, level-synchronous feedforward evaluation.

The forward pass seeds the input rows once, then walks the *static* depth levels in a plain
Python ``for`` loop, so the traced graph contains only ``gather``/``select``/``reduce_*`` and
elementwise ops — no ``scan``/``while_loop``/``cond`` — which is what lets ``evoxc.grad``
differentiate it.
Everything is batch-first (population on axis 0); the shared window reduce lives in
:class:`.base.BaseGenome` and must not be re-implemented here.
"""

import warnings
from collections.abc import Callable, Sequence
from typing import Any

import evoxc
import evoxc.numpy as enp
import numpy as np  # host-only: layout rotation and the ``forward_func`` read-out
import sympy as sp  # host-only: code generation

from ..common import State, TensorNEATError, find_useful_nodes, topological_sort
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
            TensorNEATError: If ``max_nodes``/``max_conns`` cannot hold the initial network, if
                ``init_hidden_layers`` needs an in-degree above ``max_in_degree``, if a
                gene/operator is not the expected type, or if a transform probe fails.
            ValueError: If a size argument is not a positive int.
        """
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

        1. ``values = full((P, N), NaN)`` with ``input_transform(inputs)`` written into the
           input rows (``0..num_inputs-1``) — only those rows hold a number. The seed is
           assembled as ``concatenate([seed, NaN (P, N - num_inputs)], axis=1)``: one op that
           leaves every other row at the ``NaN`` padding value, instead of a ``(P, N)`` mask.
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
        never ``max_nodes**2`` per node. A node whose level exceeds ``max_depth`` is never
        visited and keeps the ``NaN`` seed (the documented static-unroll limit; a rebuild with
        a larger ``max_depth`` is the fix, and ``Transformed.overflow`` flags it).

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
        """
        pop_size, max_nodes = (int(size) for size in transformed.node_level.shape)
        seed = inputs if self.input_transform is None else self.input_transform(inputs)
        values = evoxc.concatenate(
            [
                seed,
                enp.full(
                    (pop_size, max_nodes - self.num_inputs),
                    float("nan"),
                    dtype=evoxc.float32,
                ),
            ],
            axis=1,
        )
        hit_conn_attrs = self._hit_conn_attrs(transformed, conn_attrs)
        for level in range(1, self.max_depth + 1):
            # One level per unrolled step: the "has a live incoming slot" term is ANDed in by
            # ``_node_step`` itself, so the mask here stays a cheap int comparison.
            level_mask = evoxc.equal(
                transformed.node_level, enp.full((), level, dtype=evoxc.int32)
            )
            update_mask = evoxc.logical_and(level_mask, transformed.node_exists)
            values = self._node_step(
                state,
                transformed,
                node_attrs,
                hit_conn_attrs,
                values,
                update_mask,
            )
        return self._output_values(values, self.output_transform)

    def network_dict(
        self, state: State, genome: GenomeTensors, whether_re_cound_idx: bool = True
    ) -> dict[str, Any]:
        """Convert one individual to host dicts plus feedforward graph utilities.

        Extends :meth:`.base.BaseGenome.network_dict` with three host-only entries computed on
        the renumbered key sets — :func:`~tensorneat.common.host_graph.topological_sort` and
        :func:`~tensorneat.common.host_graph.find_useful_nodes` run in Python and never enter a
        graph.

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
            tensorneat.common.errors.ConfigError: If ``genome`` holds more than one individual.
            tensorneat.common.errors.GraphError: If the decoded topology has a cycle, which a
                feedforward genome cannot execute.
        """
        network = super().network_dict(state, genome, whether_re_cound_idx)
        node_keys = set(network["nodes"])
        conns = set(network["conns"])
        topo_order, topo_layers = topological_sort(node_keys, conns)
        network["topo_order"] = topo_order
        network["topo_layers"] = topo_layers
        network["useful_nodes"] = find_useful_nodes(
            node_keys, conns, set(self.get_output_idx())
        )
        return network

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
            TensorNEATError: If ``backend`` is not ``"numpy"`` — a compiled/JAX codegen backend
                no longer exists, and silently generating NumPy source for it would misreport
                what the caller got.
        """
        if backend != "numpy":
            raise TensorNEATError(
                "sympy_func backend must be 'numpy' (the only codegen backend of this "
                f"rewrite), got {backend!r}"
            )
        if sympy_input_transform is None:
            if self.input_transform is not None:
                warnings.warn(
                    "genome.input_transform is not None but sympy_input_transform is None: "
                    "the generated formula omits the input transform"
                )
            sympy_input_transform = lambda values: values  # noqa: E731 - identity elementwise
        if not isinstance(sympy_input_transform, list):
            sympy_input_transform = [sympy_input_transform] * self.num_inputs
        if sympy_output_transform is None and self.output_transform is not None:
            warnings.warn(
                "genome.output_transform is not None but sympy_output_transform is None: "
                "the generated formula omits the output transform"
            )
        input_idx = self.get_input_idx()
        output_idx = self.get_output_idx()
        order = network["topo_order"]
        hidden_idx = [
            key
            for key in network["nodes"]
            if key not in input_idx and key not in output_idx
        ]
        symbols = {}
        for key in network["nodes"]:
            if key in input_idx:
                # Two symbols per input: the raw ``i<k>`` and the transformed ``norm<k>``.
                offset = key - min(input_idx)
                symbols[-key - 1] = sp.Symbol(f"i{offset}")
                symbols[key] = sp.Symbol(f"norm{offset}")
            elif key in output_idx:
                symbols[key] = sp.Symbol(f"o{key - min(output_idx)}")
            else:
                symbols[key] = sp.Symbol(f"h{key - min(hidden_idx)}")
        nodes_exprs = {}
        args_symbols = {}
        for key in order:
            if key in input_idx:
                offset = key - min(input_idx)
                nodes_exprs[symbols[-key - 1]] = symbols[-key - 1]
                nodes_exprs[symbols[key]] = sympy_input_transform[offset](
                    symbols[-key - 1]
                )
                continue
            node_inputs = []
            for endpoints, conn_dict in network["conns"].items():
                if endpoints[1] != key:
                    continue
                message, conn_symbols = self.conn_gene.sympy_func(
                    state, conn_dict, symbols[endpoints[0]]
                )
                args_symbols.update(conn_symbols)
                node_inputs.append(message)
            expression, node_symbols = self.node_gene.sympy_func(
                state,
                network["nodes"][key],
                node_inputs,
                is_output_node=key in output_idx,
            )
            args_symbols.update(node_symbols)
            if key in output_idx and sympy_output_transform is not None:
                expression = sympy_output_transform(expression)
            nodes_exprs[symbols[key]] = expression
        input_symbols = [symbols[-key - 1] for key in input_idx]
        reduced_exprs = dict(nodes_exprs)
        for key in order:
            reduced_exprs[symbols[key]] = reduced_exprs[symbols[key]].subs(reduced_exprs)
        output_exprs = [reduced_exprs[symbols[key]] for key in output_idx]
        modules = ["numpy", AGG.sympy_module("numpy"), ACT.sympy_module("numpy")]
        evaluators = [
            sp.lambdify(input_symbols + list(args_symbols), expression, modules=modules)
            for expression in output_exprs
        ]

        def forward_func(values: Sequence[float]) -> np.ndarray:
            """Evaluate the symbolic network on ``num_inputs`` raw input values.

            Args:
                values: One value per input node, in input order (raw ``i<k>``, not
                    ``norm<k>`` — apply ``sympy_input_transform`` yourself if the genome has
                    one).

            Returns:
                ``(num_outputs,) float64`` numpy array with the gene values substituted.
            """
            return np.asarray(
                [
                    evaluator(*values, *args_symbols.values())
                    for evaluator in evaluators
                ],
                dtype=np.float64,
            )

        return (
            symbols,
            args_symbols,
            input_symbols,
            nodes_exprs,
            output_exprs,
            forward_func,
            order,
            network["useful_nodes"],
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
        """
        import networkx as nx
        from matplotlib import pyplot as plt

        conns = list(network["conns"])
        input_idx = self.get_input_idx()
        output_idx = self.get_output_idx()
        topo_order, topo_layers = network["topo_order"], network["topo_layers"]
        node2layer = {
            node: layer for layer, nodes in enumerate(topo_layers) for node in nodes
        }
        # Reorder the nodes of every layer by the position of their predecessors, so the
        # multipartite layout draws fewer crossing edges.
        subset_key = {}
        for layer, nodes in enumerate(topo_layers):
            if layer == 0 or len(nodes) == 1:
                subset_key[layer] = list(nodes)
                continue
            anchors = []
            for node in nodes:
                position = 0
                for row, predecessor in enumerate(topo_layers[layer - 1]):
                    if (predecessor, node) in conns:
                        position += row
                anchors.append(position)
            subset_key[layer] = [node for _, node in sorted(zip(anchors, nodes))]
        if reverse_node_order:
            subset_key = {layer: nodes[::-1] for layer, nodes in subset_key.items()}
        if not isinstance(size, tuple):
            size = (size, size, size)
        if not isinstance(color, tuple):
            color = (color, color, color)
        graph = nx.DiGraph()
        for node in topo_order:
            if node in input_idx:
                marker = (size[0], color[0])
            elif node in output_idx:
                marker = (size[2], color[2])
            else:
                marker = (size[1], color[1])
            graph.add_node(
                node, subset=node2layer[node], size=marker[0], color=marker[1]
            )
        graph.add_edges_from(conns)
        pos = nx.multipartite_layout(graph, subset_key=subset_key)
        angle = np.deg2rad(rotate)
        cos_angle, sin_angle = np.cos(angle), np.sin(angle)
        rotated_pos = {
            node: (
                cos_angle * x - sin_angle * y,
                sin_angle * x + cos_angle * y,
            )
            for node, (x, y) in pos.items()
        }
        nx.draw(
            graph,
            pos=rotated_pos,
            node_size=[data["size"] for data in graph.nodes.values()],
            node_color=[data["color"] for data in graph.nodes.values()],
            with_labels=with_labels,
            edgecolors=edgecolors,
            arrowstyle=arrowstyle,
            arrowsize=arrowsize,
            edge_color=edge_color,
            **kwargs,
        )
        plt.savefig(save_path, dpi=save_dpi)
        plt.close()
