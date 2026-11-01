"""HyperNEAT: decode a CPPN population into a fixed substrate network, batch-first.

Hypercube-based NEAT evolves only the CPPN (an ordinary NEAT network).  Each CPPN is queried once
per substrate connection at the coordinates of that connection's two endpoints, and the CPPN's single
scalar output becomes that connection's weight.  The resulting *substrate* network -- fixed topology,
coordinates in N dimensions -- is the phenotype that gets evaluated.

Roles in the tree
    * ``self.neat`` owns all evolution: ``ask`` / ``tell`` / speciation / RNG consumption.
    * ``self.hyper_genome`` is a dense fixed-topology executor (``RecurrentGenome`` here,
      ``DefaultGenome`` in :mod:`tensorneat.algorithm.hyperneat.hyperneat_feedforward`): it is never
      mutated, speciated or recombined, and its connection weights are repainted from the CPPN on
      every :meth:`HyperNEAT.transform`.
    * ``self.substrate`` is a plain **host** object (see :mod:`tensorneat.algorithm.hyperneat.substrate`).
      It exposes static geometry only, is never traced, and never appears in ``State``.

Why this module contains no ``vmap``
    ``evoxc`` refuses to vectorize region-bearing control flow and the genome kernels are full of
    ``scan``/``cond`` regions, so the query axis is *fused into the population axis* instead: the
    per-individual CPPN schedule is repeated along the query axis and flattened to ``(P * Q, ...)``,
    which is exactly the batching that ``vmap(self.neat.forward, in_axes=(None, None, 0))`` performed
    in the pre-rewrite library.  Every kernel therefore keeps the plain batch-first contract of
    ``Genome.forward``: ``inputs (P, *input_shape) -> outputs (P, *output_shape)``.

Notation used throughout (all sizes are static ints known at construction)
    ``P`` = ``neat.pop_size``; ``Q`` = ``substrate.conns_cnt``; ``N`` = ``substrate.nodes_cnt``;
    ``I`` = ``substrate.num_inputs`` (substrate input neurons *including* the trailing bias);
    ``O`` = ``substrate.num_outputs``;
    ``C`` = ``neat.num_inputs`` = ``substrate.query_coors.shape[1]`` = ``2 * CD`` (CPPN input width,
    i.e. both endpoint coordinates of dimension ``CD``; ``C`` and ``I`` are *different* numbers and
    must never be interchanged);
    ``node_len`` = ``hyper_genome.node_gene.length``; ``conn_len`` = ``hyper_genome.conn_gene.length``.

evoxc facts this module is designed against
    * ``evoxc.constant`` takes a ``core.Tensor`` (``evoxc.constant(evoxc.from_numpy(arr))``) and warns
      above ``EVOXC_LARGE_CONSTANT_BYTES`` (1 MiB default); the substrate tables are a few KiB.
    * ``enp.broadcast_to`` exists and has a batching rule; ``evoxc.tile`` exists but has none, so
      ``broadcast_to`` is the preferred way to repeat a static table.
    * ``evoxc.select`` is the mask primitive (``enp.where`` is the same op); branches are both
      evaluated, so masking never short-circuits.
    * There is no bitcast / reinterpret op anywhere in evoxc; float gene attributes therefore reach
      the payload builder as float32 values and their row encoding stays genome-owned.
"""

from typing import Any, Callable, Optional

import numpy as np

import evoxc
import evoxc.numpy as enp

from tensorneat.common import ACT, AGG, State
from tensorneat.common.errors import TensorNEATError
from tensorneat.genome import BaseConn, BaseNode, RecurrentGenome

from ..base import BaseAlgorithm
from ..neat import NEAT
from .substrate import BaseSubstrate


class HyperNEAT(BaseAlgorithm):
    """HyperNEAT with a recurrent substrate, executed by a fixed ``RecurrentGenome``.

    The wrapped :class:`~tensorneat.algorithm.neat.NEAT` evolves the CPPN; this class turns a whole
    CPPN population into one batch-first execution schedule of the substrate network.

    Attributes:
        substrate (BaseSubstrate): Static host-side geometry of the phenotype network.
        neat (NEAT): The CPPN evolution algorithm; owns ``randkey`` consumption and all state keys.
        hyper_genome (RecurrentGenome): Dense fixed-topology executor; ``max_nodes ==
            substrate.nodes_cnt`` and ``max_conns == substrate.conns_cnt``, so no NaN padding exists.
        weight_threshold (float): Dead-zone half-width of the CPPN-to-weight map.
        max_weight (float): Saturating magnitude of the mapped weight.
        aggregation (Callable): Fixed aggregation used by :class:`HyperNEATNode`.
        activation (Callable): Fixed activation used by :class:`HyperNEATNode`.
        activate_time (int): Recurrent activation steps of the substrate executor.
        output_transform (Callable): Output activation of the *substrate* network (not the CPPN).

    Note:
        ``._genome_cls``, ``._required_connection_type`` and ``._genome_extra_kwargs()`` are the three
        subclass hooks used by
        :class:`~tensorneat.algorithm.hyperneat.hyperneat_feedforward.HyperNEATFeedForward`; the
        CPPN-to-substrate decode itself is shared verbatim by both variants.
    """

    #: Genome class used for the fixed substrate executor (``DefaultGenome`` in the feedforward variant).
    _genome_cls = RecurrentGenome

    #: ``substrate.connection_type`` this algorithm requires, or ``None`` for "any".
    _required_connection_type: Optional[str] = None

    def __init__(
        self,
        substrate: BaseSubstrate,
        neat: NEAT,
        weight_threshold: float = 0.3,
        max_weight: float = 5.0,
        aggregation: Callable = AGG.sum,
        activation: Callable = ACT.sigmoid,
        activate_time: int = 10,
        output_transform: Callable = ACT.sigmoid,
    ):
        """Build the HyperNEAT decoder around an existing CPPN algorithm and a substrate.

        Args:
            substrate: Host-only substrate geometry.  Consumed at trace time through its ``nodes``,
                ``conn_index`` and ``query_coors`` arrays; it is never traced and needs no ``setup``.
            neat: The CPPN algorithm.  Its population size, input arity and (single) output arity
                define the query grid ``(P, Q, C)`` with ``C = neat.num_inputs``.
            weight_threshold: Dead-zone half-width; CPPN outputs with ``|w| < weight_threshold``
                become exactly ``0`` (pruned connections) and the remaining ones are shifted for
                continuity.  Must satisfy ``0 <= weight_threshold < 1`` because the map divides by
                ``1 - weight_threshold``.
            max_weight: Saturating weight magnitude after the map; must be ``> 0``.
            aggregation: Fixed aggregation callable ``(z, valid_mask) -> z`` used by every substrate
                node; not evolved.
            activation: Fixed activation callable used by every non-output substrate node; not evolved.
            activate_time: Recurrent activation steps of the substrate executor; must be ``>= 1``.
            output_transform: Output activation applied by the substrate genome.

        Raises:
            TensorNEATError: If ``substrate.query_coors.shape[1] != neat.num_inputs`` (the legacy
                message meaning "Query coors of Substrate should be equal to NEAT input size"), if
                ``weight_threshold`` is outside ``[0, 1)``, if ``max_weight <= 0``, if
                ``substrate.num_inputs < 2`` (a substrate without a bias coordinate cannot be fed),
                if ``activate_time < 1``, if ``neat.num_outputs != 1`` (the weight column has no
                room for a multi-output CPPN), if the substrate tables disagree on their counts, if
                the substrate node keys are not the block order ``inputs, outputs, hidden``, or if
                ``_required_connection_type`` is set and the substrate does not match it.

        Note:
            Validation is pure host Python over static configuration; nothing here builds a graph, so
            ``__init__`` must never call ``evoxc.constant`` or any ``enp`` op.
        """
        raise NotImplementedError

    def setup(self, state: State = State()) -> State:
        """Chain the CPPN setup and the substrate executor setup.

        Args:
            state: Threaded immutable state, already carrying ``randkey`` from ``Pipeline.setup``.

        Returns:
            A new state carrying the CPPN's keys (population tensors, ``generation``, ``species``)
            and the hyper genome's keys.

        Note:
            The substrate is stateless by contract -- it holds no learned state and exposes no
            ``setup`` -- so only ``neat.setup`` and ``hyper_genome.setup`` are chained here, and the
            substrate's host arrays are read at trace time (see :meth:`_query_coors`).
        """
        raise NotImplementedError

    def ask(self, state: State) -> Any:
        """Hand out the CPPN population that the problem should evaluate.

        Args:
            state: Current algorithm state.

        Returns:
            The CPPN population pytree ``(pop_nodes (P, max_nodes, node_len),
            pop_conns (P, max_conns, conn_len))``, returned by ``self.neat.ask`` unchanged.

        Note:
            The substrate is never part of the evolved population, so this costs zero ops; the
            substrate phenotype only appears inside :meth:`transform`.
        """
        raise NotImplementedError

    def transform(self, state: State, pop: Any) -> Any:
        """Decode the whole CPPN population into one substrate execution schedule.

        The decode algebra, with ``P`` CPPN individuals and ``Q`` substrate connections:

        1. ``transformed_cppn = self.neat.transform(state, pop)`` -- batch-first CPPN schedule, one
           individual per axis-0 slot of every leaf.
        2. ``query_res (P, Q) = self._query_substrate(state, pop)`` -- the CPPN is evaluated at every
           substrate connection's endpoint coordinates, flattened into the population axis.
        3. ``weights (P, Q) = self._weights_from_query(query_res)`` -- dead-zone, continuity shift and
           scaling into ``[-max_weight, max_weight]``.
        4. ``h_nodes (P, N, node_len), h_conns (P, Q, conn_len) =
           self._assemble_substrate(weights)`` -- the fixed topology painted with the new weights.
        5. ``return self.hyper_genome.transform(state, h_nodes, h_conns)`` -- the substrate genome
           derives its own schedule; its layout is genome-owned and opaque to this module.

        Args:
            state: Current algorithm state.
            pop: CPPN population pytree from :meth:`ask`, individual on axis 0.

        Returns:
            The substrate genome's batch-first ``Transformed`` pytree (individual on axis 0).  Its
            structure is defined by the genome, not by HyperNEAT: this module only appends the bias
            column in :meth:`forward` and never inspects schedule fields.

        Note:
            ``neat.pop_batch_size`` does not chunk this pass: the CPPN query is a single
            whole-population forward over the static ``(P, Q, C)`` grid, so peak memory scales with
            ``pop_size * conns_cnt`` (a few tens of thousands of floats in practice).  Shape changes
            (``pop_size``, ``conns_cnt``, ``nodes_cnt``) are static inputs to the traced graph, so the
            pipeline rebuilds the executable when they change.
        """
        raise NotImplementedError

    def forward(self, state: State, transformed: Any, inputs: Any) -> Any:
        """Run the substrate network, appending the bias input.

        Args:
            state: Current algorithm state.
            transformed: Substrate schedule returned by :meth:`transform`.
            inputs: ``(P, I - 1)`` problem inputs -- the bias neuron is *not* included.

        Returns:
            ``(P, O)`` substrate outputs.

        Note:
            The substrate's **last** input coordinate is the bias, so the column of ones is appended
            at the end: ``(P, I - 1) -> (P, I)``.  :attr:`num_inputs` is therefore
            ``substrate.num_inputs - 1``, which is what ``Pipeline`` asserts against the problem's
            ``input_shape[-1]``.  The ones block is built from the static ``pop_size``, so a chunked
            call over a sub-batch is a contract violation (chunking belongs to the genome kernels).
        """
        raise NotImplementedError

    def tell(self, state: State, fitness: Any) -> State:
        """Feed one generation of substrate fitness back into the CPPN evolution.

        Args:
            state: Current algorithm state.
            fitness: ``(P,)`` fitness of the decoded substrate networks, maximised.

        Returns:
            The new state produced by ``self.neat.tell`` (next CPPN population, advanced ``randkey``,
            incremented ``generation``, updated species bookkeeping).

        Note:
            Selection, speciation and mutation all act on CPPNs, so this is a pure delegation: the
            substrate phenotype leaves no trace in the state.
        """
        raise NotImplementedError

    def show_details(self, state: State, fitness: Any) -> None:
        """Print the CPPN generation details (host-side).

        Args:
            state: Current algorithm state.
            fitness: ``(P,)`` fitness of the generation just evaluated.
        """
        raise NotImplementedError

    @property
    def num_inputs(self) -> int:
        """Number of problem inputs, i.e. ``substrate.num_inputs - 1`` (the trailing bias removed).

        Returns:
            int: Static input arity checked by ``Pipeline`` against ``problem.input_shape[-1]``.
        """
        raise NotImplementedError

    @property
    def num_outputs(self) -> int:
        """Number of substrate outputs, i.e. ``substrate.num_outputs``.

        Returns:
            int: Static output arity checked by ``Pipeline`` against ``problem.output_shape[-1]``.
        """
        raise NotImplementedError

    @property
    def pop_size(self) -> int:
        """CPPN population size, i.e. ``neat.pop_size``.

        Returns:
            int: Length of axis 0 of every population tensor.

        Note:
            This is a read-only property (the base protocol declares it as one), not an instance
            attribute: it stays in sync with ``neat`` by construction.
        """
        raise NotImplementedError

    def _validate(self) -> None:
        """Check the host-side configuration gathered by ``__init__``.

        Raises:
            TensorNEATError: On any inconsistent arity, threshold, weight bound, activation horizon,
                substrate table shape, node-key block order, or required ``connection_type``.

        Note:
            Called once from ``__init__``; it reads only static Python/Numpy values and therefore
            never appears inside a trace.
        """
        raise NotImplementedError

    def _make_hyper_genome(self) -> Any:
        """Build the fixed-topology substrate executor genome.

        Returns:
            ``self._genome_cls(num_inputs=substrate.num_inputs, num_outputs=substrate.num_outputs,
            max_nodes=substrate.nodes_cnt, max_conns=substrate.conns_cnt,
            node_gene=HyperNEATNode(aggregation, activation), conn_gene=HyperNEATConn(),
            output_transform=output_transform, **self._genome_extra_kwargs())``.

        Note:
            ``max_nodes`` / ``max_conns`` equal the substrate's counts, so the substrate arrays are
            dense and pruning is expressed through zero weights rather than absent genes.
        """
        raise NotImplementedError

    def _genome_extra_kwargs(self) -> dict:
        """Return the variant-specific constructor kwargs of the substrate executor.

        Returns:
            ``{"activate_time": self.activate_time}`` for the recurrent variant; ``{}`` for the
            feedforward variant, whose genome has no recurrent horizon.

        Note:
            Subclass hook: it is the only place where the two HyperNEAT variants differ in genome
            construction, so the decode path stays single-sourced.
        """
        raise NotImplementedError

    def _graph_constant(self, array: "np.ndarray") -> Any:
        """Embed a host table into the traced graph as an ``evoxc`` constant.

        Args:
            array: Host-side table (already dtype- and layout-coerced by the caller).

        Returns:
            A graph tensor with the shape and dtype of ``array``.

        Note:
            The idiom is ``evoxc.constant(evoxc.from_numpy(np.ascontiguousarray(array)))``: inside a
            trace ``evoxc.from_numpy`` produces the host tensor and ``evoxc.constant`` snapshots it
            into the graph.  Constants are inlined into the compiled artifact and evoxc emits a
            ``UserWarning`` above ``EVOXC_LARGE_CONSTANT_BYTES`` (1 MiB), which is why only the small
            substrate tables go this way; a larger table would have to become an explicit executable
            input (a pipeline-level change, escalated to the parent node).
        """
        raise NotImplementedError

    def _query_coors(self) -> Any:
        """Materialise the substrate query coordinates as a batch-first query grid.

        Returns:
            ``(P, Q, C)`` float32 graph tensor: row ``q`` of ``substrate.query_coors`` -- the
            concatenated ``(source_coor, target_coor)`` pairs with ``2 * CD == C`` -- repeated once per
            CPPN individual.

        Note:
            Built with ``enp.broadcast_to`` over the static ``(Q, C)`` constant, so no Python loop
            over ``P`` and no materialisation until a consumer needs it; ``enp.broadcast_to`` is
            preferred over ``evoxc.tile`` because ``tile`` carries no batching rule in evoxc.  The
            coordinate row order is substrate-owned and must match ``substrate.conn_index`` 1:1.
        """
        raise NotImplementedError

    def _expand_schedule(self, transformed: Any, n_query: int) -> Any:
        """Repeat a per-individual schedule along a new query axis and fuse it into axis 0.

        Args:
            transformed: Batch-first schedule from ``self.neat.transform`` (individual on axis 0).
            n_query: Static query count ``Q``.

        Returns:
            The same pytree structure whose leaves carrying the population axis become
            ``(P * Q, *leaf.shape[1:])``; leaves without that axis are returned unchanged.

        Note:
            The flat row order is ``p * Q + q`` (individual slowest), matching the ``(P, Q, C) ->
            (P * Q, C)`` reshape of the query grid, so the ``(P * Q,) -> (P, Q)`` reshape in
            :meth:`_query_substrate` is exact.  Leaves holding no per-individual data (a shared
            ordering or key table) are left alone and broadcast inside the genome kernels.  This is
            the evoxc-native replacement for the legacy
            ``vmap(self.neat.forward, in_axes=(None, None, 0))``; the walk is a static Python
            recursion over the pytree, legal at trace time because the structure is static.
        """
        raise NotImplementedError

    def _query_substrate(self, state: State, pop: Any) -> Any:
        """Evaluate the CPPN population at every substrate connection's endpoint coordinates.

        Args:
            state: Current algorithm state.
            pop: CPPN population pytree, individual on axis 0.

        Returns:
            ``(P, Q)`` float32 graph tensor of raw CPPN outputs.

        Note:
            ``self.neat.transform(state, pop)`` gives the CPPN schedule, :meth:`_expand_schedule`
            repeats it ``Q`` times, the query grid is flattened to ``(P * Q, C)``, and
            ``self.neat.forward`` returns ``(P * Q, O_cppn)`` with ``O_cppn == 1`` -- reshaped to
            ``(P, Q)``.  A multi-output CPPN is rejected in ``__init__`` instead of silently
            misaligning the weight column.
        """
        raise NotImplementedError

    def _weights_from_query(self, query_res: Any) -> Any:
        """Map raw CPPN outputs to substrate connection weights.

        Args:
            query_res: ``(P, Q)`` float32 raw CPPN outputs.

        Returns:
            ``(P, Q)`` float32 weights in ``[-max_weight, max_weight]``.

        Note:
            The map is ``sign(w) * max(|w| - weight_threshold, 0) * max_weight / (1 -
            weight_threshold)``, which is exactly the legacy three-``where`` chain (dead-zone to
            zero, then the ``+/- weight_threshold`` continuity shift, then the scale) collapsed into
            two elementwise ops.  ``sign(0) == 0`` and ``max(|w| - t, 0) == 0`` for ``|w| <= t``, so a
            connection whose CPPN output ties the threshold lands on ``0`` -- a pruned edge, matching
            the legacy behaviour exactly.  NaN is *not* masked: a broken CPPN keeps NaN weights, the
            substrate output becomes NaN and the fitness is sanitised to ``-inf`` by ``NEAT.tell``,
            which is the parity oracle's contract.
        """
        raise NotImplementedError

    def _assemble_substrate(self, weights: Any) -> Any:
        """Build the dense substrate gene payload from the fixed topology plus new weights.

        Args:
            weights: ``(P, Q)`` float32 connection weights from :meth:`_weights_from_query`.

        Returns:
            ``(h_nodes, h_conns)`` with ``h_nodes (P, N, node_len)`` and
            ``h_conns (P, Q, conn_len)``, ready for ``self.hyper_genome.transform``.

        Note:
            Rows are built by the *gene-level* payload builders -- ``node_gene.from_attrs(keys)``
            over the static ``(N,)`` node keys and ``conn_gene.from_attrs(in_keys, out_keys,
            weights)`` over the static ``(Q,)`` endpoint keys plus the per-individual ``(P, Q)``
            weights -- so column order and value encoding stay genome-owned: neither this module nor
            the substrate layer ever needs to know where the weight column sits or how a float
            attribute is encoded into the row (evoxc has no bitcast op, so the encoding is the
            gene's business).  ``h_nodes`` is identical for every individual and only needs a
            ``broadcast_to`` along axis 0; the substrate arrays are dense, so no NaN padding is
            produced here.
        """
        raise NotImplementedError


class HyperNEATNode(BaseNode):
    """Substrate node gene with fixed (non-evolved) aggregation and activation.

    The substrate topology is fixed at construction and only its weights are decoded from the CPPN,
    so a substrate node carries no evolvable attributes: ``custom_attrs`` is empty and
    ``node_len == 1`` (the node identity key alone).  ``attrs`` is therefore accepted for protocol
    compatibility and never read.

    Attributes:
        aggregation (Callable): ``(z (..., max_sources), valid_mask (..., max_sources)) -> (...,)``.
        activation (Callable): Unary elementwise activation applied to non-output nodes.
    """

    #: Identity column of a node row; the substrate node carries nothing else.
    fixed_attrs = ["index"]

    #: No evolvable node attributes for a HyperNEAT substrate node.
    custom_attrs: list = []

    def __init__(self, aggregation: Callable = AGG.sum, activation: Callable = ACT.sigmoid):
        """Store the fixed aggregation and activation.

        Args:
            aggregation: Masked aggregation over the incoming-source axis, taking ``(z, valid_mask)``.
            activation: Activation applied to every node that is not an output node.

        Note:
            Both callables are configuration, not state: they are shared by every individual and
            never appear in ``State``, and they are never mutated by the evolutionary operators (the
            hyper genome is not evolved).
        """
        raise NotImplementedError

    def forward(
        self,
        state: State,
        attrs: Any,
        ins: Any,
        valid_mask: Any,
        is_output_node: Any,
    ) -> Any:
        """Aggregate a node's weighted incoming values and activate non-output nodes.

        Batch-first over *arbitrary* leading axes, so the same kernel serves the ``(P, N, S)``
        population batch and any slice of it:

        Args:
            state: Current algorithm state (unused; kept for the gene protocol).
            attrs: ``(..., node_len)`` integer node rows; unused because the node has no evolvable
                attributes.
            ins: ``(..., max_sources)`` already weighted incoming values, one slot per incoming
                edge.  Slots above the real in-degree hold garbage, not a sentinel.
            valid_mask: ``(..., max_sources)`` bool, ``True`` for real incoming edges.
            is_output_node: Bool array broadcastable to the leading batch axes, ``True`` where the
                node is an output node.

        Returns:
            ``(...)`` float32 node value: the masked aggregation, activated only for non-output
            nodes.

        Note:
            This fixes the two legacy defects: it is signature-compatible with the gene protocol
            (positional ``valid_mask`` accepted by both genome call sites) and it aggregates through
            the two-argument contract, i.e. ``evoxc.select(valid_mask, ins, 0.0)`` before
            ``self.aggregation(z, valid_mask)`` -- never the legacy one-argument
            ``aggregation(inputs)`` call, which raised ``TypeError`` because every ``AGG`` member is
            ``(z, mask)``.  Output nodes are handled by one ``evoxc.select`` on the result
            (aggregation only, no activation), not by a per-node ``evoxc.cond``: the batch is
            heterogeneous and ``cond`` requires a scalar predicate.  Masking is explicit, so absent
            edges contribute ``0`` instead of relying on NaN propagation.
        """
        raise NotImplementedError


class HyperNEATConn(BaseConn):
    """Substrate connection gene whose only evolvable attribute is the decoded CPPN weight.

    ``custom_attrs = ["weight"]`` makes ``conn_len == 3`` with rows
    ``[input_index, output_index, weight]``; the two endpoint columns are fixed by the substrate and
    only the weight is repainted on every ``HyperNEAT.transform``.
    """

    #: Endpoint key columns, fixed by the substrate topology.
    fixed_attrs = ["input_index", "output_index"]

    #: The single decoded attribute: the connection weight.
    custom_attrs = ["weight"]

    def forward(self, state: State, attrs: Any, inputs: Any) -> Any:
        """Scale a source value by the connection weight.

        Batch-first over arbitrary leading axes; the incoming-source axis is the last one:

        Args:
            state: Current algorithm state (unused; kept for the gene protocol).
            attrs: ``(..., max_sources, conn_len)`` gathered connection rows for one destination
                node.
            inputs: ``(..., max_sources)`` source values already masked to ``0`` where the edge does
                not exist.

        Returns:
            ``(..., max_sources)`` weighted contributions, ``inputs * weight``.

        Note:
            The weight column is read as the first custom attribute (``len(self.fixed_attrs)`` = 2),
            never as a literal ``-1``, so inserting a fixed attribute cannot silently shift it.
            Unlike the legacy ``HyperNEATConn.forward`` this kernel does **not** NaN-guard its
            operands: absent edges are excluded upstream by ``valid_mask`` and their slots are
            already ``0``, so masking happens once, at the node gene, and this kernel stays a single
            multiply.
        """
        raise NotImplementedError
