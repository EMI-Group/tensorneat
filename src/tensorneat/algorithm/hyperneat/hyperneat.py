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
    in the pre-rewrite library.

Notation used throughout (all sizes are static ints known at construction)
    ``P`` = the decoded payload's leading dimension (``neat.pop_size`` in an evolution step, ``1``
    for a single-individual ``Pipeline.show`` decode); ``Q`` = ``substrate.conns_cnt``;
    ``N`` = ``substrate.nodes_cnt``;
    ``I`` = ``substrate.num_inputs`` (substrate input neurons *including* the trailing bias);
    ``O`` = ``substrate.num_outputs``;
    ``C`` = ``neat.num_inputs`` = ``substrate.query_coors.shape[1]`` = ``2 * CD`` (CPPN input width,
    i.e. both endpoint coordinates of dimension ``CD``; ``C`` and ``I`` are *different* numbers and
    must never be interchanged);
    ``NA`` = ``node_gene.attr_width``, ``CK`` = ``conn_gene.key_width``, ``CA`` =
    ``conn_gene.attr_width`` (the split gene-column widths of the substrate payload).

evoxc facts this module is designed against
    * ``evoxc.select`` is the mask primitive (``enp.where`` is the same op); branches are both
      evaluated, so masking never short-circuits.
    * There is no bitcast / reinterpret op anywhere in evoxc, so a gene row is split into ``int32``
      key columns and ``float32`` attribute columns; this module assembles both halves itself (see
      :meth:`HyperNEAT._assemble_substrate`) and there is no gene-level payload builder to go
      through.
    * ``getitem`` is *static* indexing only and rejects ``Ellipsis``, so a last-axis column is taken
      with an explicit ``(slice(None),) * (rank - 1) + (index,)`` key (see
      ``tensorneat.genome.gene.base.column``, wrapped by the genes' ``column_of``).
"""

import numbers
from typing import Any, Callable, NamedTuple, Optional

import numpy as np

import evoxc
import evoxc.numpy as enp

from tensorneat.common import ACT, AGG, State
from tensorneat.common.errors import TensorNEATError
from tensorneat.common.host_graph import node_depth
from tensorneat.genome import RecurrentGenome
from tensorneat.genome.utils import GenomeTensors

from ..base import BaseAlgorithm
from ..neat import NEAT
from .genes import HyperNEATConn, HyperNEATNode
from .substrate import BaseSubstrate


def _require_real(name: str, value: Any) -> None:
    """Raise ``TensorNEATError`` unless ``value`` is a real number (``bool`` is rejected)."""
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TensorNEATError(
            f"{name} must be a real number, got {value!r} ({type(value).__name__})"
        )


class SubstrateSchedule(NamedTuple):
    """One generation's substrate phenotype: the execution schedule plus the payload it executes.

    The base algorithm protocol carries a *single* ``transformed`` object from
    :meth:`HyperNEAT.transform` through ``problem.evaluate`` into
    :meth:`HyperNEAT.forward`, while the genome's ``forward`` also needs the gene payload the
    schedule was derived from.  This container is that pair, and nothing else.

    Attributes:
        schedule: The substrate genome's ``Transformed`` schedule (topological order, per-node
            incoming-edge window, level and masks), individual on axis 0.
        node_keys: ``(P, N)`` int32 substrate node keys.
        node_attrs: ``(P, N, NA)`` float32 substrate node attributes.
        conn_keys: ``(P, Q, CK)`` int32 substrate connection keys.
        conn_attrs: ``(P, Q, CA)`` float32 substrate connection attributes (the decoded weights).

    Note:
        Every leaf keeps the population on axis 0, so ``problem.static_batch_size`` still recovers
        ``P`` by walking the leaves.
    """

    schedule: Any
    node_keys: Any
    node_attrs: Any
    conn_keys: Any
    conn_attrs: Any


    @property
    def level(self) -> Any:
        """Compatibility alias of the schedule's ``level`` (``node_level``)."""
        return self.schedule.level

    @property
    def overflow(self) -> Any:
        """Compatibility alias of the schedule's ``overflow`` (``edge_overflow > 0``).

        The pipeline reads this to decide whether the executable must be rebuilt with a wider
        in-degree window, so it has to be reachable from the object the pipeline holds.
        """
        return self.schedule.overflow


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
        self.substrate = substrate
        self.neat = neat
        self.weight_threshold = weight_threshold
        self.max_weight = max_weight
        self.aggregation = aggregation
        self.activation = activation
        self.activate_time = activate_time
        self.output_transform = output_transform
        self._validate()
        self.hyper_genome = self._make_hyper_genome()

    def setup(self, state: State | None = None) -> State:
        """Chain the CPPN setup and the substrate executor setup.

        Args:
            state: Threaded immutable state, already carrying ``randkey`` from ``Pipeline.setup``;
                ``None`` means "start from an empty :class:`~tensorneat.common.State`", which is the
                only spelling that keeps a ``State()`` default out of the signature (a default
                argument would be evaluated while this module is imported).

        Returns:
            A new state carrying the CPPN's keys (population tensors, ``generation``, ``species``)
            and the hyper genome's keys.

        Note:
            The substrate is stateless by contract (no learned state, no ``setup``), so only
            ``neat`` and ``hyper_genome`` are chained here; its host arrays are read at trace time.
        """
        if state is None:
            state = State()
        state = self.neat.setup(state)
        return self.hyper_genome.setup(state)

    def ask(self, state: State) -> Any:
        """Hand out the CPPN population that the problem should evaluate.

        Args:
            state: Current algorithm state.

        Returns:
            The CPPN population payload returned by ``self.neat.ask`` unchanged: the
            :class:`~tensorneat.genome.utils.GenomeTensors` of the CPPN genome, i.e.
            ``node_keys (P, max_nodes) int32``, ``node_attrs (P, max_nodes, NA) float32``,
            ``conn_keys (P, max_conns, CK) int32``, ``conn_attrs (P, max_conns, CA) float32``.

        Note:
            The substrate is never part of the evolved population, so this costs zero ops; the
            substrate phenotype only appears inside :meth:`transform`.
        """
        return self.neat.ask(state)

    def transform(self, state: State, pop: Any) -> Any:
        """Decode the whole CPPN population into one substrate execution schedule.

        The decode algebra, with ``P`` CPPN individuals and ``Q`` substrate connections:

        1. ``transformed_cppn = self.neat.transform(state, pop)`` -- batch-first CPPN schedule, one
           individual per axis-0 slot of every leaf.
        2. ``query_res (P, Q) = self._query_substrate(state, pop)`` -- the CPPN is evaluated at every
           substrate connection's endpoint coordinates, flattened into the population axis.
        3. ``weights (P, Q) = self._weights_from_query(query_res)`` -- dead-zone, continuity shift and
           scaling into ``[-max_weight, max_weight]``.
        4. ``payload = self._assemble_substrate(weights)`` -- the fixed topology painted with the new
           weights, as the substrate genome's four-array gene payload
           (:class:`~tensorneat.genome.utils.GenomeTensors`).
        5. ``schedule = self.hyper_genome.transform(state, payload.node_keys, payload.conn_keys)`` --
           the substrate genome derives its own schedule; its layout is genome-owned and opaque to
           this module.
        6. ``return SubstrateSchedule(schedule, *payload)`` -- the two travel together because the
           protocol forwards a single ``transformed`` object into :meth:`forward`.

        Args:
            state: Current algorithm state.
            pop: CPPN population payload from :meth:`ask`, individual on axis 0.

        Returns:
            A :class:`SubstrateSchedule`: the substrate genome's batch-first ``Transformed`` schedule
            (individual on axis 0) plus the dense gene payload it executes.  The schedule's structure
            is defined by the genome, not by HyperNEAT: this module only appends the bias column in
            :meth:`forward` and never inspects schedule fields.

        Note:
            ``neat.pop_batch_size`` does not chunk this pass: the CPPN query is a single
            whole-population forward over the static ``(P, Q, C)`` grid, so peak memory scales with
            ``pop_size * conns_cnt``; shape changes (``pop_size``, ``conns_cnt``, ``nodes_cnt``) are
            static inputs to the traced graph, so the pipeline rebuilds the executable.
        """
        query_res = self._query_substrate(state, pop)
        weights = self._weights_from_query(query_res)
        payload = self._assemble_substrate(weights)
        schedule = self.hyper_genome.transform(state, payload.node_keys, payload.conn_keys)
        return SubstrateSchedule(
            schedule=schedule,
            node_keys=payload.node_keys,
            node_attrs=payload.node_attrs,
            conn_keys=payload.conn_keys,
            conn_attrs=payload.conn_attrs,
        )

    def forward(self, state: State, transformed: Any, inputs: Any) -> Any:
        """Run the substrate network, appending the bias input.

        Args:
            state: Current algorithm state.
            transformed: :class:`SubstrateSchedule` returned by :meth:`transform`.
            inputs: ``(B, I - 1)`` problem inputs -- the bias neuron is *not* included.  ``B`` is a
                static Python int: a problem tiles its dataset over the population axis, so ``B`` is
                normally ``pop_size * samples``.

        Returns:
            ``(B, O)`` substrate outputs.

        Note:
            The substrate's **last** input coordinate is the bias, so the column of ones is appended
            at the end: ``(B, I - 1) -> (B, I)``; :attr:`num_inputs` is therefore
            ``substrate.num_inputs - 1``.  The ones column is sized from ``inputs.shape[0]``, never
            from the static ``pop_size``, because the genome expands the payload's population axis to
            ``B`` internally -- rebuilding the column from ``pop_size`` would break the tiled case.
            The concatenate axis is the explicit non-negative last axis of the static rank ``2``
            (``axis=1``), never ``-1``: the StableHLO exporter forwards the axis verbatim as
            ``stablehlo.concatenate``'s ``dimension``, which must be non-negative.
        """
        batch = int(inputs.shape[0])
        ones = enp.ones((batch, 1), dtype=evoxc.float32)
        biased = enp.concatenate([inputs, ones], axis=1)
        return self.hyper_genome.forward(
            state,
            transformed.schedule,
            transformed.node_keys,
            transformed.node_attrs,
            transformed.conn_keys,
            transformed.conn_attrs,
            biased,
        )

    def prepare_forward(
        self, state: State, transformed: Any, pop: Any = None, samples: int = 1
    ) -> Any:
        """Derive the substrate pass's loop-invariant operands, for many queries.

        The substrate schedule is at ``P`` rows and a tiled batch is the substrate genome's
        own ``samples`` axis, so ``samples`` is forwarded to the genome unchanged.

        Args:
            state: Current algorithm state.
            transformed: :class:`SubstrateSchedule` returned by :meth:`transform`.
            pop: Ignored -- the substrate payload lives inside ``transformed``; the argument
                keeps the algorithm signature uniform.
            samples: Static tiling factor of the ``pop_size * samples`` query rows.

        Returns:
            The substrate genome's opaque prepared record, for :meth:`forward_prepared`.
        """
        return self.hyper_genome.prepare_pass(
            state,
            transformed.schedule,
            transformed.node_keys,
            transformed.node_attrs,
            transformed.conn_keys,
            transformed.conn_attrs,
            samples,
        )

    def forward_prepared(
        self, state: State, transformed: Any, prepared: Any, inputs: Any
    ) -> Any:
        """Run the substrate network from a prepared record, appending the bias column.

        The per-query half of :meth:`forward`: same bias concatenation and the same genome
        call, with every loop-invariant operand taken from ``prepared``.
        """
        batch = int(inputs.shape[0])
        ones = enp.ones((batch, 1), dtype=evoxc.float32)
        biased = enp.concatenate([inputs, ones], axis=1)
        return self.hyper_genome.forward_prepared(
            state, transformed.schedule, prepared, biased
        )

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
        return self.neat.tell(state, fitness)

    def show_details(self, state: State, fitness: Any) -> None:
        """Print the CPPN generation details (host-side).

        Args:
            state: Current algorithm state.
            fitness: ``(P,)`` fitness of the generation just evaluated.
        """
        return self.neat.show_details(state, fitness)

    @property
    def num_inputs(self) -> int:
        """Number of problem inputs, i.e. ``substrate.num_inputs - 1`` (the trailing bias removed).

        Returns:
            int: Static input arity checked by ``Pipeline`` against ``problem.input_shape[-1]``.
        """
        return int(self.substrate.num_inputs) - 1

    @property
    def num_outputs(self) -> int:
        """Number of substrate outputs, i.e. ``substrate.num_outputs``.

        Returns:
            int: Static output arity checked by ``Pipeline`` against ``problem.output_shape[-1]``.
        """
        return int(self.substrate.num_outputs)

    @property
    def pop_size(self) -> int:
        """CPPN population size, i.e. ``neat.pop_size`` (read-only property, not an attribute).

        Returns:
            int: Length of axis 0 of every population tensor.
        """
        return self.neat.pop_size

    def _validate(self) -> None:
        """Check the host-side configuration gathered by ``__init__``.

        Raises:
            TensorNEATError: On any inconsistent arity, threshold, weight bound, activation horizon,
                substrate table shape, node-key block order, required ``connection_type``, or a
                non-real ``weight_threshold`` / ``max_weight`` / ``activate_time``.
        """
        substrate = self.substrate
        neat = self.neat
        num_substrate_inputs = int(substrate.num_inputs)
        num_substrate_outputs = int(substrate.num_outputs)
        nodes_cnt = int(substrate.nodes_cnt)
        conns_cnt = int(substrate.conns_cnt)
        query_coors = np.asarray(substrate.query_coors)
        conn_index = np.asarray(substrate.conn_index)
        nodes = np.asarray(substrate.nodes)

        if query_coors.ndim != 2 or query_coors.shape[1] != neat.num_inputs:
            width = query_coors.shape[1] if query_coors.ndim == 2 else None
            raise TensorNEATError(
                "Query coors of Substrate should be equal to NEAT input size: "
                f"query_coors has width {width}, neat.num_inputs is {neat.num_inputs}"
            )
        _require_real("weight_threshold", self.weight_threshold)
        if not 0 <= self.weight_threshold < 1:
            raise TensorNEATError(
                "weight_threshold must be in [0, 1) because the weight map divides by "
                f"1 - weight_threshold, got {self.weight_threshold}"
            )
        _require_real("max_weight", self.max_weight)
        if self.max_weight <= 0:
            raise TensorNEATError(f"max_weight must be > 0, got {self.max_weight}")
        if num_substrate_inputs < 2:
            raise TensorNEATError(
                "substrate.num_inputs must be >= 2 (the last substrate input is the bias "
                f"neuron fed with a constant 1), got {num_substrate_inputs}"
            )
        _require_real("activate_time", getattr(self, "activate_time", 1))
        if getattr(self, "activate_time", 1) < 1:
            raise TensorNEATError(
                f"activate_time must be >= 1, got {self.activate_time}"
            )
        if neat.num_outputs != 1:
            raise TensorNEATError(
                "a HyperNEAT CPPN must have exactly one output (it is the connection weight), "
                f"got neat.num_outputs = {neat.num_outputs}"
            )
        if query_coors.shape[0] != conns_cnt:
            raise TensorNEATError(
                f"query_coors has {query_coors.shape[0]} rows but substrate.conns_cnt is "
                f"{conns_cnt}; query row q must describe connection q"
            )
        if conn_index.shape != (conns_cnt, 2):
            raise TensorNEATError(
                f"substrate.conn_index must be (conns_cnt, 2) = ({conns_cnt}, 2), "
                f"got {conn_index.shape}"
            )
        if nodes.shape != (nodes_cnt,):
            raise TensorNEATError(
                f"substrate.nodes must be (nodes_cnt,) = ({nodes_cnt},), got {nodes.shape}"
            )
        if sorted(int(key) for key in nodes) != list(range(nodes_cnt)):
            raise TensorNEATError(
                "substrate.nodes must be a permutation of 0 .. nodes_cnt-1, got "
                f"{nodes.tolist()}"
            )
        prefix = nodes[: num_substrate_inputs + num_substrate_outputs]
        expected = np.arange(num_substrate_inputs + num_substrate_outputs)
        if prefix.shape != expected.shape or not np.array_equal(prefix, expected):
            raise TensorNEATError(
                "substrate node keys must be the block order inputs 0..I-1, then outputs "
                f"I..I+O-1 (I={num_substrate_inputs}, O={num_substrate_outputs}, "
                f"nodes_cnt={nodes_cnt}), got {nodes.tolist()}"
            )
        required = self._required_connection_type
        if required is not None and substrate.connection_type != required:
            raise TensorNEATError(
                f"{type(self).__name__} requires a substrate with connection_type="
                f"{required!r}, got {substrate.connection_type!r}"
            )

    def _static_bounds(self) -> dict:
        """Derive the executor genome's static in-degree window and depth unroll from the substrate.

        The substrate is a *dense* topology (``max_nodes == nodes_cnt``, ``max_conns == conns_cnt``),
        so its schedule fits only if the genome's static window ``K`` is at least the largest
        per-destination in-degree the substrate actually has, and -- for the feedforward executor --
        if the depth unroll ``D`` is at least its longest input-to-node path.  Neither bound is
        covered by ``Transformed.overflow``: a too-small ``K`` drops connections (the pipeline raises
        on the overflow), while a too-small ``D`` would silently leave the tail of a deep substrate
        at ``NaN``.

        Returns:
            ``{"max_in_degree": K}`` plus ``{"max_depth": D}`` for the feedforward executor only.

        Note:
            ``K`` is the exact maximum in-degree, ``bincount(conn_index[:, 1]).max()``, floored at
            ``1`` and at ``substrate.num_inputs``: the genome constructor validates ``max_in_degree``
            against its (never used) dense initial ``num_inputs -> num_outputs`` network and would
            otherwise reject the bound.  The bound is *per destination node*, so for a full or MLP
            feedforward substrate it is the number of lower-layer neurons wired into one neuron -- a
            ``FullSubstrate`` with 4 inputs and 3 fully interconnected hidden neurons therefore needs
            ``K = 7`` (4 inputs + 3 hidden, its own self-loop included), not 4.  ``D`` is the longest
            path length from the substrate inputs
            (:func:`~tensorneat.common.host_graph.node_depth`, floored at ``1``); it is only derived
            when ``_required_connection_type == "feedforward"`` because ``RecurrentGenome`` ignores
            ``max_depth`` (it ticks a fixed ``activate_time`` instead).
            A cyclic or input-unreachable substrate makes ``node_depth`` raise
            :class:`~tensorneat.common.errors.GraphError` (a ``TensorNEATError``), which propagates
            unchanged -- a broken geometry must not be papered over.
        """
        conn_index = np.asarray(self.substrate.conn_index)
        nodes_cnt = int(self.substrate.nodes_cnt)
        in_degree = np.bincount(conn_index[:, 1], minlength=nodes_cnt)
        bounds = {
            "max_in_degree": max(int(in_degree.max()), int(self.substrate.num_inputs), 1)
        }
        if self._required_connection_type == "feedforward":
            depth = node_depth(
                list(int(key) for key in self.substrate.nodes),
                [(int(src), int(dst)) for src, dst in conn_index],
                range(int(self.substrate.num_inputs)),
            )
            bounds["max_depth"] = max(max(int(value) for value in depth.values()), 1)
        return bounds

    def _make_hyper_genome(self) -> Any:
        """Build the fixed-topology substrate executor genome.

        Returns:
            ``self._genome_cls(num_inputs=substrate.num_inputs, num_outputs=substrate.num_outputs,
            max_nodes=substrate.nodes_cnt, max_conns=substrate.conns_cnt,
            node_gene=HyperNEATNode(aggregation, activation), conn_gene=HyperNEATConn(),
            output_transform=output_transform, **_static_bounds(),
            **self._genome_extra_kwargs())``, whose ``max_nodes`` / ``max_conns`` equal the
            substrate's counts (dense arrays, pruning through zero weights rather than absent genes)
            and whose static bounds come from :meth:`_static_bounds`: ``max_in_degree`` is the
            substrate's exact maximum in-degree (floored at ``num_inputs``), and ``max_depth`` --
            passed for the feedforward executor only -- is its longest path from the inputs.
        """
        return self._genome_cls(
            num_inputs=self.substrate.num_inputs,
            num_outputs=self.substrate.num_outputs,
            max_nodes=self.substrate.nodes_cnt,
            max_conns=self.substrate.conns_cnt,
            node_gene=HyperNEATNode(self.aggregation, self.activation),
            conn_gene=HyperNEATConn(),
            output_transform=self.output_transform,
            **self._static_bounds(),
            **self._genome_extra_kwargs(),
        )

    def _genome_extra_kwargs(self) -> dict:
        """Return the variant-specific constructor kwargs of the substrate executor.

        Returns:
            ``{"activate_time": self.activate_time}`` for the recurrent variant; ``{}`` for the
            feedforward variant, whose genome has no recurrent horizon.  This is the subclass hook
            where the two variants differ in genome construction, so the decode stays single-sourced.
        """
        return {"activate_time": self.activate_time}

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
        return evoxc.constant(evoxc.from_numpy(np.ascontiguousarray(array)))

    def _query_coors(self, pop_size: int) -> Any:
        """Materialise the substrate query coordinates as a batch-first query grid.

        Args:
            pop_size: Static batch size ``P`` of the *payload being decoded* -- normally
                ``self.pop_size``, but any static batch (``Pipeline.show`` decodes a single
                individual) is legal because it is read off the payload, not off the config.

        Returns:
            ``(P, Q, C)`` float32 graph tensor: row ``q`` of ``substrate.query_coors`` -- the
            concatenated ``(source_coor, target_coor)`` pairs with ``2 * CD == C`` -- repeated once per
            CPPN individual.

        Note:
            Built with ``enp.broadcast_to`` over the static ``(Q, C)`` constant, so no Python loop
            over ``P`` and no materialisation until a consumer needs it; ``enp.broadcast_to`` is
            preferred over ``evoxc.tile`` because ``tile`` carries no batching rule in evoxc.
            ``broadcast_to`` can only expand unit axes, which is why the constant is first reshaped
            to ``(1, Q, C)``; the coordinate row order is substrate-owned and must match
            ``substrate.conn_index`` 1:1.
        """
        conns_cnt = int(self.substrate.conns_cnt)
        width = int(np.asarray(self.substrate.query_coors).shape[1])
        table = self._graph_constant(self.substrate.query_coors)
        return enp.broadcast_to(
            enp.reshape(table, (1, conns_cnt, width)), (pop_size, conns_cnt, width)
        )

    def _expand_schedule(self, transformed: Any, n_query: int, pop_size: int) -> Any:
        """Repeat a per-individual schedule along a new query axis and fuse it into axis 0.

        Args:
            transformed: Batch-first schedule from ``self.neat.transform`` (individual on axis 0).
            n_query: Static query count ``Q``.
            pop_size: Static batch size ``P`` of that schedule -- the payload's own leading
                dimension, so a batch smaller than ``self.pop_size`` decodes correctly.

        Returns:
            The same pytree structure whose leaves carrying the population axis become
            ``(P * Q, *leaf.shape[1:])``; leaves without that axis are returned unchanged.

        Note:
            The flat row order is ``p * Q + q`` (individual slowest), matching the ``(P, Q, C) ->
            (P * Q, C)`` reshape of the query grid, so the ``(P * Q,) -> (P, Q)`` reshape in
            :meth:`_query_substrate` is exact.  Leaves holding no per-individual data (a shared
            ordering or key table) are left alone and broadcast inside the genome kernels; the walk is
            a static Python recursion over the pytree, legal at trace time because it is static.
        """
        return evoxc.tree_map(lambda leaf: self._expand_leaf(leaf, n_query, pop_size), transformed)

    def _expand_leaf(self, leaf: Any, n_query: int, pop_size: int) -> Any:
        """Repeat one batch-first leaf along a query axis and fuse both axes into axis 0.

        Args:
            leaf: One leaf of a batch-first pytree; either a tensor whose axis 0 is the population
                or a tensor without that axis (a shared ordering or key table).
            n_query: Static query count ``Q``.
            pop_size: Static batch size ``P`` of the batch being decoded -- the population axis
                length to rewrite, taken from the payload rather than from ``self.pop_size``.

        Returns:
            ``(P * n_query, *leaf.shape[1:])`` when ``leaf`` carries the population axis, and
            ``leaf`` itself otherwise.

        Note:
            Same flat row order as :meth:`_expand_schedule` (``p * n_query + q``, individual
            slowest), which is what makes the ``(P, Q, C) -> (P * Q, C)`` reshape of the query grid
            line up.  ``enp.broadcast_to`` cannot expand a non-unit axis, so the leaf is reshaped to
            ``(P, 1, ...)`` first; the repetition stays a broadcast until a consumer materialises it.
        """
        if not leaf.shape or int(leaf.shape[0]) != pop_size:
            return leaf
        trailing = tuple(int(size) for size in leaf.shape[1:])
        widened = enp.reshape(leaf, (pop_size, 1) + trailing)
        repeated = enp.broadcast_to(widened, (pop_size, n_query) + trailing)
        return enp.reshape(repeated, (pop_size * n_query,) + trailing)

    def _query_substrate(self, state: State, pop: Any) -> Any:
        """Evaluate the CPPN population at every substrate connection's endpoint coordinates.

        Args:
            state: Current algorithm state.
            pop: CPPN population payload, individual on axis 0.  Its leading dimension is the batch
                ``P`` -- normally ``self.pop_size``, but any static batch is accepted (``Pipeline.show``
                decodes one individual), which is why ``P`` is read from the payload.

        Returns:
            ``(P, Q)`` float32 graph tensor of raw CPPN outputs.

        Note:
            ``self.neat.transform(state, pop)`` gives the CPPN schedule, :meth:`_expand_schedule`
            repeats it ``Q`` times, the query grid is flattened to ``(P * Q, C)`` and the CPPN
            genome's ``forward`` returns ``(P * Q, O_cppn)`` with ``O_cppn == 1`` -- reshaped to
            ``(P, Q)``.  A multi-output CPPN is rejected in ``__init__`` instead of silently
            misaligning the weight column.

            The call deliberately targets ``self.neat.genome.forward`` rather than
            ``self.neat.forward``: ``NEAT.forward`` reads the *un-expanded* population payload from
            ``state`` and therefore cannot serve a ``P * Q`` batch.  That is a cross-node dependency
            on ``NEAT`` exposing its genome as ``.genome``; ``*expanded_pop`` additionally relies on
            the payload being a ``GenomeTensors``, whose field order
            (``node_keys, node_attrs, conn_keys, conn_attrs``) is exactly ``genome.forward``'s
            parameter order after ``transformed``.
        """
        pop_size = int(pop.node_keys.shape[0])
        n_query = int(self.substrate.conns_cnt)
        width = int(np.asarray(self.substrate.query_coors).shape[1])
        transformed_cppn = self.neat.transform(state, pop)
        expanded = self._expand_schedule(transformed_cppn, n_query, pop_size)
        expanded_pop = evoxc.tree_map(lambda leaf: self._expand_leaf(leaf, n_query, pop_size), pop)
        queries = enp.reshape(self._query_coors(pop_size), (pop_size * n_query, width))
        out = self.neat.genome.forward(state, expanded, *expanded_pop, queries)
        return enp.reshape(out, (pop_size, n_query))

    def _weights_from_query(self, query_res: Any) -> Any:
        """Map raw CPPN outputs to substrate connection weights.

        Args:
            query_res: ``(P, Q)`` float32 raw CPPN outputs.

        Returns:
            ``(P, Q)`` float32 weights in ``[-max_weight, max_weight]``.

        Note:
            The map is ``sign(w) * max(|w| - weight_threshold, 0) * max_weight / (1 -
            weight_threshold)``, built from
            ``abs``/``greater``/``select``/``subtract``/``sign``/``multiply``.  The dead zone is
            strict -- ``|w| <= weight_threshold`` lands on ``0``, a pruned edge -- the legacy
            ``-t < w < t`` test including its ``|w| == t`` boundary.

            NaN is *not* masked: ``sign(NaN)`` is NaN, so the final multiply carries it through even
            though the dead-zone ``select`` picks ``0`` for a NaN input.  A broken CPPN therefore
            keeps NaN weights, the substrate output becomes NaN and the fitness is sanitised to
            ``-inf`` by ``NEAT.tell``, which is the parity oracle's contract.
        """
        threshold = enp.full((), float(self.weight_threshold), dtype=evoxc.float32)
        zero = enp.full((), 0.0, dtype=evoxc.float32)
        scale = enp.full(
            (),
            float(self.max_weight) / (1.0 - float(self.weight_threshold)),
            dtype=evoxc.float32,
        )
        magnitude = enp.abs(query_res)
        mapped = evoxc.select(
            enp.greater(magnitude, threshold), enp.subtract(magnitude, threshold), zero
        )
        return enp.multiply(enp.multiply(enp.sign(query_res), mapped), scale)

    def _assemble_substrate(self, weights: Any) -> Any:
        """Build the dense substrate gene payload from the fixed topology plus new weights.

        Args:
            weights: ``(P, Q)`` float32 connection weights from :meth:`_weights_from_query`.
                Its leading dimension is the batch ``P`` this payload is built for, so a decode of
                fewer than ``self.pop_size`` individuals produces a matching ``(P, ...)`` payload.

        Returns:
            :class:`~tensorneat.genome.utils.GenomeTensors` with ``node_keys (P, N) int32``,
            ``node_attrs (P, N, NA) float32``, ``conn_keys (P, Q, CK) int32`` and
            ``conn_attrs (P, Q, CA) float32``, ready for ``self.hyper_genome.transform`` and
            ``self.hyper_genome.forward``.

        Note:
            The columns are read from the gene *tables* -- ``node_gene.attr_specs``/``key_width`` and
            ``conn_gene.attr_specs``/``key_width`` -- never from a hardcoded index, so a gene family
            that reorders its table cannot silently shift a column:

            * ``node_keys`` broadcasts the static ``(N,)`` key vector along the population axis;
            * ``node_attrs`` gets one zero column per ``node_gene.attr_specs`` entry, because no node
              attribute is decoded from the CPPN (an ``"int"`` spec stores its option index as
              float32, and index ``0`` is the first configured option -- correct for a non-evolved
              substrate, whose :class:`HyperNEATNode` declares exactly two single-option columns,
              ``aggregation`` and ``activation``, so index ``0`` selects the constructor callables);
              with ``attr_width == 0`` the array is an empty ``(P, N, 0)`` and
              ``enp.stack`` is never called on an empty column list;
            * ``conn_keys`` carries the static ``(Q, 2)`` ``conn_index`` endpoint table in its first
              two columns and a deterministic unique per-row id ``arange(Q) + 1`` in every further
              key column (the ``OriginConn`` ``historical_marker``), the extra count being derived
              from ``conn_gene.key_width - 2`` rather than from a gene-family check;
            * ``conn_attrs`` gets the ``weights`` column where ``conn_gene.attr_specs`` names
              ``"weight"`` and a zero column for every other attribute.

            Keys and attributes stay separate arrays because evoxc has no float/int bitcast, and the
            substrate topology is dense (``max_nodes == nodes_cnt``, ``max_conns == conns_cnt``), so
            no row is padding and no NaN sentinel is produced here.
        """
        pop_size = int(weights.shape[0])
        nodes_cnt = int(self.substrate.nodes_cnt)
        conns_cnt = int(self.substrate.conns_cnt)
        node_gene = self.hyper_genome.node_gene
        conn_gene = self.hyper_genome.conn_gene

        node_keys = enp.broadcast_to(
            enp.reshape(self._graph_constant(self.substrate.nodes), (1, nodes_cnt)),
            (pop_size, nodes_cnt),
        )
        node_attrs = self._attr_columns(pop_size, nodes_cnt, node_gene.attr_specs, None)
        conn_keys = self._conn_keys(pop_size, conns_cnt, conn_gene.key_width)
        conn_attrs = self._attr_columns(pop_size, conns_cnt, conn_gene.attr_specs, weights)
        return GenomeTensors(
            node_keys=node_keys,
            node_attrs=node_attrs,
            conn_keys=conn_keys,
            conn_attrs=conn_attrs,
        )

    def _attr_columns(
        self,
        pop_size: int,
        rows: int,
        specs: tuple,
        weights: Any,
    ) -> Any:
        """Stack one ``(P, rows)`` column per attribute spec into a ``(P, rows, attr_width)`` table.

        Args:
            pop_size: Static population size ``P``.
            rows: Static row count of the gene family (``N`` for nodes, ``Q`` for connections).
            specs: ``attr_specs`` of the gene, in column order.
            weights: ``(P, Q)`` decoded weights to paint into the ``"weight"`` column, or ``None``
                for a family whose attributes are not decoded from the CPPN (all-zero columns).

        Returns:
            ``(P, rows, len(specs))`` float32 attribute table.

        Raises:
            TensorNEATError: If ``weights`` is given but no spec is named ``"weight"``, i.e. the
                decoded weights would be dropped on the floor.

        Note:
            A gene with no attribute columns yields an empty ``(P, rows, 0)`` array instead of
            ``enp.stack([])``, which evoxc rejects.
            The stack axis is the explicit non-negative new last axis of the static rank ``2``
            (``axis=2``), never ``-1``, because ``stack`` is ``expand_dims`` + ``concatenate``.
        """
        if weights is not None and not any(spec.name == "weight" for spec in specs):
            raise TensorNEATError(
                "the substrate connection gene declares no attribute column named 'weight', so "
                "the decoded CPPN weights would be dropped; spec names are "
                f"{[spec.name for spec in specs]}"
            )
        if not specs:
            return enp.zeros((pop_size, rows, 0), dtype=evoxc.float32)
        columns = []
        for spec in specs:
            if weights is not None and spec.name == "weight":
                columns.append(weights)
            else:
                columns.append(enp.zeros((pop_size, rows), dtype=evoxc.float32))
        return enp.stack(columns, axis=2)

    def _conn_keys(self, pop_size: int, conns_cnt: int, key_width: int) -> Any:
        """Build the ``(P, Q, key_width)`` int32 connection key table of the substrate.

        Args:
            pop_size: Static population size ``P``.
            conns_cnt: Static connection count ``Q``.
            key_width: ``conn_gene.key_width`` -- two endpoint columns plus any extra key column.

        Returns:
            ``(P, Q, key_width)`` int32 connection keys.

        Note:
            The first two columns are the static ``(Q, 2)`` ``substrate.conn_index`` endpoint table
            of the genome's ``(input_index, output_index)`` key columns, broadcast along the
            population axis.  Any further column (the ``OriginConn`` ``historical_marker``) holds a
            deterministic unique per-row id ``arange(Q) + 1``: the substrate topology is fixed, so
            the marker only has to be distinct and reproducible, never evolvable.
            The concatenate axis is the explicit non-negative last axis of the static rank ``3``
            (``axis=2``), never ``-1``: the StableHLO exporter forwards the axis verbatim as
            ``stablehlo.concatenate``'s ``dimension``, which must be non-negative.
        """
        endpoint = enp.reshape(self._graph_constant(self.substrate.conn_index), (1, conns_cnt, 2))
        keys = enp.broadcast_to(endpoint, (pop_size, conns_cnt, 2))
        extra_width = int(key_width) - 2
        if extra_width <= 0:
            return keys
        row_ids = enp.reshape(
            enp.arange(1, conns_cnt + 1, dtype=evoxc.int32), (1, conns_cnt, 1)
        )
        extra = enp.broadcast_to(row_ids, (pop_size, conns_cnt, extra_width))
        return enp.concatenate([keys, extra], axis=2)

