"""Genome base class: payload contract, schedule glue and forward/grad API.

This module owns host-side configuration (static sizes, gene/operator objects, transforms),
the payload/row-key invariant, and the *shared* window-reduce machinery both concrete genomes
compose — ``default.py`` and ``recurrent.py`` must not duplicate it.
Every compute method is batch-first (population on axis 0) with static shapes: no
``vmap``/``vectorize`` and no ``scan``/``while_loop``/``cond`` — depth and tick loops are plain
Python ``for`` loops unrolled at trace time, which is what keeps ``evoxc.grad`` usable.

## Cross-module contracts
Every name below is provided by the ``common`` node and is never re-derived here.
``../common.tensorization.build_schedule`` derives the ``Transformed`` schedule.
``../common.functions.apply_aggregation`` is the only window-reduce kernel.
``../common.functions.apply_activation`` is the only activation kernel.
``../common.functions`` also provides ``ACT``, ``AGG`` and ``get_func_name``.
``../common.tools.I_INF`` is the ``int32`` "no index" sentinel padding every key array.
``../common.State`` and ``../common.StatefulBaseClass`` are the immutable host containers.
``../common.errors.TensorNEATError`` is raised for every configuration violation.
Host-only helpers: ``../common.topological_sort_python`` and ``../common.find_useful_nodes``.

## Cost budget (measured, numpy backend)
A forward pass costs about one unrolled level per network depth: measured 1.8 ms per level at `P=1000`, `max_nodes=50`, `K=2` with ~20 ops per level, i.e. ~14 ms for a depth-8 forward against the 157 ms legacy generation at the same population.
Op count per unrolled level — not element count — is the budget: keep the window step thin, gather connection attributes once per forward pass, and never re-derive the schedule inside a level.
Depth, tick and option loops are unrolled at trace time, so compile time grows with `max_depth`/`activate_time`/option count; single-option configurations emit no selection op at all.
"""

from collections.abc import Callable, Sequence
from typing import Any

import evoxc
import evoxc.numpy as enp  # noqa: F401 - tensor creation for the implementation wave
import numpy as np  # noqa: F401 - host-only helpers
import sympy as sp  # noqa: F401 - host-only code generation

from ..common import (
    I_INF,
    State,
    StatefulBaseClass,
    TensorNEATError,
    Transformed,
    apply_activation,
    apply_aggregation,
    derive_schedule,
    find_useful_nodes,
    to_numpy,
    topological_sort,
)
from ..common.functions import ACT, AGG
from .gene import BaseConn, BaseNode
from .operations import BaseCrossover, BaseDistance, BaseMutation
from .utils import GenomeTensors, check_cycles as _check_cycles, re_cound_idx, valid_count


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

    ``max_in_degree`` is the static incoming-window width ``K`` (default
    ``max(initial in-degree, 4)``), ``max_depth`` the static unroll count ``D``; both are
    Python ints fixed at construction, so changing either forces a rebuild.
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
    ):
        """Configure one genome family and derive its dense initial network.

        The initial network is fully connected and layered,
        ``num_inputs -> init_hidden_layers -> num_outputs``, keys assigned in layer order
        (input keys ``0..num_inputs-1``, output keys the last ``num_outputs``) and written to
        rows ``0..count-1`` of the payload by :meth:`initialize` together with the dense edges.
        ``input_transform``/``output_transform`` are elementwise host callables over the last
        axis of a ``(P, n)`` batch, probed on a ``(1, n)`` batch here so a broken transform
        fails at construction instead of inside a trace.

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

        Raises:
            TensorNEATError: If ``max_nodes``/``max_conns`` cannot hold the initial network, if
                ``init_hidden_layers`` needs an in-degree above ``max_in_degree``, or if a
                transform probe on a ``(1, n)`` batch raises or returns a wrong shape.
            ValueError: If a size argument is not a positive int.
        """
        raise NotImplementedError(
            "build the dense layered initial network, resolve max_in_degree/max_depth, and "
            "probe input_transform/output_transform on a (1, n) batch"
        )

    def setup(self, state: State = State()) -> State:
        """Register the gene/operator configuration on an immutable state.

        Args:
            state: Base state to extend; never mutated in place.

        Returns:
            A new ``State`` after chaining ``node_gene``, ``conn_gene``, ``mutation``,
            ``crossover`` and ``distance`` setup, in that order.

        Raises:
            NotImplementedError: Skeleton stub.
        """
        raise NotImplementedError(
            "chain node_gene, conn_gene, mutation, crossover, distance setup(state)"
        )

    def initialize(
        self, state: State, key: evoxc.SymbolicTensor, pop_size: int
    ) -> GenomeTensors:
        """Draw the initial population of dense layered networks.

        One rank-0 ``evoxc.random`` key plus a static ``pop_size``; every tensor is created
        with its full batched shape, so individual ``i`` gets an independent stream from that
        single draw (keys are never split per individual).  The ``historical_marker`` key
        column, when the conn gene declares it, is the row index of the connection.

        Args:
            state: Host configuration container from :meth:`setup`.
            key: Rank-0 ``evoxc.random`` key, split internally per gene family.
            pop_size: Static population size ``P``.

        Returns:
            ``GenomeTensors``: ``node_keys (P, N) int32`` (``I_INF`` padded, input/output rows
            at their invariant positions), ``node_attrs (P, N, NA) float32`` (NaN padded),
            ``conn_keys (P, E, CK) int32`` (``I_INF`` padded), ``conn_attrs (P, E, CA)
            float32`` (NaN padded).

        Raises:
            NotImplementedError: Skeleton stub.
            TensorNEATError: (implementation wave) if ``pop_size`` is not a positive int.
        """
        raise NotImplementedError(
            "fill the invariant node rows in key order and the dense initial connections, "
            "drawing attrs with one batched call per gene family"
        )

    def transform(
        self,
        state: State,
        node_keys: evoxc.SymbolicTensor,
        conn_keys: evoxc.SymbolicTensor,
    ) -> Transformed:
        """Derive the execution schedule of every individual (in-graph, no host round-trip).

        The only place the schedule is derived: it delegates to
        ``..common.tensorization.build_schedule(node_keys, conn_keys, self.max_in_degree,
        self.network_type)``.  It must stay outside any differentiated graph (``argsort`` has
        no AD rule), which is why :meth:`grad` takes an already-derived ``Transformed``.

        Args:
            state: Host configuration container.
            node_keys: ``(P, N) int32`` node keys, ``I_INF`` padded.
            conn_keys: ``(P, E, CK) int32`` connection keys, ``I_INF`` padded.

        Returns:
            ``Transformed`` with ``in_src``/``in_conn``/``in_valid (P, N, K)``,
            ``node_level``/``node_exists (P, N)``, ``overflow (P,)``; cost is
            edge-proportional, never ``max_nodes**2`` per node.

        Raises:
            NotImplementedError: Skeleton stub.
        """
        raise NotImplementedError(
            "build_schedule(node_keys, conn_keys, self.max_in_degree, self.network_type)"
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
        to cache per shape signature.  Three evoxc rules shape it: the transform needs exactly
        one scalar output, so the loss is scalarized as ``sum(loss_fn(outputs))``
        (per-individual gradients are unaffected — the population axis is independent of the
        sum); only float arguments are differentiable, so ``argnums`` selects
        ``node_attrs``/``conn_attrs`` while keys and the schedule stay non-differentiated
        ``int32``/bool inputs (a differentiated int32 argument is a hard ``TransformError``);
        and the schedule stays outside the differentiated graph because ``argsort`` has no AD
        rule.  Gradients are cleaned with ``select(isnan(g), 0, g)`` and masked to absent rows,
        so padding never carries NaN.

        Args:
            state: Host configuration container.
            transformed: Execution schedule of the population, already derived.
            node_keys: ``(P, N) int32`` node keys, non-differentiated input.
            node_attrs: ``(P, N, NA) float32`` node attributes, differentiated input.
            conn_keys: ``(P, E, CK) int32`` connection keys, non-differentiated input.
            conn_attrs: ``(P, E, CA) float32`` connection attributes, differentiated input.
            inputs: ``(P, num_inputs) float32`` network inputs.
            loss_fn: ``(P, num_outputs) -> (P, ...)`` float32 per-individual loss, summed
                before the gradient transform.

        Returns:
            ``evoxc.Graph`` mapping the same arguments to
            ``(scalar_loss, (node_attr_grads, conn_attr_grads))``, gradients taking the shapes
            and dtypes of the differentiated inputs.

        Raises:
            NotImplementedError: Skeleton stub.
        """
        raise NotImplementedError(
            "trace forward, scalarize the loss with sum(loss_fn(outputs)), take evoxc.grad "
            "over (node_attrs, conn_attrs), then NaN-clean and mask padding rows"
        )

    # ---------------------------------------------------------------- shared kernel
    def _hit_conn_attrs(
        self, transformed: Transformed, conn_attrs: evoxc.SymbolicTensor
    ) -> evoxc.SymbolicTensor:
        """Gather the attributes of every incoming edge slot, once per forward pass.

        Attributes are constant during a forward pass, so the gather is hoisted out of the
        level/tick loop instead of being repeated per step.

        Args:
            transformed: Schedule whose ``in_conn`` is the flat ``(P, N, K) int32`` index into
                ``reshape(conn_attrs, (P*max_conns,))``.
            conn_attrs: ``(P, E, CA) float32`` connection attributes, NaN padded.

        Returns:
            ``(P, N, K, CA) float32`` per-slot attributes; absent slots hold NaN, which the
            conn gene sanitises.

        Raises:
            NotImplementedError: Skeleton stub.
        """
        raise NotImplementedError(
            "gather(conn_attrs.reshape((P*E, CA)), in_conn.reshape((P*N, K))) -> (P, N, K, CA)"
        )

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

        1. ``xv = gather(values.reshape((P*N,)), in_src.reshape((P*N, K)))`` -> ``(P, N, K)``
           source values per slot; ``values`` stays ``(P, N)`` and is never widened into a
           ``max_nodes``-wide column per node.
        2. ``live = in_valid & ~isnan(xv)``; ``xs = select(live, xv, 0)`` — the NaN-to-0
           sanitisation is mandatory before any differentiable op, because a zeroed cotangent
           times a NaN local derivative is still NaN.
        3. ``msg = conn_gene.forward(state, hit_conn_attrs, xs)`` -> ``(P, N, K)``, one scalar
           message per edge slot, elementwise in the slot.
        4. ``z = node_gene.forward(state, node_attrs, msg, is_output_row, live)`` -> ``(P, N)``:
           the gene runs ``apply_aggregation(aggregation_index, msg, live,
           aggregation_options)`` (dead slots neutral-filled per option, then a ``select``
           chain), then ``combine`` (``bias + response * agg``), then the activation select
           except on output rows.
        5. ``values = select(update_mask & (live_count > 0), z, values)`` with
           ``live_count = reduce_sum(live, axis=-1)``: a node with no live incoming edge keeps
           its previous value, so NaN stays NaN and its descendants lose their only live
           source — the legacy NaN poisoning, reproduced exactly.

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

        Raises:
            NotImplementedError: Skeleton stub.
        """
        raise NotImplementedError(
            "gather in_src, live = in_valid & ~isnan(xv), xs = select(live, xv, 0), "
            "msg = conn_gene.forward(hit_conn_attrs, xs), "
            "z = node_gene.forward(node_attrs, msg, is_output_row, live), "
            "values = select(update_mask & (live_count > 0), z, values)"
        )

    def _is_output_row(self, node_keys: evoxc.SymbolicTensor) -> evoxc.SymbolicTensor:
        """Flag the node rows read out as network outputs.

        Args:
            node_keys: ``(P, N) int32`` node keys, ``I_INF`` padded.

        Returns:
            ``(P, N) bool``, True on rows ``max_nodes-num_outputs..max_nodes-1``, which the
            node gene leaves unactivated.

        Raises:
            NotImplementedError: Skeleton stub.
        """
        raise NotImplementedError(
            "True on rows [max_nodes-num_outputs, max_nodes) -> (P, N) bool"
        )

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

        Raises:
            NotImplementedError: Skeleton stub.
        """
        raise NotImplementedError(
            "take values[:, max_nodes-num_outputs:] then apply output_transform if any"
        )

    # ------------------------------------------------------------------- operators
    def execute_mutation(
        self,
        state: State,
        key: evoxc.SymbolicTensor,
        genome: GenomeTensors,
        new_node_keys: evoxc.SymbolicTensor,
        new_conn_keys: evoxc.SymbolicTensor,
    ) -> GenomeTensors:
        """Run the batched mutation operator on a whole population.

        Structural keys are allocated by the caller (not RNG draws) and must be unique across
        the population.

        Args:
            state: Host configuration container.
            key: Rank-0 ``evoxc.random`` key for this call.
            genome: ``(P, ...)`` payload to mutate; shapes and dtypes are preserved.
            new_node_keys: ``(P,) int32`` caller-allocated node keys the add-node mutation may
                insert.
            new_conn_keys: ``(P, 3) int32`` caller-allocated connection keys: column 0 feeds the
                add-node identity edge, column 1 the add-node copy edge, column 2 the
                add-connection edge.

        Returns:
            ``GenomeTensors`` with identical static shapes — the mutated population.

        Raises:
            NotImplementedError: Skeleton stub.
            TensorNEATError: (implementation wave) if the key tensors have the wrong shapes.
        """
        raise NotImplementedError(
            "self.mutation(state, self, key, *genome, new_node_keys, new_conn_keys)"
        )

    def execute_crossover(
        self,
        state: State,
        key: evoxc.SymbolicTensor,
        genome1: GenomeTensors,
        genome2: GenomeTensors,
    ) -> GenomeTensors:
        """Recombine two batches of parents into one batch of offspring.

        Args:
            state: Host configuration container.
            key: Rank-0 ``evoxc.random`` key for this call.
            genome1: ``(P, ...)`` payload of the fitter parent; its structure is kept.
            genome2: ``(P, ...)`` payload of the second parent, same shapes.

        Returns:
            ``GenomeTensors`` — parent 1's keys with key-aligned recombined attributes.

        Raises:
            NotImplementedError: Skeleton stub.
            TensorNEATError: (implementation wave) if the parents' shapes differ.
        """
        raise NotImplementedError("self.crossover(state, self, key, *genome1, *genome2)")

    def execute_distance(
        self, state: State, genome1: GenomeTensors, genome2: GenomeTensors
    ) -> evoxc.SymbolicTensor:
        """Compute one compatibility distance per aligned pair of genomes.

        Args:
            state: Host configuration container.
            genome1: ``(P, ...)`` payload of the first parent of each pair.
            genome2: ``(P, ...)`` payload of the second parent of each pair, same shapes.

        Returns:
            ``(P,) float32`` distances, ``0`` for an empty pair rather than NaN.

        Raises:
            NotImplementedError: Skeleton stub.
            TensorNEATError: (implementation wave) if the parents' shapes differ.
        """
        raise NotImplementedError("self.distance(state, self, *genome1, *genome2)")

    def check_cycles(
        self,
        node_keys: evoxc.SymbolicTensor,
        conn_keys: evoxc.SymbolicTensor,
        src_keys: evoxc.SymbolicTensor,
        dst_keys: evoxc.SymbolicTensor,
    ) -> evoxc.SymbolicTensor:
        """Ask whether adding one edge per individual would close a cycle.

        Documented delegation to ``.utils.check_cycles``; the mutation operator calls the
        predicate through the genome, so a genome can opt out by ``network_type``.

        Args:
            node_keys: ``(P, N) int32`` node keys, ``I_INF`` padded.
            conn_keys: ``(P, E, CK) int32`` connection keys, ``I_INF`` padded.
            src_keys: ``(P,) int32`` new edge source node keys.
            dst_keys: ``(P,) int32`` new edge destination node keys.

        Returns:
            ``(P,) bool`` True where the new edge would close a cycle; False for absent or
            degenerate keys.

        Raises:
            NotImplementedError: Skeleton stub.
        """
        raise NotImplementedError(
            "delegate to utils.check_cycles(node_keys, conn_keys, src_keys, dst_keys)"
        )

    # ------------------------------------------------------------------ host-only
    def get_input_idx(self) -> list[int]:
        """Keys and row positions of the input nodes.

        Returns:
            ``[0, 1, ..., num_inputs-1]``; the row/key invariant makes both coincide.

        Raises:
            NotImplementedError: Skeleton stub.
        """
        raise NotImplementedError("list(range(num_inputs))")

    def get_output_idx(self) -> list[int]:
        """Keys and row positions of the output nodes.

        Returns:
            ``[max_nodes-num_outputs, ..., max_nodes-1]``; the row/key invariant makes both
            coincide.

        Raises:
            NotImplementedError: Skeleton stub.
        """
        raise NotImplementedError("list(range(max_nodes - num_outputs, max_nodes))")

    def hash(self, genome: GenomeTensors) -> evoxc.SymbolicTensor:
        """Hash every individual's payload for fast de-duplication (in-graph).

        An xor-mix over the ``int32`` key columns.  evoxc has no float/int bitcast, so float
        attributes are quantised (scaled, rounded to ``int32``) before mixing instead of being
        hashed by bit pattern; NaN padding quantises to one sentinel, so identical genomes hash
        identically.

        Args:
            genome: ``(P, ...)`` payload batch.

        Returns:
            ``(P,) uint32`` hashes, one per individual.

        Raises:
            NotImplementedError: Skeleton stub.
        """
        raise NotImplementedError(
            "xor-mix (keys, quantised attrs) of both gene families -> (P,) uint32"
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
            NotImplementedError: Skeleton stub.
        """
        raise NotImplementedError(
            "re_cound_idx (optional) then build {nodes: ..., conns: ...} row by row"
        )

    def repr(self, state: State, genome: GenomeTensors, precision: int = 2) -> str:
        """Render one individual as a multi-line string (host-only).

        Args:
            state: Host configuration container.
            genome: Payload of one individual.
            precision: Decimal places forwarded to the gene ``repr`` hooks.

        Returns:
            ``"<ClassName>(nodes=<count>, conns=<count>):"`` then one line per node (annotated
            ``(input)``/``(output)``) and one line per connection.

        Raises:
            NotImplementedError: Skeleton stub.
        """
        raise NotImplementedError(
            "header with valid node/conn counts, then node_gene.repr / conn_gene.repr lines"
        )

    def _get_node_dict(self, state: State, node_keys: evoxc.SymbolicTensor) -> dict:
        """Map node key -> node dict for one individual (host-only).

        Args:
            state: Host configuration container.
            node_keys: Node key array of one individual, ``I_INF`` padded.

        Returns:
            ``{node_key: node_gene.to_dict(...)}``, padded rows skipped.

        Raises:
            NotImplementedError: Skeleton stub.
        """
        raise NotImplementedError("walk node rows, skip I_INF rows, call node_gene.to_dict")

    def _get_conn_dict(self, state: State, conn_keys: evoxc.SymbolicTensor) -> dict:
        """Map connection ``(in_key, out_key)`` -> conn dict for one individual (host-only).

        Args:
            state: Host configuration container.
            conn_keys: Connection key array of one individual, ``I_INF`` padded.

        Returns:
            ``{(in_key, out_key): conn_gene.to_dict(...)}``, padded rows skipped.

        Raises:
            NotImplementedError: Skeleton stub.
        """
        raise NotImplementedError("walk conn rows, skip I_INF rows, call conn_gene.to_dict")

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
