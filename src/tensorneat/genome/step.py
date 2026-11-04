"""The window/node-step machinery shared by every concrete genome.

``BaseGenome`` composes :class:`NodeStepMixin`, so ``default.py`` (one step per depth level)
and ``recurrent.py`` (one step per tick) reach the same window reduce and never duplicate it.
The split puts base.py's one remaining job — configuration, payload, schedule glue and the
operator entry points — beside the per-pass operand bundle, the per-level reduce and the
payload-row predicates this module owns.
Everything here is batch-first (population on axis 0) with static shapes, and every helper is
a plain method over already-traced ``evoxc`` tensors: no trace of its own, no host conversion.
The cost law that shapes the code: op count, not element count, is the budget on the numpy
reference backend, and an innermost-axis reduction over ``(P, N, K)`` is an order of magnitude
more expensive than a static slice fold, so the loop-invariant operands are derived once per
pass here and a level pays only for the values.
``_hit_conn_attrs`` is the single derivation point; ``_node_step`` is the single reduce point.
"""

from collections.abc import Callable
from typing import Any

import evoxc
import evoxc.numpy as enp
import numpy as np

from ..common import State, TensorNEATError, Transformed
from .window import (
    _column_index,
    _column_mask,
    _gather_prepared,
    _iinf,
    _slot_index,
    any_slot,
    by_samples,
    fill,
)


class _StepStatics:
    """Loop-invariant operands of one forward pass, built once by ``_hit_conn_attrs``.

    Both concrete genomes call :meth:`BaseGenome._hit_conn_attrs` once and then
    :meth:`BaseGenome._node_step` once per unrolled level/tick with the same attribute
    tensors. Everything that does not depend on the current node values — the gathered and
    NaN-cleaned connection attributes, the window-gather index of the value table, its
    present-slot mask, the output-row mask, and (filled on the first step) the prepared node
    attributes — lives here, so the window step spends its op budget on the values instead of
    re-deriving its own operands.

    This is a plain trace-local Python object, never a graph input or output: it holds
    ``evoxc`` tensors of the trace that created it and is not part of any pytree.

    The two genomes read every one of these at the *evaluated* row count, which a problem's
    tiling makes ``P * samples``; all of them except ``flat_index`` are derived at the
    schedule's own ``P`` rows and expanded with :func:`..window.by_samples`, because tiled row
    ``p * samples + j`` is a copy of individual ``p`` (see :meth:`BaseGenome._hit_conn_attrs`).

    Attributes:
        conn: ``ConnOperands`` of the gathered edge attributes, with the NaN of the padding
            rows cleaned, at the evaluated row count.
        flat_index: ``(rows, N, K) int32`` flat gather index of the node-value window.
        present: ``(rows, N, K) bool`` liveness of each window slot.
        output_rows: ``(rows, N) bool``, True on the rows read out as network outputs.
        samples: Static tiling factor ``S`` of the evaluated batch (``1`` for a direct call).
        mask: The ``(P, N) bool`` output-row mask the prepared ``node`` was built against.
        node_attrs: The attribute tensor the prepared ``node`` was built from, so a second
            forward pass cannot silently reuse it.
        node: ``NodeOperands`` of ``node_attrs``; ``None`` until the first step.
    """

    __slots__ = (
        "conn",
        "flat_index",
        "present",
        "output_rows",
        "samples",
        "mask",
        "node_attrs",
        "node",
    )

    def __init__(
        self,
        conn: Any,
        flat_index: evoxc.SymbolicTensor,
        present: evoxc.SymbolicTensor,
        output_rows: evoxc.SymbolicTensor,
        samples: int = 1,
    ):
        """Store the tensors derived once by :meth:`BaseGenome._hit_conn_attrs`.

        Args:
            conn: ``ConnOperands`` of the gathered connection attributes, ``P`` rows.
            flat_index: Flat gather index of the node-value window, already tiled.
            present: Liveness mask of the same slots, already tiled.
            output_rows: ``(P, N) bool`` output-row mask.
            samples: Static tiling factor of the evaluated batch.
        """
        pop_size = int(output_rows.shape[0])
        self.conn = evoxc.tree_map(
            lambda leaf: by_samples(leaf, samples, pop_size), conn
        )
        self.flat_index = flat_index
        self.present = present
        self.output_rows = by_samples(output_rows, samples, pop_size)
        self.samples = samples
        self.mask = output_rows
        self.node_attrs = None
        self.node = None

    def node_operands(
        self, genome: "BaseGenome", node_attrs: evoxc.SymbolicTensor
    ) -> Any:
        """Prepared node attributes of this forward pass, built on the first step.

        The bundle is prepared at the ``P`` rows ``node_attrs`` arrives with and then expanded
        to the evaluated row count: a column read, an option-id cast and an option mask are all
        row-local, so tiled row ``p * samples + j`` reads the copy of individual ``p`` and the
        level/tick loop still receives exactly the operands it used to prepare for itself.

        Args:
            genome: The genome whose node gene reads the attributes.
            node_attrs: ``(P, N, NA) float32`` node attributes of this pass — the same
                tensor on every step, because attributes never change during a pass.

        Returns:
            The ``NodeOperands`` bundle of ``node_attrs``, at the evaluated row count.

        Raises:
            TensorNEATError: If a step is handed a different attribute tensor than the one
                this bundle was built from, which would silently mix two forward passes.
        """
        if self.node is None:
            self.node_attrs = node_attrs
            pop_size = int(node_attrs.shape[0])
            prepared = genome.node_gene.prepare(node_attrs, self.mask)
            self.node = evoxc.tree_map(
                lambda leaf: by_samples(leaf, self.samples, pop_size), prepared
            )
        elif self.node_attrs is not node_attrs:
            raise TensorNEATError(
                "_node_step received different node attributes than the ones these step "
                "statics were built for; derive them once per forward pass and reuse them "
                "for every level/tick"
            )
        return self.node




class NodeStepMixin:
    """The shared in-graph payload machinery ``BaseGenome`` composes.

    Two families, both reading one batch-first payload and both called through the genome
    object: the forward-pass window kernel — the loop-invariant operand bundle of one pass
    (:class:`_StepStatics`, built by :meth:`_hit_conn_attrs`), the per-level reduce
    (:meth:`_node_step`) and the attribute/output row helpers around them — and the payload
    predicates that kernel and the algorithm share (:meth:`node_exists`,
    :meth:`conn_exists`, :meth:`max_node_key`, :meth:`max_conn_marker`).

    It expects the composing class to provide ``node_gene``, ``conn_gene``, ``num_outputs``
    and ``network_type`` — i.e. to be a ``BaseGenome`` (or a subclass of one) — and keeps the
    ``self``-based signatures ``default.py``, ``recurrent.py``, ``grad.py`` and the algorithm
    already call.
    """

    # -------------------------------------------------- attribute padding
    def _padding_safe_attrs(
        self, attrs: evoxc.SymbolicTensor, keys: evoxc.SymbolicTensor
    ) -> evoxc.SymbolicTensor:
        """Read the attribute rows of absent genes as ``0`` instead of ``NaN``.

        An option-index column (``activation``, ``aggregation``) is a float gene column that
        ``common.functions.apply_activation``/``apply_aggregation`` cast to ``int32``; the
        ``NaN`` padding of an absent row would be a ``NaN``-to-integer cast there.  The rows
        this touches belong to genes that do not exist, so their attributes are never read
        back into a network value — replacing them with ``0`` (option id of the first
        configured option, and the neutral value of every other column) changes nothing
        observable, while a ``NaN`` on a *live* row is left alone so a corrupt gene still
        poisons its node instead of silently reading as option ``0``.

        Applied once per forward pass (never per unrolled level), so the padding rows reach
        the node step already clean and the per-level op budget is unchanged.

        Args:
            attrs: ``(P, R, W) float32`` attribute rows, NaN padded.
            keys: ``(P, R) int32`` gene keys of the same rows, ``I_INF`` padded.

        Returns:
            ``(P, R, W) float32``: ``attrs`` with every row whose key is ``I_INF`` set to
            ``0``; live rows are returned untouched (NaN included).
        """
        shape = tuple(attrs.shape)
        if not shape[-1]:  # a family without evolvable attributes
            return attrs
        # ``select`` broadcasts the trailing-``1`` mask over the attribute columns, so the
        # widening ``broadcast_to`` op costs nothing here.
        rows = evoxc.reshape(self.node_exists(keys), (*shape[:-1], 1))
        return evoxc.select(rows, attrs, enp.zeros((), dtype=evoxc.float32))

    # ---------------------------------------------------------------- shared kernel
    @staticmethod
    def _sample_count(inputs: evoxc.SymbolicTensor, pop_size: int) -> int:
        """How many tiled samples each individual carries in this forward pass.

        A problem tiles its dataset or rollout batch over the population axis in
        *individual-major* order, so a ``forward`` can be asked for ``P * S`` rows while the
        schedule and the payload stay at ``P``. The ratio is static (both shapes are), which is
        what lets the caller expand a ``P``-row derivation to the evaluated rows and leave the
        level/tick loop's algebra untouched.

        Args:
            inputs: ``(batch, *input_shape)`` network inputs of this pass.
            pop_size: Static population ``P`` of the schedule and the payload.

        Returns:
            The static sample count ``S = batch // P``, at least ``1``.

        Raises:
            TensorNEATError: If the batch is not a positive multiple of ``P`` -- a batch that
                does not repeat whole individuals has no sample axis to expand, so the
                schedule and the inputs would silently disagree about their rows.
        """
        batch = int(inputs.shape[0])
        samples, remainder = divmod(batch, pop_size)
        if remainder or samples < 1:
            raise TensorNEATError(
                f"the network was asked for a batch of {batch} row(s), which is not a "
                f"positive multiple of the population {pop_size}; a tiled dataset must be "
                "individual-major over the population axis"
            )
        return samples

    @staticmethod
    def _gather_slots(
        x: evoxc.SymbolicTensor,
        idx: evoxc.SymbolicTensor,
        mask: evoxc.SymbolicTensor,
    ) -> evoxc.SymbolicTensor:
        """Gather ``(P, *shape)`` window slots out of one batch-first table.
        The index is flattened to a batch-wide one-dimensional gather index
        (:func:`_slot_index`) because ``evoxc.gather`` follows ``np.take`` semantics and
        splices the *whole* index shape into the axis it indexes; the flat form is the only
        one that works, so ``x`` is flattened, gathered and reshaped back.  When the index is
        derived once per forward pass (:func:`_slot_index`), the per-level half is
        :func:`_gather_prepared` — one flatten plus one gather.
        Masked-out and ``I_INF`` slots are clamped to an in-range row of their own individual
        before the gather, so the backend never sees an out-of-range index; the value they
        return is never read (see :func:`_gather_prepared`).
        Args:
            x: ``(P, R, ...)`` source table (node values, or the connection attribute payload).
            idx: ``(P, *shape) int32`` row indices into ``x``'s second axis; ``I_INF`` (or a
                masked-out position) means "no slot".
            mask: ``(P, *shape) bool`` liveness of each slot.
        Returns:
            ``(P, *shape)`` tensor with ``x``'s dtype.
        """
        flat_index, _ = _slot_index(idx, mask, int(x.shape[1]))
        return _gather_prepared(x, flat_index)
    def _hit_conn_attrs(
        self,
        transformed: Transformed,
        conn_attrs: evoxc.SymbolicTensor,
        samples: int = 1,
    ) -> _StepStatics:
        """Build the loop-invariant operands of one forward pass (called once per pass).

        The edge attributes are constant while a forward pass runs, and so is the window
        index of the node values, so both are derived here — before the level/tick loop —
        and handed to :meth:`_node_step` in a :class:`_StepStatics`. That is what keeps the
        per-level cost at the window arithmetic alone.

        Everything but the flat index is derived at the schedule's own ``P`` rows and expanded
        to the evaluated ``P * samples`` rows by :class:`_StepStatics`: a problem tiles its
        dataset over the population axis, so those rows repeat the same individual and
        re-deriving the operand at each of them would multiply the pass's element traffic by
        ``samples`` for identical numbers.

        Args:
            transformed: Schedule whose ``in_conn`` is the ``(P, N, K) int32`` connection
                **row** index of each slot, and whose ``in_src``/``in_valid`` are the source
                index and liveness of the same slots.
            conn_attrs: ``(P, E, CA) float32`` connection attributes, NaN padded.
            samples: Static tiling factor ``S`` of the batch this pass evaluates.

        Returns:
            :class:`_StepStatics` of this population: the gathered edge attributes (a dead
            slot keeps the attribute row its index was clamped to, which no consumer reads),
            the flat window index of the node values, its present-slot mask and the
            output-row mask.

        Notes:
            The flat index of slot ``(p, n, k, w)`` is ``p * (E * CA) + row * CA + w``, i.e.
            the slot row index times the attribute width plus the column index, because the
            payload is gathered from ``reshape(conn_attrs, (P, E * CA))``. The index is
            returned *block-shaped* (``(P, N, K, CA)``), so the gather of this pass and of
            every unrolled level produces its window shape directly.
        """
        pop_size, max_nodes, window = (int(size) for size in transformed.in_conn.shape)
        conn_rows, attr_width = (int(size) for size in conn_attrs.shape[1:])
        if attr_width:  # a family with evolvable attributes: gather them once per pass
            hit = self._gather_slots(
                evoxc.reshape(conn_attrs, (pop_size, conn_rows * attr_width)),
                _column_index(transformed.in_conn, attr_width, conn_rows),
                _column_mask(transformed.in_valid),
            )
        else:  # a family without evolvable attributes: nothing to gather
            hit = enp.zeros((pop_size, max_nodes, window, 0), dtype=evoxc.float32)
        return _StepStatics(
            self.conn_gene.prepare(hit),
            *_slot_index(transformed.in_src, transformed.in_valid, max_nodes, samples),
            self._output_row_mask(pop_size, max_nodes),
            samples,
        )

    def _node_step(
        self,
        state: State,
        transformed: Transformed,
        node_attrs: evoxc.SymbolicTensor,
        hit_conn_attrs: _StepStatics,
        values: evoxc.SymbolicTensor,
        update_mask: evoxc.SymbolicTensor,
    ) -> evoxc.SymbolicTensor:
        """Advance the node values by one window reduce (the composition point).

        Written here once; ``default.py`` (per depth level) and ``recurrent.py`` (per tick)
        call it with their own ``update_mask`` and must not duplicate it.  With ``P`` the
        evaluated row count (the population, or ``P * samples`` when a problem tiles its
        dataset over the population axis -- ``statics`` is expanded to that width), ``N``
        ``max_nodes`` and ``K`` the window width, and ``statics`` the
        :class:`_StepStatics` of this pass:

        1. ``xv = gather(values, statics.flat_index)`` -> ``(P, N, K)`` source values per
           slot (flatten the value table, one gather into the window block shape — the index
           table is already block-shaped, so nothing is reshaped back). ``values`` stays
           ``(P, N)`` and is never widened into a ``max_nodes``-wide column per node. The
           index table is derived once per pass, so a level pays only flatten + gather: a
           *static* dead slot is clamped to an in-range row and keeps that row's value
           instead of being overwritten with ``NaN``, which would be one ``(P, N, K)``
           ``select`` per level for a value no consumer reads (step 2 masks it out, step 3's
           aggregation neutral-fills it).
        2. ``live = statics.present & (xv == xv)`` — IEEE makes ``x == x`` False exactly on
           ``NaN``, so that equality catches a *live* slot whose source row is still ``NaN``,
           while ``statics.present`` excludes the statically dead slots whose gathered value
           is now an arbitrary in-range row. Two cheap ops on the mask replace the wide fill
           of step 1, and ``live`` is still exactly ``present & ~isnan(gathered)``. The same
           mask is what lets the connection gene skip its own sanitisation.
        3. ``msg = conn_gene.forward(state, statics.conn, xv)`` -> ``(P, N, K)``, one scalar
           message per edge slot: the gene multiplies by the weight column, which arrives
           already NaN-cleaned. A dead slot writes ``value * weight`` for a value nobody
           reads, and that is safe because every aggregation option replaces the slots of
           ``live`` with its neutral value *before* reducing
           (``common/functions/agg.py``), so nothing from a dead slot reaches a reduce or a
           differentiated op. That is why the gene needs no second sanitisation (and why the
           per-level op budget has no fill).
        4. ``z = node_gene.forward(state, statics.node, msg, statics.output_rows, live)`` ->
           ``(P, N)``: the gene runs ``apply_aggregation`` (dead slots neutral-filled per
           option, then a ``select`` chain), then ``combine`` (``bias + response * agg``), then
           the activation, whose id is ``-1`` (identity) on the output rows.
        5. ``values = select(update_mask & any_slot(live), z, values)``: a node with no live
           incoming edge keeps its previous value, so NaN stays NaN and its descendants lose
           their only live source — the legacy NaN poisoning, reproduced exactly.

        Args:
            state: Host configuration container (read-only during a trace).
            transformed: Execution schedule of the population; it must be the one whose
                fields ``statics`` was derived from (its ``in_src`` shape gives the window).
            node_attrs: ``(P, N, NA) float32`` node attributes, NaN padded -- the schedule's
                own ``P`` rows, the one argument that is not tiled (the node operands are
                prepared there and expanded).
            hit_conn_attrs: The :class:`_StepStatics` of :meth:`_hit_conn_attrs` for this
                forward pass, already at the evaluated row count.
            values: ``(rows, N) float32`` current node values, NaN where not yet computed.
            update_mask: ``(P, N) bool`` caller gate — for the feedforward genome the plain
                ``node_level == level`` comparison (the schedule already stores ``I_INF`` on
                every absent row, so a separate ``node_exists`` term is redundant), for the
                recurrent one ``node_exists & any_slot(in_valid)``; the dynamic "has a live
                slot" term is ANDed in here (step 5).

        Returns:
            ``(P, N) float32`` updated values; rows masked off are returned unchanged.

        Raises:
            TensorNEATError: If ``hit_conn_attrs`` is not the :class:`_StepStatics` of
                :meth:`_hit_conn_attrs`, which would silently reintroduce the per-level
                operand derivation this kernel exists to avoid.
        """
        if not isinstance(hit_conn_attrs, _StepStatics):
            raise TensorNEATError(
                "_node_step expects the _StepStatics returned by _hit_conn_attrs, not "
                f"{type(hit_conn_attrs).__name__}"
            )
        block = tuple(int(size) for size in transformed.in_src.shape[1:])
        if tuple(int(size) for size in hit_conn_attrs.present.shape[1:]) != block:
            # A static (host-side) check, hence no traced op: the window index and the mask
            # must come from this very schedule or the gather would read another window.
            raise TensorNEATError(
                "_node_step got step statics of window "
                f"{tuple(int(size) for size in hit_conn_attrs.present.shape[1:])} for a "
                f"schedule of window {block}"
            )
        # The gather is *unfilled*: a statically dead slot is clamped to an in-range row of
        # its own individual (see ``window._slot_index``) and keeps that row's value, so the
        # liveness mask is the static ``present`` mask ANDed with the ``x == x`` test that
        # rejects a live slot whose source row is still ``NaN``. Writing ``NaN`` into the
        # dead slots instead would cost one ``(P, N, K)`` ``select`` per level and buy
        # nothing: no consumer of ``messages`` ever reads a dead slot (the aggregation
        # neutral-fills every slot of ``live`` before reducing), so a narrow
        # ``logical_and`` over the mask replaces the wide fill for bit-identical values.
        xv = _gather_prepared(values, hit_conn_attrs.flat_index)
        live = evoxc.logical_and(hit_conn_attrs.present, evoxc.equal(xv, xv))
        messages = self.conn_gene.forward(state, hit_conn_attrs.conn, xv)
        z = self.node_gene.forward(
            state,
            hit_conn_attrs.node_operands(self, node_attrs),
            messages,
            hit_conn_attrs.output_rows,
            live,
        )
        update = evoxc.logical_and(update_mask, any_slot(live))
        return evoxc.select(update, z, values)

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
        flag = np.zeros(max_nodes, dtype=bool)
        flag[max_nodes - self.num_outputs :] = True
        row = evoxc.constant(evoxc.from_numpy(np.ascontiguousarray(flag)))
        return enp.broadcast_to(evoxc.reshape(row, (1, max_nodes)), (pop_size, max_nodes))

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

    # ------------------------------------------------------------- payload rows
    def node_exists(self, node_keys: evoxc.SymbolicTensor) -> evoxc.SymbolicTensor:
        """Mark the node rows that hold a real node key.

        Args:
            node_keys: ``(P, N) int32`` node keys, ``I_INF`` padded.

        Returns:
            ``(P, N) bool``, True where the row is not the padding sentinel.
        """
        return enp.not_equal(node_keys, _iinf(node_keys.dtype))

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
            sources, _iinf(sources.dtype)
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
            exists, node_keys, fill(tuple(node_keys.shape), 0, dtype=node_keys.dtype)
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
            return fill((pop_size,), 0, dtype=evoxc.int32)
        markers = self.conn_gene.key_column(conn_keys, "historical_marker")
        exists = self.conn_exists(conn_keys)
        safe = evoxc.select(
            exists, markers, fill(tuple(markers.shape), 0, dtype=markers.dtype)
        )
        return evoxc.cast(enp.max(safe, axis=-1), evoxc.int32)
