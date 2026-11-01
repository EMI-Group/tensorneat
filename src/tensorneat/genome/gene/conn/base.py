"""Connection gene base class: one edge's ``(attrs, source value)`` -> one message.

Every kernel is batch-first (population on axis 0) and purely elementwise in the
edge slot, so the per-level forward stays a single vectorized op and messages never
couple across edges.
This file owns the key-column contract, the split-payload assembly, the
loop-invariant attribute bundle (:class:`ConnOperands`), the forward signature and the
historical-marker hook; concrete weight machinery belongs to the families and must
never be duplicated here.
"""

from typing import Any, NamedTuple

import evoxc
import evoxc.numpy as enp

from ....common import State
from ....common.errors import ConfigError
from ..base import BaseGene, column


class ConnOperands(NamedTuple):
    """Loop-invariant view of one ``(..., CA)`` connection-attribute batch.

    The incoming-edge attributes are gathered once per forward pass and then fed to the
    family once per unrolled level/tick, so the per-column read (``slice`` + ``reshape``,
    two traced ops) and the NaN cleaning of every column (``isnan`` + ``select``, two more)
    would otherwise repeat on every level. This bundle carries the cleaned columns of the
    whole pass.

    Attributes:
        columns: One tensor per ``attr_specs`` column, in table order, with the NaN of the
            padding rows cleaned to ``0`` (the default "absent row" policy of
            :meth:`BaseConn.prepare`).
        zero: Rank-0 ``float32`` zero, created once per pass so a family can sanitise its
            ``NaN`` operand without materialising a fresh constant on every level.
    """

    columns: tuple
    zero: Any


class BaseConn(BaseGene):
    """Connection gene: two ``int32`` key columns plus float attributes.

    Key columns are ``(input_index, output_index)`` -- the source and destination
    node keys, compared exactly; the attribute columns of a concrete family (e.g.
    ``weight``) are float32. Layout of :meth:`forward`, with ``P`` the population
    size, ``N`` ``max_nodes``, ``K`` the static in-degree window width and ``CA``
    ``attr_width``::

        message = forward(attrs, source_values)     # (P, N, K), elementwise

    so one scalar message is produced per edge slot and the destination node
    aggregates them. A gene must not rely on ``NaN * 0`` arithmetic: the NaN
    sanitisation of the source value happens inside the family's ``forward``.

    :attr:`labels` renames the key columns for the inherited host-only
    :meth:`~tensorneat.genome.gene.base.BaseGene.repr` /
    :meth:`~tensorneat.genome.gene.base.BaseGene.to_dict`, so both render
    ``in``/``out`` without a family override:
    ``DefaultConn(in: 1, out: 4, weight: 0.5)`` and
    ``{"in": 1, "out": 4, "weight": 0.5}``.
    """

    fixed_attrs: tuple[str, ...] = ("input_index", "output_index")
    #: Display labels of the key columns, used by the inherited ``repr``/``to_dict``.
    labels: dict[str, str] = {"input_index": "in", "output_index": "out"}

    @property
    def uses_historical_marker(self) -> bool:
        """Whether this family carries a ``historical_marker`` key column.

        Returns:
            ``True`` iff ``"historical_marker"`` is declared in ``fixed_attrs``, i.e.
            the family aligns genes by the NEAT-paper marker
            (:class:`.origin.OriginConn`) instead of by ``(in, out)`` alone.
        """
        return "historical_marker" in self.fixed_attrs

    def from_attrs(
        self,
        state: State,
        in_keys: evoxc.SymbolicTensor,
        out_keys: evoxc.SymbolicTensor,
        attrs: evoxc.SymbolicTensor | None = None,
        historical_marker: evoxc.SymbolicTensor | None = None,
    ) -> tuple[evoxc.SymbolicTensor, evoxc.SymbolicTensor]:
        """Assemble the split connection payload pair from pre-built columns.

        The payload is always two tensors, never one packed row: the ``int32`` key
        array ``(*shape, key_width)`` and the ``float32`` attribute array
        ``(*shape, attr_width)``. ``algorithm/hyperneat`` builds its dense substrate
        payload with exactly this pair -- ``node_gene.from_attrs(keys)`` for the nodes
        and ``conn_gene.from_attrs(in_keys, out_keys, weights)`` for the edges -- so a
        connection set that was never created by mutation still enters the graphs.

        Args:
            state: Host configuration container.
            in_keys: ``(*shape,)`` int32 (or integer) keys of the source node of every
                connection.
            out_keys: ``(*shape,)`` int32 (or integer) keys of the destination node of
                every connection, same shape as ``in_keys``.
            attrs: ``(*shape, attr_width)`` float32 attribute rows, a ``(*shape,)``
                per-connection scalar (e.g. the ``(P, Q)`` weight table of a dense
                substrate) broadcast over the attribute axis, or ``None`` for zeros.
            historical_marker: ``(*shape,)`` int32 markers, used only when
                :attr:`uses_historical_marker` is ``True``; ``None`` fills them with
                the flat row index (``0 .. prod(shape) - 1``, reshaped to ``shape``),
                i.e. every connection gets a distinct marker.

        Returns:
            ``(conn_keys, conn_attrs)``: ``(*shape, key_width)`` int32 key rows and
            ``(*shape, attr_width)`` float32 attribute rows.

        Raises:
            tensorneat.common.errors.ConfigError: If the two key columns disagree in
                shape, if a ``historical_marker`` is supplied to a family that does
                not declare that key column, or if ``attrs`` cannot be interpreted as
                one attribute row per connection.
        """
        name = f"{type(self).__name__}.from_attrs"
        in_keys = evoxc.cast(in_keys, evoxc.int32)
        out_keys = evoxc.cast(out_keys, evoxc.int32)
        shape = tuple(in_keys.shape)
        if tuple(out_keys.shape) != shape:
            raise ConfigError(
                f"{name}: destination keys must have the shape of the source keys "
                f"{shape}, got {tuple(out_keys.shape)}"
            )
        columns = [in_keys, out_keys]
        if self.uses_historical_marker:
            if historical_marker is None:
                historical_marker = self._row_index_keys(shape)
            columns.append(evoxc.cast(historical_marker, evoxc.int32))
        elif historical_marker is not None:
            raise ConfigError(
                f"{name}: {type(self).__name__} declares no 'historical_marker' key "
                f"column (fixed_attrs={self.fixed_attrs}); pass at most two key columns"
            )
        keys = self._key_array(columns, shape, name)
        return keys, self._normalize_attrs(attrs, shape, name)

    def _row_index_keys(self, shape: tuple[int, ...]) -> evoxc.SymbolicTensor:
        """Build a distinct marker per connection: the flat row index.

        Args:
            shape: Leading (batch) shape of the connection rows.

        Returns:
            ``shape``-shaped int32 row indices ``0 .. prod(shape) - 1``.
        """
        count = 1
        for size in shape:
            count *= int(size)
        return evoxc.reshape(enp.arange(count, dtype=evoxc.int32), shape)

    def prepare(self, attrs: evoxc.SymbolicTensor) -> ConnOperands:
        """Clean every attribute column of one batch once per forward pass.

        Loop-invariant work hoisted out of the unrolled level/tick loop: the gathered
        edge attributes do not change while a forward pass runs, so the per-column reads
        and the NaN cleaning belong here, not in the family's ``forward``. Each column is
        read **once** (a second ``column`` call would emit its ``slice`` + ``reshape``
        pair again) and the ``0`` the cleaning writes is a single rank-0 constant shared
        by every column.

        The default policy is "a NaN attribute means an absent (padding) row, so it reads
        as ``0``", which is what the pre-rewrite library did in every family; a family
        that needs another treatment overrides this method instead of re-deriving it per
        level. ``state`` is deliberately not an argument: the cleaning is pure column
        arithmetic (like :meth:`..base.BaseGene.attr_column`), and the one production
        caller lives in ``BaseGenome._hit_conn_attrs``, which has no state to pass.

        Args:
            attrs: Per-slot connection attributes, ``(..., CA)`` float32, rows of absent
                connections NaN.

        Returns:
            The :class:`ConnOperands` of ``attrs``.

        Raises:
            tensorneat.common.errors.ConfigError: If the attribute width of ``attrs`` does
                not match the declared table.
        """
        width = int(attrs.shape[-1])
        if width != self.attr_width:
            raise ConfigError(
                f"{type(self).__name__} reads {self.attr_width} attribute column(s) but "
                f"the payload carries {width}"
            )
        zero = enp.zeros((), dtype=evoxc.float32)
        columns = []
        for index in range(width):
            values = column(attrs, index)
            columns.append(evoxc.select(evoxc.isnan(values), zero, values))
        return ConnOperands(tuple(columns), zero)

    def column_of(self, attrs: Any, name: str) -> evoxc.SymbolicTensor:
        """Read one named attribute column of a raw batch or of a prepared bundle.

        Args:
            attrs: ``(..., CA)`` float32 attribute rows, or the :class:`ConnOperands`
                bundle of :meth:`prepare`.
            name: Spec name of the column, resolved through ``attr_specs``.

        Returns:
            ``(...,)`` float32 column values.
        """
        if isinstance(attrs, ConnOperands):
            return attrs.columns[self.attr_index(name)]
        return self.attr_column(attrs, name)

    def forward(
        self,
        state: State,
        attrs: Any,
        source_values: evoxc.SymbolicTensor,
    ) -> evoxc.SymbolicTensor:
        """Turn each edge's attributes and source value into a scalar message.

        Args:
            state: Host configuration container (read-only during a trace).
            attrs: Per-slot connection attributes, ``(P, N, K, CA)`` float32, gathered
                once per generation with the incoming-edge table; rows of absent
                connections are NaN. The :class:`ConnOperands` bundle of :meth:`prepare`
                for those rows is accepted too and is what the forward pass passes, so a
                family reads its columns with :meth:`column_of` and skips the re-cleaning.
            source_values: Values of the source node of each slot, ``(P, N, K)``
                float32; NaN where the source has not been computed (unreachable or
                absent node).

        Returns:
            ``(P, N, K)`` float32 messages, elementwise in ``(attrs, source_values)``;
            dead slots carry 0 so the aggregation can neutralise them by mask.

        Raises:
            NotImplementedError: Abstract hook; connection families implement it.
        """
        raise NotImplementedError(
            "elementwise message per edge slot: sanitized source value combined with "
            "the sanitized float attributes -> (P, N, K)"
        )

    def get_historical_marker(
        self, state: State, key_row: evoxc.SymbolicTensor
    ) -> evoxc.SymbolicTensor:
        """Read the historical marker of a connection gene, if the family declares one.

        The marker is the NEAT-paper gene-alignment id: two connections with the same
        marker are homologous even if their endpoints were re-routed. It is an
        ``int32`` key column (compared exactly, ``I_INF`` for absent rows), never a
        float attribute.

        Args:
            state: Host configuration container.
            key_row: ``(*shape, key_width)`` int32 key columns of the connection gene.

        Returns:
            ``(*shape,)`` int32 historical markers.

        Raises:
            NotImplementedError: Abstract hook; only families that declare a
                ``historical_marker`` key column implement it
                (:class:`.origin.OriginConn` returns key column 2).
        """
        raise NotImplementedError(
            "return key column 2 for families that declare a historical_marker key column"
        )

    def sympy_func(self, state: State, conn_dict: dict, inputs: Any) -> tuple[Any, dict]:
        """Build the sympy expression of one connection from its ``to_dict`` output.

        Args:
            state: Host configuration container.
            conn_dict: Output of ``to_dict`` for this connection.
            inputs: sympy expression of the source node value.

        Returns:
            ``(expression, {symbol: value})``: the message expression of this edge and
            the symbol-to-weight map used to substitute concrete gene values when
            generating code.

        Raises:
            NotImplementedError: Abstract hook; connection families implement it.
        """
        raise NotImplementedError(
            "sympy expression of the edge message plus the {symbol: value} substitution map"
        )
