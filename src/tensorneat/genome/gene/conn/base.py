"""Connection gene base class: one edge's ``(attrs, source value)`` -> one message.

Every kernel is batch-first (population on axis 0) and purely elementwise in the
edge slot, so the per-level forward stays a single vectorized op and messages never
couple across edges.
This file owns the key-column contract, the forward signature and the
historical-marker hook; concrete weight machinery belongs to the families and must
never be duplicated here.
"""

from typing import Any

import evoxc

from ....common import State
from ..base import BaseGene


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
    """

    fixed_attrs: tuple[str, ...] = ("input_index", "output_index")

    def forward(
        self,
        state: State,
        attrs: evoxc.SymbolicTensor,
        source_values: evoxc.SymbolicTensor,
    ) -> evoxc.SymbolicTensor:
        """Turn each edge's attributes and source value into a scalar message.

        Args:
            state: Host configuration container (read-only during a trace).
            attrs: Per-slot connection attributes, ``(P, N, K, CA)`` float32, gathered
                once per generation with the incoming-edge table; rows of absent
                connections are NaN.
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
            NotImplementedError: Hook; only families that declare a
                ``historical_marker`` key column implement it
                (:class:`.origin.OriginConn` returns key column 2).
        """
        raise NotImplementedError(
            "return key column 2 for families that declare a historical_marker key column"
        )

    def repr(
        self,
        state: State,
        key_row: evoxc.SymbolicTensor,
        attr_row: evoxc.SymbolicTensor,
        precision: int = 2,
    ) -> str:
        """Render one connection gene as a human-readable string (host-only).

        Args:
            state: Host configuration container.
            key_row: ``(key_width,)`` int32 key columns of a single gene.
            attr_row: ``(attr_width,)`` float32 attribute columns of the same gene.
            precision: Decimal places for the float attributes.

        Returns:
            A single-line description such as ``DefaultConn(in: 1, out: 4)``.

        Raises:
            NotImplementedError: Skeleton stub.
        """
        raise NotImplementedError(
            "format the input/output key columns plus the family's float attributes"
        )

    def to_dict(
        self,
        state: State,
        key_row: evoxc.SymbolicTensor,
        attr_row: evoxc.SymbolicTensor,
    ) -> dict[str, Any]:
        """Convert one connection gene to a plain dict for host code generation.

        Args:
            state: Host configuration container.
            key_row: ``(key_width,)`` int32 key columns of a single gene.
            attr_row: ``(attr_width,)`` float32 attribute columns of the same gene.

        Returns:
            Mapping such as ``{"in": 1, "out": 4, "weight": 0.5}`` (plus
            ``"historical_marker"`` for families that declare it), consumed by the
            host-side ``network_dict`` / ``sympy_func`` code generation.

        Raises:
            NotImplementedError: Skeleton stub.
        """
        raise NotImplementedError(
            "map the input/output key columns to 'in'/'out' and the attrs to named scalars"
        )

    def sympy_func(self, state: State, conn_dict: dict, inputs: Any) -> tuple[Any, dict]:
        """Build the sympy expression of one connection from its ``to_dict`` output.

        Args:
            state: Host configuration container.
            conn_dict: Output of :meth:`to_dict` for this connection.
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
