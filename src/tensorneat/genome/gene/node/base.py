"""Node gene base class: one node's aggregated window -> that node's value.

Every kernel here is batch-first (population on axis 0) and elementwise over the
node axis: no per-node Python loop, no ``vmap``/``scan``/``cond``, and no branching
on traced values -- ``select`` chains only.
This file owns the forward contract and the host-side repr/dict scaffolding; the
attr tables and the concrete aggregation/combine algebra belong to the families.
"""

from typing import Any

import evoxc

from ....common import State
from ..base import BaseGene


class BaseNode(BaseGene):
    """Node gene: turn one node's incoming-message window into the node value.

    Contract of :meth:`forward`, where ``P`` is the population size, ``N`` is
    ``max_nodes``, ``K`` is the static in-degree window width and ``NA`` is
    ``attr_width``::

        z = combine(attrs, aggregate(attrs, messages, valid_mask))   # (P, N)
        z = activation(attrs, z)   where is_output_node is False
        z                          where is_output_node is True

    ``messages`` is already NaN-sanitized by the caller (a slot whose source value is
    NaN carries 0), so no gene sees NaN; dead slots are neutralised per aggregation
    option inside :meth:`aggregate`.
    The node is never activated when it is an output node: the final activation is
    the genome-level ``output_transform``.
    An option index of ``-1`` means *identity* in this rewrite (the legacy
    ``lax.switch(-1, ...)`` clamped to the last option, which silently picked the
    wrong activation), and every configured option is evaluated then selected with
    ``select`` because ``select`` is the only batched selection op.

    Only one ``int32`` key column is declared here; hidden node keys are arbitrary
    and may move rows, so graph identity is always compared on keys, never on row
    positions.
    """

    fixed_attrs: tuple[str, ...] = ("index",)

    def forward(
        self,
        state: State,
        attrs: evoxc.SymbolicTensor,
        messages: evoxc.SymbolicTensor,
        is_output_node: evoxc.SymbolicTensor,
        valid_mask: evoxc.SymbolicTensor,
    ) -> evoxc.SymbolicTensor:
        """Compute the value of every node row of the population in one pass.

        The caller feeds one depth level at a time: ``messages`` is the already
        gathered and NaN-sanitized incoming window of the nodes of that level.

        Args:
            state: Host configuration container (read-only during a trace).
            attrs: Node attributes, ``(P, N, NA)`` float32, one row per node slot.
            messages: One scalar message per incoming edge slot, ``(P, N, K)``
                float32, already NaN-sanitized (NaN source -> 0).
            is_output_node: ``(P, N)`` bool, True for the rows read out as network
                outputs, which must stay unactivated here.
            valid_mask: ``(P, N, K)`` bool, True where the slot holds a real, live
                connection; dead slots must not contribute to the aggregation.

        Returns:
            ``(P, N)`` float32 node values: ``combine(attrs, aggregate(...))`` with the
            activation applied only where ``is_output_node`` is False.

        Raises:
            NotImplementedError: Skeleton stub; the select-chain assembly lands in
                the implementation wave.
        """
        raise NotImplementedError(
            "z = combine(attrs, aggregate(attrs, messages, valid_mask)); apply activation "
            "only where is_output_node is False, as a select chain over the configured options"
        )

    def aggregate(
        self,
        state: State,
        attrs: evoxc.SymbolicTensor,
        messages: evoxc.SymbolicTensor,
        valid_mask: evoxc.SymbolicTensor,
    ) -> evoxc.SymbolicTensor:
        """Reduce the incoming-message window of every node with its own option.

        Each configured aggregation option reduces the last (in-degree ``K``) axis
        after replacing the dead slots of ``valid_mask`` with that option's neutral
        element (``0``, ``1``, ``+/-inf``), so every branch stays NaN-free; the
        per-row option index stored in ``attrs`` then picks the branch with a
        ``select`` chain. A single configured option emits no selection op at all.

        Args:
            state: Host configuration container.
            attrs: Node attributes, ``(P, N, NA)`` float32; the aggregation index
                column selects the branch per node row.
            messages: ``(P, N, K)`` float32, already NaN-sanitized.
            valid_mask: ``(P, N, K)`` bool, True for live slots.

        Returns:
            ``(P, N)`` float32 aggregated values, one per node row.

        Raises:
            NotImplementedError: Abstract hook; node families implement it.
        """
        raise NotImplementedError(
            "reduce the last (K) axis per aggregation option with neutral-filled dead slots, "
            "then select by the row's aggregation index"
        )

    def combine(
        self,
        state: State,
        attrs: evoxc.SymbolicTensor,
        aggregated: evoxc.SymbolicTensor,
    ) -> evoxc.SymbolicTensor:
        """Turn the aggregated incoming signal into the node's pre-activation value.

        `DefaultNode` uses ``bias + response * aggregated``, `BiasNode` uses
        ``bias + aggregated``; both are elementwise in the node axis.

        Args:
            state: Host configuration container.
            attrs: Node attributes, ``(P, N, NA)`` float32.
            aggregated: ``(P, N)`` float32 output of :meth:`aggregate`.

        Returns:
            ``(P, N)`` float32 pre-activation node values.

        Raises:
            NotImplementedError: Abstract hook; node families implement it.
        """
        raise NotImplementedError(
            "elementwise bias/response affine map of the aggregated (P, N) value"
        )

    def repr(
        self,
        state: State,
        key_row: evoxc.SymbolicTensor,
        attr_row: evoxc.SymbolicTensor,
        precision: int = 2,
    ) -> str:
        """Render one node gene as a human-readable string (host-only, never traced).

        Args:
            state: Host configuration container.
            key_row: ``(key_width,)`` int32 key columns of a single node gene; the
                node key lives in key column 0.
            attr_row: ``(attr_width,)`` float32 attribute columns of the same gene.
            precision: Decimal places for the float attributes.

        Returns:
            A single-line description such as ``DefaultNode(idx=3)``.

        Raises:
            NotImplementedError: Skeleton stub.
        """
        raise NotImplementedError(
            "format key column 0 plus the family's float attributes as one line"
        )

    def to_dict(
        self,
        state: State,
        key_row: evoxc.SymbolicTensor,
        attr_row: evoxc.SymbolicTensor,
    ) -> dict[str, Any]:
        """Convert one node gene to a plain dict for host code generation.

        Args:
            state: Host configuration container.
            key_row: ``(key_width,)`` int32 key columns of a single node gene.
            attr_row: ``(attr_width,)`` float32 attribute columns of the same gene.

        Returns:
            Mapping such as ``{"idx": 3, "bias": 0.25, "agg": "sum", "act": "sigmoid"}``
            with option indices already resolved to registered function names.

        Raises:
            NotImplementedError: Skeleton stub.
        """
        raise NotImplementedError(
            "map key column 0 to 'idx' and resolve the option indices to function names"
        )

    def sympy_func(
        self, state: State, node_dict: dict, inputs: Any, is_output_node: bool = False
    ) -> tuple[Any, dict]:
        """Build the sympy expression of one node from its ``to_dict`` output.

        Args:
            state: Host configuration container.
            node_dict: Output of :meth:`to_dict` for this node.
            inputs: sympy expression of the aggregated incoming signal (the sum of
                the incoming edge messages), used as the aggregation operand.
            is_output_node: True to leave the expression unactivated, matching
                :meth:`forward`.

        Returns:
            Tuple ``(expression, {symbol: value})``: the node's sympy expression and
            the symbol-to-attribute map used to substitute concrete gene values when
            generating Python code.

        Raises:
            NotImplementedError: Abstract hook; node families implement it.
        """
        raise NotImplementedError(
            "sympy expression of combine(agg(inputs)) plus the {symbol: value} substitution map"
        )
