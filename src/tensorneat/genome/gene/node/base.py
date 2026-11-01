"""Node gene base class: one node's aggregated window -> that node's value.

Every kernel here is batch-first (population on axis 0) and elementwise over the
node axis: no per-node Python loop, no ``vmap``/``scan``/``cond``, and no branching
on traced values -- ``select`` chains only.
This file owns the forward contract, the option-independent aggregation and the
host-side repr/dict scaffolding; the attr tables and the concrete combine algebra
belong to the families.
"""

from typing import Any, Callable, Sequence

import evoxc

from ....common import ACT, AGG, State
from ....common.errors import ConfigError
from ....common.functions import (
    FunctionManager,
    apply_activation,
    apply_aggregation,
)
from ....common.functions.act import ACT_IDENTITY
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
    The two option lists are class data with no family default (``()``), so a family
    must materialise them; :meth:`setup` rejects an empty list because an empty
    cascade would return its neutral value for every node instead of failing loudly.
    """

    fixed_attrs: tuple[str, ...] = ("index",)
    #: Node keys are called ``idx`` in ``repr``/``to_dict``; the payload column order
    #: is untouched (the label is presentation only).
    labels: dict[str, str] = {"index": "idx"}
    #: Static option tuples, filled in by the families (``DefaultNode`` /
    #: ``BiasNode`` / ``OriginNode`` resolve them in their constructor).
    aggregation_options: tuple = ()
    activation_options: tuple = ()

    def setup(self, state: State | None = None) -> State:
        """Validate the attr table and the option lists on the host.

        Chains :meth:`..base.BaseGene.setup` (which checks the attr table) and then
        checks that the two option lists a node gene dispatches on are non-empty:
        an empty list would make ``apply_aggregation`` / ``apply_activation`` return
        the cascade's neutral value for every node, silently disabling the gene.

        Args:
            state: Base state to extend; never mutated in place.

        Returns:
            The same ``State`` (or ``State()`` when ``state`` is ``None``).

        Raises:
            tensorneat.common.errors.ConfigError: If the attr table is inconsistent
                (see :meth:`..base.BaseGene.setup`) or an option list is empty.
        """
        state = super().setup(state)
        for name in ("aggregation_options", "activation_options"):
            if not getattr(self, name):
                raise ConfigError(
                    f"{type(self).__name__}.{name} must hold at least one option "
                    "callable; an empty list would neutralise every node"
                )
        return state

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
            attrs: Node attributes, ``(P, N, NA)`` float32, one row per node slot;
                the ``activation`` column holds the per-row option index.
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
            tensorneat.common.errors.ConfigError: If the attr table declares no
                ``activation`` column (families always declare one).
        """
        aggregated = self.aggregate(state, attrs, messages, valid_mask)
        z = self.combine(state, attrs, aggregated)
        act_ids = self.attr_column(attrs, "activation")
        activated = apply_activation(z, act_ids, self.activation_options)
        return evoxc.select(is_output_node, z, activated)

    def aggregate(
        self,
        state: State,
        attrs: evoxc.SymbolicTensor,
        messages: evoxc.SymbolicTensor,
        valid_mask: evoxc.SymbolicTensor,
    ) -> evoxc.SymbolicTensor:
        """Reduce the incoming-message window of every node with its own option.

        The reduction itself is option-independent and therefore implemented here
        once: :func:`~tensorneat.common.functions.apply_aggregation` evaluates every
        configured aggregation on the whole ``(P, N, K)`` window, replacing the dead
        slots of ``valid_mask`` with that option's neutral element (``0``, ``1``,
        ``+/-inf``) before reducing, and then picks the row's option with a ``select``
        cascade. A node whose window has no live slot gets exactly ``0.0`` from the
        dispatcher, never ``NaN`` and never a neutral that leaks into the value.
        Families inherit this method unchanged (a family that needs a different
        operator over the window would override it *without* copying the cascade).

        Args:
            state: Host configuration container.
            attrs: Node attributes, ``(P, N, NA)`` float32; the ``aggregation`` index
                column selects the branch per node row.
            messages: ``(P, N, K)`` float32, already NaN-sanitized.
            valid_mask: ``(P, N, K)`` bool, True for live slots.

        Returns:
            ``(P, N)`` float32 aggregated values, one per node row.

        Raises:
            tensorneat.common.errors.ConfigError: If the attr table declares no
                ``aggregation`` column (families always declare one).
        """
        return apply_aggregation(
            messages,
            valid_mask,
            self.attr_column(attrs, "aggregation"),
            self.aggregation_options,
        )

    def combine(
        self,
        state: State,
        attrs: evoxc.SymbolicTensor,
        aggregated: evoxc.SymbolicTensor,
    ) -> evoxc.SymbolicTensor:
        """Turn the aggregated incoming signal into the node's pre-activation value.

        ``DefaultNode`` uses ``bias + response * aggregated``, ``BiasNode`` uses
        ``bias + aggregated``; both are elementwise in the node axis.

        Args:
            state: Host configuration container.
            attrs: Node attributes, ``(P, N, NA)`` float32.
            aggregated: ``(P, N)`` float32 output of :meth:`aggregate`.

        Returns:
            ``(P, N)`` float32 pre-activation node values.

        Raises:
            NotImplementedError: Abstract hook -- the base class declares no attribute
                table, so it cannot know which columns to combine. Every node family
                (``DefaultNode``, ``BiasNode``, ``OriginNode``) implements it.
        """
        raise NotImplementedError(
            f"{type(self).__name__}.combine is abstract: a node family must implement "
            "the elementwise bias/response affine map of the aggregated (P, N) value"
        )

    # ------------------------------------------------------------------ #
    # host-only option-name resolution
    # ------------------------------------------------------------------ #

    def _aggregation_name(self, index: int) -> str:
        """Registry name of the aggregation option a node row selected.

        Args:
            index: Value of the ``aggregation`` column, i.e. an option index.

        Returns:
            The option's registry name, e.g. ``"sum"``.

        Raises:
            tensorneat.common.errors.ConfigError: If ``index`` is not a configured
                option index.
        """
        return self._option_name(AGG, self.aggregation_options, index, "aggregation")

    def _activation_name(self, index: int) -> str:
        """Registry name of the activation option a node row selected.

        Args:
            index: Value of the ``activation`` column: an option index, or ``-1``
                (:data:`~tensorneat.common.functions.act.ACT_IDENTITY`) for identity,
                which is a sentinel rather than an option and resolves to
                ``"identity"``.

        Returns:
            The option's registry name, e.g. ``"sigmoid"``.

        Raises:
            tensorneat.common.errors.ConfigError: If ``index`` is neither ``-1`` nor a
                configured option index.
        """
        if index == ACT_IDENTITY:
            return ACT.name_of(ACT.identity)
        return self._option_name(ACT, self.activation_options, index, "activation")

    @staticmethod
    def _option_name(
        manager: FunctionManager, options: Sequence[Callable], index: int, kind: str
    ) -> str:
        """Resolve one option id to the name ``repr``/``to_dict`` render.

        Args:
            manager: The ``ACT`` or ``AGG`` registry the option is named after.
            options: The configured option tuple of that kind.
            index: Option id stored in the gene row.
            kind: ``"aggregation"`` or ``"activation"``, for error messages.

        Returns:
            The registry name of ``options[index]``; a custom callable that is not in
            the global registry is named by its own ``__name__``.

        Raises:
            tensorneat.common.errors.ConfigError: If ``index`` is outside
                ``[0, len(options))``, or a custom option has no ``__name__`` to name
                it by.
        """
        if not 0 <= index < len(options):
            raise ConfigError(
                f"{kind} id {index} is outside [0, {len(options)}) of the configured "
                f"{kind} options"
            )
        option = options[index]
        try:
            return manager.name_of(option)
        except ValueError:
            # A custom option is a plain callable supplied by the genome, so it is not
            # in the global registry; its own ``__name__`` is the name to render.
            name = getattr(option, "__name__", None)
            if name is None:
                raise ConfigError(
                    f"{kind} option {option!r} is not registered in {manager!r} and "
                    "has no __name__ to be named by"
                ) from None
            return name

    # ------------------------------------------------------------------ #
    # host-only presentation
    # ------------------------------------------------------------------ #

    def repr(
        self,
        state: State,
        key_row: evoxc.SymbolicTensor,
        attr_row: evoxc.SymbolicTensor,
        precision: int = 2,
    ) -> str:
        """Render one node gene as a human-readable string (host-only, never traced).

        Node genes share the generic attr-table rendering; the only node-specific part
        is the column label (``index`` -> ``idx``) declared in :attr:`labels`, so this
        only delegates. Families with named option indices
        (:class:`~.default.DefaultNode`, :class:`~.bias.BiasNode`) override it to
        resolve the aggregation/activation ids to registry names.

        Args:
            state: Host configuration container.
            key_row: ``(key_width,)`` int32 key columns of a single node gene; the
                node key lives in key column 0.
            attr_row: ``(attr_width,)`` float32 attribute columns of the same gene.
            precision: Decimal places for the float attributes.

        Returns:
            A single-line description such as ``DefaultNode(idx: 3, bias: 0.25)``.
        """
        return super().repr(state, key_row, attr_row, precision)

    def to_dict(
        self,
        state: State,
        key_row: evoxc.SymbolicTensor,
        attr_row: evoxc.SymbolicTensor,
    ) -> dict[str, Any]:
        """Convert one node gene to a plain dict for host code generation.

        The generic conversion maps the key column to ``idx`` (through
        :attr:`labels`) and every attribute column to its declared name, leaving the
        option indices raw. Families whose option indices have registry names
        override it to resolve them.

        Args:
            state: Host configuration container.
            key_row: ``(key_width,)`` int32 key columns of a single node gene.
            attr_row: ``(attr_width,)`` float32 attribute columns of the same gene.

        Returns:
            Mapping such as ``{"idx": 3, "bias": 0.25, "aggregation": 0,
            "activation": 0}``; the family override replaces the raw ids with the
            registered function names (``{"idx": 3, "bias": 0.25, "agg": "sum",
            "act": "sigmoid"}``).
        """
        return super().to_dict(state, key_row, attr_row)

    def sympy_func(
        self, state: State, node_dict: dict, inputs: Any, is_output_node: bool = False
    ) -> tuple[Any, dict]:
        """Build the sympy expression of one node from its ``to_dict`` output.

        Args:
            state: Host configuration container.
            node_dict: Output of :meth:`to_dict` for this node.
            inputs: The incoming edge contributions of this node -- an iterable of
                sympy expressions, or a single expression treated as a one-term list;
                it is the *operand* of the aggregation, never an aggregated value.
            is_output_node: True to leave the expression unactivated, matching
                :meth:`forward`.

        Returns:
            Tuple ``(expression, {symbol: value})``: the node's sympy expression and
            the symbol-to-attribute map used to substitute concrete gene values when
            generating Python code.

        Raises:
            NotImplementedError: Abstract hook -- the base class declares no attribute
                table and no aggregation/activation names, so it cannot build the
                formula. Every node family implements it.
        """
        raise NotImplementedError(
            f"{type(self).__name__}.sympy_func is abstract: a node family must build "
            "combine(agg(inputs)) as a sympy expression plus its {symbol: value} map"
        )
