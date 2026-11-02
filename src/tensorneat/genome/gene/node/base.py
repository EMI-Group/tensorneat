"""Node gene base class: one node's aggregated window -> that node's value.

Every kernel here is batch-first (population on axis 0) and elementwise over the
node axis: no per-node Python loop, no ``vmap``/``scan``/``cond``, and no branching
on traced values -- ``select`` chains only.
This file owns the forward contract, the option-independent aggregation, the
loop-invariant column bundle (:class:`NodeOperands`) and the host-side repr/dict
scaffolding; the attr tables and the concrete combine algebra belong to the
families.
"""

from typing import Any, Callable, NamedTuple, Sequence

import evoxc
import evoxc.numpy as enp

from ....common import ACT, AGG, State
from ....common.errors import ConfigError
from ....common.functions import FunctionManager
from ....common.functions.act import ACT_IDENTITY
from ..base import BaseGene, column, option_masks, select_cascade
from ...window import any_slot


class NodeOperands(NamedTuple):
    """Loop-invariant view of one ``(P, N, NA)`` node-attribute batch.

    ``_node_step`` calls the gene once per unrolled level/tick with the *same* attribute
    tensor, so every per-column read (a ``slice`` plus the ``reshape`` that makes it
    differentiable, i.e. two traced ops each), the option-index casts and the option
    dispatch masks would otherwise be re-emitted on every level. This bundle is built
    once per forward pass and then read by :meth:`BaseNode.forward` /
    :meth:`BaseNode.aggregate` / ``combine`` of the families, which turn ``~10``
    per-level ops into zero.

    Attributes:
        columns: One ``(P, N) float32`` tensor per ``attr_specs`` column, in table order
            (so ``columns[i]`` is the column ``attr_index(attr_specs[i].name)`` names).
        aggregation_ids: ``(P, N) int32`` option ids of the ``aggregation`` column.
        activation_ids: ``(P, N) int32`` option ids of the ``activation`` column with
            :data:`~tensorneat.common.functions.act.ACT_IDENTITY` written on the output
            rows, which is what lets :meth:`BaseNode.forward` skip its
            ``select(is_output_node, z, activated)``: an id of ``-1`` matches no option,
            so the dispatch cascade returns ``z`` itself there.
        aggregation_masks: The ``ids == index`` masks of ``aggregation_ids``, one per
            configured aggregation option (see
            :func:`~tensorneat.genome.gene.base.option_masks`), so no level re-derives
            them.
        activation_masks: The same masks for ``activation_ids``.
        zero: Rank-0 ``float32`` zero, shared by the dispatch of every level (the
            cascade's "no option matched" value) instead of materialising one constant
            per level.
    """

    columns: tuple
    aggregation_ids: Any
    activation_ids: Any
    aggregation_masks: tuple
    activation_masks: tuple
    zero: Any


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
    #: Aggregations whose reduce of an all-dead window is already exactly ``0.0``: the
    #: masked window of :func:`~tensorneat.common.functions.agg.sum` sums to ``0.0``, and
    #: :func:`~tensorneat.common.functions.agg.mean` divides that same sum by
    #: ``maximum(count, 1)``, so the ``has_valid`` guard of ``apply_aggregation`` would
    #: return the value it already has and :meth:`aggregate` skips it.
    #: The comparison is by identity, so a wrapper callable is not recognised and simply
    #: keeps the guard -- the conservative direction.
    self_neutral_aggregations: tuple = (AGG.sum, AGG.mean)

    def setup(self, state: State | None = None) -> State:
        """Validate the attr table and the option lists on the host.

        Chains :meth:`..base.BaseGene.setup` (which checks the attr table) and then
        checks that the two option lists a node gene dispatches on are non-empty:
        an empty list would make the aggregation / activation cascade return
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

    @property
    def needs_has_valid(self) -> bool:
        """Whether :meth:`aggregate` must emit the ``has_valid`` guard (host-only).

        The guard of ``apply_aggregation`` rewrites a node with no real incoming edge to
        ``0.0``; it is only observable for an option whose all-dead-window reduction is
        not already ``0`` (``max``/``min`` reduce to ``+/-inf``, ``product`` to ``1``),
        and it is emitted for the whole cascade as soon as one configured option needs it.

        Returns:
            ``False`` iff every configured option is in
            :attr:`self_neutral_aggregations`, i.e. an empty window already reduces to
            ``0.0`` for every option the gene can select; ``True`` otherwise (including
            every custom, unrecognised kernel).
        """
        return any(
            option not in self.self_neutral_aggregations
            for option in self.aggregation_options
        )

    def forward(
        self,
        state: State,
        attrs: evoxc.SymbolicTensor,
        messages: evoxc.SymbolicTensor,
        is_output_node: evoxc.SymbolicTensor,
        valid_mask: evoxc.SymbolicTensor,
    ) -> evoxc.SymbolicTensor:
        """Compute the value of every node row of the population in one pass.

        The caller feeds one depth level at a time: ``messages`` is the gathered incoming
        window of the nodes of that level, already masked by the caller (a dead slot may
        carry ``NaN``; ``valid_mask`` is what marks it dead, and every aggregation option
        replaces the slots of that mask with its neutral value before reducing).

        ``attrs`` may be the raw ``(P, N, NA)`` attribute tensor or the
        :class:`NodeOperands` bundle of :meth:`prepare`. The bundle is what the forward
        pass uses: it carries the per-column reads, the option ids and the option
        dispatch masks of the whole unrolled pass, so this kernel emits no attribute read
        and no ``equal`` mask per level. Output rows are left unactivated by writing
        :data:`~tensorneat.common.functions.act.ACT_IDENTITY` into their activation id
        inside the bundle (an id of ``-1`` matches no option, so the dispatcher returns
        ``z`` unchanged), which is why no
        ``select(is_output_node, z, activated)`` is emitted here.

        Args:
            state: Host configuration container (read-only during a trace).
            attrs: Node attributes, ``(P, N, NA)`` float32, one row per node slot, or the
                :class:`NodeOperands` of :meth:`prepare` for those attributes.
            messages: One scalar message per incoming edge slot, ``(P, N, K)``
                float32; a dead slot may carry ``NaN`` — ``valid_mask`` marks it dead and
                the aggregation neutralises it before the reduce.
            is_output_node: ``(P, N)`` bool, True for the rows read out as network
                outputs, which must stay unactivated here.
            valid_mask: ``(P, N, K)`` bool, True where the slot holds a real, live
                connection; dead slots must not contribute to the aggregation.

        Returns:
            ``(P, N)`` float32 node values: ``combine(attrs, aggregate(...))`` with the
            activation applied only where ``is_output_node`` is False. A family with no
            attribute column (``attr_width == 0``) has no option index to dispatch on, so
            it aggregates with its first configured option and applies no activation --
            the neutral node value (see :meth:`prepare`).
        """
        if not isinstance(attrs, NodeOperands):
            attrs = self.prepare(attrs, is_output_node)
        aggregated = self.aggregate(state, attrs, messages, valid_mask)
        z = self.combine(state, attrs, aggregated)
        return select_cascade(
            attrs.activation_masks,
            [fn(z) for fn in self.activation_options],
            z,
        )

    def prepare(
        self,
        attrs: evoxc.SymbolicTensor,
        is_output_node: evoxc.SymbolicTensor,
    ) -> NodeOperands:
        """Read every attribute column and resolve the option dispatch once per forward pass.

        Loop-invariant work hoisted out of the unrolled level/tick loop: the columns of
        ``node_attrs`` do not change while a forward pass runs, so the per-column
        ``slice`` + ``reshape`` reads, the option-id casts and the per-option
        ``ids == index`` dispatch masks belong here, not in :meth:`forward` /
        :meth:`aggregate`.

        ``state`` is deliberately not an argument: the reads are pure column arithmetic
        (like :meth:`..base.BaseGene.attr_column`), and the one production caller lives in
        ``BaseGenome._node_step``, which has the state but does not need to pass it.

        Args:
            attrs: Node attributes, ``(P, N, NA)`` float32, one row per node slot.
            is_output_node: ``(P, N)`` bool, True on the rows read out as network outputs;
                their activation id is forced to
                :data:`~tensorneat.common.functions.act.ACT_IDENTITY` so the activation
                dispatch leaves them untouched.

        Returns:
            The :class:`NodeOperands` of ``attrs``.

        Raises:
            tensorneat.common.errors.ConfigError: If the attribute width of ``attrs`` does
                not match the declared table.

        Note:
            A gene that declares no attribute column (``attr_width == 0``, so ``attrs`` is
            ``(P, N, 0)``) cannot carry an option index either. The dispatch then falls
            back to the neutral configuration: the **first** configured aggregation option
            (``AGG.sum`` in the shipped families, the neutral reduction of a message
            window) and :data:`~tensorneat.common.functions.act.ACT_IDENTITY` for the
            activation, because "no activation column" must not silently apply an
            arbitrary option. Both are static ids, so the cascade still costs one
            ``select`` per option and no branch on traced values.
        """
        width = int(attrs.shape[-1])
        if width != self.attr_width:
            raise ConfigError(
                f"{type(self).__name__} reads {self.attr_width} attribute column(s) but the "
                f"payload carries {width}"
            )
        columns = tuple(column(attrs, index) for index in range(width))
        aggregation_ids = self._option_ids(columns, "aggregation", is_output_node, 0)
        activation_ids = self._option_ids(
            columns, "activation", is_output_node, ACT_IDENTITY
        )
        if self.has_attr("activation"):
            activation_ids = evoxc.select(
                is_output_node,
                enp.full((), ACT_IDENTITY, dtype=evoxc.int32),
                activation_ids,
            )
        return NodeOperands(
            columns,
            aggregation_ids,
            activation_ids,
            option_masks(aggregation_ids, len(self.aggregation_options)),
            option_masks(activation_ids, len(self.activation_options)),
            enp.zeros((), dtype=evoxc.float32),
        )

    def _option_ids(
        self,
        columns: Sequence[evoxc.SymbolicTensor],
        name: str,
        shape_source: evoxc.SymbolicTensor,
        default: int,
    ) -> evoxc.SymbolicTensor:
        """Read one option-index column, or a constant id when the table declares none.

        Args:
            columns: The ``(P, N)`` attribute columns of :meth:`prepare`, in table order.
            name: Option-column name (``aggregation`` / ``activation``).
            shape_source: ``(P, N)`` tensor whose shape the fallback constant takes (the
                output-row mask of the pass).
            default: Id written into that constant when the column is absent.

        Returns:
            ``(P, N)`` int32 option ids.
        """
        if not self.has_attr(name):
            return enp.full(
                tuple(shape_source.shape), int(default), dtype=evoxc.int32
            )
        return evoxc.cast(columns[self.attr_index(name)], evoxc.int32)

    def column_of(
        self, attrs: Any, name: str, default: float | None = None
    ) -> evoxc.SymbolicTensor:
        """Read one named attribute column of a raw batch or of a prepared bundle.

        Args:
            attrs: ``(P, N, NA)`` float32 attribute rows, or the :class:`NodeOperands`
                bundle of :meth:`prepare`.
            name: Spec name of the column, resolved through ``attr_specs``.
            default: Value of the whole column when the table declares no such
                attribute (a family with ``attr_width == 0``). A Python float is a
                scalar operand of every evoxc op, so it broadcasts over ``(P, N)`` at
                no traced cost; ``None`` keeps the strict behaviour and raises.

        Returns:
            ``(P, N)`` float32 column values, or the ``default`` scalar when the
            column is absent.

        Raises:
            tensorneat.common.errors.ConfigError: If the table declares no such column
                and no ``default`` was given.
        """
        if not self.has_attr(name):
            if default is None:
                raise ConfigError(
                    f"{type(self).__name__} declares no attribute {name!r}"
                )
            return float(default)
        if isinstance(attrs, NodeOperands):
            return attrs.columns[self.attr_index(name)]
        return self.attr_column(attrs, name)

    def aggregate(
        self,
        state: State,
        attrs: evoxc.SymbolicTensor,
        messages: evoxc.SymbolicTensor,
        valid_mask: evoxc.SymbolicTensor,
    ) -> evoxc.SymbolicTensor:
        """Reduce the incoming-message window of every node with its own option.

        The reduction itself is option-independent and therefore implemented here
        once: every configured aggregation is evaluated on the whole ``(P, N, K)``
        window -- the kernels replace the dead slots of ``valid_mask`` with that
        option's neutral element (``0``, ``1``, ``+/-inf``) before reducing -- and then
        the row's option is picked with a ``select`` cascade
        (:func:`~tensorneat.genome.gene.base.select_cascade`). A node whose window has no
        live slot gets exactly ``0.0`` from the ``has_valid`` guard, never ``NaN`` and
        never a neutral that leaks into the value. Families inherit this method
        unchanged (a family that needs a different operator over the window would
        override it *without* copying the cascade).

        This is :func:`~tensorneat.common.functions.apply_aggregation` with the
        loop-invariant half hoisted: the ``ids`` cast and the per-option ``ids == index``
        masks are built once per forward pass by :meth:`prepare`, not once per level, and
        the branch kernels are the configured option callables themselves.

        Args:
            state: Host configuration container.
            attrs: Node attributes, ``(P, N, NA)`` float32, or the :class:`NodeOperands`
                bundle of :meth:`prepare`; the ``aggregation`` index column selects the
                branch per node row.
            messages: ``(P, N, K)`` float32, already NaN-sanitized.
            valid_mask: ``(P, N, K)`` bool, True for live slots.

        Returns:
            ``(P, N)`` float32 aggregated values, one per node row.

        Note:
            A family with no ``aggregation`` column (``attr_width == 0``) aggregates with
            its first configured option, exactly as :meth:`prepare` resolves the dispatch.
        """
        if isinstance(attrs, NodeOperands):
            masks, zero = attrs.aggregation_masks, attrs.zero
        else:
            if self.has_attr("aggregation"):
                ids = evoxc.cast(self.attr_column(attrs, "aggregation"), evoxc.int32)
            else:
                ids = enp.full(
                    tuple(messages.shape[:-1]), 0, dtype=evoxc.int32
                )
            masks = option_masks(ids, len(self.aggregation_options))
            zero = enp.zeros((), dtype=evoxc.float32)
        aggregated = select_cascade(
            masks,
            [fn(messages, valid_mask) for fn in self.aggregation_options],
            zero,
        )
        if not self.needs_has_valid:
            # Every configured option already reduces an all-dead window to exactly
            # ``0.0``, so the guard below would select the value it is replacing.
            return aggregated
        # ``has_valid``, as in ``apply_aggregation``: a node with no real incoming edge gets
        # ``0.0``, so the ``-inf`` / ``+inf`` neutral padding of ``max`` / ``min`` cannot
        # leak into a node value (and a padded product cannot become ``1``). ``valid_mask``
        # is ``bool``, so its boolean "any" is the same predicate as the dispatcher's
        # ``count > 0`` at two ops instead of four, and it is folded slice-wise
        # (:func:`~tensorneat.genome.window.any_slot`) because a reduction over the
        # innermost window axis is the expensive shape on the numpy backend.
        return evoxc.select(any_slot(valid_mask), aggregated, zero)

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
            attrs: Node attributes, ``(P, N, NA)`` float32, or the :class:`NodeOperands`
                bundle of :meth:`prepare`; a family reads its columns with
                :meth:`column_of`, which accepts either spelling.
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

    def _host_column(
        self, attrs: Sequence[float], name: str, default: float
    ) -> float:
        """Read one attribute column of a *host* attribute row, or ``default``.

        The host-side mirror of :meth:`column_of`: ``to_dict``/``repr`` of a family
        with no attribute column (``attr_width == 0``) must still render the neutral
        row its forward pass computes, not fail on the missing column.

        Args:
            attrs: Attribute values of one gene, in ``attr_specs`` order (host floats).
            name: Attribute-column name.
            default: Value to render when the table declares no such column.

        Returns:
            The column's value, or ``default`` when the column is absent.
        """
        if not self.has_attr(name):
            return float(default)
        return float(attrs[self.attr_index(name)])

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
