"""Bias node gene: ``DefaultNode`` without the ``response`` column.

Batch-first like every gene kernel: one call covers the whole population and the
option indices of the attribute rows are resolved with ``select`` chains.
This file declares the response-free attribute table and the ``bias + z`` combine;
it must not import ``jax``/``torch`` and must not duplicate the aggregation or
activation machinery of :class:`.default.DefaultNode`.
"""

from typing import Callable, Sequence

import evoxc
import sympy as sp

from ....common import ACT, AGG, State
from ..base import GeneAttr
from .default import DefaultNode


class BiasNode(DefaultNode):
    """Node gene without ``response``: ``bias + aggregated``, then activation.

    Three float32 attribute columns, in this order: ``bias``, ``aggregation`` (int
    option index), ``activation`` (int option index); the aggregation, activation and
    option-table machinery is inherited unchanged from :class:`.default.DefaultNode`,
    which is also where the constructor knobs and the ``_option_list`` /
    ``_default_index`` / ``_option_spec`` declarations come from.

    Identity attributes use ``activation_default`` -- the configured option index --
    and never ``-1``: the legacy ``lax.switch(-1, ...)`` clamped to the last option,
    so a ``BiasNode`` node split silently switched to the wrong activation. In this
    rewrite ``-1`` is a first-class *identity* index handled by
    ``apply_activation``, which is why the identity row must name a real option.
    """

    attr_specs: tuple[GeneAttr, ...] = (
        GeneAttr(
            name="bias",
            kind="float",
            init_mean=0.0,
            init_std=1.0,
            mutate_power=0.15,
            mutate_rate=0.2,
            replace_rate=0.015,
            lower_bound=-5.0,
            upper_bound=5.0,
        ),
        GeneAttr(
            name="aggregation",
            kind="int",
            replace_rate=0.1,
            options=(0,),
        ),
        GeneAttr(
            name="activation",
            kind="int",
            replace_rate=0.1,
            options=(0,),
        ),
    )

    def __init__(
        self,
        bias_init_mean: float = 0.0,
        bias_init_std: float = 1.0,
        bias_mutate_power: float = 0.15,
        bias_mutate_rate: float = 0.2,
        bias_replace_rate: float = 0.015,
        bias_lower_bound: float = -5,
        bias_upper_bound: float = 5,
        aggregation_default: Callable | None = None,
        aggregation_options: Callable | Sequence[Callable] = AGG.sum,
        aggregation_replace_rate: float = 0.1,
        activation_default: Callable | None = None,
        activation_options: Callable | Sequence[Callable] = ACT.sigmoid,
        activation_replace_rate: float = 0.1,
    ):
        """Configure the response-free attribute table (host-side, called once).

        The response-carrying ``DefaultNode`` constructor is deliberately not chained:
        this family has no response knob, so storing one would advertise a
        configuration value that never reaches the payload.

        Args:
            bias_init_mean: Mean of the initial bias draw.
            bias_init_std: Standard deviation of the initial bias draw.
            bias_mutate_power: Standard deviation of the bias perturbation.
            bias_mutate_rate: Probability that the bias is perturbed.
            bias_replace_rate: Probability that the bias is re-drawn.
            bias_lower_bound: Inclusive lower clip bound of the bias.
            bias_upper_bound: Inclusive upper clip bound of the bias.
            aggregation_default: Aggregation of the identity attribute row; ``None``
                means option 0 of ``aggregation_options``.
            aggregation_options: One callable or a sequence of callables; the index of
                a callable in this list is the value stored in the ``aggregation``
                column.
            aggregation_replace_rate: Probability that the aggregation index is
                re-drawn uniformly over ``aggregation_options``.
            activation_default: Activation of the identity attribute row; ``None``
                means option 0 of ``activation_options``. Never ``-1`` (identity is
                not an option index).
            activation_options: One callable or a sequence of callables.
            activation_replace_rate: Probability that the activation index is
                re-drawn uniformly over ``activation_options``.

        Raises:
            tensorneat.common.errors.ConfigError: If an option knob is an empty
                sequence, holds a non-callable, or a ``*_default`` is not one of the
                configured options.
        """
        self.bias_init_mean = bias_init_mean
        self.bias_init_std = bias_init_std
        self.bias_mutate_power = bias_mutate_power
        self.bias_mutate_rate = bias_mutate_rate
        self.bias_replace_rate = bias_replace_rate
        self.bias_lower_bound = bias_lower_bound
        self.bias_upper_bound = bias_upper_bound

        self.aggregation_options = self._option_list(
            aggregation_options, "aggregation_options"
        )
        self.aggregation_default = self._default_index(
            aggregation_default, self.aggregation_options, "aggregation_default"
        )
        self.activation_options = self._option_list(
            activation_options, "activation_options"
        )
        self.activation_default = self._default_index(
            activation_default, self.activation_options, "activation_default"
        )

        # Column order IS the payload layout: bias, aggregation, activation.
        self.attr_specs = (
            GeneAttr(
                name="bias",
                kind="float",
                init_mean=bias_init_mean,
                init_std=bias_init_std,
                mutate_power=bias_mutate_power,
                mutate_rate=bias_mutate_rate,
                replace_rate=bias_replace_rate,
                lower_bound=bias_lower_bound,
                upper_bound=bias_upper_bound,
            ),
            self._option_spec(
                "aggregation", self.aggregation_options, aggregation_replace_rate
            ),
            self._option_spec(
                "activation", self.activation_options, activation_replace_rate
            ),
        )

    def new_identity_attrs(self, state: State) -> evoxc.SymbolicTensor:
        """Identity attribute row: ``bias=0``, default aggregation, default activation.

        Args:
            state: Host configuration container.

        Returns:
            ``(3,)`` float32 row ``(bias, aggregation_default, activation_default)``,
            with the bias clipped into its bounds; the activation stays on a real
            option index -- ``-1`` is not used here, because a split node must keep the
            activation it had before the split.
        """
        return self._attr_row(
            (
                self._clip_scalar(self.attr_specs[self.attr_index("bias")], 0.0),
                float(self.aggregation_default),
                float(self.activation_default),
            )
        )

    def combine(
        self,
        state: State,
        attrs: evoxc.SymbolicTensor,
        aggregated: evoxc.SymbolicTensor,
    ) -> evoxc.SymbolicTensor:
        """Compute the pre-activation value ``bias + aggregated``.

        Args:
            state: Host configuration container.
            attrs: Node attributes, ``(P, N, 3)`` float32; reads the ``bias`` column
                by its ``attr_specs`` position.
            aggregated: ``(P, N)`` float32 output of
                :meth:`~.base.BaseNode.aggregate`.

        Returns:
            ``(P, N)`` float32 pre-activation values.
        """
        return evoxc.add(self.attr_column(attrs, "bias"), aggregated)

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
        """Render one node gene as ``idx / bias / aggregation / activation``.

        Args:
            state: Host configuration container.
            key_row: ``(key_width,)`` int32 key columns; the node key is column 0.
            attr_row: ``(3,)`` float32 attribute columns of the same gene.
            precision: Decimal places for ``bias``.

        Returns:
            A single-line description, e.g.
            ``BiasNode(idx=3, bias=0.25, aggregation=sum, activation=sigmoid)``.
        """
        node = self.to_dict(state, key_row, attr_row)
        return (
            f"{type(self).__name__}(idx={node['idx']}, "
            f"bias={round(node['bias'], precision)}, "
            f"aggregation={node['agg']}, activation={node['act']})"
        )

    def to_dict(
        self,
        state: State,
        key_row: evoxc.SymbolicTensor,
        attr_row: evoxc.SymbolicTensor,
    ) -> dict:
        """Convert one node gene to the dict used by host code generation.

        Args:
            state: Host configuration container.
            key_row: ``(key_width,)`` int32 key columns; the node key is column 0.
            attr_row: ``(3,)`` float32 attribute columns of the same gene.

        Returns:
            ``{"idx": int, "bias": float, "agg": str, "act": str}`` -- no ``res`` key,
            unlike :meth:`.default.DefaultNode.to_dict`; ``act`` resolves ``-1`` to
            ``"identity"``.

        Raises:
            tensorneat.common.errors.ConfigError: If an option column holds an id that
                is not a configured option (see
                :meth:`~.base.BaseNode._option_name`).
        """
        keys, attrs = self._host_row(key_row, attr_row)
        return {
            "idx": keys[self.key_index("index")],
            "bias": attrs[self.attr_index("bias")],
            "agg": self._aggregation_name(
                int(round(attrs[self.attr_index("aggregation")]))
            ),
            "act": self._activation_name(
                int(round(attrs[self.attr_index("activation")]))
            ),
        }

    def sympy_func(
        self, state: State, node_dict: dict, inputs: sp.Expr, is_output_node: bool = False
    ) -> tuple[sp.Expr, dict]:
        """Build the sympy expression of one node from its ``to_dict`` output.

        Args:
            state: Host configuration container.
            node_dict: Output of :meth:`to_dict` for this node.
            inputs: The incoming edge contributions (an iterable of sympy expressions,
                or a single expression treated as a one-term list); it is the operand
                of the aggregation.
            is_output_node: True to leave the expression unactivated, matching
                :meth:`~.base.BaseNode.forward`.

        Returns:
            ``(expression, {symbol: value})`` where the expression is
            ``bias + agg(inputs)`` (plus the activation unless the node is an output
            node) and the map binds the ``n_<idx>_b`` symbol to this node's bias.
        """
        terms = list(inputs) if isinstance(inputs, (list, tuple)) else [inputs]
        bias = sp.symbols(f"n_{node_dict['idx']}_b")

        z = AGG.obtain_sympy(node_dict["agg"])(terms)
        z = bias + z
        if not is_output_node:
            z = ACT.obtain_sympy(node_dict["act"])(z)

        return z, {bias: node_dict["bias"]}
