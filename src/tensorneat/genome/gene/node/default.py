"""Default node gene: NEAT-python's ``bias / response / aggregation / activation`` row.

Batch-first: one call handles the whole population, and the option indices stored in
the attribute rows are resolved with ``select`` chains, never with branching on
traced values.
This file declares the attribute table and the family hooks; it must not import
``jax``/``torch``, must not run eager NumPy on the compute path, and must not
duplicate the table machinery of :class:`..base.BaseGene`.
"""

from typing import Callable, Sequence

import evoxc
import sympy as sp

from ....common import ACT, AGG, State
from ....common.errors import ConfigError
from ..base import GeneAttr
from .base import BaseNode


class DefaultNode(BaseNode):
    """Default node gene with the same behaviour as NEAT-python.

    Four float32 attribute columns, in this order: ``bias``, ``response``,
    ``aggregation`` (int option index), ``activation`` (int option index).
    ``combine`` is ``bias + response * agg``; the aggregation and the activation are
    dispatched by the inherited :meth:`~.base.BaseNode.aggregate` /
    :meth:`~.base.BaseNode.forward`, and the activation is applied only where the node
    is not an output node.

    The option lists are materialised into ``attr_specs`` at construction (an
    instance attribute rebuilt from the configured ``aggregation_options`` /
    ``activation_options``), so every option index of every kernel is a static
    Python int; the class-level table below records the default single-option
    layout used when the constructor knobs are left at their defaults.
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
            name="response",
            kind="float",
            init_mean=1.0,
            init_std=0.0,
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
        response_init_mean: float = 1.0,
        response_init_std: float = 0.0,
        response_mutate_power: float = 0.15,
        response_mutate_rate: float = 0.2,
        response_replace_rate: float = 0.015,
        response_lower_bound: float = -5,
        response_upper_bound: float = 5,
        aggregation_default: Callable | None = None,
        aggregation_options: Callable | Sequence[Callable] = AGG.sum,
        aggregation_replace_rate: float = 0.1,
        activation_default: Callable | None = None,
        activation_options: Callable | Sequence[Callable] = ACT.sigmoid,
        activation_replace_rate: float = 0.1,
    ):
        """Configure the gene's attribute table (host-side, called once).

        Args:
            bias_init_mean: Mean of the initial bias draw.
            bias_init_std: Standard deviation of the initial bias draw.
            bias_mutate_power: Standard deviation of the bias perturbation.
            bias_mutate_rate: Probability that the bias is perturbed.
            bias_replace_rate: Probability that the bias is re-drawn.
            bias_lower_bound: Inclusive lower clip bound of the bias.
            bias_upper_bound: Inclusive upper clip bound of the bias.
            response_init_mean: Mean of the initial response draw.
            response_init_std: Standard deviation of the initial response draw.
            response_mutate_power: Standard deviation of the response perturbation.
            response_mutate_rate: Probability that the response is perturbed.
            response_replace_rate: Probability that the response is re-drawn.
            response_lower_bound: Inclusive lower clip bound of the response.
            response_upper_bound: Inclusive upper clip bound of the response.
            aggregation_default: Aggregation of the identity attribute row;
                ``None`` means option 0 of ``aggregation_options``.
            aggregation_options: One callable or a sequence of callables; a bare
                callable is promoted to a one-element tuple. The index of a callable
                in this list is the value stored in the ``aggregation`` column.
            aggregation_replace_rate: Probability that the aggregation index is
                re-drawn uniformly over ``aggregation_options``.
            activation_default: Activation of the identity attribute row; ``None``
                means option 0 of ``activation_options``. ``-1`` is not an option
                index (it means identity) and must not be passed here.
            activation_options: One callable or a sequence of callables, promoted
                the same way as ``aggregation_options``.
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

        self.response_init_mean = response_init_mean
        self.response_init_std = response_init_std
        self.response_mutate_power = response_mutate_power
        self.response_mutate_rate = response_mutate_rate
        self.response_replace_rate = response_replace_rate
        self.response_lower_bound = response_lower_bound
        self.response_upper_bound = response_upper_bound

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

        # Column order IS the payload layout: bias, response, aggregation, activation.
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
            GeneAttr(
                name="response",
                kind="float",
                init_mean=response_init_mean,
                init_std=response_init_std,
                mutate_power=response_mutate_power,
                mutate_rate=response_mutate_rate,
                replace_rate=response_replace_rate,
                lower_bound=response_lower_bound,
                upper_bound=response_upper_bound,
            ),
            self._option_spec(
                "aggregation", self.aggregation_options, aggregation_replace_rate
            ),
            self._option_spec(
                "activation", self.activation_options, activation_replace_rate
            ),
        )

    # ------------------------------------------------------------------ #
    # option-table declarations shared by the node families
    # ------------------------------------------------------------------ #

    @staticmethod
    def _option_list(
        options: Callable | Sequence[Callable], name: str
    ) -> tuple[Callable, ...]:
        """Normalise an option knob to a non-empty tuple of callables.

        Args:
            options: One callable (promoted to a one-element tuple) or a sequence of
                callables; its order fixes the option indices of the gene column.
            name: Knob name for error messages.

        Returns:
            The option tuple, in configuration order.

        Raises:
            tensorneat.common.errors.ConfigError: If the sequence is empty or holds a
                non-callable entry.
        """
        resolved = (options,) if callable(options) else tuple(options)
        if not resolved:
            raise ConfigError(
                f"{name} must hold at least one option callable, got an empty sequence"
            )
        for option in resolved:
            if not callable(option):
                raise ConfigError(f"{name} holds the non-callable option {option!r}")
        return resolved

    @staticmethod
    def _default_index(
        default: Callable | None, options: Sequence[Callable], name: str
    ) -> int:
        """Resolve a ``*_default`` knob to its index in the option tuple.

        Args:
            default: Configured default callable, or ``None`` for option 0.
            options: The resolved option tuple of that kind.
            name: Knob name for error messages.

        Returns:
            The option index stored in the identity attribute row.

        Raises:
            tensorneat.common.errors.ConfigError: If ``default`` is not one of
                ``options`` (identity comparison -- a wrapper is not recognised).
        """
        if default is None:
            return 0
        try:
            return options.index(default)
        except ValueError:
            raise ConfigError(
                f"{name} {default!r} is not one of the configured options {options!r}"
            ) from None

    @staticmethod
    def _option_spec(
        name: str, options: Sequence[Callable], replace_rate: float
    ) -> GeneAttr:
        """Build the ``int`` option-index column of a configured option list.

        Args:
            name: Column name (``aggregation`` / ``activation``).
            options: The resolved option tuple; its length bounds the column.
            replace_rate: Probability that mutation re-draws the index.

        Returns:
            A ``GeneAttr`` whose ``options`` are the contiguous indices
            ``0..len(options)-1``.
        """
        return GeneAttr(
            name=name,
            kind="int",
            replace_rate=replace_rate,
            options=tuple(range(len(options))),
        )

    # ------------------------------------------------------------------ #
    # genes
    # ------------------------------------------------------------------ #

    def new_identity_attrs(self, state: State) -> evoxc.SymbolicTensor:
        """Identity attribute row of a node split: ``bias=0``, ``response=1``.

        The option columns keep the configured defaults, so a node inserted into an
        existing connection contributes ``1 * aggregated + 0`` and leaves the network
        output unchanged.

        Args:
            state: Host configuration container.

        Returns:
            ``(4,)`` float32 row ``(bias, response, aggregation_default,
            activation_default)``, with the two floats clipped into their bounds.
        """
        return self._attr_row(
            (
                self._clip_scalar(self.attr_specs[self.attr_index("bias")], 0.0),
                self._clip_scalar(self.attr_specs[self.attr_index("response")], 1.0),
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
        """Compute the pre-activation value ``bias + response * aggregated``.

        Args:
            state: Host configuration container.
            attrs: Node attributes, ``(P, N, 4)`` float32; reads the ``bias`` and
                ``response`` columns by their ``attr_specs`` position.
            aggregated: ``(P, N)`` float32 output of
                :meth:`~.base.BaseNode.aggregate`.

        Returns:
            ``(P, N)`` float32 pre-activation values.
        """
        bias = self.attr_column(attrs, "bias")
        response = self.attr_column(attrs, "response")
        return evoxc.add(bias, evoxc.multiply(response, aggregated))

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
        """Render one node gene as ``idx / bias / response / aggregation / activation``.

        Args:
            state: Host configuration container.
            key_row: ``(key_width,)`` int32 key columns; the node key is column 0.
            attr_row: ``(4,)`` float32 attribute columns of the same gene.
            precision: Decimal places for ``bias`` and ``response``.

        Returns:
            A single-line description, e.g.
            ``DefaultNode(idx=3, bias=0.25, response=1.0, aggregation=sum, activation=sigmoid)``.
        """
        node = self.to_dict(state, key_row, attr_row)
        return (
            f"{type(self).__name__}(idx={node['idx']}, "
            f"bias={round(node['bias'], precision)}, "
            f"response={round(node['res'], precision)}, "
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
            attr_row: ``(4,)`` float32 attribute columns of the same gene.

        Returns:
            ``{"idx": int, "bias": float, "res": float, "agg": str, "act": str}`` with
            the option indices resolved to registered function names; ``act`` resolves
            an index of ``-1`` to ``"identity"``.

        Raises:
            tensorneat.common.errors.ConfigError: If an option column holds an id that
                is not a configured option (see
                :meth:`~.base.BaseNode._option_name`).
        """
        keys, attrs = self._host_row(key_row, attr_row)
        return {
            "idx": keys[self.key_index("index")],
            "bias": attrs[self.attr_index("bias")],
            "res": attrs[self.attr_index("response")],
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
            ``bias + response * agg(inputs)`` (plus the activation unless the node is
            an output node) and the map binds the ``n_<idx>_b`` / ``n_<idx>_r`` symbols
            to this node's attribute values.
        """
        terms = list(inputs) if isinstance(inputs, (list, tuple)) else [inputs]
        bias = sp.symbols(f"n_{node_dict['idx']}_b")
        response = sp.symbols(f"n_{node_dict['idx']}_r")

        z = AGG.obtain_sympy(node_dict["agg"])(terms)
        z = bias + response * z
        if not is_output_node:
            z = ACT.obtain_sympy(node_dict["act"])(z)

        return z, {bias: node_dict["bias"], response: node_dict["res"]}
