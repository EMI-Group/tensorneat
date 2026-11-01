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

from ....common import State
from ....common.functions import (
    ACT,
    AGG,
    apply_activation,
    apply_aggregation,
    get_func_name,
)
from ..base import GeneAttr
from .base import BaseNode


class DefaultNode(BaseNode):
    """Default node gene with the same behaviour as NEAT-python.

    Four float32 attribute columns, in this order: ``bias``, ``response``,
    ``aggregation`` (int option index), ``activation`` (int option index).
    ``aggregate`` reduces the ``K``-wide incoming window per configured aggregation
    option and selects the row's option; ``combine`` is ``bias + response * agg``;
    the activation is applied only where the node is not an output node, and the
    option index ``-1`` means identity.

    The option lists are materialised into ``attr_specs`` at construction (an
    instance attribute rebuilt from the configured ``aggregation_options`` /
    ``activation_options``), so every option index of every kernel is a static
    Python int: the class-level table below records the default single-option
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
            aggregation_default: Aggregation used by ``new_identity_attrs`` and
                ``new_random_attrs`` when it is preferred over the uniform draw;
                ``None`` means option 0 of ``aggregation_options``.
            aggregation_options: One callable or a sequence of callables; a bare
                callable is promoted to a one-element list. The index of a callable
                in this list is the value stored in the ``aggregation`` column.
            aggregation_replace_rate: Probability that the aggregation index is
                re-drawn uniformly over ``aggregation_options``.
            activation_default: Activation used by identity/initial attribute rows;
                ``None`` means option 0 of ``activation_options``. ``-1`` is not an
                option index (it means identity) and must not be passed here.
            activation_options: One callable or a sequence of callables, promoted
                the same way as ``aggregation_options``.
            activation_replace_rate: Probability that the activation index is
                re-drawn uniformly over ``activation_options``.

        Raises:
            NotImplementedError: Skeleton stub; the constructor resolves the option
                sequences, records the default indices and rebuilds ``attr_specs``
                with ``options=tuple(range(len(options)))`` in the implementation
                wave.
            TensorNEATError: (implementation wave) if an option sequence is empty or
                a default is not one of the configured options.
        """
        raise NotImplementedError(
            "resolve the option sequences, store the default indices, and rebuild "
            "attr_specs with options=tuple(range(len(options)))"
        )

    def aggregate(
        self,
        state: State,
        attrs: evoxc.SymbolicTensor,
        messages: evoxc.SymbolicTensor,
        valid_mask: evoxc.SymbolicTensor,
    ) -> evoxc.SymbolicTensor:
        """Aggregate the incoming window with each configured option, then select.

        Args:
            state: Host configuration container.
            attrs: Node attributes, ``(P, N, 4)`` float32; the ``aggregation``
                column is the per-row option index.
            messages: ``(P, N, K)`` float32 incoming messages, already NaN-sanitized.
            valid_mask: ``(P, N, K)`` bool, True for live slots; dead slots are
                replaced with each option's neutral element before reduction.

        Returns:
            ``(P, N)`` float32 aggregated values.

        Raises:
            NotImplementedError: Skeleton stub; ``apply_aggregation`` plus a
                ``select`` chain over the configured options lands in the
                implementation wave.
        """
        raise NotImplementedError(
            "apply_aggregation per configured option over the last axis, then select "
            "by the aggregation column"
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
            aggregated: ``(P, N)`` float32 output of :meth:`aggregate`.

        Returns:
            ``(P, N)`` float32 pre-activation values.

        Raises:
            NotImplementedError: Skeleton stub.
        """
        raise NotImplementedError("bias + response * aggregated, elementwise over (P, N)")

    def repr(
        self,
        state: State,
        key_row: evoxc.SymbolicTensor,
        attr_row: evoxc.SymbolicTensor,
        precision: int = 2,
    ) -> str:
        """Render one node gene as ``index / bias / response / aggregation / activation``.

        Args:
            state: Host configuration container.
            key_row: ``(key_width,)`` int32 key columns; the node key is column 0.
            attr_row: ``(4,)`` float32 attribute columns of the same gene.
            precision: Decimal places for ``bias`` and ``response``.

        Returns:
            A single-line description, e.g.
            ``DefaultNode(idx=3, bias=0.25, response=1.0, aggregation=sum, activation=sigmoid)``.

        Raises:
            NotImplementedError: Skeleton stub.
        """
        raise NotImplementedError(
            "format idx plus bias/response (rounded) and the option names of aggregation/activation"
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
            the option indices resolved through ``get_func_name``; ``act`` resolves an
            index of ``-1`` to ``"identity"``.

        Raises:
            NotImplementedError: Skeleton stub.
        """
        raise NotImplementedError(
            "map idx/bias/res columns and resolve agg/act indices to registered function names"
        )

    def sympy_func(
        self, state: State, node_dict: dict, inputs: sp.Expr, is_output_node: bool = False
    ) -> tuple[sp.Expr, dict]:
        """Build the sympy expression of one node from its ``to_dict`` output.

        Args:
            state: Host configuration container.
            node_dict: Output of :meth:`to_dict` for this node.
            inputs: sympy expression of the aggregated incoming signal.
            is_output_node: True to leave the expression unactivated, matching
                :meth:`forward`.

        Returns:
            ``(expression, {symbol: value})`` where the expression is
            ``bias + response * agg(inputs)`` (plus the activation unless the node is
            an output node) and the map binds the ``bias``/``response`` symbols to
            this node's attribute values.

        Raises:
            NotImplementedError: Skeleton stub; ``AGG.obtain_sympy`` /
                ``ACT.obtain_sympy`` wiring lands in the implementation wave.
        """
        raise NotImplementedError(
            "sympy: bias + response * AGG.obtain_sympy(agg)(inputs), activated unless output node"
        )
