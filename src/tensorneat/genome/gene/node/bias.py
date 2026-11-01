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

from ....common import State
from ....common.functions import (
    ACT,
    AGG,
    apply_activation,
    apply_aggregation,
)
from ..base import GeneAttr
from .default import DefaultNode


class BiasNode(DefaultNode):
    """Node gene without ``response``: ``bias + aggregated``, then activation.

    Three float32 attribute columns, in this order: ``bias``, ``aggregation`` (int
    option index), ``activation`` (int option index); the aggregation and activation
    handling is inherited unchanged from :class:`.default.DefaultNode`.

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

        Args:
            bias_init_mean: Mean of the initial bias draw.
            bias_init_std: Standard deviation of the initial bias draw.
            bias_mutate_power: Standard deviation of the bias perturbation.
            bias_mutate_rate: Probability that the bias is perturbed.
            bias_replace_rate: Probability that the bias is re-drawn.
            bias_lower_bound: Inclusive lower clip bound of the bias.
            bias_upper_bound: Inclusive upper clip bound of the bias.
            aggregation_default: Aggregation used by identity/initial attribute rows;
                ``None`` means option 0 of ``aggregation_options``.
            aggregation_options: One callable or a sequence of callables; the index of
                a callable in this list is the value stored in the ``aggregation``
                column.
            aggregation_replace_rate: Probability that the aggregation index is
                re-drawn uniformly over ``aggregation_options``.
            activation_default: Activation used by identity/initial attribute rows;
                ``None`` means option 0 of ``activation_options``. Never ``-1``
                (identity is not an option index).
            activation_options: One callable or a sequence of callables.
            activation_replace_rate: Probability that the activation index is
                re-drawn uniformly over ``activation_options``.

        Raises:
            NotImplementedError: Skeleton stub; the implementation resolves the
                option sequences, stores the default indices and rebuilds
                ``attr_specs`` with ``options=tuple(range(len(options)))``.
            TensorNEATError: (implementation wave) if an option sequence is empty or
                a default is not one of the configured options.
        """
        raise NotImplementedError(
            "resolve the option sequences, store the default indices, and rebuild "
            "attr_specs without the response column"
        )

    def new_identity_attrs(self, state: State) -> evoxc.SymbolicTensor:
        """Identity attribute row: ``bias=0``, default aggregation, default activation.

        Args:
            state: Host configuration container.

        Returns:
            ``(3,)`` float32 row ``(0, aggregation_default, activation_default)``;
            the activation stays on a real option index -- ``-1`` is not used here,
            because a split node must keep the activation it had before the split.

        Raises:
            NotImplementedError: Skeleton stub.
        """
        raise NotImplementedError(
            "(3,) identity row: bias 0, aggregation_default, activation_default"
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
                :meth:`.default.DefaultNode.aggregate`.

        Returns:
            ``(P, N)`` float32 pre-activation values.

        Raises:
            NotImplementedError: Skeleton stub.
        """
        raise NotImplementedError("bias + aggregated, elementwise over (P, N)")

    def repr(
        self,
        state: State,
        key_row: evoxc.SymbolicTensor,
        attr_row: evoxc.SymbolicTensor,
        precision: int = 2,
    ) -> str:
        """Render one node gene as ``index / bias / aggregation / activation``.

        Args:
            state: Host configuration container.
            key_row: ``(key_width,)`` int32 key columns; the node key is column 0.
            attr_row: ``(3,)`` float32 attribute columns of the same gene.
            precision: Decimal places for ``bias``.

        Returns:
            A single-line description, e.g.
            ``BiasNode(idx=3, bias=0.25, aggregation=sum, activation=sigmoid)``.

        Raises:
            NotImplementedError: Skeleton stub.
        """
        raise NotImplementedError(
            "format idx, rounded bias and the option names of aggregation/activation"
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
            NotImplementedError: Skeleton stub.
        """
        raise NotImplementedError(
            "map idx/bias columns and resolve agg/act indices to registered function names"
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
                :meth:`.default.DefaultNode.forward`.

        Returns:
            ``(expression, {symbol: value})`` where the expression is
            ``bias + agg(inputs)`` (plus the activation unless the node is an output
            node) and the map binds the ``bias`` symbol to this node's value.

        Raises:
            NotImplementedError: Skeleton stub.
        """
        raise NotImplementedError(
            "sympy: bias + AGG.obtain_sympy(agg)(inputs), activated unless output node"
        )
