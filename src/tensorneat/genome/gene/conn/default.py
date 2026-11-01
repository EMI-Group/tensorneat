"""Default connection gene: a single ``weight`` attribute column.

Batch-first: one call computes the message of every edge slot of the whole
population, purely elementwise in ``(weight, source value)``.
This file declares the weight table and the NaN-safe message; it must not import
``jax``/``torch`` and must not duplicate the key-column or crossover machinery of
:class:`..base.BaseConn`.
"""

import evoxc
import sympy as sp

from ....common import State
from ..base import GeneAttr
from .base import BaseConn


class DefaultConn(BaseConn):
    """Default connection gene with the same behaviour as NEAT-python.

    One float32 attribute column, ``weight``; the message of an edge is
    ``weight * source_value`` with both operands NaN-sanitized first. Key columns are
    inherited: ``(input_index, output_index)``.

    Attribute values of this family: ``new_identity_attrs`` yields ``weight = 1.0``
    (an added connection reproduces the signal of its source) and
    ``new_zero_attrs`` yields ``weight = 0.0`` (a new connection that does not
    disturb existing behaviour).
    """

    attr_specs: tuple[GeneAttr, ...] = (
        GeneAttr(
            name="weight",
            kind="float",
            init_mean=0.0,
            init_std=1.0,
            mutate_power=0.15,
            mutate_rate=0.8,
            replace_rate=0.1,
            lower_bound=None,
            upper_bound=None,
        ),
    )

    def __init__(
        self,
        weight_init_mean: float = 0.0,
        weight_init_std: float = 1.0,
        weight_mutate_power: float = 0.15,
        weight_mutate_rate: float = 0.8,
        weight_replace_rate: float = 0.1,
        weight_lower_bound: float | None = None,
        weight_upper_bound: float | None = None,
    ):
        """Configure the weight attribute column (host-side, called once).

        Args:
            weight_init_mean: Mean of the initial weight draw.
            weight_init_std: Standard deviation of the initial weight draw, which is
                also the standard deviation of the ``replace_rate`` re-draw.
            weight_mutate_power: Standard deviation of the additive weight
                perturbation.
            weight_mutate_rate: Probability that the weight is perturbed; weights are
                mutated more often than node attributes, as in NEAT-python.
            weight_replace_rate: Probability that the weight is re-drawn from the init
                distribution.
            weight_lower_bound: Optional inclusive lower clip bound; ``None`` means
                unclipped.
            weight_upper_bound: Optional inclusive upper clip bound; ``None`` means
                unclipped.

        Raises:
            NotImplementedError: Skeleton stub; the implementation stores the knobs
                and rebuilds ``attr_specs`` from them in the implementation wave.
            TensorNEATError: (implementation wave) if ``lower_bound`` is not below
                ``upper_bound``.
        """
        raise NotImplementedError(
            "store the weight init/mutate knobs and rebuild attr_specs from them"
        )

    def forward(
        self,
        state: State,
        attrs: evoxc.SymbolicTensor,
        source_values: evoxc.SymbolicTensor,
    ) -> evoxc.SymbolicTensor:
        """Compute the NaN-safe edge message ``sanitized(source) * sanitized(weight)``.

        Both operands are sanitized before the product: a NaN source value (source
        not computed, absent node) or a NaN weight (absent connection row) becomes 0,
        so the aggregation never multiplies a zeroed cotangent by a NaN local
        derivative during reverse-mode differentiation.

        Args:
            state: Host configuration container.
            attrs: Per-slot connection attributes, ``(P, N, K, 1)`` float32, gathered
                with the incoming-edge table.
            source_values: Source node values, ``(P, N, K)`` float32.

        Returns:
            ``(P, N, K)`` float32 messages, ``select(isnan(source), 0, source) *
            select(isnan(weight), 0, weight)`` per slot.

        Raises:
            NotImplementedError: Skeleton stub.
        """
        raise NotImplementedError(
            "select(isnan(source), 0, source) * select(isnan(weight), 0, weight) -> (P, N, K)"
        )

    def repr(
        self,
        state: State,
        key_row: evoxc.SymbolicTensor,
        attr_row: evoxc.SymbolicTensor,
        precision: int = 2,
    ) -> str:
        """Render one connection gene as ``in / out / weight``.

        Args:
            state: Host configuration container.
            key_row: ``(key_width,)`` int32 key columns of a single gene.
            attr_row: ``(1,)`` float32 attribute columns of the same gene.
            precision: Decimal places for the weight.

        Returns:
            A single-line description, e.g. ``DefaultConn(in: 1, out: 4, weight: 0.50)``.

        Raises:
            NotImplementedError: Skeleton stub.
        """
        raise NotImplementedError("format the in/out keys and the rounded weight")

    def to_dict(
        self,
        state: State,
        key_row: evoxc.SymbolicTensor,
        attr_row: evoxc.SymbolicTensor,
    ) -> dict:
        """Convert one connection gene to the dict used by host code generation.

        Args:
            state: Host configuration container.
            key_row: ``(key_width,)`` int32 key columns of a single gene.
            attr_row: ``(1,)`` float32 attribute columns of the same gene.

        Returns:
            ``{"in": int, "out": int, "weight": float}``.

        Raises:
            NotImplementedError: Skeleton stub.
        """
        raise NotImplementedError("map the in/out keys and the weight column to a dict")

    def sympy_func(
        self, state: State, conn_dict: dict, inputs: sp.Expr
    ) -> tuple[sp.Expr, dict]:
        """Build the sympy expression ``inputs * weight`` of one connection.

        Args:
            state: Host configuration container.
            conn_dict: Output of :meth:`to_dict` for this connection.
            inputs: sympy expression of the source node value.

        Returns:
            ``(inputs * weight, {weight_symbol: conn_dict["weight"]})``.

        Raises:
            NotImplementedError: Skeleton stub.
        """
        raise NotImplementedError(
            "sympy: inputs * weight_symbol plus {weight_symbol: conn_dict['weight']}"
        )
