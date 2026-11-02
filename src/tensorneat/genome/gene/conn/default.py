"""Default connection gene: a single ``weight`` attribute column.

Batch-first: one call computes the message of every edge slot of the whole
population, purely elementwise in ``(weight, source value)``.
This file declares the weight table and the NaN-safe message; it must not import
``jax``/``torch`` and must not duplicate the key-column or crossover machinery of
:class:`.base.BaseConn`.
"""

from typing import Any

import evoxc
import sympy as sp

from ....common import State
from ....common.errors import ConfigError
from ..base import GeneAttr
from .base import BaseConn, ConnOperands


class DefaultConn(BaseConn):
    """Default connection gene with the same behaviour as NEAT-python.

    One float32 attribute column, ``weight``; the message of an edge is
    ``weight * source_value``. The weight column is NaN-cleaned once per forward pass
    (``prepare``) and the source value arrives already masked by the window step's liveness
    mask (dead slots are ``NaN``, discarded by the aggregation). Key columns are
    inherited: ``(input_index, output_index)``.

    Attribute values of this family: ``new_identity_attrs`` yields ``weight = 1.0``
    (an added connection reproduces the signal of its source) and
    ``new_zero_attrs`` yields ``weight = 0.0`` (a new connection that does not
    disturb existing behaviour).
    """

    #: Default weight table; :meth:`__init__` rebuilds it from its knobs per instance.
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

        The knobs are stored as attributes and the instance's ``attr_specs`` is
        rebuilt from them, so :meth:`~.base.BaseConn.attr_column`,
        ``new_random_attrs``, ``mutate`` and ``distance`` all follow the configured
        distribution and bounds without knowing about this constructor.

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
            tensorneat.common.errors.ConfigError: If both bounds are given and
                ``weight_lower_bound >= weight_upper_bound``, which would make the
                bound interval empty.
        """
        self.weight_init_mean = float(weight_init_mean)
        self.weight_init_std = float(weight_init_std)
        self.weight_mutate_power = float(weight_mutate_power)
        self.weight_mutate_rate = float(weight_mutate_rate)
        self.weight_replace_rate = float(weight_replace_rate)
        self.weight_lower_bound = weight_lower_bound
        self.weight_upper_bound = weight_upper_bound
        if (
            weight_lower_bound is not None
            and weight_upper_bound is not None
            and float(weight_lower_bound) >= float(weight_upper_bound)
        ):
            raise ConfigError(
                f"{type(self).__name__}: weight_lower_bound {weight_lower_bound} must "
                f"be below weight_upper_bound {weight_upper_bound}"
            )
        self.attr_specs = (
            GeneAttr(
                name="weight",
                kind="float",
                init_mean=self.weight_init_mean,
                init_std=self.weight_init_std,
                mutate_power=self.weight_mutate_power,
                mutate_rate=self.weight_mutate_rate,
                replace_rate=self.weight_replace_rate,
                lower_bound=weight_lower_bound,
                upper_bound=weight_upper_bound,
            ),
        )

    def new_identity_attrs(self, state: State) -> evoxc.SymbolicTensor:
        """Attribute row of a connection that reproduces its source signal.

        Used by the add-node split of ``genome/operations/mutation``, which replaces a
        connection ``a -> b`` by ``a -> new -> b``: the new edge must carry
        ``weight = 1.0`` so the split preserves the signal. The row is the fixed
        legacy value ``1.0`` and is deliberately **not** clipped into the weight
        bounds, so a split never silently rescales the network (legacy parity).

        Args:
            state: Host configuration container.

        Returns:
            ``(1,)`` float32 row holding ``weight = 1.0``.
        """
        return self._attr_row([1.0])

    def new_zero_attrs(self, state: State) -> evoxc.SymbolicTensor:
        """Attribute row of a brand-new connection: ``weight = 0.0``.

        Used by add-connection mutation, so a newly created edge is inert until
        mutation gives it a weight.

        Args:
            state: Host configuration container.

        Returns:
            ``(1,)`` float32 row holding ``weight = 0.0``.
        """
        return self._attr_row([0.0])

    def forward(
        self,
        state: State,
        attrs: Any,
        source_values: evoxc.SymbolicTensor,
    ) -> evoxc.SymbolicTensor:
        """Compute the edge message ``source * cleaned(weight)`` in one multiply.

        Only the *weight* is sanitized here, and that happens once per forward pass in
        :meth:`~.base.BaseConn.prepare`; the source values arrive already masked by the
        caller's liveness mask (dead and uncomputed slots are ``NaN``). Those ``NaN``
        messages never reach a reduction: the node gene's aggregation replaces every dead
        slot with its option's neutral value *before* reducing, so a ``NaN`` here cannot
        poison a node (nor a reverse-mode sweep through a zeroed cotangent). Sanitising the
        source a second time inside the gene would cost an extra ``isnan`` + ``select`` on
        every unrolled level for exactly the slots the mask already discards.

        Args:
            state: Host configuration container.
            attrs: Per-slot connection attributes, ``(P, N, K, 1)`` float32, gathered
                with the incoming-edge table, or the
                :class:`~.base.ConnOperands` bundle of those rows, whose ``weight``
                column is already NaN-cleaned.
            source_values: Source node values, ``(P, N, K)`` float32, ``NaN`` where the
                slot is dead or its source row has no value yet.

        Returns:
            ``(P, N, K)`` float32 messages, ``source * weight`` per slot; ``NaN`` on the
            slots the node gene's ``valid_mask`` discards. A family that declares no
            ``weight`` column has the identity weight ``1``, so its message is the source
            value itself (an unweighted edge).
        """
        prepared = attrs if isinstance(attrs, ConnOperands) else self.prepare(attrs)
        weight = self.column_of(prepared, "weight", 1.0)
        return evoxc.multiply(source_values, weight)

    def sympy_func(
        self, state: State, conn_dict: dict, inputs: sp.Expr
    ) -> tuple[sp.Expr, dict]:
        """Build the sympy expression ``inputs * weight`` of one connection.

        The weight symbol is named ``c_{in}_{out}_w`` from the connection's endpoint
        keys, so the generated formula stays readable and every edge of a decoded
        network gets its own symbol.

        Args:
            state: Host configuration container.
            conn_dict: Output of ``to_dict`` for this connection.
            inputs: sympy expression of the source node value.

        Returns:
            ``(inputs * weight, {weight_symbol: conn_dict["weight"]})``; a family that
            declares no ``weight`` column has the identity weight ``1`` (as in
            :meth:`forward`), so the expression is the source value itself and the
            substitution map is empty.
        """
        if "weight" not in conn_dict:  # no attribute column: an unweighted edge
            return inputs, {}
        weight = sp.symbols(f"c_{conn_dict['in']}_{conn_dict['out']}_w")
        return inputs * weight, {weight: conn_dict["weight"]}
