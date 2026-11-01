"""Substrate gene kernels of HyperNEAT: the node gene and the connection gene.

HyperNEAT evolves only the CPPN, so the substrate phenotype never mutates: its gene payload is
repainted from the decoded weights on every :meth:`HyperNEAT.transform` (see ``./hyperneat.py``).
That makes both genes *execution* genes -- they implement ``forward`` (and, for the node gene, the
``combine`` hook of the node algebra) and nothing of the genetic operator surface is ever invoked.

Roles
    * ``HyperNEATNode`` -- masked aggregation of the incoming-message window plus a fixed activation,
      inherited verbatim from :class:`~tensorneat.genome.BaseNode`; only the option tables, the
      identity ``combine`` and the host-side presentation live here.
    * ``HyperNEATConn`` -- one multiply by the decoded weight, read by spec name so the gene works
      with the prepared ``ConnOperands`` bundle the genome actually passes.

Constraints
    * Batch-first over arbitrary leading axes, no ``vmap``/``cond``/``scan``, no branching on
      traced values.
    * Host helpers (:func:`_key_int`, :func:`_attr_float`) run only in ``repr``/``to_dict`` and never
      inside a trace.
    * No ``jax``/``torch``; ``numpy`` is only used to read host rows.
"""

from typing import Any, Callable

import numpy as np

from tensorneat.common import ACT, AGG, State
from tensorneat.common.errors import ConfigError
from tensorneat.common.tools import to_numpy
from tensorneat.genome import BaseConn, BaseNode
from tensorneat.genome.gene.base import GeneAttr


def _key_int(key_row: Any, column: int) -> int:
    """Read one ``int32`` key column of a single gene row on the host (never inside a trace)."""
    return int(np.asarray(to_numpy(key_row)).reshape(-1)[column])


def _attr_float(attr_row: Any, column: int) -> float:
    """Read one ``float32`` attribute column of a single gene row on the host (never in a trace)."""
    return float(np.asarray(to_numpy(attr_row)).reshape(-1)[column])


def _require_callable(name: str, value: Any) -> None:
    """Raise ``ConfigError`` unless ``value`` is callable (host-side constructor validation).

    Args:
        name: Constructor argument name, for the error message.
        value: The candidate aggregation/activation option.

    Raises:
        tensorneat.common.errors.ConfigError: If ``value`` is not callable, in which case the
            option cascade would fail only once a graph is traced.
    """
    if not callable(value):
        raise ConfigError(
            f"{name} must be a callable, got {value!r} ({type(value).__name__})"
        )


#: One-element option-index columns of :class:`HyperNEATNode`, in payload column order
#: (``aggregation`` then ``activation``).  This is the class-level default layout; ``__init__``
#: reinstalls the very same table per instance, because every row of a substrate payload stores
#: option index ``0`` -- the single aggregation/activation pair fixed at construction.
_NODE_OPTION_SPECS: tuple[GeneAttr, ...] = (
    GeneAttr(name="aggregation", kind="int", replace_rate=0.1, options=(0,)),
    GeneAttr(name="activation", kind="int", replace_rate=0.1, options=(0,)),
)


class HyperNEATNode(BaseNode):
    """Substrate node gene with fixed (non-evolved) aggregation and activation.

    The substrate topology is fixed at construction and only its weights are decoded from the CPPN,
    so a substrate node has no evolvable *value*: the gene declares the two attribute columns the
    :class:`~tensorneat.genome.BaseNode` contract requires -- the ``aggregation`` and ``activation``
    option indices -- and every payload row carries index ``0``, which selects the callables this
    gene was constructed with.  ``key_width == 1`` (the identity key alone) and ``attr_width == 2``.

    Attributes:
        aggregation (Callable): ``(z (..., max_sources), valid_mask (..., max_sources)) -> (...,)``.
        activation (Callable): Unary elementwise activation applied to non-output nodes.
        aggregation_options (tuple): One-element option tuple ``(aggregation,)``.
        activation_options (tuple): One-element option tuple ``(activation,)``.
        attr_specs (tuple[GeneAttr, ...]): ``(aggregation, activation)`` int option columns, each
            with the single legal index ``0``.
    """

    #: Identity column of a node row; the substrate node carries nothing else in its keys.
    fixed_attrs = ("index",)

    #: Default single-option table; the instance table installed by ``__init__`` is identical.
    attr_specs: tuple[GeneAttr, ...] = _NODE_OPTION_SPECS

    def __init__(self, aggregation: Callable = AGG.sum, activation: Callable = ACT.sigmoid):
        """Store the fixed aggregation and activation as the gene's only two options.

        Args:
            aggregation: Masked aggregation over the incoming-source axis, taking ``(z, valid_mask)``.
            activation: Activation applied to every node that is not an output node.

        Raises:
            tensorneat.common.errors.ConfigError: If either option is not callable.

        Note:
            Both callables are configuration shared by every individual: they never enter ``State``
            and are never mutated, because the hyper genome is not evolved.  The option tuples hold
            exactly one element, so the option columns of the payload -- which
            :meth:`HyperNEAT._assemble_substrate` paints with ``0`` -- have exactly one legal index
            and a substrate node can never select an aggregation or activation it was not built for.
        """
        _require_callable("aggregation", aggregation)
        _require_callable("activation", activation)
        self.aggregation = aggregation
        self.activation = activation
        self.aggregation_options = (aggregation,)
        self.activation_options = (activation,)
        # Column order IS the payload layout: aggregation, then activation.
        self.attr_specs = _NODE_OPTION_SPECS

    def combine(
        self,
        state: State,
        attrs: Any,
        aggregated: Any,
    ) -> Any:
        """Return the aggregated value unchanged (a substrate node has no affine map).

        ``DefaultNode`` adds ``bias + response * aggregated`` and ``BiasNode`` adds ``bias``; a
        substrate node has neither column, because no node attribute is decoded from the CPPN.

        Args:
            state: Host configuration container (read-only during a trace).
            attrs: ``(P, N, 2)`` float32 node attribute rows, or the
                :class:`~tensorneat.genome.gene.node.base.NodeOperands` bundle of the same rows;
                accepted for the family contract and never read.
            aggregated: ``(P, N)`` float32 output of
                :meth:`~tensorneat.genome.gene.node.base.BaseNode.aggregate`.

        Returns:
            ``aggregated`` itself, so :meth:`~tensorneat.genome.gene.node.base.BaseNode.forward`
            emits no extra op for the substrate.
        """
        return aggregated

    def repr(
        self,
        state: State,
        key_row: Any,
        attr_row: Any,
        precision: int = 2,
    ) -> str:
        """Render one substrate node gene as one line (host-only, never traced).

        Args:
            state: Host configuration container.
            key_row: ``(key_width,)`` int32 key columns of a single node gene.
            attr_row: ``(2,)`` float32 option-index columns of the same gene.
            precision: Unused; the two rendered values are option names, not floats.

        Returns:
            A single-line description, e.g.
            ``HyperNEATNode(idx=3, aggregation=sum, activation=sigmoid)``.
        """
        node = self.to_dict(state, key_row, attr_row)
        return (
            f"{type(self).__name__}(idx={node['idx']}, "
            f"aggregation={node['agg']}, activation={node['act']})"
        )

    def to_dict(
        self,
        state: State,
        key_row: Any,
        attr_row: Any,
    ) -> dict[str, Any]:
        """Convert one substrate node gene to ``{"idx", "agg", "act"}`` (host-only, never traced).

        Args:
            state: Host configuration container.
            key_row: ``(key_width,)`` int32 key columns of a single node gene.
            attr_row: ``(2,)`` float32 option-index columns of the same gene.

        Returns:
            ``{"idx": int, "agg": str, "act": str}``, with both option indices resolved to registry
            names through the inherited ``_aggregation_name`` / ``_activation_name`` helpers
            (``act`` resolves the ``-1`` identity sentinel to ``"identity"``).

        Raises:
            tensorneat.common.errors.ConfigError: If an option column holds an id that is not a
                configured option (see ``BaseNode._option_name``), or a row has the wrong width
                (see ``BaseGene._host_row``).
        """
        keys, attrs = self._host_row(key_row, attr_row)
        return {
            "idx": keys[self.key_index("index")],
            "agg": self._aggregation_name(
                int(round(attrs[self.attr_index("aggregation")]))
            ),
            "act": self._activation_name(
                int(round(attrs[self.attr_index("activation")]))
            ),
        }


class HyperNEATConn(BaseConn):
    """Substrate connection gene whose only evolvable attribute is the decoded CPPN weight.

    The two endpoint columns are ``int32`` keys fixed by the substrate topology; ``attr_specs``
    declares the single float32 ``weight`` column that :meth:`HyperNEAT.transform` repaints from the
    CPPN, so ``key_width == 2`` and ``attr_width == 1``.
    """

    #: Endpoint key columns, fixed by the substrate topology.
    fixed_attrs = ("input_index", "output_index")

    #: The single decoded attribute: the connection weight, with ``DefaultConn``'s weight semantics.
    attr_specs: tuple[GeneAttr, ...] = (
        GeneAttr(
            name="weight",
            kind="float",
            init_mean=0.0,
            init_std=1.0,
            mutate_power=0.15,
            mutate_rate=0.8,
            replace_rate=0.1,
        ),
    )

    def __init__(self):
        """Resolve ``self.weight_index`` from the ``attr_specs`` names (never from an offset)."""
        self.weight_index = [spec.name for spec in self.attr_specs].index("weight")

    def forward(self, state: State, attrs: Any, source_values: Any) -> Any:
        """Scale a source value by the connection weight.

        Batch-first over arbitrary leading axes; the incoming-source axis is the last one:

        Args:
            state: Current algorithm state (unused; kept for the gene protocol).
            attrs: ``(..., max_sources, attr_width)`` gathered connection attribute rows for one
                destination node, or the ``ConnOperands`` bundle
                :meth:`~tensorneat.genome.gene.conn.base.BaseConn.prepare` builds from them -- which
                is what the forward pass actually passes, so the weight column is read through
                :meth:`~tensorneat.genome.gene.conn.base.BaseConn.column_of` by spec *name* and the
                bundle's NaN-cleaned columns are used instead of re-cleaning the raw rows.
            source_values: ``(..., max_sources)`` source values already masked to ``0`` where the
                edge does not exist.

        Returns:
            ``(..., max_sources)`` weighted contributions, ``source_values * weight``.

        Note:
            The weight column is resolved through ``attr_specs`` (by name, never a literal index),
            so inserting an attribute column cannot silently shift it.  Unlike the pre-rewrite
            ``HyperNEATConn.forward`` this kernel does **not** NaN-guard its operands: absent edges
            are excluded by ``valid_mask`` and their slots are already neutralised upstream, so
            masking happens once, at the node gene, and this kernel stays a single multiply.
        """
        return source_values * self.column_of(attrs, "weight")

    def repr(
        self,
        state: State,
        key_row: Any,
        attr_row: Any,
        precision: int = 2,
    ) -> str:
        """Render one substrate connection gene as ``in / out / weight`` (host-only).

        Args:
            state: Host configuration container.
            key_row: ``(key_width,)`` int32 key columns of a single connection gene.
            attr_row: ``(attr_width,)`` float32 attribute columns of the same gene.
            precision: Decimal places for the weight.

        Returns:
            A single-line description, e.g. ``HyperNEATConn(in: 1, out: 4, weight: 0.50)``.
        """
        weight = _attr_float(attr_row, self.weight_index)
        return (
            f"{type(self).__name__}(in: {_key_int(key_row, 0)}, "
            f"out: {_key_int(key_row, 1)}, weight: {weight:.{precision}f})"
        )

    def to_dict(
        self,
        state: State,
        key_row: Any,
        attr_row: Any,
    ) -> dict[str, Any]:
        """Convert one substrate connection gene to ``{"in", "out", "weight"}`` (host-only)."""
        return {
            "in": _key_int(key_row, 0),
            "out": _key_int(key_row, 1),
            "weight": _attr_float(attr_row, self.weight_index),
        }
