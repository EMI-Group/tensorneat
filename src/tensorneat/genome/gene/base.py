"""Gene base classes: the attr-table contract shared by node and conn genes.

A gene row is split into integer *key columns* (``fixed_attrs``) and float32
*attribute columns* (``attr_specs``); every gene operation is batch-first with the
population on axis 0 and shapes that are static at trace time.
This module declares the table (specifications, widths, row hash) only: it must
never import ``jax``/``torch``, never run eager NumPy on the compute path, and
never implement node/connection semantics (those live in ``node/`` and ``conn/``).
"""

from dataclasses import dataclass
from typing import Any

import evoxc

from ...common import State, StatefulBaseClass


@dataclass(frozen=True)
class GeneAttr:
    """Declaration of one attribute column of a gene.

    The instance data reproduces the legacy ``mutate_float`` / ``mutate_int``
    distributions, so genetic operators stay compatible with the pre-rewrite library
    while being expressed as batched evoxc ops.
    A ``"float"`` spec is initialised with ``normal(init_mean, init_std)`` clipped
    into ``[lower_bound, upper_bound]``; mutation adds ``normal(0, mutate_power)``
    noise with probability ``mutate_rate`` and, independently with probability
    ``replace_rate``, re-draws the value from the init distribution.
    An ``"int"`` spec is an option index: ``init_mean``/``init_std``/``mutate_power``
    are ignored, initialisation draws uniformly from ``options`` and mutation
    re-draws uniformly from ``options`` with probability ``replace_rate`` (an index
    never receives ``normal`` noise).

    Attributes:
        name: Column name; its position in ``attr_specs`` is the column index of
            ``*_attrs``, never a hardcoded offset.
        kind: ``"float"`` for a float32 attribute column, ``"int"`` for a float32
            column holding a discrete option index.
        init_mean: Mean of the initialisation Gaussian (``"float"`` only).
        init_std: Std of the initialisation Gaussian, also used by the
            ``replace_rate`` re-draw (``"float"`` only).
        mutate_power: Std of the zero-mean perturbation applied with probability
            ``mutate_rate`` (``"float"`` only).
        mutate_rate: Probability that the additive perturbation is applied
            (``"float"`` only).
        replace_rate: Probability that the value is re-drawn from scratch instead of
            perturbed (``"float"``: init distribution, ``"int"``: uniform over
            ``options``).
        lower_bound: Inclusive lower clip bound (``"float"`` only; ``None`` means
            unclipped), mirrored by ``upper_bound``.
        upper_bound: Inclusive upper clip bound (``"float"`` only).
        options: Static tuple of allowed option indices for an ``"int"`` spec, e.g.
            ``(0, 1, 2)``; non-empty for ``"int"`` and ``None`` for ``"float"``.
    """

    name: str
    kind: str
    init_mean: float = 0.0
    init_std: float = 1.0
    mutate_power: float = 0.15
    mutate_rate: float = 0.2
    replace_rate: float = 0.015
    lower_bound: float | None = None
    upper_bound: float | None = None
    options: tuple[int, ...] | None = None


def hash_int32(keys: evoxc.SymbolicTensor) -> evoxc.SymbolicTensor:
    """Reduce the integer key columns of a gene row to one xor-mixed row hash.

    No bitcast between float and int exists in evoxc, so the hash is taken over the
    ``int32`` key columns only; float attributes are never hashed by bit pattern.

    Args:
        keys: Key columns of one gene family, ``(*shape, key_width)`` int32.

    Returns:
        ``(*shape,)`` uint32 row hashes, one per gene row.

    Raises:
        NotImplementedError: Skeleton stub; the xor-mix kernel lands in the
            implementation wave.
    """
    raise NotImplementedError("xor-mix the int32 key columns over the last axis -> uint32")


class BaseGene(StatefulBaseClass):
    """Table-driven gene: declare key columns and attribute columns.

    Subclasses declare class data only: ``fixed_attrs`` (names of the ``int32`` key
    columns, row order, ``key_width = len(fixed_attrs)``, absent rows ``I_INF``) and
    ``attr_specs`` (``GeneAttr`` per ``float32`` attribute column, row order,
    ``attr_width = len(attr_specs)``, absent rows NaN).
    Payload layout, batch-first with the population on axis 0 and static shapes:
    ``node_keys (P, N) int32``, ``node_attrs (P, N, NA) float32``,
    ``conn_keys (P, E, key_width) int32``, ``conn_attrs (P, E, attr_width) float32``.
    Every operator below is batched over the leading axes of ``attrs``: no per-row
    Python loop, no key split, no ``vmap``/``scan``/``cond``. An ``evoxc.random`` key
    is rank-0, so randomness is drawn for the full batched shape in one call per spec.
    """

    fixed_attrs: tuple[str, ...] = ()
    attr_specs: tuple[GeneAttr, ...] = ()

    @property
    def key_width(self) -> int:
        """Number of ``int32`` key columns declared by ``fixed_attrs``.

        Returns:
            ``len(self.fixed_attrs)``; the last axis length of a ``*_keys`` tensor.
        """
        return len(self.fixed_attrs)

    @property
    def attr_width(self) -> int:
        """Number of ``float32`` attribute columns declared by ``attr_specs``.

        Returns:
            ``len(self.attr_specs)``; the last axis length of a ``*_attrs`` tensor.
        """
        return len(self.attr_specs)

    @property
    def length(self) -> int:
        """Total column count of a gene, key columns first (display only).

        Returns:
            ``key_width + attr_width``; the compute path never uses this, keys and
            attributes are separate tensors with separate dtypes.
        """
        return self.key_width + self.attr_width

    def setup(self, state: State = State()) -> State:
        """Register host-side gene configuration on an immutable state.

        Args:
            state: Base state to extend; never mutated in place.

        Returns:
            A new ``State`` carrying the resolved option lists and default option
            indices, so trace-time kernels only read static Python data.

        Raises:
            NotImplementedError: Skeleton stub; validation (non-empty ``options``,
                indices inside ``[0, len(options))``, consistent bounds) is pending.
        """
        raise NotImplementedError("register the resolved option lists and default indices")

    def new_random_attrs(
        self, state: State, key: evoxc.SymbolicTensor, shape: tuple[int, ...]
    ) -> evoxc.SymbolicTensor:
        """Draw initial attribute rows for a batch of newly created genes.

        One batched draw per spec (float: clipped Gaussian, int: uniform option
        index); the rank-0 ``key`` is never split per row, so each individual gets an
        independent stream from the single batched draw.

        Args:
            state: Host configuration container.
            key: Rank-0 ``evoxc.random`` key.
            shape: Batch shape of the genes, e.g. ``(P,)`` or ``(P, n_new)``.

        Returns:
            ``(*shape, attr_width)`` float32 attribute rows.

        Raises:
            NotImplementedError: Skeleton stub.
        """
        raise NotImplementedError("one batched draw per attr spec -> (*shape, attr_width)")

    def new_identity_attrs(self, state: State) -> evoxc.SymbolicTensor:
        """Attribute row whose gene is an identity transform (host ``add_node`` split).

        Args:
            state: Host configuration container.

        Returns:
            ``(attr_width,)`` float32 row; it must leave the network output unchanged
            when it replaces a split connection.

        Raises:
            NotImplementedError: Skeleton stub.
        """
        raise NotImplementedError("(attr_width,) identity row: no-op aggregation, identity act")

    def new_zero_attrs(self, state: State) -> evoxc.SymbolicTensor:
        """Attribute row with the least possible influence (host ``add_conn``).

        Args:
            state: Host configuration container.

        Returns:
            ``(attr_width,)`` float32 row; a new edge built from it must not disturb
            existing behaviour until mutation changes its attributes.

        Raises:
            NotImplementedError: Skeleton stub.
        """
        raise NotImplementedError("(attr_width,) zero-influence row")

    def mutate(
        self, state: State, key: evoxc.SymbolicTensor, attrs: evoxc.SymbolicTensor
    ) -> evoxc.SymbolicTensor:
        """Mutate attribute rows of the genes they belong to.

        Per spec: add ``normal(0, mutate_power)`` noise with probability
        ``mutate_rate`` (float only), then with probability ``replace_rate`` re-draw
        from the init distribution (float) or uniformly from ``options`` (int);
        floats are clipped into their bounds. Selection is a ``select`` chain over the
        batched attrs -- no branching on traced values.

        Args:
            state: Host configuration container.
            key: Rank-0 ``evoxc.random`` key for the whole batch.
            attrs: Attribute rows, ``(*shape, attr_width)`` float32.

        Returns:
            ``(*shape, attr_width)`` float32 mutated rows.

        Raises:
            NotImplementedError: Skeleton stub.
        """
        raise NotImplementedError(
            "per spec: perturb with mutate_rate, re-draw with replace_rate, clip to bounds"
        )

    def crossover(
        self,
        state: State,
        key: evoxc.SymbolicTensor,
        attrs1: evoxc.SymbolicTensor,
        attrs2: evoxc.SymbolicTensor,
    ) -> evoxc.SymbolicTensor:
        """Crossover the attributes of two homologous genes.

        Default behaviour is a per-attribute 50/50 pick; the origin families override
        it with a whole-gene pick (one draw selects the whole attribute vector).

        Args:
            state: Host configuration container.
            key: Rank-0 ``evoxc.random`` key for the whole batch.
            attrs1: Attribute rows of the first parent, ``(*shape, attr_width)``.
            attrs2: Key-aligned attribute rows of the second parent, same shape.

        Returns:
            ``(*shape, attr_width)`` float32 child attribute rows.

        Raises:
            NotImplementedError: Skeleton stub.
        """
        raise NotImplementedError(
            "per-attribute 50/50 select between attrs1 and attrs2 -> (*shape, attr_width)"
        )

    def distance(
        self, state: State, attrs1: evoxc.SymbolicTensor, attrs2: evoxc.SymbolicTensor
    ) -> evoxc.SymbolicTensor:
        """Attribute part of the compatibility distance between two aligned genes.

        Sums over the attribute axis: ``|delta|`` for float specs and ``!=`` for int
        specs, so a changed option index contributes exactly 1.

        Args:
            state: Host configuration container.
            attrs1: Attribute rows of the first gene, ``(*shape, attr_width)``.
            attrs2: Attribute rows of the second, key-aligned gene, same shape.

        Returns:
            ``(*shape,)`` float32 distances, one per aligned pair.

        Raises:
            NotImplementedError: Skeleton stub.
        """
        raise NotImplementedError("sum over the attr axis of |delta| (float) and != (int)")

    def hash(self, keys: evoxc.SymbolicTensor) -> evoxc.SymbolicTensor:
        """Hash gene key rows for fast key alignment in crossover/distance.

        Delegates to :func:`hash_int32`, an xor-mix over the ``int32`` key columns;
        float attributes are compared only after the keys align.

        Args:
            keys: Key columns of one gene family, ``(*shape, key_width)`` int32.

        Returns:
            ``(*shape,)`` uint32 row hashes.

        Raises:
            NotImplementedError: Skeleton stub.
        """
        raise NotImplementedError("xor-mix the int32 key columns over the last axis -> uint32")

    def repr(
        self,
        state: State,
        key_row: evoxc.SymbolicTensor,
        attr_row: evoxc.SymbolicTensor,
        precision: int = 2,
    ) -> str:
        """Render one gene as a human-readable string (host-only, never traced).

        Args:
            state: Host configuration container.
            key_row: ``(key_width,)`` int32 key columns of a single gene.
            attr_row: ``(attr_width,)`` float32 attribute columns of the same gene.
            precision: Decimal places for float attributes.

        Returns:
            A single-line description such as ``DefaultNode(idx=3, bias=0.25, ...)``.

        Raises:
            NotImplementedError: Skeleton stub.
        """
        raise NotImplementedError("format the key/attribute row of one gene as one line")

    def to_dict(
        self, state: State, key_row: evoxc.SymbolicTensor, attr_row: evoxc.SymbolicTensor
    ) -> dict[str, Any]:
        """Convert one gene to a plain Python/NumPy dict (host-only, never traced).

        Args:
            state: Host configuration container.
            key_row: ``(key_width,)`` int32 key columns of a single gene.
            attr_row: ``(attr_width,)`` float32 attribute columns of the same gene.

        Returns:
            Mapping of column name to scalar, e.g.
            ``{"idx": 3, "bias": 0.25, "agg": "sum", "act": "sigmoid"}``, consumed by
            the host-side ``network_dict`` / ``sympy_func`` code generation.

        Raises:
            NotImplementedError: Skeleton stub.
        """
        raise NotImplementedError(
            "map key/attribute columns to scalars using the fixed_attrs/attr_specs names"
        )
