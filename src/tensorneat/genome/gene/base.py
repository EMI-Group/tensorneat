"""Gene base classes: the attr-table contract shared by node and conn genes.

A gene row is split into integer *key columns* (``fixed_attrs``) and float32
*attribute columns* (``attr_specs``); every gene operation is batch-first with the
population on axis 0 and shapes that are static at trace time.
This module owns the table (specifications, widths, row hash), the generic genetic
operators (initialisation / mutation / crossover / distance) and the split-payload
assembly helpers: it must never import ``jax``/``torch``, never run eager NumPy on
the compute path, and never implement node/connection semantics (those live in
``node/`` and ``conn/``).
"""

from dataclasses import dataclass
from typing import Any, Sequence

import evoxc
import evoxc.numpy as enp

from ...common import State, StatefulBaseClass
from ...common.errors import ConfigError
from ...common.tools import mutate_float, mutate_int, to_numpy

#: Fold constants of the key-row hash; identical to
#: :func:`tensorneat.common.tools.hash_rows` so an in-graph hash and the host-side
#: legacy hash agree on the same key rows.
HASH_GOLDEN: int = 0x9E3779B9
HASH_SHIFT_LEFT: int = 6
HASH_SHIFT_RIGHT: int = 2


def column(tensor: evoxc.SymbolicTensor, index: int) -> evoxc.SymbolicTensor:
    """Read one static column of the last axis of a traced tensor.

    evoxc's static indexing rejects ``Ellipsis``/``None``, so ``tensor[..., i]`` is
    not available inside a graph; this builds the explicit key for the tensor's
    actual rank. The slice keeps the indexed axis (``i:i+1``) and the trailing
    ``reshape`` drops it, because the reverse-mode rule of an axis-dropping integer
    index has no usable cotangent shape -- a sliced-then-reshaped read is
    differentiable, which is what ``Genome.grad`` needs for every gene attribute.

    Args:
        tensor: Any traced tensor whose last axis is the gene column axis.
        index: Static (non-negative) column index.

    Returns:
        ``tensor.shape[:-1]``-shaped view of column ``index``.
    """
    rank = len(tensor.shape)
    slicer = (slice(None),) * (rank - 1) + (slice(index, index + 1),)
    return evoxc.reshape(tensor[slicer], tuple(tensor.shape[:-1]))


def hash_int32(keys: evoxc.SymbolicTensor) -> evoxc.SymbolicTensor:
    """Reduce the integer key columns of a gene row to one xor-mixed row hash.

    No bitcast between float and int exists in evoxc, so the hash is taken over the
    ``int32`` key columns only; float attributes are never hashed by bit pattern.
    The fold is the legacy one (``hash ^= word + 0x9E3779B9 + (hash << 6) + (hash >> 2)``,
    left to right) evaluated in ``uint32``, so the in-graph value equals
    :func:`tensorneat.common.tools.hash_rows` on the same key rows and equal key rows
    hash equal across runs.

    Args:
        keys: Key columns of one gene family, ``(*shape, key_width)`` int32.

    Returns:
        ``(*shape,)`` uint32 row hashes, one per gene row.
    """
    hashes = enp.zeros(tuple(keys.shape[:-1]), dtype=evoxc.uint32)
    width = int(keys.shape[-1])
    if width == 0:  # the empty fold: no word is mixed in
        return hashes
    golden = enp.full((), HASH_GOLDEN, dtype=evoxc.uint32)
    shift_left = enp.full((), HASH_SHIFT_LEFT, dtype=evoxc.uint32)
    shift_right = enp.full((), HASH_SHIFT_RIGHT, dtype=evoxc.uint32)
    for index in range(width):
        word = evoxc.cast(column(keys, index), evoxc.uint32)
        mixed = evoxc.add(
            evoxc.add(word, golden),
            evoxc.add(
                evoxc.bitwise_left_shift(hashes, shift_left),
                evoxc.bitwise_right_shift(hashes, shift_right),
            ),
        )
        hashes = evoxc.bitwise_xor(hashes, mixed)
    return hashes


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
    #: Display label per column name used by :meth:`repr`; families rename entries
    #: (``input_index`` -> ``in``) without touching the payload column order.
    labels: dict[str, str] = {}

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

    def setup(self, state: State | None = None) -> State:
        """Validate the attr table on the host.

        Gene configuration is constructor-level and read off ``self`` during a trace,
        so nothing is registered: the gene adds no ``State`` entries and returns the
        incoming state unchanged (a fresh empty one when ``None``). The table itself
        is checked here, once, before any graph is traced.

        Args:
            state: Base state to extend; never mutated in place.

        Returns:
            The same ``State`` (or ``State()`` when ``state`` is ``None``).

        Raises:
            tensorneat.common.errors.ConfigError: If an ``int`` spec has no
                ``options``, its options are not ``0..n-1``, a ``float`` spec carries
                ``options``, bounds are inverted, a rate falls outside ``[0, 1]``, or
                two columns share a name.
        """
        self._validate_specs()
        return State() if state is None else state

    def _validate_specs(self) -> None:
        """Check ``fixed_attrs`` / ``attr_specs`` for internal consistency.

        Raises:
            tensorneat.common.errors.ConfigError: See :meth:`setup`.
        """
        names = list(self.fixed_attrs) + [spec.name for spec in self.attr_specs]
        if len(set(names)) != len(names):
            raise ConfigError(f"{type(self).__name__} declares duplicate column names {names}")
        for spec in self.attr_specs:
            if spec.kind == "float":
                if spec.options is not None:
                    raise ConfigError(
                        f"float attribute {spec.name!r} must not declare options, "
                        f"got {spec.options!r}"
                    )
            elif spec.kind == "int":
                if not spec.options:
                    raise ConfigError(
                        f"int attribute {spec.name!r} needs a non-empty options tuple"
                    )
                expected = tuple(range(len(spec.options)))
                if tuple(spec.options) != expected:
                    raise ConfigError(
                        f"int attribute {spec.name!r} options must be the contiguous "
                        f"indices {expected}, got {tuple(spec.options)}"
                    )
            else:
                raise ConfigError(
                    f"attribute {spec.name!r} has unknown kind {spec.kind!r}; "
                    "expected 'float' or 'int'"
                )
            if (
                spec.lower_bound is not None
                and spec.upper_bound is not None
                and spec.lower_bound > spec.upper_bound
            ):
                raise ConfigError(
                    f"attribute {spec.name!r} lower_bound {spec.lower_bound} exceeds "
                    f"upper_bound {spec.upper_bound}"
                )
            for rate_name in ("mutate_rate", "replace_rate"):
                rate = getattr(spec, rate_name)
                if not 0.0 <= float(rate) <= 1.0:
                    raise ConfigError(
                        f"attribute {spec.name!r} {rate_name} must lie in [0, 1], got {rate}"
                    )

    # ------------------------------------------------------------------ #
    # column addressing
    # ------------------------------------------------------------------ #

    def attr_index(self, name: str) -> int:
        """Return the attribute-column index of ``name``.

        Args:
            name: A spec name declared in ``attr_specs``.

        Returns:
            The position in ``attr_specs``, i.e. the column of ``*_attrs``.

        Raises:
            tensorneat.common.errors.ConfigError: If no spec carries that name.
        """
        for index, spec in enumerate(self.attr_specs):
            if spec.name == name:
                return index
        raise ConfigError(f"{type(self).__name__} declares no attribute {name!r}")

    def key_index(self, name: str) -> int:
        """Return the key-column index of ``name``.

        Args:
            name: A name declared in ``fixed_attrs``.

        Returns:
            The position in ``fixed_attrs``, i.e. the column of ``*_keys``.

        Raises:
            tensorneat.common.errors.ConfigError: If no key column carries that name.
        """
        try:
            return self.fixed_attrs.index(name)
        except ValueError:
            raise ConfigError(
                f"{type(self).__name__} declares no key column {name!r}"
            ) from None

    def attr_column(self, attrs: evoxc.SymbolicTensor, name: str) -> evoxc.SymbolicTensor:
        """Read one named attribute column of a traced ``*_attrs`` tensor.

        Args:
            attrs: ``(*shape, attr_width)`` float32 attribute rows.
            name: Spec name of the column, resolved through ``attr_specs``.

        Returns:
            ``(*shape,)`` float32 column values.
        """
        return column(attrs, self.attr_index(name))

    def key_column(self, keys: evoxc.SymbolicTensor, name: str) -> evoxc.SymbolicTensor:
        """Read one named key column of a traced ``*_keys`` tensor.

        Args:
            keys: ``(*shape, key_width)`` int32 key rows.
            name: Key-column name, resolved through ``fixed_attrs``.

        Returns:
            ``(*shape,)`` int32 column values.
        """
        return column(keys, self.key_index(name))

    # ------------------------------------------------------------------ #
    # split-payload assembly
    # ------------------------------------------------------------------ #

    def _normalize_attrs(
        self,
        attrs: evoxc.SymbolicTensor | None,
        shape: Sequence[int],
        name: str,
    ) -> evoxc.SymbolicTensor:
        """Broadcast/validate the attribute part of a hand-built gene payload.

        Args:
            attrs: ``(*shape, attr_width)`` attribute rows (already split per column),
                a ``(*shape,)`` row value broadcast over the attribute axis, or
                ``None`` for "all attributes at their neutral value" (zeros).
            shape: Leading (batch) shape of the gene rows.
            name: Caller name for error messages.

        Returns:
            ``(*shape, attr_width)`` float32 attribute rows.

        Raises:
            tensorneat.common.errors.ConfigError: If ``attrs`` cannot be interpreted
                as one attribute row per gene.
        """
        shape = tuple(shape)
        if self.attr_width == 0:  # a family without evolvable attributes
            return enp.zeros((*shape, 0), dtype=evoxc.float32)
        if attrs is None:
            return enp.zeros((*shape, self.attr_width), dtype=evoxc.float32)
        attrs = evoxc.cast(attrs, evoxc.float32)
        if tuple(attrs.shape) == (*shape, self.attr_width):
            return attrs
        if tuple(attrs.shape) == shape:
            per_row = evoxc.reshape(attrs, (*shape, 1))
            return enp.broadcast_to(per_row, (*shape, self.attr_width))
        raise ConfigError(
            f"{name}: attribute rows must have shape {(*shape, self.attr_width)} or "
            f"{shape}, got {tuple(attrs.shape)}"
        )

    def _key_array(
        self,
        columns: Sequence[evoxc.SymbolicTensor],
        shape: Sequence[int],
        name: str,
    ) -> evoxc.SymbolicTensor:
        """Stack per-column key tensors into one ``(*shape, key_width)`` int32 array.

        Args:
            columns: One ``(*shape,)`` tensor per key column, in ``fixed_attrs`` order.
            shape: Leading (batch) shape of the gene rows.
            name: Caller name for error messages.

        Returns:
            ``(*shape, key_width)`` int32 key rows.

        Raises:
            tensorneat.common.errors.ConfigError: If the column count or a column
                shape does not match the declared key width.
        """
        if len(columns) != self.key_width:
            raise ConfigError(
                f"{name}: expected {self.key_width} key column(s) "
                f"{self.fixed_attrs}, got {len(columns)}"
            )
        shape = tuple(shape)
        reshaped = []
        for label, values in zip(self.fixed_attrs, columns):
            if tuple(values.shape) != shape:
                raise ConfigError(
                    f"{name}: key column {label!r} must have shape {shape}, "
                    f"got {tuple(values.shape)}"
                )
            reshaped.append(evoxc.reshape(evoxc.cast(values, evoxc.int32), (*shape, 1)))
        return evoxc.concatenate(reshaped, axis=-1)

    def from_attrs(
        self,
        state: State,
        keys: evoxc.SymbolicTensor,
        attrs: evoxc.SymbolicTensor | None = None,
    ) -> tuple[evoxc.SymbolicTensor, evoxc.SymbolicTensor]:
        """Assemble a split gene payload pair from pre-built columns.

        Args:
            state: Host configuration container.
            keys: ``(*shape, key_width)`` int32 key columns.
            attrs: ``(*shape, attr_width)`` float32 attribute columns, a
                ``(*shape,)`` row value broadcast over the attribute axis, or
                ``None`` for zeros.

        Returns:
            ``(keys, attrs)``: the int32 key array and the float32 attribute array of
            the same gene rows.
        """
        keys = evoxc.cast(keys, evoxc.int32)
        if int(keys.shape[-1]) != self.key_width:
            raise ConfigError(
                f"{type(self).__name__}.from_attrs: expected {self.key_width} key "
                f"columns {self.fixed_attrs}, got shape {tuple(keys.shape)}"
            )
        shape = tuple(keys.shape[:-1])
        return keys, self._normalize_attrs(attrs, shape, f"{type(self).__name__}.from_attrs")

    # ------------------------------------------------------------------ #
    # initialisation
    # ------------------------------------------------------------------ #

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
        """
        shape = tuple(int(size) for size in shape)
        if not self.attr_specs:
            return enp.zeros((*shape, 0), dtype=evoxc.float32)
        # A rank-0 random draw is rejected by the numpy reference backend (it builds
        # its output tensor from a NumPy scalar), so a single-gene batch is drawn as
        # shape ``(1,)`` and reshaped back to ``()``.
        draw_shape = shape or (1,)
        draw_keys = evoxc.random.split_n(key, len(self.attr_specs))
        columns = []
        for spec, draw_key in zip(self.attr_specs, draw_keys):
            if spec.kind == "float":
                values = evoxc.random.normal(
                    draw_key, draw_shape, spec.init_mean, spec.init_std, dtype=evoxc.float32
                )
                values = self._clip(spec, values)
            else:
                count = len(spec.options)
                values = evoxc.cast(
                    evoxc.random.randint(draw_key, draw_shape, 0, count, dtype=evoxc.int32),
                    evoxc.float32,
                )
            columns.append(evoxc.reshape(values, shape) if not shape else values)
        return evoxc.stack(columns, axis=-1)

    def new_identity_attrs(self, state: State) -> evoxc.SymbolicTensor:
        """Attribute row whose gene is an identity transform (host ``add_node`` split).

        The generic row is "neutral by construction": a float column takes its init
        mean (``bias = 0``, ``response = 1``) and an int column takes the first
        configured option. Families whose identity row is not the init distribution
        override this (a connection's identity weight is ``1``, not ``0``).

        Args:
            state: Host configuration container.

        Returns:
            ``(attr_width,)`` float32 row; it must leave the network output unchanged
            when it replaces a split connection.
        """
        values = []
        for spec in self.attr_specs:
            if spec.kind == "float":
                values.append(self._clip_scalar(spec, float(spec.init_mean)))
            else:
                values.append(float(self._default_option(spec)))
        return self._attr_row(values)

    def new_zero_attrs(self, state: State) -> evoxc.SymbolicTensor:
        """Attribute row with the least possible influence (host ``add_conn``).

        Every float column is ``0`` (a zero weight cannot disturb an existing signal)
        and an int column keeps the first configured option, so a brand-new gene is
        inert until mutation changes its attributes.

        Args:
            state: Host configuration container.

        Returns:
            ``(attr_width,)`` float32 row; a new edge built from it must not disturb
            existing behaviour until mutation changes its attributes.
        """
        values = []
        for spec in self.attr_specs:
            values.append(0.0 if spec.kind == "float" else float(self._default_option(spec)))
        return self._attr_row(values)

    def _default_option(self, spec: GeneAttr) -> int:
        """Option index an ``int`` spec takes in an identity/zero attribute row.

        Args:
            spec: An ``"int"`` spec.

        Returns:
            The first configured option index.
        """
        return int(spec.options[0])

    def _attr_row(self, values: Sequence[float]) -> evoxc.SymbolicTensor:
        """Build one ``(attr_width,)`` float32 attribute row from static values.

        Args:
            values: One Python float per attribute column, in ``attr_specs`` order.

        Returns:
            ``(attr_width,)`` float32 row; an empty row when the family has no
            evolvable attributes.
        """
        if not values:
            return enp.zeros((0,), dtype=evoxc.float32)
        scalars = [enp.full((), value, dtype=evoxc.float32) for value in values]
        return evoxc.stack(scalars, axis=-1)

    def _clip(self, spec: GeneAttr, values: evoxc.SymbolicTensor) -> evoxc.SymbolicTensor:
        """Clip a traced float column into the inclusive bounds of ``spec``.

        Args:
            spec: The float spec supplying the bounds.
            values: Traced values of that column.

        Returns:
            ``values`` clipped into ``[lower_bound, upper_bound]``; unchanged when the
            spec is unbounded.
        """
        if spec.lower_bound is None and spec.upper_bound is None:
            return values
        return enp.clip(values, spec.lower_bound, spec.upper_bound)

    @staticmethod
    def _clip_scalar(spec: GeneAttr, value: float) -> float:
        """Clip one host-side scalar into the inclusive bounds of ``spec``.

        Args:
            spec: The float spec supplying the bounds.
            value: Python float to clip.

        Returns:
            ``value`` clipped into ``[lower_bound, upper_bound]``.
        """
        if spec.lower_bound is not None:
            value = max(value, float(spec.lower_bound))
        if spec.upper_bound is not None:
            value = min(value, float(spec.upper_bound))
        return value

    # ------------------------------------------------------------------ #
    # genetic operators
    # ------------------------------------------------------------------ #

    def mutate(
        self, state: State, key: evoxc.SymbolicTensor, attrs: evoxc.SymbolicTensor
    ) -> evoxc.SymbolicTensor:
        """Mutate attribute rows of the genes they belong to.

        Per spec: add ``normal(0, mutate_power)`` noise with probability
        ``mutate_rate`` (float only), then with probability ``replace_rate`` re-draw
        from the init distribution (float) or uniformly from ``options`` (int);
        floats are clipped into their bounds. Selection is a ``select`` chain over the
        batched attrs -- no branching on traced values. A ``NaN`` column (absent gene
        row) stays ``NaN``: mutation never revives padding.

        Args:
            state: Host configuration container.
            key: Rank-0 ``evoxc.random`` key for the whole batch.
            attrs: Attribute rows, ``(*shape, attr_width)`` float32.

        Returns:
            ``(*shape, attr_width)`` float32 mutated rows.
        """
        if not self.attr_specs:
            return attrs
        draw_keys = evoxc.random.split_n(key, len(self.attr_specs))
        columns = []
        for index, (spec, draw_key) in enumerate(zip(self.attr_specs, draw_keys)):
            current = column(attrs, index)
            # A rank-0 row is drawn as shape ``(1,)``: the numpy reference backend
            # rejects a 0-d random output (see ``new_random_attrs``).
            batch_shape = tuple(current.shape)
            batched = evoxc.reshape(current, (1,)) if not batch_shape else current
            if spec.kind == "float":
                mutated = mutate_float(
                    draw_key,
                    batched,
                    spec.init_mean,
                    spec.init_std,
                    spec.mutate_power,
                    spec.mutate_rate,
                    spec.replace_rate,
                )
                mutated = self._clip(spec, mutated)
            else:
                options = enp.arange(len(spec.options), dtype=evoxc.int32)
                mutated = mutate_int(draw_key, batched, options, spec.replace_rate)
            if not batch_shape:
                mutated = evoxc.reshape(mutated, ())
            columns.append(evoxc.select(evoxc.isnan(current), current, mutated))
        return evoxc.stack(columns, axis=-1)

    def crossover(
        self,
        state: State,
        key: evoxc.SymbolicTensor,
        attrs1: evoxc.SymbolicTensor,
        attrs2: evoxc.SymbolicTensor,
    ) -> evoxc.SymbolicTensor:
        """Crossover the attributes of two homologous genes.

        Default behaviour is a per-attribute 50/50 pick (one normal draw per column
        entry, ``> 0`` selects the first parent); the origin families override it with
        a whole-gene pick (one draw selects the whole attribute vector).

        Args:
            state: Host configuration container.
            key: Rank-0 ``evoxc.random`` key for the whole batch.
            attrs1: Attribute rows of the first parent, ``(*shape, attr_width)``.
            attrs2: Key-aligned attribute rows of the second parent, same shape.

        Returns:
            ``(*shape, attr_width)`` float32 child attribute rows.
        """
        if not self.attr_specs:
            return attrs1
        shape = tuple(attrs1.shape)
        draw = evoxc.random.normal(
            key, shape or (1,), 0.0, 1.0, dtype=evoxc.float32
        )
        if not shape:  # the numpy backend rejects a 0-d random output
            draw = evoxc.reshape(draw, ())
        return evoxc.select(evoxc.greater(draw, 0.0), attrs1, attrs2)

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
        """
        total = enp.zeros(tuple(attrs1.shape[:-1]), dtype=evoxc.float32)
        for index, spec in enumerate(self.attr_specs):
            first, second = column(attrs1, index), column(attrs2, index)
            if spec.kind == "float":
                term = evoxc.abs(evoxc.subtract(first, second))
            else:
                term = evoxc.cast(evoxc.not_equal(first, second), evoxc.float32)
            total = evoxc.add(total, term)
        return total

    def hash(self, keys: evoxc.SymbolicTensor) -> evoxc.SymbolicTensor:
        """Hash gene key rows for fast key alignment in crossover/distance.

        Delegates to :func:`hash_int32`, an xor-mix over the ``int32`` key columns;
        float attributes are compared only after the keys align.

        Args:
            keys: Key columns of one gene family, ``(*shape, key_width)`` int32.

        Returns:
            ``(*shape,)`` uint32 row hashes.
        """
        return hash_int32(keys)

    # ------------------------------------------------------------------ #
    # host-only presentation
    # ------------------------------------------------------------------ #

    def _label(self, name: str) -> str:
        """Display label of a column name (``labels`` overrides the bare name).

        Args:
            name: Column name as declared in ``fixed_attrs``/``attr_specs``.

        Returns:
            The configured label, or ``name`` itself.
        """
        return self.labels.get(name, name)

    def _host_row(
        self, key_row: evoxc.SymbolicTensor, attr_row: evoxc.SymbolicTensor
    ) -> tuple[list[int], list[float]]:
        """Convert one gene's key and attribute rows to Python scalars.

        Args:
            key_row: ``(key_width,)`` int32 key columns of a single gene.
            attr_row: ``(attr_width,)`` float32 attribute columns of the same gene.

        Returns:
            ``(keys, attrs)`` as Python ``int``/``float`` lists, in column order.

        Raises:
            tensorneat.common.errors.ConfigError: If a row has the wrong width.
        """
        keys = to_numpy(key_row).reshape(-1)
        attrs = to_numpy(attr_row).reshape(-1)
        if keys.size != self.key_width:
            raise ConfigError(
                f"{type(self).__name__}: expected {self.key_width} key columns, "
                f"got {keys.size}"
            )
        if attrs.size != self.attr_width:
            raise ConfigError(
                f"{type(self).__name__}: expected {self.attr_width} attribute columns, "
                f"got {attrs.size}"
            )
        return [int(value) for value in keys], [float(value) for value in attrs]

    def repr(
        self,
        state: State,
        key_row: evoxc.SymbolicTensor,
        attr_row: evoxc.SymbolicTensor,
        precision: int = 2,
    ) -> str:
        """Render one gene as a human-readable string (host-only, never traced).

        The generic rendering lists every key column and every attribute column by
        name; families whose option indices have names (node genes) override it.

        Args:
            state: Host configuration container.
            key_row: ``(key_width,)`` int32 key columns of a single gene.
            attr_row: ``(attr_width,)`` float32 attribute columns of the same gene.
            precision: Decimal places for float attributes.

        Returns:
            A single-line description such as ``DefaultConn(in: 1, out: 4, weight: 0.5)``.
        """
        keys, attrs = self._host_row(key_row, attr_row)
        parts = [
            f"{self._label(name)}: {value}"
            for name, value in zip(self.fixed_attrs, keys)
        ]
        for spec, value in zip(self.attr_specs, attrs):
            label = self._label(spec.name)
            if spec.kind == "int":
                parts.append(f"{label}: {int(round(value))}")
            else:
                parts.append(f"{label}: {round(value, precision)}")
        return f"{type(self).__name__}({', '.join(parts)})"

    def to_dict(
        self, state: State, key_row: evoxc.SymbolicTensor, attr_row: evoxc.SymbolicTensor
    ) -> dict[str, Any]:
        """Convert one gene to a plain Python/NumPy dict (host-only, never traced).

        The generic conversion maps every key column and every attribute column to its
        declared name; families whose option indices have names (node genes) override
        it to resolve them.

        Args:
            state: Host configuration container.
            key_row: ``(key_width,)`` int32 key columns of a single gene.
            attr_row: ``(attr_width,)`` float32 attribute columns of the same gene.

        Returns:
            Mapping of column name to scalar, e.g. ``{"in": 1, "out": 4, "weight": 0.5}``,
            consumed by the host-side ``network_dict`` / ``sympy_func`` code generation.
        """
        keys, attrs = self._host_row(key_row, attr_row)
        result: dict[str, Any] = {
            self._label(name): value for name, value in zip(self.fixed_attrs, keys)
        }
        for spec, value in zip(self.attr_specs, attrs):
            result[spec.name] = int(round(value)) if spec.kind == "int" else value
        return result
