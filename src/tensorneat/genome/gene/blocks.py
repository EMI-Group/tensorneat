"""Batched attribute-block operators: one draw family per genetic-operator stage.

A gene row is ``(*shape, attr_width)``: ``attr_width`` attribute columns that share a
batch shape (``(P, N)`` for nodes, ``(P, E)`` for connections).
The straight table-driven implementation walks the columns one by one -- one draw,
one ``clip`` and one ``select`` chain per column -- which is correct but pays the
interpreter's per-op overhead ``attr_width`` times, and ``attr_width`` is a
configuration constant, not a work measure.
The operators here draw **one column per participating spec** and then run **one**
rank-(d+1) ``select``/``clip`` cascade over the whole block, with the per-column
parameters (rates, bounds, replacement constants) as ``(*batch_shape, attr_width)``
static tables.
So the traced-op cost of a mutation pass is independent of ``attr_width``.

Two measured properties of the reference interpreter shape the code (both at
``P=1000``, ``max_nodes=50``, so a column is 50 k entries and a 4-column block 200 k):

* A ``evoxc.random`` draw is much cheaper per element below ~2**17 elements than
  above it -- a 200 k-element draw costs about nine 50 k-element draws -- so every
  draw stays **per column** and the block is assembled from columns, while the
  elementwise ops that combine them are cheap at any size.
* An elementwise op against a *broadcast* ``(1, ..., 1, attr_width)`` operand is
  several times slower than the same op against a fully materialised block operand
  (``less`` 0.26 vs 0.06 ms, ``clip`` 0.53 vs 0.21 ms), so a parameter table is
  materialised to the block shape once as a graph constant.

Behaviour is unchanged; what is hoisted is only the *shape* of the computation:

* A float column is ``N(init_mean, init_std)`` on creation; mutation adds
  ``N(0, mutate_power)`` with probability ``mutate_rate`` and, independently with
  probability ``replace_rate``, re-draws from the init distribution. The draws are
  made per column with the column's own key, so a column keeps an independent stream
  and every entry keeps an independent sample.
* An ``"int"`` column (an option index) is uniform over ``range(len(options))`` on
  creation and on a ``replace_rate`` hit, and never receives Gaussian noise.
* ``NaN`` (an absent gene row) is never written: the re-draw branch's mask is
  restricted to the live (non-``NaN``) entries, so mutation neither revives padding
  nor reorders it, and no attribute-block pass is spent restoring it afterwards.

Degenerate specs are simplified statically, which is exact rather than approximate:
a zero ``mutate_rate``/``replace_rate`` removes its branch (a probability-0 branch
cannot change a value), a single-option ``"int"`` column is the constant
``options[0]`` instead of a draw, and an ``init_std`` of ``0`` makes the
replacement ``N(mean, 0) = mean`` exactly.
Two further simplications are *distributional* rather than bit-level, and both leave
the law of every mutated entry unchanged:

* A column that both perturbs and re-draws its value takes **one** standard-normal
  sample ``Z`` per entry and derives ``noise = mutate_power * Z`` and
  ``replacement = init_mean + init_std * Z`` from it. A mutated entry is never a
  function of both (the gate picks one), so its marginal law is exactly the two
  independent draws' law; only the *joint* law of the two unselected candidates
  changes, and neither is observable.
* The scale/shift simplifications (``1 * Z`` is ``Z``, ``Z + 0`` is ``Z``) are exact
  in floating point, and an ``"int"`` column's uniform replacement is a
  ``randint``-over-options cast rather than a ``gather`` of the option table, which
  is the same discrete uniform because options are the contiguous indices
  ``0 .. len(options) - 1``.

When no spec can change at all, :func:`mutate_attrs` returns its input untouched.

Every parameter table is *static data* built on the host once per traced graph, so
it is created with ``evoxc.constant`` (never with eager NumPy on the compute path);
shapes are static and come from the traced tensors, and randomness is drawn from
``evoxc.random`` keys only.

Two further op cuts are value-identical rather than merely cheaper, and both are made
once per pass (:class:`_ScalarPool`, :class:`_ColumnPool`):

* ``evoxc`` never deduplicates constants and every RNG parameter (``low``/``high`` of
  ``uniform``, ``mean``/``std`` of ``normal``, ``low``/``high`` of ``randint``) is a
  fresh rank-0 constant when it is passed as a Python number, so a pass that draws one
  gate per column bakes the same ``0.0``/``1.0`` pair once per column, three ops per
  draw. Handing ``evoxc.random`` a *pooled* constant instead costs one op per draw and
  the operand's value and dtype are unchanged (the pool builds the same ``float64``
  scalar ``_param`` would), so every drawn number is bit-identical.
* A per-column table whose entries are all equal becomes one pooled rank-0 constant,
  and a broadcast ``(*batch, 1)`` column that several specs share (two single-option
  ``"int"`` columns both re-draw the constant ``0.0``) is built once.
* A branch whose ``U[0, 1)`` cutoff is not positive can never be selected, so it is not
  traced at all and its operand slot borrows another column of the same shape: the
  selected ``select`` operand is the only consumer, and it is never reached.
"""

from typing import TYPE_CHECKING, Any, Sequence

import evoxc
import evoxc.numpy as enp
import numpy as np

from ...common.errors import ConfigError
from ..window import fill

if TYPE_CHECKING:  # pragma: no cover - import cycle guard, typing only
    from .base import GeneAttr

#: Probability threshold of a branch that can never fire (``U[0, 1)`` is never
#: below it), used as the perturbation cutoff of an ``"int"`` column.
_NEVER: float = -1.0


# --------------------------------------------------------------------------- #
# per-pass constant pools
# --------------------------------------------------------------------------- #


def _pooled(cache: dict, value: Any, dtype: Any) -> evoxc.SymbolicTensor:
    """Return the rank-0 ``dtype`` constant ``value``, building it at most once.

    ``evoxc`` emits one ``constant`` op per ``enp.full`` call and never deduplicates
    them, so a pass that hands ``evoxc.random`` the same ``0.0``/``1.0`` pair to every
    column's draw would pay two ops per draw. The cache is keyed by ``repr`` rather
    than by the number itself, because ``0.0 == -0.0`` compares (and hashes) equal and
    a value-keyed cache would silently hand out the wrong sign of zero.

    Args:
        cache: One per-dtype dict of one traced pass (never module-global: a traced
            tensor belongs to the builder that produced it).
        value: Python ``float`` or ``int`` baked into the trace.
        dtype: ``evoxc`` dtype of the constant.

    Returns:
        Rank-0 ``dtype`` tensor, the same object on every call with the same ``value``.
    """
    key = repr(value)
    tensor = cache.get(key)
    if tensor is None:
        tensor = enp.full((), value, dtype=dtype)
        cache[key] = tensor
    return tensor


class _ScalarPool:
    """Rank-0 constants of one traced block pass, one per distinct dtype and value.

    Attributes:
        _f64 (dict): ``repr`` of a float to a rank-0 ``float64`` constant -- the dtype
            ``evoxc.random``'s ``low``/``high``/``mean``/``std`` parameters take when
            they are passed as Python floats.
        _f32 (dict): ``repr`` of a float to a rank-0 ``float32`` constant, the dtype a
            scalar operand of ``multiply``/``add``/``clip`` is weak-promoted to.
        _i64 (dict): ``int`` to a rank-0 ``int64`` constant, the dtype of
            ``randint``'s ``low``/``high``.
    """

    __slots__ = ("_f64", "_f32", "_i64")

    def __init__(self) -> None:
        """Start an empty pool for one traced block pass."""
        self._f64: dict[str, evoxc.SymbolicTensor] = {}
        self._f32: dict[str, evoxc.SymbolicTensor] = {}
        self._i64: dict[str, evoxc.SymbolicTensor] = {}

    def f64(self, value: float) -> evoxc.SymbolicTensor:
        """Return the pooled rank-0 ``float64`` constant ``value``."""
        return _pooled(self._f64, float(value), evoxc.float64)

    def f32(self, value: float) -> evoxc.SymbolicTensor:
        """Return the pooled rank-0 ``float32`` constant ``value``."""
        return _pooled(self._f32, float(value), evoxc.float32)

    def i64(self, value: int) -> evoxc.SymbolicTensor:
        """Return the pooled rank-0 ``int64`` constant ``value``."""
        return _pooled(self._i64, int(value), evoxc.int64)


class _ColumnPool:
    """``(*batch_shape, 1)`` broadcast columns of one pass, one per distinct value.

    Attributes:
        _shape (tuple): ``(*batch_shape, 1)`` shape every column of the pass has.
        _columns (dict): ``repr`` of a float to the ``..window.fill`` column holding it.
    """

    __slots__ = ("_shape", "_columns")

    def __init__(self, column_shape: tuple[int, ...]) -> None:
        """Start an empty pool for the given column shape."""
        self._shape: tuple[int, ...] = tuple(int(size) for size in column_shape)
        self._columns: dict[str, evoxc.SymbolicTensor] = {}

    def column(self, value: float) -> evoxc.SymbolicTensor:
        """Return the pooled ``column_shape`` column holding ``value``."""
        value = float(value)
        key = repr(value)
        column = self._columns.get(key)
        if column is None:
            column = column_constant(value, self._shape)
            self._columns[key] = column
        return column


def _uniform_value(values: Sequence[float]) -> float | None:
    """The single value every entry of a per-column table holds, or ``None``.

    Args:
        values: One Python float per attribute column.

    Returns:
        ``values[0]`` when every entry is *bit-wise* the same number (compared through
        ``repr``, so ``0.0`` and ``-0.0`` -- and two NaNs -- are not collapsed),
        otherwise ``None``.
    """
    first = repr(float(values[0]))
    for value in values[1:]:
        if repr(float(value)) != first:
            return None
    return float(values[0])


# --------------------------------------------------------------------------- #
# static operands
# --------------------------------------------------------------------------- #


def static_row(values: Sequence[float]) -> evoxc.SymbolicTensor:
    """Build one ``(len(values),)`` float32 row from host-side constants.

    A gene attribute row whose every column is a configuration constant (an
    identity/zero row for a node split or a new connection) is *static data*, not a
    computation: one ``evoxc.constant`` replaces the former one-``full``-op-plus-
    one-``stack``-op-per-column construction, and the value of every column is a
    Python float that cannot depend on the trace.

    Args:
        values: One Python float per attribute column, in ``attr_specs`` order.

    Returns:
        ``(len(values),)`` float32 row; an empty row when the family has no
        evolvable attributes.
    """
    if not values:
        return enp.zeros((0,), dtype=evoxc.float32)
    return evoxc.constant(evoxc.from_numpy(np.asarray(values, dtype=np.float32)))


def column_constant(value: float, column_shape: tuple[int, ...]) -> evoxc.SymbolicTensor:
    """Broadcast one scalar to a full attribute column.

    One element plus ``broadcast_to`` (``..window.fill``) rather than a materialised
    ``enp.full``: every consumer of a column here (``concatenate`` in :func:`_assemble`,
    ``clip``, ``select``) materialises the block it needs anyway, so a materialised
    column would only bake ``prod(column_shape)`` bytes into the graph — ``(P, 50, 1)``
    float32 is 200 KB at ``P=1000``, per column, per graph.

    Args:
        value: The constant every entry of the column takes.
        column_shape: ``(*batch_shape, 1)`` shape of one attribute column.

    Returns:
        ``column_shape`` float32 tensor holding ``value``.
    """
    return fill(tuple(column_shape), float(value), dtype=evoxc.float32)


def _parameter_table(
    values: Sequence[float], block_shape: tuple[int, ...], scalars: "_ScalarPool"
) -> evoxc.SymbolicTensor:
    """Materialise a per-column parameter table at the full block shape.

    Per-column parameters (the clip bounds; the mutation cutoffs compare per column
    instead, see :func:`_mask_block`) differ only along the attribute axis, but a
    *broadcast* ``(1, ..., 1, attr_width)`` operand is several times slower to consume
    in the reference interpreter than a materialised block, so the table is expanded
    once as a graph constant.
    The values are host constants, hence ``evoxc.constant``.

    When every column shares the same value (the only possibility for a single-column
    gene, and the common case for a whole family's bounds) the table carries no
    per-column information at all: a pooled rank-0 constant broadcasts to exactly the
    same operand values in one op instead of a ``constant`` plus a ``broadcast``, and
    bakes four bytes instead of ``prod(block_shape)``.

    Args:
        values: One Python float per attribute column, in ``attr_specs`` order.
        block_shape: ``(*batch_shape, attr_width)``, i.e. ``attrs.shape``.
        scalars: The pass's rank-0 constant pool.

    Returns:
        ``block_shape`` float32 tensor holding ``values`` along the attribute axis.
    """
    shared = _uniform_value(values)
    if shared is not None:
        return scalars.f32(shared)
    row = np.asarray(values, dtype=np.float32).reshape((1,) * (len(block_shape) - 1) + (len(values),))
    return enp.broadcast_to(evoxc.constant(evoxc.from_numpy(row)), tuple(block_shape))


def _assemble(
    columns: Sequence[evoxc.SymbolicTensor], width: int
) -> evoxc.SymbolicTensor:
    """Join one column per attribute spec into the ``(*batch, width)`` block.

    Args:
        columns: ``width`` tensors of shape ``(*batch_shape, 1)``, in column order.
        width: Number of attribute columns (``len(columns)``).

    Returns:
        ``(*batch_shape, width)`` float32 block; a single-column family is its own
        block, so no join op is traced for it.
    """
    if width == 1:
        return columns[0]
    return evoxc.concatenate(list(columns), axis=len(columns[0].shape) - 1)


def _mask_block(
    columns: Sequence[evoxc.SymbolicTensor],
    cutoffs: Sequence[float],
    scalars: "_ScalarPool",
) -> evoxc.SymbolicTensor:
    """Select the entries whose gate falls below their own column's cutoff.

    One ``less`` per attribute column against a **rank-0** pooled cutoff, then one
    join of the mask columns -- rather than one join of the gates plus a single
    ``less`` against a ``(*batch, attr_width)`` cutoff table. Both forms compare each
    gate against the same ``float32`` constant (the table stores ``float32(cutoff)``,
    exactly the pooled scalar's value), so every mask entry is bit-identical.

    The per-column form is the one that pays: in the reference interpreter a
    comparison whose second operand is a stride-0 broadcast of a rank-0 constant is
    3-4x slower per element than the same comparison against a bare rank-0 scalar
    (measured inside the ``execute_mutation`` graph at ``(500, 50, 4)``: 0.068 ms per
    op against the broadcast table versus 0.010 ms per column against the scalar),
    which more than covers the extra joins, and it bakes no ``(*batch, attr_width)``
    graph constant.

    Args:
        columns: One ``(*batch_shape, 1)`` gate per attribute column, in column order.
        cutoffs: One probability cutoff per attribute column, same order.
        scalars: The pass's rank-0 constant pool.

    Returns:
        ``(*batch_shape, attr_width)`` bool mask; a single-column family's mask is
        its own column, so no join op is traced for it.
    """
    masks = [
        evoxc.less(column, scalars.f32(cutoff))
        for column, cutoff in zip(columns, cutoffs)
    ]
    if len(masks) == 1:
        return masks[0]
    return evoxc.concatenate(masks, axis=len(masks[0].shape) - 1)


def _resolved(
    columns: Sequence[evoxc.SymbolicTensor | None], pool: "_ColumnPool"
) -> list[evoxc.SymbolicTensor]:
    """Give every unobservable operand slot a column of the block's column shape.

    A slot whose branch mask is statically empty never reaches the ``select`` that
    would consume it, so it needs *some* tensor of the right shape and nothing else;
    borrowing the pass's pooled zero column keeps one broadcast for all of them.

    Args:
        columns: One column (or ``None`` for "unobservable") per attribute spec.
        pool: The pass's column pool.

    Returns:
        One concrete ``(*batch_shape, 1)`` column per spec, in column order.
    """
    return [pool.column(0.0) if column is None else column for column in columns]


def _clip_bounds(
    specs: Sequence["GeneAttr"],
) -> tuple[list[float], list[float]] | None:
    """Collect the per-column inclusive clip bounds of a spec table.

    Args:
        specs: The family's attribute specs, in column order.

    Returns:
        ``(lower, upper)`` as one Python float list per column, with ``-inf``/``inf``
        for a column without a bound (and for every ``"int"`` column, which is an
        option index and is never clipped); ``None`` when no column is bounded, so
        an unbounded family traces no ``clip`` at all.
    """
    lower: list[float] = []
    upper: list[float] = []
    bounded = False
    for spec in specs:
        low = spec.lower_bound if spec.kind == "float" else None
        high = spec.upper_bound if spec.kind == "float" else None
        lower.append(-np.inf if low is None else float(low))
        upper.append(np.inf if high is None else float(high))
        bounded = bounded or low is not None or high is not None
    return (lower, upper) if bounded else None


def _common_window(specs: Sequence["GeneAttr"]) -> tuple[float, float] | None:
    """The one ``(low, high)`` window that clips every column, or ``None``.

    A per-column bound table costs a ``constant`` and a ``broadcast`` op; when one
    window serves the whole block the two bounds are pooled rank-0 constants instead.
    Two cases make that *exact* rather than merely cheaper:

    * a ``"float"`` column must carry exactly the window's bounds -- nothing is known
      about the values it can take, and a finite window would clip an unbounded column;
    * an ``"int"`` column holds an option index in ``0 .. len(options) - 1`` (that is
      the column's contract), so the window leaves it untouched whenever that whole
      range is inside the window. The per-column table leaves an ``"int"`` column
      unclipped, and clipping to a window that contains every reachable value is the
      identity.

    Args:
        specs: The family's attribute specs, in column order.

    Returns:
        ``(low, high)`` when one window clips the whole block, else ``None`` (the
        caller falls back to the per-column tables, which are always exact).
    """
    window: tuple[float, float] | None = None
    for spec in specs:
        if spec.kind != "float":
            continue
        if spec.lower_bound is None or spec.upper_bound is None:
            return None
        bounds = (float(spec.lower_bound), float(spec.upper_bound))
        if window is None:
            window = bounds
        elif window != bounds:  # also False for a NaN bound: falls back to the table
            return None
    if window is None:
        return None
    low, high = window
    for spec in specs:
        if spec.kind != "int":
            continue
        if low > 0.0 or float(len(spec.options) - 1) > high:
            return None
    return window


def _clip_block(
    block: evoxc.SymbolicTensor,
    specs: Sequence["GeneAttr"],
    scalars: "_ScalarPool",
) -> evoxc.SymbolicTensor:
    """Clip the ``(*batch, attr_width)`` attribute block into its bounds.

    Args:
        block: The attribute block to clip.
        specs: The family's attribute specs, in column order.
        scalars: The pass's rank-0 constant pool.

    Returns:
        ``block`` clipped per column -- unchanged (and no op traced) when no column
        of the family declares a bound.
    """
    bounds = _clip_bounds(specs)
    if bounds is None:
        return block
    window = _common_window(specs)
    if window is not None:
        low, high = scalars.f32(window[0]), scalars.f32(window[1])
    else:
        lower, upper = bounds
        block_shape = tuple(block.shape)
        low = _parameter_table(lower, block_shape, scalars)
        high = _parameter_table(upper, block_shape, scalars)
    return enp.clip(block, low, high)


# --------------------------------------------------------------------------- #
# per-column draws
# --------------------------------------------------------------------------- #


def _initial_column(
    spec: "GeneAttr",
    key: evoxc.SymbolicTensor,
    column_shape: tuple[int, ...],
    scalars: "_ScalarPool",
    columns: "_ColumnPool",
) -> evoxc.SymbolicTensor:
    """Draw the initial value of one attribute column.

    A float column is ``N(init_mean, init_std)``; an ``init_std`` of ``0`` makes
    that draw the exact constant ``init_mean``, so no random work is traced for it
    (``response_init_std`` is ``0`` in the default node gene).
    An ``"int"`` column is a uniform option index, i.e. the exact constant
    ``options[0]`` when only one option is configured, otherwise a ``randint`` cast
    into the float attribute layout.

    Args:
        spec: The column's attribute spec.
        key: ``evoxc.random`` key of this column (one per column, so every column
            has an independent stream).
        column_shape: ``(*batch_shape, 1)`` shape of the column.
        scalars: The pass's rank-0 constant pool (``low``/``high``/``mean``/``std``).
        columns: The pass's column pool, so two single-option columns that re-draw the
            same constant share one broadcast.

    Returns:
        ``column_shape`` float32 column.
    """
    if spec.kind == "int":
        count = len(spec.options)
        if count == 1:
            return columns.column(float(spec.options[0]))
        index = evoxc.random.randint(
            key, column_shape, scalars.i64(0), scalars.i64(count), dtype=evoxc.int32
        )
        return evoxc.cast(index, evoxc.float32)
    if float(spec.init_std) == 0.0:
        return columns.column(float(spec.init_mean))
    return evoxc.random.normal(
        key,
        column_shape,
        scalars.f64(float(spec.init_mean)),
        scalars.f64(float(spec.init_std)),
        dtype=evoxc.float32,
    )


def _replacement_column(
    spec: "GeneAttr",
    key: evoxc.SymbolicTensor,
    column_shape: tuple[int, ...],
    scalars: "_ScalarPool",
    columns: "_ColumnPool",
) -> evoxc.SymbolicTensor:
    """Draw the ``replace_rate`` re-draw of one attribute column.

    Same distribution as :func:`_initial_column` (a float column re-draws from its
    init distribution, an ``"int"`` column re-draws uniformly from its options), so
    the two share one implementation.

    Args:
        spec: The column's attribute spec.
        key: ``evoxc.random`` key of this column's replacement stream.
        column_shape: ``(*batch_shape, 1)`` shape of the column.
        scalars: The pass's rank-0 constant pool.
        columns: The pass's column pool.

    Returns:
        ``column_shape`` float32 column.
    """
    return _initial_column(spec, key, column_shape, scalars, columns)


def _scaled(
    sample: evoxc.SymbolicTensor, scale: float, scalars: "_ScalarPool"
) -> evoxc.SymbolicTensor:
    """Scale a standard-normal column (``1 * z`` is ``z``, exactly).

    The scale is a pooled ``float32`` constant: an elementwise op weak-promotes a
    Python scalar to its other operand's dtype, so the pooled operand is the same
    ``float32`` value the promotion would have built, one op cheaper.

    Args:
        sample: ``(*batch_shape, 1)`` standard-normal columns.
        scale: Standard deviation to attach.
        scalars: The pass's rank-0 constant pool.

    Returns:
        ``(*batch_shape, 1)`` float32 column ``scale * sample``.
    """
    if float(scale) == 1.0:
        return sample
    return evoxc.multiply(sample, scalars.f32(scale))


def _affine(
    sample: evoxc.SymbolicTensor, mean: float, std: float, scalars: "_ScalarPool"
) -> evoxc.SymbolicTensor:
    """Turn a standard-normal column into ``N(mean, std)`` (``+ 0``/``* 1`` are exact).

    Args:
        sample: ``(*batch_shape, 1)`` standard-normal columns.
        mean: Mean of the wanted Gaussian.
        std: Standard deviation of the wanted Gaussian.
        scalars: The pass's rank-0 constant pool.

    Returns:
        ``(*batch_shape, 1)`` float32 column ``mean + std * sample``.
    """
    scaled = _scaled(sample, std, scalars)
    if float(mean) == 0.0:
        return scaled
    return evoxc.add(scaled, scalars.f32(mean))


def _perturbs(spec: "GeneAttr") -> bool:
    """Whether a spec can apply its ``mutate_rate`` additive perturbation.

    Args:
        spec: The column's attribute spec.

    Returns:
        True for a float column with a positive rate and a positive power; an
        ``"int"`` column is never perturbed and a zero power/rate makes the branch
        unobservable, so both are skipped statically.
    """
    return (
        spec.kind == "float"
        and float(spec.mutate_rate) > 0.0
        and float(spec.mutate_power) != 0.0
    )


def _replaces(spec: "GeneAttr") -> bool:
    """Whether a spec can re-draw its value at ``replace_rate``.

    Args:
        spec: The column's attribute spec.

    Returns:
        True for a positive ``replace_rate``.
    """
    return float(spec.replace_rate) > 0.0


def _cutoffs(specs: Sequence["GeneAttr"]) -> tuple[list[float], list[float]]:
    """Per-column probability cutoffs of the ``U[0, 1)`` mutation gate.

    The legacy three-way rule compares one draw per entry against two cutoffs:
    ``draw < mutate_rate`` takes the perturbed value and ``draw < mutate_rate +
    replace_rate`` the re-drawn one (floats), while an ``"int"`` column only ever
    replaces (``draw < replace_rate``).
    Both bounds are broadcast over the block here, so one comparison per cutoff
    covers every column.

    Args:
        specs: The family's attribute specs, in column order.

    Returns:
        ``(perturb, replace)`` cutoffs as one Python float list per column.
    """
    perturb: list[float] = []
    replace: list[float] = []
    for spec in specs:
        if spec.kind == "int":
            perturb.append(_NEVER)
            replace.append(float(spec.replace_rate))
        else:
            rate = float(spec.mutate_rate)
            perturb.append(rate)
            replace.append(rate + float(spec.replace_rate))
    return perturb, replace


# --------------------------------------------------------------------------- #
# operators
# --------------------------------------------------------------------------- #


def init_attrs(
    key: evoxc.SymbolicTensor,
    shape: tuple[int, ...],
    specs: Sequence["GeneAttr"],
) -> evoxc.SymbolicTensor:
    """Draw initial attribute rows for a batch of newly created genes.

    One column per spec, each from its own ``evoxc.random`` stream (the rank-0 key
    is split once, never per row), joined into one block and clipped once against
    the per-column bounds.

    Args:
        key: Rank-0 ``evoxc.random`` key for the whole batch.
        shape: Batch shape of the genes, e.g. ``(P,)`` or ``(P, n_new)``.
        specs: The family's attribute specs, in column order.

    Returns:
        ``(*shape, len(specs))`` float32 attribute rows.
    """
    shape = tuple(int(size) for size in shape)
    if not specs:
        return enp.zeros((*shape, 0), dtype=evoxc.float32)
    column_shape = (*shape, 1)
    scalars = _ScalarPool()
    pool = _ColumnPool(column_shape)
    keys = evoxc.random.split_n(key, len(specs))
    columns = [
        _initial_column(spec, column_key, column_shape, scalars, pool)
        for spec, column_key in zip(specs, keys)
    ]
    block = _assemble(columns, len(specs))
    return _clip_block(block, specs, scalars)


def mutate_attrs(
    key: evoxc.SymbolicTensor,
    attrs: evoxc.SymbolicTensor,
    specs: Sequence["GeneAttr"],
) -> evoxc.SymbolicTensor:
    """Mutate attribute rows of the genes they belong to.

    Per column: add ``N(0, mutate_power)`` noise with probability ``mutate_rate``
    (float only), then with probability ``replace_rate`` re-draw from the init
    distribution (float) or uniformly from ``options`` (int); floats are clipped
    into their bounds. The gate is drawn per column, so each column's mask is one
    rank-0 comparison (:func:`_mask_block`) and the two ``select`` ops that resolve
    the branches cover the whole attribute block; an absent gene row (``NaN``) is
    never written, because the re-draw mask is folded together with the liveness
    mask -- mutation never revives padding.

    Two of the three branches are dropped statically when no column can reach them:
    a cutoff that is not positive is never above ``U[0, 1)`` (so the branch's mask is
    all-``False`` and the ``select`` it feeds is the identity), and the liveness mask
    only protects the re-drawn branch, because a perturbed entry is ``NaN + x``
    -- still ``NaN``, and the clip propagates ``NaN``. The draws are unchanged either
    way: the key split is fixed by ``attr_width`` and a column whose gate or candidate
    cannot be observed traces no draw, exactly like the full form.

    Args:
        key: Rank-0 ``evoxc.random`` key for the whole batch.
        attrs: Attribute rows, ``(*shape, attr_width)`` float32.
        specs: The family's attribute specs, in column order.

    Returns:
        ``(*shape, attr_width)`` float32 mutated rows.

    Raises:
        tensorneat.common.errors.ConfigError: If ``attrs`` does not have one column
            per spec (the block ops would otherwise silently resize the payload).
    """
    width = len(specs)
    if width == 0:
        return attrs
    shape = tuple(attrs.shape)
    if int(shape[-1]) != width:
        raise ConfigError(
            f"mutate: expected {width} attribute columns, got shape {shape}"
        )
    perturb_flags = [_perturbs(spec) for spec in specs]
    replace_flags = [_replaces(spec) for spec in specs]
    if not any(perturb_flags) and not any(replace_flags):
        return attrs
    column_shape = (*shape[:-1], 1)
    scalars = _ScalarPool()
    pool = _ColumnPool(column_shape)
    perturb_cutoff, replace_cutoff = _cutoffs(specs)
    # ``U[0, 1)`` is never below a non-positive cutoff, so such a column's branch can
    # never be selected; and the perturb branch wins wherever both masks fire, so a
    # replacement is unobservable once its cutoff reaches the perturb one.
    perturb_live = [cutoff > 0.0 for cutoff in perturb_cutoff]
    replace_live = [
        cutoff > 0.0 and cutoff > perturb_cutoff[index]
        for index, cutoff in enumerate(replace_cutoff)
    ]
    # One split for every draw of the pass, and every draw is *per column* (see the
    # module docstring: a block-sized draw is super-linear in the reference
    # interpreter). A rank-0 key is never split per row.
    keys = evoxc.random.split_n(key, 3 * width)
    replacements: list[evoxc.SymbolicTensor | None] = []
    noises: list[evoxc.SymbolicTensor | None] = []
    gates = []
    for index, spec in enumerate(specs):
        perturb = perturb_flags[index]
        replace = replace_flags[index]
        gaussian_replace = replace and spec.kind == "float" and float(spec.init_std) > 0.0
        if perturb and gaussian_replace:
            # Both candidates are affine in one standard-normal sample: the entry
            # ends up on one of them, so the pair's joint dependence is unobservable
            # while its marginal law is the two-independent-draws law.
            sample = evoxc.random.normal(
                keys[index],
                column_shape,
                scalars.f64(0.0),
                scalars.f64(1.0),
                dtype=evoxc.float32,
            )
            noises.append(_scaled(sample, float(spec.mutate_power), scalars))
            replacements.append(
                _affine(sample, float(spec.init_mean), float(spec.init_std), scalars)
            )
        else:
            if perturb:
                noises.append(
                    evoxc.random.normal(
                        keys[width + index],
                        column_shape,
                        scalars.f64(0.0),
                        scalars.f64(float(spec.mutate_power)),
                        dtype=evoxc.float32,
                    )
                )
            elif perturb_live[index]:
                # The gate can fire while the power is ``0``: the perturbation is then
                # exactly ``attrs + 0``, so this slot needs a real zero.
                noises.append(pool.column(0.0))
            else:
                noises.append(None)  # statically never selected: borrow a column
            replacements.append(
                _replacement_column(spec, keys[index], column_shape, scalars, pool)
                if replace
                else None
            )
        if perturb or replace:
            gates.append(
                evoxc.random.uniform(
                    keys[2 * width + index],
                    column_shape,
                    scalars.f64(0.0),
                    scalars.f64(1.0),
                    dtype=evoxc.float32,
                )
            )
        else:
            # Nothing on this column can change, so the gate is not drawn (the draw
            # would never be consulted) and a constant stands in for it.
            gates.append(pool.column(0.0))
    mutated = attrs
    if any(replace_live):
        # The absent-row guard is folded into the re-draw mask instead of a
        # trailing ``select(isnan(attrs), attrs, mutated)``: a padded row is then
        # never written, which is the same value as restoring it afterwards, and
        # it costs one ``logical_and`` on the mask instead of a full attribute
        # block pass. Only this branch can revive a NaN row -- a perturbed entry
        # is ``NaN + x``, which stays NaN, and every other entry is ``attrs``.
        mutated = evoxc.select(
            evoxc.logical_and(
                _mask_block(gates, replace_cutoff, scalars),
                evoxc.equal(attrs, attrs),
            ),
            _assemble(_resolved(replacements, pool), width),
            attrs,
        )
    if any(perturb_live):
        for index in range(width):
            if noises[index] is None and replacements[index] is not None:
                noises[index] = replacements[index]
        mutated = evoxc.select(
            _mask_block(gates, perturb_cutoff, scalars),
            evoxc.add(attrs, _assemble(_resolved(noises, pool), width)),
            mutated,
        )
    return _clip_block(mutated, specs, scalars)


def crossover_attrs(
    key: evoxc.SymbolicTensor,
    attrs1: evoxc.SymbolicTensor,
    attrs2: evoxc.SymbolicTensor,
) -> evoxc.SymbolicTensor:
    """Recombine two parents' attribute rows with a per-entry 50/50 pick.

    One ``U[0, 1)`` block draw of the attribute shape, one comparison against
    ``0.5``, one ``select``: the entries are independent Bernoulli(0.5) draws, which
    is the documented per-attribute rule (and the cheapest way to express it --
    a Gaussian draw is unnecessary for a two-way pick and costs twice the random
    work per entry).

    Args:
        key: Rank-0 ``evoxc.random`` key for the whole batch.
        attrs1: Attribute rows of the first parent, ``(*shape, attr_width)``.
        attrs2: Key-aligned attribute rows of the second parent, same shape.

    Returns:
        ``(*shape, attr_width)`` float32 child attribute rows; a gene with no
        attribute column (``attr_width == 0``) has nothing to recombine, so the
        first parent's empty rows are returned without tracing a draw.
    """
    shape = tuple(attrs1.shape)
    if shape and int(shape[-1]) == 0:  # no attribute column to recombine
        return attrs1
    # The numpy reference backend rejects a 0-d random output, so a rank-0 row is
    # drawn as shape ``(1,)`` and reshaped back (see ``..base.BaseGene``).
    draw = evoxc.random.uniform(key, shape or (1,), 0.0, 1.0, dtype=evoxc.float32)
    if not shape:
        draw = evoxc.reshape(draw, ())
    return evoxc.select(evoxc.less(draw, 0.5), attrs1, attrs2)


__all__ = [
    "column_constant",
    "crossover_attrs",
    "init_attrs",
    "mutate_attrs",
    "static_row",
]
