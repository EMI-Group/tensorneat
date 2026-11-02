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
* ``NaN`` (an absent gene row) is restored by a final ``isnan`` select, so mutation
  never revives padding and never reorders it.

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
"""

from typing import TYPE_CHECKING, Sequence

import evoxc
import evoxc.numpy as enp
import numpy as np

from ...common.errors import ConfigError
from ..window import fill
from ..window import fill

if TYPE_CHECKING:  # pragma: no cover - import cycle guard, typing only
    from .base import GeneAttr

#: Probability threshold of a branch that can never fire (``U[0, 1)`` is never
#: below it), used as the perturbation cutoff of an ``"int"`` column.
_NEVER: float = -1.0


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
    values: Sequence[float], block_shape: tuple[int, ...]
) -> evoxc.SymbolicTensor:
    """Materialise a per-column parameter table at the full block shape.

    Per-column parameters (mutation cutoffs, clip bounds) differ only along the
    attribute axis, but a *broadcast* ``(1, ..., 1, attr_width)`` operand is several
    times slower to consume in the reference interpreter than a materialised block,
    so the table is expanded once as a graph constant.
    The values are host constants, hence ``evoxc.constant``.

    Args:
        values: One Python float per attribute column, in ``attr_specs`` order.
        block_shape: ``(*batch_shape, attr_width)``, i.e. ``attrs.shape``.

    Returns:
        ``block_shape`` float32 tensor holding ``values`` along the attribute axis.
    """
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


# --------------------------------------------------------------------------- #
# per-column draws
# --------------------------------------------------------------------------- #


def _initial_column(
    spec: "GeneAttr", key: evoxc.SymbolicTensor, column_shape: tuple[int, ...]
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

    Returns:
        ``column_shape`` float32 column.
    """
    if spec.kind == "int":
        count = len(spec.options)
        if count == 1:
            return column_constant(float(spec.options[0]), column_shape)
        index = evoxc.random.randint(key, column_shape, 0, count, dtype=evoxc.int32)
        return evoxc.cast(index, evoxc.float32)
    if float(spec.init_std) == 0.0:
        return column_constant(float(spec.init_mean), column_shape)
    return evoxc.random.normal(
        key, column_shape, spec.init_mean, spec.init_std, dtype=evoxc.float32
    )


def _replacement_column(
    spec: "GeneAttr", key: evoxc.SymbolicTensor, column_shape: tuple[int, ...]
) -> evoxc.SymbolicTensor:
    """Draw the ``replace_rate`` re-draw of one attribute column.

    Same distribution as :func:`_initial_column` (a float column re-draws from its
    init distribution, an ``"int"`` column re-draws uniformly from its options), so
    the two share one implementation.

    Args:
        spec: The column's attribute spec.
        key: ``evoxc.random`` key of this column's replacement stream.
        column_shape: ``(*batch_shape, 1)`` shape of the column.

    Returns:
        ``column_shape`` float32 column.
    """
    return _initial_column(spec, key, column_shape)


def _scaled(
    sample: evoxc.SymbolicTensor, scale: float
) -> evoxc.SymbolicTensor:
    """Scale a standard-normal column (``1 * z`` is ``z``, exactly).

    Args:
        sample: ``(*batch_shape, 1)`` standard-normal columns.
        scale: Standard deviation to attach.

    Returns:
        ``(*batch_shape, 1)`` float32 column ``scale * sample``.
    """
    if scale == 1.0:
        return sample
    return evoxc.multiply(sample, scale)


def _affine(
    sample: evoxc.SymbolicTensor, mean: float, std: float
) -> evoxc.SymbolicTensor:
    """Turn a standard-normal column into ``N(mean, std)`` (``+ 0``/``* 1`` are exact).

    Args:
        sample: ``(*batch_shape, 1)`` standard-normal columns.
        mean: Mean of the wanted Gaussian.
        std: Standard deviation of the wanted Gaussian.

    Returns:
        ``(*batch_shape, 1)`` float32 column ``mean + std * sample``.
    """
    scaled = _scaled(sample, std)
    if mean == 0.0:
        return scaled
    return evoxc.add(scaled, mean)


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
    keys = evoxc.random.split_n(key, len(specs))
    columns = [
        _initial_column(spec, column_key, column_shape)
        for spec, column_key in zip(specs, keys)
    ]
    block = _assemble(columns, len(specs))
    bounds = _clip_bounds(specs)
    if bounds is not None:
        lower, upper = bounds
        block = enp.clip(
            block,
            _parameter_table(lower, block.shape),
            _parameter_table(upper, block.shape),
        )
    return block


def mutate_attrs(
    key: evoxc.SymbolicTensor,
    attrs: evoxc.SymbolicTensor,
    specs: Sequence["GeneAttr"],
) -> evoxc.SymbolicTensor:
    """Mutate attribute rows of the genes they belong to.

    Per column: add ``N(0, mutate_power)`` noise with probability ``mutate_rate``
    (float only), then with probability ``replace_rate`` re-draw from the init
    distribution (float) or uniformly from ``options`` (int); floats are clipped
    into their bounds. The three branches are selected by one ``U[0, 1)`` gate
    block against two per-column cutoff tables, so the same two ``select`` ops serve
    every column, and a ``NaN`` entry (absent gene row) is restored by the final
    ``isnan`` select -- mutation never revives padding.

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
    # One split for every draw of the pass, and every draw is *per column* (see the
    # module docstring: a block-sized draw is super-linear in the reference
    # interpreter). A rank-0 key is never split per row.
    keys = evoxc.random.split_n(key, 3 * width)
    zero = None
    replacements = []
    noises = []
    gates = []
    for index, spec in enumerate(specs):
        perturb = perturb_flags[index]
        replace = replace_flags[index]
        zero = zero if zero is not None else column_constant(0.0, column_shape)
        gaussian_replace = replace and spec.kind == "float" and float(spec.init_std) > 0.0
        if perturb and gaussian_replace:
            # Both candidates are affine in one standard-normal sample: the entry
            # ends up on one of them, so the pair's joint dependence is unobservable
            # while its marginal law is the two-independent-draws law.
            sample = evoxc.random.normal(
                keys[index], column_shape, 0.0, 1.0, dtype=evoxc.float32
            )
            noises.append(_scaled(sample, float(spec.mutate_power)))
            replacements.append(
                _affine(sample, float(spec.init_mean), float(spec.init_std))
            )
        else:
            if perturb:
                noises.append(
                    evoxc.random.normal(
                        keys[width + index],
                        column_shape,
                        0.0,
                        spec.mutate_power,
                        dtype=evoxc.float32,
                    )
                )
            else:
                noises.append(zero)
            if replace:
                replacements.append(
                    _replacement_column(spec, keys[index], column_shape)
                )
            else:
                replacements.append(zero)
        if perturb or replace:
            gates.append(
                evoxc.random.uniform(
                    keys[2 * width + index], column_shape, 0.0, 1.0, dtype=evoxc.float32
                )
            )
        else:
            gates.append(zero)
    replacement = _assemble(replacements, width)
    noise = _assemble(noises, width)
    gate = _assemble(gates, width)
    perturb_cutoff, replace_cutoff = _cutoffs(specs)
    replaced = evoxc.select(
        evoxc.less(gate, _parameter_table(replace_cutoff, shape)),
        replacement,
        attrs,
    )
    mutated = evoxc.select(
        evoxc.less(gate, _parameter_table(perturb_cutoff, shape)),
        evoxc.add(attrs, noise),
        replaced,
    )
    bounds = _clip_bounds(specs)
    if bounds is not None:
        lower, upper = bounds
        mutated = enp.clip(
            mutated,
            _parameter_table(lower, shape),
            _parameter_table(upper, shape),
        )
    return evoxc.select(evoxc.isnan(attrs), attrs, mutated)


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
