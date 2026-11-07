"""Grouped (``k``-centre) distance: one ``measure`` call for ``k`` population batches.
Speciation measures one centre genome against the whole population many times per generation,
always with the same fixed scaffolding, so a caller holding ``k`` centre batches can pay for it
once: :func:`group_digest` turns one :class:`~.digest.DistanceDigest` into a
:class:`GroupedDigest` — the eight digest arrays tiled ``k`` times on the population axis plus
the two :class:`~.layout.MeasureStatics` bundles rebuilt at ``rows = k * P`` — and one
:meth:`~.base.BaseDistance.measure` call on a ``k * P``-row centre answers every pair.
Row ``j * P + p`` of the result is the distance of centre row ``j * P + p`` against population
individual ``p``, which is **bit-identical** to ``k`` separate ``P``-row calls: every op of the
merge, of the aligned-attribute gather, of ``gene.distance`` and of the scalar tail is
row-local, and the tiled statics carry each group's own ``p * stride`` row offsets, so neither
the values nor the float summation order move.
There is no new arithmetic here — only the static scaffolding that used to hard-code ``P`` (the
merge stride table, its ``+ 1`` twin, the merge's trailing padding column, the
aligned-attribute row base and the ``(P * M, A)`` attribute row view) is rebuilt at ``rows`` by
:func:`~.layout.build_grouped_statics`, which reuses every rank-0 scalar of the base digest.
The module also owns :func:`validate_family`, the per-family geometry check both digest types
run, so the plain and the grouped route refuse a mismatched centre with one wording.
The route is the **accelerator leg's** shape and the host leg's rejected one: :func:`group_size`
answers ``k = 1`` for a state whose ``kernel_mode`` is ``"host"``, so :func:`for_state` hands the
numpy pipeline its plain digest unchanged and the CPU leg's dispatch count, wall time and peak
RSS do not move, while a device leg groups up to :data:`ACCELERATOR_GROUP` centre rows per call
and cuts the per-call launch count by up to that factor.
The reason is the two legs' cost laws, not a preference: ``measure`` is *sort/bandwidth* bound on
the reference interpreter, where the ``k``-times wider ``(k * P, 2M + 1)`` merge intermediates
cost more than the dispatches they save (``CONTEXT.md``, ``## Grouped route``), and
*launch* bound on a compiled backend, where every dispatch is charged the same floor however
little work it does.
Wiring is the caller's: it owns the static slot count, so it builds the digest once, groups it
with :func:`for_state`, merges ``k`` centre payloads with :func:`centre_batch` per call and
slices the ``k * P`` answer back per slot (``algorithm/neat/_distance.py``).
"""

from typing import NamedTuple

import evoxc

from ....common import ConfigError
from ....common.schedule_accelerator import is_accelerator, kernel_mode
from . import layout as _layout
from .digest import DistanceDigest
from .layout import MeasureStatics


class GroupedDigest(NamedTuple):
    """One population digest tiled for a ``k``-centre ``measure`` call.

    The ten tensor/statics fields keep :class:`~.digest.DistanceDigest`'s names (node family
    first), because :meth:`~.base.BaseDistance.measure` reads them straight off the digest
    object: only the population axis grew, from ``P`` to ``rows = pop_size * group``.

    Attributes:
        node_keys (evoxc.SymbolicTensor): ``(rows, N) int64`` packed node-key composites — the
            base digest's block repeated ``group`` times on axis 0, each copy still ascending
            with its padding rows last, which is the run the merge's stable sort exploits.
        node_attrs (evoxc.SymbolicTensor): ``(rows, N, NA) float32`` node attributes, same order.
        node_count (evoxc.SymbolicTensor): ``(rows,) int32`` real node genes per row.
        node_key_cols (evoxc.SymbolicTensor): ``(rows, N, NK) int32`` node key columns, the
            exact-match oracle for a ``historical_marker`` gene.
        conn_keys (evoxc.SymbolicTensor): ``(rows, E) int64`` packed connection composites.
        conn_attrs (evoxc.SymbolicTensor): ``(rows, E, CA) float32`` connection attributes.
        conn_count (evoxc.SymbolicTensor): ``(rows,) int32`` real connection genes per row.
        conn_key_cols (evoxc.SymbolicTensor): ``(rows, E, CK) int32`` connection key columns.
        node_statics (MeasureStatics): the node family's scaffold, every rank-0 scalar and fold
            table reused from the base digest and the rows-scaled tables rebuilt at ``rows``.
        conn_statics (MeasureStatics): the same scaffold for the connection family.
        pop_size (int): Static population batch ``P`` of the base digest.
        group (int): Static number of centre batches ``k`` the digest was tiled for.

    Notes:
        A grouped digest belongs to the trace that built it: it comes from one generation's
        :meth:`~.base.BaseDistance.prepare` digest and is consumed by that generation's
        ``measure`` calls, so it is never a long-lived graph-boundary value.
        Its leaves are ordinary tensors, so a host harness can still pass the whole value as a
        pytree argument (the test runner does), but nothing in the library stores one on a
        :class:`~tensorneat.common.state.State`.
        ``pop_size`` and ``group`` are static Python ints — validation metadata that cost no op,
        because every geometry the kernels read is already baked into the tiled leaves.
    """

    node_keys: evoxc.SymbolicTensor
    node_attrs: evoxc.SymbolicTensor
    node_count: evoxc.SymbolicTensor
    node_key_cols: evoxc.SymbolicTensor
    conn_keys: evoxc.SymbolicTensor
    conn_attrs: evoxc.SymbolicTensor
    conn_count: evoxc.SymbolicTensor
    conn_key_cols: evoxc.SymbolicTensor
    node_statics: MeasureStatics
    conn_statics: MeasureStatics
    pop_size: int
    group: int

    @property
    def field_names(self) -> tuple[str, ...]:
        """Field names in payload order, for generic digest walks.

        Returns:
            Every field name of the grouped digest, node family first.
        """
        return self._fields


def group_digest(digest: DistanceDigest, group: int) -> GroupedDigest:
    """Tile one population digest for a ``k``-centre ``measure`` call (once per reused digest).

    Args:
        digest (DistanceDigest): The generation's population digest from
            :meth:`~.base.BaseDistance.prepare`; it is neither modified nor retained.
        group (int): Static number of centre batches ``k`` the tiled digest answers, i.e. the
            result covers ``k * P`` centre rows against the same ``P`` individuals.

    Returns:
        GroupedDigest: the eight digest arrays repeated ``group`` times on the population axis
        plus the two statics bundles rebuilt at ``rows = group * P``.

    Raises:
        ConfigError: If ``digest`` is not a :class:`~.digest.DistanceDigest`, or if ``group`` is
            not a positive ``int`` (``0``, a negative value, a ``bool`` and a float are refused:
            a non-positive group would make the merge read an empty row).
        evoxc.ShapeError: Propagated unchanged if a tiled block's gene axis disagrees with the
            statics rebuilt for it, which the base digest's own contract already excludes.

    Notes:
        A plain traced helper, never an ``evoxc.defn``: it runs inside the caller's graph, where
        the digest is already a pytree of ``SymbolicTensor`` leaves.
        One ``concatenate`` per tiled array (``group`` repeated operands on axis 0, one traced
        op — ``evoxc.concatenate`` accepts the same operand several times and the reference
        backend copies the block ``group`` times), and none at all for ``group == 1``, where the
        base digest's block is already the answer.
        Nothing here runs per ``measure`` call: the tile and the statics are paid once per
        reused digest, so a grouped call spends its ops on value-dependent work only.
        ``group`` is a static Python int and no traced value drives the static gate.
    """
    if not isinstance(digest, DistanceDigest):
        raise ConfigError(
            "group_digest: expected a DistanceDigest from prepare(), "
            f"got {type(digest).__name__}"
        )
    if isinstance(group, bool) or not isinstance(group, int) or group <= 0:
        raise ConfigError(
            "group_digest: group must be a positive int (the number of centre batches one "
            f"measure call covers), got {group!r}"
        )
    pop_size = int(digest.node_count.shape[0])
    rows = pop_size * group
    node_keys = _repeat(digest.node_keys, group)
    node_attrs = _repeat(digest.node_attrs, group)
    node_count = _repeat(digest.node_count, group)
    node_key_cols = _repeat(digest.node_key_cols, group)
    conn_keys = _repeat(digest.conn_keys, group)
    conn_attrs = _repeat(digest.conn_attrs, group)
    conn_count = _repeat(digest.conn_count, group)
    conn_key_cols = _repeat(digest.conn_key_cols, group)
    return GroupedDigest(
        node_keys,
        node_attrs,
        node_count,
        node_key_cols,
        conn_keys,
        conn_attrs,
        conn_count,
        conn_key_cols,
        _layout.build_grouped_statics(
            digest.node_statics,
            rows=rows,
            count=int(node_keys.shape[1]),
            attr_width=int(node_attrs.shape[2]),
            attrs=node_attrs,
        ),
        _layout.build_grouped_statics(
            digest.conn_statics,
            rows=rows,
            count=int(conn_keys.shape[1]),
            attr_width=int(conn_attrs.shape[2]),
            attrs=conn_attrs,
        ),
        pop_size,
        group,
    )


def _repeat(block: evoxc.SymbolicTensor, group: int) -> evoxc.SymbolicTensor:
    """Repeat a batch-first block ``group`` times on axis 0.

    Args:
        block (evoxc.SymbolicTensor): ``(P, ...)`` digest leaf.
        group (int): Static repetition count, ``>= 1``.

    Returns:
        evoxc.SymbolicTensor: the ``(group * P, ...)`` block — the operand itself for
        ``group == 1``, otherwise one ``concatenate`` of ``group`` copies.

    Notes:
        One traced op whatever ``group`` is, and none for ``group == 1``, so the identity case
        costs exactly nothing.
        The copies are plain data replication — no axis is permuted and no row is reordered —
        which is what keeps every row-local kernel's answer bit-identical per group.
    """
    if group == 1:
        return block
    return evoxc.concatenate((block,) * group, axis=0)


#: Centre rows one grouped ``measure`` call covers on the accelerator leg: a call spanning up to
#: eight of them cuts the per-call launch count to an eighth, which is the lever on a backend
#: that charges every dispatch a fixed floor however little work it does, while the tiled digest
#: grows only ``k`` times on the population axis — the trade a launch-bound device pays for and
#: a bandwidth-bound one does not.
#: The two other caps still bound the ``k * P`` centre width, and with it peak memory:
#: ``k = max(1, min(ACCELERATOR_GROUP, GROUP_ROW_BUDGET // pop_size, species_size))``, so a
#: 20-slot table with 20 species is two grouped calls plus a ``< 8``-row tail at ``pop <= 512``
#: and five grouped calls at pop 1000, where ``4096 // 1000 = 4`` binds.
ACCELERATOR_GROUP = 8
#: Rows one grouped call may carry.  The grouping amortises a per-launch cost, never a per-element
#: one, so a population wide enough that ``k`` batches would double the intermediates keeps
#: today's dispatch count instead (:func:`group_size` then answers ``1``).
GROUP_ROW_BUDGET = 4096


def group_size(state, pop_size: int, species_size: int | None = None) -> int:
    """Static number of centre batches one grouped ``measure`` call should cover.

    Args:
        state: The trace-time configuration ``State``; only its ``kernel_mode`` entry is read.
        pop_size (int): Static population batch ``P`` of the digest, i.e. the centre batch one
            plain ``measure`` call carries.
        species_size (int | None): The caller's static slot count, when it has one.  More groups
            than slots would measure nothing, so the answer is capped at it.

    Returns:
        int: ``1`` on the host leg (the plain route, unchanged), else
        ``max(1, min(ACCELERATOR_GROUP, GROUP_ROW_BUDGET // pop_size, species_size))`` — ``8``
        at pop 100 (where the constant binds, ``4096 // 100 = 40``), ``8`` at pop 500 (where the
        row budget binds at exactly 8) and ``4`` at pop 1000 (``4096 // 1000``), i.e. 800 /
        4 000 / 4 000 centre rows for a 20-slot table.

    Raises:
        ConfigError: If ``pop_size`` is not a positive ``int``, or if ``species_size`` is neither
            ``None`` nor a positive ``int``.
        tensorneat.common.errors.ConfigError: If the state's ``kernel_mode`` entry is not one of
            its documented values (raised by the shared reader, never defaulted here).

    Notes:
        Host-only and state-free on the compute path: it reads the same ``kernel_mode`` static
        :func:`tensorneat.common.schedule_accelerator.kernel_mode` exposes, emits no traced op,
        and is a pure function of statics, so a caller can hoist it above its scan.
        ``k == 1`` is exactly today's route — :func:`for_state` then hands the caller the plain
        digest, so no op and no byte is added on the host leg.
        The two caps are what keep the trade honest on a wide population: ``GROUP_ROW_BUDGET //
        pop_size`` bounds the ``k * P`` centre width (and with it the ``(k * P, 2M + 1)`` merge
        intermediates and the tiled digest's memory), and ``species_size`` bounds it by the slots
        the caller can actually fill, so a slot count of ``1`` degenerates to ``1`` and a
        population at or above ``GROUP_ROW_BUDGET`` keeps a single ``P``-row call.
    """
    if isinstance(pop_size, bool) or not isinstance(pop_size, int) or pop_size <= 0:
        raise ConfigError(
            f"group_size: pop_size must be a positive int, got {pop_size!r}"
        )
    if species_size is not None and (
        isinstance(species_size, bool) or not isinstance(species_size, int) or species_size <= 0
    ):
        raise ConfigError(
            f"group_size: species_size must be None or a positive int, got {species_size!r}"
        )
    if not is_accelerator(kernel_mode(state)):
        return 1
    group = GROUP_ROW_BUDGET // pop_size
    if species_size is not None:
        group = min(group, species_size)
    return max(1, min(ACCELERATOR_GROUP, group))


def for_state(state, digest: DistanceDigest, *, species_size: int | None = None):
    """Pair a prepared digest with the group size this state's leg wants.

    Args:
        state: The trace-time configuration ``State`` handed to ``measure``; only its
            ``kernel_mode`` entry is read.
        digest (DistanceDigest): The generation's population digest from
            :meth:`~.base.BaseDistance.prepare`.
        species_size (int | None): Optional static slot count, forwarded to :func:`group_size`.

    Returns:
        tuple: ``(digest, group)`` — the :class:`GroupedDigest` and the ``k`` it answers when
        ``k > 1``, else the argument digest unchanged and ``1``.

    Raises:
        ConfigError: For the reasons :func:`group_size`/:func:`group_digest` raise.

    Notes:
        The whole seam is this function plus :func:`centre_batch`: a caller that owns ``k``
        centre batches (a speciation scan over static slots) builds the digest once, groups at
        ``k``, concatenates ``k`` centre payloads per call and slices the ``k * P`` answer back
        into its slots.  Every step is a static decision on a host string and a host int, so the
        grouped form never reaches a graph the host leg builds.
        Grouping does not change the arithmetic: ``measure`` on the grouped digest is
        bit-identical to ``k`` plain ``P``-row calls (``CONTEXT.md``, ``## Grouped route``), so a
        caller may mix grouped and plain calls inside one generation without moving a value.
    """
    group = group_size(state, int(digest.node_count.shape[0]), species_size)
    if group == 1:
        return digest, 1
    return group_digest(digest, group), group


def centre_batch(centres) -> tuple:
    """Concatenate ``k`` centre payloads into the ``k * P`` batch a grouped digest takes.

    Args:
        centres: The centre payloads to merge, each a four-tuple of ``(P, ...)`` tensors in
            ``GenomeTensors`` order (``node_keys``, ``node_attrs``, ``conn_keys``,
            ``conn_attrs``) — the shape one plain ``measure`` call takes.

    Returns:
        tuple: the four arrays concatenated on axis 0, i.e. ``(k * P, ...)`` per field, in the
        same order — so group ``j`` occupies rows ``j * P .. (j + 1) * P - 1`` of every field and
        of the ``(k * P,)`` distance vector that comes back.

    Raises:
        ConfigError: If ``centres`` is empty, or if a payload does not hold exactly four arrays.

    Notes:
        One ``concatenate`` per field and none at all for a single centre, and the row order is
        the caller's: :meth:`~.base.BaseDistance.measure` answers row ``j * P + p`` for centre row
        ``j * P + p``, so slicing the result per group with ``evoxc.slice`` in the same order
        recovers the per-slot rows verbatim.
    """
    rows = [tuple(centre) for centre in centres]
    if not rows:
        raise ConfigError("centre_batch: at least one centre payload is required")
    for row in rows:
        if len(row) != 4:
            raise ConfigError(
                f"centre_batch: a centre payload must hold the four gene arrays, got {len(row)}"
            )
    if len(rows) == 1:
        return rows[0]
    return tuple(
        evoxc.concatenate(tuple(row[field] for row in rows), axis=0) for field in range(4)
    )


def validate_centre(name: str, payload: tuple, digest: GroupedDigest) -> int:
    """Check a centre payload against a grouped digest and return the row count.

    Args:
        name (str): Operator class name, used in the error message.
        payload (tuple): The four centre arrays, in ``GenomeTensors`` order.
        digest (GroupedDigest): The tiled digest built by :func:`group_digest`.

    Returns:
        int: ``rows = digest.pop_size * digest.group`` — the centre batch a grouped ``measure``
        call requires.

    Raises:
        ConfigError: If ``digest`` is not a :class:`GroupedDigest`, if the centre batch is not
            exactly ``rows``, if the digest's own leaves disagree with ``rows``, if a family's
            gene or attribute axis disagrees with the centre's (:func:`validate_family`), or if
            a rebuilt statics table is not shaped for ``rows``.

    Notes:
        Host-only: it reads ``.shape`` and the static ``pop_size``/``group`` ints only, so it
        emits no traced op and behaves identically inside and outside a trace.
        The message names the centre's batch *and* the required one plus ``group`` and
        ``pop_size``, so a caller that forgot to tile the centre — or tiled it a different
        number of times — is told exactly what to pass instead.
        The per-family checks are :func:`validate_family`, the helper the plain route runs, so
        the two digest types cannot drift apart in wording or in what they accept.
    """
    if not isinstance(digest, GroupedDigest):
        raise ConfigError(
            f"{name}.measure: expected a GroupedDigest from group_digest(), "
            f"got {type(digest).__name__}"
        )
    pop_size = int(digest.pop_size)
    group = int(digest.group)
    rows = pop_size * group
    centre_batch = int(payload[0].shape[0])
    if centre_batch != rows:
        raise ConfigError(
            f"{name}.measure: a grouped digest holds {rows} rows ({group} groups of pop_size "
            f"{pop_size}) so the centre batch must be {rows}, but it is {centre_batch}; "
            f"concatenate {group} centre batches of {pop_size} rows, or measure them one at a "
            f"time against the plain digest"
        )
    for family, count in (("node", digest.node_count), ("conn", digest.conn_count)):
        if int(count.shape[0]) != rows:
            raise ConfigError(
                f"{name}.measure: the grouped digest's {family}_count holds "
                f"{int(count.shape[0])} rows but pop_size * group is {rows} "
                f"({pop_size} * {group})"
            )
    for family, centre, leaves, statics in (
        ("node", payload[:2], digest[:4], digest.node_statics),
        ("conn", payload[2:], digest[4:8], digest.conn_statics),
    ):
        validate_family(name, family, centre[0], centre[1], *leaves)
        _validate_statics(name, family, statics, rows, int(centre[0].shape[1]))
    return rows


def _validate_statics(
    name: str, family: str, statics: MeasureStatics, rows: int, count: int
) -> None:
    """Reject a statics bundle that was not rebuilt for the grouped row count.

    Args:
        name (str): Operator class name, used in the error message.
        family (str): ``"node"`` or ``"conn"``, naming the family in the message.
        statics (MeasureStatics): The family's scaffold off the grouped digest.
        rows (int): Required row count ``pop_size * group``.
        count (int): Required gene count ``M`` of the family.

    Raises:
        ConfigError: If a rows-scaled table (the merge strides, its ``+ 1`` twin, the trailing
            padding column, the attribute row base or the attribute row view) is not shaped for
            ``rows`` rows.

    Notes:
        Pure static geometry, so a digest assembled by hand — not by :func:`group_digest` — is
        refused here instead of reaching a gather whose index block belongs to another row
        count, which would silently pair centre rows with the wrong individuals.
        A family with no attribute column (``attr_width == 0``) keeps the empty placeholder
        base, exactly as :func:`~.layout.build_grouped_statics` does, so its attribute checks
        are skipped.
    """
    for field, expected in (
        ("offsets", (rows, 1)),
        ("offsets_plus_one", (rows, 1)),
        ("pad_column_i64", (rows, 1)),
    ):
        block = getattr(statics, field)
        if tuple(int(size) for size in block.shape) != expected:
            raise ConfigError(
                f"{name}.measure: the {family} grouped statics {field} is "
                f"{tuple(block.shape)} but a {rows}-row ({count}-gene) grouped digest needs "
                f"{expected}; rebuild it with group_digest()"
            )
    base = statics.attr.base
    if tuple(int(size) for size in base.shape) not in ((rows, 1), (0, 1)):
        raise ConfigError(
            f"{name}.measure: the {family} grouped statics attr.base is "
            f"{tuple(base.shape)} but a {rows}-row ({count}-gene) grouped digest needs "
            f"{(rows, 1)}; rebuild it with group_digest()"
        )
    attrs_rows = statics.attr.attrs_rows
    if attrs_rows is not None and int(base.shape[0]) == rows:
        attr_width = int(attrs_rows.shape[1])
        if tuple(int(size) for size in attrs_rows.shape) != (rows * count, attr_width):
            raise ConfigError(
                f"{name}.measure: the {family} grouped statics attr.attrs_rows is "
                f"{tuple(attrs_rows.shape)} but a {rows}-row ({count}-gene) grouped digest "
                f"needs {(rows * count, attr_width)}; rebuild it with group_digest()"
            )


def validate_family(
    name: str,
    family: str,
    centre_keys: evoxc.SymbolicTensor,
    centre_attrs: evoxc.SymbolicTensor,
    digest_keys: evoxc.SymbolicTensor,
    digest_attrs: evoxc.SymbolicTensor,
    digest_count: evoxc.SymbolicTensor,
    digest_key_cols: evoxc.SymbolicTensor,
) -> None:
    """Reject a digest family whose gene geometry cannot be paired with this centre.

    Args:
        name (str): Operator class name, used in the error message.
        family (str): ``"node"`` or ``"conn"``, naming the family in the message.
        centre_keys (evoxc.SymbolicTensor): ``(R, M)`` or ``(R, M, K) int32`` centre key array.
        centre_attrs (evoxc.SymbolicTensor): ``(R, M, A) float32`` centre attribute rows.
        digest_keys (evoxc.SymbolicTensor): ``(R, M) int64`` composites of this family.
        digest_attrs (evoxc.SymbolicTensor): ``(R, M, A) float32`` population attribute rows in
            the digest's gene order.
        digest_count (evoxc.SymbolicTensor): ``(R,) int32`` real genes per row.
        digest_key_cols (evoxc.SymbolicTensor): ``(R, M, K) int32`` population key columns in
            the digest's gene order.

    Raises:
        ConfigError: If the digest's gene axis disagrees with the centre's key array, if the two
            attribute blocks disagree on ``(M, A)``, if the count is not rank 1, or if the
            key-column block's gene axis disagrees with the centre's keys.

    Notes:
        The single spelling of these four checks: the plain route
        (:func:`~.base._validate_digest`) and the grouped one (:func:`validate_centre`) both
        call it, so the two digest types refuse a mismatched centre with the same message.
        Static shapes only — a mismatch is a call-site configuration error, never a shape error
        deep inside a gather that would silently compare the wrong rows.
        The population axis is deliberately *not* compared: the plain route broadcasts a batch-1
        centre to the digest's batch, and the grouped route has already checked its row count.
    """
    if int(digest_keys.shape[1]) != int(centre_keys.shape[1]):
        raise ConfigError(
            f"{name}.measure: the {family} digest holds {int(digest_keys.shape[1])} "
            f"gene rows but the centre's key array is {tuple(centre_keys.shape)}"
        )
    if tuple(int(size) for size in digest_attrs.shape[1:]) != tuple(
        int(size) for size in centre_attrs.shape[1:]
    ):
        raise ConfigError(
            f"{name}.measure: the {family} digest attributes are "
            f"{(int(digest_attrs.shape[1]), int(digest_attrs.shape[2]))} (M, A) but the "
            f"centre's are {(int(centre_attrs.shape[1]), int(centre_attrs.shape[2]))}"
        )
    if len(digest_count.shape) != 1:
        raise ConfigError(
            f"{name}.measure: the {family} digest count must be (P,), got "
            f"{tuple(digest_count.shape)}"
        )
    if int(digest_key_cols.shape[1]) != int(centre_keys.shape[1]):
        raise ConfigError(
            f"{name}.measure: the {family} digest key columns hold "
            f"{int(digest_key_cols.shape[1])} gene rows but the centre's key array is "
            f"{tuple(centre_keys.shape)}"
        )


__all__ = [
    "ACCELERATOR_GROUP",
    "GROUP_ROW_BUDGET",
    "GroupedDigest",
    "centre_batch",
    "for_state",
    "group_digest",
    "group_size",
    "validate_centre",
]
