"""The structural write plan: one ``select`` per payload-array slot for one stage.

The four structural operations are **independent** per individual (four rows of one gate
draw, each its own Bernoulli decision) and run in sequence add-node → delete-node →
add-conn → delete-conn, each stage planning against the payload the previous stage left
behind.  One stage therefore writes at most **one** node row and **three** connection rows
per individual, and it may not need a node row or a cleared connection row at all.
Expressing a stage as a plan — "which row of which array receives which value", plus the
data-dependent set of connections a node deletion detaches — replaces sequential
full-array rewrites (each of which had to re-gate a whole array with ``select``) by one
write per slot, with the slot-0 clear and the node-deletion detach sharing a single
clearing pass over the connection arrays.  A stage's :class:`Edit` is the
single-operation form of the same machinery.

A node slot is written by comparing the array's row-index constant against the slot's row:
a row index equal to the array's own row count matches nothing, which is the "no write" case,
so no dump row and no flat index block is needed.  A connection slot goes through one flat
``scatter`` instead, because a ``select`` pays whole-array traffic.  Both shapes keep the
stage's cost proportional to the number of *distinct* accesses and to the cells actually
written.
"""

from typing import NamedTuple

import numpy as np
import evoxc
import evoxc.numpy as enp

from ...utils import GenomeTensors
from .arrays import graph_constant

class AddNodePlan(NamedTuple):
    """Candidate rows and values of one node split.

    Attributes:
        ok: ``(P,) bool`` "this individual may split".
        node_row: ``(P,) int32`` free node row receiving the new node.
        delete_row: ``(P,) int32`` connection row consumed by the split.
        identity_row: ``(P,) int32`` free connection row of the ``old_src -> new`` edge.
        copy_row: ``(P,) int32`` free connection row of the ``new -> old_dst`` edge.
        in_key: ``(P,) int32`` source key of the consumed connection.
        out_key: ``(P,) int32`` destination key of the consumed connection.
        copy_attrs: ``(P, 1, CA) float32`` attributes the copy edge inherits, gathered as
            the row block the write slot consumes (no reshape in the merge).
    """

    ok: evoxc.SymbolicTensor
    node_row: evoxc.SymbolicTensor
    delete_row: evoxc.SymbolicTensor
    identity_row: evoxc.SymbolicTensor
    copy_row: evoxc.SymbolicTensor
    in_key: evoxc.SymbolicTensor
    out_key: evoxc.SymbolicTensor
    copy_attrs: evoxc.SymbolicTensor


class DeleteNodePlan(NamedTuple):
    """Candidate row and detach mask of one node deletion.

    Attributes:
        ok: ``(P,) bool`` "this individual may delete a hidden node".
        node_row: ``(P,) int32`` node row to clear.
        drop: ``(P, E) bool`` connection rows touching the deleted key (already gated).
    """

    ok: evoxc.SymbolicTensor
    node_row: evoxc.SymbolicTensor
    drop: evoxc.SymbolicTensor


class AddConnPlan(NamedTuple):
    """Candidate row and endpoints of one added connection.

    Attributes:
        ok: ``(P,) bool`` "this individual may add this connection".
        conn_row: ``(P,) int32`` free connection row receiving the edge.
        in_key: ``(P,) int32`` source node key.
        out_key: ``(P,) int32`` destination node key.
    """

    ok: evoxc.SymbolicTensor
    conn_row: evoxc.SymbolicTensor
    in_key: evoxc.SymbolicTensor
    out_key: evoxc.SymbolicTensor


class DeleteConnPlan(NamedTuple):
    """Candidate row of one connection deletion.

    Attributes:
        ok: ``(P,) bool`` "this individual may delete this connection".
        conn_row: ``(P,) int32`` connection row to clear.
    """

    ok: evoxc.SymbolicTensor
    conn_row: evoxc.SymbolicTensor


__all__ = [
    "AddConnPlan",
    "AddNodePlan",
    "DeleteConnPlan",
    "DeleteNodePlan",
    "Edit",
    "EditContext",
    "apply",
]


class EditContext(NamedTuple):
    """Row bookkeeping shared by every operation of one structural stage.

    Built once per stage inside :meth:`~.base.BaseMutation.mutate_structure` — against the
    payload that stage starts from, because the previous stage may have moved a row.  The
    presence masks, the row-index constants, the ingredients of the shared masks and the
    flat index blocks are identical for all operations of the stage, so sharing them is
    what keeps the stage's op count proportional to the number of distinct accesses.

    Attributes:
        node_row_index: ``(1, N) int32`` constant ``[0, 1, ..., N-1]`` (row-range masks and
            the rank-2 write of ``node_keys``).
        node_row_column: ``(1, N, 1) int32`` the same constant reshaped for the rank-3 write
            of ``node_attrs``.
        conn_row_index: ``(1, E) int32`` constant ``[0, 1, ..., E-1]``.
        conn_row_column: ``(1, E, 1) int32`` the same constant reshaped for the rank-3 write
            of the connection arrays.
        conn_src: ``(P, E) int32`` source-key column of ``conn_keys``.
        conn_dst: ``(P, E) int32`` destination-key column of ``conn_keys``.
        present_node: ``(P, N) bool`` live node rows (``node_keys != I_INF``).
        free_node: ``(P, N) bool`` absent node rows (``node_keys == I_INF``).
        deletable_node: ``(P, N) bool`` hidden live rows (input and output rows excluded).
        targetable_node: ``(P, N) bool`` live rows that may receive an in-edge.
        present_conn: ``(P, E) bool`` live connection rows (source key ``!= I_INF``).
        free_conn: ``(P, E) bool`` absent connection rows.
        present_conn_float: ``(P, E) float32`` ``present_conn`` as a draw-weight mask, shared
            by the two operations that sample a present connection.
        present_conn_any: ``(P,) bool`` "this individual has a present connection".
        free_conn_row: ``(P,) int32`` first free connection row (``0`` where there is none).
        free_conn_any: ``(P,) bool`` "this individual has a free connection row".
        sentinel: Rank-0 ``int32`` ``I_INF``, the "no key" value written into cleared rows.
        nan_fill: Rank-0 ``float32`` NaN, the "no value" value written into cleared rows.
        in_degree_limit: Rank-0 ``int32`` ``genome.max_in_degree``.
        node_row_limit: Rank-0 ``int32`` ``N``, the "no write" node row.
        conn_row_limit: Rank-0 ``int32`` ``E``, the "no write" connection row.
        node_flat: ``(P * N,)`` flattened ``node_keys`` (endpoint and deleted-key reads).
        node_offsets: ``(P, 1)`` batch strides of ``node_keys``.
        conn_key_flat: ``(P * E * CK,)`` flattened ``conn_keys``.
        conn_key_base: ``(P, 1, CK) int32`` constant ``p * E * CK + column``.
        conn_attr_flat: ``(P * E * CA,)`` flattened ``conn_attrs``.
        conn_attr_base: ``(P, 1, CA) int32`` constant ``p * E * CA + column``.
        key_width: Rank-0 ``int32`` ``CK``, the row width of ``conn_key_flat``.
        attr_width: Rank-0 ``int32`` ``CA``, the row width of ``conn_attr_flat``.
    """

    node_row_index: evoxc.SymbolicTensor
    node_row_column: evoxc.SymbolicTensor
    conn_row_index: evoxc.SymbolicTensor
    conn_row_column: evoxc.SymbolicTensor
    conn_src: evoxc.SymbolicTensor
    conn_dst: evoxc.SymbolicTensor
    present_node: evoxc.SymbolicTensor
    free_node: evoxc.SymbolicTensor
    deletable_node: evoxc.SymbolicTensor
    targetable_node: evoxc.SymbolicTensor
    present_conn: evoxc.SymbolicTensor
    free_conn: evoxc.SymbolicTensor
    present_conn_float: evoxc.SymbolicTensor
    present_conn_any: evoxc.SymbolicTensor
    free_conn_row: evoxc.SymbolicTensor
    free_conn_any: evoxc.SymbolicTensor
    sentinel: evoxc.SymbolicTensor
    nan_fill: evoxc.SymbolicTensor
    in_degree_limit: evoxc.SymbolicTensor
    node_row_limit: evoxc.SymbolicTensor
    conn_row_limit: evoxc.SymbolicTensor
    node_flat: evoxc.SymbolicTensor
    node_offsets: evoxc.SymbolicTensor
    conn_key_flat: evoxc.SymbolicTensor
    conn_key_base: evoxc.SymbolicTensor
    conn_attr_flat: evoxc.SymbolicTensor
    conn_attr_base: evoxc.SymbolicTensor
    key_width: evoxc.SymbolicTensor
    attr_width: evoxc.SymbolicTensor


class Edit(NamedTuple):
    """The writes one structural stage performs, one row slot per array.

    Row sentinels (the array's own length) mean "no write" and match no row of
    :func:`apply`'s row-index constants; values at such a slot are irrelevant, never
    duplicated into a real row.  Fields an individual **stage** statically never uses are
    ``None`` and are skipped by :func:`apply`: the ``node_*`` trio for the two connection
    stages, and ``drop`` for every stage but delete-node (a ``None`` ``drop`` merges no
    detach mask into the clearing pass, which is the same thing as an all-``False`` one).

    Attributes:
        node_rows: ``(P,) int32`` node row to write, ``N`` for "none".
        node_keys: ``(P,) int32`` key value written at ``node_rows`` — or the rank-0
            ``I_INF`` sentinel, broadcast by the write's ``select``, when the stage only
            ever clears a row.
        node_attrs: ``(P, NA) float32`` attribute value written at ``node_rows`` — or the
            rank-0 NaN sentinel under the same rule.
        conn_rows: Connection rows to write, ``E`` for "none".  Slot 0 is the cleared row,
            then the scatter slots (identity / add / copy).  Slots are applied in order, so
            a later slot wins where two slots name the same row.
        conn_keys: Key values written at ``conn_rows``: the rank-0 ``I_INF`` sentinel (a
            cleared row) after the ``(P, 1, CK)`` blocks.
        conn_attrs: Attribute values written at ``conn_rows``: the rank-0 NaN sentinel after
            the ``(P, 1, CA)`` blocks.
        drop: ``(P, E) bool`` connection rows a node deletion detaches, cleared in the same
            pass as slot 0 — or ``None`` for a stage that detaches nothing (a deletion
            writes no row of its own, and slot 0 always carries the same cleared value, so
            the two masks never overlap).
    """

    node_rows: evoxc.SymbolicTensor
    node_keys: evoxc.SymbolicTensor
    node_attrs: evoxc.SymbolicTensor
    conn_rows: tuple
    conn_keys: tuple
    conn_attrs: tuple
    drop: evoxc.SymbolicTensor


def _slot_block(
    value: evoxc.SymbolicTensor, shape: tuple[int, ...]
) -> evoxc.SymbolicTensor:
    """Shape a slot value to the rank :func:`apply`'s ``select`` broadcasts against.

    A stage that only ever clears a row hands over the shared **rank-0** ``I_INF``/NaN
    constant — one op, and no population-scaled graph constant — while a stage that writes
    a real value hands over a ``(P, ...)`` block.  The two need different ops: a rank-0
    operand is grown with ``broadcast_to`` (a trailing-axis expansion is not a
    ``broadcast_to``) and a block is reshaped to insert the trailing axes.

    Args:
        value: Rank-0 sentinel or a ``(P, ...)`` block.
        shape: Destination shape of the write slot.

    Returns:
        ``SymbolicTensor`` of ``shape``.
    """
    if len(value.shape) == 0:
        return enp.broadcast_to(value, shape)
    return enp.reshape(value, shape)


def _dump_cells(
    pop_size: int, row_count: int, width: int, slots: int
) -> np.ndarray:
    """Flat indices of the ``slots`` throw-away cells appended after a payload.

    ``scatter`` is set-only, so a "no write" slot still has to name *some* cell; it names
    one of these trailing cells instead, which the write path slices off again.  One cell
    per slot (not one per slot *and* individual) keeps the cell count independent of the
    population, and giving every slot its own cell means no two entries of one scatter
    ever share an index — the result then does not depend on duplicate-index ordering.

    Args:
        pop_size: Population size ``P`` (static).
        row_count: Rows per individual ``R`` of the array this index block addresses.
        width: Row width ``W`` of that array.
        slots: Number of written rows per individual (static).

    Returns:
        ``(1, slots, W) int32`` host array for :func:`~.arrays.graph_constant`.
    """
    base = pop_size * row_count * width
    return (
        np.arange(slots, dtype=np.int32).reshape(1, slots, 1) * np.int32(width)
        + np.arange(width, dtype=np.int32).reshape(1, 1, width)
        + np.int32(base)
    ).astype(np.int32)


def _scatter_slots(
    x: evoxc.SymbolicTensor,
    rows: evoxc.SymbolicTensor,
    valid: evoxc.SymbolicTensor,
    values: evoxc.SymbolicTensor,
    column_base: evoxc.SymbolicTensor,
    width_value: evoxc.SymbolicTensor,
    dump: evoxc.SymbolicTensor,
    pop_size: int,
    row_count: int,
    width: int,
    slots: int,
) -> evoxc.SymbolicTensor:
    """Write one row per slot per individual with a single flat ``scatter``.

    A ``select`` pays for the whole ``(P, R, W)`` array even though it rewrites at most
    ``slots`` rows of each individual, and the reference interpreter is element-volume
    bound at that size (a ``(500, 100, 2)`` ``np.where`` costs ~0.4 ms whether or not a row
    is touched).  One flat ``scatter`` addresses only the written cells, so the stage pays
    one copy (``pad``) plus one indexed write instead of ``slots`` full-array passes.

    The target is the payload flattened with ``slots * width`` trailing dump cells; a slot
    that names no row (``valid`` false, i.e. the "no write" row ``R``) writes its own dump
    cell, which is sliced off at the end.  Every index is distinct, so the write order
    inside the ``scatter`` is unobservable.

    Args:
        x: ``(P, R, W)`` array to write into, after the clearing pass.
        rows: ``(P, slots, 1) int32`` destination rows, ``R`` meaning "no write".
        valid: ``(P, slots, 1) bool`` "this slot names a real row".
        values: ``(P, slots, W)`` values written at those rows.
        column_base: ``(P, 1, W) int32`` constant ``p * R * W + column``.
        width_value: Rank-0 ``int32`` row width ``W``.
        dump: ``(1, slots, W) int32`` constant from :func:`_dump_cells`.
        pop_size: Population size ``P`` (static).
        row_count: Rows per individual ``R`` (static).
        width: Row width ``W`` (static).
        slots: Written rows per individual (static).

    Returns:
        ``(P, R, W)`` array with the ``slots`` rows per individual overwritten.
    """
    if width == 0:
        # A zero-width row has no cell to write; the plan's key/attribute values are empty.
        return x
    index = evoxc.add(column_base, evoxc.multiply(rows, width_value))
    index = evoxc.select(valid, index, dump)
    padded = evoxc.pad(
        enp.reshape(x, (pop_size * row_count * width,)), ((0, slots * width),)
    )
    written = evoxc.scatter(
        padded,
        enp.reshape(index, (slots * pop_size * width,)),
        enp.reshape(values, (slots * pop_size * width,)),
        axis=0,
    )
    kept = evoxc.slice(written, (0,), (pop_size * row_count * width,))
    return enp.reshape(kept, (pop_size, row_count, width))


def apply(
    edit: Edit,
    context: EditContext,
    node_keys: evoxc.SymbolicTensor,
    node_attrs: evoxc.SymbolicTensor,
    conn_keys: evoxc.SymbolicTensor,
    conn_attrs: evoxc.SymbolicTensor,
    write_nodes: bool = True,
    clear_conn: bool = True,
) -> GenomeTensors:
    """Write one stage's plan into the payload: one clear pass, one scatter, two node writes.

    Connection slot 0 always writes the *cleared* value when the stage clears — the
    :meth:`~.base.BaseMutation._stage_edit` of that stage fills its key/attribute value with
    the shared rank-0 ``I_INF``/NaN constants — so it carries exactly what a
    node-deletion detach clears and the two share a single ``select`` pass.
    That is bit-identical to clearing slot 0 and then ``drop``: an individual whose slot 0
    really names a row (an add-node or a delete-connection) has an all-``False`` ``drop``
    (``drop`` is ``None`` for such a stage, so no mask is merged at all), and a delete-node
    individual points every slot at the "no write" row, which matches no row of the
    row-index constant.  The masks cannot overlap, so their order is unobservable.
    Slots 1 and 2 name two different rows of the same individual (an add-node split writes
    both of its new edges), so they go into one flat ``scatter``
    (:func:`_scatter_slots`): it addresses only the written cells, where a ``select`` per
    slot would pay for the whole array twice under the element-volume-bound interpreter.
    Slot order inside the scatter is unobservable because the two slots never name the same
    row (the copy edge goes to a *later* free row than the identity edge).

    The two keyword flags are **static** specializations of the same machinery, not runtime
    branches: a stage that statically cannot write a node row passes ``write_nodes=False``
    and gets its ``node_*`` arrays back untouched, and a stage that clears no connection row
    of its own (add-conn, whose slot 0 names the "no write" row) passes
    ``clear_conn=False`` and feeds the arrays straight into the scatter.  Both flags
    default to ``True``, which is exactly the behaviour of the full clear-and-write form, so
    flipping either one can never change an individual's bits — only the ops that are
    provably invisible.

    Args:
        edit (Edit): The plan.
        context (EditContext): Shared row-index constants and the two "clear" values.
        node_keys (evoxc.SymbolicTensor): ``(P, N) int32`` node key rows, ``I_INF`` padded.
        node_attrs (evoxc.SymbolicTensor): ``(P, N, NA) float32`` node attribute rows, NaN
            padded.
        conn_keys (evoxc.SymbolicTensor): ``(P, E, CK) int32`` connection key rows.
        conn_attrs (evoxc.SymbolicTensor): ``(P, E, CA) float32`` connection attribute rows.
        write_nodes (bool): Static flag; ``False`` skips both node-array ``select`` calls
            because this stage writes no node row.
        clear_conn (bool): Static flag; ``False`` skips the two connection-clearing
            ``select`` calls because this stage's slot 0 names no row.

    Returns:
        GenomeTensors: the four payload arrays after the plan, same shapes and dtypes.  An
        individual whose plan writes nothing comes out bit-identical (``-0.0`` and ``NaN``
        included), because nothing is recomputed for it.
    """
    pop_size = node_keys.shape[0]
    edge_count = conn_keys.shape[1]
    key_width, attr_width = conn_keys.shape[2], conn_attrs.shape[2]
    written_node_keys = node_keys
    written_node_attrs = node_attrs
    if write_nodes:
        # The two node arrays share the row layout, so the row-index comparison is built
        # once and broadcast over the attribute axis instead of being recomputed per array.
        node_write = evoxc.equal(
            context.node_row_index, enp.reshape(edit.node_rows, (pop_size, 1))
        )
        written_node_keys = evoxc.select(
            node_write, _slot_block(edit.node_keys, (pop_size, 1)), node_keys
        )
        written_node_attrs = evoxc.select(
            enp.reshape(
                node_write, (pop_size, node_keys.shape[1], 1)
            ),
            _slot_block(edit.node_attrs, (pop_size, 1, node_attrs.shape[2])),
            node_attrs,
        )
    written_conn_keys = conn_keys
    written_conn_attrs = conn_attrs
    if clear_conn:
        # One clearing pass over the connection arrays for slot 0 plus the detach mask, so
        # the stage pays one full-array select per array instead of two.
        clear_mask = evoxc.equal(
            context.conn_row_column,
            _slot_block(edit.conn_rows[0], (pop_size, 1, 1)),
        )
        if edit.drop is not None:
            clear_mask = evoxc.logical_or(
                clear_mask, enp.reshape(edit.drop, (pop_size, edge_count, 1))
            )
        written_conn_keys = evoxc.select(clear_mask, context.sentinel, conn_keys)
        written_conn_attrs = evoxc.select(clear_mask, context.nan_fill, conn_attrs)
    # Slots 1 and 2 name *different* rows for the same individual (an add-node split writes
    # both of its new edges), so no single ``select`` can express them; one flat ``scatter``
    # over the two rows addresses only the written cells and replaces two full-array passes
    # per array.  A stage with a single slot has nothing to scatter.
    slots = len(edit.conn_rows) - 1
    if slots > 0:
        slot_rows = enp.stack(tuple(edit.conn_rows[1:]), axis=1)
        slot_valid = enp.reshape(
            evoxc.less(slot_rows, context.conn_row_limit), (pop_size, slots, 1)
        )
        slot_rows = enp.reshape(slot_rows, (pop_size, slots, 1))
        written_conn_keys = _scatter_slots(
            written_conn_keys,
            slot_rows,
            slot_valid,
            enp.concatenate(list(edit.conn_keys[1:]), axis=1),
            context.conn_key_base,
            context.key_width,
            graph_constant(_dump_cells(pop_size, edge_count, key_width, slots)),
            pop_size,
            edge_count,
            key_width,
            slots,
        )
        written_conn_attrs = _scatter_slots(
            written_conn_attrs,
            slot_rows,
            slot_valid,
            enp.concatenate(list(edit.conn_attrs[1:]), axis=1),
            context.conn_attr_base,
            context.attr_width,
            graph_constant(_dump_cells(pop_size, edge_count, attr_width, slots)),
            pop_size,
            edge_count,
            attr_width,
            slots,
        )
    return GenomeTensors(
        written_node_keys,
        written_node_attrs,
        written_conn_keys,
        written_conn_attrs,
    )
