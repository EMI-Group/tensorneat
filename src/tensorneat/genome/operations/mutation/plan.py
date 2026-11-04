"""The structural write plan: one ``select`` per payload array slot for the whole stage.

The four structural operations are mutually exclusive per individual (one shared gate draw),
so at most **one** node row and **three** connection rows change per individual and
generation.  Expressing the stage as a plan — "which row of which array receives which
value", plus the data-dependent set of connections a node deletion detaches — therefore
replaces the four helpers' sequential full-array rewrites (each of which had to re-gate a
whole array with ``select``) by one write per slot, with the slot-0 clear and the
node-deletion detach sharing a single clearing pass over the connection arrays.

A slot is written by comparing the array's row-index constant against the slot's row
:func:`~.arrays.write_rows`: a row index equal to the array's own row count matches nothing,
which is the "no write" case, so no dump row and no flat index block is needed and the
number of traced ops stays proportional to the number of *distinct* accesses.  That is the
whole point of this module under the reference interpreter: a traced op costs ~80 µs largely
independent of element count, so the stage's cost is its *op count*.
"""

from typing import NamedTuple

import evoxc
import evoxc.numpy as enp

from ...utils import GenomeTensors
from .arrays import write_rows

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

    Built once per :meth:`~.base.BaseMutation.mutate_structure` call: the presence masks,
    the row-index constants, the masked-draw ingredients of the shared masks and the flat
    index blocks are identical for all four operations, so sharing them is what keeps the
    stage's op count proportional to the number of distinct accesses.

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
        zero: Rank-0 ``int32`` ``0``.
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
    zero: evoxc.SymbolicTensor
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
    duplicated into a real row.

    Attributes:
        node_rows: ``(P,) int32`` node row to write, ``N`` for "none".
        node_keys: ``(P,) int32`` key value written at ``node_rows``.
        node_attrs: ``(P, NA) float32`` attribute value written at ``node_rows``.
        conn_rows: Three ``(P,) int32`` connection rows to write, ``E`` for "none".  Slot
            order is clear / identity-or-add / copy, applied in that order, so a later slot
            wins where two slots name the same row.
        conn_keys: Three key values written at ``conn_rows``: the rank-0 ``I_INF`` sentinel
            (a cleared row) and two ``(P, 1, CK)`` blocks.
        conn_attrs: Three attribute values written at ``conn_rows``: the rank-0 NaN sentinel
            and two ``(P, 1, CA)`` blocks.
        drop: ``(P, E) bool`` connection rows a node deletion detaches, cleared in the same
            pass as slot 0 (a deletion writes no row of its own, and slot 0 always carries
            the same cleared value, so the two masks never overlap).
    """

    node_rows: evoxc.SymbolicTensor
    node_keys: evoxc.SymbolicTensor
    node_attrs: evoxc.SymbolicTensor
    conn_rows: tuple
    conn_keys: tuple
    conn_attrs: tuple
    drop: evoxc.SymbolicTensor


def apply(
    edit: Edit,
    context: EditContext,
    node_keys: evoxc.SymbolicTensor,
    node_attrs: evoxc.SymbolicTensor,
    conn_keys: evoxc.SymbolicTensor,
    conn_attrs: evoxc.SymbolicTensor,
) -> GenomeTensors:
    """Write one plan into the payload: one ``select`` per slot, one clear pass for the rest.

    Connection slot 0 always writes the *cleared* value — the ``_merge_plan`` of the
    operator fills its key/attribute value with the shared rank-0 ``I_INF``/NaN constants —
    so it carries exactly what the node-deletion detach clears and the two share a single
    pass.
    That is bit-identical to clearing slot 0 and then ``drop``: an individual whose slot 0
    really names a row (an add-node or a delete-connection) has an all-False ``drop``, and a
    delete-node individual points every slot at the "no write" row, which matches no row of
    the row-index constant.  The masks cannot overlap, so their order is unobservable.
    Slots 1 and 2 are then applied in their current order, because a later slot wins where
    two name the same row.

    Args:
        edit (Edit): The plan.
        context (EditContext): Shared row-index constants and the two "clear" values.
        node_keys (evoxc.SymbolicTensor): ``(P, N) int32`` node key rows, ``I_INF`` padded.
        node_attrs (evoxc.SymbolicTensor): ``(P, N, NA) float32`` node attribute rows, NaN
            padded.
        conn_keys (evoxc.SymbolicTensor): ``(P, E, CK) int32`` connection key rows.
        conn_attrs (evoxc.SymbolicTensor): ``(P, E, CA) float32`` connection attribute rows.

    Returns:
        GenomeTensors: the four payload arrays after the plan, same shapes and dtypes.  An
        individual whose plan writes nothing comes out bit-identical (``-0.0`` and ``NaN``
        included), because nothing is recomputed for it.
    """
    pop_size = node_keys.shape[0]
    edge_count, attr_width = conn_keys.shape[1], conn_attrs.shape[2]
    written_node_keys = write_rows(
        node_keys,
        context.node_row_index,
        edit.node_rows,
        enp.reshape(edit.node_keys, (pop_size, 1)),
    )
    written_node_attrs = write_rows(
        node_attrs,
        context.node_row_column,
        edit.node_rows,
        enp.reshape(edit.node_attrs, (pop_size, 1, node_attrs.shape[2])),
    )
    # One clearing pass over the connection arrays for slot 0 plus the detach mask, so the
    # whole stage pays one full-array select per array instead of two.
    clear_mask = evoxc.logical_or(
        evoxc.equal(
            context.conn_row_column,
            enp.reshape(edit.conn_rows[0], (pop_size, 1, 1)),
        ),
        enp.reshape(edit.drop, (pop_size, edge_count, 1)),
    )
    written_conn_keys = evoxc.select(clear_mask, context.sentinel, conn_keys)
    written_conn_attrs = evoxc.select(clear_mask, context.nan_fill, conn_attrs)
    # One row comparison per remaining connection slot, reused by the key and the attribute
    # array: the two arrays share their row layout, so the mask is built once and both writes
    # consume it.
    for rows, key_values, attr_values in zip(
        edit.conn_rows[1:], edit.conn_keys[1:], edit.conn_attrs[1:]
    ):
        mask = evoxc.equal(
            context.conn_row_column, enp.reshape(rows, (pop_size, 1, 1))
        )
        written_conn_keys = evoxc.select(mask, key_values, written_conn_keys)
        written_conn_attrs = evoxc.select(mask, attr_values, written_conn_attrs)
    return GenomeTensors(
        written_node_keys,
        written_node_attrs,
        written_conn_keys,
        written_conn_attrs,
    )
