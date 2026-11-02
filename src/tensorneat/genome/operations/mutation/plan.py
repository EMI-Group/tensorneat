"""The structural write plan: one scatter per payload array for the whole stage.

The four structural operations are mutually exclusive per individual (one shared gate draw),
so at most **one** node row and **three** connection rows change per individual and
generation.  Expressing the stage as a plan — "which row of which array receives which
value", plus the data-dependent set of connections a node deletion detaches — therefore
replaces the four helpers' sequential full-array rewrites (each of which had to re-gate a
whole array with ``select``) by exactly four scatters and one clearing ``select``.

That is the whole point of this module under the reference interpreter: a traced op costs
~80 µs largely independent of element count, so the stage's cost is its *op count*, and one
plan-wide write pass per array is the minimum a scatter-based implementation can reach.
"""

from typing import NamedTuple

import evoxc
import evoxc.numpy as enp

from ....common import I_INF
from ...utils import GenomeTensors
from .arrays import scatter_rows

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
        copy_attrs: ``(P, CA) float32`` attributes the copy edge inherits.
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

    Built once per :meth:`~.base.BaseMutation.mutate_structure` call: the presence masks and
    the flat index blocks are identical for all four operations, so sharing them is what
    keeps the stage's op count proportional to the number of distinct accesses.

    Attributes:
        node_row_index: ``(1, N) int32`` constant ``[0, 1, ..., N-1]`` (row-range masks).
        conn_row_index: ``(1, E) int32`` constant ``[0, 1, ..., E-1]``.
        node_columns: ``(1, NA) int32`` constant column indices of one node attribute row.
        conn_key_columns: ``(1, CK) int32`` constant column indices of one connection key row.
        conn_attr_columns: ``(1, CA) int32`` constant column indices of one connection
            attribute row.
        present_node: ``(P, N) bool`` live node rows (``node_keys != I_INF``).
        free_node: ``(P, N) bool`` absent node rows (``node_keys == I_INF``).
        present_conn: ``(P, E) bool`` live connection rows (first key column ``!= I_INF``).
        free_conn: ``(P, E) bool`` absent connection rows.
        node_flat: ``(P * N,)`` flattened ``node_keys``.
        node_offsets: ``(P, 1)`` batch strides of ``node_keys``.
        conn_key_flat: ``(P * E * CK,)`` flattened ``conn_keys``.
        conn_key_offsets: ``(P, 1)`` batch strides of ``conn_keys``.
        conn_attr_flat: ``(P * E * CA,)`` flattened ``conn_attrs``.
        conn_attr_offsets: ``(P, 1)`` batch strides of ``conn_attrs``.
    """

    node_row_index: evoxc.SymbolicTensor
    conn_row_index: evoxc.SymbolicTensor
    conn_key_columns: evoxc.SymbolicTensor
    conn_attr_columns: evoxc.SymbolicTensor
    present_node: evoxc.SymbolicTensor
    free_node: evoxc.SymbolicTensor
    deletable_node: evoxc.SymbolicTensor
    targetable_node: evoxc.SymbolicTensor
    present_conn: evoxc.SymbolicTensor
    free_conn: evoxc.SymbolicTensor
    sentinel: evoxc.SymbolicTensor
    zero: evoxc.SymbolicTensor
    in_degree_limit: evoxc.SymbolicTensor
    node_flat: evoxc.SymbolicTensor
    node_offsets: evoxc.SymbolicTensor
    conn_key_flat: evoxc.SymbolicTensor
    conn_key_offsets: evoxc.SymbolicTensor
    conn_attr_flat: evoxc.SymbolicTensor
    conn_attr_offsets: evoxc.SymbolicTensor


class Edit(NamedTuple):
    """The writes one structural stage performs, one row slot per array.

    Row sentinels (the array's own length) mean "no write" and are absorbed by the trailing
    dump row of :func:`scatter_rows`; values at such a slot are irrelevant, never duplicated
    into a real row.

    Attributes:
        node_rows: ``(P,) int32`` node row to write, ``N`` for "none".
        node_keys: ``(P,) int32`` key value written at ``node_rows``.
        node_attrs: ``(P, NA) float32`` attribute value written at ``node_rows``.
        conn_rows: ``(P, 3) int32`` connection rows to write, ``E`` for "none".  Slot order
            is delete / identity / copy, so a later slot wins where two slots name the same
            row (``evoxc.scatter`` resolves duplicates last-write-wins).
        conn_keys: ``(P, 3, CK) int32`` key rows written at ``conn_rows``.
        conn_attrs: ``(P, 3, CA) float32`` attribute rows written at ``conn_rows``.
        drop: ``(P, E) bool`` connection rows a node deletion detaches (cleared after the
            scatter; a deletion writes no row of its own).
    """

    node_rows: evoxc.SymbolicTensor
    node_keys: evoxc.SymbolicTensor
    node_attrs: evoxc.SymbolicTensor
    conn_rows: evoxc.SymbolicTensor
    conn_keys: evoxc.SymbolicTensor
    conn_attrs: evoxc.SymbolicTensor
    drop: evoxc.SymbolicTensor


def apply(
    edit: Edit,
    node_keys: evoxc.SymbolicTensor,
    node_attrs: evoxc.SymbolicTensor,
    conn_keys: evoxc.SymbolicTensor,
    conn_attrs: evoxc.SymbolicTensor,
) -> GenomeTensors:
    """Write one plan into the payload: four scatters plus one clearing select.

    Args:
        edit (Edit): The plan.
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
    pop_size, node_count = node_keys.shape
    edge_count = conn_keys.shape[1]
    node_attr_width = node_attrs.shape[2]
    node_slot = enp.reshape(edit.node_rows, (pop_size, 1))
    written_node_keys = scatter_rows(
        node_keys, node_slot, enp.reshape(edit.node_keys, (pop_size, 1)), node_count
    )
    written_node_attrs = scatter_rows(
        node_attrs,
        node_slot,
        enp.reshape(edit.node_attrs, (pop_size, 1, node_attr_width)),
        node_count,
    )
    written_conn_keys = scatter_rows(conn_keys, edit.conn_rows, edit.conn_keys, edge_count)
    written_conn_attrs = scatter_rows(conn_attrs, edit.conn_rows, edit.conn_attrs, edge_count)
    drop = enp.reshape(edit.drop, (pop_size, edge_count, 1))
    return GenomeTensors(
        written_node_keys,
        written_node_attrs,
        evoxc.select(drop, enp.full((), I_INF, dtype=conn_keys.dtype), written_conn_keys),
        evoxc.select(drop, enp.full((), float("nan"), dtype=conn_attrs.dtype), written_conn_attrs),
    )
