"""The gene payload record, its per-individual row count and host-only renumbering.

``GenomeTensors`` is the four-array, batch-first payload record the whole package takes and
returns (population on axis 0, static shapes): ``node_keys``/``conn_keys`` in ``int32`` with
``I_INF`` (``int32`` max) on absent rows, ``node_attrs``/``conn_attrs`` in ``float32`` with
NaN on absent rows.  The two sentinels never mix.

``valid_count`` is the one in-graph kernel left here: the number of existing rows of a gene
family, per individual, so a caller never loops over the population in Python.
``re_cound_idx`` is host-only: it converts with ``Tensor.numpy()`` and never feeds its result
back into a graph.
"""

from typing import NamedTuple

import evoxc
import evoxc.numpy as enp
import numpy as np

from ..common import I_INF, ConfigError, to_numpy


class GenomeTensors(NamedTuple):
    """The four payload arrays of a batch of genomes, population on axis 0.

    ``P`` is the population size, ``N`` ``max_nodes``, ``E`` ``max_conns``; ``NA``/``CK``/
    ``CA`` are ``node_gene.attr_width``, ``conn_gene.key_width`` and
    ``conn_gene.attr_width``.  All four shapes are static at trace time, so a change forces
    a rebuild of the cached executable.

    Row/key invariant: ``initialize`` writes nodes in key order, so the input keys occupy
    rows ``0..num_inputs-1`` and the output keys occupy rows
    ``max_nodes-num_outputs..max_nodes-1``.  Those rows are never deleted or moved, which is
    what makes ``get_input_idx()``/``get_output_idx()`` usable as *row indices*.  Hidden
    node keys are arbitrary and may move rows, so graph identity is compared on keys, never
    on row positions.

    Attributes:
        node_keys: ``(P, N) int32`` node key per row, ``I_INF`` for an absent row.
        node_attrs: ``(P, N, NA) float32`` node attribute rows, NaN for an absent row.
        conn_keys: ``(P, E, CK) int32`` ``(in_key, out_key)`` (plus ``historical_marker``
            when the conn gene declares it), ``I_INF`` for an absent row.
        conn_attrs: ``(P, E, CA) float32`` connection attribute rows (e.g. ``weight``),
            NaN for an absent row.
    """

    node_keys: evoxc.SymbolicTensor
    node_attrs: evoxc.SymbolicTensor
    conn_keys: evoxc.SymbolicTensor
    conn_attrs: evoxc.SymbolicTensor

    @property
    def field_names(self) -> tuple[str, ...]:
        """Field names in payload order, for generic payload walks.

        Returns:
            ``("node_keys", "node_attrs", "conn_keys", "conn_attrs")``.
        """
        return self._fields


# --------------------------------------------------------------------------- #
# private plumbing: int32 constants, key columns
# --------------------------------------------------------------------------- #


def _i32(value: int) -> evoxc.SymbolicTensor:
    """Rank-0 ``int32`` constant holding ``value``.

    Written out instead of a bare Python literal because ``int32`` tensor + Python ``int``
    promotes to ``int64`` in ``evoxc`` (plain ``np.result_type``), which would silently
    widen every index computation here.

    Args:
        value: Static integer baked into the trace.

    Returns:
        Rank-0 ``int32`` tensor inside the active trace.
    """
    return enp.full((), value, dtype=evoxc.int32)


def _key_column(kind_keys: evoxc.SymbolicTensor) -> evoxc.SymbolicTensor:
    """First key column of one gene family, ``(P, M)`` whatever its row width is.

    Args:
        kind_keys: ``(P, M)`` node keys, or ``(P, M, CK)`` connection keys.

    Returns:
        ``(P, M)`` ``int32`` column: both families store their primary key first
        (``in_key`` for a connection), so the ``I_INF`` "absent row" test reads the same
        column for either width.
    """
    if len(kind_keys.shape) == 3:
        return kind_keys[:, :, 0]
    return kind_keys


# --------------------------------------------------------------------------- #
# row bookkeeping
# --------------------------------------------------------------------------- #


def valid_count(kind_keys: evoxc.SymbolicTensor) -> evoxc.SymbolicTensor:
    """Count the existing rows of one gene family, per individual.

    Args:
        kind_keys: ``(P, M) int32`` key rows of a node *or* connection family (only the
            leading key column is inspected, so a ``(P, M, CK)`` connection key array is
            accepted too); ``I_INF`` marks an absent row.

    Returns:
        ``(P,) int32`` number of rows whose first key column is not ``I_INF``.

    Notes:
        The reduce returns ``int64`` in ``evoxc`` whatever the input dtype, so the result is
        cast back to ``int32`` to keep every downstream index computation narrow.
    """
    present = evoxc.not_equal(_key_column(kind_keys), _i32(I_INF))
    counted = evoxc.reduce_sum(evoxc.cast(present, evoxc.int32), axes=1)
    return evoxc.cast(counted, evoxc.int32)


# --------------------------------------------------------------------------- #
# host-only renumbering
# --------------------------------------------------------------------------- #


def re_cound_idx(
    node_keys: evoxc.SymbolicTensor,
    node_attrs: evoxc.SymbolicTensor,
    conn_keys: evoxc.SymbolicTensor,
    input_idx: list[int],
    output_idx: list[int],
) -> tuple[evoxc.SymbolicTensor, evoxc.SymbolicTensor, evoxc.SymbolicTensor]:
    """Renumber hidden node keys to be contiguous above the input/output keys (host-only).

    Host-only display helper: it converts its arguments with ``Tensor.numpy()``, applies the
    renumbering in Python and returns plain ``numpy`` arrays, which must never be fed back
    into a graph.  Input and output keys are left untouched (they are the external
    interface); hidden keys are replaced by consecutive integers starting at
    ``max(*input_idx, *output_idx) + 1``, in row order, and every connection key referring
    to a hidden node is rewritten accordingly.

    Args:
        node_keys: ``(N,) int32`` node key row of one genome, ``I_INF`` padded.
        node_attrs: ``(N, NA) float32`` node attribute row of the same genome, NaN padded.
        conn_keys: ``(E, CK) int32`` connection key rows of the same genome, ``I_INF``
            padded.
        input_idx: Keys (not rows) of the input nodes, kept verbatim.
        output_idx: Keys of the output nodes, kept verbatim.

    Returns:
        ``(node_keys, node_attrs, conn_keys)`` as ``numpy`` arrays: renumbered node keys,
        the unchanged node attributes (carried along so a payload can be rebuilt in one
        step) and renumbered connection keys.

    Raises:
        ConfigError: If ``input_idx`` and ``output_idx`` are both empty — there is then no
            external interface to renumber the hidden keys above — or if the arrays are
            batched (``node_keys`` not ``(N,)`` / ``conn_keys`` not ``(E, CK)``): a batch
            would otherwise be renumbered row by row through the wrong axis.

    Notes:
        Only columns 0 and 1 of ``conn_keys`` are rewritten: a third column is the
        ``historical_marker``, an identity, not a node reference.
    """
    keys = to_numpy(node_keys)
    attrs = to_numpy(node_attrs)
    conns = to_numpy(conn_keys)
    if keys.ndim != 1 or conns.ndim != 2:
        raise ConfigError(
            "re_cound_idx takes one genome: node_keys (N,) and conn_keys (E, CK), got "
            f"{keys.shape} and {conns.shape}"
        )
    fixed = [int(key) for key in (*input_idx, *output_idx)]
    if not fixed:
        raise ConfigError(
            "re_cound_idx needs at least one input or output key to renumber hidden keys above"
        )
    fixed_set = set(fixed)
    next_key = max(fixed) + 1
    old2new: dict[int, int] = {}
    new_keys = np.array(keys, copy=True)
    for row in range(new_keys.shape[0]):
        key = int(new_keys[row])
        if key == I_INF or key in fixed_set:  # padding row or external interface
            continue
        old2new[key] = next_key
        new_keys[row] = next_key
        next_key += 1
    new_conns = np.array(conns, copy=True)
    for row in range(new_conns.shape[0]):
        for column in (0, 1):  # (in_key, out_key); a marker column is not a reference
            key = int(new_conns[row, column])
            if key in old2new:
                new_conns[row, column] = old2new[key]
    return new_keys, attrs, new_conns
