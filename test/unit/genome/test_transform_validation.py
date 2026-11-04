"""Unit tests for ``BaseGenome.transform``'s batch-first shape validation.

What this module proves
-----------------------
``transform`` is the genome's batch-first entry point: the population is axis 0 of
``node_keys (P, max_nodes)`` and ``conn_keys (P, max_conns, key_width)``. A caller
that slices one individual out of a payload hands it bare rows (``node_keys
(max_nodes,)``, ``conn_keys (max_conns, key_width)``); those calls must raise
:class:`TensorNEATError` naming the expected rank and the missing leading
population axis, never the raw ``ValueError`` of the shape unpack. Two arrays of
different population sizes are rejected the same way, while the two correct
spellings — a ``GenomeTensors`` payload and the ``(node_keys, conn_keys)`` pair —
still derive a ``Transformed``.

Oracles used here (never the implementation)
--------------------------------------------
The contract is the documented shape tuple itself: ``(P, max_nodes)``,
``(P, max_conns, key_width)`` and the two arrays' common leading dimension. The
positive cases assert the returned ``Transformed`` type and its leading
dimension, not a schedule value (``./test_payload.py`` owns the values).

Gotchas this module encodes
---------------------------
* The check reads only ``.shape``, so it behaves the same on host arrays and on
  ``evoxc.SymbolicTensor`` inputs; the negative cases run through
  ``support.runner`` to pin that the error propagates out of ``evoxc.build``.
* The population size is validated against the *other* array, never against the
  genome's static sizes: a payload built for a different batch is legal.
Backend is ``"numpy"`` throughout; no ``jax``/``torch``.
"""

from __future__ import annotations

import numpy as np
import pytest


def _initialized_genome(payloads, run_kernel, evoxc_key, pop_size=2):
    """A small ``DefaultGenome``, its state, and one initialized payload batch.

    Returns:
        ``(genome, state, payload)`` where ``payload`` is the ``GenomeTensors``
        returned by ``initialize`` (host numpy leaves, batch-first).
    """
    genome = payloads.default_genome(
        num_inputs=2, num_outputs=1, max_nodes=8, max_conns=16
    )
    state = payloads.genome_state(genome)
    payload = run_kernel(genome.initialize, state, evoxc_key, pop_size)
    return genome, state, payload


def test_transform_rejects_a_per_individual_node_row(
    payloads, common, runner, run_kernel, evoxc_key
):
    """Invariant: a rank-1 ``node_keys`` names the missing population axis.

    Oracle: the documented ``(P, max_nodes)`` spelling and the ``Raises`` clause
    of ``transform``; the assertion checks the exception type plus the stable
    shape and population substrings, not the whole message.
    """
    genome, state, payload = _initialized_genome(payloads, run_kernel, evoxc_key)

    with pytest.raises(common.TensorNEATError) as excinfo:
        runner.run_defn(
            lambda node_keys, conn_keys: genome.transform(state, node_keys, conn_keys),
            payload.node_keys[0],
            payload.conn_keys[0],
        )

    message = str(excinfo.value)
    assert "(P, max_nodes)" in message
    assert "population" in message


def test_transform_rejects_conn_keys_without_a_population_axis(
    payloads, common, runner, run_kernel, evoxc_key
):
    """Invariant: a rank-2 ``conn_keys`` names the missing population axis.

    Oracle: the documented ``(P, max_conns, key_width)`` spelling; the batched
    ``node_keys`` on its own is valid, so only the connection rank can raise.
    """
    genome, state, payload = _initialized_genome(payloads, run_kernel, evoxc_key)

    with pytest.raises(common.TensorNEATError) as excinfo:
        runner.run_defn(
            lambda node_keys, conn_keys: genome.transform(state, node_keys, conn_keys),
            payload.node_keys,
            payload.conn_keys[0],
        )

    message = str(excinfo.value)
    assert "(P, max_conns, key_width)" in message
    assert "population" in message


def test_transform_rejects_mismatched_population_sizes(payloads, common, runner):
    """Invariant: ``node_keys`` and ``conn_keys`` must agree on the population size.

    Oracle: the documented rule that only the *internal* agreement of one call is
    a contract; the arrays are hand-built so the mismatch cannot come from a
    payload factory.
    """
    genome = payloads.default_genome(
        num_inputs=2, num_outputs=1, max_nodes=8, max_conns=16
    )
    state = payloads.genome_state(genome)
    node_keys = np.full((4, 8), payloads.I_INF, dtype=np.int32)
    conn_keys = np.full((3, 16, 2), payloads.I_INF, dtype=np.int32)

    with pytest.raises(common.TensorNEATError) as excinfo:
        genome.transform(state, node_keys, conn_keys)

    message = str(excinfo.value)
    assert "different population sizes" in message
    assert "population" in message


def test_transform_accepts_the_batched_genome_tensors_payload(
    payloads, common, runner, run_kernel, evoxc_key
):
    """Invariant: a batch-first ``GenomeTensors`` still derives a ``Transformed``.

    Oracle: the documented return type and field tuple; the leading dimension of
    every field is the payload's population size.
    """
    genome, state, payload = _initialized_genome(payloads, run_kernel, evoxc_key)

    transformed = runner.run_defn(lambda batch: genome.transform(state, batch), payload)

    assert isinstance(transformed, common.Transformed)
    assert tuple(transformed._fields) == common.Transformed._fields
    for name in transformed._fields:
        assert np.asarray(getattr(transformed, name)).shape[0] == payload.node_keys.shape[0]


def test_transform_accepts_the_batched_node_and_conn_key_pair(
    payloads, common, runner, run_kernel, evoxc_key
):
    """Invariant: the ``(node_keys, conn_keys)`` pair derives the same schedule.

    Oracle: both accepted spellings are compared field by field, and the result
    keeps one leading population axis per ``Transformed`` field.
    """
    genome, state, payload = _initialized_genome(payloads, run_kernel, evoxc_key)

    pair = runner.run_defn(
        lambda node_keys, conn_keys: genome.transform(state, node_keys, conn_keys),
        payload.node_keys,
        payload.conn_keys,
    )
    whole = runner.run_defn(lambda batch: genome.transform(state, batch), payload)

    assert isinstance(pair, common.Transformed)
    assert isinstance(whole, common.Transformed)
    for name in pair._fields:
        assert np.array_equal(
            np.asarray(getattr(pair, name)), np.asarray(getattr(whole, name))
        ), name
