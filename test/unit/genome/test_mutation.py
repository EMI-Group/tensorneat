"""Unit tests for structural and value mutation — invariants only.

What this module proves
-----------------------
Mutation is stochastic and batch-wide, so the tests assert the *invariants* an
operator must keep rather than a particular outcome (the exact draws differ from
the legacy library by construction — ``evoxc.random`` vs ``jax.random`` — and are
covered by parity only where they are deterministic):

* capacity is hard: ``max_nodes``, ``max_conns`` and the genome's
  ``max_in_degree`` are never exceeded. A refused add is a *no-op for that
  individual*, never a silent repair and never a raised error;
* at most **one** structural change per individual per generation (the four
  probabilities gate one shared draw), while value mutation always runs
  afterwards on every existing attribute row;
* keys stay unique and ``I_INF``-padded; structural keys are **caller-allocated**
  and honoured verbatim (``new_node_keys (P,)``, ``new_conn_keys (P, 3)``), so
  ``(seed, config, shape)`` — not an internal counter — decides the innovation
  numbers;
* rows the operator did not touch are bit-identical: no accidental rewrite of an
  attribute or a key column, which is what makes a mutation diff reviewable;
* a probability of ``0`` removes the operation entirely — the identity kernel.

Oracles used here (never the kernel itself)
-------------------------------------------
* counting and set algebra on the returned payload: node/connection counts before
  and after, per-individual ``(in_key, out_key)`` set membership, ``len(set(keys))``,
  ``isnan`` on the attribute rows and ``== I_INF`` on the key rows;
* hand-built topologies from ``support.payloads`` (dense chain, isolated node,
  ring) whose edge sets are known exactly, so "did the add-connection step
  duplicate an edge?" is a set question, not a statistical one;
* ``support.payloads``' padding convention (``I_INF`` for keys, ``NaN`` for
  attributes), which every operator must leave intact.

Gotchas this module encodes
---------------------------
* ``validate_schedule``-style reasoning does not apply here: a mutation may leave
  a genome temporarily inconsistent *as a network* (a node with no edges) — that
  is legal, and the forward pass handles it through ``NaN``. The invariants are
  the payload's, not the topology's.
* ``node_add`` consumes **two** connection rows (the identity edge and the copy
  edge) plus one node row, so it must be refused when only one connection row is
  free. ``first_free_row`` returning ``I_INF`` is the signal, and the operator is
  documented to gate the result with ``select`` — writing garbage into a padded
  row is the failure mode the tests look for.
* ``delete_node`` must never remove an input or output row; those are the
  genome's external interface and the row/key invariant depends on them. A
  mutation that deleted one would still pass a "count decreased" assertion, so
  the test checks the surviving key set explicitly.
* ``max_in_degree`` is enforced *per destination node*, not per genome: the check
  is "this destination would gain one more incoming edge than ``K``".

Backend is ``"numpy"`` throughout, through the shared runner only; no
``jax``/``torch``, no network, no GPU.
"""

from __future__ import annotations

import pytest


# --------------------------------------------------------------------------- #
# configuration
# --------------------------------------------------------------------------- #


def test_probabilities_outside_the_unit_interval_are_rejected(genome):
    """Invariant: a probability outside ``[0, 1]`` raises ``TensorNEATError`` at construction.

    Oracle: the documented configuration contract — validated in
    ``DefaultMutation.__init__`` so a bad configuration fails before the first
    ``evoxc.build``, not on the first generation. Each of the four knobs
    (``conn_add``, ``conn_delete``, ``node_add``, ``node_delete``) is exercised
    with ``-0.1`` and ``1.1``, and the error type is asserted (a bare
    ``ValueError`` escapes the package's hierarchy).
    """
    pytest.skip(
        "not implemented: out-of-range mutation probabilities raise TensorNEATError "
        "(owner: genome/operations/mutation/default.DefaultMutation.__init__)"
    )


# --------------------------------------------------------------------------- #
# the identity and shape contract
# --------------------------------------------------------------------------- #


def test_zero_probabilities_are_the_identity(genome, payloads, run_kernel, evoxc_key):
    """Invariant: with all four probabilities ``0`` the payload is returned unchanged.

    Oracle: bit equality against the input payload (all four arrays, ``-0.0`` and
    ``NaN`` included). A probability of ``0`` is documented to remove its
    operation from the graph, so even a *legal* structural change is a failure
    here; the check also catches an operator that resets padding or re-sorts rows
    as a side effect of running.
    """
    pytest.skip(
        "not implemented: zero probabilities leave the payload bit-identical "
        "(owner: genome/operations/mutation/default.DefaultMutation.mutate_structure)"
    )


def test_shapes_and_dtypes_are_preserved(genome, payloads, run_kernel, evoxc_key):
    """Invariant: mutation never resizes the payload.

    Oracle: shape and dtype equality with the input for all four arrays —
    ``node_keys (P, N) int32``, ``node_attrs (P, N, NA) float32``,
    ``conn_keys (P, E, CK) int32``, ``conn_attrs (P, E, CA) float32``. Shapes are
    static at build time, so an operator that grew an array would force a rebuild
    (and break the pipeline's per-shape executable cache); dtypes are part of the
    payload contract, so a float key column is a failure even at the right shape.
    """
    pytest.skip(
        "not implemented: mutation preserves the four payload shapes and dtypes "
        "(owner: genome/base.BaseGenome.execute_mutation, genome/utils.GenomeTensors)"
    )


# --------------------------------------------------------------------------- #
# capacity invariants
# --------------------------------------------------------------------------- #


def test_node_capacity_is_never_exceeded(genome, payloads, run_kernel, evoxc_key):
    """Invariant: ``node_add=1.0`` repeated over many generations never exceeds ``max_nodes``.

    Oracle: a row count per individual — the number of ``node_keys != I_INF``
    entries — must stay ``<= max_nodes``, and once the family is full (a payload
    whose only free rows are gone) the operator must leave every individual
    unchanged instead of writing into a padded row. Several generations from the
    same payload are run so the "full" case is reached deterministically.
    """
    pytest.skip(
        "not implemented: node count stays within max_nodes, a full family is a no-op "
        "(owner: genome/utils.first_free_row, "
        "genome/operations/mutation/default.DefaultMutation._mutate_add_node)"
    )


def test_connection_capacity_is_never_exceeded(genome, payloads, run_kernel, evoxc_key):
    """Invariant: ``conn_add=1.0`` repeated over many generations never exceeds ``max_conns``.

    Oracle: the connection row count per individual (``conn_keys[:, :, 0] !=
    I_INF``) stays ``<= max_conns``; a full connection family refuses the add.
    ``node_add`` needs two free connection rows at once, so the same assertion is
    run with ``node_add=1.0`` on a payload with exactly one free row: the split
    must be refused rather than half-applied (node without its two edges).
    """
    pytest.skip(
        "not implemented: connection count stays within max_conns, "
        "a half-splittable genome is unchanged "
        "(owner: genome/operations/mutation/default.DefaultMutation._mutate_add_node, "
        "_mutate_add_conn)"
    )


def test_in_degree_never_exceeds_the_genome_window(genome, payloads, run_kernel, evoxc_key):
    """Invariant: no destination node gains more than ``max_in_degree`` incoming edges.

    Oracle: the in-degree histogram of every individual's ``conn_keys``, compared
    against ``genome.max_in_degree``. The window width is the schedule's static
    ``K``; a mutation that exceeded it would make the next ``transform`` report a
    positive ``edge_overflow`` and silently drop edges, so the operator must refuse
    that candidate (a no-op for the individual) rather than rely on the schedule
    to absorb it.
    """
    pytest.skip(
        "not implemented: added connections respect the max_in_degree window "
        "(owner: genome/operations/mutation/default.DefaultMutation._mutate_add_conn)"
    )


# --------------------------------------------------------------------------- #
# keys
# --------------------------------------------------------------------------- #


def test_new_node_and_connection_keys_are_caller_allocated_and_honoured(
    genome, payloads, run_kernel, evoxc_key
):
    """Invariant: an inserted gene carries exactly the key the caller allocated.

    Oracle: set membership. With ``node_add=1.0`` the inserted node's key is
    ``new_node_keys[p]``; with ``conn_add=1.0`` the new edge's
    ``(in_key, out_key)`` is the chosen pair and its ``historical_marker`` column
    (when the conn gene declares one) is ``new_conn_keys[p, 2]``; an add-node split
    uses ``new_conn_keys[p, 0]`` for the identity edge and ``[p, 1]`` for the copy
    edge. Nothing is drawn from the RNG: the operator must not invent keys, because
    innovation numbers have to be globally unique across the population and are
    allocated by the algorithm.
    """
    pytest.skip(
        "not implemented: caller-allocated new_node_keys/new_conn_keys are inserted verbatim "
        "(owner: genome/operations/mutation/default.DefaultMutation.mutate_structure, "
        "genome/utils.add_node)"
    )


def test_keys_stay_unique_and_padding_sentinels_are_intact(
    genome, payloads, run_kernel, evoxc_key
):
    """Invariant: no duplicate key within an individual, and absent rows keep their sentinels.

    Oracle: ``len(set(node_keys[p]))`` over the present rows, and the pairing
    "``node_keys == I_INF`` iff ``node_attrs`` is all-``NaN``" (likewise for
    connections). Both sentinels are checked together because a half-cleared row —
    an ``I_INF`` key with finite attributes, or a finite key with ``NaN``
    attributes — breaks every downstream kernel while looking correct in one array.
    Keys allocated above the current maximum (the algorithm's convention) must also
    come out *strictly greater* than every pre-existing key of that individual, so
    "new gene" stays decidable from the key alone and no inserted key aliases an
    older gene.
    """
    pytest.skip(
        "not implemented: keys stay unique and I_INF/NaN padding stays paired "
        "(owner: genome/utils.delete_conn_by_pos, delete_node_by_pos, add_conn, add_node)"
    )


# --------------------------------------------------------------------------- #
# locality
# --------------------------------------------------------------------------- #


def test_rows_outside_the_mutated_gene_are_bit_identical(
    genome, payloads, run_kernel, evoxc_key
):
    """Invariant: an operator touches only the rows it names.

    Oracle: a row-by-row diff against the input payload. With ``conn_delete=1.0``
    exactly the chosen connection's ``CA`` attribute columns and ``CK`` key columns
    change (to ``NaN``/``I_INF``), every other row is bit-identical; with
    ``node_delete=1.0`` exactly the deleted node's row and the rows of the
    connections touching it change. A kernel that rebuilt the whole table (a
    ``gather``-then-``scatter`` over all rows) would show up here as a diff in
    untouched rows even when the values happen to look plausible.
    """
    pytest.skip(
        "not implemented: structural mutation changes exactly the rows it names "
        "(owner: genome/operations/mutation/default.DefaultMutation._mutate_delete_conn, "
        "_mutate_delete_node)"
    )


def test_value_mutation_never_touches_key_columns(genome, payloads, run_kernel, evoxc_key):
    """Invariant: ``mutate_values`` changes float attributes only.

    Oracle: ``np.array_equal`` on ``node_keys``/``conn_keys`` before and after the
    stage, plus "the attribute rows of present genes stay finite" (a mutation of a
    padded row would produce a finite value where ``NaN`` belongs, and a mutation
    of a key column would corrupt the topology). Value mutation is documented to be
    one batched draw per gene family, so it cannot be a no-op even at the default
    rates; the test therefore asserts *which* columns moved, not that they moved.
    """
    pytest.skip(
        "not implemented: value mutation leaves both key arrays untouched "
        "(owner: genome/operations/mutation/default.DefaultMutation.mutate_values)"
    )


# --------------------------------------------------------------------------- #
# structural correctness of each operation
# --------------------------------------------------------------------------- #


def test_at_most_one_structural_change_per_individual(
    genome, payloads, run_kernel, evoxc_key
):
    """Invariant: the four gates share one draw, so at most one structural change applies.

    Oracle: per-individual counts — a node count that moved by at most ``1``, a
    connection count that moved by at most ``2`` (an add-node split is one delete
    plus two adds), and never both a node and a connection *deletion* in the same
    individual. With all four probabilities at ``1.0`` a per-operation draw would
    make each individual change several times per generation; the documented
    "independent probabilities, one shared draw" rule forbids that.
    """
    pytest.skip(
        "not implemented: at most one structural mutation per individual per generation "
        "(owner: genome/operations/mutation/default.DefaultMutation.mutate_structure)"
    )


def test_add_connection_never_duplicates_an_existing_edge(
    genome, payloads, run_kernel, evoxc_key
):
    """Invariant: the added ``(in_key, out_key)`` pair is not already present.

    Oracle: set membership on the payload's own connection keys, computed per
    individual before and after. A duplicate edge would double that source's
    contribution in the aggregation without changing the topology, which is the
    classic silent bug of a random-pair add; ``support.payloads`` gives a dense
    network where most pairs exist, so a naive draw collides often.
    """
    pytest.skip(
        "not implemented: add-connection refuses an existing edge "
        "(owner: genome/operations/mutation/default.DefaultMutation._mutate_add_conn)"
    )


def test_add_connection_never_closes_a_cycle_in_a_feedforward_genome(
    genome, payloads, run_kernel, evoxc_key
):
    """Invariant: a feedforward genome never gains an edge that closes a cycle.

    Oracle: the genome's own ``check_cycles`` predicate is *not* used as the
    oracle (it is part of the code under test); the test recomputes reachability
    in Python on the returned topology — after the mutation, the directed graph
    must still be acyclic. On the dense chain every backward edge would close a
    cycle, so any accepted candidate is a visible violation; on a recurrent genome
    the very same mutation is legal, which the test asserts on the ring payload.
    """
    pytest.skip(
        "not implemented: feedforward mutation keeps the graph acyclic "
        "(owner: genome/operations/mutation/default.DefaultMutation._mutate_add_conn, "
        "genome/utils.check_cycles)"
    )


def test_add_node_splits_one_connection_into_two(
    genome, payloads, run_kernel, evoxc_key
):
    """Invariant: the split node sits between the old endpoints with an identity and a copy edge.

    Oracle: hand enumeration on a payload with a single known edge ``a->b``:
    ``node_add=1.0`` must leave ``a->b`` deleted, add ``a->n`` with
    ``conn_gene.new_identity_attrs`` (``weight = 1.0``: it reproduces its source)
    and ``n->b`` with the *original* attributes of ``a->b`` (so the network's
    output is preserved). The new node's attribute row is the node gene's identity
    row. Counts: nodes ``+1``, connections ``+1``; the source and destination key
    sets otherwise unchanged.
    """
    pytest.skip(
        "not implemented: add-node splits the chosen edge into identity + copy "
        "(owner: genome/operations/mutation/default.DefaultMutation._mutate_add_node)"
    )


def test_delete_node_never_removes_an_input_or_output_row(
    genome, payloads, run_kernel, evoxc_key
):
    """Invariant: ``node_delete`` clears a hidden row and every edge touching it, nothing else.

    Oracle: the surviving key set of the input and output rows — ``0..num_inputs-1``
    and ``max_nodes-num_outputs..max_nodes-1`` must still be present and unchanged
    — plus "no connection row references the deleted key any more" (the
    ``delete_conns_by_key`` contract, which is data-dependent because a node may
    have any number of edges). On ``payloads.isolated_payload`` the only hidden row
    has no edges, so the delete must clear it and change no connection row at all.
    """
    pytest.skip(
        "not implemented: node deletion keeps the I/O rows and detaches every touching edge "
        "(owner: genome/operations/mutation/default.DefaultMutation._mutate_delete_node, "
        "genome/utils.delete_conns_by_key)"
    )
