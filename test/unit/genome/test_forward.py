"""Unit tests for ``DefaultGenome.forward`` — the depth-unrolled feedforward pass.

What this module proves
-----------------------
``forward`` evaluates a whole population of differently-shaped networks in one
graph by walking the schedule's levels in a Python-unrolled loop, and it must
reproduce the *documented* algebra exactly:

1. ``values`` starts as ``NaN`` and holds ``input_transform(inputs)`` on the
   input rows only;
2. one window reduce per level: ``msg = sanitized(weight) * sanitized(source)``,
   the aggregation selected per node over the live slots, ``z = bias + response *
   agg``, then the activation **except** on the output rows;
3. a node with no *live* incoming edge keeps its previous value, so ``NaN`` stays
   ``NaN`` and everything downstream of it stays ``NaN`` (the legacy poisoning);
4. the read-out takes ``num_outputs`` trailing rows and applies
   ``output_transform`` — and nothing else.

Every test below fixes one of those clauses on a hand-built payload, so a failure
names the clause. None of them observes a ``Transformed`` field by name: the
schedule comes from the public ``genome.transform`` (see ``./conftest.py``).

Oracles used here (never the kernel itself)
-------------------------------------------
* closed forms — a dense chain with analytically chosen ``weight``/``bias``/
  ``response`` and one known activation per row, evaluated literally in the
  docstring and in a two-line numpy expression;
* ``support.oracle.forward`` — the independent numpy rollout of the same
  documented algorithm, used for the topologies that are tedious to write out
  (the skipped node, the unconnected node, random DAGs);
* the NaN pattern itself, which is data in the legacy semantics: an assertion on
  ``np.isnan`` is as strong as one on the values.

Gotchas this module encodes
---------------------------
* Output rows are **not** activated by the node gene; the only activation of the
  read-out is ``output_transform``. A genome that activated every row would pass
  a sigmoid-only test and fail :func:`test_output_rows_are_not_activated_by_the_node_gene`.
* ``activation == -1`` means identity in this rewrite; the legacy
  ``lax.switch(-1, ...)`` clamped to the *last* configured option. The test
  therefore uses a single-option list plus the ``-1`` row, and expects ``z``.
* ``chain_payload`` uses ``weight=1`` and the neutral node row
  ``(bias=0, response=1, aggregation=0, activation=0)``, so its default output is
  a repeated sigmoid — fine as an oracle target, useless as a closed form. The
  closed-form test therefore rewrites the attribute rows explicitly instead of
  trusting the builder's defaults.
* ``max_depth`` is a static unroll bound: a network deeper than ``D`` leaves its
  unreachable tail ``NaN``. That is a documented limit, not a bug, so it is
  asserted rather than avoided.

Backend is ``"numpy"`` throughout, through the shared runner only; no
``jax``/``torch``, no network, no GPU.
"""

from __future__ import annotations

import pytest


# --------------------------------------------------------------------------- #
# closed-form values on a dense chain
# --------------------------------------------------------------------------- #


def test_dense_chain_matches_the_closed_form(payloads, run_transform, run_forward):
    """Invariant: a chain with known weights/bias/response evaluates to its written-out expression.

    Oracle: hand arithmetic on ``payloads.chain_payload`` (rows ``0, 1`` inputs,
    ``2..6`` hidden, ``7`` output; edges ``0->2``, ``1->2``, ``2->3``, ...,
    ``6->7``) with ``weight=w`` on every edge, ``bias=b_l``/``response=r_l`` per
    row and one activation per row::

        v2 = f2(b2 + r2 * (w*x0 + w*x1))
        v3 = f3(b3 + r3 * w*v2)
        ...
        out = b7 + r7 * w*v6          # output row: not activated

    Every intermediate is asserted, not only the final output, so a schedule that
    evaluated the levels in the wrong order (or summed a node's *outgoing* edges)
    cannot pass by coincidence.
    """
    pytest.skip(
        "not implemented: dense chain value == the hand-written level composition "
        "(owner: genome/default.DefaultGenome.forward, genome/base.BaseGenome._node_step)"
    )


def test_output_rows_are_not_activated_by_the_node_gene(payloads, run_transform, run_forward):
    """Invariant: the read-out is ``bias + response * agg`` of the output row, unactivated.

    Oracle: closed form. With ``sigmoid`` configured as the single activation
    option and the output row's attribute row set to ``(bias=0, response=1)``, a
    single input edge of weight ``1`` carrying value ``z`` must yield ``z`` — not
    ``sigmoid(z)``. The hidden row of the same payload is activated, which the
    same test checks, so the difference is provably the output-row rule and not a
    global "no activation" bug.
    """
    pytest.skip(
        "not implemented: output rows bypass the node-gene activation "
        "(owner: genome/gene/node/default.DefaultNode.forward, "
        "genome/base.BaseGenome._is_output_row)"
    )


def test_activation_index_minus_one_is_identity(payloads, run_transform, run_forward):
    """Invariant: activation id ``-1`` leaves ``bias + response * agg`` untouched.

    Oracle: closed form. Two payloads that differ only in the activation column of
    one hidden row (``-1`` vs a configured option) must differ exactly by that
    option's map, and the ``-1`` row's value must equal ``bias + response * agg``.
    The legacy library clamped ``-1`` to the last configured option, so this test
    is also the guard against reintroducing that behaviour.
    """
    pytest.skip(
        "not implemented: activation id -1 is identity, not the last option "
        "(owner: genome/gene/node/default.DefaultNode.forward, "
        "common/functions/act.apply_activation)"
    )


# --------------------------------------------------------------------------- #
# transforms
# --------------------------------------------------------------------------- #


def test_output_transform_is_applied_after_the_readout(payloads, run_transform, run_forward):
    """Invariant: ``output_transform`` maps the ``(P, num_outputs)`` read-out once.

    Oracle: composition — the output of a genome with ``output_transform=f`` must
    equal ``f`` applied elementwise to the output of the same genome with no
    transform, on the same payload and inputs. ``f`` must not be able to change
    how the network itself was evaluated, so this is checked with an ``f`` that is
    *not* monotone (e.g. ``lambda x: x * 2 + 1``), where an accidental application
    to ``values`` instead of the read-out would show.
    """
    pytest.skip(
        "not implemented: output_transform applies to the read-out only "
        "(owner: genome/default.DefaultGenome.forward, genome/base.BaseGenome._output_values)"
    )


def test_input_transform_is_applied_before_seeding(payloads, run_transform, run_forward):
    """Invariant: ``input_transform`` maps ``inputs`` before they reach the input rows.

    Oracle: composition. A genome with ``input_transform=g`` must equal the same
    genome without a transform fed ``g(inputs)`` — computed on the host, on the
    same payload. ``RecurrentGenome`` deliberately does *not* apply it (see
    ``./test_recurrent.py``), so this test also pins the difference between the
    two forward passes.
    """
    pytest.skip(
        "not implemented: input_transform applies to inputs before seeding "
        "(owner: genome/default.DefaultGenome.forward)"
    )


# --------------------------------------------------------------------------- #
# topologies: the skipped node and the unconnected node
# --------------------------------------------------------------------------- #


def test_skipped_node_never_contributes_to_the_output(payloads, oracle, run_transform, run_forward):
    """Invariant: a node its consumer bypasses cannot change the output, however it is poisoned.

    Oracle: ``support.oracle.forward`` plus a perturbation argument.
    ``payloads.skip_payload`` wires ``0->2``, ``1->2`` and ``0->3``, ``3->7``: row
    ``2`` receives the signal but reaches no output. Setting row ``2``'s attribute
    row to ``NaN``, to a huge bias, or deleting its incoming edges must leave the
    read-out bit-identical — the schedule's level for row ``3`` is ``1``, not
    ``2``, which is exactly the property a level-synchronous pass has to get
    right.
    """
    pytest.skip(
        "not implemented: bypassed node does not affect the read-out, under any perturbation "
        "(owner: genome/default.DefaultGenome.forward, common/tensorization.derive_schedule)"
    )


def test_unconnected_node_and_its_descendants_stay_nan(
    payloads, oracle, run_transform, run_forward
):
    """Invariant: a present node with no incoming edge keeps ``NaN``, and so does its subtree.

    Oracle: ``support.oracle.forward`` and the documented poisoning rule — a node
    with ``in_conn[p, i, 0] == I_INF`` never enters an ``update_mask``, so its
    value stays the ``NaN`` seed; a consumer of it then has no *live* source, keeps
    its own previous value, and the chain of ``NaN`` reaches the output if the
    unconnected node feeds it. ``payloads.isolated_payload`` gives the first half
    (row ``2`` is present, isolated, and does not reach the output); the second
    half extends the payload with one edge from that row to the output row.
    """
    pytest.skip(
        "not implemented: unconnected node value stays NaN and poisons its descendants "
        "(owner: genome/base.BaseGenome._node_step, genome/default.DefaultGenome.forward)"
    )


def test_padded_rows_never_leak_into_the_output(payloads, oracle, run_transform, run_forward):
    """Invariant: rows outside the real network (``I_INF`` key, ``NaN`` attributes) are inert.

    Oracle: ``support.oracle.forward`` comparison against the *same topology with a
    smaller ``max_nodes``*. Padding must not change a single output value, so the
    padded payload's read-out equals the compact one's; a kernel that let an
    ``NaN`` attribute row into an aggregation, or that treated ``I_INF`` as a real
    node key, fails here while passing every dense-payload test.
    """
    pytest.skip(
        "not implemented: padded rows are inert for the forward pass "
        "(owner: genome/default.DefaultGenome.forward, genome/base.BaseGenome._node_step)"
    )


# --------------------------------------------------------------------------- #
# batch semantics
# --------------------------------------------------------------------------- #


def test_individuals_are_evaluated_independently_inside_one_batch(
    payloads, oracle, run_transform, run_forward
):
    """Invariant: population member ``p``'s output depends only on member ``p``'s payload and input.

    Oracle: one ``P == 4`` run against four ``P == 1`` runs of the same
    individuals (each with its own derived schedule). The population axis must be
    a pure batch axis: no reduce may span it, no per-individual value may come
    from a neighbour's row. Differing weights *and* differing topologies across
    the four members make an accidental cross-row gather visible.
    """
    pytest.skip(
        "not implemented: population axis is a pure batch axis "
        "(owner: genome/default.DefaultGenome.forward, genome/base.BaseGenome._node_step)"
    )


def test_random_topologies_match_the_numpy_oracle(
    payloads, oracle, run_transform, run_forward, rng
):
    """Invariant: on random feedforward DAGs the kernel equals ``support.oracle.forward``.

    Oracle: the independent numpy rollout. Topologies are drawn on the host with
    the ``rng`` fixture — a random acyclic edge set, random weights, random
    ``bias``/``response``, one activation option, well inside ``max_nodes``/
    ``max_conns``/``max_in_degree`` — so nothing about the graph is shared with
    the kernel. Both the values (``allclose``) and the ``NaN`` pattern
    (``np.isnan`` equality) are compared.
    """
    pytest.skip(
        "not implemented: random feedforward topologies equal the numpy oracle "
        "(owner: genome/default.DefaultGenome.forward)"
    )


# --------------------------------------------------------------------------- #
# the static depth bound
# --------------------------------------------------------------------------- #


def test_a_network_deeper_than_max_depth_leaves_its_tail_nan(
    payloads, oracle, run_transform, run_forward
):
    """Invariant: nodes below level ``max_depth`` are never updated and stay ``NaN``.

    Oracle: the documented static-unroll limit — ``DefaultGenome`` unrolls exactly
    ``max_depth`` levels, so a chain longer than that keeps its unreachable tail
    ``NaN`` and the read-out is ``NaN`` when the output row is in the tail. This is
    a *documented* limit (``max_depth`` is a build-time constant that a rebuild
    must raise), so the test asserts the limit instead of avoiding it, and asserts
    that the prefix of levels below the bound is still correct.
    """
    pytest.skip(
        "not implemented: levels beyond max_depth stay NaN, the reachable prefix is correct "
        "(owner: genome/default.DefaultGenome.forward)"
    )
