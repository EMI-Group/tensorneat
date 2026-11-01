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
* closed forms — a chain with analytically chosen ``weight``/``bias``/
  ``response`` and one known activation per row, evaluated literally in the
  docstring and in a short numpy expression;
* ``support.oracle.forward`` — the independent numpy rollout of the same
  documented algorithm, used for the topologies that are tedious to write out
  (the skipped node, the unconnected node, random DAGs);
* the NaN pattern itself, which is data in the legacy semantics: an assertion on
  ``np.isnan`` is as strong as one on the values.

Gotchas this module encodes
---------------------------
* The read-out is only the last ``num_outputs`` rows, so "every intermediate
  value" is asserted with *probe* output rows: the chain keeps its own hidden
  rows and additionally feeds each of them to a distinct output row through an
  edge of weight ``1`` and a ``(bias=0, response=1)`` attribute row, which makes
  the unactivated read-out equal that hidden row's value. The same layout is
  what lets :func:`test_a_network_deeper_than_max_depth_leaves_its_tail_nan`
  compare a computed level against the ``NaN`` tail.
* Output rows are **not** activated by the node gene; the only activation of the
  read-out is ``output_transform``. A genome that activated every row would pass
  a sigmoid-only test and fail
  :func:`test_output_rows_are_not_activated_by_the_node_gene`.
* ``activation == -1`` means identity in this rewrite; the legacy
  ``lax.switch(-1, ...)`` clamped to the *last* configured option. The test
  therefore uses a single-option genome plus the ``-1`` row, and expects ``z``.
* ``inputs`` is *not* tiled over the population: the kernel documents
  ``(P, num_inputs)`` and the caller batches, which is why the population test
  passes ``(P, num_inputs)`` data while its single-individual runs pass ``(1,
  num_inputs)``.
* The payload's ``max_nodes`` must agree with the genome's static ``max_nodes``:
  the schedule's ``output_slot`` is derived from the genome's own output *keys*
  (the last ``num_outputs`` rows), so a payload padded past the genome's
  capacity leaves that slot unmatched. The padding test therefore builds one
  genome per shape and compares the two *read-outs*, not one genome across two
  capacities.
* ``max_depth`` is a static unroll bound: a network deeper than ``D`` leaves its
  unreachable tail ``NaN``. That is a documented limit, not a bug, so it is
  asserted rather than avoided.

Backend is ``"numpy"`` throughout, through the shared runner only; no
``jax``/``torch``, no network, no GPU.
"""

from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------- #
# shared helpers: one genome factory, one payload attribute row, one evaluator
# --------------------------------------------------------------------------- #


def _genome(
    genome,
    *,
    num_inputs: int = 2,
    num_outputs: int = 1,
    max_nodes: int = 8,
    max_conns: int = 16,
    max_in_degree: int = 4,
    **kwargs,
):
    """A ``DefaultGenome`` whose static shapes match the payload of the same test.

    ``max_in_degree`` is the window width ``K`` the schedule allocates per node;
    every payload here keeps its real in-degree at or below ``4``, so nothing is
    dropped into ``edge_overflow`` (which would quietly change the oracle's
    answer). ``kwargs`` are forwarded to the constructor (``node_gene``,
    ``input_transform``, ``output_transform``, ``max_depth`` ...).

    Args:
        genome: The ``tensorneat.genome`` package (the module-scoped fixture).
        num_inputs: Input node count / input width.
        num_outputs: Output node count / read-out width.
        max_nodes: Static node capacity ``N``; must equal the payload's.
        max_conns: Static connection capacity ``E``; must equal the payload's.
        max_in_degree: Window width ``K`` of every node.
        **kwargs: Extra ``DefaultGenome`` keywords.

    Returns:
        The constructed genome (``DefaultGenome`` builds its genes and operators
        per instance when they are not given).
    """
    return genome.DefaultGenome(
        num_inputs=num_inputs,
        num_outputs=num_outputs,
        max_nodes=max_nodes,
        max_conns=max_conns,
        max_in_degree=max_in_degree,
        **kwargs,
    )


def _node_row(bias=0.0, response=1.0, aggregation=0.0, activation=0.0):
    """One ``DefaultNode`` attribute row, in the documented column order.

    ``support.payloads`` reads ``(bias, response, aggregation id, activation id)``
    — ``genome/gene/node/default.py::DefaultNode.attr_specs`` order — and both ids
    are *positions* in the gene's configured option lists (``-1`` is identity and
    is not an option index).

    Args:
        bias: Additive term of ``bias + response * agg``.
        response: Multiplicative term of ``bias + response * agg``.
        aggregation: Aggregation option position.
        activation: Activation option position, or ``-1`` for identity.

    Returns:
        A 4-tuple of floats, exactly what :func:`support.payloads.set_node` writes.
    """
    return (float(bias), float(response), float(aggregation), float(activation))


def _evaluate(run_transform, run_forward, payloads, genome_object, payload, inputs):
    """Derive the schedule of ``payload`` and evaluate it through the public kernels.

    Args:
        run_transform: The ``run(genome_object, state, payload) -> Transformed``
            fixture of ``./conftest.py``.
        run_forward: The ``run(genome_object, state, transformed, payload, inputs)
            -> (P, num_outputs)`` fixture of ``./conftest.py``.
        payloads: The ``support.payloads`` module (the module-scoped fixture).
        genome_object: The genome whose statics match ``payload``.
        payload: The payload batch to evaluate.
        inputs: ``(P, num_inputs)`` network inputs, already batch-sized.

    Returns:
        ``(transformed, outputs)``: the schedule (host numpy, for the *oracle*
        comparisons) and the ``(P, num_outputs)`` read-out of the kernel.
    """
    state = payloads.genome_state(genome_object)
    transformed = run_transform(genome_object, state, payload)
    return transformed, run_forward(genome_object, state, transformed, payload, inputs)


def _sigmoid(z):
    """The reference logistic sigmoid (the genome's ``ACT.sigmoid`` option)."""
    return 1.0 / (1.0 + np.exp(-z))


#: Inputs used by every small hand-built network below: ``x0 + x1 = 0.1``, so a
#: sigmoid of the raw sum (``0.525``) is far from a sigmoid of a sigmoid of it.
INPUTS = np.array([[0.3, -0.2]], dtype=np.float32)


# --------------------------------------------------------------------------- #
# closed-form values on a dense chain
# --------------------------------------------------------------------------- #


def test_dense_chain_matches_the_closed_form(
    payloads, common, genome, oracle, run_transform, run_forward
):
    """Invariant: a chain with known weights/bias/response evaluates to its written-out expression.

    Oracle: hand arithmetic on a 13-row layout (rows ``0, 1`` inputs, ``2..6``
    hidden, ``7..12`` output) with ``weight=w`` on every edge, ``bias=b_l``/
    ``response=r_l`` per row and one activation per row::

        v2 = f2(b2 + r2 * (w*x0 + w*x1))
        v3 = f3(b3 + r3 * w*v2)
        ...
        v6 = f6(b6 + r6 * w*v5)
        out_j = v_{2+j}                    # probe rows 7..11: weight 1, (0, 1)
        out_5 = b12 + r12 * w*v6           # row 12: the chain's real output

    Rows ``7..11`` are output rows fed by an edge of weight ``1`` from hidden row
    ``2+j``, with the neutral ``(bias=0, response=1)`` attribute row; because
    output rows are read out unactivated, ``out_j`` *is* ``v_{2+j}``. That is how
    every intermediate is asserted, not only the final output, so a schedule that
    evaluated the levels in the wrong order (or summed a node's *outgoing* edges)
    cannot pass by coincidence. Two activation options are configured
    (``sigmoid`` at position ``0``, ``tanh`` at ``1``), and the hidden rows
    alternate between them.

    ``support.oracle.forward`` is cross-checked on the same schedule as a second,
    independent rollout of the documented algorithm.
    """
    #: ``(bias, response, activation position)`` of rows ``2..6``.
    hidden = (
        (0.25, 1.0, 0.0),
        (-0.4, 0.8, 1.0),
        (0.1, -0.6, 0.0),
        (0.7, 1.3, 1.0),
        (-0.2, 0.5, 0.0),
    )
    final_row = (-0.3, 1.4)
    weight = 0.5
    max_nodes = 13
    num_outputs = 6

    payload = payloads.new_payload(1, max_nodes, 16)
    payloads.set_node(payload, 0, 0, 0, _node_row())
    payloads.set_node(payload, 0, 1, 1, _node_row())
    for offset, (bias, response, activation) in enumerate(hidden):
        row = 2 + offset
        payloads.set_node(
            payload, 0, row, row, _node_row(bias, response, 0.0, activation)
        )
    for row in range(7, 12):  # probe read-outs of rows 2..6
        payloads.set_node(payload, 0, row, row, _node_row())
    payloads.set_node(
        payload, 0, 12, 12, _node_row(final_row[0], final_row[1], 0.0, 0.0)
    )
    # The chain itself, then one probe edge per hidden row, then the real read-out.
    edges = [(0, 2), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6)]
    for row, (source, target) in enumerate(edges):
        payloads.set_conn(payload, 0, row, source, target, (weight,))
    for offset, row in enumerate(range(2, 7)):
        payloads.set_conn(payload, 0, len(edges) + offset, row, 7 + offset, (1.0,))
    payloads.set_conn(payload, 0, len(edges) + 5, 6, 12, (weight,))

    genome_object = _genome(
        genome,
        num_outputs=num_outputs,
        max_nodes=max_nodes,
        node_gene=genome.DefaultNode(
            activation_options=[common.ACT.sigmoid, common.ACT.tanh]
        ),
    )
    transformed, outputs = _evaluate(
        run_transform, run_forward, payloads, genome_object, payload, INPUTS
    )

    # The hand-written composition, row by row, unactivated at the read-out.
    functions = (_sigmoid, np.tanh)
    x = np.asarray(INPUTS[0], dtype=np.float64)
    bias, response, activation = hidden[0]
    z = bias + response * (weight * x[0] + weight * x[1])
    values = [functions[int(activation)](z)]
    for bias, response, activation in hidden[1:]:
        z = bias + response * weight * values[-1]
        values.append(functions[int(activation)](z))
    expected = np.array(values + [final_row[0] + final_row[1] * weight * values[-1]])

    assert outputs.shape == (1, num_outputs)
    np.testing.assert_allclose(outputs[0], expected, rtol=1e-5, atol=1e-6)

    reference = oracle.forward(
        payload,
        transformed,
        INPUTS,
        activation=[oracle.act_sigmoid, oracle.act_tanh],
    )
    np.testing.assert_allclose(outputs, reference, rtol=1e-5, atol=1e-6)


def test_output_rows_are_not_activated_by_the_node_gene(
    payloads, genome, oracle, run_transform, run_forward
):
    """Invariant: the read-out is ``bias + response * agg`` of the output row, unactivated.

    Oracle: closed form. With ``sigmoid`` configured as the single activation
    option and the output row's attribute row set to ``(bias=0, response=1)``, a
    single input edge of weight ``1`` carrying value ``z`` must yield ``z`` — not
    ``sigmoid(z)``. The hidden row of the same payload is activated, which the
    same test checks, so the difference is provably the output-row rule and not a
    global "no activation" bug.
    """
    payload = payloads.new_payload(1, 4, 8)
    payloads.set_node(payload, 0, 0, 0, _node_row())
    payloads.set_node(payload, 0, 1, 1, _node_row())
    # ``activation=0`` on *both* rows: the default gene maps position 0 to sigmoid.
    payloads.set_node(payload, 0, 2, 2, _node_row(0.0, 1.0, 0.0, 0.0))
    payloads.set_node(payload, 0, 3, 3, _node_row(0.0, 1.0, 0.0, 0.0))
    payloads.set_conn(payload, 0, 0, 0, 2, (1.0,))
    payloads.set_conn(payload, 0, 1, 1, 2, (1.0,))
    payloads.set_conn(payload, 0, 2, 2, 3, (1.0,))

    genome_object = _genome(genome, max_nodes=4, max_conns=8)
    transformed, outputs = _evaluate(
        run_transform, run_forward, payloads, genome_object, payload, INPUTS
    )

    z = float(INPUTS[0, 0] + INPUTS[0, 1])
    activated = _sigmoid(z)
    # v2 is sigmoid(z) (the hidden row *is* activated) ...
    np.testing.assert_allclose(outputs, [[activated]], rtol=1e-6, atol=1e-7)
    # ... and the output row adds nothing on top of it (its own sigmoid is ignored).
    assert not np.allclose(outputs, [[_sigmoid(activated)]], rtol=1e-4, atol=1e-4), (
        "the read-out was activated by the node gene: got the sigmoid of the "
        "hidden value instead of the hidden value itself"
    )
    assert not np.allclose(outputs, [[z]], rtol=1e-4, atol=1e-4), (
        "the read-out is the raw aggregation: the hidden row was not activated"
    )
    np.testing.assert_allclose(
        outputs, oracle.forward(payload, transformed, INPUTS), rtol=1e-6, atol=1e-7
    )


def test_activation_index_minus_one_is_identity(
    payloads, genome, oracle, run_transform, run_forward
):
    """Invariant: activation id ``-1`` leaves ``bias + response * agg`` untouched.

    Oracle: closed form. Two payloads that differ only in the activation column of
    one hidden row (``-1`` vs a configured option) must differ exactly by that
    option's map, and the ``-1`` row's value must equal ``bias + response * agg``.
    The legacy library clamped ``-1`` to the last configured option, so this test
    is also the guard against reintroducing that behaviour: the genome below
    configures a *single* activation option (``sigmoid`` at position ``0``), which
    is precisely the table a clamping ``lax.switch(-1, ...)`` would have hit.

    One edge per row (``0 -> 2 -> out``) keeps the aggregation trivial:
    ``agg = x0``, so ``bias + response * x0`` is the whole expression.
    """
    bias, response = 0.5, 0.8
    accepted = float(bias + response * float(INPUTS[0, 0]))

    def evaluated(activation_id):
        payload = payloads.new_payload(1, 4, 8)
        payloads.set_node(payload, 0, 0, 0, _node_row())
        payloads.set_node(payload, 0, 1, 1, _node_row())
        payloads.set_node(payload, 0, 2, 2, _node_row(bias, response, 0.0, activation_id))
        payloads.set_node(payload, 0, 3, 3, _node_row())
        payloads.set_conn(payload, 0, 0, 0, 2, (1.0,))
        payloads.set_conn(payload, 0, 1, 2, 3, (1.0,))
        genome_object = _genome(genome, max_nodes=4, max_conns=8)
        transformed, outputs = _evaluate(
            run_transform, run_forward, payloads, genome_object, payload, INPUTS
        )
        return payload, transformed, outputs

    _, _, identity = evaluated(-1.0)
    np.testing.assert_allclose(identity, [[accepted]], rtol=1e-6, atol=1e-7)

    payload, transformed, configured = evaluated(0.0)
    np.testing.assert_allclose(
        configured, [[_sigmoid(accepted)]], rtol=1e-6, atol=1e-7
    )
    assert not np.allclose(configured, identity, rtol=1e-4, atol=1e-4), (
        "activation -1 produced the configured option's value; the legacy "
        "lax.switch(-1, ...) clamp is back"
    )
    # The oracle dispatches the same way: ``-1`` -> identity, ``0`` -> sigmoid.
    np.testing.assert_allclose(
        configured,
        oracle.forward(payload, transformed, INPUTS, activation=[oracle.act_sigmoid]),
        rtol=1e-6,
        atol=1e-7,
    )


# --------------------------------------------------------------------------- #
# transforms
# --------------------------------------------------------------------------- #


def _affine(x):
    """The single legal map of the two transform tests: ``x -> 2x + 1``.

    Used on both sides of the comparison on purpose: as the genome's
    ``input_transform``/``output_transform`` it is called on a traced tensor
    (``*``/``+`` promote to the graph), and on the host it is the plain numpy map
    the no-transform genome is fed in :func:`test_input_transform_is_applied_before_seeding`.
    """
    return x * 2.0 + 1.0


def _transformed_payload(payloads):
    """The 4-row ``in -> hidden -> out`` payload shared by the two transform tests.

    Row ``2`` is a ``(bias=0, response=1)`` sigmoid node, so the read-out is
    ``sigmoid(x0 + x1)`` — a value far enough from ``1`` for an affine map of the
    read-out to be distinguishable from a map of the internals.

    Args:
        payloads: The ``support.payloads`` module (the module-scoped fixture).

    Returns:
        The filled payload.
    """
    payload = payloads.new_payload(1, 4, 8)
    payloads.set_node(payload, 0, 0, 0, _node_row())
    payloads.set_node(payload, 0, 1, 1, _node_row())
    payloads.set_node(payload, 0, 2, 2, _node_row())
    payloads.set_node(payload, 0, 3, 3, _node_row())
    payloads.set_conn(payload, 0, 0, 0, 2, (1.0,))
    payloads.set_conn(payload, 0, 1, 1, 2, (1.0,))
    payloads.set_conn(payload, 0, 2, 2, 3, (1.0,))
    return payload


def test_output_transform_is_applied_after_the_readout(
    payloads, genome, run_transform, run_forward
):
    """Invariant: ``output_transform`` maps the ``(P, num_outputs)`` read-out once.

    Oracle: composition — the output of a genome with ``output_transform=f`` must
    equal ``f`` applied elementwise to the output of the same genome with no
    transform, on the same payload and inputs. ``f`` is :func:`_affine` (``x * 2 +
    1``): affine and not the identity, while the read-out sits strictly inside
    ``(0, 1)``, so a transform applied to ``values`` instead of to the read-out
    would show: ``sigmoid(2 * (x0 + x1) + 1)`` is nowhere near ``2 *
    sigmoid(x0 + x1) + 1``.
    """
    payload = _transformed_payload(payloads)

    plain = _genome(genome, max_nodes=4, max_conns=8)
    _, untransformed = _evaluate(
        run_transform, run_forward, payloads, plain, payload, INPUTS
    )

    shifted = _genome(
        genome, max_nodes=4, max_conns=8, output_transform=_affine
    )
    _, transformed = _evaluate(
        run_transform, run_forward, payloads, shifted, payload, INPUTS
    )

    # The untransformed run is the network's own read-out, not something mapped.
    np.testing.assert_allclose(
        untransformed, [[_sigmoid(float(INPUTS[0, 0] + INPUTS[0, 1]))]], rtol=1e-6
    )
    np.testing.assert_allclose(transformed, untransformed * 2.0 + 1.0, rtol=1e-6)
    # Applying the map to the *values* would have produced this instead.
    assert not np.allclose(
        transformed, [[_sigmoid(2.0 * float(INPUTS[0, 0] + INPUTS[0, 1]) + 1.0)]],
        rtol=1e-4,
        atol=1e-4,
    ), "output_transform was applied to the node values instead of the read-out"


def test_input_transform_is_applied_before_seeding(
    payloads, genome, oracle, run_transform, run_forward
):
    """Invariant: ``input_transform`` maps ``inputs`` before they reach the input rows.

    Oracle: composition. A genome with ``input_transform=g`` must equal the same
    genome without a transform fed ``g(inputs)`` — computed on the host, on the
    same payload. ``RecurrentGenome`` deliberately does *not* apply it (see
    ``./test_recurrent.py``), so this test also pins the difference between the
    two forward passes.

    ``support.oracle.forward`` documents the same split: it expects ``inputs``
    already carrying the transform, which is why the reference rollout is fed
    ``g(inputs)`` too.
    """
    payload = _transformed_payload(payloads)

    plain = _genome(genome, max_nodes=4, max_conns=8)
    transformed_plain, untransformed = _evaluate(
        run_transform, run_forward, payloads, plain, payload, INPUTS
    )
    shifted_inputs = _affine(INPUTS)

    transformed_genome = _genome(
        genome, max_nodes=4, max_conns=8, input_transform=_affine
    )
    schedule, transformed = _evaluate(
        run_transform, run_forward, payloads, transformed_genome, payload, INPUTS
    )

    # g(inputs) into the untransformed genome, on the same network.
    _, reference = _evaluate(
        run_transform, run_forward, payloads, plain, payload, shifted_inputs
    )
    np.testing.assert_allclose(transformed, reference, rtol=1e-6)
    # And the transform really moved the answer (raw inputs give a different one).
    assert not np.allclose(transformed, untransformed, rtol=1e-4, atol=1e-4), (
        "input_transform had no effect: the input rows received the raw inputs"
    )
    # The oracle expects inputs that already carry the transform.
    np.testing.assert_allclose(
        transformed,
        oracle.forward(payload, schedule, shifted_inputs),
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        untransformed,
        oracle.forward(payload, transformed_plain, INPUTS),
        rtol=1e-6,
    )


# --------------------------------------------------------------------------- #
# topologies: the skipped node and the unconnected node
# --------------------------------------------------------------------------- #


def test_skipped_node_never_contributes_to_the_output(
    payloads, genome, oracle, run_transform, run_forward
):
    """Invariant: a node its consumer bypasses cannot change the output, however it is poisoned.

    Oracle: ``support.oracle.forward`` plus a perturbation argument.
    ``payloads.skip_payload`` wires ``0 -> 2``, ``1 -> 2`` and ``0 -> 3``,
    ``3 -> 7``: row ``2`` receives the signal but reaches no output. Setting row
    ``2``'s attribute row to ``NaN``, to a huge bias, or deleting its incoming
    edges must leave the read-out bit-identical — the schedule's level for row
    ``3`` is ``1``, not ``2``, which is exactly the property a level-synchronous
    pass has to get right.

    The ``NaN`` perturbation keeps the *option-id* columns finite (``0``) and
    poisons only the value columns: a live row's ``NaN`` is deliberately left
    alone by ``_padding_safe_attrs``, and a ``NaN`` option id would additionally
    emit a ``NaN``-to-``int32`` cast warning for a row that never matters.
    """
    shape = dict(max_nodes=8, max_conns=16)

    def read_out(payload):
        genome_object = _genome(genome, **shape)
        schedule, outputs = _evaluate(
            run_transform, run_forward, payloads, genome_object, payload, INPUTS
        )
        return schedule, outputs

    base = payloads.skip_payload(**shape)
    schedule, outputs = read_out(base)
    np.testing.assert_allclose(
        outputs, oracle.forward(base, schedule, INPUTS), rtol=1e-6, atol=1e-7
    )

    poisoned_values = payloads.skip_payload(**shape)
    payloads.set_node(poisoned_values, 0, 2, 2, _node_row(np.nan, np.nan, 0.0, 0.0))
    huge_bias = payloads.skip_payload(**shape)
    payloads.set_node(huge_bias, 0, 2, 2, _node_row(1e9, 1e9, 0.0, 0.0))
    no_incoming = payloads.skip_payload(**shape)
    payloads.set_conn(no_incoming, 0, 0, payloads.I_INF, payloads.I_INF)
    payloads.set_conn(no_incoming, 0, 1, payloads.I_INF, payloads.I_INF)

    for label, perturbed in (
        ("NaN value columns", poisoned_values),
        ("huge bias/response", huge_bias),
        ("incoming edges deleted", no_incoming),
    ):
        perturbed_schedule, perturbed_outputs = read_out(perturbed)
        assert np.array_equal(perturbed_outputs, outputs), (
            f"perturbing the bypassed row ({label}) changed the read-out: "
            f"{perturbed_outputs} != {outputs}"
        )
        # The perturbations do move the *schedule* of the dead row, so the
        # invariance above is about the read-out, not about a no-op edit.
        if label == "incoming edges deleted":
            assert not np.array_equal(
                perturbed_schedule.node_level, schedule.node_level
            ), "the deleted-edge payload derived an identical schedule; the test is vacuous"


def test_unconnected_node_and_its_descendants_stay_nan(
    payloads, genome, oracle, run_transform, run_forward
):
    """Invariant: a present node with no incoming edge keeps ``NaN``, and so does its subtree.

    Oracle: ``support.oracle.forward`` and the documented poisoning rule — a node
    with no live incoming slot never enters an ``update_mask``, so its value stays
    the ``NaN`` seed; a consumer of it then has no *live* source, keeps its own
    previous value, and the chain of ``NaN`` reaches the output if the unconnected
    node feeds it.

    ``payloads.isolated_payload`` gives the first half (row ``2`` is present,
    isolated, and does not reach the output, so the dense input-to-output layer
    still reads out); the second half wires ``2 -> 3 -> out`` and checks that the
    ``NaN`` crosses both hops, with a finite attribute row *and* with a huge
    ``bias``/``response`` on the isolated row (attributes cannot rescue a node
    that never runs).
    """
    shape = dict(max_nodes=8, max_conns=16)

    def read_out(payload):
        genome_object = _genome(genome, **shape)
        schedule, outputs = _evaluate(
            run_transform, run_forward, payloads, genome_object, payload, INPUTS
        )
        return schedule, outputs

    isolated = payloads.isolated_payload(**shape)
    schedule, outputs = read_out(isolated)
    assert isolated.node_keys[0, 2] != payloads.I_INF, "row 2 must be a present node"
    np.testing.assert_allclose(
        outputs, [[float(INPUTS[0, 0] + INPUTS[0, 1])]], rtol=1e-6, atol=1e-7
    )
    np.testing.assert_allclose(
        outputs, oracle.forward(isolated, schedule, INPUTS), rtol=1e-6, atol=1e-7
    )

    def poisoned(hidden_row):
        """Rows ``0, 1`` in, row ``2`` isolated, row ``3`` its consumer, ``7`` the read-out."""
        payload = payloads.new_payload(1, 8, 16)
        payloads.set_node(payload, 0, 0, 0, _node_row())
        payloads.set_node(payload, 0, 1, 1, _node_row())
        payloads.set_node(payload, 0, 2, 2, hidden_row)
        payloads.set_node(payload, 0, 3, 3, _node_row(0.5, 1.0, 0.0, 0.0))
        payloads.set_node(payload, 0, 7, 7, _node_row())
        payloads.set_conn(payload, 0, 0, 2, 3, (1.0,))
        payloads.set_conn(payload, 0, 1, 3, 7, (1.0,))
        return payload

    for label, hidden_row in (
        ("finite attributes", _node_row(0.5, 1.0, 0.0, 0.0)),
        ("huge attributes", _node_row(1e9, 1e9, 0.0, 0.0)),
    ):
        payload = poisoned(hidden_row)
        schedule, outputs = read_out(payload)
        assert np.isnan(outputs).all(), (
            f"an unconnected node with {label} produced a finite read-out {outputs}; "
            "the NaN poisoning did not reach the output"
        )
        reference = oracle.forward(payload, schedule, INPUTS)
        assert np.array_equal(np.isnan(outputs), np.isnan(reference)), (
            "the kernel and the oracle disagree on which outputs are NaN"
        )
        assert np.isnan(reference).all()


def test_padded_rows_never_leak_into_the_output(
    payloads, genome, oracle, run_transform, run_forward
):
    """Invariant: rows outside the real network (``I_INF`` key, ``NaN`` attributes) are inert.

    Oracle: ``support.oracle.forward`` comparison against the *same topology with a
    smaller ``max_nodes``*. Padding must not change a single output value, so the
    padded payload's read-out equals the compact one's; a kernel that let an
    ``NaN`` attribute row into an aggregation, or that treated ``I_INF`` as a real
    node key, fails here while passing every dense-payload test.

    Each capacity gets its own genome: the schedule's ``output_slot`` is derived
    from the genome's own output *keys* (the last ``num_outputs`` rows of *its*
    ``max_nodes``), so a genome of capacity ``8`` cannot read out a payload of
    capacity ``4``. The two networks are isomorphic — ``0, 1 -> 2 -> out`` with
    the same attribute rows and weights — so their read-outs must agree exactly.
    """

    def build(max_nodes):
        """The compact topology, replicated into ``max_nodes`` rows."""
        output_row = max_nodes - 1
        payload = payloads.new_payload(1, max_nodes, 16)
        payloads.set_node(payload, 0, 0, 0, _node_row(0.1, 1.0, 0.0, 0.0))
        payloads.set_node(payload, 0, 1, 1, _node_row(0.2, 1.0, 0.0, 0.0))
        payloads.set_node(payload, 0, 2, 2, _node_row(0.7, 1.2, 0.0, 0.0))
        payloads.set_node(payload, 0, output_row, output_row, _node_row(-0.1, 1.5))
        payloads.set_conn(payload, 0, 0, 0, 2, (0.5,))
        payloads.set_conn(payload, 0, 1, 1, 2, (0.25,))
        payloads.set_conn(payload, 0, 2, 2, output_row, (1.7,))
        return payload

    padded = build(8)
    compact = build(4)
    # The padding really is padding: rows 3..6 are absent in the padded payload.
    assert (padded.node_keys[0, 3:7] == payloads.I_INF).all()
    assert np.isnan(padded.node_attrs[0, 3:7]).all()

    def read_out(payload):
        genome_object = _genome(genome, max_nodes=payload.max_nodes, max_conns=16)
        schedule, outputs = _evaluate(
            run_transform, run_forward, payloads, genome_object, payload, INPUTS
        )
        return schedule, outputs

    padded_schedule, padded_outputs = read_out(padded)
    compact_schedule, compact_outputs = read_out(compact)

    assert np.array_equal(padded_outputs, compact_outputs), (
        "padded rows changed the read-out: "
        f"{padded_outputs} != {compact_outputs}"
    )
    np.testing.assert_allclose(
        padded_outputs,
        oracle.forward(padded, padded_schedule, INPUTS),
        rtol=1e-6,
        atol=1e-7,
    )
    np.testing.assert_allclose(
        compact_outputs,
        oracle.forward(compact, compact_schedule, INPUTS),
        rtol=1e-6,
        atol=1e-7,
    )


# --------------------------------------------------------------------------- #
# batch semantics
# --------------------------------------------------------------------------- #


def test_individuals_are_evaluated_independently_inside_one_batch(
    payloads, genome, oracle, run_transform, run_forward
):
    """Invariant: population member ``p``'s output depends only on member ``p``'s payload and input.

    Oracle: one ``P == 4`` run against four ``P == 1`` runs of the same
    individuals (each with its own derived schedule). The population axis must be
    a pure batch axis: no reduce may span it, no per-individual value may come
    from a neighbour's row. Differing weights *and* differing topologies across
    the four members make an accidental cross-row gather visible — member ``2``
    has lost its ``0 -> 2`` edge, member ``3`` its ``6 -> out`` edge, so the four
    read-outs are three different finite numbers plus one ``NaN``.
    """
    edges = [(0, 2), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7)]
    weights = (1.0, 0.5, 2.0, 2.0)
    batch = payloads.chain_payload(pop_size=4, max_nodes=8, max_conns=16)
    for member, weight in enumerate(weights):
        for row, (source, target) in enumerate(edges):
            payloads.set_conn(batch, member, row, source, target, (weight,))
    # Two structural edits: a topology the neighbours do not have.
    payloads.set_conn(batch, 2, 0, payloads.I_INF, payloads.I_INF)
    payloads.set_conn(batch, 3, 6, payloads.I_INF, payloads.I_INF)

    genome_object = _genome(genome)
    tiled = np.tile(INPUTS, (4, 1))
    schedule, outputs = _evaluate(
        run_transform, run_forward, payloads, genome_object, batch, tiled
    )
    assert outputs.shape == (4, 1)
    reference = oracle.forward(batch, schedule, tiled)
    assert np.array_equal(np.isnan(outputs), np.isnan(reference))
    np.testing.assert_allclose(outputs, reference, rtol=1e-6, atol=1e-7)

    singles = []
    for member in range(4):
        individual = batch.of(member)
        _, single = _evaluate(
            run_transform, run_forward, payloads, genome_object, individual, INPUTS
        )
        singles.append(single)
    for member, single in enumerate(singles):
        assert np.array_equal(outputs[member : member + 1], single, equal_nan=True), (
            f"member {member} of the batch ({outputs[member]}) differs from its own "
            f"single-individual run ({single[0]})"
        )

    finite = outputs[np.isfinite(outputs)]
    assert np.isnan(outputs[3]).all(), "member 3 lost its read-out edge: expected NaN"
    assert len(np.unique(finite)) == 3, (
        "the four members must produce three distinct finite read-outs, got "
        f"{finite.ravel()}"
    )


def test_random_topologies_match_the_numpy_oracle(
    payloads, genome, oracle, run_transform, run_forward, rng
):
    """Invariant: on random feedforward DAGs the kernel equals ``support.oracle.forward``.

    Oracle: the independent numpy rollout. Topologies are drawn on the host with
    the ``rng`` fixture — a random acyclic edge set (edges only ever point at a
    higher row, so acyclicity is structural), random weights, random
    ``bias``/``response``, one activation option, well inside ``max_nodes``/
    ``max_conns``/``max_in_degree`` — so nothing about the graph is shared with
    the kernel. Both the values (``allclose``) and the ``NaN`` pattern
    (``np.isnan`` equality) are compared.

    Member ``0`` deliberately receives no edge into its output row, and the test
    asserts that the batch contains both a finite and a ``NaN`` read-out: a
    rollout that agreed only on the finite members would not prove the poisoning
    rule. ``edge_overflow`` is asserted empty so a dropped edge cannot be the
    reason for a disagreement.
    """
    max_nodes, max_conns, num_inputs, num_outputs, max_in_degree = 8, 16, 2, 1, 4
    pop_size = 6
    output_row = max_nodes - 1

    payload = payloads.new_payload(pop_size, max_nodes, max_conns)
    for member in range(pop_size):
        hidden_count = int(rng.integers(1, 4))
        hidden = list(range(num_inputs, num_inputs + hidden_count))
        for row in range(num_inputs):  # inputs: never computed, neutral rows
            payloads.set_node(payload, member, row, row, _node_row())
        for row in hidden + [output_row]:
            payloads.set_node(
                payload,
                member,
                row,
                row,
                _node_row(
                    rng.normal(0.0, 1.0),
                    rng.uniform(0.5, 1.5),
                    0.0,  # the single configured aggregation (sum)
                    0.0,  # the single configured activation (sigmoid)
                ),
            )
        in_degree = {row: 0 for row in hidden + [output_row]}
        candidates = [
            (source, target)
            for source in list(range(num_inputs)) + hidden
            for target in hidden + [output_row]
            if source < target
        ]
        rng.shuffle(candidates)
        connection_row = 0
        for source, target in candidates:
            if connection_row >= max_conns:
                break
            if member == 0 and target == output_row:
                continue  # member 0 must keep its NaN read-out
            if in_degree[target] >= max_in_degree:
                continue
            if rng.random() < 0.45:
                payloads.set_conn(
                    payload,
                    member,
                    connection_row,
                    source,
                    target,
                    (rng.uniform(-2.0, 2.0),),
                )
                connection_row += 1
                in_degree[target] += 1

    inputs = rng.normal(0.0, 1.0, size=(pop_size, num_inputs)).astype(np.float32)
    genome_object = _genome(
        genome,
        num_inputs=num_inputs,
        num_outputs=num_outputs,
        max_nodes=max_nodes,
        max_conns=max_conns,
        max_in_degree=max_in_degree,
    )
    schedule, outputs = _evaluate(
        run_transform, run_forward, payloads, genome_object, payload, inputs
    )
    assert not np.asarray(schedule.edge_overflow).any(), (
        "the drawn topologies overflow max_in_degree; the comparison would be vacuous"
    )

    reference = oracle.forward(payload, schedule, inputs)
    assert np.array_equal(np.isnan(outputs), np.isnan(reference)), (
        "kernel and oracle disagree on the NaN pattern: "
        f"{np.isnan(outputs).ravel()} vs {np.isnan(reference).ravel()}"
    )
    np.testing.assert_allclose(outputs, reference, rtol=1e-5, atol=1e-6)
    assert np.isnan(outputs).any() and np.isfinite(outputs).any(), (
        "the drawn batch must exercise both a poisoned and a computed read-out, got "
        f"{outputs.ravel()}"
    )


# --------------------------------------------------------------------------- #
# the static depth bound
# --------------------------------------------------------------------------- #


def test_a_network_deeper_than_max_depth_leaves_its_tail_nan(
    payloads, genome, oracle, run_transform, run_forward
):
    """Invariant: nodes below level ``max_depth`` are never updated and stay ``NaN``.

    Oracle: the documented static-unroll limit — ``DefaultGenome`` unrolls exactly
    ``max_depth`` levels, so a chain longer than that keeps its unreachable tail
    ``NaN`` and the read-out is ``NaN`` when the output row is in the tail. This is
    a *documented* limit (``max_depth`` is a build-time constant that a rebuild
    must raise), so the test asserts the limit instead of avoiding it, and asserts
    that the prefix of levels below the bound is still correct.

    The chain of 12 rows ``0, 1 -> 2 -> 3 -> 4 -> 5 -> 6`` puts the hidden rows at
    levels ``1..5`` and probes each of them into its own output row ``7..11`` (edge
    weight ``1``, neutral attribute row), so the five read-out columns are the
    activated values of rows ``2..6``. With ``max_depth=2`` only column ``0`` is
    computed; the deep genome evaluates all five, and column ``0`` must agree
    exactly between the two builds.
    """
    max_nodes, num_outputs, max_conns = 12, 5, 16
    payload = payloads.new_payload(1, max_nodes, max_conns)
    payloads.set_node(payload, 0, 0, 0, _node_row())
    payloads.set_node(payload, 0, 1, 1, _node_row())
    for row in range(2, 7):
        payloads.set_node(payload, 0, row, row, _node_row(0.1 * row, 1.0, 0.0, 0.0))
    for row in range(7, 12):
        payloads.set_node(payload, 0, row, row, _node_row())
    edges = [(0, 2), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6)]
    for row, (source, target) in enumerate(edges):
        payloads.set_conn(payload, 0, row, source, target, (1.0,))
    for offset, row in enumerate(range(2, 7)):  # one probe per hidden row
        payloads.set_conn(payload, 0, len(edges) + offset, row, 7 + offset, (1.0,))

    def read_out(max_depth):
        genome_object = _genome(
            genome,
            num_outputs=num_outputs,
            max_nodes=max_nodes,
            max_conns=max_conns,
            max_depth=max_depth,
        )
        return _evaluate(
            run_transform, run_forward, payloads, genome_object, payload, INPUTS
        )

    shallow_schedule, shallow = read_out(2)
    deep_schedule, deep = read_out(16)

    assert shallow.shape == (1, num_outputs)
    assert np.isfinite(deep).all(), (
        f"the deep genome must compute the whole chain, got {deep.ravel()}"
    )
    # The reachable prefix (levels 1 and 2) is identical; the tail never ran.
    assert np.isfinite(shallow[0, 0]), "the level-2 probe must be computed"
    assert np.array_equal(shallow[0, :1], deep[0, :1]), (
        "the computed prefix differs between the shallow and the deep build: "
        f"{shallow[0, :1]} != {deep[0, :1]}"
    )
    assert np.isnan(shallow[0, 1:]).all(), (
        "rows below the level bound were updated: "
        f"{shallow[0, 1:]} should be NaN (levels 3..6 exceed max_depth=2)"
    )
    # The shallow schedule is the same schedule; only the unroll bound differs.
    assert np.array_equal(shallow_schedule.node_level, deep_schedule.node_level)
    np.testing.assert_allclose(
        deep, oracle.forward(payload, deep_schedule, INPUTS), rtol=1e-6, atol=1e-7
    )
