"""Unit tests for ``BaseGenome.grad`` — the differentiated forward pass.

What this module proves
-----------------------
``grad`` is a **graph factory**, not a kernel: it traces one forward pass, takes
reverse-mode derivatives with respect to the float attribute tensors, and returns
the traced ``evoxc.Graph`` for the pipeline to cache per shape signature. The
tests therefore check three separate things:

* the mathematics — the returned gradients agree with central finite differences
  of the same loss through the plain ``BaseGenome.forward``, on a genome small
  enough to perturb one attribute entry at a time;
* the evoxc rules the factory has to respect — the loss is scalarized as
  ``sum(loss_fn(outputs))`` (which must not change a per-individual gradient),
  only ``node_attrs``/``conn_attrs`` are differentiated (they are the graph's
  only float inputs, and the gradient pair mirrors them), and no ``evoxc.random``
  op may appear inside the differentiated graph;
* the NaN contract — padding never carries ``NaN`` into a gradient, because a
  zeroed cotangent times a ``NaN`` local derivative is still ``NaN``, which would
  poison the whole attribute row of that individual.

Oracles used here (never the gradient path itself)
--------------------------------------------------
* **central finite differences through the plain forward**: ``(L(+h) - L(-h)) / 2h``
  on one payload entry, with ``L`` recomputed by ``genome.forward``.  ``grad``
  differentiates the dense-weight twin of that pass (``genome/grad.py``), which
  reaches the same values through *different ops* (a per-individual weight matrix
  and a batched ``dot`` instead of the incoming-window gather), so the difference
  quotient and the analytic gradient are two independent computations of the same
  algebra.  ``h = 1e-2`` with an absolute tolerance of ``1e-3`` leaves float32
  headroom: the measured agreement on the genomes below is <= 5e-5, so the
  tolerance absorbs finite-difference truncation only, never a wrong sign or a
  wrong factor;
* **closed form** for the bias-only network (all connection weights zero), where
  the read-out is literally ``bias + response * 0`` and both partial derivatives
  can be written down;
* ``sum`` linearity for the scalarization check, and shape/dtype equality for the
  structural check.

Gotchas this module encodes
---------------------------
* ``grad`` takes an **already-derived** ``Transformed``: ``argsort`` has no AD
  rule, so the schedule must stay outside the differentiated graph. That is why
  every test derives the schedule first (``run_transform``) and hands it in.
* The scalarization is ``sum(loss_fn(outputs))`` over the whole ``(P, ...)`` loss,
  not a mean: the population axis is independent of the sum, so per-individual
  gradients are unchanged — but a test that compared *absolute* loss values
  instead of gradients would see the factor ``P``.
* A node whose value is ``NaN`` (no live incoming edge) contributes a cotangent of
  exactly ``0``, not ``NaN``: the sanitization happens on the attribute tensors at
  the graph input (``select(exists & ~isnan(a), a, 0)``), which keeps every
  differentiable op NaN-free, so the reverse sweep never multiplies by a ``NaN``
  local derivative.
* Gradients are taken with respect to the *float* attribute tensors only; the
  keys and the schedule are non-differentiated ``int32``/``bool`` inputs of the
  same graph. A gradient tuple of length two (node attributes, connection
  attributes) is therefore the contract, not an implementation detail.
* The loss is written as ``residual * residual`` rather than ``residual ** 2``:
  evoxc's ``power`` pullback emits ``log(base)`` unconditionally, so a ``** 2``
  loss would evaluate ``log`` of a negative residual and warn (that term belongs
  to the constant exponent and is discarded, so it is noise only) — the suite
  stays warning-free by squaring with a ``multiply``.

Backend is ``"numpy"`` throughout, through the shared runner only; no
``jax``/``torch``, no network, no GPU.
"""

from __future__ import annotations

import numpy as np
import pytest


#: Static geometry of the hand-built networks below: two inputs, three hidden
#: rows and one output row of a six-row payload.
NUM_INPUTS = 2
NUM_OUTPUTS = 1
MAX_NODES = 6
MAX_CONNS = 8

#: The activation id meaning "identity" (never an option index), so the read-out
#: of the test networks is a plain affine composition.
IDENTITY = -1.0

#: Weights of the five chain edges (``0->2, 1->2, 2->3, 3->4, 4->5``): distinct
#: magnitudes and both signs, so no two partial derivatives can be confused.
CHAIN_WEIGHTS = (0.5, -0.75, 1.25, -1.5, 0.8)

#: ``row -> (bias, response)`` of the non-input rows: non-neutral, so a finite
#: difference on a hidden row sees a value that actually moves the read-out.
NODE_ATTRS = {2: (0.1, 1.0), 3: (-0.2, 0.9), 4: (0.15, 1.1), 5: (0.05, 1.0)}

#: Central-difference step and the absolute tolerance the comparison allows.
FD_STEP = 1e-2
FD_TOLERANCE = 1e-3

#: Target of the quadratic loss every test here differentiates.
TARGET = 0.5


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _loss(outputs):
    """Per-individual quadratic loss ``(outputs - TARGET) * (outputs - TARGET)``.

    Used both as the ``loss_fn`` handed to ``grad`` and as the scalar function of
    the finite-difference reference, so the two derivatives belong to the same
    objective. It works on the numpy read-out of ``forward`` and on the traced
    tensor ``grad`` passes in, and it squares with a ``multiply`` on purpose (see
    the module docstring).

    Args:
        outputs: ``(P, num_outputs)`` read-out, numpy or traced.

    Returns:
        ``(P, num_outputs)`` squared residual — the same shape, so ``grad``'s
        ``sum`` over it is the total objective and ``[:, 0]`` is one individual's.
    """
    residual = outputs - TARGET
    return residual * residual


def _chain(payloads, run_transform, pop_size=1):
    """The canonical live network of this module: a six-row chain, distinct weights.

    Edges ``0->2, 1->2, 2->3, 3->4, 4->5`` make every non-input row sit on its own
    level, so a finite difference on any attribute entry has a single,
    well-conditioned path to the read-out. Activations are identity, biases and
    responses are non-neutral, and the five weights differ in magnitude and sign,
    so no partial derivative is structurally zero except the ones a test names.

    Args:
        payloads: The ``support.payloads`` module.
        run_transform: The conftest fixture deriving the schedule.
        pop_size: Number of identical individuals; a test that needs distinct
            members edits ``payload`` afterwards (attribute edits do not change
            the schedule, which only reads the key columns).

    Returns:
        ``(genome_object, state, payload, transformed)`` — everything ``grad``,
        ``forward`` and the runner need. Every node row of this payload is
        present, so writing ``-1`` into the activation column breaks no padding
        invariant.
    """
    genome_object = payloads.default_genome(
        num_inputs=NUM_INPUTS,
        num_outputs=NUM_OUTPUTS,
        max_nodes=MAX_NODES,
        max_conns=MAX_CONNS,
        max_in_degree=4,
    )
    state = payloads.genome_state(genome_object)
    payload = payloads.chain_payload(
        pop_size=pop_size,
        max_nodes=MAX_NODES,
        max_conns=MAX_CONNS,
        num_inputs=NUM_INPUTS,
        num_outputs=NUM_OUTPUTS,
    )
    payload.node_attrs[:, :, 3] = IDENTITY
    for row, (bias, response) in NODE_ATTRS.items():
        payload.node_attrs[:, row, 0] = bias
        payload.node_attrs[:, row, 1] = response
    payload.conn_attrs[:, : len(CHAIN_WEIGHTS), 0] = CHAIN_WEIGHTS
    transformed = run_transform(genome_object, state, payload)
    return genome_object, state, payload, transformed


def _run_grad(runner, genome_object, state, payload, transformed, inputs):
    """Build the gradient graph of one call and run it, returning plain numpy.

    Returns:
        ``(node_grad, conn_grad)`` — ``(P, max_nodes, NA)`` and
        ``(P, max_conns, CA)`` float32 arrays, which is the documented output of
        ``BaseGenome.grad``.
    """
    graph = genome_object.grad(
        state, transformed, *payload.args(), inputs, _loss
    )
    node_grad, conn_grad = runner.run_graph(graph, *payload.args(), inputs)
    return node_grad, conn_grad


def _central_difference(
    payload,
    run_forward,
    genome_object,
    state,
    transformed,
    inputs,
    field,
    row,
    column,
    individual=slice(None),
    step=FD_STEP,
):
    """Central difference of the per-individual loss w.r.t. one payload attribute entry.

    The entry is perturbed in place by ``±step``, the loss is re-evaluated by the
    plain ``forward`` (never by the gradient path under test) and the entry is
    restored, so a caller can sweep every coordinate of one payload.

    Args:
        payload: The ``support.payloads.Payload`` to perturb; restored on exit.
        run_forward: The conftest fixture running ``genome_object.forward``.
        genome_object: The genome whose plain forward is the reference.
        state: Host configuration container of that genome.
        transformed: The already-derived schedule of ``payload``.
        inputs: ``(P, num_inputs)`` network inputs.
        field: ``"node_attrs"`` or ``"conn_attrs"`` — which tensor to perturb.
        row: Gene row of the entry.
        column: Attribute column of the entry.
        individual: Individual index, or ``slice(None)`` for the whole batch.
        step: Perturbation step, kept float32-sized.

    Returns:
        ``(P,)`` numpy array of ``(L(+step) - L(-step)) / (2 * step)``, one entry
        per individual (the loss is read per individual, so a batching mistake
        shows up as a wrong entry rather than as a silent sum).
    """
    tensor = getattr(payload, field)
    original = tensor[individual, row, column].copy()
    shifted = []
    try:
        for delta in (step, -step):
            tensor[individual, row, column] = original + np.float32(delta)
            outputs = run_forward(genome_object, state, transformed, payload, inputs)
            shifted.append(_loss(outputs)[:, 0])
    finally:
        tensor[individual, row, column] = original
    plus, minus = shifted
    return (plus - minus) / (2 * step)


def _compare_with_finite_differences(
    payloads, run_forward, genome_object, state, payload, transformed, inputs,
    node_grad, conn_grad,
):
    """Assert every present attribute entry equals its central difference, per member.

    Reference: the plain ``forward`` (never the gradient path), one entry at a time at
    ``rtol=0``/``atol=FD_TOLERANCE``; each differentiated entry must also be *not*
    approximately zero, so two zeros cannot pass, while an input row — seeded, not
    computed — must be exactly ``0`` on both sides. Returns the largest difference seen.
    """
    nodes = [r for r in range(payload.max_nodes) if payload.node_keys[0, r] != payloads.I_INF]
    conns = [r for r in range(payload.max_conns) if payload.conn_keys[0, r, 0] != payloads.I_INF]
    cases = [("conn_attrs", row, 0) for row in conns] + [
        ("node_attrs", row, column) for row in nodes for column in (0, 1)
    ]
    worst = 0.0
    for member in range(payload.pop_size):
        for field, row, column in cases:
            measured = float(
                (conn_grad if field == "conn_attrs" else node_grad)[member, row, column]
            )
            fd = float(
                _central_difference(
                    payload, run_forward, genome_object, state, transformed, inputs,
                    field, row, column, individual=member,
                )[member]
            )
            np.testing.assert_allclose(
                measured, fd, rtol=0.0, atol=FD_TOLERANCE,
                err_msg=f"individual {member}, {field} row {row}, column {column}",
            )
            if field == "node_attrs" and row < NUM_INPUTS:
                assert measured == 0.0 and fd == 0.0  # seeded, not computed
            else:
                assert abs(measured) > FD_TOLERANCE, f"{field} {row}/{column} is ~0"
                worst = max(worst, abs(measured - fd))
    return worst


def _expect_raises(error, fragment, call):
    """Assert ``call()`` raises ``error`` whose message contains ``fragment`` verbatim.

    The fragment is a distinctive part of the documented message, never the exception type
    alone, so a refusal that moved to another check fails here.
    """
    with pytest.raises(error) as info:
        call()
    assert fragment in str(info.value), f"{fragment!r} not in {info.value}"


# --------------------------------------------------------------------------- #
# the factory contract
# --------------------------------------------------------------------------- #


def test_grad_returns_a_graph_factory_not_a_value(
    payloads, runner, run_transform
):
    """Invariant: ``grad`` returns an ``evoxc.Graph`` and does not evaluate anything.

    Oracle: the structure of the returned object. It is an ``evoxc.Graph`` (named
    by its own type, since only ``support.runner`` may import ``evoxc``), its
    tensor inputs are exactly the five tensors of one call — ``node_keys``,
    ``node_attrs``, ``conn_keys``, ``conn_attrs`` and ``inputs``, in that order
    and with those dtypes — and it carries no static input leaves, because the
    schedule is baked into the graph. The loss handed in is *traced*, not called
    on numbers: the argument it sees is a symbolic tensor, so no gradient value
    exists until ``runner.run_graph`` executes the graph, and a kernel that
    returned an array instead would fail every assertion here.
    """
    genome_object, state, payload, transformed = _chain(payloads, run_transform)
    inputs = np.array([[0.4, -0.9]], dtype=np.float32)
    traced = []

    def loss(outputs):
        traced.append(outputs)
        return _loss(outputs)

    graph = genome_object.grad(
        state, transformed, *payload.args(), inputs, loss
    )

    # 1. a traced graph, not a numeric result of any kind.
    assert type(graph).__name__ == "Graph"
    assert type(graph).__module__.split(".")[0] == "evoxc"
    assert not isinstance(graph, (np.ndarray, tuple, list, dict))

    # 2. a function of the five tensors only: keys and inputs are non-differentiated
    #    inputs, the schedule is not an input at all.
    specs = graph.tensor_specs
    assert [tuple(int(size) for size in spec.shape) for spec in specs] == [
        payload.node_keys.shape,
        payload.node_attrs.shape,
        payload.conn_keys.shape,
        payload.conn_attrs.shape,
        inputs.shape,
    ]
    assert [str(spec.dtype) for spec in specs] == [
        "int32",
        "float32",
        "int32",
        "float32",
        "float32",
    ]
    assert graph.static_values == ()

    # 3. building it traced the loss instead of evaluating it: nothing numeric was
    #    lost or produced on the way.
    assert traced, "grad did not trace the loss"
    assert all(not isinstance(item, np.ndarray) for item in traced)

    # 4. running the graph is the only way to the two gradients.
    node_grad, conn_grad = runner.run_graph(graph, *payload.args(), inputs)
    assert node_grad.shape == payload.node_attrs.shape
    assert conn_grad.shape == payload.conn_attrs.shape


def test_grad_graph_contains_no_random_op_and_is_deterministic(
    payloads, runner, run_transform
):
    """Invariant: the differentiated graph is random-free, hence run-to-run identical.

    Oracle: two runs of the *same* compiled graph on the *same* inputs return
    bit-identical gradients and loss. The observable form of "no random op inside
    the differentiated graph" is exactly this: an ``evoxc.random`` draw inside a
    differentiated region is a hard ``evoxc.TransformError`` at lowering time, so a
    graph that lowers and then reproduces its own output cannot contain one. Any
    sampling the genome needs (initialization, mutation) belongs to another graph.
    A second, freshly built graph from the same factory is compared too, which
    pins that the build itself carries no hidden state.
    """
    genome_object, state, payload, transformed = _chain(payloads, run_transform)
    inputs = np.array([[0.4, -0.9]], dtype=np.float32)
    graph = genome_object.grad(
        state, transformed, *payload.args(), inputs, _loss
    )

    first = runner.run_graph(graph, *payload.args(), inputs)
    second = runner.run_graph(graph, *payload.args(), inputs)
    assert len(first) == 2
    for left, right in zip(first, second):
        assert np.array_equal(left, right)  # bit-identical, not merely allclose

    rebuilt = genome_object.grad(
        state, transformed, *payload.args(), inputs, _loss
    )
    assert rebuilt is not graph  # the factory call returns a fresh graph
    for left, right in zip(first, runner.run_graph(rebuilt, *payload.args(), inputs)):
        assert np.array_equal(left, right)


def test_gradient_shapes_and_dtypes_follow_the_differentiated_inputs(
    payloads, runner, run_transform
):
    """Invariant: the gradient pair has the shapes and dtypes of ``node_attrs``/``conn_attrs``.

    Oracle: structural equality with the differentiated inputs —
    ``(P, N, NA) float32`` and ``(P, E, CA) float32`` — and a tuple of exactly two
    arrays, because keys and the schedule stay non-differentiated ``int32``/``bool``
    graph inputs. A float64 or broadcast gradient silently changes what the
    algorithm's optimizer does, so dtype is part of the contract here.
    """
    genome_object, state, payload, transformed = _chain(payloads, run_transform, pop_size=2)
    inputs = np.array([[0.4, -0.9], [-0.3, 0.6]], dtype=np.float32)
    grads = _run_grad(runner, genome_object, state, payload, transformed, inputs)

    assert isinstance(grads, tuple)
    assert len(grads) == 2  # one per differentiated argument, never the keys
    node_grad, conn_grad = grads
    assert node_grad.shape == payload.node_attrs.shape == (2, MAX_NODES, 4)
    assert conn_grad.shape == payload.conn_attrs.shape == (2, MAX_CONNS, 1)
    assert node_grad.dtype == payload.node_attrs.dtype == np.float32
    assert conn_grad.dtype == payload.conn_attrs.dtype == np.float32


# --------------------------------------------------------------------------- #
# finite differences
# --------------------------------------------------------------------------- #


def test_conn_attr_gradient_matches_finite_differences(
    payloads, runner, run_transform, run_forward
):
    """Invariant: ``d loss / d conn_attrs`` equals central differences of the same loss.

    Oracle: the plain ``forward`` as a function of the weight column — perturb one
    weight by ``±h = ±1e-2``, recompute the loss through it, and compare
    ``(L(+h) - L(-h)) / 2h`` with the kernel's gradient entry under an absolute
    tolerance of ``1e-3``. The chain's five edges live on five levels, so each
    weight has one clean path to the read-out and every quotient is well
    conditioned; the data for the finite difference is *never* taken from the
    gradient path, so the two derivatives are independent computations. The
    measured maximum difference on this genome is ≈5e-5, i.e. float32 noise.
    """
    genome_object, state, payload, transformed = _chain(payloads, run_transform)
    inputs = np.array([[0.4, -0.9]], dtype=np.float32)
    _, conn_grad = _run_grad(
        runner, genome_object, state, payload, transformed, inputs
    )

    for row in range(len(CHAIN_WEIGHTS)):
        fd = _central_difference(
            payload,
            run_forward,
            genome_object,
            state,
            transformed,
            inputs,
            "conn_attrs",
            row,
            0,
        )
        np.testing.assert_allclose(
            conn_grad[0, row, 0],
            fd[0],
            rtol=0.0,
            atol=FD_TOLERANCE,
            err_msg=f"connection row {row}",
        )
        # a weight the read-out depends on cannot have a zero derivative, so a pair
        # of all-zero numbers would not pass for agreement
        assert abs(float(conn_grad[0, row, 0])) > FD_TOLERANCE


def test_node_attr_gradient_matches_finite_differences(
    payloads, runner, run_transform, run_forward
):
    """Invariant: ``d loss / d node_attrs`` equals central differences of the same loss.

    Oracle: the same finite-difference procedure applied to the ``bias``/``response``
    columns of every present row. The ``aggregation``/``activation`` columns are
    option *ids* stored as floats: they have no derivative (they select a kernel,
    they do not scale one), so they are held constant and their gradient entries
    are asserted to be exactly ``0`` and finite, never to match a difference
    quotient. The input rows carry no partial derivative — their value is seeded
    after the payload is read — which the loop pins as ``0`` on both sides.
    The measured maximum difference on the differentiated columns is ≈3e-5.
    """
    genome_object, state, payload, transformed = _chain(payloads, run_transform)
    inputs = np.array([[0.4, -0.9]], dtype=np.float32)
    node_grad, _ = _run_grad(
        runner, genome_object, state, payload, transformed, inputs
    )

    present_rows = [
        row
        for row in range(MAX_NODES)
        if payload.node_keys[0, row] != payloads.I_INF
    ]
    assert len(present_rows) == MAX_NODES  # the chain has no padding row
    for row in present_rows:
        for column in (0, 1):  # bias, response
            fd = _central_difference(
                payload,
                run_forward,
                genome_object,
                state,
                transformed,
                inputs,
                "node_attrs",
                row,
                column,
            )
            np.testing.assert_allclose(
                node_grad[0, row, column],
                fd[0],
                rtol=0.0,
                atol=FD_TOLERANCE,
                err_msg=f"node row {row}, column {column}",
            )

    # the option-id columns are gene data, not differentiated inputs
    id_columns = [2, 3]
    assert np.all(np.isfinite(node_grad[:, :, id_columns]))
    assert np.all(node_grad[:, :, id_columns] == 0.0)

    # and the differentiated columns are not uniformly zero: the hidden and output
    # rows really do move the read-out
    for row in NODE_ATTRS:
        for column in (0, 1):
            assert abs(float(node_grad[0, row, column])) > FD_TOLERANCE


def test_bias_only_network_gradient_is_closed_form(
    payloads, runner, run_transform
):
    """Invariant: with zero weights the read-out is the bias and both gradients are closed form.

    Oracle: closed form. ``DefaultConn``'s message is ``weight * source``, so with
    every weight ``0`` a node aggregates ``0`` regardless of its sources; with the
    identity activation and the output row read out unactivated, the read-out is
    ``bias + response * 0 = bias``. For ``L = loss(readout)`` that gives
    ``dL/dbias = dL/dreadout``, ``dL/dresponse = 0`` (the aggregate it multiplies
    is zero) and ``dL/dweight = dL/dreadout * response * source_value`` — three
    expressions a test can state without any reference implementation. This is the
    case where the *product* rule through the weight is still exercised while the
    value path is trivial.
    """
    bias_out = 0.75
    response_out = 2.0
    inputs = np.array([[0.4, -0.9]], dtype=np.float32)

    genome_object = payloads.default_genome(
        num_inputs=NUM_INPUTS,
        num_outputs=NUM_OUTPUTS,
        max_nodes=4,
        max_conns=3,
        max_in_degree=4,
    )
    state = payloads.genome_state(genome_object)
    payload = payloads.new_payload(1, 4, 3)
    payloads.set_node(payload, 0, 0, 0)  # input row 0, key 0
    payloads.set_node(payload, 0, 1, 1)  # input row 1, key 1
    payloads.set_node(
        payload, 0, 3, 3, attrs=(bias_out, response_out, 0.0, IDENTITY)
    )  # output row 3, key 3
    payloads.set_conn(payload, 0, 0, 0, 3, attrs=(0.0,))  # zero weight: live, but silent
    payloads.set_conn(payload, 0, 1, 1, 3, attrs=(0.0,))
    transformed = run_transform(genome_object, state, payload)

    node_grad, conn_grad = _run_grad(
        runner, genome_object, state, payload, transformed, inputs
    )

    # the read-out the closed form refers to, read off the payload itself
    readout = bias_out + response_out * 0.0
    d_readout = 2.0 * (readout - TARGET)

    np.testing.assert_allclose(node_grad[0, 3, 0], d_readout, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(node_grad[0, 3, 1], 0.0, rtol=0.0, atol=1e-6)
    for conn_row, source in enumerate(inputs[0]):
        np.testing.assert_allclose(
            conn_grad[0, conn_row, 0],
            d_readout * response_out * float(source),
            rtol=1e-6,
            atol=1e-6,
            err_msg=f"edge from input {conn_row}",
        )
    # absent rows of a padded family stay zero, and input rows never consume their bias
    np.testing.assert_allclose(conn_grad[0, 2, 0], 0.0, rtol=0.0, atol=1e-6)
    np.testing.assert_allclose(node_grad[0, :NUM_INPUTS, :], 0.0, rtol=0.0, atol=1e-6)


# --------------------------------------------------------------------------- #
# NaN and padding
# --------------------------------------------------------------------------- #


def test_nan_padding_never_poisons_a_gradient(
    payloads, runner, run_transform
):
    """Invariant: a padded or dead node contributes a cotangent of ``0``, not ``NaN``.

    Oracle: the documented cleaning rule. ``node_attrs``/``conn_attrs`` padding rows
    are ``NaN``, and a present node with no live incoming edge holds a ``NaN``
    *value* — whose attribute row is additionally ``NaN`` biased here, so the
    input-boundary mask (``select(exists & ~isnan(a), a, 0)``) is exercised on a
    live key rather than only on padding. Every returned gradient entry must be
    finite: the test asserts finiteness over the whole gradient pair — on a payload
    that deliberately mixes padding rows, an isolated node and a live network —
    rather than on selected entries.
    """
    genome_object = payloads.default_genome(
        num_inputs=NUM_INPUTS,
        num_outputs=NUM_OUTPUTS,
        max_nodes=MAX_NODES,
        max_conns=4,
        max_in_degree=4,
    )
    state = payloads.genome_state(genome_object)
    payload = payloads.new_payload(1, MAX_NODES, 4)
    payloads.set_node(payload, 0, 0, 0)  # input row 0, key 0
    payloads.set_node(payload, 0, 1, 1)  # input row 1, key 1
    # present, but on no edge at all: never updated, so its value stays NaN, and its
    # bias is NaN as well so the sanitizer has to clean a *live* row too
    payloads.set_node(
        payload, 0, 2, 2, attrs=(float("nan"), 1.0, 0.0, IDENTITY)
    )
    # rows 3 and 4 stay absent (I_INF keys, NaN attributes) — the padding case
    payloads.set_node(
        payload, 0, 5, 5, attrs=(0.25, 1.0, 0.0, IDENTITY)
    )  # output row 5, key 5
    payloads.set_conn(payload, 0, 0, 0, 5, attrs=(0.75,))
    payloads.set_conn(payload, 0, 1, 1, 5, attrs=(-0.5,))
    transformed = run_transform(genome_object, state, payload)

    inputs = np.array([[0.4, -0.9]], dtype=np.float32)
    node_grad, conn_grad = _run_grad(
        runner, genome_object, state, payload, transformed, inputs
    )

    assert np.all(np.isfinite(node_grad)), node_grad
    assert np.all(np.isfinite(conn_grad)), conn_grad
    # the live edges and the output read-out do carry a derivative, so the mask above
    # is not hiding an all-zero graph
    assert abs(float(conn_grad[0, 0, 0])) > FD_TOLERANCE
    assert abs(float(node_grad[0, 5, 0])) > FD_TOLERANCE
    # the dead node's row and the NaN-biased column survive as exact zeros
    np.testing.assert_array_equal(node_grad[0, 2, :], np.zeros(4, dtype=np.float32))


def test_padded_rows_receive_a_zero_gradient(payloads, runner, run_transform):
    """Invariant: gradient entries on absent rows are exactly ``0``.

    Oracle: the padding mask. Rows whose ``node_keys``/``conn_keys`` are ``I_INF``
    (and whose attribute rows are ``NaN``) never take part in the forward pass, so
    their gradient is masked to ``0`` — not to ``NaN`` and not to a finite value
    that an optimizer would then apply to a nonexistent gene. The live rows of the
    same payload are asserted to be nonzero, so the zeros can only come from the
    mask.
    """
    genome_object = payloads.default_genome(
        num_inputs=NUM_INPUTS,
        num_outputs=NUM_OUTPUTS,
        max_nodes=MAX_NODES,
        max_conns=4,
        max_in_degree=4,
    )
    state = payloads.genome_state(genome_object)
    payload = payloads.new_payload(1, MAX_NODES, 4)
    payloads.set_node(payload, 0, 0, 0)
    payloads.set_node(payload, 0, 1, 1)
    payloads.set_node(payload, 0, 5, 5, attrs=(0.25, 1.0, 0.0, IDENTITY))
    payloads.set_conn(payload, 0, 0, 0, 5, attrs=(0.75,))
    payloads.set_conn(payload, 0, 1, 1, 5, attrs=(-0.5,))
    # connection rows 2 and 3 stay absent, node rows 2, 3 and 4 too
    transformed = run_transform(genome_object, state, payload)

    inputs = np.array([[0.4, -0.9]], dtype=np.float32)
    node_grad, conn_grad = _run_grad(
        runner, genome_object, state, payload, transformed, inputs
    )

    absent_nodes = payload.node_keys[0] == payloads.I_INF
    absent_conns = payload.conn_keys[0, :, 0] == payloads.I_INF
    assert absent_nodes.any() and absent_conns.any()

    np.testing.assert_array_equal(
        node_grad[:, absent_nodes, :], np.zeros_like(node_grad[:, absent_nodes, :])
    )
    np.testing.assert_array_equal(
        conn_grad[:, absent_conns, :], np.zeros_like(conn_grad[:, absent_conns, :])
    )
    # the present rows are not zero, so the mask (not the arithmetic) produced the zeros
    assert np.all(conn_grad[0, ~absent_conns, 0] != 0.0)
    assert np.all(node_grad[0, 5, :2] != 0.0)


# --------------------------------------------------------------------------- #
# scalarization and the population axis
# --------------------------------------------------------------------------- #


def test_scalarized_loss_leaves_per_individual_gradients_unchanged(
    payloads, runner, run_transform, run_forward
):
    """Invariant: the population sum in ``sum(loss_fn(outputs))`` does not couple individuals.

    Oracle: finite differences per individual (as in
    :func:`test_conn_attr_gradient_matches_finite_differences`, but perturbing one
    member's entry only) compared against the gradients of one ``P == 4`` run whose
    members differ in weights and inputs. The scalarization adds a constant
    multiple of the other members' losses to the objective, whose derivative with
    respect to *this* member's attributes is zero, so every entry must be
    unchanged. It also pins the sign and the scale of the accumulated term: a mean
    instead of a sum would scale every gradient by ``1/P`` and fail the absolute
    comparison by a factor of four.
    """
    pop_size = 4
    genome_object, state, payload, transformed = _chain(
        payloads, run_transform, pop_size=pop_size
    )
    # four distinct members: distinct weights *and* distinct inputs, so a coupled
    # population axis could not pass by symmetry. The scale stays near 1 on purpose:
    # the five weights compound, and a member whose read-out is O(100) would push the
    # float32 difference quotient past the absolute tolerance for reasons that have
    # nothing to do with the scalarization.
    member_scale = np.array([1.0, 1.1, 1.2, 1.3], dtype=np.float32)
    payload.conn_attrs[:, : len(CHAIN_WEIGHTS), 0] *= member_scale[:, None]
    inputs = np.array(
        [[0.3, -0.4], [-0.2, 0.5], [0.5, 0.2], [-0.4, -0.3]], dtype=np.float32
    )

    _, conn_grad = _run_grad(
        runner, genome_object, state, payload, transformed, inputs
    )
    assert conn_grad.shape[0] == pop_size

    for member in range(pop_size):
        for row in range(len(CHAIN_WEIGHTS)):
            fd = _central_difference(
                payload,
                run_forward,
                genome_object,
                state,
                transformed,
                inputs,
                "conn_attrs",
                row,
                0,
                individual=member,
            )
            np.testing.assert_allclose(
                conn_grad[member, row, 0],
                fd[member],
                rtol=0.0,
                atol=FD_TOLERANCE,
                err_msg=f"individual {member}, connection row {row}",
            )

    # the four objectives really were four different objectives: a broadcast or a
    # cross-individual accumulation would collapse these rows
    distinct = {
        tuple(np.round(conn_grad[member, :, 0], 6)) for member in range(pop_size)
    }
    assert len(distinct) == pop_size


# --------------------------------------------------------------------------- #
# grad_from_payload: the host-only wrapper, its refusals, and dense_grad_plan's
# --------------------------------------------------------------------------- #


def test_grad_from_payload_matches_finite_differences_for_one_individual(
    payloads, run_forward, run_transform
):
    """Invariant: the host wrapper returns the same derivatives as ``grad`` for ``P == 1``.

    Oracle: central differences of the loss through the *plain* ``forward``, entry by
    entry. The wrapper derives its own schedule from the payload, then builds, compiles and
    runs the gradient graph, so a schedule that disagreed with ``transform`` would put a
    weight on another ``(destination, source)`` cell and fail here.
    """
    genome_object, state, payload, transformed = _chain(payloads, run_transform)
    inputs = np.array([[0.4, -0.9]], dtype=np.float32)

    node_grad, conn_grad = genome_object.grad_from_payload(
        state, payload.args(), inputs, _loss
    )

    assert type(node_grad) is np.ndarray and type(conn_grad) is np.ndarray
    assert node_grad.shape == payload.node_attrs.shape
    assert conn_grad.shape == payload.conn_attrs.shape
    assert node_grad.dtype == conn_grad.dtype == np.float32

    worst = _compare_with_finite_differences(
        payloads, run_forward, genome_object, state, payload, transformed, inputs,
        node_grad, conn_grad,
    )
    assert worst < FD_TOLERANCE  # the headroom the float32 pair left


def test_grad_from_payload_matches_finite_differences_for_a_batch(
    payloads, runner, run_forward, run_transform
):
    """Invariant: on distinct members the wrapper stays per-individual, never a broadcast.

    Oracle: (a) bit equality with the two-step recipe (the fixture's ``transform`` plus
    ``genome.grad``), since the wrapper derives the schedule from the payload itself; (b)
    central differences of the plain ``forward`` per member, which a gradient leaking
    across the population axis would fail. The three members differ in weights, biases and
    inputs, and their gradient rows are asserted pairwise distinct.
    """
    pop_size = 3
    genome_object, state, payload, transformed = _chain(
        payloads, run_transform, pop_size=pop_size
    )
    payload.conn_attrs[:, : len(CHAIN_WEIGHTS), 0] *= np.array(
        [1.0, 1.1, 1.2], dtype=np.float32
    )[:, None]
    payload.node_attrs[:, 2:, 0] += np.array([0.0, 0.05, 0.1], dtype=np.float32)[:, None]
    inputs = np.array([[0.3, -0.4], [-0.2, 0.5], [0.5, 0.2]], dtype=np.float32)

    pair_node, pair_conn = _run_grad(
        runner, genome_object, state, payload, transformed, inputs
    )
    node_grad, conn_grad = genome_object.grad_from_payload(
        state, payload.args(), inputs, _loss
    )

    assert np.array_equal(node_grad, pair_node)
    assert np.array_equal(conn_grad, pair_conn)
    _compare_with_finite_differences(
        payloads, run_forward, genome_object, state, payload, transformed, inputs,
        node_grad, conn_grad,
    )
    # three different objectives: a broadcast along the population axis collapses these
    weights = {tuple(np.round(conn_grad[p, :, 0], 6)) for p in range(pop_size)}
    nodes = {tuple(np.round(node_grad[p, 2:, :2], 6).ravel()) for p in range(pop_size)}
    assert len(weights) == pop_size and len(nodes) == pop_size


def test_grad_from_payload_padding_gradients_are_exactly_zero(payloads):
    """Invariant: the wrapper's padding rows gradient to exactly ``0``, in every member.

    Oracle: the padding mask against ``np.zeros_like`` *exactly* (never ``allclose``), on a
    payload whose node rows 5 and 6 and connection rows 5..7 are absent while a live chain
    reads out at row 7 (the row/key invariant puts an output row at ``max_nodes - 1``).
    The present rows are asserted nonzero, so the zeros can only come from the mask.
    """
    genome_object = payloads.default_genome(
        num_inputs=NUM_INPUTS, num_outputs=NUM_OUTPUTS, max_nodes=8, max_conns=8,
        max_in_degree=4,
    )
    state = payloads.genome_state(genome_object)
    payload = payloads.new_payload(pop_size=2, max_nodes=8, max_conns=8)
    attrs = {2: (0.1, 1.0), 3: (-0.2, 0.9), 4: (0.15, 1.1), 7: (0.05, 1.0)}
    edges = [(0, 2), (1, 2), (2, 3), (3, 4), (4, 7)]
    for p in range(payload.pop_size):
        payloads.set_node(payload, p, 0, 0)
        payloads.set_node(payload, p, 1, 1)
        for row, (bias, response) in attrs.items():
            payloads.set_node(payload, p, row, row, attrs=(bias, response, 0.0, IDENTITY))
        for row, (source, target) in enumerate(edges):
            payloads.set_conn(payload, p, row, source, target, attrs=(CHAIN_WEIGHTS[row],))

    node_grad, conn_grad = genome_object.grad_from_payload(
        state, payload.args(), np.array([[0.4, -0.9], [0.2, 0.3]], np.float32), _loss
    )

    rows = np.arange(payload.max_nodes)
    for member in range(payload.pop_size):
        absent_nodes = payload.node_keys[member] == payloads.I_INF
        absent_conns = payload.conn_keys[member, :, 0] == payloads.I_INF
        assert absent_nodes.any() and absent_conns.any()
        np.testing.assert_array_equal(
            node_grad[member, absent_nodes], np.zeros_like(node_grad[member, absent_nodes])
        )
        np.testing.assert_array_equal(
            conn_grad[member, absent_conns], np.zeros_like(conn_grad[member, absent_conns])
        )
        assert np.all(conn_grad[member, ~absent_conns, 0] != 0.0)
        assert np.all(node_grad[member, (rows >= NUM_INPUTS) & ~absent_nodes, :2] != 0.0)


def test_grad_from_payload_rejects_bare_per_individual_rows(
    payloads, common, run_transform
):
    """Invariant: only the batch-first spelling is accepted, never the bare rows.

    Oracle: the documented ``TensorNEATError`` naming the spelling (``must be
    batch-first``); a checkpoint stores the bare rows, so accepting them would read an
    ``(N,)`` key row as ``N`` individuals.
    """
    genome_object, state, payload, _ = _chain(payloads, run_transform)
    bare = tuple(array[0] for array in payload.args())

    _expect_raises(
        common.TensorNEATError, "must be batch-first",
        lambda: genome_object.grad_from_payload(
            state, bare, np.array([[0.4, -0.9]], np.float32), _loss
        ),
    )


def test_grad_from_payload_rejects_inputs_with_the_wrong_row_count(
    payloads, common, run_transform
):
    """Invariant: ``inputs`` must carry one row per individual.

    Oracle: the documented ``TensorNEATError`` naming the counts (``one input row per
    individual``); the wrapper cannot broadcast an input row.
    """
    genome_object, state, payload, _ = _chain(payloads, run_transform, pop_size=2)
    wrong = np.zeros((3, NUM_INPUTS), dtype=np.float32)

    _expect_raises(
        common.TensorNEATError, "one input row per individual",
        lambda: genome_object.grad_from_payload(state, payload.args(), wrong, _loss),
    )


def test_grad_from_payload_rejects_a_traced_payload_leaf(
    payloads, common, genome, runner, run_transform
):
    """Invariant: the helper is host-only and refuses a traced ``SymbolicTensor``.

    Oracle: the documented ``TensorNEATError`` containing ``host-only``, reached by calling
    the helper inside an active trace, where building and running another executable is
    exactly what it must refuse.
    """
    genome_object, state, payload, _ = _chain(payloads, run_transform)
    inputs = np.array([[0.4, -0.9]], dtype=np.float32)

    def traced(*arrays):
        """Call the host-only helper from inside a trace: it must refuse."""
        return genome_object.grad_from_payload(
            state, genome.GenomeTensors(*arrays), inputs, _loss
        )

    _expect_raises(
        common.TensorNEATError, "host-only", lambda: runner.run_defn(traced, *payload.args())
    )


def _expect_grad_refusal(
    payloads, run_transform, genome_object, error, fragment, payload=None
):
    """Assert ``grad`` refuses ``payload`` (default: the chain) of ``genome_object``.

    Returns:
        ``(state, payload, transformed)`` of the refused call.
    """
    state = payloads.genome_state(genome_object)
    if payload is None:
        payload = payloads.chain_payload(
            pop_size=1, max_nodes=MAX_NODES, max_conns=MAX_CONNS,
            num_inputs=NUM_INPUTS, num_outputs=NUM_OUTPUTS,
        )
    transformed = run_transform(genome_object, state, payload)
    inputs = np.array([[0.4, -0.9]], dtype=np.float32)
    _expect_raises(
        error, fragment,
        lambda: genome_object.grad(state, transformed, *payload.args(), inputs, _loss),
    )
    return state, payload, transformed


def test_grad_rejects_a_recurrent_genome(payloads, common, run_transform):
    """Invariant: a non-feedforward genome cannot use the dense gradient pass.

    Oracle: the documented ``ConfigError`` naming the sweep (``level-synchronous``); a
    recurrent ring has no finite node level for the level loop to follow.
    """
    _expect_grad_refusal(
        payloads, run_transform, payloads.recurrent_genome(), common.ConfigError,
        "level-synchronous", payload=payloads.ring_payload(pop_size=1),
    )


def test_grad_rejects_a_node_gene_without_a_single_sum_or_mean_aggregation(
    payloads, common, genome, run_transform
):
    """Invariant: only a one-option ``AGG.sum``/``AGG.mean`` node gene is differentiable.

    Oracle: the documented ``ConfigError`` naming the options (``AGG.sum and AGG.mean``),
    for a gene configured with *both*: the traced pass selects one aggregation per node,
    while the dense matrix carries a single ``dot``, so two options would drop the choice.
    """
    genome_object = genome.DefaultGenome(
        num_inputs=NUM_INPUTS, num_outputs=NUM_OUTPUTS, max_nodes=MAX_NODES,
        max_conns=MAX_CONNS, max_in_degree=4,
        node_gene=genome.DefaultNode(
            aggregation_options=[common.AGG.sum, common.AGG.mean]
        ),
    )
    _expect_grad_refusal(
        payloads, run_transform, genome_object, common.ConfigError, "AGG.sum and AGG.mean"
    )


def test_grad_rejects_a_connection_gene_without_a_weight_column(
    payloads, common, genome, run_transform
):
    """Invariant: the dense weight matrix is built from the declared ``weight`` column.

    Oracle: the documented ``ConfigError`` naming that column (``weight' column``), for a
    family whose only attribute column is called ``gain``.
    """

    class GainConn(genome.DefaultConn):
        """A connection gene whose attribute column is ``gain`` instead of ``weight``."""

        def __init__(self):
            super().__init__()
            self.attr_specs = (genome.GeneAttr(name="gain", kind="float"),)

    genome_object = genome.DefaultGenome(
        num_inputs=NUM_INPUTS, num_outputs=NUM_OUTPUTS, max_nodes=MAX_NODES,
        max_conns=MAX_CONNS, max_in_degree=4, conn_gene=GainConn(),
    )
    _expect_grad_refusal(
        payloads, run_transform, genome_object, common.ConfigError, "weight' column"
    )


def test_grad_rejects_two_connections_with_the_same_endpoints(
    payloads, common, run_transform
):
    """Invariant: one weight per ``(destination, source)`` cell, or nothing.

    Oracle: the documented ``ConfigError`` naming the collision (``same endpoints``). A
    chain plus one extra row repeating ``0 -> 2`` with another weight is a topology a
    scatter would silently resolve to whichever row it kept.
    """
    genome_object = payloads.default_genome(
        num_inputs=NUM_INPUTS, num_outputs=NUM_OUTPUTS, max_nodes=MAX_NODES,
        max_conns=MAX_CONNS + 2, max_in_degree=4,
    )
    payload = payloads.chain_payload(
        pop_size=1, max_nodes=MAX_NODES, max_conns=MAX_CONNS + 2,
        num_inputs=NUM_INPUTS, num_outputs=NUM_OUTPUTS,
    )
    payloads.set_conn(payload, 0, len(CHAIN_WEIGHTS), 0, 2, attrs=(0.7,))
    _expect_grad_refusal(
        payloads, run_transform, genome_object, common.ConfigError, "same endpoints",
        payload=payload,
    )


def test_grad_rejects_an_edge_outside_every_incoming_window(
    payloads, common, run_transform
):
    """Invariant: a window too narrow for an in-degree is a ``ShapeBoundError``, not a drop.

    Oracle: the documented ``ShapeBoundError`` naming the bound (``outside every incoming
    window``), plus the ``edge_overflow`` the same schedule reports. A lossy schedule is
    legal (pinned in ``test_payload.py``); what is refused is differentiating an edge that
    exists in the payload but in no window.
    """
    genome_object = payloads.default_genome(
        num_inputs=NUM_INPUTS, num_outputs=NUM_OUTPUTS, max_nodes=8, max_conns=8,
        max_in_degree=2,
    )
    payload = payloads.new_payload(pop_size=1, max_nodes=8, max_conns=8)
    for row in (0, 1, 2, 7):
        payloads.set_node(payload, 0, row, row)
    payloads.set_conn(payload, 0, 0, 0, 7)  # three edges into node 7, window width 2
    payloads.set_conn(payload, 0, 1, 2, 7)
    payloads.set_conn(payload, 0, 2, 1, 7)
    _, _, transformed = _expect_grad_refusal(
        payloads, run_transform, genome_object, common.ShapeBoundError,
        "outside every incoming window", payload=payload,
    )
    assert int(np.asarray(transformed.edge_overflow)[0]) == 1  # the schedule is lossy
