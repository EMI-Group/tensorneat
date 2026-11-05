"""Differentiate one evolved topology exactly, then check it against finite differences.

Run with::

    python examples/interpret_visualize/grad_descent.py

The ``Pipeline`` loop never differentiates anything: NEAT searches by mutation and
selection. This script demonstrates the other half of the library -- the exact
gradient path *through* an evolved topology from the GECCO 2024 paper
(arXiv 2404.01817), exposed as ``genome.grad_from_payload``: it derives the
execution schedule of one batch-first payload, differentiates the forward pass
built on that schedule, and returns ``d loss / d node_attrs`` and
``d loss / d conn_attrs``. That path is what lets a caller run gradient descent
on the evolved topology instead of, or alongside, selection; the evolutionary
loop stays the search algorithm.

Two operands of ``grad_from_payload`` are traced, which a first call usually
trips over: ``loss_fn`` must be built from ``evoxc`` tensors (never a numpy
target array), and the payload must stay batch-first (a population on axis 0, so
one individual is a ``P == 1`` batch). Squaring is written as a multiply,
``(a - b) * (a - b)``, because ``** 2`` lowers to ``power``, whose pullback emits
an unconditional ``log`` of a possibly negative base.

A gradient is only as good as its arithmetic, so the second half of the script
perturbs a handful of attribute entries by ``+-eps`` in the payload, re-runs the
same forward graph, and compares the central difference with the analytic value.
"""

import numpy as np
import evoxc

from tensorneat import genome
from tensorneat.genome import GenomeTensors

#: Constant predictor the loss is measured against: the linear optimum of XOR.
TARGET = 0.5
#: Central-difference step for the gradient check (float32 forward pass).
FD_EPS = 1e-3
#: Tolerance on the worst analytic-vs-finite-difference discrepancy.
FD_TOL = 1e-3
#: How many of the largest-magnitude entries of each gradient array to check.
FD_ENTRIES = 4


def squared_error(output):
    """Per-individual squared error to ``TARGET`` (traced: evoxc ops, no ``** 2``)."""
    diff = output - TARGET
    return diff * diff


def loss_value(output):
    """The scalar the gradient graph differentiates: ``sum(squared_error(output))``."""
    diff = output - TARGET
    return float(np.sum(diff * diff))


def build_forward(g, state, arrays, inputs):
    """Build one reusable forward graph: payload leaves plus query -> output.

    ``transform`` runs first inside the graph (it derives the schedule from the
    keys), then ``forward`` consumes the same leaves, so every later
    finite-difference evaluation reuses this one executable.
    """
    def kernel(node_keys, node_attrs, conn_keys, conn_attrs, query):
        """Trace-time body: derive the schedule, then run the forward pass."""
        transformed = g.transform(
            state, GenomeTensors(node_keys, node_attrs, conn_keys, conn_attrs)
        )
        return g.forward(
            state, transformed, node_keys, node_attrs, conn_keys, conn_attrs, query
        )

    specs = [
        evoxc.TensorSpec(tuple(int(size) for size in array.shape), array.dtype)
        for array in (*arrays, inputs)
    ]
    return evoxc.build(kernel, *specs, backend="numpy")


def evaluate(executable, arrays, inputs):
    """Run the forward graph on host arrays and return its ``(P, num_outputs)`` output."""
    outputs = evoxc.run(
        executable, *(evoxc.from_numpy(array) for array in (*arrays, inputs))
    )
    return np.asarray(outputs.numpy())


def perturbed_loss(executable, arrays, inputs, slot, index, delta):
    """Loss after adding ``delta`` to one flat attribute entry of the payload."""
    edited = list(arrays)
    flat = edited[slot].copy().reshape(-1)
    flat[index] += delta
    edited[slot] = flat.reshape(edited[slot].shape)
    return loss_value(evaluate(executable, tuple(edited), inputs))


def check_entries(name, slot, grads, executable, arrays, inputs):
    """Compare the largest analytic entries of one gradient array with central differences.

    Returns:
        The worst absolute analytic-vs-finite-difference discrepancy seen.
    """
    flat = grads.reshape(-1)
    print(f"{name}: {len(flat)} entries, central difference on the largest-magnitude ones")
    worst = 0.0
    checked = 0
    for index in np.argsort(-np.abs(flat)):
        if checked == FD_ENTRIES:
            break
        analytic = float(flat[index])
        if analytic == 0.0:
            continue
        finite_difference = (
            perturbed_loss(executable, arrays, inputs, slot, index, FD_EPS)
            - perturbed_loss(executable, arrays, inputs, slot, index, -FD_EPS)
        ) / (2.0 * FD_EPS)
        discrepancy = abs(analytic - finite_difference)
        worst = max(worst, discrepancy)
        checked += 1
        print(
            f"  entry {index}: analytic {analytic: .8f} |"
            f" finite difference {finite_difference: .8f} | discrepancy {discrepancy:.2e}"
        )
    assert checked > 0, f"{name}: no nonzero gradient entry to check"
    return worst


def main():
    # One small evolved topology: 2 inputs, 2 hidden nodes, 1 output, and padding
    # rows the library always carries (max_nodes/max_conns are static at build time).
    g = genome.DefaultGenome(
        num_inputs=2,
        num_outputs=1,
        max_nodes=8,
        max_conns=12,
        init_hidden_layers=(2,),
    )
    state = g.setup()

    # initialize() is a trace-time kernel, so one host graph call draws one
    # individual; pop_size=1 yields the batch-first P == 1 payload.
    initializer = evoxc.build(
        lambda key: g.initialize(state, key, 1),
        evoxc.TensorSpec((), evoxc.int64),
        backend="numpy",
    )
    payload = GenomeTensors(
        *[tensor.numpy() for tensor in evoxc.run(initializer, evoxc.random.key(0))]
    )
    arrays = tuple(payload)
    inputs = np.array([[0.5, -1.0]], dtype=np.float32)

    print(g.repr(state, payload))

    # The loss value is not part of an evoxc.grad graph (it carries gradients
    # only), so read it from the same forward pass with the same loss.
    forward = build_forward(g, state, arrays, inputs)
    output = evaluate(forward, arrays, inputs)
    print(f"forward output (1, 1): {output.ravel()}")
    print(f"loss = sum of squared errors to {TARGET}: {loss_value(output):.6e}")

    # The documented convenience wrapper: derive the schedule (argsort has no AD
    # rule, so it may not live inside the differentiated graph), build and run
    # the gradient graph, and return the two attribute gradient arrays.
    node_grads, conn_grads = g.grad_from_payload(state, payload, inputs, squared_error)
    print(f"node_attrs gradient shape: {node_grads.shape}")
    print(f"conn_attrs gradient shape: {conn_grads.shape}")
    print(f"node_attrs gradient norm: {np.linalg.norm(node_grads):.6e}")
    print(f"conn_attrs gradient norm: {np.linalg.norm(conn_grads):.6e}")
    print(f"node_attrs nonzero entries: {np.count_nonzero(node_grads)}")
    print(f"conn_attrs nonzero entries: {np.count_nonzero(conn_grads)}")
    # Zeros are contractual, not accidental: input node attributes are unused
    # (the query supplies their values) and padding/absent rows gradient to
    # exactly 0, because the attribute tensors are NaN-cleaned and masked.

    # grad_from_payload traces a fresh graph per call, so the check below reuses
    # the forward graph built above instead of tracing one per perturbation.
    print(f"\nchecking analytic gradients against a central difference (eps={FD_EPS})")
    worst = max(
        check_entries("node_attrs", 1, node_grads, forward, arrays, inputs),
        check_entries("conn_attrs", 3, conn_grads, forward, arrays, inputs),
    )
    print(f"max |analytic - finite difference| = {worst:.2e} (tolerance {FD_TOL:g})")
    assert worst < FD_TOL, f"gradient check failed: worst discrepancy {worst:.2e}"
    print("PASS")


if __name__ == "__main__":
    main()
