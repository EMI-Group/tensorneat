"""Unit tests for the host-facing helpers: ``network_dict``, ``repr``, ``hash``,
``sympy_func`` and ``visualize``.

What this module proves
-----------------------
What these five methods owe their consumers is a stable, documented shape:

* ``network_dict`` → ``{"nodes": {key: node dict}, "conns": {(in_key, out_key):
  conn dict}}``, plus ``topo_order``/``topo_layers``/``useful_nodes`` for
  ``DefaultGenome`` (the three entries the code-generation and drawing paths read);
* ``repr`` → a header line with the gene counts, then one line per node (annotated
  ``(input)``/``(output)``) and one per connection;
* ``hash`` → a **host-only** numpy fold returning ``(P,)`` ``uint32``, used for
  fast de-duplication: it must be deterministic, structure- and
  attribute-sensitive, and padding must never distinguish two payloads;
* ``sympy_func`` → the legacy **8-tuple**
  ``(symbols, args_symbols, input_symbols, nodes_exprs, output_exprs, forward_func,
  topo_order, useful_nodes)``, with the node expressions assembled into closed
  forms and ``args_symbols`` binding the gene values (``forward_func`` applies that
  substitution, and ``common.sympy_tools`` does the same with
  ``expr.subs(args_symbols)``);
* ``visualize`` → a file, drawn with ``networkx``/``matplotlib`` imported *inside*
  the method.

``network_dict``, ``repr``, ``sympy_func`` and ``visualize`` are host-only: they
convert with ``Tensor.numpy()`` and walk rows in Python, so their results must
never be fed back into a kernel — and, symmetrically, they must not be needed to
*run* one.

Oracles used here (never the kernel itself)
-------------------------------------------
* one exact formula: on a two-input/one-hidden/one-output network with identity
  activations, unit weights, zero biases and unit responses, the output expression
  must simplify to ``i0 + i1`` (and ``forward_func([1.0, 2.0]) == 3.0``). It is
  written out by hand, so no part of the symbolic path is compared with itself;
  the same network's tensors-only ``DefaultGenome.forward`` must return ``3.0``
  too, which is the one place the symbolic and the tensor paths meet — through a
  number both can be checked against by hand; the sigmoid branch is pinned on
  ``skip_payload`` (``sigmoid(i0 + i1)`` at the hidden node, ``sigmoid(i0)`` at the
  unactivated output row);
* the payload's own key/row invariant (``support.payloads``), which lets a test
  name the keys and attribute values it expects in ``network_dict``/``repr``
  without reading the implementation;
* set algebra for ``hash``: equal payloads hash equal, a one-attribute change (on
  a connection *or* on a node row) or a different topology hashes differently,
  and two payloads whose *rows are permuted* — so the padding sits on different
  rows — hash equal, because a row hash depends on the gene's key/attributes only;
* the file system for ``visualize``, plus a spy on ``networkx.draw`` for the
  drawing options: the figure existing proves ``save_path`` was honoured, the
  recorded call proves the rest of the keyword arguments were not ignored.

Gotchas this module encodes
---------------------------
* ``import tensorneat.genome`` must **not** import ``networkx`` or ``matplotlib``:
  both are imported inside ``visualize`` for exactly that reason, and the release
  gate is only the loudest version of the rule. The check is a ``sys.modules``
  membership test before the drawing call, plus ``MPLBACKEND=Agg`` so the test
  needs no display.
* ``network_dict`` takes *one individual*, not a batch: a ``P == 1`` payload is
  fine, a ``P > 1`` one is a usage error (the method's documented contract, a
  ``common.TensorNEATError`` telling the caller to slice first), so the test
  passes one individual's ``GenomeTensors``.
* the feedforward utilities are keyed in the *renumbered* space when
  ``whether_re_cound_idx=True``: ``topo_order``/``topo_layers``/``useful_nodes``
  and the node/conn dicts all use the new hidden keys. The level comparison
  against the schedule therefore uses the ``whether_re_cound_idx=False`` branch,
  where a key is also the node's *row* on these payloads (the payload's key/row
  invariant) and the schedule's ``node_level`` can be read off directly.
* ``whether_re_cound_idx=True`` renumbers hidden keys to be contiguous above
  ``max(input, output keys)`` and rewrites the connection keys accordingly; the
  input/output keys are untouched because they are the genome's external
  interface. Both branches are asserted, since a caller that renumbers twice must
  not change the graph — and the payload itself must come back untouched.
* ``RecurrentGenome`` deliberately refuses ``sympy_func`` and ``visualize``: a
  cyclic network has no finite closed form and no depth-ordered layout. Refusing
  is the documented behaviour, and the refusal is a ``common.TensorNEATError``
  naming the reason and pointing at ``network_dict()``/``DefaultGenome`` — not a
  bare ``NotImplementedError`` stub.
* ``hash`` is host-only numpy, not a kernel: it folds the host rows with a
  ``uint32`` mix, so a traced payload is refused loudly (a ``SymbolicTensor`` has
  no host bit pattern yet) instead of being silently reinterpreted. The test
  therefore calls it on plain numpy payloads and pins the refusal separately.

Backend is ``"numpy"`` throughout, through the shared runner only; no
``jax``/``torch``, no network, no GPU, no display.
"""

from __future__ import annotations

import pytest


# --------------------------------------------------------------------------- #
# network_dict
# --------------------------------------------------------------------------- #


def test_network_dict_returns_nodes_conns_and_the_feedforward_utilities(
    genome, common, payloads, run_transform
):
    """Invariant: ``network_dict`` describes exactly the genes the payload holds.

    Oracle: the payload's key/row invariant — the node key set is
    ``set(node_keys[0][node_keys[0] != I_INF])`` and the connection key set is the
    present ``(in_key, out_key)`` pairs — plus the three feedforward entries
    ``topo_order`` (a permutation of the node keys), ``topo_layers`` (a partition
    of it whose level assignments agree with the schedule's ``level``) and
    ``useful_nodes`` (the keys that reach an output; a node that reaches no output
    is *not* in it). Each node/conn dict is the gene's own ``to_dict`` output, so
    the test also checks the documented column names (``idx``/``bias``/``res``/
    ``agg``/``act`` for nodes, ``in``/``out``/``weight`` for connections).

    The ``whether_re_cound_idx=False`` branch is used so that a node key is also
    its row on this payload, which is what lets the schedule's ``node_level`` be
    read off by row; the renumbering branch is the subject of the next test.
    """
    import numpy as np

    g = payloads.default_genome(
        num_inputs=2, num_outputs=1, max_nodes=8, max_conns=16, max_in_degree=4
    )
    state = payloads.genome_state(g)
    payload = payloads.skip_payload()
    network = g.network_dict(
        state,
        genome.GenomeTensors(*payload.args()),
        whether_re_cound_idx=False,
    )

    assert set(network) == {
        "nodes",
        "conns",
        "topo_order",
        "topo_layers",
        "useful_nodes",
    }

    node_keys = payload.node_keys[0]
    present_keys = {int(key) for key in node_keys if int(key) != payloads.I_INF}
    assert set(network["nodes"]) == present_keys

    conn_keys = payload.conn_keys[0]
    present_pairs = {
        (int(in_key), int(out_key))
        for in_key, out_key in conn_keys
        if int(in_key) != payloads.I_INF
    }
    assert set(network["conns"]) == present_pairs

    # A node dict is the gene's own ``to_dict`` output: the documented column
    # names, with the values of that payload row (``DefaultNode`` resolves the two
    # option columns to the configured function names).
    for row, key in enumerate(node_keys):
        key = int(key)
        if key == payloads.I_INF:
            continue
        node = network["nodes"][key]
        assert set(node) == {"idx", "bias", "res", "agg", "act"}
        assert node["idx"] == key
        assert float(node["bias"]) == pytest.approx(float(payload.node_attrs[0, row, 0]))
        assert float(node["res"]) == pytest.approx(float(payload.node_attrs[0, row, 1]))
        assert node["agg"] == "sum"
        assert node["act"] == "sigmoid"

    # Likewise for a connection dict: ``(in_key, out_key)`` keys and the weight.
    for row, (in_key, out_key) in enumerate(conn_keys):
        if int(in_key) == payloads.I_INF:
            continue
        conn = network["conns"][(int(in_key), int(out_key))]
        assert set(conn) == {"in", "out", "weight"}
        assert conn["in"] == int(in_key)
        assert conn["out"] == int(out_key)
        assert float(conn["weight"]) == pytest.approx(
            float(payload.conn_attrs[0, row, 0])
        )

    # ``topo_order`` is a permutation of the node keys and ``topo_layers`` is a
    # partition of it, each layer ascending, in execution order.
    topo_order = network["topo_order"]
    assert len(topo_order) == len(present_keys)
    assert set(topo_order) == present_keys
    topo_layers = network["topo_layers"]
    assert [key for layer in topo_layers for key in layer] == topo_order
    for layer in topo_layers:
        assert layer == sorted(layer)

    # The layer index *is* the schedule's level: both are the longest-path depth
    # from the inputs, one derived on the host, one by ``genome.transform``.
    transformed = run_transform(g, state, payload)
    levels = np.asarray(transformed.node_level)
    level_of_key = {
        int(key): int(levels[0, row])
        for row, key in enumerate(node_keys)
        if int(key) != payloads.I_INF
    }
    for layer_index, layer in enumerate(topo_layers):
        for key in layer:
            assert level_of_key[key] == layer_index

    # ``skip_payload``: 0 -> 2, 1 -> 2, 0 -> 3, 3 -> 7, so only 0, 3 and 7 reach
    # the output row; the bypassed node 2 is present but useful to nothing.
    assert network["useful_nodes"] == {0, 3, 7}
    assert all(isinstance(key, int) for key in network["useful_nodes"])

    # A node with no edge at all is in ``nodes`` and in no output's ancestry, which
    # is the case code generation must skip (its expression would be NaN).
    isolated = payloads.isolated_payload()
    isolated_network = g.network_dict(
        state,
        genome.GenomeTensors(*isolated.args()),
        whether_re_cound_idx=False,
    )
    assert 2 in isolated_network["nodes"]
    assert isolated_network["useful_nodes"] == {0, 1, 7}

    # One individual per call: a batch is a usage error, not something to walk
    # through the wrong axis.
    batch = payloads.skip_payload(pop_size=2)
    with pytest.raises(common.TensorNEATError):
        g.network_dict(state, genome.GenomeTensors(*batch.args()))


def test_network_dict_renumbers_hidden_keys_when_asked(genome, payloads):
    """Invariant: ``whether_re_cound_idx`` renumbers hidden keys only, never the interface.

    Oracle: hand enumeration. With hidden keys ``{20, 41}`` and input/output keys
    ``{0, 1}``/``{7}``, the renumbered hidden keys must be ``{8, 9}`` (consecutive,
    starting at ``max(input, output) + 1``, in row order) and every connection key
    referring to them rewritten to match; the input and output keys are byte-identical
    in both branches, and passing ``whether_re_cound_idx=False`` returns the keys
    verbatim. Renumbering is a *display* transform, so the underlying payload must
    be untouched afterwards, and renumbering twice must yield the same graph.
    """
    import numpy as np

    g = payloads.default_genome(
        num_inputs=2,
        num_outputs=1,
        max_nodes=8,
        max_conns=16,
        max_in_degree=4,
        # The hidden keys below are far above ``max_nodes``; the span is part of
        # the shape signature and must cover them.
        key_span=64,
    )
    state = payloads.genome_state(g)
    payload = payloads.new_payload()
    for row, key in ((0, 0), (1, 1), (2, 20), (3, 41), (7, 7)):
        payloads.set_node(payload, 0, row, key)
    payloads.set_conn(payload, 0, 0, 0, 20)
    payloads.set_conn(payload, 0, 1, 1, 20)
    payloads.set_conn(payload, 0, 2, 0, 41)
    payloads.set_conn(payload, 0, 3, 41, 7)

    node_keys_before = payload.node_keys.copy()
    conn_keys_before = payload.conn_keys.copy()
    individual = genome.GenomeTensors(*payload.args())

    renumbered = g.network_dict(state, individual)
    verbatim = g.network_dict(state, individual, whether_re_cound_idx=False)

    # ``max(input, output) == 7``, so the hidden keys become 8 and 9 in row order.
    assert max(*g.get_input_idx(), *g.get_output_idx()) == 7
    assert set(renumbered["nodes"]) == {0, 1, 7, 8, 9}
    assert set(verbatim["nodes"]) == {0, 1, 7, 20, 41}
    assert renumbered["nodes"][8] == {**verbatim["nodes"][20], "idx": 8}
    assert renumbered["nodes"][9] == {**verbatim["nodes"][41], "idx": 9}

    # Every connection key referring to a hidden node is rewritten to match.
    renaming = {0: 0, 1: 1, 7: 7, 20: 8, 41: 9}
    assert set(verbatim["conns"]) == {(0, 20), (1, 20), (0, 41), (41, 7)}
    assert set(renumbered["conns"]) == {
        (renaming[in_key], renaming[out_key]) for in_key, out_key in verbatim["conns"]
    }
    for (in_key, out_key), conn in verbatim["conns"].items():
        rewritten = renumbered["conns"][(renaming[in_key], renaming[out_key])]
        assert (rewritten["in"], rewritten["out"]) == (
            renaming[in_key],
            renaming[out_key],
        )
        assert rewritten["weight"] == conn["weight"]

    # The external interface is not a display detail: those keys stay verbatim.
    for key in (*g.get_input_idx(), *g.get_output_idx()):
        assert renumbered["nodes"][key] == verbatim["nodes"][key]
    assert min(renumbered["nodes"]) == 0
    assert g.get_input_idx() == [0, 1]
    assert g.get_output_idx() == [7]

    # The feedforward utilities speak the renumbered keys, so the bypassed node
    # (row 2, key 20) is 8 and is not useful.
    assert renumbered["useful_nodes"] == {0, 9, 7}
    assert set(renumbered["useful_nodes"]) <= set(renumbered["nodes"])

    # Renumbering is idempotent: the renumbered keys are themselves contiguous above
    # ``max(input, output)``, so a payload that already carries them is a fixed point
    # of the same walk (a second pass cannot renumber again). The comparison is
    # between two *different* payloads — the original hidden keys 20/41 and the
    # renumbered 8/9 — so a helper that only renumbered once by accident would fail.
    already = payloads.new_payload()
    for row, key in ((0, 0), (1, 1), (2, 8), (3, 9), (7, 7)):
        payloads.set_node(already, 0, row, key)
    payloads.set_conn(already, 0, 0, 0, 8)
    payloads.set_conn(already, 0, 1, 1, 8)
    payloads.set_conn(already, 0, 2, 0, 9)
    payloads.set_conn(already, 0, 3, 9, 7)
    assert already.node_keys[0, 2] == 8 and already.node_keys[0, 3] == 9
    assert g.network_dict(state, genome.GenomeTensors(*already.args())) == renumbered

    # ... and the walk never rewrites the payload it was handed.
    assert np.array_equal(payload.node_keys, node_keys_before)
    assert np.array_equal(payload.conn_keys, conn_keys_before)


def test_network_dict_and_repr_accept_both_one_individual_spellings(genome, payloads):
    """Invariant: the ``(1, ...)`` batch and the bare per-individual rows render identically.

    Oracle: the two documented spellings name the same individual — the batch's
    leading population axis of one row is a container, not data — so every host walk
    must return the same dict (both renumbering branches) and the same string for
    either spelling, and the rendered counts must equal the payload's own gene mask.
    ``Pipeline.auto_run`` hands the helpers a one-individual batch and a checkpoint
    stores bare rows, so both spellings reaching the same genes is the contract, not
    a convenience: a helper that walked the wrong axis would either raise or describe
    ``max_nodes`` "individuals".
    """
    g = payloads.default_genome(
        num_inputs=2, num_outputs=1, max_nodes=8, max_conns=16, max_in_degree=4
    )
    state = payloads.genome_state(g)
    payload = payloads.skip_payload()
    batch = genome.GenomeTensors(*payload.args())
    bare = genome.GenomeTensors(*(array[0] for array in payload.args()))
    assert batch.node_keys.ndim == 2 and batch.node_keys.shape[0] == 1
    assert bare.node_keys.ndim == 1

    for renumber in (True, False):
        batched = g.network_dict(state, batch, whether_re_cound_idx=renumber)
        rows = g.network_dict(state, bare, whether_re_cound_idx=renumber)
        assert batched == rows
        # Non-degenerate: the walk found the payload's genes, not an empty graph. The
        # verbatim branch is the payload's own key set; the renumbering branch renames
        # the hidden keys above the interface (that is its subject elsewhere), so only
        # the gene *count* and the untouched input/output keys are shared.
        assert batched["nodes"] and batched["conns"]
        assert batched["topo_order"] and batched["useful_nodes"]
        verbatim_keys = {
            int(key) for key in payload.node_keys[0] if int(key) != payloads.I_INF
        }
        if renumber:
            assert len(batched["nodes"]) == len(verbatim_keys)
            assert {0, 1, *g.get_output_idx()} <= set(batched["nodes"])
        else:
            assert set(batched["nodes"]) == verbatim_keys

    # The rendered text is identical for both spellings, and its counted header is
    # the payload's own mask — so a helper that read the population axis as the gene
    # axis would print the wrong count (or 16 "individuals").
    batched_text = g.repr(state, batch)
    assert batched_text == g.repr(state, bare)
    present_nodes = int((payload.node_keys[0] != payloads.I_INF).sum())
    present_conns = int((payload.conn_keys[0, :, 0] != payloads.I_INF).sum())
    assert batched_text.startswith(
        f"DefaultGenome(nodes={present_nodes}, conns={present_conns}):"
    )
    assert len(batched_text.splitlines()) == 1 + present_nodes + present_conns


@pytest.mark.parametrize("helper", ["network_dict", "repr"])
def test_one_individual_helpers_reject_a_batch_with_the_slicing_hint(
    genome, common, payloads, helper
):
    """Invariant: ``P > 1`` is a usage error naming the fix, never a silent first row.

    Oracle: the helpers' documented contract (``ConfigError``, a ``TensorNEATError``)
    and its message, which must name the received population, the rule ("takes one
    individual"), and the exact slicing expression to use — the caller of
    ``network_dict``/``repr`` on ``state.pop`` hits this on every generation, so an
    error that does not say *how* to slice sends them to the source.
    """
    g = payloads.default_genome(
        num_inputs=2, num_outputs=1, max_nodes=8, max_conns=16, max_in_degree=4
    )
    state = payloads.genome_state(g)
    batch = payloads.skip_payload(pop_size=3)

    with pytest.raises(common.TensorNEATError) as caught:
        getattr(g, helper)(state, genome.GenomeTensors(*batch.args()))

    message = str(caught.value)
    assert "takes one individual" in message
    assert "slice it first" in message
    assert "GenomeTensors(*(array[p : p + 1] for array in payload))" in message
    # The population it reports is the tensor's, not a guess.
    assert "holds 3 individuals" in message


# --------------------------------------------------------------------------- #
# repr
# --------------------------------------------------------------------------- #


def test_repr_renders_a_header_and_one_line_per_gene(genome, payloads):
    """Invariant: ``repr`` emits a counted header, then exactly one line per present gene.

    Oracle: the documented layout — a first line naming the class and the valid
    node/connection counts, then ``len(present nodes) + len(present conns)`` lines,
    each produced by the gene's own ``repr`` (so ``precision`` reaches the float
    columns), the node lines of input/output rows carrying the ``(input)``/
    ``(output)`` annotation. The expected string is written out by hand from
    ``DefaultNode``/``DefaultConn``'s documented rendering. Padded rows must
    contribute no line at all — a ``NaN``-padded payload that printed its padding
    would make the output unreadable at ``max_nodes=50`` — which the exact string
    on this 4-of-8-node / 3-of-16-connection payload shows.
    """
    g = payloads.default_genome(
        num_inputs=2, num_outputs=1, max_nodes=8, max_conns=16, max_in_degree=4
    )
    state = payloads.genome_state(g)
    payload = payloads.new_payload()
    # ``0.123456`` is not exactly representable in float32 (it is 0.123456001...),
    # so its two roundings differ visibly and the precision knob is observable.
    payloads.set_node(payload, 0, 0, 0, attrs=[0.0, 1.0, 0.0, 0.0])
    payloads.set_node(payload, 0, 1, 1, attrs=[0.123456, 1.5, 0.0, 0.0])
    payloads.set_node(payload, 0, 2, 2, attrs=[0.25, 1.0, 0.0, 0.0])
    payloads.set_node(payload, 0, 7, 7, attrs=[0.0, 1.0, 0.0, 0.0])
    payloads.set_conn(payload, 0, 0, 0, 2, attrs=[0.123456])
    payloads.set_conn(payload, 0, 1, 1, 2, attrs=[1.0])
    payloads.set_conn(payload, 0, 2, 2, 7, attrs=[0.5])
    individual = genome.GenomeTensors(*payload.args())

    default_precision = g.repr(state, individual)
    assert default_precision == (
        "DefaultGenome(nodes=4, conns=3):\n"
        "DefaultNode(idx=0, bias=0.0, response=1.0, aggregation=sum, "
        "activation=sigmoid) (input)\n"
        "DefaultNode(idx=1, bias=0.12, response=1.5, aggregation=sum, "
        "activation=sigmoid) (input)\n"
        "DefaultNode(idx=2, bias=0.25, response=1.0, aggregation=sum, "
        "activation=sigmoid)\n"
        "DefaultNode(idx=7, bias=0.0, response=1.0, aggregation=sum, "
        "activation=sigmoid) (output)\n"
        "DefaultConn(in: 0, out: 2, weight: 0.12)\n"
        "DefaultConn(in: 1, out: 2, weight: 1.0)\n"
        "DefaultConn(in: 2, out: 7, weight: 0.5)\n"
    )

    # ``precision`` reaches both float columns (node bias and conn weight), and
    # nothing else about the rendering changes with it.
    wider = g.repr(state, individual, precision=4)
    assert wider == default_precision.replace("bias=0.12,", "bias=0.1235,").replace(
        "weight: 0.12)", "weight: 0.1235)"
    )
    assert wider.splitlines()[0] == "DefaultGenome(nodes=4, conns=3):"

    # One line per present gene, padding skipped: 1 header + 4 nodes + 3 conns.
    assert len(wider.splitlines()) == 1 + 4 + 3
    assert str(payloads.I_INF) not in wider


# --------------------------------------------------------------------------- #
# hash (host-only numpy fold)
# --------------------------------------------------------------------------- #


def test_hash_is_deterministic_and_distinguishes_different_topologies(genome, payloads):
    """Invariant: ``hash`` is a pure, structure- and attribute-sensitive ``(P,) uint32`` fold.

    Oracle: set algebra on the returned vector. The same payload hashed twice
    gives equal values; two payloads that differ in one connection key give
    different values; two payloads that differ in one *node* attribute (the
    activation id, or the bias) give different values; two payloads that differ
    only in *which rows are padding* (the same genes, permuted into other rows)
    give equal values, because a row hash depends on the gene's
    key/attributes and the total is an xor over the rows; and the dtype is
    ``uint32`` with one entry per individual.

    ``hash`` is host-only numpy (``host.py``), so the test calls it on the plain
    numpy payload — the value is a plain ``numpy`` array, never an ``evoxc``
    tensor. No literal is pinned: the digest is documented as a *heuristic
    de-duplication key*, so only the equality relations above are its contract.
    """
    import numpy as np

    g = payloads.default_genome(
        num_inputs=2, num_outputs=1, max_nodes=8, max_conns=16, max_in_degree=4
    )
    payload = payloads.skip_payload()

    def hashes(one):
        return g.hash(genome.GenomeTensors(*one.args()))

    def copy_of(one):
        return payloads.Payload(
            one.node_keys.copy(),
            one.node_attrs.copy(),
            one.conn_keys.copy(),
            one.conn_attrs.copy(),
        )

    first = hashes(payload)
    assert isinstance(first, np.ndarray)
    assert first.shape == (1,)
    assert first.dtype == np.uint32
    assert np.array_equal(hashes(payload), first)

    # One changed connection key is a different topology.
    retargeted = copy_of(payload)
    assert retargeted.conn_keys[0, 0, 0] == 0 and retargeted.conn_keys[0, 0, 1] == 2
    retargeted.conn_keys[0, 0, 1] = 3
    assert not np.array_equal(hashes(retargeted), first)

    # One changed *connection* attribute is a different genome too (the scale of
    # ``HASH_QUANT_SCALE`` keeps a visible weight apart).
    reweighted = copy_of(payload)
    reweighted.conn_attrs[0, 0, 0] = 2.0
    assert not np.array_equal(hashes(reweighted), first)

    # ... and so is one changed *node* attribute. ``skip_payload`` row 2 is the
    # hidden node, so both these edits land on a present row: the bias column
    # (float) and the activation id column (int-valued float).
    rebiased = copy_of(payload)
    rebias_node_row = 2
    original_bias = float(rebiased.node_attrs[0, rebias_node_row, 0])
    rebased = copy_of(payload)
    rebased.node_attrs[0, rebias_node_row, 0] = np.float32(original_bias + 2.0)
    assert not np.array_equal(hashes(rebased), first)

    reactivated = copy_of(payload)
    assert float(reactivated.node_attrs[0, rebias_node_row, 3]) == 0.0
    reactivated.node_attrs[0, rebias_node_row, 3] = 1.0
    assert not np.array_equal(hashes(reactivated), first)

    # The same genes on different rows — so the padding sits elsewhere — hash
    # identically: the padding rows contribute the same sentinel rows either way.
    moved = copy_of(payload)
    moved.node_keys[0, [2, 4]] = moved.node_keys[0, [4, 2]]
    moved.node_attrs[0, [2, 4]] = moved.node_attrs[0, [4, 2]]
    moved.conn_keys[0, [1, 10]] = moved.conn_keys[0, [10, 1]]
    moved.conn_attrs[0, [1, 10]] = moved.conn_attrs[0, [10, 1]]
    assert moved.node_keys[0, 4] == 2 and moved.node_keys[0, 2] == payloads.I_INF
    assert np.array_equal(hashes(moved), first)

    # An absent row is *only* padding: flipping the sentinel of one padding row
    # to a real gene is a different topology, so the xor fold is not degenerate.
    present = copy_of(payload)
    present.node_keys[0, 2] = 4
    present.node_attrs[0, 2] = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    assert present.node_keys[0, 2] != payloads.I_INF
    assert not np.array_equal(hashes(present), first)

    # Batched: one hash per individual, and the individuals are independent.
    batch = payloads.skip_payload(pop_size=2)
    batched = hashes(batch)
    assert batched.shape == (2,)
    assert np.array_equal(batched, np.array([first[0], first[0]], dtype=np.uint32))
    mixed = copy_of(batch)
    mixed.conn_keys[1, 0, 1] = 3
    mixed_hashes = hashes(mixed)
    assert mixed_hashes[0] == batched[0]
    assert mixed_hashes[1] != batched[1]


def test_hash_refuses_a_traced_payload(genome, common, payloads, runner):
    """Invariant: ``hash`` is host-only, so a traced payload raises ``TensorNEATError``.

    Oracle: the method's documented contract — the fold uses numpy bit views,
    which do not exist for an ``evoxc.SymbolicTensor``, so a traced call is a
    usage error naming the field, the "host-only" rule and the in-graph
    alternative. Silently tracing a reinterpretation (or returning a symbolic
    value) is exactly what the guard prevents.

    Assertion: building the call inside a trace raises ``TensorNEATError`` (not a
    bare ``TypeError`` from ``np.asarray`` on a symbolic tensor), and the message
    names ``node_keys`` — the first payload field — plus ``SymbolicTensor`` and
    the ``evoxc.equal`` alternative.
    """
    g = payloads.default_genome(
        num_inputs=2, num_outputs=1, max_nodes=8, max_conns=16, max_in_degree=4
    )
    payload = payloads.skip_payload()

    def traced_hash(*arrays):
        return g.hash(genome.GenomeTensors(*arrays))

    with pytest.raises(common.TensorNEATError) as caught:
        runner.run_defn(traced_hash, *payload.args())

    message = str(caught.value)
    assert "host-only" in message
    assert "node_keys" in message
    assert "SymbolicTensor" in message
    assert "evoxc.equal" in message


# --------------------------------------------------------------------------- #
# sympy_func
# --------------------------------------------------------------------------- #


def test_sympy_func_returns_the_documented_eight_tuple(genome, payloads):
    """Invariant: ``sympy_func`` returns the legacy 8-tuple with consistent members.

    Oracle: the documented contract — ``(symbols, args_symbols, input_symbols,
    nodes_exprs, output_exprs, forward_func, topo_order, useful_nodes)`` — checked
    structurally: ``len(tuple) == 8``; ``symbols`` maps every node key plus one raw
    ``i<k>`` entry per input; ``input_symbols`` is ``[symbols[-key-1] for key in
    get_input_idx()]``; every node expression is a closed form once
    ``args_symbols`` is substituted; ``output_exprs`` has ``num_outputs`` entries;
    ``topo_order`` and ``useful_nodes`` are the ones ``network_dict`` produced;
    ``forward_func`` is callable and maps a sequence of ``num_inputs`` values to
    ``num_outputs`` floats. Symbols are named by role (``i<k>`` raw, ``norm<k>``
    transformed, ``h<k>`` hidden, ``o<k>`` output), which is what the caller relies
    on to apply an input transform.
    """
    import numpy as np
    import sympy as sp

    g = payloads.default_genome(
        num_inputs=2, num_outputs=1, max_nodes=8, max_conns=16, max_in_degree=4
    )
    state = payloads.genome_state(g)
    payload = payloads.skip_payload()
    network = g.network_dict(
        state,
        genome.GenomeTensors(*payload.args()),
        whether_re_cound_idx=False,
    )

    result = g.sympy_func(state, network)
    assert len(result) == 8
    (
        symbols,
        args_symbols,
        input_symbols,
        nodes_exprs,
        output_exprs,
        forward_func,
        topo_order,
        useful_nodes,
    ) = result

    # Types of the eight members.
    assert isinstance(symbols, dict)
    assert isinstance(args_symbols, dict)
    assert isinstance(input_symbols, list)
    assert isinstance(nodes_exprs, dict)
    assert isinstance(output_exprs, list)
    assert callable(forward_func)
    assert isinstance(topo_order, list)
    assert isinstance(useful_nodes, set)
    assert all(isinstance(value, sp.Symbol) for value in symbols.values())
    assert all(isinstance(value, sp.Symbol) for value in input_symbols)
    assert all(isinstance(symbol, sp.Symbol) for symbol in args_symbols)
    assert all(isinstance(value, float) for value in args_symbols.values())

    # Symbol coverage: every gene of the payload, plus the raw ``i<k>`` entry that
    # the transformed ``norm<k>`` entry is built from.
    input_idx = g.get_input_idx()
    assert set(symbols) == set(network["nodes"]) | {-key - 1 for key in input_idx}
    assert len(input_symbols) == g.num_inputs
    assert input_symbols == [symbols[-key - 1] for key in input_idx]
    assert set(nodes_exprs) == set(symbols.values())

    # Names by role: raw/transformed inputs, then hidden keys in row order, then
    # outputs; ``skip_payload`` has hidden keys 2 and 3 and the single output 7.
    assert str(symbols[-1]) == "i0"
    assert str(symbols[-2]) == "i1"
    assert str(symbols[0]) == "norm0"
    assert str(symbols[1]) == "norm1"
    assert str(symbols[2]) == "h0"
    assert str(symbols[3]) == "h1"
    assert str(symbols[7]) == "o0"

    # The read-out is one closed expression per output: every gene value is bound in
    # ``args_symbols`` (the substitution ``common.sympy_tools`` performs), and the
    # connection weights are named by their endpoints.
    assert len(output_exprs) == g.num_outputs
    assert sp.Symbol("c_0_2_w") in args_symbols
    assert sp.Symbol("c_3_7_w") in args_symbols
    assert sp.Symbol("n_3_b") in args_symbols and sp.Symbol("n_3_r") in args_symbols
    for expression in output_exprs:
        assert isinstance(expression, sp.Expr)
        assert expression.free_symbols <= set(input_symbols) | set(args_symbols)
        assert expression.subs(args_symbols).free_symbols <= set(input_symbols)

    # The two graph utilities are the ones ``network_dict`` produced.
    assert topo_order == network["topo_order"]
    assert useful_nodes == network["useful_nodes"]

    # ``forward_func`` evaluates the symbolic network: ``num_inputs`` in,
    # ``num_outputs`` out, and finite (a sigmoid network is never NaN here).
    values = forward_func([0.3, -0.2])
    assert isinstance(values, np.ndarray)
    assert values.shape == (g.num_outputs,)
    assert np.all(np.isfinite(values))


def test_sympy_func_formula_of_a_tiny_network(
    genome, payloads, run_transform, run_forward
):
    """Invariant: a two-input/one-hidden/one-output network simplifies to ``i0 + i1``.

    Oracle: hand arithmetic. With identity activations everywhere (the documented
    ``-1`` activation id), unit weights, zero biases and unit responses, the hidden
    value is ``x0 + x1`` and the output row is the identity read-out of it, so
    ``output_exprs[0]`` — after the ``args_symbols`` substitution
    ``common.sympy_tools`` performs, with no simplification beyond it — must reduce
    to ``i0 + i1`` (``sympy.simplify(expr - (i0 + i1)) == 0``) and
    ``forward_func([1.0, 2.0])`` must return ``3.0``. The *same* network's
    ``DefaultGenome.forward`` must return ``3.0`` too, which is the one place the
    symbolic and the tensor paths are compared — through a number both can be
    checked against by hand.

    A second reading, on ``skip_payload``, pins the sigmoid branch of the same
    algebra: the hidden node aggregating both unit-weighted inputs is
    ``sigmoid(i0 + i1)``, and the output row — which the node gene never activates
    — is ``sigmoid(i0)`` because it reads the bypassed node fed by input ``0``.
    """
    import numpy as np
    import sympy as sp

    g = payloads.default_genome(
        num_inputs=2, num_outputs=1, max_nodes=8, max_conns=16, max_in_degree=4
    )
    state = payloads.genome_state(g)
    identity_row = [0.0, 1.0, 0.0, -1.0]  # bias, response, sum, identity activation
    payload = payloads.new_payload()
    for row, key in ((0, 0), (1, 1), (2, 2), (7, 7)):
        payloads.set_node(payload, 0, row, key, attrs=identity_row)
    payloads.set_conn(payload, 0, 0, 0, 2, attrs=[1.0])
    payloads.set_conn(payload, 0, 1, 1, 2, attrs=[1.0])
    payloads.set_conn(payload, 0, 2, 2, 7, attrs=[1.0])
    individual = genome.GenomeTensors(*payload.args())

    network = g.network_dict(state, individual, whether_re_cound_idx=False)
    # The precondition of the closed form: both non-output rows are identity.
    assert network["nodes"][2]["act"] == "identity"
    assert network["nodes"][2]["agg"] == "sum"
    assert network["nodes"][2]["bias"] == 0.0
    assert network["nodes"][2]["res"] == 1.0

    (
        _symbols,
        args_symbols,
        input_symbols,
        _nodes_exprs,
        output_exprs,
        forward_func,
        _topo_order,
        _useful_nodes,
    ) = g.sympy_func(state, network)

    i0, i1 = input_symbols
    expression = output_exprs[0].subs(args_symbols)
    assert sp.simplify(expression - (i0 + i1)) == 0
    assert float(forward_func([1.0, 2.0])[0]) == pytest.approx(3.0)

    # The tensors-only path must agree on the same number.
    transformed = run_transform(g, state, payload)
    outputs = run_forward(
        g,
        state,
        transformed,
        payload,
        np.asarray([[1.0, 2.0]], dtype=np.float32),
    )
    assert outputs.shape == (1, g.num_outputs)
    assert float(outputs[0, 0]) == pytest.approx(3.0)

    # A sigmoid row, also written out by hand: on ``skip_payload`` the hidden node 2
    # aggregates both unit-weighted inputs through ``sigmoid``, so it is
    # ``sigmoid(i0 + i1)``; the output row 7 reads the bypassed node 3 (fed only by
    # input 0) and — because the node gene never activates an output row — is
    # exactly ``sigmoid(i0)``, with no second activation layered on top.
    sigmoid_payload = payloads.skip_payload()
    sigmoid_network = g.network_dict(
        state,
        genome.GenomeTensors(*sigmoid_payload.args()),
        whether_re_cound_idx=False,
    )
    (
        sigmoid_symbols,
        sigmoid_args,
        sigmoid_inputs,
        sigmoid_nodes,
        sigmoid_outputs,
        _sigmoid_forward,
        _sigmoid_topo_order,
        _sigmoid_useful_nodes,
    ) = g.sympy_func(state, sigmoid_network)
    # ``norm<k>`` is the transformed input; this genome has no input transform, so
    # ``norm<k>`` is the raw ``i<k>`` (what ``common.sympy_tools`` substitutes).
    normalisation = {
        sigmoid_symbols[key]: sigmoid_symbols[-key - 1] for key in g.get_input_idx()
    }
    s0, s1 = sigmoid_inputs
    # The gene values arrive as floats (``1.0``), which sympy keeps apart from the
    # integers of the hand-derived form; ``nsimplify`` maps them back to the exact
    # rationals they denote before the difference is simplified.
    hidden = sp.nsimplify(
        sigmoid_nodes[sigmoid_symbols[2]].subs(sigmoid_args).subs(normalisation)
    )
    assert sp.simplify(hidden - 1 / (1 + sp.exp(-(s0 + s1)))) == 0
    output = sp.nsimplify(
        sigmoid_outputs[0].subs(sigmoid_args).subs(normalisation)
    )
    assert sp.simplify(output - 1 / (1 + sp.exp(-s0))) == 0


# --------------------------------------------------------------------------- #
# visualize
# --------------------------------------------------------------------------- #


def test_visualize_writes_a_figure_without_a_display(genome, payloads, tmp_path, monkeypatch):
    """Invariant: ``visualize`` writes ``save_path`` and imports its drawing stack lazily.

    Oracle: the file system — the path passed as ``save_path`` exists and is
    non-empty after the call — plus a ``sys.modules`` membership test taken *before*
    it, which must show no ``networkx``/``matplotlib``: the module docstring
    promises both are imported inside the method so ``import tensorneat`` never
    pulls them in. ``MPLBACKEND=Agg`` (and a writable ``MPLCONFIGDIR``) is set
    through ``monkeypatch`` so no display is needed, and the call is made twice to
    the same ``tmp_path`` with different drawing options; a spy on
    ``networkx.draw`` records what reached the drawing call, which proves the
    options are forwarded rather than ignored.
    """
    import sys

    monkeypatch.setenv("MPLBACKEND", "Agg")
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "mplconfig"))

    g = payloads.default_genome(
        num_inputs=2, num_outputs=1, max_nodes=8, max_conns=16, max_in_degree=4
    )
    state = payloads.genome_state(g)
    payload = payloads.skip_payload()
    network = g.network_dict(state, genome.GenomeTensors(*payload.args()))

    # The drawing stack is not imported by ``tensorneat`` (nor by this suite so
    # far); a module that imported it at module scope would break the promise.
    assert "networkx" not in sys.modules
    assert "matplotlib" not in sys.modules

    import networkx

    calls = []
    real_draw = networkx.draw

    def draw_spy(graph, **kwargs):
        calls.append(kwargs)
        return real_draw(graph, **kwargs)

    monkeypatch.setattr(networkx, "draw", draw_spy)

    default_figure = tmp_path / "network.svg"
    g.visualize(network, save_path=str(default_figure))
    assert default_figure.is_file()
    assert default_figure.stat().st_size > 0

    optioned_figure = tmp_path / "network_options.svg"
    g.visualize(
        network,
        save_path=str(optioned_figure),
        rotate=90,
        reverse_node_order=True,
        with_labels=True,
        size=(100, 200, 300),
        color=("red", "green", "blue"),
        save_dpi=72,
    )
    assert optioned_figure.is_file()
    assert optioned_figure.stat().st_size > 0
    assert "matplotlib" in sys.modules and "networkx" in sys.modules

    assert len(calls) == 2
    # Defaults of the documented signature, and the per-layer marker tables of the
    # second call (two inputs, two hidden nodes, one output): the options are used.
    assert calls[0]["with_labels"] is False
    assert calls[0]["arrowstyle"] == "->"
    assert calls[0]["node_size"] == [300, 300, 300, 300, 300]
    assert calls[0]["node_color"] == ["yellow", "yellow", "white", "white", "blue"]
    assert calls[1]["with_labels"] is True
    assert calls[1]["node_size"] == [100, 100, 200, 200, 300]
    assert calls[1]["node_color"] == ["red", "red", "green", "green", "blue"]
    # ``rotate``/``reverse_node_order`` rewrite the layout, not the draw kwargs.
    assert calls[1]["pos"] != calls[0]["pos"]


# --------------------------------------------------------------------------- #
# the recurrent refusal
# --------------------------------------------------------------------------- #


def test_recurrent_genome_has_no_symbolic_form_or_drawing(
    genome, common, payloads, tmp_path
):
    """Invariant: ``RecurrentGenome.sympy_func``/``visualize`` refuse with ``TensorNEATError``.

    Oracle: the documented refusal — a cyclic network has no finite closed form and
    no depth-ordered layout, so the rewrite keeps the legacy behaviour instead of
    emitting a truncated formula or a misleading picture. The refusal is a
    ``common.TensorNEATError`` (never a bare ``NotImplementedError``) whose message
    names the reason and points at ``network_dict()``/``DefaultGenome``, which is
    also how ``network_dict`` stays usable for a recurrent genome while its two
    downstream consumers do not.
    """
    g = payloads.recurrent_genome(
        num_inputs=2, num_outputs=1, max_nodes=8, max_conns=16, max_in_degree=4
    )
    state = payloads.genome_state(g)
    payload = payloads.ring_payload()
    network = g.network_dict(
        state,
        genome.GenomeTensors(*payload.args()),
        whether_re_cound_idx=False,
    )
    # The topology itself is still described: only the two consumers refuse.
    assert set(network) == {"nodes", "conns"}
    assert 2 in network["nodes"] and 3 in network["nodes"]

    with pytest.raises(common.TensorNEATError) as symbolic_error:
        g.sympy_func(state, network)
    assert not isinstance(symbolic_error.value, NotImplementedError)
    message = str(symbolic_error.value)
    assert "RecurrentGenome.sympy_func" in message
    assert "no finite closed form" in message
    assert "network_dict()" in message
    assert "DefaultGenome" in message

    save_path = tmp_path / "recurrent.svg"
    with pytest.raises(common.TensorNEATError) as drawing_error:
        g.visualize(network, save_path=str(save_path))
    assert not isinstance(drawing_error.value, NotImplementedError)
    message = str(drawing_error.value)
    assert "RecurrentGenome.visualize" in message
    assert "depth-ordered" in message
    assert "network_dict()" in message
    assert "DefaultGenome" in message
    assert not save_path.exists()
