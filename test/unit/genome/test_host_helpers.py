"""Unit tests for the host-only helpers: ``network_dict``, ``repr``, ``hash``,
``sympy_func`` and ``visualize``.

What this module proves
-----------------------
These five methods never enter a graph: they convert tensors with ``Tensor.numpy()``
and walk rows in Python, so their results must never be fed back into a kernel —
and, symmetrically, they must not be needed to *run* one. What they owe their
consumers is a stable, documented shape:

* ``network_dict`` → ``{"nodes": {key: node dict}, "conns": {(in_key, out_key):
  conn dict}}``, plus ``topo_order``/``topo_layers``/``useful_nodes`` for
  ``DefaultGenome`` (the three entries the code-generation and drawing paths read);
* ``repr`` → a header line with the gene counts, then one line per node (annotated
  ``(input)``/``(output)``) and one per connection;
* ``hash`` → an **in-graph** kernel (``(P,) uint32``), the one helper in this module
  that is traced: it is used for fast de-duplication and must be deterministic and
  structure-sensitive, while padding must hash identically;
* ``sympy_func`` → the legacy **8-tuple**
  ``(symbols, args_symbols, input_symbols, nodes_exprs, output_exprs, forward_func,
  topo_order, useful_nodes)``, with ``forward_func`` already substituting the gene
  values, so a caller can evaluate the symbolic network as numpy;
* ``visualize`` → a file, drawn with ``networkx``/``matplotlib`` imported *inside*
  the method.

Oracles used here (never the kernel itself)
-------------------------------------------
* one exact formula: on a two-input/one-hidden/one-output network with identity
  activations, unit weights, zero biases and unit responses, the output expression
  must simplify to ``i0 + i1`` (and ``forward_func([1.0, 2.0]) == 3.0``). It is
  written out by hand, so no part of the symbolic path is compared with itself;
* the payload's own key/row invariant (``support.payloads``), which lets a test
  name the keys it expects in ``network_dict`` without reading the implementation;
* set algebra for ``hash``: equal payloads hash equal, a one-attribute change or a
  different topology hashes differently, and two payloads that differ only in their
  padding hash equal.

Gotchas this module encodes
---------------------------
* ``import tensorneat.genome`` must **not** import ``networkx`` or ``matplotlib``:
  both are imported inside ``visualize`` for exactly that reason, and the release
  gate is only the loudest version of the rule. The check is a ``sys.modules``
  membership test before the drawing call, plus ``MPLBACKEND=Agg`` so the test
  needs no display.
* ``network_dict`` takes *one individual*, not a batch: a ``P == 1`` payload is
  fine, a ``P > 1`` one is a usage error (the method's documented contract), so the
  test passes ``payload.of(p)``.
* ``whether_re_cound_idx=True`` renumbers hidden keys to be contiguous above
  ``max(input, output keys)`` and rewrites the connection keys accordingly; the
  input/output keys are untouched because they are the genome's external
  interface. Both branches are asserted, since a caller that renumbers twice must
  not change the graph.
* ``RecurrentGenome`` deliberately refuses ``sympy_func`` and ``visualize``: a
  cyclic network has no finite closed form and no depth-ordered layout. Refusing is
  the documented behaviour, so the test asserts ``NotImplementedError`` instead of
  a degenerate result.
* ``hash`` is in-graph and therefore runs through ``support.runner.run_defn`` with
  the four payload arrays passed as data and ``GenomeTensors`` rebuilt inside the
  traced closure (the shared runner hands only top-level arrays to ``evoxc.build``;
  see ``./conftest.py``).

Backend is ``"numpy"`` throughout, through the shared runner only; no
``jax``/``torch``, no network, no GPU, no display.
"""

from __future__ import annotations

import pytest


# --------------------------------------------------------------------------- #
# network_dict
# --------------------------------------------------------------------------- #


def test_network_dict_returns_nodes_conns_and_the_feedforward_utilities(payloads):
    """Invariant: ``network_dict`` describes exactly the genes the payload holds.

    Oracle: the payload's key/row invariant — the node key set is
    ``set(node_keys[0][node_keys[0] != I_INF])`` and the connection key set is the
    present ``(in_key, out_key)`` pairs — plus the three feedforward entries
    ``topo_order`` (a permutation of the node keys), ``topo_layers`` (a partition
    of it whose level assignments agree with the schedule's ``level``) and
    ``useful_nodes`` (the keys that reach an output; the isolated node is *not* in
    it). Each node/conn dict is the gene's own ``to_dict`` output, so the test also
    checks the documented column names (``idx``/``bias``/``res``/``agg``/``act`` for
    nodes, ``in``/``out``/``weight`` for connections).
    """
    pytest.skip(
        "not implemented: network_dict holds the payload's genes plus "
        "topo_order/topo_layers/useful_nodes "
        "(owner: genome/default.DefaultGenome.network_dict, genome/base.BaseGenome.network_dict)"
    )


def test_network_dict_renumbers_hidden_keys_when_asked(payloads):
    """Invariant: ``whether_re_cound_idx`` renumbers hidden keys only, never the interface.

    Oracle: hand enumeration. With hidden keys ``{20, 41}`` and input/output keys
    ``{0, 1}``/``{7}``, the renumbered hidden keys must be ``{8, 9}`` (consecutive,
    starting at ``max(input, output) + 1``, in row order) and every connection key
    referring to them rewritten to match; the input and output keys are byte-identical
    in both branches, and passing ``whether_re_cound_idx=False`` returns the keys
    verbatim. Renumbering is a *display* transform, so the underlying payload must
    be untouched afterwards.
    """
    pytest.skip(
        "not implemented: re_cound_idx renumbers hidden keys and rewrites connection keys "
        "(owner: genome/base.BaseGenome.network_dict, genome/utils.re_cound_idx)"
    )


# --------------------------------------------------------------------------- #
# repr
# --------------------------------------------------------------------------- #


def test_repr_renders_a_header_and_one_line_per_gene(payloads):
    """Invariant: ``repr`` emits a counted header, then exactly one line per present gene.

    Oracle: the documented layout — a first line naming the class and the valid
    node/connection counts, then ``len(present nodes) + len(present conns)`` lines,
    each produced by the gene's own ``repr`` (so ``precision`` reaches the float
    columns), the node lines of input/output rows carrying the ``(input)``/
    ``(output)`` annotation. Padded rows must contribute no line at all — a
    ``NaN``-padded payload that printed its padding would make the output
    unreadable at ``max_nodes=50``.
    """
    pytest.skip(
        "not implemented: repr header + one line per present gene, padding skipped "
        "(owner: genome/base.BaseGenome.repr, genome/gene/node/default.DefaultNode.repr)"
    )


# --------------------------------------------------------------------------- #
# hash (the one traced helper)
# --------------------------------------------------------------------------- #


def test_hash_is_deterministic_and_distinguishes_different_topologies(
    genome, payloads, runner
):
    """Invariant: ``hash`` is a pure, structure-sensitive ``(P,) uint32`` kernel.

    Oracle: set algebra on the returned vector — the same payload hashed twice gives
    equal values; two payloads that differ in one connection key give different
    values; two payloads that differ only in their *padding rows* give equal values
    (padding quantises to one sentinel); and the dtype is ``uint32`` with one entry
    per individual. ``hash`` is in-graph, so it runs through ``run_defn`` with the
    four arrays as data and ``genome.GenomeTensors`` rebuilt inside the closure.
    """
    pytest.skip(
        "not implemented: hash is deterministic, padding-insensitive and structure-sensitive "
        "(owner: genome/base.BaseGenome.hash)"
    )


# --------------------------------------------------------------------------- #
# sympy_func
# --------------------------------------------------------------------------- #


def test_sympy_func_returns_the_documented_eight_tuple(payloads):
    """Invariant: ``sympy_func`` returns the legacy 8-tuple with consistent members.

    Oracle: the documented contract — ``(symbols, args_symbols, input_symbols,
    nodes_exprs, output_exprs, forward_func, topo_order, useful_nodes)`` — checked
    structurally: ``len(tuple) == 8``; ``symbols`` covers ``input_symbols`` plus one
    entry per useful node; ``output_exprs`` has ``num_outputs`` entries; ``topo_order``
    and ``useful_nodes`` are the ones ``network_dict`` produced; ``forward_func`` is
    callable and maps a sequence of ``num_inputs`` values to ``num_outputs`` floats.
    Symbols are named by role (``i<k>`` raw, ``norm<k>`` transformed, ``h<k>``
    hidden, ``o<k>`` output), which is what the caller relies on to apply an input
    transform.
    """
    pytest.skip(
        "not implemented: sympy_func returns the 8-tuple with the documented members "
        "(owner: genome/default.DefaultGenome.sympy_func)"
    )


def test_sympy_func_formula_of_a_tiny_network(payloads):
    """Invariant: a two-input/one-hidden/one-output network simplifies to ``i0 + i1``.

    Oracle: hand arithmetic. With identity activations everywhere, unit weights,
    zero biases and unit responses, the hidden value is ``x0 + x1`` and the output
    row is the identity read-out of it, so ``output_exprs[0]`` must simplify to
    ``i0 + i1`` (``sympy.simplify(expr - (i0 + i1)) == 0``) and
    ``forward_func([1.0, 2.0])`` must return ``3.0``. The *same* network's
    ``DefaultGenome.forward`` must return ``3.0`` too, which is the one place the
    symbolic and the tensor paths are compared — through a number both can be
    checked against by hand.
    """
    pytest.skip(
        "not implemented: the symbolic form of a tiny identity network is i0 + i1 "
        "(owner: genome/default.DefaultGenome.sympy_func, "
        "genome/gene/node/default.DefaultNode.sympy_func)"
    )


# --------------------------------------------------------------------------- #
# visualize
# --------------------------------------------------------------------------- #


def test_visualize_writes_a_figure_without_a_display(
    genome, payloads, common, tmp_path, monkeypatch
):
    """Invariant: ``visualize`` writes ``save_path`` and imports its drawing stack lazily.

    Oracle: the file system — the path passed as ``save_path`` exists and is
    non-empty after the call — plus a ``sys.modules`` membership test taken *before*
    it, which must show no ``networkx``/``matplotlib``: the module docstring promises
    both are imported inside the method so ``import tensorneat`` never pulls them in.
    ``MPLBACKEND=Agg`` is set through ``monkeypatch`` so no display is needed, and
    the call is made twice to the same ``tmp_path`` with different drawing options
    to prove the arguments are forwarded rather than ignored.
    """
    pytest.skip(
        "not implemented: visualize saves the figure, drawing stack imported lazily "
        "(owner: genome/default.DefaultGenome.visualize)"
    )


# --------------------------------------------------------------------------- #
# the recurrent refusal
# --------------------------------------------------------------------------- #


def test_recurrent_genome_has_no_symbolic_form_or_drawing(genome, payloads):
    """Invariant: ``RecurrentGenome.sympy_func``/``visualize`` raise ``NotImplementedError``.

    Oracle: the documented refusal — a cyclic network has no finite closed form and
    no depth-ordered layout, so the rewrite keeps the legacy behaviour instead of
    emitting a truncated formula or a misleading picture. The test asserts the
    error type (``NotImplementedError``, not a silently degenerate result), which is
    also how ``network_dict`` stays usable for a recurrent genome while its two
    downstream consumers do not.
    """
    pytest.skip(
        "not implemented: recurrent sympy_func/visualize refuse with NotImplementedError "
        "(owner: genome/recurrent.RecurrentGenome.sympy_func, visualize)"
    )
