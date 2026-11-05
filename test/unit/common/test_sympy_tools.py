"""Unit tests for ``tensorneat.common.sympy_tools`` — the two host-side renderers.

What this module proves
-----------------------
``Genome.sympy_func`` turns one evolved network into an 8-tuple of sympy
objects, and this leaf module turns that tuple into the two public artifacts:
a **runnable NumPy snippet** (:func:`~tensorneat.common.sympy_tools.to_python_code`)
and a **LaTeX ``align`` block** (:func:`~tensorneat.common.sympy_tools.to_latex_code`).
Both renderers had almost no coverage before this module (``to_latex_code`` had
none at all), so the tests are written against the *whole shipped option table*:

* every one of the 12 ``ACT`` options and every one of the 6 ``AGG`` options is
  rendered on a real genome, and the emitted snippet is ``exec``-ed and matched
  to the evoxc graph forward of the **same payload** — that forward/codegen
  equivalence is the promise the module makes;
* every option's LaTeX block is checked structurally (balanced ``align``
  environment, one ``& =`` assignment per useful non-input node, the symbol
  names present) — no LaTeX toolchain is available and none is needed;
* ``input_symbols`` drives the documented ``#     i<k> -> i[<position>]``
  binding header; ``output_exprs`` drives the output-slot allocation and the
  closed-form comment block, with the empty-list fallback counting the
  ``o<k>`` keys instead;
* a node outside ``useful_nodes`` is skipped by **both** renderers — the
  documented NaN-safety rule, because a useless node's expression may be
  ``sp.nan``;
* a sympy function NumPy cannot evaluate raises
  :class:`tensorneat.common.errors.ConfigError` naming the culprit, never a
  snippet with an undefined name.

Oracles
-------
The authority is the **evoxc graph forward** on the payload the tuple was built
from (one ``run_defn`` per ``(agg, act)`` pair, cached in the module-scoped
harness).  As a second, independent composition the tests also assert the value
of :mod:`support.functions_ref`, whose kernels are a float64 transcription of
the documented option definitions and never call the kernels under test.  The
renderer-contract tests use hand-built 8-tuples (the shape the docstrings
document) and plain Python equality on the emitted strings.

Gotchas this module encodes
---------------------------
* The hidden node is the **only activated node**: ``DefaultNode.sympy_func``
  leaves output nodes unactivated (``is_output_node=True``), so the ACT id
  belongs on the hidden row.  With two live incoming edges the AGG id is
  genuinely exercised there, and the AGG cases use the ``identity`` activation
  so no nonlinearity masks the aggregation.
* ``to_python_code`` rounds every constant at ``precision=6`` and
  ``to_latex_code`` at ``precision=2``; that is why the numeric comparison
  allows ``atol=1e-5`` (measured worst case here is ~2.5e-7, on ``log``) and
  why the LaTeX block shows ``0.12`` for a bias of ``0.125``.
* ``round_expr`` never rounds a nonzero atom to zero, so ``inv``/``log`` keep
  their exact ``1e-7`` clamps in both renderers.
* ``SympyExp`` clips its argument to ``[-EXP_CLIP, EXP_CLIP]`` while the numeric
  kernel does not; the sample below stays inside the band, so the documented
  divergence is not asserted here (``test_functions.py`` owns it).
* ``SympyMaxabs`` is **not** broken: it renders as an ``np.where``/``np.abs``/
  ``np.maximum`` chain that ``exec``-s and agrees with the graph.  This module
  pins that, because a regression there would otherwise surface only as an
  undefined name in a notebook.
* ``input_symbols`` is read for the binding *header* only; the body's
  ``i[<k>]`` mapping is fixed by the ``i<k>`` symbol names, so reordering the
  list rewrites the documentation and leaves the expression untouched (still
  runnable).
* ``output_exprs`` is the authority for the allocation width
  (``o = np.zeros(len(output_exprs))``) and the trailing comment block, not for
  the assignment: the body assigns each node from ``nodes_exprs``.  A custom
  output transform therefore reaches the number through ``nodes_exprs`` and the
  comment through ``output_exprs`` — both are asserted.

Backend is ``"numpy"``; ``sympy`` is imported through a fixture, ``jax``/``iree``
are never imported, and the shapes stay small (one individual, four node rows).
"""

from __future__ import annotations

import numpy as np
import pytest

from support import functions_ref
from support.runner import run_defn

#: Canonical ACT option order (a node gene stores the *position* in this table).
ACT_OPTIONS = (
    "scaled_sigmoid",
    "sigmoid",
    "scaled_tanh",
    "tanh",
    "sin",
    "relu",
    "lelu",
    "identity",
    "inv",
    "log",
    "exp",
    "abs",
)
#: Canonical AGG option order (aggregation has no ``-1`` identity sentinel).
AGG_OPTIONS = ("sum", "product", "max", "min", "maxabs", "mean")

#: ``sum`` — the aggregation of the ACT cases, so the ACT id is the only knob read.
SUM_AGG_ID = AGG_OPTIONS.index("sum")
#: ``identity`` — the activation of the AGG cases, so no nonlinearity masks the aggregation.
IDENTITY_ACT_ID = ACT_OPTIONS.index("identity")

#: Row and key of the hidden node in the harness payload (node key == row index).
HIDDEN_ROW = 2
HIDDEN_KEY = 2
#: :class:`~tensorneat.genome.gene.node.DefaultNode` attribute columns of that row.
HIDDEN_BIAS = 0.125
HIDDEN_RESPONSE = 1.0
#: Raw inputs, one per input node: a mixed pair so ``max``/``min``/``maxabs`` differ.
INPUTS = np.array([[0.5, -0.75]], dtype=np.float32)

#: Absolute tolerance of the codegen↔graph comparison.  The wired constants are
#: rounded at ``precision=6`` and the graph runs in float32, so the observed
#: worst case is ~2.5e-7 (``log`` against its ``1e-7`` clamp); 1e-5 is the
#: documented budget of the renderer.
ATOL = 1e-5
RTOL = 1e-5


# --------------------------------------------------------------------------- #
# fixtures: one genome, one payload family, cached renders
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def sympy():
    """The host-only ``sympy`` module the hand-built 8-tuples are written with."""
    import sympy as module

    return module


@pytest.fixture(scope="module")
def sympy_tools():
    """The module under test (``tensorneat.common.sympy_tools``), imported lazily."""
    import tensorneat.common.sympy_tools as module

    return module


class _Harness:
    """A two-input / one-hidden / one-output genome plus its cached renders.

    Topology (node key == row index): ``0 -> 2``, ``1 -> 2``, ``2 -> 3``, built
    by :func:`support.payloads.chain_payload`.  Row 2 is the only activated node
    — row 3 is an output row, which ``DefaultNode.sympy_func`` renders
    unactivated — so the ACT id lives there and the two live incoming edges make
    the AGG id observable.

    Every cache is keyed by ``(agg_id, act_id)`` so the parametrized numeric and
    LaTeX cases share one ``run_defn`` per pair instead of one per test.
    """

    def __init__(self, genome, state, payloads, tools):
        self.genome = genome
        self.state = state
        self.payloads = payloads
        self.tools = tools
        self.inputs = INPUTS
        self._payloads = {}
        self._networks = {}
        self._forwards = {}
        self._tuples = {}
        self._renders = {}

    def payload(self, agg_id, act_id):
        """The payload whose hidden row selects ``agg_id`` and ``act_id`` (cached)."""
        key = (agg_id, act_id)
        if key not in self._payloads:
            payload = self.payloads.chain_payload(
                1, 4, 8, num_inputs=2, num_outputs=1
            )
            self.payloads.set_node(
                payload,
                0,
                HIDDEN_ROW,
                HIDDEN_KEY,
                [HIDDEN_BIAS, HIDDEN_RESPONSE, agg_id, act_id],
            )
            self._payloads[key] = payload
        return self._payloads[key]

    def network(self, agg_id, act_id):
        """The host graph dict of that payload (``network_dict``, cached)."""
        key = (agg_id, act_id)
        if key not in self._networks:
            self._networks[key] = self.genome.network_dict(
                self.state,
                self.payload(agg_id, act_id).args(),
                whether_re_cound_idx=False,
            )
        return self._networks[key]

    def forward(self, agg_id, act_id):
        """The evoxc graph forward of that payload, as a Python float (cached).

        One traced closure does schedule + forward in a single ``run_defn``,
        exactly as the numeric suites do; this is the authority every emitted
        number is compared against.
        """
        key = (agg_id, act_id)
        if key not in self._forwards:
            genome, state, inputs = self.genome, self.state, self.inputs

            def graph(node_keys, node_attrs, conn_keys, conn_attrs, values):
                transformed = genome.transform(state, node_keys, conn_keys)
                return genome.forward(
                    state,
                    transformed,
                    node_keys,
                    node_attrs,
                    conn_keys,
                    conn_attrs,
                    values,
                )

            payload = self.payload(agg_id, act_id)
            result = run_defn(graph, *payload.args(), inputs)
            self._forwards[key] = float(np.asarray(result).reshape(-1)[0])
        return self._forwards[key]

    def codegen(self, agg_id, act_id):
        """The legacy 8-tuple of :meth:`Genome.sympy_func` for that payload (cached)."""
        key = (agg_id, act_id)
        if key not in self._tuples:
            self._tuples[key] = self.genome.sympy_func(
                self.state, self.network(agg_id, act_id)
            )
        return self._tuples[key]

    def render(self, agg_id, act_id):
        """``(python_source, latex_block)`` of that payload (cached together)."""
        key = (agg_id, act_id)
        if key not in self._renders:
            t8 = self.codegen(agg_id, act_id)
            self._renders[key] = (
                self.tools.to_python_code(*t8),
                self.tools.to_latex_code(*t8),
            )
        return self._renders[key]

    def execute(self, source, values=None):
        """``exec`` an emitted snippet and return its ``o`` array as float64."""
        return _execute(source, self.inputs[0] if values is None else values)


@pytest.fixture(scope="module")
def harness(payloads, sympy_tools):
    """The shared genome: ``ACT.evoxc_fns`` / ``AGG.evoxc_fns`` and the tiny payload family.

    ``DefaultNode`` defaults to a *one-option* table for each kind, which is why
    the full ``ACT``/``AGG`` tuples are passed explicitly: the option index in a
    gene column is a position in the genome's own list, and the renderers
    resolve the resulting *name* through the global registry.
    """
    from tensorneat.common.functions import ACT, AGG
    from tensorneat.genome import (
        DefaultConn,
        DefaultCrossover,
        DefaultDistance,
        DefaultGenome,
        DefaultMutation,
        DefaultNode,
    )

    genome = DefaultGenome(
        num_inputs=2,
        num_outputs=1,
        max_nodes=4,
        max_conns=8,
        node_gene=DefaultNode(
            activation_options=ACT.evoxc_fns,
            aggregation_options=AGG.evoxc_fns,
        ),
        conn_gene=DefaultConn(),
        mutation=DefaultMutation(),
        crossover=DefaultCrossover(),
        distance=DefaultDistance(),
    )
    state = payloads.genome_state(genome)
    return _Harness(genome, state, payloads, sympy_tools)


# --------------------------------------------------------------------------- #
# helpers for the hand-built 8-tuples and the emitted snippets
# --------------------------------------------------------------------------- #
def _execute(source, values):
    """``exec`` an emitted snippet with ``np`` and ``i`` bound, returning ``o``.

    Args:
        source (str): The snippet :func:`~tensorneat.common.sympy_tools.to_python_code`
            produced.
        values: One raw input per input node, in input order.

    Returns:
        numpy.ndarray: The snippet's ``o`` array as ``float64``.

    Notes:
        Executing generated source is the contract under test; the ``noqa`` is
        for the linter, not a safety exception.
    """
    namespace = {"np": np, "i": np.asarray(values, dtype=np.float64)}
    exec(source, namespace)  # noqa: S102 - executing generated source is the point
    return np.asarray(namespace["o"], dtype=np.float64)


def _replace(t8, index, value):
    """Return the 8-tuple with position ``index`` replaced by ``value``."""
    fields = list(t8)
    fields[index] = value
    return tuple(fields)


def _hand_built(sp):
    """The reference hand-built 8-tuple: two inputs, one useful hidden, one useless ``NaN`` node.

    Row layout: ``0 -> i0``, ``1 -> i1``, ``2 -> h0 = i0 + i1``, ``3 -> o0 = h0``,
    ``4 -> h1 = NaN``.  ``useful_nodes = {2, 3}`` is what keeps the ``NaN``
    expression out of both renderers (the documented NaN-safety skip).
    """
    i0, i1 = sp.symbols("i0 i1")
    h0, h1 = sp.symbols("h0 h1")
    o0 = sp.Symbol("o0")
    symbols = {0: i0, 1: i1, 2: h0, 3: o0, 4: h1}
    nodes_exprs = {
        i0: i0,
        i1: i1,
        h0: i0 + i1,
        o0: h0,
        h1: sp.nan,
    }
    return (
        symbols,
        {},
        [i0, i1],
        nodes_exprs,
        [h0],
        None,
        [0, 1, 2, 3, 4],
        {2, 3},
    )


def _single_output(sp):
    """A minimal hand-built 8-tuple: one input, one hidden node, one output slot."""
    i0 = sp.Symbol("i0")
    h0 = sp.Symbol("h0")
    o0 = sp.Symbol("o0")
    symbols = {0: i0, 1: h0, 2: o0}
    nodes_exprs = {i0: i0, h0: 2 * i0, o0: h0}
    return (symbols, {}, [i0], nodes_exprs, [h0], None, [0, 1, 2], {1, 2})


def _two_outputs(sp):
    """A hand-built 8-tuple with two ``o<k>`` keys and no closed-form list."""
    i0 = sp.Symbol("i0")
    o0 = sp.Symbol("o0")
    o1 = sp.Symbol("o1")
    symbols = {0: i0, 1: o0, 2: o1}
    nodes_exprs = {i0: i0, o0: 2 * i0, o1: 3 * i0}
    return (symbols, {}, [i0], nodes_exprs, [], None, [0, 1, 2], {1, 2})


def _assert_latex_block(latex, symbols):
    """Structural invariants of a rendered LaTeX ``align`` block.

    No LaTeX toolchain is available, so "plausibly LaTeX" is asserted from the
    text itself: one balanced ``align`` environment, exactly one ``& =``
    assignment per expected non-input symbol, and one ``\\newline`` per
    assignment.
    """
    assert isinstance(latex, str) and latex, "to_latex_code returned no text"
    assert latex.startswith("\\begin{align}\n"), latex[:40]
    assert latex.rstrip("\n").endswith("\\end{align}"), latex[-40:]
    assert latex.count("\\begin{align}") == 1, "the environment must not nest"
    assert latex.count("\\end{align}") == 1, "the environment must be balanced"
    for symbol in symbols:
        assert f"{symbol} &=" in latex, f"{symbol} missing from {latex!r}"
    assert latex.count("&=") == len(symbols), latex
    assert latex.count("\\newline") == len(symbols), latex


# --------------------------------------------------------------------------- #
# ACT options: runnable source and LaTeX for all twelve
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("act_id", range(len(ACT_OPTIONS)), ids=ACT_OPTIONS)
def test_act_option_emits_runnable_source_matching_the_graph_forward(harness, act_id):
    """Invariant: for every ACT option, the snippet reproduces the graph forward.

    Oracle: the evoxc forward of the *same* payload (the authority), plus
    :func:`support.functions_ref.ACT_FUNCS` applied to the hand-computed
    pre-activation ``bias + response * sum(inputs)`` — two independent
    compositions of the option, neither of which calls the renderer.  The
    emitted ``o[0]`` is read back by ``exec``-ing the snippet with ``np`` and
    ``i`` bound, so a source that merely *looks* right fails.
    """
    source, _latex = harness.render(SUM_AGG_ID, act_id)
    expected_forward = harness.forward(SUM_AGG_ID, act_id)

    pre_activation = HIDDEN_BIAS + HIDDEN_RESPONSE * float(np.sum(INPUTS[0]))
    expected_reference = float(
        functions_ref.ACT_FUNCS[act_id](np.array([pre_activation]))[0]
    )

    generated = harness.execute(source)
    assert generated.shape == (1,), "the snippet must allocate the one output slot"
    assert not np.any(np.isnan(generated)), "no option may emit a NaN for these inputs"

    assert source.startswith("import numpy as np\n"), source.splitlines()[0]
    assert "numpy." not in source, "every NumPy spelling must use the np alias"
    assert "h[0] =" in source, "the hidden assignment must be present"
    assert "o[0] =" in source, "the output assignment must be present"

    np.testing.assert_allclose(
        generated[0],
        expected_forward,
        rtol=RTOL,
        atol=ATOL,
        err_msg=f"generated source disagrees with the graph forward for ACT '{ACT_OPTIONS[act_id]}'",
    )
    np.testing.assert_allclose(
        generated[0],
        expected_reference,
        rtol=RTOL,
        atol=ATOL,
        err_msg=f"generated source disagrees with support.functions_ref for ACT '{ACT_OPTIONS[act_id]}'",
    )


@pytest.mark.parametrize("act_id", range(len(ACT_OPTIONS)), ids=ACT_OPTIONS)
def test_latex_block_for_every_act_option(harness, act_id):
    """Invariant: every ACT option renders a balanced ``align`` block with its nodes.

    Oracle: the structural contract of :func:`to_latex_code` — one balanced
    environment and one ``& =`` line per useful non-input node (``h_{0}`` and
    ``o_{0}`` here; input nodes are never rendered).  ``to_latex_code`` rounds at
    ``precision=2``, which is why the bias renders as ``0.12``.
    """
    _source, latex = harness.render(SUM_AGG_ID, act_id)
    _assert_latex_block(latex, ("h_{0}", "o_{0}"))
    assert "i_{0}" in latex and "i_{1}" in latex, "the input symbols must be decorated"


# --------------------------------------------------------------------------- #
# AGG options: runnable source and LaTeX for all six
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("agg_id", range(len(AGG_OPTIONS)), ids=AGG_OPTIONS)
def test_agg_option_emits_runnable_source_matching_the_graph_forward(harness, agg_id):
    """Invariant: for every AGG option, the snippet reproduces the graph forward.

    The hidden node uses the ``identity`` activation so the aggregation is not
    masked by a nonlinearity, and the output node is unactivated by design, so
    the emitted number *is* ``bias + response * agg(inputs)``.  Oracles: the
    evoxc forward of the same payload (authority) and
    :func:`support.functions_ref.AGG_FUNCS` on the same two-term window — which
    is also what makes ``maxabs`` a *signed* selection (``-0.75`` here, not the
    magnitude).
    """
    source, _latex = harness.render(agg_id, IDENTITY_ACT_ID)
    expected_forward = harness.forward(agg_id, IDENTITY_ACT_ID)

    window = np.asarray(INPUTS, dtype=np.float64).reshape(1, 1, -1)
    mask = np.ones_like(window, dtype=bool)
    aggregated = float(functions_ref.AGG_FUNCS[agg_id](window, mask).reshape(-1)[0])
    expected_reference = HIDDEN_BIAS + HIDDEN_RESPONSE * aggregated

    generated = harness.execute(source)
    assert generated.shape == (1,)
    assert not np.any(np.isnan(generated))
    assert "numpy." not in source, "every NumPy spelling must use the np alias"
    assert "h[0] =" in source
    assert "o[0] =" in source

    np.testing.assert_allclose(
        generated[0],
        expected_forward,
        rtol=RTOL,
        atol=ATOL,
        err_msg=f"generated source disagrees with the graph forward for AGG '{AGG_OPTIONS[agg_id]}'",
    )
    np.testing.assert_allclose(
        generated[0],
        expected_reference,
        rtol=RTOL,
        atol=ATOL,
        err_msg=f"generated source disagrees with support.functions_ref for AGG '{AGG_OPTIONS[agg_id]}'",
    )


@pytest.mark.parametrize("agg_id", range(len(AGG_OPTIONS)), ids=AGG_OPTIONS)
def test_latex_block_for_every_agg_option(harness, agg_id):
    """Invariant: every AGG option renders a balanced ``align`` block with its nodes.

    Oracle: the same structural contract as the ACT half.  ``maxabs`` is the
    interesting case — it renders a ``\\begin{cases}`` chain (a Piecewise, not
    an unsupported ``sp.Max(..., key=Abs)``), which must stay inside the one
    ``align`` environment.
    """
    _source, latex = harness.render(agg_id, IDENTITY_ACT_ID)
    _assert_latex_block(latex, ("h_{0}", "o_{0}"))
    if AGG_OPTIONS[agg_id] == "maxabs":
        assert "\\begin{cases}" in latex, (
            "maxabs must render its signed selection as a Piecewise branch"
        )
    else:
        assert "\\begin{cases}" not in latex, (
            "only the maxabs aggregation renders a Piecewise branch"
        )


# --------------------------------------------------------------------------- #
# input_symbols: the documented binding header
# --------------------------------------------------------------------------- #
def test_input_symbols_drives_the_binding_header(harness, sympy_tools):
    """Invariant: the header names exactly the declared input symbols, in order.

    Oracle: the documented header spelling (``#     i<k> -> i[<position>]``) on
    a real tuple, then on the same tuple with ``input_symbols`` reversed.  The
    body is a function of the ``i<k>`` *symbol names*, so reordering the list
    must rewrite the header and leave the snippet runnable and numerically
    unchanged — the documented gotcha of the module.  With an empty list the
    fallback collects the referenced ``i<k>`` symbols by index, recovering the
    same header.
    """
    t8 = harness.codegen(SUM_AGG_ID, IDENTITY_ACT_ID)
    source = sympy_tools.to_python_code(*t8)

    assert "# `i` holds the raw inputs" in source
    assert "#     i0 -> i[0]" in source
    assert "#     i1 -> i[1]" in source
    assert source.index("#     i0 -> i[0]") < source.index("#     i1 -> i[1]")
    base_result = harness.execute(source)
    assert base_result.shape == (1,)

    swapped = _replace(t8, 2, list(t8[2])[::-1])
    swapped_source = sympy_tools.to_python_code(*swapped)
    assert "#     i1 -> i[0]" in swapped_source
    assert "#     i0 -> i[1]" in swapped_source
    assert swapped_source.index("#     i1 -> i[0]") < swapped_source.index(
        "#     i0 -> i[1]"
    )
    # The header is documentation: the body's ``i[k]`` mapping is fixed by the
    # ``i<k>`` symbol names, so swapping the list leaves the number unchanged.
    np.testing.assert_array_equal(harness.execute(swapped_source), base_result)

    fallback = sympy_tools.to_python_code(*_replace(t8, 2, []))
    assert "#     i0 -> i[0]" in fallback
    assert "#     i1 -> i[1]" in fallback
    np.testing.assert_array_equal(harness.execute(fallback), base_result)


# --------------------------------------------------------------------------- #
# output_exprs: the allocation width and the closed-form comment block
# --------------------------------------------------------------------------- #
def test_output_exprs_sets_the_output_count(sympy, sympy_tools):
    """Invariant: ``o = np.zeros(len(output_exprs))``, not the number of ``o<k>`` keys.

    Oracle: a hand-built tuple with a *single* ``o0`` key but a two-entry
    ``output_exprs`` — the allocation follows the list (``np.zeros(2)``), the
    body assigns only ``o[0]``, and ``o[1]`` keeps its zero slot.  The count
    coming from the ``o<k>`` keys instead would allocate one slot and fail.
    """
    t8 = _single_output(sympy)
    base_source = sympy_tools.to_python_code(*t8)
    assert "o = np.zeros(1)" in base_source
    assert "# o[0] =" in base_source

    widened = _replace(t8, 4, [t8[4][0], 2 * t8[4][0]])
    widened_source = sympy_tools.to_python_code(*widened)
    assert "o = np.zeros(2)" in widened_source, (
        "the output count must be read from len(output_exprs)"
    )
    assert "# o[0] =" in widened_source and "# o[1] =" in widened_source

    result = _execute(widened_source, [0.5])
    assert result.shape == (2,)
    np.testing.assert_allclose(result[0], 1.0, rtol=RTOL, atol=ATOL)
    assert result[1] == 0.0, "an unassigned output slot keeps its zero"


def test_output_exprs_fallback_counts_the_o_keys(sympy, sympy_tools):
    """Invariant: with ``output_exprs=[]`` the count falls back to the ``o<k>`` keys.

    Oracle: a hand-built tuple with two output nodes and no closed-form list —
    ``o = np.zeros(2)`` from the keys, both slots assigned from ``nodes_exprs``,
    and no closed-form comment block (there is nothing to repeat).
    """
    source = sympy_tools.to_python_code(*_two_outputs(sympy))
    assert "o = np.zeros(2)" in source
    assert "# o[0] =" not in source, "an empty output_exprs emits no comment block"

    result = _execute(source, [4.0])
    np.testing.assert_allclose(result, [8.0, 12.0], rtol=RTOL, atol=ATOL)


def test_custom_output_expr_changes_the_generated_result(harness, sympy_tools):
    """Invariant: a custom output expression reaches the number and the comment.

    Oracle: the same genome rendered with ``sympy_output_transform=lambda e:
    2 * e``.  ``Genome.sympy_func`` writes the transformed expression into the
    output node's ``nodes_exprs`` entry *and* into ``output_exprs``, so the
    snippet's ``o[0]`` doubles with respect to the untransformed render and the
    closed-form comment repeats the doubled formula.
    """
    network = harness.network(SUM_AGG_ID, IDENTITY_ACT_ID)
    base = harness.execute(harness.render(SUM_AGG_ID, IDENTITY_ACT_ID)[0])

    doubled_tuple = harness.genome.sympy_func(
        harness.state, network, sympy_output_transform=lambda expression: 2 * expression
    )
    doubled_source = sympy_tools.to_python_code(*doubled_tuple)
    doubled = harness.execute(doubled_source)

    assert doubled_source.count("# o[0] =") == 1
    assert doubled_source != harness.render(SUM_AGG_ID, IDENTITY_ACT_ID)[0]
    np.testing.assert_allclose(
        doubled[0],
        2.0 * base[0],
        rtol=RTOL,
        atol=ATOL,
        err_msg="a doubled output transform must double the emitted result",
    )


# --------------------------------------------------------------------------- #
# useful_nodes: the NaN-safety skip in both renderers
# --------------------------------------------------------------------------- #
def test_useless_nodes_are_skipped_by_both_renderers(harness, sympy):
    """Invariant: omitting a hidden node drops its assignment line everywhere.

    Oracle: the real tuple with the hidden node removed from ``useful_nodes``,
    and a hand-built tuple whose omitted node's expression is ``sp.nan``.  No
    ``nan`` spelling may reach either renderer, and the remaining assignment
    lines must still be there — a useless node need not be computable.
    """
    t8 = harness.codegen(SUM_AGG_ID, IDENTITY_ACT_ID)
    trimmed = _replace(t8, 7, t8[7] - {HIDDEN_KEY})
    trimmed_source = harness.tools.to_python_code(*trimmed)
    trimmed_latex = harness.tools.to_latex_code(*trimmed)
    assert "h[0] =" not in trimmed_source, "the omitted hidden node must not be assigned"
    assert "h_{0} &=" not in trimmed_latex, "the LaTeX line must be skipped too"
    assert "o[0] =" in trimmed_source, "the useful output node must stay"
    assert "o_{0} &=" in trimmed_latex

    hand = _hand_built(sympy)
    source = harness.tools.to_python_code(*hand)
    latex = harness.tools.to_latex_code(*hand)
    assert "nan" not in source.lower(), "a useless node's NaN expression must not reach the source"
    assert "h[1]" not in source
    assert "h_{1}" not in latex
    assert "h[0] =" in source and "o[0] =" in source
    assert "h_{0} &=" in latex and "o_{0} &=" in latex


# --------------------------------------------------------------------------- #
# error path: an unrenderable sympy function names the culprit
# --------------------------------------------------------------------------- #
def test_unrenderable_function_raises_config_error(sympy, sympy_tools):
    """Invariant: a sympy function NumPy lacks raises ``ConfigError`` by name.

    Oracle: ``sp.Function("Frobnicate")`` inside a hand-built node expression.
    The renderer must raise :class:`~tensorneat.common.errors.ConfigError`
    naming both the sympy function and the missing NumPy callable — never emit
    source that fails later with a bare ``NameError``.
    """
    from tensorneat.common.errors import ConfigError

    i0 = sympy.Symbol("i0")
    h0 = sympy.Symbol("h0")
    o0 = sympy.Symbol("o0")
    frobnicate = sympy.Function("Frobnicate")
    nodes_exprs = {
        i0: i0,
        h0: frobnicate(i0),
        o0: h0,
    }
    t8 = ({0: i0, 1: h0, 2: o0}, {}, [i0], nodes_exprs, [h0], None, [0, 1, 2], {1, 2})

    with pytest.raises(ConfigError) as caught:
        sympy_tools.to_python_code(*t8)

    message = str(caught.value)
    assert "Frobnicate" in message, message
    assert "frobnicate" in message, message
    assert "numpy has no callable" in message, message
