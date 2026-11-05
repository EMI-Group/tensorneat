"""Host-only sympy code generation for the interpretability features.

Dependency DAG position: leaf of ``tensorneat.common``; imports only
``sympy``, the standard library and :mod:`.errors` (no ``evoxc``, no NumPy at
import time, no ``jax``/``torch``), and nothing from
``genome``/``algorithm``/``problem``.

Role: :meth:`Genome.sympy_func` returns an 8-tuple of host-side objects
(``symbols, args_symbols, input_symbols, nodes_exprs, output_exprs,
forward_func, topo_order, useful_nodes``) built from a sympy translation of a
genome's nodes and connections. The functions here turn that 8-tuple into
human-readable output — a LaTeX ``align`` block for papers and a genuinely
runnable NumPy snippet for notebooks — and never participate in a generation, a
trace, or a fitness evaluation.

Contract kept from the pre-rewrite library (parity): the 8-tuple parameter
order and the ``o0``/``h1``/``i2`` node-symbol naming are unchanged, so
existing tutorials and stored expressions keep working.
"""

import re

import sympy as sp  # host-only; never imported from a traced kernel
from sympy.printing.numpy import NumPyPrinter

from .errors import ConfigError

_INPUT_SYMBOL = re.compile(r"^i(\d+)$")
"""Matches a raw input node symbol (``i0``, ``i1``, ...); see :func:`Genome.sympy_func`."""

_NUMPY_ALIAS = "np"
"""The module name the emitted snippet binds NumPy to."""


def analysis_nodes_exprs(nodes_exprs):
    """Classify node expressions into input/hidden/output counts.

    Args:
        nodes_exprs (dict): Mapping of sympy symbol to the expression that
            computes that node. Keys are named ``i<j>`` (inputs), ``h<j>``
            (hidden nodes), ``o<j>`` (outputs) and ``norm<...>`` (substituted
            normalization constants that must not be counted as nodes).

    Returns:
        tuple: ``(input_cnt, hidden_cnt, output_cnt, norm_symbols)`` where the
        three counts are ``int`` and ``norm_symbols`` is a ``dict`` of the
        ``norm*`` entries (symbol → expression) to be substituted away.

    Raises:
        None: Host-only; a key that starts with none of those prefixes is
        simply not counted (it is neither a node nor a normalization symbol).
    """
    input_cnt, hidden_cnt, output_cnt = 0, 0, 0
    norm_symbols = {}
    for key in nodes_exprs.keys():
        name = str(key)
        if name.startswith("i"):
            input_cnt += 1
        elif name.startswith("h"):
            hidden_cnt += 1
        elif name.startswith("o"):
            output_cnt += 1
        elif name.startswith("norm"):
            norm_symbols[key] = nodes_exprs[key]
    return input_cnt, hidden_cnt, output_cnt, norm_symbols


def round_expr(expr, precision=2):
    """Round every numeric atom of a sympy expression.

    A **nonzero** atom is never rounded to exactly zero: it is kept as it was.
    Collapsing the gene-level guard constants that way would silently change the
    meaning of the formula — ``LOG_EPS = 1e-7`` would turn
    ``log(max(z, 1e-7))`` into ``log(max(z, 0))`` (``-inf`` at ``z == 0``) and
    ``INV_EPS = 1e-7`` would turn ``1 / clamp(z)`` into a division by zero —
    which is exactly what a *readable* rendering must not do.  Every other atom
    is rounded as before, so ``123456789 / 10**9`` still renders as ``0.123457``
    at ``precision=6``.

    Args:
        expr (sympy.Expr): Expression to round.
        precision (int): Decimal places, forwarded to Python's ``round``.
            Defaults to ``2`` for LaTeX output (``6`` is used for Python code).

    Returns:
        sympy.Expr: A new expression with each ``sp.Number`` atom replaced by
        its rounded value, except the nonzero atoms that rounding would erase
        (those keep their original value). The input expression is not modified.

    Raises:
        TypeError: Propagated unchanged if ``expr`` is not a sympy object
            (``sp.Expr.atoms`` missing).
    """
    rounded = {}
    for atom in expr.atoms(sp.Number):
        value = round(atom, precision)
        if value.is_zero and not atom.is_zero:
            # Rounding would erase a real (tiny) constant: keep the exact atom.
            value = atom
        rounded[atom] = value
    return expr.xreplace(rounded)


def replace_variable_names(expression, mode):
    """Rewrite node symbols into indexable / LaTeX-decorated names.

    ``o0`` becomes ``o[0]`` in Python mode and ``o_{0}`` in LaTeX mode; the
    same rewrite applies to ``h<j>`` and ``i<j>``.

    Args:
        expression (str): The rendered expression (already ``str``-ified).
        mode (str): ``"python"`` or ``"latex"``. Anything else is a
            programming error.

    Returns:
        str: The rewritten expression string.

    Raises:
        ValueError: If ``mode`` is neither ``"python"`` nor ``"latex"`` (the
            pre-rewrite library used a bare ``assert``; a real exception is
            raised here so a mistake in a notebook cannot be silently absorbed
            by an optimized Python run).
    """
    if mode not in ("python", "latex"):
        raise ValueError(f"mode must be 'python' or 'latex', got {mode!r}")

    expression_str = str(expression)
    if mode == "python":
        expression_str = re.sub(r"\bo(\d+)\b", r"o[\1]", expression_str)
        expression_str = re.sub(r"\bh(\d+)\b", r"h[\1]", expression_str)
        expression_str = re.sub(r"\bi(\d+)\b", r"i[\1]", expression_str)
    else:  # latex mode
        expression_str = re.sub(
            r"\bo(\d+)\b", lambda match: f"o_{{{match.group(1)}}}", expression_str
        )
        expression_str = re.sub(
            r"\bh(\d+)\b", lambda match: f"h_{{{match.group(1)}}}", expression_str
        )
        expression_str = re.sub(
            r"\bi(\d+)\b", lambda match: f"i_{{{match.group(1)}}}", expression_str
        )
    return expression_str


class _NumPySourcePrinter(NumPyPrinter):
    """Render a sympy expression as runnable NumPy source using the ``np`` alias.

    ``sympy``'s stock :class:`~sympy.printing.numpy.NumPyPrinter` is not
    directly usable in a snippet that must simply be ``exec``-ed, for two
    reasons:

    * ``Max``/``Min`` print as ``functools.reduce(numpy.maximum, [...])``,
      which is a ``NameError`` in a snippet that binds only ``np``;
    * an unregistered ``sp.Function`` prints as a bare ``SympyClip(...)`` call
      (sympy's strict mode raises instead), which is an undefined name in the
      emitted source.

    Stock ``Abs``/``Piecewise`` are *not* broken -- ``Abs`` prints as the
    builtin ``abs(...)`` and ``Piecewise`` as
    ``numpy.select([...], [...], default=...)``, both of which ``exec`` fine --
    they are overridden anyway so that the emitted source is uniformly
    ``np``-aliased and evaluates eagerly through ``np.where``.

    Every other node kind (arithmetic, powers, relationals, the known sympy
    functions) is left to sympy and then normalised to the ``np`` alias.

    Notes:
        ``numpy.`` → ``np.`` replacement: sympy keeps the known-function and
        known-constant tables on the *class*, and copies them into the separate
        ``known_functions``/``known_constants`` instance tables during
        ``Printer.__init__`` — which already ran when this subclass rewrites
        ``self._kf``/``self._kc``, so ``exp``/``log``/``sin``/``tanh`` still
        print as ``numpy.exp``/... . The finished string is therefore
        normalised once, which is the load-bearing step; the ``_kf``/``_kc``
        rewrites keep anything that does consult the instance tables in step.
        The replacement is safe because every character of the output was
        produced by this printer.
    """

    _MAXIMUM = f"{_NUMPY_ALIAS}.maximum"
    _MINIMUM = f"{_NUMPY_ALIAS}.minimum"
    _NAN = f"{_NUMPY_ALIAS}.nan"

    def __init__(self):
        """Build the printer with ``np`` as its module alias and unknown functions allowed."""
        super().__init__(settings={"allow_unknown_functions": True})
        self._module = _NUMPY_ALIAS
        self._kf = {
            name: spelling.replace("numpy.", f"{_NUMPY_ALIAS}.")
            for name, spelling in self._kf.items()
        }
        self._kc = {
            name: spelling.replace("numpy.", f"{_NUMPY_ALIAS}.")
            for name, spelling in self._kc.items()
        }

    def doprint(self, expr):
        """Return the NumPy source of ``expr`` with every ``numpy.`` spelling aliased to ``np``.

        Args:
            expr (sympy.Expr): Expression to render.

        Returns:
            str: Runnable NumPy source (one expression, no statement).

        Raises:
            ConfigError: Propagated from :meth:`_print_Function` for a sympy
                function NumPy does not provide.
        """
        return super().doprint(expr).replace("numpy.", f"{_NUMPY_ALIAS}.")

    @staticmethod
    def _fold(function, terms):
        """Right-nest ``function`` over rendered ``terms`` (no ``functools.reduce`` in the output)."""
        if not terms:
            raise ConfigError("cannot render a Max/Min without arguments")
        rendered = terms[-1]
        for term in reversed(terms[:-1]):
            rendered = f"{function}({term}, {rendered})"
        return rendered

    def _print_Max(self, expr):
        """Render ``Max(a, b, ...)`` as ``np.maximum(a, np.maximum(b, ...))``."""
        return self._fold(self._MAXIMUM, [self._print(arg) for arg in expr.args])

    def _print_Min(self, expr):
        """Render ``Min(a, b, ...)`` as ``np.minimum(a, np.minimum(b, ...))``."""
        return self._fold(self._MINIMUM, [self._print(arg) for arg in expr.args])

    def _print_Abs(self, expr):
        """Render ``Abs(z)`` as ``np.abs(z)``."""
        return f"{_NUMPY_ALIAS}.abs({self._print(expr.args[0])})"

    def _print_Piecewise(self, expr):
        """Render a ``Piecewise`` as a right-nested ``np.where(cond, value, rest)``.

        The evaluation is eager, i.e. every branch expression is computed and
        then selected — which is correct for the shipped options because every
        branch is total: the clamps are ``Max``/``Min``, the ``lelu``/``maxabs``
        terms are plain products, and ``inv`` wraps its ``Piecewise`` in a
        single division *after* the selection, so no branch can raise a
        division by zero that the guarded form would avoid. A ``Piecewise``
        without a ``True`` branch (sympy's "no condition holds" case) nests on
        ``np.nan``, matching sympy's own value.
        """
        rendered = None
        for branch in reversed(expr.args):
            value = self._print(branch.expr)
            if branch.cond is sp.true:
                # A ``True`` branch terminates the chain: later branches are dead.
                rendered = value
            elif rendered is None:
                rendered = (
                    f"{_NUMPY_ALIAS}.where({self._print(branch.cond)}, "
                    f"{value}, {self._NAN})"
                )
            else:
                rendered = (
                    f"{_NUMPY_ALIAS}.where({self._print(branch.cond)}, "
                    f"{value}, {rendered})"
                )
        return rendered if rendered is not None else self._NAN

    def _print_Function(self, expr):
        """Render an unregistered ``sp.Function`` by its NumPy-provided name.

        ``SympyClip`` → ``np.clip``: a leading ``Sympy`` is stripped (that
        prefix marks the package's own host-side helpers) and the remainder is
        lower-cased, which is also what turns ``exp``/``log``/``sin``/``tanh``
        into ``np.exp``/``np.log``/``np.sin``/``np.tanh`` when sympy does not
        route them through its known-function table.

        Args:
            expr (sympy.Function): The function application to render.

        Returns:
            str: ``np.<name>(<args>)``.

        Raises:
            ConfigError: If NumPy has no callable of that name — emitting the
                bare name instead would produce source that fails at ``exec``
                time with a ``NameError`` instead of naming the culprit.
        """
        import numpy  # lazy: importing this module must not pull in NumPy

        name = type(expr).__name__
        candidate = (name[len("Sympy"):] if name.startswith("Sympy") else name).lower()
        function = getattr(numpy, candidate, None)
        if not callable(function):
            raise ConfigError(
                f"cannot render the sympy function {name!r} as NumPy source: "
                f"numpy has no callable {candidate!r}"
            )
        arguments = ", ".join(self._print(arg) for arg in expr.args)
        return f"{_NUMPY_ALIAS}.{candidate}({arguments})"


def _render(printer, expr):
    """Render one sympy expression as a Python-mode, NumPy-aliased source string."""
    return replace_variable_names(printer.doprint(expr), "python")


def _referenced_input_symbols(expressions):
    """Collect the ``i<k>`` symbols used by ``expressions``, ordered by ``k``."""
    found = {}
    for expression in expressions:
        for symbol in getattr(expression, "free_symbols", ()):  # a Number has none
            match = _INPUT_SYMBOL.match(str(symbol))
            if match:
                found[int(match.group(1))] = symbol
    return [found[index] for index in sorted(found)]


def _input_binding_lines(input_symbols, nodes_exprs, output_exprs):
    """Render the header that binds the raw input symbols to ``i[<position>]``.

    Args:
        input_symbols (list): The raw input node symbols, in input order. When
            it is empty the symbols the emitted expressions actually reference
            are used instead, ordered by their ``i<k>`` index.
        nodes_exprs (dict): Node symbol → expression, used only by that
            fallback.
        output_exprs (list): Closed-form output expressions, used only by that
            fallback.

    Returns:
        list of str: Comment lines, empty when there is no input to bind.
    """
    symbols = list(input_symbols) if input_symbols else _referenced_input_symbols(
        list(nodes_exprs.values()) + list(output_exprs)
    )
    if not symbols:
        return []
    lines = [
        "# `i` holds the raw inputs of this network, one float per input node,"
        " in input order:"
    ]
    lines.extend(f"#     {symbol} -> i[{position}]" for position, symbol in enumerate(symbols))
    return lines


def to_latex_code(
    symbols,
    args_symbols,
    input_symbols,
    nodes_exprs,
    output_exprs,
    forward_func,
    topo_order,
    useful_nodes,
):
    """Render a genome as a LaTeX ``align`` block.

    Args:
        symbols (dict): Node index → sympy symbol (the genome's node ordering).
        args_symbols (dict): Substitutions for the aggregation/activation
            argument symbols of the generated expressions.
        input_symbols (list): sympy symbols of the input values, in order.
            Unread: the per-node assignment lines already name their inputs.
        nodes_exprs (dict): Node symbol → expression computing that node.
        output_exprs (list): Expressions of the output nodes, in order. Unread
            for the same reason.
        forward_func (callable): Host-side forward function of the genome
            (accepted for signature compatibility; LaTeX output does not call
            it).
        topo_order (list): Node indices in topological order; iteration starts
            after the input nodes (``topo_order[input_cnt:]``).
        useful_nodes (set): Node indices that actually reach an output.
            Expressions of other nodes are skipped — a useless node's
            expression may be ``NaN`` and would break ``sp.latex``.

    Returns:
        str: The ``\\begin{align} ... \\end{align}`` block, one rounded
        (``precision=2``) line per useful node. A constant that rounding would
        erase (``1e-7``) keeps its exact value, so the clamp constants of the
        ``log``/``inv`` activations still render on the page.

    Raises:
        None: Host-only presentation code.
    """
    input_cnt, _hidden_cnt, _output_cnt, norm_symbols = analysis_nodes_exprs(
        nodes_exprs
    )
    res = "\\begin{align}\n"

    for i in topo_order[input_cnt:]:
        # Skip nodes that do not contribute to an output: a useless node's
        # expression may be NaN and would break ``sp.latex``.
        if i not in useful_nodes:
            continue
        symbol = symbols[i]
        expr = nodes_exprs[symbol].subs(args_symbols).subs(norm_symbols)
        rounded_expr = round_expr(expr, 2)
        latex_expr = f"{symbol} &= {sp.latex(rounded_expr)}\\newline\n"
        latex_expr = replace_variable_names(latex_expr, "latex")
        res += latex_expr
    res += "\\end{align}\n"
    return res


def to_python_code(
    symbols,
    args_symbols,
    input_symbols,
    nodes_exprs,
    output_exprs,
    forward_func,
    topo_order,
    useful_nodes,
):
    """Render a genome as a runnable NumPy snippet.

    The emitted source is module-level statements, not a function: it imports
    NumPy as ``np``, documents the caller-bound input array ``i`` (one float per
    input node, in input order), allocates ``h = np.zeros(hidden_cnt)`` (only
    when there are hidden nodes) and ``o = np.zeros(len(output_exprs))``, then
    assigns one useful node per line with the node's expression rounded to
    ``precision=6`` and rendered through
    :class:`_NumPySourcePrinter`. Finally it repeats every closed-form output as
    a comment. ``exec``-ing it with ``np`` and ``i`` bound is therefore all a
    caller needs::

        import numpy as np

        # `i` holds the raw inputs of this network, one float per input node, in input order:
        #     i0 -> i[0]
        h = np.zeros(2)
        o = np.zeros(1)
        h[0] = np.maximum(i[0], 0.0)
        o[0] = 1/(1 + np.exp(-h[0]))

    The NumPy helpers used are ``np.zeros`` for the allocations, ``np.maximum``/
    ``np.minimum`` for ``Max``/``Min``, ``np.abs`` for ``Abs``, ``np.where`` for
    ``Piecewise``, ``np.clip`` for the ``SympyClip`` helper and ``np.<name>``
    for every other sympy function (``exp``, ``log``, ``sin``, ``tanh``, ...).
    A sympy function NumPy does not provide raises
    :class:`~tensorneat.common.errors.ConfigError` instead of emitting a bare
    undefined name.

    Only rendering precision is reproduced, not the network bit-exactly: every
    emitted constant and weight is rounded by ``round_expr(..., 6)``, so the
    snippet matches the graph forward to within that rounding — measured up to
    ~2.5e-5 absolute / ~4.7e-4 relative on an ``inv``-dominated genome, and
    float32 round-off (~1e-7) when the rounding is bypassed.

    When ``input_symbols`` is empty, the fallback header collects the ``i<k>``
    symbols from the expressions *before* the ``args_symbols``/``norm_symbols``
    substitutions are applied, so a hand-built 8-tuple whose node expressions
    carry only argument symbols can miss the binding header while the body
    still references ``i[k]``; pass ``input_symbols`` explicitly when
    hand-building the 8-tuple.

    Args:
        symbols (dict): Node index → sympy symbol (the genome's node ordering).
        args_symbols (dict): Substitutions for the aggregation/activation
            argument symbols of the generated expressions.
        input_symbols (list): The raw input node symbols, in input order. Read
            for the input-binding header: one ``#     <symbol> -> i[<position>]``
            line per entry, so every ``i[<k>]`` in the body is defined and
            ordered. When the list is empty, the ``i<k>`` symbols the emitted
            expressions actually reference are used instead, ordered by ``k``.
        nodes_exprs (dict): Node symbol → expression computing that node.
        output_exprs (list): The **closed-form** output expressions, in output
            order (every intermediate node already substituted in by
            ``Genome.sympy_func``, so an entry can be very large). Read as the
            authority for the output count and order
            (``o = np.zeros(len(output_exprs))``) and for the trailing
            closed-form comment block; when the list is empty the output count
            falls back to the number of ``o<k>`` keys in ``nodes_exprs`` and no
            comment block is emitted.
        forward_func (callable): Host-side forward function of the genome
            (accepted for signature compatibility; the emitted code does not
            call it).
        topo_order (list): Node indices in topological order; iteration starts
            after the input nodes (``topo_order[input_cnt:]``).
        useful_nodes (set): Node indices that actually reach an output; other
            nodes are skipped for the same ``NaN`` reason as in
            :func:`to_latex_code`.

    Returns:
        str: The source lines described above, newline-terminated.

    Raises:
        ConfigError: If an expression references a sympy function NumPy cannot
            evaluate (see :meth:`_NumPySourcePrinter._print_Function`).
    """
    input_cnt, hidden_cnt, output_cnt, norm_symbols = analysis_nodes_exprs(
        nodes_exprs
    )
    output_exprs = list(output_exprs or ())
    printer = _NumPySourcePrinter()

    lines = ["import numpy as np", ""]
    lines.extend(_input_binding_lines(input_symbols, nodes_exprs, output_exprs))
    if hidden_cnt > 0:
        lines.append(f"h = np.zeros({hidden_cnt})")
    lines.append(f"o = np.zeros({len(output_exprs) if output_exprs else output_cnt})")

    for i in topo_order[input_cnt:]:
        # Skip nodes that do not contribute to an output: a useless node's
        # expression may be NaN and would break the emitted source.
        if i not in useful_nodes:
            continue
        symbol = symbols[i]
        expr = nodes_exprs[symbol].subs(args_symbols).subs(norm_symbols)
        rendered = _render(printer, round_expr(expr, 6))
        lines.append(f"{replace_variable_names(symbol, 'python')} = {rendered}")

    if output_exprs:
        comment = ["# Closed-form outputs (every intermediate node eliminated):"]
        for index, expr in enumerate(output_exprs):
            closed = expr.subs(args_symbols).subs(norm_symbols)
            rendered = _render(printer, round_expr(closed, 6))
            if rendered == f"o[{index}]":
                # A degenerate entry that *is* the output symbol says nothing.
                continue
            comment.append(f"# o[{index}] = {rendered}")
        if len(comment) > 1:
            lines.extend(comment)

    return "\n".join(lines) + "\n"
