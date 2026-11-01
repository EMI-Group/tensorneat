"""Host-only sympy code generation for the interpretability features.

Dependency DAG position: leaf of ``tensorneat.common``; imports only
``sympy`` and the standard library (no ``evoxc``, no NumPy global RNG, no
``jax``/``torch``), and nothing from ``genome``/``algorithm``/``problem``.

Role: :meth:`Genome.sympy_func` returns an 8-tuple of host-side objects
(``symbols, args_symbols, input_symbols, nodes_exprs, output_exprs,
forward_func, topo_order, useful_nodes``) built from a sympy translation of a
genome's nodes and connections. The functions here turn that 8-tuple into
human-readable output — a LaTeX ``align`` block for papers and a runnable
NumPy snippet for notebooks — and never participate in a generation, a trace,
or a fitness evaluation.

Contract kept from the pre-rewrite library (parity): the 8-tuple parameter
order and the ``o0``/``h1``/``i2`` node-symbol naming are unchanged, so
existing tutorials and stored expressions keep working.
"""

import re

import sympy as sp  # host-only; never imported from a traced kernel


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

    Args:
        expr (sympy.Expr): Expression to round.
        precision (int): Decimal places, forwarded to Python's ``round``.
            Defaults to ``2`` for LaTeX output (``6`` is used for Python code).

    Returns:
        sympy.Expr: A new expression with each ``sp.Number`` atom replaced by
        its rounded value (the input expression is not modified).

    Raises:
        TypeError: Propagated unchanged if ``expr`` is not a sympy object
            (``sp.Expr.atoms`` missing).
    """
    return expr.xreplace({n: round(n, precision) for n in expr.atoms(sp.Number)})


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
        nodes_exprs (dict): Node symbol → expression computing that node.
        output_exprs (list): Expressions of the output nodes, in order.
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
        (``precision=2``) line per useful node.

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

    Args:
        symbols (dict): Node index → sympy symbol (the genome's node ordering).
        args_symbols (dict): Substitutions for the aggregation/activation
            argument symbols of the generated expressions.
        input_symbols (list): sympy symbols of the input values, in order.
        nodes_exprs (dict): Node symbol → expression computing that node.
        output_exprs (list): Expressions of the output nodes, in order.
        forward_func (callable): Host-side forward function of the genome
            (accepted for signature compatibility; the emitted code does not
            call it).
        topo_order (list): Node indices in topological order; iteration starts
            after the input nodes (``topo_order[input_cnt:]``).
        useful_nodes (set): Node indices that actually reach an output; other
            nodes are skipped for the same ``NaN`` reason as in
            :func:`to_latex_code`.

    Returns:
        str: Source lines that allocate ``h = np.zeros(hidden_cnt)`` (only when
        there are hidden nodes) and ``o = np.zeros(output_cnt)``, then assign
        each useful node's expression rounded to ``precision=6``.

    Raises:
        None: Host-only presentation code.
    """
    input_cnt, hidden_cnt, output_cnt, norm_symbols = analysis_nodes_exprs(
        nodes_exprs
    )
    res = ""
    if hidden_cnt > 0:
        res += f"h = np.zeros({hidden_cnt})\n"
    res += f"o = np.zeros({output_cnt})\n"

    for i in topo_order[input_cnt:]:
        # Skip nodes that do not contribute to an output: a useless node's
        # expression may be NaN and would break the emitted source.
        if i not in useful_nodes:
            continue
        symbol = symbols[i]
        expr = nodes_exprs[symbol].subs(args_symbols).subs(norm_symbols)
        rounded_expr = round_expr(expr, 6)
        str_expr = f"{symbol} = {rounded_expr}"
        res += replace_variable_names(str_expr, "python") + "\n"

    return res
