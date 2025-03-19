import re
import sympy as sp

def analysis_nodes_exprs(nodes_exprs):
    input_cnt, hidden_cnt, output_cnt = 0, 0, 0
    norm_symbols = {}
    for key in nodes_exprs.keys():
        if str(key).startswith('i'):
            input_cnt += 1
        elif str(key).startswith('h'):
            hidden_cnt += 1
        elif str(key).startswith('o'):
            output_cnt += 1
        elif str(key).startswith('norm'):
            norm_symbols[key] = nodes_exprs[key]
    return input_cnt, hidden_cnt, output_cnt, norm_symbols

def round_expr(expr, precision=2):
    """
    Round numerical values in a sympy expression to a given precision.
    """
    return expr.xreplace({n: round(n, precision) for n in expr.atoms(sp.Number)})


def replace_variable_names(expression, mode):
    """
    Transform sympy expression to a string with array index that can be used in python code.
    For example, `o0` will be transformed to `o[0]` in Python mode,
    and `o0` will be transformed to LaTeX format using sympy's `latex()` in LaTeX mode.
    """
    assert mode in ["python", "latex"]
    expression_str = str(expression)

    if mode == "python":
        expression_str = re.sub(r"\bo(\d+)\b", r"o[\1]", expression_str)
        expression_str = re.sub(r"\bh(\d+)\b", r"h[\1]", expression_str)
        expression_str = re.sub(r"\bi(\d+)\b", r"i[\1]", expression_str)
    else:  # latex mode
        expression_str = re.sub(r"\bo(\d+)\b", lambda m: f"o_{{{m.group(1)}}}", expression_str)
        expression_str = re.sub(r"\bh(\d+)\b", lambda m: f"h_{{{m.group(1)}}}", expression_str)
        expression_str = re.sub(r"\bi(\d+)\b", lambda m: f"i_{{{m.group(1)}}}", expression_str)

    return expression_str


def to_latex_code(symbols, args_symbols, input_symbols, nodes_exprs, output_exprs, forward_func, topo_order):
    input_cnt, hidden_cnt, output_cnt, norm_symbols = analysis_nodes_exprs(nodes_exprs)
    res = "\\begin{align}\n"
    
    for i in topo_order[input_cnt: ]:
        symbol = symbols[i]
        expr = nodes_exprs[symbol].subs(args_symbols).subs(norm_symbols)
        rounded_expr = round_expr(expr, 2)
        latex_expr = f"{symbol} &= {sp.latex(rounded_expr)}\\newline\n"
        latex_expr = replace_variable_names(latex_expr, "latex")
        res += latex_expr
    res += "\\end{align}\n"
    return res


def to_python_code(symbols, args_symbols, input_symbols, nodes_exprs, output_exprs, forward_func, topo_order):
    input_cnt, hidden_cnt, output_cnt, norm_symbols = analysis_nodes_exprs(nodes_exprs)
    res = ""
    if hidden_cnt > 0:
        res += f"h = np.zeros({hidden_cnt})\n"
    res += f"o = np.zeros({output_cnt})\n"

    for i in topo_order[input_cnt: ]:
        symbol = symbols[i]
        expr = nodes_exprs[symbol].subs(args_symbols).subs(norm_symbols)
        rounded_expr = round_expr(expr, 6)
        str_expr = f"{symbol} = {rounded_expr}"
        res += replace_variable_names(str_expr, "python") + "\n"
    
    return res