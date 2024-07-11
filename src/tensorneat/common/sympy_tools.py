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


def replace_variable_names(expression):
    """
    Transform sympy expression to a string with array index that can be used in python code.
    For example, `o0` will be transformed to `o[0]`.
    """
    expression_str = str(expression)
    expression_str = re.sub(r"\bo(\d+)\b", r"o[\1]", expression_str)
    expression_str = re.sub(r"\bh(\d+)\b", r"h[\1]", expression_str)
    expression_str = re.sub(r"\bi(\d+)\b", r"i[\1]", expression_str)
    return expression_str


def to_latex_code(symbols, args_symbols, input_symbols, nodes_exprs, output_exprs, use_hidden_nodes=True):
    input_cnt, hidden_cnt, output_cnt, norm_symbols = analysis_nodes_exprs(nodes_exprs)
    res = "\\begin{align}\n"
    
    if not use_hidden_nodes:
        for i in range(output_cnt):
            expr = output_exprs[i].subs(args_symbols)
            rounded_expr = round_expr(expr, 2)
            latex_expr = f"o_{{{sp.latex(i)}}} &= {sp.latex(rounded_expr)}\\newline\n"
            res += latex_expr
    else:
        for i in range(hidden_cnt):
            symbol = sp.symbols(f"h{i}")
            expr = nodes_exprs[symbol].subs(args_symbols).subs(norm_symbols)
            rounded_expr = round_expr(expr, 2)
            latex_expr = f"h_{{{sp.latex(i)}}} &= {sp.latex(rounded_expr)}\\newline\n"
            res += latex_expr
        for i in range(output_cnt):
            symbol = sp.symbols(f"o{i}")
            expr = nodes_exprs[symbol].subs(args_symbols).subs(norm_symbols)
            rounded_expr = round_expr(expr, 2)
            latex_expr = f"o_{{{sp.latex(i)}}} &= {sp.latex(rounded_expr)}\\newline\n"
            res += latex_expr
    res += "\\end{align}\n"
    return res


def to_python_code(symbols, args_symbols, input_symbols, nodes_exprs, output_exprs, use_hidden_nodes=True):
    input_cnt, hidden_cnt, output_cnt, norm_symbols = analysis_nodes_exprs(nodes_exprs)
    res = ""
    if not use_hidden_nodes:
        # pre-allocate space
        res += f"o = np.zeros({output_cnt})\n"
        for i in range(output_cnt):
            expr = output_exprs[i].subs(args_symbols)
            rounded_expr = round_expr(expr, 6)
            str_expr = f"o{i} = {rounded_expr}"
            res += replace_variable_names(str_expr) + "\n"
    else:
        # pre-allocate space
        res += f"h = np.zeros({hidden_cnt})\n"
        res += f"o = np.zeros({output_cnt})\n"
        for i in range(hidden_cnt):
            symbol = sp.symbols(f"h{i}")
            expr = nodes_exprs[symbol].subs(args_symbols).subs(norm_symbols)
            rounded_expr = round_expr(expr, 6)
            str_expr = f"h{i} = {rounded_expr}"
            res += replace_variable_names(str_expr) + "\n"
        for i in range(output_cnt):
            symbol = sp.symbols(f"o{i}")
            expr = nodes_exprs[symbol].subs(args_symbols).subs(norm_symbols)
            rounded_expr = round_expr(expr, 6)
            str_expr = f"o{i} = {rounded_expr}"
            res += replace_variable_names(str_expr) + "\n"
    return res