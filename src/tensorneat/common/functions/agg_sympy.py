"""Sympy counterparts of the aggregation kernels -- host-only interpretability code.

Nothing here runs inside a graph. ``sp.Function`` subclasses take an *iterable*
of incoming expressions and expand it into a single sympy expression, which is
what makes an evolved topology printable as one formula.

The semantic contract mirrors :mod:`.agg`:

* the input iterable is the list of *valid* incoming contributions (the graph
  side's masked, neutral-padded ``K`` axis collapsed to the real edges), so
  there is no mask argument here;
* ``maxabs`` returns the signed term of largest magnitude, like the numeric
  kernel, not the magnitude;
* ``mean`` divides by the number of terms actually supplied.

An empty iterable is a caller bug on this side: ``sp.Max()``/``sp.Add()`` of
nothing is not the neutral element the numeric side uses, so ``sympy_func`` must
pass ``[0]`` (or the node's bias) for a node with no incoming edges.
"""

import sympy as sp


class SympySum(sp.Function):
    """Sum of the incoming terms."""

    @classmethod
    def eval(cls, z):
        """Return ``Add(*z)`` over the iterable ``z``."""
        return sp.Add(*z)


class SympyProduct(sp.Function):
    """Product of the incoming terms."""

    @classmethod
    def eval(cls, z):
        """Return ``Mul(*z)`` over the iterable ``z``."""
        return sp.Mul(*z)


class SympyMax(sp.Function):
    """Maximum of the incoming terms."""

    @classmethod
    def eval(cls, z):
        """Return ``Max(*z)`` over the iterable ``z``."""
        return sp.Max(*z)


class SympyMin(sp.Function):
    """Minimum of the incoming terms."""

    @classmethod
    def eval(cls, z):
        """Return ``Min(*z)`` over the iterable ``z``."""
        return sp.Min(*z)


class SympyMaxabs(sp.Function):
    """Signed term of largest magnitude (the sympy form of :func:`.agg.maxabs`)."""

    @classmethod
    def eval(cls, z):
        """Return a ``Piecewise`` selecting the term whose absolute value is largest.

        Parameters
        ----------
        z : iterable of sympy expressions
            The valid incoming terms, in edge order.

        Returns
        -------
        sympy expression
            ``Piecewise((t_i, Eq(Abs(t_i), max_j Abs(t_j))), ..., (0, True))``;
        because ``Piecewise`` takes the first satisfied branch, ties resolve to
        the earliest term -- the same first-maximum rule as the numeric kernel's
        ``argmax``.

        Notes
        -----
        ``sp.Max(*z, key=sp.Abs)`` (the pre-rewrite formulation) is rejected by
        current sympy (``Basic.__new__() got an unexpected keyword argument
        'key'``), so the signed selection is spelled out explicitly.
        """
        terms = list(z)
        largest = sp.Max(*[sp.Abs(term) for term in terms])
        branches = [(term, sp.Eq(sp.Abs(term), largest)) for term in terms]
        return sp.Piecewise(*branches, (sp.Integer(0), True))


class SympyMean(sp.Function):
    """Arithmetic mean of the incoming terms."""

    @classmethod
    def eval(cls, z):
        """Return ``Add(*z) / len(z)`` for the iterable ``z``."""
        terms = list(z)
        return sp.Add(*terms) / len(terms)


AGG_SYMPY_FUNCS = (
    SympySum,
    SympyProduct,
    SympyMax,
    SympyMin,
    SympyMaxabs,
    SympyMean,
)
"""Sympy counterparts of :data:`.agg.AGG_FUNCS`, in the same order.

Position ``i`` here must pair with position ``i`` of ``agg.AGG_FUNCS``;
``FunctionManager`` validates and raises ``ValueError`` when it does not.
"""
