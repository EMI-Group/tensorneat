"""Sympy counterparts of the activation kernels -- host-only interpretability code.

Nothing here runs inside a graph: these classes are used by ``Genome.sympy_func``
/ ``node.to_dict`` to render an evolved network as a formula (``sp.lambdify``)
or LaTeX. They therefore use plain ``sympy``/``numpy`` and must stay numerically
identical to the kernels in :mod:`.act` -- every constant is imported from there
so the two representations cannot drift.

Shapes and dtypes do not apply here -- every class rewrites one symbolic
expression, and the numeric contract (shapes, dtypes, sentinels) lives in
:mod:`.act`.

Each class is a *constructor-time rewriter*: ``SympySigmoid(z)`` returns the
ordinary sympy expression ``1 / (1 + exp(-z))`` (not an instance of
``SympySigmoid``), which is what keeps the generated code readable and keeps
printer/``lambdify`` handling out of the picture -- except for
:class:`SympyClip`, which has no closed form in sympy and defines
``numerical_eval`` so ``sp.lambdify`` can evaluate it.
"""

import numpy as np
import sympy as sp

from .act import EXP_CLIP, INV_EPS, LELU_SLACK, LOG_EPS, SCALE


class SympyClip(sp.Function):
    """Three-argument clamp, ``min(max(val, min_val), max_val)``.

    Only reachable through :class:`SympyExp`; it is *not* a registered option
    (the pre-rewrite sympy table exposed a ``"clip"`` name with no numeric
    counterpart -- see ``CONTEXT.md``).
    """

    @classmethod
    def eval(cls, val, min_val, max_val):
        """Fold the clamp when all three arguments are numeric constants."""
        if val.is_Number and min_val.is_Number and max_val.is_Number:
            return sp.Piecewise(
                (min_val, val < min_val), (max_val, val > max_val), (val, True)
            )
        return None

    @staticmethod
    def numerical_eval(val, min_val, max_val, backend=np):
        """Evaluate the clamp numerically.

        Parameters
        ----------
        val, min_val, max_val : numbers or numpy arrays
            Operand and bounds, broadcastable.
        backend : module, optional
            ``numpy`` (the only backend after the evoxc rewrite); needs ``clip``.

        Returns
        -------
        numpy array or scalar
            ``backend.clip(val, min_val, max_val)``.
        """
        return backend.clip(val, min_val, max_val)

    def _sympystr(self, printer):
        """Render as ``clip(val, min_val, max_val)`` in ``sstr`` output."""
        return f"clip({self.args[0]}, {self.args[1]}, {self.args[2]})"

    def _latex(self, printer):
        """Render as ``\\mathrm{clip}(...)`` in LaTeX output."""
        return rf"\mathrm{{clip}}\left({sp.latex(self.args[0])}, {self.args[1]}, {self.args[2]}\right)"


class SympySigmoid(sp.Function):
    """Logistic sigmoid, ``1 / (1 + exp(-z))``."""

    @classmethod
    def eval(cls, z):
        """Return the sigmoid expression for ``z``."""
        return 1 / (1 + sp.exp(-z))


class SympyScaledSigmoid(sp.Function):
    """Scaled logistic sigmoid, ``SCALE / (1 + exp(-z))``."""

    @classmethod
    def eval(cls, z):
        """Return :class:`SympySigmoid` of ``z`` times :data:`.act.SCALE`."""
        return SympySigmoid(z) * SCALE


class SympyTanh(sp.Function):
    """Hyperbolic tangent."""

    @classmethod
    def eval(cls, z):
        """Return ``tanh(z)``."""
        return sp.tanh(z)


class SympyScaledTanh(sp.Function):
    """Scaled hyperbolic tangent, ``SCALE * tanh(z)``."""

    @classmethod
    def eval(cls, z):
        """Return :class:`SympyTanh` of ``z`` times :data:`.act.SCALE`."""
        return SympyTanh(z) * SCALE


class SympySin(sp.Function):
    """Sine."""

    @classmethod
    def eval(cls, z):
        """Return ``sin(z)``."""
        return sp.sin(z)


class SympyRelu(sp.Function):
    """Rectified linear unit, ``max(z, 0)``."""

    @classmethod
    def eval(cls, z):
        """Return ``sp.Max(z, 0)``."""
        return sp.Max(z, 0)


class SympyLelu(sp.Function):
    """Leaky ReLU, ``z if z > 0 else LELU_SLACK * z``."""

    @classmethod
    def eval(cls, z):
        """Return the two-branch piecewise expression."""
        return sp.Piecewise((z, z > 0), (LELU_SLACK * z, True))


class SympyIdentity(sp.Function):
    """Identity activation."""

    @classmethod
    def eval(cls, z):
        """Return ``z`` unchanged."""
        return z


class SympyInv(sp.Function):
    """Reciprocal with a sign-preserving magnitude clamp, ``1 / clamped(z)``."""

    @classmethod
    def eval(cls, z):
        """Return ``1 / Piecewise((Max(z, INV_EPS), z > 0), (Min(z, -INV_EPS), True))``.

        This is the counterpart of :func:`.act.inv`; the pre-rewrite sympy table
        paired ``"inv"`` with :class:`SympyIdentity` by mistake, which this
        module no longer does.
        """
        z = sp.Piecewise(
            (sp.Max(z, INV_EPS), z > 0), (sp.Min(z, -INV_EPS), True)
        )
        return 1 / z


class SympyLog(sp.Function):
    """Natural logarithm of a lower-clamped input, ``log(max(z, LOG_EPS))``."""

    @classmethod
    def eval(cls, z):
        """Return ``log(Max(z, LOG_EPS))``."""
        return sp.log(sp.Max(z, LOG_EPS))


class SympyExp(sp.Function):
    """Exponential of an argument clipped to ``[-EXP_CLIP, EXP_CLIP]``.

    Divergence from :func:`.act.exp`: the numeric kernel does *not* clip, so a
    node whose pre-activation exceeds ``EXP_CLIP`` is over/under-estimated by
    the generated formula. Kept for formula stability (the pre-rewrite library
    behaved the same way); recorded in ``CONTEXT.md`` (Known Divergences).
    """

    @classmethod
    def eval(cls, z):
        """Return ``exp(SympyClip(z, -EXP_CLIP, EXP_CLIP))``."""
        return sp.exp(SympyClip(z, -EXP_CLIP, EXP_CLIP))


class SympyAbs(sp.Function):
    """Absolute value."""

    @classmethod
    def eval(cls, z):
        """Return ``sp.Abs(z)``."""
        return sp.Abs(z)


ACT_SYMPY_HELPERS = (SympyClip,)
"""Sympy helpers emitted for ``sp.lambdify`` without being registry options.

``SympyExp`` builds ``SympyClip`` internally, so the class name appears in
generated expressions even though ``clip`` is not an ACT option.
"""


ACT_SYMPY_FUNCS = (
    SympyScaledSigmoid,
    SympySigmoid,
    SympyScaledTanh,
    SympyTanh,
    SympySin,
    SympyRelu,
    SympyLelu,
    SympyIdentity,
    SympyInv,
    SympyLog,
    SympyExp,
    SympyAbs,
)
"""Sympy counterparts of :data:`.act.ACT_FUNCS`, in the same order.

Position ``i`` here must pair with position ``i`` of ``act.ACT_FUNCS``;
``FunctionManager`` validates the pairing by class name and raises if it does
not hold.
"""
