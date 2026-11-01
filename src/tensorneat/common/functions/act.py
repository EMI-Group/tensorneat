"""Elementwise activation kernels (evoxc graph code) and the per-node activation dispatcher.

Every function in this module is *graph code*: it is called while an
``@evoxc.defn`` graph is being traced (or from a helper that is), so it may only
use ``evoxc`` ops -- never numpy, never Python branching on traced values.

Contract shared by all kernels here:

* **Batch-first, shape-polymorphic.** Activations are elementwise over *any*
  static shape; the canonical call site is the aggregated node tensor
  ``(P, N)`` of :func:`apply_activation`, but the same kernels serve output
  transforms with shape ``(P, out_dim)`` and HyperNEAT substrate nodes.
* **float32.** Inputs and outputs are float32. No masking, no padding, no
  sentinel handling: ``NaN`` in gives ``NaN`` out (the pre-rewrite library
  relied on that, so a node value is ``NaN`` exactly while it is uncomputed).
* **Registry names are the function names.** :data:`ACT_FUNCS` fixes the
  canonical option order; a node gene stores an *index into the user's option
  list*, so the order of :data:`ACT_FUNCS` is part of the gene encoding.

Note that ``abs`` shadows the builtin inside this module; do not rely on the
builtin ``abs`` here.
"""

from typing import Callable, Sequence

import evoxc

SCALE = 3.0
"""Multiplier applied by the two ``scaled_*`` activations."""

LELU_SLACK = 0.005
"""Slope of the leaky ReLU for non-positive inputs."""

INV_EPS = 1e-7
"""Magnitude clamp that keeps :func:`inv` finite at ``z == 0``."""

LOG_EPS = 1e-7
"""Lower clamp that keeps :func:`log` finite for ``z <= 0``."""

ACT_IDENTITY = -1
"""Node-gene sentinel meaning "no activation" (the gene column is floored to this)."""

EXP_CLIP = 10.0
"""Argument bound used by the *sympy* exponential only (see ``act_sympy.SympyExp``).

The numeric :func:`exp` kernel is unclipped, exactly as the pre-rewrite one; the
constant exists here so both representations of the activation table are
documented next to each other.
"""


def scaled_sigmoid(z):
    """Scaled logistic sigmoid, ``SCALE / (1 + exp(-z))``.

    Parameters
    ----------
    z : ``(...,) float32``
        Pre-activation node values, any static shape.

    Returns
    -------
    ``(...,) float32``
        Values in ``[0, SCALE]``; ``NaN`` propagates.

    Notes
    -----
    Equality with the pre-rewrite library is exact (same expression order:
    ``1 / (1 + exp(-z))`` then ``* SCALE``).
    """
    raise NotImplementedError("act.scaled_sigmoid: evoxc kernel, implemented in the implementation phase")


def sigmoid(z):
    """Logistic sigmoid, ``1 / (1 + exp(-z))``.

    Parameters
    ----------
    z : ``(...,) float32``
        Pre-activation node values, any static shape.

    Returns
    -------
    ``(...,) float32``
        Values in ``(0, 1)``; ``NaN`` propagates, ``+-inf`` saturates to ``1``/``0``.
    """
    raise NotImplementedError("act.sigmoid: evoxc kernel, implemented in the implementation phase")


def scaled_tanh(z):
    """Scaled hyperbolic tangent, ``SCALE * tanh(z)``.

    Parameters
    ----------
    z : ``(...,) float32``
        Pre-activation node values, any static shape.

    Returns
    -------
    ``(...,) float32``
        Values in ``[-SCALE, SCALE]``; ``NaN`` propagates.
    """
    raise NotImplementedError("act.scaled_tanh: evoxc kernel, implemented in the implementation phase")


def tanh(z):
    """Hyperbolic tangent.

    Parameters
    ----------
    z : ``(...,) float32``
        Pre-activation node values, any static shape.

    Returns
    -------
    ``(...,) float32``
        Values in ``[-1, 1]``; ``NaN`` propagates.
    """
    raise NotImplementedError("act.tanh: evoxc kernel, implemented in the implementation phase")


def sin(z):
    """Sine.

    Parameters
    ----------
    z : ``(...,) float32``
        Pre-activation node values, any static shape.

    Returns
    -------
    ``(...,) float32``
        Values in ``[-1, 1]``; ``NaN`` propagates.
    """
    raise NotImplementedError("act.sin: evoxc kernel, implemented in the implementation phase")


def relu(z):
    """Rectified linear unit, ``maximum(z, 0)``.

    Parameters
    ----------
    z : ``(...,) float32``
        Pre-activation node values, any static shape.

    Returns
    -------
    ``(...,) float32``
        Non-negative part of ``z``; ``NaN`` propagates.

    Notes
    -----
    ``maximum(z, 0)`` is used (not a ``select``) so that ``z == 0`` keeps ``0``
    and ``NaN`` stays ``NaN``, exactly as the pre-rewrite ``jnp.maximum``.
    """
    raise NotImplementedError("act.relu: evoxc kernel, implemented in the implementation phase")


def lelu(z):
    """Leaky ReLU with slope :data:`LELU_SLACK`, ``z if z > 0 else LELU_SLACK * z``.

    Parameters
    ----------
    z : ``(...,) float32``
        Pre-activation node values, any static shape.

    Returns
    -------
    ``(...,) float32``
        ``z`` on the positive side, ``0.005 * z`` otherwise.

    Notes
    -----
    Implemented with ``evoxc.select(z > 0, z, LELU_SLACK * z)``; at ``z == 0``
    both branches are ``0``, so the strict comparison is immaterial.
    """
    raise NotImplementedError("act.lelu: evoxc kernel, implemented in the implementation phase")


def identity(z):
    """Identity (no-op) activation.

    Parameters
    ----------
    z : ``(...,) float32``
        Node values, any static shape.

    Returns
    -------
    ``(...,) float32``
        ``z`` unchanged (the graph may elide this kernel entirely).

    Notes
    -----
    Also the branch selected by the :data:`ACT_IDENTITY` (``-1``) gene sentinel
    inside :func:`apply_activation`.
    """
    raise NotImplementedError("act.identity: evoxc kernel, implemented in the implementation phase")


def inv(z):
    """Reciprocal with a magnitude clamp, ``1 / clamp_away_from_zero(z)``.

    Parameters
    ----------
    z : ``(...,) float32``
        Pre-activation node values, any static shape.

    Returns
    -------
    ``(...,) float32``
        ``1 / z`` with ``|z|`` floored at :data:`INV_EPS`; finite everywhere
        except where ``z`` is ``NaN``.

    Notes
    -----
    The clamp is sign preserving, matching the pre-rewrite
    ``where(z > 0, maximum(z, 1e-7), minimum(z, -1e-7))``; at ``z == 0`` the
    second branch is taken, giving ``1 / -1e-7``.
    """
    raise NotImplementedError("act.inv: evoxc kernel, implemented in the implementation phase")


def log(z):
    """Natural logarithm of a lower-clamped input, ``log(maximum(z, LOG_EPS))``.

    Parameters
    ----------
    z : ``(...,) float32``
        Pre-activation node values, any static shape.

    Returns
    -------
    ``(...,) float32``
        ``log(z)`` floored at ``log(1e-7)`` for ``z <= 1e-7``; ``NaN`` propagates.
    """
    raise NotImplementedError("act.log: evoxc kernel, implemented in the implementation phase")


def exp(z):
    """Exponential, unclipped.

    Parameters
    ----------
    z : ``(...,) float32``
        Pre-activation node values, any static shape.

    Returns
    -------
    ``(...,) float32``
        ``exp(z)``; overflows to ``inf`` above ``~88`` in float32.

    Notes
    -----
    The pre-rewrite numeric kernel was also unclipped, so the two agree bit for
    bit -- but the sympy counterpart (:class:`~.act_sympy.SympyExp`) clips its
    argument to ``[-10, 10]`` for interpretability. See ``CONTEXT.md``
    (Known Divergences).
    """
    raise NotImplementedError("act.exp: evoxc kernel, implemented in the implementation phase")


def abs(z):
    """Absolute value.

    Parameters
    ----------
    z : ``(...,) float32``
        Node values, any static shape.

    Returns
    -------
    ``(...,) float32``
        ``|z|``; ``NaN`` propagates.
    """
    raise NotImplementedError("act.abs: evoxc kernel, implemented in the implementation phase")


ACT_FUNCS: tuple[Callable, ...] = (
    scaled_sigmoid,
    sigmoid,
    scaled_tanh,
    tanh,
    sin,
    relu,
    lelu,
    identity,
    inv,
    log,
    exp,
    abs,
)
"""Canonical ACT option order -- part of the gene encoding, do not reorder.

A node gene stores the *position in the user's option list*; when a genome is
configured with ``ACT.get_all_funcs()`` this tuple is that order, so appending
is safe, reordering or deleting is a checkpoint/parity break. ``clip`` is
deliberately absent (see ``CONTEXT.md``, Design Decisions).
"""


def apply_activation(z, act_ids, act_fns: Sequence[Callable] = ACT_FUNCS):
    """Apply a per-node activation selected by gene id, without ``lax.switch``.

    evoxc has no ``switch`` and ``cond`` requires a 0-d predicate, so the
    per-node dispatch is an elementwise ``select`` cascade over the option list
    instead of a branch.

    Parameters
    ----------
    z : ``(...,) float32``
        Pre-activation node values, batch-first, any static shape. The
        canonical call site is the aggregated node tensor ``(P, N)``.
    act_ids : ``(...,) float32`` or ``(...,) int32``
        Per-node activation ids, broadcastable against ``z``. Integral-valued
        float32 gene columns are cast to int32; :data:`ACT_IDENTITY` (``-1``)
        selects :func:`identity`.
    act_fns : sequence of callables, optional
        The node gene's ``activation_options`` -- a static, trace-time Python
        tuple of elementwise kernels, each ``z -> z``. Defaults to
        :data:`ACT_FUNCS` (the global registry order).

    Returns
    -------
    ``(...,) float32``
        ``act_fns[id](z)`` per element, with ``z`` itself where ``id == -1``.

    Notes
    -----
    Cost: one evaluation of *every* option plus one ``select`` per option over
    the ``(P, N)`` shape -- 12 + 12 ops for the canonical list, independent of
    the in-degree bound ``K`` and therefore subdominant to the ``(P, N, K)``
    gathers of the forward pass. There is no static way to compute only the
    options the population happens to use: the id is gene data, not a shape.

    An id outside ``[-1, len(act_fns))`` falls back to :func:`identity` (the
    cascade's initial value). The pre-rewrite ``lax.switch`` clamped such ids
    to the last option; the divergence is unreachable for well-formed genomes
    because mutation only samples ``[0, len(act_fns))`` and ``-1`` is written
    only for output nodes.
    """
    raise NotImplementedError("act.apply_activation: evoxc kernel, implemented in the implementation phase")
