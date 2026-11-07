"""Elementwise activation kernels (evoxc graph code) and the per-node activation dispatcher.

Every function in this module is *graph code*: it is called while an
``@evoxc.defn`` graph is being traced (or from a helper that is), so it may only
use ``evoxc`` ops -- never numpy, never Python branching on traced values.

Contract shared by all kernels here:

* **Batch-first, shape-polymorphic.** Activations are elementwise over *any*
  static shape; the canonical call site is the aggregated node tensor
  ``(P, N)`` of a node gene's per-level cascade (which calls these kernels
  directly), and the same kernels serve the dispatch of
  :func:`apply_activation`, output transforms with shape ``(P, out_dim)`` and
  HyperNEAT substrate nodes.
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
    return evoxc.multiply(
        evoxc.divide(1.0, evoxc.add(1.0, evoxc.exp(evoxc.negate(z)))), SCALE
    )


def sigmoid(z, scalars=None):
    """Logistic sigmoid, ``1 / (1 + exp(-z))``.

    Parameters
    ----------
    z : ``(...,) float32``
        Pre-activation node values, any static shape.
    scalars : mapping of ``float`` to ``() float32``, optional
        Loop-invariant rank-0 constants prepared once per forward pass
        (``genome.step._PreparedScalars.mapping``); ``None`` builds them
        locally. Only the ``1.0`` is looked up, and a mapping without it keeps
        the literal, so the default spelling is op for op the one this kernel
        always traced.

    Returns
    -------
    ``(...,) float32``
        Values in ``(0, 1)``; ``NaN`` propagates, ``+-inf`` saturates to ``1``/``0``.

    Notes
    -----
    Both ``1.0`` literals of the expression are the *same* operand, so a
    prepared pass spends one shared tensor on them and traces no ``constant``
    op at all. With ``scalars=None`` the value is the Python float ``1.0`` and
    each of the two uses is its own ``constant`` op, because ``evoxc`` merges
    no constants.
    """
    one = 1.0 if scalars is None else scalars.get(1.0, 1.0)
    return evoxc.divide(one, evoxc.add(one, evoxc.exp(evoxc.negate(z))))


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
    return evoxc.multiply(evoxc.tanh(z), SCALE)


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
    return evoxc.tanh(z)


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
    return evoxc.sin(z)


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
    return evoxc.maximum(z, 0.0)


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
    return evoxc.select(evoxc.greater(z, 0.0), z, evoxc.multiply(z, LELU_SLACK))


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
    inside :func:`apply_activation` -- which is why the kernel carries the
    ``is_identity`` flag: selecting it is a no-op, so the dispatcher emits no
    ``equal``/``select`` pair for an option whose kernel is this one.
    """
    return z


identity.is_identity = True
"""Static flag read by :func:`apply_activation`: this option returns ``z`` itself."""


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
    clamped = evoxc.select(
        evoxc.greater(z, 0.0),
        evoxc.maximum(z, INV_EPS),
        evoxc.minimum(z, -INV_EPS),
    )
    return evoxc.divide(1.0, clamped)


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
    return evoxc.log(evoxc.maximum(z, LOG_EPS))


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
    (Divergences from the frozen legacy implementation).
    """
    return evoxc.exp(z)


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
    return evoxc.abs(z)


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
configured with ``ACT.evoxc_fns`` this tuple is that order, so appending
is safe, reordering or deleting is a checkpoint/parity break. ``clip`` is
deliberately absent (see ``CONTEXT.md``, the sympy / interpretability
contract).
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
    Cost: one evaluation of *every* option plus one ``equal``/``select`` pair and
    one index ``constant`` per option over the ``(...,)`` shape -- three traced
    ops per non-identity option, independent of the in-degree bound ``K`` and
    therefore subdominant to the ``(P, N, K)`` gathers of the forward pass.
    Measured graph ops at ``(P, N) = (100, 50)`` int32 ids: 5 for a one-option
    :func:`tanh` list (``constant``/``equal``/``tanh``/``select``/``return``),
    10 for one-option :func:`sigmoid` (whose kernel carries two constants of
    its own), 1 for an identity-only list (it returns ``z`` with no ``select`` at
    all) and 74 for the canonical list -- 33 dispatch ops for its 11
    non-identity options plus 41 kernel ops. There is no static way to compute
    only the options the population happens to use: the id is gene data, not a
    shape, so the cascade is at its per-option op floor.

    Two traced ops are *not* emitted: the ``cast`` of an id column that is
    already ``int32`` (the gene's ``prepare`` casts once per forward pass, and the
    ids reach every level slice typed), and the ``equal``/``select`` pair of an
    option whose kernel is :func:`identity` (``is_identity``), because selecting
    it returns the accumulator unchanged.

    An id outside ``[-1, len(act_fns))`` falls back to :func:`identity` (the
    cascade's initial value). The pre-rewrite ``lax.switch`` clamped such ids
    to the last option; the divergence is unreachable for well-formed genomes
    because mutation only samples ``[0, len(act_fns))`` and ``-1`` is written
    only for output nodes.
    """
    ids = act_ids if act_ids.dtype == evoxc.int32 else evoxc.cast(act_ids, evoxc.int32)
    # The accumulator starts at ``z``, which *is* the identity branch: an id of
    # ``-1`` (or any id outside the option list) matches no ``select`` and keeps
    # ``z`` untouched. An ``identity`` option is skipped for the same reason --
    # selecting it would only rewrite ``z`` with itself.
    out = z
    for index, fn in enumerate(act_fns):
        if getattr(fn, "is_identity", False):
            continue
        out = evoxc.select(evoxc.equal(ids, index), fn(z), out)
    return out
