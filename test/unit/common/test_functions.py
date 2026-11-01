"""Unit tests for ``tensorneat.common.functions`` — the ACT/AGG registries and dispatchers.

What this module proves
-----------------------
The package has two jobs and both are tested here:

* **Option order is gene data.** ``ACT``/``AGG`` are ``FunctionManager`` objects
  pairing an ordered tuple of evoxc kernels with an ordered tuple of sympy
  counterparts; a node gene stores the *position* in that order, so a reordering
  or a shifted pairing silently changes what every evolved genome means. The
  order, the length, the name↔id maps and the pairing are asserted literally.
* **The dispatchers are ``select`` cascades, not branches.** evoxc has no
  ``switch`` (and ``cond`` needs a 0-d predicate), so ``apply_activation`` /
  ``apply_aggregation`` evaluate *every* option and select per node. The tests
  therefore compare them against a per-option loop written here: each option is
  executed once as a one-kernel graph through :func:`support.runner.run_defn`,
  then selected elementwise with numpy. That oracle shares no code with the
  cascade — it is a loop over options, where the kernel under test is a chain of
  ``evoxc.select`` calls — so it catches a wrong option order, a wrong id
  comparison and a wrong neutral element alike.

Gotchas this module encodes
---------------------------
* ``ACT_IDENTITY == -1`` means "no activation" and must stay a no-op; an id
  outside ``[-1, len)`` falls back to identity (not to the last option, which is
  what the pre-rewrite ``lax.switch`` did). Aggregation has **no** ``-1``.
* Each aggregation kernel pads masked-out slots with *its own* neutral element
  (``0``, ``1``, ``-inf``, ``+inf``), and ``apply_aggregation`` then forces every
  empty column to exactly ``0.0`` (the ``has_valid`` guard). Both halves are
  asserted separately, because only the second keeps ``-inf`` out of a node.
* ``maxabs`` returns the **signed** term of largest magnitude with a first-max
  tie-break — not the magnitude, and not the last maximum.
* ``mean`` divides by the *real* edge count with a ``max(count, 1)`` floor, so an
  all-``False`` column gives ``0.0`` and never ``NaN``.
* ``sympy_module`` binds ``np`` to the keyword ``backend`` (a positional partial
  would land in ``val``), only accepts ``backend="numpy"`` after the rewrite, and
  must also emit helper classes (``SympyClip``) that are not options.
* The documented divergence: the numeric ``exp`` kernel is unclipped while
  ``SympyExp`` clips its argument to ``[-EXP_CLIP, EXP_CLIP]``; the sympy/kernel
  agreement test keeps its sample inside that band and says so.

Backend is ``"numpy"``; ``sympy`` is imported lazily inside the tests that need
it, so collection stays cheap and no optional framework is involved.
"""

from __future__ import annotations

import numpy as np
import pytest

from support.runner import run_defn

#: The canonical ACT option order, spelled out: a node gene stores this position.
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

#: The canonical AGG option order (no ``-1`` sentinel: aggregation has no identity).
AGG_OPTIONS = ("sum", "product", "max", "min", "maxabs", "mean")

#: Neutral element each aggregation kernel pads masked-out slots with.
AGG_NEUTRALS = {
    "sum": 0.0,
    "product": 1.0,
    "max": float("-inf"),
    "min": float("inf"),
    "maxabs": 0.0,
    "mean": 0.0,
}


# --------------------------------------------------------------------------- #
# oracles: a loop over options, executed one option at a time through the runner
# --------------------------------------------------------------------------- #


def _activation_reference(z, act_ids, act_fns):
    """Oracle for ``apply_activation``: apply option ``i`` to every element, then select.

    One one-kernel graph per option (the loop is over the *option list*, never
    over elements), selected with ``np.where`` here. ``z`` is the starting value,
    which is what makes ``-1`` and every out-of-range id come out as the identity
    — the documented fallback.
    """
    out = np.array(z, copy=True)
    ids = np.asarray(act_ids)
    for index, fn in enumerate(act_fns):
        values = run_defn(fn, z)
        out = np.where(ids == index, values, out)
    return out


def _aggregation_reference(z, mask, agg_ids, agg_fns):
    """Oracle for ``apply_aggregation``: one masked reduction per option, then select.

    Each kernel is executed on the whole ``(P, N, K)`` tensor (masked-out slots
    are the kernel's own neutral element, so the result is well defined), then the
    ``has_valid`` guard zeroes the columns with no real edge — the two rules the
    dispatcher documents.
    """
    out = np.zeros(z.shape[:2], dtype=np.asarray(z).dtype)
    ids = np.asarray(agg_ids)
    has_valid = np.asarray(mask).any(axis=-1)
    for index, fn in enumerate(agg_fns):
        values = np.where(has_valid, run_defn(fn, z, mask), 0.0)
        out = np.where(ids == index, values, out)
    return out


def _manager_arguments(common, kernel_names, sympy_names=None):
    """Collect one kernel tuple and one sympy tuple from the canonical modules.

    Returns the two sequences to hand to ``FunctionManager`` (it validates them
    and raises ``ValueError``), so a deliberately wrong table is expressible
    without touching the global registries — which is the point: a mismatched
    table must fail at construction time, not produce a wrong formula later.
    ``sympy_names`` may name a ``None`` entry, i.e. an option with no counterpart.
    """
    kernels = tuple(getattr(common.functions.act, name) for name in kernel_names)
    if sympy_names is None:
        return kernels
    sympy_fns = tuple(
        None if name is None else getattr(common.functions.act_sympy, name)
        for name in sympy_names
    )
    return kernels, sympy_fns


# --------------------------------------------------------------------------- #
# registry: order, length, name<->id, pairing
# --------------------------------------------------------------------------- #


def test_act_option_order_is_the_documented_registry(common):
    """Invariant: ``ACT.names`` is the 12-option tuple in ``act.ACT_FUNCS`` order.

    Oracle: :data:`ACT_OPTIONS` written out literally in this module — the gene
    encoding, not an implementation detail. Also asserts ``ACT.evoxc_fns is
    ACT_FUNCS`` (the registry defaults to the canonical tuple) and
    ``len(ACT) == 12``.
    """
    pytest.skip(
        "not implemented: ACT option names/order and length vs the literal ACT_OPTIONS tuple "
        "(owner: common/functions/act.py ACT_FUNCS, functions/__init__.py ACT)"
    )


def test_agg_option_order_is_the_documented_registry(common):
    """Invariant: ``AGG.names`` is ``("sum", "product", "max", "min", "maxabs", "mean")``.

    Oracle: :data:`AGG_OPTIONS` written out literally; the order is part of the
    checkpoint encoding, so appending is safe and reordering is not.
    """
    pytest.skip(
        "not implemented: AGG option names/order and length vs the literal AGG_OPTIONS tuple "
        "(owner: common/functions/agg.py AGG_FUNCS, functions/__init__.py AGG)"
    )


@pytest.mark.parametrize("registry", ["ACT", "AGG"])
def test_manager_name_and_index_lookups_are_inverse(common, registry):
    """Invariant: ``index2name[name2index[name]] == name`` and ``getattr(m, name) is evoxc_fns[i]``.

    Oracle: the registry's own ``evoxc_fns`` tuple, indexed by the literal option
    order of this module — a name that resolves to a *different* kernel is exactly
    the ``"inv"``/``SympyIdentity`` class of bug the pairing check exists to
    prevent.
    """
    pytest.skip(
        "not implemented: name<->index round-trip and attribute access == evoxc_fns[i] "
        "(owner: common/functions/manager.FunctionManager)"
    )


@pytest.mark.parametrize("index", [-1, 12, -12])
def test_get_evoxc_func_rejects_an_out_of_range_id(common, index):
    """Invariant: an option id outside ``[0, len)`` raises ``IndexError``; negatives do not wrap.

    Oracle: the documented "a negative id is rejected rather than wrapped"; a
    wrapped ``-1`` would silently select ``abs`` for what a caller meant as
    "unset", turning a corrupted payload into plausible numbers.
    """
    pytest.skip(
        "not implemented: IndexError for get_evoxc_func(-1) and ids >= len "
        "(owner: common/functions/manager.FunctionManager.get_evoxc_func)"
    )


@pytest.mark.parametrize("registry", ["ACT", "AGG"])
def test_get_sympy_func_pairs_every_option(common, registry):
    """Invariant: every canonical option has a sympy counterpart, in the same position.

    Oracle: ``manager.sympy_fns`` itself plus the name pairing rule
    (``_snake_case(sympy_cls.__name__) == option name``), so ``inv`` can pair with
    ``SympyInv`` and not with ``SympyIdentity`` — the pre-rewrite bug.
    """
    pytest.skip(
        "not implemented: get_sympy_func(i) is the paired class for every i, "
        "including inv<->SympyInv "
        "(owner: common/functions/manager.FunctionManager.get_sympy_func)"
    )


def test_get_sympy_func_rejects_an_option_without_a_counterpart(common):
    """Invariant: a ``None`` counterpart raises ``ValueError``, not ``TypeError`` later.

    Oracle: a private manager built over a single kernel with ``sympy_fns=(None,)``
    — the documented "some options have no sympy representation". Failing at the
    lookup keeps the error at the caller that asked for the formula.
    """
    pytest.skip(
        "not implemented: ValueError from get_sympy_func for a None counterpart "
        "(owner: common/functions/manager.FunctionManager.get_sympy_func)"
    )


def test_manager_rejects_an_inconsistent_registry(common):
    """Invariant: an inconsistent registry table raises ``ValueError`` at construction.

    Oracle: the documented ``Raises`` clause of ``FunctionManager.__init__``, one
    case per rule — empty, a non-callable kernel, a duplicate name, a mismatched
    length and a mispaired counterpart. A registry table is programmer data, so a
    ``ValueError`` at import/construction time is the whole safety net for the
    gene encoding.
    """
    pytest.skip(
        "not implemented: ValueError for empty/duplicate/mismatched/mispaired registry tables "
        "(owner: common/functions/manager.FunctionManager.__init__)"
    )


def test_manager_attribute_access_rejects_private_names(common):
    """Invariant: ``ACT._x`` raises ``AttributeError``; ``name_of`` accepts only registered kernels.

    Oracle: the documented contract — underscore names must stay unresolvable so
    ``copy``/``pickle`` (which probe ``__deepcopy__`` and friends) cannot recurse,
    and ``name_of`` uses identity, so a ``functools.partial`` wrapper is a
    ``ValueError`` rather than a wrong name.
    """
    pytest.skip(
        "not implemented: AttributeError for private names and ValueError for unregistered kernels "
        "(owner: common/functions/manager.FunctionManager.__getattr__, name_of)"
    )


@pytest.mark.parametrize("registry", ["ACT", "AGG"])
def test_random_index_stays_inside_the_option_range(common, registry, seed):
    """Invariant: ``random_index(key)`` is a rank-0 id in ``[0, len)``, never ``-1``.

    Oracle: the documented draw ``randint(key, (), 0, len(self))`` (high is
    exclusive) checked over many seeded keys, plus determinism for a repeated key.
    ``-1`` is excluded on purpose: the identity sentinel is never a uniform draw.
    """
    pytest.skip(
        "not implemented: random_index in [0, len) over many keys and deterministic per key "
        "(owner: common/functions/manager.FunctionManager.random_index)"
    )


# --------------------------------------------------------------------------- #
# apply_activation
# --------------------------------------------------------------------------- #


def test_apply_activation_matches_a_per_option_loop(common):
    """Invariant: ``out[p, n] == act_fns[id[p, n]](z[p, n])`` for every node.

    Oracle: :func:`_activation_reference` — one graph per option plus an
    elementwise ``np.where`` selection — cross-checked element by element against
    the per-node sympy counterpart evaluated with ``sp.lambdify`` on the same
    sample (the cheap way to prove the numeric table and the formula table agree
    without inverting one).
    """
    pytest.skip(
        "not implemented: apply_activation == per-option loop (evoxc and sympy) "
        "(owner: common/functions/act.apply_activation)"
    )


def test_apply_activation_identity_sentinel_is_a_no_op(common):
    """Invariant: ``act_ids == -1`` returns ``z`` itself, for every option list.

    Oracle: bitwise equality with the input on a ``(P, N)`` block of ``-1`` ids,
    including ``NaN`` slots (an output node carries ``-1`` and must not be
    activated); the pre-rewrite ``lax.switch`` would have clamped ``-1`` to the
    last option here.
    """
    pytest.skip(
        "not implemented: ACT_IDENTITY (-1) is the identity, NaN included "
        "(owner: common/functions/act.apply_activation)"
    )


@pytest.mark.parametrize("act_id", [12, 99, -7])
def test_apply_activation_falls_back_to_identity_for_an_unknown_id(common, act_id):
    """Invariant: an id outside ``[-1, len)`` yields ``z``, not the last option.

    Oracle: the documented divergence — "an id outside ``[-1, len)`` falls
    back to the cascade's initial value". Mutation only samples legal ids, so this
    is a robustness property of the kernel, and it is exactly the case the
    pre-rewrite ``switch`` got differently.
    """
    pytest.skip(
        "not implemented: out-of-range act id -> identity, not the last option "
        "(owner: common/functions/act.apply_activation)"
    )


def test_apply_activation_accepts_a_custom_option_list(common):
    """Invariant: ``act_fns`` redefines the dispatch order without touching the registry.

    Oracle: a two-kernel custom tuple (``(relu, identity)``) compared with
    :func:`_activation_reference` over that tuple — the documented extension
    point for a genome with private options, and the reason the default is a
    parameter rather than a module global.
    """
    pytest.skip(
        "not implemented: apply_activation with a custom two-option act_fns tuple "
        "(owner: common/functions/act.apply_activation)"
    )


def test_apply_activation_accepts_integral_float32_gene_ids(common):
    """Invariant: an integral ``float32`` id column dispatches like an ``int32`` one.

    Oracle: the same per-option loop, fed the ``int32`` cast of the float column —
    gene columns are float32 in the payload, so the cast must happen inside the
    dispatcher and must be exact for every legal id (``KEY_MAX``-style
    integer-exactness, here limited to ``[0, len)``).
    """
    pytest.skip(
        "not implemented: float32 gene ids dispatch identically to their int32 cast "
        "(owner: common/functions/act.apply_activation)"
    )


# --------------------------------------------------------------------------- #
# apply_aggregation
# --------------------------------------------------------------------------- #


def test_apply_aggregation_matches_a_per_option_loop(common):
    """Invariant: ``out[p, n] == agg_fns[id[p, n]](z[p, n, :], mask[p, n, :])``.

    Oracle: :func:`_aggregation_reference`, which executes each kernel once and
    selects elementwise; the sample mixes all six options in one node block, so a
    shifted option order or a wrong ``select`` predicate fails.
    """
    pytest.skip(
        "not implemented: apply_aggregation == per-option masked-reduction loop "
        "(owner: common/functions/agg.apply_aggregation)"
    )


@pytest.mark.parametrize("option", AGG_OPTIONS)
def test_aggregation_kernel_pads_with_its_own_neutral_element(common, option):
    """Invariant: an all-``False`` mask column reduces to ``AGG_NEUTRALS[option]`` standalone.

    Oracle: :data:`AGG_NEUTRALS` — ``0`` for sum/maxabs/mean, ``1`` for product,
    ``-inf`` for max, ``+inf`` for min. This is the *kernel* contract (evoxc
    reductions have no ``where=``/``initial=``, so padding is the only way to mask);
    the dispatcher is what turns ``-inf``/``+inf``/``1`` back into ``0.0``, and that
    is a separate test.
    """
    pytest.skip(
        "not implemented: standalone aggregation kernel empty-column neutral element "
        "(owner: common/functions/agg.<option>)"
    )


def test_apply_aggregation_zeroes_a_node_without_incoming_edges(common):
    """Invariant: every option yields exactly ``0.0`` for a node with an all-``False`` mask column.

    Oracle: the documented ``has_valid`` guard — ``select(count > 0, result, 0.0)``
    — checked for all six options at once. Without it a ``max`` node would absorb
    ``-inf`` and poison its activation on the next level.
    """
    pytest.skip(
        "not implemented: has_valid guard forces 0.0 for an empty column under every option "
        "(owner: common/functions/agg.apply_aggregation)"
    )


def test_aggregation_maxabs_keeps_the_sign_of_the_largest_magnitude(common):
    """Invariant: ``maxabs`` returns the signed contribution with the largest ``|z|``.

    Oracle: a hand-built ``(1, 1, K)`` column with one dominant negative value and
    a deliberate tie (the first maximum wins), compared against a numpy
    ``argmax(|z|)`` + index oracle. Reducing ``|z|`` instead (or taking the last
    maximum) flips the sign of every negative-dominant node.
    """
    pytest.skip(
        "not implemented: maxabs signed value with first-maximum tie-break "
        "(owner: common/functions/agg.maxabs)"
    )


def test_aggregation_mean_divides_by_the_real_edge_count(common):
    """Invariant: ``mean`` is ``sum / count`` of the masked-in slots, ``0.0`` for an empty column.

    Oracle: ``np.sum(np.where(mask, z, 0)) / max(mask.sum(), 1)`` written here;
    the ``max(count, 1)`` floor is what keeps an empty column at ``0.0`` instead of
    ``NaN`` — and the dispatch test above would not catch a division by the padded
    ``K``, which is uniformly wrong by a constant factor.
    """
    pytest.skip(
        "not implemented: mean divides by the masked-in count with a max(count, 1) floor "
        "(owner: common/functions/agg.mean)"
    )


def test_apply_aggregation_yields_zero_for_an_unknown_id(common):
    """Invariant: an id outside ``[0, len)`` yields ``0.0`` (the cascade's initial value).

    Oracle: the documented fallback; aggregation has no ``-1`` sentinel, so an
    out-of-range id is always a corrupted payload and must not be clamped to the
    last option (the pre-rewrite ``switch`` behaviour).
    """
    pytest.skip(
        "not implemented: unknown aggregation id -> 0.0, not the last option "
        "(owner: common/functions/agg.apply_aggregation)"
    )


# --------------------------------------------------------------------------- #
# sympy counterparts: module mapping and numerical agreement
# --------------------------------------------------------------------------- #


def test_sympy_module_covers_the_registry_and_binds_numpy(common):
    """Invariant: ``sympy_module`` covers every counterpart and binds ``numpy``.

    Oracle: ``sympy_fns`` plus ``extra_sympy_classes`` (``SympyClip``, reachable
    through ``SympyExp``) of the registry itself, and ``ValueError`` for
    ``backend="jax"`` after the rewrite. The keyword binding is asserted because a
    positional partial would land in ``val`` and break every evaluation.
    """
    pytest.skip(
        "not implemented: sympy_module keys/values, SympyClip included, "
        "backend='jax' -> ValueError "
        "(owner: common/functions/manager.FunctionManager.sympy_module)"
    )


def test_act_sympy_counterparts_match_the_evoxc_kernels(common):
    """Invariant: ``sp.lambdify`` of the sympy counterpart equals the evoxc kernel on a small batch.

    Oracle: the kernel itself (one graph per option through the shared runner),
    which is the *intended* direction for this pair: the formula exists so a human
    can read what the kernel computes. The sample stays inside the documented
    ``SympyExp`` clip band and away from ``inv``/``log`` singularities; those two
    divergences are asserted in their own tests rather than smeared into a
    tolerance here.
    """
    pytest.skip(
        "not implemented: sympy ACT counterparts == evoxc kernels on an in-band sample "
        "(owner: common/functions/act_sympy.py + act.py consistency)"
    )


def test_agg_sympy_counterparts_match_the_evoxc_kernels(common):
    """Invariant: the sympy aggregation of the valid terms equals the masked evoxc reduction.

    Oracle: for each option, the terms a real in-edge set would contribute, fed
    to ``SympySum``/``SympyProduct``/``SympyMax``/``SympyMin``/``SympyMaxabs``/
    ``SympyMean`` versus the same terms with an all-``True`` mask through the
    shared runner. ``maxabs`` is compared as a signed value and ``mean`` over the
    supplied term count, matching the numeric contracts in ``agg.py``.
    """
    pytest.skip(
        "not implemented: sympy AGG counterparts == evoxc kernels on the valid terms "
        "(owner: common/functions/agg_sympy.py + agg.py consistency)"
    )


def test_to_python_code_emits_runnable_source_for_a_hand_built_expression(common):
    """Invariant: the emitted snippet is executable and evaluates each node expression.

    Oracle: a hand-built 8-tuple (two inputs ``i0``/``i1``, one hidden ``h2`` fed
    by both, one output ``o3``, ``useful_nodes`` covering all of them, values
    rounded to ``precision=6``) rendered by
    :func:`~tensorneat.common.sympy_tools.to_python_code`; the returned source is
    ``exec``-ed against numpy inputs and compared with ``sp.lambdify`` on the same
    expressions. A ``NaN`` expression of a *useless* node must be skipped, which is
    why the test also feeds a useless node whose expression would raise.
    """
    pytest.skip(
        "not implemented: to_python_code source runs and matches the sympy expressions "
        "(owner: common/sympy_tools.to_python_code)"
    )
