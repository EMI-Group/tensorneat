"""Unit tests for ``tensorneat.common.functions`` — the ACT/AGG registries and dispatchers.

What this module proves
-----------------------
The package has two jobs and both are tested here:

* **Option order is gene data.** ``ACT``/``AGG`` are ``FunctionManager`` objects
  pairing an ordered tuple of evoxc kernels with an ordered tuple of sympy
  counterparts; a node gene stores the *position* in that order, so a reordering
  or a shifted pairing silently changes what every evolved genome means. The
  order, the length, the name↔id maps, the pairing and the ``ValueError`` /
  ``IndexError`` paths are asserted literally.
* **The dispatchers are ``select`` cascades, not branches.** evoxc has no
  ``switch`` (and ``cond`` needs a 0-d predicate), so ``apply_activation`` /
  ``apply_aggregation`` evaluate *every* option and select per node. The tests
  therefore compare them against a per-option loop written here: each option is
  executed once as a one-kernel graph through :func:`support.runner.run_defn`,
  then selected elementwise with numpy. That oracle shares no code with the
  cascade — it is a loop over options, where the kernel under test is a chain of
  ``evoxc.select`` calls — so it catches a wrong option order, a wrong id
  comparison and a wrong neutral element alike.

Oracles
-------
Three independent references are used, none of which shares code with what it
checks: :mod:`support.functions_ref` (numpy kernels written from the documented
closed forms and neutral padding, reached through its own dispatcher),
:func:`_activation_reference` / :func:`_aggregation_reference` (the per-option
loop described above), and ``sp.lambdify`` of the sympy counterpart for the
formula side — plus hand-enumerated literals (:data:`ACT_OPTIONS`,
:data:`AGG_OPTIONS`, :data:`AGG_NEUTRALS`, :data:`ACT_SYMPY_OPTIONS`) where a
table *is* the contract.

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
  tie-break; ``mean`` divides by the *real* edge count with a ``max(count, 1)``
  floor, so an all-``False`` column gives ``0.0`` and never ``NaN``.
* ``FunctionManager.sympy_module`` binds ``np`` to the keyword ``backend`` (a
  positional partial would land in ``val``), accepts only ``backend="numpy"``
  after the rewrite, and emits exactly the classes that define
  ``numerical_eval`` — ``SympyClip``, reachable through ``SympyExp`` but not an
  option of its own. The counterparts are constructor-time rewriters to plain
  sympy expressions and therefore appear in no module mapping.
* The documented divergence: the numeric ``exp`` kernel is unclipped while
  ``SympyExp`` clips its argument to ``[-EXP_CLIP, EXP_CLIP]``; the sympy/kernel
  agreement test keeps its sample inside that band and asserts the divergence
  just outside it.

Backend is ``"numpy"``; ``sympy`` is imported lazily inside the tests that need it.
"""

from __future__ import annotations

import functools

import numpy as np
import pytest

import evoxc

from support import functions_ref
from support.runner import run_defn

#: The canonical ACT option order, spelled out: a node gene stores this position.
ACT_OPTIONS = (
    "scaled_sigmoid", "sigmoid", "scaled_tanh", "tanh",
    "sin", "relu", "lelu", "identity",
    "inv", "log", "exp", "abs",
)

#: The canonical AGG option order (no ``-1`` sentinel: aggregation has no identity).
AGG_OPTIONS = ("sum", "product", "max", "min", "maxabs", "mean")

#: The paired sympy class names, in the same order as the kernels above.
ACT_SYMPY_OPTIONS = (
    "SympyScaledSigmoid", "SympySigmoid", "SympyScaledTanh", "SympyTanh",
    "SympySin", "SympyRelu", "SympyLelu", "SympyIdentity",
    "SympyInv", "SympyLog", "SympyExp", "SympyAbs",
)

#: The paired sympy class names of the six aggregations.
AGG_SYMPY_OPTIONS = (
    "SympySum", "SympyProduct", "SympyMax", "SympyMin", "SympyMaxabs", "SympyMean",
)

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


def _registry(common, name):
    """Return ``common.ACT`` / ``common.AGG`` by the parametrisation string."""
    return getattr(common, name)


def _activation_sample(rng, pop=2):
    """Build a ``(P, 12)`` value block whose id column covers every ACT option.

    Values span ``[-3, 3]`` so both signs and both branches of ``relu``/``lelu``
    are exercised; one slot is exactly ``0.0`` (the ``inv``/``lelu`` boundary) and
    one is ``NaN`` (an uncomputed node value, which every kernel passes through).
    The ids are a per-row permutation of ``range(12)``, so a dispatcher that
    compares against the *column* index instead of the id fails.
    """
    z = rng.uniform(-3.0, 3.0, size=(pop, len(ACT_OPTIONS))).astype(np.float32)
    z[0, 0] = np.float32(0.0)
    z[0, 1] = np.float32(np.nan)
    ids = np.stack(
        [rng.permutation(len(ACT_OPTIONS)) for _ in range(pop)]
    ).astype(np.int32)
    return z, ids


def _aggregation_sample(rng, pop=2):
    """Build a ``(P, 6, 4)`` contribution tensor with every mask corner in it.

    Node ``j`` carries aggregation id ``j``, so one block exercises all six
    options. Column 0 is fully live, column 1 fully empty (``has_valid`` ⇒
    ``0.0``), column 2 mixes live and dead slots with ``NaN`` in the dead ones
    (the mask, not ``isnan``, decides), column 3 holds a live ``NaN`` (a real edge
    from an uncomputed node — it poisons sum/product/max/min/mean) and columns
    4/5 get random partial masks. ``maxabs`` (node 4) is kept ``NaN``-free on its
    live slots so the argmax tie-break is the only thing under test there.
    """
    nodes = len(AGG_OPTIONS)
    z = rng.normal(size=(pop, nodes, 4)).astype(np.float32)
    mask = rng.random((pop, nodes, 4)) > 0.35
    mask[:, 0, :] = True
    mask[:, 1, :] = False
    z[:, 1, :] = np.float32(np.nan)  # dead slots are NaN: only the mask matters
    mask[:, 2, 0] = False
    z[:, 2, 0] = np.float32(np.nan)
    mask[:, 2, 1:] = True
    mask[:, 3, :] = True
    z[:, 3, 0] = np.float32(np.nan)  # a *live* NaN poisons the node
    z[:, 3, 1:] = np.abs(z[:, 3, 1:]) + np.float32(0.5)
    ids = np.tile(np.arange(nodes, dtype=np.int32), (pop, 1))
    return z, mask, ids


# --------------------------------------------------------------------------- #
# registry: order, length, name<->id, pairing
# --------------------------------------------------------------------------- #


def test_act_option_order_is_the_documented_registry(common):
    """Invariant: ``ACT.names`` is the 12-option tuple in ``act.ACT_FUNCS`` order.

    Oracle: :data:`ACT_OPTIONS` written out literally in this module — the gene
    encoding, not an implementation detail. Also asserts ``ACT.evoxc_fns is
    ACT_FUNCS`` (the registry defaults to the canonical tuple), ``len(ACT) == 12``
    and that the id of an option is its position in that literal tuple.
    """
    act = common.functions.act
    ACT = common.ACT

    assert ACT.names == ACT_OPTIONS
    assert ACT.index2name == ACT_OPTIONS
    assert tuple(fn.__name__ for fn in ACT.evoxc_fns) == ACT_OPTIONS
    assert ACT.evoxc_fns is act.ACT_FUNCS
    assert len(ACT) == len(ACT_OPTIONS) == 12
    assert ACT.name2index == {name: index for index, name in enumerate(ACT_OPTIONS)}
    assert ACT.name2index["scaled_sigmoid"] == 0
    assert ACT.name2index["abs"] == 11


def test_agg_option_order_is_the_documented_registry(common):
    """Invariant: ``AGG.names`` is ``("sum", "product", "max", "min", "maxabs", "mean")``.

    Oracle: :data:`AGG_OPTIONS` written out literally; the order is part of the
    checkpoint encoding, so appending is safe and reordering is not. Also asserts
    ``AGG.evoxc_fns is AGG_FUNCS`` and ``len(AGG) == 6``.
    """
    agg = common.functions.agg
    AGG = common.AGG

    assert AGG.names == AGG_OPTIONS
    assert AGG.index2name == AGG_OPTIONS
    assert tuple(fn.__name__ for fn in AGG.evoxc_fns) == AGG_OPTIONS
    assert AGG.evoxc_fns is agg.AGG_FUNCS
    assert len(AGG) == len(AGG_OPTIONS) == 6
    assert AGG.name2index == {name: index for index, name in enumerate(AGG_OPTIONS)}


@pytest.mark.parametrize("registry", ["ACT", "AGG"])
def test_manager_name_and_index_lookups_are_inverse(common, registry):
    """Invariant: ``index2name[name2index[name]] == name`` and ``getattr(m, name) is evoxc_fns[i]``.

    Oracle: :data:`ACT_OPTIONS` / :data:`AGG_OPTIONS`, indexed by position — a
    name that resolves to a *different* kernel is exactly the
    ``"inv"``/``SympyIdentity`` class of bug the pairing check exists to prevent.
    """
    manager = _registry(common, registry)
    options = ACT_OPTIONS if registry == "ACT" else AGG_OPTIONS

    for index, name in enumerate(options):
        assert manager.index2name[index] == name
        assert manager.name2index[name] == index
        assert getattr(manager, name) is manager.evoxc_fns[index]
        assert manager.get_evoxc_func(index) is manager.evoxc_fns[index]
        assert manager.name_of(manager.evoxc_fns[index]) == name
        assert manager.index2name[manager.name2index[name]] == name


@pytest.mark.parametrize("index", [-1, 12, -12])
def test_get_evoxc_func_rejects_an_out_of_range_id(common, index):
    """Invariant: an option id outside ``[0, len)`` raises ``IndexError``; negatives do not wrap.

    Oracle: the documented "a negative id is rejected rather than wrapped"; a
    wrapped ``-1`` would silently select ``abs`` (or ``mean``) for what a caller
    meant as "unset", turning a corrupted payload into plausible numbers. Both
    lookup paths (kernel and sympy counterpart) and both registries are checked,
    and the last legal id still resolves.
    """
    for registry in ("ACT", "AGG"):
        manager = _registry(common, registry)
        with pytest.raises(IndexError):
            manager.get_evoxc_func(index)
        with pytest.raises(IndexError):
            manager.get_sympy_func(index)
        assert (
            manager.get_evoxc_func(len(manager) - 1) is manager.evoxc_fns[-1]
        ), f"{registry}: the last legal id must still resolve"


@pytest.mark.parametrize("registry", ["ACT", "AGG"])
def test_get_sympy_func_pairs_every_option(common, registry):
    """Invariant: every canonical option has a sympy counterpart, in the same position.

    Oracle: :data:`ACT_SYMPY_OPTIONS` / :data:`AGG_SYMPY_OPTIONS` written out
    literally, so ``inv`` can only pair with ``SympyInv`` and not with
    ``SympyIdentity`` (the pre-rewrite bug). ``obtain_sympy`` must reach the same
    class through an id, a name and a kernel.
    """
    import sympy as sp

    manager = _registry(common, registry)
    expected = ACT_SYMPY_OPTIONS if registry == "ACT" else AGG_SYMPY_OPTIONS

    assert tuple(cls.__name__ for cls in manager.sympy_fns) == expected
    for index, name in enumerate(expected):
        cls = manager.get_sympy_func(index)
        assert cls is manager.sympy_fns[index]
        assert issubclass(cls, sp.Function)
        assert cls.__name__ == name
        assert manager.obtain_sympy(index) is cls
        assert manager.obtain_sympy(manager.names[index]) is cls
        assert manager.obtain_sympy(manager.evoxc_fns[index]) is cls

    if registry == "ACT":
        inv_id = manager.name2index["inv"]
        assert manager.get_sympy_func(inv_id) is common.functions.act_sympy.SympyInv
        assert manager.obtain_sympy("inv").__name__ == "SympyInv"


def test_get_sympy_func_rejects_an_option_without_a_counterpart(common):
    """Invariant: a ``None`` counterpart raises ``ValueError``, not ``TypeError`` later.

    Oracle: a private manager built over a single kernel with ``sympy_fns=(None,)``
    — the documented "some options have no sympy representation". Failing at the
    lookup keeps the error at the caller that asked for the formula, and every
    address form (id, name, kernel) takes that path. The kernel lookup still works.
    """
    act = common.functions.act
    manager = common.FunctionManager((act.sigmoid,), (None,))

    assert manager.names == ("sigmoid",)
    assert manager.sympy_fns == (None,)
    assert manager.get_evoxc_func(0) is act.sigmoid
    for address in (0, np.int64(0), "sigmoid", act.sigmoid):
        with pytest.raises(ValueError):
            manager.obtain_sympy(address)
    with pytest.raises(ValueError):
        manager.get_sympy_func(0)


def test_manager_rejects_an_inconsistent_registry(common):
    """Invariant: an inconsistent registry table raises ``ValueError`` at construction.

    Oracle: the documented ``Raises`` clause of ``FunctionManager.__init__``, one
    case per rule — empty, a non-callable kernel, a duplicate name, a mismatched
    length and a mispaired counterpart. A registry table is programmer data, so a
    ``ValueError`` at construction time is the whole safety net for the gene
    encoding; the valid table built the same way must *not* raise.
    """
    act = common.functions.act
    act_sympy = common.functions.act_sympy
    manager_cls = common.FunctionManager

    kernels, sympy_fns = _manager_arguments(
        common, ("sigmoid", "inv"), ("SympySigmoid", "SympyInv")
    )
    valid = manager_cls(kernels, sympy_fns)
    assert valid.names == ("sigmoid", "inv")
    assert valid.get_sympy_func(1) is act_sympy.SympyInv

    mispaired_kernels, mispaired_sympy = _manager_arguments(
        common, ("inv",), ("SympyIdentity",)
    )
    cases = [
        # empty registry
        ((), ()),
        # non-callable kernel
        (("sigmoid",), (act_sympy.SympySigmoid,)),
        # duplicate option names
        (
            (act.sigmoid, act.sigmoid),
            (act_sympy.SympySigmoid, act_sympy.SympySigmoid),
        ),
        # mismatched table lengths
        ((act.sigmoid,), ()),
        # mispaired counterpart (the pre-rewrite inv <-> SympyIdentity bug)
        (mispaired_kernels, mispaired_sympy),
    ]
    for evoxc_fns, sympy_classes in cases:
        with pytest.raises(ValueError):
            manager_cls(evoxc_fns, sympy_classes)


def test_manager_attribute_access_rejects_private_names(common):
    """Invariant: ``ACT._x`` raises ``AttributeError``; ``name_of`` accepts only registered kernels.

    Oracle: the documented contract — underscore names must stay unresolvable so
    ``copy``/``pickle`` (which probe ``__deepcopy__``/``__setstate__``) cannot
    recurse, and ``name_of`` compares by identity, so a ``functools.partial``
    wrapper *and* the numpy reference kernel of the same option are both a
    ``ValueError`` rather than a wrong name. The removed module-level
    ``get_func_name`` must not come back.
    """
    act = common.functions.act
    ACT = common.ACT

    for name in ("_x", "_evoxc_fns", "__deepcopy__", "not_an_option"):
        with pytest.raises(AttributeError):
            getattr(ACT, name)
    assert not hasattr(common.functions.manager, "get_func_name")
    assert not hasattr(common.functions, "get_func_name")

    for index, name in enumerate(ACT_OPTIONS):
        assert ACT.name_of(act.ACT_FUNCS[index]) == name
    with pytest.raises(ValueError):
        ACT.name_of(functools.partial(act.abs))
    with pytest.raises(ValueError):
        ACT.name_of(functions_ref.act_abs)


@pytest.mark.parametrize("registry", ["ACT", "AGG"])
def test_random_index_stays_inside_the_option_range(common, registry, seed):
    """Invariant: ``random_index(key)`` is a rank-0 id in ``[0, len)``, never ``-1``.

    Oracle: the documented range ``[0, len(self))`` (``randint``'s high is
    exclusive) checked over many seeded keys, plus determinism for a repeated key
    and a draw set wider than one option — a hard-wired ``0`` or a ``-1``
    identity sentinel would pass a single-key check but not this one. The
    documented spelling is a ``(1,)`` draw reshaped to rank 0, so the result is a
    0-d ``int32``.
    """
    manager = _registry(common, registry)
    draws = []
    for offset in range(24):
        key = evoxc.random.key(seed + offset)
        value = run_defn(manager.random_index, key)
        array = np.asarray(value)
        assert array.shape == (), f"{registry}: random_index must be rank-0, got {array.shape}"
        assert array.dtype == np.int32
        assert 0 <= int(array) < len(manager)
        draws.append(int(array))

    assert len(set(draws)) > 1, f"{registry}: every draw was {draws[0]}"

    repeated = evoxc.random.key(seed)
    first = int(run_defn(manager.random_index, repeated))
    second = int(run_defn(manager.random_index, repeated))
    assert first == second == draws[0]


# --------------------------------------------------------------------------- #
# apply_activation
# --------------------------------------------------------------------------- #


def test_apply_activation_matches_a_per_option_loop(common, rng):
    """Invariant: ``out[p, n] == act_fns[id[p, n]](z[p, n])`` for every node.

    Oracle: :func:`_activation_reference` — one graph per option plus an
    elementwise ``np.where`` selection — and, independently, the closed forms of
    :mod:`support.functions_ref` run through its own dispatcher. The sample's id
    block is a permutation covering all 12 options, so a shifted table, a wrong
    id comparison or a column-indexed lookup fails; the ``NaN`` slot proves the
    kernels mask nothing.
    """
    act = common.functions.act
    z, ids = _activation_sample(rng)

    out = run_defn(act.apply_activation, z, ids)

    assert set(np.unique(ids).tolist()) == set(range(len(ACT_OPTIONS)))
    np.testing.assert_array_equal(out, _activation_reference(z, ids, act.ACT_FUNCS))
    np.testing.assert_allclose(
        out,
        functions_ref.apply_activation(z, ids, functions_ref.ACT_FUNCS),
        rtol=1e-6,
        atol=1e-6,
        equal_nan=True,
    )
    assert np.isnan(out[0, 1]), "a NaN node value must stay NaN through the cascade"


def test_apply_activation_identity_sentinel_is_a_no_op(common, rng):
    """Invariant: ``act_ids == -1`` returns ``z`` itself, for every option list.

    Oracle: bitwise equality with the input on a ``(P, N)`` block of ``-1`` ids,
    including the ``NaN`` slot (an output node carries ``-1`` and must not be
    activated); the pre-rewrite ``lax.switch`` would have clamped ``-1`` to the
    last option (``abs``), which is asserted to *not* be the answer here.
    """
    act = common.functions.act
    z, _ids = _activation_sample(rng)
    assert np.any(z < 0), "the sample must contain negative values for the abs check"

    ids = np.full(z.shape, -1, dtype=np.int32)
    out = run_defn(act.apply_activation, z, ids)
    np.testing.assert_array_equal(out, z)
    assert np.isnan(out[0, 1])
    assert not np.array_equal(
        np.nan_to_num(out), np.nan_to_num(functions_ref.act_abs(z))
    ), "the -1 lane must be identity, not the last option (abs)"

    # A mixed column: the -1 lanes are untouched, the legal ids are dispatched.
    mixed = np.tile(
        np.array([-1, 0, -1, 5, -1, 7, -1, 11, -1, 2, -1, 3], dtype=np.int32),
        (z.shape[0], 1),
    )
    mixed_out = run_defn(act.apply_activation, z, mixed)
    np.testing.assert_array_equal(mixed_out[mixed == -1], z[mixed == -1])
    np.testing.assert_array_equal(
        mixed_out, _activation_reference(z, mixed, act.ACT_FUNCS)
    )


@pytest.mark.parametrize("act_id", [12, 99, -7])
def test_apply_activation_falls_back_to_identity_for_an_unknown_id(common, act_id, rng):
    """Invariant: an id outside ``[-1, len)`` yields ``z``, not the last option.

    Oracle: the documented fallback — "an id outside ``[-1, len(act_fns))`` falls
    back to the cascade's initial value". Mutation only samples legal ids, so this
    is a robustness property of the kernel, and it is exactly the case the
    pre-rewrite ``switch`` got differently: ``abs`` (the last option) is asserted
    to differ on this sample.
    """
    act = common.functions.act
    z = rng.uniform(-3.0, 3.0, size=(2, 5)).astype(np.float32)
    z[0, 0] = np.float32(np.nan)

    ids = np.full(z.shape, act_id, dtype=np.int32)
    out = run_defn(act.apply_activation, z, ids)

    np.testing.assert_array_equal(out, z)
    assert not np.array_equal(
        np.nan_to_num(out), np.nan_to_num(functions_ref.act_abs(z))
    ), "an unknown id must not clamp to the last option"


def test_apply_activation_accepts_a_custom_option_list(common, rng):
    """Invariant: ``act_fns`` redefines the dispatch order without touching the registry.

    Oracle: a two-kernel custom tuple (``(relu, identity)``) compared with
    :func:`_activation_reference` over that tuple, and against the *default*
    registry on the same ids (which must differ) — the documented extension point
    for a genome with private options. The tuple is a trace-time static, so it is
    bound with ``functools.partial`` rather than passed as a traced argument.
    """
    act = common.functions.act
    z, _ids = _activation_sample(rng)
    custom = (act.relu, act.identity)
    ids = rng.integers(0, len(custom), size=z.shape).astype(np.int32)
    assert set(np.unique(ids).tolist()) == {0, 1}

    bound = functools.partial(act.apply_activation, act_fns=custom)
    out = run_defn(bound, z, ids)

    np.testing.assert_array_equal(out, _activation_reference(z, ids, custom))
    np.testing.assert_allclose(
        out, functions_ref.apply_activation(z, ids, (functions_ref.act_relu, functions_ref.act_identity)),
        rtol=1e-6,
        atol=1e-6,
        equal_nan=True,
    )
    assert not np.array_equal(
        np.nan_to_num(out),
        np.nan_to_num(run_defn(act.apply_activation, z, ids)),
    ), "the custom tuple must not resolve to the canonical registry order"


def test_apply_activation_accepts_integral_float32_gene_ids(common, rng):
    """Invariant: an integral ``float32`` id column dispatches like an ``int32`` one.

    Oracle: the same per-option loop, fed the ``int32`` cast of the float column —
    gene columns are float32 in the payload, so the cast must happen inside the
    dispatcher and must be exact for every legal id (``KEY_MAX``-style
    integer-exactness, here limited to ``[-1, len)``).
    """
    act = common.functions.act
    z, ids = _activation_sample(rng)

    from_int = run_defn(act.apply_activation, z, ids)
    from_float = run_defn(act.apply_activation, z, ids.astype(np.float32))

    np.testing.assert_array_equal(from_int, from_float)
    np.testing.assert_array_equal(
        from_float, _activation_reference(z, ids, act.ACT_FUNCS)
    )

    negative = np.full(z.shape, -1.0, dtype=np.float32)
    np.testing.assert_array_equal(run_defn(act.apply_activation, z, negative), z)


# --------------------------------------------------------------------------- #
# apply_aggregation
# --------------------------------------------------------------------------- #


def test_apply_aggregation_matches_a_per_option_loop(common, rng):
    """Invariant: ``out[p, n] == agg_fns[id[p, n]](z[p, n, :], mask[p, n, :])``.

    Oracle: :func:`_aggregation_reference`, which executes each kernel once and
    selects elementwise, plus the masked reductions of
    :mod:`support.functions_ref` through its own dispatcher. One node block mixes
    all six options, with a fully live column, an empty one, a ``NaN``-padded one
    and a live ``NaN``, so a shifted option order, a wrong ``select`` predicate,
    a ``where=``-style skip of ``NaN`` and a missing ``has_valid`` guard all fail.
    """
    agg = common.functions.agg
    z, mask, ids = _aggregation_sample(rng)

    out = run_defn(agg.apply_aggregation, z, mask, ids)

    assert set(np.unique(ids).tolist()) == set(range(len(AGG_OPTIONS)))
    np.testing.assert_array_equal(
        out, _aggregation_reference(z, mask, ids, agg.AGG_FUNCS)
    )
    np.testing.assert_allclose(
        out,
        functions_ref.apply_aggregation(z, mask, ids, functions_ref.AGG_FUNCS),
        rtol=1e-6,
        atol=1e-6,
        equal_nan=True,
    )
    assert np.isnan(out[:, 3]).all(), "a live NaN contribution must poison its node"
    assert np.all(out[:, 1] == 0.0), "an empty column must be exactly 0.0"


@pytest.mark.parametrize("option", AGG_OPTIONS)
def test_aggregation_kernel_pads_with_its_own_neutral_element(common, option):
    """Invariant: an all-``False`` mask column reduces to ``AGG_NEUTRALS[option]`` standalone.

    Oracle: :data:`AGG_NEUTRALS` — ``0`` for sum/maxabs/mean, ``1`` for product,
    ``-inf`` for max, ``+inf`` for min. This is the *kernel* contract (evoxc
    reductions have no ``where=``/``initial=``, so neutral padding is the only way
    to mask); the dispatcher is what turns ``-inf``/``+inf``/``1`` back into
    ``0.0``, and that is a separate test. ``NaN`` in every dead slot proves the
    pad is applied *before* the reduce.
    """
    agg = common.functions.agg
    z = np.array([[[np.nan, 4.0, -2.0, np.nan]]], dtype=np.float32)
    mask = np.zeros(z.shape, dtype=bool)

    out = run_defn(getattr(agg, option), z, mask)

    expected = np.full((1, 1), AGG_NEUTRALS[option], dtype=np.float32)
    np.testing.assert_array_equal(out, expected)
    assert not np.isnan(out).any()


def test_apply_aggregation_zeroes_a_node_without_incoming_edges(common):
    """Invariant: every option yields exactly ``0.0`` for a node with an all-``False`` mask column.

    Oracle: the documented ``has_valid`` guard — ``select(count > 0, result, 0.0)``
    — checked for all six options at once, and per node in a mixed block against
    :func:`_aggregation_reference`. Without it a ``max`` node would absorb
    ``-inf``, a ``min`` node ``+inf`` and a ``product`` node ``1``, each of which
    would poison the activation on the next level.
    """
    agg = common.functions.agg

    # One empty node per option: all six ids, every column dead.
    z_dead = np.tile(
        np.array([[[3.0, -1.0]]], dtype=np.float32), (1, len(AGG_OPTIONS), 1)
    )
    mask_dead = np.zeros(z_dead.shape, dtype=bool)
    ids_dead = np.arange(len(AGG_OPTIONS), dtype=np.int32).reshape(1, -1)
    out_dead = run_defn(agg.apply_aggregation, z_dead, mask_dead, ids_dead)
    np.testing.assert_array_equal(out_dead, np.zeros_like(out_dead))
    assert not np.isinf(out_dead).any() and not np.isnan(out_dead).any()

    # A mixed block: the empty node sits between live ``max``/``min`` nodes.
    z = np.array(
        [[[3.0, -1.0], [2.0, 5.0], [-4.0, 0.5]]], dtype=np.float32
    )
    mask = np.ones(z.shape, dtype=bool)
    mask[0, 1, :] = False
    ids = np.array([[0, 2, 3]], dtype=np.int32)

    out = run_defn(agg.apply_aggregation, z, mask, ids)

    assert out[0, 1] == np.float32(0.0)
    np.testing.assert_array_equal(
        out, _aggregation_reference(z, mask, ids, agg.AGG_FUNCS)
    )


def test_aggregation_maxabs_keeps_the_sign_of_the_largest_magnitude(common):
    """Invariant: ``maxabs`` returns the signed contribution with the largest ``|z|``.

    Oracle: a hand-built ``(1, 4, 4)`` block — one dominant negative with a
    deliberate tie (the first maximum wins), a masked-out ``NaN``, a masked-out
    larger value and an all-dead column — compared against a numpy
    ``argmax(|z|)`` + gather oracle written here and against
    :func:`support.functions_ref.agg_maxabs`. Reducing ``|z|`` instead (or taking
    the last maximum, or reading the un-zeroed tensor) flips the sign or leaks a
    padding slot.
    """
    agg = common.functions.agg
    z = np.array(
        [
            [
                [1.0, -2.0, 2.0, 0.5],
                [np.nan, 0.5, -0.25, 0.0],
                [5.0, 0.5, -0.25, 0.0],
                [np.nan, np.nan, np.nan, np.nan],
            ]
        ],
        dtype=np.float32,
    )
    mask = np.array(
        [
            [
                [True, True, True, True],
                [False, True, True, False],
                [False, True, True, False],
                [False, False, False, False],
            ]
        ]
    )

    out = run_defn(agg.maxabs, z, mask)

    zeroed = np.where(mask, z, np.float32(0.0))
    index = np.argmax(np.abs(zeroed), axis=-1)[..., None]
    oracle = np.take_along_axis(zeroed, index, axis=-1)[..., 0]
    np.testing.assert_array_equal(out, oracle)
    np.testing.assert_allclose(
        out, functions_ref.agg_maxabs(z, mask), rtol=1e-6, atol=0.0
    )

    assert out[0, 0] == np.float32(-2.0), "signed value, first-maximum tie-break"
    assert out[0, 1] == np.float32(0.5), "a masked-out NaN must not win"
    assert out[0, 2] == np.float32(0.5), "a masked-out 5.0 must not win"
    assert out[0, 3] == np.float32(0.0), "an empty column keeps the 0.0 neutral"


def test_aggregation_mean_divides_by_the_real_edge_count(common):
    """Invariant: ``mean`` is ``sum / count`` of the masked-in slots, ``0.0`` for an empty column.

    Oracle: ``np.sum(np.where(mask, z, 0)) / max(mask.sum(), 1)`` written here;
    the ``max(count, 1)`` floor is what keeps an empty column at ``0.0`` instead of
    ``NaN``, and the divide-by-``K`` variant is asserted to *differ*, because
    dividing by the padded width is uniformly wrong by a constant factor and would
    survive a tolerance-only comparison.
    """
    agg = common.functions.agg
    z = np.array(
        [
            [
                [2.0, 4.0, np.nan, np.nan],
                [1.0, 3.0, 5.0, np.nan],
                [np.nan, np.nan, np.nan, np.nan],
            ]
        ],
        dtype=np.float32,
    )
    mask = np.array(
        [
            [
                [True, True, False, False],
                [True, True, True, False],
                [False, False, False, False],
            ]
        ]
    )

    out = run_defn(agg.mean, z, mask)

    count = mask.sum(axis=-1)
    total = np.sum(np.where(mask, z.astype(np.float64), 0.0), axis=-1)
    np.testing.assert_allclose(out, total / np.maximum(count, 1), rtol=1e-6)
    assert not np.isnan(out).any(), "the max(count, 1) floor must keep NaN out"
    assert out[0, 0] == np.float32(3.0)
    assert out[0, 1] == np.float32(3.0)
    assert out[0, 2] == np.float32(0.0)
    by_width = total / z.shape[-1]
    assert out[0, 0] != np.float32(by_width[0, 0]), "must divide by the real edge count"


def test_apply_aggregation_yields_zero_for_an_unknown_id(common):
    """Invariant: an id outside ``[0, len)`` yields ``0.0`` (the cascade's initial value).

    Oracle: the documented fallback; aggregation has no ``-1`` sentinel, so an
    out-of-range id is always a corrupted payload and must not be clamped to the
    last option (the pre-rewrite ``switch`` behaviour) — ``mean`` of the live node
    is asserted to be non-zero so a clamp would be visible.
    """
    agg = common.functions.agg
    z = np.array([[[1.0, 2.0], [3.0, -4.0]]], dtype=np.float32)
    mask = np.ones(z.shape, dtype=bool)

    mixed = run_defn(
        agg.apply_aggregation, z, mask, np.array([[0, 6]], dtype=np.int32)
    )
    assert mixed[0, 0] == np.float32(3.0), "the legal id still aggregates"
    assert mixed[0, 1] == np.float32(0.0)

    clamped = run_defn(
        agg.apply_aggregation,
        z,
        mask,
        np.full((1, 2), len(agg.AGG_FUNCS) - 1, dtype=np.int32),
    )
    assert clamped[0, 1] != np.float32(0.0), "mean of a live node is non-zero"

    for unknown in (-1, len(agg.AGG_FUNCS), 99):
        ids = np.full(z.shape[:2], unknown, dtype=np.int32)
        np.testing.assert_array_equal(
            run_defn(agg.apply_aggregation, z, mask, ids),
            np.zeros(z.shape[:2], dtype=np.float32),
        )


# --------------------------------------------------------------------------- #
# sympy counterparts: module mapping and numerical agreement
# --------------------------------------------------------------------------- #


def test_sympy_module_covers_the_registry_and_binds_numpy(common):
    """Invariant: ``sympy_module`` emits every class with ``numerical_eval`` and binds ``numpy``.

    Oracle: the documented rule — a ``partial(cls.numerical_eval, backend=np)``
    for every counterpart *and* ``extra_sympy_classes`` entry that defines
    ``numerical_eval``. The counterparts are constructor-time rewriters to plain
    sympy expressions, so ``SympyClip`` (built by ``SympyExp``, not an option of
    its own) is the only such class; AGG has none. The keyword binding is asserted
    because a positional partial would land in ``val`` and break every evaluation,
    which is proved by lambdifying ``SympyExp`` outside its clip band.
    """
    import sympy as sp

    ACT = common.ACT
    AGG = common.AGG
    act_sympy = common.functions.act_sympy

    act_module = ACT.sympy_module()
    assert set(act_module) == {"SympyClip"}
    assert AGG.sympy_module() == {}

    clip = act_module["SympyClip"]
    assert isinstance(clip, functools.partial)
    assert clip.func is act_sympy.SympyClip.numerical_eval
    assert set(clip.keywords) == {"backend"}
    assert clip.keywords["backend"] is np

    x = sp.Symbol("x")
    host = sp.lambdify(x, act_sympy.SympyExp(x), modules=["numpy", act_module])
    values = np.array([-20.0, 0.0, 20.0])
    np.testing.assert_allclose(
        np.asarray(host(values)), np.exp(np.clip(values, -10.0, 10.0)), rtol=1e-9
    )

    for manager in (ACT, AGG):
        with pytest.raises(ValueError):
            manager.sympy_module("jax")
        with pytest.raises(ValueError):
            manager.sympy_module(backend="torch")


def test_act_sympy_counterparts_match_the_evoxc_kernels(common):
    """Invariant: ``sp.lambdify`` of the sympy counterpart equals the evoxc kernel.

    Oracle: the kernel itself (one graph per option through the shared runner),
    which is the *intended* direction for this pair: the formula exists so a human
    can read what the kernel computes. The sample spans both signs and stays
    inside the documented ``SympyExp`` clip band and away from the ``inv``/``log``
    clamp boundaries; the ``exp`` divergence just outside the band is asserted
    explicitly rather than smeared into the tolerance.
    """
    import sympy as sp

    ACT = common.ACT
    x = sp.Symbol("x")
    modules = ["numpy", ACT.sympy_module()]
    sample = np.array([-2.0, -1.5, -1.0, 1.0, 1.5, 2.0], dtype=np.float32)

    for index, name in enumerate(ACT_OPTIONS):
        expression = ACT.get_sympy_func(index)(x)
        numeric = np.asarray(sp.lambdify(x, expression, modules=modules)(sample))
        expected = run_defn(ACT.evoxc_fns[index], sample)
        np.testing.assert_allclose(
            np.asarray(numeric, dtype=np.float64), expected, rtol=1e-6, atol=1e-6, err_msg=name
        )

    # Documented divergence: SympyExp clips to [-EXP_CLIP, EXP_CLIP], exp does not.
    exp_index = ACT.name2index["exp"]
    host = sp.lambdify(x, ACT.get_sympy_func(exp_index)(x), modules=modules)
    out_of_band = np.array([200.0, -200.0], dtype=np.float32)
    np.testing.assert_allclose(
        np.asarray(host(out_of_band)),
        np.array([np.exp(10.0), np.exp(-10.0)]),
        rtol=1e-6,
    )
    with np.errstate(over="ignore", under="ignore"):
        kernel = run_defn(ACT.evoxc_fns[exp_index], out_of_band)
    assert np.isinf(kernel[0]) and kernel[1] == np.float32(0.0)


def test_agg_sympy_counterparts_match_the_evoxc_kernels(common):
    """Invariant: the sympy aggregation of the valid terms equals the masked evoxc reduction.

    Oracle: for each option, the terms a real in-edge set would contribute, fed to
    ``SympySum``/``SympyProduct``/``SympyMax``/``SympyMin``/``SympyMaxabs``/
    ``SympyMean`` versus the same terms with an all-``True`` mask through the
    shared runner. ``maxabs`` is compared as a *signed* value (``-4``, not ``4``)
    and ``mean`` over the supplied term count, matching ``agg.py``.
    """
    import sympy as sp

    AGG = common.AGG
    terms = sp.symbols("t0:4")
    sample = np.array([1.0, -2.0, 3.0, -0.5], dtype=np.float32)
    mask = np.ones((1, 1, len(terms)), dtype=bool)

    for index, name in enumerate(AGG_OPTIONS):
        expression = AGG.get_sympy_func(index)(terms)
        numeric = np.asarray(sp.lambdify(terms, expression, modules="numpy")(*sample))
        expected = run_defn(AGG.evoxc_fns[index], sample.reshape(1, 1, -1), mask)
        np.testing.assert_allclose(
            np.asarray(numeric, dtype=np.float64),
            np.asarray(expected).reshape(-1),
            rtol=1e-6,
            atol=1e-6,
            err_msg=name,
        )

    maxabs_index = AGG.name2index["maxabs"]
    maxabs_host = sp.lambdify(terms, AGG.get_sympy_func(maxabs_index)(terms), modules="numpy")
    assert np.float32(maxabs_host(*sample)) == np.float32(3.0)
    negative_dominant = np.array([1.0, -4.0, 0.5, -0.25], dtype=np.float32)
    assert np.float32(maxabs_host(*negative_dominant)) == np.float32(-4.0), (
        "the signed term of largest magnitude, not its magnitude"
    )
    kernel_maxabs = run_defn(
        AGG.evoxc_fns[maxabs_index], negative_dominant.reshape(1, 1, -1), mask
    )
    np.testing.assert_array_equal(np.asarray(kernel_maxabs).reshape(-1), np.float32(-4.0))

    # ``mean`` divides by the number of terms *supplied*, not by the padded width.
    mean_index = AGG.name2index["mean"]
    three_terms = terms[:3]
    mean_host = sp.lambdify(
        three_terms, AGG.get_sympy_func(mean_index)(three_terms), modules="numpy"
    )
    three = sample[:3]
    three_mask = np.ones((1, 1, 3), dtype=bool)
    kernel_mean = run_defn(common.functions.agg.mean, three.reshape(1, 1, -1), three_mask)
    np.testing.assert_allclose(
        np.float32(mean_host(*three)), np.asarray(kernel_mean).reshape(-1), rtol=1e-6
    )


def test_to_python_code_emits_runnable_source_for_a_hand_built_expression(common):
    """Invariant: the emitted snippet is executable and evaluates each node expression.

    Oracle: a hand-built 8-tuple (two inputs ``i0``/``i1``, one hidden ``h0`` fed
    by both, one output ``o0``, and a *useless* ``h1`` whose expression is
    ``sp.nan``) rendered by
    :func:`~tensorneat.common.sympy_tools.to_python_code`; the returned source is
    ``exec``-ed against numpy inputs and compared with ``sp.lambdify`` of the same
    expressions composed by hand. A coefficient with nine decimals pins
    ``precision=6``; ``h1`` being skipped is what keeps the ``nan`` name out of the
    source, because a useless node's expression need not be computable.
    """
    import sympy as sp

    from tensorneat.common.sympy_tools import to_python_code

    i0, i1 = sp.symbols("i0 i1")
    h0, h1 = sp.symbols("h0 h1")
    o0 = sp.Symbol("o0")
    symbols = {0: i0, 1: i1, 2: h0, 3: o0, 4: h1}
    nodes_exprs = {
        i0: i0,
        i1: i1,
        h0: sp.Rational(123456789, 10**9) * (i0 + i1),
        o0: 1 / (1 + sp.exp(-h0)),
        h1: sp.nan,
    }

    source = to_python_code(
        symbols, {}, [i0, i1], nodes_exprs, [o0], None, [0, 1, 2, 3, 4], {2, 3}
    )

    assert "h = np.zeros(2)" in source
    assert "o = np.zeros(1)" in source
    assert "0.123457" in source, "the emitted expression must be rounded to precision=6"
    assert "h[0]" in source and "o[0]" in source
    assert "h[1]" not in source, "a useless node must be skipped"
    assert "nan" not in source, "a useless node's NaN expression must not reach the source"

    namespace = {
        name: getattr(np, name) for name in dir(np) if not name.startswith("_")
    }
    namespace["np"] = np
    namespace["i"] = np.array([0.5, 1.25])
    exec(source, namespace)  # noqa: S102 - executing generated source is the point

    expected_h0 = sp.lambdify((i0, i1), nodes_exprs[h0], modules="numpy")(*namespace["i"])
    closed_o0 = nodes_exprs[o0].subs(h0, nodes_exprs[h0])
    expected_o0 = sp.lambdify((i0, i1), closed_o0, modules="numpy")(*namespace["i"])
    np.testing.assert_allclose(namespace["h"][0], expected_h0, rtol=1e-5)
    np.testing.assert_allclose(namespace["o"][0], expected_o0, rtol=1e-5)
    assert namespace["h"][1] == 0.0, "the skipped node keeps its zero slot"
