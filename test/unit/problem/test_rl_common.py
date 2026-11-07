"""Unit tests for the host-side RL helpers of ``problem/rl/_common.py``.

What this module proves
-----------------------
Every RL environment repeats the same constructor validation, and that
validation lives in :func:`require_finite` and its three refinement helpers, so
the *knob list* of each environment is the contract this module pins:

* a **float** knob (physics constant, threshold, ``fitness_target``) that is not
  a finite number -- ``NaN``, ``+inf``, ``-inf``, a string, ``None``, a sequence
  -- raises ``TensorNEATError`` whose message is prefixed with the owning
  environment and names the knob;
* an **int** knob (``max_step``, ``repeat_times``) that is not a Python/numpy
  ``int`` -- including ``bool``, which *is* an ``int`` in Python but never a
  horizon -- raises the same way.

Why it matters numerically: ``NaN`` compares false against every ``<``/``>``
gate, so before ``require_finite`` a ``gravity = nan`` would pass every range
check and then poison every episode through the dynamics, and a
``fitness_target = nan`` would never terminate a run. The failure would surface
as ``-inf`` fitness on the whole population with no diagnostic at all.

Oracles used here (never the code under test)
---------------------------------------------
* the constructor signatures in ``src/tensorneat/problem/rl/*.py``, read as the
  documented knob tables below (one entry per validated knob, so a knob that
  silently stops being validated fails here);
* the ``Raises:`` clauses: ``"must be a finite number"`` for the four float
  helpers and ``"must be an int"`` for the integer one, with the owner prefix
  ``"<Class>: <knob> "`` that the environments pass as ``owner``/``name``.

No graph is built and the backend is irrelevant: every assertion is host-side
constructor behaviour. No ``jax``/``torch``, no network, no GPU.
"""

from __future__ import annotations

import numpy as np
import pytest


#: Every float knob each environment validates, in constructor order.
#: ``fitness_target`` is last everywhere because it is checked last.
FLOAT_KNOBS = {
    "CartPole": (
        "gravity",
        "masscart",
        "masspole",
        "length",
        "force_mag",
        "dt",
        "x_threshold",
        "theta_threshold",
        "reset_scale",
        "fitness_target",
    ),
    "Pendulum": (
        "gravity",
        "mass",
        "length",
        "max_speed",
        "max_torque",
        "dt",
        "fitness_target",
    ),
    "MountainCarContinuous": (
        "min_position",
        "max_position",
        "max_speed",
        "goal_position",
        "goal_velocity",
        "power",
        "gravity",
        "fitness_target",
    ),
}

#: Every integer knob each environment validates.
INT_KNOBS = {
    "CartPole": ("max_step", "repeat_times"),
    "Pendulum": ("max_step", "repeat_times"),
    "MountainCarContinuous": ("max_step", "repeat_times"),
}

#: ``(value, id)`` for numbers that are not finite. ``NaN`` is the important
#: one: every ``<``/``>`` gate is false against it, so a range check alone would
#: pass it.
NON_FINITE = (
    (float("nan"), "nan"),
    (float("inf"), "inf"),
    (float("-inf"), "neg_inf"),
)
#: ``(value, id)`` for values that are not numbers at all. A *numeric string* is
#: deliberately absent: ``float("1.5") == 1.5``, so ``require_finite`` accepts it
#: (documented: "anything ``float`` can convert, except ``bool``").
NON_NUMERIC = (
    ("abc", "str"),
    (None, "none"),
    ([1.0], "list"),
)

#: ``(value, id)`` values rejected by ``require_positive_int``: no integer, plus
#: the ``bool`` trap.
NON_INTEGERS = (
    (2.5, "float"),
    ("3", "str"),
    (None, "none"),
    (float("nan"), "nan"),
    (float("inf"), "inf"),
    (True, "bool_true"),
    (False, "bool_false"),
)


def _float_cases():
    """Flatten :data:`FLOAT_KNOBS` into ``(class name, knob, value)`` parameters.

    ``fitness_target = None`` is skipped: it is the documented "disable the
    check" spelling (asserted accepted in
    :func:`test_fitness_target_none_disables_the_check`), not a bad value.
    """
    cases = []
    for class_name, knobs in FLOAT_KNOBS.items():
        for knob in knobs:
            for value, value_id in NON_FINITE + NON_NUMERIC:
                if knob == "fitness_target" and value is None:
                    continue
                cases.append(
                    pytest.param(
                        class_name,
                        knob,
                        value,
                        id=f"{class_name}-{knob}-{value_id}",
                    )
                )
    return cases


def _int_cases():
    """Flatten :data:`INT_KNOBS` into ``(class name, knob, value)`` parameters."""
    cases = []
    for class_name, knobs in INT_KNOBS.items():
        for knob in knobs:
            for value, value_id in NON_INTEGERS:
                cases.append(
                    pytest.param(
                        class_name,
                        knob,
                        value,
                        id=f"{class_name}-{knob}-{value_id}",
                    )
                )
    return cases


@pytest.mark.parametrize("class_name, knob, value", _float_cases())
def test_every_float_knob_rejects_non_finite_and_non_numeric_values(
    problem, common, class_name, knob, value
):
    """Invariant: each float knob rejects ``NaN``/``±inf``/non-numbers with an owner-prefixed message.

    Oracle: :func:`require_finite`'s contract -- "a knob that is not a number at
    all ... raises the same ``TensorNEATError`` as one that is a number but not
    finite ... instead of escaping as a bare ``TypeError`` or ``ValueError``" --
    applied to every knob of every environment. The message must *start* with
    ``"<owner>: <knob> must be a finite number"``, so the two halves of the
    prefix are asserted and a message about a different knob (or none) fails.

    Notes:
        ``fitness_target`` is included: it is checked only when not ``None``, and
        a non-finite target would never terminate a run. The full cross product
        is asserted rather than a sample, because a knob that lost its
        ``require_finite`` call is exactly the regression this module exists for.
    """
    factory = getattr(problem, class_name)

    with pytest.raises(common.TensorNEATError) as excinfo:
        factory(**{knob: value})

    message = str(excinfo.value)
    assert message.startswith(f"{class_name}: {knob} must be a finite number"), message


@pytest.mark.parametrize(
    "class_name", sorted(FLOAT_KNOBS), ids=sorted(FLOAT_KNOBS)
)
def test_fitness_target_none_disables_the_check(problem, class_name):
    """Invariant: ``fitness_target = None`` is the documented "no target" spelling.

    Oracle: the constructor docstring -- "``None`` disables the check" -- for
    every environment, so the ``None`` rejection asserted for the other float
    knobs is shown to be a *value* rule and not a blanket "no ``None``" rule.
    A finite target inside the domain is accepted too, which is what makes
    ``None`` a special case rather than the only legal value.
    """
    factory = getattr(problem, class_name)

    assert factory(fitness_target=None).fitness_target is None
    assert factory(fitness_target=0.0).fitness_target == 0.0


@pytest.mark.parametrize("class_name, knob, value", _int_cases())
def test_every_int_knob_rejects_non_integers(
    problem, common, class_name, knob, value
):
    """Invariant: each integer knob rejects floats, strings, ``None`` and ``bool``.

    Oracle: :func:`require_positive_int`'s contract -- "if ``value`` is not an
    ``int`` ... ``bool`` is rejected explicitly: ``True`` is an ``int`` in Python
    but never a meaningful horizon or repeat count". The message starts with
    ``"<owner>: <knob> must be an int"``.

    Notes:
        ``nan``/``inf`` are floats, so they land here rather than in the
        finiteness branch: an integer knob never reaches ``require_finite``.
        ``True``/``False`` are asserted separately from the other non-integers
        because they are the only values an ``isinstance(value, int)`` check
        would wrongly accept.
    """
    factory = getattr(problem, class_name)

    with pytest.raises(common.TensorNEATError) as excinfo:
        factory(**{knob: value})

    message = str(excinfo.value)
    assert message.startswith(f"{class_name}: {knob} must be an int"), message


def test_require_finite_converts_numbers_and_rejects_the_documented_inputs(problem, common):
    """Invariant: ``require_finite`` returns a Python ``float`` and rejects bool/NaN/±inf/non-numbers.

    Oracle: its own ``Raises:`` clause plus the conversion contract ("the value
    as a Python ``float``", "a ``bool`` is rejected explicitly"). Checked
    directly, because the environments only ever pass literals: a numpy scalar
    must be accepted and converted, and the returned type must be ``float`` so
    downstream comparisons are not numpy-bool objects.
    """
    require_finite = problem.rl._common.require_finite

    converted = require_finite("Owner", "knob", np.float32(1.5))
    assert isinstance(converted, float) and converted == 1.5
    assert require_finite("Owner", "knob", 3) == 3.0

    # A numeric *string* is accepted: the helper's job is one conversion, so
    # ``float``'s own parsing rules are the contract.
    assert require_finite("Owner", "knob", "1.5") == 1.5

    for value in (True, False, float("nan"), float("inf"), float("-inf"), "abc", None, [1.0]):
        with pytest.raises(common.TensorNEATError) as excinfo:
            require_finite("Owner", "knob", value)
        assert str(excinfo.value) == (
            f"Owner: knob must be a finite number, got {value!r}"
        )


def test_require_positive_and_non_negative_split_only_at_zero(problem, common):
    """Invariant: ``require_positive`` accepts ``> 0``; ``require_non_negative`` also accepts ``0``.

    Oracle: the two helpers' ``Raises:`` clauses -- "must be positive" and "must
    be non-negative" -- on a small grid around zero, so the documented
    asymmetry is pinned from both sides (``0.0`` is the only value that
    distinguishes them) and a non-finite argument reports the *finiteness*
    failure, not the sign one.
    """
    require_positive = problem.rl._common.require_positive
    require_non_negative = problem.rl._common.require_non_negative

    for value in (1e-9, 1.0, np.float64(2.5)):
        assert require_positive("Owner", "knob", value) == float(value)
        assert require_non_negative("Owner", "knob", value) == float(value)

    assert require_non_negative("Owner", "knob", 0.0) == 0.0
    with pytest.raises(common.TensorNEATError, match="must be positive, got 0.0"):
        require_positive("Owner", "knob", 0.0)

    for helper, fragment in (
        (require_positive, "must be positive, got -1.0"),
        (require_non_negative, "must be non-negative, got -1.0"),
    ):
        with pytest.raises(common.TensorNEATError, match=fragment):
            helper("Owner", "knob", -1.0)
        # A non-finite value is caught by the finiteness rule, before the sign.
        with pytest.raises(common.TensorNEATError, match="must be a finite number"):
            helper("Owner", "knob", float("nan"))


def test_require_positive_int_accepts_numpy_integers_and_returns_int(problem, common):
    """Invariant: numpy integers are accepted and converted; ``0``/negatives are rejected.

    Oracle: the helper's contract -- "a Python or numpy integer", returning "the
    value as a Python ``int``", and rejecting non-positive values. The returned
    type matters: the value becomes a static shape/horizon, and a ``np.int64``
    leaking into a graph constant is a dtype surprise on the stablehlo path.
    """
    require_positive_int = problem.rl._common.require_positive_int

    for value in (1, np.int32(4), np.int64(7)):
        converted = require_positive_int("Owner", "knob", value)
        assert type(converted) is int
        assert converted == int(value)

    for value, fragment in ((0, "must be positive, got 0"), (-3, "must be positive, got -3")):
        with pytest.raises(common.TensorNEATError, match=fragment):
            require_positive_int("Owner", "knob", value)


def test_to_numpy_unwraps_evoxc_tensors_and_coerces_the_rest(problem, evoxc):
    """Invariant: ``to_numpy`` returns the data of an ``evoxc.Tensor`` and ``np.asarray`` of anything else.

    Oracle: the helper's documented split -- "a concrete ``evoxc.Tensor``
    exposes its data through ``numpy()`` ... while everything else is coerced
    with ``np.asarray``". This is the accessor every RL ``show`` uses to print a
    rollout, so both branches are asserted: a wrapper that returned the
    ``Tensor`` itself would format as an object, and one that called
    ``np.asarray`` on a symbolic value would fail outside a trace.
    """
    to_numpy = problem.rl._common.to_numpy

    tensor = evoxc.from_numpy(np.array([1.0, 2.0], dtype=np.float32))
    unwrapped = to_numpy(tensor)
    assert isinstance(unwrapped, np.ndarray)
    np.testing.assert_array_equal(unwrapped, np.array([1.0, 2.0], dtype=np.float32))

    host = np.array([[1, 2], [3, 4]], dtype=np.int32)
    np.testing.assert_array_equal(to_numpy(host), host)

    scalar = to_numpy(np.float32(1.5))
    assert isinstance(scalar, np.ndarray) and scalar.shape == ()
    assert float(scalar) == 1.5


# --------------------------------------------------------------------------- #
# the shared fixed-horizon trip condition of every rollout loop
# --------------------------------------------------------------------------- #
#: The static horizon the condition table is built against.
HORIZON = 5
#: ``(counter, other slot)`` pairs, chosen so a wrong carry slot cannot pass: at
#: ``counter == 0`` the condition is True while ``other == HORIZON`` makes the
#: *other* slot's comparison False, and the boundary rows pin strict ``<`` (the loop
#: runs exactly ``max_step`` iterations, never one more).
COUNTER_CASES = ((0, HORIZON), (HORIZON - 1, HORIZON), (HORIZON, 0), (HORIZON + 1, 0))


def test_horizon_cond_is_the_removed_per_environment_loop_condition(run_in_graph, evoxc):
    """Invariant: ``horizon_cond`` reproduces ``less(carry[index], horizon)`` exactly.

    Oracle: the closure each environment used to spell for itself (``_loop_cond`` in
    ``cartpole.py``/``mountaincar.py``/``pendulum.py`` before the dedup) —
    ``cond(carry) = enp.less(carry[index], consts.horizon)`` at index ``3`` for the
    shared four-leaf carry and ``1`` for ``Pendulum``'s packed ``(row, step)`` carry.
    Both are transcribed here, on the same operands, so the assertion is against the
    old behaviour rather than against a second copy of the new one.

    Assertion: for every ``(counter, other)`` case the host-carry condition equals
    ``counter < HORIZON`` and the packed one equals the same value read from its own
    slot; the strict boundary (``counter == HORIZON`` ⇒ False) is included; and at
    ``counter == 0`` the *other* slot's comparison is ``False`` while the condition is
    ``True``, so a condition reading the wrong slot cannot pass. The load-bearing
    semantic is asserted too: the carry's ``alive`` slot is all-``False`` (every
    episode dead) and the loop still asks for the next step, because the horizon is
    static config and never "is any episode still alive".

    Notes:
        ``consts`` is a stand-in bundle: the only entry ``horizon_cond`` reads is
        ``horizon``, so a ``SimpleNamespace`` pins that read — a helper reaching for
        a second field would raise here instead of silently using a shipped bundle.
    """
    from types import SimpleNamespace

    from evoxc import numpy as enp

    from tensorneat.problem.rl._common import horizon_cond

    from tensorneat.problem.rl import _common

    assert "horizon_cond" in _common.__all__, (
        "horizon_cond must stay on problem.rl._common's public surface"
    )
    assert _common.horizon_cond is horizon_cond

    def traced():
        """Every case of the table in one graph: condition, old form, wrong slot."""
        horizon = enp.full((), HORIZON, dtype=evoxc.int32)
        consts = SimpleNamespace(horizon=horizon)
        dead = enp.zeros((3,), dtype=evoxc.bool_)
        out = []
        for counter, other in COUNTER_CASES:
            step = enp.full((), counter, dtype=evoxc.int32)
            alt = enp.full((), other, dtype=evoxc.int32)
            host_carry = (dead, alt, alt, step)
            packed_carry = (alt, step)
            out.append(horizon_cond(consts)(host_carry))
            out.append(enp.less(host_carry[3], consts.horizon))  # the old spelling
            out.append(enp.less(alt, consts.horizon))  # what slot 1 would answer
            out.append(horizon_cond(consts, 1)(packed_carry))
            out.append(enp.less(packed_carry[1], consts.horizon))  # the old packed form
        return tuple(out)

    results = [bool(value) for value in run_in_graph(traced)]
    assert len(results) == 5 * len(COUNTER_CASES)
    for index, (counter, other) in enumerate(COUNTER_CASES):
        cond3, old3, wrong3, cond1, old1 = results[5 * index : 5 * index + 5]
        expected = counter < HORIZON
        assert cond3 is expected, (
            f"counter={counter}: horizon_cond(consts) answered {cond3}, expected "
            f"{expected}"
        )
        assert cond1 is expected, (
            f"counter={counter}: horizon_cond(consts, 1) answered {cond1}, expected "
            f"{expected}"
        )
        assert old3 is expected and old1 is expected, "the transcription disagrees"
        assert wrong3 is (other < HORIZON), "the wrong-slot control lost its value"

    first = results[0:5]
    assert first[0] is True, "a dead episode set must not stop the static horizon"
    assert first[2] is False, (
        "the wrong-slot control must differ at counter=0, else the index is unpinned"
    )
    boundary = results[5 * 2 : 5 * 2 + 5]
    assert boundary[0] is False and boundary[3] is False, (
        "the horizon is a strict bound: counter == HORIZON must stop the loop"
    )


# --------------------------------------------------------------------------- #
# every environment builds its rollout from the shared condition
# --------------------------------------------------------------------------- #
def _state(common, **entries):
    """A host ``State`` with the given entries (module ``State()`` otherwise)."""
    return common.State(**entries)


@pytest.mark.parametrize(
    "class_name, output_dim, kernel_mode, expected_index",
    [
        ("CartPole", 2, None, 3),
        ("Pendulum", 1, None, 3),
        ("Pendulum", 1, "accelerator", 1),
        ("MountainCarContinuous", 1, None, 3),
    ],
    ids=["cartpole", "pendulum_host", "pendulum_packed", "mountaincar"],
)
def test_each_environment_builds_its_loop_from_horizon_cond(
    problem, common, enp, evoxc, runner, schedule_type, fake_schedule,
    monkeypatch, class_name, output_dim, kernel_mode, expected_index,
):
    """Invariant: all three rollouts ask ``_common.horizon_cond`` at their own counter slot.

    Oracle: the recorded call. Each environment module holds the *module-local* name
    imported from ``._common``, so replacing that name with a recorder that delegates
    to the real helper proves which helper the rollout built its ``while_loop`` trip
    condition from — a module that reintroduced a private ``_loop_cond`` would leave
    the recorder empty.

    Assertion: exactly one call per rollout, at the documented slot — ``3`` for the
    shared four-leaf ``(obs, alive, total, step)`` carry of all three environments and
    ``1`` for ``Pendulum``'s packed ``(row, step)`` accelerator carry — and the
    rollout still produces one finite fitness per individual.
    """
    module = getattr(problem.rl, {
        "CartPole": "cartpole",
        "Pendulum": "pendulum",
        "MountainCarContinuous": "mountaincar",
    }[class_name])
    calls = []
    real = module.horizon_cond

    def recorder(consts, step_index=3):
        """Record the slot, then hand back the real trip condition."""
        calls.append(step_index)
        return real(consts, step_index)

    monkeypatch.setattr(module, "horizon_cond", recorder)
    problem_object = getattr(problem, class_name)(max_step=4, repeat_times=1)
    entries = {} if kernel_mode is None else {"kernel_mode": kernel_mode}
    state = _state(common, **entries)
    schedule = fake_schedule(2)

    def act_fn(batch):
        """A zero-logit policy: finite on every step, no NaN poison."""
        return enp.zeros((int(batch.shape[0]), output_dim), dtype=evoxc.float32)

    def kernel(*arrays):
        """One ``evaluate`` call on the fake schedule."""
        return problem_object.evaluate(
            state, evoxc.random.key(0), act_fn, schedule_type(*arrays)
        )

    fitness = runner.run_defn(kernel, *tuple(schedule))

    assert calls == [expected_index], (
        f"{class_name} ({kernel_mode or 'host'}) built its trip condition at "
        f"{calls} instead of [{expected_index}]"
    )
    assert fitness.shape == (2,)
    assert np.isfinite(fitness).all(), f"fitness came back non-finite: {fitness}"
