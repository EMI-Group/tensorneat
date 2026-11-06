"""``Pipeline(host_payload=...)``: the per-generation payload copy is an opt-out whose opt-in paths agree (marker ``slow``).

What this module proves
-----------------------
``Pipeline.step`` returns ``(new_state, previous_pop, fitnesses)``, and
``previous_pop`` is the gene payload *the generation evaluated*: the population
on axis 0, four arrays. Copying it to host memory is a per-generation
device-to-host transfer, so the constructor flag ``host_payload`` (and the
per-call ``step(state, host_payload=...)`` override) decides whether that copy
is paid. On the numpy reference interpreter both spellings compute the *same*
numbers -- the flag only changes what the host is handed -- which is exactly the
contract this module pins:

* **The opt-out changes nothing observable in the evolution.** A
  ``host_payload=False`` run and the default ``host_payload=True`` run of the
  same seed agree array-for-array on the ``(pop_size,)`` fitness and on the four
  payload fields of every generation, with the documented field names, shapes,
  dtypes and one population axis, and both report the same single build.
* **``materialize_payload`` is the explicit host step.** Applied to the payload
  an opt-out run returns, it produces the four arrays the default-mode run
  returned for the same generation, field for field. Applied to a payload that is
  *already* host numpy (the default-mode case) it is a strict no-op: the very
  same ``GenomeTensors`` object comes back and every field keeps its identity.
* **A missing payload is a ``TensorNEATError``.** ``materialize_payload(None)``
  is a real caller mistake (the step was told not to materialise it) and must be
  reported in the package's own error type, naming both the constructor flag that
  turned it off and the per-call escape.
* **The opt-out really hands back the backend's own leaves.** With
  ``host_payload=False`` every returned field *is* the pre-step ``state.pop``
  leaf (object identity, all four): on the numpy interpreter those leaves are
  already host arrays, so no copy is made. Default mode instead returns four
  fresh host ``ndarray`` objects. The negative witness is a fake device-resident
  leaf: ``materialize_payload`` must transfer each field to
  ``evoxc.Device("cpu", 0)`` exactly once and never read ``.numpy()`` off device
  memory.
* **The per-call override follows the caller.** ``step(state,
  host_payload=True)`` returns fresh host arrays on an opt-out pipeline,
  ``step(state, host_payload=False)`` returns the pre-step leaves on a default
  pipeline, and ``host_payload=None`` follows the constructor on both. Every
  spelling produces the same fitness as a fresh same-seed reference run.

The oracle
----------
The default-mode run of the same seed and configuration. Two (or three) fresh
pipelines are built with the real constructors from the one configuration
below; ``Pipeline.step`` is deterministic given ``(seed, config, shapes,
state)``, so stepping the *same* state with different payload flags must return
the same fitness. Comparisons are tolerance-free -- ``equal_nan=True`` on the
float fields, exact on the int fields -- because ``host_payload`` never touches
arithmetic.

The ``host_payload`` contract
-----------------------------
``previous_pop`` is the payload *as the backend left it*. On the numpy
interpreter that is already host numpy, so the opt-out hands back the identical
``evoxc.Tensor`` wrappers the state carried, and materialising one is a no-op.
A device backend hands back device-resident tensors instead; that is why the
copy is opt-out rather than always-on, and why ``materialize_payload`` -- the
single explicit device-to-host step -- exists for the callers that need a host
array. The fake ``_DeviceTensor`` below is the GPU-free witness of that second
leg.

Budgets (measured, never asserted)
----------------------------------
``pop_size=16`` and two generations per run on XOR with ``fitness_target=1.0``
(unreachable: XOR fitness is ``-mse <= 0``), so the whole budget is consumed and
no convergence is claimed. Every run compiles once; the module is a few seconds
on the numpy reference interpreter and never reads a clock.
"""

from __future__ import annotations

import numpy as np
import pytest

from .conftest import TINY_CONFIG

pytestmark = [pytest.mark.slow]

#: The four gene-payload field names, in ``GenomeTensors`` order.
POP_FIELDS = ("node_keys", "node_attrs", "conn_keys", "conn_attrs")

#: The population width the pipelines below are built with.
POP_SIZE = TINY_CONFIG["pop_size"]  # 16

#: Generations every comparison run consumes: one initialisation plus one bred
#: generation, so both the setup payload and a ``tell``-driven one are covered.
GENERATIONS = 2

#: The documented payload shapes of a :data:`TINY_CONFIG` genome.
EXPECTED_SHAPES = {
    "node_keys": (POP_SIZE, TINY_CONFIG["max_nodes"]),  # (16, 8)
    "node_attrs": (POP_SIZE, TINY_CONFIG["max_nodes"], 4),  # (16, 8, 4)
    "conn_keys": (POP_SIZE, TINY_CONFIG["max_conns"], 2),  # (16, 16, 2)
    "conn_attrs": (POP_SIZE, TINY_CONFIG["max_conns"], 1),  # (16, 16, 1)
}

#: The documented payload dtypes: keys are int32, attributes float32.
EXPECTED_DTYPES = {
    "node_keys": np.int32,
    "node_attrs": np.float32,
    "conn_keys": np.int32,
    "conn_attrs": np.float32,
}


def _pipeline(tn, *, host_payload):
    """Build a fresh XOR pipeline of the one configuration, no run yet.

    Args:
        tn: The imported ``tensorneat`` package (the ``tn`` fixture).
        host_payload (bool): The constructor flag under test. ``tiny_pipeline``
            deliberately has no such knob (:data:`TINY_CONFIG` mirrors only the
            documented compile keywords), so the pipeline is spelled out here and
            two runs differ in exactly this one argument.

    Returns:
        A configured ``Pipeline`` with ``seed=0``, an unreachable
        ``fitness_target=1.0`` (XOR fitness is ``-mse <= 0``) and a fixed
        ``generation_limit``, so calling :func:`_run` consumes the whole budget.
    """
    neat = tn.algorithm.NEAT(
        genome=tn.genome.DefaultGenome(
            num_inputs=TINY_CONFIG["num_inputs"],
            num_outputs=TINY_CONFIG["num_outputs"],
            max_nodes=TINY_CONFIG["max_nodes"],
            max_conns=TINY_CONFIG["max_conns"],
            max_in_degree=TINY_CONFIG["max_in_degree"],
            max_depth=TINY_CONFIG["max_depth"],
            output_transform=tn.common.ACT.sigmoid,
        ),
        pop_size=POP_SIZE,
        species_size=TINY_CONFIG["species_size"],
    )
    problem = tn.problem.XOR(error_method=TINY_CONFIG["error_method"])
    return tn.Pipeline(
        neat,
        problem,
        seed=TINY_CONFIG["seed"],
        fitness_target=1.0,
        generation_limit=GENERATIONS,
        backend="numpy",
        is_detailed=False,
        host_payload=host_payload,
    )


def _payload_arrays(runner, payload):
    """Unwrap one payload into host numpy arrays, keyed by field name.

    Args:
        runner: The ``support.runner`` module (``to_numpy`` unwraps the
            ``evoxc.Tensor`` leaves; ``np.asarray`` on a tensor leaves an object
            array, so every comparison below goes through this helper).
        payload: A ``GenomeTensors`` (or any four-element payload sequence), the
            population on axis 0, as ``step`` returned it or ``state.pop`` holds
            it.

    Returns:
        A ``{field name: ndarray}`` mapping in documented order.

    Raises:
        AssertionError: If the payload's field names are not :data:`POP_FIELDS`.
    """
    arrays = runner.to_numpy(payload)
    names = tuple(getattr(arrays, "_fields", POP_FIELDS))
    assert names == POP_FIELDS, (
        f"a gene payload must carry the four GenomeTensors fields {POP_FIELDS}, "
        f"got {names}"
    )
    return dict(zip(names, (np.asarray(array) for array in arrays)))


def _assert_payload_contract(arrays, label):
    """Assert the documented names/shapes/dtypes of one payload dictionary.

    Args:
        arrays: The ``{field name: ndarray}`` mapping :func:`_payload_arrays`
            produced.
        label: A short description of where the payload came from, quoted in a
            failure message.

    Raises:
        AssertionError: If a field has the wrong shape or dtype, or the four
            fields disagree about the population axis.
    """
    assert tuple(arrays) == POP_FIELDS, f"{label}: unexpected field order {tuple(arrays)}"
    for name, expected_shape in EXPECTED_SHAPES.items():
        array = arrays[name]
        assert array.shape == expected_shape, (
            f"{label}: payload field {name} has shape {array.shape}, expected "
            f"{expected_shape} (population on axis 0)"
        )
        assert array.dtype == EXPECTED_DTYPES[name], (
            f"{label}: payload field {name} has dtype {array.dtype}, expected "
            f"{EXPECTED_DTYPES[name]}"
        )
    leaders = {arrays[name].shape[0] for name in POP_FIELDS}
    assert leaders == {POP_SIZE}, (
        f"{label}: the four payload fields must share one population axis "
        f"{POP_SIZE}, got leading dimensions {leaders}"
    )


def _assert_same_payload(runner, expected, actual, generation):
    """Assert two payloads agree field for field.

    Args:
        runner: The ``support.runner`` module.
        expected: The reference (default-mode) payload.
        actual: The payload under test, possibly backend-resident.
        generation: The generation number both payloads describe, for messages.

    Raises:
        AssertionError: If a field's shape, dtype or value differs. ``NaN`` is
            the attribute padding sentinel of an unused gene row, so it compares
            equal to itself; no live value is tolerated to differ.
    """
    reference = _payload_arrays(runner, expected)
    candidate = _payload_arrays(runner, actual)
    _assert_payload_contract(reference, f"generation {generation} default payload")
    _assert_payload_contract(candidate, f"generation {generation} opt-out payload")
    for name in POP_FIELDS:
        first, second = reference[name], candidate[name]
        assert first.shape == second.shape and first.dtype == second.dtype, (
            f"generation {generation}: payload field {name} changed shape or "
            f"dtype: {first.shape}/{first.dtype} vs {second.shape}/{second.dtype}"
        )
        floating = np.issubdtype(first.dtype, np.floating)
        assert np.array_equal(first, second, equal_nan=floating), (
            f"generation {generation}: host_payload=False changed payload field "
            f"{name}"
        )


class _DeviceTensor:
    """A payload leaf whose data lives on a non-CPU device.

    The GPU-free stand-in for a device-resident tensor: ``.device.kind`` is not
    ``"cpu"``, so :func:`tensorneat.common.tools.to_numpy` must transfer it with
    ``.to(Device("cpu", 0))`` before reading. ``.to`` records the transfer and
    returns a real CPU tensor; ``.numpy`` raising is the proof that a transfer
    happened first (reading device memory directly is the bug this witness
    catches).
    """

    def __init__(self, array, *, device, evoxc_module):
        self._array = np.asarray(array)
        self.device = device
        self._evoxc = evoxc_module
        self.transfers = []

    def to(self, device):
        """Record the device-to-host transfer and return a host-backed tensor."""
        self.transfers.append(device)
        return self._evoxc.from_numpy(self._array)

    def numpy(self):  # pragma: no cover - the transfer must happen first
        """Fail loudly: host memory was never materialised on this leaf."""
        raise AssertionError("to_numpy called .numpy() on device memory")


def test_an_opt_out_run_matches_the_default_run(tn, runner):
    """Invariant: ``host_payload=False`` changes neither fitness nor payload.

    Oracle: the default-mode run of the same seed and configuration, stepped
    side by side. Both runs consume :data:`GENERATIONS` generations; every
    per-generation fitness vector and all four payload fields must agree exactly
    (``equal_nan=True`` on the float fields). The payload contract -- field
    names, ``(P, 8)``/``(P, 8, 4)``/``(P, 16, 2)``/``(P, 16, 1)`` shapes, int32
    keys and float32 attributes -- holds for both spellings, and each pipeline
    compiled exactly once for the shared shape signature.
    """
    default = _pipeline(tn, host_payload=True)
    opt_out = _pipeline(tn, host_payload=False)

    default_state = default.setup()
    opt_out_state = opt_out.setup()

    for generation in range(1, GENERATIONS + 1):
        default_state, default_previous, default_fitness = default.step(default_state)
        opt_out_state, opt_out_previous, opt_out_fitness = opt_out.step(opt_out_state)

        assert default_fitness.shape == opt_out_fitness.shape == (POP_SIZE,), (
            "step must report one fitness per individual on both spellings, got "
            f"{default_fitness.shape} and {opt_out_fitness.shape}"
        )
        assert np.array_equal(default_fitness, opt_out_fitness, equal_nan=True), (
            f"generation {generation}: host_payload=False changed the fitness: "
            f"{default_fitness!r} vs {opt_out_fitness!r}"
        )
        _assert_same_payload(runner, default_previous, opt_out_previous, generation)

    assert default.builds == opt_out.builds == 1, (
        "both spellings share one shape signature, so each must build exactly one "
        f"executable; got default={default.builds}, opt_out={opt_out.builds}"
    )
    final_generation = int(tn.common.tools.to_numpy(opt_out_state.generation))
    assert final_generation == GENERATIONS, (
        f"the opt-out run must consume exactly {GENERATIONS} generations, ended at "
        f"generation {final_generation}"
    )


def test_materialize_payload_reproduces_the_default_payload(tn, runner):
    """``materialize_payload`` is the explicit host step -- and a no-op when already host.

    Oracle: the default-mode payload of the same seed and generation. Applied to
    the payload an opt-out step returns, ``materialize_payload`` yields the four
    arrays the default-mode run returned (field, shape, dtype and value); applied
    to an already-host payload it is the identity -- the very same
    ``GenomeTensors`` object and the very same four arrays come back, so the
    helper never copies a host payload a second time.
    """
    import tensorneat.host_payload as host_payload

    default = _pipeline(tn, host_payload=True)
    opt_out = _pipeline(tn, host_payload=False)
    default_state = default.setup()
    opt_out_state = opt_out.setup()

    default_state, default_previous, _ = default.step(default_state)
    opt_out_state, opt_out_previous, _ = opt_out.step(opt_out_state)

    materialized = host_payload.materialize_payload(opt_out_previous)
    materialized_arrays = dict(
        zip(POP_FIELDS, (np.asarray(field) for field in materialized))
    )
    _assert_payload_contract(materialized_arrays, "materialized opt-out payload")
    default_arrays = _payload_arrays(runner, default_previous)
    for name in POP_FIELDS:
        reference, candidate = default_arrays[name], materialized_arrays[name]
        assert np.array_equal(
            reference, candidate, equal_nan=np.issubdtype(reference.dtype, np.floating)
        ), f"materialize_payload did not reproduce the default payload field {name}"

    noop = host_payload.materialize_payload(default_previous)
    assert noop is default_previous, (
        "materialize_payload must hand an already-host GenomeTensors back "
        "unchanged rather than rebuild it"
    )
    assert all(noop[index] is tuple(default_previous)[index] for index in range(4)), (
        "every field of an already-host payload must keep its identity through "
        "materialize_payload"
    )


def test_materialize_payload_rejects_a_missing_payload(tn):
    """A missing payload is a ``TensorNEATError`` naming the flag and the escape.

    ``Pipeline(host_payload=False)`` hands ``None`` to the reporting paths that
    need the payload, so ``materialize_payload(None)`` must raise the package's
    own error type -- never a bare ``TypeError``/``AttributeError`` -- and name
    both ``host_payload=False`` (what turned it off) and the per-call
    ``step(state, host_payload=True)`` escape.
    """
    import tensorneat.host_payload as host_payload

    with pytest.raises(tn.common.TensorNEATError) as excinfo:
        host_payload.materialize_payload(None)

    message = str(excinfo.value)
    assert "host_payload=False" in message, (
        "the error must name the constructor flag that turned the copy off; "
        f"message was {message!r}"
    )
    assert "host_payload=True" in message and "step(state, host_payload=True)" in message, (
        "the error must name the per-call escape `step(state, host_payload=True)`; "
        f"message was {message!r}"
    )


def test_the_opt_out_hands_back_the_backend_leaves_unchanged(tn):
    """The opt-out returns the state's own leaves; the host copy is a real copy.

    On the numpy reference interpreter ``previous_pop`` with
    ``host_payload=False`` *is* the pre-step ``state.pop``: all four leaves are
    the identical objects, so no host copy was made. Default mode returns four
    fresh host ``ndarray`` objects instead.

    The negative witness is a device-resident leaf: a four-field payload of
    ``_DeviceTensor`` stand-ins goes through ``materialize_payload``, which must
    transfer each field to ``evoxc.Device("cpu", 0)`` exactly once and return the
    four host arrays with the right values -- without ever calling ``.numpy()``
    on device memory.
    """
    import evoxc

    import tensorneat.host_payload as host_payload

    default = _pipeline(tn, host_payload=True)
    opt_out = _pipeline(tn, host_payload=False)
    default_state = default.setup()
    opt_out_state = opt_out.setup()
    default_before = tuple(default_state.pop)
    opt_out_before = tuple(opt_out_state.pop)

    default_state, default_previous, _ = default.step(default_state)
    opt_out_state, opt_out_previous, _ = opt_out.step(opt_out_state)

    for index in range(4):
        assert opt_out_previous[index] is opt_out_before[index], (
            f"host_payload=False must hand back the pre-step state.pop leaf "
            f"{POP_FIELDS[index]} unchanged, got a different object"
        )
        assert default_previous[index] is not default_before[index], (
            f"the default-mode payload leaf {POP_FIELDS[index]} must be a fresh "
            "host object, not the state's leaf"
        )
        assert isinstance(default_previous[index], np.ndarray), (
            f"the default-mode payload leaf {POP_FIELDS[index]} must be a host "
            f"ndarray, got {type(default_previous[index]).__name__}"
        )

    cuda = evoxc.Device("cuda", 0)
    expected = [
        np.full(shape, index, dtype=dtype)
        for index, (shape, dtype) in enumerate(
            zip(EXPECTED_SHAPES.values(), EXPECTED_DTYPES.values())
        )
    ]
    stand_ins = [
        _DeviceTensor(array, device=cuda, evoxc_module=evoxc) for array in expected
    ]

    materialized = host_payload.materialize_payload(stand_ins)

    for index, stand_in in enumerate(stand_ins):
        assert stand_in.transfers == [evoxc.Device("cpu", 0)], (
            f"materialize_payload must transfer {POP_FIELDS[index]} to the CPU "
            f"device exactly once, recorded transfers were {stand_in.transfers}"
        )
        field = np.asarray(materialized[index])
        assert isinstance(field, np.ndarray) and field.dtype == expected[index].dtype, (
            f"materialize_payload must return a host ndarray for "
            f"{POP_FIELDS[index]}, got {type(field).__name__}/{field.dtype}"
        )
        assert np.array_equal(field, expected[index]), (
            f"materialize_payload lost the values of {POP_FIELDS[index]}"
        )


def test_the_per_call_override_follows_the_caller(tn):
    """``step(state, host_payload=...)`` overrides the constructor for that one call.

    ``host_payload=True`` on an opt-out pipeline returns fresh host ``ndarray``
    leaves (not the state's), ``host_payload=False`` on a default pipeline
    returns the pre-step leaves by identity, and ``host_payload=None`` follows
    the constructor on both. Every spelling is stepped from the *same* setup
    state and must report the fitness of a fresh same-seed reference run:
    ``host_payload`` only decides what the host is handed, never what is
    computed.
    """
    opt_out = _pipeline(tn, host_payload=False)
    default = _pipeline(tn, host_payload=True)
    reference = _pipeline(tn, host_payload=True)

    opt_out_state = opt_out.setup()
    before = tuple(opt_out_state.pop)

    _, host_copy, host_fitness = opt_out.step(opt_out_state, host_payload=True)
    for index in range(4):
        assert isinstance(host_copy[index], np.ndarray), (
            f"host_payload=True must materialise {POP_FIELDS[index]} as a host "
            f"ndarray, got {type(host_copy[index]).__name__}"
        )
        assert host_copy[index] is not before[index], (
            f"host_payload=True must copy {POP_FIELDS[index]}, not hand the state's "
            "leaf back"
        )

    _, backend_copy, backend_fitness = opt_out.step(opt_out_state, host_payload=False)
    _, constructor_copy, constructor_fitness = opt_out.step(opt_out_state)
    for label, payload in (
        ("explicit False", backend_copy),
        ("None on an opt-out pipeline", constructor_copy),
    ):
        for index in range(4):
            assert payload[index] is before[index], (
                f"host_payload={label} must hand the pre-step state.pop leaf "
                f"{POP_FIELDS[index]} back unchanged"
            )

    default_state = default.setup()
    default_before = tuple(default_state.pop)

    _, default_off, off_fitness = default.step(default_state, host_payload=False)
    for index in range(4):
        assert default_off[index] is default_before[index], (
            "host_payload=False must override the default constructor and hand the "
            f"pre-step state.pop leaf {POP_FIELDS[index]} back unchanged"
        )
    _, default_none, none_fitness = default.step(default_state)
    for index in range(4):
        assert isinstance(default_none[index], np.ndarray), (
            f"host_payload=None on a default pipeline must materialise "
            f"{POP_FIELDS[index]} as a host ndarray"
        )
        assert default_none[index] is not default_before[index], (
            f"host_payload=None on a default pipeline must copy {POP_FIELDS[index]}"
        )

    reference_state = reference.setup()
    _, _, reference_fitness = reference.step(reference_state)

    for label, fitness in (
        ("host_payload=True", host_fitness),
        ("host_payload=False", backend_fitness),
        ("None on an opt-out pipeline", constructor_fitness),
        ("host_payload=False on a default pipeline", off_fitness),
        ("None on a default pipeline", none_fitness),
    ):
        assert np.array_equal(fitness, reference_fitness, equal_nan=True), (
            f"{label} must compute the same generation as the same-seed reference "
            "run; only the host copy may differ"
        )
