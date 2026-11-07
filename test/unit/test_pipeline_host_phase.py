"""Unit tests for the Pipeline's *host* setup phase and its backend selection.

What this module proves
-----------------------
``Pipeline.setup`` runs ``algorithm.setup`` / ``problem.setup`` on a **host** phase:
``device=None`` (the setup graphs read the host ``randkey`` and ``evoxc.run`` never
transfers across devices), the device-targeting ``compile_options`` stripped
(``_DEVICE_TARGETING_OPTIONS``), and — this module's subject — a *backend* that can
serve ``Device("cpu", 0)``. ``"xla"`` cannot: it drives a PJRT C API plugin whose
platform follows the installed plugin, so its only device kind here is ``rocm`` and a
CPU build raises. ``Pipeline._host_backend`` therefore maps ``"xla"`` to ``"numpy"``
(``_HOST_BACKEND_FALLBACKS``) and leaves every other backend alone, and
``Pipeline._host_build_options`` drops the whole option mapping for that fallback
because the options describe the requested backend's device and compiler.

The behaviours pinned, one test each:

* :meth:`Pipeline._host_backend` is the documented table — ``"xla"`` → ``"numpy"``,
  ``"iree"``/``"numpy"`` → themselves — and the table holds exactly that one entry;
* :meth:`Pipeline._host_build_options` is ``None`` for the ``xla`` fallback (rather
  than the options with the device keys removed, which the reference interpreter
  would reject) and still strips only ``device``/``target_backends`` for a backend
  that keeps its own host artifact (``iree``);
* :meth:`Pipeline._build_options` forwards the state's *registered*
  ``backend``/``device``/``compile_options`` when they are present and the
  pipeline's own values otherwise, with ``compile_options`` expanded last — the
  generation graph therefore names the requested backend while the host-reporting
  call names the host one;
* ``setup()`` really registers the host backend for the setup phase: a recording
  algorithm sees ``backend == "numpy"``, ``device is None`` and no
  ``compile_options`` for ``Pipeline(backend="xla")``, and ``backend == "iree"``
  with the device-targeting keys stripped for ``Pipeline(backend="iree")``; and
  after ``setup`` the returned state carries the **requested** configuration again,
  so generation 0 and every ``step`` use ``"xla"``/``"iree"`` as asked;
* the whole ``setup()`` for ``backend="xla"`` succeeds with the real plugin absent:
  nothing in the host phase builds an ``xla`` artifact, which is exactly the
  regression (before ``_host_backend`` existed, ``setup`` registered ``"xla"`` and
  the initialisation graph had to compile through the PJRT plugin for a CPU device).

Oracles: the values the recording algorithm receives, the returned state, and the
documented tables — never the implementation's own docstrings. No optional extra is
needed (no plugin, no ``iree`` wheel, no device): the host-phase contract is a
host-side selection, and every graph it builds runs on the numpy interpreter.

Why no execution test on the real ``xla`` leg
---------------------------------------------
A generation on ``xla`` SIGSEGVs the process (``genome/default.py``'s
``_SWEEP_UNSAFE_BACKENDS`` records 2 of 2 trials with the sweep and 2 of 3 with the
loop), so a committed test that ran one would take the whole pytest session down
instead of reporting. The fallback's *effect* is pinned through the recorded setup
state, which is the only thing this module's contract promises.
"""

from __future__ import annotations

import pytest

from tensorneat import Pipeline, algorithm, common, genome, problem
from tensorneat.pipeline import _HOST_BACKEND_FALLBACKS

#: The static payload shapes of the tiny pipeline (mirrors ``test_pipeline_device``).
MAX_NODES = 8
MAX_CONNS = 16
POP_SIZE = 16


def _gene():
    """A configured ``DefaultGenome`` for the tiny XOR pipeline."""
    return genome.DefaultGenome(
        num_inputs=2,
        num_outputs=1,
        max_nodes=MAX_NODES,
        max_conns=MAX_CONNS,
        max_in_degree=4,
        max_depth=8,
        output_transform=common.ACT.sigmoid,
    )


def _pipeline(*, backend="numpy", device=None, compile_options=None, algo=None):
    """A configured (never run) XOR pipeline on ``backend``."""
    if algo is None:
        algo = algorithm.NEAT(
            _gene(), pop_size=POP_SIZE, species_size=4, survival_threshold=0.01
        )
    return Pipeline(
        algo,
        problem.XOR(error_method="mse"),
        seed=0,
        fitness_target=-0.01,
        generation_limit=10,
        backend=backend,
        device=device,
        compile_options=compile_options,
    )


class _Recording(algorithm.NEAT):
    """Record the state the setup phase is handed, then run the real setup."""

    seen = None

    def setup(self, state=None):
        type(self).seen = (
            state.backend,
            state.device,
            None if state.compile_options is None else dict(state.compile_options),
        )
        state = super().setup(state)
        type(self).seen_state = state
        return state


# --------------------------------------------------------------------------- #
# the host backend and its build options
# --------------------------------------------------------------------------- #
def test_host_backend_falls_back_only_for_xla():
    """Invariant: ``_HOST_BACKEND_FALLBACKS`` maps ``xla`` to ``numpy`` and nothing else.

    Oracle: the documented reason — a PJRT plugin serves only its own platform kind,
    so ``"xla"`` owns no CPU device to build the host phase on, while ``numpy`` and
    ``iree`` (llvm-cpu when no device is named) build their own host artifact. The
    table itself is asserted, so adding a second fallback is a deliberate edit here.
    """
    assert _HOST_BACKEND_FALLBACKS == {"xla": "numpy"}, (
        "the documented host-backend fallback table is exactly {'xla': 'numpy'}, "
        f"got {_HOST_BACKEND_FALLBACKS}"
    )
    assert _pipeline(backend="xla")._host_backend() == "numpy"
    assert _pipeline(backend="iree")._host_backend() == "iree"
    assert _pipeline(backend="numpy")._host_backend() == "numpy"
    # A backend the table does not name keeps itself: nothing falls back silently.
    assert _pipeline(backend="tvm")._host_backend() == "tvm"


def test_host_build_options_drop_the_mapping_for_the_xla_fallback():
    """Invariant: the ``xla`` host phase receives no compile options at all.

    Oracle: the options of an ``xla`` pipeline (``target_backends``, ``plugin_path``,
    compiler flags) describe the *plugin's* device and compiler; the reference
    interpreter serves the host phase instead, so every key is dropped — not merely
    the device-targeting pair, which is all an ``iree`` host artifact has to lose.
    The empty-mapping and ``None`` spellings are asserted too, because ``setup``
    registers the result verbatim and ``None`` keeps "no options" distinguishable
    from "an empty mapping".
    """
    xla_options = {
        "target_backends": ["rocm"],
        "plugin_path": "/opt/xla/xla_rocm_plugin.so",
        "iree_compile_args": ["--flag"],
    }
    assert _pipeline(backend="xla", compile_options=xla_options)._host_build_options() is None
    assert _pipeline(backend="xla", compile_options={})._host_build_options() == {}
    assert _pipeline(backend="xla", compile_options=None)._host_build_options() is None

    iree_options = {
        "target_backends": ["rocm"],
        "device": "rocm",
        "iree_compile_args": ["--flag"],
    }
    host = _pipeline(backend="iree", compile_options=iree_options)._host_build_options()
    assert host == {"iree_compile_args": ["--flag"]}, (
        "an iree host artifact loses only the device-targeting keys, got " f"{host}"
    )


# --------------------------------------------------------------------------- #
# the compile-site options read registered state entries
# --------------------------------------------------------------------------- #
def test_build_options_prefer_the_state_and_fall_back_to_the_pipeline():
    """Invariant: ``_build_options(state)`` forwards the state's registered selection.

    Oracle: the two documented call sites — the generation graph passes the real
    state (whose entries are the requested backend/device/options) while a
    host-reporting call passes the host view (``device=None`` plus the host
    backend) — so the registered entries must win, entry by entry, and an
    unregistered one must fall back to the pipeline's own value.

    Assertion: a bare pipeline (no state) yields ``{"backend": "numpy"}``; a state
    registering a backend only yields that backend with no ``device`` key; a
    registered ``device`` is forwarded; a state registering nothing falls back to
    the pipeline's values; and a ``compile_options["device"]`` entry overrides the
    registered device because the mapping is expanded last.
    """
    pipeline = _pipeline(backend="iree", compile_options={"opt_level": 1})

    assert pipeline._build_options() == {"backend": "iree", "opt_level": 1}
    assert pipeline._build_options(common.State()) == {
        "backend": "iree",
        "opt_level": 1,
    }

    registered = common.State(
        backend="xla", device=None, compile_options={"target_backends": ["rocm"]}
    )
    assert pipeline._build_options(registered) == {
        "backend": "xla",
        "target_backends": ["rocm"],
    }, "a registered backend/compile_options must win over the pipeline's"

    with_device = common.State(backend="numpy", device="cpu", compile_options=None)
    assert pipeline._build_options(with_device) == {"backend": "numpy", "device": "cpu"}

    overridden = common.State(
        backend="iree", device="cpu", compile_options={"device": "rocm"}
    )
    assert pipeline._build_options(overridden)["device"] == "rocm", (
        "compile_options must be expanded last, so its 'device' entry overrides "
        "the registered one"
    )


# --------------------------------------------------------------------------- #
# setup registers the host phase, then restores the requested configuration
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "backend, expected_options",
    [
        ("xla", None),
        ("iree", {"opt_level": 1}),
        ("numpy", {"opt_level": 1}),
    ],
    ids=["xla", "iree", "numpy"],
)
def test_setup_registers_the_host_backend_for_the_setup_graphs(
    backend, expected_options
):
    """Invariant: the setup phase sees the host backend, no device and no device keys.

    Oracle: the recorded state plus the returned state. ``algorithm.setup`` builds
    its initialisation graph from ``state``'s own entries, so the backend the
    recording algorithm sees is the one the graph would compile with. For ``xla``
    that must be the reference interpreter and the whole option mapping must be gone
    (the plugin's keys are meaningless to numpy); for a backend that keeps its own
    host artifact the name stays and only ``device``/``target_backends`` disappear.

    Assertion: the recorded ``(backend, device, compile_options)`` triple; then the
    state returned by ``setup`` carries the *requested* backend and options again, a
    populated payload, and — through ``_build_options(state)`` — the requested
    backend for generation 0. That last step is what shows the host phase is
    confined to ``setup`` and never leaks into ``step``.
    """
    requested = {"opt_level": 1}
    if backend == "iree":
        requested = {"opt_level": 1, "target_backends": ["rocm"]}
    algo = _Recording(_gene(), pop_size=POP_SIZE, species_size=4, survival_threshold=0.01)
    pipeline = _pipeline(
        backend=backend, compile_options=requested, algo=algo
    )

    state = pipeline.setup()

    seen_backend, seen_device, seen_options = _Recording.seen
    assert seen_backend == pipeline._host_backend()
    assert seen_backend == ("numpy" if backend == "xla" else backend)
    assert seen_device is None, "the setup graphs must stay on the CPU host"
    assert seen_options == expected_options, (
        f"{backend}: the setup phase was handed {seen_options} instead of "
        f"{expected_options}"
    )

    assert state.backend == backend, "setup must restore the requested backend"
    assert state.device is None
    assert state.compile_options is requested
    assert len(tuple(state.pop)) == 4, "the setup phase must leave a populated payload"
    assert int(state.pop[0].shape[0]) == POP_SIZE
    assert pipeline._build_options(state)["backend"] == backend
    assert pipeline._build_options(_Recording.seen_state)["backend"] == seen_backend


def test_xla_setup_builds_only_on_the_host_backend(monkeypatch):
    """Invariant: ``Pipeline(backend="xla").setup()`` builds no ``xla`` artifact.

    Oracle: a spy on ``evoxc.build`` — the single compile site every setup kernel
    goes through — recording the backend of every build the host phase performs.
    Before ``_host_backend`` existed, ``setup`` registered the requested ``"xla"``
    and the initialisation graph had to compile through the PJRT plugin for a
    ``Device("cpu", 0)``, which is exactly what raises on a box whose plugin serves
    only its own platform. The spy proves the fallback instead of depending on
    whether this machine happens to have a plugin installed, so the test is not
    skipped anywhere.

    Assertion: every build of the setup phase named the reference interpreter, the
    returned state carries the requested ``xla`` backend again, the generation
    counter is 0 and the payload has the documented four arrays with the population
    axis — i.e. the initialisation really ran, on the host backend.
    """
    import evoxc

    backends = []
    real_build = evoxc.build

    def spying_build(*args, **options):
        """Record the backend of one ``evoxc.build``, then build it for real."""
        backends.append(options.get("backend", "numpy"))
        return real_build(*args, **options)

    monkeypatch.setattr(evoxc, "build", spying_build)

    pipeline = _pipeline(backend="xla")
    state = pipeline.setup()

    assert backends == ["numpy"], (
        "the xla host phase must build only on the reference interpreter, built on "
        f"{backends}"
    )
    assert state.backend == "xla"
    for name in ("node_keys", "node_attrs", "conn_keys", "conn_attrs"):
        array = getattr(state.pop, name)
        assert int(array.shape[0]) == POP_SIZE, f"{name} lost its population axis"
    assert float(common.to_numpy(state.generation)) == 0.0
