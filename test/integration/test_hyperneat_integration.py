"""``HyperNEAT`` through a whole ``Pipeline``: config fallback, substrate rebuild, checkpoint, CPPN decode (marker ``slow``).

What this module proves
-----------------------
``HyperNEAT`` is a *wrapper*: the evolvable payload is a CPPN owned by the inner
``NEAT``, while the substrate is a fixed host-side geometry.  The three tests here
drive the assembled ``Pipeline`` and pin the three places where that split shows
up in the public contract:

* **The reported configuration is the law that ran.**  ``show_config`` has no
  species table to read on the wrapper itself, so it falls back to
  ``algorithm.neat.species``; ``config.txt`` must record exactly that mapping, so
  a checkpoint names the speciation law (``species_size``,
  ``survival_threshold``, ``compatibility_threshold``, ``max_stagnation``,
  ``genome_elitism``, ``min_species_size``, ``species_elitism``,
  ``spawn_number_change_rate``, ``species_number_calculate_by``) that produced
  ``log.txt``.
* **A substrate change is a new compiled shape.**  The substrate's node/connection
  counts, coordinate tables and ``activate_time`` are traced into the graph, so
  the executable cache key carries
  ``host_payload._substrate_signature``.  Replacing the algorithm with a
  ``HyperNEAT`` built on a different hidden layout must force a second build, and
  the rebuilt executable must evaluate the *same* state exactly like a fresh
  pipeline of the new geometry -- a reused executable would silently score the
  wrong phenotype.
* **A checkpoint round-trips a CPPN and decodes it.**  ``best_genome.npz`` stores
  the four split arrays plus the scalar fitness; loading it with plain
  ``numpy.load`` and re-scoring through the documented kernel triple must
  reproduce that fitness, and ``genome.network_dict`` must turn the evolved CPPN
  into the node/connection dictionaries the visualization and the sympy/codegen
  paths consume.

The oracles
-----------
The configuration the fixtures built (:func:`integration.conftest.hyperneat_pipeline`
over the shared CPPN genome), the static ``(1, max_nodes)`` payload shape
``Pipeline`` documents for ``best_genome``, the recorded npz fitness, and the
fitness of a freshly built pipeline of the same geometry stepping the same state.

Budget (measured, never asserted)
---------------------------------
Two generations at ``pop_size=4`` per pipeline; the substrate-rebuild test builds
two (one whose algorithm is replaced in place, plus a fresh oracle) and each of
the other two builds one.  No test reads a clock.

Why nothing here is a module-scope ``tensorneat`` import
--------------------------------------------------------
``tn`` and the HyperNEAT factories arrive as fixtures resolved through
``support.lazy.require`` (``./conftest.py``), so a broken import names a skip
instead of turning this module into a collection error.  ``FullSubstrate`` and
``host_payload`` are imported inside the tests that need them.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from integration.conftest import (
    HYPERNEAT_HIDDEN_COORS,
    HYPERNEAT_INPUT_COORS,
    HYPERNEAT_OUTPUT_COORS,
    step_once,
)

pytestmark = [pytest.mark.slow]

#: A four-hidden-neuron layout: the same 2-D rows as the default three-hidden
#: substrate plus one more column, so the phenotype geometry (and therefore the
#: substrate half of the shape signature) differs while the CPPN payload does not.
FOUR_HIDDEN_COORS = ((-1.0, 0.0), (-0.33, 0.0), (0.33, 0.0), (1.0, 0.0))

#: The species law the fixture's ``NEAT`` is constructed with: ``species_size=2``
#: and ``survival_threshold=0.1`` explicitly, the rest NEAT's documented defaults.
EXPECTED_SPECIES_LAW = {
    "species_size": 2,
    "survival_threshold": 0.1,
    "compatibility_threshold": 2.0,
    "max_stagnation": 15,
    "genome_elitism": 2,
    "min_species_size": 1,
    "species_elitism": 2,
    "spawn_number_change_rate": 0.5,
    "species_number_calculate_by": "rank",
}

#: Two generations, always consumed (the ``CustomFuncFit`` fitness is ``-mse <= 0``
#: and the fixture's target is unreachable ``1.0``).
TWO_GENERATIONS = 2

#: The CPPN genome statics the fixture builds (``hyperneat_algorithm``).
CPPN_MAX_NODES = 12
CPPN_MAX_CONNS = 24
NODE_ATTR_WIDTH = 4
CONN_ATTR_WIDTH = 1
CPPN_NUM_INPUTS = 4
CPPN_NUM_OUTPUTS = 1

#: The batch-first one-individual shape of ``auto_run``'s ``best_genome``.
EXPECTED_BEST_SHAPES = (
    (1, CPPN_MAX_NODES),
    (1, CPPN_MAX_NODES, NODE_ATTR_WIDTH),
    (1, CPPN_MAX_CONNS, 2),
    (1, CPPN_MAX_CONNS, CONN_ATTR_WIDTH),
)

#: The four arrays a checkpoint stores, in ``GenomeTensors`` order.
SPLIT_PAYLOAD = ("node_keys", "node_attrs", "conn_keys", "conn_attrs")

#: The documented one-genome shape of each stored array (no population axis).
EXPECTED_STORED_SHAPES = {
    "node_keys": (CPPN_MAX_NODES,),
    "node_attrs": (CPPN_MAX_NODES, NODE_ATTR_WIDTH),
    "conn_keys": (CPPN_MAX_CONNS, 2),
    "conn_attrs": (CPPN_MAX_CONNS, CONN_ATTR_WIDTH),
}


def _generation(tn, state) -> int:
    """Read ``state.generation`` (a 0-d ``evoxc`` tensor) through the host accessor.

    Args:
        tn: The imported ``tensorneat`` package (the ``tn`` fixture).
        state: Any ``State`` a ``Pipeline`` returned.

    Returns:
        The generation counter as a Python ``int``.  ``evoxc.Tensor`` has no
        ``__array__``, so the package's documented ``common.tools.to_numpy`` is
        the only supported read.
    """
    return int(tn.common.tools.to_numpy(state.generation))


def _batch_one(payload) -> tuple[np.ndarray, ...]:
    """Promote the four bare per-individual rows of a checkpoint to a one-individual batch.

    Args:
        payload: The ``GenomeTensors`` (or four-field namedtuple) whose arrays are
            one genome each, as ``numpy.load`` returns them.

    Returns:
        The four arrays with a leading population axis of length 1, ready for
        ``evaluate_population`` and the algorithm's batch-first contract.
    """
    return tuple(np.asarray(field)[None, ...] for field in payload)


def test_show_config_reports_the_inner_neat_species_law(tn, hyperneat_pipeline, run_pipeline, tmp_path):
    """Invariant: a HyperNEAT run's configuration, and its ``config.txt``, carry the inner ``NEAT`` speciation law.

    Oracle: the ``NEAT`` the fixture constructs (``species_size=2``,
    ``survival_threshold=0.1``, NEAT's documented defaults for the rest) and
    NEAT's own controller object.  ``HyperNEAT`` owns no species table -- the
    search law lives on ``algorithm.neat`` -- so ``show_config`` must fall back
    to it rather than silently omit the law.

    Assertion: ``show_config()["algorithm"]["class"]`` is ``"HyperNEAT"``; every
    key of :data:`EXPECTED_SPECIES_LAW` is present with exactly the configured
    value; ``pipeline.algorithm`` has no ``species`` attribute while
    ``pipeline.algorithm.neat`` does (the fallback source really exists);
    ``<save_dir>/config.txt`` parses as JSON and its ``"algorithm"`` mapping
    agrees with ``show_config()["algorithm"]`` on those keys -- the checkpoint
    records the law that ran, not a copy written by the test.

    Notes:
        ``config.txt`` is written once by ``setup`` while ``show_config`` is read
        after ``auto_run``; the two must agree because the configuration is
        constructor-level and host-only.  A mismatch would mean the checkpoint
        header no longer describes the run it sits next to.

    Owner: ``pipeline.Pipeline.show_config`` (the wrapper fallback) and
    ``pipeline.Pipeline.setup`` (what it writes).
    """
    pipeline = hyperneat_pipeline(
        is_save=True, save_dir=tmp_path, generation_limit=TWO_GENERATIONS
    )
    state, _best = run_pipeline(pipeline)
    assert _generation(tn, state) == TWO_GENERATIONS, (
        "the unreachable fitness target must consume the two-generation budget, got "
        f"generation {_generation(tn, state)}"
    )

    algorithm_config = pipeline.show_config()["algorithm"]
    assert algorithm_config["class"] == "HyperNEAT", (
        "show_config must name the wrapper algorithm the pipeline drives, got "
        f"{algorithm_config.get('class')!r}"
    )
    for name, expected in EXPECTED_SPECIES_LAW.items():
        assert name in algorithm_config, (
            f"show_config()['algorithm'] omits the speciation law {name!r}; the "
            "wrapper must fall back to algorithm.neat.species instead of dropping it "
            f"(keys: {sorted(algorithm_config)})"
        )
        assert algorithm_config[name] == expected, (
            f"show_config()['algorithm'][{name!r}] must be the configured {expected!r}, "
            f"got {algorithm_config[name]!r}"
        )

    assert not hasattr(pipeline.algorithm, "species"), (
        "HyperNEAT must not own a species table: the test relies on show_config's "
        "documented fallback to the inner NEAT"
    )
    assert hasattr(pipeline.algorithm.neat, "species"), (
        "the inner NEAT must expose the species controller show_config falls back to"
    )

    config_path = tmp_path / "config.txt"
    assert config_path.is_file(), f"Pipeline.setup must write {config_path}"
    stored = json.loads(config_path.read_text())["algorithm"]
    for name in EXPECTED_SPECIES_LAW:
        assert stored.get(name) == algorithm_config[name], (
            f"{config_path} records algorithm[{name!r}]={stored.get(name)!r} but the "
            f"run reported {algorithm_config[name]!r}; the checkpoint header must "
            "describe the law that produced log.txt"
        )
    assert stored["class"] == "HyperNEAT", (
        f"{config_path} must name the HyperNEAT wrapper, got {stored.get('class')!r}"
    )


def test_a_substrate_geometry_change_forces_a_rebuild(tn, hyperneat_pipeline, hyperneat_algorithm):
    """Invariant: replacing the substrate changes the shape signature and rebuilds, and the rebuilt pass is not stale.

    Oracle: ``pipeline.builds`` (real ``evoxc`` builds) and a **fresh** pipeline
    built on the four-hidden substrate stepping the *same* state.  The fresh pass
    has no cached executable for that state, so it is the independent reference
    for what the new geometry must compute.

    Assertion: after one step of a three-hidden pipeline ``builds == 1``; after
    assigning a ``HyperNEAT`` built on :data:`FOUR_HIDDEN_COORS` and stepping the
    same state, ``builds == 2``, ``shape_signature[:6]`` (the CPPN payload half)
    is unchanged while ``shape_signature[6:]`` (the substrate half) differs, and
    the fitness array is exactly equal to the fresh pipeline's -- index for index,
    so a stale executable cannot pass by producing merely finite values.

    Notes:
        The substrate is replaced by assigning a **freshly constructed** algorithm,
        never by mutating ``algorithm.substrate`` in place: ``hyper_genome`` is
        sized at construction, so an in-place geometry change is refused by the
        construction-time shape validation.  The observation is taken through
        ``Pipeline.step`` only; nothing reaches into the cache itself.

    Owner: ``pipeline.Pipeline._shape_signature`` /
    ``pipeline.Pipeline._executable_for`` (the per-signature cache) and
    ``host_payload._substrate_signature`` (the geometry half of the key).
    """
    from tensorneat.algorithm.hyperneat import FullSubstrate

    pipeline = hyperneat_pipeline(generation_limit=TWO_GENERATIONS)
    state = pipeline.setup()
    state, _first_fitness = step_once(pipeline, state)
    assert pipeline.builds == 1, (
        "the first generation of a fresh pipeline must build exactly one executable, "
        f"got {pipeline.builds}"
    )
    first_signature = pipeline.shape_signature
    # Keep the state the rebuilt step consumes: the fresh oracle must score the
    # *same* generation, not the one after it.
    shared_state = state

    substrate = FullSubstrate(
        input_coors=HYPERNEAT_INPUT_COORS,
        hidden_coors=FOUR_HIDDEN_COORS,
        output_coors=HYPERNEAT_OUTPUT_COORS,
    )
    pipeline.algorithm = hyperneat_algorithm(substrate)
    _rebuilt_state, rebuilt_fitness = step_once(pipeline, shared_state)
    assert pipeline.builds == 2, (
        "a substrate geometry change must force exactly one new build (two in total), "
        f"got {pipeline.builds}; the substrate statics are part of the cache key"
    )

    rebuilt_signature = pipeline.shape_signature
    assert rebuilt_signature[:6] == first_signature[:6], (
        "the CPPN payload half of the shape signature must not change when only the "
        f"substrate changes: {rebuilt_signature[:6]} vs {first_signature[:6]}"
    )
    assert rebuilt_signature[6:] != first_signature[6:], (
        "the substrate half of the shape signature must change for a different hidden "
        f"layout: {rebuilt_signature[6:]} vs {first_signature[6:]}"
    )

    fresh = hyperneat_pipeline(hidden_coors=FOUR_HIDDEN_COORS, generation_limit=TWO_GENERATIONS)
    fresh.setup()
    _fresh_state, fresh_fitness = step_once(fresh, shared_state)
    np.testing.assert_array_equal(
        rebuilt_fitness,
        fresh_fitness,
        err_msg=(
            "the rewritten pipeline scored the same state differently from a fresh "
            "pipeline of the same substrate geometry: a reused executable evaluated "
            "the wrong phenotype"
        ),
    )


def test_auto_run_checkpoint_and_cppn_network_dict(tn, hyperneat_pipeline, run_pipeline, evaluate_population, tmp_path):
    """Invariant: a HyperNEAT run checkpoints a loadable CPPN that re-scores to its recorded fitness and decodes.

    Oracle: the static payload shape ``Pipeline`` documents for ``best_genome``,
    the scalar ``fitness`` stored next to the genome, and the evolved CPPN itself
    as ``genome.network_dict`` decodes it.

    Assertion: ``auto_run`` stops at generation 2 with a ``(1, max_nodes)``-style
    one-individual CPPN batch ``((1, 12), (1, 12, 4), (1, 24, 2), (1, 24, 1))``;
    ``genomes/best_genome.npz`` holds ``node_keys``/``node_attrs``/``conn_keys``/
    ``conn_attrs`` plus a scalar finite ``fitness``; the four arrays loaded with
    plain ``numpy.load``, given a leading axis of 1 and re-scored through
    :func:`integration.conftest.evaluate_population` reproduce that fitness
    (``rtol=1e-5, atol=1e-6``); and
    ``pipeline.algorithm.neat.genome.network_dict(state, best)`` returns
    ``nodes``/``conns``/``topo_order``/``topo_layers``/``useful_nodes`` with a
    non-empty node map that holds the CPPN's input and output keys and every
    connection endpoint.

    Notes:
        This is the CPPN visualization path: the evolved payload is a CPPN over
        substrate coordinates, and ``network_dict`` is what ``visualize``,
        ``sympy_func`` and the Python-code generation consume.  Decoding the
        *reported best* (not a reconstruction) is the point, so the test proves
        the checkpointed individual is the one the library can explain.
        ``network_dict`` renumbers hidden keys but never the input/output blocks,
        so the genome's own input/output index sets must survive in ``nodes``.

    Owner: ``pipeline.Pipeline.auto_run`` (checkpoint), ``genome.DefaultGenome.network_dict``
    (the host decode) and ``problem.func_fit`` (the re-scoring oracle).
    """
    pipeline = hyperneat_pipeline(
        is_save=True, save_dir=tmp_path, generation_limit=TWO_GENERATIONS
    )
    state, best = run_pipeline(pipeline)
    assert _generation(tn, state) == TWO_GENERATIONS, (
        "the unreachable fitness target must consume the two-generation budget, got "
        f"generation {_generation(tn, state)}"
    )

    best_shapes = tuple(tuple(int(size) for size in field.shape) for field in best)
    assert best_shapes == EXPECTED_BEST_SHAPES, (
        "auto_run's best must be a one-individual CPPN batch padded to the static "
        f"signature {EXPECTED_BEST_SHAPES}, got {best_shapes}"
    )

    npz_path = tmp_path / "genomes" / "best_genome.npz"
    assert npz_path.is_file(), f"Pipeline.auto_run must write {npz_path}"
    with np.load(npz_path) as archive:
        stored = {name: archive[name] for name in archive.files}

    missing = [name for name in SPLIT_PAYLOAD if name not in stored]
    assert not missing, (
        f"{npz_path} holds {sorted(stored)} but lacks {missing}: the checkpoint must "
        f"store the split gene payload {SPLIT_PAYLOAD} plus a scalar 'fitness'"
    )
    assert "fitness" in stored, (
        f"{npz_path} holds {sorted(stored)} but no 'fitness' entry; a checkpoint "
        "without the fitness it was selected for cannot be verified on load"
    )

    arrays = []
    for name in SPLIT_PAYLOAD:
        array = np.asarray(stored[name])
        expected = EXPECTED_STORED_SHAPES[name]
        assert array.shape == expected, (
            f"{npz_path}[{name!r}] must be one genome in shape {expected}, got "
            f"{array.shape}"
        )
        arrays.append(array)

    fitness_values = np.asarray(stored["fitness"])
    assert fitness_values.size == 1, (
        f"{npz_path}['fitness'] must be the scalar fitness of the stored genome, got "
        f"shape {fitness_values.shape}"
    )
    recorded = float(fitness_values.reshape(-1)[0])
    assert np.isfinite(recorded), (
        f"{npz_path} records the fitness {recorded}; a best genome is selected by a "
        "finite fitness, so the checkpoint cannot hold an invalid one"
    )

    payload_type = type(state.pop)
    loaded = payload_type(*_batch_one(arrays))
    rescored = evaluate_population(pipeline, state, loaded)
    assert rescored.shape == (1,), (
        f"re-scoring one loaded CPPN must yield a (1,) fitness array, got "
        f"{rescored.shape}"
    )
    assert np.isfinite(float(rescored[0])), (
        f"the loaded CPPN re-scores to {float(rescored[0])!r}, so the checkpoint did "
        "not reconstruct a scorable substrate network"
    )
    np.testing.assert_allclose(
        float(rescored[0]),
        recorded,
        rtol=1e-5,
        atol=1e-6,
        err_msg=(
            f"the best CPPN loaded from {npz_path} must re-score to the fitness it was "
            f"stored with ({recorded!r}), got {float(rescored[0])!r}"
        ),
    )

    network = pipeline.algorithm.neat.genome.network_dict(state, best)
    for key in ("nodes", "conns", "topo_order", "topo_layers", "useful_nodes"):
        assert key in network, (
            f"network_dict must expose {key!r} for the evolved CPPN, got "
            f"{sorted(network)}"
        )
    assert network["nodes"], (
        "network_dict returned no node for the evolved CPPN; the visualization path "
        "would have nothing to draw"
    )

    genome = pipeline.algorithm.neat.genome
    node_keys = set(network["nodes"])
    for role, keys in (("input", genome.get_input_idx()), ("output", genome.get_output_idx())):
        missing_role = sorted(set(int(key) for key in keys) - node_keys)
        assert not missing_role, (
            f"network_dict lost the CPPN {role} node(s) {missing_role}; decoded nodes "
            f"are {sorted(node_keys)}"
        )
    for connection in network["conns"]:
        assert len(connection) == 2, (
            f"network_dict connection keys must be (source, target) pairs, got "
            f"{connection!r}"
        )
        endpoints = set(int(endpoint) for endpoint in connection)
        orphan = sorted(endpoints - node_keys)
        assert not orphan, (
            f"network_dict connection {connection!r} names node(s) {orphan} that are "
            f"absent from network['nodes'] ({sorted(node_keys)})"
        )
