"""Golden values of a three-generation CartPole run at ``pop_size=100`` (marker ``slow``).

What this module proves
-----------------------
The prepared-forward change (one policy pass per rollout instead of one per
timestep) must be numerically neutral end to end. This test records the exact
per-generation fitness and a hash of every produced population for one fixed
configuration, so a regression that changes even one bit of the rollout — a
different ``while_loop`` carry, a different reset draw, a policy pass prepared
with the wrong tiling factor — fails on a number instead of on a distribution.
A deliberate change to a search contract also moves these numbers; re-derive
them by rerunning the configuration rather than widening the comparison.

The configuration is the shipped ``examples/rl/cartpole.py`` recipe cut down to
``pop_size=100``: ``DefaultGenome(4, 2)`` with a ``BiasNode(tanh, sum)`` gene, no
initial hidden layer, ``NEAT(species_size=10, survival_threshold=0.1)``,
``CartPole(max_step=200, repeat_times=1)``, ``seed=42`` on the numpy backend.

The oracle
----------
The values were recorded on the merged tree, after the prepared-forward and
tiled-evaluate changes *and* after the search contracts that decide the offspring
(the bounded ``key_span`` allocation and the four independent structural gates),
by running this configuration and printing
``(best, mean, state.pop digest)`` after each ``pipeline.step``; the test then
hard-codes what was observed. They are not copied from any log: re-deriving them
means re-running the module and replacing the fixtures below if the explanation
in a diff justifies it.

``best`` is ``max(fitness)`` and ``mean`` is ``mean(fitness)`` of the evaluated
generation; the digest is the first 16 hex digits of
``sha256`` over the four ``state.pop`` arrays' raw bytes in payload order
(``node_keys``, ``node_attrs``, ``conn_keys``, ``conn_attrs``), i.e. the
population the next generation starts from. The run is deterministic, so both
are exact comparisons, not tolerances.

Budget (measured, never asserted)
---------------------------------
Three generations at ``pop_size=100`` with ``max_step=200``: about 1.5 s on the
numpy reference interpreter. No test reads a clock.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

pytestmark = [pytest.mark.slow]

#: Population of the golden run.
POP_SIZE = 100

#: Horizon of the golden run; also the fitness ceiling (a frozen episode's return
#: is its survival time).
MAX_STEP = 200

#: Generations recorded.
GENERATIONS = 3

#: The four ``state.pop`` field names, in payload order.
POP_FIELDS = ("node_keys", "node_attrs", "conn_keys", "conn_attrs")

#: ``(generation, best, mean, state.pop digest)`` observed on this tree.
GOLDEN = (
    (1, 56.0, 14.069999694824219, "2f2f53415501087a"),
    (2, 176.0, 39.2400016784668, "5433039cd7a2e15e"),
    (3, 200.0, 82.7699966430664, "e9c66574d1390795"),
)


def _pipeline(tn):
    """Build the golden CartPole pipeline (the example's recipe at ``pop_size=100``).

    Args:
        tn: The imported ``tensorneat`` package (the ``tn`` fixture).

    Returns:
        A fresh, not-yet-set-up ``Pipeline`` with ``seed=42`` and the numpy
        backend. The target is reachable by the third generation, so
        ``auto_run`` would stop there; the test drives ``step`` itself so the
        recorded generation numbers are explicit.
    """
    neat = tn.algorithm.NEAT(
        pop_size=POP_SIZE,
        species_size=10,
        survival_threshold=0.1,
        genome=tn.genome.DefaultGenome(
            num_inputs=4,
            num_outputs=2,
            init_hidden_layers=(),
            node_gene=tn.genome.BiasNode(
                activation_options=tn.common.ACT.tanh,
                aggregation_options=tn.common.AGG.sum,
            ),
        ),
    )
    cartpole = tn.problem.CartPole(
        max_step=MAX_STEP, repeat_times=1, fitness_target=MAX_STEP
    )
    return tn.Pipeline(
        algorithm=neat,
        problem=cartpole,
        seed=42,
        generation_limit=GENERATIONS,
        fitness_target=MAX_STEP,
        backend="numpy",
        is_detailed=False,
    )


def _payload_digest(tn, state):
    """Hash the four ``state.pop`` arrays' raw bytes, in payload order.

    Args:
        tn: The imported ``tensorneat`` package (``tn.common.to_numpy`` unwraps
            one ``evoxc.Tensor`` leaf and is deliberately shallow, so it is
            applied per field; an ``evoxc.Tensor`` has no ``__array__``).
        state: The state a generation returned.

    Returns:
        The first 16 hex digits of the SHA-256 over the concatenated
        little-endian byte images of the four arrays. The digest is therefore
        sensitive to a changed key, attribute, padding sentinel or row order.

    Raises:
        AssertionError: If ``state.pop`` does not carry exactly the four
            documented fields.
    """
    pop = state.pop
    names = tuple(getattr(pop, "_fields", POP_FIELDS))
    assert names == POP_FIELDS, (
        f"state.pop must carry the four GenomeTensors fields {POP_FIELDS}, got {names}"
    )
    digest = hashlib.sha256()
    for field in pop:
        digest.update(np.ascontiguousarray(tn.common.to_numpy(field)).tobytes())
    return digest.hexdigest()[:16]


def test_cartpole_p100_three_generations(tn):
    """Invariant: the seeded CartPole run reproduces its recorded best, mean and populations.

    Oracle: :data:`GOLDEN`, recorded on this tree by stepping the same
    configuration and printing the three values per generation. The first
    generation's best (``56.0``) is a survival count — five episodes of
    ``56`` steps out of the ``100`` rolled out — so a shift in the reset draw or
    in the action convention shows immediately.

    Assertion: per generation, ``max(fitness)`` and ``mean(fitness)`` equal the
    recorded floats exactly and :func:`_payload_digest` equals the recorded
    digest. The third generation reaches the ``200``-step ceiling, which is the
    example's documented stopping point, and ``state.generation`` counts 1..3.
    """
    pipeline = _pipeline(tn)
    state = pipeline.setup()

    observed = []
    for generation in range(1, GENERATIONS + 1):
        result = pipeline.step(state)
        state = result[0]
        fitness = np.asarray(tn.common.to_numpy(result[-1]))
        assert fitness.shape == (POP_SIZE,), (
            f"generation {generation} must report {POP_SIZE} fitness values, got "
            f"{fitness.shape}"
        )
        observed.append(
            (
                generation,
                float(np.max(fitness)),
                float(np.mean(fitness)),
                _payload_digest(tn, state),
            )
        )
        assert int(tn.common.to_numpy(state.generation)) == generation, (
            f"step must advance state.generation to {generation}, got "
            f"{int(tn.common.to_numpy(state.generation))}"
        )

    for seen, (generation, best, mean, digest) in zip(observed, GOLDEN):
        assert seen[1] == best, (
            f"generation {generation}: best fitness changed from the recorded "
            f"{best} to {seen[1]}; the rollout is no longer numerically neutral"
        )
        assert seen[2] == mean, (
            f"generation {generation}: mean fitness changed from the recorded "
            f"{mean} to {seen[2]}"
        )
        assert seen[3] == digest, (
            f"generation {generation}: state.pop digest changed from the recorded "
            f"{digest} to {seen[3]}"
        )

    bests = [entry[1] for entry in observed]
    assert bests == sorted(bests), (
        f"the per-generation best fitness must be non-decreasing, got {bests}"
    )
    assert observed[-1][1] == MAX_STEP, (
        f"the third generation must reach the {MAX_STEP}-step ceiling, got "
        f"{observed[-1][1]}"
    )
