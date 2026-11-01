"""Parity of ``problem.XOR`` / ``problem.XOR3d`` fitness on a fixed genome.

What is compared
----------------
The *same* fixed genome — ``payloads.chain_payload()`` built on the host — scored
by each library's own problem layer:

* rewrite: ``new.problem.XOR(error_method=...).evaluate(state, key, act_fn,
  transformed)`` where ``act_fn`` is ``DefaultGenome.forward`` over the payload
  tiled ``S`` times (``../support`` documents the ``row p * S + s`` tiling order,
  and ``./conftest.new_fitness`` implements it), giving ``(P,)`` fitness;
* legacy: ``legacy.problem.XOR(error_method=...).evaluate(state, randkey,
  genome.forward, transformed)``, which ``vmap``s the forward over the dataset
  and reduces to one scalar per individual.

Both are reduced over the sample **and** output axes and negated (fitness is
maximised), and both bake the identical dataset literals
(``[[0,0],[0,1],[1,0],[1,1]] -> [[0],[1],[1],[0]]`` for XOR; the eight three-bit
patterns for XOR3d).  The forwards themselves are parity-tested in
``./test_forward.py``, so a mismatch here is attributable to the problem layer —
the tiling order, the reduction or the sign.

Tolerances and the non-finite cases
-----------------------------------
Finite fitness is compared with ``np.isclose(rtol=1e-5, atol=1e-6)``.  ``mape``
divides by the targets, which contain zeros, so both sides' loss is infinite and
both fitnesses are ``-inf`` — compared with ``==``, including the infinite case.
The one *deliberate* divergence is NaN fitness (see
:func:`test_nan_prediction_is_the_documented_divergence`): the rewrite maps NaN to
``-inf`` inside the problem (``problem.base.finalize_fitness``), the legacy library
leaves the NaN for the algorithm to sanitise, so the two differ there and the test
asserts the mapping instead of an equality.

Scope rules (``../CONTEXT.md``)
------------------------------
Never compared: RNG (the problem is deterministic; both keys are fixed and
unused), initial populations, mutations, crossovers, ``genome.hash()``.  Both
genomes are configured with one activation and one aggregation option, and no
attribute row uses activation id ``-1``, so the two ACT/AGG semantics agree.
"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = [pytest.mark.parity]

pytest.importorskip("jax", reason="parity needs the `bench` extra (jax)")

#: The two shipped function-fitting problems: ``(name, num_inputs)``.
PROBLEMS = [("XOR", 2), ("XOR3d", 3)]
#: The four reductions ``FuncFit`` documents.
ERROR_METHODS = ["mse", "rmse", "mae", "mape"]


@pytest.mark.parametrize("problem_name,num_inputs", PROBLEMS, ids=[p[0] for p in PROBLEMS])
@pytest.mark.parametrize("error_method", ERROR_METHODS)
def test_fitness_matches_legacy(
    problem_name, num_inputs, error_method, payloads, genome_pair, new_fitness, legacy_fitness
):
    """Invariant: on one fixed genome the two problem layers return the same fitness.

    Oracle: the legacy problem layer on the identical genome (and therefore, via
    ``./test_forward.py``, the same predictions).  The genome is
    ``payloads.chain_payload(num_inputs=num_inputs)`` — a fully connected chain
    whose read-out is finite, so the loss is finite for every method except
    ``mape`` — and the pair is built with ``num_inputs=num_inputs`` so
    ``problem.input_shape`` matches the network width (the default ``max_conns=16``
    holds the legacy dense initial network of both layouts: 15 edges for two
    inputs, 16 for three).

    Assertion: ``np.isclose(new[0], legacy, rtol=1e-5, atol=1e-6)`` for the four
    methods on both problems — eight comparisons — and the fitness is finite for
    ``mse``/``rmse``/``mae``, so a pair of equal ``NaN``/``-inf`` values cannot
    stand in for equality.  For ``mape`` both sides are ``-inf`` (the targets
    contain zeros) and the assertion is the exact ``==`` instead.
    """
    pair = genome_pair(num_inputs=num_inputs)
    payload = payloads.chain_payload(num_inputs=num_inputs)

    new = new_fitness(pair, payload, problem_name=problem_name, error_method=error_method)
    legacy = legacy_fitness(pair, payload, problem_name=problem_name, error_method=error_method)

    assert new.shape == (1,)
    if error_method == "mape":
        # Two XOR samples (four of the eight XOR3d samples) have a zero target.
        assert new[0] == -np.inf
        assert legacy == -np.inf
    else:
        assert np.isfinite(new[0])
        assert np.isfinite(legacy)
        assert np.isclose(new[0], legacy, rtol=1e-5, atol=1e-6)


def test_mape_uses_the_targets_as_the_denominator(
    payloads, genome_pair, new_fitness, legacy_fitness
):
    """Invariant: ``mape`` is ``mean(|err / targets|)`` on both sides, so a zero target is infinite.

    Oracle: the closed form on the *parity-compared predictions* and the legacy
    problem.  ``XOR`` targets are ``[[0], [1], [1], [0]]`` and the chain's
    predictions are all strictly positive, so two of the four samples divide by
    zero: the loss is ``inf`` and the fitness ``-inf`` on both sides.

    Assertion: ``new == -np.inf`` and ``legacy == -np.inf`` exactly, while the
    same genome under ``mse`` is finite — which is what proves the denominator is
    the target and not, say, ``max(targets, eps)``.  The ``0/0`` corner (a
    prediction that exactly equals a zero target) yields ``NaN`` on the legacy
    side and is called out as out of scope here; the fixed genome does not hit it.
    """
    pair = genome_pair()
    payload = payloads.chain_payload()

    mape = new_fitness(pair, payload, problem_name="XOR", error_method="mape")
    legacy_mape = legacy_fitness(pair, payload, problem_name="XOR", error_method="mape")
    mse = new_fitness(pair, payload, problem_name="XOR", error_method="mse")

    assert mape.shape == (1,)
    assert mape[0] == -np.inf
    assert legacy_mape == -np.inf
    assert np.isfinite(mse[0])


@pytest.mark.parametrize("error_method", ERROR_METHODS)
def test_nan_prediction_is_the_documented_divergence(
    error_method, payloads, genome_pair, new_fitness, legacy_fitness
):
    """Invariant (divergence, asserted on purpose): NaN prediction is ``-inf`` now, NaN before.

    Oracle: the two documented behaviours, not a shared number.  ``XOR`` on
    ``payloads.isolated_payload()`` with its two input-to-output edges replaced by
    the single edge ``2->7`` gives a ``NaN`` read-out on both sides; the rewrite's
    ``FuncFit.evaluate`` then ends with ``finalize_fitness`` (NaN -> ``-inf``),
    while the legacy ``FuncFit.evaluate`` returns ``-loss`` unchanged and leaves
    the sanitising to the algorithm's ``tell``.

    The edge rewrite is what reaches the corner: both libraries mask a ``NaN``
    *source* slot out of the aggregation (``live = ~isnan(values)``), so the
    isolated row only poisons the read-out once nothing else feeds the output —
    with ``0->7``/``1->7`` still present the read-out stays finite and both
    problem layers agree (``-1`` for ``mse``), which is a parity case, not this
    one.

    Assertion: ``np.isnan(legacy_fitness(...))`` **and**
    ``new_fitness(...) == -np.inf``, for all four ``error_method`` values.  This
    is the only place in ``./`` where the two libraries are expected to disagree,
    and it is pinned so a rewrite that started leaking ``NaN`` fitness is caught
    here rather than in a flaky speciation test.
    """
    pair = genome_pair()
    payload = payloads.isolated_payload()
    # Node row 2 is present but edge-less, so its value stays the NaN seed; make
    # it the output row's only incoming edge and drop ``1->7``.
    payloads.set_conn(payload, 0, 0, 2, 7)
    payloads.set_conn(payload, 0, 1, payloads.I_INF, payloads.I_INF)

    new = new_fitness(pair, payload, problem_name="XOR", error_method=error_method)
    legacy = legacy_fitness(pair, payload, problem_name="XOR", error_method=error_method)

    assert new.shape == (1,)
    assert np.isnan(legacy)
    assert new[0] == -np.inf
