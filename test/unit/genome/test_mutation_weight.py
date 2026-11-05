"""Unit tests for the *value* half of mutation: the law of a weight change.

What this module proves
-----------------------
``DefaultMutation``'s four probabilities gate *structure*; value mutation of the
connection gene runs unconditionally and is what this module pins, on hand-built
payloads whose weights are known exactly.  ``test_mutation.py`` pins the structural
invariants and only "some attribute changed"; here the *magnitude* is pinned against
the documented law of ``gene/blocks.py::mutate_attrs``:

* a perturbed float column is ``base + N(0, mutate_power)`` -- one standard-normal
  sample scaled by ``mutate_power`` -- so on a **zero base** the observed
  ``mutated - base`` is exactly the noise, and
  ``delta(p) == delta(0.1) * float32(p / 0.1)`` for every ``p`` that is a
  power-of-two multiple of ``0.1`` (the multiply is an exponent shift, exact in
  float32);
* a re-drawn float column is ``N(init_mean, init_std)``, which is **exactly**
  ``init_mean`` when ``init_std == 0``;
* the perturb branch is selected when ``draw < mutate_rate`` and the replace branch
  only otherwise (the perturb branch wins where both arms fire), so with
  ``mutate_rate == 1`` the replace candidate is never observable; at
  ``mutate_rate == 0`` every value is either the untouched base or the replace
  candidate and nothing else;
* both candidates are affine in **one** standard-normal sample (the documented
  joint-law simplification), which is observable when both branches win on
  different entries of the same run: the perturbed entry is
  ``base + mutate_power * Z`` and the re-drawn one is ``init_mean + init_std * Z``
  for the *same* ``Z``.

Oracles and gotchas
-------------------
* The oracle is the documented law of ``mutate_attrs`` evaluated in host numpy from
  the observed sample, never the kernel's own intermediate state.
* At a nonzero base the ``+ base`` step rounds, so **exact** proportionality holds
  only on a zero base (where ``base + noise`` is the noise).  Across bases the same
  sample is compared with ``atol=1e-6``; a different mutation power is a different
  static of the graph, hence a different ``evoxc.build``, while the RNG key, the
  shapes and every other static are identical -- which is what makes the
  proportionality a law rather than a coincidence of two unrelated draws.
* Padding rows are never revived: every test that observes a weight also checks
  that an absent connection row is still ``NaN``.
* The non-vacuity checks (both branches, or both outcomes of a rate-``0.5`` gate,
  occur somewhere) sweep ``evoxc`` keys rather than assuming a lucky draw; the
  executable is cached per shape, so a sweep costs runs, not builds.

Backend is ``"numpy"`` throughout, through the shared runner only; no
``jax``/``torch``, no network, no GPU.
"""
from __future__ import annotations

import numpy as np
import pytest

#: Input width of every genome built here (rows ``0.._NUM_INPUTS-1``).
_NUM_INPUTS = 2
#: Output width of every genome built here (the last ``_NUM_OUTPUTS`` rows).
_NUM_OUTPUTS = 1
#: Node capacity ``N`` of every payload built here.
_MAX_NODES = 8
#: Connection capacity ``E`` of every payload built here.
_MAX_CONNS = 16
#: Static key bound ``S``: the caller-allocated keys below must stay inside it.
_KEY_SPAN = 64
#: Output row of the hand-built payloads (``_MAX_NODES - _NUM_OUTPUTS``).
_OUTPUT_ROW = _MAX_NODES - _NUM_OUTPUTS
#: Five edges per individual, so 15 independent weight samples at ``pop_size=3``:
#: enough for "both branches occurred" to be a non-vacuity guard, not a lucky draw.
_EDGES = ((0, 2), (1, 2), (2, _OUTPUT_ROW), (0, _OUTPUT_ROW), (1, _OUTPUT_ROW))
#: Powers that are exact power-of-two multiples of ``0.1`` in float64, so the
#: expected ``float32(power / 0.1)`` factor is exact and the law is bit-testable.
_POWERS = (0.05, 0.1, 0.2, 0.4, 0.8)


# --------------------------------------------------------------------------- #
# hand-built fixtures (the same conventions as ``test_mutation.py``)
# --------------------------------------------------------------------------- #
def _quiet_node(genome):
    """``DefaultNode`` whose attribute mutation is the identity (every rate is ``0``)."""
    return genome.DefaultNode(
        bias_mutate_rate=0.0,
        bias_replace_rate=0.0,
        response_mutate_rate=0.0,
        response_replace_rate=0.0,
        aggregation_replace_rate=0.0,
        activation_replace_rate=0.0,
    )


def _genome(genome, *, conn_gene, node_gene=None):
    """Build a ``DefaultGenome`` around one connection gene and no structural mutation.

    Structural probabilities are all ``0`` so the traced kernel is the *value* stage
    alone; the node gene is quiet unless a test wants node attributes to move too.
    Shapes match :func:`_weight_payload` (``_MAX_NODES``/``_MAX_CONNS``, two inputs,
    output row ``7``), and ``key_span`` covers the caller-allocated keys.
    """
    return genome.DefaultGenome(
        num_inputs=_NUM_INPUTS,
        num_outputs=_NUM_OUTPUTS,
        max_nodes=_MAX_NODES,
        max_conns=_MAX_CONNS,
        node_gene=_quiet_node(genome) if node_gene is None else node_gene,
        conn_gene=conn_gene,
        mutation=genome.DefaultMutation(
            conn_add=0.0, conn_delete=0.0, node_add=0.0, node_delete=0.0
        ),
        crossover=genome.DefaultCrossover(),
        distance=genome.DefaultDistance(),
        key_span=_KEY_SPAN,
    )


def _keys(seed: int, count: int):
    """``count`` distinct rank-0 ``evoxc`` keys from the per-test ``seed``.

    A small fixed loop keeps a failing sweep reproducible; the executable cache keys
    only shapes and statics, so every key after the first reuses one build.
    """
    import evoxc

    return [evoxc.random.key(int(seed) + offset) for offset in range(count)]


def _node_slots(pop_size: int, first: int = 20) -> np.ndarray:
    """``(P,) int32`` caller-allocated node keys, above every hand-built key."""
    return np.arange(first, first + pop_size, dtype=np.int32)


def _conn_slots(pop_size: int, first: int = 40) -> np.ndarray:
    """``(P, 3) int32`` caller-allocated connection keys (identity/copy/add columns)."""
    return np.tile(np.array([first, first + 1, first + 2], dtype=np.int32), (pop_size, 1))


def _weight_payload(payloads, weight: float = 0.0, pop_size: int = 3):
    """Rows ``0, 1, 2, 7`` present with the five ``_EDGES`` all carrying ``weight``.

    Every individual is the same hand-written genome, so an assertion can name the
    value it expects; unused connection rows keep the padding sentinels, which is what
    makes "padding is never revived" assertable in the same breath.
    """
    payload = payloads.new_payload(pop_size, _MAX_NODES, _MAX_CONNS, conn_key_width=2)
    for p in range(pop_size):
        for row in (0, 1, 2, _OUTPUT_ROW):
            payloads.set_node(payload, p, row, row)
        for row, (source, target) in enumerate(_EDGES):
            payloads.set_conn(payload, p, row, source, target, [weight])
    return payload


def _present(payload, sentinel) -> np.ndarray:
    """``(P, E)`` mask of the connection rows that carry a real gene."""
    return payload.conn_keys[:, :, 0] != sentinel


def _mutate(run_kernel, genome_object, state, payload, key):
    """Run one ``execute_mutation`` call and return the four host arrays in order."""
    pop = payload.pop_size
    return run_kernel(
        genome_object.execute_mutation,
        state,
        key,
        *payload.args(),
        _node_slots(pop),
        _conn_slots(pop),
    )


def _weights(mutated, payload, sentinel) -> np.ndarray:
    """The present weights of one ``execute_mutation`` result, in individual-major order."""
    return mutated[3][..., 0][_present(payload, sentinel)]


def _mutate_until(
    run_kernel, genome_object, state, payload, sentinel, predicate, keys, what: str
):
    """Run the value stage on fresh keys until ``predicate(weights)`` holds.

    The non-vacuity guard for a stochastic branch: the invariant is always asserted,
    but at least one run has to exhibit the outcome the assertion is about, or the
    test would pass on an operator that never took that branch.  Keys are data, so the
    sweep reuses the one built executable; a genuinely broken implementation never
    satisfies the predicate and fails here with the branch named.
    """
    for key in keys:
        mutated = _mutate(run_kernel, genome_object, state, payload, key)
        weights = _weights(mutated, payload, sentinel)
        if predicate(weights):
            return mutated, weights
    pytest.fail(f"no evoxc key produced {what} in {len(keys)} runs")


# --------------------------------------------------------------------------- #
# the scale law of a perturbation
# --------------------------------------------------------------------------- #
def test_perturbation_delta_is_exactly_proportional_to_the_power(
    genome, payloads, run_kernel, evoxc_key
):
    """Invariant: on a zero base the weight delta is exactly ``mutate_power * Z``.

    Oracle: the documented perturb law.  With a zero base the ``+ base`` step is
    exact (``0 + noise``), so ``mutated - base`` *is* the noise and the law becomes
    bit-testable: for powers that are exact power-of-two multiples of ``0.1`` in
    float64, ``delta(p) == delta(0.1) * float32(p / 0.1)``.  Each power builds its own
    graph (``mutate_power`` is a static), but the RNG key, shape and every other
    static are identical, so the standard-normal sample is the same one -- a kernel
    that drew a fresh sample per power would fail here.
    Assertion: exact ``array_equal`` per power, non-zero deltas, and ``NaN`` padding
    untouched.  A wrong scale (a missing multiply, a mean/std swap) changes the
    ratio, and a sign flip is caught by the non-zero + ratio pair.
    """
    payload = _weight_payload(payloads, weight=0.0)
    sentinel = payloads.I_INF
    deltas = {}
    for power in _POWERS:
        conn_gene = genome.DefaultConn(
            weight_mutate_rate=1.0,
            weight_replace_rate=0.0,
            weight_mutate_power=power,
        )
        genome_object = _genome(genome, conn_gene=conn_gene)
        state = payloads.genome_state(genome_object)
        mutated = _mutate(run_kernel, genome_object, state, payload, evoxc_key)
        # The zero base makes the subtraction exact: delta == the noise itself.
        deltas[power] = _weights(mutated, payload, sentinel) - 0.0
        # ... and value mutation never revives an absent row.
        assert np.isnan(mutated[3][~_present(payload, sentinel)]).all()
    reference = deltas[0.1]
    assert (reference != 0.0).all()  # non-vacuous: the perturbation really happened
    assert len(np.unique(reference)) > 1  # ... with independent per-entry samples
    for power in _POWERS:
        factor = np.float32(power / 0.1)
        np.testing.assert_array_equal(deltas[power], reference * factor, err_msg=str(power))


def test_perturbation_noise_is_independent_of_the_base_to_within_the_rounding(
    genome, payloads, run_kernel, evoxc_key
):
    """Invariant: the same power draws the same noise whatever the base weight is.

    Oracle: ``mutated - base`` against the zero-base noise of the *same* key and power.
    The RNG draw depends only on the key, the output shape and ``mutate_power``, none
    of which changes with the base, so the two deltas differ only by the rounding of
    ``base + noise`` -- at most one ulp of ``base`` (measured 2e-8 at ``base = 2``;
    ``spacing(float32(2))`` is 2.4e-7).  ``atol=1e-6`` is a deliberately loose bound,
    as the exactness claim is only made on the zero base.
    Assertion: the shifted deltas are within ``atol=1e-6`` of the zero-base noise, are
    non-zero, and the padded rows stay ``NaN``.  A kernel that scaled the noise by the
    base (``base * N(0, power)``) or added ``init_mean`` here would be off by orders of
    magnitude.
    """
    sentinel = payloads.I_INF
    conn_gene = genome.DefaultConn(
        weight_mutate_rate=1.0, weight_replace_rate=0.0, weight_mutate_power=0.1
    )
    genome_object = _genome(genome, conn_gene=conn_gene)
    state = payloads.genome_state(genome_object)

    zero = _weight_payload(payloads, weight=0.0)
    noise = _weights(
        _mutate(run_kernel, genome_object, state, zero, evoxc_key), zero, sentinel
    )
    shifted_payload = _weight_payload(payloads, weight=2.0)
    shifted_mutated = _mutate(run_kernel, genome_object, state, shifted_payload, evoxc_key)
    shifted = _weights(shifted_mutated, shifted_payload, sentinel) - np.float32(2.0)

    assert (noise != 0.0).all()
    assert np.allclose(shifted, noise, rtol=0.0, atol=1e-6)
    assert np.isnan(shifted_mutated[3][~_present(shifted_payload, sentinel)]).all()


def test_zero_mutate_rate_leaves_every_weight_bit_identical(
    genome, payloads, run_kernel, evoxc_key
):
    """Invariant: ``weight_mutate_rate = 0`` with ``weight_replace_rate = 0`` is the identity.

    Oracle: byte equality against the input ``conn_attrs`` (padding ``NaN`` included).
    A rate of ``0`` is documented to remove its branch statically, so this is the
    base case the scale law builds on: the value the other tests call ``base`` is
    exactly the payload's own weight.
    """
    conn_gene = genome.DefaultConn(weight_mutate_rate=0.0, weight_replace_rate=0.0)
    genome_object = _genome(genome, conn_gene=conn_gene)
    state = payloads.genome_state(genome_object)
    payload = _weight_payload(payloads, weight=2.0)
    mutated = _mutate(run_kernel, genome_object, state, payload, evoxc_key)
    assert mutated[3].tobytes() == payload.conn_attrs.tobytes()
    assert mutated[0].tobytes() == payload.node_keys.tobytes()
    assert mutated[2].tobytes() == payload.conn_keys.tobytes()


# --------------------------------------------------------------------------- #
# the replace branch and its precedence
# --------------------------------------------------------------------------- #
def test_replace_rate_one_redraws_every_weight_from_the_init_mean(
    genome, payloads, run_kernel, evoxc_key
):
    """Invariant: a ``replace_rate`` of ``1`` re-draws every weight from ``N(mean, std)``.

    Oracle: ``init_std = 0`` makes the re-draw exactly ``init_mean`` -- the documented
    statically-simplified branch (``N(mean, 0) = mean``), so every *present* weight is
    exactly ``7.0`` regardless of the base and of the RNG stream.  ``mutate_rate = 0``
    keeps the perturb branch out of the way.
    Assertion: every present value equals ``7.0`` bit for bit (no tolerance), and every
    padded row is still ``NaN`` -- a re-draw that forgot the liveness mask would write
    ``7.0`` into padding here.
    """
    conn_gene = genome.DefaultConn(
        weight_mutate_rate=0.0,
        weight_replace_rate=1.0,
        weight_init_mean=7.0,
        weight_init_std=0.0,
    )
    genome_object = _genome(genome, conn_gene=conn_gene)
    state = payloads.genome_state(genome_object)
    payload = _weight_payload(payloads, weight=2.0)
    sentinel = payloads.I_INF
    mutated = _mutate(run_kernel, genome_object, state, payload, evoxc_key)
    np.testing.assert_array_equal(_weights(mutated, payload, sentinel), np.full(15, 7.0))
    assert np.isnan(mutated[3][~_present(payload, sentinel)]).all()


def test_replace_rate_half_only_produces_the_base_or_the_init_mean(
    genome, payloads, run_kernel, seed
):
    """Invariant: at ``replace_rate = 0.5`` no third value is reachable.

    Oracle: the two candidates of a ``mutate_rate = 0`` column are "leave the entry
    alone" and "re-draw from ``N(7, 0) = 7``", so the value set is exactly
    ``{base, 7.0}`` -- a set-membership statement, never a count.  Both outcomes must
    occur somewhere, or the gate could be stuck; that is the non-vacuity guard, and
    the keys are swept for it rather than assumed.
    Assertion: every present weight is ``2.0`` or ``7.0``, both occur, and padding is
    ``NaN``.  A replacement drawn from a different distribution (any other ``std``, or
    replacing instead of leaving the entry alone) puts a third value in the set.
    """
    conn_gene = genome.DefaultConn(
        weight_mutate_rate=0.0,
        weight_replace_rate=0.5,
        weight_init_mean=7.0,
        weight_init_std=0.0,
    )
    genome_object = _genome(genome, conn_gene=conn_gene)
    state = payloads.genome_state(genome_object)
    payload = _weight_payload(payloads, weight=2.0)
    sentinel = payloads.I_INF
    mutated, weights = _mutate_until(
        run_kernel,
        genome_object,
        state,
        payload,
        sentinel,
        lambda values: (values == 7.0).any() and (values == 2.0).any(),
        _keys(seed, 16),
        "both a re-drawn and an untouched weight",
    )
    assert np.isin(weights, (2.0, 7.0)).all()
    assert (weights == 7.0).any() and (weights == 2.0).any()
    assert np.isnan(mutated[3][~_present(payload, sentinel)]).all()


def test_the_perturb_branch_wins_where_both_arms_fire(
    genome, payloads, run_kernel, evoxc_key
):
    """Invariant: with ``mutate_rate = 1`` the replace candidate is never observable.

    Oracle: the documented precedence -- perturb if ``draw < mutate_rate``, replace
    only otherwise -- so at ``mutate_rate = replace_rate = 1`` every entry is
    perturbed and the result must be bit-identical to a ``replace_rate = 0`` run of
    the same power and key.  Both candidates are derived from one standard-normal
    sample; here the perturb one is the only observable one.
    Assertion: byte equality with the perturb-only run, every weight differs from the
    base, no weight equals the replace value ``100.0``, and padding is ``NaN``.  An
    implementation that let the replacement overwrite the perturbation (or that
    gated the two branches in the other order) produces ``100.0`` somewhere.
    """
    sentinel = payloads.I_INF
    payload = _weight_payload(payloads, weight=2.0)
    perturb_only_gene = genome.DefaultConn(
        weight_mutate_rate=1.0, weight_replace_rate=0.0, weight_mutate_power=0.5
    )
    perturb_only = _genome(genome, conn_gene=perturb_only_gene)
    both_gene = genome.DefaultConn(
        weight_mutate_rate=1.0,
        weight_replace_rate=1.0,
        weight_mutate_power=0.5,
        weight_init_mean=100.0,
        weight_init_std=0.0,
    )
    both = _genome(genome, conn_gene=both_gene)
    reference = _mutate(
        run_kernel, perturb_only, payloads.genome_state(perturb_only), payload, evoxc_key
    )
    mutated = _mutate(
        run_kernel, both, payloads.genome_state(both), payload, evoxc_key
    )
    # Same key, same power, same shape: the perturb draw is the same one.
    assert mutated[3].tobytes() == reference[3].tobytes()
    weights = _weights(mutated, payload, sentinel)
    assert (weights != 2.0).all()
    assert (weights != 100.0).all()
    assert np.isnan(mutated[3][~_present(payload, sentinel)]).all()


def test_both_branches_are_affine_in_one_shared_sample(
    genome, payloads, run_kernel, seed
):
    """Invariant: the perturb and replace candidates use the *same* standard-normal ``Z``.

    Oracle: the documented joint-law simplification of ``mutate_attrs`` -- a column
    that both perturbs and re-draws takes one sample ``Z`` and forms
    ``noise = mutate_power * Z`` and ``replacement = init_mean + init_std * Z``.  The
    sample is exposed by a second configuration of the same key: a
    ``replace_rate = 1`` run with ``init_mean = 0`` / ``init_std = 1`` writes ``Z``
    itself into every weight (the shared-sample draw uses the replace column's key).
    With ``mutate_rate = 0.5`` and ``replace_rate = 1`` both arms are observable in
    one run, and each entry must obey its own arm exactly:
    ``value == 100 + 2 Z`` where it was re-drawn and ``value == 2 + 0.5 Z`` where it
    was perturbed.  The two arms are classified by magnitude (the laws put them near
    ``100`` and near ``2``), and both must occur -- the non-vacuity guard, swept over
    keys so it is not a lucky draw.
    Assertion: exact ``array_equal`` for both arms.  Independent draws for the two
    candidates (the old behaviour) fail the replace equation, and swapping the
    coefficients fails both.
    """
    sentinel = payloads.I_INF
    payload = _weight_payload(payloads, weight=2.0)
    sample_gene = genome.DefaultConn(
        weight_mutate_rate=0.0,
        weight_replace_rate=1.0,
        weight_init_mean=0.0,
        weight_init_std=1.0,
    )
    sample_genome = _genome(genome, conn_gene=sample_gene)
    sample_state = payloads.genome_state(sample_genome)
    both_gene = genome.DefaultConn(
        weight_mutate_rate=0.5,
        weight_replace_rate=1.0,
        weight_mutate_power=0.5,
        weight_init_mean=100.0,
        weight_init_std=2.0,
    )
    both = _genome(genome, conn_gene=both_gene)
    both_state = payloads.genome_state(both)

    witness = None
    for key in _keys(seed, 16):
        # Same key for both runs: the shared sample is only comparable that way.
        sample = _weights(
            _mutate(run_kernel, sample_genome, sample_state, payload, key),
            payload,
            sentinel,
        )
        mutated = _mutate(run_kernel, both, both_state, payload, key)
        weights = _weights(mutated, payload, sentinel)
        re_drawn = weights > 50.0
        if re_drawn.any() and (~re_drawn).any():
            witness = (mutated, weights, sample, re_drawn)
            break
    assert witness is not None, (
        "no evoxc key produced both a re-drawn and a perturbed weight in 16 runs"
    )
    mutated, weights, sample, re_drawn = witness
    np.testing.assert_array_equal(
        weights[re_drawn], np.float32(100.0) + np.float32(2.0) * sample[re_drawn]
    )
    np.testing.assert_array_equal(
        weights[~re_drawn], np.float32(2.0) + np.float32(0.5) * sample[~re_drawn]
    )
    assert np.isnan(mutated[3][~_present(payload, sentinel)]).all()


def test_half_rates_reach_only_the_init_mean_or_the_perturb_law(
    genome, payloads, run_kernel, seed
):
    """Invariant: at equal half rates every weight is the replace constant or the perturb law.

    Oracle: with ``init_std = 0`` the re-draw is the constant ``init_mean = 100`` (no
    Gaussian), while the perturb arm is ``base + 0.5 Z`` with ``Z`` taken from the
    perturb column's key -- exactly the value a ``replace_rate = 0`` run of the same
    power and key writes.  So the reachable set of one entry is
    ``{100.0, perturb_only_value}`` and nothing else; both outcomes occur (swept keys).
    Assertion: each present weight is either exactly ``100.0`` or bit-identical to the
    perturb-only run, both classes are non-empty, and padding is ``NaN``.  A missing
    perturb arm leaves every weight at ``100``, and a mean/std swap puts a value in
    between.
    """
    sentinel = payloads.I_INF
    payload = _weight_payload(payloads, weight=2.0)
    perturb_only_gene = genome.DefaultConn(
        weight_mutate_rate=1.0, weight_replace_rate=0.0, weight_mutate_power=0.5
    )
    perturb_only = _genome(genome, conn_gene=perturb_only_gene)
    perturb_state = payloads.genome_state(perturb_only)
    both_gene = genome.DefaultConn(
        weight_mutate_rate=0.5,
        weight_replace_rate=0.5,
        weight_mutate_power=0.5,
        weight_init_mean=100.0,
        weight_init_std=0.0,
    )
    both = _genome(genome, conn_gene=both_gene)
    both_state = payloads.genome_state(both)

    witness = None
    for key in _keys(seed, 16):
        # Same key for both runs: a ``0.5`` gate draw is only comparable that way.
        reference = _weights(
            _mutate(run_kernel, perturb_only, perturb_state, payload, key),
            payload,
            sentinel,
        )
        mutated = _mutate(run_kernel, both, both_state, payload, key)
        weights = _weights(mutated, payload, sentinel)
        if (weights == 100.0).any() and (weights != 100.0).any():
            witness = (mutated, weights, reference)
            break
    assert witness is not None, (
        "no evoxc key produced both a re-drawn and a perturbed weight in 16 runs"
    )
    mutated, weights, reference = witness
    assert ((weights == 100.0) | (weights == reference)).all()
    assert (weights == 100.0).any() and (weights == reference).any()
    assert np.isnan(mutated[3][~_present(payload, sentinel)]).all()


# --------------------------------------------------------------------------- #
# the value stage touches attributes only
# --------------------------------------------------------------------------- #
def test_value_mutation_never_touches_key_columns(genome, payloads, run_kernel, evoxc_key):
    """Invariant: value mutation changes float attributes only.

    Oracle: ``np.array_equal`` on ``node_keys``/``conn_keys`` before and after the
    stage, plus "the attribute rows of present genes stay finite" (a mutation of a
    padded row would produce a finite value where ``NaN`` belongs, and a mutation of a
    key column would corrupt the topology).  Value mutation is documented to be one
    batched draw per gene family at the genes' configured rates, so at the defaults it
    cannot be a no-op; the test therefore asserts *which* columns moved (only float
    attributes, never a key) as well as that something moved.
    """
    # Default gene rates, no structural operation: this isolates the value stage.
    genome_object = _genome(
        genome, node_gene=genome.DefaultNode(), conn_gene=genome.DefaultConn()
    )
    state = payloads.genome_state(genome_object)
    payload = payloads.chain_payload(pop_size=3)
    sentinel = payloads.I_INF
    mutated = _mutate(run_kernel, genome_object, state, payload, evoxc_key)
    assert mutated[0].tobytes() == payload.node_keys.tobytes()
    assert mutated[2].tobytes() == payload.conn_keys.tobytes()
    node_present = payload.node_keys != sentinel
    conn_present = payload.conn_keys[:, :, 0] != sentinel
    assert np.isfinite(mutated[1][node_present]).all()
    assert np.isfinite(mutated[3][conn_present]).all()
    # Padded rows are passed through untouched — still the exact input NaNs.
    assert mutated[1][~node_present].tobytes() == payload.node_attrs[~node_present].tobytes()
    assert mutated[3][~conn_present].tobytes() == payload.conn_attrs[~conn_present].tobytes()
    # Non-vacuous: the weights (mutate rate 0.8) really were perturbed.
    assert (mutated[3][conn_present] != payload.conn_attrs[conn_present]).any()
