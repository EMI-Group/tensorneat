"""Unit tests for compatibility distance — the speciation metric.

What this module proves
-----------------------
``DefaultDistance`` answers one question per parent pair: *how different are these
two genomes?* The algorithm's speciation threshold is only meaningful if the answer
is a stable, symmetric, zero-at-identity number, so the tests assert exactly those
properties of ``node_distance + conn_distance``:

* **zero at identity** — the same genome against itself is ``0``, and so is a pair
  of empty gene families (``0``, never ``NaN``: a ``NaN`` distance would send every
  genome to its own species);
* **symmetry** — ``d(a, b) == d(b, a)``, which the ``/ max(cnt1, cnt2)``
  normalisation is what makes true; an unnormalised count is symmetric too, so the
  test also covers pairs of *different* sizes;
* **monotonicity** — adding a gene that has no homologue in the other parent
  strictly increases the distance (``compatibility_disjoint`` counts non-homologous
  genes, disjoint *and* excess alike), and adding an attribute difference increases
  it in proportion to ``compatibility_weight``;
* **composition** — the batched call on ``P`` pairs equals ``P`` single-pair calls
  (the population axis is a pure batch axis here too), and repeated calls agree.

Oracles used here (never the kernel itself)
-------------------------------------------
* hand arithmetic on the documented formula
  ``(non_homologous_count * compatibility_disjoint + homologous_attribute_distance *
  compatibility_weight) / max(cnt1, cnt2)`` with the attribute term itself
  ``sum(|delta|)`` over the float columns and ``sum(delta != 0)`` over the int
  columns (``BaseGene.distance``);
* ``support.payloads`` topologies whose gene counts and attribute values are known
  exactly, built in pairs that differ by one controlled edit;
* ``P`` single-pair runs for the batching check — the same kernel, but with the
  batch axis of size one, which is the only way to see a cross-pair reduction.

Gotchas this module encodes
---------------------------
* The two terms are **added** over gene families (node genes and connection genes),
  so a distance of ``0`` requires both families to be identical; a test that only
  varied connection weights could not tell an implemented node term from an absent
  one. Both are varied here.
* ``max(cnt1, cnt2)`` — not the sum, not the minimum. A pair where one parent has
  twice the genes is therefore normalised by the larger count; the empty-pair case
  would divide by zero, which is why the docstring pins the guarded ``0``.
* Int attribute columns (``aggregation``/``activation`` ids) contribute ``!=``,
  i.e. ``1`` per differing option, not ``|delta|``: changing activation from
  option ``0`` to option ``5`` is *one* difference, not five. ``DefaultNode`` is a
  single-option configuration by default, so the test configures two options to
  observe it.
* Nothing here is differentiable and nothing is random, so distance is the one
  operator whose output parity against the frozen implementation is exact rather
  than ``allclose``; this module only asserts its algebra.

Backend is ``"numpy"`` throughout, through the shared runner only; no
``jax``/``torch``, no network, no GPU.
"""

from __future__ import annotations

import pytest


# --------------------------------------------------------------------------- #
# configuration
# --------------------------------------------------------------------------- #


def test_negative_weights_are_rejected(genome):
    """Invariant: a negative ``compatibility_disjoint``/``compatibility_weight`` raises.

    Oracle: the documented constructor validation — the error must be
    ``TensorNEATError`` (the package hierarchy), raised at construction rather than
    at the first ``evoxc.build``. A negative weight would make "more different" mean
    "closer", which the algorithm cannot detect on its own.
    """
    pytest.skip(
        "not implemented: negative distance weights raise TensorNEATError "
        "(owner: genome/operations/distance/default.DefaultDistance.__init__)"
    )


# --------------------------------------------------------------------------- #
# the identity and symmetry axioms
# --------------------------------------------------------------------------- #


def test_identical_genomes_have_zero_distance(payloads, run_kernel):
    """Invariant: ``d(g, g) == 0`` for every individual of a non-trivial payload.

    Oracle: the identity axiom, on a payload that has both gene families populated
    (nodes *and* connections) and padding rows, so an over-counted padding row or a
    node-gene term applied to absent rows would show as a non-zero distance. Also
    asserted for the *initialized* population, where every individual is identical
    by construction.
    """
    pytest.skip(
        "not implemented: distance of a genome with itself is exactly zero "
        "(owner: genome/operations/distance/default.DefaultDistance.__call__, gene_distance)"
    )


def test_distance_is_symmetric(payloads, run_kernel):
    """Invariant: ``d(a, b) == d(b, a)`` within one ulp, including for unequal gene counts.

    Oracle: the two calls. Asymmetry is the classic symptom of an unnormalised or
    half-counted term (``cnt1`` instead of ``max(cnt1, cnt2)``), and it only shows
    on parents with different numbers of genes — so the pair is built with parent 2
    strictly larger than parent 1, and the reverse call is run on the same arrays.
    """
    pytest.skip(
        "not implemented: distance is symmetric for pairs of unequal size "
        "(owner: genome/operations/distance/default.DefaultDistance.gene_distance)"
    )


# --------------------------------------------------------------------------- #
# monotonicity
# --------------------------------------------------------------------------- #


def test_distance_grows_with_non_homologous_genes(payloads, run_kernel):
    """Invariant: one added non-homologous gene strictly increases the distance.

    Oracle: hand arithmetic of the disjoint term. Starting from identical parents,
    adding one node (or one connection) to parent 2 that parent 1 does not have
    must add ``compatibility_disjoint / max(cnt1, cnt2)``-worth of distance, and
    adding a second one must add more again — *disjoint* and *excess* genes are
    counted by the same term, so the test walks the count ``0 -> 1 -> 2`` and
    checks strict monotonicity rather than a particular value.
    """
    pytest.skip(
        "not implemented: distance is strictly monotone in the number of non-homologous genes "
        "(owner: genome/operations/distance/default.DefaultDistance.gene_distance)"
    )


def test_attribute_differences_are_weighted_by_compatibility_weight(
    payloads, run_kernel
):
    """Invariant: doubling a float attribute delta doubles that gene's distance contribution.

    Oracle: hand arithmetic of ``homologous_attribute_distance * compatibility_weight``
    with ``|delta|`` summed over the float columns. Two payloads that differ only in
    the weight of one known connection, by ``delta`` and ``2 * delta`` respectively,
    must give distances whose *differences from the identical-pair baseline* are in
    a ``1 : 2`` ratio — which pins both the absolute-value and the weight factor.
    """
    pytest.skip(
        "not implemented: the attribute term is |delta| weighted by compatibility_weight "
        "(owner: genome/operations/distance/default.DefaultDistance.gene_distance, "
        "gene/base.BaseGene.distance)"
    )


def test_int_attribute_columns_contribute_one_per_changed_option(
    genome, payloads, run_kernel
):
    """Invariant: a changed option id contributes ``1``, not ``|id1 - id2|``.

    Oracle: ``BaseGene.distance``'s documented per-column rule — ``!=`` for an
    ``"int"`` spec, ``|delta|`` for a ``"float"`` spec. A ``DefaultNode`` configured
    with three activation options, whose rows hold option ``0`` and option ``2``, must
    contribute ``1``; reading the column as a float would contribute ``2``. This is
    the difference between counting changed *genes* and measuring a meaningless
    numeric gap between option indices.
    """
    pytest.skip(
        "not implemented: int option columns contribute != rather than |delta| "
        "(owner: genome/gene/base.BaseGene.distance, "
        "genome/operations/distance/default.DefaultDistance.gene_distance)"
    )


# --------------------------------------------------------------------------- #
# degenerate and batched calls
# --------------------------------------------------------------------------- #


def test_empty_pair_distance_is_zero_not_nan(payloads, run_kernel):
    """Invariant: two empty gene families are ``0`` apart, and no pair ever yields ``NaN``.

    Oracle: the guarded ``max(cnt1, cnt2) == 0`` branch documented on
    ``gene_distance``. A payload with no connections (an isolated-node payload with
    its edge removed, or a whole payload of padding) is legal — a mutation can
    produce it — and it must not poison the speciation step with ``NaN``. The whole
    returned vector is asserted finite.
    """
    pytest.skip(
        "not implemented: empty gene families give zero distance, never NaN "
        "(owner: genome/operations/distance/default.DefaultDistance.gene_distance)"
    )


def test_batched_pairs_equal_pairwise_calls(payloads, runner, run_kernel):
    """Invariant: one ``P``-pair call equals ``P`` single-pair calls.

    Oracle: the per-pair calls, run separately with their own payloads of
    ``pop_size=1``. The population axis is a pure batch axis for every kernel of
    this package, so a reduce that accidentally spans it (a ``max`` over all pairs
    instead of per pair, say) is visible here and nowhere else. Pairs are chosen to
    have different sizes and different edit distances, so a single shared answer
    cannot pass.
    """
    pytest.skip(
        "not implemented: batched distance equals pairwise distance "
        "(owner: genome/operations/distance/default.DefaultDistance.__call__)"
    )


def test_distance_is_deterministic(payloads, runner, run_kernel):
    """Invariant: the same pair gives bit-identical distances on repeated calls.

    Oracle: two runs of the same kernel on the same arrays. Distance consumes no
    RNG, so it must be a pure function of its inputs — which is also why it is the
    one operator whose parity against the frozen implementation is exact rather
    than tolerance-based. A ``hash``/``argsort``-based alignment that leaked
    iteration order would surface as a run-to-run difference here.
    """
    pytest.skip(
        "not implemented: distance is a pure function of the two payloads "
        "(owner: genome/operations/distance/default.DefaultDistance.__call__)"
    )
