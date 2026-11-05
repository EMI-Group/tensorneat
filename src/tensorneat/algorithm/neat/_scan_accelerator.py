"""Accelerator formulation of the two speciation scans.

This module is the second spelling of the two ``evoxc.scan`` step bodies
:func:`._kernels.speciate` runs; both answer the identical question and return
byte-identical values, so the species table a caller observes does not depend on
which one a graph was traced with.  It is selected by the generic formulation
static ``state.kernel_mode`` (:func:`tensorneat.common.schedule_accelerator.kernel_mode`):
``"host"`` is the shipped form, ``"accelerator"`` is this one, and the choice is
a plain host string read at trace time, so a graph contains exactly one of them.

What a compiled backend charges
-------------------------------
On the box's MI300X a ``xor_p500`` generation is 19.6 ms of device time in 2 087
kernels, and ~2.29 ms of it (12 %) is ~120 dispatches inside these two scans,
every one launched as a single workgroup on grid ``(1, 1, 1)`` -- the dispatches
that cannot be spread over the grid are what a per-slot loop costs there.  A
dispatch costs a ~4.5 µs floor regardless of body size, and an index-tracking
reduction (``argmin``/``argmax``) is much dearer than a plain reduce, so the two
levers are **fewer dispatches** and **plain reductions instead of index scans**.
The host form has three avoidable costs, one formulation per item below:

1. **A duplicated predicate reduction.**  ``argmin_with_mask``/``fetch_first``
   report "no candidate" through their ``I_INF`` sentinel, so the value they
   return already says whether the iteration may write.  The host body computes
   that fact twice: one ``reduce_max(mask)`` of its own for the write mask (and,
   below ``GATED_ROWS``, for the lazy ``cond`` predicate) and again inside the
   helper.  This module reads the first candidate index and derives
   ``claim``/``create`` from it.
2. **Index-tracking reductions.**  ``argmin``/``argmax`` carry an index
   accumulator through the reduction and are the dearest op of the body.  A
   minimising index is recoverable from two plain reductions: minimise the
   *masked scores* to get the minimal score, then minimise the *index vector*
   over the positions that attain it.  The lowest attaining index is exactly
   ``argmin``'s (and the first ``True`` is exactly ``argmax``'s) tie order.
3. **Per-iteration ``(S,)``-cell scatters.**  Each body scattered its per-slot
   record (``closest_table``/``claim_table``, ``seed_table``/``created_mask``)
   and scan 2 scattered ``keys_ext`` twice, and a ``scatter`` is the one op a
   compiled backend lowers to a single thread.  ``evoxc.scan`` already stacks
   its ``y`` output per step (``evoxc/trace/control_flow.py::scan`` reshapes the
   step value and writes it with one ``scatter``), so the per-slot record
   travels as ``y`` instead: one single-cell stack write per step per body, i.e.
   **2 scatters per iteration pair instead of 7**, and no carried table at all.
   Scan 2's key counter becomes a rank-0 carried value with one conditional
   ``add``, and the created rows are rebuilt once after the scan with a prefix
   sum over the created mask (the exact sequence the two scatters wrote).

Bit identity
------------
Every value is the same number, not merely close:

* ``reduce_min`` of the masked scores is ``argmin``'s minimal score, and the
  minimal index over the attaining positions is ``argmin``'s index, because
  scores and indices are both ordered by the same ``pop_index`` vector and the
  minimal index is the first one attained.  The masked-out positions are
  ``+inf`` (scan 1) or ``I_INF`` (scan 2), both above every real candidate.
* The "no candidate" case is ``I_INF`` in both forms: the masked score of an
  empty row is ``+inf``, so nothing attains it and the index reduction returns
  its own ``I_INF`` sentinel.  ``argmin_with_mask`` also returns ``I_INF``
  (never a real row); the host body then clamps it to row ``0`` because its
  post-scan ``gather`` runs unconditionally.  Here the raw value is recorded and
  the post-scan reconstruction tests the sentinel itself, so the clamp is not
  needed inside the body.
* Liveness read inside a body is the **pre-scan** per-slot liveness: a scan
  iteration reads only its own slot's key (``gather(keys, slot)``) and each slot
  is visited exactly once, and scan 2's key write on a live slot is the
  identity.  Hence ``claimed = live & (closest != I_INF)`` and ``create =
  not live & (seed != I_INF)`` reconstruct the two write masks from the stacked
  ``y`` alone.
* Scan 2's created keys are ``next_species_key + k`` for the ``k``-th create in
  slot order, exactly what the carried counter accumulated -- state keys are
  integral ``float32`` values below ``2**24``, where ``+1.0`` is exact.

Correctness envelope
--------------------
``reduce_min`` + one-hot equals ``argmin``/``argmax`` for every input whose
masked candidates are **finite**, which is what the genome's compatibility
kernel produces (a distance is a sum/max of finite key and attribute terms).
A ``NaN`` score would make the minimising form report "no candidate" where
``argmin`` picks the ``NaN`` row; that is a genome-side defect the host form
would also silently mis-assign, and it is not reachable from a shipped genome.

``evoxc.scan`` peels step ``0`` at the enclosing level (its ``y`` is
``reshape(step0, (1, ...))`` + ``broadcast`` to the full stack), so the body's
own y-write runs ``S - 1`` times per scan; the op count above counts the
per-iteration write, which is what a region body pays.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import evoxc

__all__ = ["ScanOperands", "assign_step", "reserve_step"]


@dataclass(frozen=True)
class ScanOperands:
    """Operands both step bodies close over, built once per generation.

    These are trace-time constants and static values (:func:`._kernels.speciate`
    builds them before either ``evoxc.scan``), so a body emits no ``constant``
    op of its own and a region closes over the enclosing SSA values.

    Attributes:
        keys: ``(S,) float32`` the **pre-scan** ``species_keys`` -- the liveness
            authority of both bodies and, for scan 2, the key a live slot keeps.
        pop_index: ``(P,) int32`` index vector ``[0, P)``.
        iinf: 0-d ``int32`` "no candidate" sentinel (``I_INF``).
        inf_f32: 0-d ``float32`` filler above every real score.
        zero_i32: 0-d ``int32`` ``0``.
        zero_f32: 0-d ``float32`` ``0.0`` -- the distance scan 1 writes.
        one_f32: 0-d ``float32`` ``1.0`` -- one created species.
        threshold: 0-d ``float32`` the compatibility threshold.
        gated (bool): ``True`` when the row is computed lazily behind one
            ``evoxc.cond`` (``species_size * pop_size <= GATED_ROWS``).
    """

    keys: Any
    pop_index: Any
    iinf: Any
    inf_f32: Any
    zero_i32: Any
    zero_f32: Any
    one_f32: Any
    threshold: Any
    gated: bool


def reserve_step(operands: ScanOperands, row_for_slot: Callable) -> Callable:
    """Scan 1's step: claim the nearest unassigned genome for one live slot.

    Args:
        operands (ScanOperands): The closed-over operands.
        row_for_slot (Callable): ``row_for_slot(slot, active) -> (P,) float32``
            the compatibility row of the pre-scan centre ``slot``, computed only
            when the traced ``active`` predicate holds (the gated path) or
            unconditionally (the dense path).

    Returns:
        Callable: ``(carry, slot) -> ((assign, dist), closest)`` -- the carried
        assignment/distance pair and the step's ``y``, the **raw** minimal index
        (``I_INF`` when nothing was left to claim), which ``evoxc.scan`` stacks
        into ``(S,) int32``.

    Notes:
        ``claim = live and a candidate exists`` and the write is the one-hot
        ``pop_index == closest``, exactly the host body's ``(pop_index ==
        closest) and claim``; the sentinel can never equal a ``pop_index`` entry,
        so the one-hot needs no separate clamp.  The lazy ``cond`` predicate is
        ``live and any pending`` (one plain reduce, and only on the gated path):
        an iteration that cannot claim must not pay for a row.
    """

    def step(carry: tuple, slot: Any) -> tuple:
        """One scan-1 iteration."""
        assign, dist = carry
        key_slot = evoxc.gather(operands.keys, slot, axis=0)
        live = evoxc.logical_not(evoxc.isnan(key_slot))
        pending = evoxc.isnan(assign)
        active = live
        if operands.gated:
            active = evoxc.logical_and(live, evoxc.reduce_max(pending, axes=0))
        distances = row_for_slot(slot, active)
        masked = evoxc.select(pending, distances, operands.inf_f32)
        best = evoxc.reduce_min(masked, axes=0)
        match = evoxc.logical_and(pending, evoxc.equal(distances, best))
        closest = evoxc.reduce_min(
            evoxc.select(match, operands.pop_index, operands.iinf), axes=0
        )
        claim = evoxc.logical_and(live, evoxc.not_equal(closest, operands.iinf))
        onehot = evoxc.logical_and(claim, evoxc.equal(operands.pop_index, closest))
        assign = evoxc.select(onehot, key_slot, assign)
        dist = evoxc.select(onehot, operands.zero_f32, dist)
        return (assign, dist), closest

    return step


def assign_step(
    operands: ScanOperands,
    row_for_centre: Callable,
    centre_for_slot: Callable,
) -> Callable:
    """Scan 2's step: threshold membership, materialising a dead slot on demand.

    Args:
        operands (ScanOperands): The closed-over operands.
        row_for_centre (Callable): ``row_for_centre(centre, active) -> (P,)
            float32`` the compatibility row of this slot's four centre rows.
        centre_for_slot (Callable): ``centre_for_slot(create, seed, slot)`` the
            four centre rows -- ``pop[seed]`` for a creating slot, the slot's
            rebuilt scan-1 row otherwise (the host body's ``selected_centre``).

    Returns:
        Callable: ``(carry, slot) -> ((assign, dist, counter), seed)`` -- the
        carried assignment/distance pair and the advanced rank-0 species-key
        counter, plus the step's ``y``, the **raw** first unassigned index
        (``I_INF`` when the population is fully assigned).

    Notes:
        ``create = not live and a seed exists`` and the raw seed is the step's
        ``y``; a creating slot takes the carried counter as its species key and
        bumps it (``select(create, counter + 1, counter)``), exactly the two
        ``keys_ext`` scatters the host body performed.  A live slot's key write
        there was the identity, so no key array is carried at all; the created
        rows are rebuilt once after the scan from the stacked ``y`` and the
        counter's final value.
    """

    def step(carry: tuple, slot: Any) -> tuple:
        """One scan-2 iteration."""
        assign, dist, counter = carry
        key_slot = evoxc.gather(operands.keys, slot, axis=0)
        live = evoxc.logical_not(evoxc.isnan(key_slot))
        unassigned = evoxc.isnan(assign)
        seed = evoxc.reduce_min(
            evoxc.select(unassigned, operands.pop_index, operands.iinf), axes=0
        )
        absent = evoxc.equal(seed, operands.iinf)
        create = evoxc.logical_and(evoxc.logical_not(live), evoxc.logical_not(absent))
        # ``centre_for_slot`` gathers ``pop[seed]`` unconditionally, so the
        # sentinel is redirected to row ``0`` (discarded by ``create``).
        centre = centre_for_slot(create, evoxc.select(absent, operands.zero_i32, seed), slot)
        active = evoxc.logical_or(live, create)
        distances = row_for_centre(centre, active)
        closer = evoxc.logical_or(unassigned, evoxc.less(distances, dist))
        within = evoxc.less(distances, operands.threshold)
        take = evoxc.logical_and(evoxc.logical_and(active, within), closer)
        key_slot = evoxc.select(create, counter, key_slot)
        assign = evoxc.select(take, key_slot, assign)
        dist = evoxc.select(take, distances, dist)
        counter = evoxc.select(
            create, evoxc.add(counter, operands.one_f32), counter
        )
        return (assign, dist, counter), seed

    return step
