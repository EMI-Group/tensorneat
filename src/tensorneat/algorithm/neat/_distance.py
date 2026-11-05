"""Distance rows over one generation's population digest.

Speciation measures *every* species centre against the *same* population, twice
per generation, so the genome's two-phase distance API serves it directly:
:meth:`~tensorneat.genome.base.BaseGenome.prepare_distance` sorts the population
once per generation into a ``DistanceDigest``, and
:meth:`~tensorneat.genome.base.BaseGenome.measure_distance` turns one centre
genome into one ``(P,)`` distance row without re-sorting the population.

Shape contract
--------------
Every call this module makes to ``genome.measure_distance`` carries one centre
spelled by ``state.distance_centre``: a host string read at trace time
(``_validation.distance_centre_mode``), never a traced value.

* ``"tiled"`` -- **the default when the entry is absent**, i.e. what a bare
  ``State`` gets: the centre row repeated to ``pop_size`` identical rows (one
  rank-raising ``evoxc.broadcast`` per field, built by ``_kernels.centre_payload``).
  That is the population-shaped payload the genome's row-wise merge route takes,
  so no call can select its dense single-centre route -- whose per-call scatters
  a GPU lowers single-threaded, a measured 2x device-time regression.
* ``"single"``: a one-individual payload (leading axis ``1``, one
  ``evoxc.reshape`` per field), which selects that dense route.  It is the
  cheaper spelling on the numpy interpreter and the dearer one on a GPU, so the
  pipeline picks the mode from the device it runs on.

Both spellings carry the same centre row and return the same ``(P,)`` floats.
The ``group * pop_size`` batch the genome also accepts belongs to a
``group_digest``-tiled digest, which this module never builds.

:meth:`DistanceRows.__call__` is single-valued: it accepts a centre payload of
batch ``1`` or batch ``pop_size`` and rejects every other batch width; in
``"tiled"`` mode a batch-``1`` centre is tiled to ``pop_size`` rows first, so the
mode's contract holds for every caller, while in ``"single"`` mode both widths
are measured as they stand.
It **is** the production route: ``_kernels.speciate`` calls ``distance_fn(...)``
from its per-iteration ``compute_row`` bodies for the gated ``reserve`` and
``assign_members`` scans, and unconditionally for every ``assign_members``
iteration on the dense path (a small ``Pipeline`` run traced 4 ``__call__``
invocations on the gated path and 2 on the dense one, plus one :meth:`matrix`:
``evoxc.scan`` peels step 0 at the enclosing level and traces the body again, so
each scan body is traced twice).
:meth:`DistanceRows.row` is the same measurement spelled explicitly for one
centre, and :meth:`DistanceRows.matrix` is the explicit whole-table form: one
such call per centre row, run as a single fixed-length ``evoxc.scan`` over the
``S`` centre rows, so an ``(S, P)`` table is ``S`` calls of one centre row
rather than one ``S * P`` batch.  The rows are identical either way, because the
compatibility kernel is row-independent, and one centre per call keeps the
kernel's transient footprint at one centre instead of ``S`` of them, which
the scan body then releases again before the next iteration.  ``matrix`` is
never selected by ``__call__``: a caller that wants a whole table asks for it by
name, which is what the dense path's scan 1 does.

Nothing here re-implements distance arithmetic: every number comes from the
genome's own kernel, so the rows are bit-identical to
:meth:`~tensorneat.genome.base.BaseGenome.execute_distance` on the same centre and
population.

Measured cost
-------------
Numpy reference backend, ``species_size=20``, one configuration per fresh
process, 3 warmup + 20 timed ``Pipeline.step`` after the build step, interleaved
rounds alternating a tree that tiles the centre to ``pop_size`` rows (and so
takes the genome's row-wise merge) against this one, medians of 8 rounds at
``P = 500`` and of 4 at ``P = 1000`` / ``P = 100``, ``XOR``, seed 42,
``is_detailed=False``: ``P = 500`` **157.5 -> 111.4 ms** per generation and
222.3 -> 198.5 MiB peak RSS, ``P = 1000`` **278.4 -> 172.3 ms** and
352.0 -> 304.1 MiB, and the gated ``P = 100`` cell 42.6 -> 40.7 ms at
113.4 -> 112.3 MiB.
The dispatch count moves the other way: 7 678.7 -> 7 958.7 executed ops per
generation at ``P = 500`` (7 682.0 -> 7 962.0 at ``P = 1000``, 5 335.7 -> 5 497.7
on the gated cell), i.e. +7 per compatibility call, because the dense
single-centre route emits more ops than the merge (the genome records 102
against 94) while spending far less per op on a ``P``-row-wide payload
(genome-side record: 1.35 vs 2.39 ms per call at ``P = 500``).
At these populations the interpreter's per-op cost is not the binding term; the
row width of the merge's ``(rows, 2 * max_conns)`` intermediates is.
The table costs exactly one ``measure_distance`` call per centre row -- 40 calls per
generation at ``species_size = 20``, 20 here and 20 in scan 2 -- and what the
scan buys is the *region* around those calls, not fewer of them.

Pre-measuring *both* speciation scans needs a ``cond`` for scan 2's
data-dependent ``pop[seed]`` centre and costs ~50 % more peak RSS for no timed
gain.

Grouping several centres into one ``measure_distance`` call is a measured
rejection and an unwired option, not a shipped path: the genome's
``tensorneat.genome.operations.distance.group_digest`` / ``GroupedDigest`` route
is tested and bit-identical to the separate calls, and it is a row-wise-merge
option only (the tiled digest carries no dense table), so it competes with the
merge, not with the shipped one-individual call.
Measured against that merge over the whole generation at ``P = 500`` /
``species_size = 20``, scan 1 grouped two centres per call dispatched 6 589 ops
(-13 %) at 253.2 MiB peak RSS (+8.3 %) against the ungrouped 7 592 at 233.8 MiB,
and its wall time was bimodal at process level: 152-165 ms in the fast mode,
196-253 ms in 7 of 13 fresh default-environment processes, an intrinsic ~4 % win
only with glibc's mmap/trim thresholds pinned to 1 GiB.
The mechanism is the row width: the kernel's ``(rows, 2 * max_conns)``
intermediates double at ``k = 2`` and grow linearly beyond (273.8 MiB, 353.1 at
``k = 4``, 593 / 753 at ``k = 10`` / ``k = 20``), so a wider call's per-element
cost eats the dispatch it saves -- at ``k = 4`` the per-op-name profile moved
``argsort`` 25.5 -> 37.0 ms, ``gather`` 21.5 -> 36.1 ms, ``cast`` 8.0 -> 17.7 ms
and ``reshape`` 4.3 -> 8.9 ms while dispatches fell 8061 -> 4885 against a 28 %
wall-time increase, and a second pre-measured table for scan 2 at ``k = 2`` read
+58 % wall time (258.96 ms at 6 661 ops).

Peak memory
-----------
``evoxc``'s numpy interpreter keeps every SSA result of a block in that block's
environment until the block returns, and pops a nested region's environment per
iteration.  A kernel inside a ``scan`` body therefore pays one region's
transients at a time, while a kernel at the top level keeps them alive for the
rest of the generation -- which is what the whole-table scan is for, and why the
top-level spelling of the same rows is never used.
The measured footprint of the shipped arrangement is the A/B's 198.5 MiB at
``P = 500`` and 304.1 MiB at ``P = 1000``; build follows the same split, because
a loop body is traced once while a static loop over ``S`` centres traces ``S``
copies of the kernel.
Any future hoist out of the region has to report peak RSS and build next to its
wall time.
"""

from __future__ import annotations

from typing import Any

import evoxc
import evoxc.numpy as enp

from tensorneat.common.errors import TensorNEATError

from ._kernels import centre_payload
from ._validation import distance_centre_mode


class DistanceRows:
    """Compatibility distances of species centres against one digested population.

    One instance is built per generation (in :meth:`NEAT.tell`) and handed to
    :meth:`SpeciesController.speciate` as its ``distance_fn``.  It is a plain
    traced helper -- created and called while the generation graph is being
    traced.

    Args:
        genome (Any): The genome owning ``measure_distance`` and the digest's
            gene layout.
        digest (Any): ``DistanceDigest`` returned by
            ``genome.prepare_distance(state, population)``.
        population (Any): The payload the digest was prepared from.  Every call
            is checked against its static shape, so a payload from a different
            generation or population size is rejected instead of silently
            measuring against a stale digest.
        pop_size (int): Static population size ``P`` -- the digest's batch, and
            the row count each measured centre is paired with.

    Raises:
        TensorNEATError: If the digest's population axis is not ``pop_size``.
    """

    def __init__(
        self,
        genome: Any,
        digest: Any,
        population: Any,
        pop_size: int,
    ) -> None:
        self.genome = genome
        self.digest = digest
        self.population = population
        self.pop_size = int(pop_size)
        digest_pop_size = int(digest.node_count.shape[0])
        if digest_pop_size != self.pop_size:
            raise TensorNEATError(
                "DistanceRows: the distance digest holds "
                f"{digest_pop_size} genomes but pop_size is {self.pop_size}; "
                "prepare the digest from the current population"
            )
        if int(population[0].shape[0]) != self.pop_size:
            raise TensorNEATError(
                "DistanceRows: the population payload has batch "
                f"{int(population[0].shape[0])} but pop_size is {self.pop_size}"
            )

    def __call__(self, state: Any, centre: Any, pop: Any = None) -> Any:
        """Distance rows of ``centre`` against the digested population.

        Args:
            state (Any): Shared state (forwarded to the genome kernel).
            centre (Any): Centre payload of batch ``1`` (one individual, measured
                against the digest's population rows) or batch ``pop_size`` (a
                population-shaped centre, measured row-wise as it stands).
            pop (Any, optional): The population payload ``speciate`` measures
                against; its static shape must match the digested population's.

        Returns:
            Any: ``(P,)`` float32 distances.

        Raises:
            TensorNEATError: If ``pop`` does not have the digested population's
                static shape, or if the centre batch is neither ``1`` nor ``P``.

        Note:
            This is the callable ``_kernels.speciate`` invokes as
            ``distance_fn(state, centre, pop)``: the gated path reaches it from
            both scan bodies, the dense path from every ``assign_members``
            iteration.  Those call sites always hand over a batch-``1`` centre and
            the digested population, so the ``pop_size``-wide branch and both
            rejections in the body are unexercised from inside the package.
            A whole ``(S, P)`` species table is not a ``__call__`` shape either:
            ask for it by name with :meth:`matrix`, so a table can never be
            mistaken for a centre payload.
        """
        if pop is not None:
            self._check_population(pop)
        batch = int(centre[0].shape[0])
        if batch != self.pop_size and batch != 1:
            raise TensorNEATError(
                "DistanceRows: a distance row takes a centre payload of batch 1 "
                f"or pop_size ({self.pop_size}), got batch {batch}; call "
                "matrix() directly for a whole species table"
            )
        return self.row(state, centre)

    def _check_population(self, pop: Any) -> None:
        """Reject a population payload whose static shape the digest cannot serve.

        The digest is the *only* thing the distances are measured against, so the
        payload a caller hands to ``speciate`` is a witness that the two agree:
        a payload with different static shapes cannot be the one the digest was
        prepared from.

        Args:
            pop (Any): The payload ``speciate`` is measuring against.

        Raises:
            TensorNEATError: If a field's static shape differs from the digested
                population's.
        """
        for name, field, expected in zip(
            ("node_keys", "node_attrs", "conn_keys", "conn_attrs"),
            pop,
            self.population,
        ):
            if tuple(int(size) for size in field.shape) != tuple(
                int(size) for size in expected.shape
            ):
                raise TensorNEATError(
                    f"DistanceRows: the measured population's {name} has shape "
                    f"{tuple(int(size) for size in field.shape)} but the digest "
                    f"was prepared from {tuple(int(size) for size in expected.shape)}; "
                    "build the digest from the same payload the scans compare "
                    "against"
                )

    def row(self, state: Any, centre: Any) -> Any:
        """``(P,)`` distance row of one centre payload (batch ``1`` or ``P``).

        Args:
            state (Any): Shared state.
            centre (Any): Centre payload of one individual (batch ``1``) or a
                population-shaped one (batch ``pop_size``); both are measured as
                they stand.

        Returns:
            Any: ``(P,)`` float32 distances.

        Raises:
            TensorNEATError: If the centre batch is neither ``1`` nor ``P``.

        Note:
            The batch-``1`` centre goes to
            :meth:`~tensorneat.genome.base.BaseGenome.measure_distance`
            unchanged: the genome pairs that one centre against the digest's
            ``pop_size`` rows, taking its dense single-centre route when the static
            gate on the genome holds and broadcasting the centre internally
            otherwise.
        """
        batch = int(centre[0].shape[0])
        if batch != self.pop_size and batch != 1:
            raise TensorNEATError(
                f"DistanceRows: a distance row takes a centre payload of batch 1 "
                f"or pop_size ({self.pop_size}), got batch {batch}"
            )
        if batch == 1 and distance_centre_mode(state) == "tiled":
            # The mode's contract is population-shaped, so a scale-by-one centre
            # is tiled here rather than measured through the genome's dense route.
            centre = centre_payload(
                centre,
                tuple(field[0] for field in centre),
                self.pop_size,
                True,
            )
        return self.genome.measure_distance(state, centre, self.digest)

    def matrix(self, state: Any, centres: Any) -> Any:
        """``(S, P)`` distance matrix of a whole species table (explicit form).

        Args:
            state (Any): Shared state.
            centres (Any): Centre payload of ``K`` rows -- the ``(S, ...)``
                centre arrays of a speciation phase.

        Returns:
            Any: ``(K, P)`` float32 distances, row ``s`` against the whole
            population.

        Note:
            This is the explicit whole-table form, never reached through
            :meth:`__call__`: a caller holding an ``(S, ...)`` centre table asks
            for it by name, so a table cannot be mistaken for a batch-``1`` centre
            when ``S == 1``.  The one in-package caller is
            ``_kernels.speciate``'s dense path, which measures scan 1's whole
            centre table this way before the reservation scan.
            One ``measure_distance`` call per centre row, run as one fixed-length
            ``evoxc.scan`` over the ``K`` centre rows: the kernel never sees a
            ``K * P`` batch, and because a scan body's region environment is
            popped per iteration, only one centre's transients are alive at a
            time.  Every call carries the gathered row with its leading axis of
            length ``1``.  Measuring the same rows at the top level of the
            generation graph would keep all ``K`` of them resident -- see
            `Peak memory` in the module docstring.
        """
        count = int(centres[0].shape[0])
        tiled = distance_centre_mode(state) == "tiled"

        def body(carry: Any, slot: Any) -> Any:
            """One centre row of the table, measured inside the scan region."""
            rows = tuple(evoxc.gather(field, slot, axis=0) for field in centres)
            return carry, self.genome.measure_distance(
                state,
                centre_payload(self.population, rows, self.pop_size, tiled),
                self.digest,
            )

        _, stacked = evoxc.scan(
            body,
            enp.zeros((1,), dtype=evoxc.float32),
            enp.arange(0, count, dtype=evoxc.int32),
        )
        return stacked
