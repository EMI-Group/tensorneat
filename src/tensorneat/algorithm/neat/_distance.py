"""Distance rows over one generation's population digest.

Speciation measures *every* species centre against the *same* population, twice
per generation, so the genome's two-phase distance API serves it directly:
:meth:`~tensorneat.genome.base.BaseGenome.prepare_distance` sorts the population
once per generation into a ``DistanceDigest``, and
:meth:`~tensorneat.genome.base.BaseGenome.measure_distance` turns one centre
genome into one ``(P,)`` distance row without re-sorting the population.

Shape contract
--------------
Every call this module makes to ``genome.measure_distance`` carries a
**population-shaped** centre payload: exactly ``pop_size`` rows, row ``p`` being
the centre measured against population member ``p``, next to the digest of the
same batch width.  ``measure_distance`` accepts no other width besides a
broadcastable ``1``, so the population-shaped form is the one shape that works
for both a single centre and a whole species table without a second digest
tiling.

:meth:`DistanceRows.row` tiles its one centre row to ``pop_size`` identical rows
with :func:`repeat_leading` and measures it in a single call.
:meth:`DistanceRows.matrix` concatenates one such call per species slot, so an
``(S, P)`` table is ``species_size`` calls of ``pop_size`` rows rather than one
``K * P`` batch: the rows are identical either way, because the compatibility
kernel is row-independent, and a per-centre call keeps the transient footprint
of the kernel at one population instead of ``K`` of them.

Nothing here re-implements distance arithmetic: every number comes from the
genome's own kernel, so the rows are bit-identical to
:meth:`~tensorneat.genome.base.BaseGenome.execute_distance` on the same centre and
population.

Measured cost
-------------
``speciate`` on the numpy reference backend, ``species_size=20``, medians of
three runs of five after four warm generations, same box:

* ``P = 100``: 30.7-32.1 ms with the old ``K = 4`` centre grouping, 31.6-33.3 ms
  with one call per centre (parity, within the box's drift);
* ``P = 500``: 235.0-237.5 ms before, 229.5-240.0 ms now (parity);
* ``P = 1000``: 439.8-441.8 ms before, 387.3-391.4 ms now (about 12 % faster).

The per-centre call shape itself is what the population-shaped contract costs:
measured with the rows still pre-computed before each scan (``2 * species_size``
distance trace sites and scan 2's per-iteration ``create`` ``cond``, 11 111 graph
ops against 3 353), it was +18 % at ``P = 500``, because four-centre calls
amortise the interpreter's per-op overhead over four populations.  Measuring the
same rows inside the scan bodies instead -- one trace site per phase, 1 361 ops,
no ``cond`` -- wins it back and then some: the wide batches were also
materialising ``species_size * pop_size`` rows per digest leaf, which is the cost
that dominated at ``P = 1000``.  The grouping and the pre-measurement were the
expensive parts, not the per-centre call.
"""

from __future__ import annotations

from typing import Any

import evoxc
import evoxc.numpy as enp

from tensorneat.common.errors import TensorNEATError


def _rows(payload: Any, start: int, count: int) -> Any:
    """Slice ``count`` consecutive rows out of every field of a payload.

    Args:
        payload (Any): Payload whose fields are batch-first ``(K, *rest)``.
        start (int): Static first row.
        count (int): Static number of rows.

    Returns:
        Any: Payload of the same type with ``(count, *rest)`` fields.
    """
    rank = len(payload[0].shape)
    slicer = (slice(start, start + count),) + (slice(None),) * (rank - 1)
    return type(payload)(*(field[slicer] for field in payload))


def repeat_leading(x: Any, repeats: int) -> Any:
    """Repeat every row of a batch-first tensor ``repeats`` times along axis 0.

    Args:
        x (Any): Traced tensor of shape ``(K, *rest)``.
        repeats (int): Static repeat count, the population size.

    Returns:
        Any: ``(K * repeats, *rest)`` with rows ``k * repeats .. k * repeats +
        repeats - 1`` all equal to ``x[k]`` -- so a ``(1, ...)`` input becomes a
        population-shaped payload whose every row is that one centre.
    """
    shape = tuple(int(size) for size in x.shape)
    spread = evoxc.reshape(x, (shape[0], 1) + shape[1:])
    return evoxc.reshape(
        enp.broadcast_to(spread, (shape[0], repeats) + shape[1:]),
        (shape[0] * repeats,) + shape[1:],
    )


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
        pop_size (int): Static population size ``P`` (the digest's batch and the
            row count of every centre payload this class builds).
        species_size (int): Static number of species slots ``S``; a centre
            payload of exactly this batch width is measured row by row into the
            ``(S, P)`` matrix form.

    Raises:
        TensorNEATError: If the digest's population axis is not ``pop_size``.
    """

    def __init__(
        self,
        genome: Any,
        digest: Any,
        population: Any,
        pop_size: int,
        species_size: int,
    ) -> None:
        self.genome = genome
        self.digest = digest
        self.population = population
        self.pop_size = int(pop_size)
        self.species_size = int(species_size)
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
            centre (Any): Centre payload, batch ``1`` (one centre), batch ``P``
                (a population-shaped centre, measured row-wise) or batch ``S``
                (the whole species table of a phase, measured one row at a
                time).
            pop (Any, optional): The population payload ``speciate`` measures
                against; its static shape must match the digested population's.

        Returns:
            Any: ``(P,)`` float32 for a batch-1 or batch-``P`` centre, ``(S, P)``
            float32 for a batch-``S`` one.

        Raises:
            TensorNEATError: If ``pop`` does not have the digested population's
                static shape, or if the centre batch is neither ``1``, ``P`` nor
                ``S``.

        Note:
            A batch of ``P`` is a *population-shaped* centre and wins over a
            batch of ``S`` when the two coincide (``pop_size == species_size``),
            because that is the shape every ``speciate`` call carries; a whole
            centre table with more rows than one population is measured through
            :meth:`matrix`.
        """
        if pop is not None:
            self._check_population(pop)
        batch = int(centre[0].shape[0])
        if batch != self.pop_size and batch != 1 and batch == self.species_size:
            return self.matrix(state, centre)
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
            centre (Any): Centre payload; a batch of ``1`` is tiled to
                ``pop_size`` identical rows, a batch of ``pop_size`` is already
                population-shaped and measured as it stands.

        Returns:
            Any: ``(P,)`` float32 distances.

        Raises:
            TensorNEATError: If the centre batch is neither ``1`` nor ``P``.
        """
        batch = int(centre[0].shape[0])
        if batch == self.pop_size:
            prepared = centre
        elif batch == 1:
            prepared = self.population_shaped(centre)
        else:
            raise TensorNEATError(
                f"DistanceRows: a distance row takes a centre payload of batch 1 "
                f"or pop_size ({self.pop_size}), got batch {batch}"
            )
        return self.genome.measure_distance(state, prepared, self.digest)

    def population_shaped(self, centre: Any) -> Any:
        """Tile a centre payload to exactly ``pop_size`` identical rows.

        Args:
            centre (Any): Centre payload of ``K`` rows.

        Returns:
            Any: ``(pop_size, *rest)`` payload in which every row repeats the
            centre row, which is the shape ``measure_distance`` pairs against the
            digest.
        """
        return type(centre)(*(repeat_leading(field, self.pop_size) for field in centre))

    def matrix(self, state: Any, centres: Any) -> Any:
        """``(S, P)`` distance matrix of a whole species table.

        Args:
            state (Any): Shared state.
            centres (Any): Centre payload of ``K`` rows -- the ``(S, ...)``
                centre arrays of a speciation phase.

        Returns:
            Any: ``(K, P)`` float32 distances, row ``s`` against the whole
            population.

        Note:
            One ``row`` call per centre row, concatenated along axis ``0``: each
            call carries a population-shaped centre, so the kernel always sees
            the batch width the digest was prepared with.  Concatenating rows of
            single calls keeps every transient at one population instead of
            ``K`` of them; the static row loop is trace-time only and costs no
            graph op.
        """
        count = int(centres[0].shape[0])
        rows = [
            evoxc.reshape(self.row(state, _rows(centres, slot, 1)), (1, self.pop_size))
            for slot in range(count)
        ]
        if count == 1:
            return rows[0]
        return evoxc.concatenate(rows, axis=0)
