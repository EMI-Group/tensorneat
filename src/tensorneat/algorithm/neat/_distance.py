"""Distance rows over one generation's population digest.

Speciation measures *every* species centre against the *same* population, twice
per generation, so the genome's two-phase distance API serves it directly:
:meth:`~tensorneat.genome.base.BaseGenome.prepare_distance` sorts the population
once per generation into a ``DistanceDigest``, and
:meth:`~tensorneat.genome.base.BaseGenome.measure_distance` turns one centre
genome into one ``(P,)`` distance row without re-sorting the population.

This module wraps that digest in the callable :meth:`SpeciesController.speciate`
consumes and adds the **batched** form: a centre payload of ``K`` rows (the whole
species table of a phase) is measured against the population in a *single*
``measure_distance`` call, and the ``(K, P)`` matrix is handed back to the scan,
which then reads its slot's row with one ``gather``.

``measure_distance`` requires the centre batch to be either ``1`` (broadcast
internally) or the digest's batch, so the wide call tiles **both** sides to
``K * P`` rows with the matching row order ``s * P + p``:

* the digest, ``(P, ...)`` -> ``(K * P, ...)`` -- block ``s`` is the population;
* the centres, ``(K, ...)`` -> ``(K * P, ...)`` -- rows ``s * P .. s * P + P-1``
  are centre ``s``.

Both tilings are a ``broadcast_to`` plus a ``reshape``, and the tiled digest is
memoised per batch width, so a generation that measures both phases with the
same number of centres tiles the population once and the graph reuses the same
``evoxc`` values.

Nothing here re-implements distance arithmetic: every number comes from the
genome's own kernel, so the rows are bit-identical to
:meth:`~tensorneat.genome.base.BaseGenome.execute_distance` on the same centre and
population.
"""

from __future__ import annotations

from typing import Any

import evoxc
import evoxc.numpy as enp

from tensorneat.common.errors import TensorNEATError

#: Maximum number of species centres measured by one ``measure_distance`` call.
#:
#: One call over the whole species table is the cheapest in dispatch terms, but
#: the compatibility kernel's transient arrays scale with ``centres * pop_size``
#: (the ``(rows, 2 * max_conns, key_width)`` masked gather of the long-key
#: verification alone is hundreds of MB at ``20 * 1000`` rows) and the numpy
#: reference interpreter keeps every intermediate of a run alive, so a
#: generation's footprint is the sum of them.  Grouping also keeps each group's
#: ``(rows, 2 * max_conns)`` sort inside the cache: at pop 1000, 20 centres in one
#: call cost 726 ms of ``speciate`` against 568 ms at 4 per call (500: 496 vs
#: 300 ms), and 8 per call is already worse again (763 / 340 ms).  The rows are
#: identical either way, because the compatibility kernel is row-independent.
#: The gate keeps the sparse regime on its lazy per-slot path, so this only ever
#: groups the dense regime's rows.
CENTRES_PER_CALL = 4


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


def tile_leading(x: Any, blocks: int) -> Any:
    """Stack ``blocks`` copies of a batch-first tensor along axis 0.

    Args:
        x (Any): Traced tensor of shape ``(P, *rest)``.
        blocks (int): Static number of copies ``K``.

    Returns:
        Any: ``(K * P, *rest)`` with row ``s * P + p`` equal to ``x[p]``.
    """
    shape = tuple(int(size) for size in x.shape)
    return evoxc.reshape(
        enp.broadcast_to(x, (blocks,) + shape), (blocks * shape[0],) + shape[1:]
    )


def repeat_leading(x: Any, repeats: int) -> Any:
    """Repeat every row of a batch-first tensor ``repeats`` times along axis 0.

    Args:
        x (Any): Traced tensor of shape ``(K, *rest)``.
        repeats (int): Static repeat count, the population size.

    Returns:
        Any: ``(K * repeats, *rest)`` with rows ``k * repeats .. k * repeats +
        repeats - 1`` all equal to ``x[k]`` -- the row order that matches
        :func:`tile_leading`.
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
    traced -- so the digest tiling cache below lives on the host and each batch
    width is tiled at most once per graph.

    Args:
        genome (Any): The genome owning ``measure_distance`` and the digest's
            gene layout.
        digest (Any): ``DistanceDigest`` returned by
            ``genome.prepare_distance(state, population)``.
        population (Any): The payload the digest was prepared from.  Every call
            is checked against its static shape, so a payload from a different
            generation or population size is rejected instead of silently
            measuring against a stale digest.
        pop_size (int): Static population size ``P`` (the digest's batch).
        species_size (int): Static number of species slots ``S``; a centre
            payload of exactly this batch width is measured as one wide call.

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
        self._tiled: dict = {}

    def __call__(self, state: Any, centre: Any, pop: Any = None) -> Any:
        """Distance rows of ``centre`` against the digested population.

        Args:
            state (Any): Shared state (forwarded to the genome kernel).
            centre (Any): Centre payload, batch ``1`` (one centre, broadcast by
                the genome kernel), batch ``S`` (the whole species table of a
                phase) or batch ``P`` (a population-shaped centre, measured
                row-wise).
            pop (Any, optional): The population payload ``speciate`` measures
                against; its static shape must match the digested population's.

        Returns:
            Any: ``(P,)`` float32 for a batch-1 or batch-``P`` centre, ``(S, P)``
            float32 for a batch-``S`` one.

        Raises:
            TensorNEATError: If ``pop`` does not have the digested population's
                static shape, or if the genome kernel rejects the centre batch
                (anything other than ``1``, ``S`` or ``P``).
        """
        if pop is not None:
            self._check_population(pop)
        batch = int(centre[0].shape[0])
        if batch != 1 and batch == self.species_size:
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
            centre (Any): Centre payload; a batch of ``1`` is broadcast to the
                population by the genome kernel, a batch of ``P`` is measured
                row-wise.

        Returns:
            Any: ``(P,)`` float32 distances.
        """
        return self.genome.measure_distance(state, centre, self.digest)

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
            ``K == 1`` takes the narrow path: the genome kernel already
            broadcasts a single centre, so tiling would only duplicate work.
            Otherwise the centres are measured in groups of at most
            ``CENTRES_PER_CALL`` rows.  One call over the whole table is the
            fastest option, but its transient arrays grow with ``K * P`` -- an
            ``(K * P, 2 * max_conns, key_width)`` masked gather alone is
            hundreds of MB at ``K = 20``, ``P = 1000`` -- and the reference
            interpreter keeps every intermediate alive for the whole run.
            Grouping bounds that peak while costing one extra dispatch per
            group; the rows are identical either way, because the compatibility
            kernel is row-independent.
        """
        count = int(centres[0].shape[0])
        if count == 1:
            return evoxc.reshape(self.row(state, centres), (1, self.pop_size))
        block = min(count, CENTRES_PER_CALL)
        if block == count:
            return self._measure_block(state, centres, count)
        return evoxc.concatenate(
            [
                self._measure_block(
                    state,
                    _rows(centres, start, min(block, count - start)),
                    min(block, count - start),
                )
                for start in range(0, count, block)
            ],
            axis=0,
        )

    def _measure_block(self, state: Any, centres: Any, count: int) -> Any:
        """Measure ``count`` centres against the population in one wide call.

        Args:
            state (Any): Shared state.
            centres (Any): Centre payload of exactly ``count`` rows.
            count (int): Static number of centres in ``centres``.

        Returns:
            Any: ``(count, P)`` float32 distances.
        """
        wide_centres = type(centres)(
            *(repeat_leading(field, self.pop_size) for field in centres)
        )
        flat = self.genome.measure_distance(
            state, wide_centres, self._tiled_digest(count)
        )
        return evoxc.reshape(flat, (count, self.pop_size))

    def _tiled_digest(self, blocks: int) -> Any:
        """Digest tiled to ``blocks`` population copies, memoised per width.

        Args:
            blocks (int): Static number of population blocks ``K``.

        Returns:
            Any: A ``DistanceDigest`` whose leaves are ``(K * P, ...)``; the
            same traced values on every request with the same ``blocks``.
        """
        tiled = self._tiled.get(blocks)
        if tiled is None:
            tiled = type(self.digest)(
                *(tile_leading(leaf, blocks) for leaf in self.digest)
            )
            self._tiled[blocks] = tiled
        return tiled
