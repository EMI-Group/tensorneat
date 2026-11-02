"""Origin (NEAT-paper) connection gene: ``DefaultConn`` plus a historical marker.

The weight machinery is inherited unchanged; this family adds one ``int32`` key
column and changes the genetic-operator semantics (whole-gene crossover).
Batch-first: the marker is read from the key columns, so alignment stays an exact
integer comparison and never touches the float attributes.
"""

import evoxc

from ....common import State
from .default import DefaultConn


class OriginConn(DefaultConn):
    """Connection gene of the original NEAT paper (EMI-Group/tensorneat issue 11).

    Key columns are ``(input_index, output_index, historical_marker)``; the single
    float32 attribute column is the inherited ``weight``. The historical marker is
    the NEAT-paper gene-alignment id: connections created by the same historical
    split are meant to share a marker and stay homologous during crossover/distance
    even when their endpoints were re-routed.

    The gene only *reads* that column: marker identity **across genomes** is the
    allocator's responsibility, because a marker drawn independently in each
    individual makes the same structural innovation non-homologous between genomes.
    ``algorithm/neat/neat.py::_new_gene_keys`` currently derives new markers from each
    individual's own maximum marker, so on an evolved population full-tuple homology
    measures 7.5-14.6 percentage points below ``(input_index, output_index)`` homology
    (generation 100, seeds 42-44); a caller that wants exact marker homology must pass
    shared markers through :meth:`~.base.BaseConn.from_attrs` or allocate them from a
    population-wide innovation id.

    Differences from :class:`.default.DefaultConn`: key column 2 carries the marker,
    :meth:`crossover` picks a whole gene with one draw, and :meth:`get_historical_marker`
    exposes the marker. Reference: https://github.com/EMI-Group/tensorneat/issues/11
    """

    fixed_attrs: tuple[str, ...] = ("input_index", "output_index", "historical_marker")

    def crossover(
        self,
        state: State,
        key: evoxc.SymbolicTensor,
        attrs1: evoxc.SymbolicTensor,
        attrs2: evoxc.SymbolicTensor,
    ) -> evoxc.SymbolicTensor:
        """Pick one parent's whole attribute vector for each gene pair.

        A ``(*shape, 1)`` broadcast of one normal draw per gene row selects between
        the two parents' attribute rows, so an offspring gene inherits its weight
        from exactly one parent (NEAT-paper gene-level inheritance).

        Args:
            state: Host configuration container.
            key: Rank-0 ``evoxc.random`` key for the whole batch.
            attrs1: Attribute rows of the first parent, ``(*shape, 1)`` float32 for
                this family.
            attrs2: Key-aligned attribute rows of the second parent, same shape.

        Returns:
            Child attribute rows, ``(*shape, attr_width)`` float32, each row copied
            verbatim from one parent.
        """
        batch_shape = tuple(attrs1.shape[:-1])
        draw = evoxc.random.normal(
            key, (*batch_shape, 1), 0.0, 1.0, dtype=evoxc.float32
        )
        return evoxc.select(evoxc.greater(draw, 0.0), attrs1, attrs2)

    def get_historical_marker(
        self, state: State, key_row: evoxc.SymbolicTensor
    ) -> evoxc.SymbolicTensor:
        """Read the historical marker, key column 2 of this family.

        Args:
            state: Host configuration container.
            key_row: ``(*shape, 3)`` int32 key columns; column 2 is the marker and is
                ``I_INF`` for absent connection rows.

        Returns:
            ``(*shape,)`` int32 historical markers, compared exactly (never a float
            attribute), which is what makes crossover/distance alignment an integer
            match.
        """
        return self.key_column(key_row, "historical_marker")
