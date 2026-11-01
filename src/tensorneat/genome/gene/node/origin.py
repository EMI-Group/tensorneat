"""Origin (NEAT-paper) node gene: ``DefaultNode`` attributes, whole-gene crossover.

The layout is identical to :class:`.default.DefaultNode` -- the difference is pure
genetic-operator semantics, so nothing in the forward path changes.
Batch-first: crossover draws one scalar per gene pair, never one draw per attribute,
and is expressed with ``select`` rather than branching on traced values.
"""

import evoxc

from ....common import State
from .default import DefaultNode


class OriginNode(DefaultNode):
    """Node gene of the original NEAT paper (see EMI-Group/tensorneat issue 11).

    Same float32 attribute columns as :class:`.default.DefaultNode`
    (``bias``, ``response``, ``aggregation``, ``activation``), same aggregation,
    combine, activation and code-generation hooks; only :meth:`crossover` differs:
    the original paper aligns genes and copies a whole gene, so a single random draw
    picks the entire attribute vector from one parent instead of mixing attributes.

    Reference: https://github.com/EMI-Group/tensorneat/issues/11
    """

    def crossover(
        self,
        state: State,
        key: evoxc.SymbolicTensor,
        attrs1: evoxc.SymbolicTensor,
        attrs2: evoxc.SymbolicTensor,
    ) -> evoxc.SymbolicTensor:
        """Pick one parent's whole attribute vector for each gene pair.

        A ``(*shape, 1)`` draw per gene row is broadcast over the attribute axis, so an
        offspring gene never mixes the columns of two different parents (NEAT-paper
        gene-level inheritance); ``draw > 0`` selects the first parent, exactly the
        strict comparison :meth:`...base.BaseGene.crossover` uses per column.

        Args:
            state: Host configuration container.
            key: Rank-0 ``evoxc.random`` key for the whole batch.
            attrs1: Attribute rows of the first parent, ``(*shape, attr_width)``
                float32.
            attrs2: Key-aligned attribute rows of the second parent, same shape.

        Returns:
            Child attribute rows, ``(*shape, attr_width)`` float32, each row copied
            verbatim from one parent.
        """
        if not self.attr_specs:
            return attrs1
        shape = tuple(attrs1.shape[:-1])
        # One draw per gene row, broadcast over the attribute axis; a rank-0 row is
        # drawn as shape ``(1,)`` because the numpy reference backend rejects a 0-d
        # random output (see ``..base.BaseGene.new_random_attrs``).
        draw = evoxc.random.normal(
            key, (*shape, 1) if shape else (1,), 0.0, 1.0, dtype=evoxc.float32
        )
        return evoxc.select(evoxc.greater(draw, 0.0), attrs1, attrs2)
