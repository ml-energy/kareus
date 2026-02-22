"""Automatic partition formation from TensorGraphs.

Takes SEPARATE forward and backward ``TensorGraph`` instances (already built
with correct port connections) and forms interleaved nanobatch partitions.

The core algorithm:

1. **Split by communication boundaries** — each ``CommunicationOp`` in the
   flat op list terminates a segment of ``ComputeOp``s.
2. **Interleave nanobatches** — for 2 nanobatches, each segment produces
   two partitions (one per NB).  The communication-overlap assignment is:
   - NB0 partition waits for NB1's comm from the *previous* segment
   - NB1 partition waits for NB0's comm from the *current* segment
"""

from __future__ import annotations

from typing import List, Optional, Tuple, Type, Union

from .backward_partition import BackwardPartition
from .forward_partition import ForwardPartition
from .partition_base import PartitionBase
from .tensor_graph import CommunicationOp, ComputeOp, TensorGraph

# (list_of_compute_ops, trailing_communication_op_or_None)
Segment = Tuple[List[ComputeOp], Optional[CommunicationOp]]


class PartitionBuilder:
    """Build interleaved nanobatch partitions from forward/backward graphs.

    Args:
        forward_graph: ``TensorGraph`` for the forward pass (ops in
            execution order).
        backward_graph: ``TensorGraph`` for the backward pass (ops
            already in reverse execution order).
    """

    def __init__(
        self,
        forward_graph: TensorGraph,
        backward_graph: TensorGraph,
    ) -> None:
        self.forward_graph = forward_graph
        self.backward_graph = backward_graph

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #

    def build_forward_partitions(self) -> List[ForwardPartition]:
        """Form interleaved forward partitions from the forward graph."""
        return self._form_partitions(
            self.forward_graph.ops,
            ForwardPartition,
            direction="fwd",
        )

    def build_backward_partitions(self) -> List[BackwardPartition]:
        """Form interleaved backward partitions from the backward graph."""
        return self._form_partitions(
            self.backward_graph.ops,
            BackwardPartition,
            direction="bwd",
        )

    # ------------------------------------------------------------------ #
    #  Core algorithm
    # ------------------------------------------------------------------ #

    def _form_partitions(
        self,
        ops: List[Union[ComputeOp, CommunicationOp]],
        partition_class: Type[PartitionBase],
        direction: str,
    ) -> list:
        """Form interleaved 2-nanobatch partitions from a flat op list.

        For each segment ``(comp_ops, comm_after)``:
        - NB0 partition: compute ``comp_ops``, comm = ``prev_comm``
          (wait for NB1's comm from the previous segment)
        - NB1 partition: compute ``comp_ops``, comm = ``comm_after``
          (wait for NB0's comm from this segment)

        The first partition always has ``comm_op=None`` (nothing to wait for).
        """
        segments = self._split_by_communications(ops)
        partitions: list = []
        prev_comm: Optional[CommunicationOp] = None

        for seg_idx, (comp_ops, comm_after) in enumerate(segments):
            # --- NB0 partition ---
            partitions.append(partition_class(
                partition_id=len(partitions),
                partition_key=f"{direction}_seg{seg_idx}_nb0",
                nano_batch_idx=0,
                comp_ops=comp_ops,
                comm_op=prev_comm,
            ))

            # --- NB1 partition ---
            partitions.append(partition_class(
                partition_id=len(partitions),
                partition_key=f"{direction}_seg{seg_idx}_nb1",
                nano_batch_idx=1,
                comp_ops=comp_ops,
                comm_op=comm_after,
            ))

            prev_comm = comm_after

        return partitions

    # ------------------------------------------------------------------ #
    #  Segment splitting
    # ------------------------------------------------------------------ #

    @staticmethod
    def _split_by_communications(
        ops: List[Union[ComputeOp, CommunicationOp]],
    ) -> List[Segment]:
        """Split a flat op list into segments at communication boundaries.

        Each segment is ``(list_of_compute_ops, trailing_comm_op_or_None)``.

        Example::

            Input:  [A, AR, B, C, AR, D]
            Output: [([A], AR), ([B, C], AR), ([D], None)]
        """
        segments: List[Segment] = []
        current_comp_ops: List[ComputeOp] = []

        for op in ops:
            if isinstance(op, ComputeOp):
                current_comp_ops.append(op)
            elif isinstance(op, CommunicationOp):
                if current_comp_ops:
                    segments.append((current_comp_ops, op))
                    current_comp_ops = []
                else:
                    # Comm with no preceding compute ops — attach to previous
                    # segment (e.g. consecutive comms).
                    if segments:
                        segments[-1] = (segments[-1][0], op)

        # Trailing compute ops with no comm after.
        if current_comp_ops:
            segments.append((current_comp_ops, None))

        return segments
