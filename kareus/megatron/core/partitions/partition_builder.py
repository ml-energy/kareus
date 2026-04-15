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


def _clone_comm(comm: CommunicationOp) -> CommunicationOp:
    """Create an independent copy of a CommunicationOp.

    Shares port objects (immutable tensor IDs) but has an independent
    ``operator`` slot for per-partition physical comm assignment.
    """
    return CommunicationOp(
        comm_type=comm.comm_type,
        input_ports=comm.input_ports,
        output_ports=comm.output_ports,
        operator=comm.operator,
    )

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

    def build_forward_partitions(
        self, partition_keys: List[str],
    ) -> List[ForwardPartition]:
        """Form interleaved forward partitions from the forward graph."""
        return self._form_partitions(
            self.forward_graph.ops,
            ForwardPartition,
            partition_keys=partition_keys,
        )

    def build_backward_partitions(
        self, partition_keys: List[str],
    ) -> List[BackwardPartition]:
        """Form interleaved backward partitions from the backward graph."""
        return self._form_partitions(
            self.backward_graph.ops,
            BackwardPartition,
            partition_keys=partition_keys,
        )

    def _form_partitions(
        self,
        ops: List[Union[ComputeOp, CommunicationOp]],
        partition_class: Type[PartitionBase],
        partition_keys: List[str],
    ) -> list:
        """Form interleaved 2-nanobatch partitions from a flat op list.

        For each segment ``(comp_ops, comm_after)``:
        - NB0 partition: compute ``comp_ops``, comm = ``prev_comm``
          (wait for NB1's comm from the previous segment)
        - NB1 partition: compute ``comp_ops``, comm = ``comm_after``
          (wait for NB0's comm from this segment)

        The first partition always has ``comm_op=None`` (nothing to wait for).

        Args:
            partition_keys: List of semantic keys (one per partition,
                i.e. 2 per segment). Keys are used as-is for ``partition_key``
                on each partition (e.g. ``"fwd_attn"``, ``"bwd_mlp"``).
        """
        segments = self._split_by_communications(ops)

        expected = 2 * len(segments)
        if len(partition_keys) != expected:
            raise RuntimeError(
                f"Partition key count mismatch: "
                f"expected {expected} keys "
                f"(2 x {len(segments)} segments), "
                f"got {len(partition_keys)}."
            )

        partitions: list = []
        prev_comm: Optional[CommunicationOp] = None
        key_idx = 0

        for comp_ops, comm_after in segments:
            nb0_key = partition_keys[key_idx] or ""
            key_idx += 1

            nb1_key = partition_keys[key_idx] or ""
            key_idx += 1

            # NB0 partition
            # Clone prev_comm so this partition has its own CommunicationOp
            # instance with an independent ``operator`` slot.  Without this,
            # NB0 seg N+1 and NB1 seg N would share the same object and
            # _assign_comm_operators would clobber one of them.
            partitions.append(partition_class(
                partition_id=len(partitions),
                partition_key=nb0_key,
                nano_batch_idx=0,
                comp_ops=comp_ops,
                comm_op=_clone_comm(prev_comm) if prev_comm is not None else None,
            ))

            # NB1 partition
            partitions.append(partition_class(
                partition_id=len(partitions),
                partition_key=nb1_key,
                nano_batch_idx=1,
                comp_ops=comp_ops,
                comm_op=_clone_comm(comm_after) if comm_after is not None else None,
            ))

            prev_comm = comm_after

        return partitions

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
