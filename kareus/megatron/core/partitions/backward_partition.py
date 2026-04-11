from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List

import torch

from .partition_base import PartitionBase

if TYPE_CHECKING:
    from .context_manager import NanoBatchContext


@dataclass
class BackwardPartition(PartitionBase):
    """Partition for backward pass execution.

    Uses a **separate backward TensorGraph** where:
    1. Ops are in reverse order (last forward op first).
    2. Channel semantics are reversed via ``ComputeOpSpec(is_backward=True)``:
       - input channels  = grad of forward output channels
       - output channels = grad of forward input channels
    3. ``OperationContext`` (with ``saved_tensors``) is retrieved from the
       forward pass via ``ctx.get_op_context(op.op_id)``.

    Tensor routing follows the same ctx / pre_ctx pattern as ``ForwardPartition``:
      - Compute ops read/write grad tensors via ``ctx.tensor_store``
      - Comm op reads/writes grad tensors via ``pre_ctx.tensor_store``

    Compared with the old ``partition_fuser.py`` backward:
      - OLD: manual ``isinstance()`` checks for grad routing
      - NEW: port-based routing — no ``isinstance()`` checks needed
    """

    def execute(
        self,
        ctx: NanoBatchContext,
        pre_ctx: NanoBatchContext,
    ) -> Dict[int, List]:
        """Execute backward partition with automatic tensor routing.

        Args:
            ctx: NanoBatchContext for THIS nanobatch.
                - ``tensor_store``: holds grad tensors flowing through backward graph.
                - ``op_contexts``: saved from forward, contains ``saved_tensors``.
            pre_ctx: NanoBatchContext for the OTHER nanobatch.
                - ``tensor_store``: comm op reads/writes grad tensors here.

        Returns:
            ``grad_params``: mapping from ``op_id`` to list of parameter gradients.
        """
        current_stream = torch.cuda.current_stream()

        # Communication overlap setup
        comm_start, comm_end, sm_num, block_size = self._setup_comm()

        comm_output = None
        comm_extra_outputs: list = []

        # Collect parameter gradients
        grad_params: Dict[int, List] = {}

        # Iterate compute ops
        for op_idx, op in enumerate(self.comp_ops):

            # 1. Retrieve OperationContext saved during forward.
            op_ctx = ctx.get_op_context(op.op_id)

            # Stop if no more gradients are required.
            if not op_ctx.requires_grad:
                break

            # 2. Read grad input tensors via port tensor_ids from THIS nanobatch.
            #    Backward input port 0 = grad of forward output (dx)
            #    Backward input port 1+ = grad of extra forward outputs
            dx = ctx.tensor_store.get(op.input_ports[0].tensor_id)
            grad_extra_outputs: list = []
            if len(op.input_ports) > 1:
                grad_extra_outputs = [tuple(
                    ctx.tensor_store.get(p.tensor_id) for p in op.input_ports[1:]
                )]

            # 3. Launch overlapped backward comm at the scheduled op_idx
            if comm_start == op_idx:
                comm_output, comm_extra_outputs = self._launch_comm(
                    pre_ctx, sm_num, block_size, backward=True,
                )

            # 4. Execute backward compute op
            dx, fused_op_grad_params, fused_op_grad_extra_inputs = (
                op.operator.fuser_backward(
                    [op_ctx],
                    dx,
                    basic_op_grad_extra_outputs=grad_extra_outputs,
                )
            )

            # 5. Store parameter gradients and free saved tensors
            grad_params[op.op_id] = [
                t for param_tuple in fused_op_grad_params for t in param_tuple
            ]
            if not self.profiling_mode:
                op_ctx.saved_tensors = None

            # 6. Record event for next comm window
            if op_idx == comm_start - 1:
                self.comm_op.event_record(current_stream)

            # 7. Write grad output tensors via port tensor_ids to THIS nanobatch.
            #    Backward output port 0 = grad of forward input (dx)
            #    Backward output port 1+ = grad of extra forward inputs
            ctx.tensor_store.set(op.output_ports[0].tensor_id, dx)
            if fused_op_grad_extra_inputs:
                for port_idx, port in enumerate(op.output_ports[1:]):
                    grad_extra = (
                        fused_op_grad_extra_inputs[0][port_idx]
                        if fused_op_grad_extra_inputs[0]
                        else None
                    )
                    if grad_extra is not None:
                        ctx.tensor_store.set(port.tensor_id, grad_extra)

        # Handle non-overlapped backward comm
        if comm_start == -1 and self.comm_op is not None:
            comm_output, comm_extra_outputs = self._launch_comm_no_overlap(
                pre_ctx, sm_num, block_size, backward=True,
            )

        # Sync backward comm and store output to OTHER nanobatch
        if self.comm_op is not None:
            self._sync_comm(pre_ctx, comm_output, comm_extra_outputs)

        return grad_params
