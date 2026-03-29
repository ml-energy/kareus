from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from .partition_base import PartitionBase

if TYPE_CHECKING:
    from .context_manager import NanoBatchContext


@dataclass
class ForwardPartition(PartitionBase):
    """Partition for forward pass execution.

    Replaces the forward logic from ``_PartitionFuserAutogradFunction.forward``
    with automatic port-based tensor routing through ``TensorStore``.

    Tensor routing (cross-nanobatch via ctx / pre_ctx):
      - Compute ops read/write tensors via ``ctx.tensor_store``
      - Comm op reads/writes tensors via ``pre_ctx.tensor_store``

    Compared with the old ``partition_fuser.py``:
      - OLD: manual ``isinstance()`` checks per op type
      - NEW: port-based routing via ``op.input_ports`` / ``op.output_ports``
    """

    def execute(
        self,
        ctx: NanoBatchContext,
        pre_ctx: NanoBatchContext,
    ) -> None:
        """Execute forward partition with automatic tensor routing.

        Args:
            ctx: NanoBatchContext for THIS nanobatch (compute ops read/write here).
            pre_ctx: NanoBatchContext for the OTHER nanobatch (comm op reads/writes here).
        """
        current_stream = torch.cuda.current_stream()

        # Communication overlap setup
        comm_start, comm_end, sm_num, block_size = self._setup_comm()

        comm_output = None
        comm_extra_outputs: list = []

        # Iterate compute ops
        for fused_idx, op in enumerate(self.comp_ops):

            # 1. Create OperationContext for autograd save/restore.
            #    Keyed by op.op_id so backward can retrieve it.
            op_ctx = ctx.create_op_context(op.op_id)

            # 2. Read input tensors via port tensor_ids from THIS nanobatch.
            #    Port 0 = main tensor (x / hidden_states)
            #    Port 1+ = extra inputs (bias, residual, key, value, rotary, etc.)
            x = ctx.tensor_store.get(op.input_ports[0].tensor_id)
            extra_inputs: list = []
            if len(op.input_ports) > 1:
                extra_inputs = [tuple(
                    ctx.tensor_store.get(p.tensor_id) for p in op.input_ports[1:]
                )]

            # 3. Track requires_grad.
            #    Use ``self.is_grad_enabled`` (captured by the caller
            #    *before* entering torch.autograd.Function.forward(),
            #    where torch.is_grad_enabled() would return False).
            #    This mirrors TransformerEngine's OperationFuser pattern.
            is_grad_enabled = self.is_grad_enabled
            requires_grad = is_grad_enabled and x.requires_grad
            if is_grad_enabled and not requires_grad:
                requires_grad = any(p.requires_grad for p in op.operator.parameters())
            if is_grad_enabled and not requires_grad:
                requires_grad = any(
                    t is not None and t.requires_grad
                    for xs in extra_inputs for t in xs
                )
            op_ctx.requires_grad = requires_grad
            if requires_grad != x.requires_grad:
                x = x.requires_grad_() if requires_grad else x.detach()

            # 4. Launch overlapped comm at the scheduled fused_idx
            if comm_start == fused_idx:
                comm_output, comm_extra_outputs = self._launch_comm(
                    pre_ctx, sm_num, block_size,
                )

            # 5. Execute compute op
            x, fused_op_extra_outputs = op.operator.fuser_forward(
                [op_ctx],
                x,
                basic_op_extra_inputs=extra_inputs,
                basic_op_prev_ops=[None],
                basic_op_next_ops=[None],
                basic_op_kwargs=[{}],
            )

            # 6. Record event for next comm window
            if fused_idx == comm_start - 1:
                self.comm_op.event_record(current_stream)

            # 7. Write output tensors via port tensor_ids to THIS nanobatch.
            #    Port 0 = main output (x)
            #    Port 1+ = extra outputs (key, value, bias, etc.)
            x.requires_grad_(requires_grad=requires_grad)
            ctx.tensor_store.set(op.output_ports[0].tensor_id, x)
            for port_idx, port in enumerate(op.output_ports[1:]):
                extra_out = (
                    fused_op_extra_outputs[0][port_idx]
                    if fused_op_extra_outputs
                    else None
                )
                if extra_out is not None:
                    extra_out.requires_grad_(requires_grad=requires_grad)
                    ctx.tensor_store.set(port.tensor_id, extra_out)

        # Handle non-overlapped comm
        if comm_start == -1 and self.comm_op is not None:
            comm_output, comm_extra_outputs = self._launch_comm_no_overlap(
                pre_ctx, sm_num, block_size,
            )

        # Sync comm and store output to OTHER nanobatch
        if self.comm_op is not None:
            self._sync_comm(pre_ctx, comm_output, comm_extra_outputs)
