# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Fusible operation for all-reduce."""

from __future__ import annotations
from typing import Optional

import torch

from transformer_engine.pytorch.tensor import QuantizedTensor
from transformer_engine.pytorch.ops.op import BasicOperation, OperationContext


class AllReduce(BasicOperation):
    """All-reduce tensor

    Equivalent to summing tensors from all processes. It is assumed
    that the output is used in operations that are redundantly
    computed on all processes, and hence that gradients are identical
    between processes.

    Parameters
    ----------
    process_group: torch.distributed.ProcessGroup, default = world group
        Process group for communication
    async_op: bool, default = False
        Whether to perform asynchronous all-reduce operation

    """

    def __init__(
        self,
        process_group: Optional[torch.distributed.ProcessGroup] = None,
        async_op: bool = True,
    ) -> None:
        super().__init__()
        self.process_group: Optional[torch.distributed.ProcessGroup] = process_group
        self.async_op: bool = async_op
        self._work_handle: Optional[torch.distributed.Work] = None

    def op_forward(
        self,
        ctx: OperationContext,
        input_: torch.Tensor,
        prev_op: Optional[BasicOperation] = None,
        next_op: Optional[BasicOperation] = None,
    ) -> torch.Tensor:

        # Trivial case
        if torch.distributed.get_world_size(self.process_group) == 1:
            return input_

        # Perform all-reduce
        x = input_
        if isinstance(x, QuantizedTensor):
            x = x.dequantize()
        x = x.contiguous()
        
        if self.async_op:
            # Perform asynchronous all-reduce
            self._work_handle = torch.distributed.all_reduce(
                x, group=self.process_group, async_op=True
            )
        else:
            # Perform synchronous all-reduce
            torch.distributed.all_reduce(x, group=self.process_group)
            
        return x

    def sync(self) -> None:
        """Synchronize pending asynchronous all-reduce operation.
        
        This method should be called to wait for completion of asynchronous
        all-reduce operations initiated with async_op=True.
        
        Raises
        ------
        RuntimeError
            If no async operation is pending or if the operation was synchronous
        """
        if self._work_handle is None:
            raise Warning("No AllReduce operation to sync")
            # if not self.async_op:
            #     raise RuntimeError(
            #         "Cannot sync: AllReduce operation was configured as synchronous. "
            #         "Set async_op=True to use asynchronous operations."
            #     )
            # else:
            #     raise RuntimeError(
            #         "Cannot sync: No pending asynchronous all-reduce operation found."
            #     )
        
        # Wait for the async operation to complete
        self._work_handle.wait()
        self._work_handle = None

    def is_async_pending(self) -> bool:
        """Check if there is a pending asynchronous all-reduce operation.
        
        Returns
        -------
        bool
            True if there is a pending async operation, False otherwise
        """
        return self._work_handle is not None

    def op_backward(
        self,
        ctx: OperationContext,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[()]]:
        return grad_output, ()
