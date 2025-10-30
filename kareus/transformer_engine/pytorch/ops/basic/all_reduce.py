# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Fusible operation for all-reduce."""

from __future__ import annotations
from typing import Optional, Dict, Any, Union
import contextlib
import os

import torch

from transformer_engine.pytorch.tensor import QuantizedTensor
from transformer_engine.pytorch.ops.op import BasicOperation, OperationContext
try:
    import cfuser.msccl_comm as msccl_comm
    HAVE_CFUSER = True
except ImportError:
    HAVE_CFUSER = False
import kareus.msccl.msccl_comm as new_msccl_comm

X_AR: list[torch.Tensor | None] = [None, None]

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
    backend: str, default = "nccl"
        Backend to use for communication ("nccl" or "msccl")

    """

    def __init__(
        self,
        process_group: Optional[torch.distributed.ProcessGroup] = None,
        async_op: bool = True,
        backend: str = "nccl",
        rank: Optional[int] = None,
        world_size: Optional[int] = None,
        use_persistent_output: bool = False,
        input_buffer: Optional[torch.Tensor] = None,
        batch_idx: int = 0,
        tensor_size: Optional[list[int]] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        self.process_group: Optional[torch.distributed.ProcessGroup] = process_group
        self.async_op: bool = async_op
        self.backend: str = backend
        self.batch_idx: int = batch_idx
        
        if rank:
            self.rank: Optional[int] = rank
        elif process_group:
            self.rank: Optional[int] = torch.distributed.get_rank(process_group)
        else:
            raise ValueError("rank or process_group must be provided")
        if world_size:
            self.world_size: Optional[int] = world_size
        elif process_group:
            self.world_size: Optional[int] = torch.distributed.get_world_size(process_group)
        else:
            raise ValueError("world_size or process_group must be provided")
        
        self.comm_stream: Optional[torch.cuda.Stream] = None
        self._work_handle: Optional[torch.distributed.Work] = None
        self.wait_event = torch.cuda.Event()

        if self.backend == "msccl":
            if X_AR[self.batch_idx] is None:
                X_AR[self.batch_idx] = torch.randn(
                    *tensor_size, dtype=dtype, device=device,
                )
            self.input_buffer = X_AR[self.batch_idx]
            self.output_buffer = X_AR[self.batch_idx]
            new_msccl_comm.msccl_AllReduce_init(
                self.rank,
                self.world_size,
                self.input_buffer,
                self.process_group,
            )
            self.comm_stream = new_msccl_comm.AR_COMM_STREAM
    
    def set_stream(self, stream: torch.cuda.Stream):
        self.comm_stream = stream
    
    def event_record(self, stream: torch.cuda.Stream):
        if self.comm_stream is not None:
            self.wait_event.record(stream)
    
    def event_wait(self):
        if self.comm_stream is not None:
            self.comm_stream.wait_event(self.wait_event)

    def op_forward(
        self,
        ctx: OperationContext,
        input_: torch.Tensor,
        *,
        prev_op: Optional[BasicOperation] = None,
        next_op: Optional[BasicOperation] = None,
        sm_num: Optional[int] = None,
        block_size: Optional[int] = None,
        backward: bool = False,
    ) -> torch.Tensor:

        # Trivial case
        if torch.distributed.get_world_size(self.process_group) == 1:
            return input_

        # Perform all-reduce
        x = input_
        if isinstance(x, QuantizedTensor):
            x = x.dequantize()
        x = x.contiguous()
        
        if self.backend == "msccl":
            # assert sm_num is not None and block_size is not None, "sm_num and block_size must be provided for msccl backend"
            if sm_num is None or block_size is None:
                # fall back to nccl backend
                self._work_handle = torch.distributed.all_reduce(x, group=self.process_group,
                    async_op=self.async_op,
                )
                # self.backend = "nccl"
                return x
            else:
                assert x.shape == self.input_buffer.shape, "input_buffer shape must match x shape"
                new_msccl_comm.msccl_AllReduce(sm_num, block_size)
                return x
        else:
            assert self.process_group is not None, "process_group must be provided for nccl backend"
            if self.async_op:
                self._work_handle = torch.distributed.all_reduce(
                    x, group=self.process_group, async_op=True
                )
            else:
                torch.distributed.all_reduce(x, group=self.process_group)
            return x

    def sync(self, current_stream: torch.cuda.Stream = None) -> None:
        """Synchronize pending asynchronous all-reduce operation.
        
        This method should be called to wait for completion of asynchronous
        all-reduce operations initiated with async_op=True.
        
        Raises
        ------
        RuntimeError
            If no async operation is pending or if the operation was synchronous
        """
        if self.backend == "msccl":
            # if self.new_backend:
            # new_msccl_comm.msccl_AllReduce_sync()
            # else:
            #     msccl_comm.msccl_sync()
            if self._work_handle is None:
                self.wait_event.record(self.comm_stream)
                current_stream.wait_event(self.wait_event)
            else:
                self._work_handle.wait()
                self._work_handle = None
        else:
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
    
    def set_tensor_parallel_group(self, tp_group: Optional[torch.distributed.ProcessGroup]=None) -> None:
        """
        Set the tensor parallel group for the given
        module before executing the forward pass.

        Parameters
        ----------
        tp_group : ProcessGroup, default = `None`
                  tensor parallel process group.
        """
        self.process_group = tp_group
