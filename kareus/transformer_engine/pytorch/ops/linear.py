# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Fusible operation for linear layer."""

from __future__ import annotations
from collections.abc import Callable
from typing import Optional, Union

import torch

from transformer_engine.pytorch.distributed import CudaRNGStatesTracker
from transformer_engine.pytorch.ops.basic import (
    AllReduce,
    ReduceScatter,
)
from transformer_engine.pytorch.ops.op import FusedOperation

from kareus.transformer_engine.pytorch.ops.basic import (
    BasicLinear,
    Bias,
)


class Linear(FusedOperation):
    """Apply linear transformation: :math:`y = x A^T + b`

    This is a drop-in replacement for `torch.nn.Linear`.

    Parameters
    ----------
    in_features: int
        Inner dimension of input tensor
    out_features: int
        Inner dimension of output tensor
    bias: bool, default = `True`
        Apply additive bias
    device: torch.device, default = default CUDA device
        Tensor device
    dtype: torch.dtype, default = default dtype
        Tensor datatype
    tensor_parallel_mode: {`None`, "column", "row"}, default = `None`
        Mode for tensor parallelism
    tensor_parallel_group: torch.distributed.ProcessGroup, default = world group
        Process group for tensor parallelism
    sequence_parallel: bool, default = `False`
        Whether to apply sequence parallelism together with tensor
        parallelism, i.e. distributing input or output tensors along
        outer dimension (sequence or batch dim) when not distributing
        along inner dimension (embedding dim)
    rng_state_tracker_function: callable
        Function that returns CudaRNGStatesTracker, which is used for
        model-parallel weight initialization
    accumulate_into_main_grad: bool, default = `False`
        Whether to directly accumulate weight gradients into the
        weight's `main_grad` attribute instead of relying on PyTorch
        autograd. The weight's `main_grad` must be set externally and
        there is no guarantee that `grad` will be set or be
        meaningful.

    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bias: bool = True,
        device: Optional[torch.device | str] = None,
        dtype: Optional[torch.dtype] = None,
        tensor_parallel_mode: Optional[str] = None,
        tensor_parallel_group: Optional[torch.distributed.ProcessGroup] = None,
        tensor_parallel_size: Optional[int] = None,
        sequence_parallel: bool = False,
        rng_state_tracker_function: Optional[Callable[[], CudaRNGStatesTracker]] = None,
        accumulate_into_main_grad: bool = False,
    ) -> None:

        # Tensor parallel configuration
        (
            tensor_parallel_mode,
            tensor_parallel_group,
            tensor_parallel_size,
            sequence_parallel,
            local_in_features,
            local_out_features,
        ) = BasicLinear._canonicalize_tensor_parallelism(
            mode=tensor_parallel_mode,
            process_group=tensor_parallel_group,
            tensor_parallel_size=tensor_parallel_size,
            sequence_parallel=sequence_parallel,
            in_features=in_features,
            out_features=out_features,
        )

        # Construct basic ops
        ops = []
        linear_kwargs = {
            "in_features": local_in_features,
            "out_features": local_out_features,
            "device": device,
            "dtype": dtype,
            "tensor_parallel_mode": None,
            "tensor_parallel_group": None,
            "tensor_parallel_size": 1,
            "sequence_parallel": False,
            "rng_state_tracker_function": rng_state_tracker_function,
            "accumulate_into_main_grad": accumulate_into_main_grad,
        }
        bias_kwargs = {
            "size": local_out_features,
            "device": device,
            "dtype": dtype,
            "tensor_parallel": None,
            "tensor_parallel_group": None,
            "tensor_parallel_size": 1,
        }
        ops.append(BasicLinear(**linear_kwargs))
        if bias:
            ops.append(Bias(**bias_kwargs))

        # Initialize base class
        super().__init__(ops)

        self._has_bias: bool = bias

        if sequence_parallel:
            raise NotImplementedError("Sequence parallelism is not supported")
        
        self.forward_comm_op = None
        self.backward_comm_op = None
        if tensor_parallel_mode == "column":
            self.backward_comm_op = AllReduce(process_group=tensor_parallel_group)
        elif tensor_parallel_mode == "row":
            self.forward_comm_op = AllReduce(process_group=tensor_parallel_group)

    @property
    def weight(self) -> torch.nn.Parameter:
        """Weight tensor

        Parameter is owned by `BasicLinear` operation.

        """
        return self.basic_ops[0].weight

    @weight.setter
    def weight(self, value: Optional[torch.nn.Parameter]) -> None:
        self.basic_ops[0].weight = value

    @property
    def bias(self) -> Optional[torch.nn.Parameter]:
        """Bias tensor

        Parameter is owned by `Bias` operation.

        """
        if self._has_bias:
            return self.basic_ops[1].bias
        return None

    @bias.setter
    def bias(self, value: Optional[torch.nn.Parameter]) -> None:
        if self._has_bias:
            self.basic_ops[1].bias = value
        elif value is not None:
            raise ValueError(
                "Attempted to set bias parameter in Linear operation "
                "that does not have bias enabled"
            )

    # def set_tensor_parallel_group(self, tp_group: Union[torch.distributed.ProcessGroup, None]) -> None:
    #     """
    #     Set the tensor parallel group for the given
    #     module before executing the forward pass.

    #     Parameters
    #     ----------
    #     tp_group : ProcessGroup, default = `None`
    #               tensor parallel process group.
    #     """
        
    #     # Update the tensor parallel group in the underlying operations
    #     for op in self.basic_ops:
    #         if hasattr(op, 'tensor_parallel_group'):
    #             op.tensor_parallel_group = tp_group
    #         # For AllReduce and ReduceScatter operations, update the process group
    #         elif hasattr(op, 'process_group'):
    #             op.process_group = tp_group
