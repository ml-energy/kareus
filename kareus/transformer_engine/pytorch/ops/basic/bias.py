# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Fusible operation for bias."""

from __future__ import annotations
from typing import Optional

import torch

from transformer_engine.pytorch.ops.op import (
    BasicOperation,
    OperationContext,
)
from transformer_engine.pytorch.ops._common import (
    canonicalize_device,
    canonicalize_dtype,
)


class Bias(BasicOperation):
    """Apply additive bias

    This is equivalent to the additive bias in `torch.nn.Linear`.

    Parameters
    ----------
    size: int
        Inner dimension of input tensor
    device: torch.device, default = default CUDA device
        Tensor device
    dtype: torch.dtype, default = default dtype
        Tensor datatype
    tensor_parallel: bool, default = `False`
        Whether to distribute input tensor and bias tensors along
        inner dimension
    tensor_parallel_group: torch.distributed.ProcessGroup, default = world group
        Process group for tensor parallelism

    """
    num_extra_outputs: int = 1

    def __init__(
        self,
        size: int,
        *,
        has_bias: bool = True,
        apply_bias: bool = True,
        return_bias: bool = False,
        device: Optional[torch.device | str] = None,
        dtype: Optional[torch.dtype] = None,
        tensor_parallel: bool = False,
        tensor_parallel_group: Optional[torch.distributed.ProcessGroup] = None,
        tensor_parallel_size: Optional[int] = None,
    ) -> None:
        super().__init__()

        # Bias size
        self._size = size

        self.has_bias: bool = has_bias
        self.apply_bias: bool = apply_bias
        self.return_bias: bool = return_bias

        if has_bias:

            # Bias tensor device
            defer_param_init = False
            device = canonicalize_device(device)
            if device.type == "meta":
                defer_param_init = True
                device = canonicalize_device(None)
            self.device: torch.device = device

            # Tensor parallel configuration
            local_size = size
            if tensor_parallel:
                tensor_parallel = tensor_parallel_size > 1
                if size % tensor_parallel_size != 0:
                    raise ValueError(
                        "Invalid configuration for tensor parallelism "
                        f"({size=}, {tensor_parallel_size=})"
                    )
                local_size //= tensor_parallel_size
            else:
                tensor_parallel_group = None
                tensor_parallel_size = 1

            self.tensor_parallel: bool = tensor_parallel
            self.tensor_parallel_group: Optional[torch.distributed.ProcessGroup] = tensor_parallel_group
            self.tensor_parallel_size: int = tensor_parallel_size
            self.local_size: int = local_size

            # Initialize parameters if needed
            bias = torch.empty(
                local_size,
                device="meta",
                dtype=canonicalize_dtype(dtype),
            )
            bias = torch.nn.Parameter(bias)
            self.bias: torch.nn.Parameter
            self.register_parameter("bias", bias)
            if not defer_param_init:
                self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize parameter buffers and values"""

        # Make sure parameter is initialized
        bias = self.bias
        if bias.device.type != "cuda":
            bias = torch.empty_like(bias, device=self.device)
        else:
            bias = bias.to(device=self.device)

        # Initialize values
        bias.zero_()

        # Save updated parameter
        if not isinstance(bias, torch.nn.Parameter):
            bias = torch.nn.Parameter(bias)
        self.bias = bias

    def pre_forward(self, *args, **kwargs) -> None:
        super().pre_forward(*args, **kwargs)
        if self.has_bias and self.bias.device.type == "meta":
            self.reset_parameters()

    def op_forward(
        self,
        ctx: OperationContext,
        input_: torch.Tensor,
        prev_op: Optional[BasicOperation] = None,
        next_op: Optional[BasicOperation] = None,
    ) -> torch.Tensor:
        if self.apply_bias:
            x = input_
            b = self.bias.reshape([1] * (x.dim() - 1) + [self.local_size])
            out = x + b
        else:
            out = input_

        if self.return_bias and self.has_bias:
            return out, self.bias
        else:
            return out, None

    def op_backward(
        self,
        ctx: OperationContext,
        grad_output: torch.Tensor,
        grad_bias: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, tuple[()]]:
        if self.apply_bias:
            dy = grad_output
            # if dy.dim() > 1:
            #     db = dy.sum(tuple(range(dy.dim() - 1)))
            # else:
            db = dy
        else:
            dy = grad_output
            db = grad_bias

        if self.has_bias:
            return dy, (db,)
        else:
            return dy, ()

    def fuser_forward(
        self,
        basic_op_ctxs: list[OperationContext],
        input_: torch.Tensor,
        *,
        basic_op_extra_inputs: list[tuple[torch.Tensor, ...]],
        basic_op_prev_ops: list[Optional[BasicOperation]],
        basic_op_next_ops: list[Optional[BasicOperation]],
        basic_op_kwargs: list[dict[str, any]],
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor]]]:
        """Fuser forward pass with extra outputs.
        """
        main_out, extra_out = self.op_forward(
            basic_op_ctxs[0],
            input_,
            prev_op=basic_op_prev_ops[0],
            next_op=basic_op_next_ops[0],
            **basic_op_kwargs[0],
        )
        return main_out, [(extra_out,)]

    def fuser_backward(
        self,
        basic_op_ctxs: list[OperationContext],
        grad_output: torch.Tensor,
        *,
        basic_op_grad_extra_outputs: list[tuple[torch.Tensor, ...]],
    ) -> tuple[
        torch.Tensor,
        list[tuple[Optional[torch.Tensor], ...]],
        list[tuple[torch.Tensor]],
    ]:  
        if basic_op_grad_extra_outputs[0]:
            grad_bias, = basic_op_grad_extra_outputs[0]
        else:
            grad_bias = None
        grad_input, grad_params = self.op_backward(
            basic_op_ctxs[0], grad_output, grad_bias
        )
        return grad_input, [grad_params], [()]
