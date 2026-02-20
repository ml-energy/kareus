"""Bias SwiGLU operation following the BasicOperation pattern."""

import torch
from typing import List, Optional

from transformer_engine.pytorch.ops.op import BasicOperation, OperationContext
from transformer_engine.pytorch.utils import clear_tensor_data
from kareus.megatron.core.partitions.tensor_graph import (
    Channel,
    PartitionableOperator,
)


@torch.compile
def fused_swiglu_forward(
    x: torch.Tensor,
) -> torch.Tensor:
    """Compiled forward function for fused SwiGLU operation (no bias)."""
    x1, x2 = torch.chunk(x, 2, -1)
    return torch.nn.functional.silu(x1) * x2


@torch.compile
def fused_bias_swiglu_forward(
    x: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    """Compiled forward function for fused bias SwiGLU operation (with bias)."""
    x_plus_bias = x + bias
    x1, x2 = torch.chunk(x_plus_bias, 2, -1)
    return torch.nn.functional.silu(x1) * x2


@torch.compile
def fused_swiglu_backward(
    grad_output: torch.Tensor,
    x: torch.Tensor,
) -> torch.Tensor:
    """Compiled backward function for fused SwiGLU operation (no bias)."""
    x1, x2 = torch.chunk(x, 2, -1)
    silu_x1 = torch.nn.functional.silu(x1)
    sigmoid_x1 = torch.sigmoid(x1)
    silu_grad_x1 = sigmoid_x1 * (1 + x1 * (1 - sigmoid_x1))
    grad_x1 = grad_output * x2 * silu_grad_x1
    grad_x2 = grad_output * silu_x1
    grad_input = torch.cat([grad_x1, grad_x2], dim=-1)
    return grad_input


@torch.compile
def fused_bias_swiglu_backward(
    grad_output: torch.Tensor,
    x: torch.Tensor,
    bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compiled backward function for fused bias SwiGLU operation (with bias)."""
    x_plus_bias = x + bias
    x1, x2 = torch.chunk(x_plus_bias, 2, -1)
    silu_x1 = torch.nn.functional.silu(x1)
    sigmoid_x1 = torch.sigmoid(x1)
    silu_grad_x1 = sigmoid_x1 * (1 + x1 * (1 - sigmoid_x1))
    grad_x1 = grad_output * x2 * silu_grad_x1
    grad_x2 = grad_output * silu_x1
    grad_x_plus_bias = torch.cat([grad_x1, grad_x2], dim=-1)
    grad_input = grad_x_plus_bias
    grad_bias = grad_x_plus_bias
    return grad_input, grad_bias


class BiasSwigluOp(BasicOperation, PartitionableOperator):
    """Bias SwiGLU as a BasicOperation

    This operation performs: SwiGLU(input + bias) = SiLU(x1) * x2
    where x1, x2 = split(input + bias, 2, dim=-1)

    Parameters
    ----------
    fp8_input_store : bool, default = False
                     whether to store input in FP8 format for backward pass.
    """

    # BiasSwiglu has 1 extra input: bias
    num_extra_inputs: int = 1

    def get_input_channels(self) -> List[Channel]:
        return [Channel(0, "main"), Channel(1, "bias")]

    def __init__(
        self,
        fp8_input_store: bool = False,
    ) -> None:
        super().__init__()
        self.fp8_input_store = fp8_input_store

    def op_forward(
        self,
        ctx: OperationContext,
        input_: torch.Tensor,  # x
        *,
        prev_op: Optional[BasicOperation] = None,
        next_op: Optional[BasicOperation] = None,
        bias: Optional[torch.Tensor] = None,
        fp8_input_store: Optional[bool] = None,
    ) -> torch.Tensor:
        """Forward pass for bias SwiGLU."""
        
        # Use instance defaults if not provided
        if fp8_input_store is None:
            fp8_input_store = self.fp8_input_store
        
        # Handle input shape
        ori_shape = input_.shape
        assert len(ori_shape) in [2, 3], f"Input must be 2D or 3D, got {len(ori_shape)}D"
        input_reshaped = input_.view(-1, ori_shape[-1])
        
        # Store input for backward pass
        input_for_backward = input_reshaped.to(torch.float8_e4m3fn) if fp8_input_store else input_reshaped
        
        # Enable gradients for proper JIT compilation and mixed precision compatibility
        with torch.enable_grad():
            # Call compiled forward function (bias or no-bias variant)
            if bias is not None:
                output = fused_bias_swiglu_forward(
                    x=input_reshaped,
                    bias=bias,
                )
            else:
                output = fused_swiglu_forward(
                    x=input_reshaped,
                )

        # Reshape output back to original shape (with half the last dimension due to SwiGLU)
        output = output if len(ori_shape) == 2 else output.view(ori_shape[0], ori_shape[1], -1)

        # Save context for backward pass
        ctx.has_bias = bias is not None
        ctx.ori_shape = ori_shape
        ctx.ori_input_dtype = input_.dtype
        ctx.fp8_input_store = fp8_input_store
        # ctx.has_prev_op = prev_op is not None
        if bias is not None:
            ctx.save_for_backward(input_for_backward, bias)
        else:
            ctx.save_for_backward(input_for_backward)
        
        return output

    def op_backward(
        self,
        ctx: OperationContext,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[Optional[torch.Tensor]]]:
        """Backward pass for bias SwiGLU."""
        
        # Retrieve saved context
        has_bias = ctx.has_bias
        ori_input_dtype = ctx.ori_input_dtype
        fp8_input_store = ctx.fp8_input_store
        ori_shape = ctx.ori_shape
        if has_bias:
            input_for_backward, bias = ctx.saved_tensors
        else:
            input_for_backward = ctx.saved_tensors[0]
            bias = None

        # Reshape grad_output to match the flattened computation
        grad_output_reshaped = grad_output.view(-1, grad_output.shape[-1])
        
        # Restore input dtype if it was stored in FP8
        input_ = input_for_backward.to(ori_input_dtype) if fp8_input_store else input_for_backward
        
        # Call compiled backward function (bias or no-bias variant)
        if has_bias:
            grad_input, grad_bias = fused_bias_swiglu_backward(
                grad_output=grad_output_reshaped,
                x=input_,
                bias=bias,
            )
        else:
            grad_input = fused_swiglu_backward(
                grad_output=grad_output_reshaped,
                x=input_,
            )
            grad_bias = None
        
        # Reshape grad_input back to original input shape
        grad_input = grad_input if len(ori_shape) == 2 else grad_input.view(ori_shape)
        
        # # Clear saved tensors if possible
        # if ctx.has_prev_op:
        #     clear_tensor_data(*ctx.saved_tensors)
        
        return grad_input, (grad_bias,)

    def fuser_forward(
        self,
        basic_op_ctxs: list[OperationContext],
        input_: torch.Tensor,
        *,
        basic_op_extra_inputs: list[tuple[torch.Tensor, ...]],
        basic_op_prev_ops: list[Optional[BasicOperation]],
        basic_op_next_ops: list[Optional[BasicOperation]],
        basic_op_kwargs: list[dict[str, any]],
    ) -> tuple[torch.Tensor, list[tuple[()]]]:
        """Override fuser_forward since we have extra inputs."""
        
        # Extract bias from extra inputs
        (bias,) = basic_op_extra_inputs[0]
        
        # Add bias to kwargs
        kwargs = basic_op_kwargs[0].copy()
        kwargs['bias'] = bias
        
        output = self.op_forward(
            basic_op_ctxs[0],
            input_,
            prev_op=basic_op_prev_ops[0],
            next_op=basic_op_next_ops[0],
            **kwargs,
        )
        return output, [()]

    def fuser_backward(
        self,
        basic_op_ctxs: list[OperationContext],
        grad_output: torch.Tensor,
        *,
        basic_op_grad_extra_outputs: list[tuple[torch.Tensor, ...]],
    ) -> tuple[
        torch.Tensor,
        list[tuple[Optional[torch.Tensor], ...]],
        list[tuple[Optional[torch.Tensor]]],
    ]:
        """Override fuser_backward since we have extra inputs."""
        
        grad_input, grad_extra_inputs = self.op_backward(basic_op_ctxs[0], grad_output)
        return grad_input, [()], [grad_extra_inputs]
