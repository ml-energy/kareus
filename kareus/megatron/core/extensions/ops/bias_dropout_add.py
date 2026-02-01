"""Bias Dropout Add operation following the BasicOperation pattern."""

import torch
from typing import Optional, Tuple
import math

from transformer_engine.pytorch.ops.op import BasicOperation, OperationContext
from transformer_engine.pytorch.utils import clear_tensor_data


@torch.compile
def fused_bias_dropout_add_forward(
    x: torch.Tensor,
    bias: Optional[torch.Tensor],
    residual: torch.Tensor, 
    dropout_prob: float,
    training: bool,
) -> torch.Tensor:
    """Compiled forward function for fused bias dropout add operation."""
    if bias is not None:
        x = x + bias
        dropout_output, mask = torch.ops.aten.native_dropout(x, float(dropout_prob), training)
        out = residual + dropout_output
        return out, mask
    else:
        dropout_output, mask = torch.ops.aten.native_dropout(x, float(dropout_prob), training)
        out = residual + dropout_output
        return out, mask


@torch.compile  
def fused_bias_dropout_add_backward(
    grad_output: torch.Tensor,
    mask: Optional[torch.Tensor],
    scale: torch.Tensor,
) -> tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
    """Compiled backward function for fused bias dropout add operation."""
    grad_residual = grad_output
    grad_input = torch.ops.aten.native_dropout_backward(grad_output, mask, scale)
    grad_bias = grad_input
    return grad_input, grad_bias, grad_residual


class BiasDropoutAddOp(BasicOperation):
    """Bias Dropout Add as a BasicOperation
    
    This operation performs: residual + dropout(input + bias)
    
    Parameters
    ----------
    dropout_prob : float, default = 0.0
                  dropout probability for the dropout operation.
    training : bool, default = True
              whether the model is in training mode.
    """

    # BiasDropoutAdd has 2 extra inputs: bias and residual
    num_extra_inputs: int = 2

    def __init__(
        self,
        dropout_prob: float = 0.0,
        training: bool = True,
    ) -> None:
        super().__init__()
        
        self.dropout_prob = dropout_prob
        self.training = training

    def op_forward(
        self,
        ctx: OperationContext,
        input_: torch.Tensor,  # x
        *,
        prev_op: Optional[BasicOperation] = None,
        next_op: Optional[BasicOperation] = None,
        bias: torch.Tensor,
        residual: torch.Tensor,
        dropout_prob: Optional[float] = None,
        training: Optional[bool] = None,
    ) -> torch.Tensor:
        """Forward pass for bias dropout add."""
        # Use instance defaults if not provided
        if dropout_prob is None:
            dropout_prob = self.dropout_prob
        if training is None:
            training = self.training

        output, mask = fused_bias_dropout_add_forward(
            x=input_,
            bias=bias,
            residual=residual,
            dropout_prob=dropout_prob,
            training=training,
        )
        ctx.scale = 0.0 if math.isclose(1.0 - dropout_prob, 0.0) else 1.0 / (1.0 - dropout_prob)
        ctx.has_bias = bias is not None
        ctx.save_for_backward(mask)
        
        return output

    def op_backward(
        self,
        ctx: OperationContext,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Backward pass for bias dropout add."""
        # Retrieve saved context
        (mask,) = ctx.saved_tensors
        has_bias = ctx.has_bias
        scale = ctx.scale

        grad_input, grad_bias, grad_residual = fused_bias_dropout_add_backward(
            grad_output=grad_output,
            mask=mask,
            scale=scale,
        )
        if not has_bias:
            grad_bias = None
        
        # clear_tensor_data(mask)
        
        return grad_input, (grad_bias, grad_residual)

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
        
        # Extract bias and residual from extra inputs
        bias, residual = basic_op_extra_inputs[0]
        
        # Add bias and residual to kwargs
        kwargs = basic_op_kwargs[0].copy()
        kwargs['bias'] = bias
        kwargs['residual'] = residual
        
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
        list[tuple[torch.Tensor, torch.Tensor]],
    ]:
        """Override fuser_backward since we have extra inputs."""
        
        grad_input, grad_extra_inputs = self.op_backward(basic_op_ctxs[0], grad_output)
        return grad_input, [()], [grad_extra_inputs]
