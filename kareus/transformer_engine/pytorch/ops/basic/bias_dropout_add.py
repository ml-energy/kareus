"""Bias Dropout Add operation following the BasicOperation pattern."""

import torch
from typing import Optional, Tuple

from transformer_engine.pytorch.ops.op import BasicOperation, OperationContext


@torch.compile
def fused_bias_dropout_add_forward(
    x: torch.Tensor,
    bias: Optional[torch.Tensor],
    residual: torch.Tensor, 
    mask: Optional[torch.Tensor]
) -> torch.Tensor:
    """Compiled forward function for fused bias dropout add operation."""
    # Add bias if provided
    if bias is not None:
        x_plus_bias = x + bias
    else:
        x_plus_bias = x
    
    # Apply dropout mask if provided
    if mask is not None:
        dropout_output = x_plus_bias * mask
    else:
        dropout_output = x_plus_bias
    
    # Add residual connection
    out = residual + dropout_output
    return out


@torch.compile  
def fused_bias_dropout_add_backward(
    grad_output: torch.Tensor,
    mask: Optional[torch.Tensor],
    has_bias: bool,
) -> tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
    """Compiled backward function for fused bias dropout add operation."""
    # Gradient w.r.t. residual is just grad_output
    grad_residual = grad_output
    
    # Gradient w.r.t. dropout output  
    grad_dropout_output = grad_output
    
    # Apply dropout mask to gradient if provided
    if mask is not None:
        grad_x_plus_bias = grad_dropout_output * mask
    else:
        grad_x_plus_bias = grad_dropout_output
    
    # Gradient w.r.t. input
    grad_input = grad_x_plus_bias
    
    # Gradient w.r.t. bias (sum over batch and sequence dimensions if bias exists)
    if has_bias:
        # For bias shape (hidden_size,), sum over batch and sequence dimensions
        grad_bias = grad_x_plus_bias.sum(dim=tuple(range(grad_x_plus_bias.ndim - 1)))
    else:
        grad_bias = None
    
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
            
        # Validate inputs
        assert input_.is_cuda and bias.is_cuda and residual.is_cuda, \
            "BiasDropoutAdd only supports CUDA tensors."
        assert input_.dtype == bias.dtype == residual.dtype, \
            "Input, bias and residual must have the same data type!"
        
        # Generate dropout mask if needed
        mask = None
        if training and dropout_prob > 0.0:
            # Generate dropout mask
            noise = torch.rand_like(input_)
            keep_mask = noise >= dropout_prob
            
            if dropout_prob < 1.0:
                # Scale by 1/(1-p) for unbiased estimation
                mask = keep_mask.to(dtype=input_.dtype) / (1.0 - dropout_prob)
            else:
                mask = torch.zeros_like(input_)
        
        # Call compiled forward function
        output = fused_bias_dropout_add_forward(
            x=input_,
            bias=bias,
            residual=residual,
            mask=mask
        )

        # Save context for backward pass
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
        
        # Call compiled backward function
        grad_input, grad_bias, grad_residual = fused_bias_dropout_add_backward(
            grad_output=grad_output,
            mask=mask,
            has_bias=has_bias,
        )
        
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
