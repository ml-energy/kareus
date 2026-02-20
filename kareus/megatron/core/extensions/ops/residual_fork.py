"""Residual Fork operation for explicit residual connection handling."""

import torch
from typing import List, Optional

from transformer_engine.pytorch.ops.op import BasicOperation, OperationContext
from kareus.megatron.core.partitions.tensor_graph import (
    Channel,
    PartitionableOperator,
)


class ResidualForkOp(BasicOperation, PartitionableOperator):
    """Residual fork as a BasicOperation.

    Forward:  input x → main output x, extra output x (residual copy)
    Backward: grad_main + grad_residual → grad_input
    """

    # One extra output: the residual copy
    num_extra_outputs: int = 1

    def get_output_channels(self) -> List[Channel]:
        return [Channel(0, "main"), Channel(1, "residual")]

    def __init__(self) -> None:
        super().__init__()

    def op_backward(
        self,
        ctx: OperationContext,
        grad_output: torch.Tensor,
        *,
        grad_residual: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, tuple[()]]:
        """Backward pass: accumulate gradients from both paths.

        grad_output:   gradient from the LayerNorm backward path (main)
        grad_residual: gradient from the BDA backward path (residual)

        Returns the sum as the gradient for the single input.
        """
        if grad_residual is not None:
            grad_input = grad_output + grad_residual
        else:
            grad_input = grad_output
        return grad_input, ()

    def fuser_forward(
        self,
        basic_op_ctxs: list[OperationContext],
        input_: torch.Tensor,
        *,
        basic_op_extra_inputs: list[tuple[torch.Tensor, ...]],
        basic_op_prev_ops: list[Optional[BasicOperation]],
        basic_op_next_ops: list[Optional[BasicOperation]],
        basic_op_kwargs: list[dict],
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor]]]:
        """Fuser forward: return input as main output and as extra output (residual copy).

        Returns:
            (x, [(x,)]) where:
                x    = main output (goes to LayerNorm)
                (x,) = extra output tuple (residual copy for BDA)
        """
        return input_, [(input_,)]

    def fuser_backward(
        self,
        basic_op_ctxs: list[OperationContext],
        grad_output: torch.Tensor,
        *,
        basic_op_grad_extra_outputs: list[tuple[torch.Tensor, ...]],
    ) -> tuple[
        torch.Tensor,
        list[tuple[()]],
        list[tuple[()]],
    ]:
        """Fuser backward: accumulate grad_main and grad_residual.

        Args:
            grad_output: gradient from LayerNorm backward (main path)
            basic_op_grad_extra_outputs: [(grad_residual,)] from BDA backward

        Returns:
            (grad_input, [()], [()]) where:
                grad_input = grad_output + grad_residual
                [()] = no parameter gradients
                [()] = no extra input gradients
        """
        # if basic_op_grad_extra_outputs and len(basic_op_grad_extra_outputs[0]) > 0:
        grad_residual = basic_op_grad_extra_outputs[0][0]
        grad_input = grad_output + grad_residual
        # else:
        #     grad_input = grad_output
        return grad_input, [()], [()]
