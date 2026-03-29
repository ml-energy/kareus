"""
Modified from NVIDIA Apex (apex/transformer/functional/fused_rope.py).
Changes: forward/backward are called directly as static methods with an
externally managed ctx, bypassing torch.autograd.Function.apply().
"""

from typing import Tuple, Union
import torch


class FusedRoPEFunc(torch.autograd.Function):
    """
    Fused RoPE function

    This implementation assumes the input tensor to be in `sbhd` format and the RoPE tensor to be
    of shape (s, 1, 1, d). It accepts arbitrary memory layouts to avoid the expensive
    `.contiguous()` calls, thus it may not achieve the best memory access pattern.
    """

    @staticmethod
    def forward(
        ctx,
        t: torch.Tensor,
        freqs: torch.Tensor,
        transpose_output_memory: bool = False,
    ) -> torch.Tensor:
        import fused_rotary_positional_embedding

        output = fused_rotary_positional_embedding.forward(
            t, freqs, transpose_output_memory
        )
        ctx.save_for_backward(freqs)
        ctx.transpose_output_memory = transpose_output_memory

        return output

    @staticmethod
    def backward(
        ctx, grad_output: torch.Tensor
    ) -> Tuple[Union[torch.Tensor, None], ...]:
        import fused_rotary_positional_embedding

        (freqs,) = ctx.saved_tensors
        grad_input = fused_rotary_positional_embedding.backward(
            grad_output, freqs, ctx.transpose_output_memory
        )

        return grad_input, None, None


def fused_apply_rotary_pos_emb(
    ctx,
    t: torch.Tensor,
    freqs: torch.Tensor,
    transpose_output_memory: bool = False,
) -> torch.Tensor:
    """Apply rotary positional embedding to input tensor T in `sbhd` format.

    Modified from original Apex: calls `FusedRoPEFunc.forward` directly instead of
    `.apply()`, bypassing autograd. The caller must supply `ctx` and invoke the
    corresponding backward manually.

    Args:
        ctx: Externally managed context object for saving tensors needed by backward.
        t (Tensor): Input tensor of shape [s, b, h, d].
        freqs (Tensor): Rotary positional embedding tensor of shape [s, 1, 1, d],
            `float` dtype.
        transpose_output_memory (bool): Default to False. Whether to transpose the
            's' and 'b' dimension of the output's underlying memory format.

    Returns:
        Tensor: The input tensor after applying RoPE.
    """
    return FusedRoPEFunc.forward(ctx, t, freqs, transpose_output_memory)


def fused_apply_rotary_pos_emb_backward(
    ctx,
    grad_output: torch.Tensor,
) -> torch.Tensor:
    """Backward pass for fused RoPE, called directly with the same `ctx` populated
    during the forward pass. Returns gradients for t and freqs."""
    grad_out, freqs_grad, _ = FusedRoPEFunc.backward(ctx, grad_output)
    return grad_out, freqs_grad