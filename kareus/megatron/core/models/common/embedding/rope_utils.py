import torch
from torch import Tensor
from megatron.core.transformer.transformer_config import TransformerConfig
from transformer_engine.pytorch.ops.op import OperationContext
from typing import Optional


def apply_rotary_pos_emb(
    ctx,
    t: Tensor,
    freqs: Tensor,
    config: TransformerConfig,
    cu_seqlens: Optional[Tensor] = None,
    mscale: float = 1.0,
):
    """
    Reroute to the appropriate apply_rotary_pos_emb function depending on
    fused/unfused kernels, or bshd (conventional) / thd (packed seq) format
    """
    global fused_apply_rotary_pos_emb, fused_apply_rotary_pos_emb_thd

    if config.apply_rope_fusion:
        raise NotImplementedError("RoPE fusion is not available.")
        # if cu_seqlens is None:
        #     # NOTE: TE backends do not support mRoPE in bshd format when bs > 1
        #     if config.mrope_section is not None and freqs.shape[1] > 1:
        #         return _apply_rotary_pos_emb_bshd(
        #             t,
        #             freqs,
        #             rotary_interleaved=config.rotary_interleaved,
        #             multi_latent_attention=config.multi_latent_attention,
        #             mscale=mscale,
        #         )
        #     else:
        #         if config.rotary_interleaved:
        #             try:
        #                 from megatron.core.extensions.transformer_engine import (
        #                     fused_apply_rotary_pos_emb,
        #                 )

        #                 return fused_apply_rotary_pos_emb(t, freqs, interleaved=True)
        #             except ImportError:
        #                 raise ImportError(
        #                     "TE interleaved fused RoPE is not available."
        #                     "Please install TE >= 2.2.0.dev0."
        #                 )
        #         else:
        #             assert (
        #                 fused_apply_rotary_pos_emb is not None
        #             ), "apply_rope_fusion is not available."
        #             return fused_apply_rotary_pos_emb(t, freqs, transpose_output_memory=True)
        # else:
        #     assert fused_apply_rotary_pos_emb_thd is not None, "apply_rope_fusion is not available."
        #     cp_size = parallel_state.get_context_parallel_world_size()
        #     if cp_size > 1:
        #         if not is_te_min_version("1.11.0", check_equality=False):
        #             raise ValueError("Only TE >= 1.12 supports RoPE fusion for THD format with CP.")
        #         return fused_apply_rotary_pos_emb_thd(
        #             t,
        #             cu_seqlens,
        #             freqs,
        #             cp_size=cp_size,
        #             cp_rank=parallel_state.get_context_parallel_rank(),
        #         )
        #     else:
        #         return fused_apply_rotary_pos_emb_thd(t, cu_seqlens, freqs)
    else:
        if cu_seqlens is None:
            return _apply_rotary_pos_emb_bshd(
                ctx,
                t,
                freqs,
                rotary_interleaved=config.rotary_interleaved,
                multi_latent_attention=config.multi_latent_attention,
                mscale=mscale,
            )
        else:
            raise NotImplementedError("cu_seqlens is not supported")
            # return _apply_rotary_pos_emb_thd(
            #     t,
            #     cu_seqlens,
            #     freqs,
            #     rotary_interleaved=config.rotary_interleaved,
            #     multi_latent_attention=config.multi_latent_attention,
            #     mscale=mscale,
            # )

def apply_rotary_pos_emb_backward(
    ctx,
    grad_output: Tensor,
) -> Tensor:
    return _apply_rotary_pos_emb_bshd_backward(
        ctx,
        grad_output,
    )


def _apply_rotary_pos_emb_bshd(
    ctx,
    t: Tensor,
    freqs: Tensor,
    rotary_interleaved: bool = False,
    multi_latent_attention: bool = False,
    mscale: float = 1.0,
) -> Tensor:
    """Apply rotary positional embedding to input tensor T.

    check https://kexue.fm/archives/8265 for detailed formulas

    Args:
        t (Tensor): Input tensor T is of shape [seq_length, ... , dim]
        freqs (Tensor): Rotary Positional embedding tensor freq is of shape [seq_length, ..., dim]

    Returns:
        Tensor: The input tensor after applying RoPE
    """
    rot_dim = freqs.shape[-1]

    # ideally t_pass is empty so rotary pos embedding is applied to all tensor t
    t_re, t_pass = t[..., :rot_dim], t[..., rot_dim:]

    if multi_latent_attention:
        raise NotImplementedError("Multi-latent attention is not supported")
        # x1 = t[..., 0::2]
        # x2 = t[..., 1::2]
        # t = torch.cat((x1, x2), dim=-1)

    # first part is cosine component
    # second part is sine component, need to change signs with _rotate_half method
    cos_ = (torch.cos(freqs) * mscale).to(t.dtype)
    sin_ = (torch.sin(freqs) * mscale).to(t.dtype)

    r_t = _rotate_half(t_re, rotary_interleaved)

    y_re = (t_re * cos_) + (r_t * sin_)

    ctx.rot_dim = rot_dim
    ctx.save_for_backward(t_re, cos_, sin_, r_t)

    return torch.cat((y_re, t_pass), dim=-1)


def _rotate_half(x: Tensor, rotary_interleaved: bool) -> Tensor:
    """Change sign so the last dimension becomes [-odd, +even]

    Args:
        x (Tensor): Input tensor

    Returns:
        Tensor: Tensor rotated half
    """
    if not rotary_interleaved:
        x1, x2 = torch.chunk(x, 2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
    else:
        raise NotImplementedError("Interleaved rotary embedding is not supported")
        # x1 = x[:, :, :, ::2]
        # x2 = x[:, :, :, 1::2]
        # x_new = torch.stack((-x2, x1), dim=-1)
        # return x_new.view(x_new.shape[0], x_new.shape[1], x_new.shape[2], -1)


def _rotate_half_backward(grad_output: Tensor, rotary_interleaved: bool = False) -> Tensor:
    half = grad_output.shape[-1] // 2
    sg1, sg2 = grad_output[..., :half], grad_output[..., half:]
    return torch.cat((sg2, -sg1), dim=-1)
    

def _apply_rotary_pos_emb_bshd_backward(
    ctx,
    grad_output: Tensor,
) -> Tensor:
    """Apply backward pass of rotary positional embedding to gradient tensor.

    This function computes the gradient with respect to the input tensor t
    given the gradient with respect to the output of _apply_rotary_pos_emb_bshd.

    The backward pass involves rotating in the opposite direction (negative angle).

    Args:
        grad_output (Tensor): Gradient tensor with respect to output, shape [seq_length, ... , dim]

    Returns:
        Tensor: Gradient with respect to input tensor t
    """
    rot_dim = ctx.rot_dim
    t_re, cos_, sin_, r_t = ctx.saved_tensors

    grad_re = grad_output[..., :rot_dim]
    grad_pass = grad_output[..., rot_dim:]

    half = rot_dim // 2
    grad_t_re = grad_re * cos_
    s_grad = grad_re * sin_
    sg1, sg2 = s_grad[..., :half], s_grad[..., half:]
    grad_t_re[..., :half] += sg2
    grad_t_re[..., half:] -= sg1

    grad_freqs = grad_re * (-t_re * sin_ + r_t * cos_)
    grad_t = torch.cat((grad_t_re, grad_pass), dim=-1)

    return grad_t, grad_freqs


class RotaryPosEmbFunction(torch.autograd.Function):
    """Custom autograd function for rotary positional embedding with proper gradient computation."""
    
    @staticmethod
    def forward(
        ctx,
        t: Tensor,
        freqs: Tensor,
        rotary_interleaved: bool = False,
        multi_latent_attention: bool = False,
        mscale: float = 1.0,
    ) -> Tensor:
        """Forward pass of rotary positional embedding."""
        # Create a context object for the rope function to save tensors
        # rope_ctx = type('Context', (), {})()
        rope_ctx = OperationContext()
        
        # Call the rope function with the rope context
        result = _apply_rotary_pos_emb_bshd(
            rope_ctx, t, freqs, rotary_interleaved, multi_latent_attention, mscale
        )
        
        # Save the rope context and other parameters for backward pass
        ctx.rope_ctx = rope_ctx
        ctx.rotary_interleaved = rotary_interleaved
        ctx.multi_latent_attention = multi_latent_attention
        ctx.mscale = mscale
        
        return result
    
    @staticmethod
    def backward(ctx, grad_output: Tensor) -> tuple:
        """Backward pass of rotary positional embedding."""
        # Use the saved rope context for backward pass
        grad_t, grad_freqs = _apply_rotary_pos_emb_bshd_backward(
            ctx.rope_ctx, grad_output
        )
        
        # Return gradients for all inputs (None for non-tensor inputs)
        return grad_t, grad_freqs, None, None, None


def apply_rotary_pos_emb_bshd_with_grad(
    t: Tensor,
    freqs: Tensor,
    rotary_interleaved: bool = False,
    multi_latent_attention: bool = False,
    mscale: float = 1.0,
) -> Tensor:
    """Apply rotary positional embedding with proper gradient computation.
    
    This is a wrapper around the autograd function that provides the same interface
    as the original _apply_rotary_pos_emb_bshd but with correct gradients.
    
    Args:
        t (Tensor): Input tensor T is of shape [seq_length, ... , dim]
        freqs (Tensor): Rotary Positional embedding tensor freq is of shape [seq_length, ..., dim]
        rotary_interleaved (bool): Whether to use interleaved rotary embedding
        multi_latent_attention (bool): Whether using multi-latent attention
        mscale (float): Scaling factor for the rotation
        
    Returns:
        Tensor: The input tensor after applying RoPE with proper gradients
    """
    return RotaryPosEmbFunction.apply(
        t, freqs, rotary_interleaved, multi_latent_attention, mscale
    )