"""
Modified from Megatron-LM (megatron/core/models/common/embeddings/rope_utils.py) by NVIDIA.
Changes: apply_rotary_pos_emb and _apply_rotary_pos_emb_bshd forward/backward
are split into separate functions with an externally managed ctx, bypassing
torch.autograd.Function.apply().
"""

import torch
from torch import Tensor
from megatron.core.transformer.transformer_config import TransformerConfig
from typing import Optional

from kareus.apex.transformer.functional import fused_apply_rotary_pos_emb, fused_apply_rotary_pos_emb_backward


def apply_rotary_pos_emb(
    ctx,
    t: Tensor,
    freqs: Tensor,
    config: TransformerConfig,
    cu_seqlens: Optional[Tensor] = None,
    mscale: float = 1.0,
):
    """
    Forward pass: reroute to the appropriate apply_rotary_pos_emb function depending on
    fused/unfused kernels, or bshd (conventional) / thd (packed seq) format

    Modified: accepts an externally managed ctx for saving tensors needed by
    the backward pass.
    """

    if config.apply_rope_fusion:
        if cu_seqlens is None:
            # NOTE: TE backends do not support mRoPE in bshd format when bs > 1
            if config.mrope_section is not None and freqs.shape[1] > 1:
                return _apply_rotary_pos_emb_bshd(
                    ctx,
                    t,
                    freqs,
                    rotary_interleaved=config.rotary_interleaved,
                    multi_latent_attention=config.multi_latent_attention,
                    mscale=mscale,
                )
            else:
                if config.rotary_interleaved:
                    raise NotImplementedError("Interleaved RoPE is not supported")
                    # try:
                    #     from megatron.core.extensions.transformer_engine import (
                    #         fused_apply_rotary_pos_emb,
                    #     )

                    #     return fused_apply_rotary_pos_emb(t, freqs, interleaved=True)
                    # except ImportError:
                    #     raise ImportError(
                    #         "TE interleaved fused RoPE is not available."
                    #         "Please install TE >= 2.2.0.dev0."
                    #     )
                else:
                    # assert (
                    #     fused_apply_rotary_pos_emb is not None
                    # ), "apply_rope_fusion is not available."
                    return fused_apply_rotary_pos_emb(ctx, t, freqs, transpose_output_memory=True)
        else:
            raise NotImplementedError("cu_seqlens is not supported")
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
    config: TransformerConfig,
    grad_output: Tensor,
) -> Tensor:
    """Backward pass of rotary positional embedding routing.

    Added: split out from the original autograd-based implementation so the
    caller can invoke the backward pass explicitly with the saved ctx.

    Returns:
        (Tensor, None): Gradient w.r.t. input tensor t; None for freqs
            (rotary frequencies are not learnable).
    """
    if config.apply_rope_fusion:
        return fused_apply_rotary_pos_emb_backward(ctx, grad_output)
    else:
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
    """Forward pass: apply rotary positional embedding to input tensor T.

    Modified: accepts an externally managed ctx and saves only cos_ and sin_
    for the backward pass. Gradient w.r.t. freqs is not computed because
    rotary frequencies are fixed positional encodings, not learnable parameters.

    check https://kexue.fm/archives/8265 for detailed formulas

    Args:
        ctx: Externally managed context for saving tensors for backward.
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

    y_re = (t_re * cos_) + (_rotate_half(t_re, rotary_interleaved) * sin_)

    ctx.rot_dim = rot_dim
    ctx.save_for_backward(cos_, sin_)

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


def _apply_rotary_pos_emb_bshd_backward(
    ctx,
    grad_output: Tensor,
) -> Tensor:
    """Added: explicit backward pass of _apply_rotary_pos_emb_bshd.

    Only computes grad_t; grad_freqs is not needed because rotary frequencies
    are fixed positional encodings, not learnable parameters.

    Args:
        ctx: Externally managed context containing saved tensors from forward.
        grad_output (Tensor): Gradient tensor with respect to output, shape [seq_length, ... , dim]

    Returns:
        (Tensor, None): Gradient with respect to input tensor t; None for freqs
    """
    rot_dim = ctx.rot_dim
    cos_, sin_ = ctx.saved_tensors

    grad_re = grad_output[..., :rot_dim]
    grad_pass = grad_output[..., rot_dim:]

    half = rot_dim // 2
    grad_t_re = grad_re * cos_
    s_grad = grad_re * sin_
    sg1, sg2 = s_grad[..., :half], s_grad[..., half:]
    grad_t_re[..., :half] += sg2
    grad_t_re[..., half:] -= sg1

    grad_t = torch.cat((grad_t_re, grad_pass), dim=-1)

    return grad_t, None