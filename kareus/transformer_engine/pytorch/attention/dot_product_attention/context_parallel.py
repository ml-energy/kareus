# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Context Parallelism."""
import os
from typing import List, Union
import torch
import transformer_engine_torch as tex

from transformer_engine.pytorch.utils import (
    combine_tensors,
    get_cudnn_version,
    nvtx_range_pop,
    nvtx_range_push,
    get_device_compute_capability,
)
from transformer_engine.pytorch.cpp_extensions.fused_attn import (
    fused_attn_fwd,
    fused_attn_bwd,
    FusedAttnBackend,
)
from transformer_engine.pytorch.float8_tensor import Float8Tensor
from transformer_engine.pytorch.jit import jit_fuser
from transformer_engine.pytorch.constants import (
    dist_group_type,
    TE_DType,
)
from transformer_engine.pytorch.distributed import (
    get_distributed_world_size,
    get_distributed_rank,
    gather_along_first_dim,
    reduce_scatter_along_first_dim,
)
from transformer_engine.pytorch.tensor.quantized_tensor import (
    prepare_for_saving,
    restore_from_saved,
)

# Import attention utils
import transformer_engine.pytorch.attention.dot_product_attention.utils as dpa_utils
from transformer_engine.pytorch.attention.dot_product_attention.utils import (
    FlashAttentionUtils as fa_utils,
)

_cu_seqlens_info_with_cp_cache = {}
_seq_chunk_ids_cache_for_reordering_before_attn = {}
_seq_chunk_ids_cache_for_reordering_after_attn = {}


def flash_attn_p2p_communicate(
    rank, send_tensor, send_dst, recv_tensor, recv_src, cp_group, batch_p2p_comm
):
    """Point-to-point communications of KV and dKV in Attention with context parallelism"""
    send_recv_ops = []

    if batch_p2p_comm:
        if rank % 2 == 0:
            send_op = torch.distributed.P2POp(
                torch.distributed.isend, send_tensor, send_dst, cp_group
            )
            recv_op = torch.distributed.P2POp(
                torch.distributed.irecv, recv_tensor, recv_src, cp_group
            )
            send_recv_ops.append(send_op)
            send_recv_ops.append(recv_op)
        else:
            recv_op = torch.distributed.P2POp(
                torch.distributed.irecv, recv_tensor, recv_src, cp_group
            )
            send_op = torch.distributed.P2POp(
                torch.distributed.isend, send_tensor, send_dst, cp_group
            )
            send_recv_ops.append(recv_op)
            send_recv_ops.append(send_op)
        send_recv_reqs = torch.distributed.batch_isend_irecv(send_recv_ops)
    else:
        if rank % 2 == 0:
            send_op = torch.distributed.isend(send_tensor, send_dst, cp_group)
            recv_op = torch.distributed.irecv(recv_tensor, recv_src, cp_group)
            send_recv_ops.append(send_op)
            send_recv_ops.append(recv_op)
        else:
            recv_op = torch.distributed.irecv(recv_tensor, recv_src, cp_group)
            send_op = torch.distributed.isend(send_tensor, send_dst, cp_group)
            send_recv_ops.append(recv_op)
            send_recv_ops.append(send_op)
        send_recv_reqs = send_recv_ops

    return send_recv_reqs


@jit_fuser
def flash_attn_fwd_out_correction_init(
    out_init_step: torch.Tensor,
    softmax_lse: torch.Tensor,
    softmax_lse_init_step: torch.Tensor,
    seq_dim: int,
):
    """Merge partial outputs of the first step in Attention with context parallelism"""
    softmax_lse_corrected_exp = torch.exp(softmax_lse_init_step - softmax_lse).movedim(2, seq_dim)
    softmax_lse_corrected_exp = softmax_lse_corrected_exp.unsqueeze(-1)
    out_corrected = out_init_step * softmax_lse_corrected_exp
    return out_corrected.to(out_init_step.dtype)


@jit_fuser
def flash_attn_fwd_out_correction(
    out: torch.Tensor,
    out_per_step: torch.Tensor,
    softmax_lse: torch.Tensor,
    softmax_lse_per_step: torch.Tensor,
    seq_dim: int,
):
    """Merge partial outputs of each step in Attention with context parallelism"""
    softmax_lse_corrected_exp = torch.exp(softmax_lse_per_step - softmax_lse).movedim(2, seq_dim)
    softmax_lse_corrected_exp = softmax_lse_corrected_exp.unsqueeze(-1)
    out_corrected = out_per_step * softmax_lse_corrected_exp
    out.add_(out_corrected)


@jit_fuser
def flash_attn_fwd_second_half_out_correction(
    out: torch.Tensor,
    out_per_step: torch.Tensor,
    softmax_lse: torch.Tensor,
    softmax_lse_per_step: torch.Tensor,
    seq_dim: int,
):
    """Merge second half of partial outputs of each step in Attention with context parallelism"""
    out_ = out.select(seq_dim, 1)
    softmax_lse_ = softmax_lse.view(*softmax_lse.shape[:-1], 2, -1)[..., 1, :]
    softmax_lse_corrected_exp = torch.exp(softmax_lse_per_step - softmax_lse_).movedim(2, seq_dim)
    softmax_lse_corrected_exp = softmax_lse_corrected_exp.unsqueeze(-1)
    out_corrected = out_per_step * softmax_lse_corrected_exp
    out_.add_(out_corrected)


@jit_fuser
def flash_attn_fwd_softmax_lse_correction(
    softmax_lse: torch.Tensor,
    softmax_lse_per_step: torch.Tensor,
):
    """Merge softmax stats of each step in Attention with context parallelism"""
    max_scale = torch.max(softmax_lse, softmax_lse_per_step)
    min_scale = torch.min(softmax_lse, softmax_lse_per_step)
    new_scale = max_scale + torch.log1p(torch.exp(min_scale - max_scale))
    softmax_lse.copy_(new_scale)


@jit_fuser
def flash_attn_fwd_second_half_softmax_lse_correction(
    softmax_lse: torch.Tensor,
    softmax_lse_per_step: torch.Tensor,
):
    """Merge second half of softmax stats of each step in Attention with context parallelism"""
    softmax_lse_ = softmax_lse[..., 1, :]
    max_scale = torch.max(softmax_lse_, softmax_lse_per_step)
    min_scale = torch.min(softmax_lse_, softmax_lse_per_step)
    new_scale = max_scale + torch.log1p(torch.exp(min_scale - max_scale))
    softmax_lse_.copy_(new_scale)


@jit_fuser
def get_cu_seqlens_on_cp_rank(
    cu_seqlens: torch.Tensor,
    cu_seqlens_padded_on_cp_rank: torch.Tensor,
    cp_size: int,
    cp_rank: int,
    first_half: bool,
    second_half: bool,
):
    """Compute cu_seqlens of a context parallelism rank"""
    seqlens = cu_seqlens[1:] - cu_seqlens[:-1]
    seqlens_padded = (cu_seqlens_padded_on_cp_rank[1:] - cu_seqlens_padded_on_cp_rank[:-1]) // 2
    zeros = torch.zeros_like(seqlens)
    cu_seqlens_on_cp_rank = torch.zeros_like(cu_seqlens)
    if first_half:
        seqlens_1 = seqlens - cp_rank * seqlens_padded
        seqlens_1 = seqlens_1.clamp(zeros, seqlens_padded)
        cu_seqlens_on_cp_rank[1:].add_(seqlens_1)
    if second_half:
        seqlens_2 = seqlens - (2 * cp_size - cp_rank - 1) * seqlens_padded
        seqlens_2 = seqlens_2.clamp(zeros, seqlens_padded)
        cu_seqlens_on_cp_rank[1:].add_(seqlens_2)
    cu_seqlens_on_cp_rank.cumsum_(dim=0)
    return cu_seqlens_on_cp_rank


@jit_fuser
def get_seq_chunk_ids_for_reordering_before_attn(cp_size, device):
    """
    Context parallelism assigns two discontiguous sequence chunks to each GPU for load balancing.
    To make sure tokens are ordered correctly for compute, we need to reorder sequence chunks to
    be contigupus before attention compute. This function is to compute sequence chunk ids for
    reordering.
    """
    global _seq_chunk_ids_cache_for_reordering_before_attn
    if (cp_size, device) not in _seq_chunk_ids_cache_for_reordering_before_attn:
        chunk_ids = torch.empty(2 * cp_size, dtype=torch.int32, device=device)
        for rank in range(cp_size):
            chunk_ids[rank] = 2 * rank
            chunk_ids[rank + cp_size] = 2 * cp_size - 2 * rank - 1
        _seq_chunk_ids_cache_for_reordering_before_attn[(cp_size, device)] = chunk_ids
    return _seq_chunk_ids_cache_for_reordering_before_attn[(cp_size, device)]


@jit_fuser
def get_seq_chunk_ids_for_reordering_after_attn(cp_size, device):
    """
    Context parallelism assigns two discontiguous sequence chunks to each GPU for load balancing.
    We need to reorder sequence chunks back to discontiguous after attention compute. This function
    is to compute sequence chunk ids for reordering.
    """
    global _seq_chunk_ids_cache_for_reordering_after_attn
    if (cp_size, device) not in _seq_chunk_ids_cache_for_reordering_after_attn:
        chunk_ids = torch.empty(2 * cp_size, dtype=torch.int32, device=device)
        for rank in range(cp_size):
            chunk_ids[2 * rank] = rank
            chunk_ids[2 * rank + 1] = 2 * cp_size - rank - 1
        _seq_chunk_ids_cache_for_reordering_after_attn[(cp_size, device)] = chunk_ids
    return _seq_chunk_ids_cache_for_reordering_after_attn[(cp_size, device)]


@jit_fuser
def reorder_seq_chunks_for_a2a_before_attn(x, chunk_ids_for_a2a, seq_dim, cp_size):
    """Reorder sequence chunk for A2A communication before attention compute."""
    # [cp, b, s, np//cp, hn] -> [b, cp, s, np//cp, hn]
    # or [cp, s, b, np//cp, hn] -> [cp, s, b, np//cp, hn]
    x = x.movedim(0, seq_dim).contiguous()
    # [b, cp, s, np//cp, hn] -> [b, cp*2, s//2, np//cp, hn]
    # or [cp, s, b, np//cp, hn] -> [cp*2, s//2, b, np//cp, hn]
    x = x.view(*x.shape[:seq_dim], cp_size * 2, -1, *x.shape[(seq_dim + 2) :])
    # reorder the sequence chunks
    x = torch.index_select(x, dim=seq_dim, index=chunk_ids_for_a2a)
    return x


@jit_fuser
def reorder_seq_chunks_for_a2a_after_attn(x, chunk_ids_for_a2a, seq_dim, cp_size):
    """Reorder sequence chunk for A2A communication after attention compute."""
    # [b, cp*2, s//2, np//cp, hn] -> [cp*2, b, s//2, np//cp, hn]
    # or [cp*2, s//2, b, np//cp, hn] -> [cp*2, s//2, b, np//cp, hn]
    x = x.movedim(seq_dim, 0).contiguous()
    # reorder the sequence chunks
    x = torch.index_select(x, dim=0, index=chunk_ids_for_a2a)
    # [cp*2, b, s//2, np//cp, hn] -> [cp, 2, b, s//2, np//cp, hn]
    # or [cp*2, s//2, b, np//cp, hn] -> [cp, 2, s//2, b, np//cp, hn]
    x = x.view(cp_size, 2, *x.shape[1:])
    return x


def flash_attn_a2a_communicate(
    a2a_inputs: Union[torch.Tensor, List[torch.Tensor]],
    chunk_ids_for_a2a: torch.Tensor,
    seq_dim: int,
    cp_size: int,
    cp_group: dist_group_type,
    cp_stream: torch.cuda.Stream,
    before_attn: bool,
) -> Union[torch.Tensor, List[torch.Tensor]]:
    """A2A communication for context parallelism."""
    a2a_inputs = [a2a_inputs] if not isinstance(a2a_inputs, list) else a2a_inputs
    a2a_outputs, a2a_reqs = [None] * len(a2a_inputs), [None] * len(a2a_inputs)
    if before_attn:
        for i in range(len(a2a_inputs) + 2):
            if 0 < i < len(a2a_inputs) + 1:
                a2a_outputs[i - 1] = torch.empty_like(a2a_inputs[i - 1])
                a2a_reqs[i - 1] = torch.distributed.all_to_all_single(
                    a2a_outputs[i - 1], a2a_inputs[i - 1], group=cp_group, async_op=True
                )
            if i > 1:
                with torch.cuda.stream(cp_stream):
                    a2a_reqs[i - 2].wait()
                    x = a2a_outputs[i - 2]
                    # reorder the sequence chunks
                    x = reorder_seq_chunks_for_a2a_before_attn(
                        x, chunk_ids_for_a2a, seq_dim, cp_size
                    )
                    # [b, cp*2, s//2, np//cp, hn] -> [b, cp*s, np//cp, hn]
                    # or [cp*2, s//2, b, np//cp, hn] -> [cp*s, b, np//cp, hn]
                    a2a_outputs[i - 2] = x.view(*x.shape[:seq_dim], -1, *x.shape[(seq_dim + 2) :])
            if i < len(a2a_inputs):
                x = a2a_inputs[i]
                # [b, s, np, hn] -> [b, s, cp, np//cp, hn]
                # or [s, b, np, hn] -> [s, b, cp, np//cp, hn]
                x = x.view(*x.shape[:-2], cp_size, x.shape[-2] // cp_size, x.shape[-1])
                # [b, s, cp, np//cp, hn] -> [cp, b, s, np//cp, hn]
                # or [s, b, cp, np//cp, hn] -> [cp, s, b, np//cp, hn]
                a2a_inputs[i] = x.movedim(-3, 0).contiguous()
    else:
        for i in range(len(a2a_inputs) + 2):
            if 0 < i < len(a2a_inputs) + 1:
                a2a_outputs[i - 1] = torch.empty_like(a2a_inputs[i - 1])
                a2a_reqs[i - 1] = torch.distributed.all_to_all_single(
                    a2a_outputs[i - 1], a2a_inputs[i - 1], group=cp_group, async_op=True
                )
            if i < len(a2a_inputs):
                x = a2a_inputs[i]
                # [b, cp*s, np//cp, hn] -> [b, cp*2, s//2, np//cp, hn]
                # or [cp*s, b, np//cp, hn] -> [cp*2, s//2, b, np//cp, hn]
                x = x.view(*x.shape[:seq_dim], cp_size * 2, -1, *x.shape[(seq_dim + 1) :])
                # reorder the sequence chunks
                a2a_inputs[i] = reorder_seq_chunks_for_a2a_after_attn(
                    x, chunk_ids_for_a2a, seq_dim, cp_size
                )
            if i > 1:
                with torch.cuda.stream(cp_stream):
                    a2a_reqs[i - 2].wait()
                    x = a2a_outputs[i - 2]
                    # [cp, 2, b, s//2, np//cp, hn] -> [b, 2, s//2, cp, np//cp, hn]
                    # or [cp, 2, s//2, b, np//cp, hn] -> [2, s//2, b, cp, np//cp, hn]
                    x = x.movedim(0, -3).movedim(0, seq_dim).contiguous()
                    # [b, 2, s//2, cp, np//cp, hn] -> [b*s, np, hn]
                    # or [2, s//2, b, cp, np//cp, hn] -> [s*b, np, hn]
                    a2a_outputs[i - 2] = x.view(-1, x.shape[-3] * x.shape[-2], x.shape[-1])
    torch.cuda.current_stream().wait_stream(cp_stream)
    return a2a_outputs[0] if len(a2a_inputs) == 1 else a2a_outputs


def get_fa_args(
    forward: bool,
    use_flash_attn_3: bool,
    qkv_format: str,
    cu_seqlens_q=None,
    cu_seqlens_kv=None,
    max_seqlen_q=None,
    max_seqlen_kv=None,
    dq=None,
    dk=None,
    dv=None,
):
    """Get forward/backward arguments for flash-attn v2 and v3."""
    if use_flash_attn_3:
        if forward:
            if qkv_format == "thd":
                return [
                    *[None] * 4,  # k_new, v_new, qv, out
                    cu_seqlens_q,
                    cu_seqlens_kv,
                    *[None] * 3,  # cu_seqlens_k_new, seqused_q, seqused_k
                    max_seqlen_q,
                    max_seqlen_kv,
                    *[None]
                    * 8,  # page_table, kv_batch_idx, leftpad_k, rotary_cos, rotary_sin, q_descale, k_descale, v_descale
                ]
            return [
                *[None]
                * 9,  # k_new, v_new, qv, out, cu_seqlens_q, cu_seqlens_kv, cu_seqlens_k_new, seqused_q, seqused_k
                max_seqlen_q,
                max_seqlen_kv,
                *[None]
                * 8,  # page_table, kv_batch_idx, leftpad_k, rotary_cos, rotary_sin, q_descale, k_descale, v_descale
            ]
        if qkv_format == "thd":
            return [
                cu_seqlens_q,
                cu_seqlens_kv,
                None,  # sequed_q
                None,  # sequed_k
                max_seqlen_q,
                max_seqlen_kv,
                dq,
                dk,
                dv,
            ]
        return [
            None,  # cu_seqlens_q
            None,  # cu_seqlens_kv
            None,  # sequed_q
            None,  # sequed_k
            max_seqlen_q,
            max_seqlen_kv,
            dq,
            dk,
            dv,
        ]
    if forward:
        if qkv_format == "thd":
            return [
                cu_seqlens_q,
                cu_seqlens_kv,
                max_seqlen_q,
                max_seqlen_kv,
            ]
        return []
    if qkv_format == "thd":
        return [
            dq,
            dk,
            dv,
            cu_seqlens_q,
            cu_seqlens_kv,
            max_seqlen_q,
            max_seqlen_kv,
        ]
    return [
        dq,
        dk,
        dv,
    ]


def get_kv_seq_info_after_all_gather(
    local_chunk_id, cp_size, max_seqlen_q, max_seqlen_kv, window_size, causal
):
    """Compute KV sequence index range and update window size after all-gather."""
    local_chunk_end_idx = (local_chunk_id + 1) * max_seqlen_kv
    full_seq_end_idx = max_seqlen_kv * cp_size * 2

    if window_size is None:
        window_size = (-1, 0) if causal else (-1, -1)

    if window_size[1] == -1:
        seq_end_idx = full_seq_end_idx
        window_size_right = -1
    else:
        seq_end_idx = min(full_seq_end_idx, local_chunk_end_idx + window_size[1])
        window_size_right = local_chunk_end_idx + window_size[1] - seq_end_idx

    if window_size[0] == -1:
        seq_start_idx = 0
        window_size_left = -1
    else:
        seq_start_idx = max(0, local_chunk_end_idx - max_seqlen_q - window_size[0])
        window_size_left = window_size[0] + seq_end_idx - local_chunk_end_idx

    return (seq_start_idx, seq_end_idx), (window_size_left, window_size_right)


class AttnFuncWithCPAndKVAllGather(torch.autograd.Function):
    """
    Attention implementation with context parallelism. KV all-gather between CP ranks is exposed.
    Refer section 3.3.2 of `The Llama 3 Herd of Models <https://arxiv.org/abs/2407.21783>`_.
    """

    @staticmethod
    def forward(
        ctx,
        is_training,
        q,
        k,
        v,
        cu_seqlens_q,
        max_seqlen_q,
        max_seqlen_kv,
        cu_seqlens_q_padded,
        dropout_p,
        softmax_scale,
        qkv_format,
        attn_mask_type,
        attn_bias_type,
        attn_bias,
        deterministic,
        use_fused_attention,
        window_size,
        cp_group,
        cp_stream,
        use_flash_attn_3,
    ):
        # pylint: disable=missing-function-docstring
        nvtx_range_push("transformer_engine.AttnFuncWithCPAndKVAllGather.forward")
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)

        cp_size = get_distributed_world_size(cp_group)
        rank = get_distributed_rank(cp_group)

        qkv_dtype = q.dtype

        causal = "causal" in attn_mask_type
        padding = "padding" in attn_mask_type
        assert not padding, f"{attn_mask_type} mask type is not supported!"
        if use_fused_attention and causal and "bottom_right" not in attn_mask_type:
            attn_mask_type = attn_mask_type + "_bottom_right"
        assert attn_bias_type == "no_bias", f"{attn_bias_type} bias type is not supported!"
        assert q.shape[-1] % 8 == 0, "Hidden size per attention head should be multiple of 8!"
        assert (
            use_fused_attention or fa_utils.v2_3_plus
        ), "Sliding window attention only can work with FusedAttention or FlashAttention >= 2.3!"

        flash_attn_fwd = None
        if not use_fused_attention:
            fa_forward_kwargs = {"softmax_scale": softmax_scale}
            if use_flash_attn_3:
                from transformer_engine.pytorch.attention.dot_product_attention.backends import (
                    _flash_attn_fwd_v3,
                )

                flash_attn_fwd = _flash_attn_fwd_v3
            else:
                if qkv_format == "thd":
                    from transformer_engine.pytorch.attention.dot_product_attention.backends import (
                        _flash_attn_varlen_fwd,
                    )

                    flash_attn_fwd = _flash_attn_varlen_fwd
                else:
                    from transformer_engine.pytorch.attention.dot_product_attention.backends import (
                        _flash_attn_fwd,
                    )

                    flash_attn_fwd = _flash_attn_fwd
                fa_forward_kwargs["dropout_p"] = dropout_p
                fa_forward_kwargs["return_softmax"] = False
                if fa_utils.v2_4_plus:
                    fa_forward_kwargs["alibi_slopes"] = None
                if fa_utils.v2_5_7_plus and qkv_format == "thd":
                    fa_forward_kwargs["block_table"] = None
                if fa_utils.v2_6_0_plus:
                    fa_forward_kwargs["softcap"] = 0.0

        assert qkv_format != "thd", f"{qkv_format} format is not supported!"
        qkv_layout = qkv_format + "_" + qkv_format + "_" + qkv_format

        seq_dim = qkv_format.index("s")
        assert (
            q.shape[seq_dim] % 2 == 0 and k.shape[seq_dim] % 2 == 0
        ), "Sequence length per GPU needs to be divisible by 2!"

        max_seqlen_q = max_seqlen_q // (2 * cp_size)
        max_seqlen_kv = max_seqlen_kv // (2 * cp_size)
        if use_fused_attention or qkv_format == "thd":
            cu_seqlens_q = cu_seqlens_q // (2 * cp_size)
        if cu_seqlens_q_padded is not None and qkv_format == "thd":
            cu_seqlens_q_padded = cu_seqlens_q_padded // (2 * cp_size)
        else:
            cu_seqlens_q_padded = None

        # [b, s, np, hn] -> [b, 2, s//2, np, hn] or [s, b, np, hn] -> [2, s//2, b, np, hn]
        q = q.view(*q.shape[:seq_dim], 2, q.shape[seq_dim] // 2, *q.shape[(seq_dim + 1) :])
        # [b, s, np, hn] or [s, b, np, hn] -> [s, b, np, hn]
        k, v = [x.movedim(seq_dim, 0).contiguous() for x in [k, v]]

        # [s, b, np, hn] -> [cp, s, b, np, hn]
        k_ag, _ = gather_along_first_dim(k, cp_group)
        v_ag, _ = gather_along_first_dim(v, cp_group)

        # [cp, s, b, np, hn] -> [cp*2, s//2, b, np, hn]
        k_ag = k_ag.view(2 * cp_size, k.shape[0] // 2, *k.shape[1:])
        v_ag = v_ag.view(2 * cp_size, v.shape[0] // 2, *v.shape[1:])
        chunk_ids_for_kv_ag = get_seq_chunk_ids_for_reordering_before_attn(cp_size, k.device)
        k_ag = torch.index_select(k_ag, dim=0, index=chunk_ids_for_kv_ag)
        v_ag = torch.index_select(v_ag, dim=0, index=chunk_ids_for_kv_ag)
        # [cp*2, s//2, b, np, hn] -> [cp*s, b, np, hn]
        k_ag = k_ag.view(-1, *k.shape[1:])
        v_ag = v_ag.view(-1, *v.shape[1:])
        cp_stream.wait_stream(torch.cuda.current_stream())

        # create two streams to resolve wave quantization issue of Flash Attn in each step
        flash_attn_streams = [torch.cuda.current_stream(), cp_stream]

        local_seq_chunk_ids = [rank, 2 * cp_size - rank - 1]
        kv_seq_range_per_step = [None, None]
        window_size_per_step = [None, None]
        cu_seqlens_kv_per_step = [None, None]
        out_per_step = [None, None]
        softmax_lse_per_step = [None, None]
        rng_states = [None, None]
        out = torch.empty_like(q)

        for i in range(len(local_seq_chunk_ids) + 1):
            if i < len(local_seq_chunk_ids):
                with torch.cuda.stream(flash_attn_streams[i]):
                    # [b, 2, sq//2, np, hn] -> [b, sq//2, np, hn]
                    # or [2, sq//2, b, np, hn] -> [sq//2, b, np, hn]
                    q_ = q.select(seq_dim, i).contiguous()
                    kv_seq_range_per_step[i], window_size_per_step[i] = (
                        get_kv_seq_info_after_all_gather(
                            local_seq_chunk_ids[i],
                            cp_size,
                            max_seqlen_q,
                            max_seqlen_kv,
                            window_size,
                            causal,
                        )
                    )
                    seq_start_idx, seq_end_idx = (
                        kv_seq_range_per_step[i][0],
                        kv_seq_range_per_step[i][1],
                    )
                    max_seqlen_kv_ = seq_end_idx - seq_start_idx
                    if use_fused_attention or qkv_format == "thd":
                        cu_seqlens_kv_per_step[i] = dpa_utils.get_full_cu_seqlens(
                            k.shape[1], max_seqlen_kv_, k.device
                        )
                    k_, v_ = [x[seq_start_idx:seq_end_idx] for x in [k_ag, v_ag]]
                    # [s_range, b, np, hn] -> [b, s_range, np, hn] or [s_range, b, np, hn]
                    k_, v_ = [x.movedim(0, seq_dim).contiguous() for x in [k_, v_]]
                    if use_fused_attention:
                        out_per_step[i], [softmax_lse_per_step[i], rng_states[i]] = fused_attn_fwd(
                            is_training,
                            max_seqlen_q,
                            max_seqlen_kv_,
                            cu_seqlens_q,
                            cu_seqlens_kv_per_step[i],
                            q_,
                            k_,
                            v_,
                            qkv_dtype,
                            tex.NVTE_Fused_Attn_Backend.NVTE_F16_arbitrary_seqlen,
                            attn_scale=softmax_scale,
                            dropout=dropout_p,
                            qkv_layout=qkv_layout,
                            attn_mask_type=attn_mask_type,
                            attn_bias_type=attn_bias_type,
                            attn_bias=attn_bias,
                            cu_seqlens_q_padded=cu_seqlens_q_padded,
                            cu_seqlens_kv_padded=cu_seqlens_kv_per_step[i],
                            window_size=window_size_per_step[i],
                        )
                    else:
                        fa_forward_args_thd = get_fa_args(
                            True,
                            use_flash_attn_3,
                            qkv_format,
                            cu_seqlens_q=cu_seqlens_q,
                            cu_seqlens_kv=cu_seqlens_kv_per_step[i],
                            max_seqlen_q=max_seqlen_q,
                            max_seqlen_kv=max_seqlen_kv_,
                        )
                        if use_flash_attn_3 or (fa_utils.v2_3_plus and not fa_utils.v2_7_0_plus):
                            fa_forward_kwargs["window_size"] = window_size_per_step[i]
                        elif fa_utils.v2_7_0_plus:
                            fa_forward_kwargs["window_size_left"] = window_size_per_step[i][0]
                            fa_forward_kwargs["window_size_right"] = window_size_per_step[i][1]
                        fa_outputs = flash_attn_fwd(
                            q_,
                            k_,
                            v_,
                            *fa_forward_args_thd,
                            causal=causal,
                            **fa_forward_kwargs,
                        )
                        if not fa_utils.v2_7_0_plus:
                            out_per_step[i] = fa_outputs[4]
                            softmax_lse_per_step[i] = fa_outputs[5]
                            if not use_flash_attn_3:
                                rng_states[i] = fa_outputs[7]
                        else:
                            out_per_step[i] = fa_outputs[0]
                            softmax_lse_per_step[i] = fa_outputs[1]
                            if not use_flash_attn_3:
                                rng_states[i] = fa_outputs[3]

            if i > 0:
                with torch.cuda.stream(flash_attn_streams[i - 1]):
                    if qkv_format == "bshd":
                        out[:, i - 1].copy_(out_per_step[i - 1])
                    elif qkv_format == "sbhd":
                        out[i - 1].copy_(out_per_step[i - 1])

        torch.cuda.current_stream().wait_stream(cp_stream)

        if use_fused_attention:
            if qkv_format == "bshd":
                out = out.view(out.shape[0], -1, *out.shape[-2:])
            elif qkv_format == "sbhd":
                out = out.view(-1, *out.shape[-3:])
        else:
            out = out.view(-1, *out.shape[-2:])

        ctx.save_for_backward(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_q_padded,
            *cu_seqlens_kv_per_step,
            *out_per_step,
            *softmax_lse_per_step,
            *rng_states,
        )

        ctx.qkv_dtype = qkv_dtype
        ctx.kv_seq_range_per_step = kv_seq_range_per_step
        ctx.window_size_per_step = window_size_per_step
        ctx.cp_group = cp_group
        ctx.cp_stream = cp_stream
        ctx.dropout_p = dropout_p
        ctx.max_seqlen_q = max_seqlen_q
        ctx.softmax_scale = softmax_scale
        ctx.qkv_format = qkv_format
        ctx.attn_bias_type = attn_bias_type
        ctx.attn_mask_type = attn_mask_type
        ctx.deterministic = deterministic
        ctx.use_fused_attention = use_fused_attention
        ctx.use_flash_attn_3 = use_flash_attn_3
        nvtx_range_pop("transformer_engine.AttnFuncWithCPAndKVAllGather.forward")
        return out

    @staticmethod
    def backward(ctx, dout):
        # pylint: disable=missing-function-docstring
        nvtx_range_push("transformer_engine.AttnFuncWithCPAndKVAllGather.backward")
        cp_size = get_distributed_world_size(ctx.cp_group)
        rank = get_distributed_rank(ctx.cp_group)

        (*saved_tensors,) = ctx.saved_tensors
        (q, k, v, cu_seqlens_q, cu_seqlens_q_padded) = saved_tensors[:5]
        cu_seqlens_kv_per_step = saved_tensors[5:7]
        out_per_step = saved_tensors[7:9]
        softmax_lse_per_step = saved_tensors[9:11]
        rng_states = saved_tensors[11:13]
        kv_seq_range_per_step = ctx.kv_seq_range_per_step
        window_size_per_step = ctx.window_size_per_step

        seq_dim = ctx.qkv_format.index("s")
        qkv_layout = ctx.qkv_format + "_" + ctx.qkv_format + "_" + ctx.qkv_format

        dout = dout.view(q.shape)
        dq = torch.empty_like(q)
        dk = torch.zeros((k.shape[0] * cp_size, *k.shape[1:]), dtype=k.dtype, device=k.device)
        dv = torch.zeros_like(dk)
        dq_per_step = [None, None]
        dk_per_step = [None, None]
        dv_per_step = [None, None]

        # create two streams to resolve wave quantization issue of Flash Attn in each step
        flash_attn_streams = [torch.cuda.current_stream(), ctx.cp_stream]
        # synchronize dkv update across steps
        dkv_update_done = torch.cuda.Event()

        # [s, b, np, hn] -> [cp, s, b, np, hn]
        k_ag, _ = gather_along_first_dim(k, ctx.cp_group)
        v_ag, _ = gather_along_first_dim(v, ctx.cp_group)

        # [cp, s, b, np, hn] -> [cp*2, s//2, b, np, hn]
        k_ag = k_ag.view(2 * cp_size, k.shape[0] // 2, *k.shape[1:])
        v_ag = v_ag.view(2 * cp_size, v.shape[0] // 2, *v.shape[1:])
        chunk_ids_for_kv_ag = get_seq_chunk_ids_for_reordering_before_attn(cp_size, k.device)
        k_ag = torch.index_select(k_ag, dim=0, index=chunk_ids_for_kv_ag)
        v_ag = torch.index_select(v_ag, dim=0, index=chunk_ids_for_kv_ag)
        # [cp*2, s//2, b, np, hn] -> [cp*s, b, np, hn]
        k_ag = k_ag.view(-1, *k.shape[1:])
        v_ag = v_ag.view(-1, *v.shape[1:])
        ctx.cp_stream.wait_stream(torch.cuda.current_stream())

        local_seq_chunk_ids = [rank, 2 * cp_size - rank - 1]

        flash_attn_bwd = None
        if not ctx.use_fused_attention:
            fa_backward_kwargs = {"softmax_scale": ctx.softmax_scale}
            if ctx.use_flash_attn_3:
                from transformer_engine.pytorch.attention.dot_product_attention.backends import (
                    _flash_attn_bwd_v3,
                )

                flash_attn_bwd = _flash_attn_bwd_v3
                fa_backward_kwargs["deterministic"] = ctx.deterministic
            else:
                if ctx.qkv_format == "thd":
                    from transformer_engine.pytorch.attention.dot_product_attention.backends import (
                        _flash_attn_varlen_bwd,
                    )

                    flash_attn_bwd = _flash_attn_varlen_bwd
                else:
                    from transformer_engine.pytorch.attention.dot_product_attention.backends import (
                        _flash_attn_bwd,
                    )

                    flash_attn_bwd = _flash_attn_bwd
                fa_backward_kwargs["dropout_p"] = ctx.dropout_p
                if fa_utils.v2_4_plus:
                    fa_backward_kwargs["alibi_slopes"] = None
                if fa_utils.v2_4_1_plus:
                    fa_backward_kwargs["deterministic"] = ctx.deterministic
                if fa_utils.v2_6_0_plus:
                    fa_backward_kwargs["softcap"] = 0.0

        for i in range(len(local_seq_chunk_ids) + 1):
            if i < len(local_seq_chunk_ids):
                with torch.cuda.stream(flash_attn_streams[i]):
                    # [b, 2, sq//2, np, hn] -> [b, sq//2, np, hn]
                    # or [2, sq//2, b, np, hn] -> [sq//2, b, np, hn]
                    q_ = q.select(seq_dim, i).contiguous()
                    seq_start_idx, seq_end_idx = (
                        kv_seq_range_per_step[i][0],
                        kv_seq_range_per_step[i][1],
                    )
                    max_seqlen_kv = seq_end_idx - seq_start_idx
                    k_, v_ = [x[seq_start_idx:seq_end_idx] for x in [k_ag, v_ag]]
                    # [cp*s, b, np, hn] -> [b, s_range, np, hn] or [s_range, b, np, hn]
                    k_, v_ = [x.movedim(0, seq_dim).contiguous() for x in [k_, v_]]
                    out_ = out_per_step[i]
                    dout_ = dout.select(seq_dim, i).contiguous().view(out_.shape)
                    if ctx.use_fused_attention:
                        aux_ctx_tensors = [softmax_lse_per_step[i], rng_states[i]]
                        dq_per_step[i], dk_per_step[i], dv_per_step[i], _ = fused_attn_bwd(
                            ctx.max_seqlen_q,
                            max_seqlen_kv,
                            cu_seqlens_q,
                            cu_seqlens_kv_per_step[i],
                            q_,
                            k_,
                            v_,
                            out_,
                            dout_,
                            ctx.qkv_dtype,
                            TE_DType[dout.dtype],
                            aux_ctx_tensors,
                            tex.NVTE_Fused_Attn_Backend.NVTE_F16_arbitrary_seqlen,
                            cu_seqlens_q_padded=cu_seqlens_q_padded,
                            cu_seqlens_kv_padded=cu_seqlens_kv_per_step[i],
                            attn_scale=ctx.softmax_scale,
                            dropout=ctx.dropout_p,
                            qkv_layout=qkv_layout,
                            attn_mask_type=ctx.attn_mask_type,
                            attn_bias_type=ctx.attn_bias_type,
                            window_size=window_size_per_step[i],
                            deterministic=ctx.deterministic,
                        )
                    else:
                        dq_per_step[i], dk_per_step[i], dv_per_step[i] = [
                            torch.empty_like(x) for x in [q_, k_, v_]
                        ]
                        fa_backward_args_thd = get_fa_args(
                            False,
                            ctx.use_flash_attn_3,
                            ctx.qkv_format,
                            cu_seqlens_q=cu_seqlens_q,
                            cu_seqlens_kv=cu_seqlens_kv_per_step[i],
                            max_seqlen_q=ctx.max_seqlen_q,
                            max_seqlen_kv=max_seqlen_kv,
                            dq=dq_per_step[i],
                            dk=dk_per_step[i],
                            dv=dv_per_step[i],
                        )
                        if not ctx.use_flash_attn_3:
                            fa_backward_kwargs["rng_state"] = rng_states[i]
                        if ctx.use_flash_attn_3 or (
                            fa_utils.v2_3_plus and not fa_utils.v2_7_0_plus
                        ):
                            fa_backward_kwargs["window_size"] = window_size_per_step[i]
                        elif fa_utils.v2_7_0_plus:
                            fa_backward_kwargs["window_size_left"] = window_size_per_step[i][0]
                            fa_backward_kwargs["window_size_right"] = window_size_per_step[i][1]
                        flash_attn_bwd(
                            dout_,
                            q_,
                            k_,
                            v_,
                            out_,
                            softmax_lse_per_step[i],
                            *fa_backward_args_thd,
                            causal="causal" in ctx.attn_mask_type,
                            **fa_backward_kwargs,
                        )

            if i > 0:
                with torch.cuda.stream(flash_attn_streams[i - 1]):
                    if ctx.qkv_format == "bshd":
                        dq[:, i - 1].copy_(dq_per_step[i - 1])
                    elif ctx.qkv_format == "sbhd":
                        dq[i - 1].copy_(dq_per_step[i - 1])
                    # [b, s_range, np, hn] or [s_range, b, np, hn] -> [s_range, b, np, hn]
                    dk_per_step[i - 1], dv_per_step[i - 1] = [
                        x.movedim(seq_dim, 0).contiguous()
                        for x in [dk_per_step[i - 1], dv_per_step[i - 1]]
                    ]
                    # wait until dkv update of last step is done
                    if i > 1:
                        flash_attn_streams[i - 1].wait_event(dkv_update_done)
                    seq_start_idx, seq_end_idx = (
                        kv_seq_range_per_step[i - 1][0],
                        kv_seq_range_per_step[i - 1][1],
                    )
                    dk[seq_start_idx:seq_end_idx].add_(dk_per_step[i - 1])
                    dv[seq_start_idx:seq_end_idx].add_(dv_per_step[i - 1])
                    if i < len(local_seq_chunk_ids):
                        flash_attn_streams[i - 1].record_event(dkv_update_done)

        torch.cuda.current_stream().wait_stream(ctx.cp_stream)

        # [cp*s, b, np, hn] -> [cp*2, s//2, b, np, hn]
        dk = dk.view(2 * cp_size, -1, *dk.shape[-3:])
        dv = dv.view(2 * cp_size, -1, *dv.shape[-3:])
        chunk_ids_for_kv_ag = get_seq_chunk_ids_for_reordering_after_attn(cp_size, dk.device)
        dk = torch.index_select(dk, dim=0, index=chunk_ids_for_kv_ag)
        dv = torch.index_select(dv, dim=0, index=chunk_ids_for_kv_ag)
        # [cp*2, s//2, b, np, hn] -> [cp*s, b, np, hn]
        dk = dk.view(-1, *dk.shape[-3:])
        dv = dv.view(-1, *dv.shape[-3:])
        dk, _ = reduce_scatter_along_first_dim(dk, ctx.cp_group)
        dv, _ = reduce_scatter_along_first_dim(dv, ctx.cp_group)

        dq = dq.view(*dq.shape[:seq_dim], -1, *dq.shape[(seq_dim + 2) :])
        dk = dk.movedim(0, seq_dim).contiguous()
        dv = dv.movedim(0, seq_dim).contiguous()
        nvtx_range_pop("transformer_engine.AttnFuncWithCPAndKVAllGather.backward")

        return (
            None,
            dq,
            dk,
            dv,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def attn_forward_func_with_cp(
    is_training,
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_kv,
    max_seqlen_q,
    max_seqlen_kv,
    cu_seqlens_q_padded,
    cu_seqlens_kv_padded,
    dropout_p,
    cp_group,
    cp_global_ranks,
    cp_stream,
    cp_comm_type,
    softmax_scale=None,
    qkv_format="bshd",
    attn_mask_type="causal",
    attn_bias_type="no_bias",
    attn_bias=None,
    deterministic=False,
    use_fused_attention=False,
    window_size=None,
    fp8=False,
    fp8_meta=None,
    quantizers=None,
    pad_between_seqs=False,
    use_flash_attn_3=False,
) -> torch.Tensor:
    """
    Attention implementation with context parallelism (CP). CP partitions tensors along the sequence
    dimension, and by reducing the memory and computational pressure on each GPU, it enables long-context
    LLMs in a distributed fashion. Transformer Engine's PyTorch CP implementation currently utilizes
    the DualChunkSwap strategy to ensure load balancing across CP ranks. It is applied to all `attn_mask_type`s
    and all `qkv_format`s, and it requires sequence lengths to be, or are padded to be, divisible by
    (cp_size * 2). It also requires tokens to be re-ordered before entering this function.

    For qkv_format = {'bshd', 'sbhd'}, the token re-ordering is illustrated as below, for an example
    use case of s = 12, attn_mask_type = 'causal', and cp_size = 2. seq_pos indicates each token's position
    in their corresponding sequence.

                   GPU0        |      GPU1                            GPU0        |      GPU1
    seq_pos | 0  1  2  3  4  5 | 6  7  8  9 10 11      seq_pos | 0  1  2  9 10 11 | 3  4  5  6  7  8
    ---------------------------|-----------------      ---------------------------|------------------
          0 | 1, 0, 0, 0, 0, 0,| 0, 0, 0, 0, 0, 0            0 | 1, 0, 0, 0, 0, 0,| 0, 0, 0, 0, 0, 0,
    G     1 | 1, 1, 0, 0, 0, 0,| 0, 0, 0, 0, 0, 0      G     1 | 1, 1, 0, 0, 0, 0,| 0, 0, 0, 0, 0, 0,
    P     2 | 1, 1, 1, 0, 0, 0,| 0, 0, 0, 0, 0, 0      P     2 | 1, 1, 1, 0, 0, 0,| 0, 0, 0, 0, 0, 0,
    U     3 | 1, 1, 1, 1, 0, 0,| 0, 0, 0, 0, 0, 0      U     9 | 1, 1, 1, 1, 0, 0,| 1, 1, 1, 1, 1, 1,
    0     4 | 1, 1, 1, 1, 1, 0,| 0, 0, 0, 0, 0, 0  ->  0    10 | 1, 1, 1, 1, 1, 0,| 1, 1, 1, 1, 1, 1,
          5 | 1, 1, 1, 1, 1, 1,| 0, 0, 0, 0, 0, 0           11 | 1, 1, 1, 1, 1, 1,| 1, 1, 1, 1, 1, 1,
    ---------------------------|-----------------      ---------------------------|------------------
          6 | 1, 1, 1, 1, 1, 1,| 1, 0, 0, 0, 0, 0            3 | 1, 1, 1, 0, 0, 0,| 1, 0, 0, 0, 0, 0,
    G     7 | 1, 1, 1, 1, 1, 1,| 1, 1, 0, 0, 0, 0      G     4 | 1, 1, 1, 0, 0, 0,| 1, 1, 0, 0, 0, 0,
    P     8 | 1, 1, 1, 1, 1, 1,| 1, 1, 1, 0, 0, 0,     P     5 | 1, 1, 1, 0, 0, 0,| 1, 1, 1, 0, 0, 0,
    U     9 | 1, 1, 1, 1, 1, 1,| 1, 1, 1, 1, 0, 0,     U     6 | 1, 1, 1, 0, 0, 0,| 1, 1, 1, 1, 0, 0,
    1    10 | 1, 1, 1, 1, 1, 1,| 1, 1, 1, 1, 1, 0,     1     7 | 1, 1, 1, 0, 0, 0,| 1, 1, 1, 1, 1, 0,
         11 | 1, 1, 1, 1, 1, 1,| 1, 1, 1, 1, 1, 1,           8 | 1, 1, 1, 0, 0, 0,| 1, 1, 1, 1, 1, 1,

    For qkv_format = 'thd', multiple sequences may be packed into the batch, and they may be of different
    lengths. DualChunkSwap divides each sequence into (cp_size * 2) chunks and distributes 2 chunks of
    every sequence onto a CP rank. The token matrix transformation is shown as follows, for an example of
    batch_size = 2, seq_ids = [0, 1], seq_lens = [8, 4], t = 12, attn_mask_type = 'padding_causal', and
    cp_size = 2.

                   GPU0        |      GPU1                            GPU0        |      GPU1
    seq_id  | 0  0  0  0  0  0 | 0  0  1  1  1  1      seq_id  | 0  0  0  0  1  1 | 0  0  0  0  1  1
    seq_pos | 0  1  2  3  4  5 | 6  7  0  1  2  3      seq_pos | 0  1  6  7  0  3 | 2  3  4  5  1  2
    ---------------------------|-----------------      ---------------------------|------------------
        0 0 | 1, 0, 0, 0, 0, 0,| 0, 0, 0, 0, 0, 0          0 0 | 1, 0, 0, 0, 0, 0,| 0, 0, 0, 0, 0, 0,
    G   0 1 | 1, 1, 0, 0, 0, 0,| 0, 0, 0, 0, 0, 0      G   0 1 | 1, 1, 0, 0, 0, 0,| 0, 0, 0, 0, 0, 0,
    P   0 2 | 1, 1, 1, 0, 0, 0,| 0, 0, 0, 0, 0, 0      P   0 6 | 1, 1, 1, 0, 0, 0,| 1, 1, 1, 1, 0, 0,
    U   0 3 | 1, 1, 1, 1, 0, 0,| 0, 0, 0, 0, 0, 0      U   0 7 | 1, 1, 1, 1, 0, 0,| 1, 1, 1, 1, 0, 0,
    0   0 4 | 1, 1, 1, 1, 1, 0,| 0, 0, 0, 0, 0, 0  ->  0   1 0 | 0, 0, 0, 0, 2, 0,| 0, 0, 0, 0, 0, 0,
        0 5 | 1, 1, 1, 1, 1, 1,| 0, 0, 0, 0, 0, 0          1 3 | 0, 0, 0, 0, 2, 2,| 0, 0, 0, 0, 2, 2,
    ---------------------------|-----------------      ---------------------------|------------------
        0 6 | 1, 1, 1, 1, 1, 1,| 1, 0, 0, 0, 0, 0          0 2 | 1, 1, 0, 0, 0, 0,| 1, 0, 0, 0, 0, 0,
    G   0 7 | 1, 1, 1, 1, 1, 1,| 1, 1, 0, 0, 0, 0      G   0 3 | 1, 1, 0, 0, 0, 0,| 1, 1, 0, 0, 0, 0,
    P   1 0 | 0, 0, 0, 0, 0, 0,| 0, 0, 2, 0, 0, 0      P   0 4 | 1, 1, 0, 0, 0, 0,| 1, 1, 1, 0, 0, 0,
    U   1 1 | 0, 0, 0, 0, 0, 0,| 0, 0, 2, 2, 0, 0      U   0 5 | 1, 1, 0, 0, 0, 0,| 1, 1, 1, 1, 0, 0,
    1   1 2 | 0, 0, 0, 0, 0, 0,| 0, 0, 2, 2, 2, 0      1   1 1 | 0, 0, 0, 0, 2, 0,| 0, 0, 0, 0, 2, 0,
        1 3 | 0, 0, 0, 0, 0, 0,| 0, 0, 2, 2, 2, 2          1 2 | 0, 0, 0, 0, 2, 0,| 0, 0, 0, 0, 2, 2,

    When all transformer layers in a model share the same CP configuration, i.e. cp_group, cp_global_ranks,
    cp_comm_type and cp_stream, token re-ordering can take place in the dataloader, i.e. only once for
    all the layers. An example of the re-ordering code is `get_batch_on_this_cp_rank
    <https://github.com/NVIDIA/Megatron-LM/blob/d6eb60b5ea1efca47401c0be97f456fbe3a55bcd/megatron/core/utils.py#L1725>`_
    in Megatron-LM.

    """
    if cp_comm_type == "a2a+p2p":
        raise NotImplementedError(f"Not supported cp_comm_type: {cp_comm_type}")
        # assert isinstance(
        #     cp_group, list
        # ), "Hierarchical CP implementation needs multi-level CP groups!"
        # assert len(cp_group) == 2, "Current implementation only supports two-level CP groups!"
        # if get_distributed_world_size(cp_group[0]) == 1:
        #     cp_group = cp_group[1]
        #     cp_comm_type = "p2p"
        # elif get_distributed_world_size(cp_group[1]) == 1:
        #     cp_group = cp_group[0]
        #     cp_comm_type = "a2a"
    else:
        assert isinstance(
            cp_group, dist_group_type
        ), f"Unsupported process group for CP communication type {cp_comm_type}!"

    assert qkv_format in [
        "bshd",
        "sbhd",
        "thd",
    ], f"QKV format of {qkv_format} is not supported with context parallelism!"
    assert (
        qkv_format != "sbhd" or use_fused_attention
    ), "FlashAttention does not support sbhd format!"
    assert attn_bias is None or (use_fused_attention and "padding" not in attn_mask_type), (
        """Attention bias is only supported with FusedAttention and "causal" """
        """or "no_mask" mask types!"""
    )
    assert qkv_format != "thd" or (
        cu_seqlens_q_padded is not None and cu_seqlens_kv_padded is not None
    ), "cu_seqlens_padded cannot be None with context parallelism + THD format!"

    sliding_window_attn = (
        window_size is not None and window_size != (-1, 0) and window_size != (-1, -1)
    )
    assert not sliding_window_attn or cp_comm_type in [
        "a2a",
        "all_gather",
    ], "The context parallel running configs cannot support sliding window attetnion!"

    args = [
        is_training,
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_kv,
        max_seqlen_q,
        max_seqlen_kv,
        cu_seqlens_q_padded,
        cu_seqlens_kv_padded,
        dropout_p,
        softmax_scale,
        qkv_format,
        attn_mask_type,
        attn_bias_type,
        attn_bias,
        deterministic,
        use_fused_attention,
    ]

    if cp_comm_type in ["p2p", "a2a+p2p"]:
        raise NotImplementedError(f"Not supported cp_comm_type: {cp_comm_type}")
        # args += [
        #     fp8,
        #     fp8_meta,
        #     cp_group,
        #     cp_global_ranks,
        #     cp_stream,
        #     quantizers,
        #     pad_between_seqs,
        #     use_flash_attn_3,
        # ]
        # out = AttnFuncWithCPAndKVP2P.apply(*args)
    elif cp_comm_type == "all_gather":
        args.pop(5)
        args.pop(8)
        args += [window_size, cp_group, cp_stream, use_flash_attn_3]
        out = AttnFuncWithCPAndKVAllGather.apply(*args)
    elif cp_comm_type == "a2a":
        raise NotImplementedError(f"Not supported cp_comm_type: {cp_comm_type}")
        # args += [window_size, fp8, fp8_meta, cp_group, cp_stream, quantizers, use_flash_attn_3]
        # out = AttnFuncWithCPAndQKVOA2A.apply(*args)
    else:
        raise ValueError(f"Unsupported communication type: {cp_comm_type}!")

    return out
