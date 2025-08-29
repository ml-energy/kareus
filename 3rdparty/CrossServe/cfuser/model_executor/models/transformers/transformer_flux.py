import inspect
from typing import Any, Dict, Optional, Union

import torch
import torch.distributed
import torch.distributed as dist
from diffusers.models.transformers.transformer_2d import Transformer2DModelOutput
from diffusers.models.transformers.transformer_flux import FluxTransformer2DModel
from diffusers.models.transformers.transformer_flux import FluxSingleTransformerBlock, FluxTransformerBlock
from diffusers.utils import USE_PEFT_BACKEND, scale_lora_layers, unscale_lora_layers

from cfuser.core.distributed import get_sequence_parallel_world_size, get_world_group
from cfuser.core.long_ctx_attention.comm import all_to_all_4D, uneven_decoupled_all_to_all_4D
from cfuser.core.utils import nvtx_range
from cfuser.envs import PACKAGES_CHECKER
from cfuser.logger import init_logger
from cfuser.model_executor.layers.attention_processor import apply_rotary_emb
from cfuser.model_executor.models.transformers.base_transformer import (
    cFuserTransformerBaseWrapper,
)
from cfuser.model_executor.models.transformers.register import (
    cFuserTransformerWrappersRegister,
)
from cfuser.core.distributed.globals import PROCESS_GROUP, ULYSSES_OFF

from contextlib import nullcontext
from typing import Union

env_info = PACKAGES_CHECKER.get_packages_info()
HAS_LONG_CTX_ATTN = env_info["has_long_ctx_attn"]
HAS_FLASH_ATTN = env_info["has_flash_attn"]

logger = init_logger(__name__)

CUSTOM_ALL_TO_ALL_SINGLE_BACKEND = "python"


# NOTE(runyu): this will invoke internal sync of all the tensors in the dict, use for temporary debugging only
def check_nan(dict_of_tensors):
    for name, tensor in dict_of_tensors.items():
        assert torch.all(torch.isfinite(tensor)), f"{name} contains nan"


########################## Shared SP helpers  #############################
# No benefit in using CUDA graph on small comm regions
# @torch.compile(mode="reduce-overhead")
def comm_ulysses_qkv(
    block: Union[FluxSingleTransformerBlock, FluxTransformerBlock],
    query,
    key,
    value,
    async_op=True,
    index_req=0,
    pack_qkv=True,
):

    ranks_send = PROCESS_GROUP.get_non_attn_ranks(index_req)
    ranks_recv = PROCESS_GROUP.get_ulysses_ranks(index_req)
    head_dim = block.attn.processor.hybrid_seq_parallel_attn.scatter_idx
    seq_dim = block.attn.processor.hybrid_seq_parallel_attn.gather_idx

    # TODO(runyu): this is a hack, we should fix it later
    if PROCESS_GROUP.get_ulysses_size(index_req) == 1 and PROCESS_GROUP.get_ring_size(index_req) > 1:
        return query, key, value

    if len(ranks_send) == len(ranks_recv):
        if pack_qkv:
            qkv = torch.cat([query, key, value])
            # (3* bs, seq_len, head_cnt/N, head_size) -> (3* bs, seq_len, head_cnt/N, head_size)
            qkv = all_to_all_4D(
                qkv,
                scatter_idx=head_dim,
                gather_idx=seq_dim,
                group=PROCESS_GROUP.get_ulysses_pg(index_req),
                async_op=async_op,
            )
        else:
            query_layer = all_to_all_4D(
                query,
                scatter_idx=head_dim,
                gather_idx=seq_dim,
                group=PROCESS_GROUP.get_ulysses_pg(index_req),
                async_op=async_op,
            )
            key_layer = all_to_all_4D(
                key,
                scatter_idx=head_dim,
                gather_idx=seq_dim,
                group=PROCESS_GROUP.get_ulysses_pg(index_req),
                async_op=async_op,
            )
            value_layer = all_to_all_4D(
                value,
                scatter_idx=head_dim,
                gather_idx=seq_dim,
                group=PROCESS_GROUP.get_ulysses_pg(index_req),
                async_op=async_op,
            )

    else:
        if pack_qkv:
            qkv = torch.cat([query, key, value])
            qkv = uneven_decoupled_all_to_all_4D(
                qkv,
                ranks_mlp=PROCESS_GROUP.get_non_attn_ranks(index_req),
                ranks_attn=PROCESS_GROUP.get_attn_ranks(index_req),
                ranks_ulysses=PROCESS_GROUP.get_ulysses_ranks(index_req),
                ranks_ring=PROCESS_GROUP.get_ring_ranks(index_req),
                group_mlp=PROCESS_GROUP.get_non_attn_pg(index_req),
                dtype=query.dtype,
                scatter_idx=block.attn.processor.hybrid_seq_parallel_attn.scatter_idx,
                gather_idx=block.attn.processor.hybrid_seq_parallel_attn.gather_idx,
                async_op=async_op,
                custom_backend=CUSTOM_ALL_TO_ALL_SINGLE_BACKEND,
            )
        else:
            query_layer = uneven_decoupled_all_to_all_4D(
                query,
                ranks_mlp=PROCESS_GROUP.get_non_attn_ranks(index_req),
                ranks_attn=PROCESS_GROUP.get_attn_ranks(index_req),
                ranks_ulysses=PROCESS_GROUP.get_ulysses_ranks(index_req),
                ranks_ring=PROCESS_GROUP.get_ring_ranks(index_req),
                group_mlp=PROCESS_GROUP.get_non_attn_pg(index_req),
                dtype=query.dtype,
                scatter_idx=block.attn.processor.hybrid_seq_parallel_attn.scatter_idx,
                gather_idx=block.attn.processor.hybrid_seq_parallel_attn.gather_idx,
                async_op=async_op,
                custom_backend=CUSTOM_ALL_TO_ALL_SINGLE_BACKEND,
            )
            key_layer = uneven_decoupled_all_to_all_4D(
                key,
                ranks_mlp=PROCESS_GROUP.get_non_attn_ranks(index_req),
                ranks_attn=PROCESS_GROUP.get_attn_ranks(index_req),
                ranks_ulysses=PROCESS_GROUP.get_ulysses_ranks(index_req),
                ranks_ring=PROCESS_GROUP.get_ring_ranks(index_req),
                group_mlp=PROCESS_GROUP.get_non_attn_pg(index_req),
                dtype=key.dtype,
                scatter_idx=block.attn.processor.hybrid_seq_parallel_attn.scatter_idx,
                gather_idx=block.attn.processor.hybrid_seq_parallel_attn.gather_idx,
                async_op=async_op,
                custom_backend=CUSTOM_ALL_TO_ALL_SINGLE_BACKEND,
            )
            value_layer = uneven_decoupled_all_to_all_4D(
                value,
                ranks_mlp=PROCESS_GROUP.get_non_attn_ranks(index_req),
                ranks_attn=PROCESS_GROUP.get_attn_ranks(index_req),
                ranks_ulysses=PROCESS_GROUP.get_ulysses_ranks(index_req),
                ranks_ring=PROCESS_GROUP.get_ring_ranks(index_req),
                group_mlp=PROCESS_GROUP.get_non_attn_pg(index_req),
                dtype=value.dtype,
                scatter_idx=block.attn.processor.hybrid_seq_parallel_attn.scatter_idx,
                gather_idx=block.attn.processor.hybrid_seq_parallel_attn.gather_idx,
                async_op=async_op,
                custom_backend=CUSTOM_ALL_TO_ALL_SINGLE_BACKEND,
            )

    if pack_qkv:
        if async_op:
            (query_layer, key_layer, value_layer), handle = qkv[0].chunk(3, dim=2), qkv[1]
            query_layer = (query_layer, handle)
            key_layer = (key_layer, handle)
            value_layer = (value_layer, handle)
        else:
            query_layer, key_layer, value_layer = qkv.chunk(3)

    return query_layer, key_layer, value_layer


def comm_ulysses_mlp(
    block: "FluxSingleTransformerBlock",
    context_layer,
    dtype,
    async_op=True,
    index_req=0,
    pack_qkv=True,
    joint_tensor_query=None,
):
    # (bs, seq_len, head_cnt/N, head_size) -> (bs, seq_len/N, head_cnt, head_size)
    # scatter 1, gather 2
    # output = all_to_all_4D(
    #     context_layer,
    #     block.attn.processor.hybrid_seq_parallel_attn.gather_idx,
    #     block.attn.processor.hybrid_seq_parallel_attn.scatter_idx,
    #     group=block.attn.processor.hybrid_seq_parallel_attn.ulysses_pg,
    #     async_op=async_op,
    # )
    # logger.info(f"rank in world group: {get_world_group().rank_in_group}")
    # logger.info(f"attn ranks: {PROCESS_GROUP.get_attn_ranks(index_req)}")
    # logger.info(f"non_attn ranks: {PROCESS_GROUP.get_non_attn_ranks(index_req)}")
    assert not (pack_qkv and joint_tensor_query is None), "joint_tensor_query must be provided when pack_qkv is True"

    # Do NOT a2a cond
    seq_dim = block.attn.processor.hybrid_seq_parallel_attn.gather_idx
    head_dim = block.attn.processor.hybrid_seq_parallel_attn.scatter_idx
    if pack_qkv:
        seq_len_cond = joint_tensor_query.shape[1]
        cond_layer, context_layer = context_layer.split(
            [seq_len_cond, context_layer.shape[1] - seq_len_cond], dim=seq_dim
        )
        cond_output = allgather_ulysses_cond(cond_layer, async_op=async_op, index_req=index_req, head_dim=head_dim)

    output = uneven_decoupled_all_to_all_4D(
        context_layer,
        ranks_mlp=PROCESS_GROUP.get_non_attn_ranks(index_req),
        ranks_attn=PROCESS_GROUP.get_attn_ranks(index_req),
        ranks_ulysses=PROCESS_GROUP.get_ulysses_ranks(index_req),
        ranks_ring=PROCESS_GROUP.get_ring_ranks(index_req),
        group_mlp=PROCESS_GROUP.get_non_attn_pg(index_req),
        dtype=dtype,
        scatter_idx=block.attn.processor.hybrid_seq_parallel_attn.gather_idx,
        gather_idx=block.attn.processor.hybrid_seq_parallel_attn.scatter_idx,
        async_op=async_op,
        custom_backend=CUSTOM_ALL_TO_ALL_SINGLE_BACKEND,
    )

    if pack_qkv:
        if async_op:
            cond_output_list, cond_handle = cond_output
            output, output_handle = output
            output_handle.append(cond_handle)
            return (output, cond_output_list), output_handle
        else:
            output = torch.cat([cond_output, output], dim=seq_dim)
            return output
    return output


def allgather_ulysses_cond(cond_layer, async_op=True, index_req=0, head_dim=2):
    cond_output_list = [
        torch.empty_like(cond_layer) for _ in range(PROCESS_GROUP.get_ulysses_size(index_req=index_req))
    ]
    # Not sure if 2 * a2a will be more efficient than one ring all-gather using NVSwitch
    handle = dist.all_gather(
        cond_output_list,
        cond_layer.contiguous(),
        group=PROCESS_GROUP.get_ulysses_pg(index_req=index_req),
        async_op=async_op,
    )
    if async_op:
        return cond_output_list, handle
    else:
        cond_output = torch.cat(cond_output_list, dim=head_dim)
        return cond_output


def comm_ulysses_epilogue(
    block,
    outputs,
    handles,
    output_shape,
    async_op,
    pack_qkv,
    joint_strategy="front",
):
    # Or just hardcode 2, 1?
    head_dim = block.attn.processor.hybrid_seq_parallel_attn.scatter_idx
    seq_dim = block.attn.processor.hybrid_seq_parallel_attn.gather_idx
    if async_op:
        if pack_qkv:
            for handle in handles:
                handle.wait()
            output, cond_output_list = outputs
            output = output.reshape(output_shape[0], -1, output_shape[2], output_shape[3]).transpose(0, 2).contiguous()
            cond_output_list = torch.cat(cond_output_list, dim=head_dim)
            if joint_strategy == "front":
                output = torch.cat([cond_output_list, output], dim=seq_dim)
            elif joint_strategy == "rear":
                output = torch.cat([output, cond_output_list], dim=seq_dim)
            else:
                raise NotImplementedError(f"joint_strategy {joint_strategy} not supported")
            return output
        else:
            for handle in handles:  # or handle[0]
                handle.wait()
            output = outputs.reshape(output_shape).transpose(0, 2).contiguous()
            return output
    else:
        return outputs


def comp_ring(
    block: Union[FluxSingleTransformerBlock, FluxTransformerBlock],
    query_layer,
    key_layer,
    value_layer,
    joint_tensor_key,
    joint_tensor_value,
    joint_tensor_query=None,
    dropout_p=0.0,
    causal=False,
    joint_strategy="front",
    index_req=0,
):
    # dist.barrier(group=PROCESS_GROUP.get_ring_pg(index_req))
    # logger.info(
    #     f"rank in group: {get_world_group().rank_in_group}, ring pg: {PROCESS_GROUP.get_ring_pg(index_req)}, ring ranks: {PROCESS_GROUP.get_ring_ranks(index_req)}"
    # )

    out = block.attn.processor.hybrid_seq_parallel_attn.ring_attn_fn(
        query_layer,
        key_layer,
        value_layer,
        dropout_p=dropout_p,
        softmax_scale=None,
        causal=causal,
        window_size=(-1, -1),
        alibi_slopes=None,
        deterministic=False,
        return_attn_probs=False,
        group=PROCESS_GROUP.get_ring_pg(index_req),
        attn_layer=None,
        joint_tensor_key=joint_tensor_key,
        joint_tensor_value=joint_tensor_value,
        joint_tensor_query=joint_tensor_query,
        joint_strategy=joint_strategy,
    )

    if type(out) == tuple:
        context_layer, _, _ = out
    else:
        context_layer = out

    return context_layer


# refer to diffusers/models/attention_processor.py
def flux_blk_attn_forward_prologue(
    block: Union[FluxSingleTransformerBlock, FluxTransformerBlock],
    hidden_states: torch.Tensor,
    encoder_hidden_states: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    **cross_attention_kwargs,
) -> torch.Tensor:
    # The `Attention` class can call different attention processors / attention functions
    # here we simply pass along all tensors to the selected processor class
    # For standard processors that are defined here, `**cross_attention_kwargs` is empty

    attn_parameters = set(inspect.signature(block.attn.processor.__call__).parameters.keys())
    quiet_attn_parameters = {"ip_adapter_masks"}
    unused_kwargs = [
        k for k, _ in cross_attention_kwargs.items() if k not in attn_parameters and k not in quiet_attn_parameters
    ]
    if len(unused_kwargs) > 0:
        logger.warning(
            f"cross_attention_kwargs {unused_kwargs} are not expected by {block.attn.processor.__class__.__name__} and will be ignored."
        )
    cross_attention_kwargs = {k: w for k, w in cross_attention_kwargs.items() if k in attn_parameters}

    return (
        hidden_states,
        encoder_hidden_states,
        attention_mask,
        cross_attention_kwargs,
    )


def flux_blk_attn_processor_sp_forward_prologue(
    block: Union[FluxSingleTransformerBlock, FluxTransformerBlock],
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    joint_tensor_query,
    joint_tensor_key,
    joint_tensor_value,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    joint_strategy="front",
    pack_qkv=True,
    index_req=0,
):
    # cfuser/core/long_ctx_attention/hybrid/attn_layer.py
    # block.attn.processor.hybrid_seq_parallel_attn.renew_process_group()
    # 3 X (bs, seq_len/N, head_cnt, head_size) -> 3 X (bs, seq_len, head_cnt/N, head_size)
    # scatter 2, gather 1

    ulysses_world_size = PROCESS_GROUP.get_ulysses_size(index_req)
    ulysses_rank = PROCESS_GROUP.get_ulysses_rank(index_req)
    if ulysses_rank != ULYSSES_OFF:
        attn_heads_per_ulysses_rank = joint_tensor_key.shape[-2] // ulysses_world_size
        joint_tensor_key = joint_tensor_key[
            ...,
            attn_heads_per_ulysses_rank * ulysses_rank : attn_heads_per_ulysses_rank * (ulysses_rank + 1),
            :,
        ]
        joint_tensor_value = joint_tensor_value[
            ...,
            attn_heads_per_ulysses_rank * ulysses_rank : attn_heads_per_ulysses_rank * (ulysses_rank + 1),
            :,
        ]

    if pack_qkv:
        attn_heads_per_ulysses_rank = joint_tensor_query.shape[-2] // ulysses_world_size
        joint_tensor_query = joint_tensor_query[
            ...,
            attn_heads_per_ulysses_rank * ulysses_rank : attn_heads_per_ulysses_rank * (ulysses_rank + 1),
            :,
        ]
        return query, key, value, joint_tensor_key, joint_tensor_value, joint_tensor_query
    else:
        query = torch.cat([joint_tensor_query, query], dim=1)
        return query, key, value, joint_tensor_key, joint_tensor_value, None


def flux_tf_blk_attn_processor_epilogue(
    block: Union[FluxSingleTransformerBlock, FluxTransformerBlock],
    hidden_states: torch.Tensor,
    head_dim,
    dtype,
    encoder_hidden_states: torch.Tensor = None,
):
    batch_size = hidden_states.shape[0]
    hidden_states = hidden_states.reshape(batch_size, -1, block.attn.heads * head_dim)
    hidden_states = hidden_states.to(dtype)

    if encoder_hidden_states is not None:
        encoder_hidden_states, hidden_states = (
            hidden_states[:, : encoder_hidden_states.shape[1]],
            hidden_states[:, encoder_hidden_states.shape[1] :],
        )

        # linear proj
        hidden_states = block.attn.to_out[0](hidden_states)
        # dropout
        hidden_states = block.attn.to_out[1](hidden_states)
        encoder_hidden_states = block.attn.to_add_out(encoder_hidden_states)

        return hidden_states, encoder_hidden_states
    else:
        return hidden_states


# NOTE: no benefit in compiling too small regions. Dynamo doesn't fuse these linear ops.
def to_qkv(block, hidden_states):
    query = block.attn.to_q(hidden_states)
    key = block.attn.to_k(hidden_states)
    value = block.attn.to_v(hidden_states)
    return query, key, value


def to_qkv_cond(block, hidden_states):

    query = block.attn.add_q_proj(hidden_states)
    key = block.attn.add_k_proj(hidden_states)
    value = block.attn.add_v_proj(hidden_states)
    return query, key, value


############################### Multimodal forward helpers ###############################
# @torch.compile(mode="max-autotune-no-cudagraphs", dynamic=True)
def flux_multimodal_blk_forward_prologue(
    block: FluxTransformerBlock,
    hidden_states: torch.FloatTensor,
    encoder_hidden_states: torch.FloatTensor,
    time_embd: torch.FloatTensor,
    image_rotary_emb=None,
    joint_attention_kwargs=None,
    launch_event=None,
):

    # refer to diffusers/models/transformers/transformer_flux.py

    hidden_states = hidden_states
    encoder_hidden_states = encoder_hidden_states
    time_embd = time_embd
    image_rotary_emb = image_rotary_emb

    norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = block.norm1(hidden_states, emb=time_embd)
    # Hopefully start overlap here
    if launch_event is not None:
        launch_event.record()
    (
        norm_encoder_hidden_states,
        c_gate_msa,
        c_shift_mlp,
        c_scale_mlp,
        c_gate_mlp,
    ) = block.norm1_context(encoder_hidden_states, emb=time_embd)
    joint_attention_kwargs = joint_attention_kwargs or {}

    return (
        norm_hidden_states,
        norm_encoder_hidden_states,
        gate_msa,
        shift_mlp,
        scale_mlp,
        gate_mlp,
        c_gate_msa,
        c_shift_mlp,
        c_scale_mlp,
        c_gate_mlp,
        joint_attention_kwargs,
    )


# refer to cfuser/model_executor/layers/attention_processor.py
# @torch.compile(mode="max-autotune-no-cudagraphs", dynamic=True)
def flux_multimodal_blk_attn_processor_prologue(
    block: FluxTransformerBlock,
    hidden_states: torch.FloatTensor,
    encoder_hidden_states: torch.FloatTensor = None,
    attention_mask: Optional[torch.FloatTensor] = None,
    image_rotary_emb: Optional[torch.Tensor] = None,
    *args,
    **kwargs,
):

    batch_size, _, _ = hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape

    query, key, value = to_qkv(block, hidden_states)
    inner_dim = key.shape[-1]
    head_dim = inner_dim // block.attn.heads

    query = query.view(batch_size, -1, block.attn.heads, head_dim).transpose(1, 2).contiguous()
    key = key.view(batch_size, -1, block.attn.heads, head_dim).transpose(1, 2).contiguous()
    value = value.view(batch_size, -1, block.attn.heads, head_dim).transpose(1, 2).contiguous()

    if block.attn.norm_q is not None:
        query = block.attn.norm_q(query)
    if block.attn.norm_k is not None:
        key = block.attn.norm_k(key)

    if encoder_hidden_states is not None:
        # `context` projections.

        # encoder_hidden_states_query_proj = block.attn.add_q_proj(encoder_hidden_states)
        # encoder_hidden_states_key_proj = block.attn.add_k_proj(encoder_hidden_states)
        # encoder_hidden_states_value_proj = block.attn.add_v_proj(encoder_hidden_states)

        encoder_hidden_states_query_proj, encoder_hidden_states_key_proj, encoder_hidden_states_value_proj = (
            to_qkv_cond(block, encoder_hidden_states)
        )

        encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
            batch_size, -1, block.attn.heads, head_dim
        ).transpose(1, 2)
        encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
            batch_size, -1, block.attn.heads, head_dim
        ).transpose(1, 2)
        encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
            batch_size, -1, block.attn.heads, head_dim
        ).transpose(1, 2)

        if block.attn.norm_added_q is not None:
            encoder_hidden_states_query_proj = block.attn.norm_added_q(encoder_hidden_states_query_proj)
        if block.attn.norm_added_k is not None:
            encoder_hidden_states_key_proj = block.attn.norm_added_k(encoder_hidden_states_key_proj)

        num_encoder_hidden_states_tokens = encoder_hidden_states_query_proj.shape[2]
        num_query_tokens = query.shape[2]

        # attention
        query = torch.cat([encoder_hidden_states_query_proj, query], dim=2)
        key = torch.cat([encoder_hidden_states_key_proj, key], dim=2)
        value = torch.cat([encoder_hidden_states_value_proj, value], dim=2)
    else:
        num_encoder_hidden_states_tokens = 0
        num_query_tokens = query.shape[2]

    if image_rotary_emb is not None:
        query = apply_rotary_emb(query, image_rotary_emb)
        key = apply_rotary_emb(key, image_rotary_emb)

        #! ---------------------------------------- ATTENTION ----------------------------------------

        # assert HAS_LONG_CTX_ATTN and get_sequence_parallel_world_size() > 1
        query = query.transpose(1, 2).contiguous()
        key = key.transpose(1, 2).contiguous()
        value = value.transpose(1, 2).contiguous()
        encoder_hidden_states_query_proj, query = query.split(
            [num_encoder_hidden_states_tokens, num_query_tokens], dim=1
        )
        encoder_hidden_states_key_proj, key = key.split([num_encoder_hidden_states_tokens, num_query_tokens], dim=1)
        encoder_hidden_states_value_proj, value = value.split(
            [num_encoder_hidden_states_tokens, num_query_tokens], dim=1
        )

    return (
        query,
        key,
        value,
        encoder_hidden_states_query_proj,
        encoder_hidden_states_key_proj,
        encoder_hidden_states_value_proj,
        head_dim,
    )


# @torch.compile()
def multimodal_comp_prologue(
    block: FluxTransformerBlock,
    hidden_states: torch.FloatTensor,
    encoder_hidden_states: torch.FloatTensor,
    time_embd: torch.FloatTensor,
    image_rotary_emb=None,
    index_req=0,
    launch_event=None,
    pack_qkv=True,
):
    (
        norm_hidden_states,
        norm_encoder_hidden_states,
        gate_msa,
        shift_mlp,
        scale_mlp,
        gate_mlp,
        c_gate_msa,
        c_shift_mlp,
        c_scale_mlp,
        c_gate_mlp,
        joint_attention_kwargs,
    ) = flux_multimodal_blk_forward_prologue(
        block,
        hidden_states=hidden_states,
        encoder_hidden_states=encoder_hidden_states,
        time_embd=time_embd,
        image_rotary_emb=image_rotary_emb,
        launch_event=launch_event,
    )

    (
        hidden_states,
        encoder_hidden_states,
        attention_mask,
        cross_attention_kwargs,
    ) = flux_blk_attn_forward_prologue(
        block,
        hidden_states=norm_hidden_states,
        encoder_hidden_states=norm_encoder_hidden_states,
        image_rotary_emb=image_rotary_emb,
        **joint_attention_kwargs,
    )

    # comp_prologue_event_first_req_batch.record(
    #     stream_default
    # )  # TODO(@lry89757) the position of this line is important , we may place it in comp_prologue function to enable cross-req comp-comp overlap

    (
        query,
        key,
        value,
        encoder_hidden_states_query_proj,
        encoder_hidden_states_key_proj,
        encoder_hidden_states_value_proj,
        head_dim,
    ) = flux_multimodal_blk_attn_processor_prologue(
        block,
        hidden_states,
        encoder_hidden_states=encoder_hidden_states,
        attention_mask=attention_mask,
        **cross_attention_kwargs,
    )

    query, key, value, joint_tensor_key, joint_tensor_value, joint_tensor_query = (
        flux_blk_attn_processor_sp_forward_prologue(
            block,
            query,
            key,
            value,
            dropout_p=0.0,
            causal=False,
            joint_tensor_query=encoder_hidden_states_query_proj,
            joint_tensor_key=encoder_hidden_states_key_proj,
            joint_tensor_value=encoder_hidden_states_value_proj,
            joint_strategy="front",
            index_req=index_req,
            pack_qkv=pack_qkv,
        )
    )

    return (
        query,
        key,
        value,
        joint_tensor_key,
        joint_tensor_value,
        joint_tensor_query,
        head_dim,
        gate_msa,
        scale_mlp,
        shift_mlp,
        gate_mlp,
        c_gate_mlp,
        c_gate_msa,
        c_scale_mlp,
        c_shift_mlp,
    )


# TODO
# @torch.compile(mode="reduce-overhead", dynamic=True)
def flux_multimodal_tf_blk_epilogue(
    block: FluxTransformerBlock,
    attn_output,
    context_attn_output,
    hidden_states,
    encoder_hidden_states,
    gate_msa,
    scale_mlp,
    shift_mlp,
    gate_mlp,
    c_gate_mlp,
    c_gate_msa,
    c_scale_mlp,
    c_shift_mlp,
):
    # Process attention outputs for the `hidden_states`.
    attn_output = gate_msa.unsqueeze(1) * attn_output
    hidden_states = hidden_states + attn_output

    norm_hidden_states = block.norm2(hidden_states)

    norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
    ff_output = block.ff(norm_hidden_states)
    ff_output = gate_mlp.unsqueeze(1) * ff_output

    hidden_states = hidden_states + ff_output

    # Process attention outputs for the `encoder_hidden_states`.

    context_attn_output = c_gate_msa.unsqueeze(1) * context_attn_output
    encoder_hidden_states = encoder_hidden_states + context_attn_output

    norm_encoder_hidden_states = block.norm2_context(encoder_hidden_states)
    norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]

    context_ff_output = block.ff_context(norm_encoder_hidden_states)
    encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(1) * context_ff_output
    if encoder_hidden_states.dtype == torch.float16:
        encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)

    return encoder_hidden_states, hidden_states


# TODO: why precision error?
# @torch.compile(mode="reduce-overhead")
def multimodal_comp_epilogue(
    block: FluxTransformerBlock,
    sp_attn_output,
    hidden_states,
    encoder_hidden_states,
    head_dim,
    gate_msa,
    scale_mlp,
    shift_mlp,
    gate_mlp,
    c_gate_mlp,
    c_gate_msa,
    c_scale_mlp,
    c_shift_mlp,
    dtype,
):

    # context_attn_output correspond to the attn output of encoder_hidden_states

    attn_output, context_attn_output = flux_tf_blk_attn_processor_epilogue(
        block, sp_attn_output, head_dim, dtype=dtype, encoder_hidden_states=encoder_hidden_states
    )

    # Use context_attn_output to update encoder_hidden_states for the next layer's qkv
    encoder_hidden_states, hidden_states = flux_multimodal_tf_blk_epilogue(
        block,
        attn_output,
        context_attn_output,
        hidden_states,
        encoder_hidden_states,
        gate_msa,
        scale_mlp,
        shift_mlp,
        gate_mlp,
        c_gate_mlp,
        c_gate_msa,
        c_scale_mlp,
        c_shift_mlp,
    )
    # if _stance.stance != "force_eager": # compile (cuda graph) enabled
    #     print("clone")
    # encoder_hidden_states = encoder_hidden_states.clone()
    # hidden_states = hidden_states.clone()

    return encoder_hidden_states, hidden_states


############################ Unimodal Forward Helpers ############################
# refer to cfuser/model_executor/layers/attention_processor.py
# @torch.compile(mode="max-autotune-no-cudagraphs", dynamic=True)
def flux_unimodal_blk_attn_processor_prologue(
    block: FluxSingleTransformerBlock,
    hidden_states: torch.FloatTensor,
    encoder_hidden_states: torch.FloatTensor = None,
    attention_mask: Optional[torch.FloatTensor] = None,
    image_rotary_emb: Optional[torch.Tensor] = None,
    *args,
    **kwargs,
) -> torch.FloatTensor:

    batch_size, _, _ = hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape

    query, key, value = to_qkv(block, hidden_states)
    inner_dim = key.shape[-1]
    head_dim = inner_dim // block.attn.heads

    query = query.view(batch_size, -1, block.attn.heads, head_dim).transpose(1, 2)
    key = key.view(batch_size, -1, block.attn.heads, head_dim).transpose(1, 2)
    value = value.view(batch_size, -1, block.attn.heads, head_dim).transpose(1, 2)

    if block.attn.norm_q is not None:
        query = block.attn.norm_q(query)
    if block.attn.norm_k is not None:
        key = block.attn.norm_k(key)

    num_encoder_hidden_states_tokens = 0
    num_query_tokens = query.shape[2]

    if image_rotary_emb is not None:
        query = apply_rotary_emb(query, image_rotary_emb)
        key = apply_rotary_emb(key, image_rotary_emb)

    #! ---------------------------------------- ATTENTION ----------------------------------------
    # assert HAS_LONG_CTX_ATTN and get_sequence_parallel_world_size() > 1
    # if HAS_LONG_CTX_ATTN and get_sequence_parallel_world_size() > 1:
    if HAS_LONG_CTX_ATTN:
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        encoder_hidden_states_query_proj, query = query.split(
            [num_encoder_hidden_states_tokens, num_query_tokens], dim=1
        )
        encoder_hidden_states_key_proj, key = key.split([num_encoder_hidden_states_tokens, num_query_tokens], dim=1)
        encoder_hidden_states_value_proj, value = value.split(
            [num_encoder_hidden_states_tokens, num_query_tokens], dim=1
        )

    return (
        query,
        key,
        value,
        encoder_hidden_states_query_proj,
        encoder_hidden_states_key_proj,
        encoder_hidden_states_value_proj,
        head_dim,
    )


# refer to diffusers/models/transformers/transformer_flux.py
# CUDA graph tree can handle control flow
# see https://pytorch.org/docs/stable/torch.compiler_cudagraph_trees.html
# @torch.compile(mode="max-autotune-no-cudagraphs", dynamic=True)
def flux_unimodal_blk_forward_prologue(
    block: FluxSingleTransformerBlock,
    hidden_states,
    time_embd,
    image_rotary_emb,
    joint_attention_kwargs,
    launch_event=None,
):

    residual = hidden_states

    # TODO: why not compilable?
    # @torch.compile(dynamic=True)
    norm_hidden_states, gate = norm_hidden_states, gate = block.norm(hidden_states, emb=time_embd)
    if launch_event is not None:
        launch_event.record()
    mlp_hidden_states = block.act_mlp(block.proj_mlp(norm_hidden_states))
    joint_attention_kwargs = joint_attention_kwargs or {}

    return (
        residual,
        gate,
        norm_hidden_states,
        mlp_hidden_states,
        joint_attention_kwargs,
    )


def unimodal_comp_prologue(
    block: FluxSingleTransformerBlock,
    hidden_states,
    time_embd,
    image_rotary_emb,
    joint_attention_kwargs,
    index_req,
    launch_event=None,
    pack_qkv=True,
):

    (
        residual,
        gate,
        norm_hidden_states,
        mlp_hidden_states,
        joint_attention_kwargs,
    ) = flux_unimodal_blk_forward_prologue(
        block, hidden_states, time_embd, image_rotary_emb, joint_attention_kwargs, launch_event=launch_event
    )

    (
        hidden_states,
        encoder_hidden_states,
        attention_mask,
        cross_attention_kwargs,
    ) = flux_blk_attn_forward_prologue(
        block,
        hidden_states=norm_hidden_states,
        image_rotary_emb=image_rotary_emb,
        **joint_attention_kwargs,
    )

    (
        query,
        key,
        value,
        encoder_hidden_states_query_proj,
        encoder_hidden_states_key_proj,
        encoder_hidden_states_value_proj,
        head_dim,
    ) = flux_unimodal_blk_attn_processor_prologue(
        block,
        hidden_states=hidden_states,
        encoder_hidden_states=encoder_hidden_states,
        attention_mask=attention_mask,
        image_rotary_emb=image_rotary_emb,
    )

    if get_sequence_parallel_world_size() > 1:
        query, key, value, joint_tensor_key, joint_tensor_value, joint_tensor_query = (
            flux_blk_attn_processor_sp_forward_prologue(
                block,
                query=query,
                key=key,
                value=value,
                dropout_p=0.0,
                causal=False,
                joint_tensor_query=encoder_hidden_states_query_proj,
                joint_tensor_key=encoder_hidden_states_key_proj,
                joint_tensor_value=encoder_hidden_states_value_proj,
                index_req=index_req,
                pack_qkv=pack_qkv,
            )
        )
    else:
        joint_tensor_key = encoder_hidden_states_key_proj
        joint_tensor_value = encoder_hidden_states_value_proj
        joint_tensor_query = None
    batch_size = hidden_states.shape[0]
    return (
        query,
        key,
        value,
        joint_tensor_key,
        joint_tensor_value,
        joint_tensor_query,
        head_dim,
        batch_size,
        residual,
        gate,
        mlp_hidden_states,
    )


# @torch.compile
def flux_unimodal_blk_epilogue(block: FluxSingleTransformerBlock, residual, gate, mlp_hidden_states, attn_output):
    hidden_states = torch.cat([attn_output, mlp_hidden_states], dim=2)
    gate = gate.unsqueeze(1)
    # TODO: Why precision error with only a linear layer??
    # @torch.compile()
    hidden_states = gate * block.proj_out(hidden_states)
    hidden_states = residual + hidden_states
    if hidden_states.dtype == torch.float16:
        hidden_states = hidden_states.clip(-65504, 65504)

    return hidden_states


def unimodal_comp_epilogue(
    block: FluxSingleTransformerBlock, attn_output, residual, gate, mlp_hidden_states, head_dim, dtype
):

    attn_output = flux_tf_blk_attn_processor_epilogue(
        block,
        hidden_states=attn_output,
        head_dim=head_dim,
        dtype=dtype,
    )

    hidden_states = flux_unimodal_blk_epilogue(block, residual, gate, mlp_hidden_states, attn_output)
    return hidden_states


############################################################################################################


@cFuserTransformerWrappersRegister.register(FluxTransformer2DModel)
class cFuserFluxTransformer2DWrapper(cFuserTransformerBaseWrapper):
    def __init__(
        self,
        transformer: FluxTransformer2DModel,
    ):
        super().__init__(
            transformer=transformer,
            submodule_classes_to_wrap=[],
            submodule_name_to_wrap=["attn"],
        )
        self.encoder_hidden_states_cache = [None for _ in range(len(self.transformer_blocks))]
        self.stream_default = torch.cuda.default_stream()
        self.stream_1 = torch.cuda.Stream()

    # NOTE(runyu): like 'inline', we need to implement the cross-req forward, so we inline all the forward code logic here
    def inline_multimodal_transformer_blocks_forward(
        self,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor,
        time_embd: torch.FloatTensor,
        image_rotary_emb=None,
        joint_attention_kwargs=None,
        hidden_states_1: torch.FloatTensor = None,
        encoder_hidden_states_1: torch.FloatTensor = None,
        time_embd_1: torch.FloatTensor = None,
        image_rotary_emb_1: torch.FloatTensor = None,
        async_op: bool = False,
        pack_qkv: bool = True,
    ):

        stream_default = self.stream_default
        if hidden_states_1 is not None:
            stream_1 = self.stream_1
        else:
            stream_1 = None

        # async_op = async_op or hidden_states_1 is not None

        skip_attn = False
        skip_attn_1 = False

        # set all handles to None
        query_layer_handle = None
        key_layer_handle = None
        value_layer_handle = None
        output_handle = None

        query_layer_handle_1 = None
        key_layer_handle_1 = None
        value_layer_handle_1 = None
        output_handle_1 = None

        # set some events to sync

        comp_prologue_event_first_req_batch = torch.cuda.Event()
        comp_ring_event_first_req_batch = torch.cuda.Event()

        for index_block, block in enumerate(self.transformer_blocks):
            # NOTE(runyu): launch the first compute kernel for first hidden_states
            with torch.cuda.stream(stream_default):
                with nvtx_range(f"comp_prologue for first Requests_batch in layer {index_block}"):

                    (
                        query,
                        key,
                        value,
                        joint_tensor_key,
                        joint_tensor_value,
                        joint_tensor_query,
                        head_dim,
                        gate_msa,
                        scale_mlp,
                        shift_mlp,
                        gate_mlp,
                        c_gate_mlp,
                        c_gate_msa,
                        c_scale_mlp,
                        c_shift_mlp,
                    ) = multimodal_comp_prologue(
                        block=block,
                        hidden_states=hidden_states,
                        encoder_hidden_states=encoder_hidden_states,
                        time_embd=time_embd,
                        image_rotary_emb=image_rotary_emb,
                        launch_event=comp_prologue_event_first_req_batch,
                        pack_qkv=pack_qkv,
                    )

                # NOTE(runyu): launch the communication kernel for first hidden_states, here we first launch the communication kernel for first Requests_batch and then launch the compute kernel for the second Requests_batch, so that the communication kernel and compute kernel can overlap(don't let compute kernel satuates all the SM)
                with nvtx_range(f"comm_ulysses_qkv for first Requests_batch in layer {index_block}"):
                    bs_q, shard_seq_len_q, hc_q, hs_q = query.shape
                    bs_k, shard_seq_len_k, hc_k, hs_k = key.shape
                    bs_v, shard_seq_len_v, hc_v, hs_v = value.shape

                    attn_world_size = len(PROCESS_GROUP.get_attn_ranks(index_req=0))
                    non_attn_world_size = len(PROCESS_GROUP.get_non_attn_ranks(index_req=0))
                    query_layer, key_layer, value_layer = comm_ulysses_qkv(
                        block, query, key, value, async_op=async_op, pack_qkv=pack_qkv
                    )

                    # comm_ulysses_qkv_kernel_launch_event_first_request_batch.record(stream_default)
                    if async_op:
                        query_layer, query_layer_handle = query_layer
                        key_layer, key_layer_handle = key_layer
                        value_layer, value_layer_handle = value_layer

            # NOTE(runyu): launch the first compute kernel for second hidden_states
            if stream_1 is not None:
                with torch.cuda.stream(stream_1):
                    stream_1.wait_event(comp_prologue_event_first_req_batch)
                    with nvtx_range(f"comp_prologue for second Requests_batch in layer {index_block}"):
                        (
                            query_1,
                            key_1,
                            value_1,
                            joint_tensor_key_1,
                            joint_tensor_value_1,
                            joint_tensor_query_1,
                            head_dim_1,
                            gate_msa_1,
                            scale_mlp_1,
                            shift_mlp_1,
                            gate_mlp_1,
                            c_gate_mlp_1,
                            c_gate_msa_1,
                            c_scale_mlp_1,
                            c_shift_mlp_1,
                        ) = multimodal_comp_prologue(
                            block=block,
                            hidden_states=hidden_states_1,
                            encoder_hidden_states=encoder_hidden_states_1,
                            time_embd=time_embd_1,
                            image_rotary_emb=image_rotary_emb_1,
                            pack_qkv=pack_qkv,
                        )
                        # comp_prologue_event_second_request_batch.record(stream_1)
                        # check_nan({f"layer{index_block}_query_1": query_1, "key_1": key_1, "value_1": value_1})

                    with nvtx_range(f"comm_ulysses_qkv for second Requests_batch in layer {index_block}"):
                        # NOTE(runyu): launch the communication kernel for second hidden_states before comp_ring kernel for first Requests_batch launch
                        bs_q_1, shard_seq_len_q_1, hc_q_1, hs_q_1 = query_1.shape
                        bs_k_1, shard_seq_len_k_1, hc_k_1, hs_k_1 = key_1.shape
                        bs_v_1, shard_seq_len_v_1, hc_v_1, hs_v_1 = value_1.shape
                        # parallel_degree_1 = dist.get_world_size(
                        #     block.attn.processor.hybrid_seq_parallel_attn.ulysses_pg
                        # )
                        attn_world_size_1 = len(PROCESS_GROUP.get_attn_ranks(index_req=1))
                        non_attn_world_size_1 = len(PROCESS_GROUP.get_non_attn_ranks(index_req=1))
                        query_layer_1, key_layer_1, value_layer_1 = comm_ulysses_qkv(
                            block, query_1, key_1, value_1, async_op=async_op, index_req=1, pack_qkv=pack_qkv
                        )
                        # comm_ulysses_qkv_event_second_req_batch.record(stream_1)

                        if async_op:
                            query_layer_1, query_layer_handle_1 = query_layer_1
                            key_layer_1, key_layer_handle_1 = key_layer_1
                            value_layer_1, value_layer_handle_1 = value_layer_1

            if async_op:
                with torch.cuda.stream(stream_default):
                    query_layer_handle.wait()
                    key_layer_handle.wait()
                    value_layer_handle.wait()

                    if query_layer.numel() > 0:
                        query_layer = (
                            query_layer.reshape(
                                non_attn_world_size * shard_seq_len_q,
                                bs_q,
                                hc_q // attn_world_size,
                                hs_q,
                            )
                            .transpose(0, 1)
                            .contiguous()
                        )
                        key_layer = (
                            key_layer.reshape(
                                non_attn_world_size * shard_seq_len_k,
                                bs_k,
                                hc_k // attn_world_size,
                                hs_k,
                            )
                            .transpose(0, 1)
                            .contiguous()
                        )
                        value_layer = (
                            value_layer.reshape(
                                non_attn_world_size * shard_seq_len_v,
                                bs_v,
                                hc_v // attn_world_size,
                                hs_v,
                            )
                            .transpose(0, 1)
                            .contiguous()
                        )
                    else:
                        skip_attn = True
                        query_layer = None
                        key_layer = None
                        value_layer = None

            with torch.cuda.stream(stream_default):
                with nvtx_range(f"comp_ring for first Requests_batch in layer {index_block}"):
                    # if stream_1 is not None:
                    # stream_default.wait_event(comp_prologue_event_second_request_batch)
                    # stream_default.wait_event(comm_ulysses_qkv_event_second_req_batch)
                    if not skip_attn:
                        context_layer = comp_ring(
                            block,
                            query_layer=query_layer,
                            key_layer=key_layer,
                            value_layer=value_layer,
                            joint_tensor_key=joint_tensor_key,
                            joint_tensor_value=joint_tensor_value,
                            joint_tensor_query=joint_tensor_query,
                            index_req=0,
                        )
                    else:
                        context_layer = torch.Size(
                            [
                                bs_q,
                                shard_seq_len_q * len(PROCESS_GROUP.get_non_attn_ranks(index_req=0)),
                                hc_q // len(PROCESS_GROUP.get_attn_ranks(index_req=0)),
                                hs_q,
                            ]
                        )
                    comp_ring_event_first_req_batch.record(stream_default)

                with nvtx_range(f"comm_ulysses_mlp for first Requests_batch in layer {index_block}"):
                    # bs_o, seq_len_o, shard_hc_o, hs_o = context_layer.shape
                    bs_o, seq_len_o, shard_hc_o, hs_o = (
                        bs_q,
                        shard_seq_len_q * attn_world_size,
                        hc_q // attn_world_size,
                        hs_q,
                    )
                    output = comm_ulysses_mlp(
                        block,
                        context_layer,
                        async_op=async_op,
                        dtype=context_layer.dtype,
                        index_req=0,
                        pack_qkv=pack_qkv,
                        joint_tensor_query=joint_tensor_query,
                    )
                    del joint_tensor_query
                    if async_op:
                        output, output_handle = output

            if stream_1 is not None:
                with torch.cuda.stream(stream_1):
                    if async_op:
                        query_layer_handle_1.wait()
                        key_layer_handle_1.wait()
                        value_layer_handle_1.wait()

                        if query_layer_1.numel() > 0:
                            query_layer_1 = (
                                query_layer_1.reshape(
                                    non_attn_world_size_1 * shard_seq_len_q_1,
                                    bs_q_1,
                                    hc_q_1 // attn_world_size_1,
                                    hs_q_1,
                                )
                                .transpose(0, 1)
                                .contiguous()
                            )
                            key_layer_1 = (
                                key_layer_1.reshape(
                                    non_attn_world_size_1 * shard_seq_len_k_1,
                                    bs_k_1,
                                    hc_k_1 // attn_world_size_1,
                                    hs_k_1,
                                )
                                .transpose(0, 1)
                                .contiguous()
                            )
                            value_layer_1 = (
                                value_layer_1.reshape(
                                    non_attn_world_size_1 * shard_seq_len_v_1,
                                    bs_v_1,
                                    hc_v_1 // attn_world_size_1,
                                    hs_v_1,
                                )
                                .transpose(0, 1)
                                .contiguous()
                            )
                        else:
                            skip_attn_1 = True
                            query_layer_1 = None
                            key_layer_1 = None
                            value_layer_1 = None

                    with nvtx_range(f"comp_ring for second Requests_batch in layer {index_block}"):
                        # stream_1.wait_event(comp_ring_event_first_req_batch)
                        if not skip_attn_1:
                            context_layer_1 = comp_ring(
                                block=block,
                                query_layer=query_layer_1,
                                key_layer=key_layer_1,
                                value_layer=value_layer_1,
                                joint_tensor_key=joint_tensor_key_1,
                                joint_tensor_value=joint_tensor_value_1,
                                joint_tensor_query=joint_tensor_query_1,
                                index_req=1,
                            )
                        else:
                            context_layer_1 = torch.Size(
                                [
                                    bs_q_1,
                                    shard_seq_len_q_1 * len(PROCESS_GROUP.get_non_attn_ranks(index_req=1)),
                                    hc_q_1 // len(PROCESS_GROUP.get_attn_ranks(index_req=1)),
                                    hs_q_1,
                                ]
                            )
                        # comp_ring_event_second_req_batch.record(stream_1)

                    with nvtx_range(f"comm_ulysses_mlp for second Requests_batch in layer {index_block}"):
                        # bs_o_1, seq_len_o_1, shard_hc_o_1, hs_o_1 = context_layer_1.shape
                        bs_o_1, seq_len_o_1, shard_hc_o_1, hs_o_1 = (
                            bs_q_1,
                            shard_seq_len_q_1 * attn_world_size_1,
                            hc_q_1 // attn_world_size_1,
                            hs_q_1,
                        )

                        output_1 = comm_ulysses_mlp(
                            block,
                            context_layer_1,
                            async_op=async_op,
                            dtype=context_layer_1.dtype,
                            index_req=1,
                            pack_qkv=pack_qkv,
                            joint_tensor_query=joint_tensor_query_1,
                        )
                        del joint_tensor_query_1
                        if async_op:
                            output_1, output_handle_1 = output_1

            with torch.cuda.stream(stream_default):
                output_shape = (shard_hc_o * attn_world_size, seq_len_o // non_attn_world_size, bs_o, hs_o)
                output = comm_ulysses_epilogue(
                    block, output, output_handle, output_shape, async_op=async_op, pack_qkv=pack_qkv
                )
                #     # check_nan(
                #     #     {
                #     #         "output": output,
                #     #     }
                #     # )
                with nvtx_range(f"comp_epilogue for first Requests_batch in layer {index_block}"):
                    encoder_hidden_states, hidden_states = multimodal_comp_epilogue(
                        block,
                        output,
                        hidden_states,
                        encoder_hidden_states,
                        head_dim,
                        gate_msa,
                        scale_mlp,
                        shift_mlp,
                        gate_mlp,
                        c_gate_mlp,
                        c_gate_msa,
                        c_scale_mlp,
                        c_shift_mlp,
                        dtype=query.dtype,
                    )

            if stream_1 is not None:
                with torch.cuda.stream(stream_1):
                    output_shape_1 = (
                        shard_hc_o_1 * attn_world_size_1,
                        seq_len_o_1 // non_attn_world_size_1,
                        bs_o_1,
                        hs_o_1,
                    )
                    output_1 = comm_ulysses_epilogue(
                        block,
                        output_1,
                        output_handle_1,
                        output_shape_1,
                        async_op=async_op,
                        pack_qkv=pack_qkv,
                    )
                    # check_nan(
                    #     {
                    #         "output_1": output_1,
                    #     }
                    # )

                    with nvtx_range(f"comp_epilogue for second Requests_batch in layer {index_block}"):
                        # stream_1.wait_event(comp_epilogue_event_first_req_batch)
                        # check_nan({f"layer{index_block}_encoder_hidden_states_1": encoder_hidden_states_1, "hidden_states_1": hidden_states_1})
                        encoder_hidden_states_1, hidden_states_1 = multimodal_comp_epilogue(
                            block,
                            output_1,  # contain partial outliers
                            hidden_states_1,
                            encoder_hidden_states_1,
                            head_dim_1,
                            gate_msa_1,
                            scale_mlp_1,
                            shift_mlp_1,
                            gate_mlp_1,
                            c_gate_mlp_1,
                            c_gate_msa_1,
                            c_scale_mlp_1,
                            c_shift_mlp_1,
                            dtype=query_1.dtype,
                        )
                        # check_nan(
                        #     {
                        #         f"layer{index_block}_encoder_hidden_states_1": encoder_hidden_states_1,
                        #         "hidden_states_1": hidden_states_1,
                        #     }
                        # )

        if stream_1 is not None:
            stream_default.wait_stream(stream_1)

        return (
            encoder_hidden_states,
            hidden_states,
            encoder_hidden_states_1,
            hidden_states_1,
        )

    def inline_unimodal_transformer_blocks_forward(
        self,
        hidden_states: torch.FloatTensor,
        time_embd: torch.FloatTensor,
        image_rotary_emb=None,
        joint_attention_kwargs=None,
        hidden_states_1: torch.FloatTensor = None,  # for the second request batch
        time_embd_1: torch.FloatTensor = None,
        image_rotary_emb_1=None,
        joint_attention_kwargs_1=None,
        async_op=False,
        pack_qkv=True,
    ):
        """
        Inline forward for all transformer blocks,with special stream handling to overlap across requests.
        """
        stream_default = self.stream_default
        if hidden_states_1 is not None:
            stream_1 = self.stream_1
            # TODO precision error between out1 and out2 for shape >= 1024 * 1024 without this
            stream_1.synchronize()
        else:
            stream_1 = None
        # async_op = async_op or hidden_states_1 is not None

        skip_attn = False
        skip_attn_1 = False

        # set all handles to None
        query_layer_handle = None
        key_layer_handle = None
        value_layer_handle = None
        output_handle = None

        query_layer_handle_1 = None
        key_layer_handle_1 = None
        value_layer_handle_1 = None
        output_handle_1 = None

        # set some events to sync
        comp_prologue_event_first_req_batch = torch.cuda.Event()
        comp_ring_event_first_req_batch = torch.cuda.Event()

        for index_block, block in enumerate(self.single_transformer_blocks):
            with torch.cuda.stream(stream_default):
                with nvtx_range(f"Single comp_prologue for first Requests_batch in layer {index_block}"):

                    # if stream_1 is not None and index_block != 0:
                    # stream_default.wait_event(comp_epilogue_event_second_req_batch)

                    (
                        query,
                        key,
                        value,
                        joint_tensor_key,
                        joint_tensor_value,
                        joint_tensor_query,
                        head_dim,
                        batch_size,
                        residual,
                        gate,
                        mlp_hidden_states,
                    ) = unimodal_comp_prologue(
                        block=block,
                        hidden_states=hidden_states,
                        time_embd=time_embd,
                        image_rotary_emb=image_rotary_emb,
                        joint_attention_kwargs=joint_attention_kwargs,
                        index_req=0,
                        launch_event=comp_prologue_event_first_req_batch,
                        pack_qkv=pack_qkv,
                    )
                    # check_nan({"hidden_states": hidden_states})

                with nvtx_range(f"comm_ulysses_qkv for first Requests_batch in layer {index_block}"):
                    bs_q, shard_seq_len_q, hc_q, hs_q = query.shape
                    bs_k, shard_seq_len_k, hc_k, hs_k = key.shape
                    bs_v, shard_seq_len_v, hc_v, hs_v = value.shape
                    # parallel_degree = dist.get_world_size(block.attn.processor.hybrid_seq_parallel_attn.ulysses_pg)
                    attn_world_size = len(PROCESS_GROUP.get_attn_ranks(index_req=0))
                    non_attn_world_size = len(PROCESS_GROUP.get_non_attn_ranks(index_req=0))

                    query_layer, key_layer, value_layer = comm_ulysses_qkv(
                        block, query, key, value, async_op=async_op, pack_qkv=pack_qkv
                    )

                    if async_op:
                        query_layer, query_layer_handle = query_layer
                        key_layer, key_layer_handle = key_layer
                        value_layer, value_layer_handle = value_layer

            # NOTE(runyu): launch the first compute kernel for second hidden_states
            if stream_1 is not None:
                with torch.cuda.stream(stream_1):
                    with nvtx_range(f"comp_prologue for second Requests_batch in layer {index_block}"):
                        # comp_prologue_event_first_req_batch.wait()

                        (
                            query_1,
                            key_1,
                            value_1,
                            joint_tensor_key_1,
                            joint_tensor_value_1,
                            joint_tensor_query_1,
                            head_dim_1,
                            batch_size_1,
                            residual_1,
                            gate_1,
                            mlp_hidden_states_1,
                        ) = unimodal_comp_prologue(
                            block=block,
                            hidden_states=hidden_states_1,
                            time_embd=time_embd_1,
                            image_rotary_emb=image_rotary_emb_1,
                            joint_attention_kwargs=joint_attention_kwargs_1,
                            index_req=1,
                            pack_qkv=pack_qkv,
                        )

                        # comp_prologue_event_second_request_batch.record(stream_1)

                    with nvtx_range(f"comm_ulysses_qkv for second Requests_batch in layer {index_block}"):
                        bs_q_1, shard_seq_len_q_1, hc_q_1, hs_q_1 = query_1.shape
                        bs_k_1, shard_seq_len_k_1, hc_k_1, hs_k_1 = key_1.shape
                        bs_v_1, shard_seq_len_v_1, hc_v_1, hs_v_1 = value_1.shape
                        # parallel_degree_1 = dist.get_world_size(
                        #     block.attn.processor.hybrid_seq_parallel_attn.ulysses_pg
                        # )
                        attn_world_size_1 = len(PROCESS_GROUP.get_attn_ranks(index_req=1))
                        non_attn_world_size_1 = len(PROCESS_GROUP.get_non_attn_ranks(index_req=1))

                        query_layer_1, key_layer_1, value_layer_1 = comm_ulysses_qkv(
                            block,
                            query_1,
                            key_1,
                            value_1,
                            async_op=async_op,
                            pack_qkv=pack_qkv,
                        )

                        if async_op:
                            query_layer_1, query_layer_handle_1 = query_layer_1
                            key_layer_1, key_layer_handle_1 = key_layer_1
                            value_layer_1, value_layer_handle_1 = value_layer_1

            if async_op:
                with torch.cuda.stream(stream_default):
                    query_layer_handle.wait()
                    key_layer_handle.wait()
                    value_layer_handle.wait()

                    if query_layer.numel() > 0:
                        query_layer = (
                            query_layer.reshape(
                                non_attn_world_size * shard_seq_len_q,
                                bs_q,
                                hc_q // attn_world_size,
                                hs_q,
                            )
                            .transpose(0, 1)
                            .contiguous()
                        )
                        key_layer = (
                            key_layer.reshape(
                                non_attn_world_size * shard_seq_len_k,
                                bs_k,
                                hc_k // attn_world_size,
                                hs_k,
                            )
                            .transpose(0, 1)
                            .contiguous()
                        )
                        value_layer = (
                            value_layer.reshape(
                                non_attn_world_size * shard_seq_len_v,
                                bs_v,
                                hc_v // attn_world_size,
                                hs_v,
                            )
                            .transpose(0, 1)
                            .contiguous()
                        )
                    else:
                        skip_attn = True
                        query_layer = None
                        key_layer = None
                        value_layer = None

            with torch.cuda.stream(stream_default):

                with nvtx_range(f"Single comp_ring for first Requests_batch in layer {index_block}"):
                    # if stream_1 is not None:
                    # stream_default.wait_event(comp_prologue_event_second_request_batch)
                    if not skip_attn:
                        context_layer = comp_ring(
                            block,
                            query_layer=query_layer,
                            key_layer=key_layer,
                            value_layer=value_layer,
                            joint_tensor_key=joint_tensor_key,
                            joint_tensor_value=joint_tensor_value,
                            joint_tensor_query=joint_tensor_query,
                            index_req=0,
                        )
                    else:
                        context_layer = torch.Size(
                            [
                                bs_q,
                                shard_seq_len_q * len(PROCESS_GROUP.get_non_attn_ranks(index_req=0)),
                                hc_q // len(PROCESS_GROUP.get_attn_ranks(index_req=0)),
                                hs_q,
                            ]
                        )

                    # comp_ring_event_first_req_batch.record(stream_default)

                with nvtx_range(f"Single comm_ulysses_mlp for first Requests_batch in layer {index_block}"):
                    # bs_o, seq_len_o, shard_hc_o, hs_o = context_layer.shape
                    bs_o, seq_len_o, shard_hc_o, hs_o = (
                        bs_q,
                        shard_seq_len_q * attn_world_size,
                        hc_q // attn_world_size,
                        hs_q,
                    )
                    output = comm_ulysses_mlp(
                        block,
                        context_layer,
                        async_op=async_op,
                        dtype=context_layer.dtype,
                        index_req=0,
                        pack_qkv=pack_qkv,
                        joint_tensor_query=joint_tensor_query,
                    )
                    del joint_tensor_query
                    if async_op:
                        output, output_handle = output

            if stream_1 is not None:
                with torch.cuda.stream(stream_1):
                    if async_op:
                        query_layer_handle_1.wait()
                        key_layer_handle_1.wait()
                        value_layer_handle_1.wait()

                        if query_layer_1.numel() > 0:
                            query_layer_1 = (
                                query_layer_1.reshape(
                                    non_attn_world_size_1 * shard_seq_len_q_1,
                                    bs_q_1,
                                    hc_q_1 // attn_world_size_1,
                                    hs_q_1,
                                )
                                .transpose(0, 1)
                                .contiguous()
                            )
                            key_layer_1 = (
                                key_layer_1.reshape(
                                    non_attn_world_size_1 * shard_seq_len_k_1,
                                    bs_k_1,
                                    hc_k_1 // attn_world_size_1,
                                    hs_k_1,
                                )
                                .transpose(0, 1)
                                .contiguous()
                            )
                            value_layer_1 = (
                                value_layer_1.reshape(
                                    non_attn_world_size_1 * shard_seq_len_v_1,
                                    bs_v_1,
                                    hc_v_1 // attn_world_size_1,
                                    hs_v_1,
                                )
                                .transpose(0, 1)
                                .contiguous()
                            )
                        else:
                            skip_attn_1 = True
                            query_layer_1 = None
                            key_layer_1 = None
                            value_layer_1 = None

                    with nvtx_range(f"comp_ring for second Requests_batch in layer {index_block}"):
                        stream_1.wait_event(comp_ring_event_first_req_batch)
                        if not skip_attn_1:
                            context_layer_1 = comp_ring(
                                block=block,
                                query_layer=query_layer_1,
                                key_layer=key_layer_1,
                                value_layer=value_layer_1,
                                joint_tensor_key=joint_tensor_key_1,
                                joint_tensor_value=joint_tensor_value_1,
                                index_req=1,
                            )
                        else:
                            context_layer_1 = torch.Size(
                                [
                                    bs_q_1,
                                    shard_seq_len_q_1 * len(PROCESS_GROUP.get_non_attn_ranks(index_req=1)),
                                    hc_q_1 // len(PROCESS_GROUP.get_attn_ranks(index_req=1)),
                                    hs_q_1,
                                ]
                            )
                        # comp_ring_event_second_req_batch.record(stream_1)

                    with nvtx_range(f"Single comm_ulysses_mlp for second Requests_batch in layer {index_block}"):
                        # bs_o_1, seq_len_o_1, shard_hc_o_1, hs_o_1 = context_layer_1.shape
                        bs_o_1, seq_len_o_1, shard_hc_o_1, hs_o_1 = (
                            bs_q_1,
                            shard_seq_len_q_1 * attn_world_size_1,
                            hc_q_1 // attn_world_size_1,
                            hs_q_1,
                        )

                        output_1 = comm_ulysses_mlp(
                            block,
                            context_layer_1,
                            async_op=async_op,
                            dtype=context_layer_1.dtype,
                            index_req=1,
                            pack_qkv=pack_qkv,
                            joint_tensor_query=joint_tensor_query_1,
                        )
                        del joint_tensor_query_1
                        if async_op:
                            output_1, output_handle_1 = output_1

            with torch.cuda.stream(stream_default):
                output_shape = (shard_hc_o * attn_world_size, seq_len_o // non_attn_world_size, bs_o, hs_o)
                output = comm_ulysses_epilogue(
                    block, output, output_handle, output_shape, async_op=async_op, pack_qkv=pack_qkv
                )
                with nvtx_range(f"comp_epilogue for first Requests_batch in layer {index_block}"):
                    hidden_states = unimodal_comp_epilogue(
                        block,
                        attn_output=output,
                        residual=residual,
                        gate=gate,
                        mlp_hidden_states=mlp_hidden_states,
                        head_dim=head_dim,
                        dtype=query.dtype,
                    )

            if stream_1 is not None:
                with torch.cuda.stream(stream_1):
                    output_shape_1 = (
                        shard_hc_o_1 * attn_world_size_1,
                        seq_len_o_1 // non_attn_world_size_1,
                        bs_o_1,
                        hs_o_1,
                    )
                    output_1 = comm_ulysses_epilogue(
                        block,
                        output_1,
                        output_handle_1,
                        output_shape_1,
                        async_op=async_op,
                        pack_qkv=pack_qkv,
                    )

                    # check_nan({"output_1": output_1})
                    with nvtx_range(f"comp_epilogue for second Requests_batch in layer {index_block}"):
                        # stream_1.wait_event(comp_epilogue_event_first_req_batch)
                        hidden_states_1 = unimodal_comp_epilogue(
                            block,
                            attn_output=output_1,
                            residual=residual_1,
                            gate=gate_1,
                            mlp_hidden_states=mlp_hidden_states_1,
                            head_dim=head_dim_1,
                            dtype=query.dtype,
                        )
                        # comp_epilogue_event_second_req_batch.record(stream_1)

        if stream_1 is not None:
            stream_default.wait_stream(stream_1)

        return hidden_states, hidden_states_1

    def prepare_input(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        pooled_projections: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        img_ids: torch.Tensor = None,
        txt_ids: torch.Tensor = None,
        guidance: torch.Tensor = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    ):
        if joint_attention_kwargs is not None:
            joint_attention_kwargs = joint_attention_kwargs.copy()
            lora_scale = joint_attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if joint_attention_kwargs is not None and joint_attention_kwargs.get("scale", None) is not None:
                logger.warning(
                    "Passing `scale` via `joint_attention_kwargs` when not using the PEFT backend is ineffective."
                )
        hidden_states = self.x_embedder(hidden_states)

        timestep = timestep.to(hidden_states.dtype) * 1000
        if guidance is not None:
            guidance = guidance.to(hidden_states.dtype) * 1000
        else:
            guidance = None
        time_embd = (
            self.time_text_embed(timestep, pooled_projections)
            if guidance is None
            else self.time_text_embed(timestep, guidance, pooled_projections)
        )
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)

        if txt_ids.ndim == 3:
            logger.warning(
                "Passing `txt_ids` 3d torch.Tensor is deprecated."
                "Please remove the batch dimension and pass it as a 2d torch Tensor"
            )
            txt_ids = txt_ids[0]
        if img_ids.ndim == 3:
            logger.warning(
                "Passing `img_ids` 3d torch.Tensor is deprecated."
                "Please remove the batch dimension and pass it as a 2d torch Tensor"
            )
            img_ids = img_ids[0]

        ids = torch.cat((txt_ids, img_ids), dim=0)
        image_rotary_emb = self.pos_embed(ids)

        return hidden_states, encoder_hidden_states, time_embd, image_rotary_emb, lora_scale

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        pooled_projections: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        img_ids: torch.Tensor = None,
        txt_ids: torch.Tensor = None,
        guidance: torch.Tensor = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
        hidden_states_1: torch.Tensor = None,
        encoder_hidden_states_1: torch.Tensor = None,
        pooled_projections_1: torch.Tensor = None,
        timestep_1: torch.LongTensor = None,
        img_ids_1: torch.Tensor = None,
        txt_ids_1: torch.Tensor = None,
        guidance_1: torch.Tensor = None,
        joint_attention_kwargs_1: Optional[Dict[str, Any]] = None,
        overlap: bool = False,
        inline_inference: bool = False,
        no_stream: bool = False,
        async_op: bool = False,
        pack_qkv: bool = True,
    ) -> Union[torch.FloatTensor, Transformer2DModelOutput]:
        """
        The [`FluxTransformer2DModel`] forward method. See model flow chart: https://www.reddit.com/media?url=https%3A%2F%2Fpreview.redd.it%2Ffluxs-architecture-diagram-dont-think-theres-a-paper-so-had-v0-7bggr77f7t0e1.png%3Fwidth%3D1023%26format%3Dpng%26auto%3Dwebp%26s%3D9673e4e7cdbb7b7779e2931dcc189f2211c794c2

        Args:
            hidden_states (`torch.FloatTensor` of shape `(batch size, channel, height, width)`):
                Input `hidden_states`.
            encoder_hidden_states (`torch.FloatTensor` of shape `(batch size, sequence_len, embed_dims)`):
                Conditional embeddings (embeddings computed from the input conditions such as prompts) to use.
            pooled_projections (`torch.FloatTensor` of shape `(batch_size, projection_dim)`): Embeddings projected
                from the embeddings of input conditions.
            timestep ( `torch.LongTensor`):
                Used to indicate denoising step.
            block_controlnet_hidden_states: (`list` of `torch.Tensor`):
                A list of tensors that if specified are added to the residuals of transformer blocks.
            joint_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.transformer_2d.Transformer2DModelOutput`] instead of a plain
                tuple.

        Returns:
            If `return_dict` is True, an [`~models.transformer_2d.Transformer2DModelOutput`] is returned, otherwise a
            `tuple` where the first element is the sample tensor.
        """
        # logger.info(f"hidden_states: {hidden_states.shape}")
        # logger.info(f"encoder_hidden_states: {encoder_hidden_states.shape}")
        if get_sequence_parallel_world_size() <= 1:
            pack_qkv = False
        hidden_states, encoder_hidden_states, time_embd, image_rotary_emb, lora_scale = self.prepare_input(
            hidden_states,
            encoder_hidden_states,
            pooled_projections,
            timestep,
            img_ids,
            txt_ids,
            guidance,
            joint_attention_kwargs,
        )

        if hidden_states_1 is not None:
            # assert inline_inference, "inline_inference must be True when hidden_states_1 is not None"
            (
                hidden_states_1,
                encoder_hidden_states_1,
                time_embd_1,
                image_rotary_emb_1,
                lora_scale_1,
            ) = self.prepare_input(
                hidden_states_1,
                encoder_hidden_states_1,
                pooled_projections_1,
                timestep_1,
                img_ids_1,
                txt_ids_1,
                guidance_1,
                joint_attention_kwargs_1,
            )
        else:
            time_embd_1 = None
            image_rotary_emb_1 = None

        if inline_inference:
            # logger.info(f"use inline_inference, this may takes more memory") # TODO(runyu): figure out the memory usage.
            if not no_stream:
                with nvtx_range("multimodal forward"):
                    (
                        encoder_hidden_states,
                        hidden_states,
                        encoder_hidden_states_1,
                        hidden_states_1,
                    ) = self.inline_multimodal_transformer_blocks_forward(
                        hidden_states=hidden_states,
                        encoder_hidden_states=encoder_hidden_states,
                        time_embd=time_embd,
                        image_rotary_emb=image_rotary_emb,
                        hidden_states_1=hidden_states_1,
                        encoder_hidden_states_1=encoder_hidden_states_1,
                        time_embd_1=time_embd_1,
                        image_rotary_emb_1=image_rotary_emb_1,
                        async_op=async_op,
                        pack_qkv=pack_qkv,
                    )
            else:
                raise ValueError(
                    "no_stream has been deprecated for inline inference. You should use multi-stream to get better overlap"
                )
        else:
            for index_block, block in enumerate(self.transformer_blocks):
                with nvtx_range(f"transformer_block for first Requests_batch in layer {index_block}"):
                    encoder_hidden_states, hidden_states = block(
                        hidden_states=hidden_states,
                        encoder_hidden_states=encoder_hidden_states,
                        temb=time_embd,
                        image_rotary_emb=image_rotary_emb,
                    )

                if hidden_states_1 is not None:
                    with nvtx_range(f"transformer_block for second Requests_batch in layer {index_block}"):
                        encoder_hidden_states_1, hidden_states_1 = block(
                            hidden_states=hidden_states_1,
                            encoder_hidden_states=encoder_hidden_states_1,
                            temb=time_embd_1,
                            image_rotary_emb=image_rotary_emb_1,
                        )

        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
        if hidden_states_1 is not None:
            hidden_states_1 = torch.cat([encoder_hidden_states_1, hidden_states_1], dim=1)

        if inline_inference:
            with nvtx_range("unimodal forward"):
                hidden_states, hidden_states_1 = self.inline_unimodal_transformer_blocks_forward(
                    hidden_states=hidden_states,
                    time_embd=time_embd,
                    image_rotary_emb=image_rotary_emb,
                    hidden_states_1=hidden_states_1,
                    time_embd_1=time_embd_1,
                    image_rotary_emb_1=image_rotary_emb_1,
                    async_op=async_op,
                    pack_qkv=pack_qkv,
                )
        else:
            for index_block, block in enumerate(self.single_transformer_blocks):
                with nvtx_range(f"single_transformer_block for first Requests_batch in layer {index_block}"):
                    hidden_states = block(
                        hidden_states=hidden_states,
                        temb=time_embd,
                        image_rotary_emb=image_rotary_emb,
                    )

                if hidden_states_1 is not None:
                    with nvtx_range(f"single_transformer_block for second Requests_batch in layer {index_block}"):
                        hidden_states_1 = block(
                            hidden_states=hidden_states_1,
                            temb=time_embd_1,
                            image_rotary_emb=image_rotary_emb_1,
                        )

        hidden_states = hidden_states[:, encoder_hidden_states.shape[1] :, ...]
        if hidden_states_1 is not None:
            hidden_states_1 = hidden_states_1[:, encoder_hidden_states_1.shape[1] :, ...]

        hidden_states = self.norm_out(hidden_states, time_embd)
        if hidden_states_1 is not None:
            hidden_states_1 = self.norm_out(hidden_states_1, time_embd_1)
        output = self.proj_out(hidden_states)
        if hidden_states_1 is not None:
            output_1 = self.proj_out(hidden_states_1)

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            if hidden_states_1 is not None:
                return (output, output_1)
            else:
                return (output,)

        if hidden_states_1 is not None:
            return Transformer2DModelOutput(sample=output), Transformer2DModelOutput(sample=output_1)
        else:
            return Transformer2DModelOutput(sample=output)


if __name__ == "__main__":
    # print module in FluxTransformer2DModel
    model = FluxTransformer2DModel()
