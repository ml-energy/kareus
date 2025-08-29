import inspect
from typing import Any, Dict, Optional, Union, List

import torch
import torch.distributed
import torch.distributed as dist
from diffusers.models.transformers.transformer_2d import Transformer2DModelOutput
from diffusers.models.transformers.transformer_flux import FluxTransformer2DModel
from diffusers.models.transformers.transformer_flux import FluxSingleTransformerBlock, FluxTransformerBlock
from diffusers.utils import USE_PEFT_BACKEND, scale_lora_layers, unscale_lora_layers

from cfuser.core.distributed import get_world_group
from cfuser.core.long_ctx_attention.comm import all_to_all_4D, uneven_all_to_all_4D
from cfuser.core.utils import nvtx_range
from cfuser.envs import PACKAGES_CHECKER
from cfuser.logger import init_logger
from cfuser.model_executor.layers.attention_processor import apply_rotary_emb
from cfuser.core.distributed.globals import PROCESS_GROUP
from cfuser.model_executor.models.transformers.transformer_flux import cFuserFluxTransformer2DWrapper
from cfuser.model_executor.models.transformers.transformer_flux import (
    multimodal_comp_prologue,
    comm_ulysses_qkv,
    comp_ring,
    comm_ulysses_mlp,
    multimodal_comp_epilogue,
    unimodal_comp_prologue,
    unimodal_comp_epilogue,
)
from typing import Union

env_info = PACKAGES_CHECKER.get_packages_info()
HAS_LONG_CTX_ATTN = env_info["has_long_ctx_attn"]
HAS_FLASH_ATTN = env_info["has_flash_attn"]

logger = init_logger(__name__)


def inline_multimodal_transformer_blocks_forward(
    self: cFuserFluxTransformer2DWrapper,
    hidden_states_list: List[torch.Tensor],
    encoder_hidden_states_list: List[torch.Tensor] = None,
    time_embd_list: List[torch.Tensor] = None,
    image_rotary_emb_list: List[torch.Tensor] = None,
    lora_scale_list: List[torch.Tensor] = None,
    async_op: bool = False,
):
    for index_block, block in enumerate(self.transformer_blocks):

        query_list = []
        key_list = []
        value_list = []
        joint_tensor_key_list = []
        joint_tensor_value_list = []
        head_dim_list = []
        gate_msa_list = []
        scale_mlp_list = []
        shift_mlp_list = []
        gate_mlp_list = []
        c_gate_mlp_list = []
        c_gate_msa_list = []
        c_scale_mlp_list = []
        c_shift_mlp_list = []

        query_layer_list = []
        key_layer_list = []
        value_layer_list = []

        q_shape_list = []

        skip_attn_list = []

        a2a_dtype = hidden_states_list[0].dtype  # NOTE: Crucial for all_to_all to send the correct amount of data
        for index_req, (hidden_states, encoder_hidden_states, time_embd, image_rotary_emb, lora_scale) in enumerate(
            zip(hidden_states_list, encoder_hidden_states_list, time_embd_list, image_rotary_emb_list, lora_scale_list)
        ):
            with nvtx_range(f"multimodal_comp_prologue for req {index_req} in layer {index_block}"):
                (
                    query,
                    key,
                    value,
                    joint_tensor_key,
                    joint_tensor_value,
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
                    index_req=index_req,
                )

                query_list.append(query)
                key_list.append(key)
                value_list.append(value)
                joint_tensor_key_list.append(joint_tensor_key)
                joint_tensor_value_list.append(joint_tensor_value)
                head_dim_list.append(head_dim)
                gate_msa_list.append(gate_msa)
                scale_mlp_list.append(scale_mlp)
                shift_mlp_list.append(shift_mlp)
                gate_mlp_list.append(gate_mlp)
                c_gate_mlp_list.append(c_gate_mlp)
                c_gate_msa_list.append(c_gate_msa)
                c_scale_mlp_list.append(c_scale_mlp)
                c_shift_mlp_list.append(c_shift_mlp)

            with nvtx_range(f"comm_ulysses_qkv for req {index_req} in layer {index_block}"):
                bs_q, shard_seq_len_q, hc_q, hs_q = query.shape
                q_shape_list.append((bs_q, shard_seq_len_q, hc_q, hs_q))
                bs_k, shard_seq_len_k, hc_k, hs_k = key.shape
                bs_v, shard_seq_len_v, hc_v, hs_v = value.shape
                ulysses_world_size = PROCESS_GROUP.get_ulysses_size(index_req=index_req)
                non_attn_world_size = len(PROCESS_GROUP.get_non_attn_ranks(index_req=index_req))

                query_layer, key_layer, value_layer = comm_ulysses_qkv(
                    block, query, key, value, async_op=async_op, index_req=index_req
                )

                skip_attn = False
                if async_op:
                    query_layer, query_layer_handle = query_layer
                    key_layer, key_layer_handle = key_layer
                    value_layer, value_layer_handle = value_layer

                    query_layer_handle.wait()
                    key_layer_handle.wait()
                    value_layer_handle.wait()

                    if query_layer.numel() > 0:
                        query_layer = (
                            query_layer.reshape(
                                non_attn_world_size * shard_seq_len_q,
                                bs_q,
                                hc_q // ulysses_world_size,
                                hs_q,
                            )
                            .transpose(0, 1)
                            .contiguous()
                        )
                        key_layer = (
                            key_layer.reshape(
                                non_attn_world_size * shard_seq_len_k,
                                bs_k,
                                hc_k // ulysses_world_size,
                                hs_k,
                            )
                            .transpose(0, 1)
                            .contiguous()
                        )
                        value_layer = (
                            value_layer.reshape(
                                non_attn_world_size * shard_seq_len_v,
                                bs_v,
                                hc_v // ulysses_world_size,
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
                else:
                    if query_layer is None:
                        skip_attn = True
                    else:
                        ring_world_size = PROCESS_GROUP.get_ring_size(index_req=index_req)
                        assert query_layer.shape == (
                            bs_q,
                            shard_seq_len_q * non_attn_world_size // ring_world_size,
                            hc_q // ulysses_world_size,
                            hs_q,
                        ), f"rank in world group: {get_world_group().rank_in_group}, query_layer shape: {query_layer.shape}"

                query_layer_list.append(query_layer)
                key_layer_list.append(key_layer)
                value_layer_list.append(value_layer)
                skip_attn_list.append(skip_attn)

        context_layer_list = []
        for index_req, (query_layer, key_layer, value_layer, skip_attn) in enumerate(
            zip(query_layer_list, key_layer_list, value_layer_list, skip_attn_list)
        ):
            with nvtx_range(f"comp_ring for req {index_req} in layer {index_block}"):
                non_attn_world_size = len(PROCESS_GROUP.get_non_attn_ranks(index_req=index_req))
                if not skip_attn:

                    context_layer = comp_ring(
                        block,
                        query_layer,
                        key_layer,
                        value_layer,
                        joint_tensor_key_list[index_req],
                        joint_tensor_value_list[index_req],
                        index_req=index_req,
                    )

                    ring_world_size = PROCESS_GROUP.get_ring_size(index_req=index_req)
                    assert context_layer.shape == (
                        bs_q,
                        shard_seq_len_q * non_attn_world_size // ring_world_size,
                        hc_q // ulysses_world_size,
                        hs_q,
                    ), f"rank in world group: {get_world_group().rank_in_group}"
                else:
                    bs_q, shard_seq_len_q, hc_q, hs_q = q_shape_list[index_req]
                    ulysses_world_size = PROCESS_GROUP.get_ulysses_size(index_req=index_req)
                    context_layer = torch.Size(
                        [bs_q, shard_seq_len_q * non_attn_world_size, hc_q // ulysses_world_size, hs_q]
                    )

                context_layer_list.append(context_layer)

        output_list = []
        # logger.info(
        #     f"rank {dist.get_rank()} start comm_ulysses_mlp in layer {index_block}/{len(self.transformer_blocks) - 1}"
        # )
        for index_req, (context_layer, skip_attn) in enumerate(zip(context_layer_list, skip_attn_list)):
            with nvtx_range(f"comm_ulysses_mlp for req {index_req} in layer {index_block}"):
                bs_q, shard_seq_len_q, hc_q, hs_q = q_shape_list[index_req]
                ulysses_world_size = PROCESS_GROUP.get_ulysses_size(index_req=index_req)
                non_attn_world_size = len(PROCESS_GROUP.get_non_attn_ranks(index_req=index_req))
                bs_o, seq_len_o, shard_hc_o, hs_o = (
                    bs_q,
                    shard_seq_len_q * non_attn_world_size,
                    hc_q // ulysses_world_size,
                    hs_q,
                )

                # gather idx = 2, scatter idx = 1
                output = comm_ulysses_mlp(block, context_layer, dtype=a2a_dtype, async_op=async_op, index_req=index_req)
                if async_op:
                    output, output_handle = output
                    output_handle.wait()

                    output = output.reshape(
                        shard_hc_o * ulysses_world_size,
                        seq_len_o // non_attn_world_size,
                        bs_o,
                        hs_o,
                    )
                    output = output.transpose(0, 2).contiguous()

                assert output.shape == (
                    bs_o,
                    seq_len_o // non_attn_world_size,
                    shard_hc_o * ulysses_world_size,
                    hs_o,
                ), f"rank in world group: {get_world_group().rank_in_group}, output shape: {output.shape}"

                output_list.append(output)

        new_hidden_states_list = []
        new_encoder_hidden_states_list = []
        for index_req, (
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
        ) in enumerate(
            zip(
                output_list,
                hidden_states_list,
                encoder_hidden_states_list,
                head_dim_list,
                gate_msa_list,
                scale_mlp_list,
                shift_mlp_list,
                gate_mlp_list,
                c_gate_mlp_list,
                c_gate_msa_list,
                c_scale_mlp_list,
                c_shift_mlp_list,
            )
        ):
            with nvtx_range(f"multimodal_comp_epilogue for req {index_req} in layer {index_block}"):
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
                    dtype=hidden_states.dtype,
                )
                new_hidden_states_list.append(hidden_states)
                new_encoder_hidden_states_list.append(encoder_hidden_states)

        hidden_states_list = new_hidden_states_list
        encoder_hidden_states_list = new_encoder_hidden_states_list
    # logger.info(f"rank {dist.get_rank()} finished multimodal transformer blocks")
    return hidden_states_list, encoder_hidden_states_list


def inline_unimodal_transformer_blocks_forward(
    self: cFuserFluxTransformer2DWrapper,
    hidden_states_list: List[torch.FloatTensor],
    time_embd_list: List[torch.FloatTensor],
    image_rotary_emb_list: List[torch.FloatTensor],
    joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    async_op: bool = False,
):

    for index_block, block in enumerate(self.single_transformer_blocks):

        query_list = []
        key_list = []
        value_list = []
        joint_tensor_key_list = []
        joint_tensor_value_list = []
        head_dim_list = []
        batch_size_list = []
        residual_list = []
        gate_list = []
        mlp_hidden_states_list = []

        a2a_dtype = hidden_states_list[0].dtype
        for index_req, (hidden_states, time_embd, image_rotary_emb) in enumerate(
            zip(hidden_states_list, time_embd_list, image_rotary_emb_list)
        ):
            with nvtx_range(f"Single comp_prologue for req {index_req} in layer {index_block}"):
                (
                    query,
                    key,
                    value,
                    joint_tensor_key,
                    joint_tensor_value,
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
                    index_req=index_req,
                )

            query_list.append(query)
            key_list.append(key)
            value_list.append(value)
            joint_tensor_key_list.append(joint_tensor_key)
            joint_tensor_value_list.append(joint_tensor_value)
            head_dim_list.append(head_dim)
            batch_size_list.append(batch_size)
            residual_list.append(residual)
            gate_list.append(gate)
            mlp_hidden_states_list.append(mlp_hidden_states)

        # dist.barrier(group=PROCESS_GROUP.get_non_attn_pg(index_req=index_req))

        query_layer_list = []
        q_shape_list = []
        key_layer_list = []
        value_layer_list = []
        skip_attn_list = []

        for index_req, (
            query,
            key,
            value,
            joint_tensor_key,
            joint_tensor_value,
            head_dim,
            batch_size,
            residual,
            gate,
            mlp_hidden_states,
        ) in enumerate(
            zip(
                query_list,
                key_list,
                value_list,
                joint_tensor_key_list,
                joint_tensor_value_list,
                head_dim_list,
                batch_size_list,
                residual_list,
                gate_list,
                mlp_hidden_states_list,
            )
        ):
            with nvtx_range(f"Single comm_ulysses_qkv for req {index_req} in layer {index_block}"):
                # TODO: fix this
                bs_q, shard_seq_len_q, hc_q, hs_q = query.shape
                q_shape_list.append((bs_q, shard_seq_len_q, hc_q, hs_q))
                bs_k, shard_seq_len_k, hc_k, hs_k = key.shape
                bs_v, shard_seq_len_v, hc_v, hs_v = value.shape

                query_layer, key_layer, value_layer = comm_ulysses_qkv(
                    block, query, key, value, async_op=async_op, index_req=index_req
                )

                skip_attn = False
                if async_op:
                    query_layer, query_layer_handle = query_layer
                    key_layer, key_layer_handle = key_layer
                    value_layer, value_layer_handle = value_layer

                    query_layer_handle.wait()
                    key_layer_handle.wait()
                    value_layer_handle.wait()

                    ulysses_world_size = PROCESS_GROUP.get_ulysses_size(index_req=index_req)
                    non_attn_world_size = len(PROCESS_GROUP.get_non_attn_ranks(index_req=index_req))

                    if query_layer.numel() > 0:
                        query_layer = (
                            query_layer.reshape(
                                non_attn_world_size * shard_seq_len_q,
                                bs_q,
                                hc_q // ulysses_world_size,
                                hs_q,
                            )
                            .transpose(0, 1)
                            .contiguous()
                        )
                        key_layer = (
                            key_layer.reshape(
                                non_attn_world_size * shard_seq_len_k,
                                bs_k,
                                hc_k // ulysses_world_size,
                                hs_k,
                            )
                            .transpose(0, 1)
                            .contiguous()
                        )
                        value_layer = (
                            value_layer.reshape(
                                non_attn_world_size * shard_seq_len_v,
                                bs_v,
                                hc_v // ulysses_world_size,
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
                else:
                    if query_layer is None:
                        skip_attn = True
                    else:
                        ring_world_size = PROCESS_GROUP.get_ring_size(index_req=index_req)
                        non_attn_world_size = len(PROCESS_GROUP.get_non_attn_ranks(index_req=index_req))
                        ulysses_world_size = PROCESS_GROUP.get_ulysses_size(index_req=index_req)
                        assert query_layer.shape == (
                            bs_q,
                            shard_seq_len_q * non_attn_world_size // ring_world_size,
                            hc_q // ulysses_world_size,
                            hs_q,
                        ), f"rank in world group: {get_world_group().rank_in_group}"

                query_layer_list.append(query_layer)
                key_layer_list.append(key_layer)
                value_layer_list.append(value_layer)
                skip_attn_list.append(skip_attn)

        dist.barrier(group=PROCESS_GROUP.get_non_attn_pg(index_req=index_req))
        # logger.info(
        #     f"rank {get_world_group().rank_in_group} unimodal finished comm_ulysses_qkv in layer {index_block}/{len(self.single_transformer_blocks) - 1}"
        # )

        context_layer_list = []
        for index_req, (query_layer, key_layer, value_layer, skip_attn) in enumerate(
            zip(query_layer_list, key_layer_list, value_layer_list, skip_attn_list)
        ):
            with nvtx_range(f"Single comp_ring for req {index_req} in layer {index_block}"):
                non_attn_world_size = len(PROCESS_GROUP.get_non_attn_ranks(index_req=index_req))
                if not skip_attn:
                    context_layer = comp_ring(
                        block,
                        query_layer,
                        key_layer,
                        value_layer,
                        joint_tensor_key_list[index_req],
                        joint_tensor_value_list[index_req],
                        index_req=index_req,
                    )
                else:
                    bs_q, shard_seq_len_q, hc_q, hs_q = q_shape_list[index_req]
                    ulysses_world_size = PROCESS_GROUP.get_ulysses_size(index_req=index_req)
                    context_layer = torch.Size(
                        [bs_q, shard_seq_len_q * non_attn_world_size, hc_q // ulysses_world_size, hs_q]
                    )

                context_layer_list.append(context_layer)

        # dist.barrier(group=PROCESS_GROUP.get_non_attn_pg(index_req=index_req))

        output_list = []
        for index_req, (context_layer) in enumerate(context_layer_list):
            with nvtx_range(f"Single comm_ulysses_mlp for req {index_req} in layer {index_block}"):
                bs_q, shard_seq_len_q, hc_q, hs_q = q_shape_list[index_req]
                ulysses_world_size = PROCESS_GROUP.get_ulysses_size(index_req=index_req)
                non_attn_world_size = len(PROCESS_GROUP.get_non_attn_ranks(index_req=index_req))
                bs_o, seq_len_o, shard_hc_o, hs_o = (
                    bs_q,
                    shard_seq_len_q * non_attn_world_size,
                    hc_q // ulysses_world_size,
                    hs_q,
                )

                output = comm_ulysses_mlp(block, context_layer, dtype=a2a_dtype, async_op=async_op, index_req=index_req)
                if async_op:
                    output, output_handle = output
                    output_handle.wait()

                    output = output.reshape(
                        shard_hc_o * ulysses_world_size,
                        seq_len_o // non_attn_world_size,
                        bs_o,
                        hs_o,
                    )
                    output = output.transpose(0, 2).contiguous()

                output_list.append(output)

        for index_req, (output, hidden_states, residual, gate, mlp_hidden_states, head_dim) in enumerate(
            zip(output_list, hidden_states_list, residual_list, gate_list, mlp_hidden_states_list, head_dim_list)
        ):
            with nvtx_range(f"Single comp_epilogue for req {index_req} in layer {index_block}"):
                hidden_states = unimodal_comp_epilogue(
                    block,
                    attn_output=output,
                    residual=residual,
                    gate=gate,
                    mlp_hidden_states=mlp_hidden_states,
                    head_dim=head_dim,
                    dtype=residual.dtype,
                )
                hidden_states_list[index_req] = hidden_states

    return hidden_states_list


def transformer_flux_infer_req_batch_forward(
    self: cFuserFluxTransformer2DWrapper,
    hidden_states_list: List[torch.Tensor],
    encoder_hidden_states_list: List[torch.Tensor] = None,
    pooled_projections_list: List[torch.Tensor] = None,
    timestep_list: List[torch.LongTensor] = None,
    img_ids_list: List[torch.Tensor] = None,
    txt_ids_list: List[torch.Tensor] = None,
    guidance_list: List[torch.Tensor] = None,
    joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    return_dict: bool = True,
) -> Union[torch.FloatTensor, Transformer2DModelOutput]:

    time_embd_list = [None] * len(hidden_states_list)
    image_rotary_emb_list = [None] * len(hidden_states_list)
    lora_scale_list = [None] * len(hidden_states_list)
    new_hidden_states_list = []
    new_encoder_hidden_states_list = []

    for i, (
        hidden_states,
        encoder_hidden_states,
        pooled_projections,
        timestep,
        img_ids,
        txt_ids,
        guidance,
    ) in enumerate(
        zip(
            hidden_states_list,
            encoder_hidden_states_list,
            pooled_projections_list,
            timestep_list,
            img_ids_list,
            txt_ids_list,
            guidance_list,
        )
    ):
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

        new_hidden_states_list.append(hidden_states)
        new_encoder_hidden_states_list.append(encoder_hidden_states)
        time_embd_list[i] = time_embd
        image_rotary_emb_list[i] = image_rotary_emb
        lora_scale_list[i] = lora_scale

    hidden_states_list = new_hidden_states_list
    encoder_hidden_states_list = new_encoder_hidden_states_list

    hidden_states_list, encoder_hidden_states_list = inline_multimodal_transformer_blocks_forward(
        self,
        hidden_states_list,
        encoder_hidden_states_list,
        time_embd_list,
        image_rotary_emb_list,
        lora_scale_list,
        async_op=False,
    )

    new_hidden_states_list = []
    for i, (hidden_states, encoder_hidden_states) in enumerate(zip(hidden_states_list, encoder_hidden_states_list)):
        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
        new_hidden_states_list.append(hidden_states)

    hidden_states_list = new_hidden_states_list
    hidden_states_list = inline_unimodal_transformer_blocks_forward(
        self, hidden_states_list, time_embd_list, image_rotary_emb_list, joint_attention_kwargs, async_op=False
    )

    hidden_states_list = [
        hidden_states[:, encoder_hidden_states.shape[1] :, ...]
        for hidden_states, encoder_hidden_states in zip(hidden_states_list, encoder_hidden_states_list)
    ]

    hidden_states_list = [
        self.norm_out(hidden_states, time_embd) for hidden_states, time_embd in zip(hidden_states_list, time_embd_list)
    ]

    output_list = [self.proj_out(hidden_states) for hidden_states in hidden_states_list]

    if USE_PEFT_BACKEND:
        unscale_lora_layers(self, lora_scale)

    if not return_dict:
        return output_list
    else:
        return [Transformer2DModelOutput(sample=output) for output in output_list]
