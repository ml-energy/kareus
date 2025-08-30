from contextlib import nullcontext
import math
import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import warnings
import logging
from packaging.version import Version as PkgVersion

import torch

# import transformer_engine_torch as tex
from transformer_engine.pytorch.utils import get_cudnn_version
from transformer_engine.pytorch.fp8 import get_fp8_te_dtype
from transformer_engine.pytorch.float8_tensor import Float8Tensor
from transformer_engine.pytorch.ops.op import BasicOperation, OperationContext
from transformer_engine.pytorch.constants import (
    AttnMaskTypes,
    AttnTypes,
    QKVLayouts,
    dist_group_type,
)
from transformer_engine.pytorch.distributed import (
    get_distributed_world_size,
    checkpoint,
    set_all_rng_states,
    CudaRNGStatesTracker,
    graph_safe_rng_available,
)
from transformer_engine.pytorch.jit import no_torch_dynamo
from transformer_engine.pytorch.graph import is_graph_capturing
from transformer_engine.pytorch.attention.inference import InferenceParams

# Import attention utils
import transformer_engine.pytorch.attention.dot_product_attention.utils as dpa_utils
from transformer_engine.pytorch.attention.dot_product_attention.utils import (
    AttentionLogging as attn_log,
    FlashAttentionUtils as fa_utils,
)

# Import FlashAttention functions directly
from kareus.flash_attn.flash_attn_interface import flash_attn_func, flash_attn_varlen_func
from kareus.flash_attn.flash_attn_interface import flash_attn_func_backward, flash_attn_varlen_func_backward
try:
    from kareus.flash_attn.hopper.flash_attn_interface import flash_attn_func as flash_attn_func_v3
    from kareus.flash_attn.hopper.flash_attn_interface import flash_attn_func_backward as flash_attn_func_backward_v3
    # from kareus.flash_attn.hopper.flash_attn_interface import flash_attn_varlen_func as flash_attn_varlen_func_v3
    # from kareus.flash_attn.hopper.flash_attn_interface import flash_attn_with_kvcache as flash_attn_with_kvcache_v3
except ImportError:
    flash_attn_func_v3 = None
    flash_attn_func_backward_v3 = None
    # flash_attn_varlen_func_v3 = None
    # flash_attn_with_kvcache_v3 = None

from transformer_engine.pytorch.attention.dot_product_attention.context_parallel import (
    attn_forward_func_with_cp,
)


def flash_attention_forward(
    query_layer: torch.Tensor,
    key_layer: torch.Tensor,
    value_layer: torch.Tensor,
    attention_mask: Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]] = None,
    qkv_layout: str = "sbh3d",
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_kv: Optional[torch.Tensor] = None,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_kv: Optional[int] = None,
    attn_mask_type: str = "causal",
    window_size: Optional[Tuple[int, int]] = None,
    alibi_slopes: Optional[torch.Tensor] = None,
    cp_group: Optional[Union[dist_group_type, List[dist_group_type]]] = None,
    cp_global_ranks: List[int] = None,
    cp_stream: torch.cuda.Stream = None,
    cp_comm_type: str = "p2p",
    fp8: bool = False,
    fp8_meta: Optional[Dict[str, Any]] = None,
    quantizers=None,
    inference_params: Optional[InferenceParams] = None,
    flash_attention_backend: Optional[PkgVersion] = PkgVersion("0"),
    softmax_scale: float = 1.0,
    attention_dropout: float = 0.0,
    attention_dropout_ctx: Optional[Callable] = nullcontext,
    attention_type: str = "self",
    deterministic: bool = False,
    training: bool = True,
    ctx: Optional[OperationContext] = None,
) -> torch.Tensor:
    """Standalone FlashAttention forward function extracted from FlashAttention module.
    
    This function contains the core FlashAttention logic without the torch.nn.Module wrapper.
    """
    
    assert all(
        x.dtype in [torch.float16, torch.bfloat16] or isinstance(x, Float8Tensor)
        for x in [query_layer, key_layer, value_layer]
    ), "FlashAttention only supports FP16 and BF16 data types, or Float8Tensors."
    assert (
        query_layer.is_cuda and key_layer.is_cuda and value_layer.is_cuda
    ), "FlashAttention currently only supports CUDA tensors."
    assert (
        qkv_layout in QKVLayouts
    ), f"FlashAttention does not support qkv_layout = {qkv_layout}!"

    cp_size = 1
    if isinstance(cp_group, dist_group_type):
        cp_size = get_distributed_world_size(cp_group)
    elif isinstance(cp_group, list):
        for group in cp_group:
            cp_size *= get_distributed_world_size(group)
    context_parallel = cp_size > 1

    # get q_format and kv_format for training and inference
    qkv_format, q_format, kv_format = dpa_utils.get_qkv_format(qkv_layout, inference_params)

    # convert q, k, v to bshd if they are in sbhd; qkv_format doesn't change
    if all(not isinstance(x, Float8Tensor) for x in [query_layer, key_layer, value_layer]):
        if qkv_format == "sbhd":
            # For now just 128, will make it more general in the future
            if (
                query_layer.shape[-1] == 128
                and query_layer.shape[0] * query_layer.shape[1] >= 512
                and qkv_layout == "sbh3d"
            ):  
                raise NotImplementedError("FlashAttention does not support \
                    query_layer.shape[-1] == 128 and query_layer.shape[0] * query_layer.shape[1] >= 512 and qkv_layout == 'sbh3d'.")
                # # Use _PrepareQKVForFA if available
                # try:
                #     from transformer_engine.pytorch.attention.dot_product_attention.backends import _PrepareQKVForFA
                #     query_layer, key_layer, value_layer = _PrepareQKVForFA.apply(
                #         query_layer, key_layer, value_layer
                #     )
                # except ImportError:
                #     query_layer, key_layer, value_layer = [
                #         x.transpose(0, 1).contiguous()
                #         for x in (query_layer, key_layer, value_layer)
                #     ]
            else:
                query_layer, key_layer, value_layer = [
                    x.transpose(0, 1).contiguous()
                    for x in (query_layer, key_layer, value_layer)
                ]
        else:
            raise NotImplementedError(f"qkv_format: {qkv_format} is not supported in FlashAttention.")
        # elif q_format == "sbhd" and kv_format == "bshd":
        #     query_layer = query_layer.transpose(0, 1).contiguous()
        # if context_parallel:
        #     query_layer, key_layer, value_layer = [
        #         x.contiguous() for x in (query_layer, key_layer, value_layer)
        #     ]
    else:
        raise NotImplementedError("Float8Tensors are not supported in FlashAttention.")
        # if qkv_format == "sbhd":
        #     query_layer._data, key_layer._data, value_layer._data = [
        #         x.transpose(0, 1).contiguous()
        #         for x in (query_layer._data, key_layer._data, value_layer._data)
        #     ]
        #     query_layer, key_layer, value_layer = [
        #         Float8Tensor.make_like(x, data=x._data, shape=x._data.shape)
        #         for x in (query_layer, key_layer, value_layer)
        #     ]
        # elif q_format == "sbhd" and kv_format == "bshd":
        #     query_layer._data = query_layer._data.transpose(0, 1).contiguous()
        #     query_layer = Float8Tensor.make_like(
        #         query_layer, data=query_layer._data, shape=query_layer._data.shape
        #     )
        # if context_parallel:
        #     query_layer._data, key_layer._data, value_layer._data = [
        #         x.contiguous() for x in (query_layer._data, key_layer._data, value_layer._data)
        #     ]

    # get batch_size, max_seqlen and cu_seqlens
    batch_size, context_len = None, None
    indices_q = None  # Initialize indices_q
    if inference_params is None:
        if qkv_format in ["sbhd", "bshd"]:
            batch_size = query_layer.shape[0]
            max_seqlen_q, max_seqlen_kv = query_layer.shape[1], key_layer.shape[1]
            max_seqlen_q *= cp_size
            max_seqlen_kv *= cp_size

            if "padding" in attn_mask_type:
                raise NotImplementedError("Padding mask not supported with context parallelism!")
                # assert (
                #     not context_parallel
                # ), "Padding mask not supported with context parallelism!"

                # # [b * s, h, d]
                # query_layer, key_layer, value_layer = [
                #     x.reshape(x.shape[0] * x.shape[1], *x.shape[2:])
                #     for x in [query_layer, key_layer, value_layer]
                # ]

                # if attention_type == "self":
                #     assert (
                #         max_seqlen_q == max_seqlen_kv
                #     ), "Maximum sequence length for Q and KV should be the same."
                #     if cu_seqlens_q is None:
                #         assert (
                #             attention_mask is not None
                #         ), "Please provide attention_mask for padding!"
                #         cu_seqlens_q, indices_q = dpa_utils.get_cu_seqlens_and_indices(
                #             attention_mask
                #         )
                #     else:
                #         indices_q = dpa_utils.get_indices(max_seqlen_q, cu_seqlens_q)
                #     cu_seqlens_kv = cu_seqlens_q
                #     query_layer, key_layer, value_layer = dpa_utils.PackTensors.apply(
                #         indices_q, query_layer, key_layer, value_layer
                #     )
                # else:
                #     if cu_seqlens_q is None or cu_seqlens_kv is None:
                #         assert (
                #             attention_mask is not None
                #         ), "Please provide attention_mask for padding!"
                #         cu_seqlens_q, indices_q = dpa_utils.get_cu_seqlens_and_indices(
                #             attention_mask[0]
                #         )
                #         cu_seqlens_kv, indices_kv = dpa_utils.get_cu_seqlens_and_indices(
                #             attention_mask[1]
                #         )
                #     else:
                #         indices_q = dpa_utils.get_indices(max_seqlen_q, cu_seqlens_q)
                #         indices_kv = dpa_utils.get_indices(max_seqlen_kv, cu_seqlens_kv)
                #     query_layer = dpa_utils.PackTensors.apply(indices_q, query_layer)
                #     key_layer, value_layer = dpa_utils.PackTensors.apply(
                #         indices_kv, key_layer, value_layer
                #     )
            # else:
            #     # Cumulative sequence lengths for unpadded data
            #     if cu_seqlens_q is None:
            #         cu_seqlens_q = dpa_utils.get_full_cu_seqlens(
            #             batch_size,
            #             max_seqlen_q,
            #             query_layer.device,
            #         )
            #     if cu_seqlens_kv is None:
            #         cu_seqlens_kv = dpa_utils.get_full_cu_seqlens(
            #             batch_size,
            #             max_seqlen_kv,
            #             key_layer.device,
            #         )
        elif qkv_format == "thd":
            raise NotImplementedError("FlashAttention does not support qkv_format = thd.")
            # assert (
            #     cu_seqlens_q is not None and cu_seqlens_kv is not None
            # ), "cu_seqlens_q and cu_seqlens_kv can not be None when qkv_format = thd!"
            # if max_seqlen_q is None:
            #     seqlens_q = cu_seqlens_q[1:] - cu_seqlens_q[:-1]
            #     max_seqlen_q = seqlens_q.max().item()
            # if max_seqlen_kv is None:
            #     seqlens_kv = cu_seqlens_kv[1:] - cu_seqlens_kv[:-1]
            #     max_seqlen_kv = seqlens_kv.max().item()
    else:
        raise NotImplementedError("FlashAttention does not support inference_params is not None.")
        # if qkv_format in ["sbhd_2bshd", "bshd"]:
        #     # q is in bshd in both cases from conversion above or the original input
        #     batch_size, context_len = query_layer.shape[:2]
        #     cu_seqlens_q = cu_seqlens_q[: batch_size + 1]
        #     cu_seqlens_kv = cu_seqlens_kv[: batch_size + 1]
        #     # convert from bshd to thd_2bshd for flash_attn_varlen_func/_with_kvcache;
        #     # kernel assumes tensor is contiguous
        #     if isinstance(query_layer, Float8Tensor):
        #         import transformer_engine_torch as tex
        #         query_layer._data = tex.convert_bshd_to_thd(
        #             query_layer._data,
        #             cu_seqlens_q,
        #             batch_size * context_len,
        #         )
        #         query_layer = Float8Tensor.make_like(
        #             query_layer, data=query_layer._data, shape=query_layer._data.shape
        #         )
        #     else:
        #         import transformer_engine_torch as tex
        #         query_layer = tex.convert_bshd_to_thd(
        #             query_layer,
        #             cu_seqlens_q,
        #             batch_size * context_len,
        #         )

    use_flash_attn_3 = False
    if flash_attention_backend is not None and flash_attention_backend > PkgVersion("3.0.0b"):
        use_flash_attn_3 = True
    elif flash_attention_backend is None and flash_attn_func_v3 is not None:
        # Auto-detect Flash Attention 3 when backend is None
        use_flash_attn_3 = True
    
    if context_parallel and all(
        not isinstance(x, Float8Tensor) for x in [query_layer, key_layer, value_layer]
    ):
        raise NotImplementedError("Context parallelism is not supported with FlashAttention.")
        # assert (
        #     alibi_slopes is None
        # ), "Alibi slope bias addition is not supported with context parallelism."
        # with attention_dropout_ctx():
        #     output = attn_forward_func_with_cp(
        #         training,
        #         query_layer,
        #         key_layer,
        #         value_layer,
        #         cu_seqlens_q,
        #         cu_seqlens_kv,
        #         max_seqlen_q,
        #         max_seqlen_kv,
        #         cu_seqlens_q if qkv_format == "thd" else None,
        #         cu_seqlens_kv if qkv_format == "thd" else None,
        #         attention_dropout if training else 0.0,
        #         cp_group,
        #         cp_global_ranks,
        #         cp_stream,
        #         cp_comm_type,
        #         softmax_scale=softmax_scale,
        #         qkv_format="bshd" if qkv_format == "sbhd" else qkv_format,
        #         attn_mask_type=attn_mask_type,
        #         deterministic=deterministic,
        #         window_size=window_size,
        #         quantizers=quantizers,
        #         pad_between_seqs=False,
        #         use_flash_attn_3=use_flash_attn_3,
        #     )
    else:
        from transformer_engine.pytorch.cpu_offload import (
            CPUOffloadEnabled,
            mark_activation_offload,
        )

        if CPUOffloadEnabled:
            raise NotImplementedError("FlashAttention does not support CPUOffload.")
            # mark_activation_offload(
            #     query_layer, key_layer, value_layer, cu_seqlens_q, cu_seqlens_kv
            # )

        with attention_dropout_ctx():
            #       | API                     | use cases
            # ----------------------------------------------------------------------
            # FA v2 | flash_attn_func         | bshd/sbhd + not padding
            #       | flash_attn_varlen_func  | bshd/sbhd + padding
            #       |                         | thd + padding
            #       |                         | KV cache (not-paged/paged), i.e.
            #       |                         |     bshd/sbhd/thd + padding
            # FA v3 | flash_attn_func         | bshd/sbhd + not padding
            #       | flash_attn_varlen_func  | bshd/sbhd + padding
            #       |                         | thd + padding
            #       | flash_attn_with_kvcache | KV cache (not-paged/paged), i.e.
            #       |                         |     bshd/sbhd/thd + padding
            fa_optional_forward_args_thd = []
            if qkv_format in ["bshd", "sbhd"] and "padding" not in attn_mask_type:
                func = (
                    flash_attn_func if not use_flash_attn_3 else flash_attn_func_v3
                )
            else:
                raise NotImplementedError("FlashAttention does not support flash_attn_varlen_func.")
                # if not use_flash_attn_3:
                #     func = flash_attn_varlen_func
                # elif inference_params is None:
                #     func = flash_attn_varlen_func_v3
                # else:
                #     func = flash_attn_with_kvcache_v3
                # if not use_flash_attn_3 or inference_params is None:
                #     fa_optional_forward_args_thd.append(cu_seqlens_q)
                #     fa_optional_forward_args_thd.append(cu_seqlens_kv)
                #     fa_optional_forward_args_thd.append(max_seqlen_q)
                #     fa_optional_forward_args_thd.append(max_seqlen_kv)
            
            if not use_flash_attn_3:
                fa_optional_forward_kwargs = {}
                if fa_utils.v2_3_plus:
                    fa_optional_forward_kwargs["window_size"] = window_size
                if fa_utils.v2_4_plus:
                    fa_optional_forward_kwargs["alibi_slopes"] = alibi_slopes
                if fa_utils.v2_4_1_plus:
                    fa_optional_forward_kwargs["deterministic"] = deterministic
                if inference_params is not None:
                    # use block_table kwarg to support thd_2bshd for non-paged
                    fa_optional_forward_kwargs["block_table"] = (
                        inference_params.cache_manager.page_table[:batch_size]
                        if inference_params.is_paged
                        else inference_params.cache_manager.batch_indices_post_step.unsqueeze(
                            1
                        )[:batch_size]
                    )
                output = func(
                    ctx,
                    query_layer,
                    key_layer,
                    value_layer,
                    *fa_optional_forward_args_thd,
                    attention_dropout if training else 0.0,
                    softmax_scale=softmax_scale,
                    causal="causal" in attn_mask_type,
                    **fa_optional_forward_kwargs,
                )
            else:
                fa_3_optional_forward_kwargs = {}
                fa_3_optional_forward_kwargs["window_size"] = window_size
                if inference_params is None:
                    fa_3_optional_forward_kwargs["deterministic"] = deterministic
                # else:
                #     fa_3_optional_forward_kwargs["cu_seqlens_q"] = cu_seqlens_q
                #     fa_3_optional_forward_kwargs["max_seqlen_q"] = max_seqlen_q
                #     cache_seqlens = cu_seqlens_kv[1:] - cu_seqlens_kv[:-1]
                #     fa_3_optional_forward_kwargs["cache_seqlens"] = cache_seqlens
                #     # flash_attn_with_kvcache accepts thd_2bshd for non-paged
                #     if inference_params.is_paged:
                #         fa_3_optional_forward_kwargs["page_table"] = (
                #             inference_params.cache_manager.page_table[:batch_size]
                #         )
                
                # if fp8:
                #     from transformer_engine.pytorch.fp8 import get_fp8_torch_dtype
                #     from transformer_engine.pytorch.cpp_extensions.fused_attn import META_QKV
                    
                #     QKV_quantizer = quantizers["scaling_fwd"][META_QKV]
                #     torch_dtype = get_fp8_torch_dtype(fp8_meta["recipe"], fprop_tensor=True)
                #     torch_orig_dtype = query_layer.dtype

                #     def convert_to_torch_float8(tensor, dtype):
                #         out = torch.Tensor().to(device=tensor.device, dtype=dtype)
                #         out.set_(
                #             tensor._data.untyped_storage(),
                #             tensor._data.storage_offset(),
                #             tensor._data.shape,
                #             tensor._data.stride(),
                #         )
                #         return out

                #     # "fp8_mha" decides outputs in fp8, while inputs are inferred from
                #     # the real dtype
                #     assert isinstance(key_layer, query_layer.__class__) and isinstance(
                #         value_layer, query_layer.__class__
                #     ), "q, k, and v must have the same type."
                #     if not isinstance(query_layer, Float8Tensor):
                #         query_layer, key_layer, value_layer = (
                #             QKV_quantizer(x) for x in [query_layer, key_layer, value_layer]
                #         )
                #     batch_size = cu_seqlens_q.shape[0] - 1
                #     num_heads_k = key_layer.shape[-2]
                #     fa_3_optional_forward_kwargs["q_descale"] = (
                #         query_layer._scale_inv.unsqueeze(0).repeat(batch_size, num_heads_k)
                #     )
                #     fa_3_optional_forward_kwargs["k_descale"] = key_layer._scale_inv.unsqueeze(
                #         0
                #     ).repeat(batch_size, num_heads_k)
                #     fa_3_optional_forward_kwargs["v_descale"] = (
                #         value_layer._scale_inv.unsqueeze(0).repeat(batch_size, num_heads_k)
                #     )
                #     query_layer, key_layer, value_layer = (
                #         convert_to_torch_float8(x, torch_dtype)
                #         for x in [query_layer, key_layer, value_layer]
                #     )
                
                try:
                    output = func(
                        ctx,
                        query_layer,
                        key_layer,
                        value_layer,
                        *fa_optional_forward_args_thd,
                        softmax_scale=softmax_scale,
                        causal="causal" in attn_mask_type,
                        **fa_3_optional_forward_kwargs,
                    )
                    if isinstance(output, (List, Tuple)):
                        output = output[0]
                except TypeError as e:
                    if fa_utils.v3_0_0_beta:
                        e.args = (
                            e.args[0]
                            + ". Please update your flash-attn v3 (beta) installation as it "
                            + "may have added more supported arguments to its API. \n"
                            + fa_utils.v3_installation_steps,
                        ) + e.args[1:]
                    raise

                # if fp8:
                #     output = output.to(dtype=torch_orig_dtype)
                # if fp8 and fp8_meta["recipe"].fp8_mha:
                #     from transformer_engine.pytorch.cpp_extensions.fused_attn import META_O
                #     O_quantizer = quantizers["scaling_fwd"][META_O]
                #     output = O_quantizer(output)

    # if inference_params is None:
    #     if qkv_format in ["sbhd", "bshd"] and "padding" in attn_mask_type and indices_q is not None:
    #         output = dpa_utils.UnpackTensor.apply(indices_q, batch_size * max_seqlen_q, output)
    # elif qkv_format in ["bshd", "sbhd_2bshd"]:
    #     # all KV caching cases use thd_2bshd for calculation
    #     # convert results back to bshd from thd_2bshd
    #     if isinstance(query_layer, Float8Tensor):
    #         import transformer_engine_torch as tex
    #         output._data = tex.convert_thd_to_bshd(
    #             output._data,
    #             cu_seqlens_q,
    #             batch_size,
    #             context_len,
    #         )
    #         output = Float8Tensor.make_like(output, data=output._data, shape=output._data.shape)
    #     else:
    #         import transformer_engine_torch as tex
    #         output = tex.convert_thd_to_bshd(
    #             output,
    #             cu_seqlens_q,
    #             batch_size,
    #             context_len,
    #         )

    if q_format == "sbhd":
        # (bs)hd -> bs(hd) -> sb(hd)
        if fp8 and fp8_meta["recipe"].fp8_mha:
            raise NotImplementedError("FlashAttention does not support fp8.")
            # output_data = (
            #     output._data.reshape(batch_size, max_seqlen_q // cp_size, -1)
            #     .transpose(0, 1)
            #     .contiguous()
            # )
            # output = Float8Tensor.make_like(
            #     output,
            #     data=output_data,
            #     shape=output_data.shape,
            # )
        else:
            ctx.original_output_shape = output.shape
            output = output.view(batch_size, max_seqlen_q // cp_size, -1).transpose(0, 1)
    else:
        raise NotImplementedError("FlashAttention does not support q_format != sbhd.")
    # elif q_format == "bshd":
    #     # (bs)hd -> bs(hd)
    #     output = output.reshape(batch_size, max_seqlen_q // cp_size, -1)
    # elif q_format == "thd":
    #     # thd -> t(hd)
    #     output = output.reshape(output.shape[0], -1)

    return output.contiguous() 

def flash_attention_backward(
    ctx: OperationContext,
    grad_output: torch.Tensor,
    qkv_format: str,
    q_format: str,
    context_parallel: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Backward pass for FlashAttention that reverses all forward transformations."""

    # Step 1: Reverse the final output format transformations
    if q_format == "sbhd":
        grad_output_transformed = grad_output.transpose(0, 1).contiguous()
        original_shape = ctx.original_output_shape
        grad_output_transformed = grad_output_transformed.view(original_shape)
    else:
        raise NotImplementedError("FlashAttention only supports q_format == 'sbhd'.")

    # Step 2: Call the appropriate FlashAttention backward function
    if context_parallel:
        raise NotImplementedError("Context parallelism backward is not supported.")
    else:
        # Determine if FlashAttention 3 is available and should be used
        use_flash_attn_3 = False
        if flash_attn_func_v3 is not None:
            use_flash_attn_3 = True
        
        if use_flash_attn_3 and flash_attn_func_backward_v3 is not None:
            dq, dk, dv, _, _, _, _, _, _, _, _ = flash_attn_func_backward_v3(ctx, grad_output_transformed)
        else:
            dq, dk, dv, _, _, _, _, _, _, _, _ = flash_attn_func_backward(ctx, grad_output_transformed)

    # Step 3: Reverse the input tensor format transformations
    if qkv_format == "sbhd":
        dq, dk, dv = [x.transpose(0, 1).contiguous() for x in (dq, dk, dv)]
    else:
        raise NotImplementedError("Simplified FlashAttention only supports qkv_format == 'sbhd'.")

    return dq, dk, dv 