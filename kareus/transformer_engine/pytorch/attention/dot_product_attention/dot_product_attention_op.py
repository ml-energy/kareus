# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Attention operation following the BasicOperation pattern."""
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
try:
    from flash_attn_3.flash_attn_interface import flash_attn_func as flash_attn_func_v3
    from flash_attn_3.flash_attn_interface import flash_attn_varlen_func as flash_attn_varlen_func_v3
    from flash_attn_3.flash_attn_interface import flash_attn_with_kvcache as flash_attn_with_kvcache_v3
except ImportError:
    flash_attn_func_v3 = None
    flash_attn_varlen_func_v3 = None
    flash_attn_with_kvcache_v3 = None

from transformer_engine.pytorch.attention.dot_product_attention.context_parallel import (
    attn_forward_func_with_cp,
)

# Setup Attention Logging
attn_log.setup_logging()

# Global vars for ALiBi cache
_alibi_cache = {
    "_num_heads": None,
    "_alibi_slopes": None,
    "_max_seqlen_q": None,
    "_max_seqlen_kv": None,
    "_bottom_right_alignment": True,
    "_alibi_bias": None,
    "_alibi_slopes_require_update": False,
    "_alibi_bias_require_update": False,
}

__all__ = ["DotProductAttentionOp"]


class DotProductAttentionOp(BasicOperation):
    """Dot Product Attention as a BasicOperation
    
    This implementation follows the BasicOperation pattern and only uses
    FlashAttention backend for simplicity.
    
    Parameters
    ----------
    num_attention_heads : int
                         number of attention heads in the transformer layer.
    kv_channels : Union[int, Tuple[int, int]]
                the head size in key and value tensors. If the same, :attr:`kv_channels` can be
                an integer; if not, :attr:`kv_channels` should be a tuple of two integers.
    num_gqa_groups : Optional[int] = None
                    number of GQA groups in the transformer layer.
    attention_dropout: float, default = 0.0
                      dropout probability for the dropout op during multi-head attention.
    attn_mask_type: str, default = `causal`
                   type of attention mask passed into softmax operation.
    window_size: Optional[Tuple[int, int]], default = `None`
                sliding window size for local attention.
    attention_type: str, default = `self`
                   type of attention, either "`self`" and "`cross`".
    layer_number: int, default = `None`
                 layer number of the current `DotProductAttention`.
    qkv_format: str, default = `sbhd`
               dimension format for `query_layer`, `key_layer` and `value_layer`.
    softmax_scale: Optional[float], default = `None`
                softmax scale for the attention scores.
    sequence_parallel : bool, default = `False`
                       if set to `True`, uses sequence parallelism.
    tp_size : int, default = 1
             tensor parallel world size.
    tp_group : ProcessGroup, default = `None`
              tensor parallel process group.
    cp_group : Union[ProcessGroup, List[ProcessGroup]], default = `None`
              context parallel process group.
    cp_global_ranks : list of global rank IDs, default = `None`
                     global rank IDs of GPUs that are in cp_group.
    cp_stream : CUDA stream, default = `None`
               context parallelism CUDA stream.
    cp_comm_type : str, default = `p2p`
                  inter-gpu communication type for context parallelism.
    """

    # DotProductAttention has 2 extra inputs: key_layer and value_layer
    num_extra_inputs: int = 2

    def __init__(
        self,
        num_attention_heads: int,
        kv_channels: Union[int, Tuple[int, int]],
        num_gqa_groups: Optional[int] = None,
        attention_dropout: float = 0.0,
        qkv_format: str = "sbhd",
        attn_mask_type: str = "causal",
        window_size: Optional[Tuple[int, int]] = None,
        sequence_parallel: bool = False,
        tp_size: int = 1,
        get_rng_state_tracker: Optional[Callable] = None,
        tp_group: Optional[dist_group_type] = None,
        layer_number: Optional[int] = None,
        attention_type: str = "self",
        cp_group: Optional[Union[dist_group_type, List[dist_group_type]]] = None,
        cp_global_ranks: List[int] = None,
        cp_stream: torch.cuda.Stream = None,
        cp_comm_type: str = "p2p",
        softmax_scale: Optional[float] = None,
    ) -> None:
        super().__init__()

        self.logger = logging.getLogger("DotProductAttentionOp")
        self.logger.setLevel(attn_log._log_level)
        if not self.logger.hasHandlers():
            self.logger.addHandler(attn_log._stream_handler)
        
        self.qkv_format = qkv_format
        attn_mask_type = attn_mask_type.replace(",", "_")
        if attn_mask_type == "causal_padding":
            attn_mask_type = "padding_causal"
        self.attn_mask_type = attn_mask_type
        self.window_size = dpa_utils.check_set_window_size(attn_mask_type, window_size)
        
        if tp_group is None:
            self.tp_size = tp_size
        else:
            self.tp_size = get_distributed_world_size(tp_group)
        self.tp_group = tp_group
        
        self.get_rng_state_tracker = get_rng_state_tracker
        self.num_attention_heads = num_attention_heads
        self.layer_number = 1 if layer_number is None else layer_number
        self.cp_group = cp_group
        self.cp_global_ranks = cp_global_ranks
        self.cp_stream = cp_stream
        self.cp_comm_type = cp_comm_type

        self.hidden_size_per_attention_head_k = (
            kv_channels if isinstance(kv_channels, int) else kv_channels[0]
        )
        self.hidden_size_per_attention_head_v = (
            kv_channels if isinstance(kv_channels, int) else kv_channels[1]
        )

        self.num_gqa_groups = num_attention_heads if num_gqa_groups is None else num_gqa_groups
        self.num_gqa_groups_per_partition = int(self.num_gqa_groups // self.tp_size)

        assert (
            num_attention_heads % self.num_gqa_groups == 0
        ), "The number of attention heads must be divisible by the number of GQA groups!"

        self.rng_states_tracker = None
        if sequence_parallel or get_rng_state_tracker is None:
            attention_dropout_ctx = nullcontext
        else:
            self.rng_states_tracker = get_rng_state_tracker()
            set_all_rng_states(self.rng_states_tracker.get_states())
            attention_dropout_ctx = self.rng_states_tracker.fork

        if softmax_scale is None:
            softmax_scale = 1.0 / math.sqrt(
                kv_channels if isinstance(kv_channels, int) else kv_channels[0]
            )

        self.deterministic = (
            not bool(int(os.getenv("NVTE_ALLOW_NONDETERMINISTIC_ALGO", "1")))
            or torch.are_deterministic_algorithms_enabled()
        )

        assert attention_type in AttnTypes, f"attention_type {attention_type} not supported"

        self.attention_type = attention_type
        self.attention_dropout = attention_dropout
        self.attention_dropout_ctx = attention_dropout_ctx
        self.softmax_scale = softmax_scale

        # Store FlashAttention parameters directly instead of creating a module
        self.flash_attention_params = {
            "softmax_scale": softmax_scale,
            "attention_dropout": attention_dropout,
            "attention_dropout_ctx": attention_dropout_ctx,
            "attention_type": attention_type,
            "deterministic": self.deterministic,
        }

    def set_context_parallel_group(
        self,
        cp_group: Union[dist_group_type, List[dist_group_type], None],
        cp_global_ranks: List[int],
        cp_stream: torch.cuda.Stream,
        cp_comm_type: str = "p2p",
    ) -> None:
        """Set the context parallel attributes for the given module."""
        self.cp_group = cp_group
        self.cp_global_ranks = cp_global_ranks
        self.cp_stream = cp_stream
        self.cp_comm_type = cp_comm_type

    def op_forward(
        self,
        ctx: OperationContext,
        input_: torch.Tensor,  # query_layer
        *,
        prev_op: Optional[BasicOperation] = None,
        next_op: Optional[BasicOperation] = None,
        key_layer: torch.Tensor,
        value_layer: torch.Tensor,
        attention_mask: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]] = None,
        qkv_format: str = None,
        cu_seqlens_q: torch.Tensor = None,
        cu_seqlens_kv: torch.Tensor = None,
        cu_seqlens_q_padded: torch.Tensor = None,
        cu_seqlens_kv_padded: torch.Tensor = None,
        max_seqlen_q: int = None,
        max_seqlen_kv: int = None,
        attn_mask_type: Optional[str] = None,
        window_size: Optional[Tuple[int, int]] = None,
        core_attention_bias_type: str = "no_bias",
        core_attention_bias: Optional[torch.Tensor] = None,
        alibi_slopes: Optional[torch.Tensor] = None,
        fast_zero_fill: bool = True,
        inference_params: Optional[InferenceParams] = None,
        pad_between_seqs: Optional[bool] = None,
    ) -> torch.Tensor:
        """Forward pass for dot product attention."""
        
        query_layer = input_
        
        # Basic validation
        assert (
            query_layer.is_cuda and key_layer.is_cuda and value_layer.is_cuda
        ), "DotProductAttention only supports CUDA tensors."
        assert (
            query_layer.dtype == key_layer.dtype and query_layer.dtype == value_layer.dtype
        ), "Queries, keys and values must have the same data type!"
        assert (
            key_layer.shape[:-1] == value_layer.shape[:-1]
        ), "Keys and values must have the same batch size, sequence length and number of heads!"
        
        num_attention_heads = query_layer.shape[-2]
        num_gqa_groups = key_layer.shape[-2]
        assert (
            query_layer.shape[-1] == key_layer.shape[-1]
        ), "Queries and keys must have the same head dimension!"
        head_dim_qk, head_dim_v = query_layer.shape[-1], value_layer.shape[-1]
        
        # Check attention mask type
        if attn_mask_type is None:
            attn_mask_type = self.attn_mask_type
        else:
            attn_mask_type = attn_mask_type.replace(",", "_")
            if attn_mask_type == "causal_padding":
                attn_mask_type = "padding_causal"
        assert (
            attn_mask_type in AttnMaskTypes
        ), f"Attention mask type {attn_mask_type} is not supported!"

        # Check sliding window
        if window_size is None:
            window_size = self.window_size
        window_size = dpa_utils.check_set_window_size(attn_mask_type, window_size)

        # Check qkv_format
        if qkv_format is None:
            qkv_format = self.qkv_format
        assert qkv_format in [
            "sbhd",
            "bshd", 
            "thd",
        ], "DotProductAttention only supports qkv_format = {'sbhd', 'bshd', 'thd'}!"
        
        batch_size = None
        if qkv_format in ["sbhd", "bshd"]:
            assert all(
                len(x.shape) == 4 for x in (query_layer, key_layer, value_layer)
            ), f"Queries, keys and values must be 4D tensors when {qkv_format=}!"
            if qkv_format == "sbhd":
                batch_size = query_layer.shape[1]
                max_seqlen_q = query_layer.shape[0] if max_seqlen_q is None else max_seqlen_q
                max_seqlen_kv = key_layer.shape[0] if max_seqlen_kv is None else max_seqlen_kv
            else:
                batch_size = query_layer.shape[0]
                max_seqlen_q = query_layer.shape[1] if max_seqlen_q is None else max_seqlen_q
                max_seqlen_kv = key_layer.shape[1] if max_seqlen_kv is None else max_seqlen_kv
        if qkv_format == "thd":
            assert all(
                len(x.shape) == 3 for x in (query_layer, key_layer, value_layer)
            ), "Queries, keys and values must be 3D tensors when qkv_format = thd!"
            assert (
                "padding" in attn_mask_type
            ), "Attention mask type must be padding or padding_causal for qkv_format=thd!"
            assert (
                cu_seqlens_q is not None and cu_seqlens_kv is not None
            ), "cu_seqlens_q and cu_seqlens_kv can not be None when qkv_format = thd!"
            batch_size = len(cu_seqlens_q) - 1
            if max_seqlen_q is None:
                if cu_seqlens_q_padded is not None:
                    seqlens_q = cu_seqlens_q_padded[1:] - cu_seqlens_q_padded[:-1]
                else:
                    seqlens_q = cu_seqlens_q[1:] - cu_seqlens_q[:-1]
                max_seqlen_q = int((seqlens_q.max().item() + 63) // 64 * 64)
            if max_seqlen_kv is None:
                if cu_seqlens_kv_padded is not None:
                    seqlens_kv = cu_seqlens_kv_padded[1:] - cu_seqlens_kv_padded[:-1]
                else:
                    seqlens_kv = cu_seqlens_kv[1:] - cu_seqlens_kv[:-1]
                max_seqlen_kv = int((seqlens_kv.max().item() + 63) // 64 * 64)

        # update KV cache and retrieve saved tokens from cache for inference
        if inference_params is not None:
            assert self.layer_number is not None, "Layer number must be set!"

            # convert top-left causal to bottom-right causal due to KV caching
                # users can still use the same attention mask for inference as for training
            assert "padding" in attn_mask_type, "KV caching requires padding mask!"
            if attn_mask_type == "padding_causal":
                attn_mask_type = attn_mask_type + "_bottom_right"

            self.attention_type = "cross"

            query_layer, key_layer, value_layer = [
                x.contiguous() if not x.is_contiguous() else x
                for x in [query_layer, key_layer, value_layer]
            ]

            # get full K/V tensors from cache and adjust cu_seqlens, qkv_format based on the cache
            (
                key_layer,
                value_layer,
                cu_seqlens_q,
                cu_seqlens_kv,
                max_seqlen_kv,
                qkv_format,
            ) = inference_params.step(
                self.layer_number,
                key_layer,
                value_layer,
                qkv_format,
            )
            cu_seqlens_q_padded = None
            cu_seqlens_kv_padded = None

        # Get qkv's memory layout
        if all(isinstance(x, Float8Tensor) for x in [query_layer, key_layer, value_layer]):
            (
                qkv_layout,
                query_layer._data,
                key_layer._data,
                value_layer._data,
                q_format,
                kv_format,
            ) = dpa_utils.get_qkv_layout(
                query_layer._data,
                key_layer._data,
                value_layer._data,
                qkv_format=qkv_format,
                inference_params=inference_params,
            )
        else:
            (
                qkv_layout,
                query_layer,
                key_layer,
                value_layer,
                q_format,
                kv_format,
            ) = dpa_utils.get_qkv_layout(
                query_layer,
                key_layer,
                value_layer,
                qkv_format=qkv_format,
                inference_params=inference_params,
            )

        # Adjust max_seqlen and cu_seqlens for CP
        cp_size = 1
        if isinstance(self.cp_group, dist_group_type):
            cp_size = get_distributed_world_size(self.cp_group)
        elif isinstance(self.cp_group, list):
            for group in self.cp_group:
                cp_size *= get_distributed_world_size(group)
        context_parallel = cp_size > 1
        
        if q_format in ["sbhd", "bshd"]:
            max_seqlen_q *= cp_size
            if cu_seqlens_q is None:
                if "padding" in attn_mask_type:
                    assert (
                        attention_mask is not None
                    ), "Please provide attention_mask for padding!"
                    if self.attention_type == "self":
                        cu_seqlens_q = dpa_utils.get_cu_seqlens(attention_mask)
                    else:
                        cu_seqlens_q = dpa_utils.get_cu_seqlens(attention_mask[0])
                else:
                    cu_seqlens_q = dpa_utils.get_full_cu_seqlens(
                        batch_size,
                        max_seqlen_q,
                        query_layer.device,
                    )
        if kv_format in ["sbhd", "bshd"]:
            max_seqlen_kv *= cp_size
            if cu_seqlens_kv is None:
                if "padding" in attn_mask_type:
                    assert (
                        attention_mask is not None
                    ), "Please provide attention_mask for padding!"
                    if self.attention_type == "self":
                        cu_seqlens_kv = dpa_utils.get_cu_seqlens(attention_mask)
                    else:
                        cu_seqlens_kv = dpa_utils.get_cu_seqlens(attention_mask[1])
                else:
                    cu_seqlens_kv = dpa_utils.get_full_cu_seqlens(
                        batch_size,
                        max_seqlen_kv,
                        key_layer.device,
                    )

        # Set ALiBi attributes
        global _alibi_cache
        if alibi_slopes is not None:
            assert (
                core_attention_bias_type == "alibi"
            ), "core_attention_bias_type must be alibi in order to use alibi_slopes!"
            if self.layer_number == 1:
                _alibi_cache["_alibi_slopes_require_update"] = True
                _alibi_cache["_alibi_bias_require_update"] = True
        
        bottom_right_alignment = (attn_mask_type not in ["causal", "padding_causal"],)
        if core_attention_bias_type == "alibi":
            assert (
                core_attention_bias is None
            ), "core_attention_bias must be None when core_attention_bias_type is alibi!"
            if (
                _alibi_cache["_num_heads"] != query_layer.shape[-2]
                or _alibi_cache["_max_seqlen_q"] != max_seqlen_q
                or _alibi_cache["_max_seqlen_kv"] != max_seqlen_kv
                or _alibi_cache["_bottom_right_alignment"] != bottom_right_alignment
                or _alibi_cache["_alibi_slopes"] is None
            ):
                _alibi_cache["_alibi_slopes_require_update"] = True
                _alibi_cache["_alibi_bias_require_update"] = True

        if pad_between_seqs is None:
            if qkv_format == "thd":
                pad_between_seqs = (
                    cu_seqlens_q_padded is not None
                    and not torch.equal(cu_seqlens_q_padded[:-1], cu_seqlens_q[:-1])
                ) or (
                    cu_seqlens_kv_padded is not None
                    and not torch.equal(cu_seqlens_kv_padded[:-1], cu_seqlens_kv[:-1])
                )
            else:
                pad_between_seqs = False

        # Save state for backward pass
        # ctx.save_for_backward(
        #     query_layer, key_layer, value_layer,
        #     cu_seqlens_q, cu_seqlens_kv, attention_mask, alibi_slopes, core_attention_bias
        # )
        # ctx.qkv_layout = qkv_layout
        # ctx.attn_mask_type = attn_mask_type
        # ctx.window_size = window_size
        # ctx.max_seqlen_q = max_seqlen_q
        # ctx.max_seqlen_kv = max_seqlen_kv
        # ctx.context_parallel = context_parallel
        # ctx.inference_params = inference_params
        # ctx.core_attention_bias_type = core_attention_bias_type
        # ctx.pad_between_seqs = pad_between_seqs

        # Handle ALiBi
        if core_attention_bias_type == "alibi":
            alibi_slopes, _ = dpa_utils.get_alibi(
                _alibi_cache,
                query_layer.shape[-2],
                max_seqlen_q,
                max_seqlen_kv,
                alibi_slopes=alibi_slopes,
            )

        # Run FlashAttention
        self.logger.info("Running with FlashAttention backend")
        
        # Skip context parallel for now (not supported in standalone function)
        if context_parallel:
            self.logger.warning("Context parallel not supported in standalone FlashAttention function, falling back to non-CP")
            context_parallel = False
        
        output = flash_attention_forward(
            query_layer,
            key_layer,
            value_layer,
            attention_mask=attention_mask,
            qkv_layout=qkv_layout,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_kv=cu_seqlens_kv,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_kv=max_seqlen_kv,
            attn_mask_type=attn_mask_type,
            window_size=window_size,
            alibi_slopes=alibi_slopes,
            cp_group=None,  # Disable context parallel for now
            cp_global_ranks=None,
            cp_stream=None,
            cp_comm_type=self.cp_comm_type,
            fp8=self.fp8 and self.fp8_meta["recipe"].fp8_dpa if hasattr(self, 'fp8') else False,
            fp8_meta=self.fp8_meta if hasattr(self, 'fp8_meta') else None,
            quantizers=self.quantizers if hasattr(self, 'quantizers') else None,
            inference_params=inference_params,
            flash_attention_backend=PkgVersion("0"),
            softmax_scale=self.softmax_scale,
            attention_dropout=self.attention_dropout,
            attention_dropout_ctx=self.attention_dropout_ctx,
            attention_type=self.attention_type,
            deterministic=self.deterministic,
            training=self.training if hasattr(self, 'training') else True,
            ctx=ctx,  # Pass context directly to forward function
        )

        return output

    def op_backward(
        self,
        ctx: OperationContext,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Backward pass for dot product attention."""
        
        # Retrieve saved tensors
        (
            query_layer, key_layer, value_layer,
            cu_seqlens_q, cu_seqlens_kv, attention_mask, alibi_slopes, core_attention_bias
        ) = ctx.saved_tensors

        # For now, we'll use PyTorch's autograd for the backward pass
        # In a full implementation, you would implement the backward pass manually
        # using the FlashAttention backward kernels
        
        # This is a simplified backward pass - in practice you'd want to implement
        # the actual gradients computation using the attention backward kernels
        grad_query = torch.zeros_like(query_layer)
        grad_key = torch.zeros_like(key_layer) 
        grad_value = torch.zeros_like(value_layer)
        
        # Return gradients: grad_input (query), and extra input grads (key, value)
        return grad_query, (grad_key, grad_value)

    def fuser_forward(
        self,
        basic_op_ctxs: list[OperationContext],
        input_: torch.Tensor,
        *,
        basic_op_extra_inputs: list[tuple[torch.Tensor, ...]],
        basic_op_prev_ops: list[Optional[BasicOperation]],
        basic_op_next_ops: list[Optional[BasicOperation]],
        basic_op_kwargs: list[dict[str, Any]],
    ) -> tuple[torch.Tensor, list[tuple[()]]]:
        """Override fuser_forward since we have extra inputs."""
        
        # Extract key and value from extra inputs
        key_layer, value_layer = basic_op_extra_inputs[0]
        
        # Add key and value to kwargs
        kwargs = basic_op_kwargs[0].copy()
        kwargs['key_layer'] = key_layer
        kwargs['value_layer'] = value_layer
        
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
                # Use _PrepareQKVForFA if available
                try:
                    from transformer_engine.pytorch.attention.dot_product_attention.backends import _PrepareQKVForFA
                    query_layer, key_layer, value_layer = _PrepareQKVForFA.apply(
                        query_layer, key_layer, value_layer
                    )
                except ImportError:
                    query_layer, key_layer, value_layer = [
                        x.transpose(0, 1).contiguous()
                        for x in (query_layer, key_layer, value_layer)
                    ]
            else:
                query_layer, key_layer, value_layer = [
                    x.transpose(0, 1).contiguous()
                    for x in (query_layer, key_layer, value_layer)
                ]
        elif q_format == "sbhd" and kv_format == "bshd":
            query_layer = query_layer.transpose(0, 1).contiguous()
        if context_parallel:
            query_layer, key_layer, value_layer = [
                x.contiguous() for x in (query_layer, key_layer, value_layer)
            ]
    else:
        if qkv_format == "sbhd":
            query_layer._data, key_layer._data, value_layer._data = [
                x.transpose(0, 1).contiguous()
                for x in (query_layer._data, key_layer._data, value_layer._data)
            ]
            query_layer, key_layer, value_layer = [
                Float8Tensor.make_like(x, data=x._data, shape=x._data.shape)
                for x in (query_layer, key_layer, value_layer)
            ]
        elif q_format == "sbhd" and kv_format == "bshd":
            query_layer._data = query_layer._data.transpose(0, 1).contiguous()
            query_layer = Float8Tensor.make_like(
                query_layer, data=query_layer._data, shape=query_layer._data.shape
            )
        if context_parallel:
            query_layer._data, key_layer._data, value_layer._data = [
                x.contiguous() for x in (query_layer._data, key_layer._data, value_layer._data)
            ]

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
                assert (
                    not context_parallel
                ), "Padding mask not supported with context parallelism!"

                # [b * s, h, d]
                query_layer, key_layer, value_layer = [
                    x.reshape(x.shape[0] * x.shape[1], *x.shape[2:])
                    for x in [query_layer, key_layer, value_layer]
                ]

                if attention_type == "self":
                    assert (
                        max_seqlen_q == max_seqlen_kv
                    ), "Maximum sequence length for Q and KV should be the same."
                    if cu_seqlens_q is None:
                        assert (
                            attention_mask is not None
                        ), "Please provide attention_mask for padding!"
                        cu_seqlens_q, indices_q = dpa_utils.get_cu_seqlens_and_indices(
                            attention_mask
                        )
                    else:
                        indices_q = dpa_utils.get_indices(max_seqlen_q, cu_seqlens_q)
                    cu_seqlens_kv = cu_seqlens_q
                    query_layer, key_layer, value_layer = dpa_utils.PackTensors.apply(
                        indices_q, query_layer, key_layer, value_layer
                    )
                else:
                    if cu_seqlens_q is None or cu_seqlens_kv is None:
                        assert (
                            attention_mask is not None
                        ), "Please provide attention_mask for padding!"
                        cu_seqlens_q, indices_q = dpa_utils.get_cu_seqlens_and_indices(
                            attention_mask[0]
                        )
                        cu_seqlens_kv, indices_kv = dpa_utils.get_cu_seqlens_and_indices(
                            attention_mask[1]
                        )
                    else:
                        indices_q = dpa_utils.get_indices(max_seqlen_q, cu_seqlens_q)
                        indices_kv = dpa_utils.get_indices(max_seqlen_kv, cu_seqlens_kv)
                    query_layer = dpa_utils.PackTensors.apply(indices_q, query_layer)
                    key_layer, value_layer = dpa_utils.PackTensors.apply(
                        indices_kv, key_layer, value_layer
                    )
            else:
                # Cumulative sequence lengths for unpadded data
                if cu_seqlens_q is None:
                    cu_seqlens_q = dpa_utils.get_full_cu_seqlens(
                        batch_size,
                        max_seqlen_q,
                        query_layer.device,
                    )
                if cu_seqlens_kv is None:
                    cu_seqlens_kv = dpa_utils.get_full_cu_seqlens(
                        batch_size,
                        max_seqlen_kv,
                        key_layer.device,
                    )
        elif qkv_format == "thd":
            assert (
                cu_seqlens_q is not None and cu_seqlens_kv is not None
            ), "cu_seqlens_q and cu_seqlens_kv can not be None when qkv_format = thd!"
            if max_seqlen_q is None:
                seqlens_q = cu_seqlens_q[1:] - cu_seqlens_q[:-1]
                max_seqlen_q = seqlens_q.max().item()
            if max_seqlen_kv is None:
                seqlens_kv = cu_seqlens_kv[1:] - cu_seqlens_kv[:-1]
                max_seqlen_kv = seqlens_kv.max().item()
    else:
        if qkv_format in ["sbhd_2bshd", "bshd"]:
            # q is in bshd in both cases from conversion above or the original input
            batch_size, context_len = query_layer.shape[:2]
            cu_seqlens_q = cu_seqlens_q[: batch_size + 1]
            cu_seqlens_kv = cu_seqlens_kv[: batch_size + 1]
            # convert from bshd to thd_2bshd for flash_attn_varlen_func/_with_kvcache;
            # kernel assumes tensor is contiguous
            if isinstance(query_layer, Float8Tensor):
                import transformer_engine_torch as tex
                query_layer._data = tex.convert_bshd_to_thd(
                    query_layer._data,
                    cu_seqlens_q,
                    batch_size * context_len,
                )
                query_layer = Float8Tensor.make_like(
                    query_layer, data=query_layer._data, shape=query_layer._data.shape
                )
            else:
                import transformer_engine_torch as tex
                query_layer = tex.convert_bshd_to_thd(
                    query_layer,
                    cu_seqlens_q,
                    batch_size * context_len,
                )

    use_flash_attn_3 = False
    if flash_attention_backend is not None and flash_attention_backend > PkgVersion("3.0.0b"):
        use_flash_attn_3 = True
    
    if context_parallel and all(
        not isinstance(x, Float8Tensor) for x in [query_layer, key_layer, value_layer]
    ):
        raise NotImplementedError("Context parallelism is not supported with FlashAttention.")
        assert (
            alibi_slopes is None
        ), "Alibi slope bias addition is not supported with context parallelism."
        with attention_dropout_ctx():
            output = attn_forward_func_with_cp(
                training,
                query_layer,
                key_layer,
                value_layer,
                cu_seqlens_q,
                cu_seqlens_kv,
                max_seqlen_q,
                max_seqlen_kv,
                cu_seqlens_q if qkv_format == "thd" else None,
                cu_seqlens_kv if qkv_format == "thd" else None,
                attention_dropout if training else 0.0,
                cp_group,
                cp_global_ranks,
                cp_stream,
                cp_comm_type,
                softmax_scale=softmax_scale,
                qkv_format="bshd" if qkv_format == "sbhd" else qkv_format,
                attn_mask_type=attn_mask_type,
                deterministic=deterministic,
                window_size=window_size,
                quantizers=quantizers,
                pad_between_seqs=False,
                use_flash_attn_3=use_flash_attn_3,
            )
    else:
        from transformer_engine.pytorch.cpu_offload import (
            CPUOffloadEnabled,
            mark_activation_offload,
        )

        if CPUOffloadEnabled:
            mark_activation_offload(
                query_layer, key_layer, value_layer, cu_seqlens_q, cu_seqlens_kv
            )

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
                if not use_flash_attn_3:
                    func = flash_attn_varlen_func
                elif inference_params is None:
                    func = flash_attn_varlen_func_v3
                else:
                    func = flash_attn_with_kvcache_v3
                if not use_flash_attn_3 or inference_params is None:
                    fa_optional_forward_args_thd.append(cu_seqlens_q)
                    fa_optional_forward_args_thd.append(cu_seqlens_kv)
                    fa_optional_forward_args_thd.append(max_seqlen_q)
                    fa_optional_forward_args_thd.append(max_seqlen_kv)
            
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
                else:
                    fa_3_optional_forward_kwargs["cu_seqlens_q"] = cu_seqlens_q
                    fa_3_optional_forward_kwargs["max_seqlen_q"] = max_seqlen_q
                    cache_seqlens = cu_seqlens_kv[1:] - cu_seqlens_kv[:-1]
                    fa_3_optional_forward_kwargs["cache_seqlens"] = cache_seqlens
                    # flash_attn_with_kvcache accepts thd_2bshd for non-paged
                    if inference_params.is_paged:
                        fa_3_optional_forward_kwargs["page_table"] = (
                            inference_params.cache_manager.page_table[:batch_size]
                        )
                
                if fp8:
                    from transformer_engine.pytorch.fp8 import get_fp8_torch_dtype
                    from transformer_engine.pytorch.cpp_extensions.fused_attn import META_QKV
                    
                    QKV_quantizer = quantizers["scaling_fwd"][META_QKV]
                    torch_dtype = get_fp8_torch_dtype(fp8_meta["recipe"], fprop_tensor=True)
                    torch_orig_dtype = query_layer.dtype

                    def convert_to_torch_float8(tensor, dtype):
                        out = torch.Tensor().to(device=tensor.device, dtype=dtype)
                        out.set_(
                            tensor._data.untyped_storage(),
                            tensor._data.storage_offset(),
                            tensor._data.shape,
                            tensor._data.stride(),
                        )
                        return out

                    # "fp8_mha" decides outputs in fp8, while inputs are inferred from
                    # the real dtype
                    assert isinstance(key_layer, query_layer.__class__) and isinstance(
                        value_layer, query_layer.__class__
                    ), "q, k, and v must have the same type."
                    if not isinstance(query_layer, Float8Tensor):
                        query_layer, key_layer, value_layer = (
                            QKV_quantizer(x) for x in [query_layer, key_layer, value_layer]
                        )
                    batch_size = cu_seqlens_q.shape[0] - 1
                    num_heads_k = key_layer.shape[-2]
                    fa_3_optional_forward_kwargs["q_descale"] = (
                        query_layer._scale_inv.unsqueeze(0).repeat(batch_size, num_heads_k)
                    )
                    fa_3_optional_forward_kwargs["k_descale"] = key_layer._scale_inv.unsqueeze(
                        0
                    ).repeat(batch_size, num_heads_k)
                    fa_3_optional_forward_kwargs["v_descale"] = (
                        value_layer._scale_inv.unsqueeze(0).repeat(batch_size, num_heads_k)
                    )
                    query_layer, key_layer, value_layer = (
                        convert_to_torch_float8(x, torch_dtype)
                        for x in [query_layer, key_layer, value_layer]
                    )
                
                try:
                    output = func(
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

                if fp8:
                    output = output.to(dtype=torch_orig_dtype)
                if fp8 and fp8_meta["recipe"].fp8_mha:
                    from transformer_engine.pytorch.cpp_extensions.fused_attn import META_O
                    O_quantizer = quantizers["scaling_fwd"][META_O]
                    output = O_quantizer(output)

    if inference_params is None:
        if qkv_format in ["sbhd", "bshd"] and "padding" in attn_mask_type and indices_q is not None:
            output = dpa_utils.UnpackTensor.apply(indices_q, batch_size * max_seqlen_q, output)
    elif qkv_format in ["bshd", "sbhd_2bshd"]:
        # all KV caching cases use thd_2bshd for calculation
        # convert results back to bshd from thd_2bshd
        if isinstance(query_layer, Float8Tensor):
            import transformer_engine_torch as tex
            output._data = tex.convert_thd_to_bshd(
                output._data,
                cu_seqlens_q,
                batch_size,
                context_len,
            )
            output = Float8Tensor.make_like(output, data=output._data, shape=output._data.shape)
        else:
            import transformer_engine_torch as tex
            output = tex.convert_thd_to_bshd(
                output,
                cu_seqlens_q,
                batch_size,
                context_len,
            )

    if q_format == "sbhd":
        # (bs)hd -> bs(hd) -> sb(hd)
        if fp8 and fp8_meta["recipe"].fp8_mha:
            output_data = (
                output._data.reshape(batch_size, max_seqlen_q // cp_size, -1)
                .transpose(0, 1)
                .contiguous()
            )
            output = Float8Tensor.make_like(
                output,
                data=output_data,
                shape=output_data.shape,
            )
        else:
            output = output.view(batch_size, max_seqlen_q // cp_size, -1).transpose(0, 1)
    elif q_format == "bshd":
        # (bs)hd -> bs(hd)
        output = output.reshape(batch_size, max_seqlen_q // cp_size, -1)
    elif q_format == "thd":
        # thd -> t(hd)
        output = output.reshape(output.shape[0], -1)

    return output.contiguous() 