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

from kareus.transformer_engine.pytorch.attention.dot_product_attention.backends import (
    flash_attention_forward, 
    flash_attention_backward,
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
            raise NotImplementedError("DotProductAttentionOp does not support inference_params is not None.")
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

        ctx.qkv_format = qkv_format
        ctx.q_format = q_format
        ctx.context_parallel = context_parallel

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
        # self.logger.info("Running with FlashAttention backend")
        
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
            ctx=ctx,  # Pass context to save tensors for backward
        )

        return output

    def op_backward(
        self,
        ctx: OperationContext,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Backward pass for dot product attention using FlashAttention backward functions."""
        
        # Retrieve saved context information
        qkv_format = ctx.qkv_format
        q_format = ctx.q_format
        context_parallel = ctx.context_parallel
        
        # Call the comprehensive backward function that handles all transformations
        dq, dk, dv = flash_attention_backward(
            ctx=ctx,
            grad_output=grad_output,
            qkv_format=qkv_format,
            q_format=q_format,
            context_parallel=context_parallel,
        )
        
        return dq, (dk, dv)

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
