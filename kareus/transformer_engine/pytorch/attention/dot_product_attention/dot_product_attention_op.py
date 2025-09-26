from __future__ import annotations
from typing import Any, Callable, List, Optional, Tuple, Union

import torch

from transformer_engine.pytorch.ops.op import FusedOperation
from transformer_engine.pytorch.constants import dist_group_type

from kareus.transformer_engine.pytorch.attention.dot_product_attention.basic import (
    BasicDotProductAttention,
)


class DotProductAttentionOp(FusedOperation):
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
        cp_size: int = 1,
        cp_group: Optional[Union[dist_group_type, List[dist_group_type]]] = None,
        cp_global_ranks: List[int] = None,
        cp_stream: torch.cuda.Stream = None,
        cp_comm_type: str = "p2p",
        softmax_scale: Optional[float] = None,
    ) -> None:
        # if cp_size > 1:
        #     pass
        # else:
        basic_op = BasicDotProductAttention(
            num_attention_heads=num_attention_heads,
            kv_channels=kv_channels,
            num_gqa_groups=num_gqa_groups,
            attention_dropout=attention_dropout,
            qkv_format=qkv_format,
            attn_mask_type=attn_mask_type,
            window_size=window_size,
            sequence_parallel=sequence_parallel,
            tp_size=tp_size,
            get_rng_state_tracker=get_rng_state_tracker,
            tp_group=tp_group,
            layer_number=layer_number,
            attention_type=attention_type,
            cp_group=cp_group,
            cp_global_ranks=cp_global_ranks,
            cp_stream=cp_stream,
            cp_comm_type=cp_comm_type,
            softmax_scale=softmax_scale,
        )
        super().__init__([basic_op])
