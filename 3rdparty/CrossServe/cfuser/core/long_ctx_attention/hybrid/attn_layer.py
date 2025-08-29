from typing import Any
import torch
from torch import Tensor

import torch.distributed
from cfuser.core.long_ctx_attention.comm import all_to_all_4D

from cfuser.core.distributed.globals import PROCESS_GROUP

from cfuser.logger import init_logger
from ..ring import ring_flash_attn_func

logger = init_logger(__name__)


class cFuserLongContextAttention(torch.nn.Module):
    def __init__(
        self,
        scatter_idx: int = 2,
        gather_idx: int = 1,
        use_pack_qkv: bool = False,
    ) -> None:
        super().__init__()
        self.ring_pg = PROCESS_GROUP.get_ring_pg(PROCESS_GROUP.index_req)
        self.ulysses_pg = PROCESS_GROUP.get_ulysses_pg(PROCESS_GROUP.index_req)

        self.use_pack_qkv = use_pack_qkv
        self.use_sync = False

        assert (
            self.ulysses_pg is not None or self.ring_pg is not None
        ), f"use set_seq_parallel_pg() first. Now ulysses pg {self.ulysses_pg} and ring pg {self.ring_pg}"

        self.scatter_idx = scatter_idx
        self.gather_idx = gather_idx

        self.ring_attn_fn = ring_flash_attn_func

    def renew_process_group(self):
        self.ring_pg = PROCESS_GROUP.get_ring_pg(PROCESS_GROUP.index_req)
        self.ulysses_pg = PROCESS_GROUP.get_ulysses_pg(PROCESS_GROUP.index_req)

    @torch.compiler.disable
    def forward(
        self,
        attn,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        *,
        joint_tensor_query=None,
        joint_tensor_key=None,
        joint_tensor_value=None,
        dropout_p=0.0,
        softmax_scale=None,
        causal=False,
        window_size=(-1, -1),
        alibi_slopes=None,
        deterministic=False,
        return_attn_probs=False,
        joint_strategy="none",
    ) -> Tensor:
        """forward

        Arguments:
            query (Tensor): query input to the layer
            key (Tensor): key input to the layer
            value (Tensor): value input to the layer
            args: other args

        Returns:
            * output (Tensor): context output
        """

        self.renew_process_group()

        # 3 X (bs, seq_len/N, head_cnt, head_size) -> 3 X (bs, seq_len, head_cnt/N, head_size)
        # scatter 2, gather 1
        if self.use_pack_qkv:
            # (3*bs, seq_len/N, head_cnt, head_size)
            qkv = torch.cat([query, key, value]).continous()
            # (3*bs, seq_len, head_cnt/N, head_size)
            qkv = all_to_all_4D(qkv, self.scatter_idx, self.gather_idx, group=self.ulysses_pg)
            qkv = torch.chunk(qkv, 3, dim=0)
            query_layer, key_layer, value_layer = qkv

        else:
            query_layer = all_to_all_4D(query, self.scatter_idx, self.gather_idx, group=self.ulysses_pg)
            key_layer = all_to_all_4D(key, self.scatter_idx, self.gather_idx, group=self.ulysses_pg)
            value_layer = all_to_all_4D(value, self.scatter_idx, self.gather_idx, group=self.ulysses_pg)

        out = self.ring_attn_fn(
            query_layer,
            key_layer,
            value_layer,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            alibi_slopes=alibi_slopes,
            deterministic=deterministic,
            return_attn_probs=return_attn_probs,
            group=self.ring_pg,
            attn_layer=None,
        )

        if type(out) == tuple:
            context_layer, _, _ = out
        else:
            context_layer = out

        # (bs, seq_len, head_cnt/N, head_size) -> (bs, seq_len/N, head_cnt, head_size)
        # scatter 1, gather 2
        output = all_to_all_4D(context_layer, self.gather_idx, self.scatter_idx, group=self.ulysses_pg)

        # out e.g., [s/p::h]
        return output


class cFuserFluxLongContextAttention(cFuserLongContextAttention):
    @torch.compiler.disable
    def forward(
        self,
        attn,
        query: Tensor,
        key: Tensor,
        value: Tensor,
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
    ) -> Tensor:
        """forward

        Arguments:
            query (Tensor): query input to the layer
            key (Tensor): key input to the layer
            value (Tensor): value input to the layer
            args: other args

        Returns:
            * output (Tensor): context output
        """

        self.renew_process_group()
        # 3 X (bs, seq_len/N, head_cnt, head_size) -> 3 X (bs, seq_len, head_cnt/N, head_size)
        # scatter 2, gather 1
        query = torch.cat([joint_tensor_query, query], dim=1)
        ulysses_world_size = torch.distributed.get_world_size(self.ulysses_pg)
        ulysses_rank = torch.distributed.get_rank(self.ulysses_pg)
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

        if self.use_pack_qkv:
            # (3*bs, seq_len/N, head_cnt, head_size)
            qkv = torch.cat([query, key, value]).continous()
            # (3*bs, seq_len, head_cnt/N, head_size)
            qkv = all_to_all_4D(qkv, self.scatter_idx, self.gather_idx, group=self.ulysses_pg)
            qkv = torch.chunk(qkv, 3, dim=0)
            query_layer, key_layer, value_layer = qkv

        else:
            query_layer = all_to_all_4D(query, self.scatter_idx, self.gather_idx, group=self.ulysses_pg)
            key_layer = all_to_all_4D(key, self.scatter_idx, self.gather_idx, group=self.ulysses_pg)
            value_layer = all_to_all_4D(value, self.scatter_idx, self.gather_idx, group=self.ulysses_pg)

        out = self.ring_attn_fn(
            query_layer,
            key_layer,
            value_layer,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            alibi_slopes=alibi_slopes,
            deterministic=deterministic,
            return_attn_probs=return_attn_probs,
            group=self.ring_pg,
            attn_layer=None,
            joint_tensor_key=joint_tensor_key,
            joint_tensor_value=joint_tensor_value,
            joint_strategy=joint_strategy,
        )

        if type(out) == tuple:
            context_layer, _, _ = out
        else:
            context_layer = out

        # (bs, seq_len, head_cnt/N, head_size) -> (bs, seq_len/N, head_cnt, head_size)
        # scatter 1, gather 2
        output = all_to_all_4D(context_layer, self.gather_idx, self.scatter_idx, group=self.ulysses_pg)

        # out e.g., [s/p::h]
        return output
