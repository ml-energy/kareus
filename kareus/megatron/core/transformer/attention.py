"""
Modified from Megatron-LM (megatron/core/transformer/attention.py) by NVIDIA.
Changes: QKV post-processing and rotary embedding use dedicated Op objects for
graph-based partition execution; get_compute_ops added for the partition system;
forward() not used (execution handled by TransformerBlockAutogradFunction);
cross-attention and inference code paths removed.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Union

import torch

from megatron.core import parallel_state
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import divide

from kareus.megatron.core.extensions.ops import create_qkv_postprocess_op
from kareus.megatron.core.extensions.ops import create_rotary_embedding_op


@dataclass
class SelfAttentionSubmodules:
    """
    Configuration class for specifying the submodules of a self-attention.
    """

    linear_qkv: Union[ModuleSpec, type] = None
    core_attention: Union[ModuleSpec, type] = None
    linear_proj: Union[ModuleSpec, type] = None
    q_layernorm: Union[ModuleSpec, type] = None
    k_layernorm: Union[ModuleSpec, type] = None


class Attention(MegatronModule, ABC):
    """Attention layer abstract class.

    This layer only contains common modules required for the "self attn"
    specialization. The layer does NOT execute forward passes directly --
    instead, ``TransformerBlock`` uses ``get_compute_ops()`` to collect
    operators for the graph-based partition system.
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: SelfAttentionSubmodules,
        layer_number: int,
        attn_mask_type: AttnMaskType,
        attention_type: str,
        cp_comm_type: str = None,
    ):
        super().__init__(config=config)

        self.config = config
        self.layer_number = layer_number
        self.attn_mask_type = attn_mask_type
        self.attention_type = attention_type

        # For normal attention without groups, num_query_groups == num_attention_heads,
        # so these two will be the same
        self.query_projection_size = self.config.kv_channels * self.config.num_attention_heads
        self.kv_projection_size = self.config.kv_channels * self.config.num_query_groups

        # Per attention head and per partition values.
        world_size = parallel_state.get_tensor_model_parallel_world_size()
        self.hidden_size_per_attention_head = divide(
            self.query_projection_size, self.config.num_attention_heads
        )
        self.num_attention_heads_per_partition = divide(self.config.num_attention_heads, world_size)
        self.num_query_groups_per_partition = divide(self.config.num_query_groups, world_size)

        self.rotary_embedding_op = create_rotary_embedding_op(self.config)

        self.core_attention = build_module(
            submodules.core_attention,
            config=self.config,
            layer_number=self.layer_number,
            attn_mask_type=self.attn_mask_type,
            attention_type=self.attention_type,
            cp_comm_type=cp_comm_type,
            softmax_scale=self.config.softmax_scale,
        )

        # Output.
        self.linear_proj = build_module(
            submodules.linear_proj,
            self.query_projection_size,
            self.config.hidden_size,
            config=self.config,
            init_method=self.config.output_layer_init_method,
            bias=self.config.add_bias_linear,
            input_is_parallel=True,
            skip_bias_add=True,
            is_expert=False,
            tp_comm_buffer_name='proj',
        )

    def get_compute_ops(self) -> List:
        qkv_ops = self.get_query_key_value_tensors_ops()
        return qkv_ops + [self.rotary_embedding_op, self.core_attention, self.linear_proj]

    @abstractmethod
    def get_query_key_value_tensors_ops(self) -> List:
        """Return the list of operators for QKV computation."""

    def forward(self, *args, **kwargs):
        """Not used in graph-based execution mode.

        TransformerBlock executes operators directly via
        TransformerBlockAutogradFunction using the TensorGraph.
        """
        raise NotImplementedError(
            "Attention.forward() is not supported. "
            "The graph-based partition system in TransformerBlock "
            "executes operators directly via TensorGraph."
        )


class SelfAttention(Attention):
    """Self-attention layer class

    Self-attention layer takes input with size [s, b, h]
    and returns output of the same size.
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: SelfAttentionSubmodules,
        layer_number: int,
        attn_mask_type=AttnMaskType.padding,
        cp_comm_type: str = None,
    ):
        super().__init__(
            config=config,
            submodules=submodules,
            layer_number=layer_number,
            attn_mask_type=attn_mask_type,
            attention_type="self",
            cp_comm_type=cp_comm_type,
        )

        self.linear_qkv = build_module(
            submodules.linear_qkv,
            self.config.hidden_size,
            self.query_projection_size + 2 * self.kv_projection_size,
            config=self.config,
            init_method=self.config.init_method,
            gather_output=False,
            bias=self.config.add_bias_linear or self.config.add_qkv_bias,
            skip_bias_add=False,
            is_expert=False,
            tp_comm_buffer_name='qkv',
        )

        if submodules.q_layernorm is not None:
            self.q_layernorm = build_module(
                submodules.q_layernorm,
                hidden_size=self.hidden_size_per_attention_head,
                config=self.config,
                eps=self.config.layernorm_epsilon,
            )
        else:
            self.q_layernorm = None

        if submodules.k_layernorm is not None:
            self.k_layernorm = build_module(
                submodules.k_layernorm,
                hidden_size=self.hidden_size_per_attention_head,
                config=self.config,
                eps=self.config.layernorm_epsilon,
            )
        else:
            self.k_layernorm = None

        self.qkv_postprocess_op = create_qkv_postprocess_op(
            num_query_groups_per_partition=self.num_query_groups_per_partition,
            num_attention_heads_per_partition=self.num_attention_heads_per_partition,
            hidden_size_per_attention_head=self.hidden_size_per_attention_head,
            q_layernorm=self.q_layernorm,
            k_layernorm=self.k_layernorm,
            test_mode=self.config.test_mode,
        )

    def get_query_key_value_tensors_ops(self) -> List:
        return [self.linear_qkv, self.qkv_postprocess_op]
