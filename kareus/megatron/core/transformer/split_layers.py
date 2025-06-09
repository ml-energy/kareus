# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

import warnings
from abc import ABC
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Union, Tuple

import torch
import torch.distributed
from torch import Tensor

from megatron.core import parallel_state, tensor_parallel
from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.dist_checkpointing.utils import apply_prefix_mapping
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.cuda_graphs import CudaGraphManager
from megatron.core.transformer.identity_op import IdentityFuncOp, IdentityOp
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import (
    get_transformer_layer_offset,
    TransformerLayerSubmodules,
    BaseTransformerLayer,
)
from megatron.core.utils import deprecate_inference_params, is_te_min_version, make_viewless_tensor


@dataclass
class AttentionLayerSubmodules:
    """Configuration class for specifying the submodules of an attention layer."""
    
    input_layernorm: Union[ModuleSpec, type] = IdentityOp
    self_attention: Union[ModuleSpec, type] = IdentityOp
    self_attn_bda: Union[ModuleSpec, type] = IdentityFuncOp
    pre_cross_attn_layernorm: Union[ModuleSpec, type] = IdentityOp
    cross_attention: Union[ModuleSpec, type] = IdentityOp
    cross_attn_bda: Union[ModuleSpec, type] = IdentityFuncOp
    pre_mlp_layernorm: Union[ModuleSpec, type] = IdentityOp
    
    # Mapping for sharded tensor keys to be applied in `sharded_state_dict` method
    sharded_state_dict_keys_map: Dict[str, str] = field(default_factory=dict)


@dataclass
class MLPLayerSubmodules:
    """Configuration class for specifying the submodules of an MLP layer."""
    
    mlp: Union[ModuleSpec, type] = IdentityOp
    mlp_bda: Union[ModuleSpec, type] = IdentityFuncOp
    
    # Mapping for sharded tensor keys to be applied in `sharded_state_dict` method
    sharded_state_dict_keys_map: Dict[str, str] = field(default_factory=dict)


class AttentionLayer(MegatronModule, BaseTransformerLayer):
    """Attention layer containing self-attention and cross-attention operations."""

    def __init__(
        self,
        config: TransformerConfig,
        submodules: AttentionLayerSubmodules,
        layer_number: int = 1,
        hidden_dropout: Optional[float] = None,
    ):
        super().__init__(config=config)

        self.submodules_config = submodules
        self.layer_number = layer_number + get_transformer_layer_offset(self.config)
        self.hidden_dropout = config.hidden_dropout if hidden_dropout is None else hidden_dropout

        # [Module 1: Input Layernorm] Optional Layernorm on the input data
        self.input_layernorm = build_module(
            submodules.input_layernorm,
            config=self.config,
            hidden_size=self.config.hidden_size,
            eps=self.config.layernorm_epsilon,
        )

        attention_optional_kwargs = {}
        if config.context_parallel_size > 1 and config.cp_comm_type is not None:
            if isinstance(config.cp_comm_type, list):
                attention_optional_kwargs["cp_comm_type"] = config.cp_comm_type[self.layer_number]
            else:
                attention_optional_kwargs["cp_comm_type"] = config.cp_comm_type

        # [Module 2: SelfAttention]
        self.self_attention = build_module(
            submodules.self_attention,
            config=self.config,
            layer_number=layer_number,
            **attention_optional_kwargs,
        )

        # [Module 3: BiasDropoutFusion]
        self.self_attn_bda = build_module(submodules.self_attn_bda)

        # [Module 4: Post SelfAttention] Optional Layernorm after self-attn
        self.pre_cross_attn_layernorm = build_module(
            submodules.pre_cross_attn_layernorm,
            config=self.config,
            hidden_size=self.config.hidden_size,
            eps=self.config.layernorm_epsilon,
        )

        # [Module 5: CrossAttention]
        self.cross_attention = build_module(
            submodules.cross_attention,
            config=self.config,
            layer_number=layer_number,
            **attention_optional_kwargs,
        )

        # [Module 6: BiasDropoutFusion]
        self.cross_attn_bda = build_module(submodules.cross_attn_bda, config=self.config)

        # [Module 7: Pre MLP] Optional Layernorm before MLP
        self.pre_mlp_layernorm = build_module(
            submodules.pre_mlp_layernorm,
            config=self.config,
            hidden_size=self.config.hidden_size,
            eps=self.config.layernorm_epsilon,
        )

        self.recompute_input_layernorm = False
        self.recompute_pre_mlp_layernorm = False
        if self.config.recompute_granularity == 'selective':
            if "layernorm" in self.config.recompute_modules:
                if not isinstance(self.input_layernorm, IdentityOp):
                    self.recompute_input_layernorm = True
                if not isinstance(self.pre_mlp_layernorm, IdentityOp):
                    self.recompute_pre_mlp_layernorm = True

        # Set bias+dropout+add fusion grad_enable execution handler.
        self.bias_dropout_add_exec_handler = torch.enable_grad

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        context: Optional[Tensor] = None,
        context_mask: Optional[Tensor] = None,
        rotary_pos_emb: Optional[Tensor] = None,
        rotary_pos_cos: Optional[Tensor] = None,
        rotary_pos_sin: Optional[Tensor] = None,
        attention_bias: Optional[Tensor] = None,
        inference_context: Optional[Any] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
        sequence_len_offset: Optional[Tensor] = None,
        *,
        inference_params: Optional[Any] = None,
    ):
        """
        Perform a forward pass through the attention layer.

        Returns:
            Tuple[Tensor, Tensor, Tensor]: A tuple containing:
                pre_mlp_layernorm_output (Tensor): Transformed hidden states before the MLP.
                residual (Tensor): Residual connection.
                context (Tensor): Updated context tensor if cross-attention is used.
        """
        inference_context = deprecate_inference_params(inference_context, inference_params)

        # Residual connection.
        residual = hidden_states

        # Optional Input Layer norm
        if self.recompute_input_layernorm:
            self.input_layernorm_checkpoint = tensor_parallel.CheckpointWithoutOutput()
            input_layernorm_output = self.input_layernorm_checkpoint.checkpoint(
                self.input_layernorm, hidden_states
            )
        else:
            input_layernorm_output = self.input_layernorm(hidden_states)

        # Self attention.
        attention_output_with_bias = self.self_attention(
            input_layernorm_output,
            attention_mask=attention_mask,
            inference_context=inference_context,
            rotary_pos_emb=rotary_pos_emb,
            rotary_pos_cos=rotary_pos_cos,
            rotary_pos_sin=rotary_pos_sin,
            attention_bias=attention_bias,
            packed_seq_params=packed_seq_params,
            sequence_len_offset=sequence_len_offset,
        )

        if self.recompute_input_layernorm:
            # discard the output of the input layernorm and register the recompute
            # as a gradient hook of attention_output_with_bias[0]
            self.input_layernorm_checkpoint.discard_output_and_register_recompute(
                attention_output_with_bias[0]
            )

        # TODO: could we move `bias_dropout_add_exec_handler` itself
        # inside the module provided in the `bias_dropout_add_spec` module?
        with self.bias_dropout_add_exec_handler():
            hidden_states = self.self_attn_bda(self.training, self.config.bias_dropout_fusion)(
                attention_output_with_bias, residual, self.hidden_dropout
            )

        # Residual connection.
        residual = hidden_states

        # Optional Layer norm after self-attention
        pre_cross_attn_layernorm_output = self.pre_cross_attn_layernorm(hidden_states)

        # Cross attention.
        attention_output_with_bias = self.cross_attention(
            pre_cross_attn_layernorm_output,
            attention_mask=context_mask,
            key_value_states=context,
            inference_context=inference_context,
        )

        if isinstance(attention_output_with_bias, dict) and "context" in attention_output_with_bias:
            context = attention_output_with_bias["context"]

        # TODO: could we move `bias_dropout_add_exec_handler` itself
        # inside the module provided in the `bias_dropout_add_spec` module?
        with self.bias_dropout_add_exec_handler():
            hidden_states = self.cross_attn_bda(self.training, self.config.bias_dropout_fusion)(
                attention_output_with_bias, residual, self.hidden_dropout
            )

        # Residual connection.
        residual = hidden_states

        # Optional Layer norm post the cross-attention.
        if self.recompute_pre_mlp_layernorm:
            self.pre_mlp_norm_checkpoint = tensor_parallel.CheckpointWithoutOutput()
            pre_mlp_layernorm_output = self.pre_mlp_norm_checkpoint.checkpoint(
                self.pre_mlp_layernorm, hidden_states
            )
        else:
            pre_mlp_layernorm_output = self.pre_mlp_layernorm(hidden_states)

        return pre_mlp_layernorm_output, residual, context

    def sharded_state_dict(
        self, prefix: str = '', sharded_offsets: tuple = (), metadata: Optional[dict] = None
    ) -> ShardedStateDict:
        """Generate a sharded state dictionary for the attention layer."""
        sharded_state_dict = super().sharded_state_dict(prefix, sharded_offsets, metadata)
        prefixed_map = {
            f'{prefix}{k}': f'{prefix}{v}'
            for k, v in self.submodules_config.sharded_state_dict_keys_map.items()
        }
        if prefixed_map:
            apply_prefix_mapping(sharded_state_dict, prefixed_map)
        return sharded_state_dict


class MLPLayer(MegatronModule, BaseTransformerLayer):
    """MLP layer containing feed-forward operations."""

    def __init__(
        self,
        config: TransformerConfig,
        submodules: MLPLayerSubmodules,
        layer_number: int = 1,
        hidden_dropout: Optional[float] = None,
    ):
        super().__init__(config=config)

        self.submodules_config = submodules
        self.layer_number = layer_number + get_transformer_layer_offset(self.config)
        self.hidden_dropout = config.hidden_dropout if hidden_dropout is None else hidden_dropout

        # [Module 8: MLP block]
        self.mlp = build_module(submodules.mlp, config=self.config)
        if hasattr(self.mlp, 'set_layer_number'):
            self.mlp.set_layer_number(self.layer_number)

        # [Module 9: BiasDropoutFusion]
        self.mlp_bda = build_module(submodules.mlp_bda)

        self.recompute_mlp = False
        if self.config.recompute_granularity == 'selective':
            if "mlp" in self.config.recompute_modules:
                from megatron.core.transformer.moe.moe_layer import MoELayer
                if not isinstance(self.mlp, MoELayer):
                    self.recompute_mlp = True

        # Set bias+dropout+add fusion grad_enable execution handler.
        self.bias_dropout_add_exec_handler = torch.enable_grad

    def forward(self, pre_mlp_layernorm_output, residual):
        """
        Perform a forward pass through the feed-forward layer.

        Args:
            pre_mlp_layernorm_output (Tensor): Transformed hidden states before the MLP.
            residual (Tensor): Residual connection.

        Returns:
            output (Tensor): Transformed hidden states of shape [s, b, h].
        """

        # MLP.
        if self.recompute_mlp:
            mlp_output_with_bias = tensor_parallel.checkpoint(
                self.mlp, False, pre_mlp_layernorm_output
            )
        else:
            mlp_output_with_bias = self.mlp(pre_mlp_layernorm_output)

        # TODO: could we move `bias_dropout_add_exec_handler` itself
        # inside the module provided in the `bias_dropout_add_spec` module?
        with self.bias_dropout_add_exec_handler():
            hidden_states = self.mlp_bda(self.training, self.config.bias_dropout_fusion)(
                mlp_output_with_bias, residual, self.hidden_dropout
            )

        # Jit compiled function creates 'view' tensor. This tensor
        # potentially gets saved in the MPU checkpoint function context,
        # which rejects view tensors. While making a viewless tensor here
        # won't result in memory savings (like the data loader, or
        # p2p_communication), it serves to document the origin of this
        # 'view' tensor.
        output = make_viewless_tensor(
            inp=hidden_states, requires_grad=hidden_states.requires_grad, keep_graph=True
        )

        return output

    def sharded_state_dict(
        self, prefix: str = '', sharded_offsets: tuple = (), metadata: Optional[dict] = None
    ) -> ShardedStateDict:
        """Generate a sharded state dictionary for the MLP layer."""
        sharded_state_dict = super().sharded_state_dict(prefix, sharded_offsets, metadata)
        prefixed_map = {
            f'{prefix}{k}': f'{prefix}{v}'
            for k, v in self.submodules_config.sharded_state_dict_keys_map.items()
        }
        if prefixed_map:
            apply_prefix_mapping(sharded_state_dict, prefixed_map)
        return sharded_state_dict


def create_attention_and_mlp_layers_from_transformer_submodules(
    transformer_submodules: TransformerLayerSubmodules,
) -> Tuple[AttentionLayerSubmodules, MLPLayerSubmodules]:
    """
    Create AttentionLayerSubmodules and MLPLayerSubmodules from TransformerLayerSubmodules.
    
    Args:
        transformer_submodules: The original transformer layer submodules
        
    Returns:
        Tuple of (AttentionLayerSubmodules, MLPLayerSubmodules)
    """
    attention_submodules = AttentionLayerSubmodules(
        input_layernorm=transformer_submodules.input_layernorm,
        self_attention=transformer_submodules.self_attention,
        self_attn_bda=transformer_submodules.self_attn_bda,
        pre_cross_attn_layernorm=transformer_submodules.pre_cross_attn_layernorm,
        cross_attention=transformer_submodules.cross_attention,
        cross_attn_bda=transformer_submodules.cross_attn_bda,
        pre_mlp_layernorm=transformer_submodules.pre_mlp_layernorm,
        sharded_state_dict_keys_map=transformer_submodules.sharded_state_dict_keys_map,
    )
    
    mlp_submodules = MLPLayerSubmodules(
        mlp=transformer_submodules.mlp,
        mlp_bda=transformer_submodules.mlp_bda,
        sharded_state_dict_keys_map=transformer_submodules.sharded_state_dict_keys_map,
    )
    
    return attention_submodules, mlp_submodules


def create_attention_and_mlp_layers_from_module_spec(
    module_spec: ModuleSpec,
) -> Tuple[AttentionLayerSubmodules, MLPLayerSubmodules]:
    """
    Create AttentionLayerSubmodules and MLPLayerSubmodules from a ModuleSpec.
    
    Args:
        module_spec: The ModuleSpec for a transformer layer
        
    Returns:
        Tuple of (AttentionLayerSubmodules, MLPLayerSubmodules)
    """
    # Extract submodules from the ModuleSpec
    if hasattr(module_spec, 'submodules') and module_spec.submodules is not None:
        transformer_submodules = module_spec.submodules
    else:
        # If no submodules specified, use default TransformerLayerSubmodules
        from megatron.core.transformer.transformer_layer import TransformerLayerSubmodules
        transformer_submodules = TransformerLayerSubmodules()
    
    return create_attention_and_mlp_layers_from_transformer_submodules(transformer_submodules) 