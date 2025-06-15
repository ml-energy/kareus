from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Union, Tuple

import torch
from torch import Tensor

from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.dist_checkpointing.utils import apply_prefix_mapping
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.identity_op import IdentityFuncOp, IdentityOp
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import (
    get_transformer_layer_offset,
    BaseTransformerLayer,
)
from megatron.core.utils import deprecate_inference_params, make_viewless_tensor

from kareus.megatron.core.extensions.transformer_engine import create_operation_fuser
from kareus.utils.debug import save_tensors

@dataclass
class AttentionLayerSubmodules:
    """Configuration class for specifying the submodules of an attention layer."""
    
    prev_mlp_bda: Union[ModuleSpec, type] = IdentityFuncOp
    input_layernorm: Union[ModuleSpec, type] = IdentityOp
    self_attention: Union[ModuleSpec, type] = IdentityOp
    # pre_cross_attn_layernorm: Union[ModuleSpec, type] = IdentityOp
    # cross_attention: Union[ModuleSpec, type] = IdentityOp
    # cross_attn_bda: Union[ModuleSpec, type] = IdentityFuncOp
    
    # Mapping for sharded tensor keys to be applied in `sharded_state_dict` method
    sharded_state_dict_keys_map: Dict[str, str] = field(default_factory=dict)


class AttentionLayer(MegatronModule, BaseTransformerLayer):
    """Attention layer containing self-attention operations."""

    def __init__(
        self,
        config: TransformerConfig,
        submodules: AttentionLayerSubmodules,
        layer_number: int = 1,
        hidden_dropout: Optional[float] = None,
    ):
        super().__init__(config=config)

        if config.enable_cuda_graph or config.external_cuda_graph:
            raise NotImplementedError("Cuda graph not implemented")

        self.submodules_config = submodules
        self.is_first_layer = layer_number == 1
        self.layer_number = layer_number + get_transformer_layer_offset(self.config)
        self.hidden_dropout = config.hidden_dropout if hidden_dropout is None else hidden_dropout

        # [Module 1: Prev MLP BDA] Optional BDA on the previous MLP output
        self.prev_mlp_bda = build_module(
            submodules.prev_mlp_bda if not self.is_first_layer else IdentityFuncOp,
            config=self.config,
            hidden_size=self.config.hidden_size,
        )

        # [Module 2: Input Layernorm] Optional Layernorm on the input data
        self.input_layernorm = build_module(
            submodules.input_layernorm,
            config=self.config,
            hidden_size=self.config.hidden_size,
            eps=self.config.layernorm_epsilon,
        )
        # self.input_layernorm = create_operation_fuser(self.input_layernorm)

        attention_optional_kwargs = {}
        if config.context_parallel_size > 1 and config.cp_comm_type is not None:
            if isinstance(config.cp_comm_type, list):
                attention_optional_kwargs["cp_comm_type"] = config.cp_comm_type[self.layer_number]
            else:
                attention_optional_kwargs["cp_comm_type"] = config.cp_comm_type
        
        # [Module 3: SelfAttention]
        self.self_attention = build_module(
            submodules.self_attention,
            config=self.config,
            layer_number=layer_number,
            **attention_optional_kwargs,
        )
    
        self.recompute_input_layernorm = False
        if self.config.recompute_granularity == 'selective':
            raise NotImplementedError("Selective recompute not implemented")
        
        # Set bias+dropout+add fusion grad_enable execution handler.
        self.bias_dropout_add_exec_handler = torch.enable_grad
        
    def forward(
        self,
        hidden_states: Union[Tensor, Tuple[Tensor, Tensor]],
        residual: Tensor = None,
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

        if not self.is_first_layer:
            with self.bias_dropout_add_exec_handler():
                hidden_states = self.prev_mlp_bda(self.training, self.config.bias_dropout_fusion)(
                    hidden_states, residual, self.hidden_dropout
                )
            # hidden_states = make_viewless_tensor(
            #     inp=hidden_states, requires_grad=hidden_states.requires_grad, keep_graph=True
            # )

        # Residual connection.
        residual = hidden_states

        # Optional Input Layer norm
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

        return attention_output_with_bias, residual, context
    
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
    