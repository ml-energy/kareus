from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Union

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
from megatron.core.transformer.transformer_block import get_num_layers_to_build
from megatron.core.utils import deprecate_inference_params, make_viewless_tensor


@dataclass
class MLPLayerSubmodules:
    """Configuration class for specifying the submodules of an MLP layer."""
    
    prev_self_attn_bda: Union[ModuleSpec, type] = IdentityFuncOp
    pre_mlp_layernorm: Union[ModuleSpec, type] = IdentityOp
    mlp: Union[ModuleSpec, type] = IdentityOp
    post_mlp_bda: Union[ModuleSpec, type] = IdentityFuncOp
    
    # Mapping for sharded tensor keys to be applied in `sharded_state_dict` method
    sharded_state_dict_keys_map: Dict[str, str] = field(default_factory=dict)


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

        if config.enable_cuda_graph or config.external_cuda_graph:
            raise NotImplementedError("Cuda graph not implemented")

        self.submodules_config = submodules
        self.layer_number = layer_number + get_transformer_layer_offset(self.config)
        self.hidden_dropout = config.hidden_dropout if hidden_dropout is None else hidden_dropout

        num_layers = get_num_layers_to_build(config)
        self.is_last_layer = layer_number == num_layers
        
        # [Module 1: Prev attention BDA] Optional BDA on the previous attention output
        self.prev_self_attn_bda = build_module(submodules.prev_self_attn_bda)

        # [Module 2: Pre MLP Layernorm] Optional Layernorm before MLP
        self.pre_mlp_layernorm = build_module(
            submodules.pre_mlp_layernorm,
            config=self.config,
            hidden_size=self.config.hidden_size,
            eps=self.config.layernorm_epsilon,
        )

        # [Module 3: MLP]
        self.mlp = build_module(submodules.mlp, config=self.config)
        if hasattr(self.mlp, 'set_layer_number'):
            self.mlp.set_layer_number(self.layer_number)

        # [Module 4: Post MLP BDA] Optional BDA on the MLP output
        
        self.post_mlp_bda = build_module(
            submodules.post_mlp_bda if self.is_last_layer else IdentityFuncOp,
        )

        self.recompute_pre_mlp_layernorm = False
        self.recompute_mlp = False
        if self.config.recompute_granularity == 'selective':
            raise NotImplementedError("Selective recompute not implemented")

        # Set bias+dropout+add fusion grad_enable execution handler.
        # Note: BiasDropoutAddOp now handles torch.enable_grad() internally
        # self.bias_dropout_add_exec_handler = torch.enable_grad
        
    def forward(self, hidden_states, residual):
        """
        Perform a forward pass through the feed-forward layer.

        Args:
            hidden_states (Tensor): Transformed hidden states before the MLP.
            residual (Tensor): Residual connection.

        Returns:
            output (Tensor): Transformed hidden states of shape [s, b, h].
        """
        
        hidden_states = self.prev_self_attn_bda(hidden_states[0], hidden_states[1], residual,
                                            training=self.training, dropout_prob=self.hidden_dropout)
        
        # Residual connection.
        residual = hidden_states
        
        # Optional Pre MLP Layer norm
        input_layernorm_output = self.pre_mlp_layernorm(hidden_states)

        # MLP.
        mlp_output_with_bias = self.mlp(input_layernorm_output)
        
        if self.is_last_layer:
            hidden_states = self.post_mlp_bda(mlp_output_with_bias[0], mlp_output_with_bias[1], residual,
                                            training=self.training, dropout_prob=self.hidden_dropout)
            # Jit compiled function creates 'view' tensor. This tensor
            # potentially gets saved in the MPU checkpoint function context,
            # which rejects view tensors. While making a viewless tensor here
            # won't result in memory savings (like the data loader, or
            # p2p_communication), it serves to document the origin of this
            # 'view' tensor.
            output = make_viewless_tensor(
                inp=hidden_states, requires_grad=hidden_states.requires_grad, keep_graph=True
            )
            return output, residual
        else:
            return mlp_output_with_bias, residual