
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union, Tuple

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
    TransformerLayerSubmodules,
)
from megatron.core.transformer.transformer_block import get_num_layers_to_build
from megatron.core.utils import make_viewless_tensor

from kareus.megatron.core.extensions.ops.residual_fork import ResidualForkOp


class TransformerLayer(MegatronModule, BaseTransformerLayer):
    """A single transformer layer.

    Contains both self-attention and MLP submodules. The layer does NOT
    execute forward passes directly -- instead, ``TransformerBlock`` uses
    ``get_all_operators()`` to collect operators for the graph-based
    partition system, which handles tensor routing, communication overlap,
    and nanobatch interleaving automatically.

    Submodule structure (same as Megatron's TransformerLayerSubmodules):
        self_attn_bda -> attn_residual_fork -> input_layernorm
            -> self_attention (qkv, qkv_post, rotary, core_attn, proj)
        mlp_bda -> mlp_residual_fork -> pre_mlp_layernorm
            -> mlp (fc1, activation, fc2)
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: TransformerLayerSubmodules,
        layer_number: int = 1,
        hidden_dropout: Optional[float] = None,
    ):
        super().__init__(config=config)

        if config.enable_cuda_graph or config.external_cuda_graph:
            raise NotImplementedError(
                "CUDA graph not implemented for Kareus TransformerLayer"
            )

        if (
            submodules.pre_cross_attn_layernorm is not IdentityOp
            or submodules.cross_attention is not IdentityOp
            or submodules.cross_attn_bda is not IdentityFuncOp
        ):
            raise NotImplementedError(
                "Cross attention is not supported in Kareus TransformerLayer"
            )

        self.submodules_config = submodules
        self.layer_number = layer_number + get_transformer_layer_offset(self.config)
        self.hidden_dropout = (
            config.hidden_dropout if hidden_dropout is None else hidden_dropout
        )

        num_layers = get_num_layers_to_build(config)
        self.is_first_layer = layer_number == 1
        self.is_last_layer = layer_number == num_layers

        # =================================================================
        # Attention submodules
        # =================================================================

        self.attn_residual_fork = ResidualForkOp()

        # [Module 1: Input Layernorm]
        self.input_layernorm = build_module(
            submodules.input_layernorm,
            config=self.config,
            hidden_size=self.config.hidden_size,
            eps=self.config.layernorm_epsilon,
        )

        attention_optional_kwargs = {}
        if config.context_parallel_size > 1 and config.cp_comm_type is not None:
            if isinstance(config.cp_comm_type, list):
                attention_optional_kwargs["cp_comm_type"] = config.cp_comm_type[
                    self.layer_number
                ]
            else:
                attention_optional_kwargs["cp_comm_type"] = config.cp_comm_type

        # [Module 2: Self Attention]
        self.self_attention = build_module(
            submodules.self_attention,
            config=self.config,
            layer_number=layer_number,
            **attention_optional_kwargs,
        )

        # [Module 3: Self-Attention BiasDropoutAdd]
        self.self_attn_bda = build_module(
            submodules.self_attn_bda,
            has_bias=config.add_bias_linear,
            dropout_prob=self.hidden_dropout,
        )

        # =================================================================
        # MLP submodules
        # =================================================================

        self.mlp_residual_fork = ResidualForkOp()

        # [Module 4: Pre-MLP Layernorm]
        self.pre_mlp_layernorm = build_module(
            submodules.pre_mlp_layernorm,
            config=self.config,
            hidden_size=self.config.hidden_size,
            eps=self.config.layernorm_epsilon,
        )

        # [Module 5: MLP]
        self.mlp = build_module(submodules.mlp, config=self.config)
        if hasattr(self.mlp, "set_layer_number"):
            self.mlp.set_layer_number(self.layer_number)

        # [Module 6: MLP BiasDropoutAdd]
        self.mlp_bda = build_module(
            submodules.mlp_bda,
            has_bias=config.add_bias_linear,
            dropout_prob=self.hidden_dropout,
        )

        # Recompute flags
        self.recompute_input_layernorm = False
        self.recompute_pre_mlp_layernorm = False
        self.recompute_mlp = False
        if self.config.recompute_granularity == "selective":
            raise NotImplementedError(
                "Selective recompute not implemented for Kareus TransformerLayer"
            )

    # =================================================================
    # Operator access (for TensorGraph / partition building)
    # =================================================================

    def get_all_operators(self) -> List:
        """Return all operators in forward execution order.

        Order:
          attn_residual_fork, input_ln,
          qkv, qkv_post, rotary, core_attn, proj, [AR],
          self_attn_bda,
          mlp_residual_fork, pre_mlp_ln,
          fc1, activation, fc2, [AR],
          mlp_bda

        ResidualForkOp sits before each LayerNorm so that:
          Forward:  fork x into (x_main->LN, x_copy->residual for later BDA)
          Backward: accumulate grad_main + grad_residual at the fork point
        """
        ops: List = []
        # Attention layer
        ops.append(self.attn_residual_fork)
        ops.append(self.input_layernorm)
        ops.extend(self.self_attention.get_compute_ops())
        ops.append(self.self_attn_bda)
        # MLP layer
        ops.append(self.mlp_residual_fork)
        ops.append(self.pre_mlp_layernorm)
        ops.extend(self.mlp.get_compute_ops())
        ops.append(self.mlp_bda)
        return ops

    # =================================================================
    # Forward (not used — block-level graph execution handles this)
    # =================================================================

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
    ):
        """Not used in graph-based execution mode.

        TransformerBlock executes operators directly via
        TransformerBlockAutogradFunction using the TensorGraph.
        """
        raise NotImplementedError(
            "TransformerLayer.forward() is not supported. "
            "The graph-based partition system in TransformerBlock "
            "executes operators directly via TensorGraph."
        )

    # =================================================================
    # State dict
    # =================================================================

    def sharded_state_dict(
        self,
        prefix: str = "",
        sharded_offsets: tuple = (),
        metadata: Optional[dict] = None,
    ) -> ShardedStateDict:
        """Generate a sharded state dictionary for this transformer layer."""
        sharded_state_dict = super().sharded_state_dict(
            prefix, sharded_offsets, metadata
        )
        prefixed_map = {
            f"{prefix}{k}": f"{prefix}{v}"
            for k, v in self.submodules_config.sharded_state_dict_keys_map.items()
        }
        if prefixed_map:
            apply_prefix_mapping(sharded_state_dict, prefixed_map)
        return sharded_state_dict
