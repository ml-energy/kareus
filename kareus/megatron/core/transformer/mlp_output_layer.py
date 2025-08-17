from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, Union

import torch
from torch import Tensor

from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.dist_checkpointing.utils import apply_prefix_mapping
from megatron.core.transformer.identity_op import IdentityFuncOp
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import (
    BaseTransformerLayer,
    get_transformer_layer_offset,
)
from megatron.core.transformer.transformer_block import get_num_layers_to_build
from megatron.core.utils import make_viewless_tensor


@dataclass
class MLPOutputLayerSubmodules:
    """Configuration for the MLP output layer (applied after the MLP).

    This layer is responsible for the post-MLP Bias+Dropout+Add operation.
    It should be applied only on the last transformer layer; for all other
    layers it must be a no-op that forwards the tuple (output, bias).
    """

    post_mlp_bda: Union[ModuleSpec, type] = IdentityFuncOp

    # Optional mapping for sharded tensor keys in state dict
    sharded_state_dict_keys_map: Dict[str, str] = field(default_factory=dict)


class MLPOutputLayer(MegatronModule, BaseTransformerLayer):
    """A layer applied after the MLP to perform post-MLP BDA on the last layer.

    For non-last layers, this layer forwards the input tuple unchanged so that
    the next attention layer can consume it as its input and apply the
    Bias+Dropout+Add in its own `prev_mlp_bda` op.
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: MLPOutputLayerSubmodules,
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

        # Only the last layer performs post-MLP BDA; other layers must be a no-op.
        self.post_mlp_bda = build_module(
            submodules.post_mlp_bda if self.is_last_layer else IdentityFuncOp,
        )

    def forward(
        self,
        mlp_output_with_bias: Tuple[Tensor, Tensor],
        residual_1: Tensor,
        comm_output_with_bias: Optional[Tuple[Tensor, Tensor]] = None, # TODO: to be all-reduced
        residual_2: Optional[Tensor] = None,
    ):
        """Apply post-MLP BDA on the final layer; otherwise pass-through.

        Args:
            mlp_output_with_bias_1: Tuple[tensor, bias] from first nano-batch MLP.
            residual_1: Residual for first nano-batch.
            mlp_output_with_bias_2: Optional second nano-batch tuple.
            residual_2: Optional residual for second nano-batch.

        Returns:
            When last layer and both halves provided: final hidden states tensor and concatenated residual.
            When last layer and single input provided: final hidden states tensor and residual_1.
            Otherwise: the original tuple (tensor, bias) and residual_1.
        """
        if self.is_last_layer:
            hidden_states_1 = self.post_mlp_bda(
                mlp_output_with_bias[0], mlp_output_with_bias[1], residual_1,
                training=self.training, dropout_prob=self.hidden_dropout,
            )
            hidden_states_2 = self.post_mlp_bda(
                comm_output_with_bias[0], comm_output_with_bias[1], residual_2,
                training=self.training, dropout_prob=self.hidden_dropout,
            )
            output_1 = make_viewless_tensor(
                inp=hidden_states_1, requires_grad=hidden_states_1.requires_grad, keep_graph=True
            )
            output_2 = make_viewless_tensor(
                inp=hidden_states_2, requires_grad=hidden_states_2.requires_grad, keep_graph=True
            )
            output = torch.cat([output_1, output_2], dim=1)
            return output
        else:
            raise ValueError("MLPOutputLayer is not the last layer")

    def sharded_state_dict(
        self, prefix: str = '', sharded_offsets: tuple = (), metadata: Optional[dict] = None
    ) -> ShardedStateDict:
        sharded_state_dict = super().sharded_state_dict(prefix, sharded_offsets, metadata)
        prefixed_map = {
            f'{prefix}{k}': f'{prefix}{v}'
            for k, v in self.submodules_config.sharded_state_dict_keys_map.items()
        }
        if prefixed_map:
            apply_prefix_mapping(sharded_state_dict, prefixed_map)
        return sharded_state_dict


