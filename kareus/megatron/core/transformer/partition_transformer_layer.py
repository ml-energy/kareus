from typing import Tuple

from megatron.core.transformer.identity_op import IdentityFuncOp, IdentityOp
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_layer import TransformerLayerSubmodules

from kareus.megatron.core.transformer.attention_layer import AttentionLayerSubmodules
from kareus.megatron.core.transformer.mlp_layer import MLPLayerSubmodules
from kareus.megatron.core.transformer.mlp_output_layer import MLPOutputLayerSubmodules


def create_attention_and_mlp_layers_from_transformer_submodules(
    transformer_submodules: TransformerLayerSubmodules,
) -> Tuple[AttentionLayerSubmodules, MLPLayerSubmodules, MLPOutputLayerSubmodules]:
    """
    Create AttentionLayerSubmodules and MLPLayerSubmodules from TransformerLayerSubmodules.
    
    Args:
        transformer_submodules: The original transformer layer submodules
        
    Returns:
        Tuple of (AttentionLayerSubmodules, MLPLayerSubmodules, MLPOutputLayerSubmodules)
    """
    # check if they ARE the Identity classes
    if (transformer_submodules.pre_cross_attn_layernorm is not IdentityOp) \
        or (transformer_submodules.cross_attention is not IdentityOp) \
        or (transformer_submodules.cross_attn_bda is not IdentityFuncOp):
        raise NotImplementedError("Cross attention not implemented")

    attention_submodules = AttentionLayerSubmodules(
        prev_mlp_bda=transformer_submodules.mlp_bda,
        input_layernorm=transformer_submodules.input_layernorm,
        self_attention=transformer_submodules.self_attention,
        sharded_state_dict_keys_map=transformer_submodules.sharded_state_dict_keys_map,
    )
        # pre_cross_attn_layernorm=transformer_submodules.pre_cross_attn_layernorm,
        # cross_attention=transformer_submodules.cross_attention,
        # cross_attn_bda=transformer_submodules.cross_attn_bda,
    
    mlp_submodules = MLPLayerSubmodules(
        prev_self_attn_bda=transformer_submodules.self_attn_bda,
        pre_mlp_layernorm=transformer_submodules.pre_mlp_layernorm,
        mlp=transformer_submodules.mlp,
        sharded_state_dict_keys_map=transformer_submodules.sharded_state_dict_keys_map,
    )
    mlp_output_submodules = MLPOutputLayerSubmodules(
        post_mlp_bda=transformer_submodules.mlp_bda,
        sharded_state_dict_keys_map=transformer_submodules.sharded_state_dict_keys_map,
    )
    
    return attention_submodules, mlp_submodules, mlp_output_submodules


def create_attention_and_mlp_layers_from_module_spec(
    module_spec: ModuleSpec,
) -> Tuple[AttentionLayerSubmodules, MLPLayerSubmodules, MLPOutputLayerSubmodules]:
    """
    Create AttentionLayerSubmodules and MLPLayerSubmodules from a ModuleSpec.
    
    Args:
        module_spec: The ModuleSpec for a transformer layer
        
    Returns:
        Tuple of (AttentionLayerSubmodules, MLPLayerSubmodules, MLPOutputLayerSubmodules)
    """
    # Extract submodules from the ModuleSpec
    if hasattr(module_spec, 'submodules') and module_spec.submodules is not None:
        transformer_submodules = module_spec.submodules
    else:
        raise ValueError("ModuleSpec does not contain submodules")
    
    return create_attention_and_mlp_layers_from_transformer_submodules(transformer_submodules) 