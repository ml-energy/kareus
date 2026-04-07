"""
Modified from Megatron-LM (megatron/core/models/gpt/gpt_layer_specs.py).
Changes: replaces upstream TE modules (TEColumnParallelLinear, TERowParallelLinear,
TEDotProductAttention, TENorm) with BasicOperation-based + PartitionableOperator
variants (TEColumnParallelLinearOp, TERowParallelLinearOp, TEDotProductAttentionOp,
TENormOp), and swaps SelfAttention/MLP/bias-dropout-add with Kareus versions that
wire through the partition-aware fuser_forward/fuser_backward path.
"""

import warnings
from typing import Optional

from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_layer import (
    TransformerLayer,
    TransformerLayerSubmodules,
)

from kareus.megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
from kareus.megatron.core.transformer.mlp import MLP, MLPSubmodules
from kareus.megatron.core.extensions.ops import (
    TEColumnParallelLinearOp,
    TERowParallelLinearOp,
)
from kareus.megatron.core.extensions.ops import TEDotProductAttentionOp
from kareus.megatron.core.extensions.ops import TENormOp
from kareus.megatron.core.extensions.ops import BiasDropoutAddOp


def get_gpt_layer_with_transformer_engine_spec(
    num_experts: Optional[int] = None,
    moe_grouped_gemm: Optional[bool] = False,
    qk_layernorm: Optional[bool] = False,
    multi_latent_attention: Optional[bool] = False,
    fp8: Optional[str] = None,  # pylint: disable=unused-arguments
    moe_use_legacy_grouped_gemm: Optional[bool] = False,
    qk_l2_norm: Optional[bool] = False,
) -> ModuleSpec:
    """Use this spec to use lower-level Transformer Engine modules (required for fp8 training).


    Args:
        num_experts (int, optional): Number of experts. Defaults to None.
        moe_grouped_gemm (bool, optional): To use Grouped GEMM. Defaults to False.
        qk_layernorm (bool, optional): To use layernorm for queries/keys. Defaults to False.
        fp8 (str, optional): Deprecated. For temporary Nemo compatibility.
        moe_use_legacy_grouped_gemm (bool, optional): Force use the legacy GroupedMLP.
                                                      Defaults to False.
        qk_l2_norm (bool, optional): To use l2 norm for queries/keys. Defaults to False.

    Returns:
        ModuleSpec: Module specification with TE modules
    """
    if fp8 is not None:
        warnings.warn(
            'The fp8 argument in "get_gpt_layer_with_transformer_engine_spec" has been deprecated'
            ' and will be removed soon. Please update your code accordingly.'
        )

    mlp = get_mlp_module_spec(
        use_te=True,
        num_experts=num_experts,
        moe_grouped_gemm=moe_grouped_gemm,
        moe_use_legacy_grouped_gemm=moe_use_legacy_grouped_gemm,
    )

    if qk_l2_norm:
        raise NotImplementedError("qk_l2_norm is not supported")

    if multi_latent_attention:
        raise NotImplementedError("MLA is not supported.")
        # assert qk_l2_norm is False, "qk_l2_norm is not supported with MLA."
        # return ModuleSpec(
        #     module=TransformerLayer,
        #     submodules=TransformerLayerSubmodules(
        #         input_layernorm=TENorm,
        #         self_attention=ModuleSpec(
        #             module=MLASelfAttention,
        #             params={"attn_mask_type": AttnMaskType.causal},
        #             submodules=MLASelfAttentionSubmodules(
        #                 linear_q_proj=TEColumnParallelLinear,
        #                 linear_q_down_proj=TEColumnParallelLinear,
        #                 linear_q_up_proj=(
        #                     TELayerNormColumnParallelLinear
        #                     if qk_layernorm
        #                     else TEColumnParallelLinear
        #                 ),
        #                 linear_kv_down_proj=TEColumnParallelLinear,
        #                 linear_kv_up_proj=(
        #                     TELayerNormColumnParallelLinear
        #                     if qk_layernorm
        #                     else TEColumnParallelLinear
        #                 ),
        #                 core_attention=TEDotProductAttention,
        #                 linear_proj=TERowParallelLinear,
        #                 q_layernorm=IdentityOp,
        #                 kv_layernorm=IdentityOp,
        #             ),
        #         ),
        #         self_attn_bda=get_bias_dropout_add,
        #         pre_mlp_layernorm=TENorm if num_experts else IdentityOp,
        #         mlp=mlp,
        #         mlp_bda=get_bias_dropout_add,
        #     ),
        # )
    else:
        return ModuleSpec(
            module=TransformerLayer,
            submodules=TransformerLayerSubmodules(
                input_layernorm=TENormOp,
                self_attention=ModuleSpec(
                    module=SelfAttention,
                    params={"attn_mask_type": AttnMaskType.causal},
                    submodules=SelfAttentionSubmodules(
                        linear_qkv=TEColumnParallelLinearOp,
                        core_attention=TEDotProductAttentionOp,
                        linear_proj=TERowParallelLinearOp,
                        q_layernorm=TENormOp if qk_layernorm else None,
                        k_layernorm=TENormOp if qk_layernorm else None,
                    ),
                ),
                self_attn_bda=BiasDropoutAddOp,
                pre_mlp_layernorm=TENormOp,
                mlp=mlp,
                mlp_bda=BiasDropoutAddOp,
            ),
        )


def get_mlp_module_spec(
    use_te: Optional[bool] = True,
    num_experts: Optional[int] = None,
    moe_grouped_gemm: Optional[bool] = False,
    fp8: Optional[str] = None,  # pylint: disable=unused-arguments
    moe_use_legacy_grouped_gemm: Optional[bool] = False,
) -> ModuleSpec:
    """Helper function to get module spec for MLP/MoE"""
    if fp8 is not None:
        warnings.warn(
            'The fp8 argument in "_get_mlp_module_spec" has been deprecated'
            ' and will be removed soon. Please update your code accordingly.'
        )

    if num_experts is None:
        # Dense MLP w/ or w/o TE modules.
        return ModuleSpec(
            module=MLP,
            submodules=MLPSubmodules(
                linear_fc1=TEColumnParallelLinearOp,
                linear_fc2=TERowParallelLinearOp
            ),
        )
    else:
        raise NotImplementedError("MoE is not supported")
        # # Mixture of experts with modules in megatron core.
        # return get_moe_module_spec(
        #     use_te=use_te,
        #     num_experts=num_experts,
        #     moe_grouped_gemm=moe_grouped_gemm,
        #     moe_use_legacy_grouped_gemm=moe_use_legacy_grouped_gemm,
        # )
