"""
Common configuration for fuser test scripts (Llama 3.1 70B-style config).

This module provides a centralized way to create TransformerConfig instances
with consistent parameters across all fuser test scripts.
"""

import torch
import torch.nn.functional as F
from megatron.core.transformer.transformer_config import TransformerConfig


class FuserTestConfig:
    """Configuration factory for fuser test scripts."""

    MODEL_NAME = "llama3.3_70b"
    
    # Model dimensions from Llama 3.3 70B config
    HIDDEN_SIZE = 8192
    NUM_ATTENTION_HEADS = 64
    HEAD_DIM = 128  # hidden_size / num_attention_heads
    NUM_QUERY_GROUPS = 8  # num_key_value_heads (GQA)
    FFN_HIDDEN_SIZE = 28672  # intermediate_size
    VOCAB_SIZE = 128256
    DROP_RATE = 0.0  # attention_dropout from config
    NUM_LAYERS = 80  # num_hidden_layers
    
    # Default test parameters
    DEFAULT_TENSOR_PARALLEL_SIZE = 8
    DEFAULT_CONTEXT_PARALLEL_SIZE = 1
    DEFAULT_BATCH_SIZE = 2
    DEFAULT_SEQ_LENGTH = 4096

    DEFAULT_STAGES = 2
    DEFAULT_NUM_MICROBATCHES = 8
    num_layers_in_first_pipeline_stage = 40
    num_layers_in_last_pipeline_stage = 40
    
    # GPU p2p power (W) configuration
    P2P_POWER_W_BY_GPU = {
        'A40': 70.0,
        'A100': 85.0,
    }

    GPU_TYPE = "A100"

    @staticmethod
    def get_p2p_power(gpu_type: str) -> float:
        """Return default p2p power (W) for given GPU type, with a conservative fallback."""
        return float(FuserTestConfig.P2P_POWER_W_BY_GPU.get(gpu_type, 70.0))
    
    @staticmethod
    def get_comm_sm_values() -> list[int]:
        """Return candidate SM counts for communication kernels."""
        return list(FuserTestConfig.COMM_SM_VALUES)
    
    @staticmethod
    def create_transformer_config(
        context_parallel_size: int = None,
        tensor_parallel_size: int = None,
        dtype: torch.dtype = torch.bfloat16,
        # Model architecture parameters
        num_layers: int = 1,
        hidden_size: int = None,
        num_attention_heads: int = None,
        head_dim: int = None,
        num_query_groups: int = None,
        ffn_hidden_size: int = None,
        vocab_size: int = None,
        # Training parameters
        drop_rate: float = None,
        layernorm_epsilon: float = 1e-5,  # rms_norm_eps from config
        # Feature flags
        qk_layernorm: bool = False,
        apply_query_key_layer_scaling: bool = False,
        rotary_interleaved: bool = False,
        flash_decode: bool = False,
        apply_rope_fusion: bool = True,
        gated_linear_unit: bool = True,  # Set to True for MLP tests
        activation_func = F.silu,  # F.silu for MLP tests (hidden_act: silu)
        bias_activation_fusion: bool = True,  # True for MLP tests
        add_bias_linear: bool = False,  # attention_bias and mlp_bias are false
        # Cross entropy parameters (for postprocess tests)
        cross_entropy_loss_fusion: bool = True,
        cross_entropy_fusion_impl: str = 'te',
        use_cpu_initialization: bool = False,
        **kwargs
    ) -> TransformerConfig:
        """
        Create a TransformerConfig with common defaults for fuser tests.
        
        Args:
            context_parallel_size: Context parallel size
            tensor_parallel_size: Tensor parallel size
            dtype: Model data type
            **kwargs: Additional parameters to override defaults
            
        Returns:
            TransformerConfig instance
        """
        # Use class defaults if not provided
        if context_parallel_size is None:
            context_parallel_size = FuserTestConfig.DEFAULT_CONTEXT_PARALLEL_SIZE
        if tensor_parallel_size is None:
            tensor_parallel_size = FuserTestConfig.DEFAULT_TENSOR_PARALLEL_SIZE
        if hidden_size is None:
            hidden_size = FuserTestConfig.HIDDEN_SIZE
        if num_attention_heads is None:
            num_attention_heads = FuserTestConfig.NUM_ATTENTION_HEADS
        if head_dim is None:
            head_dim = FuserTestConfig.HEAD_DIM
        if num_query_groups is None:
            num_query_groups = FuserTestConfig.NUM_QUERY_GROUPS
        if ffn_hidden_size is None:
            ffn_hidden_size = FuserTestConfig.FFN_HIDDEN_SIZE
        if vocab_size is None:
            vocab_size = FuserTestConfig.VOCAB_SIZE
        if drop_rate is None:
            drop_rate = FuserTestConfig.DROP_RATE
            
        config_params = {
            'num_layers': num_layers,
            'hidden_size': hidden_size,
            'num_attention_heads': num_attention_heads,
            'kv_channels': head_dim,
            'num_query_groups': num_query_groups,
            'ffn_hidden_size': ffn_hidden_size,
            'layernorm_epsilon': layernorm_epsilon,
            'hidden_dropout': drop_rate,
            'attention_dropout': drop_rate,
            'qk_layernorm': qk_layernorm,
            'apply_query_key_layer_scaling': apply_query_key_layer_scaling,
            'rotary_interleaved': rotary_interleaved,
            'flash_decode': flash_decode,
            'apply_rope_fusion': apply_rope_fusion,
            'params_dtype': dtype,
            'tensor_model_parallel_size': tensor_parallel_size,
            'context_parallel_size': context_parallel_size,
            'add_bias_linear': add_bias_linear,
        }
        
        # Add MLP-specific parameters if enabled
        if gated_linear_unit:
            config_params.update({
                'gated_linear_unit': gated_linear_unit,
                'activation_func': activation_func or F.silu,
                'bias_activation_fusion': bias_activation_fusion,
            })
            
        # Add cross entropy parameters if enabled
        if cross_entropy_loss_fusion or use_cpu_initialization:
            config_params.update({
                'cross_entropy_loss_fusion': cross_entropy_loss_fusion,
                'cross_entropy_fusion_impl': cross_entropy_fusion_impl,
                'use_cpu_initialization': use_cpu_initialization,
            })
            
        # Override with any additional kwargs
        config_params.update(kwargs)
        
        return TransformerConfig(**config_params)
    
    @staticmethod
    def create_attention_config(context_parallel_size: int = None, tensor_parallel_size: int = None, dtype: torch.dtype = torch.bfloat16, **kwargs) -> TransformerConfig:
        """Create config optimized for attention tests."""
        return FuserTestConfig.create_transformer_config(
            context_parallel_size=context_parallel_size,
            tensor_parallel_size=tensor_parallel_size,
            dtype=dtype,
            **kwargs
        )
    
    @staticmethod
    def create_mlp_config(context_parallel_size: int = None, tensor_parallel_size: int = None, dtype: torch.dtype = torch.bfloat16, **kwargs) -> TransformerConfig:
        """Create config optimized for MLP tests."""
        return FuserTestConfig.create_transformer_config(
            context_parallel_size=context_parallel_size,
            tensor_parallel_size=tensor_parallel_size,
            dtype=dtype,
            gated_linear_unit=True,
            activation_func=F.silu,
            bias_activation_fusion=True,
            **kwargs
        )
    
    @staticmethod
    def create_postprocess_config(context_parallel_size: int = None, tensor_parallel_size: int = None, dtype: torch.dtype = torch.bfloat16, **kwargs) -> TransformerConfig:
        """Create config optimized for postprocess tests."""
        return FuserTestConfig.create_transformer_config(
            context_parallel_size=context_parallel_size,
            tensor_parallel_size=tensor_parallel_size,
            dtype=dtype,
            use_cpu_initialization=True,
            **kwargs
        )
    
    @staticmethod
    def create_loss_config(context_parallel_size: int = None, tensor_parallel_size: int = None, dtype: torch.dtype = torch.bfloat16, **kwargs) -> TransformerConfig:
        """Create config optimized for loss computation tests."""
        return FuserTestConfig.create_transformer_config(
            context_parallel_size=context_parallel_size,
            tensor_parallel_size=tensor_parallel_size,
            dtype=dtype,
            cross_entropy_loss_fusion=True,
            cross_entropy_fusion_impl='te',
            use_cpu_initialization=True,
            **kwargs
        )