"""
Common configuration for fuser test scripts.

This module provides a centralized way to create TransformerConfig instances
with consistent parameters across all fuser test scripts.
"""

import torch
import torch.nn.functional as F
from megatron.core.transformer.transformer_config import TransformerConfig


class FuserTestConfig:
    """Configuration factory for fuser test scripts."""
    
    # Default model dimensions (Llama-like)
    HIDDEN_SIZE = 2048
    NUM_ATTENTION_HEADS = 32
    NUM_QUERY_GROUPS = 8  # For grouped query attention
    FFN_HIDDEN_SIZE = 8192
    VOCAB_SIZE = 128256
    DROP_RATE = 0.0

    NUM_LAYERS = 16
    
    # Default test parameters
    DEFAULT_WORLD_SIZE = 4
    DEFAULT_BATCH_SIZE = 16
    DEFAULT_SEQ_LENGTH = 4096

    DEFAULT_STAGES = 4
    DEFAULT_NUM_MICROBATCHES = 16
    num_layers_in_first_pipeline_stage = 4
    num_layers_in_last_pipeline_stage = 2
    
    # Default Bayesian Optimization parameters
    BO_DEFAULT_N_INIT = 96
    BO_DEFAULT_BATCHES = 8
    BO_DEFAULT_ACQ_BATCH = 32
    
    # GPU p2p power (W) configuration
    P2P_POWER_W_BY_GPU = {
        'A40': 90.0,
        'A100': 86.35,
    }

    @staticmethod
    def get_p2p_power(gpu_type: str) -> float:
        """Return default p2p power (W) for given GPU type, with a conservative fallback."""
        return float(FuserTestConfig.P2P_POWER_W_BY_GPU.get(gpu_type, 70.0))
    
    # Architecture presets
    ARCH_PRESETS = {
        'llama': {
            'hidden_size': 2048,
            'num_attention_heads': 32,
            'num_query_groups': 8,
            'ffn_hidden_size': 8192,
            'apply_rope_fusion': True,
            'apply_query_key_layer_scaling': False,
        },
        'gpt3': {
            'hidden_size': 2560,
            'num_attention_heads': 32,
            'num_query_groups': 32,  # no GQA
            'ffn_hidden_size': 10240,
            'apply_rope_fusion': False,
            'apply_query_key_layer_scaling': True,
        },
    }
    
    @staticmethod
    def get_arch_preset(arch: str) -> dict:
        """Return preset dictionary for a given architecture name."""
        return dict(FuserTestConfig.ARCH_PRESETS.get(arch, FuserTestConfig.ARCH_PRESETS['llama']))
    
    @staticmethod
    def create_transformer_config(
        world_size: int,
        dtype: torch.dtype = torch.bfloat16,
        arch: str | None = None,
        # Model architecture parameters
        num_layers: int = 1,
        hidden_size: int = None,
        num_attention_heads: int = None,
        num_query_groups: int = None,
        ffn_hidden_size: int = None,
        vocab_size: int = None,
        # Training parameters
        drop_rate: float = None,
        layernorm_epsilon: float = 1e-5,
        # Feature flags
        qk_layernorm: bool = False,
        apply_query_key_layer_scaling: bool = False,
        rotary_interleaved: bool = False,
        flash_decode: bool = False,
        apply_rope_fusion: bool = True,
        gated_linear_unit: bool = True,  # Set to True for MLP tests
        activation_func = F.silu,  # F.silu for MLP tests
        bias_activation_fusion: bool = True,  # True for MLP tests
        add_bias_linear: bool = False,
        # Cross entropy parameters (for postprocess tests)
        cross_entropy_loss_fusion: bool = True,
        cross_entropy_fusion_impl: str = 'te',
        use_cpu_initialization: bool = False,
        **kwargs
    ) -> TransformerConfig:
        """
        Create a TransformerConfig with common defaults for fuser tests.
        
        Args:
            world_size: Number of processes for tensor parallelism
            dtype: Model data type
            arch: Optional architecture preset name (e.g., 'llama', 'gpt3')
            **kwargs: Additional parameters to override defaults
            
        Returns:
            TransformerConfig instance
        """
        # Apply architecture presets if provided and fields are not explicitly set
        arch_preset = FuserTestConfig.get_arch_preset(arch) if arch else None

        # Use class defaults if not provided (possibly overridden by arch preset)
        if hidden_size is None:
            hidden_size = arch_preset['hidden_size'] if arch_preset else FuserTestConfig.HIDDEN_SIZE
        if num_attention_heads is None:
            num_attention_heads = arch_preset['num_attention_heads'] if arch_preset else FuserTestConfig.NUM_ATTENTION_HEADS
        if num_query_groups is None:
            num_query_groups = arch_preset['num_query_groups'] if arch_preset else FuserTestConfig.NUM_QUERY_GROUPS
        if ffn_hidden_size is None:
            ffn_hidden_size = arch_preset['ffn_hidden_size'] if arch_preset else FuserTestConfig.FFN_HIDDEN_SIZE
        if vocab_size is None:
            vocab_size = FuserTestConfig.VOCAB_SIZE
        if drop_rate is None:
            drop_rate = FuserTestConfig.DROP_RATE
        # Feature toggles via arch preset (if user hasn't explicitly set them)
        if arch_preset is not None:
            if 'apply_rope_fusion' in arch_preset and 'apply_rope_fusion' not in kwargs and apply_rope_fusion is True:
                apply_rope_fusion = arch_preset['apply_rope_fusion']
            if 'apply_query_key_layer_scaling' in arch_preset and 'apply_query_key_layer_scaling' not in kwargs and apply_query_key_layer_scaling is False:
                apply_query_key_layer_scaling = arch_preset['apply_query_key_layer_scaling']
            
        config_params = {
            'num_layers': num_layers,
            'hidden_size': hidden_size,
            'num_attention_heads': num_attention_heads,
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
            'tensor_model_parallel_size': world_size,
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
    def create_attention_config(world_size: int, dtype: torch.dtype = torch.bfloat16, arch: str | None = None, **kwargs) -> TransformerConfig:
        """Create config optimized for attention tests."""
        return FuserTestConfig.create_transformer_config(
            world_size=world_size,
            dtype=dtype,
            arch=arch,
            **kwargs
        )
    
    @staticmethod
    def create_mlp_config(world_size: int, dtype: torch.dtype = torch.bfloat16, arch: str | None = None, **kwargs) -> TransformerConfig:
        """Create config optimized for MLP tests."""
        return FuserTestConfig.create_transformer_config(
            world_size=world_size,
            dtype=dtype,
            arch=arch,
            gated_linear_unit=True,
            activation_func=F.silu,
            bias_activation_fusion=True,
            **kwargs
        )
    
    @staticmethod
    def create_postprocess_config(world_size: int, dtype: torch.dtype = torch.bfloat16, arch: str | None = None, **kwargs) -> TransformerConfig:
        """Create config optimized for postprocess tests."""
        return FuserTestConfig.create_transformer_config(
            world_size=world_size,
            dtype=dtype,
            arch=arch,
            use_cpu_initialization=True,
            **kwargs
        )
    
    @staticmethod
    def create_loss_config(world_size: int, dtype: torch.dtype = torch.bfloat16, arch: str | None = None, **kwargs) -> TransformerConfig:
        """Create config optimized for loss computation tests."""
        return FuserTestConfig.create_transformer_config(
            world_size=world_size,
            dtype=dtype,
            arch=arch,
            cross_entropy_loss_fusion=True,
            cross_entropy_fusion_impl='te',
            use_cpu_initialization=True,
            **kwargs
        )
