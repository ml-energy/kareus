"""Model architecture registry, GPU constants, and test-parameter defaults."""

from __future__ import annotations

import dataclasses
from typing import Dict

import torch
import torch.nn.functional as F
from megatron.core.transformer.transformer_config import TransformerConfig


@dataclasses.dataclass(frozen=True)
class ModelConfig:
    """Immutable model architecture definition.

    Test parameters (tp, cp, batch_size, seq_len) are intentionally excluded;
    they are passed at runtime via CLI arguments.
    """
    name: str
    hidden_size: int
    num_attention_heads: int
    head_dim: int
    num_query_groups: int
    ffn_hidden_size: int
    num_layers: int
    vocab_size: int = 128256
    drop_rate: float = 0.0
    layernorm_epsilon: float = 1e-5
    qk_layernorm: bool = False
    add_bias_linear: bool = False

    def create_transformer_config(
        self,
        tensor_parallel_size: int = 1,
        context_parallel_size: int = 1,
        dtype: torch.dtype = torch.bfloat16,
        partition_type: str = "attention",
        **kwargs,
    ) -> TransformerConfig:
        """Build a Megatron TransformerConfig from this model + parallelism params.

        ``partition_type`` selects sensible defaults:
          * ``"attention"`` — base config (no MLP-specific flags)
          * ``"mlp"`` — additionally enables ``gated_linear_unit`` / ``bias_activation_fusion``
        """
        config_params: dict = {
            "num_layers": 1,
            "hidden_size": self.hidden_size,
            "num_attention_heads": self.num_attention_heads,
            "kv_channels": self.head_dim,
            "num_query_groups": self.num_query_groups,
            "ffn_hidden_size": self.ffn_hidden_size,
            "layernorm_epsilon": self.layernorm_epsilon,
            "hidden_dropout": self.drop_rate,
            "attention_dropout": self.drop_rate,
            "qk_layernorm": self.qk_layernorm,
            "apply_query_key_layer_scaling": False,
            "rotary_interleaved": False,
            "flash_decode": False,
            "apply_rope_fusion": True,
            "params_dtype": dtype,
            "tensor_model_parallel_size": tensor_parallel_size,
            "context_parallel_size": context_parallel_size,
            "add_bias_linear": self.add_bias_linear,
        }

        if partition_type == "mlp":
            config_params.update({
                "gated_linear_unit": True,
                "activation_func": F.silu,
                "bias_activation_fusion": True,
            })

        config_params.update(kwargs)
        return TransformerConfig(**config_params)


MODEL_REGISTRY: Dict[str, ModelConfig] = {
    "llama3.3_70b": ModelConfig(
        name="llama3.3_70b",
        hidden_size=8192, num_attention_heads=64, head_dim=128,
        num_query_groups=8, ffn_hidden_size=28672, num_layers=80,
        vocab_size=128256,
    ),
    "llama3.2_3b": ModelConfig(
        name="llama3.2_3b",
        hidden_size=3072, num_attention_heads=24, head_dim=128,
        num_query_groups=8, ffn_hidden_size=8192, num_layers=28,
        vocab_size=128256,
    ),
    "qwen3_1.7b": ModelConfig(
        name="qwen3_1.7b",
        hidden_size=2048, num_attention_heads=16, head_dim=128,
        num_query_groups=8, ffn_hidden_size=6144, num_layers=28,
        vocab_size=151936, layernorm_epsilon=1e-6, qk_layernorm=True,
    ),
}


def get_model_config(name: str) -> ModelConfig:
    """Look up a model by name."""
    if name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model {name!r}. Available: {list(MODEL_REGISTRY)}"
        )
    return MODEL_REGISTRY[name]

GPU_CONFIGS: Dict[str, dict] = {
    "A40":  {"p2p_power_w": 70.0, "freq_range": (900, 1740, 30)},
    "A100": {"p2p_power_w": 85.0, "freq_range": (900, 1410, 30)},
}
DEFAULT_GPU = "A100"


def get_p2p_power(gpu_type: str) -> float:
    """Return p2p power (W) for *gpu_type*, with a conservative fallback."""
    return float(GPU_CONFIGS.get(gpu_type, {}).get("p2p_power_w", 70.0))


