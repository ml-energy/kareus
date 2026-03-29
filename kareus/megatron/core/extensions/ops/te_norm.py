"""
Modified from Megatron-LM (megatron/core/extensions/transformer_engine.py::TENorm).
Changes: replaces the original te.pytorch.LayerNorm / te.pytorch.RMSNorm bases
with the BasicOperation-based LayerNorm / RMSNorm ops and adds the
PartitionableOperator mixin so normalization layers can participate in the
partition scheduler's compute/communication graph.
"""

from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.extensions.transformer_engine import _get_extra_te_kwargs
from kareus.transformer_engine.pytorch.ops import LayerNorm, RMSNorm
from kareus.megatron.core.partitions.tensor_graph import PartitionableOperator


class PartitionableLayerNorm(LayerNorm, PartitionableOperator):
    """BasicOperation-based LayerNorm with the PartitionableOperator interface."""
    pass


class PartitionableRMSNorm(RMSNorm, PartitionableOperator):
    """BasicOperation-based RMSNorm with the PartitionableOperator interface."""
    pass


class TENormOp:
    """
    A conditional wrapper to initialize an instance of Transformer-Engine's
    `LayerNorm` or `RMSNorm` based on input.

    Modified from Megatron-LM's `TENorm`: the returned instances inherit
    from BasicOperation-based norm ops and `PartitionableOperator`, so
    they expose `fuser_forward` / `fuser_backward` with externally
    managed ctx and can declare their compute graph for partition scheduling.
    """

    # TODO should we ditch normalization config and just use spec to choose LayerNorm vs RMSNorm?
    def __new__(cls, config: TransformerConfig, hidden_size: int, eps: float = 1e-5):
        extra_kwargs = _get_extra_te_kwargs(config)
        if "params_dtype" in extra_kwargs:
            extra_kwargs["dtype"] = extra_kwargs["params_dtype"]
            del extra_kwargs["params_dtype"]

        if config.normalization == "LayerNorm":
            instance = PartitionableLayerNorm(
                hidden_size,
                eps=eps,
                # sequence_parallel=config.sequence_parallel,
                zero_centered_gamma=config.layernorm_zero_centered_gamma,
                **extra_kwargs,
            )
        elif config.normalization == "RMSNorm":
            instance = PartitionableRMSNorm(
                hidden_size,
                eps=eps,
                # sequence_parallel=config.sequence_parallel,
                zero_centered_gamma=config.layernorm_zero_centered_gamma,
                **extra_kwargs,
            )
        else:
            raise Exception('Only LayerNorm and RMSNorm are curently supported')

        return instance