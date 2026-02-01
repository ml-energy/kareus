from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.extensions.transformer_engine import _get_extra_te_kwargs
from kareus.transformer_engine.pytorch.ops import LayerNorm, RMSNorm

class TENormOp:
    """
    A conditional wrapper to initialize an instance of Transformer-Engine's
    `LayerNorm` or `RMSNorm` based on input
    """

    # TODO should we ditch normalization config and just use spec to choose LayerNorm vs RMSNorm?
    def __new__(cls, config: TransformerConfig, hidden_size: int, eps: float = 1e-5):
        extra_kwargs = _get_extra_te_kwargs(config)
        if "params_dtype" in extra_kwargs:
            extra_kwargs["dtype"] = extra_kwargs["params_dtype"]
            del extra_kwargs["params_dtype"]

        if config.normalization == "LayerNorm":
            instance = LayerNorm(
                hidden_size,
                eps=eps,
                # sequence_parallel=config.sequence_parallel,
                zero_centered_gamma=config.layernorm_zero_centered_gamma,
                **extra_kwargs,
            )
        elif config.normalization == "RMSNorm":
            instance = RMSNorm(
                hidden_size,
                eps=eps,
                # sequence_parallel=config.sequence_parallel,
                zero_centered_gamma=config.layernorm_zero_centered_gamma,
                **extra_kwargs,
            )
        else:
            raise Exception('Only LayerNorm and RMSNorm are curently supported')

        return instance