import inspect
from typing import Optional, Union, Tuple

import torch
from torch import nn
import torch.distributed
from torch.nn import functional as F
from diffusers.utils import deprecate
from diffusers.models.attention import Attention
from diffusers.models.attention_processor import (
    AttnProcessor2_0,
    JointAttnProcessor2_0,
    FluxAttnProcessor2_0,
    FluxSingleAttnProcessor2_0,
    HunyuanAttnProcessor2_0,
)

try:
    from diffusers.models.attention_processor import CogVideoXAttnProcessor2_0
except ImportError:
    CogVideoXAttnProcessor2_0 = None

from cfuser.core.distributed import (
    get_sequence_parallel_world_size,
    get_sequence_parallel_rank,
    get_sp_group,
)

from cfuser.core.distributed.runtime_state import get_runtime_state
from cfuser.model_executor.layers import cFuserLayerBaseWrapper
from cfuser.model_executor.layers import cFuserLayerWrappersRegister
from cfuser.logger import init_logger
from cfuser.envs import PACKAGES_CHECKER

logger = init_logger(__name__)

env_info = PACKAGES_CHECKER.get_packages_info()
HAS_LONG_CTX_ATTN = env_info["has_long_ctx_attn"]
HAS_FLASH_ATTN = env_info["has_flash_attn"]


def is_v100():
    if not torch.cuda.is_available():
        return False
    device_name = torch.cuda.get_device_name(torch.cuda.current_device())
    return "V100" in device_name


def torch_compile_disable_if_v100(func):
    if is_v100():
        return torch.compiler.disable(func)
    return func


# Not compilable


def apply_rotary_emb(
    x: torch.Tensor,
    freqs_cis: Union[torch.Tensor, Tuple[torch.Tensor]],
    use_real: bool = True,
    use_real_unbind_dim: int = -1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor. This function applies rotary embeddings
    to the given query or key 'x' tensors using the provided frequency tensor 'freqs_cis'. The input tensors are
    reshaped as complex numbers, and the frequency tensor is reshaped for broadcasting compatibility. The resulting
    tensors contain rotary embeddings and are returned as real tensors.

    Args:
        x (`torch.Tensor`):
            Query or key tensor to apply rotary embeddings. [B, H, S, D] xk (torch.Tensor): Key tensor to apply
        freqs_cis (`Tuple[torch.Tensor]`): Precomputed frequency tensor for complex exponentials. ([S, D], [S, D],)

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.
    """
    if use_real:
        cos, sin = freqs_cis  # [S, D]
        cos = cos[None, None]
        sin = sin[None, None]
        cos, sin = cos.to(x.device), sin.to(x.device)

        if use_real_unbind_dim == -1:
            # Used for flux, cogvideox, hunyuan-dit
            x_real, x_imag = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)  # [B, S, H, D//2]
            x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(3)
        elif use_real_unbind_dim == -2:
            # Used for Stable Audio
            x_real, x_imag = x.reshape(*x.shape[:-1], 2, -1).unbind(-2)  # [B, S, H, D//2]
            x_rotated = torch.cat([-x_imag, x_real], dim=-1)
        else:
            raise ValueError(f"`use_real_unbind_dim={use_real_unbind_dim}` but should be -1 or -2.")

        out = (x.float() * cos + x_rotated.float() * sin).to(x.dtype)

        return out
    else:
        # used for lumina
        x_rotated = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        freqs_cis = freqs_cis.unsqueeze(2)
        x_out = torch.view_as_real(x_rotated * freqs_cis).flatten(3)

        return x_out.type_as(x)


class cFuserAttentionBaseWrapper(cFuserLayerBaseWrapper):
    def __init__(
        self,
        attention: Attention,
    ):
        super().__init__(module=attention)

        to_k = self.module.to_k
        to_v = self.module.to_v
        assert isinstance(to_k, nn.Linear)
        assert isinstance(to_v, nn.Linear)
        assert (to_k.bias is None) == (to_v.bias is None)
        assert to_k.weight.shape == to_v.weight.shape


class cFuserAttentionProcessorRegister:
    _XFUSER_ATTENTION_PROCESSOR_MAPPING = {}

    @classmethod
    def register(cls, origin_processor_class):
        def decorator(cfuser_processor):
            if not issubclass(cfuser_processor, origin_processor_class):
                raise ValueError(
                    f"{cfuser_processor.__class__.__name__} is not a subclass of origin class {origin_processor_class.__class__.__name__}"
                )
            cls._XFUSER_ATTENTION_PROCESSOR_MAPPING[origin_processor_class] = cfuser_processor
            # logger.info(
            #     f"{cls.__name__} Register {cfuser_processor.__name__} to {origin_processor_class.__name__}"
            # )
            return cfuser_processor

        return decorator

    @classmethod
    def get_processor(cls, processor):
        for (
            origin_processor_class,
            cfuser_processor,
        ) in cls._XFUSER_ATTENTION_PROCESSOR_MAPPING.items():
            if isinstance(processor, origin_processor_class):
                return cfuser_processor
        raise ValueError(f"Attention Processor class {processor.__class__.__name__} is not supported by cFuser")


@cFuserLayerWrappersRegister.register(Attention)
class cFuserAttentionWrapper(cFuserAttentionBaseWrapper):
    def __init__(
        self,
        attention: Attention,
        latte_temporal_attention: bool = False,
    ):
        super().__init__(attention=attention)
        self.processor = cFuserAttentionProcessorRegister.get_processor(attention.processor)()
        self.latte_temporal_attention = latte_temporal_attention

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **cross_attention_kwargs,
    ) -> torch.Tensor:
        r"""
        The forward method of the `Attention` class.

        Args:
            hidden_states (`torch.Tensor`):
                The hidden states of the query.
            encoder_hidden_states (`torch.Tensor`, *optional*):
                The hidden states of the encoder.
            attention_mask (`torch.Tensor`, *optional*):
                The attention mask to use. If `None`, no mask is applied.
            **cross_attention_kwargs:
                Additional keyword arguments to pass along to the cross attention.

        Returns:
            `torch.Tensor`: The output of the attention layer.
        """
        # The `Attention` class can call different attention processors / attention functions
        # here we simply pass along all tensors to the selected processor class
        # For standard processors that are defined here, `**cross_attention_kwargs` is empty
        attn_parameters = set(inspect.signature(self.processor.__call__).parameters.keys())
        quiet_attn_parameters = {"ip_adapter_masks"}
        unused_kwargs = [
            k for k, _ in cross_attention_kwargs.items() if k not in attn_parameters and k not in quiet_attn_parameters
        ]
        if len(unused_kwargs) > 0:
            logger.warning(
                f"cross_attention_kwargs {unused_kwargs} are not expected by {self.processor.__class__.__name__} and will be ignored."
            )
        cross_attention_kwargs = {k: w for k, w in cross_attention_kwargs.items() if k in attn_parameters}

        return self.processor(
            self,
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            latte_temporal_attention=self.latte_temporal_attention,
            **cross_attention_kwargs,
        )


@cFuserAttentionProcessorRegister.register(FluxAttnProcessor2_0)
class cFuserFluxAttnProcessor2_0(FluxAttnProcessor2_0):
    """Attention processor used typically in processing the SD3-like self-attention projections."""

    def __init__(self):
        super().__init__()
        if HAS_LONG_CTX_ATTN:
            from cfuser.core.long_ctx_attention import (
                cFuserFluxLongContextAttention,
            )

            if HAS_FLASH_ATTN:
                self.hybrid_seq_parallel_attn = cFuserFluxLongContextAttention()

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> torch.FloatTensor:
        batch_size, _, _ = hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape

        # `sample` projections.
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # the attention in FluxSingleTransformerBlock does not use `encoder_hidden_states`
        if encoder_hidden_states is not None:
            # `context` projections.
            encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
            encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
            encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

            encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)

            if attn.norm_added_q is not None:
                encoder_hidden_states_query_proj = attn.norm_added_q(encoder_hidden_states_query_proj)
            if attn.norm_added_k is not None:
                encoder_hidden_states_key_proj = attn.norm_added_k(encoder_hidden_states_key_proj)

            num_encoder_hidden_states_tokens = encoder_hidden_states_query_proj.shape[2]
            num_query_tokens = query.shape[2]

            # attention
            query = torch.cat([encoder_hidden_states_query_proj, query], dim=2)
            key = torch.cat([encoder_hidden_states_key_proj, key], dim=2)
            value = torch.cat([encoder_hidden_states_value_proj, value], dim=2)
        else:
            num_encoder_hidden_states_tokens = 0
            num_query_tokens = query.shape[2]

        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb)
            key = apply_rotary_emb(key, image_rotary_emb)

        #! ---------------------------------------- ATTENTION ----------------------------------------
        if HAS_LONG_CTX_ATTN and get_sequence_parallel_world_size() > 1:
            query = query.transpose(1, 2)
            key = key.transpose(1, 2)
            value = value.transpose(1, 2)
            encoder_hidden_states_query_proj, query = query.split(
                [num_encoder_hidden_states_tokens, num_query_tokens], dim=1
            )
            encoder_hidden_states_key_proj, key = key.split([num_encoder_hidden_states_tokens, num_query_tokens], dim=1)
            encoder_hidden_states_value_proj, value = value.split(
                [num_encoder_hidden_states_tokens, num_query_tokens], dim=1
            )
            hidden_states = self.hybrid_seq_parallel_attn(
                attn,
                query,
                key,
                value,
                dropout_p=0.0,
                causal=False,
                joint_tensor_query=encoder_hidden_states_query_proj,
                joint_tensor_key=encoder_hidden_states_key_proj,
                joint_tensor_value=encoder_hidden_states_value_proj,
                joint_strategy="front",
            )
            hidden_states = hidden_states.reshape(batch_size, -1, attn.heads * head_dim)

        else:
            if HAS_FLASH_ATTN:
                from flash_attn import flash_attn_func

                query = query.transpose(1, 2)
                key = key.transpose(1, 2)
                value = value.transpose(1, 2)
                hidden_states = flash_attn_func(query, key, value, dropout_p=0.0, causal=False)
                hidden_states = hidden_states.reshape(batch_size, -1, attn.heads * head_dim)
            else:
                hidden_states = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)
                hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        #! ---------------------------------------- ATTENTION ----------------------------------------

        hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            encoder_hidden_states, hidden_states = (
                hidden_states[:, : encoder_hidden_states.shape[1]],
                hidden_states[:, encoder_hidden_states.shape[1] :],
            )

            # linear proj
            hidden_states = attn.to_out[0](hidden_states)
            # dropout
            hidden_states = attn.to_out[1](hidden_states)
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

            return hidden_states, encoder_hidden_states
        else:
            return hidden_states
