import torch
import torch.nn as nn
from diffusers.models.transformers.transformer_flux import (
    FluxAttnProcessor2_0,
    Attention,
)
from deepspeed_profiler import FlopsProfiler
from torch.profiler import profile, record_function, ProfilerActivity

# from deepspeed.profiling.flops_profiler import FlopsProfiler
# from analyzer.utils import Timer
import time


class Timer:
    def __init__(self, name):
        self.name = name
        self.start_time = None
        self.duration = None

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        end_time = time.perf_counter()
        self.duration = end_time - self.start_time


class FluxAttention(nn.Module):
    r"""
    A Transformer block following the MMDiT architecture, introduced in Stable Diffusion 3.

    Reference: https://arxiv.org/abs/2403.03206

    Parameters:
        dim (`int`): The number of channels in the input and output.
        num_attention_heads (`int`): The number of heads to use for multi-head attention.
        attention_head_dim (`int`): The number of channels in each head.
        context_pre_only (`bool`): Boolean to determine if we should add some blocks associated with the
            processing of `context` conditions.
    """

    def __init__(self, dim, num_attention_heads, attention_head_dim):
        super().__init__()
        processor = FluxAttnProcessor2_0()
        self.attn = Attention(
            query_dim=dim,
            cross_attention_dim=None,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            bias=True,
            processor=processor,
            qk_norm="rms_norm",
            eps=1e-6,
            pre_only=True,
        )

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        image_rotary_emb=None,
    ):
        hidden_states = self.attn(
            hidden_states=hidden_states,
            image_rotary_emb=image_rotary_emb,
        )
        return hidden_states


def main():
    attn = FluxAttention(dim=3072, num_attention_heads=24, attention_head_dim=128)
    attn = attn.cuda().to(torch.float16)

    batch_size = 1
    seq_length = 16384
    hidden_states = torch.randn(batch_size, seq_length, 3072).cuda().to(torch.float16)
    image_rotary_emb = [
        torch.randn(batch_size, seq_length, 128).cuda().to(torch.float16),
        torch.randn(batch_size, seq_length, 128).cuda().to(torch.float16),
    ]

    USE_TORCH_PROFILER = True
    USE_TORCH_COMPILE = True

    if USE_TORCH_COMPILE:
        attn = torch.compile(attn)
        for _ in range(3):
            out = attn(hidden_states, image_rotary_emb)

    with torch.inference_mode():
        for _ in range(5):
            out = attn(hidden_states, image_rotary_emb)

        if USE_TORCH_COMPILE:
            attn = torch.compile(attn)

        if USE_TORCH_PROFILER:
            prof = torch.profiler.profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                profile_memory=True,
                with_stack=True,
                with_flops=True,
                with_modules=True,
                record_shapes=True,
            )
            prof.start()
        else:
            prof = FlopsProfiler(attn)
            prof.start_profile()

        with Timer("FluxAttention") as t:
            out = attn(hidden_states, image_rotary_emb)
        print(f"FluxAttention: {t.duration}s")

        if USE_TORCH_PROFILER:
            prof.step()
            prof.stop()
            prof.export_chrome_trace(f"flux/flux_llama/flux_attn_compile_{USE_TORCH_COMPILE}.json")
        else:
            prof.stop_profile()
            prof.print_model_profile()


if __name__ == "__main__":
    main()
