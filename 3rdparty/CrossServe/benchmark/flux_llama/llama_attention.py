import torch
from transformers.models.llama.modeling_llama import LlamaSdpaAttention
from transformers import LlamaConfig
from deepspeed_profiler import FlopsProfiler
from torch.profiler import profile, record_function, ProfilerActivity

# from deepspeed.profiling.flops_profiler import FlopsProfiler
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


def main():
    config = LlamaConfig.from_pretrained(
        "meta-llama/Llama-2-13b-hf",
        hidden_size=3072,
        num_attention_heads=24,
        num_key_value_heads=24,
        num_hidden_layers=2,
    )
    attn = LlamaSdpaAttention(config)
    attn = attn.cuda().to(torch.float16)

    batch_size = 1
    seq_length = 16384
    hidden_states = torch.randn(batch_size, seq_length, 3072).cuda().to(torch.float16)
    pos_emb = [
        torch.randn(batch_size, seq_length, 128).cuda().to(torch.float16),
        torch.randn(batch_size, seq_length, 128).cuda().to(torch.float16),
    ]
    # Create position_ids
    position_ids = torch.arange(seq_length, dtype=torch.long).unsqueeze(0).cuda()

    USE_TORCH_PROFILER = True
    USE_TORCH_COMPILE = True

    if USE_TORCH_COMPILE:
        attn = torch.compile(attn)
        for _ in range(3):
            out = attn(hidden_states, position_ids=position_ids)

    with torch.inference_mode():
        for _ in range(5):
            # out = attn(hidden_states, position_embeddings=pos_emb)
            out = attn(hidden_states, position_ids=position_ids)

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
            pass

        with Timer("LlamaSdpaAttention") as t:
            out = attn(hidden_states, position_ids=position_ids)
        print(f"LlamaSdpaAttention: {t.duration}s")

        if USE_TORCH_PROFILER:
            prof.step()
            prof.stop()
            prof.export_chrome_trace(f"flux/flux_llama/llama_attn_like_flux_causal_compile_{USE_TORCH_COMPILE}.json")
        else:
            prof.stop_profile()
            prof.print_model_profile()


if __name__ == "__main__":
    main()
