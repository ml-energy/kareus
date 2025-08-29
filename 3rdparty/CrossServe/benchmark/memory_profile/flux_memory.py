import logging
import socket
from datetime import datetime

import torch

from diffusers import FluxPipeline

from cfuser.core.utils.memory_profiler import (
    start_record_memory_history,
    stop_record_memory_history,
    export_memory_snapshot,
)

torch.cuda.empty_cache()

memory_allocated_before_init_model = torch.cuda.memory_allocated() / (1024**3)
memory_reserved_before_init_model = torch.cuda.memory_reserved() / (1024**3)
free_memory_before_init_model, total_memory_before_init_model = torch.cuda.mem_get_info()

print(
    f"before model init: torch cuda memory allocated: {memory_allocated_before_init_model:.2f} GB, torch cuda memory reserved: {memory_reserved_before_init_model:.2f} GB, torch cuda free memory: {free_memory_before_init_model / (1024**3):.2f} GB, torch cuda total memory: {total_memory_before_init_model / (1024**3):.2f} GB"
)

pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16).to("cuda")
# pipe.enable_model_cpu_offload() #save some VRAM by offloading the model to CPU. Remove this if you have enough GPU power


memory_allocated_after_init_model = torch.cuda.memory_allocated() / (1024**3)
memory_reserved_after_init_model = torch.cuda.memory_reserved() / (1024**3)
memory_allocated_after_init_model_max = torch.cuda.max_memory_allocated() / (1024**3)
memory_reserved_after_init_model_max = torch.cuda.max_memory_reserved() / (1024**3)

free_memory_after_init_model, total_memory_after_init_model = torch.cuda.mem_get_info()

print(
    f"after model init: torch cuda memory allocated: {memory_allocated_after_init_model:.2f} GB, torch cuda memory reserved: {memory_reserved_after_init_model:.2f} GB, torch cuda free memory: {free_memory_after_init_model / (1024**3):.2f} GB, torch cuda total memory: {total_memory_after_init_model / (1024**3):.2f} GB"
)
print(
    f"after model init max: torch cuda memory max allocated: {memory_allocated_after_init_model_max:.2f} GB, torch cuda memory max reserved: {memory_reserved_after_init_model_max:.2f} GB"
)

torch.cuda.reset_max_memory_allocated()
# torch.cuda.reset_max_memory_reserved()

print(
    f"after model init reset: torch cuda max memory allocated: {torch.cuda.max_memory_allocated() / (1024**3):.2f} GB, torch cuda max memory reserved: {torch.cuda.max_memory_reserved() / (1024**3):.2f} GB"
)

# https://github.com/vllm-project/vllm/blob/e25810ae29058299b7bf845c7ed572f2474a1d85/vllm/worker/worker.py#L177
# Profile the memory usage of the model and get the maximum number of
# cache blocks that can be allocated with the remaining free memory.
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()


def single_run(batch_size=1, height=1024, width=1024, num_inference_steps=50, output_type="pil"):
    prompt = "A cat holding a sign that says hello world"
    image = pipe(
        prompt=[prompt] * batch_size,
        height=height,
        width=width,
        guidance_scale=3.5,
        num_inference_steps=num_inference_steps,
        max_sequence_length=512,
        generator=torch.Generator("cpu").manual_seed(0),
        output_type=output_type,
    ).images[0]
    # image.save("flux-dev.png")
    return image


def run_flux_with_memory_profiling(batch_size=1, height=1024, width=1024, num_inference_steps=5, output_type="pil"):
    with torch.no_grad():
        # Start recording memory snapshot history
        start_record_memory_history()

        # Run Flux
        single_run(
            batch_size=batch_size,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            output_type=output_type,
        )

        # Create the memory snapshot file
        export_memory_snapshot(f"bs{batch_size}-h{height}-w{width}")

        # Stop recording memory snapshot history
        stop_record_memory_history()


# warmup
def warmup(times=3):
    for _ in range(times):
        single_run(num_inference_steps=5)


"""
CUDA_VISIBLE_DEVICES=4 python benchmark/memory_profile/flux_memory.py --batch_size 4 --height 4096 --width 4096 --num_inference_steps 5 --output_type latent
CUDA_VISIBLE_DEVICES=4 python benchmark/memory_profile/flux_memory.py --batch_size 4 --height 4096 --width 4096 --num_inference_steps 5 --output_type pil
CUDA_VISIBLE_DEVICES=4 python benchmark/memory_profile/flux_memory.py --batch_size 4 --height 4096 --width 4096 --num_inference_steps 5 --output_type pil


CUDA_VISIBLE_DEVICES=0 python benchmark/memory_profile/flux_memory.py --batch_size 1 --height 2048 --width 2048 --num_inference_steps 5 --output_type pil
"""


def parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--num_inference_steps", type=int, default=5)
    parser.add_argument("--output_type", type=str, default="pil")
    return parser.parse_args()


if __name__ == "__main__":
    warmup(3)
    # Run Flux with memory profiling
    args = parse_args()
    run_flux_with_memory_profiling(
        batch_size=args.batch_size,
        height=args.height,
        width=args.width,
        num_inference_steps=args.num_inference_steps,
        output_type=args.output_type,
    )
    memory_allocated_after_run = torch.cuda.memory_allocated() / (1024**3)
    memory_reserved_after_run = torch.cuda.memory_reserved() / (1024**3)
    memory_allocated_after_run_max = torch.cuda.max_memory_allocated() / (1024**3)
    memory_reserved_after_run_max = torch.cuda.max_memory_reserved() / (1024**3)
    free_memory_after_run, total_memory_after_run = torch.cuda.mem_get_info()
    print(
        f"after run: torch cuda memory allocated: {memory_allocated_after_run:.2f} GB, torch cuda memory reserved: {memory_reserved_after_run:.2f} GB, torch cuda free memory: {free_memory_after_run / (1024**3):.2f} GB, torch cuda total memory: {total_memory_after_run / (1024**3):.2f} GB"
    )
    print(
        f"after run max: torch cuda memory max allocated: {memory_allocated_after_run_max:.2f} GB, torch cuda memory max reserved: {memory_reserved_after_run_max:.2f} GB"
    )

    # Get the peak memory allocation recorded by torch
    peak_memory = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
    print(
        f"peak memory between init and run for bs {args.batch_size} height {args.height} width {args.width}: {peak_memory / (1024**3):.2f} GB"
    )
