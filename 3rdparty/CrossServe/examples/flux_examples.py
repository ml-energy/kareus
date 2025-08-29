import os
import logging
import time
import torch
import torch.distributed
from cfuser import cFuserFluxPipeline, cFuserArgs
from cfuser.config import FlexibleArgumentParser
from torch.profiler import profile, record_function, ProfilerActivity
from contextlib import nullcontext
from cfuser.core.distributed import (
    get_world_group,
    get_runtime_state,
)

from cfuser.logger import init_logger

logger = init_logger(__name__)


def main():
    parser = FlexibleArgumentParser(description="cFuser Arguments")
    parser.add_argument("--torch_profiler", action="store_true")
    args = cFuserArgs.add_cli_args(parser).parse_args()

    engine_args = cFuserArgs.from_cli_args(args)
    engine_config, input_config = engine_args.create_config()
    local_rank = get_world_group().local_rank

    pipe = cFuserFluxPipeline.from_pretrained(
        pretrained_model_name_or_path=engine_config.model_config.model,
        engine_config=engine_config,
        torch_dtype=torch.bfloat16,
    )

    if args.enable_sequential_cpu_offload:
        pipe.enable_sequential_cpu_offload(gpu_id=local_rank)
        logging.info(f"rank {local_rank} sequential CPU offload enabled")
    else:
        pipe = pipe.to(f"cuda:{local_rank}")

    if args.use_torch_compile:
        torch._dynamo.config.accumulated_cache_size_limit = 256

    # pipe.prepare_run(input_config, steps=2)

    logger.info(f"logger init over")

    ctx = nullcontext()
    if args.torch_profiler:
        ctx = torch.profiler.profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            profile_memory=True,
            with_stack=True,
            with_flops=True,
            with_modules=True,
            record_shapes=True,
        )

    print(f"input_config: {input_config}")
    torch.cuda.reset_peak_memory_stats()
    start_time = time.time()
    with ctx as prof:
        output = pipe(
            height=input_config.height,
            width=input_config.width,
            prompt=input_config.prompt,
            num_inference_steps=input_config.num_inference_steps,
            output_type=input_config.output_type,
            max_sequence_length=256,
            guidance_scale=0.0,
            generator=torch.Generator(device="cuda").manual_seed(input_config.seed),
        )
    end_time = time.time()
    elapsed_time = end_time - start_time
    peak_memory = torch.cuda.max_memory_allocated(device=f"cuda:{local_rank}")

    parallel_info = f"ulysses{engine_args.ulysses_degree}_ring{engine_args.ring_degree}"

    if args.torch_profiler and local_rank == 0:
        prof.export_chrome_trace("./results/cfuser_flux_trace.json")

    if input_config.output_type == "pil":
        if get_world_group().rank == 0:
            os.makedirs("./results", exist_ok=True)
            for i, image in enumerate(output.images):
                image_rank = i
                image_name = f"cfuser_flux_result_{parallel_info}_{image_rank}_tc_{engine_args.use_torch_compile}.png"
                image.save(f"./results/{image_name}")
                print(f"image {i} saved to ./results/{image_name}")

    if get_world_group().rank == 0:
        print(f"epoch time: {elapsed_time:.2f} sec, memory: {peak_memory/1e9} GB")
    get_runtime_state().destory_distributed_env()


"""
# latent
NCCL_DEBUG=WARN CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=1049 examples/flux_examples.py --prompt 'hello world' --output_type 'latent' --model "black-forest-labs/FLUX.1-dev" \
--batch_size 16 --height 2048 --width 2048 --ulysses_degree 4 --ring_degree 1 --num_inference_steps 3

NCCL_DEBUG=WARN CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=1049 examples/flux_examples.py --prompt 'hello world' --output_type 'latent' --model "black-forest-labs/FLUX.1-dev" \
--height 1024 --width 1024 --ulysses_degree 4 --ring_degree 1 --num_inference_steps 3

NCCL_DEBUG=WARN CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 --master_port=1038 examples/flux_examples.py --prompt 'hello world' --output_type 'latent' --model "black-forest-labs/FLUX.1-dev" \
--batch_size 4 --height 4096 --width 4096 --ulysses_degree 4 --ring_degree 1 --num_inference_steps 3

NCCL_DEBUG=WARN CUDA_VISIBLE_DEVICES=5 torchrun --nproc_per_node=1 --master_port=1037 examples/flux_examples.py --prompt 'hello world' --output_type 'latent' --model "black-forest-labs/FLUX.1-dev" \
--batch_size 4 --height 1024 --width 1024 --ulysses_degree 1 --ring_degree 1 --num_inference_steps 3

# pil
NCCL_DEBUG=WARN CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 --master_port=2037 examples/flux_examples.py --prompt 'hello world' --output_type 'pil' --model "black-forest-labs/FLUX.1-dev" \
--batch_size 1 --height 1024 --width 1024 --ulysses_degree 4 --ring_degree 1 --num_inference_steps 50

NCCL_DEBUG=WARN CUDA_VISIBLE_DEVICES=3 torchrun --nproc_per_node=1 --master_port=2037 examples/flux_examples.py --prompt 'hello world' --output_type 'pil' --model "black-forest-labs/FLUX.1-dev" \
--batch_size 1 --height 1024 --width 1024 --ulysses_degree 1 --ring_degree 1 --num_inference_steps 50

# compile
NCCL_DEBUG=WARN CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=1037 examples/flux_examples.py --prompt 'hello world' --output_type 'latent' --model "black-forest-labs/FLUX.1-dev" \
--batch_size 4 --height 4096 --width 4096 --ulysses_degree 4 --ring_degree 1 --num_inference_steps 3 --use_torch_compile

NCCL_DEBUG=WARN CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=1037 examples/flux_examples.py --prompt 'hello world' --output_type 'latent' --model "black-forest-labs/FLUX.1-dev" \
--batch_size 4 --height 1024 --width 1024 --ulysses_degree 2 --ring_degree 1 --num_inference_steps 3 --use_torch_compile

# profiler
NCCL_DEBUG=WARN CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=1037 examples/flux_examples.py --prompt 'hello world' --output_type 'latent' --model "black-forest-labs/FLUX.1-dev" \
--batch_size 4 --height 1024 --width 1024 --ulysses_degree 4 --ring_degree 1 --num_inference_steps 3 --torch_profiler

# nsys
# NCCL_DEBUG=WARN CUDA_VISIBLE_DEVICES=0,1,2,3 nsys profile --gpu-metrics-devices all \ # A40 doesn't support this parameter
NCCL_DEBUG=WARN CUDA_VISIBLE_DEVICES=0,1,2,3 nsys profile \
    --force-overwrite true \
    --output flux_profile \
    torchrun --nproc_per_node=4 --master_port=1037 examples/flux_examples.py --prompt 'hello world' --output_type 'latent' --model "black-forest-labs/FLUX.1-dev" \
    --batch_size 1 --height 1024 --width 1024 --ulysses_degree 4 --ring_degree 1 --num_inference_steps 3
"""

if __name__ == "__main__":
    main()
