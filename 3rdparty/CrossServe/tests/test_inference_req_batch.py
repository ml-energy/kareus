# This script is for testing the correctness and performance of the inference requests batch

import json
import os

# os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
import random
from copy import deepcopy

import torch
import torch.distributed
from torch.profiler import profile, ProfilerActivity, tensorboard_trace_handler, schedule
from cfuser import cFuserFluxPipeline
from cfuser.config import (
    EngineConfig,
    InputConfig,
    ModelConfig,
    ParallelConfig,
    RuntimeConfig,
    SequenceParallelConfig,
)
from cfuser.core.distributed import (
    get_runtime_state,
    get_world_group,
    init_distributed_environment,
)
from cfuser.core.utils import nvtx_range
from cfuser.logger import init_logger
from cfuser.testing import assert_close
from contextlib import nullcontext

logger = init_logger(__name__)


def warmup_pipe(pipe, input_config, args, steps=2):
    for _ in range(steps):
        pipe(
            height=input_config.height,
            width=input_config.width,
            prompt=input_config.prompt,
            num_inference_steps=input_config.num_inference_steps,
            output_type=input_config.output_type,
            max_sequence_length=256,
            guidance_scale=0.0,
            generator=torch.Generator(device="cuda").manual_seed(input_config.seed),
            return_dict=False,
        )
        pipe.inference_requests_batch(
            input_configs=[input_config, input_config],
            generators=[
                torch.Generator(device="cuda").manual_seed(input_config.seed),
                torch.Generator(device="cuda").manual_seed(input_config.seed),
            ],
        )


def test_inference_req_batch(rank, world_size, args, master_port):
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = f"{master_port}"

    init_distributed_environment(rank=rank, local_rank=rank, world_size=world_size)

    torch.cuda.set_device(rank)

    engine_config = EngineConfig(
        model_config=ModelConfig(
            model="black-forest-labs/FLUX.1-dev",
        ),
        runtime_config=RuntimeConfig(
            dtype=torch.float16,
            use_torch_compile=False,
        ),
        parallel_config=ParallelConfig(
            sp_config=SequenceParallelConfig(
                ulysses_degree=args.ulysses_degree,
                ring_degree=args.ring_degree,
            )
        ),
    )

    output_type = "latent"
    prompt = "a beautiful cyberpunk city with high-rise buildings and a river"
    input_config = InputConfig(
        height=args.height,
        width=args.width,
        prompt=[prompt] * args.batch_size,
        num_inference_steps=args.num_inference_steps,
        output_type=output_type,
        seed=42,
    )

    input_config_1 = InputConfig(
        height=args.height,
        width=args.width,
        prompt=[prompt] * args.batch_size,
        num_inference_steps=args.num_inference_steps,
        output_type=output_type,
        seed=42,
    )

    input_config_2 = InputConfig(
        height=args.height,
        width=args.width,
        prompt=[prompt] * args.batch_size,
        num_inference_steps=args.num_inference_steps,
        output_type=output_type,
        seed=42,
    )

    local_rank = get_world_group().local_rank

    pipe = cFuserFluxPipeline.from_pretrained(
        pretrained_model_name_or_path=engine_config.model_config.model,
        engine_config=engine_config,
        torch_dtype=torch.bfloat16,
        # device_map="auto",  # Add device_map parameter
    ).to(f"cuda:{local_rank}")

    # warmup
    warmup_pipe(pipe, input_config, args, steps=args.warmup_steps)
    timer_start = torch.cuda.Event(enable_timing=True)
    timer_end = torch.cuda.Event(enable_timing=True)

    torch.cuda.cudart().cudaProfilerStart()
    torch.cuda.reset_peak_memory_stats()
    timer_start.record()
    with nvtx_range(f"baseline {args.repeat} forward"):
        for index_repeat in range(args.repeat):
            output = pipe(
                height=input_config.height,
                width=input_config.width,
                prompt=input_config.prompt,
                num_inference_steps=input_config.num_inference_steps,
                output_type=input_config.output_type,
                max_sequence_length=256,
                guidance_scale=0.0,
                generator=torch.Generator(device="cuda").manual_seed(input_config.seed),
                return_dict=False,
            )
            # two requests
            output = pipe(
                height=input_config.height,
                width=input_config.width,
                prompt=input_config.prompt,
                num_inference_steps=input_config.num_inference_steps,
                output_type=input_config.output_type,
                max_sequence_length=256,
                guidance_scale=0.0,
                generator=torch.Generator(device="cuda").manual_seed(input_config.seed),
                return_dict=False,
            )
        output = output[0]

    timer_end.record()
    timer_end.synchronize()
    elapsed_time = timer_start.elapsed_time(timer_end) / 1e3 / args.repeat
    peak_memory = torch.cuda.max_memory_allocated(device=f"cuda:{local_rank}")

    torch.cuda.reset_peak_memory_stats()
    timer_start.record()
    with nvtx_range(f"two Requests batch inline inference"):
        for index_repeat in range(args.repeat):
            output_batch_inline = pipe.inference_requests_batch(
                input_configs=[input_config_1, input_config_2],
                generators=[
                    torch.Generator(device="cuda").manual_seed(input_config_1.seed),
                    torch.Generator(device="cuda").manual_seed(input_config_2.seed),
                ],
            )
    timer_end.record()
    timer_end.synchronize()
    elapsed_time_batch_inline = timer_start.elapsed_time(timer_end) / 1e3 / args.repeat
    peak_memory_batch_inline = torch.cuda.max_memory_allocated(device=f"cuda:{local_rank}")

    if get_world_group().rank == 0:
        results = {
            "repeat": args.repeat,
            "batch_size": args.batch_size,
            "height": args.height,
            "width": args.width,
            "num_inference_steps": args.num_inference_steps,
            "ulysses_degree": args.ulysses_degree,
            "ring_degree": args.ring_degree,
            "baseline": {"time": elapsed_time, "memory": peak_memory / 1e9},
            "batch_inline": {"time": elapsed_time_batch_inline, "memory": peak_memory_batch_inline / 1e9},
        }

        # Print results
        print(f"repeat: {args.repeat}")
        print(
            f"batch_size: {args.batch_size}, height: {args.height}, width: {args.width}, num_inference_steps: {args.num_inference_steps}, ulysses_degree: {args.ulysses_degree}, ring_degree: {args.ring_degree}"
        )
        print(f"epoch time baseline: {elapsed_time:.2f} sec, memory: {peak_memory/1e9} GB")
        print(
            f"epoch time batch inline: {elapsed_time_batch_inline:.2f} sec, memory: {peak_memory_batch_inline/1e9} GB"
        )

        if args.output_json:
            try:
                existing_data = []
                try:
                    with open(args.output_json, "r") as f:
                        existing_data = json.load(f)
                except (FileNotFoundError, json.JSONDecodeError):
                    existing_data = []

                if not isinstance(existing_data, list):
                    existing_data = [existing_data]

                existing_data.append(results)

                with open(args.output_json, "w") as f:
                    json.dump(existing_data, f, indent=4)
            except Exception as e:
                print(f"Error writing to JSON file: {e}")

    assert not torch.isnan(output_batch_inline[0]).any()
    assert not torch.isnan(output_batch_inline[1]).any()

    assert_close(output, output_batch_inline[0], atol=1e-3, rtol=1e-3)
    assert_close(output, output_batch_inline[1], atol=1e-3, rtol=1e-3)

    get_runtime_state().destory_distributed_env()


"""
Example usages:
# check correctness
CUDA_VISIBLE_DEVICES=0,1 python tests/test_inference_req_batch.py --ulysses_degree 2 --ring_degree 1 --height 512 --width 512 --num_inference_steps 1 --repeat 1 --warmup_steps 1

# check performance
## with stream
export nsys_args="--force-overwrite true -w true -s cpu --python-backtrace=cuda --cudabacktrace=all --capture-range=cudaProfilerApi"
CUDA_VISIBLE_DEVICES=0,1,2,3 nsys profile ${nsys_args} -o log/tests/test_inference_req_batch_bs1_u4_r1_h1024_w1024 python tests/test_inference_req_batch.py --ulysses_degree 4 --ring_degree 1 --height 1024 --width 1024 --num_inference_steps 8 --repeat 3 -b 4
CUDA_VISIBLE_DEVICES=4,5,6,7 nsys profile ${nsys_args}  -o log/tests/test_inference_req_batch_bs1_u4_r1_h512_w512 python tests/test_inference_req_batch.py --ulysses_degree 4 --ring_degree 1 --height 512 --width 512 --num_inference_steps 8
NCCL_MAX_CTAS=1 NCCL_MIN_CTAS=1 CUDA_VISIBLE_DEVICES=0,1 nsys profile ${nsys_args} -o log/tests/test_inference_req_batch_max_ctas_1_1_bs1_u2_r1_h2048_w2048 python tests/test_inference_req_batch.py --ulysses_degree 2 --ring_degree 1 --height 2048 --width 2048 --num_inference_steps 3 --repeat 2
NCCL_MAX_CTAS=1 NCCL_MIN_CTAS=1 CUDA_VISIBLE_DEVICES=0,1,2,3 nsys profile ${nsys_args} -o log/tests/test_inference_req_batch_max_ctas_1_1_bs1_u4_r1_h2048_w2048 python tests/test_inference_req_batch.py --ulysses_degree 4 --ring_degree 1 --height 2048 --width 2048 --num_inference_steps 3 --repeat 2

# check performance without nsys
## with stream
CUDA_VISIBLE_DEVICES=0,1 python tests/test_inference_req_batch.py --ulysses_degree 2 --ring_degree 1 --height 1024 --width 1024 --num_inference_steps 8 --repeat 3
NCCL_MAX_CTAS=4 NCCL_MIN_CTAS=4 CUDA_VISIBLE_DEVICES=0,1 python tests/test_inference_req_batch.py --ulysses_degree 2 --ring_degree 1 --height 1024 --width 1024 --num_inference_steps 8 --repeat 3
"""


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--ulysses_degree", "-u", type=int, default=1)
    parser.add_argument("--ring_degree", "-r", type=int, default=1)
    parser.add_argument("--batch_size", "-b", type=int, default=1)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--num_inference_steps", type=int, default=10)
    parser.add_argument("--warmup_steps", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--output_json", type=str, default=None, help="Path to output JSON file")
    args = parser.parse_args()

    from torch.multiprocessing import spawn

    nprocs = args.ulysses_degree * args.ring_degree
    spawn(
        test_inference_req_batch,
        args=(
            nprocs,
            args,
            random.randint(8000, 65535),
        ),
        nprocs=nprocs,
    )
