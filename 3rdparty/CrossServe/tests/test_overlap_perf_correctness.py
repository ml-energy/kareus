# This script is for testing the correctness and performance of the overlap inference

import json
import os

# NOTE: the below line causes significant perf. degradation

# os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
import random
from copy import deepcopy

import torch
import torch.distributed as dist
from torch.profiler import profile, ProfilerActivity, tensorboard_trace_handler, schedule
from cfuser import cFuserFluxPipeline
from diffusers import FluxPipeline
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
import gc
from cfuser.core.utils import nvtx_range
from cfuser.logger import init_logger
from cfuser.testing import assert_close
from contextlib import nullcontext

logger = init_logger(__name__)


def warmup_pipe(pipe, input_config, args, steps=20):
    input_config_1 = deepcopy(input_config)
    input_config_2 = deepcopy(input_config)
    input_config_1.num_inference_steps = steps
    input_config_2.num_inference_steps = steps
    pipe.prepare_run(input_config_1, steps=steps)
    pipe.inference(
        input_config_1,
        input_config_2,
        return_dict=False,
        async_op=False,
        inline_inference=False,
        pack_qkv=args.pack_qkv,
    )
    pipe.inference(
        input_config_1, input_config_2, return_dict=False, async_op=False, inline_inference=True, pack_qkv=args.pack_qkv
    )
    pipe.inference(
        input_config_1, input_config_2, return_dict=False, async_op=True, inline_inference=True, pack_qkv=args.pack_qkv
    )


def test_overlap_perf_correctness(rank, world_size, args, master_port):
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

    if args.compile:
        print("Enabling torch.compile")
    else:
        # torch.compiler.set_stance("force_eager")
        print("Disabling torch.compile")

    output_type = "pil_latent" if args.save_image else "latent"
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

    # initialize_profiler()
    # profiler_data = {}
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
        if args.save_image:
            org_image = output[1][0]  # the first PIL image in the batch
        output = output[0]

    timer_end.record()
    timer_end.synchronize()
    elapsed_time = timer_start.elapsed_time(timer_end) / 1e3 / args.repeat
    peak_memory = torch.cuda.max_memory_allocated(device=f"cuda:{local_rank}")

    torch.cuda.cudart().cudaProfilerStart()

    torch.cuda.reset_peak_memory_stats()
    timer_start.record()
    with nvtx_range(f"two Requests interleaved inference"):
        for index_repeat in range(args.repeat):
            output_overlap_1, output_overlap_2 = pipe.inference(input_config_1, input_config_2, return_dict=False)
        if input_config_1.output_type == "pil_latent":
            output_overlap_1 = output_overlap_1[0]
            output_overlap_2 = output_overlap_2[0]

    timer_end.record()
    timer_end.synchronize()
    elapsed_time_overlap = timer_start.elapsed_time(timer_end) / 1e3 / args.repeat
    peak_memory_overlap = torch.cuda.max_memory_allocated(device=f"cuda:{local_rank}")

    # if get_world_group().rank == 0:
    #     print_time_distribution()``
    #     profiler_data["overlap_batch"] = get_time()
    # clear_profiler()
    torch.cuda.reset_peak_memory_stats()
    timer_start.record()
    with nvtx_range(f"two Requests Streams inline interleaved inference"):
        for index_repeat in range(args.repeat):
            output_overlap_1_inline, output_overlap_2_inline = pipe.inference(
                input_config_1,
                input_config_2,
                return_dict=False,
                inline_inference=True,
                pack_qkv=args.pack_qkv,
            )
        if input_config_1.output_type == "pil_latent":
            output_overlap_1_inline = output_overlap_1_inline[0]
            output_overlap_2_inline = output_overlap_2_inline[0]

    timer_end.record()
    timer_end.synchronize()
    elapsed_time_overlap_inline = timer_start.elapsed_time(timer_end) / 1e3 / args.repeat
    peak_memory_overlap_inline = torch.cuda.max_memory_allocated(device=f"cuda:{local_rank}")

    # if get_world_group().rank == 0:
    #     print_time_distribution()
    #     profiler_data["overlap_inline"] = get_time()
    # clear_profiler()

    torch.cuda.reset_peak_memory_stats()

    if args.torch_prof:
        rank = torch.distributed.get_rank()
        prof = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=schedule(wait=0, warmup=1, active=1),
            on_trace_ready=tensorboard_trace_handler(
                f"log/tests",
                worker_name=f"rank{rank}_test_overlap_u{args.ulysses_degree}_r{args.ring_degree}_bs{args.batch_size}_h{args.height}_w{args.width}",
            ),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        )
    else:
        prof = nullcontext()

    timer_start.record()
    with nvtx_range(f"two Requests Streams overlap inference"), prof:
        for index_repeat in range(args.repeat):
            output_overlap_1_inline_async, output_overlap_2_inline_async = pipe.inference(
                input_config_1,
                input_config_2,
                return_dict=False,
                inline_inference=True,
                async_op=True,
                no_stream=args.no_stream,
                pack_qkv=args.pack_qkv,
            )
            if args.torch_prof:
                prof.step()
        if input_config_1.output_type == "pil_latent":
            overlap_image = output_overlap_1_inline_async[1][0]
            output_overlap_1_inline_async = output_overlap_1_inline_async[0]
            output_overlap_2_inline_async = output_overlap_2_inline_async[0]

    timer_end.record()
    timer_end.synchronize()
    elapsed_time_overlap_inline_async = timer_start.elapsed_time(timer_end) / 1e3 / args.repeat
    peak_memory_overlap_inline_async = torch.cuda.max_memory_allocated(device=f"cuda:{local_rank}")

    # if get_world_group().rank == 0:
    #     print_time_distribution()
    #     profiler_data["overlap_inline_async"] = get_time()
    # clear_profiler()

    torch.cuda.cudart().cudaProfilerStop()
    # parallel_info = f"ulysses{ulysses_degree}_ring{ring_degree}"

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
            "overlap_batch": {
                "time": elapsed_time_overlap,
                "memory": peak_memory_overlap / 1e9,
            },
            "overlap_inline": {
                "time": elapsed_time_overlap_inline,
                "memory": peak_memory_overlap_inline / 1e9,
            },
            "overlap_inline_async": {
                "time": elapsed_time_overlap_inline_async,
                "memory": peak_memory_overlap_inline_async / 1e9,
            },
        }

        # Print results
        print(f"repeat: {args.repeat}")
        print(
            f"batch_size: {args.batch_size}, height: {args.height}, width: {args.width}, num_inference_steps: {args.num_inference_steps}, ulysses_degree: {args.ulysses_degree}, ring_degree: {args.ring_degree}"
        )
        print(f"epoch time: {elapsed_time:.2f} sec, memory: {peak_memory/1e9} GB")
        print(f"overlap batch epoch time: {elapsed_time_overlap:.2f} sec, memory: {peak_memory_overlap/1e9} GB")
        print(
            f"overlap inline epoch time: {elapsed_time_overlap_inline:.2f} sec, memory: {peak_memory_overlap_inline/1e9} GB"
        )
        print(
            f"overlap inline async epoch time: {elapsed_time_overlap_inline_async:.2f} sec, memory: {peak_memory_overlap_inline_async/1e9} GB"
        )

        # for msg, dict_nvtx_time in profiler_data.items():
        #     print(f"{msg}:")
        #     smart_time_distribution(dict_nvtx_time)

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

        if args.save_image:
            sample_dir = "overlap_samples"
            os.makedirs(sample_dir, exist_ok=True)
            org_image.save(f"{sample_dir}/org_image_u{args.ulysses_degree}_r{args.ring_degree}.png")
            overlap_image.save(f"{sample_dir}/overlap_image_u{args.ulysses_degree}_r{args.ring_degree}.png")

    assert_close(output_overlap_1, output_overlap_2, atol=1e-3, rtol=1e-3)
    assert_close(output, output_overlap_1, atol=1e-3, rtol=1e-3)

    assert not torch.isnan(output_overlap_1_inline).any()
    assert_close(output, output_overlap_1_inline, atol=1e-3, rtol=1e-3)

    assert not torch.isnan(output_overlap_1_inline_async).any()
    assert_close(output, output_overlap_1_inline_async, atol=1e-3, rtol=1e-3)

    assert not torch.isnan(output_overlap_2_inline).any()

    assert not torch.isnan(output_overlap_2_inline_async).any()

    assert_close(output_overlap_1_inline, output_overlap_2_inline, atol=1e-3, rtol=1e-3)

    assert_close(
        output_overlap_1_inline_async,
        output_overlap_2_inline_async,
        atol=1e-3,
        rtol=1e-3,
    )

    # dist.barrier()
    get_runtime_state().destory_distributed_env()


"""
Dominant shapes from OpenVid-1M: 1280x720, 1920x1080, 1920x960

Example usages:
# check correctness
CUDA_VISIBLE_DEVICES=0,1 python tests/test_overlap_perf_correctness.py --ulysses_degree 2 --ring_degree 1 --height 512 --width 512 --num_inference_steps 1 --repeat 1 --warmup_steps 1
# check performance traces
export nsys_args="--force-overwrite true -w true -s cpu --python-backtrace=cuda --capture-range=cudaProfilerApi --cudabacktrace=all:500 -x true"
CUDA_VISIBLE_DEVICES=0,1,2,3 nsys profile ${nsys_args} -o log/tests/test_overlap_bs4_u4_r1_h1024_w1024_$(date +"%m_%d")_$(git rev-parse --short HEAD) python tests/test_overlap_perf_correctness.py --ulysses_degree 4 --ring_degree 1 --height 1024 --width 1024 --num_inference_steps 2 --repeat 1 -b 4


# check performance without nsys
CUDA_VISIBLE_DEVICES=0,1 python tests/test_overlap_perf_correctness.py --ulysses_degree 2 --ring_degree 1 --height 1024 --width 1024 --num_inference_steps 8 --repeat 3
CUDA_VISIBLE_DEVICES=0,1,2,3 python tests/test_overlap_perf_correctness.py --ulysses_degree 4 --ring_degree 1 --height 1024 --width 1024 --num_inference_steps 8 --repeat 4 -b 4
CUDA_VISIBLE_DEVICES=4,5,6,7 python tests/test_overlap_perf_correctness.py --ulysses_degree 4 --ring_degree 1 --height 2048 --width 2048 --num_inference_steps 8 --repeat 4 -b 2
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
    parser.add_argument("--warmup_steps", type=int, default=1)
    parser.add_argument("--no_stream", action="store_true", default=False)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--output_json", type=str, default=None, help="Path to output JSON file")
    parser.add_argument("--save_image", action="store_true", default=False)
    parser.add_argument("--torch_prof", action="store_true", default=False, help="Use torch profiler for async inline")
    parser.add_argument("--pack_qkv", action="store_true", default=False, help="Use torch profiler for async inline")
    parser.add_argument(
        "--compile",
        action="store_true",
        default=False,
        help="Use torch compile. If False, will disable for the entire forward",
    )
    args = parser.parse_args()
    assert (
        args.ulysses_degree > 0 or args.ring_degree > 0
    ), "Either ulysses_degree or ring_degree must be greater than 0"

    from torch.multiprocessing import spawn

    nprocs = args.ulysses_degree * args.ring_degree
    spawn(
        test_overlap_perf_correctness,
        args=(
            nprocs,
            args,
            random.randint(8000, 65535),
        ),
        nprocs=nprocs,
    )
