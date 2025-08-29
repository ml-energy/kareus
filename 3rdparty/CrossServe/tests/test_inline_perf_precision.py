# This script is for testing the correctness and performance of the inline inference
import os
import torch
import random

import time
import torch
import torch.distributed
from cfuser import cFuserFluxPipeline
from cfuser.testing import assert_close
from cfuser.config import (
    EngineConfig,
    ModelConfig,
    RuntimeConfig,
    ParallelConfig,
    SequenceParallelConfig,
    InputConfig,
)
from cfuser.core.distributed import (
    get_world_group,
    get_runtime_state,
    init_distributed_environment,
)

from cfuser.core.utils import (
    initialize_profiler,
    clear_profiler,
    print_time_distribution,
)

from cfuser.core.utils import nvtx_range

from cfuser.logger import init_logger

logger = init_logger(__name__)


def test_inline_perf_precision(
    rank,
    world_size,
    ulysses_degree: int = 1,
    ring_degree: int = 1,
    batch_size: int = 1,
    height: int = 1024,
    width: int = 1024,
    num_inference_steps: int = 10,
    repeat: int = 3,
    warmup_steps: int = 20,
    master_port: int = 12355,
):
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(master_port)

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
                ulysses_degree=ulysses_degree,
                ring_degree=ring_degree,
            )
        ),
    )

    input_config = InputConfig(
        height=height,
        width=width,
        prompt=["a beautiful image"] * batch_size,
        num_inference_steps=num_inference_steps,
        output_type="latent",
        seed=42,
    )

    local_rank = get_world_group().local_rank

    pipe = cFuserFluxPipeline.from_pretrained(
        pretrained_model_name_or_path=engine_config.model_config.model,
        engine_config=engine_config,
        torch_dtype=torch.bfloat16,
    ).to(f"cuda:{local_rank}")

    # warmup
    pipe.prepare_run(input_config, steps=warmup_steps)

    torch.cuda.cudart().cudaProfilerStart()

    torch.cuda.reset_peak_memory_stats()
    start_time = time.time()
    with nvtx_range("original inference"):
        for index_repeat in range(repeat):
            if index_repeat == 1:
                torch.cuda.cudart().cudaProfilerStart()
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
            )[0]
            if index_repeat == repeat - 1:
                torch.cuda.cudart().cudaProfilerStop()
    end_time = time.time()
    elapsed_time = end_time - start_time
    peak_memory = torch.cuda.max_memory_allocated(device=f"cuda:{local_rank}")

    # torch.cuda.reset_peak_memory_stats()
    # start_time = time.time()
    # with nvtx_range("inline inference"):
    #     for index_repeat in range(repeat):
    #         if index_repeat == 1:
    #             torch.cuda.cudart().cudaProfilerStart()
    #         output_inline = pipe.inference(
    #             input_config,
    #             inline_inference=True,
    #             return_dict=False,
    #             async_op=False,
    #         )[0]
    #         if index_repeat == repeat - 1:
    #             torch.cuda.cudart().cudaProfilerStop()
    # end_time = time.time()
    # elapsed_time_inline = end_time - start_time
    # peak_memory_inline = torch.cuda.max_memory_allocated(device=f"cuda:{local_rank}")

    # torch.cuda.reset_peak_memory_stats()
    # start_time = time.time()
    # with nvtx_range("inline async inference"):
    #     for index_repeat in range(repeat):
    #         if index_repeat == 1:
    #             torch.cuda.cudart().cudaProfilerStart()
    #         output_inline_async = pipe.inference(
    #             input_config,
    #             inline_inference=True,
    #             return_dict=False,
    #             async_op=True,
    #         )[0]
    #         if index_repeat == repeat - 1:
    #             torch.cuda.cudart().cudaProfilerStop()
    # end_time = time.time()
    # elapsed_time_inline_async = end_time - start_time
    # peak_memory_inline_async = torch.cuda.max_memory_allocated(device=f"cuda:{local_rank}")

    torch.cuda.reset_peak_memory_stats()
    start_time = time.time()
    with nvtx_range("inference_requests_batch inference"):
        for index_repeat in range(repeat):
            if index_repeat == 1:
                torch.cuda.cudart().cudaProfilerStart()
            output_inference_requests_batch = pipe.inference_requests_batch(
                [input_config],
                [torch.Generator(device="cuda").manual_seed(input_config.seed)],
            )[0]
            if index_repeat == repeat - 1:
                torch.cuda.cudart().cudaProfilerStop()
    end_time = time.time()
    elapsed_time_inference_requests_batch = end_time - start_time
    peak_memory_inference_requests_batch = torch.cuda.max_memory_allocated(device=f"cuda:{local_rank}")

    torch.cuda.cudart().cudaProfilerStop()

    parallel_info = f"ulysses{ulysses_degree}_ring{ring_degree}"

    if get_world_group().rank == 0:
        print(f"repeat: {repeat}")
        print(f"epoch time: {elapsed_time:.2f} sec, memory: {peak_memory/1e9} GB")
        # print(f"inline epoch time: {elapsed_time_inline:.2f} sec, memory: {peak_memory_inline/1e9} GB")
        # print(
        # f"inline async epoch time: {elapsed_time_inline_async:.2f} sec, memory: {peak_memory_inline_async/1e9} GB"
        # )
        print(
            f"inference_requests_batch epoch time: {elapsed_time_inference_requests_batch:.2f} sec, memory: {peak_memory_inference_requests_batch/1e9} GB"
        )

    # ensure no nan
    assert not torch.isnan(output).any()
    # assert not torch.isnan(output_inline).any()
    # assert not torch.isnan(output_inline_async).any()

    # assert_close(output, output_inline)
    # assert torch.allclose(output, output_inline)
    # assert_close(output, output_inline_async)
    # assert torch.allclose(output, output_inline_async)

    assert torch.allclose(output, output_inference_requests_batch)
    assert_close(output, output_inference_requests_batch)

    get_runtime_state().destory_distributed_env()


"""
# test correctness
## with stream
CUDA_VISIBLE_DEVICES=0,1,2,3 python tests/test_inline_perf_precision.py --ulysses_degree 4 --ring_degree 1 --height 512 --width 512 --num_inference_steps 3 --repeat 1 --warmup_steps 2
CUDA_VISIBLE_DEVICES=0,1,2,3 python tests/test_inline_perf_precision.py --ulysses_degree 4 --ring_degree 1 --height 2048 --width 2048 --num_inference_steps 3 --repeat 1 --warmup_steps 1 --batch_size 2 2>&1 | tee log.log

# test performance

CUDA_VISIBLE_DEVICES=0,1,2,3 nsys profile --force-overwrite true -w true -s cpu  --capture-range=cudaProfilerApi -o log/tests/test_inline_perf_precision_bs1_u4_r1_h1024_w1024 python tests/test_inline_perf_precision.py --ulysses_degree 4 --ring_degree 1 --height 1024 --width 1024 --num_inference_steps 8 --repeat 3 --batch_size 2

# test performance without nsys
## with stream
CUDA_VISIBLE_DEVICES=0,1,2,3 python tests/test_inline_perf_precision.py --ulysses_degree 4 --ring_degree 1 --height 1024 --width 1024 --num_inference_steps 8 --repeat 3 --batch_size 4

"""

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--ulysses_degree", type=int, default=1)
    parser.add_argument("--ring_degree", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--num_inference_steps", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--warmup_steps", type=int, default=20)
    args = parser.parse_args()

    from torch.multiprocessing import spawn

    nprocs = args.ulysses_degree * args.ring_degree

    spawn(
        test_inline_perf_precision,
        args=(
            nprocs,
            args.ulysses_degree,
            args.ring_degree,
            args.batch_size,
            args.height,
            args.width,
            args.num_inference_steps,
            args.repeat,
            args.warmup_steps,
            random.randint(8000, 65535),
        ),
        nprocs=nprocs,
    )
