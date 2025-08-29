import logging
import os
import time
import torch
import torch.distributed
from cfuser import cFuserFluxPipeline, cFuserArgs
from cfuser.config import FlexibleArgumentParser
from cfuser.core.distributed import (
    get_world_group,
    get_runtime_state,
)
from torch.profiler import profile, record_function, ProfilerActivity
import signal
from contextlib import contextmanager

from cfuser.logger import init_logger
from cfuser.core.utils import dimension_generator, any_dimension_generator

logger = init_logger(__name__)


class TimeoutException(Exception):
    pass


@contextmanager
def timeout(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")

    # Register a function to raise a TimeoutException on the signal
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)

    try:
        yield
    finally:
        # Disable the alarm
        signal.alarm(0)


def single_run(pipe, input_config, timeout_seconds=300):  # 5 minutes default timeout
    try:
        with timeout(timeout_seconds):
            output = pipe(
                height=input_config.height,
                width=input_config.width,
                prompt=input_config.prompt,
                output_type=input_config.output_type,
                num_inference_steps=input_config.num_inference_steps,
                max_sequence_length=256,
                guidance_scale=0.0,
                generator=torch.Generator(device="cuda").manual_seed(input_config.seed),
            )
            return output
    except TimeoutException:
        logger.error(f"Pipeline execution timed out after {timeout_seconds} seconds")


def warmup(pipe, input_config, times=1, timeout_seconds=300):
    for _ in range(times):
        single_run(pipe, input_config, timeout_seconds)


def main():
    parser = FlexibleArgumentParser(description="cFuser Arguments")
    parser.add_argument("--gpu_count", type=int, default=4, help="Number of GPUs to use (default: 4)")
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Timeout in seconds for each run (default: 300)",
    )
    args = cFuserArgs.add_cli_args(parser).parse_args()
    engine_args = cFuserArgs.from_cli_args(args)
    engine_config, input_config = engine_args.create_config()
    local_rank = get_world_group().local_rank

    torch.cuda.empty_cache()
    free_memory, total_memory = torch.cuda.mem_get_info()
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

    free_memory_after_init, total_memory_after_init = torch.cuda.mem_get_info()
    if local_rank == 0:
        logger.info(f"GPU Memory Total: {total_memory / 1024**3} GB")
        logger.info(f"Memory of Model Init: {(free_memory - free_memory_after_init) / 1024**3} GB")

    ulysses_degree = [1, 2, 4, 8]
    # ring_degree = [1, 2, 4, 8]
    ring_degree = [1]
    # ring_degree = [2, 4]

    for u, r in any_dimension_generator([ulysses_degree, ring_degree]):

        if u * r != args.gpu_count:
            continue

        print(f"Ulysses Degree: {u}, Ring Degree: {r}")

        pipe.change_runtime_config(ulysses_degree=u, ring_degree=r)
        engine_config = pipe.engine_config

        # custom_batches = (1, 2, 4, 8, 16, 32)
        custom_batches = (16, 32, 64)
        custom_resolutions = {
            "360p": 360,
            "720p": 720,
            "512p": 512,
            "1024p": 1024,
            "2048p": 2048,
            "4096p": 4096,
        }
        custom_ratios = {"1:1": 1}

        for batch_size, width, height in dimension_generator(
            batch_sizes=custom_batches,
            resolution_map=custom_resolutions,
            ratio_map=custom_ratios,
        ):
            torch._dynamo.reset()

            if local_rank == 0:
                print(f"Batch Size: {batch_size}, Dimensions: {width}x{height}")

            input_config.batch_size = batch_size
            prompt = ["Draw"] * batch_size
            input_config.prompt = prompt
            input_config.height = height
            input_config.width = width

            folder = (
                f"./benchmark/benchmark_data/pytorch_profiler/ulysses_{engine_config.parallel_config.ulysses_degree}_"
            )
            folder += f"ring_{engine_config.parallel_config.ring_degree}/"
            os.makedirs(folder, exist_ok=True)

            json_file = (
                folder
                + f"cfuser_flux_trace_steps_{input_config.num_inference_steps}_rank_{local_rank}_bs{batch_size}_height{input_config.height}_width{input_config.width}.json"
            )

            if args.use_torch_compile:
                json_file = json_file.replace(".json", "_torch_compile.json")

            if os.path.exists(json_file):
                if local_rank == 0:
                    logger.info(f"Skipping {json_file} because it already exists")
                continue
            else:
                if local_rank == 0:
                    logger.info(f"Running {json_file}")

            try:
                # Warmup
                warmup(pipe, input_config, timeout_seconds=args.timeout)

            except TimeoutException:
                if local_rank == 0:
                    logger.error("Timeout during warmup, skipping this configuration")
                continue
            except Exception as e:
                if local_rank == 0:
                    logger.error(f"Error in warmup: {e}")
                continue

            try:
                # Modify the profiling section to use the new argument
                if args.use_profiler:
                    with profile(
                        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                        profile_memory=True,
                        with_stack=True,
                        with_flops=True,
                        with_modules=True,
                        record_shapes=True,
                    ) as prof:
                        with record_function("cfuser_flux_pipeline"):
                            torch.cuda.reset_peak_memory_stats()
                            start_time = time.time()
                            output = single_run(pipe, input_config, args.timeout)
                            end_time = time.time()
                else:
                    torch.cuda.reset_peak_memory_stats()
                    start_time = time.time()
                    output = single_run(pipe, input_config, args.timeout)
                    end_time = time.time()

                elapsed_time = end_time - start_time
                peak_memory = torch.cuda.max_memory_allocated(device=f"cuda:{local_rank}")

                if local_rank == 0:
                    logger.info(f"Execution time: {elapsed_time:.2f} seconds")
                    logger.info(f"Peak Activation Memory Usage: {peak_memory / 1024**3:.2f} GB")

                if args.use_profiler:
                    # Export Chrome trace
                    if local_rank == 0:
                        prof.export_chrome_trace(json_file)
                    # Print key averages
                    if local_rank == 0:
                        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
            except TimeoutException:
                if local_rank == 0:
                    logger.error(f"Pipeline execution timed out")
                continue
            except Exception as e:
                if local_rank == 0:
                    logger.error(f"Error in profiling: {e}")
                continue

    get_runtime_state().destory_distributed_env()


"""
NCCL_DEBUG=WARN CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 --master_port=1036 benchmark/pytorch_profiler/flux_profiler.py --use_profiler --gpu_count 8 --ulysses_degree 1 --ring_degree 8 \
--output_type 'latent' --model "black-forest-labs/FLUX.1-dev" --num_inference_steps 3 2>&1 | tee log_gpu_count_8.log

NCCL_DEBUG=WARN CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 --master_port=1039 benchmark/pytorch_profiler/flux_profiler.py --use_profiler --gpu_count 4 --ulysses_degree 4 --ring_degree 1 --timeout 1200 \
--height 4096 --width 4096 --output_type 'latent' --model "black-forest-labs/FLUX.1-dev" --num_inference_steps 3 2>&1 | tee log_gpu_count_4.log

NCCL_DEBUG=WARN CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node=2 --master_port=1038 benchmark/pytorch_profiler/flux_profiler.py --use_profiler --gpu_count 2 --ulysses_degree 2 --ring_degree 1 \
--output_type 'latent' --model "black-forest-labs/FLUX.1-dev" --num_inference_steps 3 2>&1 | tee log_gpu_count_2.log

NCCL_DEBUG=WARN CUDA_VISIBLE_DEVICES=1 torchrun --nproc_per_node=1 --master_port=1037 benchmark/pytorch_profiler/flux_profiler.py --use_profiler --gpu_count 1 --ulysses_degree 1 --ring_degree 1 \
--output_type 'latent' --model "black-forest-labs/FLUX.1-dev" --num_inference_steps 3 2>&1 | tee log_gpu_count_1.log

# torch compile
NCCL_DEBUG=WARN CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 --master_port=1036 benchmark/pytorch_profiler/flux_profiler.py --use_profiler --gpu_count 8 --ulysses_degree 1 --ring_degree 8 \
--output_type 'latent' --model "black-forest-labs/FLUX.1-dev" --num_inference_steps 3 --use_torch_compile 2>&1 | tee log_gpu_count_8_torch_compile.log

NCCL_DEBUG=WARN CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=1040 benchmark/pytorch_profiler/flux_profiler.py --use_profiler --gpu_count 4 --ulysses_degree 4 --ring_degree 1 --timeout 1200 \
--height 4096 --width 4096 --output_type 'latent' --model "black-forest-labs/FLUX.1-dev" --num_inference_steps 3 --use_torch_compile 2>&1 | tee log_gpu_count_4_torch_compile.log

NCCL_DEBUG=WARN CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=1037 benchmark/pytorch_profiler/flux_profiler.py --use_profiler --gpu_count 2 --ulysses_degree 2 --ring_degree 1 \
--output_type 'latent' --model "black-forest-labs/FLUX.1-dev" --num_inference_steps 3 --use_torch_compile 2>&1 | tee log_gpu_count_2_torch_compile.log

NCCL_DEBUG=WARN CUDA_VISIBLE_DEVICES=4 torchrun --nproc_per_node=1 --master_port=2037 benchmark/pytorch_profiler/flux_profiler.py --use_profiler --gpu_count 1 --ulysses_degree 1 --ring_degree 1 \
--output_type 'latent' --model "black-forest-labs/FLUX.1-dev" --num_inference_steps 3 --use_torch_compile 2>&1 | tee log_gpu_count_1_torch_compile.log
"""

if __name__ == "__main__":
    main()
