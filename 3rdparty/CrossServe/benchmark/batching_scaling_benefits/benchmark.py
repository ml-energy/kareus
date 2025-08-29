import os
import time
import torch
import torch.distributed
from cfuser import cFuserFluxPipeline
from cfuser.core.distributed import (
    get_world_group,
    get_runtime_state,
    init_distributed_environment,
)

from cfuser.config import (
    EngineConfig,
    ModelConfig,
    RuntimeConfig,
    ParallelConfig,
    SequenceParallelConfig,
    InputConfig,
)
import argparse
from typing import Union
from cfuser.logger import init_logger
from cfuser.core.utils import dimension_generator, any_dimension_generator
import json

logger = init_logger(__name__)


def result_writer(
    model: str,
    batch_size: int,
    height: int,
    width: int,
    single_latent_seq_len: int,
    ulysses_degree: int,
    ring_degree: int,
    use_compile: bool,
    num_inference_steps: int,
    output_type: str,
    device: str,
    dtype: Union[torch.dtype, str],
    duration: float,
    peak_memory: float,
    notes: str = "",
):
    result_dict = {
        "model": model,
        "batch_size": batch_size,
        "height": height,
        "width": width,
        "single_latent_seq_len": single_latent_seq_len,
        "ulysses_degree": ulysses_degree,
        "ring_degree": ring_degree,
        "torch_compile": use_compile,
        "num_inference_steps": num_inference_steps,
        "output_type": output_type,
        "device": device,
        "dtype": str(dtype),
        "duration(s)": duration,
        "peak_memory(GB)": peak_memory,
        "notes": notes,
    }
    print(result_dict)
    return result_dict


def save_result(ulysses_degree, ring_degree, use_compile, result_dict):
    os.makedirs("benchmark/benchmark_data/batching_scaling_benefits", exist_ok=True)
    result_filename = f"benchmark/benchmark_data/batching_scaling_benefits/benchmark_u{ulysses_degree}_r{ring_degree}_c{use_compile}.json"
    with open(result_filename, "w") as f:
        json.dump(result_dict, f, indent=2)
    logger.info(f"All results saved to {result_filename}")


def single_run(pipe, input_config, timeout_seconds=300):  # 5 minutes default timeout
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


def warmup(pipe, input_config, times=1, timeout_seconds=300):
    for _ in range(times):
        single_run(pipe, input_config, timeout_seconds)


def check_result_exists(ulysses_degree, ring_degree, use_compile):
    result_filename = f"benchmark/benchmark_data/batching_scaling_benefits/benchmark_u{ulysses_degree}_r{ring_degree}_c{use_compile}.json"
    if os.path.exists(result_filename):
        with open(result_filename, "r") as f:
            return json.load(f)
    return None


def config_exists_in_results(
    results,
    batch_size,
    height,
    width,
    ulysses_degree,
    ring_degree,
    use_compile,
    device,
    model,
):
    for result in results:
        if (
            result["batch_size"] == batch_size
            and result["height"] == height
            and result["width"] == width
            and result["ulysses_degree"] == ulysses_degree
            and result["ring_degree"] == ring_degree
            and result["torch_compile"] == use_compile
            and result["device"] == device
            and result["model"] == model
        ):
            return True
    return False


def perf_batching_scaling_benefits(
    ulysses_degree: int = 1,
    ring_degree: int = 1,
    use_compile: bool = False,
    dtype=torch.bfloat16,  # bf16 seems a better default; ppl have reported range clipping issues with fp16 (https://github.com/huggingface/diffusers/pull/9097#issuecomment-2272292516)
):
    # Load existing results if they exist
    all_results = check_result_exists(ulysses_degree, ring_degree, use_compile) or []
    if all_results:
        logger.info(f"Loaded {len(all_results)} existing results")

    def initialize_environment():
        init_distributed_environment()
        local_rank = get_world_group().local_rank
        rank = get_world_group().rank
        world_size = get_world_group().world_size
        torch.cuda.set_device(rank)
        return local_rank, rank, world_size

    def create_pipeline(local_rank):
        engine_config = EngineConfig(
            model_config=ModelConfig(
                model="black-forest-labs/FLUX.1-dev",
            ),
            runtime_config=RuntimeConfig(
                use_torch_compile=use_compile,
            ),
            parallel_config=ParallelConfig(
                sp_config=SequenceParallelConfig(
                    ulysses_degree=ulysses_degree,
                    ring_degree=ring_degree,
                )
            ),
        )

        pipe = cFuserFluxPipeline.from_pretrained(
            pretrained_model_name_or_path=engine_config.model_config.model,
            engine_config=engine_config,
            torch_dtype=dtype,
        ).to(f"cuda:{local_rank}")

        return pipe, engine_config

    local_rank, rank, world_size = initialize_environment()
    pipe, engine_config = create_pipeline(local_rank)

    def remake_pipeline(pipe, local_rank):
        # TODO(@lry89757): I planned to use this function to remake the pipeline when the batch size is too large, but it's not working.
        logger.info("Remaking pipeline......")
        get_runtime_state().destory_distributed_env()
        del pipe
        torch.cuda.empty_cache()
        time.sleep(5)
        local_rank, rank, world_size = initialize_environment()
        pipe, engine_config = create_pipeline(local_rank)
        return pipe, engine_config

    if use_compile:
        torch._dynamo.config.accumulated_cache_size_limit = 256

    custom_batches = (1, 2, 4, 8, 16, 32, 64)

    custom_resolutions = {
        "360p": 360,
        "720p": 720,
        "512p": 512,
        "1024p": 1024,
        "2048p": 2048,
        "4096p": 4096,
    }
    custom_ratios = {"1:1": 1, "16:9": 16 / 9, "4:3": 4 / 3}

    for batch_size, width, height in dimension_generator(
        batch_sizes=custom_batches,
        resolution_map=custom_resolutions,
        ratio_map=custom_ratios,
    ):

        if local_rank == 0:
            print(f"Batch Size: {batch_size}, Dimensions: {width}x{height}")
            print(f"Saved {len(all_results)} results")
            save_result(ulysses_degree, ring_degree, use_compile, all_results)

        device_name = f"{torch.cuda.get_device_name(f'cuda:{local_rank}')}"

        # Skip if this configuration already exists
        if config_exists_in_results(
            all_results,
            batch_size,
            height,
            width,
            ulysses_degree,
            ring_degree,
            use_compile,
            device_name,
            engine_config.model_config.model,
        ):
            if local_rank == 0:
                logger.info(
                    f"Skipping existing configuration: batch={batch_size}, "
                    f"{width}x{height}, u{ulysses_degree}_r{ring_degree}_c{use_compile}"
                )
            continue

        if use_compile:
            torch._dynamo.reset()

        input_config = InputConfig(
            height=height,
            width=width,
            prompt=["a delicate apple made of opal hung on branch in the early morning light"] * batch_size,
            num_inference_steps=10,
            output_type="latent",
            seed=42,
        )

        try:
            input_config.num_inference_steps = 3
            warmup(pipe, input_config)
            input_config.num_inference_steps = 10
        except Exception as e:
            if local_rank == 0:
                logger.error(f"Error in warmup: {e}")

            if local_rank == 0:
                result_dict = result_writer(
                    model=engine_config.model_config.model,
                    batch_size=batch_size,
                    height=height,
                    width=width,
                    single_latent_seq_len=(int(height) // pipe.vae_scale_factor)
                    * (int(width) // pipe.vae_scale_factor),
                    ulysses_degree=ulysses_degree,
                    ring_degree=ring_degree,
                    use_compile=use_compile,
                    num_inference_steps=input_config.num_inference_steps,
                    output_type=input_config.output_type,
                    device=f"{torch.cuda.get_device_name(f'cuda:{local_rank}')}",
                    dtype=dtype,
                    duration=None,
                    peak_memory=None,
                    notes=f"Error {e}",
                )
                print(result_dict)
                all_results.append(result_dict)
                save_result(ulysses_degree, ring_degree, use_compile, all_results)

            if "`height` and `width` have to be divisible by 8" in str(e):
                continue

            logger.info("please check the log file for more details and rerun the benchmark with a smaller batch size")
            get_runtime_state().destory_distributed_env()
            return

        try:
            torch.cuda.reset_peak_memory_stats()
            start_time = time.time()
            output = single_run(pipe, input_config)
            end_time = time.time()

            elapsed_time = end_time - start_time
            peak_memory = torch.cuda.max_memory_allocated(device=f"cuda:{local_rank}")

            if local_rank == 0:
                logger.info(f"Execution time: {elapsed_time:.2f} seconds")
                logger.info(f"Peak Activation Memory Usage: {peak_memory / 1024**3:.2f} GB")

            if local_rank == 0:
                result_dict = result_writer(
                    model=engine_config.model_config.model,
                    batch_size=batch_size,
                    height=height,
                    width=width,
                    single_latent_seq_len=(int(height) // pipe.vae_scale_factor)
                    * (int(width) // pipe.vae_scale_factor),
                    ulysses_degree=ulysses_degree,
                    ring_degree=ring_degree,
                    use_compile=use_compile,
                    num_inference_steps=input_config.num_inference_steps,
                    output_type=input_config.output_type,
                    device=f"{torch.cuda.get_device_name(f'cuda:{local_rank}')}",
                    dtype=dtype,
                    duration=elapsed_time,
                    peak_memory=f"{peak_memory / 1024**3:.2f}",
                    notes="",
                )
                print(result_dict)
                all_results.append(result_dict)

        except Exception as e:
            if local_rank == 0:
                logger.error(f"Error in profiling: {e}")
                result_dict = result_writer(
                    model=engine_config.model_config.model,
                    batch_size=batch_size,
                    height=height,
                    width=width,
                    single_latent_seq_len=(int(height) // pipe.vae_scale_factor)
                    * (int(width) // pipe.vae_scale_factor),
                    ulysses_degree=ulysses_degree,
                    ring_degree=ring_degree,
                    use_compile=use_compile,
                    num_inference_steps=input_config.num_inference_steps,
                    output_type=input_config.output_type,
                    device=f"{torch.cuda.get_device_name(f'cuda:{local_rank}')}",
                    dtype=dtype,
                    duration=None,
                    peak_memory=None,
                    notes=f"Error {e}",
                )
                print(result_dict)
                all_results.append(result_dict)
                save_result(ulysses_degree, ring_degree, use_compile, all_results)

            if "`height` and `width` have to be divisible by 8" in str(e):
                continue

            logger.info("please check the log file for more details and rerun the benchmark with a smaller batch size")
            get_runtime_state().destory_distributed_env()
            return

    # Save all results at the end
    if local_rank == 0:
        save_result(ulysses_degree, ring_degree, use_compile, all_results)

    get_runtime_state().destory_distributed_env()


"""
python benchmark/batching_scaling_benefits/launcher.py

torchrun --standalone --nproc_per_node 4 benchmark/batching_scaling_benefits/benchmark.py --ulysses_degree 4 --ring_degree 1 --use_compile
"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ulysses_degree", type=int, default=1)
    parser.add_argument("--ring_degree", type=int, default=1)
    parser.add_argument("--use_compile", action="store_true", default=False)
    args = parser.parse_args()

    perf_batching_scaling_benefits(
        ulysses_degree=args.ulysses_degree,
        ring_degree=args.ring_degree,
        use_compile=args.use_compile,
    )
