import os
import logging
import time
import torch
import torch.distributed
from xfuser import xFuserFluxPipeline, xFuserArgs
from xfuser.config import FlexibleArgumentParser
from xfuser.core.distributed import (
    get_world_group,
    get_runtime_state,
)

from xfuser.logger import init_logger

from torch.profiler import profile, record_function, ProfilerActivity

logger = init_logger(__name__)


def single_run(pipe, input_config):
    output = pipe(
        height=input_config.height,
        width=input_config.height,
        prompt=input_config.prompt,
        num_inference_steps=input_config.num_inference_steps,
        output_type=input_config.output_type,
        max_sequence_length=256,
        guidance_scale=0.0,
        generator=torch.Generator(device="cuda").manual_seed(input_config.seed),
    )
    return output


def create_json_file(engine_config, input_config, local_rank):
    folder = f"./benchmark/benchmark_data/pytorch_profiler/ulysses_{engine_config.parallel_config.ulysses_degree}_"
    folder += f"ring_{engine_config.parallel_config.ring_degree}/"
    os.makedirs(folder, exist_ok=True)

    json_file = (
        folder
        + f"xfuser_flux_trace_steps_{input_config.num_inference_steps}_rank_{local_rank}_bs{input_config.batch_size}_height{input_config.height}_width{input_config.width}.json"
    )
    if os.path.exists(json_file):
        if local_rank == 0:
            logger.info(f"Skipping {json_file} because it already exists")
        return False, json_file
    else:
        if local_rank == 0:
            logger.info(f"Running {json_file}")
        return True, json_file


def main():
    parser = FlexibleArgumentParser(description="xfuser Arguments")
    args = xFuserArgs.add_cli_args(parser).parse_args()
    print(args)
    engine_args = xFuserArgs.from_cli_args(args)
    engine_config, input_config = engine_args.create_config()
    local_rank = get_world_group().local_rank

    # bs_list = [1, 2, 4, 8, 16, 32, 64]
    bs_list = [1, 2, 4, 8, 32, 64]

    file_created = False
    for bs in bs_list:
        input_config.batch_size = bs
        input_config.prompt = ["Draw how UMich looks like"] * bs
        file_created, json_file = create_json_file(engine_config, input_config, local_rank)
        if not file_created:
            continue
        else:
            break

    if not file_created:
        return

    pipe = xFuserFluxPipeline.from_pretrained(
        pretrained_model_name_or_path=engine_config.model_config.model,
        engine_config=engine_config,
        torch_dtype=torch.bfloat16,
    )

    logger.info(f"logger init over")

    if args.enable_sequential_cpu_offload:
        pipe.enable_sequential_cpu_offload(gpu_id=local_rank)
        logging.info(f"rank {local_rank} sequential CPU offload enabled")
    else:
        pipe = pipe.to(f"cuda:{local_rank}")

    torch.cuda.reset_peak_memory_stats()
    start_time = time.time()

    end_time = time.time()
    elapsed_time = end_time - start_time
    peak_memory = torch.cuda.max_memory_allocated(device=f"cuda:{local_rank}")

    parallel_info = f"ulysses{engine_args.ulysses_degree}_ring{engine_args.ring_degree}"
    logger.info(f"parallel_info: {parallel_info}")

    for bs in bs_list:
        input_config.batch_size = bs
        input_config.prompt = ["Draw how UMich looks like"] * bs

        file_created, json_file = create_json_file(engine_config, input_config, local_rank)
        if not file_created:
            continue

        # warmup
        single_run(pipe, input_config)

        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            profile_memory=True,
            with_stack=True,
            with_flops=True,
            with_modules=True,
            record_shapes=True,
        ) as prof:
            with record_function("xfuser_flux_pipeline"):
                output = single_run(pipe, input_config)

        if local_rank == 0:
            prof.export_chrome_trace(json_file)

        if input_config.output_type == "pil":
            if get_world_group().rank == 0:
                for i, image in enumerate(output.images):
                    image_rank = i
                    image_name = f"flux_result_{parallel_info}_{image_rank}_tc_{engine_args.use_torch_compile}.png"
                    image.save(f"./results/{image_name}")
                    print(f"image {i} saved to ./results/{image_name}")

        if get_world_group().rank == 0:
            print(f"epoch time: {elapsed_time:.2f} sec, memory: {peak_memory/1e9} GB")
    get_runtime_state().destory_distributed_env()


"""
NCCL_DEBUG=WARN CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 --master_port=1049 examples/xfuser_examples.py --prompt 'hello world' --output_type 'latent' --model "black-forest-labs/FLUX.1-dev" \
--height 2048 --width 2048 --ulysses_degree 4 --ring_degree 1 --num_inference_steps 3

NCCL_DEBUG=WARN CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node=2 --master_port=1048 examples/xfuser_examples.py --prompt 'hello world' --output_type 'latent' --model "black-forest-labs/FLUX.1-dev" \
--height 2048 --width 2048 --ulysses_degree 2 --ring_degree 1 --num_inference_steps 3


NCCL_DEBUG=WARN CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 --master_port=1050 examples/xfuser_examples.py --prompt 'hello world' --output_type 'latent' --model "black-forest-labs/FLUX.1-dev" \
--height 2048 --width 2048 --ulysses_degree 8 --ring_degree 1 --num_inference_steps 3


NCCL_DEBUG=WARN CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node=2 --master_port=1048 examples/xfuser_examples.py --prompt 'hello world' --output_type 'latent' --model "black-forest-labs/FLUX.1-dev" \
--height 4096 --width 4096 --ulysses_degree 2 --ring_degree 1 --num_inference_steps 3

NCCL_DEBUG=WARN CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 --master_port=1049 examples/xfuser_examples.py --prompt 'hello world' --output_type 'latent' --model "black-forest-labs/FLUX.1-dev" \
--height 720 --width 720 --ulysses_degree 4 --ring_degree 1 --num_inference_steps 3


NCCL_DEBUG=WARN CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 --master_port=1050 examples/xfuser_examples.py --prompt 'hello world' --output_type 'latent' --model "black-forest-labs/FLUX.1-dev" \
--height 4096 --width 4096 --ulysses_degree 8 --ring_degree 1 --num_inference_steps 3
"""

if __name__ == "__main__":
    main()
