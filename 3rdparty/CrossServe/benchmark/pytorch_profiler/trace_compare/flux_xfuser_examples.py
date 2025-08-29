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

from viztracer import VizTracer
from torch.profiler import profile, record_function, ProfilerActivity

logger = init_logger(__name__)


def main():
    parser = FlexibleArgumentParser(description="xFuser Arguments")
    parser.add_argument("--batch_size", type=int, default=1)
    args = xFuserArgs.add_cli_args(parser).parse_args()

    engine_args = xFuserArgs.from_cli_args(args)
    engine_config, input_config = engine_args.create_config()
    local_rank = get_world_group().local_rank

    pipe = xFuserFluxPipeline.from_pretrained(
        pretrained_model_name_or_path=engine_config.model_config.model,
        engine_config=engine_config,
        torch_dtype=torch.bfloat16,
    )

    if args.enable_sequential_cpu_offload:
        pipe.enable_sequential_cpu_offload(gpu_id=local_rank)
        logging.info(f"rank {local_rank} sequential CPU offload enabled")
    else:
        pipe = pipe.to(f"cuda:{local_rank}")

    logger.info(f"logger init over")

    # viztracer = VizTracer()
    # viztracer.start()
    prof = profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    )

    input_config.batch_size = args.batch_size
    input_config.prompt = ["hello world"] * args.batch_size

    for i in range(3):
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

    prof.start()
    # print(f"input_config: {input_config}")
    torch.cuda.reset_peak_memory_stats()
    start_time = time.time()
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

    prof.stop()
    if get_world_group().rank == 0:
        prof.export_chrome_trace(
            f"./benchmark/benchmark_data/pytorch_profiler/trace_compare/xfuser_flux_trace_{parallel_info}_bs_{input_config.batch_size}_height_{input_config.height}_width_{input_config.width}.json"
        )

    # viztracer.stop()
    # if get_world_group().rank == 0:
    #     viztracer.save(f"./benchmark/benchmark_data/pytorch_profiler/trace_compare/xfuser_flux_trace_{parallel_info}_bs_{input_config.batch_size}_height_{input_config.height}_width_{input_config.width}.json")

    if input_config.output_type == "pil":
        if get_world_group().rank == 0:
            for i, image in enumerate(output.images):
                image_rank = i
                image_name = f"xfuser_flux_result_{parallel_info}_{image_rank}_tc_{engine_args.use_torch_compile}.png"
                image.save(f"./results/{image_name}")
                print(f"image {i} saved to ./results/{image_name}")

    if get_world_group().rank == 0:
        print(f"epoch time: {elapsed_time:.2f} sec, memory: {peak_memory/1e9} GB")
    get_runtime_state().destory_distributed_env()


"""
NCCL_DEBUG=WARN CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=2 --master_port=1037 benchmark/pytorch_profiler/trace_compare/flux_xfuser_examples.py --prompt 'hello world' --output_type 'latent' --model "black-forest-labs/FLUX.1-dev" \
--batch_size 4 --height 2048 --width 2048 --ulysses_degree 2 --ring_degree 1 --num_inference_steps 1
"""

if __name__ == "__main__":
    main()
