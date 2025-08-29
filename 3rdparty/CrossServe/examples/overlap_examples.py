import time
import torch
import torch.distributed
from copy import deepcopy
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

from cfuser.logger import init_logger
from cfuser.core.utils import nvtx_range

logger = init_logger(__name__)


def single_run(pipe, input_config, input_config_2=None, inline_inference=False, async_op=False):
    if input_config_2 is None:
        output = pipe(
            height=input_config.height,
            width=input_config.width,
            prompt=input_config.prompt,
            output_type=input_config.output_type,
            num_inference_steps=input_config.num_inference_steps,
            max_sequence_length=256,
            guidance_scale=0.0,
            generator=torch.Generator(device="cuda").manual_seed(input_config.seed),
            return_dict=False,
        )
    else:
        output = pipe.inference(
            input_config=input_config,
            input_config_2=input_config_2,
            inline_inference=inline_inference,
            async_op=async_op,
        )

    return output


def warmup(pipe, input_config, times=1, num_inference_steps=10):
    input_config_for_warmup = deepcopy(input_config)
    input_config_for_warmup.num_inference_steps = num_inference_steps
    for _ in range(times):
        single_run(pipe, input_config_for_warmup)


def perf_overlap_benefits(
    height: int = 1024,
    width: int = 1024,
    batch_size: int = 2,
    ulysses_degree: int = 1,
    ring_degree: int = 1,
    use_overlap: bool = False,
    use_inline_inference: bool = False,
    async_op: bool = False,
    use_compile: bool = False,
):
    assert batch_size % 2 == 0

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
            torch_dtype=torch.bfloat16,
        ).to(f"cuda:{local_rank}")

        return pipe, engine_config

    local_rank, rank, world_size = initialize_environment()
    pipe, engine_config = create_pipeline(local_rank)

    input_config = InputConfig(
        height=height,
        width=width,
        prompt=["a beautiful image"] * batch_size,
        num_inference_steps=10,
        output_type="latent",
        seed=42,
    )

    if use_overlap:
        input_config_2 = deepcopy(input_config)
    else:
        input_config_2 = None

    with nvtx_range("warmup"):
        warmup(pipe, input_config, times=1, num_inference_steps=20)

    torch.cuda.reset_peak_memory_stats()
    start_time = time.time()
    with nvtx_range("single_run"):
        output = single_run(
            pipe,
            input_config,
            input_config_2=input_config_2,
            inline_inference=use_inline_inference,
            async_op=async_op,
        )
    end_time = time.time()
    elapsed_time = end_time - start_time
    peak_memory = torch.cuda.max_memory_allocated(device=f"cuda:{local_rank}")

    if get_world_group().rank == 0:
        print(f"epoch time: {elapsed_time:.2f} sec, memory: {peak_memory/1e9} GB")
    get_runtime_state().destory_distributed_env()


"""
CUDA_VISIBLE_DEVICES=4,5,6,7 nsys profile \
    -w true -t cuda,nvtx,osrt,cudnn,cublas -s cpu  \
    --capture-range=cudaProfilerApi --cudabacktrace=true -x true \
    --force-overwrite true \
    --output log/benchmark/overlap_benefits/benchmark_overlap_benefits_bs2_ulysses4_ring1_overlap_true_inline_inference_true_async_op_true \
    torchrun --standalone --nproc_per_node 4 examples/overlap_examples.py --ulysses_degree 4 --ring_degree 1 --height 1024 --width 1024 --batch_size 2 --use_overlap --use_inline_inference --async_op
"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ulysses_degree", type=int, default=1)
    parser.add_argument("--ring_degree", type=int, default=1)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_inference_steps", type=int, default=10)
    parser.add_argument("--use_inline_inference", action="store_true", default=False)
    parser.add_argument("--async_op", action="store_true", default=False)
    parser.add_argument("--use_overlap", action="store_true", default=False)
    args = parser.parse_args()

    perf_overlap_benefits(
        height=args.height,
        width=args.width,
        batch_size=args.batch_size,
        ulysses_degree=args.ulysses_degree,
        ring_degree=args.ring_degree,
        async_op=args.async_op,
        use_overlap=args.use_overlap,
        use_inline_inference=args.use_inline_inference,
    )
