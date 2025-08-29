import logging
import time
import torch
import torch.distributed
from cfuser import cFuserFluxPipeline, cFuserArgs
from cfuser.config import FlexibleArgumentParser
from cfuser.core.distributed import (
    get_world_group,
    get_runtime_state,
)

from cfuser.logger import init_logger

import torch.distributed as dist

from cfuser.core.utils.memory_profiler import (
    start_record_memory_history,
    stop_record_memory_history,
    export_memory_snapshot,
)

import os

logger = init_logger(__name__)


def main():
    parser = FlexibleArgumentParser(description="cFuser Arguments")
    args = cFuserArgs.add_cli_args(parser).parse_args()
    engine_args = cFuserArgs.from_cli_args(args)
    engine_config, input_config = engine_args.create_config()
    local_rank = get_world_group().local_rank

    start_record_memory_history()

    pipe = cFuserFluxPipeline.from_pretrained(
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

    os.environ["MEMORY_SNAPSHOT_FILE_PREFIX"] = (
        f"u{engine_args.ulysses_degree}_r{engine_args.ring_degree}_bs{engine_args.batch_size}_h{input_config.height}_w{input_config.width}_rank{local_rank}"
    )

    print(f"input config: {input_config}")

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

    stop_record_memory_history()

    if local_rank == 0:
        export_memory_snapshot(
            f"u{engine_args.ulysses_degree}_r{engine_args.ring_degree}_bs{engine_args.batch_size}_h{input_config.height}_w{input_config.width}_rank{local_rank}"
        )

    export_memory_snapshot()

    parallel_info = f"ulysses{engine_args.ulysses_degree}_ring{engine_args.ring_degree}"

    get_runtime_state().destory_distributed_env()


"""
NCCL_DEBUG=WARN CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=1049 benchmark/memory_profile/cserve_memory.py --prompt 'hello world' --output_type 'latent' --model "black-forest-labs/FLUX.1-dev" \
--batch_size 4 --height 4096 --width 4096 --ulysses_degree 4 --ring_degree 1 --num_inference_steps 3
"""

if __name__ == "__main__":
    main()
