import os
import json
import torch
import random
import time
from cfuser.engine.runner import CServeRunner
from cfuser.scheduler.request import ScheduledRequests, ScheduledRequest
from cfuser.core.distributed.parallel_state import set_runtime_config, PROCESS_GROUP
from cfuser.core.distributed import (
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


def init_runner(rank, world_size, master_port):
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
                ulysses_degree=world_size,
                ring_degree=1,
            )
        ),
    )

    runner = CServeRunner(pretrained_model_name_or_path=engine_config.model_config.model, engine_config=engine_config)

    return runner


def run(runner: CServeRunner, input_config: InputConfig, repeat: int, rank: int):
    req = ScheduledRequest(
        req_ids=[0],
        attn_ranks=PROCESS_GROUP.get_attn_ranks(index_req=0),
        non_attn_ranks=PROCESS_GROUP.get_non_attn_ranks(index_req=0),
        attn_ulysses_degree=len(PROCESS_GROUP.get_ulysses_ranks(index_req=0)),
        attn_ring_degree=len(PROCESS_GROUP.get_ring_ranks(index_req=0)),
        non_attn_sp_degree=len(PROCESS_GROUP.get_non_attn_ranks(index_req=0)),
        input_config=input_config,
    )

    reqs = ScheduledRequests(requests=[req])

    for _ in range(repeat):
        if rank in req.non_attn_ranks:
            runner.generate(reqs)


def run_test(rank, world_size, repeat, master_port, logging):
    runner = init_runner(rank, world_size, master_port)

    json_path = (
        f"log/benchmark/component_scaling_efficiency/e2e_scaling_efficiency/e2e_scaling_efficiency_{world_size}.json"
    )
    existing_data = []
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            existing_data = json.load(f)

    # Create all combinations and sort by their product
    combinations = []
    for batch_size in [1, 2, 4, 8, 16, 32]:
        for seq_len in [1024, 2048, 4096, 8192, 16384, 32768, 65536]:
            combinations.append((batch_size, seq_len, batch_size * seq_len))
        # seqlen 1024 -> height 512, width 512
        # seqlen 65536 -> height 4096, width 4096

    # Sort by the product (bs × seq_len)
    combinations.sort(key=lambda x: x[2])
    print(combinations)

    for batch_size, seq_len, _ in combinations:
        height = 512
        width = seq_len * 16 * 16 // height

        num_inference_steps = 10
        if seq_len * batch_size > 1 * 65536:
            num_inference_steps = 3

        input_config = InputConfig(
            batch_size=batch_size,
            height=height,
            width=width,
            prompt=["a beautiful river with flowers growing on a turtle's back"] * batch_size,
            num_inference_steps=num_inference_steps,
            output_type="latent",
            seed=42,
        )

        for ulysses_degree in [1, 2, 4, 8]:
            for ring_degree in [1, 2, 4, 8]:
                if ulysses_degree * ring_degree != world_size:
                    continue

                if logging:
                    skip = False
                    # if this item is already in the log, skip
                    for item in existing_data:
                        if (
                            item["bs"] == batch_size
                            and item["seq_len"] == seq_len
                            and item["ulysses_degree"] == ulysses_degree
                            and item["ring_degree"] == ring_degree
                        ):
                            print(
                                f"skip bs {batch_size}, seq_len {seq_len}, ulysses_degree {ulysses_degree}, ring_degree {ring_degree}, world_size {world_size}"
                            )
                            skip = True
                            break

                    if skip:
                        continue
                    else:
                        print(
                            f"run bs {batch_size}, seq_len {seq_len}, ulysses_degree {ulysses_degree}, ring_degree {ring_degree}, world_size {world_size}"
                        )

                set_runtime_config(
                    ranks=list(range(world_size)),
                    ulysses_degree=ulysses_degree,
                    ring_degree=ring_degree,
                    non_attn_sp_ranks=list(range(world_size)),
                    index_req=0,
                )

                # warm up
                run(runner, input_config, 2, rank)

                # benchmark
                start_time = time.perf_counter()
                run(runner, input_config, repeat, rank)
                end_time = time.perf_counter()
                duration = (end_time - start_time) / repeat
                if rank == 0:
                    print(
                        f"batch_size: {batch_size}, seq_len: {seq_len}, ulysses_degree: {ulysses_degree}, ring_degree: {ring_degree}, time: {duration:.2f}s"
                    )
                    if logging:
                        to_save = {
                            "bs": batch_size,
                            "seq_len": seq_len,
                            "ulysses_degree": ulysses_degree,
                            "ring_degree": ring_degree,
                            "mlp_world_size": world_size,
                            "num_iter": repeat,
                            "steps": num_inference_steps,
                            "avg_e2e_time": duration,
                            "e2e_time": (end_time - start_time),
                        }

                        # Create directory if it doesn't exist
                        os.makedirs(os.path.dirname(json_path), exist_ok=True)

                        # Append new data
                        if not isinstance(existing_data, list):
                            existing_data = [existing_data]
                        existing_data.append(to_save)

                        # Save updated data
                        with open(json_path, "w") as f:
                            json.dump(existing_data, f, indent=2)


"""
python benchmark/component_scaling_efficiency/e2e_scaling_efficiency/test_e2e.py -g 2 --repeat 3 --logging
CUDA_VISIBLE_DEVICES=2,3 python benchmark/component_scaling_efficiency/e2e_scaling_efficiency/test_e2e.py -g 2 --repeat 3 --logging
"""

if __name__ == "__main__":
    from torch.multiprocessing import spawn
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-gpu-num", "-g", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--logging", action="store_true", default=False)
    args = parser.parse_args()

    nprocs = args.gpu_num

    spawn(
        run_test,
        args=(
            nprocs,
            args.repeat,
            random.randint(8000, 65535),
            args.logging,
        ),
        nprocs=nprocs,
    )
