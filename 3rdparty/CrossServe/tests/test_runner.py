import os
import torch
import random
from cfuser.engine.runner import CServeRunner
from cfuser.scheduler.request import ScheduledRequests, ScheduledRequest
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


def test_runner(rank, world_size, ulysses_degree, ring_degree, repeat, master_port):
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
        height=1024,
        width=1024,
        prompt=["a beautiful river with flowers growing on a turtle's back"] * 1,
        num_inference_steps=3,
        output_type="latent",
        seed=42,
    )

    runner = CServeRunner(pretrained_model_name_or_path=engine_config.model_config.model, engine_config=engine_config)

    req = ScheduledRequest(
        req_ids=[0],
        attn_ranks=[0],
        non_attn_ranks=[0, 1],
        attn_ulysses_degree=1,
        attn_ring_degree=1,
        non_attn_sp_degree=2,
        input_config=input_config,
    )

    reqs = ScheduledRequests(requests=[req])
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(repeat):
        if rank in req.non_attn_ranks:
            runner.generate(reqs)
    end_event.record()
    end_event.synchronize()
    print(f"[rank {rank}] time per request: {start_event.elapsed_time(end_event) / repeat / 1000} s")
    # for i in range(4):
    #     req1 = ScheduledRequest(
    #         req_ids=[0],
    #         attn_ranks=[0, 1, 2, 3],
    #         non_attn_ranks=[0, 1, 2, 3],
    #         attn_ulysses_degree=4,
    #         attn_ring_degree=1,
    #         non_attn_sp_degree=4,
    #         input_config=input_config,
    #     )

    #     reqs = ScheduledRequests(requests=[req1])
    #     runner.generate(reqs)

    # runner.generate(reqs)

    # req2 = ScheduledRequest(
    #     req_ids=[1],
    #     attn_ranks=[0, 1],
    #     non_attn_ranks=[0, 1, 2, 3],
    #     attn_ulysses_degree=2,
    #     attn_ring_degree=1,
    #     non_attn_sp_degree=4,
    #     input_config=input_config,
    # )

    # req3 = ScheduledRequest(
    #     req_ids=[2],
    #     attn_ranks=[2, 3],
    #     non_attn_ranks=[0, 1, 2, 3],
    #     attn_ulysses_degree=2,
    #     attn_ring_degree=1,
    #     non_attn_sp_degree=4,
    #     input_config=input_config,
    # )

    # reqs = ScheduledRequests(requests=[req2])
    # reqs = ScheduledRequests(requests=[req2, req3])
    # runner.generate(reqs)

    # req_list = []
    # for i in range(4):
    #     req_list.append(
    #         ScheduledRequest(
    #             req_ids=[2 + i],
    #             attn_ranks=[i],
    #             non_attn_ranks=[0, 1, 2, 3],
    #             attn_ulysses_degree=1,
    #             attn_ring_degree=1,
    #             non_attn_sp_degree=4,
    #             input_config=input_config,
    #         )
    #     )
    # reqs = ScheduledRequests(requests=req_list)
    # runner.generate(reqs)

    # req4 = ScheduledRequest(
    #     req_ids=[3],
    #     attn_ranks=[0, 1],
    #     non_attn_ranks=[0, 1, 2, 3],
    #     attn_ulysses_degree=1,
    #     attn_ring_degree=2,
    #     non_attn_sp_degree=4,
    #     input_config=input_config,
    # )

    # req5 = ScheduledRequest(
    #     req_ids=[4],
    #     attn_ranks=[2, 3],
    #     non_attn_ranks=[0, 1, 2, 3],
    #     attn_ulysses_degree=1,
    #     attn_ring_degree=2,
    #     non_attn_sp_degree=4,
    #     input_config=input_config,
    # )

    # reqs = ScheduledRequests(requests=[req4, req5])
    # runner.generate(reqs)

    get_runtime_state().destory_distributed_env()


"""
python tests/test_runner.py --ulysses_degree 2 --ring_degree 1 --repeat 1
"""

if __name__ == "__main__":
    from torch.multiprocessing import spawn
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--ulysses_degree", "-u", type=int, default=2)
    parser.add_argument("--ring_degree", "-r", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=1)
    args = parser.parse_args()

    nprocs = args.ulysses_degree * args.ring_degree

    spawn(
        test_runner,
        args=(
            nprocs,
            args.ulysses_degree,
            args.ring_degree,
            args.repeat,
            random.randint(8000, 65535),
        ),
        nprocs=nprocs,
    )
